import torch
import torch.nn.functional as F
import wandb

from src.recurrent_neurons import axonal_recdel
from src.SHD.snn import dcls_module, modified_batchnorm, SNN_vanilla_recurrent
from src.utils import *

def get_spike_cost(model, normalize="NT"):
    costs = []
    for m in model.modules():
        if m.__class__.__name__ == "spike_registrator":
            spk = getattr(m, "spikes", None)
            if spk is None or not torch.is_tensor(spk):
                continue
            spk = spk.float()  # (T, B, N) 

            if spk.dim() != 3:
                costs.append(0.5 * spk.pow(2).mean())
                continue

            T, B, N = spk.shape
            if normalize == "NT":
                per_sample = 0.5 * spk.pow(2).sum(dim=(0, 2)) / (T * N)   # (B,)
                costs.append(per_sample.mean())
            else:
                costs.append(0.5 * spk.pow(2).mean())

    if not costs:
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
    return torch.stack(costs).mean()

def train(train_loader, model, optimizer, epoch, device, config, penalize_spikes=False):
    train_loss = 0
    accuracy = 0

    model.train()
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        # inputs of shape (Batch, Time, Neurons)
        
        # if batch_idx > 3:  # for fast debug
        #     break
        
        targets = F.one_hot(targets, config.output_size).float()
        
        inputs = inputs.permute(1,0,2).float().to(device)  #(time, batch, neurons)
        targets = targets.to(device)
        
        reset_states(model=model)
        outputs = model(inputs)
        loss = calc_loss_SHD(outputs, targets)
        
        if penalize_spikes:
            spike_cost = get_spike_cost(model)
            loss += config.spike_penalty * spike_cost
            
            wandb.log({"spike_cost": spike_cost.item()})

        train_loss += loss.item()
        accuracy += calc_metric_SHD(outputs, targets)

        for opt in optimizer: opt.zero_grad()
        loss.backward()
        
        if model.__class__.__name__ == 'SNN_vanilla_recurrent':
            # Gradient clipping for recurrence of 1 :
            max_grad_val = 1.0 
            for layer in model.layers:
                if isinstance(layer, axonal_recdel):
                    torch.nn.utils.clip_grad_value_(layer.recurrent_weights, max_grad_val)
                elif isinstance(layer, modified_batchnorm):
                    torch.nn.utils.clip_grad_value_(layer.weight, max_grad_val)
                    if config.bias:
                        torch.nn.utils.clip_grad_value_(layer.bias, max_grad_val)
                elif isinstance(layer, torch.nn.Linear):
                    torch.nn.utils.clip_grad_value_(layer.weight, max_grad_val)
                    if config.bias:
                        torch.nn.utils.clip_grad_value_(layer.bias, max_grad_val)
                        
        for opt in optimizer: opt.step()
        
        if hasattr(model, 'clamp_delays'):
            model.clamp_delays()

        progress_bar(
            batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), 100 * accuracy/(batch_idx+1))
            )
        
    avg_loss = train_loss / len(train_loader)
    avg_acc = 100. * accuracy / len(train_loader)
    
    wandb.log({
        'train/loss': avg_loss,
        'train/acc': avg_acc,
        'epoch': epoch,
    })
    
    model.log_params()
            
    return avg_acc, avg_loss

def test(test_loader, model, epoch, device, config, penalize_spikes=False):
    test_loss = 0
    accuracy = 0

    model.eval()
    
    if hasattr(model, 'round_pos'):
        model.round_pos()
        
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            
            # inputs of shape (Batch, Time, Neurons)
            
            # if batch_idx > 3:  # for fast debug
            #     break
            
            targets = F.one_hot(targets, config.output_size).float()
            
            inputs = inputs.permute(1,0,2).float().to(device)  #(time, batch, neurons)
            targets = targets.to(device)
            
            reset_states(model=model)
            outputs = model(inputs)
            loss = calc_loss_SHD(outputs, targets)
            
            if penalize_spikes:
                spike_cost = get_spike_cost(model)
                loss += config.spike_penalty * spike_cost

            test_loss += loss.item()
            accuracy += calc_metric_SHD(outputs, targets)

            progress_bar(
                batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100 * accuracy/(batch_idx+1))
            )
    
    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * accuracy / len(test_loader)
    
    wandb.log({
        'test/loss': avg_loss,
        'test/acc': avg_acc,
        'epoch': epoch,
    })
    
    return avg_acc, avg_loss

def init_optim_sche(model, config):
    weights_norm = []
    weights = []
    positions = []

    for m in model.layers:
        if isinstance(m, torch.nn.Linear):
            weights.append(m.weight)
            if config.bias:
                weights.append(m.bias)
                
        elif isinstance(m, axonal_recdel):
            weights.append(m.recurrent_weights)
            positions.append(m.recurrent_delays)
            
            # if hasattr(m, 'p_spread'):
            #     positions.append(m.p_spread)
            
        elif isinstance(m, dcls_module):
            weights.append(m.weight)
            if config.bias:
                weights.append(m.bias)
            positions.append(m.P)
            
        elif isinstance(m, modified_batchnorm):
            weights_norm.append(m.weight)
            if config.bias:
                weights_norm.append(m.bias)

    optimizer = []
    scheduler = []

    if config.optim == 'adamW':
        optimizer.append(torch.optim.AdamW([{'params':weights, 'lr':config.lr_w, 'weight_decay':config.weight_decay},
                                           {'params':weights_norm, 'lr':config.lr_w, 'weight_decay':0},]))
        optimizer.append(torch.optim.AdamW([{'params':positions, 'lr':config.lr_positions, 'weight_decay':0}]))
    else:
        raise NotImplementedError

    if config.scheduler_weights == 'cos':
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=config.epochs))
    elif config.scheduler_weights == 'onecycle':
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[0], max_lr=config.lr_w, total_steps=config.epochs))
    else:
        raise NotImplementedError
    
    if config.scheduler_pos == 'cos':
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=config.epochs))
    elif config.scheduler_pos == 'onecycle':
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[1], max_lr=config.lr_positions, total_steps=config.epochs))
    else:
        raise NotImplementedError
    
    return optimizer, scheduler