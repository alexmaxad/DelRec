import torch
import numpy as np
import wandb
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import os
import numpy as np

from configs.perf_SHD import Config
from src.SHD.trainer import *
from src.SHD.snn import *
from src.datasets import load_dataset

os.environ["WANDB_MODE"] = "disabled"
WANDB_KEY = None # Your key here

if __name__ == "__main__":
    
    seed_list = [0, 1, 2, 4, 5, 6, 7, 8, 9]  
    test_accuracies = []
    
    for run_seed in seed_list:

        config = Config()
        config.seed = run_seed 
        seed_everything(config.seed, is_cuda=True)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.set_default_dtype(torch.float32)
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU")
            
        train_loader, valid_loader, test_loader = load_dataset(config)
        model = SNN_recurrent_delays(config).to(device)
        optimizer, scheduler = init_optim_sche(model, config)
        count_parameters(model)
        
        model_name = getattr(model, "module", model).__class__.__name__
        cfg = {k:v for k,v in dict(vars(Config)).items() if '__' not in k}
        wandb.login(key=WANDB_KEY)
        wandb.init(
                project='SHD',
                config=cfg,
                name=f"SHD/{model_name}/{config.dataset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

        # Delay evolution :
        prev_recurrent_delays_vals = []
        prev_ff_delays_vals = []
        for m in model.layers:
            if isinstance(m, axonal_recdel):
                prev_recurrent_delays_vals.append(m.recurrent_delays.detach().clone())
            if isinstance(m, dcls_module):
                prev_ff_delays_vals.append(m.P.detach().clone())
                
        # for logs
        if config.results_dir == '':
            config.results_dir = './exp/' + config.dataset + '/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        with open(config.results_dir + '/config.json', 'w') as fid:
            json.dump(config.__dict__, fid, indent=2)
            
        # For storing results
        train_res = pd.DataFrame()
        val_res = pd.DataFrame()
        best_val_acc = 0.0

        for epoch in range(config.epochs):
            # If update sigma at each epoch :
            for m in model.layers:
                if isinstance(m, axonal_recdel):
                    m.update_sigma(epoch)
                    
            train_acc, train_loss = train(train_loader, model, optimizer, epoch, device, config)
            val_acc, val_loss = test(valid_loader, model, epoch, device, config)

            for sc in scheduler: sc.step()
            
            #for logs
            train_res[str(epoch)] = [train_acc, train_loss]
            val_res[str(epoch)] = [val_acc, val_loss]

            train_res.to_csv(os.path.join(config.results_dir, 'train_res.csv'), index=True)
            val_res.to_csv(os.path.join(config.results_dir, 'val_res.csv'), index=True)
            
            # Following delay variation
            rec_delay_layer_idx = 0
            ff_delay_layer_idx = 0
            for idx, m in enumerate(model.layers):
                if isinstance(m, axonal_recdel):
                    # Log mean recurrent delay variation
                    current = m.recurrent_delays.detach()
                    prev = prev_recurrent_delays_vals[rec_delay_layer_idx]
                    delta = (current - prev).abs().mean().item()

                    wandb.log({f'recurrent_delay_variation/layer{rec_delay_layer_idx}': delta})

                    # Update reference for next epoch
                    prev_recurrent_delays_vals[rec_delay_layer_idx] = current.clone()
                    rec_delay_layer_idx += 1
                
                if isinstance(m, dcls_module):
                    # Log mean feedforward delay variation
                    current = m.P.detach()
                    prev = prev_ff_delays_vals[ff_delay_layer_idx]
                    delta = (current - prev).abs().mean().item()

                    wandb.log({f'feedforward_delay_variation/layer{ff_delay_layer_idx}': delta})

                    # Update reference for next epoch
                    prev_ff_delays_vals[ff_delay_layer_idx] = current.clone()
                    ff_delay_layer_idx += 1
                    
            state = {'net': model.state_dict(), 'acc': val_acc, 'epoch': epoch,}
            torch.save(state, os.path.join(config.results_dir, 'last.pth'))
            
            if val_acc >= best_val_acc:
                torch.save(state, os.path.join(config.results_dir, 'best.pth'))
                best_val_acc = val_acc

            print(
                'Val Epoch: [{}/{}], lr: {:.6f}, lr_pos: {:.6f}, acc: {:.4f}, best: {:.4f}'
                .format(
                    epoch,
                    config.epochs,
                    optimizer[0].param_groups[0]['lr'],
                    optimizer[1].param_groups[0]['lr'],
                    val_acc,
                    best_val_acc
                )
            )
            
        ### Testing the best model ###
        best_ckpt = torch.load(os.path.join(config.results_dir, 'best.pth'))
        model.load_state_dict(best_ckpt['net'])
        best_epoch = best_ckpt['epoch']
        best_val   = best_ckpt['acc']

        final_test_acc, final_test_loss = test(
            test_loader, model, epoch, device, config,
        )

        print(f"\nFinal Test (seed {run_seed}) ► Acc: {final_test_acc:.4f}, "
              f"Loss: {final_test_loss:.4f} "
              f"(from best val_acc={best_val:.4f}@epoch={best_epoch})")

        wandb.log({
            'final_test/acc':  final_test_acc,
            'final_test/loss': final_test_loss
        })
        
        test_accuracies.append(final_test_acc)
        
        wandb.finish()
        
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    print(f"Seeds: {seed_list}")
    print(f"Mean final test accuracy: {mean_acc:.4f} ± {std_acc:.4f}")