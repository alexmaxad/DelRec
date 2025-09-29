import os
import json
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import numpy as np
import wandb

from tqdm import tqdm

from configs.equiparam_SHD import Config
from src.SHD.trainer import *  
from src.SHD.snn import *  
from src.datasets import load_dataset
from src.utils import reset_states

WANDB_KEY = None # Your key here

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


@torch.no_grad()
def mean_spikes_on_test(model, test_loader, device):
    
    model.eval()

    registrators = [m for m in model.modules() if m.__class__.__name__ == "spike_registrator"]
    print(f'Found: {len(registrators)} module(s) named "spike_registrator" in {model.__class__.__name__}.')
    if not registrators:
        raise RuntimeError("No module named 'spike_registrator' found in model.")

    total_spike_sum = 0.0      # sum of all spikes over layers & batches
    total_norm = 0.0           # sum of (B*N*T) over layers & batches

    for batch in test_loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch

        x = x.permute(1, 0, 2).float().to(device)  # (T, B, N)

        for r in registrators:
            if hasattr(r, "spikes"):
                r.spikes = None

        reset_states(model=model)
        _ = model(x)

        for m in registrators:
            spk = getattr(m, "spikes", None)  # (T, B, N)
            if spk is None or not torch.is_tensor(spk):
                continue
            assert spk.dim() == 3, "spikes should have shape (T, B, N)"
            T, B, N = spk.shape
            total_spike_sum += spk.float().detach().sum().item()
            total_norm += float(B) * float(N) * float(T)

    return float(total_spike_sum / max(1.0, total_norm))

def run_one(model_class, seed, lambda_spike, base_results_dir=None, wandb_project="Penalize spikes SHD"):
    config = Config()
    config.seed = seed
    config.spike_penalty = float(lambda_spike)
    seed_everything(seed=config.seed, is_cuda=torch.cuda.is_available())

    if model_class == SNN:
        config.hidden_layers = [52, 52]
    elif model_class == SNN_feedforward_delays:
        config.hidden_layers = [42, 42]
    elif model_class == SNN_recurrent_delays:
        config.hidden_layers = [42, 42]
    elif model_class == SNN_recurrent_and_feedforward_delays:
        config.hidden_layers = [38, 38]
    elif model_class == SNN_fixed_recurrent_delays:
        config.hidden_layers = [42, 42]
    elif model_class == SNN_vanilla_recurrent:
        config.hidden_layers = [42, 42]

    train_loader, valid_loader, test_loader = load_dataset(config)
    model = model_class(config).to(device)
    optimizer, scheduler = init_optim_sche(model, config)
    count_parameters(model)

    model_name = getattr(model, "module", model).__class__.__name__
    cfg = {k: v for k, v in dict(vars(Config)).items() if "__" not in k}
    cfg.update({"seed": seed, "model": model_name, "lambda_spike": lambda_spike})

    wandb.login(key=WANDB_KEY)
    wandb.init(
        project=wandb_project,
        config=cfg,
        group=f"{model_name}|λ={lambda_spike:g}",
        name=f"{model_name}/seed{seed}_{config.dataset}_lam{lambda_spike:g}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        reinit=True,
    )

    # Track delay evolution
    prev_recurrent_delays_vals, prev_ff_delays_vals = [], []
    for m in model.layers:
        if isinstance(m, axonal_recdel):
            prev_recurrent_delays_vals.append(m.recurrent_delays.detach().clone())
        if isinstance(m, dcls_module):
            prev_ff_delays_vals.append(m.P.detach().clone())

    if not base_results_dir:
        base_results_dir = f'./exp/{config.dataset}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    results_dir = os.path.join(base_results_dir, f"lam{lambda_spike:g}", f"{model_name}_seed{seed}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    config.results_dir = results_dir
    with open(os.path.join(results_dir, "config.json"), "w") as fid:
        json.dump(config.__dict__, fid, indent=2)

    # For storing per-epoch results
    train_res, val_res = pd.DataFrame(), pd.DataFrame()
    best_val_acc = 0.0

    for epoch in range(config.epochs):
        for m in model.layers:
            if isinstance(m, axonal_recdel):
                m.update_sigma(epoch)

        train_acc, train_loss = train(train_loader, model, optimizer, epoch, device, config, penalize_spikes=True)
        val_acc, val_loss = test(valid_loader, model, epoch, device, config, penalize_spikes=True)

        for sc in scheduler:
            sc.step()

        # Save epoch logs
        train_res[str(epoch)] = [train_acc, train_loss]
        val_res[str(epoch)] = [val_acc, val_loss]
        train_res.to_csv(os.path.join(results_dir, "train_res.csv"), index=True)
        val_res.to_csv(os.path.join(results_dir, "val_res.csv"), index=True)

        # Track delay variations
        rec_delay_layer_idx, ff_delay_layer_idx = 0, 0
        for layer in model.layers:
            if isinstance(layer, axonal_recdel):
                current = layer.recurrent_delays.detach()
                prev = prev_recurrent_delays_vals[rec_delay_layer_idx]
                delta = (current - prev).abs().mean().item()
                wandb.log({f"recurrent_delay_variation/layer{rec_delay_layer_idx}": delta})
                prev_recurrent_delays_vals[rec_delay_layer_idx] = current.clone()
                rec_delay_layer_idx += 1

            if isinstance(layer, dcls_module):
                current = layer.P.detach()
                prev = prev_ff_delays_vals[ff_delay_layer_idx]
                delta = (current - prev).abs().mean().item()
                wandb.log({f"feedforward_delay_variation/layer{ff_delay_layer_idx}": delta})
                prev_ff_delays_vals[ff_delay_layer_idx] = current.clone()
                ff_delay_layer_idx += 1

        state = {"net": model.state_dict(), "acc": val_acc, "epoch": epoch}
        torch.save(state, os.path.join(results_dir, "last.pth"))
        if val_acc >= best_val_acc:
            torch.save(state, os.path.join(results_dir, "best.pth"))
            best_val_acc = val_acc

        print(
            "Val Epoch: [{}/{}], lr: {:.6f}, lr_pos: {:.6f}, acc: {:.4f}, best: {:.4f}".format(
                epoch,
                config.epochs,
                optimizer[0].param_groups[0]["lr"],
                optimizer[1].param_groups[0]["lr"],
                val_acc,
                best_val_acc,
            )
        )

    # Final test on best checkpoint
    best_ckpt = torch.load(os.path.join(results_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["net"])
    best_epoch = best_ckpt["epoch"]
    best_val = best_ckpt["acc"]

    final_test_acc, final_test_loss = test(test_loader, model, epoch, device, config)
    print(
        f"\n[{model_name} | seed {seed} | λ={lambda_spike:g}] Final Test ► Acc: {final_test_acc:.4f}, "
        f"Loss: {final_test_loss:.4f} (from best val_acc={best_val:.4f}@epoch={best_epoch})"
    )
    wandb.log({"final_test/acc": final_test_acc, "final_test/loss": final_test_loss})

    # Compute spike metric on test using the best checkpointed model
    mean_spk = mean_spikes_on_test(model, test_loader, device)
    wandb.log({"final_test/mean_spikes_per_sample_per_neuron_per_timestep": mean_spk})

    wandb.finish()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        model_name, seed,
        float(final_test_acc), float(final_test_loss),
        float(best_val), int(best_epoch),
        results_dir, float(lambda_spike),
        float(mean_spk),
    )


if __name__ == "__main__":
    
    model_classes = [
        SNN,
        SNN_feedforward_delays,
        SNN_recurrent_delays,
        SNN_fixed_recurrent_delays,
        SNN_recurrent_and_feedforward_delays,
        SNN_vanilla_recurrent
    ]

    seeds = [0, 1, 2]
    lambda_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10] 

    # Base results directory
    master_results_dir = f'./exp/SHD_penalize_spikes/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    Path(master_results_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for lam in lambda_grid:
        for model_class in model_classes:
            for seed in seeds:
                out = run_one(
                    model_class=model_class,
                    seed=seed,
                    lambda_spike=lam,
                    base_results_dir=master_results_dir,
                    wandb_project="Penalize spikes SHD",
                )
                rows.append(out)

    all_cols = [
        "model", "seed", "final_test_acc", "final_test_loss",
        "best_val_acc", "best_epoch", "results_dir", "lambda_spike",
        "mean_spikes_test",
    ]
    df = pd.DataFrame(rows, columns=all_cols)

    agg_model = (df.groupby("model")
                   .agg(
                       mean_test_acc=("final_test_acc", "mean"),
                       std_test_acc=("final_test_acc", "std"),
                       mean_test_loss=("final_test_loss", "mean"),
                       std_test_loss=("final_test_loss", "std"),
                       mean_best_val=("best_val_acc", "mean"),
                       std_best_val=("best_val_acc", "std"),
                       mean_spikes_test=("mean_spikes_test", "mean"),
                       std_spikes_test=("mean_spikes_test", "std"),
                   ).reset_index())

    agg_model_lambda = (df.groupby(["model", "lambda_spike"])
                          .agg(
                              mean_test_acc=("final_test_acc", "mean"),
                              std_test_acc=("final_test_acc", "std"),
                              mean_test_loss=("final_test_loss", "mean"),
                              std_test_loss=("final_test_loss", "std"),
                              mean_best_val=("best_val_acc", "mean"),
                              std_best_val=("best_val_acc", "std"),
                              mean_spikes_test=("mean_spikes_test", "mean"),
                              std_spikes_test=("mean_spikes_test", "std"),
                          ).reset_index())

    # Save CSVs
    df.to_csv(os.path.join(master_results_dir, "all_runs.csv"), index=False)
    agg_model.to_csv(os.path.join(master_results_dir, "summary_by_model.csv"), index=False)
    agg_model_lambda.to_csv(os.path.join(master_results_dir, "summary_by_model_and_lambda.csv"), index=False)

    print(f"\n=== Summary by model : {len(seeds)} seeds ===")
    for _, r in agg_model.iterrows():
        print(
            f"{r['model']}: "
            f"test_acc = {r['mean_test_acc']:.4f} ± {r['std_test_acc']:.4f}, "
            f"test_loss = {r['mean_test_loss']:.4f} ± {r['std_test_loss']:.4f}, "
            f"best_val = {r['mean_best_val']:.4f} ± {r['std_best_val']:.4f}, "
            f"mean_spikes_test = {r['mean_spikes_test']:.6f} ± {r['std_spikes_test']:.6f}"
        )

    print(f"\n=== Summary by model & λ : {len(seeds)} seeds ===")
    for _, r in agg_model_lambda.iterrows():
        print(
            f"{r['model']} | λ={r['lambda_spike']}: "
            f"test_acc = {r['mean_test_acc']:.4f} ± {r['std_test_acc']:.4f}, "
            f"test_loss = {r['mean_test_loss']:.4f} ± {r['std_test_loss']:.4f}, "
            f"best_val = {r['mean_best_val']:.4f} ± {r['std_best_val']:.4f}, "
            f"mean_spikes_test = {r['mean_spikes_test']:.6f} ± {r['std_spikes_test']:.6f}"
        )
