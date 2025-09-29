import os
import json
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import numpy as np
import wandb

from configs.equiparam_SHD import Config
from src.SHD.trainer import *
from src.SHD.snn import *  # SNN, SNN_feedforward_delays, SNN_recurrent_delays, SNN_recurrent_and_feedforward_delays, SNN_fixed_recurrent_delays, SNN_vanilla_recurrent
from src.datasets import load_dataset

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

def _default_hidden_for_model(model_class):
    if model_class == SNN:
        return [52, 52]  # ~11k params
    elif model_class == SNN_feedforward_delays:
        return [42, 42]  # ~10.4k params
    elif model_class in [SNN_recurrent_delays, SNN_vanilla_recurrent, SNN_fixed_recurrent_delays]:
        return [42, 42]  # ~10.4k params
    elif model_class == SNN_recurrent_and_feedforward_delays:
        return [38, 38]  # ~10.6k params
    else:
        raise ValueError(f"No default hidden sizes for {model_class}")

def run_one(model_class, seed, hidden_layers=None, base_results_dir=None, wandb_project="Equiparam SHD"):
    config = Config()
    config.seed = seed
    seed_everything(seed=config.seed, is_cuda=True)

    if hidden_layers is not None:
        assert isinstance(hidden_layers, (list, tuple)) and len(hidden_layers) == 2
        config.hidden_layers = list(map(int, hidden_layers))
    else:
        config.hidden_layers = _default_hidden_for_model(model_class)

    train_loader, valid_loader, test_loader = load_dataset(config)
    model = model_class(config).to(device)
    optimizer, scheduler = init_optim_sche(model, config)
    num_params = count_parameters(model)

    model_name = getattr(model, "module", model).__class__.__name__
    cfg = {k: v for k, v in dict(vars(Config)).items() if "__" not in k}
    cfg.update({"seed": seed, "model": model_name, "hidden_layers": config.hidden_layers})

    h1, h2 = config.hidden_layers
    run_tag = f"h{h1}-{h2}"
    wandb.init(
        project=wandb_project,
        config=cfg,
        group=f"{model_name}/{run_tag}",
        name=f"{model_name}/{run_tag}/seed{seed}_{config.dataset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        reinit=True,
    )

    prev_recurrent_delays_vals, prev_ff_delays_vals = [], []
    for m in model.layers:
        if isinstance(m, axonal_recdel):
            prev_recurrent_delays_vals.append(m.recurrent_delays.detach().clone())
        if isinstance(m, dcls_module):
            prev_ff_delays_vals.append(m.P.detach().clone())

    if not base_results_dir:
        base_results_dir = f'./exp/{config.dataset}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    
    results_dir = os.path.join(base_results_dir, f"{model_name}_{run_tag}_seed{seed}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    config.results_dir = results_dir
    with open(os.path.join(results_dir, "config.json"), "w") as fid:
        json.dump(config.__dict__, fid, indent=2)

    
    train_res, val_res = pd.DataFrame(), pd.DataFrame()
    best_val_acc = 0.0

    for epoch in range(config.epochs):
        for m in model.layers:
            if isinstance(m, axonal_recdel):
                m.update_sigma(epoch)

        train_acc, train_loss = train(train_loader, model, optimizer, epoch, device, config)
        val_acc, val_loss = test(valid_loader, model, epoch, device, config)

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
                wandb.log({f"recurrent_delay_variation/layer{rec_delay_layer_idx}": delta, "epoch": epoch})
                prev_recurrent_delays_vals[rec_delay_layer_idx] = current.clone()
                rec_delay_layer_idx += 1

            if isinstance(layer, dcls_module):
                current = layer.P.detach()
                prev = prev_ff_delays_vals[ff_delay_layer_idx]
                delta = (current - prev).abs().mean().item()
                wandb.log({f"feedforward_delay_variation/layer{ff_delay_layer_idx}": delta, "epoch": epoch})
                prev_ff_delays_vals[ff_delay_layer_idx] = current.clone()
                ff_delay_layer_idx += 1

        state = {"net": model.state_dict(), "acc": val_acc, "epoch": epoch, "hidden_layers": config.hidden_layers}
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
        f"\n[{model_name} | {run_tag} | seed {seed}] Final Test ► Acc: {final_test_acc:.4f}, "
        f"Loss: {final_test_loss:.4f} (from best val_acc={best_val:.4f}@epoch={best_epoch})"
    )
    wandb.log({
        "final_test/acc": final_test_acc,
        "final_test/loss": final_test_loss,
        "hidden/h1": h1, "hidden/h2": h2
    })
    wandb.finish()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        model_name,
        seed,
        h1,
        h2,
        num_params,
        float(final_test_acc),
        float(final_test_loss),
        float(best_val),
        int(best_epoch),
        results_dir,
        )

if __name__ == "__main__":
    
    wandb.login(key=WANDB_KEY)

    model_classes = [
        SNN,
        SNN_recurrent_delays,
        SNN_feedforward_delays,
        SNN_recurrent_and_feedforward_delays,
        SNN_fixed_recurrent_delays,
        SNN_vanilla_recurrent,
    ]

    seeds = [0, 1, 2]

    # Base results directory
    master_results_dir = f'./exp/SHD_equiparams/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    def upper_hidden(model_class):
        return _default_hidden_for_model(model_class)

    # Collect raw results
    rows = []
    for model_class in model_classes:
        upper = upper_hidden(model_class)  # [u, u]
        u = int(upper[0])
        # include the upper limit + 5 reductions of 4 neurons each: u, u-6, u-12, u-18, u-24, u-30
        if model_class is SNN:
            widths = [u - 6*k for k in range(0, 8)]
        else:
            widths = [u - 6*k for k in range(0, 6)]
        for hidden in widths:
            hidden_layers = [hidden, hidden]
            for seed in seeds:
                out = run_one(
                    model_class=model_class,
                    seed=seed,
                    hidden_layers=hidden_layers,  
                    base_results_dir=master_results_dir,
                    wandb_project="Equiparam SHD",
                )
                rows.append(out)

    all_cols = [
        "model", "seed", "hidden_h1", "hidden_h2", "num_params",
        "final_test_acc", "final_test_loss", "best_val_acc", "best_epoch", "results_dir"
    ]
    df = pd.DataFrame(rows, columns=all_cols)

    agg = df.groupby(["model", "hidden_h1", "hidden_h2", "num_params"]).agg(
        mean_test_acc=("final_test_acc", "mean"),
        std_test_acc=("final_test_acc", "std"),
        mean_test_loss=("final_test_loss", "mean"),
        std_test_loss=("final_test_loss", "std"),
        mean_best_val=("best_val_acc", "mean"),
        std_best_val=("best_val_acc", "std"),
    ).reset_index()

    Path(master_results_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(master_results_dir, "all_runs.csv"), index=False)
    agg.to_csv(os.path.join(master_results_dir, "summary_by_model_and_hidden.csv"), index=False)

    print(f"\n=== Summary by model & hidden ({len(seeds)} seeds each) ===")
    for _, r in agg.iterrows():
        print(
        f"{r['model']} | h={int(r['hidden_h1'])}-{int(r['hidden_h2'])} "
        f"({r['num_params']} params): "
        f"test_acc = {r['mean_test_acc']:.4f} ± {r['std_test_acc']:.4f}, "
        f"test_loss = {r['mean_test_loss']:.4f} ± {r['std_test_loss']:.4f}, "
        f"best_val = {r['mean_best_val']:.4f} ± {r['std_best_val']:.4f}"
    )
