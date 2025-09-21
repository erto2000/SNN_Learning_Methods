#!/usr/bin/env python3
# Spiking PEPITA-style training with dynamic datasets (HAR/WISDM/Speech Commands)
# Toggle accumulation vs. original update with ACCUMULATION_MODE.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import utils

# dynamic dataset loader from your datasets.py
from datasets import get_dataloaders

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Dataset
DATASET_NAME = "har"          # "har", "wisdm", or "speech_commands"
DATA_ROOT    = "../data"
WINDOW_LEN   = 128              # used by HAR/WISDM, ignored for Speech Commands
NUM_WORKERS  = 2                # set to 0 when debugging on Windows if needed
MAX_SAMPLES  = None             # e.g., 5000 for quick tests (SC caps to 10k if None)

# Training
num_epochs  = 10
batch_size  = 128
hidden_dim  = 128
beta        = 0.9
lr          = 0.01
f_factor    = 0.05
SEED        = 123

# Update rule mode: "original" (mean of outer products over time)
#                 or "accum"    (outer product of time-averaged stats)
ACCUMULATION_MODE = "accum"     # "original" or "accum"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL ────────────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, output_dim: int, beta: float):
        super().__init__()
        self.fc1  = nn.Linear(in_channels, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2  = nn.Linear(hidden_dim, output_dim)

        # He/Kaiming initialization
        init.kaiming_uniform_(self.fc1.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        init.zeros_(self.fc1.bias)
        init.kaiming_uniform_(self.fc2.weight, a=0.0, mode="fan_in", nonlinearity="linear")
        init.zeros_(self.fc2.bias)

    def reset(self):
        utils.reset(self.lif1)

# ─── TRAIN / EVAL LOGIC ───────────────────────────────────────────────────────
@torch.no_grad()
def train_epoch(model: SNN, train_loader: DataLoader, projection: torch.Tensor,
                n_classes: int, mode: str) -> float:
    """One training epoch with PEPITA-like two-pass update.
       mode ∈ {"original", "accum"}"""
    model.train()
    running_acc = 0.0

    for series, labels in train_loader:
        # series: [B, T, C]
        series = series.to(DEVICE)
        labels = labels.to(DEVICE)
        B, T, C = series.shape

        # ─ First pass ─
        model.reset()

        if mode == "accum":
            # Accumulate sums only
            rates      = torch.zeros((B, n_classes),                device=DEVICE)
            h_norm_sum = torch.zeros((B, model.fc1.out_features),   device=DEVICE)
            for t in range(T):
                x_t = series[:, t, :]                # [B, C]
                h_t = model.lif1(model.fc1(x_t))     # [B, H]
                o_t = model.fc2(h_t)                 # [B, K]
                rates      += o_t
                h_norm_sum += h_t
            h_norm_avg = h_norm_sum / T              # [B, H]
        else:
            # Keep full sequences to do mean of outer products
            h_rec, o_rec = [], []
            for t in range(T):
                x_t = series[:, t, :]
                h_t = model.lif1(model.fc1(x_t))
                o_t = model.fc2(h_t)
                h_rec.append(h_t)
                o_rec.append(o_t)
            h_norm = torch.stack(h_rec, dim=0)       # [T, B, H]
            rates  = torch.stack(o_rec,  dim=0).sum(dim=0)  # [B, K]
            # (we’ll compute h_norm_avg later only if needed)

        # Probabilities & error
        p      = F.softmax(rates, dim=1)                 # [B, K]
        onehot = F.one_hot(labels, n_classes).float()    # [B, K]
        e      = p - onehot                              # [B, K]

        # ─ Input modulation ─
        proj_err   = e @ projection                      # [B, C]
        series_mod = series + proj_err.unsqueeze(1)      # [B, T, C]

        # ─ Second pass ─
        model.reset()

        if mode == "accum":
            # Accumulate only means
            h_mod_sum = torch.zeros((B, model.fc1.out_features), device=DEVICE)
            x_mod_sum = torch.zeros((B, C), device=DEVICE)
            for t in range(T):
                x_mod = series_mod[:, t, :]
                h_mod = model.lif1(model.fc1(x_mod))
                h_mod_sum += h_mod
                x_mod_sum += x_mod
            h_mod_avg = h_mod_sum / T                   # [B, H]
            x_mod_avg = x_mod_sum / T                   # [B, C]

            # ΔW1 (outer product of means)
            delta_w1 = - (h_norm_avg - h_mod_avg).t() @ x_mod_avg / B   # [H, C]
            model.fc1.weight.add_(lr * delta_w1)

        else:
            # Keep full sequences to compute mean of outer products
            h_mod_rec = []
            for t in range(T):
                x_mod = series_mod[:, t, :]
                h_mod_rec.append(model.lif1(model.fc1(x_mod)))
            h_mod = torch.stack(h_mod_rec, dim=0)       # [T, B, H]

            # ΔW1 (mean over time of outer products)
            diff     = (h_norm - h_mod)                 # [T, B, H]
            x_mod_T  = series_mod.permute(1, 0, 2)      # [T, B, C]
            mult     = diff.unsqueeze(3) * x_mod_T.unsqueeze(2)  # [T,B,H,C]
            delta_w1 = - mult.sum(dim=(0,1)) / (B * T)  # [H, C]
            model.fc1.weight.add_(lr * delta_w1)

            # for ΔW2 we still need h_mod_avg:
            h_mod_avg = h_mod.mean(dim=0)               # [B, H]

        # ΔW2 (shared for both modes)
        delta_w2 = - (e.t() @ h_mod_avg) / B            # [K, H]
        model.fc2.weight.add_(lr * delta_w2)

        # batch accuracy (from first pass)
        preds = p.argmax(dim=1)
        running_acc += (preds == labels).float().mean().item()

    return 100.0 * running_acc / max(1, len(train_loader))


@torch.no_grad()
def evaluate(model: SNN, loader: DataLoader) -> float:
    """Sum firing readout over time, then argmax."""
    model.eval()
    total_acc = 0.0
    for series, labels in loader:
        series, labels = series.to(DEVICE), labels.to(DEVICE)
        B, T, C = series.shape
        model.reset()
        out_rec = []
        for t in range(T):
            x_t = series[:, t, :]
            h_t = model.lif1(model.fc1(x_t))
            out_rec.append(model.fc2(h_t))
        rates = torch.stack(out_rec, dim=0).sum(dim=0)   # [B, K]
        preds = rates.argmax(dim=1)
        total_acc += (preds == labels).float().mean().item()
    return 100.0 * total_acc / max(1, len(loader))

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)

    # Data: dynamic loaders
    train_loader, test_loader, meta = get_dataloaders(
        DATASET_NAME,
        root=DATA_ROOT,
        batch_size=batch_size,
        window_len=WINDOW_LEN,
        num_workers=NUM_WORKERS,
        max_samples=MAX_SAMPLES,
    )
    in_channels = meta["input_dim"]   # C
    n_classes   = meta["n_classes"]   # K
    print(f"[Data] {DATASET_NAME.upper()} | C={in_channels} | K={n_classes} | ~T={meta['time_steps']}")
    print(f"[Classes] {meta['class_names']}")
    print(f"[Mode]  ACCUMULATION_MODE = '{ACCUMULATION_MODE}'")

    # Model
    model = SNN(in_channels=in_channels, hidden_dim=hidden_dim, output_dim=n_classes, beta=beta).to(DEVICE)

    # Projection matrix K→C (error to channel space)
    projection = torch.empty(n_classes, in_channels, device=DEVICE)
    init.kaiming_normal_(projection, nonlinearity="relu")
    projection.mul_(f_factor)

    # Train
    for epoch in range(1, num_epochs + 1):
        train_acc = train_epoch(model, train_loader, projection, n_classes, ACCUMULATION_MODE)
        print(f"Epoch {epoch:02d}/{num_epochs} | Train Acc: {train_acc:.2f}%")

    # Eval
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

# Windows / PyCharm debug safe entry point
if __name__ == "__main__":
    main()
