#!/usr/bin/env python3
# Spiking PEPITA-style training with segmented datasets (HAR/Speech Commands/MNIST)
# Toggle accumulation vs. original update with ACCUMULATION_MODE.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import snntorch as snn
from snntorch import utils

# dynamic dataset loader + helpers from your datasets.py
from datasets import get_dataloaders, flatten_segments, majority_vote

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Dataset
DATASET_NAME  = "har"
DATA_ROOT     = "../data"
SAMPLE_LENGTH = None

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
def train_epoch(model: SNN,
                train_loader,
                projection: torch.Tensor,
                n_classes: int,
                mode: str) -> float:
    """One training epoch with PEPITA-like two-pass update on SEGMENTS.
       mode ∈ {"original", "accum"}."""
    model.train()
    running_acc = 0.0

    for X, y in train_loader:
        # X: [B,S,T,D], y: [B] → flatten to segments
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        X_segs, y_segs, _, _ = flatten_segments(X, y)  # [Nseg,T,D], [Nseg]
        if X_segs.numel() == 0:
            continue

        N, T, C = X_segs.shape

        # ─ First pass ─
        model.reset()

        if mode == "accum":
            rates      = torch.zeros((N, n_classes),              device=DEVICE)
            h_norm_sum = torch.zeros((N, model.fc1.out_features), device=DEVICE)
            for t in range(T):
                x_t = X_segs[:, t, :]              # [N, C]
                h_t = model.lif1(model.fc1(x_t))   # [N, H]
                o_t = model.fc2(h_t)               # [N, K]
                rates      += o_t
                h_norm_sum += h_t
            h_norm_avg = h_norm_sum / T            # [N, H]
        else:
            h_rec, o_rec = [], []
            for t in range(T):
                x_t = X_segs[:, t, :]
                h_t = model.lif1(model.fc1(x_t))
                o_t = model.fc2(h_t)
                h_rec.append(h_t)
                o_rec.append(o_t)
            h_norm = torch.stack(h_rec, dim=0)           # [T, N, H]
            rates  = torch.stack(o_rec,  dim=0).sum(0)   # [N, K]

        # Probabilities & error (segment labels)
        p      = F.softmax(rates, dim=1)                       # [N, K]
        onehot = F.one_hot(y_segs, n_classes).float()          # [N, K]
        e      = p - onehot                                    # [N, K]

        # ─ Input modulation ─
        proj_err   = e @ projection                            # [N, C]
        X_mod      = X_segs + proj_err.unsqueeze(1)            # [N, T, C]

        # ─ Second pass ─
        model.reset()

        if mode == "accum":
            h_mod_sum = torch.zeros((N, model.fc1.out_features), device=DEVICE)
            x_mod_sum = torch.zeros((N, C), device=DEVICE)
            for t in range(T):
                x_mod = X_mod[:, t, :]
                h_mod = model.lif1(model.fc1(x_mod))
                h_mod_sum += h_mod
                x_mod_sum += x_mod
            h_mod_avg = h_mod_sum / T                          # [N, H]
            x_mod_avg = x_mod_sum / T                          # [N, C]

            # ΔW1 (outer product of means)
            # shapes: (H,N) @ (N,C) → [H,C]
            delta_w1 = - (h_norm_avg - h_mod_avg).t() @ x_mod_avg / N
            model.fc1.weight.add_(lr * delta_w1)

        else:
            # Keep full sequences to compute mean of outer products
            h_mod_rec = []
            for t in range(T):
                x_mod = X_mod[:, t, :]
                h_mod_rec.append(model.lif1(model.fc1(x_mod)))
            h_mod = torch.stack(h_mod_rec, dim=0)              # [T, N, H]

            # ΔW1 (mean over time of outer products)
            diff     = (h_norm - h_mod)                        # [T, N, H]
            x_mod_T  = X_mod.permute(1, 0, 2)                  # [T, N, C]
            mult     = diff.unsqueeze(3) * x_mod_T.unsqueeze(2)  # [T,N,H,C]
            delta_w1 = - mult.sum(dim=(0,1)) / (N * T)         # [H, C]
            model.fc1.weight.add_(lr * delta_w1)

            # for ΔW2 we still need h_mod_avg:
            h_mod_avg = h_mod.mean(dim=0)                      # [N, H]

        # ΔW2 (shared for both modes): (K,N) @ (N,H) → [K,H]
        delta_w2 = - (e.t() @ h_mod_avg) / N
        model.fc2.weight.add_(lr * delta_w2)

        # batch segment-accuracy (from first pass)
        preds = p.argmax(dim=1)
        running_acc += (preds == y_segs).float().mean().item()

    return 100.0 * running_acc / max(1, len(train_loader))


@torch.no_grad()
def evaluate_with_voting(model: SNN, loader, n_classes: int) -> float:
    """Sum firing readout over time per segment, majority vote to sample."""
    model.eval()
    total_acc = 0.0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)               # X:[B,S,T,D], y:[B]
        B = y.size(0)

        X_segs, _, sample_ids, _ = flatten_segments(X, y)  # [Nseg,T,D], [Nseg], [Nseg]
        if X_segs.numel() == 0:
            continue

        model.reset()
        N, T, C = X_segs.shape
        out_rec = []
        for t in range(T):
            x_t = X_segs[:, t, :]
            h_t = model.lif1(model.fc1(x_t))
            out_rec.append(model.fc2(h_t))
        rates = torch.stack(out_rec, dim=0).sum(dim=0)  # [Nseg, K]
        preds_seg = rates.argmax(dim=1)                 # [Nseg]

        preds_sample = majority_vote(preds_seg, sample_ids.to(DEVICE), n_classes, B)  # [B]
        total_acc += (preds_sample == y).float().mean().item()

    return 100.0 * total_acc / max(1, len(loader))


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)

    # Data: segmented loaders
    train_loader, test_loader, meta = get_dataloaders(
        DATASET_NAME,
        root=DATA_ROOT,
        batch_size=batch_size,
        sample_length=SAMPLE_LENGTH,   # set to int to cap/split (e.g., 128)
    )
    in_channels = meta["input_dim"]   # D
    n_classes   = meta["n_classes"]   # K
    print(f"[Data] {DATASET_NAME.upper()} | D={in_channels} | K={n_classes} | segment_T≈{meta['time_steps']}")
    print(f"[Classes] {meta['class_names']}")
    print(f"[Mode]  ACCUMULATION_MODE = '{ACCUMULATION_MODE}'")

    # Model
    model = SNN(in_channels=in_channels, hidden_dim=hidden_dim, output_dim=n_classes, beta=beta).to(DEVICE)

    # Projection matrix K→D (error to channel space)
    projection = torch.empty(n_classes, in_channels, device=DEVICE)
    init.kaiming_normal_(projection, nonlinearity="relu")
    projection.mul_(f_factor)

    # Train
    for epoch in range(1, num_epochs + 1):
        train_acc = train_epoch(model, train_loader, projection, n_classes, ACCUMULATION_MODE)
        print(f"Epoch {epoch:02d}/{num_epochs} | Train Seg Acc: {train_acc:.2f}%")

    # Eval (majority vote)
    test_acc = evaluate_with_voting(model, test_loader, n_classes)
    print(f"Test Accuracy (vote): {test_acc:.2f}%")

# Windows / PyCharm debug safe entry point
if __name__ == "__main__":
    main()
