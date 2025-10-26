#!/usr/bin/env python3
# Multi-layer Spiking PEPITA-style training with segmented datasets
# Supports "original" and "accum" update modes; DIMS defines hidden stack.

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
beta        = 0.9
lr          = 0.01
f_factor    = 0.05
SEED        = 12

# Update rule mode: "original" (mean of outer products over time)
#                 or "accum"    (outer product of time-averaged stats)
ACCUMULATION_MODE = "accum"     # "original" or "accum"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture: input_dim -> DIMS... -> output_dim
DIMS = [128]   # e.g. [256, 128]

# ─── MODEL ────────────────────────────────────────────────────────────────────
class SNNMulti(nn.Module):
    """
    Multi-layer SNN: [Linear -> LIF]*L  then Linear(out)
    States are owned by LIF (init_hidden=True); call reset() per segment batch.
    """
    def __init__(self, in_channels: int, hidden_dims, output_dim: int, beta: float):
        super().__init__()
        hidden_dims = list(hidden_dims)
        assert len(hidden_dims) >= 1, "DIMS must contain at least one hidden layer."

        # Build hidden stack
        self.fcs  = nn.ModuleList()
        self.lifs = nn.ModuleList()
        prev = in_channels
        for h in hidden_dims:
            fc = nn.Linear(prev, h)
            init.kaiming_uniform_(fc.weight, a=0.0, mode="fan_in", nonlinearity="relu")
            init.zeros_(fc.bias)
            self.fcs.append(fc)
            self.lifs.append(snn.Leaky(beta=beta, init_hidden=True))
            prev = h

        # Output (Linear only; logits per timestep)
        self.fc_out = nn.Linear(prev, output_dim)
        init.kaiming_uniform_(self.fc_out.weight, a=0.0, mode="fan_in", nonlinearity="linear")
        init.zeros_(self.fc_out.bias)

    @property
    def hidden_dims(self):
        return [fc.out_features for fc in self.fcs]

    def reset(self):
        for lif in self.lifs:
            utils.reset(lif)

    @torch.no_grad()
    def step_hidden(self, x_t):
        """
        One timestep through hidden stack.
        Returns:
          h_list: list of hidden activations per layer (after LIF), length L.
        """
        h_list = []
        h = x_t
        for fc, lif in zip(self.fcs, self.lifs):
            h = lif(fc(h))
            h_list.append(h)
        return h_list

# ─── TRAIN / EVAL LOGIC ───────────────────────────────────────────────────────
@torch.no_grad()
def train_epoch(model: SNNMulti,
                train_loader,
                projection: torch.Tensor,
                n_classes: int,
                mode: str) -> float:
    """One training epoch with PEPITA-like two-pass update on SEGMENTS (multi-layer)."""
    model.train()
    running_acc = 0.0

    L = len(model.fcs)
    Hs = model.hidden_dims

    for X, y in train_loader:
        # Flatten to segments
        X = X.to(DEVICE)  # [B,S,T,D]
        y = y.to(DEVICE)  # [B]
        X_segs, y_segs, _, _ = flatten_segments(X, y)        # [Nseg,T,D], [Nseg]
        if X_segs.numel() == 0:
            continue

        N, T, C = X_segs.shape
        # ───────────── First pass ─────────────
        model.reset()

        if mode == "accum":
            # time-averaged per-layer activities
            h_sum_layers = [torch.zeros((N, H), device=DEVICE) for H in Hs]
            rates = torch.zeros((N, n_classes), device=DEVICE)

            for t in range(T):
                x_t = X_segs[:, t, :]               # [N, C]
                h_list = model.step_hidden(x_t)     # list of [N, H_l]
                for l, h_t in enumerate(h_list):
                    h_sum_layers[l] += h_t
                rates += model.fc_out(h_list[-1])   # [N, K]

            h_avg_layers = [h_sum / T for h_sum in h_sum_layers]  # each [N, H_l]

        else:  # "original": keep full sequences
            h_rec_layers = [ [] for _ in range(L) ]  # each list of T tensors [N, H_l]
            o_rec = []
            for t in range(T):
                x_t = X_segs[:, t, :]
                h_list = model.step_hidden(x_t)
                for l, h_t in enumerate(h_list):
                    h_rec_layers[l].append(h_t)
                o_rec.append(model.fc_out(h_list[-1]))

            rates = torch.stack(o_rec, dim=0).sum(0)                # [N, K]
            # stack to [T, N, H_l]
            h_seq_layers = [torch.stack(seq, dim=0) for seq in h_rec_layers]
            h_avg_layers = [seq.mean(dim=0) for seq in h_seq_layers]  # [N, H_l]

        # Probabilities & error (segment labels, from first pass)
        p      = F.softmax(rates, dim=1)                       # [N, K]
        onehot = F.one_hot(y_segs, n_classes).float()          # [N, K]
        e      = p - onehot                                    # [N, K]

        # ───────────── Input modulation (project error to input space) ─────────────
        proj_err = e @ projection                              # [N, C]
        X_mod    = X_segs + proj_err.unsqueeze(1)              # [N, T, C]

        # ───────────── Second pass ─────────────
        model.reset()

        if mode == "accum":
            # time-averaged per-layer activities on modulated input
            h_mod_sum_layers = [torch.zeros((N, H), device=DEVICE) for H in Hs]
            x_mod_sum = torch.zeros((N, C), device=DEVICE)

            for t in range(T):
                x_mod_t = X_mod[:, t, :]
                h_mod_list = model.step_hidden(x_mod_t)
                for l, h_mod_t in enumerate(h_mod_list):
                    h_mod_sum_layers[l] += h_mod_t
                x_mod_sum += x_mod_t

            h_mod_avg_layers = [s / T for s in h_mod_sum_layers]     # [N, H_l]
            x_mod_avg        = x_mod_sum / T                         # [N, C]

            # --- ΔW for each hidden layer (outer product of means) ---
            # Layer 0: pre = x_mod_avg, post = h^0
            diff0 = (h_avg_layers[0] - h_mod_avg_layers[0])          # [N, H0]
            delta_w1 = - (diff0.t() @ x_mod_avg) / N                 # [H0, C]
            model.fcs[0].weight.add_(lr * delta_w1)

            # Layers 1..L-1: pre = h_mod_avg^{l-1}, post = h^{l}
            for l in range(1, L):
                pre  = h_mod_avg_layers[l-1]                         # [N, H_{l-1}]
                diff = (h_avg_layers[l] - h_mod_avg_layers[l])       # [N, H_l]
                delta_w = - (diff.t() @ pre) / N                     # [H_l, H_{l-1}]
                model.fcs[l].weight.add_(lr * delta_w)

        else:
            # original: mean over time of outer products
            # Collect second-pass per-layer sequences
            h_mod_rec_layers = [ [] for _ in range(L) ]  # lists of [N, H_l]
            for t in range(T):
                x_mod_t = X_mod[:, t, :]
                h_mod_list = model.step_hidden(x_mod_t)
                for l, h_mod_t in enumerate(h_mod_list):
                    h_mod_rec_layers[l].append(h_mod_t)

            h_mod_seq_layers = [torch.stack(seq, dim=0) for seq in h_mod_rec_layers]  # [T, N, H_l]

            # --- ΔW for each hidden layer (mean of outer products over time) ---
            # Layer 0: pre = X_mod[:, t, :], post diff = (h - h_mod)
            diff0 = (h_seq_layers[0] - h_mod_seq_layers[0])          # [T, N, H0]
            x_mod_T = X_mod.permute(1, 0, 2)                         # [T, N, C]
            # (T,N,H0,C) via broadcasting, then average over (T,N)
            mult0 = diff0.unsqueeze(3) * x_mod_T.unsqueeze(2)        # [T,N,H0,C]
            delta_w1 = - mult0.sum(dim=(0,1)) / (N * T)              # [H0, C]
            model.fcs[0].weight.add_(lr * delta_w1)

            # Layers 1..L-1: pre = h_mod^{l-1}_t, post diff = (h^l_t - h_mod^l_t)
            for l in range(1, L):
                pre_l  = h_mod_seq_layers[l-1]                       # [T, N, H_{l-1}]
                diff_l = (h_seq_layers[l] - h_mod_seq_layers[l])     # [T, N, H_l]
                mult_l = diff_l.unsqueeze(3) * pre_l.unsqueeze(2)    # [T, N, H_l, H_{l-1}]
                delta_w = - mult_l.sum(dim=(0,1)) / (N * T)          # [H_l, H_{l-1}]
                model.fcs[l].weight.add_(lr * delta_w)

            # also compute second-pass averages for W_out update
            h_mod_avg_layers = [seq.mean(dim=0) for seq in h_mod_seq_layers]  # [N, H_l]

        # --- ΔW_out uses second-pass last-layer average ---
        h_last_mod_avg = h_mod_avg_layers[-1]                        # [N, H_{L-1}]
        delta_w_out = - (e.t() @ h_last_mod_avg) / N                 # [K, H_{L-1}]
        model.fc_out.weight.add_(lr * delta_w_out)

        # batch segment-accuracy (from first pass predictions)
        preds = p.argmax(dim=1)
        running_acc += (preds == y_segs).float().mean().item()

    return 100.0 * running_acc / max(1, len(train_loader))


@torch.no_grad()
def evaluate_with_voting(model: SNNMulti, loader, n_classes: int) -> float:
    """Sum logits over time per segment, majority vote to per-sample."""
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
            h_list = model.step_hidden(x_t)
            out_rec.append(model.fc_out(h_list[-1]))
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
    print(f"[Arch]  DIMS = {DIMS}")

    # Model
    model = SNNMulti(in_channels=in_channels, hidden_dims=DIMS, output_dim=n_classes, beta=beta).to(DEVICE)

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
