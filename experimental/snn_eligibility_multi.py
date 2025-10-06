#!/usr/bin/env python3
# E-PROP with a simple recurrent SNN (PyTorch, no autograd through time)
# Uses segmented dataset loading via datasets.get_dataloaders
# Batches are shaped [B, S, T, D]; we flatten to segments [Nseg, T, D].

import math
import numpy as np
import torch
import torch.nn.functional as F

# ---- Dynamic loaders + helpers (from datasets.py) ----
from datasets import get_dataloaders, flatten_segments, majority_vote

# ---------------------------- Config ----------------------------
# Dataset config
DATASET_NAME  = "har"
DATA_ROOT     = "../data"
BATCH_SIZE    = 128
SAMPLE_LENGTH = None

# Training
seed        = 11
num_epochs  = 10
hidden_size = 128              # try 128..512
lr_in       = 5e-4
lr_rec      = 5e-4
lr_out      = 1e-3
weight_clip = 1.5              # clip recurrent weights for stability
drop_diag   = True             # zero W_rec diagonal

# Neuron / e-prop
alpha       = 0.9              # membrane leak (like beta)
v_th        = 1.0              # threshold
slope       = 25.0             # surrogate sharpness (fast-sigmoid)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_recurrence = False

# ----------------------- Utilities -----------------------
def set_seed(s=seed):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ----------------------- E-prop SNN -----------------------
class EpropSNN(torch.nn.Module):
    """
    One-layer LIF SNN with optional recurrence and manual e-prop updates.
    - Input:  x_t in R^C
    - Hidden: spikes z_t in {0,1}^H (with or without recurrent weights)
    - Readout: logits = sum_t z_t @ W_out + b_out  (no output spikes)

    Note: All weights are updated with manual e-prop; requires_grad=False.
    """
    def __init__(self, input_dim, hidden_dim, n_classes, alpha=0.9, v_th=1.0, slope=25.0,
                 rec_enabled=True):
        super().__init__()
        self.C, self.H, self.K = input_dim, hidden_dim, n_classes
        self.alpha = alpha
        self.v_th  = v_th
        self.slope = slope
        self.rec_enabled = rec_enabled

        # Xavier-like init
        k_in  = math.sqrt(1.0 / self.C)
        k_rec = math.sqrt(1.0 / max(1, self.H))
        k_out = math.sqrt(1.0 / max(1, self.H))

        # Parameters updated manually (requires_grad=False)
        self.W_in  = torch.nn.Parameter(torch.empty(self.C, self.H).uniform_(-k_in,  k_in),  requires_grad=False)
        self.W_rec = torch.nn.Parameter(torch.empty(self.H, self.H).uniform_(-k_rec, k_rec), requires_grad=False)
        self.W_out = torch.nn.Parameter(torch.empty(self.H, self.K).uniform_(-k_out, k_out), requires_grad=False)
        self.b_out = torch.nn.Parameter(torch.zeros(self.K), requires_grad=False)

        if not self.rec_enabled:
            with torch.no_grad():
                self.W_rec.zero_()

    @torch.no_grad()
    def clamp_rec(self):
        if not self.rec_enabled:
            return
        if drop_diag:
            self.W_rec.fill_diagonal_(0.0)
        if weight_clip is not None:
            self.W_rec.clamp_(-weight_clip, weight_clip)

    @staticmethod
    def _surrogate_fast_sigmoid(u, slope):
        sig = torch.sigmoid(slope * u)
        return slope * sig * (1.0 - sig)

    @torch.no_grad()
    def forward_unrolled(self, series):
        """
        series: [N, T, C]  (N can be batch or number of segments)
        Returns:
          z_seq: [T, N, H]  (hidden spikes)
          u_seq: [T, N, H]  (pre-threshold voltage)
        """
        N, T, C = series.shape
        assert C == self.C

        v = torch.zeros(N, self.H, device=series.device)  # membrane
        z = torch.zeros(N, self.H, device=series.device)  # previous spikes

        zs, us = [], []
        for t in range(T):
            x_t = series[:, t, :]                         # [N, C]
            rec_term = (z @ self.W_rec) if self.rec_enabled else 0.0
            v = self.alpha * v + x_t @ self.W_in + rec_term
            u = v - self.v_th
            spk = (u > 0).float()
            v = v - self.v_th * spk

            zs.append(spk)
            us.append(u)
            z = spk

        z_seq = torch.stack(zs, dim=0)   # [T, N, H]
        u_seq = torch.stack(us, dim=0)   # [T, N, H]
        return z_seq, u_seq

    @torch.no_grad()
    def logits_from_spikes(self, z_seq):
        """
        z_seq: [T, N, H] -> logits: [N, K]
        """
        z_sum = z_seq.sum(dim=0)          # [N, H]
        logits = z_sum @ self.W_out + self.b_out  # [N, K]
        return logits, z_sum


# --------------------------- Training on SEGMENTS ---------------------------
@torch.no_grad()
def train_epoch_eprop(model, loader):
    """
    Train on segments:
      - Loader yields X:[B,S,T,D], y:[B]
      - Flatten into X_segs:[Nseg,T,D], y_segs:[Nseg]
      - E-prop updates computed online across time on segment batches
    Returns avg loss per time step and segment-level accuracy.
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0
    n_batches, total_time = 0, 0

    for X, y in loader:
        # Flatten to segments
        X = X.to(device)  # [B,S,T,D]
        y = y.to(device)  # [B]
        X_segs, y_segs, _, _ = flatten_segments(X, y)        # [Nseg,T,D], [Nseg]
        if X_segs.numel() == 0:
            continue

        N, T, C = X_segs.shape
        H, K = model.H, model.K

        # States per segment batch
        v = torch.zeros(N, H, device=device)
        z = torch.zeros(N, H, device=device)
        r_sum = torch.zeros(N, H, device=device)  # cumulative spikes for readout

        # Eligibility traces (online)
        e_in  = torch.zeros(N, C, H, device=device)                 # for W_in
        e_rec = torch.zeros(N, H, H, device=device) if model.rec_enabled else None

        # Gradient accumulators
        dW_in  = torch.zeros_like(model.W_in)
        dW_rec = torch.zeros_like(model.W_rec) if model.rec_enabled else None
        dW_out = torch.zeros_like(model.W_out)
        db_out = torch.zeros_like(model.b_out)

        last_logits = None

        for t in range(T):
            x_t = X_segs[:, t, :]                           # [N, C]

            # LIF update
            rec_term = (z @ model.W_rec) if model.rec_enabled else 0.0
            v = model.alpha * v + x_t @ model.W_in + rec_term
            u = v - model.v_th
            spk = (u > 0).float()
            v = v - model.v_th * spk
            psi = model._surrogate_fast_sigmoid(u, model.slope)  # [N,H]

            # Eligibility traces
            e_in  = model.alpha * e_in  + x_t.unsqueeze(2) * psi.unsqueeze(1)   # [N,C,H]
            if model.rec_enabled:
                e_rec = model.alpha * e_rec + z.unsqueeze(2) * psi.unsqueeze(1) # [N,H,H]

            # Readout from cumulative spikes (sum-over-time)
            r_sum = r_sum + spk
            logits_t = r_sum @ model.W_out + model.b_out   # [N,K]
            last_logits = logits_t

            # Learning signal (symmetric feedback)
            probs_t = torch.softmax(logits_t, dim=1)
            grad_logits_t = probs_t - F.one_hot(y_segs, num_classes=K).float()  # [N,K]
            L_t = grad_logits_t @ model.W_out.T                                  # [N,H]

            # Accumulate synaptic gradients
            dW_in  += torch.einsum('nch,nh->ch', e_in,  L_t)          # [C,H]
            if model.rec_enabled:
                dW_rec += torch.einsum('nij,nj->ij', e_rec, L_t)      # [H,H]
            dW_out += r_sum.T @ grad_logits_t                         # [H,K]
            db_out += grad_logits_t.sum(dim=0)                        # [K]

            # log CE for diagnostics
            total_loss += F.cross_entropy(logits_t, y_segs, reduction="mean").item()

            z = spk  # next

        # SGD step (normalize by #segments)
        norm = max(1, N)
        model.W_in  -= (lr_in  / norm) * dW_in
        if model.rec_enabled:
            model.W_rec -= (lr_rec / norm) * dW_rec
        model.W_out -= (lr_out / norm) * dW_out
        model.b_out -= (lr_out / norm) * db_out
        model.clamp_rec()  # no-op if recurrence disabled

        # Segment-level accuracy from final step logits
        preds = last_logits.argmax(dim=1)
        total_acc += (preds == y_segs).float().mean().item()
        n_batches += 1
        total_time += T

    # Average CE over all time steps across all (segment) batches
    avg_loss = total_loss / max(1, total_time)
    avg_acc  = 100.0 * total_acc / max(1, n_batches)
    return avg_loss, avg_acc


@torch.no_grad()
def eval_epoch_with_voting(model, loader, n_classes: int):
    """
    Evaluate with majority vote over segments per sample:
      - Flatten to segments
      - Compute per-segment logits
      - Majority vote back to per-sample predictions
    """
    model.eval()
    total_acc, n_batches = 0.0, 0

    for X, y in loader:
        X = X.to(device)  # [B,S,T,D]
        y = y.to(device)  # [B]
        B = y.size(0)

        X_segs, _, sample_ids, _ = flatten_segments(X, y)  # [Nseg,T,D], [Nseg], [Nseg]
        if X_segs.numel() == 0:
            continue

        # Forward unrolled on segments
        z_seq, _ = model.forward_unrolled(X_segs)          # [T, Nseg, H]
        logits, _ = model.logits_from_spikes(z_seq)        # [Nseg, K]
        preds_seg = logits.argmax(dim=1)                   # [Nseg]

        # Majority vote back to samples
        preds_sample = majority_vote(preds_seg, sample_ids.to(device), n_classes, B)  # [B]
        total_acc += (preds_sample == y).float().mean().item()
        n_batches += 1

    return 100.0 * total_acc / max(1, n_batches)


# ------------------------------ Main -----------------------------
if __name__ == "__main__":
    set_seed()

    # ---- Segmented dataset loading ----
    train_loader, test_loader, meta = get_dataloaders(
        DATASET_NAME,
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        sample_length=SAMPLE_LENGTH,   # set to an int to cap/split sequences
    )
    n_classes  = meta["n_classes"]
    input_dim  = meta["input_dim"]
    time_steps = meta["time_steps"]       # segment length (for info)
    class_names = meta["class_names"]

    # Build model
    model = EpropSNN(
        input_dim=input_dim,
        hidden_dim=hidden_size,
        n_classes=n_classes,
        alpha=alpha, v_th=v_th, slope=slope,
        rec_enabled=use_recurrence
    ).to(device)
    model.clamp_rec()

    print(f"[Data] {DATASET_NAME.upper()} | classes={n_classes} | D={input_dim} | segment_T≈{time_steps}")
    print(f"[Classes] {class_names}")
    print(f"Device: {device}; Hidden: {hidden_size}; alpha={alpha}; slope={slope}; recurrence={use_recurrence}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc_seg = train_epoch_eprop(model, train_loader)
        test_acc = eval_epoch_with_voting(model, test_loader, n_classes)
        print(f"Epoch {epoch:02d}/{num_epochs}  "
              f"Loss/step: {train_loss:.4f}  Train Seg Acc: {train_acc_seg:.2f}%  "
              f"Test (vote): {test_acc:.2f}%")
