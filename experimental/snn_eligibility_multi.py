#!/usr/bin/env python3
# E-PROP with a simple recurrent SNN (PyTorch, no autograd through time)
# Uses dynamic dataset loading via datasets.get_dataloaders

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- NEW: dynamic loaders ----
from datasets import get_dataloaders

# ---------------------------- Config ----------------------------
# Dataset config
DATASET_NAME = "har"           # ← "har", "wisdm", or "speech_commands"
DATA_ROOT    = "../data"
BATCH_SIZE   = 128
NUM_WORKERS  = 2
WINDOW_LEN   = 128             # used for WISDM/HAR, ignored for Speech Commands
MAX_SAMPLES  = None            # e.g., 5000 for quick runs (SC caps to 10k by default)

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
        series: [B, T, C]
        Returns:
          z_seq: [T, B, H]  (hidden spikes)
          u_seq: [T, B, H]  (pre-threshold voltage)
        """
        B, T, C = series.shape
        assert C == self.C

        v = torch.zeros(B, self.H, device=series.device)  # membrane
        z = torch.zeros(B, self.H, device=series.device)  # previous spikes

        zs, us = [], []
        for t in range(T):
            x_t = series[:, t, :]                          # [B, C]
            rec_term = (z @ self.W_rec) if self.rec_enabled else 0.0
            v = self.alpha * v + x_t @ self.W_in + rec_term
            u = v - self.v_th
            spk = (u > 0).float()
            v = v - self.v_th * spk

            zs.append(spk)
            us.append(u)
            z = spk

        z_seq = torch.stack(zs, dim=0)   # [T, B, H]
        u_seq = torch.stack(us, dim=0)   # [T, B, H]
        return z_seq, u_seq

    @torch.no_grad()
    def logits_from_spikes(self, z_seq):
        z_sum = z_seq.sum(dim=0)          # [B, H]
        logits = z_sum @ self.W_out + self.b_out  # [B, K]
        return logits, z_sum

# --------------------------- Training ---------------------------
@torch.no_grad()
def train_epoch_eprop(model, loader):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    n_batches, total_time = 0, 0

    for series, labels in loader:
        series = series.to(device)  # [B,T,C]
        labels = labels.to(device)
        B, T, C = series.shape
        H, K = model.H, model.K

        # States
        v = torch.zeros(B, H, device=device)
        z = torch.zeros(B, H, device=device)
        r_sum = torch.zeros(B, H, device=device)  # cumulative spikes for readout

        # Eligibility traces (online)
        e_in  = torch.zeros(B, C, H, device=device)                 # for W_in
        e_rec = torch.zeros(B, H, H, device=device) if model.rec_enabled else None

        # Gradient accumulators
        dW_in  = torch.zeros_like(model.W_in)
        dW_rec = torch.zeros_like(model.W_rec) if model.rec_enabled else None
        dW_out = torch.zeros_like(model.W_out)
        db_out = torch.zeros_like(model.b_out)

        last_logits = None

        for t in range(T):
            x_t = series[:, t, :]
            # LIF step
            rec_term = (z @ model.W_rec) if model.rec_enabled else 0.0
            v = model.alpha * v + x_t @ model.W_in + rec_term
            u = v - model.v_th
            spk = (u > 0).float()
            v = v - model.v_th * spk
            psi = model._surrogate_fast_sigmoid(u, model.slope)  # [B,H]

            # Eligibility
            e_in  = model.alpha * e_in  + x_t.unsqueeze(2) * psi.unsqueeze(1)  # [B,C,H]
            if model.rec_enabled:
                e_rec = model.alpha * e_rec + z.unsqueeze(2) * psi.unsqueeze(1)  # [B,H,H]

            # Readout from cumulative spikes (sum-over-time)
            r_sum = r_sum + spk
            logits_t = r_sum @ model.W_out + model.b_out   # [B,K]
            last_logits = logits_t

            # Learning signal (symmetric feedback)
            probs_t = torch.softmax(logits_t, dim=1)
            grad_logits_t = probs_t - F.one_hot(labels, num_classes=K).float()  # [B,K]
            L_t = grad_logits_t @ model.W_out.T                                  # [B,H]

            # Accumulate synaptic gradients
            dW_in  += torch.einsum('bch,bh->ch', e_in,  L_t)          # [C,H]
            if model.rec_enabled:
                dW_rec += torch.einsum('bij,bj->ij', e_rec, L_t)      # [H,H]
            dW_out += r_sum.T @ grad_logits_t                         # [H,K]
            db_out += grad_logits_t.sum(dim=0)                        # [K]

            # logging loss per timestep
            total_loss += F.cross_entropy(logits_t, labels, reduction="mean").item()

            z = spk  # next

        # SGD step (normalize by current batch size)
        model.W_in  -= (lr_in  / B) * dW_in
        if model.rec_enabled:
            model.W_rec -= (lr_rec / B) * dW_rec
        model.W_out -= (lr_out / B) * dW_out
        model.b_out -= (lr_out / B) * db_out
        model.clamp_rec()  # no-op if recurrence disabled

        # Metrics from final step
        preds = last_logits.argmax(dim=1)
        total_acc += (preds == labels).float().mean().item()
        n_batches += 1
        total_time += T

    # Average CE over all time steps across all batches
    avg_loss = total_loss / max(1, total_time)
    avg_acc  = 100.0 * total_acc / max(1, n_batches)
    return avg_loss, avg_acc

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_acc, n_batches = 0.0, 0
    for series, labels in loader:
        series = series.to(device); labels = labels.to(device)
        z_seq, _ = model.forward_unrolled(series)
        logits, _ = model.logits_from_spikes(z_seq)
        preds = logits.argmax(dim=1)
        total_acc += (preds == labels).float().mean().item()
        n_batches += 1
    return 100.0 * total_acc / max(1, n_batches)

# ------------------------------ Main -----------------------------
if __name__ == "__main__":
    set_seed()

    # ---- NEW: dynamic dataset loading ----
    train_loader, test_loader, meta = get_dataloaders(
        DATASET_NAME,
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        window_len=WINDOW_LEN,
        num_workers=NUM_WORKERS,
        max_samples=MAX_SAMPLES
    )
    n_classes  = meta["n_classes"]
    input_dim  = meta["input_dim"]
    time_steps = meta["time_steps"]       # not required by training; sequences can vary
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

    # Train
    print(f"[Data] {DATASET_NAME.upper()} | classes={n_classes} | input_dim={input_dim} | ~T={time_steps}")
    print(f"[Classes] {class_names}")
    print(f"Device: {device}; Hidden: {hidden_size}; alpha={alpha}; slope={slope}; recurrence={use_recurrence}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch_eprop(model, train_loader)
        test_acc = eval_epoch(model, test_loader)
        print(f"Epoch {epoch:02d}/{num_epochs}  "
              f"Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")
