#!/usr/bin/env python3
# Multi-layer E-PROP SNN with one-step backprop spatial credit (no random feedback)
# Uses segmented dataset loading via datasets.get_dataloaders
# Batches are shaped [B, S, T, D]; we flatten to segments [Nseg, T, D].

import math
import numpy as np
import torch
import torch.nn.functional as F

# ---- Dynamic loaders + helpers (from datasets.py) ----
from datasets import get_dataloaders, flatten_segments, majority_vote

# ---------------------------- Config ----------------------------
# Dataset
DATASET_NAME  = "har"
DATA_ROOT     = "../data"
BATCH_SIZE    = 128
SAMPLE_LENGTH = None

# Training
seed        = 11
num_epochs  = 10
lr_in       = 5e-4         # input->h1 (and generally all feedforward layers)
lr_rec      = 5e-4         # recurrent layers (if enabled)
lr_out      = 1e-3         # readout
weight_clip = 1.5          # clip recurrent weights for stability
drop_diag   = True         # zero W_rec diagonal

# Neuron / e-prop
alpha       = 0.9          # membrane leak
v_th        = 1.0          # threshold
slope       = 25.0         # surrogate sharpness (fast-sigmoid)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture
DIMS = [128, 128]          # e.g., input_dim -> 512 -> 256 -> n_classes
use_recurrence = False     # per-hidden-layer recurrence on/off

# ----------------------- Utilities -----------------------
def set_seed(s=seed):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ----------------------- E-prop SNN -----------------------
class EpropSNN(torch.nn.Module):
    """
    Multi-layer LIF SNN with optional per-layer recurrence and manual e-prop updates.
    - Hidden layers: spikes z^l_t in {0,1}^{H_l}
    - Readout: logits = (sum_t z^{L-1}_t) @ W_out + b_out
    - Spatial credit: one-step backprop via actual forward weights (no random feedback).
    - All weights updated manually (requires_grad=False).
    """
    def __init__(self, input_dim, hidden_dims, n_classes,
                 alpha=0.9, v_th=1.0, slope=25.0, rec_enabled=False):
        super().__init__()
        self.C = input_dim
        self.hidden_dims = list(hidden_dims)
        self.L = len(self.hidden_dims)             # number of hidden layers
        assert self.L >= 1, "Provide at least one hidden layer in DIMS."
        self.K = n_classes

        self.alpha = alpha
        self.v_th  = v_th
        self.slope = slope
        self.rec_enabled = rec_enabled

        # ---------- Parameter initialization ----------
        # Feedforward dims: [C, H0, H1, ..., H_{L-1}]
        in_dims  = [self.C] + self.hidden_dims[:-1]
        out_dims = self.hidden_dims

        # Feedforward weights: W_ff[i] maps in_dims[i] -> out_dims[i]
        self.W_ff = torch.nn.ParameterList()
        for Cin, Cout in zip(in_dims, out_dims):
            k = math.sqrt(1.0 / Cin)
            W = torch.empty(Cin, Cout).uniform_(-k, k)
            self.W_ff.append(torch.nn.Parameter(W, requires_grad=False))

        # Optional recurrent weights per layer
        self.W_rec = torch.nn.ParameterList()
        for H in self.hidden_dims:
            k = math.sqrt(1.0 / max(1, H))
            W = torch.empty(H, H).uniform_(-k, k)
            if not rec_enabled:
                W.zero_()
            self.W_rec.append(torch.nn.Parameter(W, requires_grad=False))

        # Readout
        k_out = math.sqrt(1.0 / max(1, self.hidden_dims[-1]))
        self.W_out = torch.nn.Parameter(
            torch.empty(self.hidden_dims[-1], self.K).uniform_(-k_out, k_out),
            requires_grad=False
        )
        self.b_out = torch.nn.Parameter(torch.zeros(self.K), requires_grad=False)

    @torch.no_grad()
    def clamp_rec(self):
        if not self.rec_enabled:
            return
        for W in self.W_rec:
            if drop_diag:
                W.fill_diagonal_(0.0)
            if weight_clip is not None:
                W.clamp_(-weight_clip, weight_clip)

    # --- surrogate derivative for spike function ---
    @staticmethod
    def _surrogate_fast_sigmoid(u, slope):
        sig = torch.sigmoid(slope * u)
        return slope * sig * (1.0 - sig)

    # --- Unrolled forward to get hidden spikes (for eval) ---
    @torch.no_grad()
    def forward_unrolled(self, series):
        """
        series: [N, T, C]
        Returns per-layer spike sequences:
          z_seq_list: list over layers l of [T, N, H_l]
        """
        N, T, C = series.shape
        assert C == self.C

        # States per layer
        v = [torch.zeros(N, H, device=series.device) for H in self.hidden_dims]
        z = [torch.zeros(N, H, device=series.device) for H in self.hidden_dims]

        zs_per_layer = [ [] for _ in range(self.L) ]

        for t in range(T):
            x = series[:, t, :]  # [N, C]
            # Layer 0
            rec0 = (z[0] @ self.W_rec[0]) if self.rec_enabled else 0.0
            v[0] = self.alpha * v[0] + x @ self.W_ff[0] + rec0
            u0 = v[0] - self.v_th
            spk0 = (u0 > 0).float()
            v[0] = v[0] - self.v_th * spk0
            zs_per_layer[0].append(spk0)
            z[0] = spk0

            # Deeper layers
            for l in range(1, self.L):
                h_in = z[l-1]
                rec = (z[l] @ self.W_rec[l]) if self.rec_enabled else 0.0
                v[l] = self.alpha * v[l] + h_in @ self.W_ff[l] + rec
                ul = v[l] - self.v_th
                spkl = (ul > 0).float()
                v[l] = v[l] - self.v_th * spkl
                zs_per_layer[l].append(spkl)
                z[l] = spkl

        # Stack time
        z_seq_list = [ torch.stack(zs_per_layer[l], dim=0) for l in range(self.L) ]
        return z_seq_list  # each: [T, N, H_l]

    @torch.no_grad()
    def logits_from_spikes(self, z_last_seq):
        """
        z_last_seq: [T, N, H_{L-1}] -> logits: [N, K]
        """
        z_sum = z_last_seq.sum(dim=0)             # [N, H_{L-1}]
        logits = z_sum @ self.W_out + self.b_out  # [N, K]
        return logits, z_sum


# --------------------------- Training on SEGMENTS ---------------------------
@torch.no_grad()
def train_epoch_eprop(model, loader):
    """
    Online e-prop over segments with one-step backprop spatial credit (symmetric).
    - For each layer i, eligibility traces accumulate: e_ff[i] and (optional) e_rec[i].
    - Learning signals:
        grad_logits_t = softmax(logits_t) - onehot(y)
        L_sig[L-1] = grad_logits_t @ W_out^T
        L_sig[i]   = L_sig[i+1] @ W_ff[i+1]^T   (for i = L-2 .. 0)
      (No random feedback matrices.)
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0
    n_batches, total_time = 0, 0

    for X, y in loader:
        X = X.to(device)  # [B,S,T,D]
        y = y.to(device)  # [B]
        X_segs, y_segs, _, _ = flatten_segments(X, y)  # [Nseg,T,D], [Nseg]
        if X_segs.numel() == 0:
            continue

        N, T, C = X_segs.shape
        assert C == model.C
        Hs = model.hidden_dims
        L  = model.L
        K  = model.K

        # ---------- States ----------
        v = [torch.zeros(N, H, device=device) for H in Hs]
        z = [torch.zeros(N, H, device=device) for H in Hs]
        r_sum = torch.zeros(N, Hs[-1], device=device)  # cumulative spikes for readout

        # ---------- Eligibility traces ----------
        # Feedforward: e_ff[i] shape [N, in_i, H_i]
        in_dims = [model.C] + Hs[:-1]
        e_ff = [torch.zeros(N, in_dims[i], Hs[i], device=device) for i in range(L)]

        # Recurrent: e_rec[i] shape [N, H_i, H_i]
        e_rec = [torch.zeros(N, Hs[i], Hs[i], device=device) if model.rec_enabled else None
                 for i in range(L)]

        # ---------- Gradient accumulators ----------
        dW_ff  = [torch.zeros_like(model.W_ff[i]) for i in range(L)]
        dW_rec = [torch.zeros_like(model.W_rec[i]) if model.rec_enabled else None for i in range(L)]
        dW_out = torch.zeros_like(model.W_out)
        db_out = torch.zeros_like(model.b_out)

        last_logits = None

        for t in range(T):
            x_t = X_segs[:, t, :]   # [N, C]

            # ----- Forward step (layer 0) -----
            rec0 = (z[0] @ model.W_rec[0]) if model.rec_enabled else 0.0
            v[0] = model.alpha * v[0] + x_t @ model.W_ff[0] + rec0
            u0 = v[0] - model.v_th
            spk0 = (u0 > 0).float()
            v[0] = v[0] - model.v_th * spk0
            psi0 = model._surrogate_fast_sigmoid(u0, model.slope)
            z[0] = spk0

            # ----- Forward deeper -----
            psis = [psi0]  # store all psi^l
            for l in range(1, L):
                h_in = z[l-1]
                rec = (z[l] @ model.W_rec[l]) if model.rec_enabled else 0.0
                v[l] = model.alpha * v[l] + h_in @ model.W_ff[l] + rec
                ul = v[l] - model.v_th
                spkl = (ul > 0).float()
                v[l] = v[l] - model.v_th * spkl
                psil = model._surrogate_fast_sigmoid(ul, model.slope)
                z[l] = spkl
                psis.append(psil)

            # ----- Update eligibilities (feedforward & recurrent) -----
            # Layer 0: pre = x_t
            e_ff[0] = model.alpha * e_ff[0] + x_t.unsqueeze(2) * psis[0].unsqueeze(1)  # [N,C,H0]
            if model.rec_enabled:
                e_rec[0] = model.alpha * e_rec[0] + z[0].unsqueeze(2) * psis[0].unsqueeze(1)  # [N,H0,H0]

            # Layers 1..L-1: pre = z[l-1]
            for l in range(1, L):
                e_ff[l] = model.alpha * e_ff[l] + z[l-1].unsqueeze(2) * psis[l].unsqueeze(1)  # [N,H_{l-1},H_l]
                if model.rec_enabled:
                    e_rec[l] = model.alpha * e_rec[l] + z[l].unsqueeze(2) * psis[l].unsqueeze(1)  # [N,H_l,H_l]

            # ----- Readout from cumulative spikes -----
            r_sum = r_sum + z[-1]                                   # [N, H_{L-1}]
            logits_t = r_sum @ model.W_out + model.b_out            # [N, K]
            last_logits = logits_t

            # ----- Learning signals (one-step backprop, symmetric) -----
            probs_t = torch.softmax(logits_t, dim=1)                # [N, K]
            grad_logits_t = probs_t - F.one_hot(y_segs, num_classes=K).float()  # [N, K]

            # Top hidden layer learning signal
            L_sig = [None for _ in range(L)]
            L_sig[L-1] = grad_logits_t @ model.W_out.T              # [N, H_{L-1}]

            # Propagate one layer at a time using REAL forward weights (transpose)
            for l in range(L-2, -1, -1):
                L_sig[l] = L_sig[l+1] @ model.W_ff[l+1].T           # [N, H_l]

            # ----- Accumulate synaptic grads -----
            # Feedforward layers
            for l in range(L):
                dW_ff[l] += torch.einsum('nih,nh->ih', e_ff[l], L_sig[l])  # [in_l, H_l]

            # Recurrent layers
            if model.rec_enabled:
                for l in range(L):
                    dW_rec[l] += torch.einsum('nij,nj->ij', e_rec[l], L_sig[l])  # [H_l, H_l]

            # Readout
            dW_out += r_sum.T @ grad_logits_t                        # [H_{L-1}, K]
            db_out += grad_logits_t.sum(dim=0)                       # [K]

            # CE for diagnostics
            total_loss += F.cross_entropy(logits_t, y_segs, reduction="mean").item()

        # ----- SGD step (normalize by #segments) -----
        norm = max(1, N)
        for l in range(L):
            model.W_ff[l] -= (lr_in  / norm) * dW_ff[l]
        if model.rec_enabled:
            for l in range(L):
                model.W_rec[l] -= (lr_rec / norm) * dW_rec[l]
        model.W_out -= (lr_out / norm) * dW_out
        model.b_out -= (lr_out / norm) * db_out
        model.clamp_rec()  # (no-op if recurrence disabled)

        # Segment-level accuracy from final step logits
        preds = last_logits.argmax(dim=1)
        total_acc += (preds == y_segs).float().mean().item()
        n_batches += 1
        total_time += T

    avg_loss = total_loss / max(1, total_time)          # CE averaged per time step
    avg_acc  = 100.0 * total_acc / max(1, n_batches)    # segment-level accuracy
    return avg_loss, avg_acc


@torch.no_grad()
def eval_epoch_with_voting(model, loader, n_classes: int):
    """
    Evaluate with majority vote over segments per sample:
      - Forward unrolled to per-segment logits
      - Vote back to per-sample prediction
    """
    model.eval()
    total_acc, n_batches = 0.0, 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        B = y.size(0)

        X_segs, _, sample_ids, _ = flatten_segments(X, y)  # [Nseg,T,D], [Nseg], [Nseg]
        if X_segs.numel() == 0:
            continue

        z_seq_list = model.forward_unrolled(X_segs)        # list of [T, Nseg, H_l]
        logits, _   = model.logits_from_spikes(z_seq_list[-1])  # [Nseg, K]
        preds_seg   = logits.argmax(dim=1)

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
        sample_length=SAMPLE_LENGTH,
    )
    n_classes  = meta["n_classes"]
    input_dim  = meta["input_dim"]
    time_steps = meta["time_steps"]
    class_names = meta["class_names"]

    # Build model
    model = EpropSNN(
        input_dim=input_dim,
        hidden_dims=DIMS,
        n_classes=n_classes,
        alpha=alpha, v_th=v_th, slope=slope,
        rec_enabled=use_recurrence
    ).to(device)
    model.clamp_rec()

    print(f"[Data] {DATASET_NAME.upper()} | classes={n_classes} | D={input_dim} | segment_T≈{time_steps}")
    print(f"[Classes] {class_names}")
    print(f"Device: {device}; DIMS={DIMS}; alpha={alpha}; slope={slope}; recurrence={use_recurrence}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc_seg = train_epoch_eprop(model, train_loader)
        test_acc = eval_epoch_with_voting(model, test_loader, n_classes)
        print(f"Epoch {epoch:02d}/{num_epochs}  "
              f"Loss/step: {train_loss:.4f}  Train Seg Acc: {train_acc_seg:.2f}%  "
              f"Test (vote): {test_acc:.2f}%")
