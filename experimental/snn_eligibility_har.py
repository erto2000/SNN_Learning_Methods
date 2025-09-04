#!/usr/bin/env python3
# E-PROP on UCI-HAR with a simple recurrent SNN (PyTorch, no autograd through time)

import os
import urllib.request
import zipfile
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------- Config ----------------------------
DATA_URL    = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_PATH    = "../data/UCI_HAR.zip"
DATA_DIR    = "../data/UCI_HAR_Dataset"

WINDOW_LEN  = 128
CLASS_NAMES = ["Walking","Walking Upstairs","Walking Downstairs","Sitting","Standing","Laying"]

# Training
seed        = 11
num_epochs  = 10
batch_size  = 128
hidden_size = 128            # try 128..512
lr_in       = 5e-4
lr_rec      = 5e-4
lr_out      = 1e-3
weight_clip = 1.5            # clip recurrent weights for stability
drop_diag   = True           # zero W_rec diagonal

# Neuron / e-prop
alpha       = 0.9            # membrane leak (like beta)
v_th        = 1.0            # threshold
slope       = 25.0           # surrogate sharpness (fast-sigmoid)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Utilities & Data -----------------------
def set_seed(s=seed):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def download_and_extract():
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print("Downloading UCI HAR Dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    if not os.path.exists(DATA_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(os.path.dirname(ZIP_PATH))
        os.rename(os.path.join(os.path.dirname(ZIP_PATH), "UCI HAR Dataset"), DATA_DIR)
    print(f"Dataset ready at {DATA_DIR}")

def load_split(split="train"):
    # Channels as provided by UCI HAR naming
    channels = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x","body_gyro_y","body_gyro_z",
        "total_acc_x","total_acc_y","total_acc_z"
    ]
    folder = os.path.join(DATA_DIR, split, "Inertial Signals")
    arrays = []
    for ch in channels:
        path = os.path.join(folder, f"{ch}_{split}.txt")
        arr  = np.loadtxt(path)             # [N, T]
        arrays.append(arr[..., np.newaxis]) # [N, T, 1]
    X = np.concatenate(arrays, axis=2)     # [N, T, C=9]
    y_path = os.path.join(DATA_DIR, split, f"y_{split}.txt")
    y      = np.loadtxt(y_path).astype(int) - 1
    return X, y

class HARTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()  # [N, T, C]
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ----------------------- E-prop SNN Module ----------------------
class EpropSNN(torch.nn.Module):
    """
    One-layer recurrent LIF SNN with manual e-prop updates.
    - Input:  x_t in R^C
    - Hidden: LIF spikes z_t in {0,1}^H with recurrent weights
    - Readout: logits = sum_t z_t @ W_out + b_out  (no output spikes)
    """
    def __init__(self, input_dim, hidden_dim, n_classes, alpha=0.9, v_th=1.0, slope=25.0):
        super().__init__()
        self.C, self.H, self.K = input_dim, hidden_dim, n_classes
        self.alpha = alpha
        self.v_th  = v_th
        self.slope = slope

        # Parameters we will update manually (requires_grad=False)
        # Xavier-like init
        k_in  = math.sqrt(1.0 / self.C)
        k_rec = math.sqrt(1.0 / self.H)
        k_out = math.sqrt(1.0 / self.H)

        self.W_in  = torch.nn.Parameter(torch.empty(self.C, self.H).uniform_(-k_in,  k_in),  requires_grad=False)
        self.W_rec = torch.nn.Parameter(torch.empty(self.H, self.H).uniform_(-k_rec, k_rec), requires_grad=False)
        self.W_out = torch.nn.Parameter(torch.empty(self.H, self.K).uniform_(-k_out, k_out), requires_grad=False)
        self.b_out = torch.nn.Parameter(torch.zeros(self.K), requires_grad=False)

    @torch.no_grad()
    def clamp_rec(self):
        if drop_diag:
            # avoid self-connections causing ping-pong
            self.W_rec.fill_diagonal_(0.0)
        if weight_clip is not None:
            self.W_rec.clamp_(-weight_clip, weight_clip)

    @staticmethod
    def _surrogate_fast_sigmoid(u, slope):
        # derivative of sigmoid(slope*u): slope * sigm * (1-sigm)
        sig = torch.sigmoid(slope * u)
        return slope * sig * (1.0 - sig)

    @torch.no_grad()
    def forward_unrolled(self, series):
        """
        series: [B, T, C]
        Returns:
          z_seq: [T, B, H]  (hidden spikes)
          u_seq: [T, B, H]  (pre-threshold voltage u = v - v_th, for surrogate)
        """
        B, T, C = series.shape
        assert C == self.C

        v = torch.zeros(B, self.H, device=series.device)  # membrane
        z = torch.zeros(B, self.H, device=series.device)  # previous spikes

        zs, us = [], []
        for t in range(T):
            x_t = series[:, t, :]                          # [B, C]
            # membrane update with soft reset
            v = self.alpha * v + x_t @ self.W_in + z @ self.W_rec
            u = v - self.v_th                              # distance to threshold
            spk = (u > 0).float()                          # hard spike
            v = v - self.v_th * spk                        # soft reset

            zs.append(spk)
            us.append(u)
            z = spk

        z_seq = torch.stack(zs, dim=0)   # [T, B, H]
        u_seq = torch.stack(us, dim=0)   # [T, B, H]
        return z_seq, u_seq

    @torch.no_grad()
    def logits_from_spikes(self, z_seq):
        """
        Integrate hidden spikes over time and compute logits.
        """
        z_sum = z_seq.sum(dim=0)          # [B, H]
        logits = z_sum @ self.W_out + self.b_out  # [B, K]
        return logits, z_sum

# --------------------------- Training ---------------------------
def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()

@torch.no_grad()
def train_epoch_eprop(model, loader):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for series, labels in loader:
        series = series.to(device)  # [B,T,C]
        labels = labels.to(device)
        B, T, C = series.shape
        H, K = model.H, model.K

        # States
        v = torch.zeros(B, H, device=device)
        z = torch.zeros(B, H, device=device)
        r_sum = torch.zeros(B, H, device=device)  # time-integrated spikes for readout

        # Eligibility traces (kept online)
        e_in  = torch.zeros(B, C, H, device=device)  # for W_in
        e_rec = torch.zeros(B, H, H, device=device)  # for W_rec

        # Gradient accumulators
        dW_in  = torch.zeros_like(model.W_in)
        dW_rec = torch.zeros_like(model.W_rec)
        dW_out = torch.zeros_like(model.W_out)
        db_out = torch.zeros_like(model.b_out)

        last_logits = None

        for t in range(T):
            x_t = series[:, t, :]                            # [B,C]
            # LIF step
            v = model.alpha * v + x_t @ model.W_in + z @ model.W_rec
            u = v - model.v_th
            spk = (u > 0).float()
            v = v - model.v_th * spk
            psi = model._surrogate_fast_sigmoid(u, model.slope)  # [B,H]

            # --- e-prop eligibility traces (simple LIF version) ---
            # Decayed traces + new local factors (no reset term for simplicity)
            e_in  = model.alpha * e_in  + x_t.unsqueeze(2) * psi.unsqueeze(1)  # [B,C,H]
            e_rec = model.alpha * e_rec + z.unsqueeze(2)   * psi.unsqueeze(1)  # [B,H,H]

            # Readout (use cumulative spikes like your sum-over-time logits)
            r_sum = r_sum + spk
            logits_t = r_sum @ model.W_out + model.b_out   # [B,K]
            last_logits = logits_t

            # Learning signal at this step (symmetric feedback)
            probs_t = torch.softmax(logits_t, dim=1)
            grad_logits_t = probs_t - F.one_hot(labels, num_classes=K).float()  # [B,K]
            L_t = grad_logits_t @ model.W_out.T                                  # [B,H]

            # Accumulate synaptic gradients via eligibilities
            dW_in  += torch.einsum('bch,bh->ch', e_in,  L_t)  # [C,H]
            dW_rec += torch.einsum('bij,bj->ij', e_rec, L_t)  # [H,H]
            dW_out += r_sum.T @ grad_logits_t                 # [H,K]
            db_out += grad_logits_t.sum(dim=0)                # [K]

            # next
            z = spk
            # (Optional) you can also step weights here every few steps instead of at sequence end.

            # For logging: average CE across time
            total_loss += F.cross_entropy(logits_t, labels, reduction="mean").item()

        # SGD step
        model.W_in  -= (lr_in  / B) * dW_in
        model.W_rec -= (lr_rec / B) * dW_rec
        model.W_out -= (lr_out / B) * dW_out
        model.b_out -= (lr_out / B) * db_out
        model.clamp_rec()

        # Metrics from final step
        preds = last_logits.argmax(dim=1)
        total_acc += (preds == labels).float().mean().item()
        n_batches += 1

    # Average loss per batch per time step
    return total_loss / max(1, n_batches * T), 100.0 * total_acc / max(1, n_batches)


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
    download_and_extract()

    # Load data
    X_train, y_train = load_split("train")
    X_test,  y_test  = load_split("test")

    # Datasets & loaders
    train_ds = HARTimeSeriesDataset(X_train, y_train)
    test_ds  = HARTimeSeriesDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Build model
    input_dim  = X_train.shape[2]     # 9 channels
    time_steps = X_train.shape[1]     # 128 (not directly needed here)
    n_classes  = len(CLASS_NAMES)

    model = EpropSNN(input_dim, hidden_size, n_classes, alpha=alpha, v_th=v_th, slope=slope).to(device)
    model.clamp_rec()

    # Train
    print(f"Device: {device}; Hidden: {hidden_size}; alpha={alpha}; slope={slope}")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch_eprop(model, train_loader)
        test_acc = eval_epoch(model, test_loader)
        print(f"Epoch {epoch:02d}/{num_epochs}  "
              f"Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")
