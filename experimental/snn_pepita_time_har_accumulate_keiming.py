import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init

import snntorch as snn
from snntorch import utils, surrogate

# ─── HAR PARAMETERS ────────────────────────────────────────────────────────────
DATA_URL    = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_PATH    = "../data/UCI_HAR.zip"
DATA_DIR    = "../data/UCI_HAR_Dataset"
WINDOW_LEN  = 128
CHANNELS    = [               # C = 9
    "body_acc_x","body_acc_y","body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z"
]
CLASS_NAMES = [               # K = 6
    "Walking","Walking Upstairs","Walking Downstairs",
    "Sitting","Standing","Laying"
]

# SNN hyper-params
num_epochs  = 20
batch_size  = 128
hidden_dim  = 128
beta        = 0.9
lr          = 0.01
f_factor    = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── DATA DOWNLOAD / LOAD ──────────────────────────────────────────────────────
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
    folder = os.path.join(DATA_DIR, split, "Inertial Signals")
    arrays = []
    for ch in CHANNELS:
        arr = np.loadtxt(os.path.join(folder, f"{ch}_{split}.txt"))  # [N, T]
        arrays.append(arr[..., None])                                # [N, T, 1]
    X = np.concatenate(arrays, axis=2)  # → [N, T, C]
    y = np.loadtxt(os.path.join(DATA_DIR, split, f"y_{split}.txt")).astype(int) - 1
    return X, y

download_and_extract()
X_train, y_train = load_split("train")
X_test,  y_test  = load_split("test")

class HARTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()   # [N, T, C]
        self.y = torch.from_numpy(y).long()    # [N]
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = HARTimeSeriesDataset(X_train, y_train)
test_ds  = HARTimeSeriesDataset(X_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

series_len  = WINDOW_LEN        # 128
in_channels = len(CHANNELS)     # 9
n_classes   = len(CLASS_NAMES)  # 6

# ─── SNN DEFINITION WITH HE/KAIMING INIT ───────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, output_dim, beta):
        super().__init__()
        # now accept C channels
        self.fc1  = nn.Linear(in_channels, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2  = nn.Linear(hidden_dim, output_dim)

        # ─── HE / KAIMING INITIALIZATION ─────────────────────────────────
        init.kaiming_uniform_(self.fc1.weight, a=0, mode="fan_in", nonlinearity="relu")
        init.zeros_(self.fc1.bias)

        init.kaiming_uniform_(self.fc2.weight, a=0, mode="fan_in", nonlinearity="linear")
        init.zeros_(self.fc2.bias)

    def reset(self):
        utils.reset(self.lif1)

model = SNN(
    in_channels=in_channels,
    hidden_dim=hidden_dim,
    output_dim=n_classes,
    beta=beta
).to(device)

# ─── PROJECTION MATRIX ─────────────────────────────────────────────────────────
# now project error into each of the C input channels
projection = torch.empty(n_classes, in_channels, device=device)
init.kaiming_normal_(projection, nonlinearity='relu')
projection *= f_factor

# ─── TRAINING LOOP (Approximate single‐difference update) ─────────────────────
for epoch in range(1, num_epochs + 1):
    model.train()
    running_acc = 0.0

    for series, labels in train_loader:
        # series: [B, T, C]
        series = series.to(device)
        labels = labels.to(device)
        B, T, C = series.shape

        # ─ First Pass: accumulate outputs & hidden sums ────────────────────────
        model.reset()
        rates      = torch.zeros((B, n_classes),         device=device)
        h_norm_sum = torch.zeros((B, hidden_dim),        device=device)

        for t in range(T):
            x_t = series[:, t, :]               # [B, C]
            h_t = model.lif1(model.fc1(x_t))    # [B, H]
            o_t = model.fc2(h_t)                # [B, n_classes]
            rates      += o_t
            h_norm_sum += h_t

        # Softmax & error
        p      = F.softmax(rates, dim=1)               # [B, K]
        onehot = F.one_hot(labels, n_classes).float()  # [B, K]
        e      = p - onehot                            # [B, K]

        # Modulate inputs
        proj_err   = e @ projection                    # [B, C]
        # broadcast along time axis
        series_mod = series + proj_err.unsqueeze(1)    # [B, T, C]

        # ─ Second Pass: accumulate modulated hidden & input sums ─────────────
        model.reset()
        h_mod_sum = torch.zeros((B, hidden_dim), device=device)
        x_mod_sum = torch.zeros((B, C),          device=device)

        for t in range(T):
            x_mod = series_mod[:, t, :]              # [B, C]
            h_mod = model.lif1(model.fc1(x_mod))     # [B, H]
            h_mod_sum += h_mod
            x_mod_sum += x_mod

        # ─ Compute per‐time averages ───────────────────────────────────────────
        h_norm_avg = h_norm_sum / T                  # [B, H]
        h_mod_avg  = h_mod_sum  / T                  # [B, H]
        x_mod_avg  = x_mod_sum  / T                  # [B, C]

        # ─ Manual Weight Updates ──────────────────────────────────────────────
        delta_w1 = - (h_norm_avg - h_mod_avg).t() @ x_mod_avg / B  # [H, C]
        model.fc1.weight.data += lr * delta_w1

        delta_w2 = - (e.t() @ h_mod_avg) / B                     # [K, H]
        model.fc2.weight.data += lr * delta_w2

        # ─ Accumulate Training Accuracy ─────────────────────────────────────
        preds = p.argmax(dim=1)
        running_acc += (preds == labels).float().mean().item()

    train_acc = 100.0 * running_acc / len(train_loader)
    print(f"Epoch {epoch:02d}/{num_epochs}, Train Acc: {train_acc:.2f}%")

# ─── FINAL EVALUATION ─────────────────────────────────────────────────────────
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series, labels = series.to(device), labels.to(device)
        model.reset()
        out_rec = []
        for t in range(series_len):
            x_t = series[:, t, :]
            h_t = model.lif1(model.fc1(x_t))
            out_rec.append(model.fc2(h_t))
        rates = torch.stack(out_rec, dim=0).sum(dim=0)
        preds = rates.argmax(dim=1)
        test_acc += (preds == labels).float().mean().item()

test_acc = 100.0 * test_acc / len(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
