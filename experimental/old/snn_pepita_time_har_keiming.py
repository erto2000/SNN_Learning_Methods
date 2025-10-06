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
ZIP_PATH    = "../../data/UCI_HAR.zip"
DATA_DIR    = "../../data/UCI_HAR_Dataset"
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
hidden_size = 128
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

# prepare data
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

series_len = WINDOW_LEN      # 128
n_classes  = len(CLASS_NAMES)  # 6
in_channels = len(CHANNELS)    # 9

# ─── SNN DEFINITION ─────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, beta):
        super().__init__()
        # <-- CHANGE: input dim = C instead of 1
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2  = nn.Linear(hidden_dim, out_dim)

        # He init
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.zeros_(self.fc2.bias)

    def reset(self):
        utils.reset(self.lif1)

model = SNN(
    in_dim=in_channels,
    hidden_dim=hidden_size,
    out_dim=n_classes,
    beta=beta
).to(device)

# ─── PROJECTION MATRIX ─────────────────────────────────────────────────────────
# <-- CHANGE: projection maps error → C channels
projection = torch.empty(n_classes, in_channels, device=device)
init.kaiming_normal_(projection, nonlinearity='relu')
projection *= f_factor

# ─── TRAINING LOOP (Two-Pass Manual Updates) ───────────────────────────────────
for epoch in range(1, num_epochs + 1):
    model.train()
    running_acc = 0.0

    for series, labels in train_loader:
        # series: [B, T, C]
        series = series.to(device)
        labels = labels.to(device)
        B = series.size(0)

        # ─ First Pass ─
        h_rec, o_rec = [], []
        model.reset()
        for t in range(series_len):
            x_t = series[:, t, :]         # [B, C]
            h_t = model.lif1(model.fc1(x_t))
            o_t = model.fc2(h_t)
            h_rec.append(h_t)
            o_rec.append(o_t)

        h_norm = torch.stack(h_rec,   dim=0)  # [T, B, H]
        o_norm = torch.stack(o_rec,   dim=0)  # [T, B, K]
        rates  = o_norm.sum(dim=0)            # [B, K]

        p      = F.softmax(rates, dim=1)                # [B, K]
        onehot = F.one_hot(labels, n_classes).float()   # [B, K]
        e      = p - onehot                             # [B, K]

        # ─ Modulate Input ─
        proj_err   = e @ projection                     # [B, C]
        series_mod = series + proj_err.unsqueeze(1)     # [B, T, C]

        # ─ Second Pass ─
        h_mod_rec = []
        model.reset()
        for t in range(series_len):
            x_t = series_mod[:, t, :]
            h_mod_rec.append(model.lif1(model.fc1(x_t)))
        h_mod = torch.stack(h_mod_rec, dim=0)           # [T, B, H]

        # ─ Manual Updates ─
        diff     = (h_norm - h_mod)                     # [T, B, H]
        x_mod_T  = series_mod.permute(1,0,2)            # [T, B, C]
        mult     = diff.unsqueeze(3) * x_mod_T.unsqueeze(2)
        # → [T, B, H, C]  (we correlate each hidden unit with each channel)
        delta_w1 = - mult.sum(dim=(0,1)) / (B * series_len)  # [H, C]
        model.fc1.weight.data += lr * delta_w1

        h_mod_avg = h_mod.sum(dim=0) / series_len       # [B, H]
        delta_w2  = - (e.transpose(0,1) @ h_mod_avg) / B  # [K, H]
        model.fc2.weight.data += lr * delta_w2

        preds = p.argmax(dim=1)
        running_acc += (preds == labels).float().mean().item()

    train_acc = 100.0 * running_acc / len(train_loader)
    print(f"Epoch {epoch:02d}/{num_epochs}, Train Acc: {train_acc:.2f}%")

# ─── EVALUATION ────────────────────────────────────────────────────────────────
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series, labels = series.to(device), labels.to(device)
        model.reset()
        out_spikes = []
        for t in range(series_len):
            x_t = series[:, t, :]
            h_t = model.lif1(model.fc1(x_t))
            out_spikes.append(model.fc2(h_t))
        rates = torch.stack(out_spikes, dim=0).sum(dim=0)
        preds = rates.argmax(dim=1)
        test_acc += (preds == labels).float().mean().item()

test_acc = 100.0 * test_acc / len(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
