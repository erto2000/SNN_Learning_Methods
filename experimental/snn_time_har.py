#!/usr/bin/env python3
import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import utils, surrogate


# ─── PARAMETERS ───────────────────────────────────────────────────────────────
DATA_URL    = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_PATH    = "../data/UCI_HAR.zip"
DATA_DIR    = "../data/UCI_HAR_Dataset"
WINDOW_LEN  = 128
CHANNELS    = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z"
]
CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]

# SNN hyper-params
num_epochs  = 10
batch_size  = 128
hidden_size = 128
beta        = 0.9
spike_grad  = surrogate.fast_sigmoid(slope=25)
# ────────────────────────────────────────────────────────────────────────────────

def download_and_extract():
    """Download and unzip the dataset if not already present."""
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
    """
    Load X and y for a given split.
    Returns:
      X : np.ndarray, shape = [N, WINDOW_LEN, len(CHANNELS)]
      y : np.ndarray, shape = [N]
    """
    folder = os.path.join(DATA_DIR, split, "Inertial Signals")
    # Load each channel: array shape [N, WINDOW_LEN]
    arrays = []
    for ch in CHANNELS:
        path = os.path.join(folder, f"{ch}_{split}.txt")
        arr  = np.loadtxt(path)               # → [N, WINDOW_LEN]
        arrays.append(arr[..., np.newaxis])   # → [N, WINDOW_LEN, 1]
    X = np.concatenate(arrays, axis=2)       # → [N, WINDOW_LEN, C]
    # Load labels (1..6) → zero-indexed
    y_path = os.path.join(DATA_DIR, split, f"y_{split}.txt")
    y      = np.loadtxt(y_path).astype(int) - 1
    return X, y

class HARTimeSeriesDataset(Dataset):
    """Torch Dataset for UCI HAR windows."""
    def __init__(self, X, y):
        # X: np.ndarray [N, T, C]
        self.X = torch.from_numpy(X).float()   # → [N, T, C]
        self.y = torch.from_numpy(y).long()    # → [N]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        # returns: series [T, C], label
        return self.X[idx], self.y[idx]

class SNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_steps, beta, spike_grad):
        super().__init__()
        self.time_steps = time_steps
        # a two-layer SNN: Linear → LIF → Linear
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        # data: [B, T, input_dim]
        utils.reset(self.net)   # reset hidden state
        spk_rec = []
        for t in range(self.time_steps):
            x_t = data[:, t, :]       # [B, input_dim]
            spk = self.net(x_t)       # [B, n_classes]
            spk_rec.append(spk)
        # → [T, B, n_classes]
        return torch.stack(spk_rec, dim=0)

if __name__ == "__main__":
    # 1. Download & extract
    download_and_extract()

    # 2. Load HAR data
    X_train, y_train = load_split("train")
    X_test,  y_test  = load_split("test")

    # 3. Prepare DataLoaders
    train_ds = HARTimeSeriesDataset(X_train, y_train)
    test_ds  = HARTimeSeriesDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # 4. Build model
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim  = X_train.shape[2]             # 9 channels
    time_steps = X_train.shape[1]             # 128
    n_classes  = len(CLASS_NAMES)             # 6

    model   = SNNModel(input_dim, hidden_size, n_classes,
                       time_steps, beta, spike_grad).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.CrossEntropyLoss()

    # 5. Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc  = 0.0

        for series, labels in train_loader:
            series, labels = series.to(device), labels.to(device)
            optimizer.zero_grad()

            spk_rec = model(series)        # [T, B, C]
            rates   = spk_rec.sum(dim=0)   # integrate spikes: [B, C]

            loss = loss_fn(rates, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = rates.argmax(dim=1)
            total_acc += (preds == labels).float().mean().item()

        avg_loss = total_loss / len(train_loader)
        avg_acc  = 100.0 * total_acc  / len(train_loader)
        print(f"Epoch {epoch:02d}/{num_epochs}  "
              f"Loss: {avg_loss:.4f}  "
              f"Train Acc: {avg_acc:.2f}%")

    # 6. Final evaluation
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for series, labels in test_loader:
            series, labels = series.to(device), labels.to(device)
            rates = model(series).sum(dim=0)
            preds = rates.argmax(dim=1)
            test_acc += (preds == labels).float().mean().item()

    test_acc = 100.0 * test_acc / len(test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")
