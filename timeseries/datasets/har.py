# timeseries/datasets/har.py
from __future__ import annotations
import os, zipfile, urllib.request
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

_HAR_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
_HAR_ZIP   = "UCI_HAR.zip"
_HAR_DIR   = "UCI_HAR_Dataset"
_HAR_CHS   = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z"
]
_HAR_CLASSES = ["Walking","Walking Upstairs","Walking Downstairs","Sitting","Standing","Laying"]

def _download_and_extract(root: str) -> str:
    data_dir = os.path.join(root, _HAR_DIR)
    zip_path = os.path.join(root, _HAR_ZIP)
    if not os.path.exists(data_dir):
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(zip_path):
            print("[HAR] Downloading...")
            urllib.request.urlretrieve(_HAR_URL, zip_path)
        print("[HAR] Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)
        src = os.path.join(root, "UCI HAR Dataset")
        if not os.path.exists(data_dir):
            os.rename(src, data_dir)
    return data_dir

def _load_split(data_dir: str, split: str):
    folder = os.path.join(data_dir, split, "Inertial Signals")
    arrays = []
    for ch in _HAR_CHS:
        arr = np.loadtxt(os.path.join(folder, f"{ch}_{split}.txt"))  # [N,T]
        arrays.append(arr[..., np.newaxis])
    X = np.concatenate(arrays, axis=2)  # [N,T,C]
    y = np.loadtxt(os.path.join(data_dir, split, f"y_{split}.txt")).astype(int) - 1
    return X, y

class HARDatasetRaw(Dataset):
    """IO-only: returns raw per-sample time series [T,D] (D=9)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()      # [N,T,C]
        self.y = torch.from_numpy(y).long()       # [N]
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]            # [T,C]
        y = int(self.y[i])
        info = {"id": i, "length": x.shape[0], "channels": x.shape[1]}
        return x, y, info

def build_har_raw(root: str, max_samples: Optional[int] = None):
    data_dir = _download_and_extract(root)
    X_tr, y_tr = _load_split(data_dir, "train")
    X_te, y_te = _load_split(data_dir, "test")
    if max_samples is not None:
        n_tr = min(max_samples, len(X_tr))
        n_te = max(1, min(max_samples // 4 if (max_samples and max_samples > 4) else 1, len(X_te)))
        X_tr, y_tr = X_tr[:n_tr], y_tr[:n_tr]
        X_te, y_te = X_te[:n_te], y_te[:n_te]
    return HARDatasetRaw(X_tr, y_tr), HARDatasetRaw(X_te, y_te), list(_HAR_CLASSES)
