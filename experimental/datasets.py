#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets.py
Unified loaders for:
- UCI HAR (accelerometer/gyroscope)
- WISDM (accelerometer)
- Google Speech Commands (audio)

Public API:
    get_dataloaders(dataset, root="./data", batch_size=128,
                    window_len=128, num_workers=2, max_samples=None)

Returns:
    train_loader, test_loader, meta
where meta = {
    "n_classes": int,
    "input_dim": int,   # feature/channel dimension per time step
    "time_steps": int,  # sequence length (T)
    "class_names": List[str]
}
"""

from typing import Tuple, Dict, List, Optional
import os
import io
import csv
import math
import zipfile
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Optional (only needed for Speech Commands)
try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False


# ───────────────────────────────────────────────────────────────────────────────
# Common dataset wrappers
# ───────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """Generic time-series dataset: X [N, T, C], y [N]."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3, f"X must be [N,T,C], got {X.shape}"
        assert y.ndim == 1, f"y must be [N], got {y.shape}"
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _make_loaders(train_ds: Dataset, test_ds: Dataset,
                  batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    return train_loader, test_loader


# ───────────────────────────────────────────────────────────────────────────────
# UCI HAR
# ───────────────────────────────────────────────────────────────────────────────

_HAR_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
_HAR_ZIP   = "UCI_HAR.zip"
_HAR_DIR   = "UCI_HAR_Dataset"
_HAR_CHS   = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z"
]
_HAR_CLASSES = ["Walking","Walking Upstairs","Walking Downstairs","Sitting","Standing","Laying"]

def _har_download_and_extract(root: str):
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
        os.rename(os.path.join(root, "UCI HAR Dataset"), data_dir)
    return data_dir

def _har_load_split(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    folder = os.path.join(data_dir, split, "Inertial Signals")
    arrays = []
    for ch in _HAR_CHS:
        path = os.path.join(folder, f"{ch}_{split}.txt")
        arr  = np.loadtxt(path)               # [N, T]
        arrays.append(arr[..., np.newaxis])   # [N, T, 1]
    X = np.concatenate(arrays, axis=2)       # [N, T, C]
    y_path = os.path.join(data_dir, split, f"y_{split}.txt")
    y      = np.loadtxt(y_path).astype(int) - 1
    return X, y

def load_har(root: str, batch_size: int, num_workers: int = 2,
             max_samples: Optional[int] = None):
    data_dir = _har_download_and_extract(root)
    X_train, y_train = _har_load_split(data_dir, "train")
    X_test,  y_test  = _har_load_split(data_dir, "test")

    # Optional downsampling (keep evaluation small but non-zero)
    if max_samples is not None:
        n_tr = min(max_samples, len(X_train))
        n_te = max(1, min(max_samples // 4 if max_samples > 4 else 1, len(X_test)))
        X_train, y_train = X_train[:n_tr], y_train[:n_tr]
        X_test,  y_test  = X_test[:n_te],  y_test[:n_te]

    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds  = TimeSeriesDataset(X_test,  y_test)

    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
    meta = {
        "n_classes": len(_HAR_CLASSES),
        "input_dim": X_train.shape[2] if len(X_train) else X_test.shape[2],
        "time_steps": X_train.shape[1] if len(X_train) else X_test.shape[1],
        "class_names": _HAR_CLASSES
    }
    return train_loader, test_loader, meta


# ───────────────────────────────────────────────────────────────────────────────
# WISDM (accelerometer). Robust raw parser + sliding window.
# Expected raw line format (no header, trailing ';'):
#   user,Activity,Timestamp,accX,accY,accZ;
# ───────────────────────────────────────────────────────────────────────────────

_WISDM_CANON = {
    "Walking": "Walking",
    "Jogging": "Jogging",
    "Sitting": "Sitting",
    "Standing": "Standing",
    "Upstairs": "Upstairs",
    "Downstairs": "Downstairs",
}

def _wisdm_find_file(root: str) -> Optional[str]:
    """Try to locate a typical WISDM CSV/TXT file under root/WISDM."""
    candidates = []
    wroot = os.path.join(root, "WISDM")
    if os.path.isdir(wroot):
        for fn in os.listdir(wroot):
            if fn.lower().endswith((".csv", ".txt")) and "wisdm" in fn.lower():
                candidates.append(os.path.join(wroot, fn))
        if not candidates:
            for fn in os.listdir(wroot):
                if fn.lower().endswith((".csv", ".txt")):
                    candidates.append(os.path.join(wroot, fn))
    return candidates[0] if candidates else None

def _sliding_windows(arr: np.ndarray, win: int, step: int) -> np.ndarray:
    """
    Create windows over time axis.
    arr: [T, C] -> [N, win, C]
    """
    T, C = arr.shape
    if T < win:  # pad if too short
        pad = np.zeros((win - T, C), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
        T = win
    starts = np.arange(0, T - win + 1, step)
    windows = np.stack([arr[s:s+win] for s in starts], axis=0) if len(starts) else arr[None, :win]
    return windows

def _wisdm_parse_and_window(path: str, window_len: int = 128, step: Optional[int] = None):
    """
    Robust parser for WISDM raw lines like:
    33,Jogging,49105962326000,-0.6946,12.6805,0.5039;
    Builds contiguous segments per activity, then windows them into [N,T,3].
    """
    if step is None:
        step = window_len // 2  # 50% overlap

    X_all: List[np.ndarray] = []
    y_all: List[int] = []

    def flush_current(cur_act, cur_buf):
        if cur_act is None or not cur_buf:
            return None, []
        arr = np.array(cur_buf, dtype=np.float32)  # [T,3]
        wins = _sliding_windows(arr, window_len, step)  # [N,T,3]
        label = list(_WISDM_CANON.keys()).index(cur_act)
        X_all.append(wins)
        y_all.append(np.full((wins.shape[0],), label, dtype=np.int64))
        return None, []

    cur_act = None
    cur_buf: List[List[float]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(";"):
                line = line[:-1]

            parts = line.split(",")
            if len(parts) < 6:
                continue

            act_raw  = parts[1].strip()
            xs, ys, zs = parts[3:6]

            act_norm = act_raw.capitalize()
            if act_norm not in _WISDM_CANON:
                cur_act, cur_buf = flush_current(cur_act, cur_buf)
                continue

            if cur_act is not None and act_norm != cur_act:
                cur_act, cur_buf = flush_current(cur_act, cur_buf)

            cur_act = act_norm
            try:
                x = float(xs); y = float(ys); z = float(zs)
            except ValueError:
                continue
            cur_buf.append([x, y, z])

    cur_act, cur_buf = flush_current(cur_act, cur_buf)

    if not X_all:
        raise RuntimeError(f"[WISDM] No usable windows parsed from {path}.")

    X = np.concatenate(X_all, axis=0)          # [N, T, 3]
    y = np.concatenate(y_all, axis=0)          # [N]
    class_names = list(_WISDM_CANON.values())

    # Shuffle once for reproducibility
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], class_names

def load_wisdm(root: str, batch_size: int, window_len: int = 128,
               num_workers: int = 2, test_split: float = 0.2,
               max_samples: Optional[int] = None):
    """
    Looks for WISDM under {root}/WISDM/*.csv or *.txt.
    If not found, raises with a friendly message.
    """
    path = _wisdm_find_file(root)
    if path is None:
        raise FileNotFoundError(
            "[WISDM] Could not find a WISDM CSV/TXT.\n"
            f"Expected something like {os.path.join(root, 'WISDM', 'WISDM_ar_v1.1_raw.txt')}.\n"
            "Place the file there and try again. (We avoid hard-coding URLs that change.)"
        )

    X, y, class_names = _wisdm_parse_and_window(path, window_len=window_len, step=window_len // 2)

    # Optional downsampling before split
    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]

    # train/test split (simple tail split)
    N = len(X)
    n_test = max(1, int(math.ceil(N * test_split)))
    n_train = max(1, N - n_test)
    train_ds = TimeSeriesDataset(X[:n_train], y[:n_train])
    test_ds  = TimeSeriesDataset(X[n_train:n_train+n_test], y[n_train:n_train+n_test])

    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
    meta = {
        "n_classes": len(class_names),
        "input_dim": X.shape[2],       # 3 (x,y,z)
        "time_steps": X.shape[1],      # window_len
        "class_names": class_names
    }
    return train_loader, test_loader, meta


# ───────────────────────────────────────────────────────────────────────────────
# Google Speech Commands via torchaudio (downloaded automatically)
# We convert waveforms -> log-mel spectrograms and treat as time-series [T, C=mels]
# ───────────────────────────────────────────────────────────────────────────────

class _SpeechCommandsWrapper(Dataset):
    def __init__(self, subset: str, root: str, mels: int = 64, win_len: int = 25, hop_len: int = 10,
                 target_words: Optional[List[str]] = None, max_seconds: float = 1.0):
        """
        subset: "training", "validation", or "testing" (per torchaudio split)
        win_len/hop_len in ms
        """
        assert _HAS_TORCHAUDIO, "torchaudio is required for Speech Commands."
        self.ds = SPEECHCOMMANDS(root=root, download=True, subset=subset)
        self.sample_rate = 16000
        self.mels = mels
        self.max_len = int(max_seconds * self.sample_rate)

        # Transform: waveform -> log-mel (time major)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024,
            win_length=int(win_len * self.sample_rate / 1000),
            hop_length=int(hop_len * self.sample_rate / 1000),
            n_mels=mels
        )
        self.amplog = torchaudio.transforms.AmplitudeToDB()

        # Build label set (12-class common subset)
        if target_words is None:
            target_words = ["yes","no","up","down","left","right","on","off","stop","go","unknown","silence"]
        self.target_words = target_words
        self.word_to_idx = {w: i for i, w in enumerate(target_words)}

    def _label_of(self, word: str) -> int:
        w = word.lower()
        if w in self.word_to_idx:
            return self.word_to_idx[w]
        if "unknown" in self.word_to_idx:
            return self.word_to_idx["unknown"]
        return -1

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.ds[idx]
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        wav = waveform[0]
        if wav.numel() > self.max_len:
            wav = wav[:self.max_len]
        elif wav.numel() < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.numel()))

        mel = self.melspec(wav.unsqueeze(0))                 # [1, mels, T]
        logmel = self.amplog(mel).squeeze(0).transpose(0, 1) # [T, mels]
        y = self._label_of(label)
        if y < 0:
            return self.__getitem__((idx + 1) % len(self))
        return logmel, y


def _sc_collate(batch):
    # batch: list of ( [T, mels], y )
    Ts = [b[0].shape[0] for b in batch]
    Tm = max(Ts)
    mels = batch[0][0].shape[1]
    X = torch.zeros((len(batch), Tm, mels), dtype=torch.float32)
    y = torch.zeros((len(batch),), dtype=torch.long)
    for i, (x, yi) in enumerate(batch):
        T = x.shape[0]
        X[i, :T] = x
        y[i] = yi
    return X, y


def load_speech_commands(root: str, batch_size: int, num_workers: int = 2,
                         mels: int = 64,
                         max_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, Dict]:
    assert _HAS_TORCHAUDIO, "torchaudio is required for Speech Commands."
    os.makedirs(root, exist_ok=True)

    train_ds = _SpeechCommandsWrapper("training",   root=root, mels=mels)
    valid_ds = _SpeechCommandsWrapper("validation", root=root, mels=mels)
    test_ds  = _SpeechCommandsWrapper("testing",    root=root, mels=mels)

    # Combine train+valid; optionally downsample
    full_train: Dataset = torch.utils.data.ConcatDataset([train_ds, valid_ds])

    if max_samples is not None:
        max_train = max(1, min(max_samples, len(full_train)))
        max_test  = max(1, min(max_samples // 4 if max_samples > 4 else 1, len(test_ds)))
        full_train = torch.utils.data.Subset(full_train, range(max_train))
        test_ds    = torch.utils.data.Subset(test_ds,   range(max_test))

    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=_sc_collate)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=_sc_collate)

    # Peek a batch to derive meta (time_steps can vary; pick observed length)
    tmpX, _ = next(iter(train_loader))
    time_steps = tmpX.shape[1]
    input_dim  = tmpX.shape[2]
    class_names = train_ds.target_words

    meta = {
        "n_classes": len(class_names),
        "input_dim": input_dim,      # mels
        "time_steps": time_steps,    # ~100 frames for 1s audio (hop ~10ms)
        "class_names": class_names
    }
    return train_loader, test_loader, meta


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

def get_dataloaders(dataset: str,
                    root: str = "./data",
                    batch_size: int = 128,
                    window_len: int = 128,
                    num_workers: int = 2,
                    max_samples: Optional[int] = None):
    """
    dataset ∈ {"har", "wisdm", "speech_commands"} (case-insensitive)
    max_samples:
        Limit number of training samples for quick experiments.
        If provided, test set is also reduced (~25% of max_samples) but ≥1.
    """
    ds = dataset.lower()
    os.makedirs(root, exist_ok=True)

    if ds == "har":
        return load_har(root, batch_size, num_workers, max_samples=max_samples)

    elif ds == "wisdm":
        return load_wisdm(root, batch_size, window_len, num_workers,
                          test_split=0.2, max_samples=max_samples)

    elif ds == "speech_commands":
        # Default cap if user forgets (dataset is large)
        if max_samples is None:
            max_samples = 10_000
        return load_speech_commands(os.path.join(root, "SpeechCommands"),
                                    batch_size, num_workers,
                                    max_samples=max_samples)

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'har', 'wisdm', 'speech_commands'.")
