#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets.py
Unified loaders for:
- UCI HAR (accelerometer/gyroscope)
- WISDM (accelerometer)
- Google Speech Commands (audio → log-mel), labels auto-detected
- MNIST (static repeat)
- (optional) MNIST rate-coded variant

Public API:
    get_dataloaders(dataset, root="./data", batch_size=128,
                    window_len=128, num_workers=2, max_samples=None,
                    mnist_T_steps=50)

Returns:
    train_loader, test_loader, meta
where meta = {
    "n_classes": int,
    "input_dim": int,   # feature dimension per time step
    "time_steps": int,  # sequence length (T)
    "class_names": List[str],
    # present for WISDM only (stats computed on train split)
    "norm_mean": Optional[List[float]],
    "norm_std": Optional[List[float]],
}
"""

from typing import Tuple, Dict, List, Optional
import os
import math
import zipfile
import urllib.request
import sys

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

# Optional (only needed for MNIST)
try:
    from torchvision import datasets as tvds, transforms as T
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


# ───────────────────────────────────────────────────────────────────────────────
# Debugger-aware worker handling (helps on Windows + PyCharm)
# ───────────────────────────────────────────────────────────────────────────────

def _is_debugging() -> bool:
    try:
        if sys.gettrace() is not None:
            return True
    except Exception:
        pass
    return os.environ.get("PYCHARM_HOSTED") == "1"


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
    debugging = _is_debugging()
    pin_mem = torch.cuda.is_available()
    nw_train = 0 if debugging else max(0, int(num_workers))
    nw_test  = 0 if debugging else max(0, int(num_workers))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw_train, drop_last=False, pin_memory=pin_mem,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw_test, drop_last=False, pin_memory=pin_mem,
        persistent_workers=False
    )
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

    if max_samples is not None:
        n_tr = min(max_samples, len(X_train))
        n_te = max(1, min(max_samples // 4 if max_samples and max_samples > 4 else 1, len(X_test)))
        X_train, y_train = X_train[:n_tr], y_train[:n_tr]
        X_test,  y_test  = X_test[:n_te],  y_test[:n_te]

    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds  = TimeSeriesDataset(X_test,  y_test)

    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
    meta = {
        "n_classes": len(_HAR_CLASSES),
        "input_dim": X_train.shape[2] if len(X_train) else X_test.shape[2],
        "time_steps": X_train.shape[1] if len(X_train) else X_test.shape[1],
        "class_names": _HAR_CLASSES,
        "norm_mean": None,
        "norm_std": None,
    }
    return train_loader, test_loader, meta


# ───────────────────────────────────────────────────────────────────────────────
# WISDM (accelerometer) — sliding window
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
    T, C = arr.shape
    if T < win:
        pad = np.zeros((win - T, C), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
        T = win
    starts = np.arange(0, T - win + 1, step)
    windows = np.stack([arr[s:s+win] for s in starts], axis=0) if len(starts) else arr[None, :win]
    return windows

def _wisdm_parse_and_window(path: str, window_len: int = 128, step: Optional[int] = None):
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

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], class_names

def load_wisdm(root: str, batch_size: int, window_len: int = 128,
               num_workers: int = 2, test_split: float = 0.2,
               max_samples: Optional[int] = None):
    """
    Loads WISDM and **normalizes inside the loader**.
    Normalization is per-channel using train-split mean/std computed over (N,T) axes.
    """
    path = _wisdm_find_file(root)
    if path is None:
        raise FileNotFoundError(
            "[WISDM] Could not find a WISDM CSV/TXT.\n"
            f"Expected something like {os.path.join(root, 'WISDM', 'WISDM_ar_v1.1_raw.txt')}."
        )

    X, y, class_names = _wisdm_parse_and_window(path, window_len=window_len, step=window_len // 2)

    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]

    # ----- split -----
    N = len(X)
    n_test = max(1, int(math.ceil(N * float(test_split))))
    n_train = max(1, N - n_test)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test,  y_test  = X[n_train:n_train+n_test], y[n_train:n_train+n_test]

    # ----- compute train stats & normalize both splits -----
    # shapes: X_train [Ntr, T, C]
    eps = 1e-6
    mean = X_train.mean(axis=(0, 1), keepdims=True)                 # [1,1,C]
    var  = ((X_train - mean)**2).mean(axis=(0, 1), keepdims=True)   # [1,1,C]
    std  = np.sqrt(np.clip(var, eps, None))                         # [1,1,C]

    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    train_ds = TimeSeriesDataset(X_train.astype(np.float32), y_train.astype(np.int64))
    test_ds  = TimeSeriesDataset(X_test.astype(np.float32),  y_test.astype(np.int64))

    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
    meta = {
        "n_classes": len(class_names),
        "input_dim": X.shape[2],
        "time_steps": X.shape[1],
        "class_names": class_names,
        # expose stats for reproducibility / debugging
        "norm_mean": mean.reshape(-1).tolist(),
        "norm_std": std.reshape(-1).tolist(),
    }
    return train_loader, test_loader, meta


# ───────────────────────────────────────────────────────────────────────────────
# Google Speech Commands → log-mel time series (labels auto-detected)
# ───────────────────────────────────────────────────────────────────────────────

class _SpeechCommandsWrapper(Dataset):
    """
    Wrapper over torchaudio SPEECHCOMMANDS producing log-mel features [T, mels].
    Labels are auto-detected from subfolders; optional 'silence' from _background_noise_.
    """
    def __init__(self, subset: str, root: str, mels: int = 64, win_len: int = 25, hop_len: int = 10,
                 target_words: Optional[str | List[str]] = "auto", max_seconds: float = 1.0,
                 include_silence: bool = True):
        assert _HAS_TORCHAUDIO, "torchaudio is required for Speech Commands."
        self.ds = SPEECHCOMMANDS(root=root, download=True, subset=subset)
        self.sample_rate = 16000
        self.mels = mels
        self.max_len = int(max_seconds * self.sample_rate)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024,
            win_length=int(win_len * self.sample_rate / 1000),
            hop_length=int(hop_len * self.sample_rate / 1000),
            n_mels=mels
        )
        self.amplog = torchaudio.transforms.AmplitudeToDB()
        self.include_silence = bool(include_silence)

        # ----- Auto-detect labels -----
        if target_words is None or target_words == "auto":
            root_path = getattr(self.ds, "_path", None) or self.ds._path  # dataset root
            labels = sorted([d for d in os.listdir(root_path)
                             if os.path.isdir(os.path.join(root_path, d)) and not d.startswith("_")])
            if os.path.isdir(os.path.join(root_path, "_background_noise_")) and "silence" not in labels:
                labels.append("silence")
            self.target_words = labels
        elif isinstance(target_words, list):
            self.target_words = target_words
        else:
            raise ValueError("target_words must be 'auto', a list, or None.")

        self.word_to_idx = {w.lower(): i for i, w in enumerate(self.target_words)}

        # Build index; optionally extend with synthetic silence entries marked by -1
        self._indices = list(range(len(self.ds)))
        self._silence_bank: List[torch.Tensor] = []
        if self.include_silence and ("silence" in self.word_to_idx):
            try:
                noise_dir = os.path.join(getattr(self.ds, "_path", ""), "_background_noise_")
                if os.path.isdir(noise_dir):
                    import glob
                    import random
                    for nf in glob.glob(os.path.join(noise_dir, "*.wav")):
                        wav, nsr = torchaudio.load(nf)  # [C, N]
                        wav = wav.mean(dim=0, keepdim=True)  # mono
                        if nsr != self.sample_rate:
                            wav = torchaudio.functional.resample(wav, nsr, self.sample_rate)
                        wav = wav.squeeze(0)
                        if wav.numel() >= self.max_len:
                            for _ in range(6):
                                start = random.randint(0, wav.numel() - self.max_len)
                                seg = wav[start:start + self.max_len].clone()
                                self._silence_bank.append(seg)
                if self._silence_bank:
                    self._indices.extend([-1] * len(self._silence_bank))
            except Exception:
                self._silence_bank = []  # fall back silently

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        idx = self._indices[i]
        if idx == -1 and self._silence_bank:
            wav = self._silence_bank[i % len(self._silence_bank)]
            mel = self.melspec(wav.unsqueeze(0))
            logmel = self.amplog(mel).squeeze(0).transpose(0, 1)  # [T, mels]
            y = self.word_to_idx["silence"]
            return logmel, y

        waveform, sr, label, *_ = self.ds[idx]
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        wav = waveform[0]
        if wav.numel() > self.max_len:
            wav = wav[:self.max_len]
        elif wav.numel() < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.numel()))

        mel = self.melspec(wav.unsqueeze(0))
        logmel = self.amplog(mel).squeeze(0).transpose(0, 1)  # [T, mels]

        y = self.word_to_idx.get(label.lower(), None)
        if y is None:
            # Skip any label we didn't index (shouldn't happen under 'auto')
            return self.__getitem__((i + 1) % len(self))
        return logmel, y


def _sc_collate(batch):
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
                         include_silence: bool = True,
                         max_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, Dict]:
    assert _HAS_TORCHAUDIO, "torchaudio is required for Speech Commands."
    os.makedirs(root, exist_ok=True)

    # Auto-detect labels by default
    train_ds = _SpeechCommandsWrapper("training",   root=root, mels=mels,
                                      include_silence=include_silence, target_words="auto")
    valid_ds = _SpeechCommandsWrapper("validation", root=root, mels=mels,
                                      include_silence=include_silence, target_words="auto")
    test_ds  = _SpeechCommandsWrapper("testing",    root=root, mels=mels,
                                      include_silence=include_silence, target_words="auto")

    full_train: Dataset = torch.utils.data.ConcatDataset([train_ds, valid_ds])

    if max_samples is not None:
        max_train = max(1, min(max_samples, len(full_train)))
        max_test  = max(1, min(max_samples // 4 if max_samples and max_samples > 4 else 1, len(test_ds)))
        full_train = torch.utils.data.Subset(full_train, range(max_train))
        test_ds    = torch.utils.data.Subset(test_ds,   range(max_test))

    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True,
                              num_workers=(0 if _is_debugging() else num_workers),
                              collate_fn=_sc_collate, pin_memory=torch.cuda.is_available(),
                              persistent_workers=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=(0 if _is_debugging() else num_workers),
                              collate_fn=_sc_collate, pin_memory=torch.cuda.is_available(),
                              persistent_workers=False)

    # Peek to infer T and D
    tmpX, _ = next(iter(DataLoader(full_train, batch_size=1, shuffle=False,
                                   collate_fn=_sc_collate, num_workers=0)))
    time_steps = int(tmpX.shape[1])
    input_dim  = int(tmpX.shape[2])
    # class names come from the wrapped dataset
    class_names = train_ds.target_words

    meta = {
        "n_classes": len(class_names),
        "input_dim": input_dim,
        "time_steps": time_steps,
        "class_names": class_names,
        "norm_mean": None,
        "norm_std": None,
    }
    return train_loader, test_loader, meta


# ───────────────────────────────────────────────────────────────────────────────
# MNIST — static repeat
# ───────────────────────────────────────────────────────────────────────────────

class _MNISTRepeatSeq(Dataset):
    """Each example: flattened image repeated across T steps → [T, 784]."""
    def __init__(self, train: bool, root: str, T_steps: int = 50):
        assert _HAS_TORCHVISION, "torchvision is required for MNIST."
        self.ds = tvds.MNIST(root=root, train=train, download=True, transform=T.ToTensor())
        self.T_steps = int(T_steps)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]              # x: [1,28,28] in [0,1]
        x = x.view(-1)                   # [784]
        series = x.unsqueeze(0).repeat(self.T_steps, 1)  # [T,784]
        return series, int(y)

def load_mnist_static(root: str, batch_size: int, num_workers: int = 2,
                      T_steps: int = 50, max_samples: Optional[int] = None):
    os.makedirs(root, exist_ok=True)
    train_ds = _MNISTRepeatSeq(train=True,  root=root, T_steps=T_steps)
    test_ds  = _MNISTRepeatSeq(train=False, root=root, T_steps=T_steps)

    if max_samples is not None:
        max_train = max(1, min(max_samples, len(train_ds)))
        max_test  = max(1, min(max_samples // 4 if max_samples > 4 else 1, len(test_ds)))
        train_ds  = torch.utils.data.Subset(train_ds, range(max_train))
        test_ds   = torch.utils.data.Subset(test_ds,  range(max_test))

    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
    meta = {
        "n_classes": 10,
        "input_dim": 28 * 28,
        "time_steps": int(T_steps),
        "class_names": [str(i) for i in range(10)],
        "norm_mean": None,
        "norm_std": None,
    }
    return train_loader, test_loader, meta


# (Optional) MNIST rate-coded variant
class _MNISTRateSpike(Dataset):
    """Rate code: per-pixel Bernoulli spikes over T steps → [T,784]."""
    def __init__(self, train: bool, root: str, T_steps: int = 20, gain: float = 0.7, seed: int = 0):
        assert _HAS_TORCHVISION, "torchvision is required for MNIST."
        self.ds = tvds.MNIST(root=root, train=train, download=True, transform=T.ToTensor())
        self.T_steps, self.gain = int(T_steps), float(gain)
        self.rng = torch.Generator().manual_seed(seed + (0 if train else 1))
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]          # [1,28,28]
        x01 = (x - x.min()) / (x.max() - x.min() + 1e-8)
        p = torch.clamp(self.gain * x01, 0.0, 1.0)
        spikes = torch.bernoulli(p.expand(self.T_steps, -1, -1, -1), generator=self.rng)  # [T,1,28,28]
        return spikes.squeeze(1).reshape(self.T_steps, 28*28).float(), int(y)


def load_mnist_rate(root: str, batch_size: int, num_workers: int = 2,
                    T_steps: int = 20, gain: float = 0.7,
                    max_samples: Optional[int] = None):
    assert _HAS_TORCHVISION, "torchvision is required for MNIST."
    train_ds = _MNISTRateSpike(train=True,  root=root, T_steps=T_steps, gain=gain, seed=0)
    test_ds  = _MNISTRateSpike(train=False, root=root, T_steps=T_steps, gain=gain, seed=0)
    if max_samples is not None:
        train_ds  = torch.utils.data.Subset(train_ds, range(min(max_samples, len(train_ds))))
        test_ds   = torch.utils.data.Subset(test_ds,  range(max(1, min(max_samples // 4 if max_samples > 4 else 1, len(test_ds)))))
    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers)
    meta = {
        "n_classes": 10,
        "input_dim": 28*28,
        "time_steps": int(T_steps),
        "class_names": [str(i) for i in range(10)],
        "norm_mean": None,
        "norm_std": None,
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
                    max_samples: Optional[int] = None,
                    mnist_T_steps: int = 50):
    ds = dataset.lower()
    os.makedirs(root, exist_ok=True)

    if ds == "har":
        return load_har(root, batch_size, num_workers, max_samples=max_samples)

    elif ds == "wisdm":
        return load_wisdm(root, batch_size, window_len, num_workers,
                          test_split=0.2, max_samples=max_samples)

    elif ds == "speech_commands":
        # leave max_samples=None for full coverage (or set a cap for speed)
        return load_speech_commands(root, batch_size, num_workers,
                                    max_samples=max_samples)

    elif ds == "mnist":
        return load_mnist_static(root, batch_size, num_workers,
                                 T_steps=int(mnist_T_steps),
                                 max_samples=max_samples)

    elif ds == "mnist_rate":
        return load_mnist_rate(root, batch_size, num_workers,
                               T_steps=int(mnist_T_steps), gain=0.7,
                               max_samples=max_samples)

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
