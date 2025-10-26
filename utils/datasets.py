#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets.py
Unified loaders for:
- UCI HAR (accelerometer/gyroscope)
- Google Speech Commands (audio → log-mel), labels auto-detected
- MNIST (static repeat)
- (optional) MNIST rate-coded variant

Public API:
    get_dataloaders(dataset, root="./data", batch_size=128,
                    max_samples=None, sample_length=None)

sample_length semantics:
    - None: use each sample's real (native) sequence length as-is (one segment).
    - L>0: cap/split into segments of length L; the final remainder is padded
           to L within the batch collate.
Batched tensors are shaped: [batch, segment, time, value]

Returns:
    train_loader, test_loader, meta
where meta = {
    "n_classes": int,
    "input_dim": int,   # feature dimension per time step
    "time_steps": int,  # segment length used by the loader (T)
    "class_names": List[str],
    "norm_mean": Optional[List[float]],
    "norm_std": Optional[List[float]],
}
"""

from typing import Tuple, Dict, List, Optional
import os
import zipfile
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
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
# Common dataset wrappers + batching helpers
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


def _split_into_segments(x: torch.Tensor, L: Optional[int]) -> List[torch.Tensor]:
    """
    x: [T, D] (D can be 1)
    L: segment length or None
    Returns list of [t_i, D] (t_i <= L when L is not None)
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)  # [T, 1]
    T = x.shape[0]
    if L is None or L <= 0:
        return [x]  # one segment, real length
    if T <= L:
        return [x]  # one segment, shorter than or equal to cap
    segs = []
    start = 0
    while start < T:
        segs.append(x[start:start+L])
        start += L
    return segs


def _pad_time_to(x: torch.Tensor, T_pad: int) -> torch.Tensor:
    """Right-pad along time to T_pad. x: [T, D] -> [T_pad, D]"""
    T, D = x.shape
    if T == T_pad:
        return x
    out = torch.zeros((T_pad, D), dtype=x.dtype)
    out[:T] = x
    return out


def make_segment_collate(sample_length: Optional[int]):
    """
    Returns a collate_fn that:
      - splits each sequence into segments of length sample_length (if given)
      - right-pads each segment in time to (sample_length or batch max)
      - pads number of segments per sample to S_max
    Output: X [B, S, T, D], y [B]
    """
    L = sample_length

    def _collate(batch):
        # batch: list of (x:[T,D or T], y:int)
        all_segments: List[List[torch.Tensor]] = []
        ys: List[int] = []

        # First pass: split
        for x, y in batch:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            segs = _split_into_segments(x, L)
            all_segments.append(segs)
            ys.append(int(y))

        # Determine padding lengths
        S_max = max(len(segs) for segs in all_segments) if all_segments else 1
        if L is not None and L > 0:
            T_pad = L
        else:
            # no fixed length: pad to the max time across all segments in this batch
            T_pad = 1
            for segs in all_segments:
                for s in segs:
                    T_pad = max(T_pad, s.shape[0])

        # Feature dim (D) — assume consistent within batch
        D = all_segments[0][0].shape[1] if all_segments and all_segments[0] else 1
        B = len(batch)

        X = torch.zeros((B, S_max, T_pad, D), dtype=torch.float32)

        for b, segs in enumerate(all_segments):
            for s_idx, s in enumerate(segs):
                X[b, s_idx] = _pad_time_to(s, T_pad).to(torch.float32)
            # remaining [S_max - len(segs)] left as zeros

        y = torch.tensor(ys, dtype=torch.long)
        return X, y

    return _collate


# ───────────────────────────────────────────────────────────────────────────────
# Generic utilities to work with segmented batches
# ───────────────────────────────────────────────────────────────────────────────

def flatten_segments(X: torch.Tensor, y: torch.Tensor):
    """
    X: [B, S, T, D], y: [B]
    Returns:
      X_segs: [Nseg, T, D]     (valid, non-zero segments only)
      y_segs: [Nseg]
      sample_ids: [Nseg]       (original sample index per segment)
      seg_mask: [B, S]         (True for valid segments)

    A segment is considered valid if sum(|x|) over (T,D) > 0.
    Works for any dataset that uses this collate scheme.
    """
    assert X.dim() == 4, f"Expected [B,S,T,D], got {tuple(X.shape)}"
    B, S, T, D = X.shape
    seg_mask = (X.abs().sum(dim=(2, 3)) > 0)  # [B, S]
    b_idx, s_idx = torch.where(seg_mask)
    if b_idx.numel() == 0:
        return X.new_zeros((0, T, D)), y.new_zeros((0,), dtype=torch.long), b_idx, seg_mask
    X_segs = X[b_idx, s_idx]  # [Nseg, T, D]
    y_segs = y[b_idx]         # [Nseg]
    sample_ids = b_idx        # [Nseg]
    return X_segs, y_segs, sample_ids, seg_mask


def majority_vote(preds_seg: torch.Tensor, sample_ids: torch.Tensor, num_classes: int, B: int):
    """
    preds_seg:  [Nseg] predicted label per segment
    sample_ids: [Nseg] original sample index (0..B-1) per segment
    num_classes: int
    B: batch size (number of samples)

    Returns per-sample prediction: [B]
    """
    if preds_seg.numel() == 0:
        return torch.zeros(B, dtype=torch.long, device=preds_seg.device)
    counts = torch.zeros(B, num_classes, device=preds_seg.device)
    one_hot = F.one_hot(preds_seg, num_classes=num_classes).float()
    counts.index_add_(0, sample_ids, one_hot)        # sum per (sample, class)
    return counts.argmax(dim=1)                      # [B]


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

def load_har(root: str, batch_size: int,
             max_samples: Optional[int] = None,
             sample_length: Optional[int] = None):
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

    collate_fn = make_segment_collate(sample_length)
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, pin_memory=pin_mem,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=pin_mem,
                              collate_fn=collate_fn)

    # Infer meta["time_steps"]
    if sample_length is not None and sample_length > 0:
        time_steps = int(sample_length)
        input_dim  = int(X_train.shape[2] if len(X_train) else X_test.shape[2])
    else:
        # Probe one minibatch to get the padded T for None case
        tmpX, _ = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)))
        time_steps = int(tmpX.shape[2])  # [B,S,T,D]
        input_dim  = int(tmpX.shape[3])

    meta = {
        "n_classes": len(_HAR_CLASSES),
        "input_dim": input_dim,
        "time_steps": time_steps,
        "class_names": _HAR_CLASSES,
        "norm_mean": None,
        "norm_std": None,
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


def load_speech_commands(root: str, batch_size: int,
                         mels: int = 64,
                         include_silence: bool = True,
                         max_samples: Optional[int] = None,
                         sample_length: Optional[int] = None) -> Tuple[DataLoader, DataLoader, Dict]:
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

    collate_fn = make_segment_collate(sample_length)
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=pin_mem, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, pin_memory=pin_mem, drop_last=False)

    # Peek to infer T and D
    tmpX, _ = next(iter(DataLoader(full_train, batch_size=1, shuffle=False,
                                   collate_fn=collate_fn)))
    time_steps = int(tmpX.shape[2])  # [B,S,T,D]
    input_dim  = int(tmpX.shape[3])
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
# MNIST — static repeat (real T=50 by default)
# ───────────────────────────────────────────────────────────────────────────────

_MNIST_STATIC_REAL_T = 50  # "real" native length; sample_length may cap/split

class _MNISTRepeatSeq(Dataset):
    """Each example: flattened image repeated across REAL_T steps → [T, 784]."""
    def __init__(self, train: bool, root: str, real_T: int = _MNIST_STATIC_REAL_T):
        assert _HAS_TORCHVISION, "torchvision is required for MNIST."
        self.ds = tvds.MNIST(root=root, train=train, download=True, transform=T.ToTensor())
        self.real_T = int(real_T)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]              # x: [1,28,28] in [0,1]
        x = x.view(-1)                   # [784]
        series = x.unsqueeze(0).repeat(self.real_T, 1)  # [T,784]
        return series, int(y)

def load_mnist_static(root: str, batch_size: int,
                      max_samples: Optional[int] = None,
                      sample_length: Optional[int] = None):
    os.makedirs(root, exist_ok=True)
    train_ds = _MNISTRepeatSeq(train=True,  root=root, real_T=_MNIST_STATIC_REAL_T)
    test_ds  = _MNISTRepeatSeq(train=False, root=root, real_T=_MNIST_STATIC_REAL_T)

    if max_samples is not None:
        max_train = max(1, min(max_samples, len(train_ds)))
        max_test  = max(1, min(max_samples // 4 if max_samples > 4 else 1, len(test_ds)))
        train_ds  = torch.utils.data.Subset(train_ds, range(max_train))
        test_ds   = torch.utils.data.Subset(test_ds,  range(max_test))

    collate_fn = make_segment_collate(sample_length)
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, pin_memory=pin_mem,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=pin_mem,
                              collate_fn=collate_fn)
    # Probe to fill meta
    tmpX, _ = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)))
    meta = {
        "n_classes": 10,
        "input_dim": int(tmpX.shape[3]),
        "time_steps": int(tmpX.shape[2]),
        "class_names": [str(i) for i in range(10)],
        "norm_mean": None,
        "norm_std": None,
    }
    return train_loader, test_loader, meta


# (Optional) MNIST rate-coded variant (real T=20 by default)
_MNIST_RATE_REAL_T = 20

class _MNISTRateSpike(Dataset):
    """Rate code: per-pixel Bernoulli spikes over REAL_T steps → [T,784]."""
    def __init__(self, train: bool, root: str, real_T: int = _MNIST_RATE_REAL_T,
                 gain: float = 0.7, seed: int = 0):
        assert _HAS_TORCHVISION, "torchvision is required for MNIST."
        self.ds = tvds.MNIST(root=root, train=train, download=True, transform=T.ToTensor())
        self.real_T, self.gain = int(real_T), float(gain)
        self.rng = torch.Generator().manual_seed(seed + (0 if train else 1))
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]          # [1,28,28]
        x01 = (x - x.min()) / (x.max() - x.min() + 1e-8)
        p = torch.clamp(self.gain * x01, 0.0, 1.0)
        spikes = torch.bernoulli(p.expand(self.real_T, -1, -1, -1), generator=self.rng)  # [T,1,28,28]
        return spikes.squeeze(1).reshape(self.real_T, 28*28).float(), int(y)


def load_mnist_rate(root: str, batch_size: int,
                    gain: float = 0.7,
                    max_samples: Optional[int] = None,
                    sample_length: Optional[int] = None):
    assert _HAS_TORCHVISION, "torchvision is required for MNIST."
    train_ds = _MNISTRateSpike(train=True,  root=root, real_T=_MNIST_RATE_REAL_T, gain=gain, seed=0)
    test_ds  = _MNISTRateSpike(train=False, root=root, real_T=_MNIST_RATE_REAL_T, gain=gain, seed=0)
    if max_samples is not None:
        train_ds  = torch.utils.data.Subset(train_ds, range(min(max_samples, len(train_ds))))
        test_ds   = torch.utils.data.Subset(test_ds,  range(max(1, min(max_samples // 4 if max_samples > 4 else 1, len(test_ds)))))

    collate_fn = make_segment_collate(sample_length)
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, pin_memory=pin_mem,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=pin_mem,
                              collate_fn=collate_fn)
    # Probe to fill meta
    tmpX, _ = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)))
    meta = {
        "n_classes": 10,
        "input_dim": int(tmpX.shape[3]),
        "time_steps": int(tmpX.shape[2]),
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
                    max_samples: Optional[int] = None,
                    sample_length: Optional[int] = None):
    """
    dataset: one of {"har", "speech_commands", "mnist", "mnist_rate"}
    sample_length: Optional[int] used to cap/split sequences into equal segments.
                   None keeps each example as a single segment with its real length.
    """
    ds = dataset.lower()
    os.makedirs(root, exist_ok=True)

    if ds == "har":
        return load_har(root, batch_size, max_samples=max_samples,
                        sample_length=sample_length)

    elif ds == "speech_commands":
        return load_speech_commands(root, batch_size,
                                    max_samples=max_samples,
                                    sample_length=sample_length)

    elif ds == "mnist":
        return load_mnist_static(root, batch_size,
                                 max_samples=max_samples,
                                 sample_length=sample_length)

    elif ds == "mnist_rate":
        return load_mnist_rate(root, batch_size,
                               gain=0.7,
                               max_samples=max_samples,
                               sample_length=sample_length)

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
