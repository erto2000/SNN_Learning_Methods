# timeseries/transforms.py
from __future__ import annotations
from typing import Dict, Any, Sequence, Optional, List
import torch

try:
    import torchaudio
    _HAS_TA = True
except Exception:
    _HAS_TA = False

# ---------- Base ----------
class Transform:
    def __call__(self, x, y, info):
        raise NotImplementedError
    def needs_fit(self) -> bool:
        return False
    def fit(self, iterator, pre_ops: Sequence["Transform"]):
        return self

class Compose(Transform):
    def __init__(self, ops: Sequence[Transform]):
        self.ops: List[Transform] = list(ops)

    def __call__(self, x, y, info):
        for op in self.ops:
            x, y, info = op(x, y, info)
        return x, y, info

    def _apply_until(self, x, y, info, end_idx: int):
        for j in range(end_idx):
            x, y, info = self.ops[j](x, y, info)
        return x, y, info

    def fit(self, dataset, max_samples: Optional[int] = None):
        N = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        for i, op in enumerate(self.ops):
            if not op.needs_fit():
                continue
            def iterator():
                for k in range(N):
                    x, y, info = dataset[k]
                    x, y, info = self._apply_until(x, y, info, i)
                    yield x, y, info
            op.fit(iterator(), pre_ops=self.ops[:i])
        return self

# ---------- Basic ----------
class ToFloat32(Transform):
    def __call__(self, x, y, info):
        return x.to(torch.float32), y, info

class Ensure2D(Transform):
    """Make [T] -> [T,1]; pass-through if already [T,D]."""
    def __call__(self, x, y, info):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x, y, info

class Flatten(Transform):
    """If [T,H,W], make [T,H*W]."""
    def __call__(self, x, y, info):
        if x.dim() == 3:
            T, H, W = x.shape
            x = x.reshape(T, H * W)
        return x, y, info

# ---------- Normalization ----------
class ZScore(Transform):
    """
    With args: explicit ZScore(mean,std).
    No args: auto-fit on train via Compose.fit().
    """
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None, eps: float = 1e-8):
        self._mean = mean.clone() if mean is not None else None
        self._std  = std.clone()  if std  is not None else None
        self.eps = eps

    def needs_fit(self) -> bool:
        return (self._mean is None) or (self._std is None)

    @torch.no_grad()
    def fit(self, iterator, pre_ops: Sequence[Transform]):
        n = 0
        sum_c = None
        sumsq_c = None
        for x, _, _ in iterator:
            assert x.dim() == 2, "ZScore.fit expects [T,D]; place SlidingWindow AFTER ZScore."
            m = x.mean(dim=0)
            v = x.var(dim=0, unbiased=False)
            if sum_c is None:
                sum_c = m.clone()
                sumsq_c = v.clone()
            else:
                sum_c += m
                sumsq_c += v
            n += 1
        if n == 0:
            raise RuntimeError("ZScore.fit: empty iterator.")
        self._mean = (sum_c / n).to(torch.float32)
        self._std  = (sumsq_c / n).sqrt().clamp_min(1e-6).to(torch.float32)
        return self

    def __call__(self, x, y, info):
        assert self._mean is not None and self._std is not None, \
            "ZScore not fit. Call Compose.fit(train_dataset) or pass mean/std."
        mean = self._mean.view(1, -1).to(x.dtype).to(x.device)
        std  = self._std.view(1, -1).to(x.dtype).to(x.device)
        return (x - mean) / (std + self.eps), y, info

# ---------- Audio features ----------
class ToLogMel(Transform):
    """Waveform [T,1] -> Log-mel [F, M]."""
    def __init__(self, sample_rate: int, n_mels: int = 64, win_len_ms: int = 25, hop_ms: int = 10):
        assert _HAS_TA, "torchaudio required for ToLogMel."
        self.sr = int(sample_rate)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=1024,
            win_length=int(win_len_ms * self.sr / 1000),
            hop_length=int(hop_ms * self.sr / 1000),
            n_mels=int(n_mels)
        )
        self.amplog = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, x, y, info):
        X = x.transpose(0, 1)                    # [1,T]
        m = self.melspec(X)                      # [1, M, F]
        m = self.amplog(m).squeeze(0).transpose(0, 1)  # [F,M]
        return m.to(torch.float32), y, info

# ---------- Spiking / static helpers ----------
class DeterministicSpikes(Transform):
    """[1,D] in [0,1] -> [T,D] spikes with per-sample deterministic RNG."""
    def __init__(self, gain: float, T: int, base_seed: int = 0):
        self.gain, self.T, self.base_seed = float(gain), int(T), int(base_seed)
    def __call__(self, x, y, info):
        x01 = (x - x.min()) / (x.max() - x.min() + 1e-8)
        p = torch.clamp(self.gain * x01, 0.0, 1.0)
        sid = int(info.get("id", 0))
        g = torch.Generator().manual_seed(self.base_seed + sid)
        spikes = torch.bernoulli(p.expand(self.T, -1), generator=g)
        return spikes.to(torch.float32), y, info

class RepeatStatic(Transform):
    """Repeat [1,D] to [T,D]."""
    def __init__(self, T: int): self.T = int(T)
    def __call__(self, x, y, info):
        return x.repeat(self.T, 1), y, info

# ---------- Segmentation ----------
class SlidingWindow(Transform):
    """
    [T,D] -> [S, L, D] with stride 'hop'.
    Place AFTER any ops that expect unsegmented [T,D] (e.g., ZScore).
    """
    def __init__(self, length: int, hop: Optional[int] = None):
        self.L = int(length)
        self.hop = int(hop) if hop and hop > 0 else int(length)
    def __call__(self, x, y, info):
        T, D = x.shape
        segs = []
        start = 0
        while start < T:
            segs.append(x[start:start+self.L])
            start += self.hop
        Lmax = max(s.shape[0] for s in segs) if segs else self.L
        out = []
        for s in segs:
            if s.shape[0] == Lmax:
                out.append(s)
            else:
                pad = s.new_zeros((Lmax, D)); pad[:s.shape[0]] = s; out.append(pad)
        X = torch.stack(out, dim=0) if out else x.new_zeros((1, self.L, D))
        return X, y, info
