# timeseries/transforms.py
from __future__ import annotations
from typing import Dict, Any, Sequence, Optional, List
import torch
import math

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
        self.win_length = int(win_len_ms * self.sr / 1000)
        self.hop_length = int(hop_ms * self.sr / 1000)
        self.n_fft = 2 ** math.ceil(math.log2(self.win_length))
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
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

class Resample(Transform):
    def __init__(self, orig_sr: int, new_sr: int):
        assert _HAS_TA, "torchaudio required for Resample."
        import torchaudio
        self.orig_sr, self.new_sr = int(orig_sr), int(new_sr)
    def __call__(self, x, y, info):
        # x:[T,1] waveform
        import torchaudio
        X = x.transpose(0,1)  # [1,T]
        X = torchaudio.functional.resample(X, self.orig_sr, self.new_sr)
        return X.transpose(0,1).contiguous(), y, {**info, "sample_rate": self.new_sr}

class RandomTimeCrop(Transform):
    """Random crop waveform to fixed T (pads if short)."""
    def __init__(self, T: int): self.T = int(T)
    def __call__(self, x, y, info):
        T = x.shape[0]
        if T == self.T: return x, y, info
        if T < self.T:
            pad = x.new_zeros((self.T, x.shape[1])); pad[:T] = x
            return pad, y, info
        start = torch.randint(0, T-self.T+1, (1,)).item()
        return x[start:start+self.T], y, info

class SpecAugmentLike(Transform):
    """Apply simple time masking on [F,M] log-mels."""
    def __init__(self, max_time_mask: int = 20, p: float = 0.5):
        self.max_time_mask, self.p = int(max_time_mask), float(p)
    def __call__(self, x, y, info):
        if x.dim() == 2 and torch.rand(()) < self.p:
            F, M = x.shape
            w = torch.randint(1, self.max_time_mask+1, ()).item()
            s = torch.randint(0, max(1, M - w + 1), ()).item()
            x = x.clone(); x[:, s:s+w] = x.min()
        return x, y, info

class EventToVoxel(Transform):
    def __init__(self, H:int, W:int, bins:int, t_min:float=None, t_max:float=None, polarity:bool=True):
        self.H, self.W, self.bins = int(H), int(W), int(bins)
        self.t_min, self.t_max = t_min, t_max
        self.polarity = bool(polarity)

    def __call__(self, x, y, info):
        ev = info.get("events", None)
        assert ev is not None, "EventToVoxel expects info['events']=[N,4]"
        # ev: [N,4] with (t,x,y,p), t in seconds (float32)

        # pull fields
        t  = ev[:, 0]
        xs = ev[:, 1].long().clamp_(0, self.W - 1)
        ys = ev[:, 2].long().clamp_(0, self.H - 1)
        ps = ev[:, 3].long().clamp_(0, 1)  # make LONG for indexing

        # handle empty events gracefully
        N = ev.shape[0]
        device = ev.device
        vox = torch.zeros((self.bins, self.H, self.W, 2 if self.polarity else 1),
                          dtype=torch.float32, device=device)

        if N == 0:
            return vox.view(self.bins, -1), y, info

        # time binning
        t0 = float(self.t_min if self.t_min is not None else (t.min().item()))
        t1 = float(self.t_max if self.t_max is not None else (t.max().item() + 1e-6))
        tb = ((t - t0) / (t1 - t0) * self.bins).long().clamp_(0, self.bins - 1)

        # values to add (must be a tensor, not an int)
        vals = torch.ones_like(tb, dtype=vox.dtype, device=device)

        if self.polarity:
            vox.index_put_((tb, ys, xs, ps), vals, accumulate=True)
        else:
            ch0 = torch.zeros_like(ps, dtype=torch.long, device=device)
            vox.index_put_((tb, ys, xs, ch0), vals, accumulate=True)

        return vox.view(self.bins, -1), y, info

class DownsampleEvents(Transform):
    """
    Downsample event coordinates by an integer factor.
    Assumes info['events'] is [N,4] = (t,x,y,p) and info['H'], info['W'] exist.
    """
    def __init__(self, factor: int):
        self.factor = int(factor)

    def __call__(self, x, y, info):
        ev = info.get("events", None)
        if ev is None:
            return x, y, info

        f = self.factor
        H = int(info.get("H", 128))
        W = int(info.get("W", 128))
        new_H = max(1, H // f)
        new_W = max(1, W // f)

        ev = ev.clone()
        # x, y in columns 1,2
        ev[:, 1] = torch.clamp((ev[:, 1] / f).floor(), 0, new_W - 1)
        ev[:, 2] = torch.clamp((ev[:, 2] / f).floor(), 0, new_H - 1)

        info = {**info, "events": ev, "H": new_H, "W": new_W}
        return x, y, info
