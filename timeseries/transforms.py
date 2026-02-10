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


class AdaptiveSlidingWindow(Transform):
    """
    GLOBAL adaptive windowing based on autocorrelation.

    fit(): estimates ONE global L/hop from training data (prints once, optional plot)
    __call__(): uses fixed L/hop for all samples

    Output: [S, L, D]
    """

    def __init__(
        self,
        L_min: int = 10,
        L_max: int = 1000,
        hop_ratio: float = 0.5,
        summary: str = "energy",          # "energy" | "absmean" | "mean"
        downsample_to: int = 1024,
        plot_examples: bool = True,
        fit_samples: int = 256,           # how many train samples to estimate global params
        eps: float = 1e-8,
    ):
        self.L_min = int(L_min)
        self.L_max = int(L_max)
        self.hop_ratio = float(hop_ratio)
        self.summary = str(summary)
        self.downsample_to = int(downsample_to) if downsample_to is not None else None
        self.plot_examples = bool(plot_examples)
        self.fit_samples = int(fit_samples)
        self.eps = float(eps)

        self.L_global: int | None = None
        self.hop_global: int | None = None

        assert self.L_min > 0 and self.L_max >= self.L_min
        assert 0.0 < self.hop_ratio <= 1.0
        assert self.summary in ("energy", "absmean", "mean")

    # Compose.fit() will call this
    def needs_fit(self) -> bool:
        return True

    # ---------------- helpers ----------------
    def _summary_signal(self, x: torch.Tensor) -> torch.Tensor:
        if self.summary == "energy":
            return (x * x).mean(dim=1)
        if self.summary == "absmean":
            return x.abs().mean(dim=1)
        return x.mean(dim=1)

    def _downsample(self, s: torch.Tensor) -> torch.Tensor:
        if self.downsample_to is None:
            return s
        if s.numel() <= self.downsample_to:
            return s
        stride = max(1, s.numel() // self.downsample_to)
        return s[::stride]

    def _acf(self, s: torch.Tensor) -> torch.Tensor:
        # normalized autocorrelation, r[0]=1
        s = s.to(torch.float32)
        s = s - s.mean()
        n = s.numel()
        nfft = 1 << (2 * n - 1).bit_length()
        S = torch.fft.rfft(s, n=nfft)
        P = S * torch.conj(S)
        r = torch.fft.irfft(P, n=nfft)[:n].real
        r = r / (r[0] + self.eps)
        return r

    # ---------------- fit (global) ----------------
    @torch.no_grad()
    def fit(self, iterator, pre_ops=None):
        Ls: list[int] = []
        Hs: list[int] = []
        acfs_to_plot: list[torch.Tensor] = []

        max_samples = self.fit_samples

        for i, (x, _, _) in enumerate(iterator):
            if i >= max_samples:
                break
            if not isinstance(x, torch.Tensor) or x.dim() != 2:
                continue

            T = int(x.shape[0])

            s = self._summary_signal(x)
            s_ds = self._downsample(s)
            r = self._acf(s_ds)

            # ---- simple period estimate ----
            # first peak after lag 8 with r>0.25, else decay to <=0.15
            lag = None
            for j in range(8, max(9, len(r) - 1)):
                if r[j] > 0.25 and r[j] > r[j - 1] and r[j] >= r[j + 1]:
                    lag = int(j)
                    break
            if lag is None:
                hits = (r <= 0.15).nonzero(as_tuple=False)
                lag = int(hits[0].item()) if hits.numel() > 0 else max(1, len(r) // 4)

            # map ds-lag back to original units
            stride = max(1, T // int(s_ds.numel()))
            L = int(max(self.L_min, min(self.L_max, 2 * lag * stride, T)))
            hop = int(max(1, min(L, int(round(self.hop_ratio * L)))))

            Ls.append(L)
            Hs.append(hop)

            if self.plot_examples and len(acfs_to_plot) < 6:
                acfs_to_plot.append(r.detach().cpu())

        if not Ls:
            # fallback
            self.L_global = self.L_min
            self.hop_global = max(1, int(round(self.hop_ratio * self.L_global)))
            print("\n=== AdaptiveWindow GLOBAL PARAMETERS (fallback) ===")
            print(f"L_global = {self.L_global}")
            print(f"hop_global = {self.hop_global}")
            return self

        # FIX: compute mean in float space
        L_mean = sum(Ls) / float(len(Ls))
        H_mean = sum(Hs) / float(len(Hs))
        self.L_global = int(round(L_mean))
        self.hop_global = int(round(H_mean))

        print("\n=== AdaptiveWindow GLOBAL PARAMETERS ===")
        print(f"L_global = {self.L_global}")
        print(f"hop_global = {self.hop_global}")

        if self.plot_examples and acfs_to_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4))
            for r in acfs_to_plot:
                plt.plot(r.numpy(), alpha=0.7)
            plt.axvline(max(1, self.L_global // max(1, (Ls[0] // max(1, (len(acfs_to_plot[0]) if acfs_to_plot else 1))))), color="r", alpha=0.3)
            plt.title("Example autocorrelations (fit samples)")
            plt.xlabel("Lag (downsampled)")
            plt.ylabel("ACF")
            plt.tight_layout()
            plt.show()

        return self

    # ---------------- apply fixed window ----------------
    def __call__(self, x, y, info):
        assert x.dim() == 2, f"AdaptiveSlidingWindow expects [T,D], got {tuple(x.shape)}"
        assert self.L_global is not None and self.hop_global is not None, \
            "AdaptiveSlidingWindow not fit. Ensure it's inside Compose(...) so Compose.fit() runs."

        L = int(self.L_global)
        hop = int(self.hop_global)

        T, D = x.shape
        segs = []
        start = 0
        while start < T:
            seg = x[start:start + L]
            if seg.shape[0] < L:
                pad = x.new_zeros((L, D))
                pad[:seg.shape[0]] = seg
                seg = pad
            segs.append(seg)
            start += hop

        X = torch.stack(segs, dim=0) if segs else x.new_zeros((1, L, D))
        info = {**info, "adaptive_L": L, "adaptive_hop": hop}
        return X, y, info
