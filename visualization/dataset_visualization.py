# visualization/dataset_vizualization.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, Subset
import json

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_class_distribution(counts: List[int], class_names: List[str], path: str) -> str:
    fig, ax = plt.subplots(figsize=(8,4), dpi=140)
    idx = np.arange(len(counts))
    bars = ax.bar(idx, counts, color="#1f77b4")
    ax.set_xticks(idx)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("# samples")
    ax.set_title("Class distribution")
    ax.bar_label(bars, padding=2)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig); return path

def save_length_hist(lengths: List[int], path: str) -> str:
    if not lengths:
        return ""
    fig, ax = plt.subplots(figsize=(6,4), dpi=140)
    ax.hist(lengths, bins=30, color="#7f7f7f")
    ax.set_xlabel("T (time steps per sample)")
    ax.set_ylabel("Count")
    ax.set_title("Sequence length histogram (raw)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig); return path

def save_pad_ratio(pad_time: float, pad_seg: Optional[float], path: str) -> str:
    labels, vals = ["time_mask"], [pad_time*100.0]
    if pad_seg is not None:
        labels.append("seg_mask"); vals.append(pad_seg*100.0)
    fig, ax = plt.subplots(figsize=(5,3), dpi=140)
    bars = ax.bar(labels, vals, color=["#1f77b4","#2ca02c"][:len(vals)])
    ax.set_ylabel("Mean padding (%)")
    ax.set_ylim(0, max(5.0, max(vals)*1.2))
    ax.set_title("Padding utilization (post-pipeline)")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig); return path

def _iter_subset(sub: Subset, max_items: int = 48):
    n = min(len(sub), max_items)
    for i in range(n):
        yield sub[i]

def save_examples_har_traces(sub: Subset, class_names: List[str], path: str, overlay_segments: bool=False) -> str:
    # Expect x: [T,D] raw or [S,T,D] post (overlay_segments=True)
    rows, cols = 3, 3  # show up to 9 small multiples
    fig, axes = plt.subplots(rows, cols, figsize=(10,8), dpi=140, sharex=False)
    axes = axes.flatten()
    i = 0
    for x, y, _ in _iter_subset(sub, max_items=rows*cols):
        ax = axes[i]; i += 1
        if x.dim() == 2:
            ax.plot(x.cpu().numpy())
        elif x.dim() == 3:
            if overlay_segments:
                for s in range(min(x.shape[0], 6)):
                    ax.plot(x[s].cpu().numpy(), alpha=0.7)
            else:
                ax.plot(x[0].cpu().numpy())
        ax.set_title(class_names[y], fontsize=9)
        ax.grid(True, alpha=0.2)
    for k in range(i, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    fig.savefig(path); plt.close(fig); return path

def save_examples_waveforms(sub: Subset, class_names: List[str], path: str) -> str:
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(10,8), dpi=140, sharex=False)
    axes = axes.flatten()
    i=0
    for x, y, _ in _iter_subset(sub, max_items=rows*cols):
        ax = axes[i]; i += 1
        if x.dim()==2: ax.plot(x.squeeze(-1).cpu().numpy())
        elif x.dim()==3: ax.plot(x[0].squeeze(-1).cpu().numpy())
        ax.set_title(class_names[y], fontsize=9)
        ax.grid(True, alpha=0.2)
    for k in range(i, len(axes)): axes[k].axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig); return path

def save_examples_mnist_grid(sub: Subset, class_names: List[str], path: str) -> str:
    rows, cols = 4, 8
    fig, axes = plt.subplots(rows, cols, figsize=(10,6), dpi=140)
    axes = axes.flatten()
    i=0
    for x, y, _ in _iter_subset(sub, max_items=rows*cols):
        ax = axes[i]; i+=1
        # x: [1,784] or [T,784]
        if x.dim()==2:
            img = x[0].view(28,28).cpu().numpy()
        else:
            img = x.view(-1)[0:784].view(28,28).cpu().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(str(class_names[y]), fontsize=8)
        ax.axis("off")
    for k in range(i,len(axes)): axes[k].axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig); return path

def save_examples_mel_specs(sub: Subset, class_names: List[str], path: str) -> str:
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12,7), dpi=140)
    axes = axes.flatten()
    i=0
    for x, y, _ in _iter_subset(sub, max_items=rows*cols):
        ax = axes[i]; i+=1
        # x: [F, M] or [T,D] depending on pipeline. Expect [F,M] post-pipeline.
        X = x.cpu().numpy()
        if x.dim()==2 and X.shape[0] < X.shape[1]: # [F,M]
            im = ax.imshow(X, aspect="auto", origin="lower", cmap="magma")
        else:
            im = ax.imshow(X.T if X.ndim==2 else X[0].T, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(class_names[y], fontsize=9)
        ax.axis("off")
    for k in range(i, len(axes)): axes[k].axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig); return path

def save_examples_spike_raster(sub: Subset, class_names: List[str], path: str) -> str:
    # Works for rate-coded MNIST [T,D]
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12,7), dpi=140)
    axes = axes.flatten()
    i=0
    for x, y, _ in _iter_subset(sub, max_items=rows*cols):
        ax = axes[i]; i+=1
        if x.dim()==2:
            X = x.cpu().numpy()
            t, d = np.where(X>0.5)
            ax.scatter(t, d, s=1, alpha=0.6)
            rate = float(X.mean())
            ax.set_title(f"{class_names[y]} | rate={rate:.3f}", fontsize=8)
            ax.set_xlabel("t"); ax.set_ylabel("unit")
            ax.set_ylim(0, min(784, X.shape[1]))
            ax.grid(True, alpha=0.2)
        else:
            ax.axis("off")
    for k in range(i, len(axes)): axes[k].axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig); return path

def save_embeddings_scatter(Z: np.ndarray, y: np.ndarray, class_names: List[str], path: str) -> str:
    if Z.size == 0:
        return ""
    # ensure 2D (pad with zeros if rank-1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.shape[1] == 1:
        Z = np.concatenate([Z, np.zeros((Z.shape[0], 1), dtype=Z.dtype)], axis=1)

    fig, ax = plt.subplots(figsize=(7,6), dpi=140)
    uniq = np.unique(y)
    for k in uniq:
        mask = (y == k)
        if not np.any(mask):
            continue
        pts = Z[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=10, alpha=0.7,
            label=class_names[int(k) % len(class_names)],
            color=COLORS[int(k) % len(COLORS)]
        )
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA (post-pipeline features)")
    ax.grid(True, alpha=0.25)
    ax.legend(markerscale=2, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig); return path

def save_pipeline_summary(obj: Dict[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        import json; json.dump(obj, f, indent=2)
    return path

def save_counts_json(counts, class_names, json_path):
    total = int(sum(counts))
    rows = []
    for name, c in zip(class_names, counts):
        pct = (100.0 * c / total) if total > 0 else 0.0
        rows.append({"class": name, "count": int(c), "percent": pct})

    obj = {"total": total, "per_class": rows}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_examples_multichannel_traces(sub: Subset, class_names: List[str], path: str, overlay_segments: bool=False, max_channels: int = 6) -> str:
    """
    Generic multi-sensor line plots for [T,D] (or [S,T,D] if overlaying segments).
    For D large, we only plot the first 'max_channels' dims.
    """
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8), dpi=140, sharex=False)
    axes = axes.flatten()
    i = 0
    for x, y, _ in _iter_subset(sub, max_items=rows*cols):
        ax = axes[i]; i += 1
        if x.dim() == 2:
            X = x[:, :max_channels].cpu().numpy()
            ax.plot(X)
        elif x.dim() == 3:
            # overlay a few segments
            S = min(x.shape[0], 5)
            for s in range(S):
                X = x[s, :, :max_channels].cpu().numpy()
                ax.plot(X, alpha=0.7)
        ax.set_title(class_names[y], fontsize=9)
        ax.grid(True, alpha=0.2)
    for k in range(i, len(axes)): axes[k].axis("off")
    fig.tight_layout()
    fig.savefig(path); plt.close(fig); return path

def save_examples_voxel_slices(sub: Subset, class_names: List[str], path: str, H: int, W: int, bins_hint: int = 200) -> str:
    """
    For DVS-Gesture after EventToVoxel(+optional SlidingWindow): input can be
    [T, H*W*(1 or 2)] or [S, T, H*W*(1 or 2)]. We render a time slice (sum over polarity).
    """
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7), dpi=140)
    axes = axes.flatten()
    i = 0
    """
    For DVS-Gesture after EventToVoxel(+optional SlidingWindow):
    input can be [T, H*W*(1 or 2)] or [S, T, H*W*(1 or 2)].
    Renders a single time slice (sum over polarity).
    """
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7), dpi=140)
    axes = axes.flatten()
    i = 0

    for x, y, info in _iter_subset(sub, max_items=rows * cols):
        ax = axes[i]; i += 1

        # unwrap segments if present
        if x.dim() == 3:        # [S, T, D]
            x2 = x[0]
        elif x.dim() == 2:      # [T, D]
            x2 = x
        else:
            ax.axis("off")
            continue

        # --- FIX: get H/W from info if available ---
        H_ = int(info.get("H", H if H is not None else 128))
        W_ = int(info.get("W", W if W is not None else 128))

        T, D = x2.shape
        C = max(1, D // (H_ * W_))

        if H_ * W_ * C != D:
            ax.axis("off")
            continue

        X = x2.cpu().numpy().reshape(T, H_, W_, C)

        # collapse polarity/channel
        img = X.sum(axis=-1)   # [T, H, W]

        t_idx = min(T // 2, T - 1)
        ax.imshow(img[t_idx], cmap="magma", origin="lower")
        ax.set_title(f"{class_names[y]} | t={t_idx}/{T}", fontsize=8)
        ax.axis("off")

    for k in range(i, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_examples_dvs_events_raw(sub: Subset, class_names: List[str], path: str, max_points: int = 20_000) -> str:
    """
    Show DVS raw events as (t vs y) rasters colored by polarity, with a small x overlay.
    Expects items to have info['events'] as [N,4] (t,x,y,p) in seconds.
    """
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7), dpi=140)
    axes = axes.flatten()
    i = 0
    for _, y, info in _iter_subset(sub, max_items=rows*cols):
        ax = axes[i]; i += 1
        ev = info.get("events", None)
        if ev is None or ev.numel() == 0:
            ax.axis("off"); continue
        E = ev.cpu()
        # Subsample for speed if too many points
        if E.shape[0] > max_points:
            idx = torch.randint(0, E.shape[0], (max_points,))
            E = E.index_select(0, idx)
        t = E[:, 0].numpy()
        xs = E[:, 1].numpy()
        ys = E[:, 2].numpy()
        ps = E[:, 3].numpy()
        # Raster: time vs y, color by polarity
        ax.scatter(t, ys, s=0.3, c=np.where(ps > 0.5, "#d62728", "#1f77b4"), alpha=0.6)
        # Light x overlay to hint spatial spread (projected along a second axis)
        ax2 = ax.twinx()
        ax2.scatter(t, xs, s=0.3, c="#7f7f7f", alpha=0.2)
        ax.set_title(class_names[y], fontsize=8)
        ax.set_xlabel("time (s)"); ax.set_ylabel("y"); ax.grid(True, alpha=0.1)
        ax2.set_ylabel("x")
    for k in range(i, len(axes)): axes[k].axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig); return path
