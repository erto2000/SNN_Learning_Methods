"""Plotting helpers for the dataset inspection report."""
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

def save_length_hist(lengths: List[int], path: str, *, events: bool = False) -> str:
    if not lengths:
        return ""
    fig, ax = plt.subplots(figsize=(6,4), dpi=140)
    ax.hist(lengths, bins=30, color="#7f7f7f")
    ax.set_xlabel("Events per sample" if events else "Time steps per sample")
    ax.set_ylabel("Count")
    ax.set_title("Raw event counts (sampled)" if events
                 else "Raw sequence lengths (sampled)")
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
    return save_examples_multichannel_traces(
        sub, class_names, path, overlay_segments=overlay_segments)


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
        json.dump(obj, f, indent=2, allow_nan=False)
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

def save_examples_multichannel_traces(sub: Subset, class_names: List[str], path: str, overlay_segments: bool=False, max_channels: Optional[int] = None) -> str:
    """
    Generic multi-sensor line plots for [T,D] (or [S,T,D] if overlaying segments).
    Show all channels unless max_channels explicitly limits them.
    """
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(11, 9), dpi=140)
    axes = axes.flatten()
    handles, names = [], []
    i = 0
    for x, y, info in _iter_subset(sub, max_items=rows * cols):
        ax = axes[i]
        i += 1
        segments = [x] if x.dim() == 2 else list(x[:5] if overlay_segments else x[:1])
        for segment_index, segment in enumerate(segments):
            values = segment[:, :max_channels].detach().cpu().numpy()
            labels = info.get("channel_names", [f"channel {j + 1}" for j in range(values.shape[1])])
            rate = info.get("sample_rate")
            time = np.arange(len(values)) / rate if rate else np.arange(len(values))
            for channel in range(values.shape[1]):
                line, = ax.plot(time, values[:, channel], lw=.8,
                                color=COLORS[channel % len(COLORS)],
                                linestyle=("-", "--", ":")[channel // len(COLORS) % 3],
                                alpha=.7 if overlay_segments else 1,
                                label=labels[channel])
                if i == 1 and segment_index == 0:
                    handles.append(line)
                    names.append(labels[channel])
        ax.set_title(class_names[y], fontsize=9)
        ax.set_xlabel("Time (s)" if info.get("sample_rate") else "Time step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=.2)
    for ax in axes[i:]:
        ax.axis("off")
    legend_height = .025 * int(np.ceil(len(names) / 3))
    if handles:
        fig.legend(handles, names, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, legend_height, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def save_examples_voxel_slices(sub: Subset, class_names: List[str], path: str, H: int, W: int, bins_hint: int = 200) -> str:
    """
    For DVS-Gesture after EventToVoxel(+optional SlidingWindow): input can be
    [T, H*W*(1 or 2)] or [S, T, H*W*(1 or 2)]. We render a time slice (sum over polarity).
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


DATASET_TITLES = {
    "har": "HAR", "mnist-static": "MNIST static", "mnist-rate": "MNIST rate",
    "sc": "Speech Commands", "esc50": "ESC-50", "urban8k": "UrbanSound8K",
    "pamap2": "PAMAP2", "mitbih": "MIT-BIH", "dvs": "DVS128 Gesture",
    "large-scale-audio": "Large-scale audio",
}


def _sensor_groups(dataset_id: str):
    """Channel groups in the exact order returned by the raw loaders."""
    if dataset_id == "har":
        return [("Body acceleration", "g"), ("Body gyroscope", "rad/s"),
                ("Total acceleration", "g")]
    return [(f"{location} {sensor}", unit)
            for location in ("Hand", "Chest", "Ankle")
            for sensor, unit in (("acceleration (±16g)", "m/s²"),
                                 ("gyroscope", "rad/s"), ("magnetometer", "µT"))]


def _plot_sensor_channels(ax, x, info, dataset_id, *, processed, title):
    """Separate sensor groups so traces with different physical units never mix."""
    values = x.detach().cpu().numpy()
    groups = _sensor_groups(dataset_id)
    if values.ndim != 2 or values.shape[1] != 3 * len(groups):
        raise ValueError(f"Expected {3 * len(groups)} channels for {dataset_id}")
    sample_rate = float(info.get("sample_rate", 50 if dataset_id == "har" else 100))
    time = np.arange(len(values)) / sample_rate
    spec = ax.get_subplotspec().subgridspec(len(groups), 1)
    fig = ax.figure
    ax.remove()
    first = None
    for group, (name, unit) in enumerate(groups):
        panel = fig.add_subplot(spec[group], sharex=first)
        if first is None:
            first = panel
        for channel, axis in enumerate(("x", "y", "z")):
            panel.plot(time, values[:, 3 * group + channel], lw=.85,
                       color=COLORS[channel], label=axis)
        panel.set_ylabel("Z-score" if processed else unit, fontsize=8)
        panel.text(.01, .95, name, transform=panel.transAxes,
                   va="top", fontsize=8,
                   bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1))
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(alpha=.2)
        panel.tick_params(labelsize=7, labelbottom=(group == len(groups) - 1))
        if group == 0:
            panel.set_title(title, loc="left", fontsize=10)
            panel.legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
        if group == len(groups) - 1:
            panel.set_xlabel("Time (s)", fontsize=9)


def _plot_dvs_sequence(ax, post, post_info, raw_info, label):
    """Three spatial snapshots with elapsed time and a shared brightness scale."""
    H, W = int(post_info["H"]), int(post_info["W"])
    voxel = post.detach().cpu().numpy().reshape(len(post), H, W, -1)
    indices = np.linspace(0, len(voxel) - 1, 3, dtype=int)
    frames = np.maximum(voxel[indices], 0).sum(axis=-1)
    vmax = max(1e-6, float(np.percentile(frames, 99.5)))
    events = raw_info["events"]
    duration = (float(events[:, 0].max()) - float(events[:, 0].min()) + 1e-6
                if len(events) else None)

    fig = ax.figure
    grid = ax.get_subplotspec().subgridspec(3, 3, height_ratios=[.18, 1, .22])
    ax.remove()
    heading = fig.add_subplot(grid[0, :])
    heading.set_axis_off()
    heading.text(0, .5, f"Processed sequence · {label}",
                 transform=heading.transAxes, va="center", fontsize=10)
    for column, (index, frame) in enumerate(zip(indices, frames)):
        panel = fig.add_subplot(grid[1, column])
        panel.imshow(frame, origin="upper", cmap="magma", vmin=0, vmax=vmax)
        timestamp = (f"t = {(index + .5) * duration / len(voxel):.2f} s\n"
                     if duration is not None else "")
        panel.set_title(f"{timestamp}Bin {index + 1}/{len(voxel)}", fontsize=8)
        panel.set_xticks([0, W // 2, W - 1])
        panel.set_yticks([0, H // 2, H - 1])
        panel.tick_params(labelsize=7, labelleft=(column == 0))
        panel.set_xlabel("x (pixel)", fontsize=8)
        if column == 0:
            panel.set_ylabel("y (pixel)", fontsize=8)
    timeline = fig.add_subplot(grid[2, :])
    timeline.set_axis_off()
    timeline.annotate("", xy=(.97, .85), xytext=(.03, .85),
                      xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.2))
    timeline.text(.5, .2, "Time from first event (s)" if duration is not None
                  else "Time bin", transform=timeline.transAxes,
                  ha="center", va="center", fontsize=9)


def _plot_example_pair(left, right, example, *, show_legend=True):
    dataset_id = example["dataset_id"]
    raw, raw_y, raw_info = example["raw"]
    post, post_y, post_info = example["post"]
    if int(raw_y) != int(post_y):
        raise ValueError("Preprocessing changed the example label")
    label = example["label"]
    audio = {"sc", "esc50", "urban8k", "large-scale-audio"}
    mel_image = None
    left.set_title(f"Raw · {label}", loc="left", fontsize=10)
    right.set_title(f"Processed · {label}", loc="left", fontsize=10)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)

    if dataset_id in {"har", "pamap2"}:
        _plot_sensor_channels(left, raw, raw_info, dataset_id, processed=False,
                              title=f"Raw · {label}")
        _plot_sensor_channels(right, post, post_info, dataset_id, processed=True,
                              title=f"Processed · {label}")
    elif dataset_id == "mitbih":
        rate = float(raw_info.get("sample_rate", raw_info.get("fs", 360)))
        for ax, x, processed in ((left, raw, False), (right, post, True)):
            values = x.detach().cpu().numpy()
            ax.plot(np.arange(len(values)) / rate, values[:, 0],
                    lw=1.1, color=COLORS[0])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Z-score" if processed else "ECG (mV)")
            ax.grid(alpha=.2)
    elif dataset_id in audio:
        waveform = raw.squeeze(-1).detach().cpu().numpy()
        sr = int(raw_info["sample_rate"])
        stride = max(1, len(waveform) // 4000)
        left.plot(np.arange(0, len(waveform), stride) / sr,
                  waveform[::stride], color=COLORS[0], lw=.7)
        left.set_xlabel("Time (s)")
        left.set_ylabel("Waveform amplitude")
        left.grid(alpha=.2)
        mel = post.detach().cpu().numpy()
        if mel.ndim != 2:
            raise ValueError("Expected [frames, mel bands] after audio preprocessing")
        mel_image = right.imshow(mel.T, aspect="auto", origin="lower",
                                 cmap="magma",
                                 extent=(0, len(mel), 0, mel.shape[1]),
                                 vmin=-3, vmax=3)
        right.set_xlabel("Log-mel frame (10 ms hop)")
        right.set_ylabel("Mel band")
    elif dataset_id.startswith("mnist"):
        image = raw[0].reshape(28, 28).detach().cpu().numpy()
        left.imshow(image, cmap="gray", vmin=0, vmax=1)
        left.set_xlabel("Pixel column")
        left.set_ylabel("Pixel row")
        if dataset_id == "mnist-static":
            steps = [0, len(post) // 2, len(post) - 1]
            frames = [post[t].reshape(28, 28).detach().cpu().numpy()
                      for t in steps]
            right.imshow(np.concatenate(frames, axis=1), cmap="gray",
                         vmin=0, vmax=1)
            right.set_xticks([14, 42, 70], [f"t={t + 1}" for t in steps])
            right.set_yticks([])
            right.set_xlabel("Identical image repeated over time")
        else:
            spikes = post.detach().cpu().numpy()
            steps, pixels = np.nonzero(spikes > .5)
            right.scatter(steps, pixels, s=.8, color=COLORS[0], rasterized=True)
            right.set_xlim(-.5, len(spikes) - .5)
            right.set_ylim(0, spikes.shape[1])
            right.set_xlabel("Time step")
            right.set_ylabel("Pixel index with spike")
    elif dataset_id == "dvs":
        left.set_title(f"Raw events (all times) · {label}", loc="left", fontsize=10)
        events = raw_info["events"].detach().cpu().numpy()
        stride = max(1, len(events) // 8000)
        shown = events[::stride]
        left.scatter(shown[:, 1], shown[:, 2], s=.35, alpha=.25,
                     c=np.where(shown[:, 3] > .5, COLORS[3], COLORS[0]),
                     rasterized=True)
        left.set_xlim(0, int(raw_info.get("W", 128)))
        left.set_ylim(int(raw_info.get("H", 128)), 0)
        left.set_aspect("equal")
        left.set_xlabel("Sensor x (pixel)")
        left.set_ylabel("Sensor y (pixel)")
        if show_legend:
            left.scatter([], [], color=COLORS[3], label="positive polarity")
            left.scatter([], [], color=COLORS[0], label="negative polarity")
            left.legend(loc="upper right", frameon=False, fontsize=7)
        _plot_dvs_sequence(right, post, post_info, raw_info, label)
    else:
        raise ValueError(f"No thesis example renderer for {dataset_id}")

    return mel_image


def _example_height(dataset_id):
    return {"har": 4.2, "pamap2": 9.0}.get(dataset_id, 3.2)


def _save_example_figure(examples, path, title, *, dataset_headings):
    if not examples:
        raise ValueError("At least one dataset example is required")
    heights = [_example_height(item["dataset_id"]) for item in examples]
    fig = plt.figure(figsize=(12, sum(heights) + .6), layout="constrained")
    grid = fig.add_gridspec(len(examples), 2, height_ratios=heights)
    try:
        for row, example in enumerate(examples):
            left = fig.add_subplot(grid[row, 0])
            right = fig.add_subplot(grid[row, 1])
            if dataset_headings:
                name = DATASET_TITLES[example["dataset_id"]]
                example = {**example, "label": f"{name} · {example['label']}"}
            mel = _plot_example_pair(left, right, example, show_legend=(row == 0 or dataset_headings))
            if mel is not None:
                fig.colorbar(mel, ax=right, shrink=.72, pad=.02,
                             label="Normalized log-mel value")
        fig.suptitle(title, fontsize=13, weight="bold")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        fig.savefig(os.path.splitext(path)[0] + ".svg", bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def save_thesis_examples(raw_ds: Dataset, post_ds: Dataset, indices: List[int],
                         class_names: List[str], dataset_id: str, path: str) -> str:
    """Show the same two training records before and after preprocessing."""
    if len(indices) != 2:
        raise ValueError("Thesis comparison requires two training indices")
    examples = []
    for index in indices:
        raw, post = raw_ds[index], post_ds[index]
        examples.append(dict(dataset_id=dataset_id, raw=raw, post=post,
                             label=class_names[int(raw[1])], train_index=index))
    return _save_example_figure(
        examples, path, f"{DATASET_TITLES[dataset_id]}: input and model representation",
        dataset_headings=False)


def save_combined_examples(examples: List[Dict[str, Any]], path: str,
                           title: str = "Dataset examples: input and model representation") -> str:
    """One training record per dataset, with matching raw/processed columns."""
    return _save_example_figure(examples, path, title, dataset_headings=True)
