# visualize.py
# Unified visualization for batches from datasets.py loaders
# Edit the parameters below and run this file directly.

# ── USER PARAMETERS ────────────────────────────────────────────────────────────
DATASET            = "har"  # "har","wisdm","speech_commands","mnist","mnist_rate"
ROOT               = "../data"
BATCH_SIZE         = 16
NUM_WORKERS        = 2
WINDOW_LEN         = 128            # used by WISDM via get_dataloaders
MNIST_T_STEPS      = 50             # used by MNIST loaders via get_dataloaders
MAX_SAMPLES        = None           # set to None to discover all labels

# Visualization controls
SHOW_EVERY_LABEL   = True           # <<< ensures one tile per class using a tiny curated batch
N_SAMPLES_TO_SHOW  = 8              # ignored when SHOW_EVERY_LABEL=True (uses all classes)
GRID_COLS          = 6              # auto-adjusted when SHOW_EVERY_LABEL=True
TIME_SLICES_IMAGES = 6              # how many MNIST frames to mosaic per sample

# Optional channel names for line plots (HAR/WISDM). Leave None to auto-generate.
HAR_CHANNEL_NAMES   = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","tacc_x","tacc_y","tacc_z"]
WISDM_CHANNEL_NAMES = ["acc_x","acc_y","acc_z"]
OUT_PATH            = None          # e.g., "./viz_out/batch.png" to save; None to show
# ───────────────────────────────────────────────────────────────────────────────

from typing import Optional, List, Tuple
import math
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["font.size"] = 10


def _infer_mode(input_dim: int) -> str:
    """Choose a visualization mode based on feature/channel dimension D."""
    if input_dim == 28 * 28:
        return "image_seq"        # MNIST variants
    if input_dim >= 40:
        return "heatmap_seq"      # log-mel or other high-D per step
    return "multichannel_lines"   # classic low-D sensor streams


def _label_to_str(y: int, class_names: Optional[List[str]]) -> str:
    if class_names and 0 <= int(y) < len(class_names):
        return class_names[int(y)]
    return str(int(y))


def _plot_image_sequence(ax, seq_txd, T_show: int, title: str = ""):
    """
    seq_txd: [T, 784] (float tensor/ndarray in [0,1] ideally)
    Show a few time steps as small images left→right.
    """
    T = seq_txd.shape[0]
    idxs = np.linspace(0, T - 1, num=T_show, dtype=int) if T_show < T else np.arange(T)
    tiles = []
    for t in idxs:
        img = seq_txd[t].reshape(28, 28)
        tiles.append(img)
    mosaic = np.concatenate(tiles, axis=1)  # [28, 28*T_show]
    im = ax.imshow(mosaic, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return im


def _plot_heatmap(ax, seq_txd, title: str = "", ylabel: str = "Features"):
    """
    seq_txd: [T, D] with D relatively large (e.g., 64 mels). Plotted as T×D image.
    """
    data = seq_txd.T  # D x T for "freq × time"
    im = ax.imshow(data, origin="lower", aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return im


def _plot_multichannel(ax, seq_txd, title: str = "", ch_names: Optional[List[str]] = None):
    """
    seq_txd: [T, D] with small D (e.g., 3–9). Line plot with per-channel offsets.
    """
    x = np.arange(seq_txd.shape[0])
    D = seq_txd.shape[1]
    y = seq_txd.astype(np.float32)
    std = y.std(axis=0, keepdims=True) + 1e-8
    y_norm = (y - y.mean(axis=0, keepdims=True)) / std

    offset = 2.0
    colors = plt.cm.tab10.colors
    for c in range(D):
        ax.plot(x, y_norm[:, c] + offset * c, color=colors[c % len(colors)], lw=1.0,
                label=(ch_names[c] if ch_names and c < len(ch_names) else f"ch{c}"))
    ax.set_xlim(0, len(x) - 1)
    ax.set_yticks([])  # stacked look
    ax.set_xlabel("Time step")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)


def visualize_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    meta: dict,
    n_samples: int = 8,
    cols: int = 4,
    out_path: Optional[str] = None,
    time_slices_for_images: int = 6,
    channel_names: Optional[List[str]] = None,
):
    """
    Render a grid of examples from a batch produced by your loaders.

    Args:
        batch: (X, y) where X is [B, T, D], y is [B]
        meta: dict with keys: n_classes, input_dim, time_steps, class_names
        n_samples: max samples to display from the batch
        cols: grid columns
        out_path: if provided, saves PNG to this path; otherwise shows.
        time_slices_for_images: for MNIST-like sequences, number of frames to mosaic.
        channel_names: optional names for multichannel signals (HAR/WISDM).
    """
    X, y = batch
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    B, T, D = X.shape
    n = min(n_samples, B)
    rows = math.ceil(n / cols)
    mode = _infer_mode(int(meta.get("input_dim", D)))
    class_names = meta.get("class_names", None)

    # Prepare figure
    W = 4.2 if mode != "image_seq" else 5.2
    H = 2.8 if mode != "image_seq" else 2.6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * W, rows * H), squeeze=False)

    heatmaps = []
    for i in range(n):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        title = _label_to_str(y[i], class_names)
        seq = X[i]

        if mode == "image_seq":
            _plot_image_sequence(ax, seq, time_slices_for_images, title=title)
        elif mode == "heatmap_seq":
            im = _plot_heatmap(ax, seq, title=title, ylabel=("Mel bins" if D >= 40 else "Features"))
            heatmaps.append(im)
        else:  # multichannel_lines
            _plot_multichannel(ax, seq, title=title, ch_names=channel_names)

    # Hide any unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    # Single colorbar for heatmaps (if any)
    if heatmaps:
        cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
        fig.colorbar(heatmaps[0], cax=cax)

    fig.suptitle(f"Unified visualization • mode={mode} • T={T}, D={D}", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96] if heatmaps else [0, 0, 1, 0.96])

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# Build a tiny batch with exactly one example per class (for previews)
# ───────────────────────────────────────────────────────────────────────────────

def sample_one_per_class(
    loader: torch.utils.data.DataLoader,
    meta: dict,
    prefer_non_silence: bool = False,
):
    """
    Build a tiny batch with exactly one example per class name in meta['class_names'].
    For Speech Commands, we synthesize a 'silence' example if none exists.

    Returns (X, y): X is [K, T, D], y is [K]
    """
    class_names = list(meta.get("class_names", []))
    assert class_names, "meta['class_names'] missing"

    ds = loader.dataset
    # Infer (T, D) by grabbing one item safely
    x0, _ = ds[0] if len(ds) > 0 else (torch.zeros(meta["time_steps"], meta["input_dim"]), 0)
    if isinstance(x0, torch.Tensor):
        T, D = int(x0.shape[0]), int(x0.shape[1])
    else:
        T, D = int(meta["time_steps"]), int(meta["input_dim"])

    want = {name: None for name in class_names}
    if prefer_non_silence and "silence" in want:
        want.pop("silence")

    # Iterate through underlying dataset until we get one per class (except 'silence')
    got = set()
    for i in range(len(ds)):
        xi, yi = ds[i]
        yi = int(yi)
        cname = class_names[yi] if 0 <= yi < len(class_names) else str(yi)
        if cname in want and want[cname] is None:
            want[cname] = xi.detach().cpu() if isinstance(xi, torch.Tensor) else torch.as_tensor(xi)
            got.add(cname)
            if len(got) == len(want):
                break

    # Synthesize 'silence' if listed but not found
    if "silence" in class_names and ("silence" not in want or want["silence"] is None):
        sil_feat = torch.zeros(T, D, dtype=torch.float32)
        want["silence"] = sil_feat

    # Pack results in class order; skip truly missing (very rare unless tiny subsets)
    xs, ys = [], []
    for idx, cname in enumerate(class_names):
        xi = want.get(cname, None)
        if xi is None:
            continue
        xs.append(xi.unsqueeze(0))
        ys.append(torch.tensor([idx], dtype=torch.long))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return (X, y)


# -----------------------------
# Minimal main using constants
# -----------------------------
if __name__ == "__main__":
    from datasets import get_dataloaders  # your datasets.py module

    train_loader, _, meta = get_dataloaders(
        dataset=DATASET,
        root=ROOT,
        batch_size=BATCH_SIZE,
        window_len=WINDOW_LEN,
        num_workers=NUM_WORKERS,
        max_samples=MAX_SAMPLES,
        mnist_T_steps=MNIST_T_STEPS,
    )

    # Choose batch
    if SHOW_EVERY_LABEL:
        batch = sample_one_per_class(train_loader, meta)
        # auto-fit the grid to class count
        N_SAMPLES_TO_SHOW = len(meta["class_names"])
        GRID_COLS = max(3, min(8, int(math.ceil(math.sqrt(N_SAMPLES_TO_SHOW)))))
    else:
        batch = next(iter(train_loader))

    ch_names = None
    if DATASET == "har":
        ch_names = HAR_CHANNEL_NAMES
    elif DATASET == "wisdm":
        ch_names = WISDM_CHANNEL_NAMES

    visualize_batch(
        batch=batch,
        meta=meta,
        n_samples=N_SAMPLES_TO_SHOW,
        cols=GRID_COLS,
        out_path=OUT_PATH,
        time_slices_for_images=TIME_SLICES_IMAGES,
        channel_names=ch_names,
    )
