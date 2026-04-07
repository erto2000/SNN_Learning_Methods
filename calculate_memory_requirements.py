# calculate_memory_requirements.py
#
# Reads experiment configs from run_training.py (RUNS list), loads real dataset
# metadata by probing one transformed sample (fast, no full training needed),
# then saves memory graphs to results/memory/.
#
# Window-aware: if SlidingWindow / AdaptiveSlidingWindow is in the transform
# pipeline, meta["time_steps"] already reflects window length L — the model
# never sees the full sequence.
#
# Output layout:
#   results/memory/
#     overview_static_memory.png
#     overview_dynamic_memory.png
#     overview_total_memory.png
#     {dataset}/
#       1_static_vs_dynamic.png       base config, stacked bar per learner
#       2_memory_vs_hidden_size.png   sweep hidden width
#       3_memory_vs_batch_size.png    sweep batch size
#       4_memory_vs_time_steps.png    sweep window/sequence length
#       5_memory_vs_num_layers.png    sweep network depth

from __future__ import annotations
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, Any, Optional

from utils.training import build_cfg
from utils.runner import _make_learner, _infer_fp_bytes
from timeseries.registry import get_dataloaders
from timeseries.transforms import SlidingWindow, AdaptiveSlidingWindow, Compose
from run_training import RUNS, DEFAULT

# ── output / sweep constants ──────────────────────────────────────────────────
OUT_ROOT  = "./results/memory"

# Samples to load for metadata probing (enough for ZScore.fit; fast)
PROBE_MAX_SAMPLES = 64

HIDDEN_SIZES_SWEEP = [32, 64, 128, 256, 512, 1024]
BATCH_SIZES_SWEEP  = [8, 16, 32, 64, 128, 256, 512]
TIME_STEPS_SWEEP   = [16, 32, 64, 128, 256, 512, 1024]
NUM_LAYERS_SWEEP   = [1, 2, 3, 4, 5]

LEARNERS = ["bp", "ff", "eprop", "pepita"]
LEARNER_LABELS = {
    "bp":     "BPTT",
    "ff":     "Forward-Forward",
    "eprop":  "E-Prop",
    "pepita": "PEPITA",
}
LEARNER_COLORS = {
    "bp":     "#1f77b4",
    "ff":     "#ff7f0e",
    "eprop":  "#2ca02c",
    "pepita": "#d62728",
}
# Distinct line styles so overlapping curves remain visible
LEARNER_STYLES = {
    "bp":     {"linestyle": "-",    "marker": "o"},
    "ff":     {"linestyle": "--",   "marker": "s"},
    "eprop":  {"linestyle": "-.",   "marker": "^"},
    "pepita": {"linestyle": ":",    "marker": "D"},
}

# Fallback metadata used when a dataset cannot be loaded

# ── helpers ───────────────────────────────────────────────────────────────────

def _dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _mb(b: int) -> float:
    return b / (1024 ** 2)


def _dataset_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _detect_window(transform) -> tuple[bool, Optional[int]]:
    """
    Inspect a transform pipeline for SlidingWindow / AdaptiveSlidingWindow.
    Returns (is_windowed, L) where L is the window length if determinable.
    AdaptiveSlidingWindow requires .fit(), so L is None until it is fitted.
    """
    ops = []
    if isinstance(transform, Compose):
        ops = transform.ops
    elif transform is not None:
        ops = [transform]

    for op in ops:
        if isinstance(op, SlidingWindow):
            return True, op.L
        if isinstance(op, AdaptiveSlidingWindow):
            return True, op.L_global  # None until fitted
    return False, None


def _load_meta(g: dict) -> dict:
    """
    Load real dataset metadata by probing one transformed sample.
    meta["time_steps"] reflects window length when SlidingWindow is present.
    Raises on failure (dataset not downloaded, bad config, etc.).
    """
    _, _, meta = get_dataloaders(
        dataset=g["DATASET"],
        root=g.get("DATA_ROOT", "./data"),
        batch_size=1,
        max_samples=PROBE_MAX_SAMPLES,
        transform=g.get("TRANSFORM"),
        num_workers=0,
        pin_memory=False,
        seed=g.get("SEED", 123),
        **{k: v for k, v in g.get("DATASET_KW", {}).items()},
    )
    return meta


def _make_g(base_g: dict, learner: str, hidden_sizes: list[int]) -> dict:
    g = dict(base_g)
    g["LEARNER"]      = learner
    g["HIDDEN_SIZES"] = hidden_sizes
    return g


def _get_memory(
    base_g: dict,
    learner_name: str,
    meta: dict,
    hidden_sizes: list[int],
    batch: int,
    time_steps: int,
) -> tuple[int, int, int]:
    """Returns (static_bytes, dynamic_bytes, fp_bytes). Does NOT load any data."""
    g = _make_g(base_g, learner_name, hidden_sizes)
    cfg = build_cfg(meta["input_dim"], meta["n_classes"], g)
    m = dict(meta)
    m["time_steps"] = time_steps
    try:
        learner = _make_learner(cfg, m, torch.device("cpu"), g)
    except Exception as e:
        warnings.warn(f"Could not build {learner_name} with hidden={hidden_sizes}: {e}")
        return 0, 0, 4
    fp = _infer_fp_bytes(learner.model)
    static  = learner.get_static_memory_bytes(fp_bytes=fp)
    dynamic = learner.get_training_memory_bytes(batch=batch, time_steps=time_steps, fp_bytes=fp)
    return static, dynamic, fp


def _apply_style(ax, title: str, xlabel: str, ylabel: str = "Memory (MB)") -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.8)


# ── per-dataset graphs ────────────────────────────────────────────────────────

def _fp_label(fp_bytes: int) -> str:
    return {4: "fp32", 2: "fp16", 1: "int8"}.get(fp_bytes, f"{fp_bytes*8}-bit")


def plot_static_vs_dynamic(out_dir, ds_label, base_g, meta, base_hidden, base_batch, windowed) -> None:
    T = meta["time_steps"]
    statics, dynamics, fp_bytes = [], [], 4
    for ln in LEARNERS:
        s, d, fp = _get_memory(base_g, ln, meta, [base_hidden], base_batch, T)
        statics.append(_mb(s))
        dynamics.append(_mb(d))
        fp_bytes = fp

    labels = [LEARNER_LABELS[ln] for ln in LEARNERS]
    colors = [LEARNER_COLORS[ln] for ln in LEARNERS]
    x = np.arange(len(LEARNERS))
    w = 0.5

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.bar(x, statics,  w, label="Static (weights)",              color=colors, alpha=0.85)
    ax.bar(x, dynamics, w, bottom=statics, label="Dynamic (activations/traces)",
           color=colors, alpha=0.40, hatch="//", edgecolor="white")

    max_total = max(s + d for s, d in zip(statics, dynamics)) if statics else 1
    for i, (s, d) in enumerate(zip(statics, dynamics)):
        total = s + d
        ax.text(x[i], total + max_total * 0.01, f"{total:.3f} MB",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    t_label = f"window L={T}" if windowed else f"T={T}"
    ax.set_title(
        f"{ds_label} — Static vs Dynamic Memory\n"
        f"(hidden=[{base_hidden}], batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Memory (MB)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_static_vs_dynamic.png"))
    plt.close(fig)


def plot_vs_hidden_size(out_dir, ds_label, base_g, meta, base_batch, windowed) -> None:
    T = meta["time_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    ax_s, ax_d = axes
    fp_bytes = 4

    for ln in LEARNERS:
        vals_s, vals_d = [], []
        for h in HIDDEN_SIZES_SWEEP:
            s, d, fp = _get_memory(base_g, ln, meta, [h], base_batch, T)
            vals_s.append(_mb(s))
            vals_d.append(_mb(d))
            fp_bytes = fp
        kw = dict(linewidth=2, color=LEARNER_COLORS[ln], label=LEARNER_LABELS[ln],
                  **LEARNER_STYLES[ln])
        ax_s.plot(HIDDEN_SIZES_SWEEP, vals_s, **kw)
        ax_d.plot(HIDDEN_SIZES_SWEEP, vals_d, **kw)

    t_label = f"window L={T}" if windowed else f"T={T}"
    _apply_style(ax_s, "Static Memory vs Hidden Width", "Hidden layer width (neurons)")
    _apply_style(ax_d, "Dynamic Memory vs Hidden Width", "Hidden layer width (neurons)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(HIDDEN_SIZES_SWEEP)

    fig.suptitle(f"{ds_label} — Memory vs Hidden Size  (batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_memory_vs_hidden_size.png"))
    plt.close(fig)


def plot_vs_batch_size(out_dir, ds_label, base_g, meta, base_hidden, windowed) -> None:
    T = meta["time_steps"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    fp_bytes = 4

    for ln in LEARNERS:
        vals = []
        for b in BATCH_SIZES_SWEEP:
            _, d, fp = _get_memory(base_g, ln, meta, [base_hidden], b, T)
            vals.append(_mb(d))
            fp_bytes = fp
        ax.plot(BATCH_SIZES_SWEEP, vals, linewidth=2,
                color=LEARNER_COLORS[ln], label=LEARNER_LABELS[ln], **LEARNER_STYLES[ln])

    t_label = f"window L={T}" if windowed else f"T={T}"
    _apply_style(ax,
        f"{ds_label} — Dynamic Memory vs Batch Size\n(hidden=[{base_hidden}], {t_label}, {_fp_label(fp_bytes)})",
        "Batch size")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(BATCH_SIZES_SWEEP)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_memory_vs_batch_size.png"))
    plt.close(fig)


def plot_vs_time_steps(out_dir, ds_label, base_g, meta, base_hidden, base_batch, windowed) -> None:
    native_T = meta["time_steps"]
    xlabel = "Window length (L)" if windowed else "Sequence length (T)"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    fp_bytes = 4

    for ln in LEARNERS:
        vals = []
        for T in TIME_STEPS_SWEEP:
            _, d, fp = _get_memory(base_g, ln, meta, [base_hidden], base_batch, T)
            vals.append(_mb(d))
            fp_bytes = fp
        ax.plot(TIME_STEPS_SWEEP, vals, linewidth=2,
                color=LEARNER_COLORS[ln], label=LEARNER_LABELS[ln], **LEARNER_STYLES[ln])

    ax.axvline(native_T, color="gray", linestyle=":", linewidth=1.5,
               label=f"Data Length = {native_T}")
    _apply_style(ax,
        f"{ds_label} — Dynamic Memory vs {'Window' if windowed else 'Sequence'} Length\n"
        f"(hidden=[{base_hidden}], batch={base_batch}, {_fp_label(fp_bytes)})",
        xlabel)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_memory_vs_time_steps.png"))
    plt.close(fig)


def plot_vs_num_layers(out_dir, ds_label, base_g, meta, base_hidden, base_batch, windowed) -> None:
    T = meta["time_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    ax_s, ax_d = axes
    fp_bytes = 4

    for ln in LEARNERS:
        vals_s, vals_d = [], []
        for n in NUM_LAYERS_SWEEP:
            s, d, fp = _get_memory(base_g, ln, meta, [base_hidden] * n, base_batch, T)
            vals_s.append(_mb(s))
            vals_d.append(_mb(d))
            fp_bytes = fp
        kw = dict(linewidth=2, color=LEARNER_COLORS[ln], label=LEARNER_LABELS[ln],
                  **LEARNER_STYLES[ln])
        ax_s.plot(NUM_LAYERS_SWEEP, vals_s, **kw)
        ax_d.plot(NUM_LAYERS_SWEEP, vals_d, **kw)

    t_label = f"window L={T}" if windowed else f"T={T}"
    _apply_style(ax_s, "Static Memory vs Depth", "Number of hidden layers")
    _apply_style(ax_d, "Dynamic Memory vs Depth", "Number of hidden layers")
    for ax in axes:
        ax.set_xticks(NUM_LAYERS_SWEEP)

    fig.suptitle(
        f"{ds_label} — Memory vs Network Depth\n"
        f"(hidden={base_hidden}/layer, batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "5_memory_vs_num_layers.png"))
    plt.close(fig)


# ── cross-dataset overview graphs ─────────────────────────────────────────────

def plot_overview(all_results: dict, out_dir: str, component: str, fp_bytes: int = 4) -> None:
    """
    all_results[ds_label][learner] = (static_mb, dynamic_mb)
    component: "static" | "dynamic" | "total"
    """
    ds_names = list(all_results.keys())
    n_ds  = len(ds_names)
    n_ln  = len(LEARNERS)
    x     = np.arange(n_ds)
    w     = 0.18
    offsets = np.linspace(-(n_ln - 1) / 2, (n_ln - 1) / 2, n_ln) * w

    fig, ax = plt.subplots(figsize=(max(12, n_ds * 1.5), 6), dpi=140)

    for i, ln in enumerate(LEARNERS):
        vals = []
        for ds in ds_names:
            s, d = all_results[ds].get(ln, (0.0, 0.0))
            vals.append(s if component == "static" else d if component == "dynamic" else s + d)
        ax.bar(x + offsets[i], vals, w * 0.9,
               label=LEARNER_LABELS[ln], color=LEARNER_COLORS[ln], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Memory (MB)", fontsize=9)
    ax.set_title(f"All Datasets — {component.capitalize()} Memory per Learner  ({_fp_label(fp_bytes)})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"overview_{component}_memory.png"), bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def _group_runs_by_dataset(runs: list[dict]) -> dict[str, list[dict]]:
    """Group run configs by DATASET key, preserving insertion order."""
    groups: dict[str, list[dict]] = {}
    for g in runs:
        ds = g.get("DATASET", "unknown")
        groups.setdefault(ds, []).append(g)
    return groups


def main() -> None:
    _dir(OUT_ROOT)

    if not RUNS:
        print(
            "No runs configured. Enable at least one run in the RUNS list in run_training.py and try again."
        )
        return

    groups = _group_runs_by_dataset(RUNS)

    print(f"Output: {os.path.abspath(OUT_ROOT)}\n")

    all_results: dict[str, dict[str, tuple]] = {}
    global_fp_bytes = 4

    for dataset_key, run_list in groups.items():
        # Use the first run as the representative config for this dataset
        rep_g = run_list[0]
        ds_label = rep_g.get("RUN_ID", dataset_key).rsplit("-", 1)[0]  # strip learner suffix

        # Use the dataset key itself as label when all RUNS are present
        # (cleaner for multi-learner grouping)
        ds_label = dataset_key.upper().replace("_", " ")

        base_hidden = rep_g.get("HIDDEN_SIZES", [128])[0]  # first/only hidden layer width
        base_batch  = rep_g.get("BATCH_SIZE", 128)

        print(f"=== {ds_label} ===")

        # Detect windowing in the transform pipeline
        transform = rep_g.get("TRANSFORM")
        windowed, window_L = _detect_window(transform)

        # Load real metadata by probing one transformed sample
        meta = _load_meta(rep_g)
        print(f"  metadata: input_dim={meta['input_dim']}, "
              f"n_classes={meta['n_classes']}, time_steps={meta.get('time_steps', '?')}"
              + (f"  [windowed, L={meta.get('time_steps')}]" if windowed else ""))

        # Override window_L from probed meta when AdaptiveSlidingWindow was fitted
        if windowed and window_L is None:
            window_L = meta.get("time_steps")

        T = meta.get("time_steps") or 128

        # Compute base memory for all 4 learners
        all_results[ds_label] = {}
        ds_fp_bytes = 4
        for ln in LEARNERS:
            s, d, fp = _get_memory(rep_g, ln, meta, [base_hidden], base_batch, T)
            all_results[ds_label][ln] = (_mb(s), _mb(d))
            ds_fp_bytes = fp
            global_fp_bytes = fp
            print(f"    {LEARNER_LABELS[ln]:20s}  static={_mb(s):.4f} MB  dynamic={_mb(d):.4f} MB  [{_fp_label(fp)}]")

        out_dir = _dir(os.path.join(OUT_ROOT, _dataset_slug(dataset_key)))
        print(f"  Plotting …")
        plot_static_vs_dynamic(out_dir, ds_label, rep_g, meta, base_hidden, base_batch, windowed)
        plot_vs_hidden_size(out_dir, ds_label, rep_g, meta, base_batch, windowed)
        plot_vs_batch_size(out_dir, ds_label, rep_g, meta, base_hidden, windowed)
        plot_vs_time_steps(out_dir, ds_label, rep_g, meta, base_hidden, base_batch, windowed)
        plot_vs_num_layers(out_dir, ds_label, rep_g, meta, base_hidden, base_batch, windowed)
        print(f"  Saved: {out_dir}\n")

    print("Plotting cross-dataset overviews …")
    plot_overview(all_results, OUT_ROOT, "static",   global_fp_bytes)
    plot_overview(all_results, OUT_ROOT, "dynamic",  global_fp_bytes)
    plot_overview(all_results, OUT_ROOT, "total",    global_fp_bytes)
    print(f"Done. All graphs under {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()
