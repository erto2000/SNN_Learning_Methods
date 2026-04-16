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
#     overview_training_memory.png
#     overview_base_memory.csv
#     {dataset}/
#       1_training_memory.png
#       2_training_memory_vs_hidden_size.png
#       3_training_memory_vs_batch_size.png  (saved as 4_memory_vs_batch_size.png)
#       4_training_memory_vs_time_steps.png  (saved as 5_memory_vs_time_steps.png)
#       5_training_memory_vs_num_layers.png  (saved as 6_training_memory_vs_num_layers.png)
#       0_base_memory.csv
#       2_training_memory_vs_hidden_size.csv
#       4_dynamic_memory_vs_batch_size.csv
#       5_dynamic_memory_vs_time_steps.csv
#       6_training_memory_vs_num_layers.csv

from __future__ import annotations
import os
import csv
import warnings
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
OUT_ROOT = "./results/memory"

# Overview graphs only
OVERVIEW_LOG_SCALE = True
OVERVIEW_LOG_EPS_MB = 1e-6  # prevents log(0) issues for empty bars

# Per-dataset graphs
PER_DATASET_LOG_SCALE = True
PER_DATASET_LOG_EPS_MB = 1e-6

# Per-dataset numeric x-axis graphs only
PER_DATASET_LOG_X = True

# Samples to load for metadata probing (enough for ZScore.fit; fast)
PROBE_MAX_SAMPLES = 64

HIDDEN_SIZES_SWEEP = [32, 64, 128, 256, 512, 1024]
BATCH_SIZES_SWEEP = [8, 16, 32, 64, 128, 256, 512]
TIME_STEPS_SWEEP = [16, 32, 64, 128, 256, 512, 1024]
NUM_LAYERS_SWEEP = [1, 10, 100, 1000]

LEARNERS = ["bp", "ff", "eprop", "pepita"]
LEARNER_LABELS = {
    "bp": "BPTT",
    "ff": "Forward-Forward",
    "eprop": "E-Prop",
    "pepita": "PEPITA",
}
LEARNER_COLORS = {
    "bp": "#1f77b4",
    "ff": "#ff7f0e",
    "eprop": "#2ca02c",
    "pepita": "#d62728",
}
LEARNER_STYLES = {
    "bp": {"linestyle": "-", "marker": "o"},
    "ff": {"linestyle": "--", "marker": "s"},
    "eprop": {"linestyle": "-.", "marker": "^"},
    "pepita": {"linestyle": ":", "marker": "D"},
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _kb(b: int) -> float:
    return b / 1024


def _dataset_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _fp_label(fp_bytes: int) -> str:
    return {4: "fp32", 2: "fp16", 1: "int8"}.get(fp_bytes, f"{fp_bytes*8}-bit")


def _csv_write(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    _dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            return True, op.L_global
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
    g["LEARNER"] = learner
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
    """Returns (param_bytes, training_bytes, fp_bytes). Does NOT load any data."""
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
    param    = learner.get_param_memory_bytes(fp_bytes=fp)
    training = learner.get_training_memory_bytes(batch=batch, time_steps=time_steps, fp_bytes=fp)
    return param, training, fp


def _apply_style(ax, xlabel: str, ylabel: str = "Memory (KB)") -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.8)


def _apply_numeric_x_log(ax, ticks: list[int]) -> None:
    if PER_DATASET_LOG_X:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(ticks)


def _base_context_row(
    ds_label: str,
    learner_key: str,
    fp_bytes: int,
    base_hidden: int,
    base_batch: int,
    time_steps: int,
    windowed: bool,
    meta: dict,
) -> dict:
    return {
        "dataset": ds_label,
        "learner_key": learner_key,
        "learner_label": LEARNER_LABELS[learner_key],
        "precision": _fp_label(fp_bytes),
        "fp_bytes": fp_bytes,
        "base_hidden": base_hidden,
        "base_batch": base_batch,
        "time_steps": time_steps,
        "windowed": windowed,
        "input_dim": meta.get("input_dim"),
        "n_classes": meta.get("n_classes"),
    }


# ── per-dataset graphs + csv ──────────────────────────────────────────────────

def plot_static_vs_dynamic(out_dir, ds_label, base_g, meta, base_hidden, base_batch, windowed) -> None:
    T = meta["time_steps"]
    totals, fp_bytes = [], 4
    csv_rows = []

    for ln in LEARNERS:
        p, t, fp = _get_memory(base_g, ln, meta, [base_hidden], base_batch, T)
        total_kb = _kb(t)
        totals.append(total_kb)
        fp_bytes = fp

        row = _base_context_row(ds_label, ln, fp, base_hidden, base_batch, T, windowed, meta)
        row["training_kb"] = total_kb
        row["training_bytes"] = t
        csv_rows.append(row)

    _csv_write(
        os.path.join(out_dir, "0_base_memory.csv"),
        csv_rows,
        [
            "dataset", "learner_key", "learner_label",
            "precision", "fp_bytes",
            "base_hidden", "base_batch", "time_steps", "windowed",
            "input_dim", "n_classes",
            "training_bytes", "training_kb",
        ],
    )

    labels = [LEARNER_LABELS[ln] for ln in LEARNERS]
    colors = [LEARNER_COLORS[ln] for ln in LEARNERS]
    x = np.arange(len(LEARNERS))
    w = 0.5

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.bar(x, totals, w, color=colors, alpha=0.85)

    max_val = max(totals) if totals else 1
    for i, v in enumerate(totals):
        ax.text(x[i], v + max_val * 0.01, f"{v:.1f} KB",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    t_label = f"window L={T}" if windowed else f"T={T}"
    ax.set_title(
        f"{ds_label} — Training Memory\n"
        f"(hidden=[{base_hidden}], batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("Memory (KB)", fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_static_vs_dynamic.png"))
    plt.close(fig)


def plot_vs_hidden_size(out_dir, ds_label, base_g, meta, base_batch, windowed) -> None:
    T = meta["time_steps"]
    fp_bytes = 4
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    for ln in LEARNERS:
        vals = []
        for h in HIDDEN_SIZES_SWEEP:
            p, t, fp = _get_memory(base_g, ln, meta, [h], base_batch, T)
            total_kb = _kb(t)
            v = max(total_kb, PER_DATASET_LOG_EPS_MB) if PER_DATASET_LOG_SCALE else total_kb
            vals.append(v)
            fp_bytes = fp

            rows.append({
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "hidden_size": h,
                "batch_size": base_batch,
                "time_steps": T,
                "windowed": windowed,
                "precision": _fp_label(fp),
                "fp_bytes": fp,
                "training_bytes": t,
                "training_kb": total_kb,
            })

        ax.plot(HIDDEN_SIZES_SWEEP, vals, linewidth=2,
                color=LEARNER_COLORS[ln], label=LEARNER_LABELS[ln], **LEARNER_STYLES[ln])

    _csv_write(
        os.path.join(out_dir, "2_training_memory_vs_hidden_size.csv"),
        rows,
        [
            "dataset", "learner_key", "learner_label",
            "hidden_size", "batch_size", "time_steps", "windowed",
            "precision", "fp_bytes",
            "training_bytes", "training_kb",
        ],
    )

    _apply_style(ax, "Hidden layer width (neurons)")
    _apply_numeric_x_log(ax, HIDDEN_SIZES_SWEEP)

    if PER_DATASET_LOG_SCALE:
        ax.set_yscale("log")

    t_label = f"window L={T}" if windowed else f"T={T}"
    fig.suptitle(
        f"{ds_label} — Training Memory vs Hidden Size  (batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_training_memory_vs_hidden_size.png"))
    plt.close(fig)


def plot_vs_batch_size(out_dir, ds_label, base_g, meta, base_hidden, windowed) -> None:
    T = meta["time_steps"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    fp_bytes = 4
    rows = []

    for ln in LEARNERS:
        vals = []
        for b in BATCH_SIZES_SWEEP:
            p, t, fp = _get_memory(base_g, ln, meta, [base_hidden], b, T)
            total_kb = _kb(t)
            v = max(total_kb, PER_DATASET_LOG_EPS_MB) if PER_DATASET_LOG_SCALE else total_kb
            vals.append(v)
            fp_bytes = fp

            rows.append({
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "hidden_size": base_hidden,
                "batch_size": b,
                "time_steps": T,
                "windowed": windowed,
                "precision": _fp_label(fp),
                "fp_bytes": fp,
                "training_bytes": t,
                "training_kb": total_kb,
            })

        ax.plot(
            BATCH_SIZES_SWEEP,
            vals,
            linewidth=2,
            color=LEARNER_COLORS[ln],
            label=LEARNER_LABELS[ln],
            **LEARNER_STYLES[ln],
        )

    _csv_write(
        os.path.join(out_dir, "4_dynamic_memory_vs_batch_size.csv"),
        rows,
        [
            "dataset", "learner_key", "learner_label",
            "hidden_size", "batch_size", "time_steps", "windowed",
            "precision", "fp_bytes",
            "training_bytes", "training_kb",
        ],
    )

    t_label = f"window L={T}" if windowed else f"T={T}"
    ax.set_title(
        f"{ds_label} — Training Memory vs Batch Size\n(hidden=[{base_hidden}], {t_label}, {_fp_label(fp_bytes)})",
        fontsize=11, fontweight="bold",
    )
    _apply_style(ax, "Batch size")
    _apply_numeric_x_log(ax, BATCH_SIZES_SWEEP)

    if PER_DATASET_LOG_SCALE:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_memory_vs_batch_size.png"))
    plt.close(fig)


def plot_vs_time_steps(out_dir, ds_label, base_g, meta, base_hidden, base_batch, windowed) -> None:
    native_T = meta["time_steps"]
    xlabel = "Window length (L)" if windowed else "Sequence length (T)"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    fp_bytes = 4
    rows = []

    for ln in LEARNERS:
        vals = []
        for T in TIME_STEPS_SWEEP:
            p, t, fp = _get_memory(base_g, ln, meta, [base_hidden], base_batch, T)
            total_kb = _kb(t)
            v = max(total_kb, PER_DATASET_LOG_EPS_MB) if PER_DATASET_LOG_SCALE else total_kb
            vals.append(v)
            fp_bytes = fp

            rows.append({
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "hidden_size": base_hidden,
                "batch_size": base_batch,
                "time_steps": T,
                "native_time_steps": native_T,
                "windowed": windowed,
                "precision": _fp_label(fp),
                "fp_bytes": fp,
                "training_bytes": t,
                "training_kb": total_kb,
            })

        ax.plot(
            TIME_STEPS_SWEEP,
            vals,
            linewidth=2,
            color=LEARNER_COLORS[ln],
            label=LEARNER_LABELS[ln],
            **LEARNER_STYLES[ln],
        )

    _csv_write(
        os.path.join(out_dir, "5_dynamic_memory_vs_time_steps.csv"),
        rows,
        [
            "dataset", "learner_key", "learner_label",
            "hidden_size", "batch_size",
            "time_steps", "native_time_steps", "windowed",
            "precision", "fp_bytes",
            "training_bytes", "training_kb",
        ],
    )

    ax.axvline(
        native_T,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Data Length = {native_T}",
    )
    ax.set_title(
        f"{ds_label} — Training Memory vs {'Window' if windowed else 'Sequence'} Length\n"
        f"(hidden=[{base_hidden}], batch={base_batch}, {_fp_label(fp_bytes)})",
        fontsize=11, fontweight="bold",
    )
    _apply_style(ax, xlabel)
    _apply_numeric_x_log(ax, TIME_STEPS_SWEEP)

    if PER_DATASET_LOG_SCALE:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "5_memory_vs_time_steps.png"))
    plt.close(fig)


def plot_vs_num_layers(out_dir, ds_label, base_g, meta, base_hidden, base_batch, windowed) -> None:
    T = meta["time_steps"]
    fp_bytes = 4
    t_label = f"window L={T}" if windowed else f"T={T}"
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    for ln in LEARNERS:
        vals = []
        for n in NUM_LAYERS_SWEEP:
            p, t, fp = _get_memory(base_g, ln, meta, [base_hidden] * n, base_batch, T)
            total_kb = _kb(t)
            v = max(total_kb, PER_DATASET_LOG_EPS_MB) if PER_DATASET_LOG_SCALE else total_kb
            vals.append(v)
            fp_bytes = fp

            rows.append({
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "num_layers": n,
                "hidden_size_per_layer": base_hidden,
                "batch_size": base_batch,
                "time_steps": T,
                "windowed": windowed,
                "precision": _fp_label(fp),
                "fp_bytes": fp,
                "training_bytes": t,
                "training_kb": total_kb,
            })

        ax.plot(NUM_LAYERS_SWEEP, vals, linewidth=2,
                color=LEARNER_COLORS[ln], label=LEARNER_LABELS[ln], **LEARNER_STYLES[ln])

    _csv_write(
        os.path.join(out_dir, "6_training_memory_vs_num_layers.csv"),
        rows,
        [
            "dataset", "learner_key", "learner_label",
            "num_layers", "hidden_size_per_layer", "batch_size", "time_steps", "windowed",
            "precision", "fp_bytes",
            "training_bytes", "training_kb",
        ],
    )

    scale_label = "Log Scale" if PER_DATASET_LOG_SCALE else "Linear Scale"
    _apply_style(ax, "Number of hidden layers")
    _apply_numeric_x_log(ax, NUM_LAYERS_SWEEP)

    if PER_DATASET_LOG_SCALE:
        ax.set_yscale("log")

    fig.suptitle(
        f"{ds_label} — Training Memory vs Network Depth\n"
        f"(hidden={base_hidden}/layer, batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "6_training_memory_vs_num_layers.png"))
    plt.close(fig)


# ── cross-dataset overview graphs + csv ──────────────────────────────────────

def write_overview_csv(all_results: dict, out_dir: str, fp_bytes: int = 4) -> None:
    rows = []
    for ds_label, learner_map in all_results.items():
        for ln in LEARNERS:
            training_kb = learner_map.get(ln, 0.0)
            rows.append({
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "precision": _fp_label(fp_bytes),
                "fp_bytes": fp_bytes,
                "training_kb": training_kb,
            })

    _csv_write(
        os.path.join(out_dir, "overview_base_memory.csv"),
        rows,
        [
            "dataset", "learner_key", "learner_label",
            "precision", "fp_bytes",
            "training_kb",
        ],
    )


def plot_overview(all_results: dict, out_dir: str, fp_bytes: int = 4) -> None:
    """all_results[ds_label][learner] = training_kb (total training memory)."""
    ds_names = list(all_results.keys())
    n_ds = len(ds_names)
    n_ln = len(LEARNERS)
    x = np.arange(n_ds)
    w = 0.18
    offsets = np.linspace(-(n_ln - 1) / 2, (n_ln - 1) / 2, n_ln) * w

    fig, ax = plt.subplots(figsize=(max(12, n_ds * 1.5), 6), dpi=140)

    for i, ln in enumerate(LEARNERS):
        vals = []
        for ds in ds_names:
            v = all_results[ds].get(ln, 0.0)
            if OVERVIEW_LOG_SCALE:
                v = max(v, OVERVIEW_LOG_EPS_MB)
            vals.append(v)

        ax.bar(x + offsets[i], vals, w * 0.9,
               label=LEARNER_LABELS[ln], color=LEARNER_COLORS[ln], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Memory (KB)", fontsize=9)

    scale_label = "Log Scale" if OVERVIEW_LOG_SCALE else "Linear Scale"
    ax.set_title(
        f"All Datasets — Training Memory per Learner  ({_fp_label(fp_bytes)}, {scale_label})",
        fontsize=12,
        fontweight="bold",
    )

    if OVERVIEW_LOG_SCALE:
        ax.set_yscale("log")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview_training_memory.png"), bbox_inches="tight")
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
        rep_g = run_list[0]
        ds_label = dataset_key.upper().replace("_", " ")

        base_hidden = rep_g.get("HIDDEN_SIZES", [128])[0]
        base_batch = rep_g.get("BATCH_SIZE", 128)

        print(f"=== {ds_label} ===")

        transform = rep_g.get("TRANSFORM")
        windowed, window_L = _detect_window(transform)

        meta = _load_meta(rep_g)
        print(
            f"  metadata: input_dim={meta['input_dim']}, "
            f"n_classes={meta['n_classes']}, time_steps={meta.get('time_steps', '?')}"
            + (f"  [windowed, L={meta.get('time_steps')}]" if windowed else "")
        )

        if windowed and window_L is None:
            window_L = meta.get("time_steps")

        T = meta.get("time_steps") or 128

        all_results[ds_label] = {}
        ds_fp_bytes = 4
        for ln in LEARNERS:
            p, t, fp = _get_memory(rep_g, ln, meta, [base_hidden], base_batch, T)
            all_results[ds_label][ln] = _kb(t)
            ds_fp_bytes = fp
            global_fp_bytes = fp
            print(
                f"    {LEARNER_LABELS[ln]:20s}  training={_kb(t):.4f} KB  [{_fp_label(fp)}]"
            )

        out_dir = _dir(os.path.join(OUT_ROOT, _dataset_slug(dataset_key)))
        print(f"  Plotting and writing CSVs …")

        plot_static_vs_dynamic(out_dir, ds_label, rep_g, meta, base_hidden, base_batch, windowed)
        plot_vs_hidden_size(out_dir, ds_label, rep_g, meta, base_batch, windowed)
        plot_vs_batch_size(out_dir, ds_label, rep_g, meta, base_hidden, windowed)
        plot_vs_time_steps(out_dir, ds_label, rep_g, meta, base_hidden, base_batch, windowed)
        plot_vs_num_layers(out_dir, ds_label, rep_g, meta, base_hidden, base_batch, windowed)

        print(f"  Saved: {out_dir}\n")

    print("Plotting cross-dataset overview and writing overview CSV …")
    write_overview_csv(all_results, OUT_ROOT, global_fp_bytes)
    plot_overview(all_results, OUT_ROOT, global_fp_bytes)
    print(f"Done. All graphs and CSVs under {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()