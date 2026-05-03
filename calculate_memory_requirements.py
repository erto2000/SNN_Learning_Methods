# calculate_memory_requirements.py
#
# Uses experiment configs directly from run_training.py (RUNS list).
# Each RUN is treated as its own memory scenario keyed by RUN_ID.
#
# For every scenario:
#   - loads real dataset metadata by probing one transformed sample
#   - computes training memory for all learners on that exact scenario
#   - writes per-scenario CSV + plots
#
# Also writes:
#   - overview by scenario (RUN_ID)
#   - overview by dataset (aggregated from scenario baseline memory)
#
# Output layout:
#   results/memory/
#     overview_scenarios_training_memory.png
#     overview_scenarios_base_memory.csv
#     overview_datasets_training_memory.png
#     overview_datasets_base_memory.csv
#     {run_id}/
#       0_base_memory.csv
#       1_training_memory.png
#       2_training_memory_vs_hidden_size.csv
#       2_training_memory_vs_hidden_size.png
#       4_dynamic_memory_vs_batch_size.csv
#       4_memory_vs_batch_size.png
#       5_dynamic_memory_vs_time_steps.csv
#       5_memory_vs_time_steps.png
#       6_training_memory_vs_num_layers.csv
#       6_training_memory_vs_num_layers.png

from __future__ import annotations
import os
import csv
import warnings
from typing import Optional

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils.training import build_cfg
from utils.runner import _make_learner, _infer_fp_bytes
from timeseries.registry import get_dataloaders
from timeseries.transforms import SlidingWindow, AdaptiveSlidingWindow, Compose
from run_training import RUNS

# ── output / sweep constants ──────────────────────────────────────────────────
OUT_ROOT = "./results/memory"

# Overview graphs only
OVERVIEW_LOG_SCALE = True
OVERVIEW_LOG_EPS_KB = 1e-6  # prevents log(0) issues for empty bars

# Per-scenario graphs
PER_SCENARIO_LOG_SCALE = True
PER_SCENARIO_LOG_EPS_KB = 1e-6

# Per-scenario numeric x-axis graphs only
PER_SCENARIO_LOG_X = True

# Samples to load for metadata probing (enough for ZScore.fit; fast)
PROBE_MAX_SAMPLES = 64

HIDDEN_SIZES_SWEEP = [32, 64, 128, 256, 512, 1024]
BATCH_SIZES_SWEEP = [8, 16, 32, 64, 128, 256, 512]
TIME_STEPS_SWEEP = [16, 32, 64, 128, 256, 512, 1024]
NUM_LAYERS_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]

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
    "ff": {"linestyle": "-", "marker": "o"},
    "eprop": {"linestyle": "-", "marker": "o"},
    "pepita": {"linestyle": "-", "marker": "o"},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _kb(b: int) -> float:
    return b / 1024.0


def _slug(name: str) -> str:
    return str(name).lower().replace(" ", "_").replace("-", "_").replace("/", "_")


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
        warnings.warn(
            f"Could not build learner='{learner_name}' for run='{base_g.get('RUN_ID', '?')}' "
            f"with hidden={hidden_sizes}: {e}"
        )
        return 0, 0, 4

    fp = _infer_fp_bytes(learner.model)
    param_bytes = learner.get_param_memory_bytes(fp_bytes=fp)
    training_bytes = learner.get_training_memory_bytes(
        batch=batch,
        time_steps=time_steps,
        fp_bytes=fp,
    )
    return param_bytes, training_bytes, fp


def _apply_style(ax, xlabel: str, ylabel: str = "Memory (KB)") -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.8)


def _apply_numeric_x_log(ax, ticks: list[int]) -> None:
    if PER_SCENARIO_LOG_X:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(ticks)


def _scenario_context_row(
    run_id: str,
    ds_label: str,
    learner_key: str,
    fp_bytes: int,
    hidden_sizes: list[int],
    base_batch: int,
    time_steps: int,
    windowed: bool,
    meta: dict,
) -> dict:
    return {
        "run_id": run_id,
        "dataset": ds_label,
        "learner_key": learner_key,
        "learner_label": LEARNER_LABELS[learner_key],
        "precision": _fp_label(fp_bytes),
        "fp_bytes": fp_bytes,
        "hidden_sizes": str(hidden_sizes),
        "num_layers": len(hidden_sizes),
        "base_batch": base_batch,
        "time_steps": time_steps,
        "windowed": windowed,
        "input_dim": meta.get("input_dim"),
        "n_classes": meta.get("n_classes"),
        "native_time_steps": meta.get("time_steps"),
    }


# ── per-scenario graphs + csv ────────────────────────────────────────────────

def plot_static_training_memory(
    out_dir: str,
    run_id: str,
    ds_label: str,
    base_g: dict,
    meta: dict,
    hidden_sizes: list[int],
    base_batch: int,
    windowed: bool,
) -> None:
    T = meta["time_steps"]
    totals = []
    fp_bytes = 4
    csv_rows = []

    for ln in LEARNERS:
        p, t, fp = _get_memory(base_g, ln, meta, hidden_sizes, base_batch, T)
        total_kb = _kb(t)
        totals.append(total_kb)
        fp_bytes = fp

        row = _scenario_context_row(
            run_id, ds_label, ln, fp, hidden_sizes, base_batch, T, windowed, meta
        )
        row["param_bytes"] = p
        row["param_kb"] = _kb(p)
        row["training_bytes"] = t
        row["training_kb"] = total_kb
        csv_rows.append(row)

    _csv_write(
        os.path.join(out_dir, "0_base_memory.csv"),
        csv_rows,
        [
            "run_id", "dataset",
            "learner_key", "learner_label",
            "precision", "fp_bytes",
            "hidden_sizes", "num_layers",
            "base_batch", "time_steps", "windowed",
            "input_dim", "n_classes", "native_time_steps",
            "param_bytes", "param_kb",
            "training_bytes", "training_kb",
        ],
    )

    labels = [LEARNER_LABELS[ln] for ln in LEARNERS]
    colors = [LEARNER_COLORS[ln] for ln in LEARNERS]
    x = np.arange(len(LEARNERS))
    w = 0.5

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.bar(x, totals, w, color=colors, alpha=0.85)

    max_val = max(totals) if totals else 1.0
    for i, v in enumerate(totals):
        ax.text(
            x[i],
            v + max_val * 0.01,
            f"{v:.1f} KB",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    t_label = f"window L={T}" if windowed else f"T={T}"
    ax.set_title(
        f"{run_id} — Training Memory\n"
        f"({ds_label}, hidden={hidden_sizes}, batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("Memory (KB)", fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_training_memory.png"))
    plt.close(fig)


def plot_vs_hidden_size(
    out_dir: str,
    run_id: str,
    ds_label: str,
    base_g: dict,
    meta: dict,
    base_batch: int,
    windowed: bool,
) -> None:
    T = meta["time_steps"]
    fp_bytes = 4
    rows = []

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    for ln in LEARNERS:
        vals = []
        for h in HIDDEN_SIZES_SWEEP:
            p, t, fp = _get_memory(base_g, ln, meta, [h], base_batch, T)
            total_kb = _kb(t)
            v = max(total_kb, PER_SCENARIO_LOG_EPS_KB) if PER_SCENARIO_LOG_SCALE else total_kb
            vals.append(v)
            fp_bytes = fp

            rows.append({
                "run_id": run_id,
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "hidden_size": h,
                "batch_size": base_batch,
                "time_steps": T,
                "windowed": windowed,
                "precision": _fp_label(fp),
                "fp_bytes": fp,
                "param_bytes": p,
                "param_kb": _kb(p),
                "training_bytes": t,
                "training_kb": total_kb,
            })

        ax.plot(
            HIDDEN_SIZES_SWEEP,
            vals,
            linewidth=2,
            color=LEARNER_COLORS[ln],
            label=LEARNER_LABELS[ln],
            **LEARNER_STYLES[ln],
        )

    _csv_write(
        os.path.join(out_dir, "2_training_memory_vs_hidden_size.csv"),
        rows,
        [
            "run_id", "dataset",
            "learner_key", "learner_label",
            "hidden_size", "batch_size", "time_steps", "windowed",
            "precision", "fp_bytes",
            "param_bytes", "param_kb",
            "training_bytes", "training_kb",
        ],
    )

    _apply_style(ax, "Hidden layer width (neurons)")
    _apply_numeric_x_log(ax, HIDDEN_SIZES_SWEEP)

    if PER_SCENARIO_LOG_SCALE:
        ax.set_yscale("log")

    t_label = f"window L={T}" if windowed else f"T={T}"
    fig.suptitle(
        f"{run_id} — Training Memory vs Hidden Size\n"
        f"({ds_label}, batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_training_memory_vs_hidden_size.png"))
    plt.close(fig)


def plot_vs_batch_size(
    out_dir: str,
    run_id: str,
    ds_label: str,
    base_g: dict,
    meta: dict,
    hidden_sizes: list[int],
    windowed: bool,
) -> None:
    T = meta["time_steps"]
    fp_bytes = 4
    rows = []

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    for ln in LEARNERS:
        vals = []
        for b in BATCH_SIZES_SWEEP:
            p, t, fp = _get_memory(base_g, ln, meta, hidden_sizes, b, T)
            total_kb = _kb(t)
            v = max(total_kb, PER_SCENARIO_LOG_EPS_KB) if PER_SCENARIO_LOG_SCALE else total_kb
            vals.append(v)
            fp_bytes = fp

            rows.append({
                "run_id": run_id,
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "hidden_sizes": str(hidden_sizes),
                "batch_size": b,
                "time_steps": T,
                "windowed": windowed,
                "precision": _fp_label(fp),
                "fp_bytes": fp,
                "param_bytes": p,
                "param_kb": _kb(p),
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
            "run_id", "dataset",
            "learner_key", "learner_label",
            "hidden_sizes", "batch_size", "time_steps", "windowed",
            "precision", "fp_bytes",
            "param_bytes", "param_kb",
            "training_bytes", "training_kb",
        ],
    )

    t_label = f"window L={T}" if windowed else f"T={T}"
    ax.set_title(
        f"{run_id} — Training Memory vs Batch Size\n"
        f"({ds_label}, hidden={hidden_sizes}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=11,
        fontweight="bold",
    )
    _apply_style(ax, "Batch size")
    _apply_numeric_x_log(ax, BATCH_SIZES_SWEEP)

    if PER_SCENARIO_LOG_SCALE:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_memory_vs_batch_size.png"))
    plt.close(fig)


def plot_vs_time_steps(
    out_dir: str,
    run_id: str,
    ds_label: str,
    base_g: dict,
    meta: dict,
    hidden_sizes: list[int],
    base_batch: int,
    windowed: bool,
) -> None:
    native_T = meta["time_steps"]
    xlabel = "Window length (L)" if windowed else "Sequence length (T)"
    fp_bytes = 4
    rows = []

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    for ln in LEARNERS:
        vals = []
        for T in TIME_STEPS_SWEEP:
            p, t, fp = _get_memory(base_g, ln, meta, hidden_sizes, base_batch, T)
            total_kb = _kb(t)
            v = max(total_kb, PER_SCENARIO_LOG_EPS_KB) if PER_SCENARIO_LOG_SCALE else total_kb
            vals.append(v)
            fp_bytes = fp

            rows.append({
                "run_id": run_id,
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "hidden_sizes": str(hidden_sizes),
                "batch_size": base_batch,
                "time_steps": T,
                "native_time_steps": native_T,
                "windowed": windowed,
                "precision": _fp_label(fp),
                "fp_bytes": fp,
                "param_bytes": p,
                "param_kb": _kb(p),
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
            "run_id", "dataset",
            "learner_key", "learner_label",
            "hidden_sizes", "batch_size",
            "time_steps", "native_time_steps", "windowed",
            "precision", "fp_bytes",
            "param_bytes", "param_kb",
            "training_bytes", "training_kb",
        ],
    )

    ax.set_title(
        f"{run_id} — Training Memory vs {'Window' if windowed else 'Sequence'} Length\n"
        f"({ds_label}, hidden={hidden_sizes}, batch={base_batch}, {_fp_label(fp_bytes)})",
        fontsize=11,
        fontweight="bold",
    )
    _apply_style(ax, xlabel)
    _apply_numeric_x_log(ax, TIME_STEPS_SWEEP)

    if PER_SCENARIO_LOG_SCALE:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "5_memory_vs_time_steps.png"))
    plt.close(fig)


def plot_vs_num_layers(
    out_dir: str,
    run_id: str,
    ds_label: str,
    base_g: dict,
    meta: dict,
    base_hidden: int,
    base_batch: int,
    windowed: bool,
) -> None:
    T = meta["time_steps"]
    fp_bytes = 4
    rows = []
    t_label = f"window L={T}" if windowed else f"T={T}"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    for ln in LEARNERS:
        vals = []
        for n in NUM_LAYERS_SWEEP:
            hidden_sizes = [base_hidden] * n
            p, t, fp = _get_memory(base_g, ln, meta, hidden_sizes, base_batch, T)
            total_kb = _kb(t)
            v = max(total_kb, PER_SCENARIO_LOG_EPS_KB) if PER_SCENARIO_LOG_SCALE else total_kb
            vals.append(v)
            fp_bytes = fp

            rows.append({
                "run_id": run_id,
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
                "param_bytes": p,
                "param_kb": _kb(p),
                "training_bytes": t,
                "training_kb": total_kb,
            })

        ax.plot(
            NUM_LAYERS_SWEEP,
            vals,
            linewidth=2,
            color=LEARNER_COLORS[ln],
            label=LEARNER_LABELS[ln],
            **LEARNER_STYLES[ln],
        )

    _csv_write(
        os.path.join(out_dir, "6_training_memory_vs_num_layers.csv"),
        rows,
        [
            "run_id", "dataset",
            "learner_key", "learner_label",
            "num_layers", "hidden_size_per_layer", "batch_size", "time_steps", "windowed",
            "precision", "fp_bytes",
            "param_bytes", "param_kb",
            "training_bytes", "training_kb",
        ],
    )

    _apply_style(ax, "Number of hidden layers")
    _apply_numeric_x_log(ax, NUM_LAYERS_SWEEP)

    if PER_SCENARIO_LOG_SCALE:
        ax.set_yscale("log")

    fig.suptitle(
        f"{run_id} — Training Memory vs Network Depth\n"
        f"({ds_label}, hidden={base_hidden}/layer, batch={base_batch}, {t_label}, {_fp_label(fp_bytes)})",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "6_training_memory_vs_num_layers.png"))
    plt.close(fig)


# ── overview csv / plots ─────────────────────────────────────────────────────

def write_scenario_overview_csv(all_results: dict, out_dir: str) -> None:
    rows = []
    for run_id, info in all_results.items():
        ds_label = info["dataset"]
        for ln in LEARNERS:
            rows.append({
                "run_id": run_id,
                "dataset": ds_label,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "precision": info["precision"],
                "fp_bytes": info["fp_bytes"],
                "training_kb": info["training_kb"].get(ln, 0.0),
            })

    _csv_write(
        os.path.join(out_dir, "overview_scenarios_base_memory.csv"),
        rows,
        [
            "run_id", "dataset",
            "learner_key", "learner_label",
            "precision", "fp_bytes",
            "training_kb",
        ],
    )


def plot_scenario_overview(all_results: dict, out_dir: str) -> None:
    run_ids = list(all_results.keys())
    n_runs = len(run_ids)
    n_ln = len(LEARNERS)
    x = np.arange(n_runs)
    w = 0.18
    offsets = np.linspace(-(n_ln - 1) / 2, (n_ln - 1) / 2, n_ln) * w

    fig, ax = plt.subplots(figsize=(max(12, n_runs * 1.5), 6), dpi=140)

    for i, ln in enumerate(LEARNERS):
        vals = []
        for run_id in run_ids:
            v = all_results[run_id]["training_kb"].get(ln, 0.0)
            if OVERVIEW_LOG_SCALE:
                v = max(v, OVERVIEW_LOG_EPS_KB)
            vals.append(v)

        ax.bar(
            x + offsets[i],
            vals,
            w * 0.9,
            label=LEARNER_LABELS[ln],
            color=LEARNER_COLORS[ln],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("Memory (KB)", fontsize=9)
    ax.set_title(
        "Training Memory per Learner",
        fontsize=12,
        fontweight="bold",
    )

    if OVERVIEW_LOG_SCALE:
        ax.set_yscale("log")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview_scenarios_training_memory.png"), bbox_inches="tight")
    plt.close(fig)


def write_dataset_overview_csv(all_results: dict, out_dir: str) -> None:
    dataset_map: dict[str, dict[str, float]] = {}

    for _, info in all_results.items():
        ds = info["dataset"]
        dataset_map.setdefault(ds, {ln: 0.0 for ln in LEARNERS})
        for ln in LEARNERS:
            dataset_map[ds][ln] += info["training_kb"].get(ln, 0.0)

    rows = []
    for ds, learner_map in dataset_map.items():
        for ln in LEARNERS:
            rows.append({
                "dataset": ds,
                "learner_key": ln,
                "learner_label": LEARNER_LABELS[ln],
                "training_kb_sum_over_runs": learner_map[ln],
            })

    _csv_write(
        os.path.join(out_dir, "overview_datasets_base_memory.csv"),
        rows,
        [
            "dataset",
            "learner_key",
            "learner_label",
            "training_kb_sum_over_runs",
        ],
    )


def plot_dataset_overview(all_results: dict, out_dir: str) -> None:
    dataset_map: dict[str, dict[str, float]] = {}

    for _, info in all_results.items():
        ds = info["dataset"]
        dataset_map.setdefault(ds, {ln: 0.0 for ln in LEARNERS})
        for ln in LEARNERS:
            dataset_map[ds][ln] += info["training_kb"].get(ln, 0.0)

    ds_names = list(dataset_map.keys())
    n_ds = len(ds_names)
    n_ln = len(LEARNERS)
    x = np.arange(n_ds)
    w = 0.18
    offsets = np.linspace(-(n_ln - 1) / 2, (n_ln - 1) / 2, n_ln) * w

    fig, ax = plt.subplots(figsize=(max(12, n_ds * 1.5), 6), dpi=140)

    for i, ln in enumerate(LEARNERS):
        vals = []
        for ds in ds_names:
            v = dataset_map[ds].get(ln, 0.0)
            if OVERVIEW_LOG_SCALE:
                v = max(v, OVERVIEW_LOG_EPS_KB)
            vals.append(v)

        ax.bar(
            x + offsets[i],
            vals,
            w * 0.9,
            label=LEARNER_LABELS[ln],
            color=LEARNER_COLORS[ln],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("Memory (KB)", fontsize=9)
    ax.set_title(
        "Datasets — Sum of Scenario Baseline Training Memory per Learner",
        fontsize=12,
        fontweight="bold",
    )

    if OVERVIEW_LOG_SCALE:
        ax.set_yscale("log")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview_datasets_training_memory.png"), bbox_inches="tight")
    plt.close(fig)


# ── run/scenario iteration ────────────────────────────────────────────────────

def _run_id(g: dict, fallback_index: int) -> str:
    return g.get("RUN_ID") or f"{g.get('DATASET', 'run')}-{fallback_index:03d}"


def _scenario_label(g: dict) -> str:
    return g.get("DATASET", "unknown").upper().replace("_", " ")


def main() -> None:
    _dir(OUT_ROOT)

    if not RUNS:
        print("No runs configured. Add at least one run in run_training.py and try again.")
        return

    print(f"Output: {os.path.abspath(OUT_ROOT)}\n")

    all_results: dict[str, dict] = {}

    for idx, g in enumerate(RUNS):
        run_id = _run_id(g, idx)
        ds_label = _scenario_label(g)
        out_dir = _dir(os.path.join(OUT_ROOT, _slug(run_id)))

        hidden_sizes = list(g.get("HIDDEN_SIZES", [128]))
        base_hidden = hidden_sizes[0] if hidden_sizes else 128
        base_batch = int(g.get("BATCH_SIZE", 128))

        print(f"=== {run_id} ({ds_label}) ===")

        transform = g.get("TRANSFORM")
        windowed, window_L = _detect_window(transform)

        meta = _load_meta(g)
        if windowed and window_L is None:
            window_L = meta.get("time_steps")

        print(
            f"  metadata: input_dim={meta['input_dim']}, "
            f"n_classes={meta['n_classes']}, "
            f"time_steps={meta.get('time_steps', '?')}"
            + (f"  [windowed, L={meta.get('time_steps')}]" if windowed else "")
        )

        T = int(meta.get("time_steps") or 128)

        all_results[run_id] = {
            "dataset": ds_label,
            "precision": "fp32",
            "fp_bytes": 4,
            "training_kb": {},
        }

        for ln in LEARNERS:
            p, t, fp = _get_memory(g, ln, meta, hidden_sizes, base_batch, T)
            all_results[run_id]["training_kb"][ln] = _kb(t)
            all_results[run_id]["precision"] = _fp_label(fp)
            all_results[run_id]["fp_bytes"] = fp

            print(
                f"    {LEARNER_LABELS[ln]:20s} "
                f"training={_kb(t):.4f} KB  "
                f"hidden={hidden_sizes}  batch={base_batch}  T={T}  [{_fp_label(fp)}]"
            )

        print("  Plotting and writing CSVs …")

        plot_static_training_memory(out_dir, run_id, ds_label, g, meta, hidden_sizes, base_batch, windowed)
        plot_vs_hidden_size(out_dir, run_id, ds_label, g, meta, base_batch, windowed)
        plot_vs_batch_size(out_dir, run_id, ds_label, g, meta, hidden_sizes, windowed)
        plot_vs_time_steps(out_dir, run_id, ds_label, g, meta, hidden_sizes, base_batch, windowed)
        plot_vs_num_layers(out_dir, run_id, ds_label, g, meta, base_hidden, base_batch, windowed)

        print(f"  Saved: {out_dir}\n")

    print("Writing overview CSVs and overview plots …")
    write_scenario_overview_csv(all_results, OUT_ROOT)
    plot_scenario_overview(all_results, OUT_ROOT)
    write_dataset_overview_csv(all_results, OUT_ROOT)
    plot_dataset_overview(all_results, OUT_ROOT)
    print(f"Done. All graphs and CSVs under {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()