from __future__ import annotations
import os
import re
import glob
import json
from typing import Dict, Any, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "comparisons", "window_search")
LEARNER_ORDER = ["bp", "eprop", "ff", "pepita"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_run_id(run_id: str) -> Dict[str, Optional[int]]:
    """
    Expected pattern:
      <base>-p{phase}-optuna-t{trial}-L{L}-H{hop}
    """
    out = {"phase": None, "trial": None, "L_from_id": None, "hop_from_id": None}
    if not run_id:
        return out

    m = re.search(r"-p(?P<phase>\d+)-optuna-t(?P<trial>\d+)-L(?P<L>\d+)-H(?P<H>\d+)$", run_id)
    if not m:
        return out

    out["phase"] = int(m.group("phase"))
    out["trial"] = int(m.group("trial"))
    out["L_from_id"] = int(m.group("L"))
    out["hop_from_id"] = int(m.group("H"))
    return out


def ordered_learners(values: List[str]) -> List[str]:
    seen = set(values)
    out = [x for x in LEARNER_ORDER if x in seen]
    rest = sorted([x for x in values if x not in set(out)])
    return out + rest


def learner_color_map(learners: List[str]) -> Dict[str, Any]:
    cmap = plt.get_cmap("tab10")
    ordered = ordered_learners(learners)
    return {learner: cmap(i % 10) for i, learner in enumerate(ordered)}


def learner_marker_map(learners: List[str]) -> Dict[str, str]:
    marker_pool = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    ordered = ordered_learners(learners)
    return {learner: marker_pool[i % len(marker_pool)] for i, learner in enumerate(ordered)}


def load_run_summaries(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    pattern = os.path.join(results_dir, "runs", "*", "summary.json")
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)

        cfg = s.get("config", {}) or {}
        final = s.get("final", {}) or {}
        win = cfg.get("WINDOW", {}) or {}
        mem = s.get("memory", {}) or {}
        run_id = s.get("run_id", "")

        if not win and "-optuna-" not in run_id:
            continue

        parsed = parse_run_id(run_id)

        L = win.get("L", parsed["L_from_id"])
        hop = win.get("hop", parsed["hop_from_id"])
        hop_ratio = win.get("hop_ratio")

        if hop_ratio is None and L is not None and hop is not None and L > 0:
            hop_ratio = float(hop) / float(L)

        rows.append({
            "run_id": run_id,
            "dataset": cfg.get("DATASET"),
            "learner": cfg.get("LEARNER"),
            "epochs": cfg.get("EPOCHS"),
            "status": s.get("status"),
            "sample_acc": final.get("sample_acc"),
            "win_L": L,
            "hop": hop,
            "hop_ratio": hop_ratio,
            "phase": parsed["phase"],
            "trial": parsed["trial"],
            "fp_bytes": mem.get("fp_bytes"),
            "static_bytes": mem.get("static_bytes"),
            "training_bytes_per_batch": mem.get("training_bytes_per_batch"),
            "memory_batch_size": mem.get("batch_size"),
            "memory_time_steps": mem.get("time_steps"),
            "avg_synaptic_ops": final.get("avg_synaptic_operations"),
            "firing_rate": final.get("firing_rate"),
            "energy_per_sample_pj": final.get("energy_per_sample_pj"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[df["status"] == "ok"].copy()
    df = df[df["sample_acc"].notna()].copy()
    df = df[df["win_L"].notna()].copy()
    df = df[df["hop"].notna()].copy()
    df = df[df["hop_ratio"].notna()].copy()

    df["win_L"] = df["win_L"].astype(int)
    df["hop"] = df["hop"].astype(int)
    df["hop_ratio"] = df["hop_ratio"].astype(float)
    df["sample_acc"] = df["sample_acc"].astype(float)

    numeric_cols = [
        "phase",
        "trial",
        "fp_bytes",
        "static_bytes",
        "training_bytes_per_batch",
        "memory_batch_size",
        "memory_time_steps",
        "avg_synaptic_ops",
        "firing_rate",
        "energy_per_sample_pj",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "static_bytes" in df.columns:
        df["static_mb"] = df["static_bytes"] / (1024 ** 2)
    if "training_bytes_per_batch" in df.columns:
        df["training_mb_per_batch"] = df["training_bytes_per_batch"] / (1024 ** 2)

    return df


def load_optuna_trials(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    rows = []
    pattern = os.path.join(results_dir, "optuna", "*_trials.csv")
    for path in glob.glob(pattern):
        try:
            rows.append(pd.read_csv(path))
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    keep_cols = [
        c for c in [
            "study", "trial", "state", "value",
            "win_L", "hop_ratio", "hop", "run_id"
        ]
        if c in df.columns
    ]
    return df[keep_cols].copy()


def maybe_join_optuna(df_runs: pd.DataFrame, df_optuna: pd.DataFrame) -> pd.DataFrame:
    if df_runs.empty or df_optuna.empty or "run_id" not in df_optuna.columns:
        return df_runs

    df = df_runs.merge(df_optuna, on="run_id", how="left", suffixes=("", "_optuna"))

    if "study" in df.columns:
        def infer_phase(study: Any) -> Optional[int]:
            if not isinstance(study, str):
                return None
            m = re.search(r"_p(\d+)$", study)
            return int(m.group(1)) if m else None

        if "phase" in df.columns:
            df["phase"] = df["phase"].fillna(df["study"].map(infer_phase))

    return df


def save_all_trials_csv(df: pd.DataFrame, out_dir: str) -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "all_trials.csv")

    cols = [
        "run_id", "dataset", "learner", "phase", "trial",
        "win_L", "hop", "hop_ratio", "sample_acc",
        "avg_synaptic_ops", "firing_rate", "energy_per_sample_pj",
        "training_bytes_per_batch", "training_mb_per_batch",
        "static_bytes", "static_mb",
        "fp_bytes", "memory_batch_size", "memory_time_steps",
    ]
    cols = [c for c in cols if c in df.columns]

    sort_cols = [c for c in ["dataset", "learner", "phase", "trial", "win_L", "hop"] if c in df.columns]
    df[cols].sort_values(sort_cols).to_csv(out_path, index=False)
    return out_path


def save_best_csv(df: pd.DataFrame, out_dir: str) -> str:
    ensure_dir(out_dir)
    idx = df.groupby(["dataset", "learner"])["sample_acc"].idxmax()
    best = df.loc[idx, [
        "dataset", "learner", "phase", "trial",
        "win_L", "hop", "hop_ratio", "sample_acc",
        "avg_synaptic_ops", "firing_rate", "energy_per_sample_pj",
        "training_bytes_per_batch", "training_mb_per_batch",
        "static_bytes", "static_mb",
        "run_id",
    ]].sort_values(["dataset", "learner"])
    out_path = os.path.join(out_dir, "best_per_dataset_learner.csv")
    best.to_csv(out_path, index=False)
    return out_path


def save_best_per_phase_csv(df: pd.DataFrame, out_dir: str) -> str:
    ensure_dir(out_dir)
    data = df[df["phase"].notna()].copy()
    if data.empty:
        out_path = os.path.join(out_dir, "best_per_dataset_learner_phase.csv")
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path

    idx = data.groupby(["dataset", "learner", "phase"])["sample_acc"].idxmax()
    best = data.loc[idx, [
        "dataset", "learner", "phase", "trial",
        "win_L", "hop", "hop_ratio", "sample_acc",
        "avg_synaptic_ops", "firing_rate", "energy_per_sample_pj",
        "training_mb_per_batch", "static_mb", "run_id"
    ]].sort_values(["dataset", "learner", "phase"])

    out_path = os.path.join(out_dir, "best_per_dataset_learner_phase.csv")
    best.to_csv(out_path, index=False)
    return out_path


def plot_scatter_phase_colored(
    df_sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    out_path: str,
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    color_label: Optional[str] = None,
) -> None:
    if df_sub.empty or x_col not in df_sub.columns or y_col not in df_sub.columns or color_col not in df_sub.columns:
        return

    data = df_sub[
        df_sub[x_col].notna() &
        df_sub[y_col].notna() &
        df_sub[color_col].notna()
    ].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)

    markers = {1: "o", 2: "^"}
    phase_values = sorted(v for v in data["phase"].dropna().unique())

    vmin = data[color_col].min()
    vmax = data[color_col].max()

    if phase_values:
        first_sc = None
        for phase in phase_values:
            gp = data[data["phase"] == phase]
            if gp.empty:
                continue
            sc = ax.scatter(
                gp[x_col],
                gp[y_col],
                c=gp[color_col],
                s=80,
                marker=markers.get(int(phase), "o"),
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.4,
                label=f"phase {int(phase)}",
            )
            if first_sc is None:
                first_sc = sc

        if first_sc is not None:
            cbar = fig.colorbar(first_sc, ax=ax)
            cbar.set_label(color_label or color_col)
            ax.legend()
    else:
        sc = ax.scatter(
            data[x_col],
            data[y_col],
            c=data[color_col],
            s=80,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label or color_col)

    ax.set_title(title)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_phase_simple(
    df_sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> None:
    if df_sub.empty or x_col not in df_sub.columns or y_col not in df_sub.columns:
        return

    data = df_sub[df_sub[x_col].notna() & df_sub[y_col].notna()].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    markers = {1: "o", 2: "^"}
    phase_values = sorted(v for v in data["phase"].dropna().unique())

    if phase_values:
        for phase in phase_values:
            gp = data[data["phase"] == phase]
            if gp.empty:
                continue
            ax.scatter(
                gp[x_col],
                gp[y_col],
                s=80,
                marker=markers.get(int(phase), "o"),
                alpha=0.85,
                edgecolors="black",
                linewidths=0.4,
                label=f"phase {int(phase)}",
            )
        ax.legend()
    else:
        ax.scatter(
            data[x_col],
            data[y_col],
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods_scatter(
    df_ds: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> None:
    data = df_ds[df_ds[x_col].notna() & df_ds[y_col].notna()].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    learners = ordered_learners(list(data["learner"].dropna().unique()))
    colors = learner_color_map(learners)
    markers = {1: "o", 2: "^"}

    for learner in learners:
        gl = data[data["learner"] == learner]
        if gl.empty:
            continue
        color = colors[learner]

        phase_values = sorted(v for v in gl["phase"].dropna().unique())
        if phase_values:
            for phase in phase_values:
                gp = gl[gl["phase"] == phase]
                if gp.empty:
                    continue
                ax.scatter(
                    gp[x_col],
                    gp[y_col],
                    s=80,
                    marker=markers.get(int(phase), "o"),
                    alpha=0.85,
                    color=color,
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{learner} | phase {int(phase)}",
                )
        else:
            ax.scatter(
                gl[x_col],
                gl[y_col],
                s=80,
                alpha=0.85,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                label=learner,
            )

    ax.set_title(title)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods_scatter_colored_by_accuracy(
    df_ds: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> None:
    data = df_ds[df_ds[x_col].notna() & df_ds[y_col].notna() & df_ds["sample_acc"].notna()].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    learners = ordered_learners(list(data["learner"].dropna().unique()))
    colors = learner_color_map(learners)
    markers = learner_marker_map(learners)

    color_min = data["sample_acc"].min()
    color_max = data["sample_acc"].max()
    first_sc = None

    for learner in learners:
        gl = data[data["learner"] == learner]
        if gl.empty:
            continue

        color = colors[learner]
        marker = markers[learner]

        phase_values = sorted(v for v in gl["phase"].dropna().unique())
        if phase_values:
            for phase in phase_values:
                gp = gl[gl["phase"] == phase]
                if gp.empty:
                    continue
                sc = ax.scatter(
                    gp[x_col],
                    gp[y_col],
                    c=gp["sample_acc"],
                    s=95,
                    marker=marker,
                    vmin=color_min,
                    vmax=color_max,
                    alpha=0.92,
                    edgecolors=color,
                    linewidths=1.3,
                    label=f"{learner} | phase {int(phase)}",
                )
                if first_sc is None:
                    first_sc = sc
        else:
            sc = ax.scatter(
                gl[x_col],
                gl[y_col],
                c=gl["sample_acc"],
                s=95,
                marker=marker,
                vmin=color_min,
                vmax=color_max,
                alpha=0.92,
                edgecolors=color,
                linewidths=1.3,
                label=learner,
            )
            if first_sc is None:
                first_sc = sc

    if first_sc is not None:
        cbar = fig.colorbar(first_sc, ax=ax)
        cbar.set_label("sample_acc")

    ax.set_title(title)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods_static_memory_bar(df_ds: pd.DataFrame, out_path: str, dataset: str) -> None:
    data = df_ds[df_ds["static_mb"].notna()].copy()
    if data.empty:
        return

    bars = (
        data.groupby("learner", as_index=False)["static_mb"]
        .median()
        .copy()
    )
    if bars.empty:
        return

    learners = ordered_learners(list(bars["learner"]))
    colors = learner_color_map(learners)
    bars["learner"] = pd.Categorical(bars["learner"], categories=learners, ordered=True)
    bars = bars.sort_values("learner")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.bar(
        bars["learner"].astype(str),
        bars["static_mb"],
        color=[colors[str(x)] for x in bars["learner"].astype(str)]
    )
    ax.set_title(f"{dataset} | static memory by method")
    ax.set_xlabel("method")
    ax.set_ylabel("static memory (MB)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods_method_vs_accuracy(df_ds: pd.DataFrame, out_path: str, dataset: str) -> None:
    data = df_ds[df_ds["sample_acc"].notna()].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    learners = ordered_learners(list(data["learner"].dropna().unique()))
    colors = learner_color_map(learners)
    markers = {1: "o", 2: "^"}

    x_positions = {learner: i for i, learner in enumerate(learners)}

    for learner in learners:
        gl = data[data["learner"] == learner]
        if gl.empty:
            continue
        color = colors[learner]
        x = x_positions[learner]

        phase_values = sorted(v for v in gl["phase"].dropna().unique())
        if phase_values:
            for phase in phase_values:
                gp = gl[gl["phase"] == phase]
                if gp.empty:
                    continue
                ax.scatter(
                    [x] * len(gp),
                    gp["sample_acc"],
                    s=80,
                    marker=markers.get(int(phase), "o"),
                    alpha=0.85,
                    color=color,
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{learner} | phase {int(phase)}",
                )
        else:
            ax.scatter(
                [x] * len(gl),
                gl["sample_acc"],
                s=80,
                alpha=0.85,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                label=learner,
            )

    ax.set_title(f"{dataset} | sample_acc vs method")
    ax.set_xlabel("method")
    ax.set_ylabel("sample_acc")
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def export_trials_csv(df_sub: pd.DataFrame, out_path: str) -> None:
    cols = [
        "run_id", "phase", "trial",
        "win_L", "hop", "hop_ratio",
        "sample_acc",
        "avg_synaptic_ops", "firing_rate", "energy_per_sample_pj",
        "training_bytes_per_batch", "training_mb_per_batch",
        "static_bytes", "static_mb",
        "fp_bytes", "memory_batch_size", "memory_time_steps",
    ]
    cols = [c for c in cols if c in df_sub.columns]
    sort_cols = [c for c in ["phase", "trial", "win_L", "hop"] if c in df_sub.columns]
    df_sub[cols].sort_values(sort_cols).to_csv(out_path, index=False)


def export_cross_method_csv(df_ds: pd.DataFrame, out_path: str) -> None:
    cols = [
        "run_id", "dataset", "learner", "phase", "trial",
        "win_L", "hop", "hop_ratio",
        "sample_acc", "avg_synaptic_ops", "firing_rate", "energy_per_sample_pj",
        "training_mb_per_batch", "static_mb"
    ]
    cols = [c for c in cols if c in df_ds.columns]
    sort_cols = [c for c in ["learner", "phase", "trial", "win_L", "hop"] if c in df_ds.columns]
    df_ds[cols].sort_values(sort_cols).to_csv(out_path, index=False)


def make_per_method_plots(df: pd.DataFrame, out_root: str) -> List[str]:
    ensure_dir(out_root)
    saved: List[str] = []

    grouped = df.groupby(["dataset", "learner"], dropna=False)

    for (dataset, learner), g in grouped:
        if g.empty:
            continue

        subdir = os.path.join(out_root, str(dataset), str(learner))
        ensure_dir(subdir)

        p = os.path.join(subdir, "hop_ratio_vs_window_length.png")
        plot_scatter_phase_colored(
            g,
            x_col="win_L",
            y_col="hop_ratio",
            color_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | hop ratio vs win length",
            x_label="window length",
            y_label="hop ratio",
            color_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "sample_acc_vs_window_length.png")
        plot_scatter_phase_simple(
            g,
            x_col="win_L",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | sample_acc vs window length",
            x_label="window length",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "sample_acc_vs_hop_ratio.png")
        plot_scatter_phase_simple(
            g,
            x_col="hop_ratio",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | sample_acc vs hop ratio",
            x_label="hop ratio",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "sample_acc_vs_hop.png")
        plot_scatter_phase_simple(
            g,
            x_col="hop",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | sample_acc vs hop",
            x_label="hop",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "sample_acc_vs_training_memory.png")
        plot_scatter_phase_simple(
            g,
            x_col="training_mb_per_batch",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | sample_acc vs training memory",
            x_label="training memory per batch (MB)",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "window_length_vs_training_memory.png")
        plot_scatter_phase_colored(
            g,
            x_col="win_L",
            y_col="training_mb_per_batch",
            color_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | window length vs training memory",
            x_label="window length",
            y_label="training memory per batch (MB)",
            color_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "window_length_vs_synaptic_ops.png")
        plot_scatter_phase_colored(
            g,
            x_col="win_L",
            y_col="avg_synaptic_ops",
            color_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | window length vs synaptic ops",
            x_label="window length",
            y_label="avg synaptic operations",
            color_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "window_length_vs_firing_rate.png")
        plot_scatter_phase_colored(
            g,
            x_col="win_L",
            y_col="firing_rate",
            color_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | window length vs firing rate",
            x_label="window length",
            y_label="firing rate",
            color_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "window_length_vs_energy.png")
        plot_scatter_phase_colored(
            g,
            x_col="win_L",
            y_col="energy_per_sample_pj",
            color_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | window length vs energy",
            x_label="window length",
            y_label="energy per sample (pJ)",
            color_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "energy_vs_accuracy.png")
        plot_scatter_phase_simple(
            g,
            x_col="energy_per_sample_pj",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | accuracy vs energy per sample",
            x_label="energy per sample (pJ)",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "trials.csv")
        export_trials_csv(g, p)
        saved.append(p)

    return saved


def make_cross_method_plots(df: pd.DataFrame, out_root: str) -> List[str]:
    saved: List[str] = []

    for dataset, g in df.groupby("dataset", dropna=False):
        if g.empty:
            continue

        ds_dir = os.path.join(out_root, str(dataset), "_all_methods")
        ensure_dir(ds_dir)

        p = os.path.join(ds_dir, "hop_ratio_vs_window_length_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g,
            x_col="win_L",
            y_col="hop_ratio",
            out_path=p,
            title=f"{dataset} | all methods | hop ratio vs win length",
            x_label="window length",
            y_label="hop ratio",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_training_memory_all_methods.png")
        plot_all_methods_scatter(
            g,
            x_col="training_mb_per_batch",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | sample_acc vs training memory",
            x_label="training memory per batch (MB)",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "window_length_vs_training_memory_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g,
            x_col="win_L",
            y_col="training_mb_per_batch",
            out_path=p,
            title=f"{dataset} | all methods | window length vs training memory",
            x_label="window length",
            y_label="training memory per batch (MB)",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "static_memory_by_method.png")
        plot_all_methods_static_memory_bar(g, out_path=p, dataset=str(dataset))
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_window_length_all_methods.png")
        plot_all_methods_scatter(
            g,
            x_col="win_L",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | sample_acc vs window length",
            x_label="window length",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_hop_ratio_all_methods.png")
        plot_all_methods_scatter(
            g,
            x_col="hop_ratio",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | sample_acc vs hop ratio",
            x_label="hop ratio",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_hop_all_methods.png")
        plot_all_methods_scatter(
            g,
            x_col="hop",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | sample_acc vs hop",
            x_label="hop",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "window_length_vs_synaptic_ops_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g,
            x_col="win_L",
            y_col="avg_synaptic_ops",
            out_path=p,
            title=f"{dataset} | all methods | window length vs synaptic ops",
            x_label="window length",
            y_label="avg synaptic operations",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "window_length_vs_firing_rate_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g,
            x_col="win_L",
            y_col="firing_rate",
            out_path=p,
            title=f"{dataset} | all methods | window length vs firing rate",
            x_label="window length",
            y_label="firing rate",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "window_length_vs_energy_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g,
            x_col="win_L",
            y_col="energy_per_sample_pj",
            out_path=p,
            title=f"{dataset} | all methods | window length vs energy",
            x_label="window length",
            y_label="energy per sample (pJ)",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "energy_vs_accuracy_all_methods.png")
        plot_all_methods_scatter(
            g,
            x_col="energy_per_sample_pj",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | accuracy vs energy per sample",
            x_label="energy per sample (pJ)",
            y_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_method.png")
        plot_all_methods_method_vs_accuracy(g, out_path=p, dataset=str(dataset))
        saved.append(p)

        p = os.path.join(ds_dir, "trials.csv")
        export_cross_method_csv(g, p)
        saved.append(p)

    return saved


def main() -> None:
    ensure_dir(OUT_DIR)

    df_runs = load_run_summaries(RESULTS_DIR)
    if df_runs.empty:
        print("No window-search runs found in results/runs/*/summary.json")
        return

    df_optuna = load_optuna_trials(RESULTS_DIR)
    df = maybe_join_optuna(df_runs, df_optuna)

    all_trials_csv = save_all_trials_csv(df, OUT_DIR)
    best_csv = save_best_csv(df, OUT_DIR)
    best_phase_csv = save_best_per_phase_csv(df, OUT_DIR)

    saved = []
    saved.extend(make_per_method_plots(df, OUT_DIR))
    saved.extend(make_cross_method_plots(df, OUT_DIR))

    print("\n=== Window Search Comparison Saved ===")
    print(all_trials_csv)
    print(best_csv)
    print(best_phase_csv)
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()