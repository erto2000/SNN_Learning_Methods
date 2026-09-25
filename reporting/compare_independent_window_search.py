from __future__ import annotations

"""
Cleaned independent window/hop comparison script.

Fixes:
- Supports new hop study names like ws_<dataset>_<learner>_hop_bestL128_p1.
- More tolerant run_id parsing for independent runs.
- Prints counts by dataset/learner/experiment/phase so missing runs are obvious.
- Keeps window-only runs in window plots and hop-only runs in hop plots.
"""


import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from timeseries.splitting import split_report_fields


RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "comparisons", "window_hop_independent")
LEARNER_ORDER = ["bp", "ff", "eprop", "pepita"]
EXPERIMENT_ORDER = ["window", "hop"]
PHASE_MARKERS = {1: "o", 2: "^"}
EXP_PHASE_MARKERS = {
    ("window", 1): "o",
    ("window", 2): "s",
    ("hop", 1): "^",
    ("hop", 2): "D",
}

THEORY_TRADEOFF_PLOTS = [
    ("theory_memory_mb", "estimated memory (MB)", "sample_acc_vs_theory_memory.png", "sample_acc vs estimated memory"),
    ("theory_compute_scalars", "compute per original-sequence batch (scalar ops)", "sample_acc_vs_theory_compute.png", "sample_acc vs estimated compute"),
    ("theory_access_scalars", "accesses per original-sequence batch (scalars)", "sample_acc_vs_theory_access.png", "sample_acc vs estimated memory access"),
    ("theory_time_proxy", "estimated time proxy", "sample_acc_vs_theory_time_proxy.png", "sample_acc vs estimated time proxy"),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def parse_run_id(run_id: str) -> Dict[str, Optional[Any]]:
    """
    New independent runner pattern:
      <base>-exp{experiment}-p{phase}-{search_mode}-optuna-t{trial}-L{L}-H{hop}

    Also supports the older consecutive pattern as a fallback:
      <base>-p{phase}-{search_mode}-optuna-t{trial}-L{L}-H{hop}
    """
    out: Dict[str, Optional[Any]] = {
        "experiment": None,
        "phase": None,
        "search_mode": None,
        "trial": None,
        "L_from_id": None,
        "hop_from_id": None,
    }
    if not run_id:
        return out

    m = re.search(
        r"-exp(?P<experiment>window|hop)-p(?P<phase>\d+)-(?P<search_mode>[a-z_]+)-optuna-t(?P<trial>\d+)-L(?P<L>\d+)-H(?P<H>\d+)$",
        run_id,
    )
    if m:
        out["experiment"] = str(m.group("experiment"))
        out["phase"] = int(m.group("phase"))
        out["search_mode"] = str(m.group("search_mode"))
        out["trial"] = int(m.group("trial"))
        out["L_from_id"] = int(m.group("L"))
        out["hop_from_id"] = int(m.group("H"))
        return out

    m = re.search(
        r"-p(?P<phase>\d+)-(?P<search_mode>[a-z_]+)-optuna-t(?P<trial>\d+)-L(?P<L>\d+)-H(?P<H>\d+)$",
        run_id,
    )
    if m:
        out["phase"] = int(m.group("phase"))
        out["search_mode"] = str(m.group("search_mode"))
        out["trial"] = int(m.group("trial"))
        out["L_from_id"] = int(m.group("L"))
        out["hop_from_id"] = int(m.group("H"))
        if out["search_mode"] == "window_only":
            out["experiment"] = "window"
        elif out["search_mode"] == "hop_only":
            out["experiment"] = "hop"
        return out

    # Last-resort tolerant parser. This catches small naming changes while still
    # requiring the independent-run markers.
    m = re.search(r"-exp(?P<experiment>window|hop)-p(?P<phase>\d+)-(?P<search_mode>[a-z_]+)-", run_id)
    if m:
        out["experiment"] = str(m.group("experiment"))
        out["phase"] = int(m.group("phase"))
        out["search_mode"] = str(m.group("search_mode"))

    mt = re.search(r"-t(?P<trial>\d+)", run_id)
    if mt:
        out["trial"] = int(mt.group("trial"))

    mlh = re.search(r"-L(?P<L>\d+)-H(?P<H>\d+)", run_id)
    if mlh:
        out["L_from_id"] = int(mlh.group("L"))
        out["hop_from_id"] = int(mlh.group("H"))

    return out


def parse_study_name(study: str) -> Dict[str, Optional[Any]]:
    """
    Independent runner study names:
      ws_<dataset>_<learner>_window_p1
      ws_<dataset>_<learner>_window_p2
      ws_<dataset>_<learner>_hop_p1
      ws_<dataset>_<learner>_hop_p2

    Older consecutive fallback:
      ws_<dataset>_<learner>_p1_window_only
      ws_<dataset>_<learner>_p2_hop_only
    """
    out: Dict[str, Optional[Any]] = {
        "experiment": None,
        "phase": None,
        "search_mode": None,
    }
    if not isinstance(study, str) or not study:
        return out

    # Supports both:
    #   ws_<dataset>_<learner>_hop_p1
    #   ws_<dataset>_<learner>_hop_bestL128_p1
    m = re.search(r"_(window|hop)(?:_bestL\d+)?_p(?P<phase>\d+)$", study)
    if m:
        experiment = m.group(1)
        phase = int(m.group("phase"))
        out["experiment"] = experiment
        out["phase"] = phase
        out["search_mode"] = "window_only" if experiment == "window" else "hop_only"
        return out

    m = re.search(r"_p(?P<phase>\d+)_(?P<search_mode>window_only|hop_only)$", study)
    if m:
        phase = int(m.group("phase"))
        search_mode = str(m.group("search_mode"))
        out["phase"] = phase
        out["search_mode"] = search_mode
        if search_mode == "window_only":
            out["experiment"] = "window"
        elif search_mode == "hop_only":
            out["experiment"] = "hop"
    return out


def dataset_label(run_id: str, dataset: str) -> str:
    """Identify MNIST encodings by the Optuna run prefix, not the source dataset."""
    if run_id.startswith("mnist_static-"):
        return "mnist_static"
    if run_id.startswith("mnist_rate-"):
        return "mnist_rate"
    return dataset


def load_run_summaries(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    from timeseries.splitting import split_report_fields
    rows: List[Dict[str, Any]] = []

    pattern = os.path.join(results_dir, "runs", "*", "summary.json")
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)

        cfg = s.get("config", {}) or {}
        final = s.get("final", {}) or {}
        win = cfg.get("WINDOW", {}) or {}
        mem = s.get("memory", {}) or {}
        theory = mem.get("theory", {}) or {}
        run_id = s.get("run_id", "")

        parsed = parse_run_id(run_id)
        # Historical test-tuned runs are a different protocol.
        if final.get("evaluation_split") != "validation" or not cfg.get('DATA_SPLIT'):
            continue
        experiment = parsed["experiment"]

        # This comparison script is specifically for the independent window/hop runner.
        # Be tolerant: if the run_id does not expose experiment directly, infer it from search_mode.
        if experiment is None:
            if parsed["search_mode"] == "window_only":
                experiment = "window"
            elif parsed["search_mode"] == "hop_only":
                experiment = "hop"
            else:
                continue

        L = win.get("L", parsed["L_from_id"])
        hop = win.get("hop", parsed["hop_from_id"])
        hop_ratio = win.get("hop_ratio")
        if hop_ratio is None and L is not None and hop is not None and int(L) > 0:
            hop_ratio = float(hop) / float(L)

        rows.append({
            **split_report_fields(s.get('meta', {})),
            "run_id": run_id,
            "dataset": dataset_label(run_id, cfg.get("DATASET")),
            "learner": cfg.get("LEARNER"),
            "epochs": cfg.get("EPOCHS"),
            "status": s.get("status"),
            "sample_acc": final.get("sample_acc"),
            "evaluation_split": "validation",
            "theory_work_scope": theory.get("work_scope"),
            "windows_per_sample": theory.get("windows_per_sample"),
            "win_L": L,
            "hop": hop,
            "hop_ratio": hop_ratio,
            "experiment": experiment,
            "phase": parsed["phase"],
            "search_mode": parsed["search_mode"],
            "trial": parsed["trial"],
            "fp_bytes": mem.get("fp_bytes"),
            "static_bytes": mem.get("static_bytes"),
            "training_bytes_per_batch": mem.get("training_bytes_per_batch"),
            "memory_batch_size": mem.get("batch_size"),
            "memory_time_steps": mem.get("time_steps"),
            "theory_memory_bytes": (theory.get("memory", {}) or {}).get("total_bytes"),
            "theory_memory_scalars": (theory.get("memory", {}) or {}).get("total_scalars"),
            "theory_compute_scalars": (theory.get("compute", {}) or {}).get("total_scalars"),
            "theory_access_scalars": (theory.get("access", {}) or {}).get("total_scalars"),
            "theory_time_proxy": (theory.get("time_proxy", {}) or {}).get("value"),
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
    df = df[df["experiment"].isin(EXPERIMENT_ORDER)].copy()

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
        "theory_memory_bytes",
        "theory_memory_scalars",
        "theory_compute_scalars",
        "theory_access_scalars",
        "theory_time_proxy",
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
    if "theory_memory_bytes" in df.columns:
        df["theory_memory_mb"] = df["theory_memory_bytes"] / (1024 ** 2)

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
    if "study" in df.columns:
        parsed = df["study"].map(parse_study_name)
        df["experiment_from_study"] = parsed.map(lambda x: x.get("experiment"))
        df["phase_from_study"] = parsed.map(lambda x: x.get("phase"))
        df["search_mode_from_study"] = parsed.map(lambda x: x.get("search_mode"))

    keep_cols = [
        c
        for c in [
            "study",
            "trial",
            "state",
            "value",
            "win_L",
            "hop_ratio",
            "hop",
            "experiment",
            "phase",
            "search_mode",
            "experiment_from_study",
            "phase_from_study",
            "search_mode_from_study",
            "run_id",
        ]
        if c in df.columns
    ]
    return df[keep_cols].copy()


def maybe_join_optuna(df_runs: pd.DataFrame, df_optuna: pd.DataFrame) -> pd.DataFrame:
    if df_runs.empty or df_optuna.empty or "run_id" not in df_optuna.columns:
        return df_runs

    df = df_runs.merge(df_optuna, on="run_id", how="left", suffixes=("", "_optuna"))

    if "experiment" in df.columns and "experiment_optuna" in df.columns:
        df["experiment"] = df["experiment"].fillna(df["experiment_optuna"])
    if "phase" in df.columns and "phase_optuna" in df.columns:
        df["phase"] = df["phase"].fillna(df["phase_optuna"])
    if "search_mode" in df.columns and "search_mode_optuna" in df.columns:
        df["search_mode"] = df["search_mode"].fillna(df["search_mode_optuna"])

    if "experiment_from_study" in df.columns:
        df["experiment"] = df["experiment"].fillna(df["experiment_from_study"])
    if "phase_from_study" in df.columns:
        df["phase"] = df["phase"].fillna(df["phase_from_study"])
    if "search_mode_from_study" in df.columns:
        df["search_mode"] = df["search_mode"].fillna(df["search_mode_from_study"])

    return df


def save_all_trials_csv(df: pd.DataFrame, out_dir: str) -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "all_trials.csv")

    cols = [
        *split_report_fields({}),
        "run_id",
        "dataset",
        "learner",
        "experiment",
        "phase",
        "search_mode",
        "trial",
        "win_L",
        "hop",
        "hop_ratio",
        "sample_acc",
        "avg_synaptic_ops",
        "firing_rate",
        "energy_per_sample_pj",
        "training_bytes_per_batch",
        "training_mb_per_batch",
        "static_bytes",
        "static_mb",
        "fp_bytes",
        "memory_batch_size",
        "memory_time_steps",
        "evaluation_split", "theory_work_scope", "windows_per_sample",
        "theory_memory_bytes",
        "theory_memory_mb",
        "theory_memory_scalars",
        "theory_compute_scalars",
        "theory_access_scalars",
        "theory_time_proxy",
    ]
    cols = [c for c in cols if c in df.columns]

    sort_cols = [c for c in ["dataset", "learner", "experiment", "phase", "trial", "win_L", "hop"] if c in df.columns]
    df[cols].sort_values(sort_cols).to_csv(out_path, index=False)
    return out_path


def save_best_csv(df: pd.DataFrame, out_dir: str) -> str:
    ensure_dir(out_dir)
    # Prefer the refined training budget; fall back only if that phase has no results.
    latest = df.groupby(["dataset", "learner", "experiment"])["phase"].transform("max")
    df = df[df["phase"] == latest].copy()
    idx = df.groupby(["dataset", "learner", "experiment"])["sample_acc"].idxmax()
    best = df.loc[idx, [
        *split_report_fields({}),
        "dataset",
        "learner",
        "experiment",
        "phase",
        "trial",
        "win_L",
        "hop",
        "hop_ratio",
        "sample_acc",
        "avg_synaptic_ops",
        "firing_rate",
        "energy_per_sample_pj",
        "training_bytes_per_batch",
        "training_mb_per_batch",
        "static_bytes",
        "static_mb",
        "evaluation_split", "theory_work_scope", "windows_per_sample",
        "theory_memory_bytes",
        "theory_memory_mb",
        "theory_memory_scalars",
        "theory_compute_scalars",
        "theory_access_scalars",
        "theory_time_proxy",
        "run_id",
    ]].sort_values(["dataset", "learner", "experiment"])
    out_path = os.path.join(out_dir, "best_per_dataset_learner_experiment.csv")
    best.to_csv(out_path, index=False)
    return out_path


def save_best_per_phase_csv(df: pd.DataFrame, out_dir: str) -> str:
    ensure_dir(out_dir)
    data = df[df["phase"].notna()].copy()
    if data.empty:
        out_path = os.path.join(out_dir, "best_per_dataset_learner_experiment_phase.csv")
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path

    idx = data.groupby(["dataset", "learner", "experiment", "phase"])["sample_acc"].idxmax()
    best = data.loc[idx, [
        *split_report_fields({}),
        "dataset",
        "learner",
        "experiment",
        "phase",
        "trial",
        "win_L",
        "hop",
        "hop_ratio",
        "sample_acc",
        "avg_synaptic_ops",
        "firing_rate",
        "energy_per_sample_pj",
        "training_mb_per_batch",
        "static_mb",
        "evaluation_split", "theory_work_scope", "windows_per_sample",
        "theory_memory_bytes",
        "theory_memory_mb",
        "theory_memory_scalars",
        "theory_compute_scalars",
        "theory_access_scalars",
        "theory_time_proxy",
        "run_id",
    ]].sort_values(["dataset", "learner", "experiment", "phase"])

    out_path = os.path.join(out_dir, "best_per_dataset_learner_experiment_phase.csv")
    best.to_csv(out_path, index=False)
    return out_path


def export_trials_csv(df_sub: pd.DataFrame, out_path: str) -> None:
    cols = [
        *split_report_fields({}),
        "run_id",
        "experiment",
        "phase",
        "search_mode",
        "trial",
        "win_L",
        "hop",
        "hop_ratio",
        "sample_acc",
        "avg_synaptic_ops",
        "firing_rate",
        "energy_per_sample_pj",
        "training_bytes_per_batch",
        "training_mb_per_batch",
        "static_bytes",
        "static_mb",
        "fp_bytes",
        "memory_batch_size",
        "memory_time_steps",
        "evaluation_split", "theory_work_scope", "windows_per_sample",
        "theory_memory_bytes",
        "theory_memory_mb",
        "theory_memory_scalars",
        "theory_compute_scalars",
        "theory_access_scalars",
        "theory_time_proxy",
    ]
    cols = [c for c in cols if c in df_sub.columns]
    sort_cols = [c for c in ["experiment", "phase", "trial", "win_L", "hop"] if c in df_sub.columns]
    df_sub[cols].sort_values(sort_cols).to_csv(out_path, index=False)


def export_cross_method_csv(df_ds: pd.DataFrame, out_path: str) -> None:
    cols = [
        *split_report_fields({}),
        "run_id",
        "dataset",
        "learner",
        "experiment",
        "phase",
        "search_mode",
        "trial",
        "win_L",
        "hop",
        "hop_ratio",
        "sample_acc",
        "avg_synaptic_ops",
        "firing_rate",
        "energy_per_sample_pj",
        "training_mb_per_batch",
        "static_mb",
        "theory_memory_mb",
        "theory_memory_scalars",
        "theory_compute_scalars",
        "theory_access_scalars",
        "theory_time_proxy",
    ]
    cols = [c for c in cols if c in df_ds.columns]
    sort_cols = [c for c in ["learner", "experiment", "phase", "trial", "win_L", "hop"] if c in df_ds.columns]
    df_ds[cols].sort_values(sort_cols).to_csv(out_path, index=False)


def filter_by_experiment(df: pd.DataFrame, experiment_scope: str) -> pd.DataFrame:
    if df.empty:
        return df
    if experiment_scope == "window":
        return df[df["experiment"] == "window"].copy()
    if experiment_scope == "hop":
        return df[df["experiment"] == "hop"].copy()
    if experiment_scope == "both":
        return df.copy()
    raise ValueError(f"Unknown experiment_scope={experiment_scope!r}")


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
                marker=PHASE_MARKERS.get(int(phase), "o"),
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

    data = df_sub[df_sub[x_col].notna() & df_sub[y_col].notna() & df_sub[color_col].notna()].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    phase_values = sorted(v for v in data["phase"].dropna().unique())
    vmin = data[color_col].min()
    vmax = data[color_col].max()
    first_sc = None

    if phase_values:
        for phase in phase_values:
            gp = data[data["phase"] == phase]
            if gp.empty:
                continue
            sc = ax.scatter(
                gp[x_col],
                gp[y_col],
                c=gp[color_col],
                s=80,
                marker=PHASE_MARKERS.get(int(phase), "o"),
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
        sc = ax.scatter(data[x_col], data[y_col], c=data[color_col], s=80)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_label or color_col)

    ax.set_title(title)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_experiment_phase_colored(
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

    data = df_sub[df_sub[x_col].notna() & df_sub[y_col].notna() & df_sub[color_col].notna()].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    vmin = data[color_col].min()
    vmax = data[color_col].max()
    first_sc = None

    for experiment in EXPERIMENT_ORDER:
        ge = data[data["experiment"] == experiment]
        if ge.empty:
            continue
        phase_values = sorted(v for v in ge["phase"].dropna().unique())
        if phase_values:
            for phase in phase_values:
                gp = ge[ge["phase"] == phase]
                if gp.empty:
                    continue
                sc = ax.scatter(
                    gp[x_col],
                    gp[y_col],
                    c=gp[color_col],
                    s=90,
                    marker=EXP_PHASE_MARKERS.get((experiment, int(phase)), "o"),
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{experiment} | phase {int(phase)}",
                )
                if first_sc is None:
                    first_sc = sc
        else:
            sc = ax.scatter(
                ge[x_col],
                ge[y_col],
                c=ge[color_col],
                s=90,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.4,
                label=experiment,
            )
            if first_sc is None:
                first_sc = sc

    if first_sc is not None:
        cbar = fig.colorbar(first_sc, ax=ax)
        cbar.set_label(color_label or color_col)
        ax.legend(fontsize=9)

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
                    marker=PHASE_MARKERS.get(int(phase), "o"),
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


def plot_all_methods_mixed_scatter(
    df_ds: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    color_by_accuracy: bool = False,
) -> None:
    data = df_ds[df_ds[x_col].notna() & df_ds[y_col].notna()].copy()
    if color_by_accuracy:
        data = data[data["sample_acc"].notna()].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    learners = ordered_learners(list(data["learner"].dropna().unique()))
    colors = learner_color_map(learners)

    acc_min = data["sample_acc"].min() if color_by_accuracy else None
    acc_max = data["sample_acc"].max() if color_by_accuracy else None
    first_sc = None

    for learner in learners:
        gl = data[data["learner"] == learner]
        if gl.empty:
            continue

        for experiment in EXPERIMENT_ORDER:
            ge = gl[gl["experiment"] == experiment]
            if ge.empty:
                continue

            phase_values = sorted(v for v in ge["phase"].dropna().unique())
            if phase_values:
                for phase in phase_values:
                    gp = ge[ge["phase"] == phase]
                    if gp.empty:
                        continue
                    marker = EXP_PHASE_MARKERS.get((experiment, int(phase)), "o")
                    if color_by_accuracy:
                        sc = ax.scatter(
                            gp[x_col],
                            gp[y_col],
                            c=gp["sample_acc"],
                            s=95,
                            marker=marker,
                            vmin=acc_min,
                            vmax=acc_max,
                            alpha=0.92,
                            edgecolors=colors[learner],
                            linewidths=1.3,
                            label=f"{learner} | {experiment} | phase {int(phase)}",
                        )
                        if first_sc is None:
                            first_sc = sc
                    else:
                        ax.scatter(
                            gp[x_col],
                            gp[y_col],
                            s=85,
                            marker=marker,
                            alpha=0.88,
                            color=colors[learner],
                            edgecolors="black",
                            linewidths=0.4,
                            label=f"{learner} | {experiment} | phase {int(phase)}",
                        )
            else:
                marker = EXP_PHASE_MARKERS.get((experiment, 1), "o")
                if color_by_accuracy:
                    sc = ax.scatter(
                        ge[x_col],
                        ge[y_col],
                        c=ge["sample_acc"],
                        s=95,
                        marker=marker,
                        vmin=acc_min,
                        vmax=acc_max,
                        alpha=0.92,
                        edgecolors=colors[learner],
                        linewidths=1.3,
                        label=f"{learner} | {experiment}",
                    )
                    if first_sc is None:
                        first_sc = sc
                else:
                    ax.scatter(
                        ge[x_col],
                        ge[y_col],
                        s=85,
                        marker=marker,
                        alpha=0.88,
                        color=colors[learner],
                        edgecolors="black",
                        linewidths=0.4,
                        label=f"{learner} | {experiment}",
                    )

    if color_by_accuracy and first_sc is not None:
        cbar = fig.colorbar(first_sc, ax=ax)
        cbar.set_label("sample_acc")

    ax.set_title(title)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods_static_memory_bar(df_ds: pd.DataFrame, out_path: str, dataset: str) -> None:
    data = df_ds[df_ds["static_mb"].notna()].copy()
    if data.empty:
        return

    bars = data.groupby("learner", as_index=False)["static_mb"].median().copy()
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
        color=[colors[str(x)] for x in bars["learner"].astype(str)],
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
    x_positions = {learner: i for i, learner in enumerate(learners)}

    for learner in learners:
        gl = data[data["learner"] == learner]
        if gl.empty:
            continue

        for experiment in EXPERIMENT_ORDER:
            ge = gl[gl["experiment"] == experiment]
            if ge.empty:
                continue

            phase_values = sorted(v for v in ge["phase"].dropna().unique())
            for phase in phase_values:
                gp = ge[ge["phase"] == phase]
                if gp.empty:
                    continue
                ax.scatter(
                    [x_positions[learner]] * len(gp),
                    gp["sample_acc"],
                    s=80,
                    marker=EXP_PHASE_MARKERS.get((experiment, int(phase)), "o"),
                    alpha=0.85,
                    color=colors[learner],
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{learner} | {experiment} | phase {int(phase)}",
                )

    ax.set_title(f"{dataset} | sample_acc vs method")
    ax.set_xlabel("method")
    ax.set_ylabel("Validation sample accuracy (%)")
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def add_per_method_theory_tradeoff_plots(
    g: pd.DataFrame,
    subdir: str,
    dataset: str,
    learner: str,
    saved: List[str],
) -> None:
    for x_col, x_label, filename, title_suffix in THEORY_TRADEOFF_PLOTS:
        p = os.path.join(subdir, filename)
        plot_scatter_experiment_phase_colored(
            g,
            x_col=x_col,
            y_col="sample_acc",
            color_col="hop_ratio",
            out_path=p,
            title=f"{dataset} | {learner} | {title_suffix}",
            x_label=x_label,
            y_label="Validation sample accuracy (%)",
            color_label="hop ratio",
        )
        saved.append(p)


def add_cross_method_theory_tradeoff_plots(
    g: pd.DataFrame,
    ds_dir: str,
    dataset: str,
    saved: List[str],
) -> None:
    for x_col, x_label, filename, title_suffix in THEORY_TRADEOFF_PLOTS:
        stem, ext = os.path.splitext(filename)
        p = os.path.join(ds_dir, f"{stem}_all_methods{ext}")
        plot_all_methods_mixed_scatter(
            g,
            x_col=x_col,
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | {title_suffix}",
            x_label=x_label,
            y_label="Validation sample accuracy (%)",
            color_by_accuracy=False,
        )
        saved.append(p)


def make_per_method_plots(df: pd.DataFrame, out_root: str) -> List[str]:
    ensure_dir(out_root)
    saved: List[str] = []

    grouped = df.groupby(["dataset", "learner"], dropna=False)

    for (dataset, learner), g in grouped:
        if g.empty:
            continue

        subdir = os.path.join(out_root, str(dataset), str(learner))
        ensure_dir(subdir)

        # window-only experiment plots: include both window phases
        g_window = filter_by_experiment(g, "window")

        p = os.path.join(subdir, "sample_acc_vs_window_length.png")
        plot_scatter_phase_simple(
            g_window,
            x_col="win_L",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | sample_acc vs window length",
            x_label="window length",
            y_label="Validation sample accuracy (%)",
        )
        saved.append(p)

        p = os.path.join(subdir, "window_length_vs_training_memory.png")
        plot_scatter_phase_colored(
            g_window,
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
            g_window,
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
            g_window,
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
            g_window,
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

        # hop-only experiment plots: include both hop phases
        g_hop = filter_by_experiment(g, "hop")

        p = os.path.join(subdir, "sample_acc_vs_hop_ratio.png")
        plot_scatter_phase_simple(
            g_hop,
            x_col="hop_ratio",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | sample_acc vs hop ratio",
            x_label="hop ratio",
            y_label="Validation sample accuracy (%)",
        )
        saved.append(p)

        p = os.path.join(subdir, "sample_acc_vs_hop.png")
        plot_scatter_phase_simple(
            g_hop,
            x_col="hop",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | sample_acc vs hop",
            x_label="hop",
            y_label="Validation sample accuracy (%)",
        )
        saved.append(p)

        p = os.path.join(subdir, "hop_ratio_vs_synaptic_ops.png")
        plot_scatter_phase_colored(
            g_hop,
            x_col="hop_ratio",
            y_col="avg_synaptic_ops",
            color_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | hop ratio vs synaptic ops",
            x_label="hop ratio",
            y_label="avg synaptic operations",
            color_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "hop_ratio_vs_firing_rate.png")
        plot_scatter_phase_colored(
            g_hop,
            x_col="hop_ratio",
            y_col="firing_rate",
            color_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | hop ratio vs firing rate",
            x_label="hop ratio",
            y_label="firing rate",
            color_label="sample_acc",
        )
        saved.append(p)

        p = os.path.join(subdir, "hop_ratio_vs_energy.png")
        plot_scatter_phase_colored(
            g_hop,
            x_col="hop_ratio",
            y_col="energy_per_sample_pj",
            color_col="sample_acc",
            out_path=p,
            title=f"{dataset} | {learner} | hop ratio vs energy",
            x_label="hop ratio",
            y_label="energy per sample (pJ)",
            color_label="sample_acc",
        )
        saved.append(p)

        # mixed / neutral plots: include both experiments and both phases
        g_both = filter_by_experiment(g, "both")

        p = os.path.join(subdir, "hop_ratio_vs_window_length.png")
        plot_scatter_experiment_phase_colored(
            g_both,
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

        p = os.path.join(subdir, "sample_acc_vs_training_memory.png")
        plot_scatter_experiment_phase_colored(
            g_both,
            x_col="training_mb_per_batch",
            y_col="sample_acc",
            color_col="hop_ratio",
            out_path=p,
            title=f"{dataset} | {learner} | sample_acc vs training memory",
            x_label="training memory per batch (MB)",
            y_label="Validation sample accuracy (%)",
            color_label="hop ratio",
        )
        saved.append(p)

        add_per_method_theory_tradeoff_plots(g_both, subdir, str(dataset), str(learner), saved)

        p = os.path.join(subdir, "energy_vs_accuracy.png")
        plot_scatter_experiment_phase_colored(
            g_both,
            x_col="energy_per_sample_pj",
            y_col="sample_acc",
            color_col="win_L",
            out_path=p,
            title=f"{dataset} | {learner} | accuracy vs energy per sample",
            x_label="energy per sample (pJ)",
            y_label="Validation sample accuracy (%)",
            color_label="window length",
        )
        saved.append(p)

        p = os.path.join(subdir, "trials.csv")
        export_trials_csv(g_both, p)
        saved.append(p)

    return saved


def make_cross_method_plots(df: pd.DataFrame, out_root: str) -> List[str]:
    saved: List[str] = []

    for dataset, g in df.groupby("dataset", dropna=False):
        if g.empty:
            continue

        ds_dir = os.path.join(out_root, str(dataset), "_all_methods")
        ensure_dir(ds_dir)

        g_window = filter_by_experiment(g, "window")
        g_hop = filter_by_experiment(g, "hop")
        g_both = filter_by_experiment(g, "both")

        p = os.path.join(ds_dir, "sample_acc_vs_window_length_all_methods.png")
        plot_all_methods_scatter(
            g_window,
            x_col="win_L",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | sample_acc vs window length",
            x_label="window length",
            y_label="Validation sample accuracy (%)",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "window_length_vs_training_memory_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g_window,
            x_col="win_L",
            y_col="training_mb_per_batch",
            out_path=p,
            title=f"{dataset} | all methods | window length vs training memory",
            x_label="window length",
            y_label="training memory per batch (MB)",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "window_length_vs_synaptic_ops_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g_window,
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
            g_window,
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
            g_window,
            x_col="win_L",
            y_col="energy_per_sample_pj",
            out_path=p,
            title=f"{dataset} | all methods | window length vs energy",
            x_label="window length",
            y_label="energy per sample (pJ)",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_hop_ratio_all_methods.png")
        plot_all_methods_scatter(
            g_hop,
            x_col="hop_ratio",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | sample_acc vs hop ratio",
            x_label="hop ratio",
            y_label="Validation sample accuracy (%)",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_hop_all_methods.png")
        plot_all_methods_scatter(
            g_hop,
            x_col="hop",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | sample_acc vs hop",
            x_label="hop",
            y_label="Validation sample accuracy (%)",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "hop_ratio_vs_synaptic_ops_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g_hop,
            x_col="hop_ratio",
            y_col="avg_synaptic_ops",
            out_path=p,
            title=f"{dataset} | all methods | hop ratio vs synaptic ops",
            x_label="hop ratio",
            y_label="avg synaptic operations",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "hop_ratio_vs_firing_rate_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g_hop,
            x_col="hop_ratio",
            y_col="firing_rate",
            out_path=p,
            title=f"{dataset} | all methods | hop ratio vs firing rate",
            x_label="hop ratio",
            y_label="firing rate",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "hop_ratio_vs_energy_all_methods.png")
        plot_all_methods_scatter_colored_by_accuracy(
            g_hop,
            x_col="hop_ratio",
            y_col="energy_per_sample_pj",
            out_path=p,
            title=f"{dataset} | all methods | hop ratio vs energy",
            x_label="hop ratio",
            y_label="energy per sample (pJ)",
        )
        saved.append(p)

        p = os.path.join(ds_dir, "hop_ratio_vs_window_length_all_methods.png")
        plot_all_methods_mixed_scatter(
            g_both,
            x_col="win_L",
            y_col="hop_ratio",
            out_path=p,
            title=f"{dataset} | all methods | hop ratio vs win length",
            x_label="window length",
            y_label="hop ratio",
            color_by_accuracy=True,
        )
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_training_memory_all_methods.png")
        plot_all_methods_mixed_scatter(
            g_both,
            x_col="training_mb_per_batch",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | sample_acc vs training memory",
            x_label="training memory per batch (MB)",
            y_label="Validation sample accuracy (%)",
            color_by_accuracy=False,
        )
        saved.append(p)

        add_cross_method_theory_tradeoff_plots(g_both, ds_dir, str(dataset), saved)

        p = os.path.join(ds_dir, "energy_vs_accuracy_all_methods.png")
        plot_all_methods_mixed_scatter(
            g_both,
            x_col="energy_per_sample_pj",
            y_col="sample_acc",
            out_path=p,
            title=f"{dataset} | all methods | accuracy vs energy per sample",
            x_label="energy per sample (pJ)",
            y_label="Validation sample accuracy (%)",
            color_by_accuracy=False,
        )
        saved.append(p)

        p = os.path.join(ds_dir, "static_memory_by_method.png")
        plot_all_methods_static_memory_bar(g_both, out_path=p, dataset=str(dataset))
        saved.append(p)

        p = os.path.join(ds_dir, "sample_acc_vs_method.png")
        plot_all_methods_method_vs_accuracy(g_both, out_path=p, dataset=str(dataset))
        saved.append(p)

        p = os.path.join(ds_dir, "trials.csv")
        export_cross_method_csv(g_both, p)
        saved.append(p)

    return saved


def print_run_count_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No usable runs after filtering.")
        return

    print("\n=== Run Counts Used In Comparison ===")
    count_cols = ["dataset", "learner", "experiment", "phase"]
    have = [c for c in count_cols if c in df.columns]
    if have:
        counts = (
            df.groupby(have, dropna=False)
            .size()
            .reset_index(name="n_runs")
            .sort_values(have)
        )
        print(counts.to_string(index=False))

    total = len(df)
    print(f"\nTotal usable runs: {total}")


def main() -> None:
    ensure_dir(OUT_DIR)
    from utils.optuna_support import export_final_tests
    print(f"Final test table: {export_final_tests(RESULTS_DIR, OUT_DIR, 'independent')}")

    df_runs = load_run_summaries(RESULTS_DIR)
    if df_runs.empty:
        print("No independent window/hop runs found in results/runs/*/summary.json")
        return

    df_optuna = load_optuna_trials(RESULTS_DIR)
    df = maybe_join_optuna(df_runs, df_optuna)

    df = df[df["experiment"].isin(EXPERIMENT_ORDER)].copy()

    print_run_count_summary(df)

    all_trials_csv = save_all_trials_csv(df, OUT_DIR)
    best_csv = save_best_csv(df, OUT_DIR)
    best_phase_csv = save_best_per_phase_csv(df, OUT_DIR)

    saved: List[str] = []
    saved.extend(make_per_method_plots(df, OUT_DIR))
    saved.extend(make_cross_method_plots(df, OUT_DIR))

    print("\n=== Independent Window/Hop Comparison Saved ===")
    print(all_trials_csv)
    print(best_csv)
    print(best_phase_csv)
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
