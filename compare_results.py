# compare_results.py
"""
Per-dataset comparisons (no cross-dataset mixing), with explicit out_path.
All outputs are written RELATIVE to BASE_DIR, which is set to "results/comparisons".

Run:
    python compare_results.py
"""

from visualization.flex_compare import run_comparisons

# Write all artifacts under this directory
BASE_DIR = "results"

# ──────────────────────────────────────────────────────────────
# Custom metric functions
# ──────────────────────────────────────────────────────────────
def max_test_acc(info):
    """Return max test accuracy across epochs (ignores None)."""
    hist = (info.get("summary") or {}).get("history", {}) or {}
    vals = []
    for _, d in hist.items():
        v = (d or {}).get("acc")
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None

def best_epoch(info):
    """Epoch index at which accuracy is maximal; None if no history."""
    hist = (info.get("summary") or {}).get("history", {}) or {}
    best, best_ep = None, None
    for k, d in hist.items():
        v = (d or {}).get("acc")
        if v is None:
            continue
        if best is None or v > best:
            best, best_ep = v, int(k)
    return best_ep

def sec_per_epoch(info):
    """Wall-clock sec per epoch from duration / EPOCHS."""
    dur = info.get("summary", {}).get("duration_seconds")
    epochs = ((info.get("summary", {}) or {}).get("config") or {}).get("EPOCHS")
    if dur is None or not epochs:
        return None
    return float(dur) / float(epochs)

def hs_total(info):
    """Sum of hidden sizes (handy numeric axis)."""
    hs = ((info.get("summary", {}) or {}).get("config") or {}).get("HIDDEN_SIZES") or []
    try:
        return sum(int(h) for h in hs)
    except Exception:
        return None

def hs_depth(info):
    """Depth (# hidden layers)."""
    hs = ((info.get("summary", {}) or {}).get("config") or {}).get("HIDDEN_SIZES") or []
    return len(hs)

def hs_str(info):
    """Hidden sizes as string (for labeling)."""
    hs = ((info.get("summary", {}) or {}).get("config") or {}).get("HIDDEN_SIZES") or []
    return "x".join(str(h) for h in hs)

CUSTOM_FUNCS = {
    "max_test_acc": max_test_acc,
    "best_epoch": best_epoch,
    "sec_per_epoch": sec_per_epoch,
    "hs_total": hs_total,
    "hs_depth": hs_depth,
    "hs_str": hs_str,
}

# ──────────────────────────────────────────────────────────────
# Datasets & learners
# ──────────────────────────────────────────────────────────────
DATASETS = [
    "har", "mnist", "speech_commands", "esc50",
    "urban8k", "pamap2", "mitbih", "dvs_gesture"
]
LEARNERS = "bp|ff|eprop|pepita"

# Toggle to also export per-epoch CSV for each dataset (one row/epoch/run)
EMIT_PER_EPOCH_CSV = False

def pretty_ds(ds):
    return {
        "har": "HAR",
        "mnist": "MNIST (temporalized)",
        "speech_commands": "Speech Commands",
        "esc50": "ESC-50",
        "urban8k": "UrbanSound8K",
        "pamap2": "PAMAP2",
        "mitbih": "MIT-BIH",
        "dvs_gesture": "DVS Gesture",
    }.get(ds, ds)

def build_for_dataset(ds: str):
    """
    Build comparison specs for a single dataset.
    Each comparison sets out_path so files land under:
        results/comparisons/<ds>/<section>/
    """
    title_ds = pretty_ds(ds)
    runs_all       = f"re:^{ds}-({LEARNERS})-"
    runs_baseline  = f"re:^{ds}-({LEARNERS})-nowin-bs64-h128-e10$"
    runs_win_pair  = f"re:^{ds}-({LEARNERS})-(nowin|win)-bs64-h128-e10$"
    runs_batch     = f"re:^{ds}-({LEARNERS})-nowin-bs(64|128)-h128-e10$"
    runs_hidden    = f"re:^{ds}-({LEARNERS})-nowin-bs64-h(128|128x128|512|512x512)-e10$"

    comps = [
        # 1) Baseline: per-learner epoch curves
        {
            "name": f"[{title_ds}] Epoch curves — baseline (nowin, bs64, h=128, e10)",
            "runs": [runs_baseline],
            "dest": {
                "type": "plot",
                "out_path": f"{ds}/baseline",  # relative to BASE_DIR
                "panels": [
                    {"x": "epoch", "y": "loss", "plot": "line", "title": "Train Loss vs Epoch"},
                    {"x": "epoch", "y": "acc",  "plot": "line", "title": "Acc (%) vs Epoch"},
                    {"x": "run",   "y": "final.sample_acc", "plot": "bar",  "title": "Final Acc by Learner"},
                ],
                "style": {"dpi": 140, "figsize": [12, 6], "tight_layout": True}
            }
        },

        # 2) Window ON/OFF effect (paired bars, scatter trade-off)
        {
            "name": f"[{title_ds}] Window ON vs OFF (bs64, h=128, e10)",
            "runs": [runs_win_pair],
            "dest": {
                "type": "plot",
                "out_path": f"{ds}/window",
                "panels": [
                    {"x": "run", "y": "final.sample_acc", "plot": "bar", "title": "Final Acc — Window ON/OFF"},
                    {"x": "func:max_test_acc", "y": "func:sec_per_epoch", "plot": "scatter",
                     "title": "Max Acc vs Sec/Epoch (window trade-off)"},
                ],
                "style": {"dpi": 140, "figsize": [12, 6]}
            }
        },

        # 3) Batch sweep (64 vs 128), holding others
        {
            "name": f"[{title_ds}] Batch sweep (64 vs 128) — nowin, h=128, e10",
            "runs": [runs_batch],
            "dest": {
                "type": "plot",
                "out_path": f"{ds}/batch",
                "panels": [
                    {"x": "config.BATCH_SIZE", "y": "final.sample_acc", "plot": "scatter",
                     "title": "Final Acc vs Batch Size"},
                    {"x": "config.BATCH_SIZE", "y": "func:sec_per_epoch", "plot": "line",
                     "title": "Sec/Epoch vs Batch Size"},
                ],
                "style": {"dpi": 140, "figsize": [10, 5]}
            }
        },

        # 4) Hidden-size sweep, holding others
        {
            "name": f"[{title_ds}] Hidden-size sweep — nowin, bs64, e10",
            "runs": [runs_hidden],
            "dest": {
                "type": "plot",
                "out_path": f"{ds}/hidden",
                "panels": [
                    {"x": "func:hs_total", "y": "final.sample_acc", "plot": "line",
                     "title": "Final Acc vs Σ Hidden"},
                    {"x": "func:hs_depth", "y": "func:max_test_acc", "plot": "scatter",
                     "title": "Max Acc vs Depth (#layers)"},
                ],
                "style": {"dpi": 140, "figsize": [10, 5]}
            }
        },

        # 5) Dataset leaderboard (all runs for this dataset)
        {
            "name": f"[{title_ds}] Leaderboard (CSV) — all runs",
            "runs": [runs_all],
            "dest": {
                "type": "csv",
                "out_path": f"{ds}/tables",
                "spread_epochs": False,
                "columns": [
                    {"name": "run_id",          "value": "run_id"},
                    {"name": "learner",         "value": "config.LEARNER"},
                    {"name": "batch",           "value": "config.BATCH_SIZE"},
                    {"name": "hidden",          "func": "hs_str"},
                    {"name": "epochs",          "value": "config.EPOCHS"},
                    {"name": "final_acc",       "value": "final.sample_acc"},
                    {"name": "max_epoch_acc",   "func": "max_test_acc"},
                    {"name": "best_epoch_idx",  "func": "best_epoch"},
                    {"name": "sec_per_epoch",   "func": "sec_per_epoch"},
                    {"name": "duration_sec",    "value": "duration_seconds"},
                    {"name": "status",          "value": "status"},
                    {"name": "started_at",      "value": "started_at"},
                    {"name": "finished_at",     "value": "finished_at"},
                ]
            }
        },
    ]

    if EMIT_PER_EPOCH_CSV:
        comps.append({
            "name": f"[{title_ds}] Per-epoch (CSV) — all runs",
            "runs": [runs_all],
            "dest": {
                "type": "csv",
                "out_path": f"{ds}/tables",
                "spread_epochs": True,
                "columns": [
                    {"name": "run_id",     "value": "run_id"},
                    {"name": "learner",    "value": "config.LEARNER"},
                    {"name": "epoch",      "value": "epoch"},
                    {"name": "train_loss", "value": "history.loss"},
                    {"name": "train_acc",  "value": "history.acc"},
                    {"name": "test_acc",   "value": "history.sample_acc"},
                    {"name": "final_acc",  "value": "final.sample_acc"},
                    {"name": "max_test_acc", "func": "max_test_acc"},
                ]
            }
        })
    return comps

# ──────────────────────────────────────────────────────────────
# Build COMPARISONS: per dataset, no cross-dataset plots
# ──────────────────────────────────────────────────────────────
COMPARISONS = []
for ds in DATASETS:
    COMPARISONS.extend(build_for_dataset(ds))

def main():
    arts = run_comparisons(base_dir=BASE_DIR, comparisons=COMPARISONS, custom_funcs=CUSTOM_FUNCS)
    print("\n===== Comparison Summary =====")
    for comp_name, produced in arts.items():
        print(f"[{comp_name}]")
        for k, v in produced.items():
            if isinstance(v, list):
                for p in v: print(f"- {p}")
            elif v:
                print(f"- {k}: {v}")

if __name__ == "__main__":
    main()
