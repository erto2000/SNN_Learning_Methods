# compare_results.py
"""
Flexible comparisons (opinionated paths):
- All outputs are saved under: results/comparisons/<slug>/
- CSV file names are stems; '.csv' is appended automatically.
- Run: python compare_results.py
"""

from visualization.flex_compare import run_comparisons

BASE_DIR = "results"

COMPARISONS = [
    # 1) Multi-panel figure:
    #    - Left: per-epoch line overlays (epoch on x)
    #    - Right: per-run scatter using scalar metrics
    {
        "name": "HAR: BP vs E-Prop",
        "runs": ["dvs_gesture-bp-nowin-bs64-h128-e10", "dvs_gesture-bp-nowin-bs64-h128x128-e10"],
        "name_map": { "har-bp": "BP (H=128)", "har-eprop": "E-Prop (H=128)" },
        "dest": {
            "type": "plot",
            "panels": [
                {"x": "epoch", "y": "loss",        "plot": "line",    "title": "Train Loss vs Epoch"},
                {"x": "epoch", "y": "acc",  "plot": "line",    "title": "Test Acc (%) vs Epoch"},
                # scatter: each dot = one run, axes are scalar metrics
                {"x": "final.sample_acc", "y": "config.HIDDEN_SIZES", "plot": "scatter", "title": "Final Acc vs Hidden Size"},
                # custom function on axis:
                {"x": "func:max_test_acc", "y": "config.EPOCHS", "plot": "scatter", "title": "Max Test Acc vs EPOCHS"},
                {"x": "run", "y": "final.sample_acc", "plot": "bar"},
            ],
            "style": {"dpi": 140}
        }
    },

    # 2) CSV example unchanged—works with the new plotting rules too
    {
        "name": "HAR CSV per-epoch",
        "runs": ["re:^har-"],
        "dest": {
            "type": "csv",
            "file_stem": "har_epochwise",
            "spread_epochs": True,
            "columns": [
                {"name": "run_id",     "value": "run_id"},
                {"name": "name",       "value": "pretty_name"},
                {"name": "epoch",      "value": "epoch"},
                {"name": "info",       "template": "{epoch} --- {timestamp}"},
                {"name": "train_loss", "value": "history.loss"},
                {"name": "train_acc",  "value": "history.acc"},
                {"name": "test_acc",   "value": "history.sample_acc"},
                {"name": "final_acc",  "value": "final.sample_acc"},
                {"name": "max_test_acc", "func": "max_test_acc"},
            ]
        }
    },
]

# Custom metric functions available to CSV and to plot "func:<name>" paths
def max_test_acc(info):
    """Return max test accuracy across epochs (ignores None)."""
    hist = info["summary"].get("history", {}) or {}
    vals = []
    for _, d in hist.items():
        v = (d or {}).get("acc")
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None

CUSTOM_FUNCS = {
    "max_test_acc": max_test_acc,
}

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
