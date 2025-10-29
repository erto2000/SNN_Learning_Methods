# compare_results.py
"""
Create comparison plots from finished runs.

- Edit RUNS below (id + pretty name).
- Edit TAG to choose the output subfolder under results/_comparisons/TAG.
- Run:  python compare_results.py
"""

from io.comparison import build_comparison

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
BASE_DIR = "results"
TAG = "har_compare"

# Provide pairs of run_id and a human-friendly name
RUNS = [
    {"id": "har-bp",    "name": "BP (H=128)"},
    {"id": "har-eprop", "name": "E-Prop (H=128)"},
    # {"id": "har-ff",    "name": "FF (H=512)"},
    # {"id": "har-pepita","name": "PEPITA (H=128)"},
]
# ──────────────────────────────────────────────────────────────────────────────

def main():
    arts = build_comparison(BASE_DIR, TAG, RUNS)
    print("\n===== Comparison Summary =====")
    print(f"Output dir: {arts['out_dir']}")
    if arts.get("final_bars"):        print(f"- final_accuracy_comparison.png")
    if arts.get("overlay_loss"):      print(f"- overlay_train_loss.png")
    if arts.get("overlay_train_acc"): print(f"- overlay_train_acc.png")
    if arts.get("overlay_test_acc"):  print(f"- overlay_test_acc.png")
    if arts.get("summary_md"):        print(f"- summary.md")

if __name__ == "__main__":
    main()
