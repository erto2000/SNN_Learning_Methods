# io/comparison.py
from __future__ import annotations
import os, json
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _load_summary(base_dir: str, run_id: str) -> Dict[str, Any]:
    path = os.path.join(base_dir, run_id, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary.json for run '{run_id}' at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_epoch_series(history: Dict[str, Any], key: str) -> Tuple[list, list]:
    if not history:
        return [], []
    # handle epochs possibly as str or int keys after JSON round-trip
    epochs = sorted(int(e) for e in history.keys())
    ys = [history[str(e) if str(e) in history else e].get(key) for e in epochs]
    return epochs, ys

def compare_bar_final_acc(runs_info: List[Dict[str, Any]], out_dir: str) -> str:
    names = [r["name"] for r in runs_info]
    vals = [r["summary"].get("final", {}).get("sample_acc") for r in runs_info]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    bars = ax.bar(names, vals, color="#1f77b4")
    ax.set_ylabel("Final Test Acc (%)")
    ax.set_title("Final Accuracy Comparison")
    ymax = max([v or 0 for v in vals] + [100])
    ax.set_ylim(0, ymax * 1.05)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    fig.tight_layout()

    path = os.path.join(out_dir, "final_accuracy_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path

def compare_overlay_curve(runs_info: List[Dict[str, Any]], key: str, ylabel: str,
                          title: str, filename: str, out_dir: str) -> str | None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    any_data = False
    for r in runs_info:
        epochs, ys = _get_epoch_series(r["summary"].get("history", {}), key)
        if ys and any(y is not None for y in ys):
            any_data = True
            ax.plot(epochs, ys, label=r["name"], linewidth=2)

    if not any_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path

def write_markdown_table(runs_info: List[Dict[str, Any]], out_dir: str) -> str:
    lines = [
        "# Comparison Summary\n",
        "| Run ID | Name | Dataset | Learner | Epochs | Final Test Acc (%) | Status |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for r in runs_info:
        s = r["summary"]
        cfg = s.get("config", {}) or {}
        final = s.get("final", {}) or {}
        lines.append(
            f"| {s.get('run_id','?')} | {r['name']} | {cfg.get('DATASET','?')} "
            f"| {cfg.get('LEARNER','?')} | {cfg.get('EPOCHS','?')} "
            f"| {final.get('sample_acc','n/a')} | {s.get('status','?')} |"
        )

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return md_path

def collect_runs(base_dir: str, run_specs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    runs_info = []
    for spec in run_specs:
        rid, name = spec["id"], spec["name"]
        summary = _load_summary(base_dir, rid)
        runs_info.append({"id": rid, "name": name, "summary": summary})
    return runs_info

def build_comparison(base_dir: str, tag: str, run_specs: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Orchestrates comparison artifacts. Returns dict of produced files.
    """
    out_dir = os.path.join(base_dir, "_comparisons", tag)
    _ensure_dir(out_dir)

    runs_info = collect_runs(base_dir, run_specs)

    # Freeze the input spec
    with open(os.path.join(out_dir, "inputs.json"), "w", encoding="utf-8") as f:
        json.dump(run_specs, f, indent=2)

    artifacts = {}
    artifacts["final_bars"] = compare_bar_final_acc(runs_info, out_dir)
    artifacts["overlay_loss"] = compare_overlay_curve(
        runs_info, "loss", "Loss", "Train Loss (overlay)", "overlay_train_loss.png", out_dir
    )
    artifacts["overlay_train_acc"] = compare_overlay_curve(
        runs_info, "acc", "Train Acc (%)", "Train Accuracy (overlay)", "overlay_train_acc.png", out_dir
    )
    artifacts["overlay_test_acc"] = compare_overlay_curve(
        runs_info, "sample_acc", "Test Acc (%)", "Test Accuracy (overlay)", "overlay_test_acc.png", out_dir
    )
    artifacts["summary_md"] = write_markdown_table(runs_info, out_dir)
    artifacts["out_dir"] = out_dir
    return artifacts
