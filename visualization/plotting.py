# visualization/plotting.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import math
import json
import matplotlib.pyplot as plt

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _epoch_series(epoch_log: Dict[int, Dict[str, float]], key: str) -> List[float]:
    # epochs are 1..E; return list aligned to epoch index
    if not epoch_log:
        return []
    E = max(int(e) for e in epoch_log.keys())
    out = []
    for e in range(1, E + 1):
        out.append(epoch_log.get(e, {}).get(key))
    return out

def save_training_curves(run_dir: str, summary: Dict[str, Any]) -> None:
    """
    Saves:
      - training_curves.png : loss / train-acc / test-acc (when available)
      - epoch_log.csv       : tabular epoch log (for quick access)
    """
    _ensure_dir(run_dir)
    epoch_log: Dict[int, Dict[str, float]] = summary.get("history", {}) or {}
    if not epoch_log:
        return

    # CSV export for convenience
    csv_path = os.path.join(run_dir, "epoch_log.csv")
    # Header
    keys = set()
    for _, d in epoch_log.items():
        keys.update(d.keys())
    ordered_cols = ["timestamp", "loss", "acc", "sample_acc"] + sorted(k for k in keys if k not in {"timestamp","loss","acc","sample_acc"})
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch," + ",".join(ordered_cols) + "\n")
        for e in sorted(epoch_log):
            row = [str(e)]
            for k in ordered_cols:
                v = epoch_log[e].get(k, "")
                row.append(str(v) if v is not None else "")
            f.write(",".join(row) + "\n")

    # Plots
    loss = _epoch_series(epoch_log, "loss")
    acc_tr = _epoch_series(epoch_log, "acc")
    acc_te = _epoch_series(epoch_log, "sample_acc")  # test accuracy (eval_epoch)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=140)
    epochs = list(range(1, len(loss) + 1))

    if any(v is not None for v in loss):
        ax.plot(epochs, loss, label="Train Loss", color="#d62728", linewidth=2)
    if any(v is not None for v in acc_tr):
        ax.plot(epochs, acc_tr, label="Train Acc (%)", color="#1f77b4", linewidth=2)
    if any(v is not None for v in acc_te):
        ax.plot(epochs, acc_te, label="Test Acc (%)", color="#2ca02c", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_title("Training Curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curves.png"))
    plt.close(fig)

def save_final_card(run_dir: str, summary: Dict[str, Any]) -> None:
    """
    Makes a small 'final_card.png' summarizing final metrics.
    """
    _ensure_dir(run_dir)
    final = summary.get("final", {}) or {}
    config = summary.get("config", {}) or {}

    label_lines = [
        f"Run ID: {summary.get('run_id','?')}",
        f"Dataset: {config.get('DATASET','?')}",
        f"Learner: {config.get('LEARNER','?')}",
        f"Epochs: {config.get('EPOCHS','?')}",
        f"Hidden: {config.get('HIDDEN_SIZES','?')}",
        f"TestEveryEpoch: {config.get('TEST_EVERY_EPOCH','?')}",
        f"Final sample_acc: {final.get('sample_acc','n/a')}",
        f"Final window_acc: {final.get('window_acc', 'n/a')}",
        f"Avg spikes/sample: {final.get('avg_spike_count', 'n/a')}",
        f"Firing rate: {final.get('firing_rate', 'n/a')}",
        f"Avg SynOps/sample: {final.get('avg_synaptic_operations', 'n/a')}",
        f"Energy per sample (uJ): {final.get('energy_per_sample_uj', 'n/a')}",
        f"Eval dtype: {final.get('eval_dtype', 'n/a')}",
        f"Eval int8 weights: {final.get('eval_int8_weights', 'n/a')}",
    ]
    text = "\n".join(label_lines)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
    ax.axis("off")
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "final_card.png"))
    plt.close(fig)

def save_run_plots(run_dir: str, summary: Dict[str, Any]) -> None:
    save_training_curves(run_dir, summary)
    save_final_card(run_dir, summary)
