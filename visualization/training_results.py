# visualization/training_results.py
from __future__ import annotations
import os, json
from typing import Dict, Any, List

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_results(results: List[Dict[str, Any]], base_dir: str = "results", make_plots: bool = True) -> List[str]:
    """
    Save each run as results/<RUN_ID>/summary.json and (optionally) plots.
    Returns list of saved summary.json paths.
    """
    saved_paths: List[str] = []
    if not results:
        print("[save_results] No runs to save.")
        return saved_paths

    for r in results:
        run_id = r.get("run_id") or "unknown-run"
        folder = os.path.join(base_dir, "runs", run_id)
        _safe_mkdir(folder)

        payload = {
            "run_id": run_id,
            "status": r.get("status", "ok"),
            "error": r.get("error"),
            "traceback": r.get("traceback"),
            "started_at": r.get("started_at"),
            "finished_at": r.get("finished_at"),
            "duration_seconds": r.get("duration_seconds"),
            "config": r.get("config"),
            "meta": r.get("meta", {}),
            "memory": r.get("memory", {}),
            "final": r.get("final", {}),
            "history": r.get("history", {}),
            "console_log": r.get("console_log", ""),
        }

        # 1) summary.json
        out_path = os.path.join(folder, "summary.json")
        _write_json(payload, out_path)
        print(f"[Saved] {out_path}")
        saved_paths.append(out_path)

        # 2) metrics.json (flat, handy for quick reads)
        metrics = {
            "run_id": run_id,
            "dataset": payload["config"].get("DATASET") if payload.get("config") else None,
            "learner": payload["config"].get("LEARNER") if payload.get("config") else None,
            "epochs": payload["config"].get("EPOCHS") if payload.get("config") else None,
            "status": payload["status"],

            "final_sample_acc": payload["final"].get("sample_acc"),
            "final_window_acc": payload["final"].get("window_acc"),

            "avg_spike_count": payload["final"].get("avg_spike_count"),
            "firing_rate": payload["final"].get("firing_rate"),
            "avg_synaptic_operations": payload["final"].get("avg_synaptic_operations"),

            "energy_per_sample_pj": payload["final"].get("energy_per_sample_pj"),
            "energy_breakdown_pct": payload["final"].get("energy_breakdown_pct"),

            "eval_dtype": payload["final"].get("eval_dtype"),
            "eval_int8_weights": payload["final"].get("eval_int8_weights"),
        }
        _write_json(metrics, os.path.join(folder, "metrics.json"))

        # 3) optional plots
        if make_plots and payload["status"] == "ok":
            try:
                from .plotting import save_run_plots
                save_run_plots(folder, payload)
            except Exception as e:
                print(f"[save_results] Plotting failed for {run_id}: {e}")

    return saved_paths
