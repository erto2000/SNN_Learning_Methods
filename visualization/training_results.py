# visualization/training_results.py
from __future__ import annotations
import os, json
from typing import Dict, Any, List
from timeseries.splitting import split_report_fields

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _resolve_eval_dtype(final: Dict[str, Any]) -> Any:
    # Producer uses "dtype", older/newer output code may expect "eval_dtype"
    return final.get("eval_dtype", final.get("dtype"))

def save_results(results: List[Dict[str, Any]], base_dir: str = "results", make_plots: bool = True,
                 quiet: bool = False) -> List[str]:
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
        if not quiet:
            print(f"Saved: {out_path}")
        saved_paths.append(out_path)

        final = payload.get("final", {}) or {}
        memory = payload.get("memory", {}) or {}
        theory = memory.get("theory", {}) or {}

        # 2) metrics.json (flat, handy for quick reads)
        metrics = {
            **split_report_fields(payload.get('meta', {})),
            "run_id": run_id,
            "dataset": payload["config"].get("DATASET") if payload.get("config") else None,
            "learner": payload["config"].get("LEARNER") if payload.get("config") else None,
            "epochs": payload["config"].get("EPOCHS") if payload.get("config") else None,
            "status": payload["status"],

            "evaluation_split": final.get("evaluation_split", "test"),
            "final_sample_acc": final.get("sample_acc"),
            "final_window_acc": final.get("window_acc"),

            "avg_spike_count": final.get("avg_spike_count"),
            "firing_rate": final.get("firing_rate"),
            "avg_synaptic_operations": final.get("avg_synaptic_operations"),

            "energy_per_sample_pj": final.get("energy_per_sample_pj"),
            "energy_breakdown_pct": final.get("energy_breakdown_pct"),

            # compatibility with actual producer field
            "eval_dtype": _resolve_eval_dtype(final),
            "eval_int8_weights": final.get("eval_int8_weights"),

            "theory_model_version": theory.get("model_version"),
            "theory_work_scope": theory.get("work_scope"),
            "windows_per_sample": theory.get("windows_per_sample"),
            "theory_memory_bytes": theory.get("memory", {}).get("total_bytes"),
            "theory_memory_scalars": theory.get("memory", {}).get("total_scalars"),
            "theory_compute_scalars": theory.get("compute", {}).get("total_scalars"),
            "theory_access_scalars": theory.get("access", {}).get("total_scalars"),
            "theory_time_proxy": theory.get("time_proxy", {}).get("value"),
        }
        _write_json(metrics, os.path.join(folder, "metrics.json"))

        # 3) optional plots
        if make_plots and payload["status"] == "ok":
            try:
                from .plotting import save_run_plots
                save_run_plots(folder, payload)
            except Exception as e:
                print(f"Plotting failed for {run_id}: {e}")

    return saved_paths
