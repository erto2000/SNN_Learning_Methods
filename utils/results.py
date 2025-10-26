# utils/results_io.py
from __future__ import annotations
import os, json
from typing import Dict, Any, List

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_results(results: List[Dict[str, Any]], base_dir: str = "results") -> List[str]:
    """
    Save each run as results/<RUN_ID>/summary.json.
    The function is intentionally side-effect-only (like summarize) and returns
    the list of saved file paths for convenience.
    """
    saved_paths: List[str] = []
    if not results:
        print("[save_results] No runs to save.")
        return saved_paths

    for r in results:
        run_id = r.get("run_id") or "unknown-run"
        folder = os.path.join(base_dir, run_id)
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
            "final": r.get("final", {}),
            "history": r.get("history", {}),
            "console_log": r.get("console_log", ""),
        }

        out_path = os.path.join(folder, "summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out_path}")
        saved_paths.append(out_path)

    return saved_paths
