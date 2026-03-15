"""
optuna_window_grid_two_phase.py

Two-phase Optuna tuning for window length (L) and hop on 28 combinations:
Datasets: HAR, Speech Commands, ESC-50, UrbanSound8K, PAMAP2, MIT-BIH, DVS Gesture
Learners: bp, eprop, ff, pepita

Phase 1: 20 trials, 5 epochs (broad)
Phase 2: 10 trials, 10 epochs (refine around best of Phase 1)

- Resumable via sqlite storage
- One Optuna study per (dataset, learner, phase)
- Each trial saved via visualization.training_results.save_results (same as you do today)

Run:
  python optuna_window_grid_two_phase.py
"""

from __future__ import annotations
from copy import deepcopy
import os
import math
import json
import csv
from typing import Dict, Any, Tuple, Optional

import optuna

from utils.runner import run_one
from visualization.training_results import save_results
import timeseries.transforms as transforms


# -----------------------------
# Pipeline helpers
# -----------------------------
def _as_compose(pipeline) -> transforms.Compose:
    if pipeline is None:
        return transforms.Compose([])
    if isinstance(pipeline, transforms.Compose):
        return pipeline
    return transforms.Compose([pipeline])


def remove_window_ops(pipeline) -> transforms.Compose:
    base = _as_compose(pipeline)
    ops = [
        op for op in base.ops
        if not isinstance(op, (transforms.SlidingWindow, transforms.AdaptiveSlidingWindow))
    ]
    return transforms.Compose(ops)


def append_sliding_window(pipeline, *, L: int, hop: int) -> transforms.Compose:
    base = _as_compose(pipeline)
    ops = list(base.ops)
    ops.append(transforms.SlidingWindow(length=int(L), hop=int(hop)))
    return transforms.Compose(ops)


# -----------------------------
# Dataset hints (from your comments)
# These guide sensible L ranges so Optuna isn't wasting trials.
# -----------------------------
DATASET_T_HINT = {
    "har": 128,
    "speech_commands": 101,
    "esc50": 101,
    "urban8k": 436,
    "pamap2": 128,
    "mitbih": 360,
    "dvs_gesture": 200,
}


def _step_for_T(T: int) -> int:
    # Reasonable step so search is not too granular
    # T=101 -> step ~ 8, T=128 -> 8/16, T=360 -> 16/32, T=436 -> 16/32
    raw = max(4, T // 16)
    # round to power-of-two-ish
    p = 2 ** int(round(math.log2(raw)))
    return int(max(4, min(p, max(8, T // 8))))


def bounds_phase1(dataset: str) -> Tuple[int, int, int]:
    """
    Phase 1 broad bounds for L and step.
    """
    T = int(DATASET_T_HINT.get(dataset.lower(), 256))
    L_min = max(10, int(round(T * 0.10)))
    L_max = max(L_min, int(round(T * 1.0)))
    step = _step_for_T(T)
    # clamp to make sure min/max align to step reasonably
    return L_min, L_max, step


def bounds_phase2_from_best(dataset: str, best_L: int) -> Tuple[int, int, int]:
    """
    Phase 2 narrower bounds centered around best_L from phase 1.
    """
    T = int(DATASET_T_HINT.get(dataset.lower(), 256))
    step = _step_for_T(T)

    # +/- 25% neighborhood (clamped)
    lo = int(round(best_L * 0.75))
    hi = int(round(best_L * 1.25))

    # clamp to sensible absolute bounds for that dataset
    abs_lo, abs_hi, _ = bounds_phase1(dataset)
    lo = max(abs_lo, lo)
    hi = min(abs_hi, hi)

    # ensure lo <= hi and at least one step
    if hi < lo:
        lo, hi = abs_lo, abs_hi

    return lo, hi, step


def suggest_L_hop(trial: optuna.Trial, *, L_min: int, L_max: int, step: int) -> Tuple[int, int, float]:
    """
    Suggest (L, hop, hop_ratio). Hop derived from hop_ratio for stability.
    """
    L = trial.suggest_int("win_L", int(L_min), int(L_max), step=int(step))
    hop_ratio = trial.suggest_float("hop_ratio", 0.25, 1.0)
    hop = max(1, int(round(L * hop_ratio)))
    trial.set_user_attr("hop", hop)
    return L, hop, hop_ratio


# -----------------------------
# Persistence helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_study_csv(study: optuna.Study, out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv))
    fieldnames = ["study", "trial", "state", "value", "win_L", "hop_ratio", "hop", "run_id", "status", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in study.trials:
            w.writerow({
                "study": study.study_name,
                "trial": t.number,
                "state": str(t.state),
                "value": t.value,
                "win_L": (t.params or {}).get("win_L"),
                "hop_ratio": (t.params or {}).get("hop_ratio"),
                "hop": (t.user_attrs or {}).get("hop"),
                "run_id": (t.user_attrs or {}).get("run_id"),
                "status": (t.user_attrs or {}).get("status"),
                "error": (t.user_attrs or {}).get("error"),
            })


def export_best_json(study: optuna.Study, out_json: str) -> None:
    ensure_dir(os.path.dirname(out_json))
    payload = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "n_trials": len(study.trials),
        "best_value": study.best_value if len(study.trials) else None,
        "best_params": study.best_params if len(study.trials) else None,
        "best_trial": study.best_trial.number if len(study.trials) else None,
        "best_user_attrs": dict(study.best_trial.user_attrs) if len(study.trials) else None,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def completed_trials(study: optuna.Study) -> int:
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


# -----------------------------
# Objective factory
# -----------------------------
def make_objective(
    base_cfg: Dict[str, Any],
    base_pipeline,
    *,
    dataset: str,
    learner: str,
    phase: int,
    metric: str,
    results_dir: str,
    L_min: int,
    L_max: int,
    step: int,
):
    """
    Each trial:
      - sample L/hop
      - pipeline = base_pipeline + SlidingWindow(L, hop)
      - run_one + save_results
      - return metric
    """
    assert metric in ("sample_acc", "window_acc")

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)

        L, hop, hop_ratio = suggest_L_hop(trial, L_min=L_min, L_max=L_max, step=step)

        cfg["WINDOW"] = {"type": "sliding", "L": int(L), "hop": int(hop), "hop_ratio": float(hop_ratio)}

        pipe0 = remove_window_ops(base_pipeline)
        cfg["TRANSFORM"] = append_sliding_window(pipe0, L=L, hop=hop)

        base_id = cfg.get("RUN_ID", f"{dataset}-{learner}")
        run_id = f"{base_id}-p{phase}-optuna-t{trial.number:04d}-L{L}-H{hop}"
        cfg["RUN_ID"] = run_id

        result = run_one(cfg)
        save_results([result], base_dir=results_dir, make_plots=True)

        trial.set_user_attr("run_id", run_id)
        trial.set_user_attr("status", result.get("status"))

        if result.get("status") != "ok":
            err = result.get("error") or "unknown error"
            trial.set_user_attr("error", err)
            raise optuna.TrialPruned(err)

        return float(result.get("final", {}).get(metric, 0.0))

    return objective


# -----------------------------
# Main runner
# -----------------------------
def run_all_two_phase(
    *,
    results_dir: str = "results",
    db_path: str = "results/optuna/window_hop_two_phase.db",
    metric: str = "sample_acc",
    phase1_trials: int = 20,
    phase1_epochs: int = 5,
    phase2_trials: int = 10,
    phase2_epochs: int = 10,
    max_samples_override: Optional[int] = 8000,
):
    """
    Runs all combos in order, resumable.

    Phase 1: broad L bounds
    Phase 2: narrow around Phase 1 best L

    NOTE: We intentionally separate Phase 1 and Phase 2 into different studies,
    because epochs differ (objective distribution shifts). That keeps Optuna’s
    modeling clean and makes results easier to interpret.
    """
    ensure_dir(os.path.dirname(db_path))
    storage = f"sqlite:///{db_path}"

    # import your pipelines + defaults
    from run_training import (
        DEFAULT,
        HAR_PIPELINE,
        SC_PIPELINE,
        ESC50_PIPELINE,
        URBAN8K_PIPELINE,
        PAMAP2_PIPELINE,
        MITBIH_PIPELINE,
        DVS_GESTURE_PIPELINE,
    )

    dataset_to_pipeline = {
        "har": HAR_PIPELINE,
        "speech_commands": SC_PIPELINE,
        "esc50": ESC50_PIPELINE,
        "urban8k": URBAN8K_PIPELINE,
        "pamap2": PAMAP2_PIPELINE,
        "mitbih": MITBIH_PIPELINE,
        "dvs_gesture": DVS_GESTURE_PIPELINE,
    }

    datasets = ["har", "speech_commands", "esc50", "urban8k", "pamap2", "mitbih", "dvs_gesture"]
    learners = ["bp", "eprop", "ff", "pepita"]

    for ds in datasets:
        base_pipeline = dataset_to_pipeline[ds]

        for learner in learners:
            print(f"\n==============================")
            print(f"Combo: dataset={ds} | learner={learner}")
            print(f"==============================")

            # -----------------
            # Phase 1
            # -----------------
            p1_name = f"ws_{ds}_{learner}_p1"
            L1_min, L1_max, step1 = bounds_phase1(ds)

            cfg1 = {
                **DEFAULT,
                "RUN_ID": f"{ds}-{learner}",
                "DATASET": ds,
                "LEARNER": learner,
                "EPOCHS": int(phase1_epochs),
                "TEST_EVERY_EPOCH": False,
            }
            if max_samples_override is not None:
                cfg1["MAX_SAMPLES"] = int(max_samples_override)

            study1 = optuna.create_study(
                study_name=p1_name,
                direction="maximize",
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=int(cfg1.get("SEED", 123))),
                pruner=optuna.pruners.NopPruner(),
            )

            done1 = completed_trials(study1)
            rem1 = max(0, phase1_trials - done1)
            if rem1 > 0:
                print(f"[Phase 1] {p1_name} | L in [{L1_min},{L1_max}] step={step1} | epochs={phase1_epochs}")
                print(f"[Phase 1] completed={done1} remaining={rem1} target={phase1_trials}")

                obj1 = make_objective(
                    cfg1, base_pipeline,
                    dataset=ds, learner=learner, phase=1,
                    metric=metric, results_dir=results_dir,
                    L_min=L1_min, L_max=L1_max, step=step1,
                )
                study1.optimize(obj1, n_trials=rem1, catch=(Exception,))
                export_study_csv(study1, os.path.join(results_dir, "optuna", f"{p1_name}_trials.csv"))
                export_best_json(study1, os.path.join(results_dir, "optuna", f"{p1_name}_best.json"))
            else:
                print(f"[Phase 1] Skip (already has >= {phase1_trials} COMPLETE trials).")

            # If phase 1 still has no completed trials (e.g., repeated failures), skip phase 2.
            if completed_trials(study1) == 0:
                print("[Phase 2] Skipped because Phase 1 has no COMPLETE trials.")
                continue

            best_L = int(study1.best_params.get("win_L"))
            best_hr = float(study1.best_params.get("hop_ratio"))
            best_hop = int(study1.best_trial.user_attrs.get("hop", max(1, int(round(best_L * best_hr)))))

            # -----------------
            # Phase 2
            # -----------------
            p2_name = f"ws_{ds}_{learner}_p2"
            L2_min, L2_max, step2 = bounds_phase2_from_best(ds, best_L)

            cfg2 = {
                **DEFAULT,
                "RUN_ID": f"{ds}-{learner}",
                "DATASET": ds,
                "LEARNER": learner,
                "EPOCHS": int(phase2_epochs),
                "TEST_EVERY_EPOCH": False,
            }
            # usually keep same cap; if you want full data in phase2, set max_samples_override=None
            if max_samples_override is not None:
                cfg2["MAX_SAMPLES"] = int(max_samples_override)

            study2 = optuna.create_study(
                study_name=p2_name,
                direction="maximize",
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=int(cfg2.get("SEED", 123))),
                pruner=optuna.pruners.NopPruner(),
            )

            # Seed phase 2 with the phase 1 best to make refinement start strong (and reproducible)
            if len(study2.trials) == 0:
                study2.enqueue_trial({"win_L": best_L, "hop_ratio": best_hr})

                # also enqueue a couple neighbors (optional but helpful)
                neigh = []
                for dL in (-step2, step2):
                    Lcand = max(L2_min, min(L2_max, best_L + dL))
                    neigh.append({"win_L": int(Lcand), "hop_ratio": best_hr})
                for x in neigh:
                    study2.enqueue_trial(x)

            done2 = completed_trials(study2)
            rem2 = max(0, phase2_trials - done2)
            if rem2 > 0:
                print(f"[Phase 2] {p2_name} | refine around best_L={best_L}, best_hop={best_hop}")
                print(f"[Phase 2] L in [{L2_min},{L2_max}] step={step2} | epochs={phase2_epochs}")
                print(f"[Phase 2] completed={done2} remaining={rem2} target={phase2_trials}")

                obj2 = make_objective(
                    cfg2, base_pipeline,
                    dataset=ds, learner=learner, phase=2,
                    metric=metric, results_dir=results_dir,
                    L_min=L2_min, L_max=L2_max, step=step2,
                )
                study2.optimize(obj2, n_trials=rem2, catch=(Exception,))
                export_study_csv(study2, os.path.join(results_dir, "optuna", f"{p2_name}_trials.csv"))
                export_best_json(study2, os.path.join(results_dir, "optuna", f"{p2_name}_best.json"))
            else:
                print(f"[Phase 2] Skip (already has >= {phase2_trials} COMPLETE trials).")

            if completed_trials(study2) > 0:
                print(f"[Best Phase2] value={study2.best_value:.4f} params={study2.best_params} hop={study2.best_trial.user_attrs.get('hop')}")
            else:
                print(f"[Best Phase1] value={study1.best_value:.4f} params={study1.best_params} hop={study1.best_trial.user_attrs.get('hop')}")

    print("\n=== Done: all combos processed (two-phase) ===")
    print(f"Optuna DB: {db_path}")
    print(f"Optuna exports: {os.path.join(results_dir, 'optuna')}")


if __name__ == "__main__":
    run_all_two_phase(
        results_dir="results",
        db_path="results/optuna/window_hop_two_phase.db",
        metric="sample_acc",
        phase1_trials=20,
        phase1_epochs=5,
        phase2_trials=10,
        phase2_epochs=10,
        max_samples_override=None
    )