from __future__ import annotations

from copy import deepcopy
import os
import math
import json
import csv
from typing import Dict, Any, Tuple, Optional

import optuna
from utils.optuna_support import create_study, preferred_study, run_final_test, export_final_tests
from utils.window_search import inspect_training_lengths
from utils.console import panel, close, timestamp

from utils.runner import run_one
from visualization.training_results import save_results
import timeseries.transforms as transforms


# =========================================================
# Pipeline helpers
# =========================================================
def _as_compose(pipeline) -> transforms.Compose:
    if pipeline is None:
        return transforms.Compose([])
    if isinstance(pipeline, transforms.Compose):
        return pipeline
    return transforms.Compose([pipeline])


def remove_window_ops(pipeline) -> transforms.Compose:
    """
    Remove any existing SlidingWindow / AdaptiveSlidingWindow ops from a pipeline.
    This ensures Optuna controls windowing and nothing else changes.
    """
    base = _as_compose(pipeline)
    ops = [
        op for op in base.ops
        if not isinstance(op, (transforms.SlidingWindow, transforms.AdaptiveSlidingWindow))
    ]
    return transforms.Compose(ops)


def append_sliding_window(pipeline, *, L: int, hop: int) -> transforms.Compose:
    """
    Append one SlidingWindow op at the end of the existing non-window pipeline.
    """
    base = _as_compose(pipeline)
    ops = list(base.ops)
    ops.append(transforms.SlidingWindow(length=int(L), hop=int(hop)))
    return transforms.Compose(ops)


def try_get_baseline_window_from_pipeline(pipeline) -> Optional[Dict[str, int]]:
    """
    Best-effort extraction of the baseline sliding-window config from the original
    dataset pipeline. Kept only as a utility/debug helper; hop-ratio tuning below
    now uses the best window length found by the window-length experiment.

    Returns {"L": ..., "hop": ...} or None if no static window op is found.
    """
    base = _as_compose(pipeline)
    for op in getattr(base, "ops", []):
        if isinstance(op, transforms.SlidingWindow):
            length = getattr(op, "length", None)
            hop = getattr(op, "hop", None)
            if length is not None and hop is not None:
                return {"L": int(length), "hop": int(hop)}

        if isinstance(op, transforms.AdaptiveSlidingWindow):
            length = getattr(op, "length", None)
            hop = getattr(op, "hop", None)
            if length is not None and hop is not None:
                return {"L": int(length), "hop": int(hop)}

    return None


# =========================================================
# Baseline config alignment
# =========================================================
def get_base_run_config_by_dataset(dataset: str) -> Dict[str, Any]:
    """
    Pull the matching dataset configuration used by full-sequence runs.
    """
    from experiment_config import RUNS

    dataset = dataset.lower()
    requested_prefix = dataset.replace("_", "-")
    # A dataset can have multiple temporal representations, as with MNIST
    # static and rate coding. Prefer the explicit run-prefix identity first.
    for r in RUNS:
        run_prefix = r["RUN_ID"].rsplit("-", 1)[0].lower()
        if run_prefix == requested_prefix:
            return deepcopy(r)
    for r in RUNS:
        if r["DATASET"].lower() == dataset:
            return deepcopy(r)

    raise ValueError(f"No run configuration found for dataset={dataset!r}")


# =========================================================
# Window-length bounds
# =========================================================
def _step_for_T(T: int) -> int:
    raw = max(4, T // 16)
    p = 2 ** int(round(math.log2(raw)))
    return int(max(4, min(p, max(8, T // 8))))


def bounds_phase1_window(max_time_steps: int) -> Tuple[int, int, int]:
    """
    Broad search for the independent window-length experiment.
    """
    T = int(max_time_steps)
    L_min = max(10, int(round(T * 0.10)))
    L_max = max(L_min, int(round(T * 1.0)))
    step = _step_for_T(T)
    return L_min, L_max, step


def bounds_phase2_window(best_L: int, *, max_time_steps: int) -> Tuple[int, int, int]:
    """
    Refinement search for the independent window-length experiment.
    Uses the user's intended 0.75x .. 1.25x neighborhood around the best
    phase-1 window.
    """
    T = int(max_time_steps)
    base_step = _step_for_T(T)
    step = max(2, base_step // 2)

    L_min = max(4, int(round(best_L * 0.75)))
    L_max = min(T, max(L_min, int(round(best_L * 1.25))))

    # Snap to step so Optuna gets a valid integer grid.
    L_min = max(step, int(math.floor(L_min / step) * step))
    L_max = min(T, max(L_min, int(math.ceil(L_max / step) * step)))
    return L_min, L_max, step


def bounds_phase1_hop_ratio() -> Tuple[float, float]:
    """
    Broad search for the independent hop-ratio experiment.
    """
    return 0.25, 1.25


def bounds_phase2_hop_ratio(best_hr: float) -> Tuple[float, float]:
    """
    Refinement search for the independent hop-ratio experiment.
    Uses the user's intended 0.75x .. 1.25x neighborhood around the best
    phase-1 hop ratio.
    """
    lo = max(0.05, float(best_hr) * 0.75)
    hi = min(2.0, float(best_hr) * 1.25)
    if hi <= lo:
        hi = min(2.0, lo + 0.05)
    return lo, hi


# =========================================================
# Suggestion helpers
# =========================================================
def suggest_window_only(
    trial: optuna.Trial,
    *,
    L_min: int,
    L_max: int,
    step: int,
    fixed_hop_ratio: float,
) -> Tuple[int, int, float]:
    """
    Independent window-length experiment:
      - search only window length
      - keep hop_ratio fixed
      - hop = round(L * hop_ratio)
    """
    L = trial.suggest_int("win_L", int(L_min), int(L_max), step=int(step))
    hop_ratio = float(fixed_hop_ratio)
    hop = max(1, int(round(L * hop_ratio)))

    trial.set_user_attr("search_mode", "window_only")
    trial.set_user_attr("fixed_hop_ratio", hop_ratio)
    trial.set_user_attr("hop", hop)

    return L, hop, hop_ratio


def suggest_hop_only(
    trial: optuna.Trial,
    *,
    fixed_L: int,
    hr_min: float,
    hr_max: float,
) -> Tuple[int, int, float]:
    """
    Independent hop-ratio experiment:
      - keep window length fixed
      - search only hop_ratio
    """
    L = int(fixed_L)
    hop_ratio = trial.suggest_float("hop_ratio", float(hr_min), float(hr_max))
    hop = max(1, int(round(L * hop_ratio)))

    trial.set_user_attr("search_mode", "hop_only")
    trial.set_user_attr("fixed_L", L)
    trial.set_user_attr("hop", hop)

    return L, hop, hop_ratio


# =========================================================
# Persistence helpers
# =========================================================
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def export_study_csv(study: optuna.Study, out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv))
    fieldnames = [
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
        "run_id",
        "status",
        "error",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for t in study.trials:
            w.writerow({
                "study": study.study_name,
                "trial": t.number,
                "state": str(t.state),
                "value": t.value,
                "win_L": (t.params or {}).get("win_L", (t.user_attrs or {}).get("fixed_L")),
                "hop_ratio": (t.params or {}).get("hop_ratio", (t.user_attrs or {}).get("fixed_hop_ratio")),
                "hop": (t.user_attrs or {}).get("hop"),
                "experiment": (t.user_attrs or {}).get("experiment"),
                "phase": (t.user_attrs or {}).get("phase"),
                "search_mode": (t.user_attrs or {}).get("search_mode"),
                "run_id": (t.user_attrs or {}).get("run_id"),
                "status": (t.user_attrs or {}).get("status"),
                "error": (t.user_attrs or {}).get("error"),
            })


def export_best_json(study: optuna.Study, out_json: str) -> None:
    ensure_dir(os.path.dirname(out_json))

    if len(study.trials) == 0 or completed_trials(study) == 0:
        payload = {
            "study_name": study.study_name,
            "direction": str(study.direction),
            "n_trials": len(study.trials),
            "n_complete_trials": completed_trials(study),
            "best_value": None,
            "best_params": None,
            "best_trial": None,
            "best_user_attrs": None,
        }
    else:
        payload = {
            "study_name": study.study_name,
            "direction": str(study.direction),
            "n_trials": len(study.trials),
            "n_complete_trials": completed_trials(study),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "best_user_attrs": dict(study.best_trial.user_attrs),
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def completed_trials(study: optuna.Study) -> int:
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def best_window_anchor_from_studies(
    study_w1: optuna.Study,
    study_w2: Optional[optuna.Study],
) -> Optional[Dict[str, Any]]:
    """
    Use the refined phase when available, otherwise fall back to Phase 1.
    This result is used as the fixed window length for the hop-ratio experiment.
    """
    candidates = []

    if completed_trials(study_w1) > 0:
        candidates.append({
            "value": float(study_w1.best_value),
            "L": int(study_w1.best_params["win_L"]),
            "phase": 1,
            "study_name": study_w1.study_name,
        })

    if study_w2 is not None and completed_trials(study_w2) > 0:
        candidates.append({
            "value": float(study_w2.best_value),
            "L": int(study_w2.best_params["win_L"]),
            "phase": 2,
            "study_name": study_w2.study_name,
        })

    if not candidates:
        return None

    return max(candidates, key=lambda x: x["phase"])


# =========================================================
# Objective factory
# =========================================================
def make_objective(
    base_cfg: Dict[str, Any],
    base_pipeline,
    *,
    dataset: str,
    learner: str,
    experiment: str,  # "window" or "hop"
    phase: int,       # 1 or 2
    metric: str,
    results_dir: str,
    search_mode: str,  # "window_only" or "hop_only"
    L_min: Optional[int] = None,
    L_max: Optional[int] = None,
    step: Optional[int] = None,
    fixed_L: Optional[int] = None,
    hr_min: Optional[float] = None,
    hr_max: Optional[float] = None,
    fixed_hop_ratio: Optional[float] = None,
):
    assert metric in ("sample_acc", "window_acc")
    assert search_mode in ("window_only", "hop_only")
    assert experiment in ("window", "hop")

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)

        if search_mode == "window_only":
            assert L_min is not None and L_max is not None and step is not None
            assert fixed_hop_ratio is not None
            L, hop, hop_ratio = suggest_window_only(
                trial,
                L_min=L_min,
                L_max=L_max,
                step=step,
                fixed_hop_ratio=fixed_hop_ratio,
            )
        else:
            assert fixed_L is not None and hr_min is not None and hr_max is not None
            L, hop, hop_ratio = suggest_hop_only(
                trial,
                fixed_L=fixed_L,
                hr_min=hr_min,
                hr_max=hr_max,
            )

        cfg["WINDOW"] = {
            "type": "sliding",
            "L": int(L),
            "hop": int(hop),
            "hop_ratio": float(hop_ratio),
        }

        pipe0 = remove_window_ops(base_pipeline)
        cfg["TRANSFORM"] = append_sliding_window(pipe0, L=L, hop=hop)

        # Build the identity from the current experiment variant and learner.
        # The baseline config may have been copied from the first learner entry.
        base_id = f"{dataset}-{learner}-v3"
        run_id = (
            f"{base_id}-exp{experiment}-p{phase}-{search_mode}-"
            f"optuna-t{trial.number:04d}-L{L}-H{hop}"
        )
        cfg["RUN_ID"] = run_id

        result = run_one(cfg, tuning=True)
        save_results([result], base_dir=results_dir, make_plots=True, quiet=True)

        trial.set_user_attr("experiment", experiment)
        trial.set_user_attr("phase", int(phase))
        trial.set_user_attr("run_id", run_id)
        trial.set_user_attr("status", result.get("status"))
        trial.set_user_attr("L", int(L))
        trial.set_user_attr("hop_ratio", float(hop_ratio))
        trial.set_user_attr("hop", int(hop))

        if result.get("status") != "ok":
            err = result.get("error") or "unknown error"
            trial.set_user_attr("error", err)
            print(f"|  [{timestamp()}] Trial {trial.number + 1:>2}  |  FAILED  |  {err}")
            if result.get("traceback"):
                print(result["traceback"], end="" if result["traceback"].endswith("\n") else "\n")
            raise optuna.TrialPruned(err)

        value = float(result["final"][metric])
        if not math.isfinite(value):
            raise optuna.TrialPruned("Non-finite validation score")
        print(f"|  [{timestamp()}] Trial {trial.number + 1:>2}  |  L {L}  |  Hop {hop}  |  Validation {value:.2f}%")
        return value

    return objective


# =========================================================
# Main runner
# =========================================================
def run_all_independent(
    *,
    results_dir: str = "results",
    db_path: str = "results/optuna/window_hop_independent_v3.db",
    metric: str = "sample_acc",
    phase1_trials: int = 20,
    phase1_epochs: int = 5,
    phase2_trials: int = 10,
    phase2_epochs: int = 10,
    max_samples_override: Optional[int] = None,
    window_fixed_hop_ratio: float = 1.0,
    datasets: Optional[list[str]] = None,
):
    """
    Runs two experiments per dataset x learner:

    A) Window-length experiment:
       - Phase 1: broad search on L, fixed hop_ratio
       - Phase 2: refine L around best phase-1 L using 0.75x..1.25x

    B) Hop-ratio experiment:
       - Phase 1: broad search on hop_ratio, fixed best L from the completed
         window-length experiment
       - Phase 2: refine hop_ratio around best phase-1 value using 0.75x..1.25x

    This yields phase1_trials + phase2_trials for each experiment.
    With the defaults that is 30 + 30 = 60 search runs per dataset x learner,
    plus two automatically retrained final test runs at the normal epoch budget.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ensure_dir(os.path.dirname(db_path))
    storage = f"sqlite:///{db_path}"

    datasets = list(datasets) if datasets is not None else [
        "pamap2",
        "har",
        "dvs_gesture",
        "large_scale_audio",
        "speech_commands",
        "esc50",
        "urban8k",
        "mitbih",
    ]
    learners = ["bp", "ff", "eprop", "pepita"]

    for ds in datasets:
        base_run_cfg = get_base_run_config_by_dataset(ds)
        base_run_cfg["OPTUNA_METRIC"] = metric
        base_pipeline = deepcopy(base_run_cfg["TRANSFORM"])
        bounds_cfg = deepcopy(base_run_cfg)
        if max_samples_override is not None:
            bounds_cfg["MAX_SAMPLES"] = int(max_samples_override)
        length_info = inspect_training_lengths(
            bounds_cfg,
            remove_window_ops(base_pipeline),
            max_samples=bounds_cfg.get("WINDOW_BOUND_SAMPLES", 256),
        )
        max_time_steps = int(length_info["maximum"])
        print(
            f"[Window Bounds] training-only preload "
            f"{length_info['inspected_samples']}/{length_info['training_samples']} samples | "
            f"observed T {length_info['minimum']}..{max_time_steps}"
        )

        for learner in learners:
            panel(f"SEARCH | {ds.upper()} | {learner.upper()}")

            # -----------------------------------------
            # Experiment A: independent window-length
            # -----------------------------------------
            w1_name = f"ws_{ds}_{learner}_window_p1"
            L1_min, L1_max, step1 = bounds_phase1_window(max_time_steps)

            cfg_w1 = deepcopy(base_run_cfg)
            cfg_w1["RUN_ID"] = f"{ds}-{learner}"
            cfg_w1["LEARNER"] = learner
            cfg_w1["EPOCHS"] = int(phase1_epochs)
            cfg_w1["TEST_EVERY_EPOCH"] = False
            if max_samples_override is not None:
                cfg_w1["MAX_SAMPLES"] = int(max_samples_override)

            study_w1 = create_study(
                config=cfg_w1,
                study_name=w1_name,
                direction="maximize",
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=int(cfg_w1.get("SEED", 123))),
                pruner=optuna.pruners.NopPruner(),
            )

            done_w1 = completed_trials(study_w1)
            rem_w1 = max(0, phase1_trials - done_w1)

            if rem_w1 > 0:
                print(
                    f"[Window P1] {w1_name} | L in [{L1_min},{L1_max}] step={step1} | "
                    f"fixed_hop_ratio={window_fixed_hop_ratio:.4f} | epochs={phase1_epochs}"
                )
                print(f"[Window P1] completed={done_w1} remaining={rem_w1} target={phase1_trials}")

                obj_w1 = make_objective(
                    cfg_w1,
                    base_pipeline,
                    dataset=ds,
                    learner=learner,
                    experiment="window",
                    phase=1,
                    metric=metric,
                    results_dir=results_dir,
                    search_mode="window_only",
                    L_min=L1_min,
                    L_max=L1_max,
                    step=step1,
                    fixed_hop_ratio=window_fixed_hop_ratio,
                )
                study_w1.optimize(obj_w1, n_trials=rem_w1, catch=(Exception,))
                export_study_csv(study_w1, os.path.join(results_dir, "optuna", f"{w1_name}_trials.csv"))
                export_best_json(study_w1, os.path.join(results_dir, "optuna", f"{w1_name}_best.json"))
            else:
                print(f"[Window P1] Skip (already has >= {phase1_trials} COMPLETE trials).")

            if completed_trials(study_w1) > 0:
                best_L = int(study_w1.best_params["win_L"])
                print(f"[Window P1 Best] value={study_w1.best_value:.4f} best_L={best_L}")
            else:
                print("[Window P2] Skipped because Window P1 has no COMPLETE trials.")
                best_L = None

            study_w2 = None

            if best_L is not None:
                w2_name = f"ws_{ds}_{learner}_window_p2"
                L2_min, L2_max, step2 = bounds_phase2_window(
                    best_L, max_time_steps=max_time_steps
                )

                cfg_w2 = deepcopy(base_run_cfg)
                cfg_w2["RUN_ID"] = f"{ds}-{learner}"
                cfg_w2["LEARNER"] = learner
                cfg_w2["EPOCHS"] = int(phase2_epochs)
                cfg_w2["TEST_EVERY_EPOCH"] = False
                if max_samples_override is not None:
                    cfg_w2["MAX_SAMPLES"] = int(max_samples_override)

                study_w2 = create_study(
                    config=cfg_w2,
                    study_name=w2_name,
                    direction="maximize",
                    storage=storage,
                    load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(seed=int(cfg_w2.get("SEED", 123))),
                    pruner=optuna.pruners.NopPruner(),
                )

                if len(study_w2.trials) == 0:
                    study_w2.enqueue_trial({"win_L": int(best_L)})

                done_w2 = completed_trials(study_w2)
                rem_w2 = max(0, phase2_trials - done_w2)

                if rem_w2 > 0:
                    print(
                        f"[Window P2] {w2_name} | L in [{L2_min},{L2_max}] step={step2} | "
                        f"fixed_hop_ratio={window_fixed_hop_ratio:.4f} | epochs={phase2_epochs}"
                    )
                    print(f"[Window P2] completed={done_w2} remaining={rem_w2} target={phase2_trials}")

                    obj_w2 = make_objective(
                        cfg_w2,
                        base_pipeline,
                        dataset=ds,
                        learner=learner,
                        experiment="window",
                        phase=2,
                        metric=metric,
                        results_dir=results_dir,
                        search_mode="window_only",
                        L_min=L2_min,
                        L_max=L2_max,
                        step=step2,
                        fixed_hop_ratio=window_fixed_hop_ratio,
                    )
                    study_w2.optimize(obj_w2, n_trials=rem_w2, catch=(Exception,))
                    export_study_csv(study_w2, os.path.join(results_dir, "optuna", f"{w2_name}_trials.csv"))
                    export_best_json(study_w2, os.path.join(results_dir, "optuna", f"{w2_name}_best.json"))
                else:
                    print(f"[Window P2] Skip (already has >= {phase2_trials} COMPLETE trials).")

            window_anchor = best_window_anchor_from_studies(study_w1, study_w2)
            if window_anchor is None:
                print("[Hop Search] Skipped because no COMPLETE window-length trials exist.")
                continue

            hop_anchor_L = int(window_anchor["L"])
            print(
                f"[Window Best Overall] value={window_anchor['value']:.4f} "
                f"hop_anchor_L={hop_anchor_L} "
                f"source_phase={window_anchor['phase']} "
                f"source_study={window_anchor['study_name']}"
            )

            # -----------------------------------------
            # Experiment B: hop-ratio using best window
            # -----------------------------------------
            # Include bestL in the study names so old hop runs that used baseline_L
            # are not mixed with the new intended hop experiments.
            h1_name = f"ws_{ds}_{learner}_hop_bestL{hop_anchor_L}_p1"
            hr1_min, hr1_max = bounds_phase1_hop_ratio()

            cfg_h1 = deepcopy(base_run_cfg)
            cfg_h1["RUN_ID"] = f"{ds}-{learner}"
            cfg_h1["LEARNER"] = learner
            cfg_h1["EPOCHS"] = int(phase1_epochs)
            cfg_h1["TEST_EVERY_EPOCH"] = False
            if max_samples_override is not None:
                cfg_h1["MAX_SAMPLES"] = int(max_samples_override)

            study_h1 = create_study(
                config=cfg_h1,
                study_name=h1_name,
                direction="maximize",
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=int(cfg_h1.get("SEED", 123))),
                pruner=optuna.pruners.NopPruner(),
            )

            if len(study_h1.trials) == 0:
                study_h1.enqueue_trial({"hop_ratio": 1.0})
                study_h1.enqueue_trial({"hop_ratio": 0.75})
                study_h1.enqueue_trial({"hop_ratio": 1.25})

            done_h1 = completed_trials(study_h1)
            rem_h1 = max(0, phase1_trials - done_h1)

            if rem_h1 > 0:
                print(
                    f"[Hop P1] {h1_name} | hop_ratio in [{hr1_min:.4f},{hr1_max:.4f}] | "
                    f"fixed_L={hop_anchor_L} | epochs={phase1_epochs}"
                )
                print(f"[Hop P1] completed={done_h1} remaining={rem_h1} target={phase1_trials}")

                obj_h1 = make_objective(
                    cfg_h1,
                    base_pipeline,
                    dataset=ds,
                    learner=learner,
                    experiment="hop",
                    phase=1,
                    metric=metric,
                    results_dir=results_dir,
                    search_mode="hop_only",
                    fixed_L=hop_anchor_L,
                    hr_min=hr1_min,
                    hr_max=hr1_max,
                )
                study_h1.optimize(obj_h1, n_trials=rem_h1, catch=(Exception,))
                export_study_csv(study_h1, os.path.join(results_dir, "optuna", f"{h1_name}_trials.csv"))
                export_best_json(study_h1, os.path.join(results_dir, "optuna", f"{h1_name}_best.json"))
            else:
                print(f"[Hop P1] Skip (already has >= {phase1_trials} COMPLETE trials).")

            if completed_trials(study_h1) > 0:
                best_hr = float(study_h1.best_params["hop_ratio"])
                print(f"[Hop P1 Best] value={study_h1.best_value:.4f} best_hop_ratio={best_hr:.4f}")
            else:
                print("[Hop P2] Skipped because Hop P1 has no COMPLETE trials.")
                best_hr = None

            study_h2 = None
            if best_hr is not None:
                h2_name = f"ws_{ds}_{learner}_hop_bestL{hop_anchor_L}_p2"
                hr2_min, hr2_max = bounds_phase2_hop_ratio(best_hr)

                cfg_h2 = deepcopy(base_run_cfg)
                cfg_h2["RUN_ID"] = f"{ds}-{learner}"
                cfg_h2["LEARNER"] = learner
                cfg_h2["EPOCHS"] = int(phase2_epochs)
                cfg_h2["TEST_EVERY_EPOCH"] = False
                if max_samples_override is not None:
                    cfg_h2["MAX_SAMPLES"] = int(max_samples_override)

                study_h2 = create_study(
                    config=cfg_h2,
                    study_name=h2_name,
                    direction="maximize",
                    storage=storage,
                    load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(seed=int(cfg_h2.get("SEED", 123))),
                    pruner=optuna.pruners.NopPruner(),
                )

                if len(study_h2.trials) == 0:
                    study_h2.enqueue_trial({"hop_ratio": float(best_hr)})
                    study_h2.enqueue_trial({"hop_ratio": float(max(0.05, best_hr * 0.75))})
                    study_h2.enqueue_trial({"hop_ratio": float(min(2.0, best_hr * 1.25))})

                done_h2 = completed_trials(study_h2)
                rem_h2 = max(0, phase2_trials - done_h2)

                if rem_h2 > 0:
                    print(
                        f"[Hop P2] {h2_name} | hop_ratio in [{hr2_min:.4f},{hr2_max:.4f}] | "
                        f"fixed_L={hop_anchor_L} | epochs={phase2_epochs}"
                    )
                    print(f"[Hop P2] completed={done_h2} remaining={rem_h2} target={phase2_trials}")

                    obj_h2 = make_objective(
                        cfg_h2,
                        base_pipeline,
                        dataset=ds,
                        learner=learner,
                        experiment="hop",
                        phase=2,
                        metric=metric,
                        results_dir=results_dir,
                        search_mode="hop_only",
                        fixed_L=hop_anchor_L,
                        hr_min=hr2_min,
                        hr_max=hr2_max,
                    )
                    study_h2.optimize(obj_h2, n_trials=rem_h2, catch=(Exception,))
                    export_study_csv(study_h2, os.path.join(results_dir, "optuna", f"{h2_name}_trials.csv"))
                    export_best_json(study_h2, os.path.join(results_dir, "optuna", f"{h2_name}_best.json"))
                else:
                    print(f"[Hop P2] Skip (already has >= {phase2_trials} COMPLETE trials).")

            final_cfg = deepcopy(base_run_cfg)
            if max_samples_override is not None:
                final_cfg['MAX_SAMPLES'] = int(max_samples_override)
            window_phase, window_study = preferred_study(study_w1, study_w2)
            hop_phase, hop_study = preferred_study(study_h1, study_h2)
            # Selection is complete before either held-out test evaluation.
            for experiment, phase, study, ratio in (
                ('window', window_phase, window_study, window_fixed_hop_ratio),
                ('hop', hop_phase, hop_study,
                 float(hop_study.best_params['hop_ratio']) if hop_study is not None else 1.0),
            ):
                if study is None:
                    continue
                pipeline = append_sliding_window(remove_window_ops(base_pipeline),
                    L=hop_anchor_L, hop=max(1, round(hop_anchor_L * ratio)))
                run_final_test(final_cfg, pipeline, study=study, phase=phase,
                    family='independent', experiment=experiment, dataset=ds, learner=learner,
                    length=hop_anchor_L, hop_ratio=ratio, results_dir=results_dir)
            export_final_tests(results_dir, os.path.join(results_dir, 'optuna', 'final_tests', 'independent'), 'independent')

    close("ALL INDEPENDENT SEARCHES COMPLETE")
    print(f"Optuna DB: {db_path}")
    print(f"Optuna exports: {os.path.join(results_dir, 'optuna')}")


if __name__ == "__main__":
    run_all_independent(
        results_dir="results",
        db_path="results/optuna/window_hop_independent_mnist_10k_v1.db",
        metric="sample_acc",
        phase1_trials=20,
        phase1_epochs=5,
        phase2_trials=10,
        phase2_epochs=10,
        max_samples_override=None,
        window_fixed_hop_ratio=1.0,
        datasets=["mnist_static", "mnist_rate"],
    )
