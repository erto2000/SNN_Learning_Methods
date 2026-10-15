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
def bounds_phase1_window(max_time_steps: int) -> Tuple[int, int, int]:
    """
    Broad search for the independent window-length experiment.
    """
    T = int(max_time_steps)
    if T < 1:
        raise ValueError("max_time_steps must be positive")
    L_min = max(1, int(round(T * 0.10)))
    L_max = T
    # Include every integer length, including the full-sequence endpoint.
    return L_min, L_max, 1


def bounds_phase2_window(best_L: int, *, max_time_steps: int) -> Tuple[int, int, int]:
    """
    Refinement search for the independent window-length experiment.
    Uses the user's intended 0.75x .. 1.25x neighborhood around the best
    phase-1 window.
    """
    T = int(max_time_steps)
    if T < 1:
        raise ValueError("max_time_steps must be positive")
    if not 1 <= best_L <= T:
        raise ValueError("best_L must be within the positive sequence horizon")
    L_min = max(1, int(round(best_L * 0.75)))
    L_max = min(T, max(L_min, int(round(best_L * 1.25))))
    return L_min, L_max, 1


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
    Search integer hops directly, avoiding rounded float aliases.
    """
    L = int(fixed_L)
    hop_min, hop_max = hop_bounds(L, hr_min, hr_max)
    hop = trial.suggest_int("hop", hop_min, hop_max)
    hop_ratio = hop / L

    trial.set_user_attr("search_mode", "hop_only")
    trial.set_user_attr("fixed_L", L)
    trial.set_user_attr("hop", hop)
    trial.set_user_attr("hop_ratio", hop_ratio)

    return L, hop, hop_ratio


def hop_bounds(length: int, ratio_min: float, ratio_max: float) -> Tuple[int, int]:
    """Include only integer hops whose effective ratio lies inside the bounds."""
    lo = max(1, math.ceil(length * ratio_min - 1e-12))
    hi = max(lo, math.floor(length * ratio_max + 1e-12))
    return lo, hi


def schedule_key(length: int, hop: int, source_lengths=None) -> tuple:
    """Collapse equivalent inputs only when source lengths are guaranteed.

    A subset maximum cannot prove equivalence for a variable-length dataset.
    Without a complete length profile or fixed-length contract, use the exact
    integer configuration instead.
    """
    if source_lengths is None:
        return int(length), int(hop)
    return tuple((int(total), tuple(transforms.window_spans(total, length, hop)))
                 for total in sorted(set(source_lengths)))


def optimize_unique(study, objective, *, choices, parameter, distribution,
                    target, source_lengths=None, peers=(), context=None, anchors=()):
    """Train up to target new, distinct schedules; duplicates spend no budget.

    Reuse observations from the same dataset, learner and epoch budget. TPE
    still proposes candidates normally. After repeated duplicate proposals,
    enqueue one remaining integer choice so finite spaces terminate reliably.
    """
    context = dict(context or {})
    candidates = {}
    for value, (length, hop) in choices.items():
        candidates.setdefault(schedule_key(length, hop, source_lengths),
                              (value, int(length), int(hop)))

    def trial_key(trial):
        attrs = trial.user_attrs
        length = attrs.get("L", trial.params.get("win_L", attrs.get("fixed_L")))
        hop = attrs.get("hop", trial.params.get("hop"))
        return schedule_key(length, hop, source_lengths) if length and hop else None

    known = {}
    for peer in (*peers, study):
        for trial in peer.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                known.setdefault(trial_key(trial), (peer.study_name, trial))
    present = {trial_key(trial) for trial in study.trials
               if trial.state == optuna.trial.TrialState.COMPLETE}
    for key, (value, length, hop) in candidates.items():
        if key in known and key not in present:
            source_name, source = known[key]
            attrs = {**source.user_attrs, **context, "L": length, "hop": hop,
                     "hop_ratio": hop / length, "training_reused": True,
                     "source_study": source_name, "source_trial": source.number}
            study.add_trial(optuna.trial.create_trial(value=source.value,
                params={parameter: value}, distributions={parameter: distribution},
                user_attrs=attrs))
            present.add(key)

    attempted = {trial_key(trial) for trial in study.trials
                 if trial.user_attrs.get("training_attempted")}
    remaining = set(candidates) - present - attempted
    trained = lambda: sum(t.state == optuna.trial.TrialState.COMPLETE
                          and not t.user_attrs.get("training_reused", False)
                          for t in study.trials)
    study.set_user_attr("training_budget", int(target))
    study.set_user_attr("unique_candidates", len(candidates))
    if not study.trials:
        for value in dict.fromkeys(anchors):
            if value in choices and schedule_key(*choices[value], source_lengths) in remaining:
                study.enqueue_trial({parameter: value})
    duplicate_streak = 0
    while trained() < target and remaining:
        if duplicate_streak >= 3:
            key = min(remaining, key=lambda item: candidates[item][0])
            study.enqueue_trial({parameter: candidates[key][0]})
            duplicate_streak = 0
        trial = study.ask()
        value = trial.suggest_int(parameter, distribution.low, distribution.high,
                                  step=distribution.step)
        length, hop = choices[value]
        key = schedule_key(length, hop, source_lengths)
        trial.set_user_attr("L", int(length))
        trial.set_user_attr("hop", int(hop))
        if key not in remaining:
            trial.set_user_attr("status", "duplicate_skipped")
            # TPE ignores FAIL records. An unscored PRUNED record would count
            # toward startup and incorrectly influence its observations.
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            duplicate_streak += 1
            continue
        duplicate_streak = 0
        remaining.remove(key)
        trial.set_user_attr("training_attempted", True)
        try:
            score = objective(trial)
        except optuna.TrialPruned as exc:
            trial.set_user_attr("error", str(exc))
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"[Search failure] {type(exc).__name__}: {exc}")
        else:
            study.tell(trial, score)
    study.set_user_attr("trained_complete", trained())
    study.set_user_attr("space_exhausted", not bool(remaining))
    if trained() < target and not remaining:
        print(f"[Search] Unique choices exhausted: {trained()}/{target} new trained networks.")


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
        "training_reused",
        "source_study",
        "source_trial",
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
                "hop_ratio": t.user_attrs.get("hop_ratio", t.user_attrs.get("fixed_hop_ratio")),
                "hop": (t.user_attrs or {}).get("hop"),
                "experiment": (t.user_attrs or {}).get("experiment"),
                "phase": (t.user_attrs or {}).get("phase"),
                "search_mode": (t.user_attrs or {}).get("search_mode"),
                "run_id": (t.user_attrs or {}).get("run_id"),
                "status": (t.user_attrs or {}).get("status"),
                "error": (t.user_attrs or {}).get("error"),
                "training_reused": t.user_attrs.get("training_reused", False),
                "source_study": t.user_attrs.get("source_study"),
                "source_trial": t.user_attrs.get("source_trial"),
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
        payload["n_trained_complete"] = sum(
            t.state == optuna.trial.TrialState.COMPLETE
            and not t.user_attrs.get("training_reused", False) for t in study.trials)
        payload["n_reused_complete"] = sum(
            t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get("training_reused", False) for t in study.trials)
        payload["space_exhausted"] = study.user_attrs.get("space_exhausted")
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
        base_id = f"{dataset}-{learner}-v5"
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
def select_hop_with_fallback(window_phase, window_study, hop_phase, hop_study):
    """Compare validation at the same epoch budget, retaining non-overlap."""
    candidates = [(window_phase, window_study), (hop_phase, hop_study)]
    candidates = [(phase, study) for phase, study in candidates if study is not None]
    if not candidates:
        return None, None
    # Phase 2 normally uses 10 epochs; Phase 1 uses only 5. Compare validation
    # scores only within the highest available training budget.
    highest_epochs = max(study.user_attrs.get("epochs", phase) for phase, study in candidates)
    comparable = [(phase, study) for phase, study in candidates
                  if study.user_attrs.get("epochs", phase) == highest_epochs]
    return max(comparable, key=lambda item: item[1].best_value)


def run_all_independent(
    *,
    results_dir: str = "results",
    db_path: str = "results/optuna/window_hop_independent_recording_batch_v3.db",
    metric: str = "sample_acc",
    phase1_trials: int = 20,
    phase1_epochs: int = 5,
    phase2_trials: int = 10,
    phase2_epochs: int = 10,
    max_samples_override: Optional[int] = None,
    window_fixed_hop_ratio: float = 1.0,
    datasets: Optional[list[str]] = None,
):
    """Search lengths, then integer hops, using distinct trained networks.

    Each phase trains up to its own budget. Already evaluated schedules at the
    same epoch budget are reused; exhausted finite spaces stop without moving
    the unused budget elsewhere. Full length remains a normal length candidate.
    Final settings are selected by validation before held-out test evaluation.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ensure_dir(os.path.dirname(db_path))
    storage = f"sqlite:///{db_path}"
    datasets = list(datasets) if datasets is not None else [
        "pamap2", "har", "dvs_gesture", "large_scale_audio", "speech_commands",
        "esc50", "urban8k", "mitbih", "mnist_static", "mnist_rate",
    ]
    learners = ["bp", "ff", "eprop", "pepita"]

    for ds in datasets:
        base_run_cfg = get_base_run_config_by_dataset(ds)
        base_run_cfg["OPTUNA_METRIC"] = metric
        base_pipeline = deepcopy(base_run_cfg["TRANSFORM"])
        bounds_cfg = deepcopy(base_run_cfg)
        if max_samples_override is not None:
            bounds_cfg["MAX_SAMPLES"] = int(max_samples_override)
        length_info = inspect_training_lengths(bounds_cfg, remove_window_ops(base_pipeline),
            max_samples=bounds_cfg.get("WINDOW_BOUND_SAMPLES", 256))
        max_time_steps = int(length_info["maximum"])
        if length_info.get("fixed_length") is not None:
            source_lengths = [int(length_info["fixed_length"])]
        elif length_info.get("all_lengths_inspected"):
            source_lengths = length_info["lengths"]
        else:
            source_lengths = None
        print(f"[Window Bounds] training-only preload "
              f"{length_info['inspected_samples']}/{length_info['training_samples']} samples | "
              f"observed T {length_info['minimum']}..{max_time_steps}")

        for learner in learners:
            panel(f"SEARCH | {ds.upper()} | {learner.upper()}")
            studies_by_epochs = {}

            def run_phase(experiment, phase, *, length_bounds=None, fixed_length=None,
                          ratio_bounds=None, anchors=()):
                epochs = int(phase1_epochs if phase == 1 else phase2_epochs)
                target = int(phase1_trials if phase == 1 else phase2_trials)
                cfg = deepcopy(base_run_cfg)
                cfg.update(RUN_ID=f"{ds}-{learner}", LEARNER=learner,
                           EPOCHS=epochs, TEST_EVERY_EPOCH=False)
                if max_samples_override is not None:
                    cfg["MAX_SAMPLES"] = int(max_samples_override)
                name = f"ws5_{ds}_{learner}_{experiment}"
                if experiment == "hop":
                    name += f"_bestL{fixed_length}"
                name += f"_p{phase}"
                sampler_kw = dict(seed=int(cfg.get("SEED", 123)))
                if phase == 2:
                    sampler_kw["n_startup_trials"] = 3
                study = create_study(config=cfg, study_name=name, direction="maximize",
                    storage=storage, load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(**sampler_kw),
                    pruner=optuna.pruners.NopPruner())
                study.set_user_attr("epochs", epochs)
                objective_kw = {}
                if experiment == "window":
                    lo, hi, step = length_bounds
                    parameter = "win_L"
                    choices = {length: (length, max(1, round(length * window_fixed_hop_ratio)))
                               for length in range(lo, hi + 1, step)}
                    objective_kw.update(search_mode="window_only", L_min=lo, L_max=hi,
                                        step=step, fixed_hop_ratio=window_fixed_hop_ratio)
                    context = dict(search_mode="window_only", fixed_hop_ratio=window_fixed_hop_ratio)
                else:
                    lo, hi = hop_bounds(fixed_length, *ratio_bounds)
                    step = 1
                    parameter = "hop"
                    choices = {hop: (fixed_length, hop) for hop in range(lo, hi + 1)}
                    objective_kw.update(search_mode="hop_only", fixed_L=fixed_length,
                                        hr_min=ratio_bounds[0], hr_max=ratio_bounds[1])
                    context = dict(search_mode="hop_only", fixed_L=fixed_length)
                print(f"[{experiment.title()} P{phase}] {name} | {parameter} in [{lo},{hi}] | "
                      f"epochs={epochs} | new-network budget={target}")
                objective = make_objective(cfg, base_pipeline, dataset=ds, learner=learner,
                    experiment=experiment, phase=phase, metric=metric,
                    results_dir=results_dir, **objective_kw)
                optimize_unique(study, objective, choices=choices, parameter=parameter,
                    distribution=optuna.distributions.IntDistribution(lo, hi, step=step),
                    target=target, source_lengths=source_lengths,
                    peers=studies_by_epochs.get(epochs, []),
                    context=dict(experiment=experiment, phase=phase, **context), anchors=anchors)
                studies_by_epochs.setdefault(epochs, []).append(study)
                export_study_csv(study, os.path.join(results_dir, "optuna", f"{name}_trials.csv"))
                export_best_json(study, os.path.join(results_dir, "optuna", f"{name}_best.json"))
                return study

            # No forced full-length trial: TPE samples the whole integer domain.
            study_w1 = run_phase("window", 1,
                length_bounds=bounds_phase1_window(max_time_steps))
            if not completed_trials(study_w1):
                print("[Search] No successful window trials; skipping this learner.")
                continue
            best_L = int(study_w1.best_params["win_L"])
            study_w2 = run_phase("window", 2,
                length_bounds=bounds_phase2_window(best_L, max_time_steps=max_time_steps),
                anchors=[best_L])
            window_phase, window_study = preferred_study(study_w1, study_w2)
            hop_anchor_L = int(window_study.best_params["win_L"])

            # Non-overlap is included when its integer hop lies inside the domain.
            study_h1 = run_phase("hop", 1, fixed_length=hop_anchor_L,
                ratio_bounds=bounds_phase1_hop_ratio(), anchors=[hop_anchor_L])
            study_h2 = None
            if completed_trials(study_h1):
                best_hop = int(study_h1.best_params["hop"])
                study_h2 = run_phase("hop", 2, fixed_length=hop_anchor_L,
                    ratio_bounds=bounds_phase2_hop_ratio(best_hop / hop_anchor_L),
                    anchors=[best_hop])
            hop_phase, hop_study = preferred_study(study_h1, study_h2)
            hop_phase, hop_study = select_hop_with_fallback(
                window_phase, window_study, hop_phase, hop_study)

            # Both choices are fixed before either test result is observed.
            selected = [("window", window_phase, window_study), ("hop", hop_phase, hop_study)]
            final_cfg = deepcopy(base_run_cfg)
            final_cfg["OPTUNA_METRIC"] = metric
            if max_samples_override is not None:
                final_cfg["MAX_SAMPLES"] = int(max_samples_override)
            previous_final = None
            previous_schedule = None
            for experiment, phase, study in selected:
                if study is None:
                    continue
                attrs = study.best_trial.user_attrs
                length = int(attrs.get("L", study.best_params.get("win_L", hop_anchor_L)))
                hop = int(attrs["hop"])
                ratio = hop / length
                current_schedule = schedule_key(length, hop, source_lengths)
                pipeline = append_sliding_window(remove_window_ops(base_pipeline), L=length, hop=hop)
                reuse = previous_final if current_schedule == previous_schedule else None
                result = run_final_test(final_cfg, pipeline, study=study, phase=phase,
                    family="independent", experiment=experiment, dataset=ds, learner=learner,
                    length=length, hop_ratio=ratio, results_dir=results_dir, reuse_result=reuse)
                previous_final, previous_schedule = result, current_schedule
            export_final_tests(results_dir,
                os.path.join(results_dir, "optuna", "final_tests", "independent"), "independent")

    close("ALL INDEPENDENT SEARCHES COMPLETE")
    print(f"Optuna DB: {db_path}")
    print(f"Optuna exports: {os.path.join(results_dir, 'optuna')}")


if __name__ == "__main__":
    run_all_independent()
