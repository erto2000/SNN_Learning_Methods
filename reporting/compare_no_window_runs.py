"""Compare unwindowed runs across datasets and learners."""

from __future__ import annotations

from typing import Any, Dict, Optional

from visualization.flex_compare import run_comparisons


BASE_DIR = "results"
LEARNERS = "bp|ff|eprop|pepita"
EMIT_PER_EPOCH_CSV = False


RUN_GROUPS = [
    {"key": "har", "title": "HAR", "prefix": "har"},
    {"key": "mnist_static", "title": "MNIST static", "prefix": "mnist-static"},
    {"key": "mnist_rate", "title": "MNIST rate", "prefix": "mnist-rate"},
    {"key": "speech_commands", "title": "Speech Commands", "prefix": "sc"},
    {"key": "esc50", "title": "ESC-50", "prefix": "esc50"},
    {"key": "urban8k", "title": "UrbanSound8K", "prefix": "urban8k"},
    {"key": "pamap2", "title": "PAMAP2", "prefix": "pamap2"},
    {"key": "mitbih", "title": "MIT-BIH", "prefix": "mitbih"},
    {"key": "dvs_gesture", "title": "DVS Gesture", "prefix": "dvs"},
    {"key": "large_scale_audio", "title": "Large-scale audio", "prefix": "large-scale-audio"},
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers


def _summary(info: Dict[str, Any]) -> Dict[str, Any]:
    return info.get("summary") or {}


def _config(info: Dict[str, Any]) -> Dict[str, Any]:
    return _summary(info).get("config") or {}


def _history(info: Dict[str, Any]) -> Dict[str, Any]:
    return _summary(info).get("history") or {}


def _memory(info: Dict[str, Any]) -> Dict[str, Any]:
    return _summary(info).get("memory") or {}


def _theory(info: Dict[str, Any]) -> Dict[str, Any]:
    theory = _memory(info).get("theory") or {}
    return theory if isinstance(theory, dict) else {}


def _deep_get(obj: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cfg_value(info: Dict[str, Any], name: str) -> Any:
    return _config(info).get(name)


def _theory_value(info: Dict[str, Any], path: str) -> Optional[float]:
    return _as_float(_deep_get(_theory(info), path))


# ──────────────────────────────────────────────────────────────────────────────
# Custom metrics for flex_compare


def max_train_acc(info: Dict[str, Any]) -> Optional[float]:
    vals = []
    for _, row in _history(info).items():
        v = _as_float((row or {}).get("acc"))
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def best_train_epoch(info: Dict[str, Any]) -> Optional[int]:
    best_acc, best_ep = None, None
    for epoch, row in _history(info).items():
        v = _as_float((row or {}).get("acc"))
        if v is None:
            continue
        if best_acc is None or v > best_acc:
            best_acc, best_ep = v, int(epoch)
    return best_ep


def max_sample_acc(info: Dict[str, Any]) -> Optional[float]:
    vals = []
    for _, row in _history(info).items():
        v = _as_float((row or {}).get("sample_acc"))
        if v is not None:
            vals.append(v)
    final_acc = _as_float(_deep_get(_summary(info), "final.sample_acc"))
    if final_acc is not None:
        vals.append(final_acc)
    return max(vals) if vals else None


def sec_per_epoch(info: Dict[str, Any]) -> Optional[float]:
    duration = _as_float(_summary(info).get("duration_seconds"))
    epochs = _as_float(_cfg_value(info, "EPOCHS"))
    if duration is None or not epochs:
        return None
    return duration / epochs


def hidden_total(info: Dict[str, Any]) -> Optional[int]:
    hidden = _cfg_value(info, "HIDDEN_SIZES") or []
    try:
        return sum(int(x) for x in hidden)
    except (TypeError, ValueError):
        return None


def hidden_depth(info: Dict[str, Any]) -> int:
    return len(_cfg_value(info, "HIDDEN_SIZES") or [])


def hidden_str(info: Dict[str, Any]) -> str:
    return "x".join(str(x) for x in (_cfg_value(info, "HIDDEN_SIZES") or []))


def learner_label(info: Dict[str, Any]) -> str:
    mapping = {
        "bp": "BPTT",
        "ff": "FF",
        "eprop": "E-PROP",
        "pepita": "RATE-PEPITA",
    }
    return mapping.get(str(_cfg_value(info, "LEARNER")), str(_cfg_value(info, "LEARNER")))


def dataset_label(info: Dict[str, Any]) -> str:
    """Keep MNIST's temporal encodings distinct in combined tables."""
    run_id = str(_summary(info).get("run_id") or info.get("id") or "")
    if run_id.startswith("mnist-static-"):
        return "mnist_static"
    if run_id.startswith("mnist-rate-"):
        return "mnist_rate"
    return str(_cfg_value(info, "DATASET"))


def optimizer_label(info: Dict[str, Any]) -> str:
    learner = str(_cfg_value(info, "LEARNER"))
    if learner == "bp":
        opt = _cfg_value(info, "BP_OPTIMIZER")
        lr = _cfg_value(info, "BP_LR")
        return f"{str(opt).upper()} lr={float(lr):g}" if opt is not None and lr is not None else "BPTT"
    if learner == "ff":
        opt = _cfg_value(info, "FF_OPTIMIZER")
        lr = _cfg_value(info, "FF_LR")
        return f"{str(opt).upper()} lr={float(lr):g}" if opt is not None and lr is not None else "FF"
    if learner == "eprop":
        opt = _cfg_value(info, "EP_OPTIMIZER")
        lr_in = _cfg_value(info, "EP_LR_IN")
        lr_out = _cfg_value(info, "EP_LR_OUT")
        if opt is not None and lr_in is not None and lr_out is not None:
            return f"{str(opt).upper()} lr_in={float(lr_in):g}, lr_out={float(lr_out):g}"
        return "E-PROP"
    if learner == "pepita":
        opt = _cfg_value(info, "PEP_OPTIMIZER")
        lr = _cfg_value(info, "PEP_LR")
        mode = _cfg_value(info, "PEP_MODE")
        if opt is not None and lr is not None:
            suffix = f" {mode}" if mode is not None else ""
            return f"{str(opt).upper()} lr={float(lr):g}{suffix}"
        return "RATE-PEPITA"
    return learner


def theory_memory_scalars(info: Dict[str, Any]) -> Optional[float]:
    return _theory_value(info, "memory.total_scalars")


def theory_memory_bytes(info: Dict[str, Any]) -> Optional[float]:
    return _theory_value(info, "memory.total_bytes")


def theory_memory_mb(info: Dict[str, Any]) -> Optional[float]:
    value = theory_memory_bytes(info)
    return None if value is None else value / (1024 ** 2)


def theory_compute_scalars(info: Dict[str, Any]) -> Optional[float]:
    return _theory_value(info, "compute.total_scalars")


def theory_access_scalars(info: Dict[str, Any]) -> Optional[float]:
    return _theory_value(info, "access.total_scalars")


def theory_time_proxy(info: Dict[str, Any]) -> Optional[float]:
    return _theory_value(info, "time_proxy.value")


def training_memory_mb(info: Dict[str, Any]) -> Optional[float]:
    value = _as_float(_memory(info).get("training_bytes_per_batch"))
    return None if value is None else value / (1024 ** 2)


def static_memory_mb(info: Dict[str, Any]) -> Optional[float]:
    value = _as_float(_memory(info).get("static_bytes"))
    return None if value is None else value / (1024 ** 2)


def time_eval_t1(info: Dict[str, Any]) -> Optional[float]:
    return _as_float(_deep_get(_summary(info), "final.time_eval_sample_acc_T1"))


def time_eval_p25(info: Dict[str, Any]) -> Optional[float]:
    return _as_float(_deep_get(_summary(info), "final.time_eval_sample_acc_P25"))


def time_eval_p50(info: Dict[str, Any]) -> Optional[float]:
    return _as_float(_deep_get(_summary(info), "final.time_eval_sample_acc_P50"))


def time_eval_p100(info: Dict[str, Any]) -> Optional[float]:
    return _as_float(_deep_get(_summary(info), "final.time_eval_sample_acc_P100"))


CUSTOM_FUNCS = {
    "dataset_label": dataset_label,
    "max_train_acc": max_train_acc,
    "best_train_epoch": best_train_epoch,
    "max_sample_acc": max_sample_acc,
    "sec_per_epoch": sec_per_epoch,
    "hidden_total": hidden_total,
    "hidden_depth": hidden_depth,
    "hidden_str": hidden_str,
    "learner_label": learner_label,
    "optimizer_label": optimizer_label,
    "theory_memory_scalars": theory_memory_scalars,
    "theory_memory_bytes": theory_memory_bytes,
    "theory_memory_mb": theory_memory_mb,
    "theory_compute_scalars": theory_compute_scalars,
    "theory_access_scalars": theory_access_scalars,
    "theory_time_proxy": theory_time_proxy,
    "training_memory_mb": training_memory_mb,
    "static_memory_mb": static_memory_mb,
    "time_eval_t1": time_eval_t1,
    "time_eval_p25": time_eval_p25,
    "time_eval_p50": time_eval_p50,
    "time_eval_p100": time_eval_p100,
}


# ──────────────────────────────────────────────────────────────────────────────
# Comparison specs


def run_pattern(prefix: str) -> str:
    return f"re:^{prefix}-({LEARNERS})$"


def all_current_runs_pattern() -> str:
    prefixes = "|".join(g["prefix"] for g in RUN_GROUPS)
    return f"re:^({prefixes})-({LEARNERS})$"


def leaderboard_columns():
    return [
        *[{"name": f"{name}_percent_requested", "value": f"meta.data_split.{name}"}
          for name in ("train", "validation", "test")],
        *[{"name": f"{name}_percent_actual", "value": f"meta.split_percentages.{name}"}
          for name in ("train", "validation", "test")],
        {"name": "train_samples", "value": "meta.num_train_samples"},
        {"name": "run_id", "value": "run_id"},
        {"name": "dataset", "func": "dataset_label"},
        {"name": "learner", "value": "config.LEARNER"},
        {"name": "learner_label", "func": "learner_label"},
        {"name": "optimizer", "func": "optimizer_label"},
        {"name": "batch", "value": "config.BATCH_SIZE"},
        {"name": "hidden", "func": "hidden_str"},
        {"name": "hidden_total", "func": "hidden_total"},
        {"name": "hidden_depth", "func": "hidden_depth"},
        {"name": "epochs", "value": "config.EPOCHS"},
        {"name": "evaluation_split", "value": "meta.evaluation_split"},
        {"name": "validation_samples", "value": "meta.num_validation_samples"},
        {"name": "test_samples", "value": "meta.num_test_samples"},
        {"name": "theory_model_version", "value": "memory.theory.model_version"},
        {"name": "theory_work_scope", "value": "memory.theory.work_scope"},
        {"name": "windows_per_sample", "value": "memory.theory.windows_per_sample"},
        {"name": "final_sample_acc", "value": "final.sample_acc"},
        {"name": "max_sample_acc", "func": "max_sample_acc"},
        {"name": "max_train_acc", "func": "max_train_acc"},
        {"name": "best_train_epoch", "func": "best_train_epoch"},
        {"name": "sec_per_epoch", "func": "sec_per_epoch"},
        {"name": "static_memory_mb", "func": "static_memory_mb"},
        {"name": "training_memory_mb", "func": "training_memory_mb"},
        {"name": "theory_memory_scalars", "func": "theory_memory_scalars"},
        {"name": "theory_memory_bytes", "func": "theory_memory_bytes"},
        {"name": "theory_memory_mb", "func": "theory_memory_mb"},
        {"name": "theory_compute_scalars", "func": "theory_compute_scalars"},
        {"name": "theory_access_scalars", "func": "theory_access_scalars"},
        {"name": "theory_time_proxy", "func": "theory_time_proxy"},
        {"name": "avg_synaptic_ops", "value": "final.avg_synaptic_operations"},
        {"name": "firing_rate", "value": "final.firing_rate"},
        {"name": "energy_per_sample_pj", "value": "final.energy_per_sample_pj"},
        {"name": "time_eval_sample_acc_T1", "func": "time_eval_t1"},
        {"name": "time_eval_sample_acc_P25", "func": "time_eval_p25"},
        {"name": "time_eval_sample_acc_P50", "func": "time_eval_p50"},
        {"name": "time_eval_sample_acc_P100", "func": "time_eval_p100"},
        {"name": "status", "value": "status"},
        {"name": "duration_sec", "value": "duration_seconds"},
        {"name": "started_at", "value": "started_at"},
        {"name": "finished_at", "value": "finished_at"},
    ]


def per_epoch_columns():
    return [
        {"name": "run_id", "value": "run_id"},
        {"name": "dataset", "func": "dataset_label"},
        {"name": "learner", "value": "config.LEARNER"},
        {"name": "epoch", "value": "epoch"},
        {"name": "train_loss", "value": "history.loss"},
        {"name": "train_acc", "value": "history.acc"},
        {"name": "sample_acc", "value": "history.sample_acc"},
        {"name": "window_acc", "value": "history.window_acc"},
        {"name": "evaluation_split", "value": "meta.evaluation_split"},
        {"name": "validation_samples", "value": "meta.num_validation_samples"},
        {"name": "test_samples", "value": "meta.num_test_samples"},
        {"name": "theory_model_version", "value": "memory.theory.model_version"},
        {"name": "theory_work_scope", "value": "memory.theory.work_scope"},
        {"name": "windows_per_sample", "value": "memory.theory.windows_per_sample"},
        {"name": "final_sample_acc", "value": "final.sample_acc"},
    ]


def build_for_group(group: Dict[str, str]):
    key = group["key"]
    title = group["title"]
    runs = [run_pattern(group["prefix"])]

    comps = [
        {
            "name": f"[{title}] Epoch curves",
            "runs": runs,
            "dest": {
                "type": "plot",
                "out_path": f"{key}/epoch_curves",
                "panels": [
                    {"x": "epoch", "y": "loss", "plot": "line", "title": "Train loss vs epoch"},
                    {"x": "epoch", "y": "acc", "plot": "line", "title": "Train accuracy vs epoch"},
                    {"x": "run", "y": "final.sample_acc", "plot": "bar", "title": "Final sample accuracy"},
                ],
                "style": {"dpi": 140, "figsize": [12, 6], "tight_layout": True},
            },
        },
        {
            "name": f"[{title}] Estimated cost bars",
            "runs": runs,
            "dest": {
                "type": "plot",
                "out_path": f"{key}/estimated_costs",
                "panels": [
                    {"x": "run", "y": "func:theory_memory_mb", "plot": "bar", "title": "Estimated training memory (MB)"},
                    {"x": "run", "y": "func:theory_compute_scalars", "plot": "bar", "title": "Compute per batch of original sequences"},
                    {"x": "run", "y": "func:theory_access_scalars", "plot": "bar", "title": "Accesses per batch of original sequences"},
                    {"x": "run", "y": "func:theory_time_proxy", "plot": "bar", "title": "Estimated time proxy"},
                ],
                "style": {"dpi": 140, "figsize": [12, 8], "tight_layout": True},
            },
        },
        {
            "name": f"[{title}] Accuracy-cost tradeoffs",
            "runs": runs,
            "dest": {
                "type": "plot",
                "out_path": f"{key}/accuracy_cost_tradeoffs",
                "panels": [
                    {"x": "func:theory_memory_mb", "y": "final.sample_acc", "plot": "scatter", "title": "Accuracy vs estimated memory"},
                    {"x": "func:theory_compute_scalars", "y": "final.sample_acc", "plot": "scatter", "title": "Accuracy vs estimated compute"},
                    {"x": "func:theory_access_scalars", "y": "final.sample_acc", "plot": "scatter", "title": "Accuracy vs estimated access"},
                    {"x": "func:theory_time_proxy", "y": "final.sample_acc", "plot": "scatter", "title": "Accuracy vs estimated time proxy"},
                ],
                "style": {"dpi": 140, "figsize": [12, 8], "tight_layout": True},
            },
        },
        {
            "name": f"[{title}] Time-truncation evaluation",
            "runs": runs,
            "dest": {
                "type": "plot",
                "out_path": f"{key}/time_eval",
                "panels": [
                    {"x": "run", "y": "func:time_eval_t1", "plot": "bar", "title": "Sample accuracy at T=1"},
                    {"x": "run", "y": "func:time_eval_p25", "plot": "bar", "title": "Sample accuracy at 25% T"},
                    {"x": "run", "y": "func:time_eval_p50", "plot": "bar", "title": "Sample accuracy at 50% T"},
                    {"x": "run", "y": "func:time_eval_p100", "plot": "bar", "title": "Sample accuracy at 100% T"},
                ],
                "style": {"dpi": 140, "figsize": [12, 8], "tight_layout": True},
            },
        },
        {
            "name": f"[{title}] Leaderboard CSV",
            "runs": runs,
            "dest": {
                "type": "csv",
                "out_path": f"{key}/tables/leaderboard",
                "spread_epochs": False,
                "columns": leaderboard_columns(),
            },
        },
    ]

    if EMIT_PER_EPOCH_CSV:
        comps.append({
            "name": f"[{title}] Per-epoch CSV",
            "runs": runs,
            "dest": {
                "type": "csv",
                "out_path": f"{key}/tables/per_epoch",
                "spread_epochs": True,
                "columns": per_epoch_columns(),
            },
        })

    return comps


COMPARISONS = []
for run_group in RUN_GROUPS:
    COMPARISONS.extend(build_for_group(run_group))

COMPARISONS.append({
    "name": "[All current runs] Leaderboard CSV",
    "runs": [all_current_runs_pattern()],
    "dest": {
        "type": "csv",
        "out_path": "all_current_runs/leaderboard",
        "spread_epochs": False,
        "columns": leaderboard_columns(),
    },
})

COMPARISONS.append({
    "name": "[All current runs] Accuracy-cost tradeoffs",
    "runs": [all_current_runs_pattern()],
    "dest": {
        "type": "plot",
        "out_path": "all_current_runs/accuracy_cost_tradeoffs",
        "panels": [
            {"x": "func:theory_memory_mb", "y": "final.sample_acc", "plot": "scatter", "title": "Accuracy vs estimated memory"},
            {"x": "func:theory_compute_scalars", "y": "final.sample_acc", "plot": "scatter", "title": "Accuracy vs estimated compute"},
            {"x": "func:theory_access_scalars", "y": "final.sample_acc", "plot": "scatter", "title": "Accuracy vs estimated access"},
            {"x": "func:theory_time_proxy", "y": "final.sample_acc", "plot": "scatter", "title": "Accuracy vs estimated time proxy"},
        ],
        "style": {"dpi": 140, "figsize": [12, 8], "tight_layout": True},
    },
})


# ──────────────────────────────────────────────────────────────────────────────


def main():
    artifacts = run_comparisons(
        base_dir=BASE_DIR,
        comparisons=COMPARISONS,
        custom_funcs=CUSTOM_FUNCS,
    )

    print("\n===== Comparison Summary =====")
    for comp_name, produced in artifacts.items():
        print(f"[{comp_name}]")
        for key, value in produced.items():
            if isinstance(value, list):
                for item in value:
                    print(f"- {item}")
            elif value:
                print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
