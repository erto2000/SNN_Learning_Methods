# utils/runner.py
from typing import Dict, Any, List
from copy import deepcopy
import traceback
import io
import os
import torch
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

from utils.common import set_seed, select_device
from timeseries.registry import get_dataloaders
from utils.training import build_cfg, run_train_loop, str_to_dtype
from learners.registry import LEARNER_REGISTRY
from utils.console import panel, line, section, close, timestamp


# ---------- tiny tee to capture console while echoing ----------
class _TeeIO(io.StringIO):
    def __init__(self, real_stdout=None):
        super().__init__()
        self._real = real_stdout

    def write(self, s):
        if self._real is not None:
            self._real.write(s)
        return super().write(s)

    def flush(self):
        if self._real is not None:
            self._real.flush()
        return super().flush()


def _infer_fp_bytes(model: torch.nn.Module) -> int:
    """Infer element size in bytes from model params."""
    p = next(model.parameters(), None)
    if p is None:
        return 4
    dt = p.dtype
    if dt == torch.float32:
        return 4
    if dt in (torch.float16, torch.bfloat16):
        return 2
    if dt == torch.float64:
        return 8
    return 4


# ---------- registry helpers (unchanged) ----------
def _make_learner(cfg, meta, device, g: Dict[str, Any]):
    name = g["LEARNER"]
    LearnerCls = LEARNER_REGISTRY.get(name)
    if LearnerCls is None:
        raise ValueError(f"Unknown learner: {name}")

    if name == "bp":
        return LearnerCls(cfg, meta, device, agg=g["BP_AGG"], lr=g["BP_LR"], optimizer=g.get("BP_OPTIMIZER", "adam"))
    if name == "ff":
        return LearnerCls(cfg, meta, device, alpha=g["FF_ALPHA"], lr=g["FF_LR"], total_epochs=g["EPOCHS"], optimizer=g.get("FF_OPTIMIZER", "adam"))
    if name == "eprop":
        return LearnerCls(
            cfg, meta, device,
            lr_in=g["EP_LR_IN"], lr_rec=g["EP_LR_REC"], lr_out=g["EP_LR_OUT"],
            drop_diag=g["EP_DROP_DIAG"], weight_clip=g["EP_WEIGHT_CLIP"],
            optimizer=g.get("EP_OPTIMIZER", "adam"),
        )
    if name == "pepita":
        return LearnerCls(
            cfg, meta, device,
            mode=g["PEP_MODE"],
            lr=g["PEP_LR"],
            max_rel_step=g["PEP_MAX_REL_STEP"],
            target_modulation_ratio=g["PEP_MOD_RATIO"],
            optimizer=g.get("PEP_OPTIMIZER", "adam"),
        )
    raise ValueError(f"Unhandled learner: {name}")


def _print_header(run_id: str, g: Dict[str, Any], meta: Dict[str, Any]) -> None:
    panel(f"RUN | {run_id}")
    line(
        f"Started {timestamp()}  |  Evaluate {meta['evaluation_split']}  |  Seed {g['SEED']}  |  "
        f"Split {meta['num_train_samples']}/{meta['num_validation_samples']}/{meta['num_test_samples']} "
        f"(train/validation/test)"
    )
    line(
        f"{g['DATASET'].upper()}  |  Input {meta['input_dim']} x {meta.get('time_steps', '?')}  |  "
        f"{meta['n_classes']} classes"
    )
    line(
        f"{g['LEARNER'].upper()}  |  Hidden {g['HIDDEN_SIZES']}  |  "
        f"Batch {g['BATCH_SIZE']}  |  {g['EPOCHS']} epochs  |  {g.get('DTYPE', 'fp32').upper()}"
    )
    section("TRAINING")


def _print_memory_info(
    static_bytes: int,
    train_bytes: int | None,
    batch_size: int,
    time_steps: int | None,
    fp_bytes: int,
    theory: Dict[str, Any] | None = None,
) -> None:
    mb = 1024 ** 2
    static_mb = static_bytes / mb if static_bytes is not None else float("nan")
    if train_bytes is not None and time_steps is not None:
        train_mb = train_bytes / mb
        print(
            f"[Memory] dtype={fp_bytes*8}-bit | "
            f"param={static_mb:.2f} MB | "
            f"train_batch={train_mb:.2f} MB (B={batch_size}, T={time_steps})"
        )
    else:
        print(
            f"[Memory] dtype={fp_bytes*8}-bit | "
            f"param={static_mb:.2f} MB | train_batch=n/a"
        )

    if theory:
        if "error" in theory:
            print(f"[Theory] unavailable: {theory['error']}")
        else:
            compute = theory.get("compute", {}).get("total_scalars")
            access = theory.get("access", {}).get("total_scalars")
            proxy = theory.get("time_proxy", {}).get("value")
            print(f"[Theory] compute={compute} | access={access} | time_proxy={proxy}")


# ---------- core ----------
def run_one(config: Dict[str, Any], *, tuning: bool = False) -> Dict[str, Any]:
    """
    Runs a single experiment and RETURNS a dict with everything
    (including captured console_log). No file writing here.
    """
    run_id = config.get("RUN_ID", f"{config['DATASET']}-{config['LEARNER']}-seed{config['SEED']}")
    try:
        g = deepcopy(config)

        # Pull out the transform object (avoid serializing it later)
        # and deepcopy to ensure per-run state (e.g., ZScore fit stats) are isolated.
        transform = g.pop("TRANSFORM", None)
        if transform is not None:
            from copy import deepcopy as _dc
            transform = _dc(transform)

        start_dt = datetime.now()
        started_at = start_dt.isoformat(timespec="seconds")

        # Search trials keep a complete saved log but emit one concise result
        # from the search script instead of repeating every training detail.
        tee = _TeeIO(real_stdout=None if tuning else os.sys.stdout)
        with redirect_stdout(tee), redirect_stderr(tee):
            try:
                # Repro + device
                set_seed(g["SEED"])
                device = select_device()

                # Data
                loaders, meta = get_dataloaders(
                    g["DATASET"],
                    root=g["DATA_ROOT"],
                    batch_size=g["BATCH_SIZE"],
                    max_samples=g["MAX_SAMPLES"],
                    transform=transform,
                    num_workers=g.get("NUM_WORKERS"),
                    pin_memory=g.get("PIN_MEMORY"),
                    seed=g["SEED"],
                    data_split=g.get("DATA_SPLIT"),
                    **g.get("DATASET_KW", {}),
                )
                train_loader = loaders['train']
                evaluated_set = 'validation' if tuning else 'test'
                test_loader = loaders[evaluated_set]
                meta['evaluation_split'] = evaluated_set
                meta['num_evaluation_samples'] = len(test_loader.dataset)

                # Model + learner
                cfg = build_cfg(meta["input_dim"], meta["n_classes"], g)
                learner = _make_learner(cfg, meta, device, g)

                # Apply selected runtime dtype before memory estimation / training
                runtime_dtype = str_to_dtype(g.get("DTYPE", "fp32"))
                learner.model.to(device=device, dtype=runtime_dtype)

                # Byte-based costs use the model's actual training dtype.
                fp_bytes = _infer_fp_bytes(learner.model)
                time_steps = meta.get("time_steps")
                static_mem_bytes = learner.get_param_memory_bytes(fp_bytes=fp_bytes)
                train_mem_bytes = None
                cost_estimate = None
                # Pretty header
                _print_header(run_id, g, meta)
                time_eval_enabled = bool(g.get("TIME_EVAL", False))
                time_eval_fracs = g.get("TIME_EVAL_FRACS") if time_eval_enabled else None
                time_eval_include_t1 = bool(g.get("TIME_EVAL_INCLUDE_T1", False)) if time_eval_enabled else False

                # Train
                from collections import Counter
                from utils.costs import training_costs
                cost_profile = Counter()
                final_stats, epoch_log = run_train_loop(
                    learner,
                    train_loader,
                    test_loader,
                    device,
                    meta["n_classes"],
                    epochs=g["EPOCHS"],
                    test_every_epoch=g["TEST_EVERY_EPOCH"],
                    dtype_str=g.get("DTYPE", "fp32"),
                    use_int8_weights=bool(g.get("EVAL_INT8_WEIGHTS", False)),
                    time_eval_fracs=time_eval_fracs,
                    time_eval_include_t1=time_eval_include_t1,
                    cost_profile=cost_profile,
                    evaluation_split=meta["evaluation_split"],
                )

                cost_estimate = training_costs(
                    learner, cost_profile, original_samples=len(train_loader.dataset),
                    batch=g["BATCH_SIZE"], fp_bytes=fp_bytes,
                    alpha=g.get("COST_COMPUTE_WEIGHT", 1.0), beta=g.get("COST_ACCESS_WEIGHT", 1.0),
                )
                train_mem_bytes = cost_estimate["memory"]["total_bytes"]
                time_steps = cost_estimate["time_steps"]
                final_stats["evaluation_split"] = meta["evaluation_split"]
                section("COST")
                line(f"{cost_estimate['windows_per_sample']:.2f} windows/sample  |  "
                     f"Compute {cost_estimate['compute']['total_scalars'] / 1e6:.2f}M  |  "
                     f"Access {cost_estimate['access']['total_scalars'] / 1e6:.2f}M  |  "
                     f"Peak {train_mem_bytes / 1024**2:.2f} MB")
                close(
                    f"{meta['evaluation_split'].upper()} | "
                    f"Sample {final_stats['sample_acc']:.2f}%  |  Window {final_stats['window_acc']:.2f}%"
                )

                status = "ok"
                error = None
                tb = None

            except Exception as e:
                status = "error"
                error = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                close(f"FAILED | {error}")
                print(tb, end="" if tb.endswith("\n") else "\n")
                final_stats, epoch_log = {}, {}
                meta = locals().get("meta", {})
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        finished_at = datetime.now().isoformat(timespec="seconds")
        duration_seconds = (datetime.fromisoformat(finished_at) - start_dt).total_seconds()
        console_text = tee.getvalue()

        memory_info = {}
        if status == "ok":
            fp_bytes_loc = locals().get("fp_bytes")
            static_loc = locals().get("static_mem_bytes")
            train_loc = locals().get("train_mem_bytes")
            time_steps_loc = locals().get("time_steps")
            cost_record = locals().get("cost_estimate")
            if fp_bytes_loc is not None and static_loc is not None:
                memory_info = {
                    "fp_bytes": fp_bytes_loc,
                    "static_bytes": static_loc,
                    "training_bytes_per_batch": train_loc,
                    "batch_size": g.get("BATCH_SIZE"),
                    "time_steps": time_steps_loc,
                    "theory": cost_record,  # Existing result files use this key.
                }

        return {
            "run_id": run_id,
            "config": g,  # note: TRANSFORM removed above (non-serializable)
            "meta": meta if isinstance(meta, dict) else {},
            "final": final_stats,
            "history": epoch_log,
            "status": status,
            "error": error,
            "traceback": tb,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "console_log": console_text,
            "memory": memory_info,
        }
    except Exception as e:
        tb = traceback.format_exc(limit=20)
        return {
            "run_id": run_id,
            "config": deepcopy(config),
            "status": "error",
            "error": str(e),
            "traceback": tb,
            "final": {},
            "history": {},
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "console_log": "",
        }


def summarize(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No runs executed.")
        return
    panel("SUMMARY")
    for r in results:
        g = r.get("config", {})
        status = r.get("status", "ok")
        run_id = r.get("run_id", "?")
        dataset = g.get("DATASET", "?")
        learner = g.get("LEARNER", "?")
        epochs = g.get("EPOCHS", "?")
        if status == "ok":
            final_acc = r.get("final", {}).get("sample_acc", float("nan"))
            line(f"{run_id}  |  {final_acc:.2f}%  |  {dataset.upper()} / {learner.upper()} / {epochs} epochs")
        else:
            err_msg = (r.get("error") or "").splitlines()[0][:120]
            line(f"{run_id}  |  FAILED  |  {dataset.upper()} / {learner.upper()}  |  {err_msg}")
    close(f"{len(results)} run{'s' if len(results) != 1 else ''}")
