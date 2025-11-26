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
from utils.training import build_cfg, run_train_loop
from learners.registry import LEARNER_REGISTRY


# ---------- tiny tee to capture console while echoing ----------
class _TeeIO(io.StringIO):
    def __init__(self, real_stdout):
        super().__init__()
        self._real = real_stdout

    def write(self, s):
        self._real.write(s)
        return super().write(s)

    def flush(self):
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
        return LearnerCls(cfg, meta, device, agg=g["BP_AGG"], lr=g["BP_LR"])
    if name == "ff":
        return LearnerCls(cfg, meta, device, alpha=g["FF_ALPHA"], lr=g["FF_LR"], total_epochs=g["EPOCHS"])
    if name == "eprop":
        return LearnerCls(
            cfg, meta, device,
            lr_in=g["EP_LR_IN"], lr_rec=g["EP_LR_REC"], lr_out=g["EP_LR_OUT"],
            drop_diag=g["EP_DROP_DIAG"], weight_clip=g["EP_WEIGHT_CLIP"],
        )
    if name == "pepita":
        return LearnerCls(cfg, meta, device, mode=g["PEP_MODE"], lr=g["PEP_LR"], max_rel_step=g["PEP_MAX_REL_STEP"],
                          target_modulation_ratio=g["PEP_MOD_RATIO"])
    raise ValueError(f"Unhandled learner: {name}")


def _print_header(run_id: str, g: Dict[str, Any], meta: Dict[str, Any]) -> None:
    print(f"\n=== Run: {run_id} ===")
    print(
        f"[Data] {g['DATASET'].upper()} | input_dim={meta['input_dim']} | classes={meta['n_classes']} | "
        f"time_steps={meta.get('time_steps', 'n/a')} | "
        f"number_of_samples(train/test)={meta['num_train_samples']}/{meta['num_test_samples']} | "
        f"epoch={g['EPOCHS']} | batch_size={g['BATCH_SIZE']}"
    )
    print(
        f"[Arch] hidden={g['HIDDEN_SIZES']} | norm={g['NORM']} | base_head={g['HEAD']} | "
        f"recurrent={g['RECURRENT']} | learner={g['LEARNER']}"
    )


def _print_memory_info(static_bytes: int,
                       train_bytes: int | None,
                       batch_size: int,
                       time_steps: int | None,
                       fp_bytes: int) -> None:
    mb = 1024 ** 2
    static_mb = static_bytes / mb if static_bytes is not None else float("nan")
    if train_bytes is not None and time_steps is not None:
        train_mb = train_bytes / mb
        print(
            f"[Memory] dtype={fp_bytes*8}-bit | "
            f"static={static_mb:.2f} MB | "
            f"train_batch={train_mb:.2f} MB (B={batch_size}, T={time_steps})"
        )
    else:
        print(
            f"[Memory] dtype={fp_bytes*8}-bit | "
            f"static={static_mb:.2f} MB | train_batch=n/a"
        )


# ---------- core ----------
def run_one(config: Dict[str, Any]) -> Dict[str, Any]:
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

        tee = _TeeIO(real_stdout=os.sys.stdout)
        with redirect_stdout(tee), redirect_stderr(tee):
            try:
                # Repro + device
                set_seed(g["SEED"])
                device = select_device()

                # Data
                train_loader, test_loader, meta = get_dataloaders(
                    g["DATASET"],
                    root=g["DATA_ROOT"],
                    batch_size=g["BATCH_SIZE"],
                    max_samples=g["MAX_SAMPLES"],
                    transform=transform,
                    num_workers=g.get("NUM_WORKERS"),
                    pin_memory=g.get("PIN_MEMORY"),
                    **g.get("DATASET_KW", {}),
                )

                # Model + learner
                cfg = build_cfg(meta["input_dim"], meta["n_classes"], g)
                learner = _make_learner(cfg, meta, device, g)

                # Memory estimates (auto: uses meta time_steps and model dtype)
                fp_bytes = _infer_fp_bytes(learner.model)
                time_steps = meta.get("time_steps")
                static_mem_bytes = learner.get_static_memory_bytes(fp_bytes=fp_bytes)
                train_mem_bytes = None
                if time_steps is not None:
                    train_mem_bytes = learner.get_training_memory_bytes(
                        batch=g["BATCH_SIZE"],
                        time_steps=time_steps,
                        fp_bytes=fp_bytes,
                    )

                # Pretty header
                _print_header(run_id, g, meta)
                _print_memory_info(
                    static_bytes=static_mem_bytes,
                    train_bytes=train_mem_bytes,
                    batch_size=g["BATCH_SIZE"],
                    time_steps=time_steps,
                    fp_bytes=fp_bytes,
                )

                # Train
                final_stats, epoch_log = run_train_loop(
                    learner,
                    train_loader,
                    test_loader,
                    device,
                    meta["n_classes"],
                    epochs=g["EPOCHS"],
                    test_every_epoch=g["TEST_EVERY_EPOCH"],
                    eval_dtype_str=g.get("EVAL_DTYPE", "fp32"),
                    use_int8_weights=bool(g.get("EVAL_INT8_WEIGHTS", False)),
                )

                if not g["TEST_EVERY_EPOCH"]:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{ts}] [Final Test] sample_acc:{final_stats['sample_acc']:.2f}% | window_acc:{final_stats['window_acc']:.2f}%")

                status = "ok"
                error = None
                tb = None

            except Exception as e:
                status = "error"
                error = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                ts_err = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{ts_err}] [Error] Run '{run_id}' failed:")
                print(tb)
                final_stats, epoch_log = {}, {}
                meta = locals().get("meta", {})
                # free CUDA for later runs
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
            # fp_bytes, static_mem_bytes, train_mem_bytes may not exist if exception
            # so fetch safely from locals()
            fp_bytes_loc = locals().get("fp_bytes")
            static_loc = locals().get("static_mem_bytes")
            train_loc = locals().get("train_mem_bytes")
            time_steps_loc = locals().get("time_steps")
            if fp_bytes_loc is not None and static_loc is not None:
                memory_info = {
                    "fp_bytes": fp_bytes_loc,
                    "static_bytes": static_loc,
                    "training_bytes_per_batch": train_loc,
                    "batch_size": g.get("BATCH_SIZE"),
                    "time_steps": time_steps_loc,
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
            "config": deepcopy(cfg),
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
    print("\n===== Summary =====")
    for r in results:
        g = r.get("config", {})
        status = r.get("status", "ok")
        run_id = r.get("run_id", "?")
        dataset = g.get("DATASET", "?")
        learner = g.get("LEARNER", "?")
        epochs = g.get("EPOCHS", "?")
        if status == "ok":
            final_acc = r.get("final", {}).get("sample_acc", float('nan'))
            print(f"{run_id:>30s} | data={dataset:<10s} | learner={learner:<7s} | status=OK     | E={epochs:<3} | acc={final_acc:6.2f}%")
        else:
            err_msg = (r.get("error") or "").splitlines()[0][:120]
            print(f"{run_id:>30s} | data={dataset:<10s} | learner={learner:<7s} | status=FAILED | E={epochs:<3} | acc=   n/a | err: {err_msg}")
