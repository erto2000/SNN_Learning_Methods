# utils/runner.py
from typing import Dict, Any, List
from copy import deepcopy
import traceback, io, os
from datetime import datetime
from contextlib import redirect_stdout
from utils.common import set_seed, select_device
from utils.datasets import get_dataloaders
from utils.training import build_cfg, run_train_loop
from utils.registry import LEARNER_REGISTRY

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

# ---------- registry helpers (unchanged) ----------
def _make_learner(cfg, meta, device, g: Dict[str, Any]):
    name = g["LEARNER"]
    LearnerCls = LEARNER_REGISTRY.get(name)
    if LearnerCls is None:
        raise ValueError(f"Unknown learner: {name}")

    if name == "bp":
        return LearnerCls(cfg, meta, device, agg=g["BP_AGG"], head=g["BP_HEAD"], lr=g["BP_LR"])
    if name == "ff":
        return LearnerCls(cfg, meta, device, alpha=g["FF_ALPHA"], lr=g["FF_LR"], total_epochs=g["EPOCHS"])
    if name == "eprop":
        return LearnerCls(
            cfg, meta, device,
            use_recurrence=g["EP_USE_REC"],
            lr_in=g["EP_LR_IN"], lr_rec=g["EP_LR_REC"], lr_out=g["EP_LR_OUT"],
            drop_diag=g["EP_DROP_DIAG"], weight_clip=g["EP_WEIGHT_CLIP"],
        )
    if name == "pepita":
        return LearnerCls(cfg, meta, device, mode=g["PEP_MODE"], lr=g["PEP_LR"], f_factor=g["PEP_F_FACTOR"])
    raise ValueError(f"Unhandled learner: {name}")

def _print_header(run_id: str, g: Dict[str, Any], meta: Dict[str, Any]) -> None:
    print(f"\n=== Run: {run_id} ===")
    print(f"[Data] {g['DATASET'].upper()} | classes={meta['n_classes']} | D={meta['input_dim']} | segment_T≈{meta['time_steps']}")
    print(f"[Arch] hidden={g['HIDDEN_SIZES']} | norm={g['NORM']} | base_head={( 'logits' if g['LEARNER']!='ff' else None)} | learner={g['LEARNER']}")
    if g["LEARNER"] == "eprop":
        print(f"[E-Prop] recurrence={g['EP_USE_REC']}")

# ---------- core ----------
def run_one(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a single experiment and RETURNS a dict with everything
    (including captured console_log). No file writing here.
    """
    g = deepcopy(config)
    run_id = g.get("RUN_ID", f"{g['DATASET']}-{g['LEARNER']}-seed{g['SEED']}")
    start_dt = datetime.now()
    started_at = start_dt.isoformat(timespec="seconds")

    tee = _TeeIO(real_stdout=os.sys.stdout)
    with redirect_stdout(tee):
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
                sample_length=g["SAMPLE_LENGTH"],
                stride=g["STRIDE"],
            )

            # Model + learner
            cfg = build_cfg(meta["input_dim"], meta["n_classes"], g)
            learner = _make_learner(cfg, meta, device, g)

            # Pretty header
            _print_header(run_id, g, meta)

            # Train
            final_stats, epoch_log = run_train_loop(
                learner, train_loader, test_loader, device, meta["n_classes"],
                epochs=g["EPOCHS"], test_end_only=g["TEST_END_ONLY"],
            )

            if g["TEST_END_ONLY"]:
                from datetime import datetime as _dt
                ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] [Final Test] sample_acc:{final_stats['sample_acc']:.2f}%")

            status = "ok"
            error = None
            tb = None

        except Exception as e:
            status = "error"
            error = str(e)
            tb = traceback.format_exc(limit=50)
            print(f"\n[Error] Run '{run_id}' failed:\n{error}\n")
            final_stats, epoch_log = {}, {}
            meta = locals().get("meta", {})
            # free CUDA for later runs
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    finished_at = datetime.now().isoformat(timespec="seconds")
    duration_seconds = (datetime.fromisoformat(finished_at) - start_dt).total_seconds()
    console_text = tee.getvalue()

    return {
        "run_id": run_id,
        "config": g,
        "meta": meta if isinstance(meta, dict) else {},
        "final": final_stats,
        "history": epoch_log,
        "status": status,
        "error": error,
        "traceback": tb,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "console_log": console_text,  # <— so save_results can write it
    }

def run_all(run_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for cfg in run_list:
        run_id = cfg.get("RUN_ID", f"{cfg.get('DATASET','?')}-{cfg.get('LEARNER','?')}-seed{cfg.get('SEED','?')}")
        try:
            out = run_one(cfg)
            results.append(out)
        except Exception as e:
            tb = traceback.format_exc(limit=20)
            results.append({
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
                "console_log": "",  # nothing captured at this level
            })
    return results

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
