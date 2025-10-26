# utils/training.py
from typing import Dict, Any, Tuple
import torch
from datetime import datetime
from networks.specs import NetConfig, LayerSpec
from utils.segments import iter_pieces, majority_vote

# ---------------------------
# Model config + evaluation
# ---------------------------
def build_cfg(D: int, K: int, g: Dict[str, Any]) -> NetConfig:
    layers = []
    din = D
    for h in g["HIDDEN_SIZES"]:
        layers.append(LayerSpec(dim_in=din, dim_out=h, recurrent=g["RECURRENT"], norm=g["NORM"]))
        din = h
    return NetConfig(
        layers=layers,
        beta=g["BETA"],
        spike_grad=g["SPIKE_GRAD"],
        slope=g["SLOPE"],
        threshold=g["THRESHOLD"],
        head=g["HEAD"],
        init=g["INIT_TYPE"],
    )

@torch.no_grad()
def eval_epoch(learner, test_loader, device, n_classes) -> Dict[str, float]:
    acc_sum, nb = 0.0, 0
    for Xp, yp, sample_ids, B in iter_pieces(test_loader, device):
        preds = learner.predict_batch(Xp)                 # [N]
        yb = majority_vote(preds, sample_ids, n_classes, B)
        acc_sum += (yb == yp.view(B, -1)[:, 0]).float().mean().item()
        nb += 1
    return {"sample_acc": 100.0 * acc_sum / max(1, nb)}

def run_train_loop(
    learner, train_loader, test_loader, device, n_classes: int, *,
    epochs: int, test_every_epoch: bool
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, float]]]:
    epoch_log: Dict[int, Dict[str, float]] = {}

    for epoch in range(1, epochs + 1):
        n_tr, acc_tr_sum, loss_sum, aux_msg = 0, 0.0, 0.0, ""
        for Xp, yp, _, _ in iter_pieces(train_loader, device):
            stats = learner.train_step(Xp, yp)
            n_tr += 1
            if "acc" in stats:           acc_tr_sum += stats["acc"]
            if "loss" in stats:          loss_sum   += stats["loss"]
            if "layer" in stats:         aux_msg = f" | layer:{stats['layer']}"

        if hasattr(learner, "on_epoch_end"):
            ff_state = learner.on_epoch_end()
            if ff_state.get("layer_advanced", False):
                aux_msg += " | layer_advanced"

        loss_avg = loss_sum / max(1, n_tr)
        acc_avg  = acc_tr_sum / max(1, n_tr)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if test_every_epoch:
            stats_te = eval_epoch(learner, test_loader, device, n_classes)
            print(f"[{ts}] Epoch {epoch:02d} | loss:{loss_avg:.4f} | acc:{acc_avg:.2f}% | test:{stats_te['sample_acc']:.2f}%{aux_msg}")
            epoch_log[epoch] = {"loss": loss_avg, "acc": acc_avg, **stats_te, "timestamp": ts}
        else:
            print(f"[{ts}] Epoch {epoch:02d} | loss:{loss_avg:.4f} | acc:{acc_avg:.2f}%{aux_msg}")
            epoch_log[epoch] = {"loss": loss_avg, "acc": acc_avg, "timestamp": ts}

    final_stats = epoch_log[epochs] if test_every_epoch else eval_epoch(learner, test_loader, device, n_classes)
    return final_stats, epoch_log