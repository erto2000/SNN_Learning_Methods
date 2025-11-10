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
def eval_epoch(learner, test_loader, device, n_classes):
    learner.model.eval()
    n_tr, acc_tr_sum = 0, 0.0
    sample_correct = sample_total = 0

    for Xw, yw, sample_ids, B in iter_pieces(test_loader, device):
        logits = learner.forward(Xw)        # [Nseg, K]
        preds_w = logits.argmax(dim=-1)     # [Nseg]
        acc_tr_sum += (preds_w == yw).float().mean().item()
        n_tr += 1

        # majority vote per original sample
        gt_per_sample = torch.empty(B, dtype=torch.long, device=yw.device)
        gt_per_sample[:] = -1
        gt_per_sample.index_copy_(0, sample_ids, yw)
        gt_per_sample = torch.where(gt_per_sample < 0, torch.zeros_like(gt_per_sample), gt_per_sample)

        preds_sample = majority_vote(preds_w, sample_ids, num_classes=n_classes, B=B)
        sample_correct += (preds_sample == gt_per_sample).sum().item()
        sample_total   += B

    return {"window_acc": 100 * acc_tr_sum / max(1, n_tr),
            "sample_acc": 100 * sample_correct / max(1, sample_total)}

def run_train_loop(
    learner, train_loader, test_loader, device, n_classes: int, *,
    epochs: int, test_every_epoch: bool
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, float]]]:
    epoch_log: Dict[int, Dict[str, float]] = {}

    for epoch in range(1, epochs + 1):
        n_tr, acc_tr_sum, loss_sum, aux_msg = 0, 0.0, 0.0, ""
        for Xp, yp, _, _ in iter_pieces(train_loader, device, chunk_segments=True):
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
            print(f"[{ts}] Epoch {epoch:02d} | loss:{loss_avg:.4f} | acc:{acc_avg:.2f}% | test_sample_acc:{stats_te['sample_acc']:.2f}% | test_window_acc:{stats_te['window_acc']:.2f}%{aux_msg}")
            epoch_log[epoch] = {"loss": loss_avg, "acc": acc_avg, **stats_te, "timestamp": ts}
        else:
            print(f"[{ts}] Epoch {epoch:02d} | loss:{loss_avg:.4f} | acc:{acc_avg:.2f}%{aux_msg}")
            epoch_log[epoch] = {"loss": loss_avg, "acc": acc_avg, "timestamp": ts}

    final_stats = epoch_log[epochs] if test_every_epoch else eval_epoch(learner, test_loader, device, n_classes)
    return final_stats, epoch_log