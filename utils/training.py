# utils/training.py
import torch
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from networks.specs import NetConfig, LayerSpec
from utils.segments import iter_pieces, majority_vote
from networks.int8_linear import convert_linear_to_int8
from utils.energy import compute_energy_breakdown


def str_to_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in ("fp32", "float32", "32"):
        return torch.float32
    if name in ("fp16", "float16", "16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown dtype string: {name}")


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


def _time_eval_steps(T: int, time_eval_fracs=None, time_eval_include_t1: bool = False):
    steps = []

    if time_eval_include_t1:
        steps.append(1)

    if time_eval_fracs:
        for frac in time_eval_fracs:
            frac = float(frac)
            if frac > 1.0:
                frac = frac / 100.0
            t = int(round(frac * T))
            t = max(1, min(T, t))
            steps.append(t)

    return sorted(set(steps))


@torch.no_grad()
def eval_epoch(
    learner,
    test_loader,
    device,
    n_classes: int,
    *,
    dtype_str: str = None,
    use_int8_weights: bool = False,
    time_eval_fracs=None,
    time_eval_include_t1: bool = False,
) -> Dict[str, float]:
    learner.model.eval()

    runtime_dtype = str_to_dtype(dtype_str)

    if use_int8_weights:
        print("[Int8] Converting Linear weights to int8 for inference-only final eval...")
        convert_linear_to_int8(learner.model, per_channel=True)

    orig_dtype = None
    orig_device = None
    p = next(learner.model.parameters(), None)
    if p is not None:
        orig_dtype = p.dtype
        orig_device = p.device

    learner.model.to(device=device, dtype=runtime_dtype)

    correct_windows = 0
    total_windows = 0
    sample_correct = 0
    sample_total = 0

    time_eval_window_correct = {}
    time_eval_window_total = {}
    time_eval_sample_correct = {}
    time_eval_sample_total = {}

    total_spike_count = 0.0
    total_synops = 0.0
    total_neuron_slots = 0.0
    total_neuron_updates = 0.0
    total_input_mac_ops = 0.0
    total_activity_samples = 0

    try:
        for Xw, yw, sample_ids, B in iter_pieces(test_loader, device, dtype=runtime_dtype):
            logits, activity = learner.forward(Xw, return_activity=True)

            preds_w = logits.argmax(dim=-1)
            correct_windows += (preds_w == yw).sum().item()
            total_windows += yw.numel()

            gt_per_sample = torch.empty(B, dtype=torch.long, device=yw.device)
            gt_per_sample[:] = -1
            gt_per_sample.index_copy_(0, sample_ids, yw)
            gt_per_sample = torch.where(
                gt_per_sample < 0,
                torch.zeros_like(gt_per_sample),
                gt_per_sample,
            )

            preds_sample = majority_vote(preds_w, sample_ids, num_classes=n_classes, B=B)

            sample_correct += (preds_sample == gt_per_sample).sum().item()
            sample_total += B

            for T_eval in _time_eval_steps(Xw.shape[1], time_eval_fracs, time_eval_include_t1):
                key = f"T{T_eval}"

                if T_eval == Xw.shape[1]:
                    logits_t = logits
                else:
                    logits_t = learner.forward(Xw[:, :T_eval, :], return_activity=False)

                preds_w_t = logits_t.argmax(dim=-1)

                time_eval_window_correct[key] = time_eval_window_correct.get(key, 0) + (
                    preds_w_t == yw
                ).sum().item()
                time_eval_window_total[key] = time_eval_window_total.get(key, 0) + yw.numel()

                preds_sample_t = majority_vote(preds_w_t, sample_ids, num_classes=n_classes, B=B)

                time_eval_sample_correct[key] = time_eval_sample_correct.get(key, 0) + (
                    preds_sample_t == gt_per_sample
                ).sum().item()
                time_eval_sample_total[key] = time_eval_sample_total.get(key, 0) + B

            total_spike_count += float(activity["total_spike_count"])
            total_synops += float(activity["synaptic_operations"])
            total_neuron_slots += float(activity["num_neuron_slots"])
            total_neuron_updates += float(activity["neuron_updates"])
            total_input_mac_ops += float(activity["input_mac_ops"])
            total_activity_samples += int(activity["num_samples"])

    finally:
        if orig_dtype is not None:
            learner.model.to(device=orig_device or device, dtype=orig_dtype)

    firing_rate = total_spike_count / max(1.0, total_neuron_slots)
    avg_spike_count = total_spike_count / max(1, total_activity_samples)
    avg_synops = total_synops / max(1, total_activity_samples)

    aggregated_activity = {
        "synaptic_operations": total_synops,
        "neuron_updates": total_neuron_updates,
        "input_mac_ops": total_input_mac_ops,
        "num_samples": total_activity_samples,
    }

    energy = compute_energy_breakdown(aggregated_activity)

    result = {
        "window_acc": 100.0 * correct_windows / max(1, total_windows),
        "sample_acc": 100.0 * sample_correct / max(1, sample_total),

        "avg_spike_count": avg_spike_count,
        "avg_synaptic_operations": avg_synops,
        "firing_rate": firing_rate,

        **energy,

        "dtype": str(dtype_str),
        "eval_int8_weights": bool(use_int8_weights),
    }

    for key in sorted(time_eval_window_correct.keys(), key=lambda x: int(x[1:])):
        result[f"time_eval_window_acc_{key}"] = (
            100.0 * time_eval_window_correct[key] / max(1, time_eval_window_total[key])
        )
        result[f"time_eval_sample_acc_{key}"] = (
            100.0 * time_eval_sample_correct[key] / max(1, time_eval_sample_total[key])
        )

    return result


def run_train_loop(
    learner,
    train_loader,
    test_loader,
    device,
    n_classes: int,
    *,
    epochs: int,
    test_every_epoch: bool,
    dtype_str: str = None,
    use_int8_weights: bool = False,
    time_eval_fracs=None,
    time_eval_include_t1: bool = False,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, float]]]:
    epoch_log: Dict[int, Dict[str, float]] = {}

    runtime_dtype = str_to_dtype(dtype_str)

    # Keep model in selected dtype for the whole training loop
    learner.model.to(device=device, dtype=runtime_dtype)

    for epoch in range(1, epochs + 1):
        learner.model.train()

        n_tr, acc_tr_sum, loss_sum, aux_msg = 0, 0.0, 0.0, ""
        for Xp, yp, _, _ in iter_pieces(train_loader, device, chunk_segments=True, dtype=runtime_dtype):
            stats = learner.train_step(Xp, yp)
            n_tr += 1
            if "acc" in stats:
                acc_tr_sum += stats["acc"]
            if "loss" in stats:
                loss_sum += stats["loss"]
            if "layer" in stats:
                aux_msg = f" | layer:{stats['layer']}"

        if hasattr(learner, "on_epoch_end"):
            ff_state = learner.on_epoch_end()
            if ff_state.get("layer_advanced", False):
                aux_msg += " | layer_advanced"

        loss_avg = loss_sum / max(1, n_tr)
        acc_avg = acc_tr_sum / max(1, n_tr)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if test_every_epoch:
            do_time_eval = epoch == epochs
            stats_te = eval_epoch(
                learner,
                test_loader,
                device,
                n_classes,
                dtype_str=dtype_str,
                use_int8_weights=use_int8_weights,
                time_eval_fracs=time_eval_fracs if do_time_eval else None,
                time_eval_include_t1=time_eval_include_t1 if do_time_eval else False,
            )

            print(
                f"[{ts}] Epoch {epoch:02d} | loss:{loss_avg:.4f} | acc:{acc_avg:.2f}% | "
                f"test_sample_acc:{stats_te['sample_acc']:.2f}% | "
                f"test_window_acc:{stats_te['window_acc']:.2f}%{aux_msg}"
            )
            epoch_log[epoch] = {
                "loss": loss_avg,
                "acc": acc_avg,
                **stats_te,
                "timestamp": ts,
            }
        else:
            print(f"[{ts}] Epoch {epoch:02d} | loss:{loss_avg:.4f} | acc:{acc_avg:.2f}%{aux_msg}")
            epoch_log[epoch] = {
                "loss": loss_avg,
                "acc": acc_avg,
                "dtype": str(dtype_str),
                "timestamp": ts,
            }

    if test_every_epoch:
        final_stats = epoch_log[epochs]
    else:
        final_stats = eval_epoch(
            learner,
            test_loader,
            device,
            n_classes,
            dtype_str=dtype_str,
            use_int8_weights=use_int8_weights,
            time_eval_fracs=time_eval_fracs,
            time_eval_include_t1=time_eval_include_t1,
        )

    return final_stats, epoch_log
