# utils/training.py
import torch
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from networks.specs import NetConfig, LayerSpec
from utils.segments import iter_pieces, iter_piece_batches, majority_vote
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
        layers.append(LayerSpec(dim_in=din, dim_out=h, recurrent=g["RECURRENT"], norm=g["NORM"], bias=g.get("HIDDEN_BIAS", False)))
        din = h
    return NetConfig(
        layers=layers,
        beta=g["BETA"],
        spike_grad=g["SPIKE_GRAD"],
        slope=g["SLOPE"],
        threshold=g["THRESHOLD"],
        head=g["HEAD"],
        head_bias=g.get("HEAD_BIAS", False),
        init=g["INIT_TYPE"],
    )


def _time_eval_specs(T, time_eval_fracs=None, time_eval_include_t1: bool = False):
    specs = []
    seen = set()

    if time_eval_include_t1:
        specs.append(("T1", 1))
        seen.add("T1")

    for frac in time_eval_fracs or []:
        frac = float(frac)
        if frac > 1.0:
            frac = frac / 100.0
        if frac <= 0.0:
            continue

        key = f"P{int(round(frac * 100))}"
        if key in seen:
            continue

        if torch.is_tensor(T):
            T_eval = torch.minimum((frac * T).round().long().clamp_min(1), T)
        else:
            T_eval = max(1, min(T, int(round(frac * T))))

        specs.append((key, T_eval))
        seen.add(key)

    return specs


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
        for pieces in iter_piece_batches(test_loader, device, chunk_segments=True, dtype=runtime_dtype):
            batch_preds, batch_ids, batch_lengths = [], [], []
            partial_preds, partial_lengths = {}, {}
            gt_per_sample = None
            for Xw, yw, sample_ids, B, time_mask in pieces:
                logits, activity = learner.forward(Xw, return_activity=True, time_mask=time_mask)
                preds_w = logits.argmax(dim=-1)
                correct_windows += (preds_w == yw).sum().item()
                total_windows += yw.numel()
                batch_preds.append(preds_w)
                batch_ids.append(sample_ids)
                if gt_per_sample is None:
                    gt_per_sample = torch.full((B,), -1, dtype=torch.long, device=yw.device)
                gt_per_sample.index_copy_(0, sample_ids, yw)

                lengths = (time_mask.sum(dim=1) if time_mask is not None else
                           torch.full((len(yw),), Xw.shape[1], device=Xw.device, dtype=torch.long))
                batch_lengths.append(lengths)
                for key, T_eval in _time_eval_specs(lengths, time_eval_fracs, time_eval_include_t1):
                    if torch.is_tensor(T_eval):
                        horizon = int(T_eval.max())
                        partial_mask = torch.arange(horizon, device=Xw.device)[None, :] < T_eval[:, None]
                        unchanged = torch.equal(T_eval, lengths)
                    else:
                        horizon, partial_mask = T_eval, None
                        unchanged = time_mask is None and horizon == Xw.shape[1]
                    logits_t = (logits if unchanged else
                                learner.forward(Xw[:, :horizon, :], return_activity=False,
                                                time_mask=partial_mask))
                    preds_t = logits_t.argmax(dim=-1)
                    partial_preds.setdefault(key, []).append(preds_t)
                    vote_lengths = (T_eval if torch.is_tensor(T_eval) else
                                    torch.full_like(lengths, T_eval))
                    partial_lengths.setdefault(key, []).append(vote_lengths)
                    time_eval_window_correct[key] = time_eval_window_correct.get(key, 0) + (preds_t == yw).sum().item()
                    time_eval_window_total[key] = time_eval_window_total.get(key, 0) + yw.numel()

                total_spike_count += float(activity["total_spike_count"])
                total_synops += float(activity["synaptic_operations"])
                total_neuron_slots += float(activity["num_neuron_slots"])
                total_neuron_updates += float(activity["neuron_updates"])
                total_input_mac_ops += float(activity["input_mac_ops"])
                total_activity_samples += int(activity["num_samples"])

            if not batch_preds:
                continue
            if (gt_per_sample < 0).any():
                raise ValueError("Evaluation batch contains a sample with no valid windows")
            ids = torch.cat(batch_ids)
            preds_sample = majority_vote(torch.cat(batch_preds), ids, num_classes=n_classes, B=B,
                                         weights=torch.cat(batch_lengths))
            sample_correct += (preds_sample == gt_per_sample).sum().item()
            sample_total += B
            for key, predictions in partial_preds.items():
                preds_t = majority_vote(torch.cat(predictions), ids, num_classes=n_classes, B=B,
                                         weights=torch.cat(partial_lengths[key]))
                time_eval_sample_correct[key] = time_eval_sample_correct.get(key, 0) + (preds_t == gt_per_sample).sum().item()
                time_eval_sample_total[key] = time_eval_sample_total.get(key, 0) + B

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

    for key in time_eval_window_correct:
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
    cost_profile=None,
    cost_source_profile=None,
    evaluation_split: str = "test",
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, float]]]:
    epoch_log: Dict[int, Dict[str, float]] = {}

    runtime_dtype = str_to_dtype(dtype_str)

    # Keep model in selected dtype for the whole training loop
    learner.model.to(device=device, dtype=runtime_dtype)

    for epoch in range(1, epochs + 1):
        learner.model.train()

        n_tr, acc_tr_sum, loss_sum, loss_weight_sum, aux_msg = 0, 0.0, 0.0, 0.0, ""
        for recording_batch in train_loader:
            X_batch, _, extra = recording_batch
            segment_mask = extra.get("seg_mask")
            multiple = X_batch.dim() == 4 and (bool((segment_mask.sum(1) > 1).any()) if segment_mask is not None else X_batch.shape[1] > 1)
            accumulate = multiple and getattr(learner, "mode", None) != "original"
            if accumulate:
                learner.begin_recording_batch()
            for Xp, yp, _, original_B, time_mask, sample_weight in iter_pieces(
                    [recording_batch], device, chunk_segments=True, dtype=runtime_dtype,
                    source_length_profile=cost_source_profile if epoch == 1 else None,
                    include_weights=True):
                if epoch == 1 and cost_profile is not None:
                    if time_mask is None:
                        cost_profile[int(Xp.shape[1])] += int(Xp.shape[0])
                    else:
                        for length in time_mask.sum(dim=1).tolist():
                            cost_profile[int(length)] += 1
                learning_weight = sample_weight * (len(yp) / original_B) if accumulate else sample_weight
                stats = learner.train_step(Xp, yp, time_mask=time_mask, sample_weight=learning_weight)
                if accumulate and "loss" in stats:
                    stats["loss"] *= original_B / len(yp)
                n_tr += len(yp)
                if "acc" in stats:
                    acc_tr_sum += stats["acc"] * len(yp)
                if "loss" in stats:
                    loss_sum += stats["loss"] * len(yp)
                    loss_weight_sum += float(sample_weight.sum())
                if "layer" in stats:
                    aux_msg = f" | layer:{stats['layer']}"
            if accumulate:
                learner.end_recording_batch()

        if hasattr(learner, "on_epoch_end"):
            ff_state = learner.on_epoch_end()
            if ff_state.get("layer_advanced", False):
                aux_msg += " | layer_advanced"

        # Learners retain a batch-mean gradient. Report the weighted loss per
        # original recording so extra windows do not artificially lower it.
        loss_avg = loss_sum / max(1.0, loss_weight_sum)
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

            width = len(str(epochs))
            print(f"|  [{ts}] Epoch {epoch:>{width}}/{epochs}  |  Loss {loss_avg:.4f}  |  Train {acc_avg:.2f}%  |  "
                  f"{evaluation_split.title()} {stats_te['sample_acc']:.2f}%{aux_msg}")
            epoch_log[epoch] = {
                "loss": loss_avg,
                "acc": acc_avg,
                **stats_te,
                "timestamp": ts,
            }
        else:
            width = len(str(epochs))
            print(f"|  [{ts}] Epoch {epoch:>{width}}/{epochs}  |  Loss {loss_avg:.4f}  |  Train {acc_avg:.2f}%{aux_msg}")
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
