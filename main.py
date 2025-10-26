import torch

from utils.datasets import get_dataloaders  # must return X:[B,S,T,D], y:[B], and meta dict
from networks.specs import NetConfig, LayerSpec
from learners.backprop import BackpropLearner
from learners.forward_forward import FFLearner
from learners.eprop import EpropLearner
from learners.pepita import PepitaLearner
from utils.segments import iter_pieces, majority_vote
from utils.common import set_seed, select_device

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
DATASET          = "har"
DATA_ROOT        = "./data"
BATCH_SIZE       = 128
EPOCHS           = 10
SAMPLE_LENGTH    = None            # e.g., 128 or None

HIDDEN_SIZES     = [128]           # backbone hidden dims
BETA             = 0.9
SPIKE_GRAD       = "fast_sigmoid"  # "fast_sigmoid" | "atan" | "sigmoid"
SLOPE            = 25.0            # surrogate slope for learners that need it
THRESHOLD        = 1.0             # used by e-prop (u = pre - THRESHOLD)
NORM             = None            # None | "layernorm" | "batchnorm"
INIT_TYPE        = "kaiming_uniform"  # "kaiming_normal", "xavier_uniform", etc.

LEARNER          = "bp"            # "bp" | "ff" | "eprop" | "pepita"

# Backprop specific
BP_AGG           = "sum"           # "sum" | "mean" | "last"
BP_HEAD          = "logits"        # "logits" | "lif"
BP_LR            = 1e-3

# FF specific
FF_ALPHA         = 0.6
FF_LR            = 1e-3

# E-Prop specific
EP_USE_REC       = False
EP_LR_IN         = 5e-4
EP_LR_REC        = 5e-4
EP_LR_OUT        = 1e-3
EP_DROP_DIAG     = True
EP_WEIGHT_CLIP   = 1.5

# PEPITA specific
PEP_MODE         = "original"      # "original" | "accum"
PEP_LR           = 1e-2
PEP_F_FACTOR     = 0.05

SEED             = 123
# ──────────────────────────────────────────────────────────────────────────────


def build_cfg(D: int, K: int) -> NetConfig:
    layers = []
    din = D
    for h in HIDDEN_SIZES:
        layers.append(LayerSpec(dim_in=din, dim_out=h, recurrent=False, norm=NORM))
        din = h
    return NetConfig(
        layers=layers,
        beta=BETA,
        spike_grad=SPIKE_GRAD,
        slope=SLOPE,
        threshold=THRESHOLD,
        head=("logits" if LEARNER != "ff" else None),
        init=INIT_TYPE,
    )


@torch.no_grad()
def eval_epoch(learner, test_loader, device, n_classes):
    acc_sum, nb = 0.0, 0
    for Xp, yp, sample_ids, B in iter_pieces(test_loader, device):
        preds = learner.predict_batch(Xp)                 # [N]
        yb = majority_vote(preds, sample_ids, n_classes, B)
        acc_sum += (yb == yp.view(B, -1)[:, 0]).float().mean().item()
        nb += 1
    return {"sample_acc": 100 * acc_sum / max(1, nb)}


def main():
    set_seed(SEED)
    device = select_device()

    train_loader, test_loader, meta = get_dataloaders(
        DATASET,
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        sample_length=SAMPLE_LENGTH,
    )
    D = meta["input_dim"]; K = meta["n_classes"]

    cfg = build_cfg(D, K)

    # Instantiate learner
    if LEARNER == "bp":
        learner = BackpropLearner(cfg, meta, device, agg=BP_AGG, head=BP_HEAD, lr=BP_LR)
    elif LEARNER == "ff":
        learner = FFLearner(cfg, meta, device, alpha=FF_ALPHA, lr=FF_LR, total_epochs=EPOCHS)
    elif LEARNER == "eprop":
        learner = EpropLearner(
            cfg, meta, device,
            use_recurrence=EP_USE_REC,
            lr_in=EP_LR_IN, lr_rec=EP_LR_REC, lr_out=EP_LR_OUT,
            drop_diag=EP_DROP_DIAG, weight_clip=EP_WEIGHT_CLIP,
        )
    elif LEARNER == "pepita":
        learner = PepitaLearner(
            cfg, meta, device,
            mode=PEP_MODE, lr=PEP_LR, f_factor=PEP_F_FACTOR
        )
    else:
        raise ValueError(f"Unknown learner: {LEARNER}")

    print(f"[Data] {DATASET.upper()} | classes={K} | D={D} | segment_T≈{meta['time_steps']}")
    print(f"[Arch] hidden={HIDDEN_SIZES} | norm={NORM} | base_head={cfg.head} | learner={LEARNER}")
    if LEARNER == "eprop":
        print(f"[E-Prop] recurrence={EP_USE_REC}")

    # ── training controlled only here ────────────────────────
    for epoch in range(1, EPOCHS + 1):
        n_tr, acc_tr_sum, loss_sum, aux_msg = 0, 0.0, 0.0, ""
        for Xp, yp, _, _ in iter_pieces(train_loader, device):
            stats = learner.train_step(Xp, yp)
            n_tr += 1
            if "acc" in stats:     acc_tr_sum += stats["acc"]
            if "loss" in stats:    loss_sum   += stats["loss"]
            if "loss_per_step" in stats: loss_sum += stats["loss_per_step"]
            # optional short aux status (e.g., current FF layer)
            if "layer" in stats:
                aux_msg = f" | layer:{stats['layer']}"
        # epoch boundary hook for FF
        if hasattr(learner, "on_epoch_end"):
            ff_state = learner.on_epoch_end()
            if ff_state.get("layer_advanced", False):
                aux_msg += " | layer_advanced"

        stats_te = eval_epoch(learner, test_loader, device, K)
        loss_avg = loss_sum / max(1, n_tr)
        acc_avg  = acc_tr_sum / max(1, n_tr)
        msg = f"Epoch {epoch:02d} | loss:{loss_avg:.4f} | acc:{acc_avg:.2f}% | test:{stats_te['sample_acc']:.2f}%{aux_msg}"
        print(msg)


if __name__ == "__main__":
    main()
