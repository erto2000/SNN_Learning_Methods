#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from snntorch import surrogate
import snntorch as snn

from datasets import get_dataloaders, flatten_segments, majority_vote

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_NAME  = "har"
DATA_ROOT     = "../data"
SAMPLE_LENGTH = None

BATCH_SIZE    = 128
NUM_EPOCHS    = 10
BETA          = 0.9
SPIKE_GRAD    = surrogate.fast_sigmoid(slope=25)
LR            = 1e-3

DIMS = [128]
INCLUDE_OUTPUT_LIF = False
# ───────────────────────────────────────────────────────────────────────────────


# ─── MODEL ────────────────────────────────────────────────────────────────────
class SNNModel(nn.Module):
    """
    Multi-layer SNN built from (Linear -> optional LIF) blocks.

    - Expects sequences as [N, T, D]; unrolls over TRUE T timesteps.
    - Manages membrane states EXTERNALLY per forward (fresh zeros each call),
      so no state is retained across iterations/mini-batches.

    Layout:
      input_dim -> DIMS[0] -> ... -> DIMS[-1] -> output_dim
      After each Linear, we add a LIF, except the last layer when
      INCLUDE_OUTPUT_LIF is False.

    Forward output:
      - If INCLUDE_OUTPUT_LIF=False: returns per-timestep LOGITS [T, N, C]
      - If INCLUDE_OUTPUT_LIF=True : returns per-timestep SPIKES [T, N, C]
    """
    def __init__(self, input_dim, dims, output_dim, beta, spike_grad, include_output_lif: bool):
        super().__init__()

        # Assemble full dimension list
        layer_dims = [input_dim] + list(dims) + [output_dim]
        num_layers = len(layer_dims) - 1

        self.include_output_lif = include_output_lif
        self.num_layers = num_layers

        # Linear layers
        self.linears = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(num_layers)
        ])

        # Decide where LIFs exist: after every Linear except possibly the last
        self.lif_after = [
            True if (i < num_layers - 1) else include_output_lif
            for i in range(num_layers)
        ]

        # LIF modules only for places where lif_after[i] is True
        # We'll keep indexing aligned with layers; put placeholders for easier logic
        self.lifs = nn.ModuleList([
            snn.Leaky(beta=beta, spike_grad=spike_grad) if self.lif_after[i] else nn.Identity()
            for i in range(num_layers)
        ])

    def forward(self, data_seq: torch.Tensor) -> torch.Tensor:
        """
        data_seq: [N, T, D]
        returns: per-timestep activations at the network head:
                 - logits if INCLUDE_OUTPUT_LIF=False
                 - spikes if INCLUDE_OUTPUT_LIF=True
                 shape: [T, N, C]
        """
        N, T, _ = data_seq.shape
        device = data_seq.device
        dtype  = data_seq.dtype

        # Fresh membrane states per forward, for each REAL LIF (else None)
        mem_states = []
        for i in range(self.num_layers):
            if self.lif_after[i]:
                # mem size = output width of corresponding Linear
                out_dim = self.linears[i].out_features
                mem_states.append(torch.zeros(N, out_dim, device=device, dtype=dtype))
            else:
                mem_states.append(None)

        outs = []

        # Unroll in time
        for t in range(T):
            z = data_seq[:, t, :]  # [N, D]
            for i in range(self.num_layers):
                z = self.linears[i](z)  # linear pass
                if self.lif_after[i]:
                    # Apply LIF with explicit state passing
                    z, mem_states[i] = self.lifs[i](z, mem_states[i])
                # else: pure linear, keep z as logits for that stage
            # z is either logits (no output LIF) or spikes (with output LIF)
            outs.append(z)

        return torch.stack(outs, dim=0)  # [T, N, C]


# ─── TRAIN / EVAL ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, device, optimizer, loss_fn):
    """
    Train on segments:
      - Loader yields X:[B,S,T,D], y:[B]
      - Flatten into X_segs:[Nseg,T,D], y_segs:[Nseg]
      - Optimize CE over time-aggregated head outputs:
          * If output has no LIF: sum logits over time
          * If output has LIF:    sum spikes (rate code) over time
    Returns segment-level loss/accuracy.
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X_segs, y_segs, _, _ = flatten_segments(X, y)  # [Nseg,T,D], [Nseg]
        if X_segs.numel() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        out_rec = model(X_segs)        # [T, Nseg, C] (spikes if include_output_lif else logits)
        rates   = out_rec.sum(dim=0)   # [Nseg, C]  (sum over time)

        loss    = loss_fn(rates, y_segs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = rates.argmax(dim=1)
        total_acc += (preds == y_segs).float().mean().item()

    n_batches = max(1, len(loader))
    return total_loss / n_batches, 100.0 * total_acc / n_batches


@torch.no_grad()
def evaluate_with_voting(model, loader, device, n_classes: int):
    """
    Evaluate with majority vote over segments per sample:
      - Predict per segment
      - Vote back to per-sample prediction
      - Report sample-level accuracy
    """
    model.eval()
    acc = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)         # X:[B,S,T,D], y:[B]
        B = y.size(0)

        X_segs, _, sample_ids, _ = flatten_segments(X, y)  # [Nseg,T,D], [Nseg], [Nseg]
        if X_segs.numel() == 0:
            continue

        out_rec = model(X_segs)         # [T, Nseg, C]
        rates   = out_rec.sum(dim=0)    # [Nseg, C]
        preds_seg = rates.argmax(dim=1) # [Nseg]

        preds_sample = majority_vote(preds_seg, sample_ids.to(device), n_classes, B)  # [B]
        acc += (preds_sample == y).float().mean().item()

    return 100.0 * acc / max(1, len(loader))


def main():
    # 1) Data: returns [B, S, T, D]; set sample_length to cap/split if desired
    train_loader, test_loader, meta = get_dataloaders(
        DATASET_NAME,
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        sample_length=SAMPLE_LENGTH,
    )
    n_classes  = meta["n_classes"]
    input_dim  = meta["input_dim"]   # per-timestep feature dim (D)
    time_steps = meta["time_steps"]  # segment length (T) chosen by loader
    class_names = meta["class_names"]

    print(f"[Data] {DATASET_NAME.upper()} | classes={n_classes} | D={input_dim} | segment_T={time_steps}")
    print(f"[Classes] {class_names}")
    print(f"[Arch ] DIMS={DIMS} | INCLUDE_OUTPUT_LIF={INCLUDE_OUTPUT_LIF}")

    # 2) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNModel(
        input_dim=input_dim,
        dims=DIMS,
        output_dim=n_classes,
        beta=BETA,
        spike_grad=SPIKE_GRAD,
        include_output_lif=INCLUDE_OUTPUT_LIF
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # 3) Train (segment-level objective)
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc_seg = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  Loss: {tr_loss:.4f}  Train Seg Acc: {tr_acc_seg:.2f}%")

    # 4) Test (majority vote to sample-level)
    test_acc = evaluate_with_voting(model, test_loader, device, n_classes)
    print(f"Test Accuracy (vote): {test_acc:.2f}%")


if __name__ == "__main__":
    main()
