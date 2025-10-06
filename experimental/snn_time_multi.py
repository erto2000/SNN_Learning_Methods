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
HIDDEN_SIZE   = 128
BETA          = 0.9
SPIKE_GRAD    = surrogate.fast_sigmoid(slope=25)
LR            = 1e-3
# ───────────────────────────────────────────────────────────────────────────────


# ─── MODEL ────────────────────────────────────────────────────────────────────
class SNNModel(nn.Module):
    """
    Two-layer SNN: Linear -> Leaky (LIF) -> Linear
    Expects sequences as [N, T, D]; unrolls over the TRUE T timesteps.

    IMPORTANT: We manage membrane state EXTERNALLY (no init_hidden, no utils.reset),
    to avoid retaining graphs across iterations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, beta, spike_grad):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)  # no init_hidden
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data_seq: torch.Tensor) -> torch.Tensor:
        """
        data_seq: [N, T, D]
        returns:  logits per timestep -> spk_rec [T, N, C]
        """
        N, T, _ = data_seq.shape
        # Fresh membrane state per forward (detached from any previous graph)
        mem = torch.zeros(N, self.fc1.out_features, device=data_seq.device, dtype=data_seq.dtype)

        outs = []
        for t in range(T):
            h_t = self.fc1(data_seq[:, t, :])   # [N, H]
            spk_t, mem = self.lif(h_t, mem)     # explicit state passing
            out_t = self.fc2(spk_t)             # [N, C] logits
            outs.append(out_t)
        return torch.stack(outs, dim=0)         # [T, N, C]


# ─── TRAIN / EVAL ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, device, optimizer, loss_fn):
    """
    Train on segments:
      - Loader yields X:[B,S,T,D], y:[B]
      - Flatten into X_segs:[Nseg,T,D], y_segs:[Nseg]
      - Optimize CE over per-segment logits (sum over time)
    Returns segment-level loss/accuracy (useful to track learning on segments).
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X_segs, y_segs, _, _ = flatten_segments(X, y)  # [Nseg,T,D], [Nseg]
        if X_segs.numel() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        spk_rec = model(X_segs)            # [T, Nseg, C]
        rates   = spk_rec.sum(dim=0)       # [Nseg, C] logits aggregated over time
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

        spk_rec = model(X_segs)            # [T, Nseg, C]
        rates   = spk_rec.sum(dim=0)       # [Nseg, C]
        preds_seg = rates.argmax(dim=1)    # [Nseg]

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

    # 2) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNModel(input_dim, HIDDEN_SIZE, n_classes, BETA, SPIKE_GRAD).to(device)
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
