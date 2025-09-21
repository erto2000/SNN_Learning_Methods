#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from snntorch import utils, surrogate
import snntorch as snn

from datasets import get_dataloaders

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_NAME = "har"              # ← switch to "wisdm" or "speech_commands"
DATA_ROOT    = "../data"
BATCH_SIZE   = 128
NUM_EPOCHS   = 10
HIDDEN_SIZE  = 128
BETA         = 0.9
SPIKE_GRAD   = surrogate.fast_sigmoid(slope=25)
LR           = 1e-3
NUM_WORKERS  = 2
# For WISDM/HAR:
WINDOW_LEN   = 128                # ignored for Speech Commands
# ────────────────────────────────────────────────────────────────────────────────


# ─── MODEL ────────────────────────────────────────────────────────────────────
class SNNModel(nn.Module):
    """
    Two-layer SNN: Linear -> Leaky (LIF) -> Linear
    Input expects [B, T, input_dim]; we unroll over T.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, time_steps, beta, spike_grad):
        super().__init__()
        self.time_steps = time_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        utils.reset(self.net)   # reset hidden state at start of sequence
        spk_rec = []
        for t in range(self.time_steps):
            x_t = data[:, t, :]      # [B, input_dim]
            spk = self.net(x_t)      # [B, n_classes]
            spk_rec.append(spk)
        return torch.stack(spk_rec, dim=0)  # [T, B, C]


# ─── TRAIN / EVAL ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for series, labels in loader:
        series, labels = series.to(device), labels.to(device)

        optimizer.zero_grad()
        spk_rec = model(series)            # [T, B, C]
        rates   = spk_rec.sum(dim=0)       # [B, C]
        loss    = loss_fn(rates, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = rates.argmax(dim=1)
        total_acc += (preds == labels).float().mean().item()
    return total_loss / len(loader), 100.0 * total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc = 0.0
    for series, labels in loader:
        series, labels = series.to(device), labels.to(device)
        rates = model(series).sum(dim=0)
        preds = rates.argmax(dim=1)
        acc += (preds == labels).float().mean().item()
    return 100.0 * acc / len(loader)


def main():
    # 1) Data
    train_loader, test_loader, meta = get_dataloaders(
        DATASET_NAME, root=DATA_ROOT, batch_size=BATCH_SIZE,
        window_len=WINDOW_LEN, num_workers=NUM_WORKERS
    )

    n_classes  = meta["n_classes"]
    input_dim  = meta["input_dim"]
    time_steps = meta["time_steps"]
    class_names = meta["class_names"]

    print(f"[Data] {DATASET_NAME.upper()} | classes={n_classes} | input_dim={input_dim} | T={time_steps}")
    print(f"[Classes] {class_names}")

    # 2) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNModel(input_dim, HIDDEN_SIZE, n_classes, time_steps, BETA, SPIKE_GRAD).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # 3) Train
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%")

    # 4) Test
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
