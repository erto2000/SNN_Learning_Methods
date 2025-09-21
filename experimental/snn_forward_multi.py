#!/usr/bin/env python3
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate

# NEW: dynamic dataset loading
from datasets import get_dataloaders

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
# Dataset config
DATASET_NAME = "har"        # "har", "wisdm", or "speech_commands"
DATA_ROOT    = "../data"
NUM_WORKERS  = 2
WINDOW_LEN   = 128
MAX_SAMPLES  = None         # e.g., 5000 for quick runs (SC caps to 10k if None)

BATCH_SIZE       = 128
TIME_STEPS       = 10         # internal SNN steps per forward (constant current)
BETA             = 0.9
SPIKE_GRAD       = surrogate.fast_sigmoid(slope=25)
ALPHA            = 0.6        # FF loss scale
LR               = 1e-3
EPOCHS_PER_LAYER = 10         # layerwise pretraining epochs
CURRENT_GAIN     = 5.0        # scale after per-sample L2 norm
SEED             = 123
ON_WINDOWS       = platform.system() == "Windows"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

# ─── FF BUILDING BLOCKS ───────────────────────────────────────────────────────
def overlay_y_on_x_flat(x_flat: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Overwrite the first n_classes features with a one-hot label vector
    scaled by the per-sample max over features.
    x_flat : [B, D]
    y      : [B]
    """
    x_ = x_flat.clone()
    B = x_.size(0)
    m = x_.max(dim=1).values.clamp_min(1e-6)  # per-sample max
    x_[:, :n_classes] = 0.0
    x_[torch.arange(B, device=x_.device), y] = m
    return x_

class LeakyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc    = nn.Linear(in_features, out_features, bias=False)
        self.lif   = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.T     = TIME_STEPS
        self.alpha = ALPHA
        self.epochs = EPOCHS_PER_LAYER
        self.opt   = None  # set after .to(DEVICE)

    def forward(self, x):
        """
        x: [B, in_features]; constant current across T steps
        returns spike_count: [B, out_features] accumulated over T
        """
        B = x.size(0)
        mem = torch.zeros(B, self.fc.out_features, device=x.device, dtype=x.dtype)
        spike_count = torch.zeros(B, self.fc.out_features, device=x.device, dtype=x.dtype)

        cur = self.fc(x)
        cur = cur / (cur.norm(p=2, dim=1, keepdim=True) + 1e-4) * CURRENT_GAIN

        for _ in range(self.T):
            spk, mem = self.lif(cur, mem)
            spike_count += spk
        return spike_count

class FFNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            LeakyLayer(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])

    @torch.no_grad()
    def predict(self, x_flat, n_classes: int):
        """
        Vectorized prediction with label overlay.
        x_flat : [B, D]
        returns predicted labels: [B]
        """
        B, D = x_flat.shape
        device = x_flat.device

        x_rep = x_flat.unsqueeze(1).repeat(1, n_classes, 1).view(B * n_classes, D)
        labels = torch.arange(n_classes, device=device).repeat(B)
        x_lbl = overlay_y_on_x_flat(x_rep, labels, n_classes)  # [B*C, D]

        h = x_lbl
        totals = torch.zeros(B * n_classes, device=device)
        for layer in self.layers:
            spk = layer.forward(h)
            totals += (spk**2).mean(dim=1)
            h = spk

        goodness = totals.view(B, n_classes)
        return goodness.argmax(dim=1)

def build_optimizers(net, lr=LR):
    for layer in net.layers:
        layer.opt = optim.Adam(layer.parameters(), lr=lr)

# ─── FF TRAINING ──────────────────────────────────────────────────────────────
def pretrain_layers(net, train_loader, n_classes: int):
    """
    Greedy layerwise forward-forward pretraining.
    Returns: list of per-layer loss histories: [[..], ..]
    """
    all_losses = []
    for idx, layer in enumerate(net.layers, start=1):
        print(f"\n⏳ Pre-training Layer {idx}/{len(net.layers)}: "
              f"{layer.fc.in_features}→{layer.fc.out_features}")
        layer_losses = []
        for epoch in range(1, layer.epochs + 1):
            layer.train()
            running_loss, count = 0.0, 0

            for series, labels in train_loader:
                series, labels = series.to(DEVICE), labels.to(DEVICE)
                # series: [B, T, C]  -> flatten time & channel
                x_flat = series.reshape(series.size(0), -1)  # [B, D]

                # Positive/negative with overlay
                x_pos = overlay_y_on_x_flat(x_flat, labels, n_classes)
                perm  = torch.randperm(labels.size(0), device=DEVICE)
                x_neg = overlay_y_on_x_flat(x_flat, labels[perm], n_classes)

                # Pass through previous layers (frozen)
                h_pos, h_neg = x_pos, x_neg
                if idx > 1:
                    with torch.no_grad():
                        for prev in net.layers[:idx-1]:
                            h_pos = prev.forward(h_pos)
                            h_neg = prev.forward(h_neg)

                spk_pos = layer.forward(h_pos)
                spk_neg = layer.forward(h_neg)
                Gpos = (spk_pos ** 2).mean(dim=1)
                Gneg = (spk_neg ** 2).mean(dim=1)
                delta = Gpos - Gneg

                loss = F.softplus(-layer.alpha * delta).mean()

                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()

                running_loss += loss.item() * series.size(0)
                count += series.size(0)

            epoch_loss = running_loss / max(count, 1)
            layer_losses.append(epoch_loss)
            print(f"  Layer {idx} Epoch {epoch}/{layer.epochs} — loss: {epoch_loss:.4f}")
        all_losses.append(layer_losses)
    return all_losses

@torch.inference_mode()
def evaluate_ff(net, loader, n_classes: int):
    net.eval()
    correct = total = 0
    for series, labels in loader:
        series, labels = series.to(DEVICE), labels.to(DEVICE)
        x_flat = series.reshape(series.size(0), -1)
        pred = net.predict(x_flat, n_classes)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(SEED)

    # 1) Data — dynamic loaders
    train_loader, test_loader, meta = get_dataloaders(
        DATASET_NAME,
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        window_len=WINDOW_LEN,
        num_workers=NUM_WORKERS,
        max_samples=MAX_SAMPLES
    )
    N_CLASSES = meta["n_classes"]
    input_dim = meta["input_dim"]     # C
    time_steps = meta["time_steps"]   # T (approximate for SC; consistent for HAR/WISDM)
    class_names = meta["class_names"]

    print(f"[Data] {DATASET_NAME.upper()} | classes={N_CLASSES} | input_dim={input_dim} | T≈{time_steps}")
    print(f"[Classes] {class_names}")

    # 2) Network dims
    D_IN = time_steps * input_dim
    dims = [D_IN, 512]  # tweak as desired

    # guard: auto-fix if user forgot to start with D_IN
    if dims[0] != D_IN:
        print(f"[warn] dims[0] ({dims[0]}) != input size ({D_IN}). Overriding.")
        dims = [D_IN] + list(dims[1:])

    net = FFNet(dims).to(DEVICE)
    build_optimizers(net, lr=LR)

    # 3) Forward-Forward pretraining
    print("⏳ Starting full-dataset, layerwise pre-training…")
    losses = pretrain_layers(net, train_loader, N_CLASSES)

    # 4) Evaluation
    train_acc = evaluate_ff(net, train_loader, N_CLASSES)
    test_acc  = evaluate_ff(net, test_loader,  N_CLASSES)
    print(f"\n▶️  Final Train Accuracy: {train_acc:.2f}%")
    print(f"▶️   Final Test Accuracy: {test_acc:.2f}%")
