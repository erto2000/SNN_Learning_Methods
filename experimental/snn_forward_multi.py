#!/usr/bin/env python3
"""
Forward-Forward SNN on segmented time-series.

- Uses datasets.get_dataloaders which returns [B, S, T, D]
- Trains on segments (each segment inherits the sample's label)
- Label info is appended as extra channels (non-destructive)
- First SNN layer consumes true segment timesteps T
- Evaluation uses majority vote across segments per sample
"""

import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate

# Import loaders + generic helpers
from datasets import get_dataloaders, flatten_segments, majority_vote

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
DATASET_NAME = "MNIST"
DATA_ROOT    = "../data"
SAMPLE_LENGTH = None

BATCH_SIZE       = 128
DIMS             = [512]      # hidden layer sizes; first auto-set to per-timestep input size (+ n_classes)
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


# ─── HELPERS: non-destructive label concatenation ─────────────────────────────
def add_label_channels(x_seq: torch.Tensor, y: torch.Tensor, n_classes: int, scale="max") -> torch.Tensor:
    """
    x_seq : [B, T, D] real data (unchanged)
    y     : [B]
    returns x_cat: [B, T, D + n_classes] with label channels appended
    """
    B, T, D = x_seq.shape
    hot = F.one_hot(y, num_classes=n_classes).to(x_seq.dtype)   # [B, C]
    if scale == "max":
        s = x_seq.amax(dim=(1, 2)).clamp_min(1e-6)             # [B]
        hot = hot * s.unsqueeze(1)
    label_seq = hot.unsqueeze(1).expand(B, T, n_classes)       # [B, T, C]
    return torch.cat([x_seq, label_seq], dim=-1)               # [B, T, D+C]


# ─── FF BUILDING BLOCKS (sequence-aware) ──────────────────────────────────────
class LeakyLayer(nn.Module):
    """
    If input is [B, T, Din]: run over the TRUE T timesteps (no artificial repeat).
    If input is [B, Din]:    do a single step (no loop).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc    = nn.Linear(in_features, out_features, bias=False)
        self.lif   = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.alpha = ALPHA
        self.epochs = EPOCHS_PER_LAYER
        self.opt   = None  # set after .to(DEVICE)

    def _normalize_current(self, cur):
        # L2 normalize per sample then scale
        return cur / (cur.norm(p=2, dim=1, keepdim=True) + 1e-4) * CURRENT_GAIN

    def forward(self, x):
        """
        x: [B, T, Din] or [B, Din]
        returns spike_count: [B, Dout] (accumulated over time when sequence)
        """
        if x.dim() == 3:
            B, T, _ = x.shape
            mem = torch.zeros(B, self.fc.out_features, device=x.device, dtype=x.dtype)
            spike_count = torch.zeros_like(mem)
            for t in range(T):
                cur_t = self.fc(x[:, t, :])
                cur_t = self._normalize_current(cur_t)
                spk, mem = self.lif(cur_t, mem)
                spike_count += spk
            return spike_count
        elif x.dim() == 2:
            B = x.size(0)
            mem = torch.zeros(B, self.fc.out_features, device=x.device, dtype=x.dtype)
            cur = self._normalize_current(self.fc(x))
            spk, mem = self.lif(cur, mem)  # single step (no repetition)
            return spk
        else:
            raise ValueError(f"Unsupported input rank {x.dim()} for LeakyLayer.forward")


class FFNet(nn.Module):
    def __init__(self, dims):
        """
        dims: [Din, H1, H2, ...]
        """
        super().__init__()
        self.layers = nn.ModuleList([
            LeakyLayer(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])

    @torch.no_grad()
    def predict_segments(self, x_seq: torch.Tensor, n_classes: int):
        """
        Vectorized prediction over segments using label concatenation.
        x_seq : [Nseg, T, Din]   (Din is per-timestep feature size without labels)
        returns predicted labels per segment: [Nseg]
        """
        N, T, D = x_seq.shape
        device = x_seq.device

        # Replicate per class and append one-hot label channels
        x_rep = x_seq.unsqueeze(1).expand(N, n_classes, T, D).reshape(N * n_classes, T, D)
        labels = torch.arange(n_classes, device=device).unsqueeze(0).expand(N, -1).reshape(-1)
        x_lbl = add_label_channels(x_rep, labels, n_classes)  # [N*C, T, D+C]

        h = x_lbl
        totals = torch.zeros(N * n_classes, device=device)
        for layer in self.layers:
            spk = layer.forward(h)  # first pass seq -> vector; then vectors onward
            totals += (spk**2).mean(dim=1)
            h = spk

        goodness = totals.view(N, n_classes)
        return goodness.argmax(dim=1)


def build_optimizers(net, lr=LR):
    for layer in net.layers:
        layer.opt = optim.Adam(layer.parameters(), lr=lr)


# ─── FF TRAINING ON SEGMENTS (with label concatenation) ───────────────────────
def pretrain_layers_on_segments(net, train_loader, n_classes: int):
    """
    Greedy layerwise forward-forward pretraining using segments with label concatenation.
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

            for X, y in train_loader:
                # X: [B, S, T, D], y: [B]
                X = X.to(DEVICE)
                y = y.to(DEVICE)

                X_segs, y_segs, _, _ = flatten_segments(X, y)  # [Nseg, T, D], [Nseg]
                if X_segs.numel() == 0:
                    continue

                # Positive/negative with label concatenation on sequences
                x_pos = add_label_channels(X_segs, y_segs, n_classes)             # [Nseg, T, D+C]
                perm  = torch.randperm(y_segs.size(0), device=DEVICE)
                x_neg = add_label_channels(X_segs, y_segs[perm], n_classes)        # [Nseg, T, D+C]

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

                running_loss += loss.item() * X_segs.size(0)
                count += X_segs.size(0)

            epoch_loss = running_loss / max(count, 1)
            layer_losses.append(epoch_loss)
            print(f"  Layer {idx} Epoch {epoch}/{layer.epochs} — loss: {epoch_loss:.4f}")
        all_losses.append(layer_losses)
    return all_losses


@torch.inference_mode()
def evaluate_ff_with_voting(net, loader, n_classes: int):
    """
    Segment-level predictions + majority vote per sample.
    Returns % accuracy at the sample level.
    """
    net.eval()
    correct = total = 0
    for X, y in loader:
        X = X.to(DEVICE)          # [B, S, T, D]
        y = y.to(DEVICE)          # [B]
        B = y.size(0)

        X_segs, _, sample_ids, _ = flatten_segments(X, y)  # [Nseg, T, D], [Nseg], [Nseg]
        preds_seg = net.predict_segments(X_segs, n_classes) if X_segs.numel() > 0 \
                    else X_segs.new_zeros((0,), dtype=torch.long)
        preds_sample = majority_vote(preds_seg, sample_ids.to(DEVICE), n_classes, B)  # [B]

        correct += (preds_sample == y).sum().item()
        total   += B
    return 100.0 * correct / max(total, 1)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(SEED)

    # 1) Data — dynamic loaders (pass sample_length=... to cap/split)
    train_loader, test_loader, meta = get_dataloaders(
        DATASET_NAME,
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        sample_length=SAMPLE_LENGTH,
    )
    N_CLASSES  = meta["n_classes"]
    D_TIMESTEP = meta["input_dim"]     # per-timestep feature dimension (D)
    T_SEG      = meta["time_steps"]    # segment length used by loader (T)
    class_names = meta["class_names"]

    print(f"[Data] {DATASET_NAME.upper()} | classes={N_CLASSES} | D={D_TIMESTEP} | segment_T={T_SEG}")
    print(f"[Classes] {class_names}")

    # 2) Network dims — use per-timestep features + label channels
    D_IN = D_TIMESTEP + N_CLASSES
    dims = [D_IN] + DIMS
    if dims[0] != D_IN:
        print(f"[warn] dims[0] ({dims[0]}) != input size ({D_IN}). Overriding.")
        dims = [D_IN] + list(dims[1:])

    net = FFNet(dims).to(DEVICE)
    build_optimizers(net, lr=LR)

    # 3) Forward-Forward pretraining (on segments with label concatenation)
    print("⏳ Starting layerwise pre-training on segments (label concatenation)…")
    _ = pretrain_layers_on_segments(net, train_loader, N_CLASSES)

    # 4) Evaluation (segment votes -> sample prediction)
    train_acc = evaluate_ff_with_voting(net, train_loader, N_CLASSES)
    test_acc  = evaluate_ff_with_voting(net, test_loader,  N_CLASSES)
    print(f"\n▶️  Final Train Accuracy (vote): {train_acc:.2f}%")
    print(f"▶️   Final Test  Accuracy (vote): {test_acc:.2f}%")
