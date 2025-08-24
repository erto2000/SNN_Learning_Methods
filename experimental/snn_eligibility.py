# eprop_ecg5000_snntorch_full.py
# Minimal e-prop-ish SNN on UCR ECG5000 with manual eligibility updates
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate, utils
from tslearn.datasets import UCR_UEA_datasets

# ─── Repro / Device ───────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
dataset_name = "ECG5000"
num_epochs   = 15
batch_size   = 128
hidden_size  = 128

# Neuron & learning settings
beta_mem     = 0.9          # LIF membrane leak
beta_e       = 0.99         # eligibility decay (credit assignment timescale)
thr          = 1.0          # LIF threshold (align ψ(v) center)
slope        = 25.0         # surrogate derivative slope
cur_scale    = 1.25         # scales fc1 current -> helps hover near threshold

# Learning rates (separate for hidden/readout)
lr_w1        = 2e-2
lr_w2        = 1e-2

# snnTorch spike surrogate (used by the neuron itself)
spike_grad = surrogate.fast_sigmoid(slope=slope)

# ─── Helper: explicit surrogate derivative ψ(v) (numeric, forward usable) ─────
@torch.no_grad()
def psi_fast_sigmoid(mem, thr=1.0, slope=25.0):
    # logistic derivative: slope * σ(z) * (1-σ(z)), z = slope*(v - thr)
    z = slope * (mem - thr)
    s = torch.sigmoid(z)
    return slope * s * (1.0 - s)

# ─── Load UCR ECG5000 ─────────────────────────────────────────────────────────
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)

# labels 1..C -> 0..C-1
y_train = y_train.astype(int) - 1
y_test  = y_test.astype(int)  - 1

# Ensure shape [N, T, 1]
def ensure_3d(x_np):
    x = torch.from_numpy(x_np).float()
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    return x

X_train_t = ensure_3d(X_train)    # [N, T, 1]
X_test_t  = ensure_3d(X_test)     # [N, T, 1]
series_len = X_train_t.shape[1]
n_classes  = int(np.unique(y_train).size)

# Per-series standardization (helps keep membrane near threshold)
def standardize_per_series(X):  # X: [N, T, 1] torch
    mu = X.mean(dim=1, keepdim=True)
    sd = X.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (X - mu) / sd

X_train_t = standardize_per_series(X_train_t)
X_test_t  = standardize_per_series(X_test_t)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(
    TimeSeriesDataset(X_train_t, y_train),
    batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    TimeSeriesDataset(X_test_t, y_test),
    batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available()
)

# ─── Model ────────────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1  = nn.Linear(1, hidden_dim, bias=True)
        # IMPORTANT: output=True to get (spk, mem) with init_hidden=True
        self.lif1 = snn.Leaky(beta=beta_mem, spike_grad=spike_grad, init_hidden=True, output=True)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
    def reset(self):
        utils.reset(self.lif1)

model = SNN(hidden_dim=hidden_size, output_dim=n_classes).to(device)

# ─── Train (manual e-prop-like updates) ───────────────────────────────────────
model.train()
for epoch in range(num_epochs):
    epoch_loss, epoch_acc = 0.0, 0.0

    for series, labels in train_loader:
        series, labels = series.to(device), labels.to(device)
        B = series.size(0)

        # Reset recurrent state & traces
        model.reset()
        e_w1 = torch.zeros_like(model.fc1.weight, device=device)  # [H,1]
        e_b1 = torch.zeros_like(model.fc1.bias,   device=device)  # [H]

        final_spk = None

        # Unroll over time
        for t in range(series_len):
            x_t = series[:, t, :]                     # [B, 1]
            cur = cur_scale * model.fc1(x_t)          # [B, H] scaled
            spk, mem = model.lif1(cur)                # both tensors

            if t == series_len - 1:
                final_spk = spk.detach()              # [B, H]

            # Eligibility update using explicit ψ(v) (numeric, forward)
            psi = psi_fast_sigmoid(mem, thr=thr, slope=slope)  # [B, H]
            # e_w1(t) = beta_e * e_w1(t-1) + (psi^T @ x_t)/B  -> [H,1]
            e_w1.mul_(beta_e).add_(psi.t() @ x_t / B)
            # e_b1(t) = beta_e * e_b1(t-1) + mean_b(psi)      -> [H]
            e_b1.mul_(beta_e).add_(psi.mean(dim=0))

        # Readout from last spike vector
        out = model.fc2(final_spk)                     # [B, C]
        loss = F.cross_entropy(out, labels)

        # Local errors + manual parameter updates
        with torch.no_grad():
            # δ_out = softmax - one_hot  (grad of CE wrt logits)
            probs = F.softmax(out, dim=1)
            delta_out = probs
            delta_out[torch.arange(B), labels] -= 1     # [B, C]

            # Hidden error (batch mean) to pair with eligibilities
            # δ_hidden = δ_out @ W2 -> [B, H]
            delta_hidden = delta_out @ model.fc2.weight   # [B, H]
            delta_h_mean_col = delta_hidden.mean(dim=0, keepdim=True).t()  # [H,1]
            delta_h_mean_vec = delta_hidden.mean(dim=0)                     # [H]

            # Layer 1 updates
            model.fc1.weight.add_(-lr_w1 * (delta_h_mean_col * e_w1))       # [H,1] .* [H,1]
            model.fc1.bias  .add_(-lr_w1 * (delta_h_mean_vec * e_b1))       # [H]   .* [H]

            # Readout (standard gradient from last timestep)
            grad_W2 = (delta_out.t() @ final_spk) / B                        # [C,H]
            grad_b2 = delta_out.mean(dim=0)                                  # [C]
            model.fc2.weight.add_(-lr_w2 * grad_W2)
            model.fc2.bias  .add_(-lr_w2 * grad_b2)

        # Logging
        epoch_loss += loss.item()
        epoch_acc  += (out.argmax(1) == labels).float().mean().item()

    print(f"Epoch {epoch+1}/{num_epochs}  "
          f"Loss: {epoch_loss/len(train_loader):.3f}  "
          f"Acc: {100*epoch_acc/len(train_loader):.2f}%")

# ─── Evaluation ───────────────────────────────────────────────────────────────
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series, labels = series.to(device), labels.to(device)
        model.reset()
        spk = None
        for t in range(series_len):
            x_t = series[:, t, :]
            cur = cur_scale * model.fc1(x_t)
            spk, mem = model.lif1(cur)
        logits = model.fc2(spk)                        # last-timestep spikes
        preds = logits.argmax(dim=1)
        test_acc += (preds == labels).float().mean().item()

test_acc = 100.0 * test_acc / len(test_loader)
print(f"\nTest Accuracy: {test_acc:.2f}%")
