import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate, utils
from tslearn.datasets import UCR_UEA_datasets
import torch.nn.functional as F


# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
num_epochs     = 20
batch_size     = 128
hidden_size    = 128
beta           = 0.9
spike_grad     = surrogate.fast_sigmoid(slope=25)
dataset_name   = "ECG5000"
lr             = 0.01     # learning rate for manual updates
f_factor       = 0.5     # error‐to‐input scaling


# ─── DEVICE ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── LOAD UCR DATA ─────────────────────────────────────────────────────────────
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)
y_train = y_train.astype(int) - 1
y_test  = y_test.astype(int)  - 1

series_len = X_train.shape[1]
n_classes  = len(torch.unique(torch.tensor(y_train)))


# ─── CUSTOM DATASET ───────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X: [N, T]
        self.X = torch.from_numpy(X).float()  # now [N, T, 1]
        self.y = torch.from_numpy(y).long()   # [N]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train, y_train)
test_ds  = TimeSeriesDataset(X_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)


# ─── SNN DEFINITION ────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, beta, spike_grad):
        super().__init__()
        self.fc1  = nn.Linear(1, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2  = nn.Linear(hidden_dim, output_dim)

    def reset(self):
        utils.reset(self.lif1)

model = SNN(hidden_dim=hidden_size,
            output_dim=n_classes,
            beta=beta,
            spike_grad=spike_grad).to(device)


# ─── PROJECTION MATRIX ─────────────────────────────────────────────────────────
# maps error (n_classes) → scalar input
projection = (torch.rand(n_classes, 1, device=device) * f_factor)


# ─── TRAINING LOOP (Two‑Pass, Manual Updates) ─────────────────────────────────
for epoch in range(1, num_epochs + 1):
    model.train()
    running_acc = 0.0

    for series, labels in train_loader:
        series = series.to(device)      # [B, T, 1]
        labels = labels.to(device)      # [B]
        B = series.size(0)

        # ─ First Pass ─────────────────────────────────────────────────────────────
        h_norm_rec   = []   # will store [T, B, hidden_dim]
        out_norm_rec = []   # will store [T, B, n_classes]
        model.reset()
        for t in range(series_len):
            x_t = series[:, t, :]           # [B, 1]
            h_t = model.lif1(model.fc1(x_t))
            o_t = model.fc2(h_t)
            h_norm_rec.append(h_t)
            out_norm_rec.append(o_t)

        h_norm = torch.stack(h_norm_rec,   dim=0)    # [T, B, H]
        o_norm = torch.stack(out_norm_rec, dim=0)    # [T, B, C]
        rates  = o_norm.sum(dim=0)                   # [B, C]

        # Softmax + error
        p       = F.softmax(rates, dim=1)            # [B, C]
        onehot  = F.one_hot(labels, n_classes).float()  # [B, C]
        e       = p - onehot                         # [B, C]

        # ─ Modulate Input ─────────────────────────────────────────────────────────
        proj_err   = e @ projection                  # [B, 1]
        series_mod = series + proj_err.unsqueeze(1)  # [B, T, 1]

        # ─ Second Pass ────────────────────────────────────────────────────────────
        h_mod_rec = []
        model.reset()
        for t in range(series_len):
            x_t = series_mod[:, t, :]
            h_mod = model.lif1(model.fc1(x_t))
            h_mod_rec.append(h_mod)

        h_mod = torch.stack(h_mod_rec, dim=0)        # [T, B, H]

        # ─ Manual Weight Updates ──────────────────────────────────────────────────
        # -- fc1.weight: correlate hidden‐difference with modulated inputs across t & batch
        diff      = (h_norm - h_mod)                 # [T, B, H]
        x_mod_T   = series_mod.permute(1, 0, 2)      # [T, B, 1]
        mult      = diff * x_mod_T                   # [T, B, H]
        delta_w1  = - mult.sum(dim=(0,1)) / (B * series_len)  # [H]
        model.fc1.weight.data += lr * delta_w1.unsqueeze(1)

        # -- fc2.weight: correlate output error with average modulated hidden spikes
        h_mod_avg = h_mod.sum(dim=0) / series_len    # [B, H]
        delta_w2  = - (e.transpose(0,1) @ h_mod_avg) / B   # [C, H]
        model.fc2.weight.data += lr * delta_w2

        # ─ Accumulate Training Accuracy ───────────────────────────────────────────
        preds = p.argmax(dim=1)
        running_acc += (preds == labels).float().mean().item()

    train_acc = 100.0 * running_acc / len(train_loader)
    print(f"Epoch {epoch:02d}/{num_epochs}, Train Acc: {train_acc:.2f}%")


# ─── FINAL EVALUATION ───────────────────────────────────────────────────────────
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series = series.to(device)
        labels = labels.to(device)

        # simply do one forward (no modulation) and sum spikes
        model.reset()
        out_rec = []
        for t in range(series_len):
            x_t = series[:, t, :]
            h_t = model.lif1(model.fc1(x_t))
            out_rec.append(model.fc2(h_t))
        rates = torch.stack(out_rec, dim=0).sum(dim=0)
        preds = rates.argmax(dim=1)
        test_acc += (preds == labels).float().mean().item()

test_acc = 100.0 * test_acc / len(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
