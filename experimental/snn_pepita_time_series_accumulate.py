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
beta           = 0.9
spike_grad     = surrogate.fast_sigmoid(slope=25)
dataset_name   = "ECG5000"
lr             = 0.01     # learning rate for manual updates
f_factor       = 0.5     # error‐to‐input scaling
hidden_dim     = 128

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
        self.X = torch.from_numpy(X).float()  # [N, T]
        self.y = torch.from_numpy(y).long()   # [N]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        # return [T,1] sequence, label
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

model = SNN(hidden_dim=hidden_dim,
            output_dim=n_classes,
            beta=beta,
            spike_grad=spike_grad).to(device)

# ─── PROJECTION MATRIX ─────────────────────────────────────────────────────────
projection = (torch.rand(n_classes, 1, device=device) * f_factor)

# ─── TRAINING LOOP (Approximate single‐difference update) ─────────────────────
for epoch in range(1, num_epochs + 1):
    model.train()
    running_acc = 0.0

    for series, labels in train_loader:
        # series: [B, T, 1], labels: [B]
        series = series.to(device)
        labels = labels.to(device)
        B, T, _ = series.shape

        # ─ First Pass: accumulate outputs & hidden sums ────────────────────────
        model.reset()
        rates        = torch.zeros(B, n_classes, device=device)
        h_norm_sum   = torch.zeros(B, hidden_dim, device=device)

        for t in range(T):
            x_t    = series[:, t, :]               # [B,1]
            h_t    = model.lif1(model.fc1(x_t))    # [B,H]
            o_t    = model.fc2(h_t)                # [B,C]
            rates  += o_t
            h_norm_sum += h_t

        # Softmax & error
        p       = F.softmax(rates, dim=1)                 # [B,C]
        onehot  = F.one_hot(labels, n_classes).float()    # [B,C]
        e       = p - onehot                              # [B,C]

        # Modulate inputs
        proj_err    = e @ projection                       # [B,1]
        series_mod  = series + proj_err.unsqueeze(1)       # [B,T,1]

        # ─ Second Pass: accumulate modulated hidden & input sums ─────────────
        model.reset()
        h_mod_sum = torch.zeros(B, hidden_dim, device=device)
        x_mod_sum = torch.zeros(B, 1, device=device)

        for t in range(T):
            x_mod = series_mod[:, t, :]                   # [B,1]
            h_mod = model.lif1(model.fc1(x_mod))          # [B,H]
            h_mod_sum += h_mod
            x_mod_sum += x_mod

        # ─ Compute per‐time averages ───────────────────────────────────────────
        h_norm_avg = h_norm_sum / T                       # [B,H]
        h_mod_avg  = h_mod_sum  / T                       # [B,H]
        x_mod_avg  = x_mod_sum  / T                       # [B,1]

        # ─ Manual Weight Updates ──────────────────────────────────────────────
        # -- fc1.weight: approximate product‐of‐means update
        #    δw1 ∝ - (mean h_norm - mean h_mod)ᵀ @ mean x_mod
        delta_w1 = - (h_norm_avg - h_mod_avg).t() @ x_mod_avg / B  # [H,1]
        model.fc1.weight.data += lr * delta_w1

        # -- fc2.weight: same as before (exact)
        delta_w2 = - (e.t() @ h_mod_avg) / B                     # [C,H]
        model.fc2.weight.data += lr * delta_w2

        # ─ Accumulate Training Accuracy ─────────────────────────────────────
        preds = p.argmax(dim=1)
        running_acc += (preds == labels).float().mean().item()

    train_acc = 100.0 * running_acc / len(train_loader)
    print(f"Epoch {epoch:02d}/{num_epochs}, Train Acc: {train_acc:.2f}%")

# ─── FINAL EVALUATION ─────────────────────────────────────────────────────────
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series, labels = series.to(device), labels.to(device)
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
