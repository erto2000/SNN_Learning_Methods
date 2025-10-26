import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import snntorch as snn
from snntorch import surrogate, utils

from tslearn.datasets import UCR_UEA_datasets

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
num_epochs   = 20
batch_size   = 128
hidden_size  = 128
beta         = 0.9                       # decay for eligibility traces
spike_grad   = surrogate.fast_sigmoid(slope=25)
dataset_name = "ECG5000"
lr           = 0.01                     # learning rate
f_factor     = 0.005                  # error‐to‐input gain

# ─── DEVICE ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD UCR DATA ─────────────────────────────────────────────────────────────
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)
y_train = y_train.astype(int) - 1
y_test  = y_test.astype(int)  - 1

series_len = X_train.shape[1]
n_classes  = len(torch.unique(torch.tensor(y_train)))

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X: [N, T]
        self.X = torch.from_numpy(X).float()  # [N, T, 1]
        self.y = torch.from_numpy(y).long()                # [N]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TimeSeriesDataset(X_test,  y_test),
                          batch_size=batch_size, shuffle=False)

# ─── SNN DEFINITION ────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, beta, spike_grad):
        super().__init__()
        self.fc1  = nn.Linear(1, hidden_dim, bias=True)
        self.lif1 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              init_hidden=True)
        self.fc2  = nn.Linear(hidden_dim, output_dim, bias=True)

    def reset(self):
        utils.reset(self.lif1)

model = SNN(hidden_dim=hidden_size,
            output_dim=n_classes,
            beta=beta,
            spike_grad=spike_grad).to(device)

# ─── PROJECTION MATRIX ─────────────────────────────────────────────────────────
# maps error (n_classes) → scalar input modulation
projection = torch.rand(n_classes, 1, device=device) * f_factor

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_acc = 0.0

    for series, labels in train_loader:
        series, labels = series.to(device), labels.to(device)
        B = series.size(0)

        # ─── PASS 1: NORMAL ────────────────────────────────────────────────────
        model.reset()
        # eligibility trace for fc1: shape [H_out, H_in]
        e_trace1_norm = torch.zeros_like(model.fc1.weight, device=device)
        # accumulate read‐out over time
        o_sum = torch.zeros(B, n_classes, device=device)

        for t in range(series_len):
            x_t = series[:, t, :]               # [B, 1]
            cur = model.fc1(x_t)                # [B, H]
            spk = model.lif1(cur)               # [B, H]
            spk_det = spk.detach()

            # e‑prop update for layer1
            e_trace1_norm.mul_(beta).add_(spk_det.t() @ x_t / B)

            # accumulate output
            o_sum += model.fc2(spk)

            if t == series_len - 1:
                final_spk_norm = spk_det         # for layer2 eligibility

        # compute classification error once
        rates   = o_sum                       # [B, C]
        probs   = F.softmax(rates, dim=1)     # [B, C]
        onehot  = F.one_hot(labels, n_classes).float()
        error   = probs - onehot             # [B, C]

        # ─── MODULATE INPUT ─────────────────────────────────────────────────────
        # project error → scalar correction for each sample
        proj_err   = error @ projection       # [B, 1]
        series_mod = series + proj_err.unsqueeze(1)  # [B, T, 1]

        # ─── PASS 2: MODULATED ─────────────────────────────────────────────────
        model.reset()
        e_trace1_mod = torch.zeros_like(model.fc1.weight, device=device)
        h_mod_sum    = torch.zeros(B, hidden_size, device=device)

        for t in range(series_len):
            x_t = series_mod[:, t, :]
            cur = model.fc1(x_t)
            spk = model.lif1(cur)
            spk_det = spk.detach()

            # eligibility for modulated pass
            e_trace1_mod.mul_(beta).add_(spk_det.t() @ x_t / B)

            # also accumulate hidden spikes to update fc2
            h_mod_sum += spk

            if t == series_len - 1:
                final_spk_mod = spk_det

        h_mod_avg = h_mod_sum / series_len    # [B, H]

        # ─── WEIGHT UPDATES ─────────────────────────────────────────────────────
        with torch.no_grad():
            # -- fc1: difference of eligibility traces
            delta_W1 = (e_trace1_norm - e_trace1_mod)
            model.fc1.weight.add_(-lr * delta_W1)
            if model.fc1.bias is not None:
                # bias‐eligibility is just sum of rows
                delta_b1 = delta_W1.sum(dim=1)
                model.fc1.bias.add_(-lr * delta_b1)

            # -- fc2: same as original PEPITA (only uses modulated hidden avg)
            #    ΔW2 = -lr * (error^T @ h_mod_avg) / B
            grad_W2 = (error.t() @ h_mod_avg) / B  # [C, H]
            model.fc2.weight.add_(-lr * grad_W2)

            # bias update for fc2
            grad_b2 = error.mean(dim=0)            # [C]
            model.fc2.bias.add_(-lr * grad_b2)

        # track accuracy on pass 1
        epoch_acc += (probs.argmax(dim=1) == labels).float().mean().item()

    train_acc = 100.0 * epoch_acc / len(train_loader)
    print(f"Epoch {epoch:02d}/{num_epochs}, Train Acc: {train_acc:.2f}%")

# ─── EVALUATION ────────────────────────────────────────────────────────────────
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series, labels = series.to(device), labels.to(device)
        model.reset()
        out_sum = torch.zeros(series.size(0), n_classes, device=device)

        for t in range(series_len):
            x_t = series[:, t, :]
            spk = model.lif1(model.fc1(x_t))
            out_sum += model.fc2(spk)

        preds = out_sum.argmax(dim=1)
        test_acc += (preds == labels).float().mean().item()

test_acc = 100.0 * test_acc / len(test_loader)
print(f"Test  Acc: {test_acc:.2f}%")
