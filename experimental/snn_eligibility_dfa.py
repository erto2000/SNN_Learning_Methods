import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate, utils
from tslearn.datasets import UCR_UEA_datasets
import torch.nn.functional as F

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
num_epochs   = 10
batch_size   = 128
hidden_size  = 128
beta         = 0.9
spike_grad   = surrogate.fast_sigmoid(slope=25)
dataset_name = "ECG5000"
lr           = 0.01   # learning rate
f_factor     = 0.5    # scale for random feedback

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
        self.X = torch.from_numpy(X).float()  # [N, T, 1]
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train,y_train),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TimeSeriesDataset(X_test, y_test),
                          batch_size=batch_size, shuffle=False)

# ─── SNN DEFINITION ────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, beta, spike_grad):
        super().__init__()
        self.fc1  = nn.Linear(1, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              init_hidden=True)
        self.fc2  = nn.Linear(hidden_dim, output_dim)

    def reset(self):
        utils.reset(self.lif1)

model = SNN(hidden_dim=hidden_size,
            output_dim=n_classes,
            beta=beta,
            spike_grad=spike_grad).to(device)

# ─── RANDOM FEEDBACK ──────────────────────────────────────────────────────────
# maps output‐error → hidden‐learning‐signal
feedback = torch.randn(n_classes, hidden_size, device=device) * f_factor

# ─── TRAINING LOOP (e‑prop) ────────────────────────────────────────────────────
for epoch in range(1, num_epochs + 1):
    model.train()
    running_acc = 0.0

    for series, labels in train_loader:
        series = series.to(device)          # [B, T, 1]
        labels = labels.to(device)          # [B]
        B = series.size(0)

        # Reset states and traces
        model.reset()
        elig_h = torch.zeros(B, hidden_size, device=device)  # [B, H]
        elig_o = torch.zeros(B, hidden_size, device=device)  # [B, H]
        out_rec = []

        # ── Single Forward Pass ───────────────────────────────────────────────
        for t in range(series_len):
            x_t = series[:, t, :]                    # [B, 1]
            h_t = model.lif1(model.fc1(x_t))         # [B, H]
            mem = model.lif1.mem                      # [B, H]

            # Eligibility for fc1: e_h[t] = surrogate'(mem-θ) * x_t
            grad_sur = spike_grad(mem - model.lif1.threshold)  # [B, H]
            elig_h += grad_sur * x_t                        # [B, H]

            # Eligibility for fc2: simply sum hidden spikes
            elig_o += h_t                                   # [B, H]

            # Record output pre‐activations
            out_rec.append(model.fc2(h_t))                  # [B, C]

        # Compute rates, probabilities, error
        rates = torch.stack(out_rec, dim=0).sum(dim=0)      # [B, C]
        p     = F.softmax(rates, dim=1)                     # [B, C]
        onehot= F.one_hot(labels, n_classes).float()        # [B, C]
        e     = p - onehot                                 # [B, C]

        # ── Weight Updates ────────────────────────────────────────────────────
        # -- fc2 weights (hidden→output)
        #    ΔW2[k,j] ∝ −∑_i e[i,k] * elig_o[i,j]
        grad_w2 = (e.transpose(0,1) @ elig_o) / B           # [C, H]
        model.fc2.weight.data += -lr * grad_w2

        # -- fc1 weights (input→hidden)
        #    first compute per‐sample learning signal for hidden
        L_hidden = e @ feedback                             # [B, H]
        #    then ΔW1[j] ∝ −∑_i L_hidden[i,j] * elig_h[i,j]
        grad_w1 = (L_hidden * elig_h).sum(dim=0) / B         # [H]
        model.fc1.weight.data += -lr * grad_w1.unsqueeze(1)

        # ── Accumulate Training Accuracy ───────────────────────────────────
        preds = p.argmax(dim=1)
        running_acc += (preds == labels).float().mean().item()

    train_acc = 100.0 * running_acc / len(train_loader)
    print(f"Epoch {epoch:02d}/{num_epochs}, Train Acc: {train_acc:.2f}%")

# ─── FINAL EVALUATION ─────────────────────────────────────────────────────────
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series = series.to(device)
        labels = labels.to(device)

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
