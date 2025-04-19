import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate, utils
import torch.nn.functional as F
from tslearn.datasets import UCR_UEA_datasets

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
num_epochs   = 20
batch_size   = 128
hidden_size  = 128
beta         = 0.9          # decay for eligibility traces
spike_grad   = surrogate.fast_sigmoid(slope=25)
dataset_name = "ECG5000"
lr           = 0.01         # learning rate

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
        self.X = torch.from_numpy(X).float()  # [N, T, 1]
        self.y = torch.from_numpy(y).long()                # [N]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train,y_train),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TimeSeriesDataset(X_test, y_test),
                          batch_size=batch_size, shuffle=False)

# ─── SNN DEFINITION ────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, beta, spike_grad):
        super().__init__()
        self.fc1  = nn.Linear(1, hidden_dim, bias=True)
        self.lif1 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              init_hidden=True)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
    def reset(self):
        utils.reset(self.lif1)

model = SNN(hidden_dim=hidden_size,
            output_dim=n_classes,
            beta=beta,
            spike_grad=spike_grad).to(device)

# ─── TRAINING LOOP ────────────────────────────────
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_acc  = 0.0

    for series, labels in train_loader:
        series, labels = series.to(device), labels.to(device)
        bsz = series.size(0)

        # 1) reset state & eligibility
        model.reset()
        # eligibility for W1 and b1
        e_trace1 = torch.zeros_like(model.fc1.weight, device=device)
        # eligibility for W2 and b2 (only depends on final spike)
        # we'll accumulate a single-step eligibility after the last time-step
        e_trace2 = None

        # 2) forward through time, detach spikes for eligibility
        for t in range(series_len):
            x_t = series[:, t, :]             # [B, 1]
            cur = model.fc1(x_t)              # [B, H]
            spk = model.lif1(cur)             # [B, H]
            spk_det = spk.detach()

            # 2a) eligibility for layer1: β e + (spk ⊺ x) / B
            e_trace1.mul_(beta).add_(spk_det.t() @ x_t / bsz)

            if t == series_len - 1:
                final_spk = spk_det           # freeze for read‑out
                # build one‑step eligibility for layer2: final_spkᵀ has shape [H, B]
                # so e_trace2 will be [C, H] after multiplying with one‑hot?
                # we'll do that in the update step

        # 3) compute read‑out by hand (still a linear layer)
        #    out = final_spk @ W2ᵀ + b2
        W2, b2 = model.fc2.weight, model.fc2.bias
        out = F.linear(final_spk, W2, b2)    # [B, C]

        # 4) compute cross‐entropy loss *value* for logging
        loss = F.cross_entropy(out, labels)

        # 5) compute modulatory δ from the read‑out
        with torch.no_grad():
            # a) softmax + one‑hot error
            probs = F.softmax(out, dim=1)            # [B, C]
            δ_out = probs
            δ_out[torch.arange(bsz), labels] -= 1    # [B, C]

            # b) hidden error for layer1
            δ_hidden = δ_out @ W2                    # [B, H]
            δ_sum1 = δ_hidden.sum(0).unsqueeze(1) / bsz  # [H, 1]

            # c) manual update W1:
            #    ΔW1 = -lr * (δ_sum1 ⊙ e_trace1)
            model.fc1.weight.add_(-lr * (δ_sum1 * e_trace1))
            if model.fc1.bias is not None:
                # bias eligibility for LIF is just sum of δ_sum1 across hidden units
                # here we use the same δ_sum1 but collapsed
                model.fc1.bias.add_(-lr * δ_sum1.squeeze())

            # d) manual update W2 & b2:
            #    ∂L/∂W2 = δ_outᵀ @ final_spk / B
            grad_W2 = (δ_out.t() @ final_spk) / bsz    # [C, H]
            model.fc2.weight.add_(-lr * grad_W2)
            # bias gradient is the mean of δ_out over batch
            grad_b2 = δ_out.mean(dim=0)               # [C]
            model.fc2.bias.add_(-lr * grad_b2)

        # 6) logging
        epoch_loss += loss.item()
        epoch_acc  += (out.argmax(1) == labels).float().mean().item()

    print(f"Epoch {epoch+1}/{num_epochs}  "
          f"Loss: {epoch_loss/len(train_loader):.3f}  "
          f"Acc: {100*epoch_acc/len(train_loader):.2f}%")

# ─── FINAL EVALUATION ───────────────────────────────────────────────────────────
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series, labels = series.to(device), labels.to(device)
        model.reset()
        for t in range(series_len):
            x_t = series[:, t, :]
            h_t = model.lif1(model.fc1(x_t))
        out = model.fc2(h_t)
        preds = out.argmax(dim=1)
        test_acc += (preds == labels).float().mean().item()

test_acc = 100.0 * test_acc / len(test_loader)
print(f"\nTest Accuracy: {test_acc:.2f}%")
