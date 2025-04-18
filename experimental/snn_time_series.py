import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import snntorch as snn
from snntorch import utils
from snntorch import surrogate

from tslearn.datasets import UCR_UEA_datasets

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
num_epochs   = 10
batch_size   = 128
beta         = 0.9
spike_grad   = surrogate.fast_sigmoid(slope=25)
dataset_name = "ECG5000"   # UCR archive
# ────────────────────────────────────────────────────────────────────────────────

# ─── DEVICE ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD UCR DATA ─────────────────────────────────────────────────────────────
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)

# zero‑base the labels
y_train = y_train.astype(int) - 1
y_test  = y_test.astype(int)  - 1

series_len = X_train.shape[1]
n_classes  = len(torch.unique(torch.tensor(y_train)))

# ─── CUSTOM DATASET ───────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X: [N, T]
        self.X = torch.from_numpy(X).float()  # [N, T, 1]
        self.y = torch.from_numpy(y).long()                # [N]
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
    def __init__(self, input_dim, hidden_dim, output_dim, time_steps, beta, spike_grad):
        super().__init__()
        self.time_steps = time_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        # data: [B, T, 1]
        utils.reset(self.net)
        spk_rec = []
        for t in range(self.time_steps):
            x_t = data[:, t, :]         # [B, 1]
            spk = self.net(x_t)         # [B, n_classes]
            spk_rec.append(spk)
        return torch.stack(spk_rec, dim=0)  # [T, B, n_classes]

model = SNN(
    input_dim=1,
    hidden_dim=128,
    output_dim=n_classes,
    time_steps=series_len,
    beta=beta,
    spike_grad=spike_grad
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

# ─── TRAIN & EVAL ──────────────────────────────────────────────────────────────
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    running_acc  = 0.0

    for series, labels in train_loader:
        series, labels = series.to(device), labels.to(device)

        optimizer.zero_grad()
        spk_rec = model(series)               # [T, B, C]
        rates   = spk_rec.sum(dim=0)          # [B, C]

        loss = loss_fn(rates, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = rates.argmax(dim=1)
        running_acc += (preds == labels).float().mean().item()

    avg_loss = running_loss / len(train_loader)
    avg_acc  = 100.0 * running_acc / len(train_loader)
    print(f"Epoch {epoch:02d}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}%")

# final test
model.eval()
test_acc = 0.0
with torch.no_grad():
    for series, labels in test_loader:
        series, labels = series.to(device), labels.to(device)
        rates = model(series).sum(dim=0)
        preds = rates.argmax(dim=1)
        test_acc += (preds == labels).float().mean().item()

test_acc = 100.0 * test_acc / len(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
