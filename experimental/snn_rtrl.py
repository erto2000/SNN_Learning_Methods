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
beta         = 0.9
spike_grad   = surrogate.fast_sigmoid(slope=25)
lr           = 0.01
dataset_name = "ECG5000"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
ucr = UCR_UEA_datasets()
X_tr, y_tr, X_te, y_te = ucr.load_dataset(dataset_name)
y_tr -= 1; y_te -= 1
T = X_tr.shape[1]
n_classes = len(torch.unique(torch.tensor(y_tr)))

class TS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()  # [N,T,1]
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(TS(X_tr,y_tr), batch_size, shuffle=True)
test_loader  = DataLoader(TS(X_te,y_te), batch_size, shuffle=False)

# ─── MODEL ───────────────────────────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, H, C):
        super().__init__()
        self.fc1  = nn.Linear(1, H)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2  = nn.Linear(H, C)

    def reset(self):
        utils.reset(self.lif1)

model = SNN(hidden_size, n_classes).to(device)

# ─── TRAIN: MANUAL RTRL ───────────────────────────────────────────────────────
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss    = 0.0
    train_correct = 0
    train_total   = 0

    for X, y in train_loader:
        X = X.to(device)    # [B, T, 1]
        y = y.to(device)    # [B]
        B = X.size(0)

        # reset state & sensitivities
        model.reset()
        # P_w1: dV/dW1 for fc1.weight, shape [B, H, 1]
        P_w1 = torch.zeros(B, hidden_size, 1, device=device)
        # P_b1: dV/db1 for bias, shape [B, H]
        P_b1 = torch.zeros(B, hidden_size,   device=device)

        # buffers for RTRL‐chain rule
        P_w1_list, P_b1_list = [], []
        ds_dV_list  = []
        spk_list    = []

        # forward in time, accumulate sensitivities
        V_prev = torch.zeros(B, hidden_size, device=device)
        s_prev = torch.zeros(B, hidden_size, device=device)

        for t in range(T):
            x_t = X[:, t, :]                # [B,1]
            net1 = model.fc1(x_t)           # [B,H]

            # RTRL sensitivity update for W1,b1:
            #  V_t = β(1−s_{t−1}) V_{t−1} + net1
            # ⇒ dV_t/dW1 = β(1−s_prev)⋅P_w1 + d net1/dW1
            P_w1 = beta * (1 - s_prev).unsqueeze(-1) * P_w1 + x_t.unsqueeze(1)  # [B,H,1]
            P_b1 = beta * (1 - s_prev) * P_b1 + 1.0  # [B,H]

            # membrane + spike
            V_t   = beta * (1 - s_prev) * V_prev + net1
            ds_dV = spike_grad(V_t - 1.0)  # surrogate′(V−θ)
            s_t   = (V_t >= 1.0).float()

            # store for gradient‐chain
            P_w1_list.append(P_w1)
            P_b1_list.append(P_b1)
            ds_dV_list.append(ds_dV)
            spk_list.append(s_t)

            # update for next step
            V_prev, s_prev = V_t, s_t

        # sum spikes → logits → loss
        H_sum = torch.stack(spk_list, dim=0).sum(0)  # [B,H]
        logits = model.fc2(H_sum)                    # [B,C]
        loss   = F.cross_entropy(logits, y)
        total_loss += loss.item()

        # compute training accuracy for this batch
        preds = logits.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total   += B

        # compute error signal e = p−onehot
        p      = F.softmax(logits, dim=1)
        onehot = F.one_hot(y, n_classes).float()
        e      = (p - onehot)                   # [B,C]

        # GRADIENTS
        # fc2: ∂L/∂W2 = eᵀ @ H_sum / B,  ∂L/∂b2 = e.mean(0)
        dW2 = (e.t() @ H_sum) / B
        db2 = e.mean(0)

        # fc1: chain‐rule over t
        dW1 = torch.zeros_like(model.fc1.weight)  # [H,1]
        db1 = torch.zeros_like(model.fc1.bias)    # [H]
        W2  = model.fc2.weight                    # [C,H]


        for t in range(T):
            # error‐signal on hidden spikes: (e @ W2) ⊙ ds/dV_t
            err_t = (e @ W2) * ds_dV_list[t]  # [B,H]

            # dW1: [H,1] ← (err_t.unsqueeze(-1) * P_w1_list[t]).sum(0)/B
            dW1 += (err_t.unsqueeze(-1) * P_w1_list[t]).sum(0) / B

            # db1 stays the same
            db1 += (err_t * P_b1_list[t]).sum(0) / B

        # MANUAL SGD STEP
        model.fc1.weight.data -= lr * dW1
        model.fc1.bias.data   -= lr * db1
        model.fc2.weight.data -= lr * dW2
        model.fc2.bias.data   -= lr * db2

    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    print(f"Epoch {epoch:02d}/{num_epochs}  "
          f"Loss: {avg_loss:.4f}  "
          f"Train Acc: {train_acc:.2f}%")

# ─── EVALUATION ────────────────────────────────────────────────────────────────
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        model.reset()
        spikes = []
        for t in range(T):
            spk = model.lif1(model.fc1(X[:,t,:]))
            spikes.append(spk)
        H_sum = torch.stack(spikes,0).sum(0)
        preds = model.fc2(H_sum).argmax(1)
        correct += (preds==y).sum().item()
        total   += y.size(0)

print(f"Test Acc: {100*correct/total:.2f}%")
