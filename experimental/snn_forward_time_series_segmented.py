import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tslearn.datasets import UCR_UEA_datasets

# ----------------------------
#  Hyperparameters & Architecture
# ----------------------------
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE       = 128
LR               = 1e-3
EPOCHS_PER_LAYER = 10
TIME_STEPS       = 10    # number of SNN time‐steps = window length
ALPHA            = 0.6
BETA             = 0.9

SEG_LEN   = 10          # raw window length
NUM_WINS  = 140 // 10   # 14 windows per series
NUM_CLSES = 5           # ECG5000 has 5 classes
dims = [SEG_LEN + NUM_CLSES, 128, 128]
# ----------------------------

# 1) Load the UCR ECG5000 data and fix shapes
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("ECG5000")

# squeeze away the singleton feature dimension → (n_samples, 140)
X_train = np.squeeze(X_train)
X_test  = np.squeeze(X_test)

# convert labels from 1…5 → 0…4
y_train = (np.array(y_train).astype(int) - 1)
y_test  = (np.array(y_test).astype(int) - 1)

# 2) Build a Dataset that chops into windows of length SEG_LEN
class ECGWindowDataset(Dataset):
    def __init__(self, X, y, seg_len=SEG_LEN):
        n, L = X.shape
        assert L % seg_len == 0, "Series length must be multiple of segment length"
        # reshape (n, L) → (n, L/seg_len, seg_len) → merge to (n * (L/seg_len), seg_len)
        self.segments = X.reshape(n, L//seg_len, seg_len).reshape(-1, seg_len)
        # repeat each label once per window
        self.labels   = np.repeat(y, L//seg_len)
        # torch tensors
        self.segments = torch.from_numpy(self.segments).float()
        self.labels   = torch.from_numpy(self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

train_ds = ECGWindowDataset(X_train, y_train)
test_ds  = ECGWindowDataset(X_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# 3) Function to append a one-hot label spike to each segment
def overlay_label_on_segment(x, y, num_classes=NUM_CLSES):
    """
    x: [batch, SEG_LEN]
    y: [batch] in {0,…,num_classes-1}
    returns [batch, SEG_LEN + num_classes]
    """
    batch = x.size(0)
    out = torch.zeros(batch, x.size(1) + num_classes, device=x.device)
    out[:, :x.size(1)] = x
    # use max amplitude in the window as spike height
    max_val = x.max(dim=1, keepdim=True)[0]
    # scatter the spike for each label
    out.scatter_(1, y.view(-1,1) + x.size(1), max_val)
    return out

# 4) Define your LeakyLayer and Net
class LeakyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc    = nn.Linear(in_features, out_features, bias=False)
        self.lif   = snn.Leaky(beta=BETA, spike_grad=surrogate.atan())
        self.opt   = Adam(self.parameters(), lr=LR)
        self.T     = TIME_STEPS
        self.alpha = ALPHA
        self.epochs = EPOCHS_PER_LAYER

    def forward(self, x):
        # x: [batch, in_features]
        mem = self.lif.init_leaky()
        cur = self.fc(x)
        # normalize currents
        cur = cur / (cur.norm(p=2, dim=1, keepdim=True) + 1e-4) * 10
        spike_count = torch.zeros_like(cur)
        for _ in range(self.T):
            spk, mem = self.lif(cur, mem)
            spike_count += spk
        return spike_count

class Net(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            LeakyLayer(dims[i], dims[i+1])
            for i in range(len(dims)-1)
        ])

    @torch.no_grad()
    def predict(self, X):
        """
        X: [batch, 140] full series
        Returns: [batch] predicted labels 0…4
        """
        batch = X.size(0)
        # reshape to windows → [batch, 14, SEG_LEN]
        wins = X.view(batch, NUM_WINS, SEG_LEN)
        goodness = torch.zeros(batch, NUM_CLSES, device=X.device)

        for lbl in range(NUM_CLSES):
            total_lbl = torch.zeros(batch, device=X.device)
            for w in range(NUM_WINS):
                seg = wins[:, w, :]                     # [batch, SEG_LEN]
                inp = overlay_label_on_segment(seg,
                                               torch.full((batch,), lbl, device=X.device))
                h = inp
                for layer in self.layers:
                    spk = layer(h)
                    total_lbl += (spk**2).mean(dim=1)
                    h = spk
            goodness[:, lbl] = total_lbl

        return goodness.argmax(dim=1)

def pretrain_layers(net, train_loader):
    all_losses = []
    for idx, layer in enumerate(net.layers, start=1):
        print(f"\n⏳ Pre-training Layer {idx}/{len(net.layers)}: "
              f"{layer.fc.in_features}→{layer.fc.out_features}")
        layer_losses = []
        for epoch in range(1, layer.epochs+1):
            running_loss = 0.0
            count = 0
            for x_seg, y in train_loader:
                x_seg, y = x_seg.to(DEVICE), y.to(DEVICE)

                # create positive & negative inputs
                x_pos = overlay_label_on_segment(x_seg, y)
                rnd   = torch.randperm(x_seg.size(0), device=DEVICE)
                x_neg = overlay_label_on_segment(x_seg, y[rnd])

                # if not the first layer, run through previous ones
                if idx > 1:
                    with torch.no_grad():
                        h_pos, h_neg = x_pos, x_neg
                        for prev in net.layers[:idx-1]:
                            h_pos = prev(h_pos)
                            h_neg = prev(h_neg)
                else:
                    h_pos, h_neg = x_pos, x_neg

                spk_pos = layer(h_pos)
                spk_neg = layer(h_neg)
                Gpos = (spk_pos ** 2).mean(dim=1)
                Gneg = (spk_neg ** 2).mean(dim=1)
                δ    = Gpos - Gneg
                loss = -(layer.alpha * δ / (1 + torch.exp(layer.alpha * δ))).mean()

                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()

                running_loss += loss.item() * x_seg.size(0)
                count += x_seg.size(0)

            epoch_loss = running_loss / count
            layer_losses.append(epoch_loss)
            print(f"  Layer {idx} Epoch {epoch}/{layer.epochs} — loss: {epoch_loss:.4f}")
        all_losses.append(layer_losses)
    return all_losses

# 5) Training & Evaluation
if __name__ == "__main__":
    torch.manual_seed(0)
    net = Net(dims).to(DEVICE)

    print("⏳ Starting full-dataset, layerwise pre-training…")
    losses = pretrain_layers(net, train_loader)

    # Evaluate on full-series test set
    net.eval()
    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_t = torch.from_numpy(y_test).long().to(DEVICE)
    preds = net.predict(X_test_t)
    test_acc = 100 * (preds == y_test_t).float().mean().item()
    print(f"\n▶️  Final Test Accuracy on full 140-step series: {test_acc:.2f}%")
