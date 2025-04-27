import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Lambda, Compose
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange
import matplotlib.pyplot as plt

# ----------------------------
#  Hyperparameters & Architecture
# ----------------------------
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE       = 4096
LR               = 1e-3
EPOCHS_PER_LAYER = 100
TIME_STEPS       = 10
ALPHA            = 0.6
BETA             = 0.99

# dims = [input_dim, hidden1, hidden2, ...]
# Here: 784 → 500 → 500
dims = [784, 500, 500]
# ----------------------------

# Data transforms
transform = Compose([
    ToTensor(),
    Normalize((0.0,), (1.0,)),
    Lambda(lambda x: x.view(-1))
])

train_ds = MNIST('../data', train=True,  download=True, transform=transform)
test_ds  = MNIST('../data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

def overlay_y_on_x(x, y):
    """
    Zero out first 10 dims of x, then set the y-th to max(x).
    x: [batch, 784], y: [batch] in {0..9}
    """
    x_ = x.clone()
    x_[:, :10] = 0.0
    x_[torch.arange(x.size(0)), y] = x.max()
    return x_

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
        batch = x.size(0)
        # initialize membrane potential
        mem = self.lif.init_leaky()
        spike_count = torch.zeros(batch, self.fc.out_features, device=x.device)

        # normalize to unit norm (orientation only)
        cur = self.fc(x)
        cur = cur / (cur.norm(p=2, dim=1, keepdim=True) + 1e-4) * 10
        for _ in range(self.T):
            spk, mem = self.lif(cur, mem)
            spike_count += spk

        return spike_count

    def train_layer(self, x_pos, x_neg, layer_idx=None):
        """
        Trains this layer with positive and negative samples,
        showing a tqdm progress bar over epochs.
        """
        if layer_idx is None:
            desc = f"Training layer [{self.fc.in_features}→{self.fc.out_features}]"
        else:
            desc = f"Layer {layer_idx} [{self.fc.in_features}→{self.fc.out_features}]"

        losses = []
        for _ in trange(self.epochs, desc=desc, leave=True):
            spk_pos = self.forward(x_pos)
            spk_neg = self.forward(x_neg)

            Gpos = (spk_pos ** 2).mean(dim=1)
            Gneg = (spk_neg ** 2).mean(dim=1)
            delta = Gpos - Gneg

            loss = -(self.alpha * delta / (1 + torch.exp(self.alpha * delta))).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())

        # detach so next layer sees only the spikes
        return (spk_pos.detach(), spk_neg.detach()), losses

class Net(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            LeakyLayer(dims[i], dims[i+1])
            for i in range(len(dims)-1)
        ])

    def train(self, x, y):
        # positive samples
        x_pos = overlay_y_on_x(x, y)
        # random negatives
        rnd   = torch.randperm(x.size(0), device=x.device)
        x_neg = overlay_y_on_x(x, y[rnd])

        h_pos, h_neg = x_pos, x_neg
        all_losses = []
        for idx, layer in enumerate(self.layers, start=1):
            print(f"\n⏳ Beginning training of layer {idx}/{len(self.layers)}")
            (h_pos, h_neg), losses = layer.train_layer(h_pos, h_neg, layer_idx=idx)
            all_losses.append(losses)
        return all_losses

    @torch.no_grad()
    def predict(self, x):
        batch = x.size(0)
        goodness = torch.zeros(batch, 10, device=x.device)

        for lbl in range(10):
            x_lbl = overlay_y_on_x(x, torch.full((batch,), lbl, device=x.device))
            total = 0
            h = x_lbl
            for layer in self.layers:
                spk = layer.forward(h)
                total += (spk**2).mean(dim=1)
                h = spk
            goodness[:, lbl] = total

        return goodness.argmax(dim=1)

if __name__ == "__main__":
    torch.manual_seed(123)
    net = Net(dims).to(DEVICE)

    # Demo on one batch
    x, y = next(iter(train_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)

    print("⏳ Training layer-wise (this will take a minute)...")
    losses = net.train(x, y)

    # Check shapes to confirm no mismatch
    # print(" First layer weight:", net.layers[0].fc.weight.shape)
    # print("Second layer weight:", net.layers[1].fc.weight.shape)
    # print(" Input to second layer:", losses and losses[0][0])

    # Train & test accuracy on that batch
    with torch.no_grad():
        train_pred = net.predict(x)
        train_acc  = 100*(train_pred==y).float().mean().item()
        xt, yt     = next(iter(test_loader))
        xt, yt     = xt.to(DEVICE), yt.to(DEVICE)
        test_pred  = net.predict(xt)
        test_acc   = 100*(test_pred==yt).float().mean().item()

    print(f"Train acc (batch): {train_acc:.2f}%")
    print(f" Test acc (batch): {test_acc:.2f}%")

    # Plot per-layer loss curves
    for idx, layer_losses in enumerate(losses, start=1):
        plt.plot(layer_losses, label=f"Layer {idx}")
    plt.xlabel("Epoch")
    plt.ylabel("Local Loss")
    plt.legend()
    plt.show()
