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
BATCH_SIZE       = 128
LR               = 1e-3
EPOCHS_PER_LAYER = 10
TIME_STEPS       = 10
ALPHA            = 0.6
BETA             = 0.9

dims = [784, 128]
# ----------------------------

transform = Compose([
    ToTensor(),
    Normalize((0.0,), (1.0,)),
    Lambda(lambda x: x.view(-1))
])

train_ds = MNIST('../../data', train=True, download=True, transform=transform)
test_ds  = MNIST('../../data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

def overlay_y_on_x(x, y):
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
        mem = self.lif.init_leaky()
        spike_count = torch.zeros(batch, self.fc.out_features, device=x.device)
        cur = self.fc(x)
        cur = cur / (cur.norm(p=2, dim=1, keepdim=True) + 1e-4) * 10
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

def pretrain_layers(net, train_loader):
    """
    Greedy layerwise pretraining over the entire dataset.
    Returns a list of per-layer loss histories: [[epoch1_loss, ...], ...].
    """
    all_losses = []
    for idx, layer in enumerate(net.layers, start=1):
        print(f"\n⏳ Pre-training Layer {idx}/{len(net.layers)}: "
              f"{layer.fc.in_features}→{layer.fc.out_features}")
        layer_losses = []
        for epoch in range(1, layer.epochs+1):
            running_loss = 0.0
            count = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                # build positive/negative inputs
                x_pos = overlay_y_on_x(x, y)
                rnd   = torch.randperm(x.size(0), device=DEVICE)
                x_neg = overlay_y_on_x(x, y[rnd])
                # feed through previous layers
                h_pos, h_neg = x_pos, x_neg
                if idx > 1:
                    with torch.no_grad():
                        for prev in net.layers[:idx-1]:
                            h_pos = prev.forward(h_pos)
                            h_neg = prev.forward(h_neg)
                # forward current layer
                spk_pos = layer.forward(h_pos)
                spk_neg = layer.forward(h_neg)
                Gpos = (spk_pos ** 2).mean(dim=1)
                Gneg = (spk_neg ** 2).mean(dim=1)
                delta = Gpos - Gneg
                loss = -(layer.alpha * delta / (1 + torch.exp(layer.alpha * delta))).mean()
                # step
                layer.opt.zero_grad()
                loss.backward()
                layer.opt.step()
                running_loss += loss.item() * x.size(0)
                count += x.size(0)
            epoch_loss = running_loss / count
            layer_losses.append(epoch_loss)
            print(f"  Layer {idx} Epoch {epoch}/{layer.epochs} — loss: {epoch_loss:.4f}")
        all_losses.append(layer_losses)
    return all_losses

if __name__ == "__main__":
    torch.manual_seed(123)
    net = Net(dims).to(DEVICE)

    print("⏳ Starting full-dataset, layerwise pre-training…")
    losses = pretrain_layers(net, train_loader)

    # Evaluate on the full training set
    net.eval()
    def evaluate(loader):
        correct = total = 0
        for xt, yt in loader:
            xt, yt = xt.to(DEVICE), yt.to(DEVICE)
            pred = net.predict(xt)
            correct += (pred == yt).sum().item()
            total += yt.size(0)
        return 100 * correct / total

    train_acc = evaluate(train_loader)
    test_acc  = evaluate(test_loader)
    print(f"\n▶️  Final Train Accuracy: {train_acc:.2f}%")
    print(f"▶️   Final Test Accuracy: {test_acc:.2f}%")

    # Plot per‐layer losses
    for idx, layer_losses in enumerate(losses, start=1):
        plt.plot(layer_losses, label=f"Layer {idx}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.show()
