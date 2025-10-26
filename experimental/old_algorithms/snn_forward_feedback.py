import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Lambda, Compose
from torch.utils.data import DataLoader


def overlay_y_on_x(x, y):
    """
    Zero out first 10 dims of x, then set the y-th to max(x).
    x: [batch, 784], y: [batch] in {0..9}
    """
    x_ = x.clone()
    x_[:, :10] = 0.0
    x_[torch.arange(x.size(0)), y] = x.max()
    return x_


class FeedbackLeakyLayer(nn.Module):
    def __init__(self, in_features, out_features, lr, beta):
        super().__init__()
        # feed‐forward weights
        self.fc_ff = nn.Linear(in_features, out_features, bias=False)
        # feedback weights (maps next‐layer spikes → this layer’s drive)
        self.fc_fb = nn.Linear(out_features, out_features, bias=False)
        # leaky‐LIF neuron
        self.lif   = snn.Leaky(beta=beta, spike_grad=surrogate.atan())
        # optimizer for *both* ff and fb
        self.opt   = torch.optim.Adam(self.parameters(), lr=lr)

    def init_state(self, batch_size, device):
        # initialize membrane for this layer
        return torch.zeros(batch_size, self.fc_ff.out_features, device=device)

class FeedbackFFNet(nn.Module):
    def __init__(self, dims, T=10, α=0.6, β=0.99, η=1.0, lr=1e-3):
        """
        dims: list of layer sizes, e.g. [784, 500, 500]
        T: number of time‐steps to unroll
        α: forward‐forward alpha
        β: LIF leak
        η: feedback gain
        """
        super().__init__()
        self.T     = T
        self.alpha = α
        self.eta   = η

        # build layers
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(
                FeedbackLeakyLayer(
                    in_features  = dims[i],
                    out_features = dims[i+1],
                    lr           = lr,
                    beta         = β
                )
            )

    def forward_run(self, x):
        """
        Run the network unrolled for T time‐steps, with both
        feed‐forward and feedback connections.
        Returns:
            spike_counts: list of length L,
                          each a tensor [batch, layer_size].
        """
        batch = x.size(0)
        device = x.device
        L = len(self.layers)

        # initialize mem & spike‐counts
        mems = [layer.init_state(batch, device) for layer in self.layers]
        spike_counts = [
            torch.zeros(batch, layer.fc_ff.out_features, device=device)
            for layer in self.layers
        ]
        # for feedback we need “previous” spikes; start as zeros
        prev_spks = [
            torch.zeros(batch, layer.fc_ff.out_features, device=device)
            for layer in self.layers
        ]

        for t in range(self.T):
            cur_spks = []
            for ℓ, layer in enumerate(self.layers):
                # feed‐forward drive
                if ℓ == 0:
                    ff_drive = layer.fc_ff(x)
                else:
                    # detach ensures no gradient flows into earlier layers
                    ff_drive = layer.fc_ff(prev_spks[ℓ-1].detach())

                # top‐down feedback (only for ℓ < L-1)
                if ℓ < L-1:
                    fb_drive = layer.fc_fb(prev_spks[ℓ+1].detach())
                    drive = ff_drive + self.eta * fb_drive
                else:
                    drive = ff_drive

                # drive = drive / (drive.norm(p=2, dim=1, keepdim=True) + 1e-4) * 10

                # step the LIF neuron
                spk, mems[ℓ] = layer.lif(drive, mems[ℓ])

                # accumulate
                spike_counts[ℓ] += spk
                cur_spks.append(spk)

            # prepare next time‐step
            prev_spks = cur_spks

        return spike_counts

    def train_step(self, x, y):
        """
        One forward‐forward update over a single batch.
        """
        # positive & negative examples
        x_pos = overlay_y_on_x(x, y)
        rnd   = torch.randperm(x.size(0), device=x.device)
        x_neg = overlay_y_on_x(x, y[rnd])

        # run the whole network (no parameter updates yet)
        spk_pos_list = self.forward_run(x_pos)
        spk_neg_list = self.forward_run(x_neg)

        losses = []
        # compute & apply each layer's local loss
        for layer, spk_pos, spk_neg in zip(self.layers, spk_pos_list, spk_neg_list):
            Gpos  = (spk_pos**2).mean(dim=1)
            Gneg  = (spk_neg**2).mean(dim=1)
            delta = Gpos - Gneg
            loss  = -( self.alpha * delta / (1 + torch.exp(self.alpha * delta)) ).mean()

            layer.opt.zero_grad()
            loss.backward()
            layer.opt.step()

            losses.append(loss.item())

        return losses

    @torch.no_grad()
    def predict(self, x):
        # identical to your original, or you can add feedback here if desired
        batch = x.size(0)
        goodness = torch.zeros(batch, 10, device=x.device)

        for lbl in range(10):
            x_lbl = overlay_y_on_x(x, torch.full((batch,), lbl, device=x.device))
            total = 0
            mems  = [layer.init_state(batch, x.device) for layer in self.layers]
            prev_spks = [torch.zeros_like(m) for m in mems]

            for t in range(self.T):
                cur_spks = []
                for ℓ, layer in enumerate(self.layers):
                    if ℓ == 0:
                        ff_drive = layer.fc_ff(x_lbl)
                    else:
                        ff_drive = layer.fc_ff(prev_spks[ℓ-1])

                    if ℓ < len(self.layers)-1:
                        fb_drive = layer.fc_fb(prev_spks[ℓ+1])
                        drive = ff_drive + self.eta * fb_drive
                    else:
                        drive = ff_drive

                    # drive = drive / (drive.norm(p=2, dim=1, keepdim=True) + 1e-4) * 10

                    spk, mems[ℓ] = layer.lif(drive, mems[ℓ])
                    total += (spk**2).mean(dim=1)
                    cur_spks.append(spk)
                prev_spks = cur_spks

            goodness[:, lbl] = total

        return goodness.argmax(dim=1)


DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE       = 128
LR               = 1e-3
EPOCH            = 10
TIME_STEPS       = 10
ALPHA            = 0.6
BETA             = 0.9
ETA              = 1
DIMS             = [784, 102, 102]

# Data transforms
transform = Compose([
    ToTensor(),
    Normalize((0.0,), (1.0,)),
    Lambda(lambda x: x.view(-1))
])

train_ds = MNIST('../../data', train=True, download=True, transform=transform)
test_ds  = MNIST('../../data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

net = FeedbackFFNet(dims=DIMS, T=TIME_STEPS, α=ALPHA, β=BETA, η=ETA, lr=LR).to(DEVICE)


for epoch in range(EPOCH):
    # ——— Training ———
    epoch_losses = []
    net.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        losses = net.train_step(x, y)
        epoch_losses.append(losses)   # list of per‐layer losses

    # average loss per layer this epoch
    mean_losses = torch.tensor(epoch_losses).mean(dim=0).tolist()

    # ——— Evaluation ———
    net.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
            preds = net.predict(x_test)
            correct += (preds == y_test).sum().item()
            total   += y_test.size(0)
    test_acc = 100.0 * correct / total

    print(f"Epoch {epoch:3d}  losses per layer: {[f'{l:.4f}' for l in mean_losses]}  —  Test Acc: {test_acc:.2f}%")
