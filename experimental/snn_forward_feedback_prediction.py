import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Lambda, Compose
from torch.utils.data import DataLoader

# ——— helper for +/− overlays ———————————————————————————————
def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] = 0.0
    x_[torch.arange(x.size(0)), y] = x.max()
    return x_

# ——— one layer: FF + FB + Leaky LIF + its own optimizer —————————————————
class FeedbackLeakyLayer(nn.Module):
    def __init__(self, in_features, out_features, lr, beta):
        super().__init__()
        self.fc_ff = nn.Linear(in_features,  out_features, bias=False)
        self.fc_fb = nn.Linear(out_features, out_features, bias=False)
        self.lif   = snn.Leaky(beta=beta, spike_grad=surrogate.atan())
        self.opt   = torch.optim.Adam(self.parameters(), lr=lr)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.fc_ff.out_features, device=device)

# ——— the full network with logistic loss ———————————————————————
class FeedbackFFNet(nn.Module):
    def __init__(self, dims, T=10, beta=0.99, eta=1.0, lr=1e-3):
        super().__init__()
        self.T   = T
        self.eta = eta
        self.layers = nn.ModuleList([
            FeedbackLeakyLayer(dims[i], dims[i+1], lr, beta)
            for i in range(len(dims)-1)
        ])

    def forward_drives(self, x):
        batch, device = x.size(0), x.device
        L = len(self.layers)
        mems    = [lyr.init_state(batch,device) for lyr in self.layers]
        prev_sp = [torch.zeros_like(m)    for m   in mems]
        ff_sum  = [torch.zeros_like(m)    for m   in mems]
        fb_sum  = [torch.zeros_like(m)    for m   in mems]

        for _ in range(self.T):
            cur_sp = []
            for ℓ, lyr in enumerate(self.layers):
                # feed-forward drive
                ff = lyr.fc_ff(x if ℓ==0 else prev_sp[ℓ-1].detach())
                # feedback drive (zero on top layer)
                fb = (lyr.fc_fb(prev_sp[ℓ+1].detach())
                      if ℓ < L-1 else torch.zeros_like(ff))

                ff_sum[ℓ] += ff
                fb_sum[ℓ] += self.eta * fb

                # step LIF
                spk, mems[ℓ] = lyr.lif(ff + self.eta*fb, mems[ℓ])
                cur_sp.append(spk)
            prev_sp = cur_sp

        # average over time
        ff_avg = [s / self.T for s in ff_sum]
        fb_avg = [s / self.T for s in fb_sum]
        return ff_avg, fb_avg

    def train_step(self, x, y):
        # create positive / negative overlays
        x_pos = overlay_y_on_x(x, y)
        perm  = torch.randperm(x.size(0), device=x.device)
        x_neg = overlay_y_on_x(x, y[perm])

        # get averaged drives
        ff_p, fb_p = self.forward_drives(x_pos)
        ff_n, fb_n = self.forward_drives(x_neg)

        losses = []
        # for each layer compute E⁺, E⁻, then logistic( E⁺ − E⁻ )
        for lyr, ffp, fbp, ffn, fbn in zip(self.layers, ff_p, fb_p, ff_n, fb_n):
            err_p = ffp - fbp
            err_n = ffn - fbn
            Epos = err_p.pow(2).mean()
            Eneg = err_n.pow(2).mean()

            # logistic contrastive loss
            eps = 1e-8
            logp = torch.log(Epos + eps) - torch.log(Eneg + eps)
            loss = F.softplus(logp)

            lyr.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lyr.parameters(), 1.0)
            lyr.opt.step()

            losses.append((Epos.item(), Eneg.item(), loss.item()))

        return losses

    @torch.no_grad()
    def predict(self, x):
        batch = x.size(0)
        device = x.device
        errors = torch.zeros(batch, 10, device=device)

        for lbl in range(10):
            x_lbl = overlay_y_on_x(x, torch.full((batch,), lbl, device=device))
            ff, fb = self.forward_drives(x_lbl)
            # sum MSE across layers, per‐sample
            err = sum((f - b).pow(2).mean(dim=1) for (f, b) in zip(ff, fb))
            errors[:, lbl] = err

        # choose label with smallest error
        return errors.argmin(dim=1)

# ——— set up data, model, and training loop —————————————————————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([ToTensor(), Normalize((0,), (1,)), Lambda(lambda x: x.view(-1))])
train_loader = DataLoader(MNIST('../data', train=True,  download=True, transform=transform),
                          batch_size=512, shuffle=True)
test_loader  = DataLoader(MNIST('../data', train=False, download=True, transform=transform),
                          batch_size=512)

net = FeedbackFFNet([784, 250, 250], T=10, beta=0.99, eta=0, lr=1e-4).to(device)

for epoch in range(10):
    net.train()
    all_stats = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        stats = net.train_step(x, y)
        all_stats.extend(stats)

    # unpack Epos, Eneg, L
    Epos = torch.tensor([s[0] for s in all_stats]).mean()
    Eneg = torch.tensor([s[1] for s in all_stats]).mean()
    L    = torch.tensor([s[2] for s in all_stats]).mean()
    print(f"Epoch {epoch:2d}  ⟨E⁺⟩={Epos:.4f}  ⟨E⁻⟩={Eneg:.4f}  ⟨ℓ⟩={L:.4f}", end='')

    # eval acc
    net.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            preds = net.predict(x_test)
            correct += (preds == y_test).sum().item()
            total   += y_test.size(0)
    acc = 100.0 * correct / total
    print(f"  → Acc: {acc:.2f}%")
