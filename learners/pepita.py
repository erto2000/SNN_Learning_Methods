import torch
import torch.nn.functional as F
from .base import BaseLearner
from networks.snn_core import SNNCore

class PepitaLearner(BaseLearner):
    """
    PEPITA-like two-pass update with feedback matrix F.
    Forward returns accumulated logits over time.
    """
    def __init__(self, net_cfg, meta, device, mode="original", lr=0.01, f_factor=0.05):
        super().__init__(net_cfg, meta, device)
        self.mode = mode
        self.lr = lr

        K = meta["n_classes"]
        D0 = net_cfg.layers[0].dim_in
        self.F = torch.empty(K, D0, device=device).normal_() * f_factor

    def _build_model(self):
        return SNNCore(self.cfg, self.meta["n_classes"])

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X:[B,T,D] -> logits:[B,K] (sum of head outputs over time)."""
        B, T, _ = X.shape
        X = X.to(self.device)
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device)
        for t in range(T):
            _, head_out, state, head_mem, _, _ = self.model.forward_step(X[:, t, :], state, head_mem)
            logits += head_out
        return logits

    @torch.no_grad()
    def _first_pass(self, X):
        B, T, _ = X.shape
        state, head_mem = self.model.init_state(B, X.device, X.dtype)
        L = len(self.model.fcs)
        h_rec = [[] for _ in range(L)]
        logits = torch.zeros(B, self.meta["n_classes"], device=X.device)
        for t in range(T):
            _, head_out, state, head_mem, layer_spikes, _ = self.model.forward_step(X[:, t, :], state, head_mem)
            for i, h in enumerate(layer_spikes): h_rec[i].append(h)
            logits += head_out
        h_seq = [torch.stack(seq, 0) for seq in h_rec]  # list of [T,B,H_l]
        return {"h_seq": h_seq, "logits": logits}

    def train_step(self, X: torch.Tensor, y: torch.Tensor):
        K = self.meta["n_classes"]
        X, y = X.to(self.device), y.to(self.device)
        B, T, D = X.shape

        fp = self._first_pass(X)
        logits = fp["logits"]
        p = torch.softmax(logits, dim=1)
        e = p - F.one_hot(y, num_classes=K).float()
        loss = F.cross_entropy(logits, y)

        X_mod = X + (e @ self.F).unsqueeze(1)    # [B,T,D]

        L = len(self.model.fcs)
        sp = self._first_pass(X_mod)
        h_seq     = fp["h_seq"]                       # list [T,B,H_l]
        h_mod_seq = sp["h_seq"]                       # list [T,B,H_l]

        # layer 0
        diff0 = (h_seq[0] - h_mod_seq[0])             # [T,B,H0]
        x_mod_T = X_mod.permute(1,0,2)                # [T,B,D]
        mult0 = diff0.unsqueeze(3) * x_mod_T.unsqueeze(2)  # [T,B,H0,D]
        dW0 = - mult0.sum(dim=(0,1)) / max(1, B*T)    # [H0,D]
        self.model.fcs[0].weight.data.add_(self.lr * dW0)

        # deeper layers
        for l in range(1, L):
            pre  = h_mod_seq[l-1]                     # [T,B,H_{l-1}]
            diff = (h_seq[l] - h_mod_seq[l])          # [T,B,H_l]
            mult = diff.unsqueeze(3) * pre.unsqueeze(2)   # [T,B,H_l,H_{l-1}]
            dWl  = - mult.sum(dim=(0,1)) / max(1, B*T)    # [H_l, H_{l-1}]
            self.model.fcs[l].weight.data.add_(self.lr * dWl)

        # readout (use second-pass last layer average over time)
        h_last_mod = h_mod_seq[-1].mean(0)            # [B, H_{L-1}]
        dWo = - (e.t() @ h_last_mod) / max(1, B)      # [K, H_{L-1}]
        self.model.head.weight.data.add_(self.lr * dWo)

        acc = (logits.argmax(1) == y).float().mean().item() * 100.0
        return {"acc": acc, "loss": loss.item()}
