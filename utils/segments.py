# utils/segments.py
import torch

def iter_pieces(loader, device):
    """
    Convert dataset batches [B,S,T,D], y:[B] into piece batches [N,T,D], y:[N], sample_ids:[N], B.
    """
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        B,S,T,D = X.shape
        Xp = X.view(B*S, T, D)
        yp = y.unsqueeze(1).expand(B, S).reshape(-1)
        sample_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, S).reshape(-1)
        yield Xp, yp, sample_ids, B

@torch.no_grad()
def majority_vote(preds: torch.Tensor, sample_ids: torch.Tensor, n_classes: int, B: int):
    # preds:[N], sample_ids:[N] with values in 0..B-1
    votes = torch.zeros(B, n_classes, device=preds.device, dtype=torch.int32)
    votes[sample_ids, preds] += 1
    return votes.argmax(1)  # [B]
