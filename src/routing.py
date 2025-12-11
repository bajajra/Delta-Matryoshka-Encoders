import torch
from torch import nn

class ResidualPreview(nn.Module):
    """A small predictor that estimates whether delta would be beneficial at token level.
    Returns a scalar per token; higher => apply delta."""
    def __init__(self, hidden_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):  # h: (B, T, H)
        score = self.net(h).squeeze(-1)  # (B, T)
        return score

def token_topk_mask(scores, ratio: float):
    """Return boolean mask (B, T) selecting top-k fraction per batch element."""
    B, T = scores.shape
    k = max(1, int(T * ratio))
    # Get top-k per row
    topk = torch.topk(scores, k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    rows = torch.arange(B)[:, None].to(scores.device)
    mask[rows, topk] = True
    return mask
