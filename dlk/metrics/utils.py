import torch


def pairwise_distances(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    x: [N, d], y: [M, d]
    p=1 -> Euclidean distance
    p=2 -> squared Euclidean distance
    """
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    sq = torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)
    if p == 2:
        return sq
    elif p == 1:
        return torch.sqrt(sq + 1e-12)
    else:
        raise ValueError(f"p must be 1 (Euclidean) or 2 (squared Euclidean), got {p=}")
