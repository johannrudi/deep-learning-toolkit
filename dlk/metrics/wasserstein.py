import torch

from dlk.metrics.utils import pairwise_distances


@torch.no_grad()
def sinkhorn_wasserstein_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 300,
    p: int = 1,
) -> torch.Tensor:
    """
    Entropic OT cost between two empirical measures with uniform weights.
    Returns scalar cost.
    """
    device = x.device
    dtype = x.dtype
    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    N = x.shape[0]
    M = y.shape[0]

    a = torch.full((N,), 1.0 / N, device=device, dtype=dtype)
    b = torch.full((M,), 1.0 / M, device=device, dtype=dtype)
    log_a = torch.log(a + 1e-32)
    log_b = torch.log(b + 1e-32)

    C = pairwise_distances(x, y, p=p)  # [N,M]

    # Log-domain Sinkhorn dual updates
    f = torch.zeros(N, device=device, dtype=dtype)
    g = torch.zeros(M, device=device, dtype=dtype)
    for _ in range(n_iters):
        f = epsilon * (log_a - torch.logsumexp((g[None, :] - C) / epsilon, dim=1))
        g = epsilon * (log_b - torch.logsumexp((f[:, None] - C) / epsilon, dim=0))

    P = torch.exp((f[:, None] + g[None, :] - C) / epsilon)
    cost = torch.sum(P * C)
    return cost
