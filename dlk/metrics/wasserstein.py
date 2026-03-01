import torch

from dlk.metrics.utils import pairwise_distances


@torch.no_grad()
def sinkhorn_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 300,
    p: int = 1,
) -> torch.Tensor:
    """
    Entropic OT cost OT_eps(P,Q) between uniform empirical measures.
    Returns scalar OT cost (biased for eps>0).
    """
    device, dtype = x.device, x.dtype
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


@torch.no_grad()
def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 300,
    p: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sinkhorn divergence:
        S_eps(P,Q) = OT_eps(P,Q) - 0.5*OT_eps(P,P) - 0.5*OT_eps(Q,Q)

    Nonnegative, symmetric, and S_eps(P,P)=0 (up to numerical error).
    """
    _xy = sinkhorn_cost(x, y, epsilon=epsilon, n_iters=n_iters, p=p)
    _xx = sinkhorn_cost(x, x, epsilon=epsilon, n_iters=n_iters, p=p)
    _yy = sinkhorn_cost(y, y, epsilon=epsilon, n_iters=n_iters, p=p)
    divergence = _xy - 0.5 * _xx - 0.5 * _yy
    cost = _xy
    return divergence, cost
