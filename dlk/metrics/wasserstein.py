"""Compute optimal transport costs and divergence metrics."""

from typing import Literal

import torch

from dlk.metrics.utils import pairwise_distances


@torch.no_grad()
def sinkhorn_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 300,
    p: Literal[1, 2] = 1,
) -> torch.Tensor:
    r"""Compute entropic OT cost between two uniform empirical measures.

    Args:
        x: Source samples with shape ``[N, D]``.
        y: Target samples with shape ``[M, D]``.
        epsilon: Positive entropy regularization strength.
        n_iters: Number of Sinkhorn fixed-point iterations.
        p: Ground-cost mode passed to `pairwise_distances`.
            Use ``1`` for Euclidean distance and ``2`` for squared Euclidean distance.

    Returns:
        A scalar tensor equal to :math:`OT_{\epsilon}(P, Q)`.

    Raises:
        ValueError: If ``x`` or ``y`` is not 2D.
        ValueError: If ``x`` or ``y`` has zero rows.
        ValueError: If ``x`` and ``y`` do not share the same feature dimension.
        ValueError: If ``epsilon <= 0``.
        ValueError: If ``n_iters < 1``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D with shape [N, D], got shape {tuple(x.shape)}")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D with shape [M, D], got shape {tuple(y.shape)}")
    if x.shape[0] == 0:
        raise ValueError("x must contain at least one sample")
    if y.shape[0] == 0:
        raise ValueError("y must contain at least one sample")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "x and y must have the same feature dimension, "
            f"got x.shape[1]={x.shape[1]} and y.shape[1]={y.shape[1]}"
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")

    device = x.device
    dtype = torch.promote_types(x.dtype, y.dtype)
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32

    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    n_samples_x = x.shape[0]
    n_samples_y = y.shape[0]

    a = torch.full((n_samples_x,), 1.0 / n_samples_x, device=device, dtype=dtype)
    b = torch.full((n_samples_y,), 1.0 / n_samples_y, device=device, dtype=dtype)

    # stabilize log computations with dtype-aware lower bounds
    tiny = torch.finfo(dtype).tiny
    log_a = torch.log(a.clamp_min(tiny))
    log_b = torch.log(b.clamp_min(tiny))

    C = pairwise_distances(x, y, p=p)  # [N, M]

    # update dual potentials in the log domain
    f = torch.zeros(n_samples_x, device=device, dtype=dtype)
    g = torch.zeros(n_samples_y, device=device, dtype=dtype)
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
    p: Literal[1, 2] = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the Sinkhorn divergence and the cross OT cost.

    The divergence is:
    :math:`S_{\epsilon}(P,Q)=OT_{\epsilon}(P,Q)-0.5\,OT_{\epsilon}(P,P)-0.5\,OT_{\epsilon}(Q,Q)`.

    Args:
        x: Source samples with shape ``[N, D]``.
        y: Target samples with shape ``[M, D]``.
        epsilon: Positive entropy regularization strength.
        n_iters: Number of Sinkhorn fixed-point iterations for each OT term.
        p: Ground-cost mode passed to `pairwise_distances`.
            Use ``1`` for Euclidean distance and ``2`` for squared Euclidean distance.

    Returns:
        A tuple ``(divergence, cost)`` where:
            - ``divergence`` is :math:`S_{\epsilon}(P,Q)`.
            - ``cost`` is :math:`OT_{\epsilon}(P,Q)`.

    Raises:
        ValueError: If any validation in `sinkhorn_cost` fails.
    """
    _xy = sinkhorn_cost(x, y, epsilon=epsilon, n_iters=n_iters, p=p)
    _xx = sinkhorn_cost(x, x, epsilon=epsilon, n_iters=n_iters, p=p)
    _yy = sinkhorn_cost(y, y, epsilon=epsilon, n_iters=n_iters, p=p)
    divergence = _xy - 0.5 * _xx - 0.5 * _yy
    cost = _xy
    return divergence, cost
