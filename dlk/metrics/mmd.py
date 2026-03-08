"""Compute maximum mean discrepancy metrics."""

from collections.abc import Sequence

import torch

from dlk.metrics.utils import pairwise_distances


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute an RBF kernel matrix between two sets of vectors.

    Args:
        x: Source tensor with shape ``[N, D]``.
        y: Target tensor with shape ``[M, D]``.
        sigma: Positive kernel bandwidth.

    Returns:
        A kernel matrix with shape ``[N, M]``.

    Raises:
        ValueError: If ``x`` or ``y`` is not 2D.
        ValueError: If ``x`` and ``y`` do not share feature dimension ``D``.
        ValueError: If ``sigma`` is not positive.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D with shape [N, D], got shape {tuple(x.shape)}")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D with shape [M, D], got shape {tuple(y.shape)}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "x and y must have the same feature dimension, "
            f"got x.shape[1]={x.shape[1]} and y.shape[1]={y.shape[1]}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma=}")
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    sq = torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)
    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)
    return torch.exp(-gamma * sq)


@torch.no_grad()
def mmd2_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float | None = None,
    biased: bool = True,
    median_heuristic_max_samples: int = 512,
) -> torch.Tensor:
    """Compute MMD² with an RBF kernel.

    Args:
        x: First sample matrix with shape ``[N, D]``.
        y: Second sample matrix with shape ``[M, D]``.
        sigma: RBF bandwidth. If ``None``, estimate via median heuristic.
        biased: If ``True``, use the biased estimator. If ``False``, use the
            unbiased estimator.
        median_heuristic_max_samples: Maximum number of pooled samples used to
            estimate ``sigma`` when ``sigma is None``.

    Returns:
        A scalar tensor with the estimated MMD² value.

    Raises:
        ValueError: If sample matrices are empty.
        ValueError: If ``median_heuristic_max_samples`` is not positive.
        ValueError: If ``sigma`` is provided and not positive.
        ValueError: If unbiased estimation is requested with fewer than 2
            samples in either set.
    """
    device = x.device
    x = x.float()
    y = y.float()
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("x and y must each contain at least one sample")
    if median_heuristic_max_samples <= 0:
        raise ValueError(
            "median_heuristic_max_samples must be positive, "
            f"got {median_heuristic_max_samples=}"
        )

    if sigma is None:
        z = torch.cat([x, y], dim=0)
        n = z.shape[0]
        idx = torch.randperm(n, device=device)[: min(n, median_heuristic_max_samples)]
        zz = z[idx]
        D = pairwise_distances(zz, zz, p=1)  # Euclidean distances
        triu_indices = torch.triu_indices(
            D.size(0), D.size(1), offset=1, device=D.device
        )
        triu = D[triu_indices[0], triu_indices[1]]
        if triu.numel() == 0:
            sigma = 1.0
        else:
            med = torch.median(triu)
            sigma = float(torch.clamp(med, min=1e-3).item())
    elif sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma=}")

    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)

    if biased:
        return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    N = x.shape[0]
    M = y.shape[0]
    if N < 2 or M < 2:
        raise ValueError(
            "unbiased estimation requires at least two samples per set, "
            f"got {N=}, {M=}"
        )
    return (
        (Kxx.sum() - Kxx.diag().sum()) / (N * (N - 1) + 1e-12)
        + (Kyy.sum() - Kyy.diag().sum()) / (M * (M - 1) + 1e-12)
        - 2.0 * Kxy.mean()
    )


@torch.no_grad()
def mmd2_rbf_multi_sigma(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Sequence[float] | None = None,
    biased: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute multi-kernel MMD² by averaging across RBF bandwidths.

    Args:
        x: First sample matrix with shape ``[N, D]``.
        y: Second sample matrix with shape ``[M, D]``.
        sigmas: Sequence of positive RBF bandwidths. If ``None``, use default
            values ``[0.1, 0.2, 0.4, 0.8]``.
        biased: If ``True``, use biased estimator per kernel.

    Returns:
        A tuple ``(mmd2_multi, mmd2_single)`` where ``mmd2_multi`` is the
        averaged scalar MMD² and ``mmd2_single`` contains per-bandwidth scalar
        MMD² values.

    Raises:
        ValueError: If ``sigmas`` is empty.
        ValueError: If any ``sigma`` in ``sigmas`` is not positive.
    """
    # set default sigmas
    if sigmas is None:
        sigmas = [0.1, 0.2, 0.4, 0.8]
    if len(sigmas) == 0:
        raise ValueError("sigmas must contain at least one bandwidth value")
    if any(s <= 0.0 for s in sigmas):
        raise ValueError(f"all sigma values must be positive, got {list(sigmas)=}")

    mmd2_single: list[torch.Tensor] = []
    mmd2_multi = torch.zeros((), device=x.device, dtype=torch.float32)
    for s in sigmas:
        mmd2 = mmd2_rbf(x, y, sigma=s, biased=biased)
        mmd2_single.append(mmd2)
        mmd2_multi = mmd2_multi + mmd2
    mmd2_multi = mmd2_multi / len(sigmas)

    return mmd2_multi, mmd2_single
