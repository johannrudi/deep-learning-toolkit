import torch

from dlk.metrics.utils import pairwise_distances


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    sq = torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)
    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)
    return torch.exp(-gamma * sq)


@torch.no_grad()
def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float | None = None,
    biased: bool = True,
    median_heuristic_max_samples: int = 512,
) -> torch.Tensor:
    """
    Returns MMD^2 (RBF kernel).
    """
    device = x.device
    x = x.float()
    y = y.float()

    if sigma is None:
        z = torch.cat([x, y], dim=0)
        n = z.shape[0]
        idx = torch.randperm(n, device=device)[: min(n, median_heuristic_max_samples)]
        zz = z[idx]
        D = pairwise_distances(zz, zz, p=1)  # Euclidean distances
        triu = D[torch.triu_indices(D.size(0), D.size(1), offset=1).unbind()]
        med = torch.median(triu)
        sigma = float(torch.clamp(med, min=1e-3).item())

    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)

    if biased:
        return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    else:
        N = x.shape[0]
        M = y.shape[0]
        return (
            (Kxx.sum() - Kxx.diag().sum()) / (N * (N - 1) + 1e-12)
            + (Kyy.sum() - Kyy.diag().sum()) / (M * (M - 1) + 1e-12)
            - 2.0 * Kxy.mean()
        )


@torch.no_grad()
def mmd_rbf_multi_sigma(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: list[float] | None = None,
    biased: bool = True,
) -> torch.Tensor:
    """
    Multi-kernel MMD^2 using a sum/average of RBF kernels with different bandwidths.
    """
    # set default sigmas
    if sigmas is None:
        sigmas = [0.1, 0.2, 0.4, 0.8]

    mmd2 = torch.Tensor([0.0])
    for s in sigmas:
        mmd2 = mmd2 + mmd_rbf(x, y, sigma=s, biased=biased)
    return mmd2 / len(sigmas)
