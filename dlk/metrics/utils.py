"""Provide shared utilities for metric computations."""

from collections.abc import Sequence
from numbers import Real
from typing import Literal, TypeGuard

import torch


def as_floating_2d(
    samples: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Validate and normalize a 2D samples tensor.

    Args:
        samples: Tensor with shape `(n_samples, n_features)`.
        name: Argument name used in error messages.

    Returns:
        Floating-point tensor with unchanged shape.

    Raises:
        ValueError: If `samples` is not 2D.
    """
    if samples.ndim != 2:
        raise ValueError(
            f"{name} must be 2D with shape (n_samples, n_features), got {samples.shape}."
        )
    if not torch.is_floating_point(samples):
        samples = samples.float()
    return samples


def resolve_hist_bins(
    hist_bins: int | Sequence[int] | None,
    n_features: int,
) -> list[int]:
    """Resolve histogram bin counts for each feature dimension.

    Args:
        hist_bins: Number of bins per feature as an int or per-dimension sequence.
        n_features: Number of feature dimensions.

    Returns:
        List of per-dimension bin counts.

    Raises:
        ValueError: If bin counts are non-positive or incompatible with `n_features`.
    """
    if hist_bins is None:
        bins = [16] * n_features
    elif isinstance(hist_bins, int):
        bins = [hist_bins] * n_features
    else:
        bins = [int(n_bins) for n_bins in hist_bins]
        if len(bins) != n_features:
            raise ValueError(
                "hist_bins must have one value per feature; "
                f"expected {n_features}, got {len(bins)}."
            )
    if any(n_bins <= 0 for n_bins in bins):
        raise ValueError("hist_bins values must be positive.")
    return bins


def _is_global_hist_range(
    hist_range: tuple[float, float] | Sequence[tuple[float, float]],
) -> TypeGuard[tuple[float, float]]:
    """Check whether histogram range is a single global `(min, max)` pair."""
    if not isinstance(hist_range, tuple) or len(hist_range) != 2:
        return False
    low, high = hist_range
    return isinstance(low, Real) and isinstance(high, Real)


def resolve_hist_range(
    hist_range: tuple[float, float] | Sequence[tuple[float, float]] | None,
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    n_features: int,
) -> list[tuple[float, float]]:
    """Resolve histogram value ranges for each feature dimension.

    Args:
        hist_range: Global `(min, max)` range or per-dimension ranges.
        samples1: First sample tensor.
        samples2: Second sample tensor.
        n_features: Number of feature dimensions.

    Returns:
        Per-dimension list of `(min, max)` tuples.

    Raises:
        ValueError: If ranges are malformed or invalid.
    """
    if hist_range is None:
        low = torch.minimum(samples1.min(dim=0).values, samples2.min(dim=0).values)
        high = torch.maximum(samples1.max(dim=0).values, samples2.max(dim=0).values)
        return [
            (float(low_i.item()), float(high_i.item()))
            for low_i, high_i in zip(low, high, strict=True)
        ]

    if _is_global_hist_range(hist_range):
        low = float(hist_range[0])
        high = float(hist_range[1])
        if low >= high:
            raise ValueError("hist_range must satisfy min < max.")
        return [(low, high)] * n_features

    if len(hist_range) != n_features:
        raise ValueError(
            "hist_range must provide one (min, max) pair per feature; "
            f"expected {n_features}, got {len(hist_range)}."
        )

    resolved_range: list[tuple[float, float]] = []
    for dim, bounds in enumerate(hist_range):
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(
                f"hist_range[{dim}] must be a (min, max) pair, got {bounds}."
            )
        low_value = bounds[0]
        high_value = bounds[1]
        if not isinstance(low_value, Real) or not isinstance(high_value, Real):
            raise ValueError(
                f"hist_range[{dim}] must be a (min, max) pair, got {bounds}."
            )
        low = float(low_value)
        high = float(high_value)
        if low >= high:
            raise ValueError(f"hist_range[{dim}] must satisfy min < max.")
        resolved_range.append((low, high))
    return resolved_range


def resolve_hist_bin_edges(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    hist_bins: int | Sequence[int] | None = None,
    hist_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
) -> list[torch.Tensor]:
    n_features = samples1.shape[1]
    histogram_bins = resolve_hist_bins(
        hist_bins=hist_bins,
        n_features=n_features,
    )
    histogram_range = resolve_hist_range(
        hist_range=hist_range,
        samples1=samples1,
        samples2=samples2,
        n_features=n_features,
    )
    bin_edges = [
        torch.linspace(
            low,
            high,
            n_bins + 1,
            device=samples1.device,
            dtype=samples1.dtype,
        )
        for n_bins, (low, high) in zip(histogram_bins, histogram_range, strict=True)
    ]
    return bin_edges


@torch.no_grad()
def kdedd_on_grid(
    samples: torch.Tensor,
    grid_limits: torch.Tensor,
    grid_size: int = 32,
    bandwidth: float | str = 1.0,
    algorithm: str = "standard",
    kernel: str = "gaussian",
    batch_size: int = 128,
    log_density: bool = False,
) -> torch.Tensor:
    """Estimate KDE values on a Cartesian grid using torch-kde.

    Args:
        samples: Sample tensor with shape `(n_samples, n_features)`.
        grid_limits: Per-feature bounds with shape `(n_features, 2)` where each
            row is `(low, high)`. The transposed shape `(2, n_features)` is also
            accepted.
        grid_size: Number of evaluation points per feature dimension.
        bandwidth: Bandwidth argument forwarded to `torchkde.KernelDensity`.
        algorithm: Algorithm argument forwarded to `torchkde.KernelDensity`.
        kernel: Kernel argument forwarded to `torchkde.KernelDensity`.
        batch_size: Number of grid points per batch in `score_samples`.

    Returns:
        Flattened density values with shape `(grid_size**n_features,)`.

    Raises:
        ValueError: If tensor shapes or scalar arguments are invalid.
        ImportError: If `torchkde` is not available.
    """
    samples = as_floating_2d(samples=samples, name="samples")
    n_features = samples.shape[1]

    # resolve grid limits
    if grid_limits.ndim != 2:
        raise ValueError(
            "grid_limits must be 2D with shape (n_features, 2) "
            f"or (2, n_features), got {tuple(grid_limits.shape)}."
        )
    if grid_limits.shape == (n_features, 2):
        resolved_grid_limits = grid_limits
    elif grid_limits.shape == (2, n_features):
        resolved_grid_limits = grid_limits.T
    else:
        raise ValueError(
            "grid_limits must have shape (n_features, 2) "
            f"or (2, n_features), got {tuple(grid_limits.shape)}."
        )
    if not torch.is_floating_point(resolved_grid_limits):
        resolved_grid_limits = resolved_grid_limits.float()
    resolved_grid_limits = resolved_grid_limits.to(
        device=samples.device,
        dtype=samples.dtype,
    )

    low = resolved_grid_limits[:, 0]
    high = resolved_grid_limits[:, 1]
    if torch.any(low >= high):
        raise ValueError("Each grid_limits row must satisfy low < high.")

    if grid_size < 1:
        raise ValueError(f"grid_size must be positive, got {grid_size}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    try:
        from torchkde import KernelDensity
    except ImportError as exc:
        raise ImportError(
            "torchkde is required for kdedd_on_grid. Install with `pip install torch-kde`."
        ) from exc

    # create a KDE estimator from the input samples
    kernel_density = KernelDensity(
        bandwidth=bandwidth,
        algorithm=algorithm,
        kernel=kernel,
    )
    kernel_density.fit(samples)

    # construct Cartesian grid points using ij indexing
    grid_1d = [
        torch.linspace(
            low_i.item(),
            high_i.item(),
            grid_size,
            device=samples.device,
            dtype=samples.dtype,
        )
        for low_i, high_i in zip(low, high, strict=True)
    ]
    grid_mesh = torch.meshgrid(*grid_1d, indexing="ij")
    grid_points = torch.stack(grid_mesh, dim=0).reshape(n_features, -1).T

    # evaluate log-density and convert it to density values
    log_grid_vals = kernel_density.score_samples(grid_points, batch_size=batch_size)
    if log_density:
        return log_grid_vals
    return torch.exp(log_grid_vals)


def pairwise_distances(
    x: torch.Tensor,
    y: torch.Tensor,
    p: Literal[1, 2] = 2,
) -> torch.Tensor:
    """Compute pairwise distances between rows of two matrices.

    Args:
        x: Input tensor with shape ``[N, D]``.
        y: Input tensor with shape ``[M, D]``.
        p: Distance mode selector. Use ``1`` for Euclidean distance and ``2`` for
            squared Euclidean distance.

    Returns:
        A tensor with shape ``[N, M]`` where entry ``(i, j)`` is the distance
        between ``x[i]`` and ``y[j]``.

    Raises:
        ValueError: If ``x`` or ``y`` is not 2D.
        ValueError: If ``x`` and ``y`` do not share the same feature dimension.
        ValueError: If ``p`` is not one of ``{1, 2}``.
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
    if not p in [1, 2]:
        raise ValueError(f"p must be 1 (Euclidean) or 2 (squared Euclidean), got {p=}")

    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    sq = torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)
    if p == 2:
        return sq
    if p == 1:
        return torch.sqrt(sq + 1e-12)
