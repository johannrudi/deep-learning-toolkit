"""Provide helper utilities for histogram, KDE, and distance metrics."""

from collections.abc import Sequence
from numbers import Real
from typing import Literal, TypeAlias, TypeGuard

import torch

# --------------------------------------
# Types
# --------------------------------------

CountsLike: TypeAlias = int | Sequence[int] | None
LimitsLike: TypeAlias = tuple[float, float] | Sequence[tuple[float, float]] | None

# --------------------------------------


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


def resolve_counts(
    counts: CountsLike,
    n_features: int,
) -> list[int]:
    """Resolve histogram bin counts or grid point counts for each feature dimension.

    Args:
        counts: Number of bins/points per feature as an int or per-dimension sequence.
        n_features: Number of feature dimensions.

    Returns:
        List of per-dimension counts.

    Raises:
        ValueError: If counts are non-positive or incompatible with `n_features`.
    """
    if counts is None:
        resolved_counts = [16] * n_features
    elif isinstance(counts, int):
        resolved_counts = [counts] * n_features
    else:
        resolved_counts = [int(n) for n in counts]
        if len(resolved_counts) != n_features:
            raise ValueError(
                "counts must have one value per feature; "
                f"expected {n_features}, got {len(counts)}."
            )
    if any(n <= 0 for n in resolved_counts):
        raise ValueError(f"counts must be positive, got {counts}.")
    return resolved_counts


def _are_global_limits(
    limits: tuple[float, float] | Sequence[tuple[float, float]],
) -> TypeGuard[tuple[float, float]]:
    """Check whether limits are a single global `(min, max)` pair."""
    if not isinstance(limits, tuple) or len(limits) != 2:
        return False
    low, high = limits
    return isinstance(low, Real) and isinstance(high, Real)


def resolve_limits(
    limits: LimitsLike,
    n_features: int,
    samples1: torch.Tensor | None = None,
    samples2: torch.Tensor | None = None,
) -> list[tuple[float, float]]:
    """Resolve histogram value ranges or grid point limits for each feature dimension.

    Args:
        limits: Global `(min, max)` limits or per-dimension limits.
        n_features: Number of feature dimensions.
        samples1: First sample tensor.
        samples2: Second sample tensor.

    Returns:
        Per-dimension list of `(min, max)` tuples.

    Raises:
        ValueError: If limits are malformed or invalid.
    """
    if limits is None:
        if samples1 is None:
            raise ValueError("limits and samples1 cannot both be None.")
        low = samples1.min(dim=0).values
        high = samples1.max(dim=0).values
        if samples2 is not None:
            low = torch.minimum(low, samples2.min(dim=0).values)
            high = torch.maximum(high, samples2.max(dim=0).values)
        return [
            (float(low_i.item()), float(high_i.item()))
            for low_i, high_i in zip(low, high, strict=True)
        ]

    if _are_global_limits(limits):
        low = float(limits[0])
        high = float(limits[1])
        if low >= high:
            raise ValueError("limits must satisfy min < max.")
        return [(low, high)] * n_features

    if len(limits) != n_features:
        raise ValueError(
            "limits must provide one (min, max) pair per feature; "
            f"expected {n_features}, got {len(limits)}."
        )

    resolved_limits: list[tuple[float, float]] = []
    for dim, bounds in enumerate(limits):
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(f"limits[{dim}] must be a (min, max) pair, got {bounds}.")
        low_value = bounds[0]
        high_value = bounds[1]
        if not isinstance(low_value, Real) or not isinstance(high_value, Real):
            raise ValueError(f"limits[{dim}] must be a (min, max) pair, got {bounds}.")
        low = float(low_value)
        high = float(high_value)
        if low >= high:
            raise ValueError(f"limits[{dim}] must satisfy min < max.")
        resolved_limits.append((low, high))
    return resolved_limits


def resolve_points(
    samples1: torch.Tensor,
    samples2: torch.Tensor | None,
    counts: CountsLike = None,
    limits: LimitsLike = None,
    counts_are_intervals: bool = False,
) -> list[torch.Tensor]:
    """Resolve per-feature 1D coordinate grids from counts and limits.

    Args:
        samples1: First sample tensor with shape `(n_samples, n_features)`.
        samples2: Optional second sample tensor used when inferring limits.
        counts: Number of points/intervals per feature as an int or per-feature sequence.
        limits: Global `(min, max)` limits or per-feature limits.
        counts_are_intervals: `True` if counts represent intervals, or points otherwise.

    Returns:
        List of per-feature coordinate tensors. Each tensor has shape
        `(count_d + 1,)` for feature dimension `d` if `counts_are_intervals=True` and
        `(count_d,)` otherwise.

    Raises:
        ValueError: If counts or limits are invalid.
    """
    n_features = samples1.shape[1]
    resolved_counts = resolve_counts(
        counts=counts,
        n_features=n_features,
    )
    resolved_limits = resolve_limits(
        limits=limits,
        n_features=n_features,
        samples1=samples1,
        samples2=samples2,
    )

    if counts_are_intervals:
        add = 1
    else:
        add = 0
    points = [
        torch.linspace(lo, hi, n + add, device=samples1.device, dtype=samples1.dtype)
        for n, (lo, hi) in zip(resolved_counts, resolved_limits, strict=True)
    ]
    return points


@torch.no_grad()
def histogramdd(
    samples: torch.Tensor,
    bin_counts: CountsLike = None,
    bin_ranges: LimitsLike = None,
    to_density: bool = False,
    to_mass: bool = False,
) -> torch.Tensor:
    """Estimate a normalized multidimensional histogram from samples.

    Args:
        samples: Sample tensor with shape `(n_samples, n_features)`.
        bin_counts: Number of bins as one int or a per-feature sequence.
        bin_ranges: Global `(min, max)` range or per-feature ranges.
        to_density: If `True`, return density values whose integral over the
            histogram domain is approximately 1.
        to_mass: If `True`, divide the histogram output by `n_samples` so each
            bin stores empirical probability mass.

    Returns:
        Histogram density tensor with one axis per feature dimension.

    Raises:
        ValueError: If samples, bin_counts, or bin_ranges are invalid.
    """
    samples = as_floating_2d(samples=samples, name="samples")
    n_features = samples.shape[1]

    resolved_bin_counts = resolve_counts(
        counts=bin_counts,
        n_features=n_features,
    )
    resolved_bin_ranges = resolve_limits(
        limits=bin_ranges,
        n_features=n_features,
        samples1=samples,
    )
    bin_edges = resolve_points(
        samples1=samples,
        samples2=None,
        counts=resolved_bin_counts,
        limits=resolved_bin_ranges,
        counts_are_intervals=True,
    )

    hist, _ = torch.histogramdd(samples, bins=bin_edges, density=to_density)
    if to_mass:
        hist = hist / samples.shape[0]
    return hist


@torch.no_grad()
def kde_density(
    samples: torch.Tensor,
    grid_size: CountsLike = None,
    grid_limits: LimitsLike = None,
    bandwidth: float | str = 1.0,
    algorithm: str = "standard",
    kernel: str = "gaussian",
    batch_size: int = 256,
    log_density: bool = False,
    to_mass: bool = False,
) -> torch.Tensor:
    """Estimate KDE values on a Cartesian grid using torch-kde.

    Args:
        samples: Sample tensor with shape `(n_samples, n_features)`.
        grid_size: Number of intervals as one int or per-feature-dimension
            sequence. Each grid dimension contains `grid_size + 1` points.
        grid_limits: Global `(min, max)` limits or per-feature limits.
        bandwidth: Bandwidth argument forwarded to `torchkde.KernelDensity`.
        algorithm: Algorithm argument forwarded to `torchkde.KernelDensity`.
        kernel: Kernel argument forwarded to `torchkde.KernelDensity`.
        batch_size: Number of grid points per batch in `score_samples`.
        log_density: If `True`, return log-density values instead of density values.
        to_mass: If `True`, convert density values into approximate probability
            mass per grid cell by multiplying by cell volume. Ignored when
            `log_density=True`.

    Returns:
        Density or log-density values on the Cartesian grid with shape
        `(grid_size_0 + 1, ..., grid_size_{n_features - 1} + 1)`.

    Raises:
        ValueError: If grid_size, grid_limits, or batch_size are invalid.
        ImportError: If `torchkde` is not available.
    """
    samples = as_floating_2d(samples=samples, name="samples")
    n_features = samples.shape[1]

    resolved_grid_size = resolve_counts(
        counts=grid_size,
        n_features=n_features,
    )
    resolved_grid_limits = resolve_limits(
        limits=grid_limits,
        n_features=n_features,
        samples1=samples,
    )

    try:
        from torchkde import KernelDensity
    except ImportError as exc:
        raise ImportError(
            "torchkde is required for kde_on_grid. Install with `pip install torch-kde`."
        ) from exc

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    # create a KDE estimator from the input samples
    kernel_density = KernelDensity(
        bandwidth=bandwidth,
        algorithm=algorithm,
        kernel=kernel,
    )
    kernel_density.fit(samples)

    # construct Cartesian grid points using ij indexing
    grids_1d = resolve_points(
        samples1=samples,
        samples2=None,
        counts=resolved_grid_size,
        limits=resolved_grid_limits,
        counts_are_intervals=False,
    )
    grid_mesh = torch.meshgrid(*grids_1d, indexing="ij")
    grid_points = torch.stack(grid_mesh, dim=0).reshape(n_features, -1).T
    grid_shape = tuple(grid_1d.numel() for grid_1d in grids_1d)

    # evaluate log-density
    log_grid_vals = kernel_density.score_samples(grid_points, batch_size=batch_size)
    log_grid_vals = log_grid_vals.reshape(grid_shape)
    if log_density:
        return log_grid_vals

    # convert to density values
    kde_density = torch.exp(log_grid_vals)
    if to_mass:
        # compute the volume of each cell
        cell_volume = 1.0
        for dim, (siz, lim) in enumerate(
            zip(resolved_grid_size, resolved_grid_limits, strict=True)
        ):
            cell_volume *= (lim[1] - lim[0]) / (siz - 1)
        return kde_density * cell_volume
    return kde_density


def kde_density_to_mass(
    kde_density: torch.Tensor,
    grid_size: CountsLike = None,
    grid_limits: LimitsLike = None,
):
    """Convert grid-based KDE values into approximate probability masses.

    Args:
        kde_density: KDE values on a Cartesian grid with one axis per feature.
        grid_size: Number of grid points as one int or per-feature sequence.
        grid_limits: Global `(min, max)` limits or per-feature limits for the grid.

    Returns:
        Tensor with the same shape as `kde_density` containing approximate
        probability mass values per grid cell.

    Raises:
        ValueError: If `grid_size` or `grid_limits` are invalid.
    """
    n_features = kde_density.ndim
    resolved_grid_size = resolve_counts(counts=grid_size, n_features=n_features)
    resolved_grid_limits = resolve_limits(limits=grid_limits, n_features=n_features)

    # compute the volume of each cell
    cell_volume = 1.0
    for dim, (siz, lim) in enumerate(
        zip(resolved_grid_size, resolved_grid_limits, strict=True)
    ):
        cell_volume *= (lim[1] - lim[0]) / (siz - 1)
    return kde_density * cell_volume


@torch.no_grad()
def pairwise_distances(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    p: Literal[1, 2] = 2,
) -> torch.Tensor:
    """Compute pairwise distances between rows of two matrices.

    Args:
        samples1: Input tensor with shape ``[N, D]``.
        samples2: Input tensor with shape ``[M, D]``.
        p: Distance mode selector. Use ``1`` for Euclidean distance and ``2`` for
            squared Euclidean distance.

    Returns:
        A tensor with shape ``[N, M]`` where entry ``(i, j)`` is the distance
        between ``samples1[i]`` and ``samples2[j]``.

    Raises:
        ValueError: If ``samples1`` or ``samples2`` is not 2D.
        ValueError: If ``samples1`` and ``samples2`` do not share the same feature dimension.
        ValueError: If ``p`` is not one of ``{1, 2}``.
    """
    if samples1.ndim != 2:
        raise ValueError(
            f"samples1 must be 2D with shape [N, D], got shape {tuple(samples1.shape)}"
        )
    if samples2.ndim != 2:
        raise ValueError(
            f"samples2 must be 2D with shape [M, D], got shape {tuple(samples2.shape)}"
        )
    if samples1.shape[1] != samples2.shape[1]:
        raise ValueError(
            "samples1 and samples2 must have the same feature dimension, "
            f"got samples1.shape[1]={samples1.shape[1]} and samples2.shape[1]={samples2.shape[1]}"
        )
    if not p in [1, 2]:
        raise ValueError(f"p must be 1 (Euclidean) or 2 (squared Euclidean), got {p=}")

    x2 = (samples1 * samples1).sum(dim=1, keepdim=True)
    y2 = (samples2 * samples2).sum(dim=1, keepdim=True).T
    sq = torch.clamp(x2 + y2 - 2.0 * (samples1 @ samples2.T), min=0.0)
    if p == 2:
        return sq
    if p == 1:
        return torch.sqrt(sq + 1e-12)
