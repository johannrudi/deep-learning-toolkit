from collections.abc import Sequence
from numbers import Real

import torch


def _as_floating_2d(samples: torch.Tensor, name: str) -> torch.Tensor:
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


def _resolve_hist_bins(
    hist_bins: int | Sequence[int] | None, n_features: int
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


def _resolve_hist_range(
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

    if len(hist_range) == 2 and all(isinstance(value, Real) for value in hist_range):
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
        if len(bounds) != 2:
            raise ValueError(f"hist_range[{dim}] must be a (min, max) pair.")
        low = float(bounds[0])
        high = float(bounds[1])
        if low >= high:
            raise ValueError(f"hist_range[{dim}] must satisfy min < max.")
        resolved_range.append((low, high))
    return resolved_range


@torch.no_grad()
def hellinger_distance_hist(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    hist_bins: int | Sequence[int] | None = None,
    hist_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Estimate Hellinger distance between two sample sets using histograms.

    Args:
        samples1: First sample tensor with shape `(n_samples_1, n_features)`.
        samples2: Second sample tensor with shape `(n_samples_2, n_features)`.
        hist_bins: Number of bins as an int or per-feature sequence. Defaults to 16 bins per feature.
        hist_range: Histogram range as a global `(min, max)` tuple or per-feature `(min, max)` tuples.
            When omitted, the range is inferred from both sample sets.

    Returns:
        Scalar tensor containing the Hellinger distance.

    Raises:
        ValueError: If sample shapes/devices are incompatible or histogram settings are invalid.
    """
    samples1 = _as_floating_2d(samples=samples1, name="samples1")
    samples2 = _as_floating_2d(samples=samples2, name="samples2")

    if samples1.device != samples2.device:
        raise ValueError("samples1 and samples2 must be on the same device.")
    if samples1.shape[1] != samples2.shape[1]:
        raise ValueError(
            "samples1 and samples2 must have the same number of features; "
            f"got {samples1.shape[1]} and {samples2.shape[1]}."
        )

    n_features = samples1.shape[1]
    histogram_bins = _resolve_hist_bins(hist_bins=hist_bins, n_features=n_features)
    histogram_range = _resolve_hist_range(
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

    hist1, _ = torch.histogramdd(samples1, bins=bin_edges, density=True)
    hist2, _ = torch.histogramdd(samples2, bins=bin_edges, density=True)
    diff = torch.sqrt(torch.clamp(hist1, min=0.0)) - torch.sqrt(
        torch.clamp(hist2, min=0.0)
    )
    return torch.sqrt(torch.sum(diff * diff) / 2.0)


@torch.no_grad()
def hellinger_distance_hist_marginals(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    hist_bins: int | Sequence[int] | None = None,
    hist_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Estimate per-feature 1D Hellinger distances from sample marginals.

    Args:
        samples1: First sample tensor with shape `(n_samples_1, n_features)`.
        samples2: Second sample tensor with shape `(n_samples_2, n_features)`.
        hist_bins: Number of bins as an int or per-feature sequence. Defaults to 16 bins per feature.
        hist_range: Histogram range as a global `(min, max)` tuple or per-feature `(min, max)` tuples.
            When omitted, the range is inferred from both sample sets.

    Returns:
        Tensor with shape `(n_features,)` containing one 1D Hellinger distance per feature.

    Raises:
        ValueError: If sample shapes/devices are incompatible or histogram settings are invalid.
    """
    samples1 = _as_floating_2d(samples=samples1, name="samples1")
    samples2 = _as_floating_2d(samples=samples2, name="samples2")

    if samples1.device != samples2.device:
        raise ValueError("samples1 and samples2 must be on the same device.")
    if samples1.shape[1] != samples2.shape[1]:
        raise ValueError(
            "samples1 and samples2 must have the same number of features; "
            f"got {samples1.shape[1]} and {samples2.shape[1]}."
        )

    n_features = samples1.shape[1]
    histogram_bins = _resolve_hist_bins(hist_bins=hist_bins, n_features=n_features)
    histogram_range = _resolve_hist_range(
        hist_range=hist_range,
        samples1=samples1,
        samples2=samples2,
        n_features=n_features,
    )
    distances = torch.empty(n_features, device=samples1.device, dtype=samples1.dtype)

    # compute one-dimensional histograms for each feature independently.
    for dim, (n_bins, (low, high)) in enumerate(
        zip(histogram_bins, histogram_range, strict=True)
    ):
        bin_edges = torch.linspace(
            low,
            high,
            n_bins + 1,
            device=samples1.device,
            dtype=samples1.dtype,
        )
        hist1, _ = torch.histogram(samples1[:, dim], bins=bin_edges, density=True)
        hist2, _ = torch.histogram(samples2[:, dim], bins=bin_edges, density=True)
        diff = torch.sqrt(torch.clamp(hist1, min=0.0)) - torch.sqrt(
            torch.clamp(hist2, min=0.0)
        )
        distances[dim] = torch.sqrt(torch.sum(diff * diff) / 2.0)

    return distances
