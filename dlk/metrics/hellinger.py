r"""Compute Hellinger distances using histograms and kernel density estimation.

The Hellinger distance (https://en.wikipedia.org/wiki/Hellinger_distance) is defined as

.. math::
    H^{2}(P,Q) =
    {\frac {1}{2}}\displaystyle \int _{\mathcal {X}}\left({\sqrt {p(x)}}-{\sqrt {q(x)}}\right)^{2}\lambda (dx).

To compute this distance from samples, histograms or kernel density estimation provides
discrete distributions.  The Hellinger distance for two discrete probability distributions
:math:`{\displaystyle P=(p_{1},\ldots ,p_{k})}` and
:math:`{\displaystyle Q=(q_{1},\ldots ,q_{k})}` is

.. math::
    H(P,Q) =
    {\frac {1}{\sqrt {2}}}\;{\sqrt {\sum _{i=1}^{k}({\sqrt {p_{i}}}-{\sqrt {q_{i}}})^{2}}} =
    {\frac {1}{\sqrt {2}}}\;{\bigl \|}{\sqrt {P}}-{\sqrt {Q}}{\bigr \|}_{2}.

"""

import math
from collections.abc import Sequence

import torch

from dlk.metrics.utils import as_floating_2d, resolve_hist_bin_edges


@torch.no_grad()
def hellinger_distance_hist(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    hist_bins: int | Sequence[int] | None = None,
    hist_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
    scale_invariant: bool = False,
) -> torch.Tensor:
    r"""Estimate Hellinger distance between two sample sets using histograms.
    When `scale_invariant=True`, the metric rescales the square-root histogram of
    `samples2` before computing the norm by minimizing:

    .. math::
        d_{\text{SI}}(x, y)
        = \frac{1}{\sqrt{2}}\left\lVert \sqrt{x} - \alpha\sqrt{y} \right\rVert_2

    with respect to :math:`\alpha`, where:

    .. math::
        \alpha
        = \left(\frac{\sum_i \sqrt{x_i}\sqrt{y_i}}{\sum_i y_i}\right)^2
        \quad (\text{for } \sum_i y_i > 0)

    Args:
        samples1: First sample tensor with shape `(n_samples_1, n_features)`.
        samples2: Second sample tensor with shape `(n_samples_2, n_features)`.
        hist_bins: Number of bins as an int or per-feature sequence. Defaults to 16 bins per feature.
        hist_range: Histogram range as a global `(min, max)` tuple or per-feature `(min, max)` tuples.
            When omitted, the range is inferred from both sample sets.
        scale_invariant: If `True`, apply the scale-invariant variant defined above.
            If `False`, compute the standard Hellinger distance.

    Returns:
        Scalar tensor containing the histogram-based Hellinger distance.

    Raises:
        ValueError: If sample shapes/devices are incompatible or histogram settings are invalid.
    """
    samples1 = as_floating_2d(samples=samples1, name="samples1")
    samples2 = as_floating_2d(samples=samples2, name="samples2")

    if samples1.device != samples2.device:
        raise ValueError("samples1 and samples2 must be on the same device.")
    if samples1.shape[1] != samples2.shape[1]:
        raise ValueError(
            "samples1 and samples2 must have the same number of features; "
            f"got {samples1.shape[1]} and {samples2.shape[1]}."
        )

    bin_edges = resolve_hist_bin_edges(
        samples1,
        samples2,
        hist_bins=hist_bins,
        hist_range=hist_range,
    )

    # compute one multi-dimensional histogram over the entire feature space
    hist1, _ = torch.histogramdd(samples1, bins=bin_edges, density=True)
    hist2, _ = torch.histogramdd(samples2, bins=bin_edges, density=True)
    hist1_sqrt = torch.sqrt(torch.clamp(hist1, min=0.0))
    hist2_sqrt = torch.sqrt(torch.clamp(hist2, min=0.0))
    if scale_invariant:
        scale = torch.sum(hist1_sqrt * hist2_sqrt) / torch.sum(hist2_sqrt**2)
        diff = hist1_sqrt - scale * hist2_sqrt
        distance = torch.linalg.vector_norm(diff) / math.sqrt(2.0)
    else:
        diff = hist1_sqrt - hist2_sqrt
        distance = torch.linalg.vector_norm(diff) / math.sqrt(2.0)

    return distance


@torch.no_grad()
def hellinger_distance_hist_marginals(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    hist_bins: int | Sequence[int] | None = None,
    hist_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
    scale_invariant: bool = False,
) -> torch.Tensor:
    """Estimate per-feature 1D Hellinger distances from sample marginals.

    Args:
        samples1: First sample tensor with shape `(n_samples_1, n_features)`.
        samples2: Second sample tensor with shape `(n_samples_2, n_features)`.
        hist_bins: Number of bins as an int or per-feature sequence. Defaults to 16 bins per feature.
        hist_range: Histogram range as a global `(min, max)` tuple or per-feature `(min, max)` tuples.
            When omitted, the range is inferred from both sample sets.
        scale_invariant: If `True`, apply the scale-invariant variant described in
            `hellinger_distance_hist`. If `False`, compute the standard Hellinger distance.

    Returns:
        Tensor with shape `(n_features,)` containing one 1D Hellinger distance per feature.

    Raises:
        ValueError: If sample shapes/devices are incompatible or histogram settings are invalid.
    """
    samples1 = as_floating_2d(samples=samples1, name="samples1")
    samples2 = as_floating_2d(samples=samples2, name="samples2")

    if samples1.device != samples2.device:
        raise ValueError("samples1 and samples2 must be on the same device.")
    if samples1.shape[1] != samples2.shape[1]:
        raise ValueError(
            "samples1 and samples2 must have the same number of features; "
            f"got {samples1.shape[1]} and {samples2.shape[1]}."
        )

    bin_edges = resolve_hist_bin_edges(
        samples1,
        samples2,
        hist_bins=hist_bins,
        hist_range=hist_range,
    )
    n_features = samples1.shape[1]
    distances = torch.empty(n_features, device=samples1.device, dtype=samples1.dtype)

    # compute one-dimensional histograms for each feature independently
    for dim, bin_edges_per_dim in enumerate(bin_edges):
        hist1, _ = torch.histogram(
            samples1[:, dim], bins=bin_edges_per_dim, density=True
        )
        hist2, _ = torch.histogram(
            samples2[:, dim], bins=bin_edges_per_dim, density=True
        )
        hist1_sqrt = torch.sqrt(torch.clamp(hist1, min=0.0))
        hist2_sqrt = torch.sqrt(torch.clamp(hist2, min=0.0))
        if scale_invariant:
            scale = torch.sum(hist1_sqrt * hist2_sqrt) / torch.sum(hist2_sqrt**2)
            diff = hist1_sqrt - scale * hist2_sqrt
        else:
            diff = hist1_sqrt - hist2_sqrt
        distances[dim] = torch.linalg.vector_norm(diff) / math.sqrt(2.0)

    return distances
