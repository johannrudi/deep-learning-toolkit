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
    {\frac {1}{\sqrt {2}}}\;{\bigl \|}{\sqrt {P}}-{\sqrt {Q}}{\bigr \|}_{2} =
    \sqrt {1 - \sum _{i=1}^{k}{\sqrt {p_{i}q_{i}}}}.

"""

import math
from collections.abc import Sequence
from typing import Literal, cast

import torch

from dlk.metrics.utils import as_floating_2d, resolve_hist_bin_edges

# --------------------------------------
# Types
# --------------------------------------

Method = Literal[
    "direct",
    "direct_scale_invariant",
    "bc",
]

METHOD_VALUES: set[str] = set(Method.__args__)


# --------------------------------------


def _validate_method(method: Method | str) -> Method:
    """Validate the Hellinger distance method selector.

    Args:
        method: Hellinger distance implementation name.

    Returns:
        The validated method name.

    Raises:
        ValueError: If `method` is not one of the supported names.
    """
    if method not in METHOD_VALUES:
        valid_methods = ", ".join(f'"{value}"' for value in METHOD_VALUES)
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}.")
    return cast(Method, method)


def _compute_h_dist_direct_method(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute the Hellinger distance directly:

    .. math::
        {\frac {1}{\sqrt {2}}}\;{\bigl \|}{\sqrt {P}}-{\sqrt {Q}}{\bigr \|}_{2}
    """
    pm1_sqrt = torch.sqrt(torch.clamp(probmass1, min=0.0))
    pm2_sqrt = torch.sqrt(torch.clamp(probmass2, min=0.0))
    diff = pm1_sqrt - pm2_sqrt
    return torch.linalg.vector_norm(diff) / math.sqrt(2.0)


def _compute_h_dist_direct_scale_invariant_method(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute a scale invariant metric by minimizing:

    .. math::
        d_{\text{SI}}(x, y)
        = \frac{1}{\sqrt{2}}\left\lVert \sqrt{x} - \alpha\sqrt{y} \right\rVert_2

    with respect to :math:`\alpha`, where:

    .. math::
        \alpha
        = \frac{\sum_i \sqrt{x_i}\sqrt{y_i}}{\sum_i y_i}
        \quad (\text{for } \sum_i y_i > 0)

    If :math:`\sum_i y_i = 0`, this method uses :math:`\alpha = 0`.
    """
    pm1_sqrt = torch.sqrt(torch.clamp(probmass1, min=0.0))
    pm2_sqrt = torch.sqrt(torch.clamp(probmass2, min=0.0))
    n = torch.sum(pm1_sqrt * pm2_sqrt)
    d = torch.sum(pm2_sqrt.square())
    scale = n / d if 0.0 < d else torch.zeros_like(d)
    diff = pm1_sqrt - scale * pm2_sqrt
    return torch.linalg.vector_norm(diff) / math.sqrt(2.0)


def _compute_h_dist_bc_method(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute a scale invariant metric with the generalized Bhattacharyya coefficient (BC):

    .. math::
        BC(x,y) = \frac {\sum _{i=1}^{k} \sqrt {x_{i} y_{i}}}
                        {\sqrt {\bigl(\sum _{i=1}^{k} x_{i}\bigr)
                                \bigl(\sum _{i=1}^{k} y_{i}\bigr)}}

    .. math::
        H(x,y) = \sqrt {1 - BC(x,y)}.
    """
    probmass1 = torch.clamp(probmass1, min=0.0)
    probmass2 = torch.clamp(probmass2, min=0.0)
    totalmass1 = torch.sum(probmass1)
    totalmass2 = torch.sum(probmass2)
    n = torch.sum(torch.sqrt(probmass1 * probmass2))
    d = torch.sqrt(totalmass1 * totalmass2)
    if 0.0 < d:
        bc = n / d
    elif totalmass1 <= 0.0 and totalmass2 <= 0.0:
        bc = torch.ones_like(n)
    else:
        bc = torch.zeros_like(n)
    return torch.sqrt(torch.clamp(1.0 - bc, min=0.0))


def _compute_h_dist(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
    method: Method,
) -> torch.Tensor:
    """Dispatch Hellinger distance computation by method name."""
    if method == "direct_scale_invariant":
        return _compute_h_dist_direct_scale_invariant_method(probmass1, probmass2)
    if method == "bc":
        return _compute_h_dist_bc_method(probmass1, probmass2)
    return _compute_h_dist_direct_method(probmass1, probmass2)


@torch.no_grad()
def hellinger_distance_hist(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    hist_bins: int | Sequence[int] | None = None,
    hist_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
    method: Method = "direct",
) -> torch.Tensor:
    r"""Estimate Hellinger distance between two sample sets using histograms.

    Args:
        samples1: First sample tensor with shape `(n_samples_1, n_features)`.
        samples2: Second sample tensor with shape `(n_samples_2, n_features)`.
        hist_bins: Number of bins as an int or per-feature sequence. Defaults to 16 bins per feature.
        hist_range: Histogram range as a global `(min, max)` tuple or per-feature `(min, max)` tuples.
            When omitted, the range is inferred from both sample sets.
        method: Distance implementation. Use `"direct"` for
            :math:`\frac{1}{\sqrt{2}}\lVert\sqrt{P}-\sqrt{Q}\rVert_2`,
            `"direct_scale_invariant"` for the scale-invariant direct method, or
            `"bc"` for the Bhattacharyya-coefficient form.

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
    method = _validate_method(method)

    # calculate bin edges
    bin_edges = resolve_hist_bin_edges(
        samples1,
        samples2,
        hist_bins=hist_bins,
        hist_range=hist_range,
    )

    # compute one multi-dimensional histogram over the entire feature space
    hist1, _ = torch.histogramdd(samples1, bins=bin_edges, density=True)
    hist2, _ = torch.histogramdd(samples2, bins=bin_edges, density=True)
    distance = _compute_h_dist(hist1, hist2, method=method)

    return distance


@torch.no_grad()
def hellinger_distance_hist_marginals(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    hist_bins: int | Sequence[int] | None = None,
    hist_range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
    method: Method = "direct",
) -> torch.Tensor:
    r"""Estimate per-feature 1D Hellinger distances from sample marginals.

    Args:
        samples1: First sample tensor with shape `(n_samples_1, n_features)`.
        samples2: Second sample tensor with shape `(n_samples_2, n_features)`.
        hist_bins: Number of bins as an int or per-feature sequence. Defaults to 16 bins per feature.
        hist_range: Histogram range as a global `(min, max)` tuple or per-feature `(min, max)` tuples.
            When omitted, the range is inferred from both sample sets.
        method: Distance implementation. Use `"direct"` for
            :math:`\frac{1}{\sqrt{2}}\lVert\sqrt{P}-\sqrt{Q}\rVert_2`,
            `"direct_scale_invariant"` for the scale-invariant direct method, or
            `"bc"` for the Bhattacharyya-coefficient form.

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
    method = _validate_method(method)

    # calculate bin edges
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
        distance = _compute_h_dist(hist1, hist2, method=method)
        distances[dim] = distance

    return distances
