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
from typing import Literal, cast

import torch

from dlk.metrics.utils import (
    CountsLike,
    LimitsLike,
    as_floating_2d,
    histogramdd,
    kde_density,
    resolve_counts,
    resolve_limits,
    resolve_points,
)

# --------------------------------------
# Types
# --------------------------------------

HellingerMethod = Literal[
    "direct",
    "direct_si",
    "bc",
]

DensityMethod = Literal[
    "hist",
    "kde",
]

# --------------------------------------

HELLINGER_METHOD_VALUES: set[str] = set(HellingerMethod.__args__)
DENSITY_METHOD_VALUES: set[str] = set(DensityMethod.__args__)


def _validate_hellinger_method(
    hellinger_method: HellingerMethod | str,
) -> HellingerMethod:
    """Validate the Hellinger distance method selector.

    Args:
        hellinger_method: Hellinger distance implementation name.

    Returns:
        The validated method name.

    Raises:
        ValueError: If `hellinger_method` is not one of the supported names.
    """
    if hellinger_method not in HELLINGER_METHOD_VALUES:
        valid_methods = ", ".join(f'"{value}"' for value in HELLINGER_METHOD_VALUES)
        raise ValueError(
            f"hellinger_method must be one of {valid_methods}, got {hellinger_method!r}."
        )
    return cast(HellingerMethod, hellinger_method)


def _validate_density_method(density_method: DensityMethod | str) -> DensityMethod:
    """Validate the density approximation method selector.

    Args:
        density_method: Density approximation implementation name.

    Returns:
        The validated method name.

    Raises:
        ValueError: If `density_method` is not one of the supported names.
    """
    if density_method not in DENSITY_METHOD_VALUES:
        valid_methods = ", ".join(f'"{value}"' for value in DENSITY_METHOD_VALUES)
        raise ValueError(
            f"density_method must be one of {valid_methods}, got {density_method!r}."
        )
    return cast(DensityMethod, density_method)


def _validate_samples(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    samples1 = as_floating_2d(samples=samples1, name="samples1")
    samples2 = as_floating_2d(samples=samples2, name="samples2")

    if samples1.device != samples2.device:
        raise ValueError("samples1 and samples2 must be on the same device.")
    if samples1.shape[1] != samples2.shape[1]:
        raise ValueError(
            "samples1 and samples2 must have the same number of features; "
            f"got {samples1.shape[1]} and {samples2.shape[1]}."
        )
    return samples1, samples2


def _validate_trapezoid_inputs(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
    grid_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and normalize distance inputs.

    Args:
        probmass1: First non-negative mass or density values.
        probmass2: Second non-negative mass or density values.
        grid_points: 1D grid coordinates where values are evaluated.

    Returns:
        Tuple `(probmass1, probmass2, grid_points)` as 1D floating tensors with
        matching dtype.

    Raises:
        ValueError: If devices are inconsistent.
        ValueError: If any input is not one-dimensional.
        ValueError: If inputs do not share the same number of elements.
    """
    if probmass1.device != probmass2.device:
        raise ValueError("probmass1 and probmass2 must be on the same device.")
    if probmass1.device != grid_points.device:
        raise ValueError("grid_points must be on the same device as probmass tensors.")

    if probmass1.ndim != 1:
        raise ValueError(
            f"probmass1 must be one-dimensional; got shape {tuple(probmass1.shape)}."
        )
    if probmass2.ndim != 1:
        raise ValueError(
            f"probmass2 must be one-dimensional; got shape {tuple(probmass2.shape)}."
        )
    if grid_points.ndim != 1:
        raise ValueError(
            f"grid_points must be one-dimensional; got shape {tuple(grid_points.shape)}."
        )

    resolved_probmass1 = probmass1
    resolved_probmass2 = probmass2
    resolved_grid_points = grid_points

    if resolved_probmass1.numel() != resolved_probmass2.numel():
        raise ValueError(
            "probmass1 and probmass2 must have the same number of elements; "
            f"got {resolved_probmass1.numel()} and {resolved_probmass2.numel()}."
        )
    if resolved_grid_points.numel() != resolved_probmass1.numel():
        raise ValueError(
            "grid_points must match the number of probmass values; "
            f"got {resolved_grid_points.numel()} and {resolved_probmass1.numel()}."
        )

    if not torch.is_floating_point(resolved_probmass1):
        resolved_probmass1 = resolved_probmass1.float()
    if not torch.is_floating_point(resolved_probmass2):
        resolved_probmass2 = resolved_probmass2.float()
    if not torch.is_floating_point(resolved_grid_points):
        resolved_grid_points = resolved_grid_points.float()

    resolved_grid_points = resolved_grid_points.to(dtype=resolved_probmass1.dtype)
    resolved_probmass2 = resolved_probmass2.to(dtype=resolved_probmass1.dtype)
    return resolved_probmass1, resolved_probmass2, resolved_grid_points


def _compute_h_dist_direct_method(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute the Hellinger distance directly:

    .. math::
        {\frac {1}{\sqrt {2}}}\;{\bigl \|}{\sqrt {P}}-{\sqrt {Q}}{\bigr \|}_{2}

    Args:
        probmass1: First density-like values.
        probmass2: Second density-like values.

    Returns:
        Scalar tensor for the direct Hellinger distance.
    """
    pm1_sqrt = torch.sqrt(torch.clamp(probmass1, min=0.0))
    pm2_sqrt = torch.sqrt(torch.clamp(probmass2, min=0.0))
    diff = pm1_sqrt - pm2_sqrt
    return torch.linalg.vector_norm(diff) / math.sqrt(2.0)


def _compute_h_dist_direct_method_trapezoid1d(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
    grid_points: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the direct Hellinger distance using trapezoidal integration.

    Args:
        probmass1: First density-like values evaluated on `grid_points`.
        probmass2: Second density-like values evaluated on `grid_points`.
        grid_points: 1D integration coordinates.

    Returns:
        Scalar tensor for the direct Hellinger distance.

    Raises:
        ValueError: If input devices, dimensions, or lengths are incompatible.
    """
    probmass1, probmass2, grid_points = _validate_trapezoid_inputs(
        probmass1=probmass1,
        probmass2=probmass2,
        grid_points=grid_points,
    )
    pm1_sqrt = torch.sqrt(torch.clamp(probmass1, min=0.0))
    pm2_sqrt = torch.sqrt(torch.clamp(probmass2, min=0.0))
    diff_sq = (pm1_sqrt - pm2_sqrt).square()
    integral = torch.trapezoid(diff_sq, x=grid_points)
    return torch.sqrt(torch.clamp(integral / 2.0, min=0.0))


def _compute_h_dist_direct_si_method(
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

    Args:
        probmass1: First density-like values.
        probmass2: Second density-like values.

    Returns:
        Scalar tensor for the scale-invariant direct distance.
    """
    pm1_sqrt = torch.sqrt(torch.clamp(probmass1, min=0.0))
    pm2_sqrt = torch.sqrt(torch.clamp(probmass2, min=0.0))
    n = torch.sum(pm1_sqrt * pm2_sqrt)
    d = torch.sum(pm2_sqrt.square())
    scale = n / d if 0.0 < d else torch.zeros_like(d)
    diff = pm1_sqrt - scale * pm2_sqrt
    return torch.linalg.vector_norm(diff) / math.sqrt(2.0)


def _compute_h_dist_direct_si_method_trapezoid1d(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
    grid_points: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the direct scale-invariant distance using trapezoidal integration.

    Args:
        probmass1: First density-like values evaluated on `grid_points`.
        probmass2: Second density-like values evaluated on `grid_points`.
        grid_points: 1D integration coordinates.

    Returns:
        Scalar tensor for the scale-invariant direct distance.

    Raises:
        ValueError: If input devices, dimensions, or lengths are incompatible.
    """
    probmass1, probmass2, grid_points = _validate_trapezoid_inputs(
        probmass1=probmass1,
        probmass2=probmass2,
        grid_points=grid_points,
    )
    pm1_sqrt = torch.sqrt(torch.clamp(probmass1, min=0.0))
    pm2_sqrt = torch.sqrt(torch.clamp(probmass2, min=0.0))
    n = torch.trapezoid(pm1_sqrt * pm2_sqrt, x=grid_points)
    d = torch.trapezoid(pm2_sqrt.square(), x=grid_points)
    scale = n / d if 0.0 < d else torch.zeros_like(d)
    diff_sq = (pm1_sqrt - scale * pm2_sqrt).square()
    integral = torch.trapezoid(diff_sq, x=grid_points)
    return torch.sqrt(torch.clamp(integral / 2.0, min=0.0))


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

    Args:
        probmass1: First density-like values.
        probmass2: Second density-like values.

    Returns:
        Scalar tensor for the BC-form Hellinger distance.
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


def _compute_h_dist_bc_method_trapezoid1d(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
    grid_points: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the BC-form Hellinger distance using trapezoidal integration.

    Args:
        probmass1: First density-like values evaluated on `grid_points`.
        probmass2: Second density-like values evaluated on `grid_points`.
        grid_points: 1D integration coordinates.

    Returns:
        Scalar tensor for the BC-form Hellinger distance.

    Raises:
        ValueError: If input devices, dimensions, or lengths are incompatible.
    """
    probmass1, probmass2, grid_points = _validate_trapezoid_inputs(
        probmass1=probmass1,
        probmass2=probmass2,
        grid_points=grid_points,
    )
    probmass1 = torch.clamp(probmass1, min=0.0)
    probmass2 = torch.clamp(probmass2, min=0.0)
    totalmass1 = torch.trapezoid(probmass1, x=grid_points)
    totalmass2 = torch.trapezoid(probmass2, x=grid_points)
    n = torch.trapezoid(torch.sqrt(probmass1 * probmass2), x=grid_points)
    d = torch.sqrt(totalmass1 * totalmass2)
    if 0.0 < d:
        bc = n / d
    elif totalmass1 <= 0.0 and totalmass2 <= 0.0:
        bc = torch.ones_like(n)
    else:
        bc = torch.zeros_like(n)
    return torch.sqrt(torch.clamp(1.0 - bc, min=0.0))


@torch.no_grad()
def hellinger_distance_mass(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
    hellinger_method: HellingerMethod,
) -> torch.Tensor:
    """Compute Hellinger distance from discrete mass tensors.

    Args:
        probmass1: First tensor of mass-like values.
        probmass2: Second tensor of mass-like values.
        hellinger_method: Distance variant selector (`"direct"`, `"direct_si"`, or `"bc"`).

    Returns:
        Scalar tensor containing the selected Hellinger distance value.
    """
    if hellinger_method == "direct_si":
        return _compute_h_dist_direct_si_method(probmass1, probmass2)
    if hellinger_method == "bc":
        return _compute_h_dist_bc_method(probmass1, probmass2)
    assert hellinger_method == "direct"
    return _compute_h_dist_direct_method(probmass1, probmass2)


@torch.no_grad()
def hellinger_distance_mass_trapezoid1d(
    probmass1: torch.Tensor,
    probmass2: torch.Tensor,
    counts: CountsLike,
    limits: LimitsLike,
    hellinger_method: HellingerMethod,
) -> torch.Tensor:
    """Compute 1D Hellinger distance with trapezoidal integration.

    Args:
        probmass1: First tensor of 1D mass-like values sampled on a grid.
        probmass2: Second tensor of 1D mass-like values sampled on a grid.
        counts: Number of grid points as an int or per-feature sequence.
        limits: Global `(min, max)` limits or per-feature limits for the grid.
        hellinger_method: Distance variant selector (`"direct"`, `"direct_si"`, or `"bc"`).

    Returns:
        Scalar tensor containing the selected Hellinger distance value.

    Raises:
        ValueError: If counts or limits are invalid for grid construction.
        ValueError: If trapezoid inputs have incompatible devices, dimensions, or lengths.
    """
    assert 1 == probmass1.ndim
    assert 1 == probmass2.ndim
    resolved_grid_points = resolve_points(
        samples1=probmass1.unsqueeze(1),
        samples2=None,
        counts=counts,
        limits=limits,
        counts_are_intervals=False,
    )
    assert 1 == len(resolved_grid_points)
    grid_points = resolved_grid_points[0]

    if hellinger_method == "direct_si":
        return _compute_h_dist_direct_si_method_trapezoid1d(
            probmass1, probmass2, grid_points=grid_points
        )
    if hellinger_method == "bc":
        return _compute_h_dist_bc_method_trapezoid1d(
            probmass1, probmass2, grid_points=grid_points
        )
    assert hellinger_method == "direct"
    return _compute_h_dist_direct_method_trapezoid1d(
        probmass1, probmass2, grid_points=grid_points
    )


@torch.no_grad()
def hellinger_distance_samples(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    counts: CountsLike | None = None,
    limits: LimitsLike | None = None,
    hellinger_method: HellingerMethod = "direct",
    density_method: DensityMethod = "hist",
) -> torch.Tensor:
    r"""Estimate Hellinger distance between two sample sets.

    Args:
        samples1: First sample tensor with shape `(n_samples_1, n_features)`.
        samples2: Second sample tensor with shape `(n_samples_2, n_features)`.
        counts: Number of bins (`density_method="hist"`) or grid points
            (`density_method="kde"`) as an int or per-feature sequence.
        limits: Histogram range (`density_method="hist"`) of grid limits
            (`density_method="kde"`) as a global `(min, max)` tuple or
            per-feature `(min, max)` tuples.
            When omitted, the range is inferred from both sample sets.
        hellinger_method: Distance implementation. Use `"direct"` for
            :math:`\frac{1}{\sqrt{2}}\lVert\sqrt{P}-\sqrt{Q}\rVert_2`,
            `"direct_si"` for the scale-invariant direct method, or
            `"bc"` for the Bhattacharyya-coefficient form.
        density_method: Density approximation backend. Use `"hist"` for
            histogram-based masses or `"kde"` for KDE-based masses.

    Returns:
        Scalar tensor containing the estimated Hellinger distance.

    Raises:
        ValueError: If sample shapes/devices are incompatible or histogram settings are invalid.
        ValueError: If `hellinger_method` or `density_method` is invalid.
        ImportError: If `density_method="kde"` and `torchkde` is not installed.
    """
    samples1, samples2 = _validate_samples(samples1, samples2)
    hellinger_method = _validate_hellinger_method(hellinger_method)
    density_method = _validate_density_method(density_method)

    # compute one multi-dimensional histogram over the entire feature space
    if density_method == "kde":
        probmass1 = kde_density(
            samples1, grid_size=counts, grid_limits=limits, to_mass=True
        )
        probmass2 = kde_density(
            samples2, grid_size=counts, grid_limits=limits, to_mass=True
        )
    else:
        assert density_method == "hist"
        probmass1 = histogramdd(
            samples1, bin_counts=counts, bin_ranges=limits, to_mass=True
        )
        probmass2 = histogramdd(
            samples2, bin_counts=counts, bin_ranges=limits, to_mass=True
        )
    distance = hellinger_distance_mass(
        probmass1, probmass2, hellinger_method=hellinger_method
    )

    return distance


@torch.no_grad()
def marginal_hellinger_distances_samples(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    counts: CountsLike | None = None,
    limits: LimitsLike | None = None,
    hellinger_method: HellingerMethod = "direct",
    density_method: DensityMethod = "hist",
) -> torch.Tensor:
    r"""Estimate per-feature 1D Hellinger distances from sample marginals.

    Args:
        samples1: First sample tensor with shape `(n_samples_1, n_features)`.
        samples2: Second sample tensor with shape `(n_samples_2, n_features)`.
        counts: Number of bins (`density_method="hist"`) or grid points
            (`density_method="kde"`) as an int or per-feature sequence.
        limits: Histogram range (`density_method="hist"`) of grid limits
            (`density_method="kde"`) as a global `(min, max)` tuple or
            per-feature `(min, max)` tuples.
            When omitted, the range is inferred from both sample sets.
        hellinger_method: Distance implementation. Use `"direct"` for
            :math:`\frac{1}{\sqrt{2}}\lVert\sqrt{P}-\sqrt{Q}\rVert_2`,
            `"direct_si"` for the scale-invariant direct method, or
            `"bc"` for the Bhattacharyya-coefficient form.
        density_method: Density approximation backend. Use `"hist"` for
            histogram-based masses or `"kde"` for KDE-based masses.

    Returns:
        Tensor with shape `(n_features,)` containing one 1D Hellinger distance per feature.

    Raises:
        ValueError: If sample shapes/devices are incompatible or histogram settings are invalid.
        ValueError: If `hellinger_method` or `density_method` is invalid.
        ImportError: If `density_method="kde"` and `torchkde` is not installed.
    """
    samples1, samples2 = _validate_samples(samples1, samples2)
    hellinger_method = _validate_hellinger_method(hellinger_method)
    density_method = _validate_density_method(density_method)

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
    distances = torch.empty(n_features, device=samples1.device, dtype=samples1.dtype)

    # compute one-dimensional histograms for each feature independently
    for dim, (cnt, lim) in enumerate(
        zip(resolved_counts, resolved_limits, strict=True)
    ):
        if density_method == "kde":
            probmass1 = kde_density(
                samples1[:, [dim]], grid_size=cnt, grid_limits=lim, to_mass=False
            )
            probmass2 = kde_density(
                samples2[:, [dim]], grid_size=cnt, grid_limits=lim, to_mass=False
            )
            distance = hellinger_distance_mass_trapezoid1d(
                probmass1,
                probmass2,
                counts=cnt,
                limits=lim,
                hellinger_method=hellinger_method,
            )
        else:
            assert density_method == "hist"
            probmass1 = histogramdd(
                samples1[:, [dim]], bin_counts=cnt, bin_ranges=lim, to_mass=True
            )
            probmass2 = histogramdd(
                samples2[:, [dim]], bin_counts=cnt, bin_ranges=lim, to_mass=True
            )
            distance = hellinger_distance_mass(
                probmass1, probmass2, hellinger_method=hellinger_method
            )
        distances[dim] = distance

    return distances
