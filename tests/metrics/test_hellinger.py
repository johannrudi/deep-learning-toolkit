import importlib.util
import math
from typing import cast

import pytest
import torch

from dlk.metrics.hellinger import (
    DensityMethod,
    HellingerMethod,
    hellinger_distance_samples,
    marginal_hellinger_distances_samples,
)

_HAS_TORCHKDE = importlib.util.find_spec("torchkde") is not None
_HELLINGER_METHODS: list[HellingerMethod] = ["direct", "direct_si", "bc"]
_DENSITY_METHODS: list[DensityMethod] = ["hist", "kde"]


def _skip_if_torchkde_is_unavailable(density_method: DensityMethod) -> None:
    if density_method == "kde" and not _HAS_TORCHKDE:
        pytest.skip("requires optional dependency 'torchkde'")


@pytest.mark.parametrize("hellinger_method", _HELLINGER_METHODS)
@pytest.mark.parametrize("density_method", _DENSITY_METHODS)
def test_hellinger_distance_hist_returns_zero_for_identical_inputs(
    hellinger_method: HellingerMethod,
    density_method: DensityMethod,
) -> None:
    """Return zero distance when the two input sample sets are identical."""
    _skip_if_torchkde_is_unavailable(density_method)
    samples = torch.tensor(
        [
            [0.00, 0.00],
            [0.25, 0.25],
            [0.50, 0.50],
            [0.75, 0.75],
            [1.00, 1.00],
        ],
        dtype=torch.float32,
    )

    distance = hellinger_distance_samples(
        samples1=samples,
        samples2=samples.clone(),
        counts=3,
        limits=(0.0, 1.0),
        hellinger_method=hellinger_method,
        density_method=density_method,
    )

    torch.testing.assert_close(
        distance,
        torch.tensor(0.0, dtype=samples.dtype),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("hellinger_method", ["direct", "bc"])
@pytest.mark.parametrize("density_method", ["hist"])  # TODO: "kde" fails this test
def test_hellinger_distance_hist_matches_analytic_uniform_distance(
    hellinger_method: HellingerMethod,
    density_method: DensityMethod,
) -> None:
    """Approximate the analytic distance between U(0, 1) and U(0, 2)."""
    n_samples = 20_000
    base_grid = (torch.arange(n_samples, dtype=torch.float64) + 0.5) / n_samples
    samples1 = base_grid.unsqueeze(1)
    samples2 = (2.0 * base_grid).unsqueeze(1)

    estimated_distance = hellinger_distance_samples(
        samples1=samples1,
        samples2=samples2,
        counts=2,
        limits=(0.0, 2.0),
        hellinger_method=hellinger_method,
        density_method=density_method,
    )
    true_distance = torch.tensor(
        math.sqrt(1.0 - (1.0 / math.sqrt(2.0))),
        dtype=torch.float64,
    )

    torch.testing.assert_close(estimated_distance, true_distance, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("density_method", ["hist"])  # TODO: "kde" fails this test
def test_hellinger_distance_hist_scale_invariant_for_scaled_uniform_inputs(
    density_method: DensityMethod,
) -> None:
    """Reduce distance when one input distribution is a scaled version of the other."""
    n_samples = 20_000
    base_grid = (torch.arange(n_samples, dtype=torch.float64) + 0.5) / n_samples
    samples1 = base_grid.unsqueeze(1)
    samples2 = (2.0 * base_grid).unsqueeze(1)

    standard_distance = hellinger_distance_samples(
        samples1=samples1,
        samples2=samples2,
        counts=2,
        limits=(0.0, 2.0),
        hellinger_method="direct",
        density_method=density_method,
    )
    scale_invariant_distance = hellinger_distance_samples(
        samples1=samples1,
        samples2=samples2,
        counts=2,
        limits=(0.0, 2.0),
        hellinger_method="direct_si",
        density_method=density_method,
    )

    torch.testing.assert_close(
        scale_invariant_distance,
        torch.tensor(0.5, dtype=torch.float64),
        rtol=0.0,
        atol=1e-10,
    )
    assert scale_invariant_distance < standard_distance


def test_hellinger_distance_hist_raises_for_unknown_method() -> None:
    """Raise ValueError when hellinger_method is not one of the supported names."""
    samples = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32)
    invalid_method = cast(HellingerMethod, "unsupported")

    with pytest.raises(ValueError, match="hellinger_method must be one of"):
        hellinger_distance_samples(
            samples1=samples,
            samples2=samples.clone(),
            counts=4,
            hellinger_method=invalid_method,
        )


@pytest.mark.parametrize("hellinger_method", _HELLINGER_METHODS)
@pytest.mark.parametrize("density_method", _DENSITY_METHODS)
def test_hellinger_distance_hist_marginals_returns_zero_for_identical_inputs(
    hellinger_method: HellingerMethod,
    density_method: DensityMethod,
) -> None:
    """Return zero distance for each feature when the two input sample sets are identical."""
    _skip_if_torchkde_is_unavailable(density_method)
    samples = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.25, 0.25, 0.25],
            [0.50, 0.50, 0.50],
            [0.75, 0.75, 0.75],
            [1.00, 1.00, 1.00],
        ],
        dtype=torch.float32,
    )

    distances = marginal_hellinger_distances_samples(
        samples1=samples,
        samples2=samples.clone(),
        counts=5,
        limits=(0.0, 1.0),
        hellinger_method=hellinger_method,
        density_method=density_method,
    )

    torch.testing.assert_close(
        distances,
        torch.zeros(samples.shape[1], dtype=samples.dtype),
    )


@pytest.mark.parametrize("hellinger_method", _HELLINGER_METHODS)
@pytest.mark.parametrize("density_method", ["hist"])  # TODO: "kde" fails this test
def test_hellinger_distance_hist_marginals_matches_per_feature_hist_distance(
    hellinger_method: HellingerMethod,
    density_method: DensityMethod,
) -> None:
    """Match stacked 1D distances computed by the base histogram Hellinger function."""
    samples1 = torch.tensor(
        [
            [0.10, -0.20, 12.0],
            [0.25, 0.10, 14.0],
            [0.40, 0.80, 18.0],
            [0.70, 1.20, 24.0],
            [0.90, 1.60, 26.0],
        ],
        dtype=torch.float32,
    )
    samples2 = torch.tensor(
        [
            [0.05, -0.10, 11.0],
            [0.15, 0.30, 13.0],
            [0.30, 0.40, 16.0],
            [0.50, 1.00, 20.0],
            [0.80, 1.80, 29.0],
        ],
        dtype=torch.float32,
    )
    histogram_bins = [6, 5, 8]
    histogram_range = [(0.0, 1.0), (-0.5, 2.0), (10.0, 30.0)]

    distances = marginal_hellinger_distances_samples(
        samples1=samples1,
        samples2=samples2,
        counts=histogram_bins,
        limits=histogram_range,
        hellinger_method=hellinger_method,
        density_method=density_method,
    )
    expected = torch.stack(
        [
            hellinger_distance_samples(
                samples1=samples1[:, [dim]],
                samples2=samples2[:, [dim]],
                counts=histogram_bins[dim],
                limits=histogram_range[dim],
                hellinger_method=hellinger_method,
                density_method=density_method,
            )
            for dim in range(3)
        ]
    )

    torch.testing.assert_close(distances, expected)


def test_hellinger_distance_hist_marginals_raises_for_unknown_method() -> None:
    """Raise ValueError when marginals hellinger_method is not one of the supported names."""
    samples = torch.tensor(
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
        dtype=torch.float32,
    )
    invalid_method = cast(HellingerMethod, "unsupported")

    with pytest.raises(ValueError, match="hellinger_method must be one of"):
        marginal_hellinger_distances_samples(
            samples1=samples,
            samples2=samples.clone(),
            counts=4,
            hellinger_method=invalid_method,
        )
