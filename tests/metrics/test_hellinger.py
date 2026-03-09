import math

import torch

from dlk.metrics.hellinger import (
    hellinger_distance_hist,
    hellinger_distance_hist_marginals,
)


def test_hellinger_distance_hist_returns_zero_for_identical_inputs() -> None:
    """Return zero distance when the two input sample sets are identical."""
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

    distance = hellinger_distance_hist(
        samples1=samples,
        samples2=samples.clone(),
        hist_bins=3,
        hist_range=(0.0, 1.0),
    )

    torch.testing.assert_close(distance, torch.tensor(0.0, dtype=samples.dtype))


def test_hellinger_distance_hist_matches_analytic_uniform_distance() -> None:
    """Approximate the analytic distance between U(0, 1) and U(0, 2)."""
    n_samples = 20_000
    base_grid = (torch.arange(n_samples, dtype=torch.float64) + 0.5) / n_samples
    samples1 = base_grid.unsqueeze(1)
    samples2 = (2.0 * base_grid).unsqueeze(1)

    estimated_distance = hellinger_distance_hist(
        samples1=samples1,
        samples2=samples2,
        hist_bins=2,
        hist_range=(0.0, 2.0),
    )
    true_distance = torch.tensor(
        math.sqrt(1.0 - (1.0 / math.sqrt(2.0))),
        dtype=torch.float64,
    )

    torch.testing.assert_close(estimated_distance, true_distance, rtol=0.0, atol=1e-10)


def test_hellinger_distance_hist_scale_invariant_for_scaled_uniform_inputs() -> None:
    """Reduce distance when one input distribution is a scaled version of the other."""
    n_samples = 20_000
    base_grid = (torch.arange(n_samples, dtype=torch.float64) + 0.5) / n_samples
    samples1 = base_grid.unsqueeze(1)
    samples2 = (2.0 * base_grid).unsqueeze(1)

    standard_distance = hellinger_distance_hist(
        samples1=samples1,
        samples2=samples2,
        hist_bins=2,
        hist_range=(0.0, 2.0),
        scale_invariant=False,
    )
    scale_invariant_distance = hellinger_distance_hist(
        samples1=samples1,
        samples2=samples2,
        hist_bins=2,
        hist_range=(0.0, 2.0),
        scale_invariant=True,
    )

    torch.testing.assert_close(
        scale_invariant_distance,
        torch.tensor(0.5, dtype=torch.float64),
        rtol=0.0,
        atol=1e-10,
    )
    assert scale_invariant_distance < standard_distance


def test_hellinger_distance_hist_marginals_returns_zero_for_identical_inputs() -> None:
    """Return zero distance for each feature when the two input sample sets are identical."""
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

    distances = hellinger_distance_hist_marginals(
        samples1=samples,
        samples2=samples.clone(),
        hist_bins=5,
        hist_range=(0.0, 1.0),
    )

    torch.testing.assert_close(
        distances,
        torch.zeros(samples.shape[1], dtype=samples.dtype),
    )


def test_hellinger_distance_hist_marginals_matches_per_feature_hist_distance() -> None:
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

    distances = hellinger_distance_hist_marginals(
        samples1=samples1,
        samples2=samples2,
        hist_bins=histogram_bins,
        hist_range=histogram_range,
    )
    expected = torch.stack(
        [
            hellinger_distance_hist(
                samples1=samples1[:, [0]],
                samples2=samples2[:, [0]],
                hist_bins=histogram_bins[0],
                hist_range=histogram_range[0],
            ),
            hellinger_distance_hist(
                samples1=samples1[:, [1]],
                samples2=samples2[:, [1]],
                hist_bins=histogram_bins[1],
                hist_range=histogram_range[1],
            ),
            hellinger_distance_hist(
                samples1=samples1[:, [2]],
                samples2=samples2[:, [2]],
                hist_bins=histogram_bins[2],
                hist_range=histogram_range[2],
            ),
        ]
    )

    torch.testing.assert_close(distances, expected)
