import pytest
import torch

from dlk.metrics.utils import kdedd_on_grid


def test_kdedd_on_grid_matches_histogram_bin_masses() -> None:
    """Compare KDE-implied bin masses on a grid against histogram bin masses."""
    pytest.importorskip("torchkde")

    torch.manual_seed(0)
    n_samples = 20_000
    component_1 = torch.normal(mean=-1.0, std=0.45, size=(n_samples // 2, 1))
    component_2 = torch.normal(mean=1.2, std=0.7, size=(n_samples - n_samples // 2, 1))
    samples = torch.cat((component_1, component_2), dim=0)

    low, high = -4.0, 4.0
    grid_size = 65
    grid_limits = torch.tensor([[low, high]], dtype=samples.dtype)
    kde_density = kdedd_on_grid(
        samples=samples,
        grid_limits=grid_limits,
        grid_size=grid_size,
        bandwidth=0.12,
    )

    assert kde_density.shape == (grid_size,)

    bin_width = (high - low) / (grid_size - 1)
    kde_bin_masses = 0.5 * (kde_density[:-1] + kde_density[1:]) * bin_width
    histogram = torch.histogram(
        samples.squeeze(1),
        bins=grid_size - 1,
        range=(low, high),
        density=False,
    ).hist
    histogram_bin_masses = histogram / samples.shape[0]

    torch.testing.assert_close(
        kde_bin_masses.sum(),
        torch.tensor(1.0, dtype=samples.dtype),
        rtol=0.0,
        atol=2e-2,
    )
    torch.testing.assert_close(
        histogram_bin_masses.sum(),
        torch.tensor(1.0, dtype=samples.dtype),
        rtol=0.0,
        atol=1e-6,
    )

    l1_distance = torch.abs(kde_bin_masses - histogram_bin_masses).sum()
    assert float(l1_distance) < 0.05
