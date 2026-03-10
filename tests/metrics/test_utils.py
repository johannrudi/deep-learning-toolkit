import pytest
import torch

from dlk.metrics import utils


def test_kdedd_on_2d_grid_matches_histogram_bin_masses() -> None:
    """Compare KDE-implied bin masses on a grid against histogram bin masses."""
    pytest.importorskip("torchkde")

    torch.manual_seed(0)
    n_samples = 100_000
    component_1 = torch.normal(mean=-1.0, std=0.45, size=(n_samples // 2, 1))
    component_2 = torch.normal(mean=1.2, std=0.7, size=(n_samples - n_samples // 2, 1))
    samples = torch.cat((component_1, component_2), dim=0)

    grid_size = 65
    low, high = -4.0, 4.0
    grid_limits = [(low, high)]
    kde_masses = utils.kde_density(
        samples=samples,
        grid_size=grid_size,
        grid_limits=grid_limits,
        bandwidth=0.12,
        to_mass=True,
    )
    assert kde_masses.shape == (grid_size,)

    hist_masses = utils.histogramdd(
        samples,
        bin_counts=grid_size,
        bin_ranges=grid_limits,
        to_density=False,
        to_mass=True,
    )

    print()

    kde_total_mass = kde_masses.sum()
    print(f"  {kde_total_mass.item()=}")
    torch.testing.assert_close(
        kde_total_mass,
        torch.tensor(1.0, dtype=samples.dtype),
        rtol=0.0,
        atol=1e-4,
    )
    hist_total_mass = hist_masses.sum()
    print(f"  {hist_total_mass.item()=}")
    torch.testing.assert_close(
        hist_total_mass,
        torch.tensor(1.0, dtype=samples.dtype),
        rtol=0.0,
        atol=1e-4,
    )

    l1_distance = torch.abs(kde_masses - hist_masses).sum()
    print(f"  {l1_distance.item()=}")
    assert float(l1_distance) < 0.05

    print(end="  ")


def test_kde_on_3d_grid_matches_histogram_bin_masses() -> None:
    """Compare 3D KDE-implied bin masses on a grid against histogram bin masses."""
    pytest.importorskip("torchkde")

    torch.manual_seed(1)
    n_samples = 100_000

    mean_1 = torch.tensor([-1.2, 0.6, -0.8])
    std_1 = torch.tensor([0.45, 0.35, 0.55])
    mean_2 = torch.tensor([1.1, -0.9, 1.0])
    std_2 = torch.tensor([0.70, 0.50, 0.40])
    component_1 = torch.randn(n_samples // 2, 3) * std_1 + mean_1
    component_2 = torch.randn(n_samples - n_samples // 2, 3) * std_2 + mean_2
    samples = torch.cat((component_1, component_2), dim=0)

    grid_size = [10, 12, 13]
    grid_limits = [(-4.0, 4.0), (-3.0, 2.5), (-3.5, 3.0)]
    kde_masses = utils.kde_density(
        samples=samples,
        grid_size=grid_size,
        grid_limits=grid_limits,
        bandwidth=0.16,
        to_mass=True,
    )
    assert kde_masses.shape == tuple(grid_size)

    hist_masses = utils.histogramdd(
        samples,
        bin_counts=grid_size,
        bin_ranges=grid_limits,
        to_density=False,
        to_mass=True,
    )

    print()

    kde_total_mass = kde_masses.sum()
    print(f"  {kde_total_mass.item()=}")
    torch.testing.assert_close(
        kde_total_mass,
        torch.tensor(1.0, dtype=samples.dtype),
        rtol=0.0,
        atol=1e-2,
    )
    hist_total_mass = hist_masses.sum()
    print(f"  {hist_total_mass.item()=}")
    torch.testing.assert_close(
        hist_total_mass,
        torch.tensor(1.0, dtype=samples.dtype),
        rtol=0.0,
        atol=1e-4,
    )

    l1_distance = torch.abs(kde_masses - hist_masses).sum()
    print(f"  {l1_distance.item()=}")
    assert float(l1_distance) < 0.50

    print(end="  ")
