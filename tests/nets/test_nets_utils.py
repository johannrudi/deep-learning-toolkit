"""Unit tests for the layer initialization in `dlk.nets.utils`."""

import pytest
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from dlk.nets.utils import set_init_parameters


def _weight_orig(layer: nn.Module) -> torch.Tensor:
    """Return the trainable weight that `parametrizations.spectral_norm` normalizes."""
    parametrizations = layer.parametrizations
    assert isinstance(parametrizations, nn.ModuleDict)
    weight_orig = parametrizations["weight"].original
    assert isinstance(weight_orig, torch.Tensor)
    return weight_orig.detach()


@pytest.mark.parametrize(
    ("layer", "input_shape"),
    [
        (nn.Linear(8, 16), (2, 8)),
        (nn.Linear(16, 8), (2, 16)),
        (nn.Conv1d(4, 6, 3), (2, 4, 10)),
    ],
)
def test_set_init_parameters_spectral_norm_is_orthogonal_and_survives_forward(
    layer: nn.Module, input_shape: tuple[int, ...]
) -> None:
    """A spectral-norm layer gets an orthogonal weight that its normalization keeps."""
    torch.manual_seed(0)
    layer = spectral_norm(layer)

    set_init_parameters(layer, gain=nn.init.calculate_gain("relu"))

    singular_values = torch.linalg.svdvals(_weight_orig(layer).flatten(1))
    torch.testing.assert_close(singular_values, torch.ones_like(singular_values))
    # a training forward reruns the power iteration and recomputes the weight
    layer.train()
    layer(torch.randn(input_shape))
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    torch.testing.assert_close(weight.detach(), _weight_orig(layer))


def test_set_init_parameters_spectral_norm_syncs_power_iteration() -> None:
    """Re-initialization refreshes `u` and `v`, so eval sees the new weight."""
    torch.manual_seed(0)
    layer = spectral_norm(nn.Linear(8, 16))
    with torch.no_grad():
        _weight_orig(layer).mul_(10.0)

    set_init_parameters(layer)

    # eval skips the power iteration and relies on the synced `u` and `v`
    layer.eval()
    layer(torch.randn(2, 8))
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    sigma = torch.linalg.matrix_norm(weight.detach(), ord=2)
    torch.testing.assert_close(sigma, torch.tensor(1.0))


def test_set_init_parameters_plain_layer_keeps_xavier_scale() -> None:
    """A plain layer still follows Xavier's gain-scaled uniform range."""
    layer = nn.Linear(8, 16)
    gain = 2.0

    set_init_parameters(layer, gain=gain)

    limit = gain * (6.0 / (8 + 16)) ** 0.5
    assert layer.weight.abs().max() <= limit
    assert layer.weight.abs().max() > 0.5 * limit
