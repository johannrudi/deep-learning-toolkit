"""Unit tests for the spectral normalization helpers in `dlk.nets.spectral_norm`."""

import math
from typing import Literal, cast

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from dlk.nets.spectral_norm import (
    DepthwiseSpectralNorm,
    get_depthwise_spectral_norm,
    get_spectral_norm,
    get_weight_parametrizations,
    set_depthwise_spectral_norm,
    set_spectral_norm,
)

# Input length of the dense matrices, equal to the default DFT length.
DFT_LENGTH = 64
CHANNELS = 8


def _weight_parametrizations(layer: nn.Module) -> parametrize.ParametrizationList:
    """Return the parametrizations of `layer.weight`, asserting there are some."""
    weight_parametrizations = get_weight_parametrizations(layer)
    assert weight_parametrizations is not None, f"{layer!r} has no parametrization"
    return weight_parametrizations


def _dense_matrix(layer: nn.Conv1d, length: int) -> torch.Tensor:
    """Build the matrix of `layer` acting on inputs of `length` by the identity basis.

    Args:
        layer: Convolution without bias.
        length: Input length.

    Returns:
        The matrix of shape ``(C_out * L_out, C_in * length)``.
    """
    size = layer.in_channels * length
    basis = torch.eye(size).reshape(size, layer.in_channels, length)
    with torch.no_grad():
        output = layer(basis)
    return output.reshape(size, -1).T


def _depthwise_conv(
    kernel_size: int,
    stride: int,
    padding_mode: Literal["zeros", "circular"],
    weights: Literal["random", "scaled"],
) -> nn.Conv1d:
    """Build a depthwise convolution without bias, with random or scaled weights.

    Args:
        kernel_size: Kernel size.
        stride: Stride.
        padding_mode: Padding mode, with padding ``kernel_size // 2``.
        weights: ``"random"`` keeps the default initialization; ``"scaled"``
            multiplies each channel by its own factor between 1e-2 and 1e2.

    Returns:
        The convolution.
    """
    torch.manual_seed(0)
    layer = nn.Conv1d(
        CHANNELS,
        CHANNELS,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        padding_mode=padding_mode,
        groups=CHANNELS,
        bias=False,
    )
    if weights == "scaled":
        with torch.no_grad():
            layer.weight.mul_(torch.logspace(-2, 2, CHANNELS).reshape(-1, 1, 1))
    return layer


def test_set_spectral_norm_is_idempotent() -> None:
    """Wrapping twice keeps one parametrization."""
    layer = set_spectral_norm(nn.Conv1d(4, 8, 3))

    assert set_spectral_norm(layer) is layer
    assert len(_weight_parametrizations(layer)) == 1


@pytest.mark.parametrize("weights", ["random", "scaled"])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize(
    "padding_mode, stride", [("circular", 1), ("zeros", 1), ("zeros", 2)]
)
def test_depthwise_spectral_norm_bounds_operator_norm(
    padding_mode: Literal["zeros", "circular"],
    stride: int,
    kernel_size: int,
    weights: Literal["random", "scaled"],
) -> None:
    r"""Bound a depthwise convolution's operator norm by 1, up to the Bernstein cap.

    With circular padding at input length ``dft_length``, the matrix is circulant
    and its singular values are the DFT magnitudes, so the norm is exactly 1.
    With zero padding, the norm stays below $1 / (1 - (k - 1)\pi / n)$.
    """
    layer = _depthwise_conv(kernel_size, stride, padding_mode, weights)
    original_maxima = torch.fft.rfft(layer.weight.detach()[:, 0, :], n=DFT_LENGTH)
    original_maxima = original_maxima.abs().amax(dim=1)
    # Precondition of the fixture: the channel maxima differ, so per-channel
    # division would show in the ratio check below.
    assert original_maxima.max() > 1.1 * original_maxima.min()

    set_depthwise_spectral_norm(layer)
    sigma = torch.linalg.matrix_norm(_dense_matrix(layer, DFT_LENGTH), ord=2)

    if padding_mode == "circular":
        torch.testing.assert_close(sigma, torch.tensor(1.0), atol=1e-5, rtol=0)
    else:
        cap = 1 / (1 - (kernel_size - 1) * math.pi / DFT_LENGTH)
        assert sigma <= cap, f"{sigma=} exceeds the Bernstein cap {cap}"

    # check that one scalar, the largest channel maximum, divides all channels
    original = cast(torch.Tensor, _weight_parametrizations(layer).original)
    ratio = layer.weight.detach() / original.detach()
    torch.testing.assert_close(ratio, ratio.flatten()[0].expand_as(ratio))
    torch.testing.assert_close(
        ratio.flatten()[0], 1 / original_maxima.max(), atol=0, rtol=1e-5
    )

    # check that gradients reach the trainable weight
    layer(torch.randn(2, CHANNELS, DFT_LENGTH)).square().sum().backward()
    assert original.grad is not None and original.grad.abs().sum() > 0


def test_set_depthwise_spectral_norm_is_idempotent() -> None:
    """Wrapping twice keeps one parametrization."""
    layer = set_depthwise_spectral_norm(nn.Conv1d(4, 4, 3, groups=4))

    assert set_depthwise_spectral_norm(layer) is layer
    assert len(_weight_parametrizations(layer)) == 1
    assert isinstance(_weight_parametrizations(layer)[0], DepthwiseSpectralNorm)


@pytest.mark.parametrize(
    "layer",
    [
        nn.Conv1d(4, 8, 3),
        nn.Conv1d(4, 4, 3, groups=2),
        nn.Conv1d(4, 8, 3, groups=4),
        nn.Linear(4, 4),
    ],
    ids=["dense", "grouped", "depth_multiplier", "linear"],
)
def test_set_depthwise_spectral_norm_rejects_dense_conv(layer: nn.Module) -> None:
    """Reject every layer that is not a depthwise ``nn.Conv1d``."""
    with pytest.raises(AssertionError, match="depthwise"):
        set_depthwise_spectral_norm(layer)


def test_depthwise_spectral_norm_rejects_kernel_longer_than_dft() -> None:
    """Reject a kernel that the DFT would truncate."""
    with pytest.raises(AssertionError, match="dft_length"):
        set_depthwise_spectral_norm(nn.Conv1d(4, 4, 5, groups=4), dft_length=4)


def test_get_spectral_norm_and_get_depthwise_spectral_norm_tell_c0_from_c2() -> None:
    """Find neither on a plain layer, and only its own kind on a wrapped one."""
    plain = nn.Conv1d(4, 4, 3, groups=4)
    c0 = set_spectral_norm(nn.Conv1d(4, 4, 3, groups=4))
    c2 = set_depthwise_spectral_norm(nn.Conv1d(4, 4, 3, groups=4))

    assert get_spectral_norm(plain) is None
    assert get_depthwise_spectral_norm(plain) is None
    assert get_depthwise_spectral_norm(c0) is None
    assert get_spectral_norm(c2) is None
