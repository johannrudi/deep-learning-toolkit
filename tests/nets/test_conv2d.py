"""Tests for 2D convolutional network building blocks."""

import torch
import torch.nn as nn

from dlk.nets.conv2d import (
    ConvUpsampleNet_Interpolate,
    ConvUpsampleNet_Reshuffle,
    Downsample,
    LevelBlock,
    Upsample,
)


def test_conv_upsample_net_reshuffle_forward_output_shape() -> None:
    """Validate output shape for ``ConvUpsampleNet_Reshuffle``."""
    net = ConvUpsampleNet_Reshuffle(1, 2)
    x = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])

    y = net(x)

    assert y.shape == (1, 1, 4, 4)


def test_conv_upsample_net_interpolate_forward_output_shape() -> None:
    """Validate output shape for ``ConvUpsampleNet_Interpolate``."""
    net = ConvUpsampleNet_Interpolate(1)
    x = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])

    y = net(x)

    assert y.shape == (1, 1, 4, 4)


def test_upsample_forward_output_shape() -> None:
    """Validate output shape for ``Upsample``."""
    layer = Upsample(1, 1, 3)
    row = 4 * [1.0]
    x = torch.tensor([[[row for _ in range(4)]]])

    y = layer(x)

    assert y.shape == (1, 1, 8, 8)


def test_downsample_forward_output_shape() -> None:
    """Validate output shape for ``Downsample``."""
    layer = Downsample(1, 1, 3)
    row = 8 * [1.0]
    x = torch.tensor([[[row for _ in range(8)]]])

    y = layer(x)

    assert y.shape == (1, 1, 4, 4)


def test_level_block_forward_output_shape() -> None:
    """Validate output shape for ``LevelBlock``."""
    layer = LevelBlock(1, 1, 3, activation=nn.ReLU())
    row = 8 * [1.0]
    x = torch.tensor([[[row for _ in range(8)]]])

    y = layer(x)

    assert y.shape == (1, 1, 8, 8)
