"""Tests for UNet model variants and encoder/decoder wrappers."""

import torch

from dlk.nets.unet import (
    DecoderNet1d_2021,
    EncoderNet1d_2021,
    UNet1d_2021,
    UNet2d_2021,
    UNet2d_2021_idd,
    UNet2d_2025,
)


def test_unet2d_2025_forward_output_shape() -> None:
    """Validate output shape for ``UNet2d_2025``."""
    net = UNet2d_2025(
        1,
        1,
        down_levels_conv_channels=[[2, 2], [4, 4], [8, 8]],
        up_levels_conv_channels=[[8, 8], [4, 4], [2, 2]],
    )
    x = torch.ones((1, 1, 16, 16))

    y = net(x)

    assert y.shape == (1, 1, 16, 16)


def test_unet1d_2021_forward_output_shape() -> None:
    """Validate output shape for ``UNet1d_2021``."""
    net = UNet1d_2021(1, 1, channel_mult=(1, 2, 4))
    x = torch.ones((1, 1, 16))

    y = net(x)

    assert y.shape == (1, 1, 16)


def test_unet2d_2021_forward_output_shape() -> None:
    """Validate output shape for ``UNet2d_2021``."""
    net = UNet2d_2021(1, 1, channel_mult=(1, 2, 4))
    x = torch.ones((1, 1, 16, 16))

    y = net(x)

    assert y.shape == (1, 1, 16, 16)


def test_unet2d_2021_idd_forward_output_shape() -> None:
    """Validate output shape for ``UNet2d_2021_idd`` with timestep input."""
    net = UNet2d_2021_idd(1, 1, channel_mult=(1, 2, 4))
    x = torch.ones((1, 1, 16, 16))
    t = torch.tensor([0.1])

    y = net(x, t)

    assert y.shape == (1, 1, 16, 16)


def test_encodernet1d_2021_forward_output_shape() -> None:
    """Validate output shape for ``EncoderNet1d_2021``."""
    net = EncoderNet1d_2021(1, 1, channel_mult=(1, 1, 1))
    x = torch.ones((1, 1, 16))

    y = net(x)

    assert y.shape == (1, 1, 4)


def test_decodernet1d_2021_forward_output_shape() -> None:
    """Validate output shape for ``DecoderNet1d_2021``."""
    net = DecoderNet1d_2021(1, 1, channel_mult=(1, 1, 1))
    x = torch.ones((1, 1, 4))

    y = net(x)

    assert y.shape == (1, 1, 16)
