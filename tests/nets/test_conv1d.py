from typing import cast

import pytest
import torch
import torch.nn as nn

from dlk.nets.conv1d import (
    ChannelLayerNorm,
    ConvNet,
    ConvNeXtBlock,
    ConvResNet,
    UniversalMultiLevelBlock,
)


def test_convnet_forward_output_shape() -> None:
    """Run a forward pass for ``ConvNet`` and validate output shape."""
    input_channels = 1
    input_size = 16
    hidden_conv_layers_channels_mult = [2, 4, 8]
    hidden_dense_input_size = (
        input_size - 2 * len(hidden_conv_layers_channels_mult)
    ) * hidden_conv_layers_channels_mult[-1]
    net = ConvNet(
        input_channels=input_channels,
        output_size=2,
        hidden_conv_layers_channels_mult=hidden_conv_layers_channels_mult,
        hidden_dense_input_size=hidden_dense_input_size,
        hidden_dense_layers_sizes=[32, 32],
    )
    x = torch.ones((1, input_channels, input_size))

    y = net(x)

    assert y.shape == (1, 2)


def test_convresnet_forward_without_mlp_head() -> None:
    """Validate ``ConvResNet`` output shape when only conv blocks are enabled."""
    batch_size = 4
    input_channels = 1
    input_length = 64
    channels_mult = [4, 8]
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": channels_mult,
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
    )
    x = torch.randn(batch_size, input_channels, input_length)
    expected_out_size = (
        batch_size,
        input_channels * channels_mult[-1],
        input_length // 4,
    )

    y = net(x)

    assert y.shape == expected_out_size


def test_convresnet_forward_with_mlp_head() -> None:
    """Validate ``ConvResNet`` output shape when an ``MLPResNet`` head is enabled."""
    batch_size = 4
    input_channels = 1
    input_length = 64
    output_size = 10
    channels_mult = [4, 8]
    conv_out_size = (
        batch_size,
        input_channels * channels_mult[-1],
        input_length // 4,
    )
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": channels_mult,
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
        mlp_resnet_params={
            "input_size": conv_out_size[1] * conv_out_size[2],
            "output_size": output_size,
            "residual_blocks_sizes": [(16, 16, 64, 16)],
        },
    )
    x = torch.randn(batch_size, input_channels, input_length)

    y = net(x)

    assert y.shape == (batch_size, output_size)


def test_convresnet_forward_with_hidden_inputs() -> None:
    """Validate ``ConvResNet`` output shape when hidden inputs are passed to MLP blocks."""
    batch_size = 4
    input_channels = 1
    input_length = 64
    output_size = 10
    hidden_input_size = 8
    channels_mult = [4, 8]
    conv_out_size = (
        batch_size,
        input_channels * channels_mult[-1],
        input_length // 4,
    )
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": channels_mult,
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
        mlp_resnet_params={
            "input_size": conv_out_size[1] * conv_out_size[2],
            "output_size": output_size,
            "residual_blocks_sizes": [
                (16, 16, 64, 16),
                (16 + hidden_input_size, 16, 64, 16),
            ],
        },
    )
    x = torch.randn(batch_size, input_channels, input_length)
    h1 = torch.randn(batch_size, 1, hidden_input_size)

    y = net(x, h1=h1)

    assert y.shape == (batch_size, output_size)


def test_multilevel_block_forward_shapes_without_scaling() -> None:
    """Validate ``UniversalMultiLevelBlock`` shapes for level blocks without resizing."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    x = torch.randn(batch_size, input_channels, input_length)

    net = UniversalMultiLevelBlock(input_channels=input_channels, kernel_size=3)
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=False,
        activation=False,
    )
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)

    output_channels = 32
    normalization_channels = 8
    activation_channels = 64
    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=lambda num_channels: nn.GroupNorm(2, num_channels),
        normalization_channels=normalization_channels,
        activation=nn.SiLU(),
        activation_channels=activation_channels,
        output_channels=output_channels,
    )
    y = net(x)
    assert y.shape == (batch_size, output_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        output_channels=output_channels,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == (batch_size, output_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        dropout=0.1,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)

    # concatenate two inputs along the channels
    net = UniversalMultiLevelBlock(input_channels=input_channels, kernel_size=3)
    y = net(x[:, :4], x[:, 4:])
    assert y.shape == (batch_size, input_channels, input_length)


@pytest.mark.parametrize(
    "kernel_size, conv_kwargs, scale_factor, expected_length",
    [
        (3, {"padding": 0}, None, 62),
        (3, {"padding": 0}, 0.5, 31),
        (4, None, None, 64),
        (4, None, 0.5, 32),
        (3, {"dilation": 2}, 0.5, 32),
        (3, None, 1 / 3, 22),
        (3, None, 1.5, 96),
    ],
)
def test_multilevel_block_skip_matches_main_branch_length(
    kernel_size: int,
    conv_kwargs: dict | None,
    scale_factor: float | None,
    expected_length: int,
) -> None:
    """Validate that the skip branch follows the length of the main branch."""
    x = torch.randn(2, 8, 64)
    net = UniversalMultiLevelBlock(
        input_channels=8,
        kernel_size=kernel_size,
        output_channels=16,
        scale_factor=scale_factor,
        skip_connection=True,
        conv_kwargs=conv_kwargs,
    )
    y = net(x)
    assert y.shape == (2, 16, expected_length)


def test_multilevel_block_scale_factor_and_stride() -> None:
    """Validate the coupling of ``scale_factor`` and the stride."""
    net = UniversalMultiLevelBlock(
        input_channels=8, kernel_size=3, scale_factor=0.5, conv_kwargs={"stride": 2}
    )
    assert net.scale_factor == 0.5
    net = UniversalMultiLevelBlock(
        input_channels=8, kernel_size=3, conv_kwargs={"stride": 4}
    )
    assert net.scale_factor == 0.25
    with pytest.raises(ValueError, match="requires stride"):
        UniversalMultiLevelBlock(
            input_channels=8, kernel_size=3, scale_factor=0.5, conv_kwargs={"stride": 3}
        )
    with pytest.raises(ValueError, match="inverse of an integer"):
        UniversalMultiLevelBlock(input_channels=8, kernel_size=3, scale_factor=0.3)
    with pytest.raises(ValueError, match="must be positive"):
        UniversalMultiLevelBlock(input_channels=8, kernel_size=3, scale_factor=0.0)


def test_multilevel_block_starts_as_skip_branch() -> None:
    """Validate that the main branch starts at zero with a skip branch."""
    x = torch.randn(2, 8, 64)
    net = UniversalMultiLevelBlock(
        input_channels=8, kernel_size=3, skip_connection=True, skip_scale=0.5
    )
    torch.testing.assert_close(net(x), 0.5 * x)

    net = UniversalMultiLevelBlock(
        input_channels=8, kernel_size=3, activation=False, skip_connection=True
    )
    torch.testing.assert_close(net(x), x)


def test_multilevel_block_rejects_channels_without_operation() -> None:
    """Validate that channels of a disabled operation raise an error."""
    with pytest.raises(ValueError, match="normalization_channels"):
        UniversalMultiLevelBlock(
            input_channels=16,
            kernel_size=3,
            normalization=False,
            normalization_channels=8,
        )
    with pytest.raises(ValueError, match="activation_channels"):
        UniversalMultiLevelBlock(
            input_channels=16,
            kernel_size=3,
            activation=False,
            activation_channels=64,
        )


def test_multilevel_block_forward_shapes_with_downsampling() -> None:
    """Validate ``UniversalMultiLevelBlock`` shapes when downsampling is enabled."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    expected_out_size = (batch_size, input_channels, input_length // 2)
    x = torch.randn(batch_size, input_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=0.5,
        normalization=False,
        activation=False,
    )
    y = net(x)
    assert y.shape == expected_out_size

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        skip_connection=True,
        conv_kwargs={"stride": 2},
    )
    y = net(x)
    assert y.shape == expected_out_size

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=0.5,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == expected_out_size


def test_multilevel_block_forward_shapes_with_upsampling() -> None:
    """Validate ``UniversalMultiLevelBlock`` shapes when upsampling is enabled."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    expected_out_size = (batch_size, input_channels, input_length * 2)
    x = torch.randn(batch_size, input_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=2.0,
        normalization=False,
        activation=False,
    )
    y = net(x)
    assert y.shape == expected_out_size

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=2.0,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == expected_out_size


def test_channel_layer_norm_normalizes_each_position_over_channels() -> None:
    """Validate ``ChannelLayerNorm`` against statistics over the channels."""
    x = torch.randn(4, 8, 32)
    norm = ChannelLayerNorm(8)
    y = norm(x)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    torch.testing.assert_close(y, (x - mean) / torch.sqrt(var + norm.eps))


@pytest.mark.filterwarnings("ignore:Using padding='same' with even kernel:UserWarning")
@pytest.mark.parametrize("kernel_size", [7, 4])
def test_convnext_block_keeps_input_shape(kernel_size: int) -> None:
    """Validate that ``ConvNeXtBlock`` keeps channels and length."""
    x = torch.randn(4, 16, 63)
    net = ConvNeXtBlock(input_channels=16, kernel_size=kernel_size)
    y = net(x)
    assert y.shape == x.shape


def test_convnext_block_layers() -> None:
    """Validate the layers of ``ConvNeXtBlock`` and that it starts as the identity."""
    x = torch.randn(4, 16, 64)
    net = ConvNeXtBlock(input_channels=16, conv_kwargs={"padding_mode": "replicate"})
    conv_0 = cast(nn.Conv1d, net.block.conv_0)
    assert (conv_0.groups, conv_0.padding_mode) == (16, "replicate")
    assert isinstance(net.block.normalization, ChannelLayerNorm)
    assert isinstance(net.block.activation, nn.GELU)
    assert cast(nn.Conv1d, net.block.conv_1).out_channels == 64
    assert isinstance(net.skip_connection, nn.Identity)
    torch.testing.assert_close(net(x), x)


def test_multilevel_block_drop_path() -> None:
    """Validate that drop path keeps or drops the main branch per sample."""
    torch.manual_seed(0)
    x = torch.randn(64, 16, 32)
    net = ConvNeXtBlock(input_channels=16, drop_path=0.5)
    nn.init.normal_(cast(nn.Conv1d, net.block.conv_2).weight)

    # evaluate without drop path
    net.eval()
    branch = net(x) - x

    # drop path either removes or rescales the branch of each sample
    net.train()
    y = net(x)
    dropped = torch.isclose(y, x).flatten(1).all(dim=1)
    assert 0 < dropped.sum() < x.size(0)
    torch.testing.assert_close(y[~dropped], x[~dropped] + 2.0 * branch[~dropped])

    with pytest.raises(ValueError, match="drop_path"):
        ConvNeXtBlock(input_channels=16, drop_path=1.0)
    with pytest.raises(ValueError, match="requires skip_connection"):
        UniversalMultiLevelBlock(input_channels=16, kernel_size=3, drop_path=0.1)
