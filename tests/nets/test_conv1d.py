import torch
import torch.nn as nn

from dlk.nets.conv1d import ConvNet, ConvResNet, MultiLevelBlock


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
    """Validate ``MultiLevelBlock`` shapes for level blocks without resizing."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    x = torch.randn(batch_size, input_channels, input_length)

    net = MultiLevelBlock(input_channels=input_channels, kernel_size=3)
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)

    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=False,
        normalization_layer_channels=False,
        activation_layer_channels=False,
    )
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)

    output_channels = 32
    normalization_layer_channels = 8
    activation_layer_channels = 64
    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=nn.GroupNorm(2, normalization_layer_channels),
        normalization_layer_channels=normalization_layer_channels,
        activation=nn.SiLU(),
        activation_layer_channels=activation_layer_channels,
        output_channels=output_channels,
    )
    y = net(x)
    assert y.shape == (batch_size, output_channels, input_length)

    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        output_channels=output_channels,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == (batch_size, output_channels, input_length)

    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        dropout=nn.Dropout(0.1, inplace=False),
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)


def test_multilevel_block_forward_shapes_with_downsampling() -> None:
    """Validate ``MultiLevelBlock`` shapes when downsampling is enabled."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    expected_out_size = (batch_size, input_channels, input_length // 2)
    x = torch.randn(batch_size, input_channels, input_length)

    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=0.5,
        normalization=False,
        normalization_layer_channels=False,
        activation_layer_channels=False,
    )
    y = net(x)
    assert y.shape == expected_out_size

    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=0.5,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == expected_out_size


def test_multilevel_block_forward_shapes_with_upsampling() -> None:
    """Validate ``MultiLevelBlock`` shapes when upsampling is enabled."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    expected_out_size = (batch_size, input_channels, input_length * 2)
    x = torch.randn(batch_size, input_channels, input_length)

    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=2.0,
        normalization=False,
        normalization_layer_channels=False,
        activation_layer_channels=False,
    )
    y = net(x)
    assert y.shape == expected_out_size

    net = MultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=2.0,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == expected_out_size
