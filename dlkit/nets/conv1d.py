"""
Networks with 1D convolutional layers.
"""

from collections import OrderedDict
import math
import torch
import torch.nn as nn

from dlkit.nets.mlp import MLPResNet

# --------------------------------------
# Convolutional Nets
# --------------------------------------


class ConvNet(nn.Module):
    r"""
    Network with convolutional layers followed by dense layers.

    Args:
        input_channels: number of input channels
    """

    def __init__(
        self,
        input_channels,
        hidden_conv_layers_channels_mult=[8, 16, 32],
        hidden_conv_layers_kernels=[3, 3, 3],
        hidden_conv_layers_activation=nn.ReLU(),
        hidden_conv_layers_kwargs={},
        hidden_dense_input_size=None,
        hidden_dense_layers_sizes=[],
        hidden_dense_layers_activation=nn.ReLU(),
        hidden_dense_layers_kwargs={},
        output_size=None,
        output_layer_activation=None,
        output_layer_kwargs={},
        use_dropout=False,
    ):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.hidden_conv_layers_activation = hidden_conv_layers_activation
        self.hidden_dense_layers_activation = hidden_dense_layers_activation
        self.output_layer_activation = output_layer_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers
        assert len(hidden_conv_layers_channels_mult) == len(hidden_conv_layers_kernels)
        in_channels = input_channels
        self.hidden_conv_layers = nn.ModuleList()
        for channel_mult, kernel_size in zip(
            hidden_conv_layers_channels_mult, hidden_conv_layers_kernels
        ):
            out_channels = channel_mult * input_channels
            layer = nn.Conv1d(
                in_channels, out_channels, kernel_size, **hidden_conv_layers_kwargs
            )
            self.hidden_conv_layers.append(layer)
            in_channels = out_channels
        # create hidden dense layers
        assert hidden_dense_input_size is not None or 0 == len(
            hidden_dense_layers_sizes
        )
        in_size = hidden_dense_input_size
        self.hidden_dense_layers = nn.ModuleList()
        for layer_size in hidden_dense_layers_sizes:
            layer = nn.Linear(in_size, layer_size, **hidden_dense_layers_kwargs)
            self.hidden_dense_layers.append(layer)
            in_size = layer_size
        # create output layer
        if in_size is not None and output_size is not None:
            self.output_layer = nn.Linear(in_size, output_size, **output_layer_kwargs)
        else:
            self.output_layer = None
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the model function: y = model(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = x
        # apply hidden convolutional layers
        for layer in self.hidden_conv_layers:
            h = layer(h)
            if self.hidden_conv_layers_activation is not None:
                h = self.hidden_conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        h = torch.flatten(h, 1)
        # apply hidden dense layers
        for layer in self.hidden_dense_layers:
            h = layer(h)
            if self.hidden_dense_layers_activation is not None:
                h = self.hidden_dense_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply output layer
        if self.output_layer is not None:
            y = self.output_layer(h)
            if self.output_layer_activation is not None:
                y = self.output_layer_activation(y)
        else:
            y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden convolutional layers
        gain = _get_gain(self.hidden_conv_layers_activation)
        for layer in self.hidden_conv_layers:
            _set_init_parameters(layer, gain)
        # initialize hidden dense layers
        gain = _get_gain(self.hidden_dense_layers_activation)
        for layer in self.hidden_dense_layers:
            _set_init_parameters(layer, gain)
        # initialize output layer
        gain = _get_gain(self.output_layer_activation)
        if self.output_layer is not None:
            _set_init_parameters(self.output_layer, gain, bias_scale=0.0)


class ConvResNet(nn.Module):
    r"""
    Network with residual convolutional layers followed by residual dense layers.

    Args:
        input_channels: number of input channels
        conv_layers_params: configuration for convolutional layers
        mlp_resnet_params: parameters to pass to MLPResNet constructor

    Specs:
    - doc/specify/2025-10-27a.md
    """

    def __init__(
        self,
        input_channels,
        conv_layers_params={},
        mlp_resnet_params={},
    ):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.conv_layers_params = dict(conv_layers_params)
        self.mlp_resnet_params = dict(mlp_resnet_params)

        # set default convolution parameters
        self.conv_layers_params.setdefault("channels_mult", [8, 16, 32])
        self.conv_layers_params.setdefault("kernels", [5, 5, 5])
        self.conv_layers_params.setdefault("activation", nn.ReLU())
        self.conv_layers_params.setdefault("use_dropout", False)
        self.conv_layers_params.setdefault("conv_kwargs", {})
        assert len(self.conv_layers_params["channels_mult"]) == len(
            self.conv_layers_params["kernels"]
        )

        # create dropout layer
        if self.conv_layers_params["use_dropout"]:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None

        # create convolutional layers using Downsample
        layers = list()
        in_channels = self.input_channels
        for mult, kernel_size in zip(
            self.conv_layers_params["channels_mult"], self.conv_layers_params["kernels"]
        ):
            out_channels = mult * self.input_channels
            layers.append(
                Downsample(
                    in_channels,
                    out_channels,
                    kernel_size,
                    activation=self.conv_layers_params["activation"],
                    dropout=self.dropout,
                    **self.conv_layers_params["conv_kwargs"],
                )
            )
            in_channels = out_channels
        self.conv_resnet_block = nn.Sequential(*layers)

        # create dense layers using MLPResNet if parameters provided
        if self.mlp_resnet_params:
            # check if input_size is provided in self.mlp_resnet_params
            if "input_size" not in self.mlp_resnet_params:
                raise ValueError(
                    "mlp_resnet_params must have 'input_size' (flattened conv output size)"
                )
            self.mlp_resnet_block = MLPResNet(**self.mlp_resnet_params)
        else:
            self.mlp_resnet_block = None

        # initialize parameters
        self.init_parameters()

    def forward(self, x, **h_kwargs):
        r"""Applies the forward function: y = net(x, h0=hidden_input_0, h1=hidden_input1, ...)

        Args:
            x (tensor): input tensor
            h0, h1, ... (tensor, optional): input tensors to hidden dense layers
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = x

        # apply convolutional layers
        for layer in self.conv_resnet_block:
            h = layer(h)

        # return if nothing to do
        if self.mlp_resnet_block is None:
            return h

        # flatten for dense layers
        h = torch.flatten(h, 1)

        # apply dense residual network if configured
        y = self.mlp_resnet_block(h, **h_kwargs)

        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize convolutional block
        for layer in self.conv_resnet_block:
            layer.init_parameters()
        # initialize dense block
        if self.mlp_resnet_block is not None:
            self.mlp_resnet_block.init_parameters()


# --------------------------------------
# UNet Components
# --------------------------------------


class Downsample(nn.Module):
    """
    A downsampling layer with a convolution.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        activation=None,
        dropout=None,
        scale_factor=2,
        **conv_kwargs,
    ):
        super().__init__()
        # create convolutional layer with stride=scale_factor
        # add default values only if keys don't exist
        conv_kwargs = dict(conv_kwargs)  # copy to avoid modifying input
        conv_kwargs.setdefault("padding", 1)
        conv_kwargs.setdefault("padding_mode", "replicate")
        conv_kwargs.setdefault("stride", scale_factor)
        self.layer = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            **conv_kwargs,
        )
        # set from arguments
        self.input_channels = input_channels
        self.scale_factor = scale_factor
        self.activation = activation
        self.dropout = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = self.layer(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self):
        r"""
        Initializes the values of trainable parameters.
        """
        _set_init_parameters(self.layer, _get_gain(self.activation))


class Upsample(nn.Module):
    """
    An upsampling layer with a convolution.

    Args:
      input_channels   Channels in the inputs.
      output_channels  Channels in the outputs.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        activation=None,
        dropout=None,
        scale_factor=2,
        interp_mode="nearest",
        **conv_kwargs,
    ):
        super().__init__()
        # create convolutional layer
        # add default values only if keys don't exist
        conv_kwargs = dict(conv_kwargs)  # copy to avoid modifying input
        conv_kwargs.setdefault("padding", 1)
        conv_kwargs.setdefault("padding_mode", "replicate")
        self.layer = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            **conv_kwargs,
        )
        # set from arguments
        self.input_channels = input_channels
        self.scale_factor = scale_factor
        self.interp_mode = interp_mode
        self.activation = activation
        self.dropout = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.interp_mode
        )
        h = self.layer(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self):
        r"""
        Initializes the values of trainable parameters.
        """
        _set_init_parameters(self.layer, _get_gain(self.activation))


# TODO
# class LevelBlock(nn.Module):


def Normalization(num_channels, num_groups=1):
    return nn.GroupNorm(num_groups, num_channels)


# TODO replace by implementation of LevelBlock
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param output_channels: if specified, the number of out channels.
    :param use_conv: if True and output_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    """

    def __init__(
        self,
        input_channels,
        output_channels=None,
        use_conv=False,
        normalization=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels
        # create input layers
        self.in_layers = nn.Sequential(
            normalization(input_channels),
            nn.SiLU(),
            nn.Conv1d(
                input_channels,
                self.output_channels,
                3,
                padding=1,
                padding_mode="replicate",
            ),
        )
        # create output layers
        self.out_layers = nn.Sequential(
            normalization(self.output_channels),
            nn.SiLU(),
            _set_zero_parameters(
                nn.Conv1d(
                    self.output_channels,
                    self.output_channels,
                    3,
                    padding=1,
                    padding_mode="replicate",
                )
            ),
        )
        # create skip connection
        if self.output_channels == input_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                input_channels,
                self.output_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )
        else:
            self.skip_connection = nn.Conv1d(input_channels, self.output_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


# --------------------------------------
# Utility Functions
# --------------------------------------


def _get_gain(activation):
    r"""
    Calculates the gain to be used as an argument for initializing parameter values.

    Args:
        activation: Object of activation function or None
    """
    if activation is not None:
        activation_name = type(activation).__name__.lower()
        if activation_name in ["silu", "gelu"]:
            activation_name = "relu"
        gain = nn.init.calculate_gain(activation_name)
    else:
        gain = nn.init.calculate_gain("conv1d")
    return gain


def _set_init_parameters(layer, gain=1.0, bias_scale=0.1):
    r"""
    Initializes the trainable parameters of a layer.

    Args:
        layer:      Layer of a network to be initialized (of type nn.Module)
        gain:       Gain to use for sampling initial parameters
        bias_scale: Scaling of uniform distribution for initializing the bias
    """
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        lim = bias_scale * gain / math.sqrt(layer.bias.size(0))
        nn.init.uniform_(layer.bias, a=-lim, b=+lim)


def _set_zero_parameters(layer):
    r"""
    Zeros the parameters of a layer.
    """
    for p in layer.parameters():
        torch.nn.init.zeros_(p)
    return layer


# --------------------------------------
# Tests
# --------------------------------------

# TODO use doxygen for these test


def test_ConvNet():
    print("---------------------------------------^")
    input_channels = 1
    input_size = 16
    hidden_conv_layers_channels_mult = [2, 4, 8]
    hidden_dense_input_size = (
        16 - 2 * len(hidden_conv_layers_channels_mult)
    ) * hidden_conv_layers_channels_mult[-1]
    net = ConvNet(
        input_channels,
        output_size=2,
        hidden_conv_layers_channels_mult=hidden_conv_layers_channels_mult,
        hidden_dense_input_size=hidden_dense_input_size,
        hidden_dense_layers_sizes=[32, 32],
    )
    print(net)

    print("Test 1:")
    x = torch.ones((1, input_channels, input_size))
    y = net(x)
    print("- input  x =", x, sep="\n")
    print("- output y =", y, sep="\n")
    print("---------------------------------------$")


def test_ConvResNet():
    """Test ConvResNet architecture."""
    print("---------------------------------------^")
    batch_size = 4
    input_channels = 1
    input_length = 64
    output_size = 10

    # test basic configuration without MLPResNet
    print("Test 1: basic configuration (conv layers only)")
    net = ConvResNet(
        input_channels=input_channels,
        conv_layers_params={
            "channels_mult": [4, 8],
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
    )
    print(net)
    x = torch.randn(batch_size, input_channels, input_length)
    # calculate size after conv layers with 2 downsampling layers (scale_factor=2)
    # and output channels = 8 * input_channels = 8
    expected_out_size = (batch_size, 8, input_length // 4)
    expected_flat_size = (batch_size, expected_out_size[1] * expected_out_size[2])
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.shape}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test with MLPResNet
    print("\nTest 2: with MLPResNet")
    net = ConvResNet(
        input_channels=input_channels,
        conv_layers_params={
            "channels_mult": [4, 8],
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
        mlp_resnet_params={
            "input_size": expected_flat_size[1],
            "output_size": output_size,
            "residual_blocks_sizes": [(16, 16, 64, 16)],
        },
    )
    print(net)
    x = torch.randn(batch_size, input_channels, input_length)
    expected_out_size = (batch_size, output_size)
    print(f"Expected output size: {expected_out_size}")
    y = net(x)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    # test with hidden inputs to MLPResNet
    print("\nTest 3: with hidden inputs")
    hidden_input_size = 8
    net = ConvResNet(
        input_channels=input_channels,
        conv_layers_params={
            "channels_mult": [4, 8],
            "kernels": [3, 3],
            "activation": nn.ReLU(),
        },
        mlp_resnet_params={
            "input_size": expected_flat_size[1],
            "output_size": output_size,
            "residual_blocks_sizes": [
                (16, 16, 64, 16),
                (16 + hidden_input_size, 16, 64, 16),
            ],
        },
    )
    print(net)
    x = torch.randn(batch_size, input_channels, input_length)
    h1 = torch.randn(batch_size, 1, hidden_input_size)  # input to 2nd residual block
    print(f"Expected output size: {expected_out_size}")
    y = net(x, h1=h1)
    print(f"Output size: {y.size()}")
    assert (
        y.size() == expected_out_size
    ), f"Expected {expected_out_size}, got {y.size()}"

    print("---------------------------------------$")


if __name__ == "__main__":
    r"""Runs tests."""
    test_ConvNet()
    print("\n")
    test_ConvResNet()
