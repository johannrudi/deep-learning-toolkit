"""1D convolutional network architectures and reusable blocks."""

import logging
import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from dlk.nets.mlp import MLPResNet
from dlk.nets.utils import (
    ModuleFactory,
    NormalizationFactory,
    WeightedLayer,
    get_gain,
    set_init_parameters,
    set_zero_parameters,
)

# TODO: old code, decide what to do
# def _get_conv1d_size(in_length, kernel, stride=1, padding=0, dilation=1):
#     return int((in_length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


# --------------------------------------
# Convolutional Nets
# --------------------------------------


class ConvNet(nn.Module):
    """
    Build a convolutional network with optional dense output layers.

    Args:
        input_channels: Number of input channels.
        hidden_conv_layers_channels_mult: Multipliers for hidden convolution channels.
        hidden_conv_layers_kernels: Kernel sizes for hidden convolution layers.
        hidden_conv_layers_activation: Optional activation after convolution layers.
        hidden_conv_layers_kwargs: Optional keyword arguments for convolution layers.
        hidden_dense_input_size: Input feature size for first dense layer.
        hidden_dense_layers_sizes: Width of each hidden dense layer.
        hidden_dense_layers_activation: Optional activation after dense layers.
        hidden_dense_layers_kwargs: Optional keyword arguments for dense layers.
        output_size: Output feature size of the output layer.
        output_layer_activation: Optional activation after output layer.
        output_layer_kwargs: Optional keyword arguments for output layer.
        use_dropout: Dropout probability, or ``False`` to disable dropout.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_conv_layers_channels_mult: Sequence[int] = (8, 16, 32),
        hidden_conv_layers_kernels: Sequence[int] = (3, 3, 3),
        hidden_conv_layers_activation: nn.Module | None = nn.ReLU(),
        hidden_conv_layers_kwargs: dict[str, Any] | None = None,
        hidden_dense_input_size: int | None = None,
        hidden_dense_layers_sizes: Sequence[int] = (),
        hidden_dense_layers_activation: nn.Module | None = nn.ReLU(),
        hidden_dense_layers_kwargs: dict[str, Any] | None = None,
        output_size: int | None = None,
        output_layer_activation: nn.Module | None = None,
        output_layer_kwargs: dict[str, Any] | None = None,
        use_dropout: float | bool = False,
    ) -> None:
        super().__init__()
        # set default layer kwargs
        hidden_conv_layers_kwargs = dict(hidden_conv_layers_kwargs or {})
        hidden_dense_layers_kwargs = dict(hidden_dense_layers_kwargs or {})
        output_layer_kwargs = dict(output_layer_kwargs or {})
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
            assert in_size is not None
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the model function: y = model(x)

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.

        Returns:
            Output tensor after convolutional and optional dense layers.
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = x
        # apply hidden convolutional layers
        for layer in self.hidden_conv_layers:
            h = cast(nn.Conv1d, layer)(h)
            if self.hidden_conv_layers_activation is not None:
                h = self.hidden_conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        h = torch.flatten(h, 1)
        # apply hidden dense layers
        for layer in self.hidden_dense_layers:
            h = cast(nn.Linear, layer)(h)
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

    def init_parameters(self) -> None:
        """Initialize trainable parameters with layer-aware gains."""
        # initialize hidden convolutional layers
        gain = get_gain(self.hidden_conv_layers_activation, default="conv1d")
        for layer in self.hidden_conv_layers:
            set_init_parameters(cast(nn.Conv1d, layer), gain)
        # initialize hidden dense layers
        gain = get_gain(self.hidden_dense_layers_activation, default="conv1d")
        for layer in self.hidden_dense_layers:
            set_init_parameters(cast(nn.Linear, layer), gain)
        # initialize output layer
        gain = get_gain(self.output_layer_activation, default="conv1d")
        if self.output_layer is not None:
            set_init_parameters(self.output_layer, gain, bias_scale=0.0)


class ConvResNet(nn.Module):
    """
    Build a residual 1D convolutional network with an optional residual MLP head.

    Args:
        input_channels: Number of input channels.
        conv_resnet_params: Configuration for convolutional residual layers.
        mlp_resnet_params: Parameters passed to :class:`dlk.nets.mlp.MLPResNet`.
        with_Conv: Convolution layer factory used for 1D blocks.

    Specs:
    - doc/specify/2025-10-27a.md
    """

    def __init__(
        self,
        input_channels: int,
        conv_resnet_params: dict[str, Any] | None = None,
        mlp_resnet_params: dict[str, Any] | None = None,
        with_Conv: ModuleFactory = nn.Conv1d,
    ) -> None:
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.conv_resnet_params = dict(conv_resnet_params or {})
        self.mlp_resnet_params = dict(mlp_resnet_params or {})

        # set default convolution parameters
        self.conv_resnet_params.setdefault("channels_mult", [8, 16, 32])
        self.conv_resnet_params.setdefault("kernels", [5, 5, 5])
        self.conv_resnet_params.setdefault("activation", nn.ReLU())
        self.conv_resnet_params.setdefault("use_dropout", False)
        self.conv_resnet_params.setdefault("mlb_kwargs", {})
        assert len(self.conv_resnet_params["channels_mult"]) == len(
            self.conv_resnet_params["kernels"]
        )

        # set scale factor
        if "stride" not in self.conv_resnet_params["mlb_kwargs"]:
            scale_factor = 0.5  # downsample by factor 1/2
        else:
            scale_factor = None

        # set activation
        activation = self.conv_resnet_params["activation"]

        # set up dropout
        use_dropout = self.conv_resnet_params["use_dropout"]
        if use_dropout:
            dropout = nn.Dropout(use_dropout)
        else:
            dropout = None

        # create input layer
        in_channels = self.input_channels
        out_channels = self.conv_resnet_params["channels_mult"][0] * self.input_channels
        kernel_size = self.conv_resnet_params["kernels"][0]
        self.input_layer: nn.Module = with_Conv(
            in_channels, out_channels, 1, groups=in_channels
        )
        in_channels = out_channels

        # create convolutional residual blocks using MultiLevelBlock
        layers = list()
        for mult, kernel_size in zip(
            self.conv_resnet_params["channels_mult"], self.conv_resnet_params["kernels"]
        ):
            # set defaul normalization, normalization channels, and activation channels
            # fmt: off
            if "normalization" not in self.conv_resnet_params["mlb_kwargs"]:
                self.conv_resnet_params["mlb_kwargs"]["normalization"] = nn.GroupNorm(
                    self.input_channels, in_channels
                )
            if ( "normalization_layer_channels" not in self.conv_resnet_params["mlb_kwargs"]):
                self.conv_resnet_params["mlb_kwargs"][ "normalization_layer_channels" ] = in_channels
            if "activation_layer_channels" not in self.conv_resnet_params["mlb_kwargs"]:
                self.conv_resnet_params["mlb_kwargs"]["activation_layer_channels"] = 4 * in_channels
            # fmt: on
            # create convolution block
            out_channels = mult * self.input_channels
            layers.append(
                MultiLevelBlock(
                    in_channels,
                    kernel_size,
                    activation=activation,
                    output_channels=out_channels,
                    dropout=dropout,
                    scale_factor=scale_factor,
                    skip_connection=True,
                    **self.conv_resnet_params["mlb_kwargs"],
                )
            )
            in_channels = out_channels
        self.conv_resnet: nn.Sequential = nn.Sequential(*layers)

        # create dense layers using MLPResNet if parameters provided
        if self.mlp_resnet_params:
            # check if input_size is provided in self.mlp_resnet_params
            if "input_size" not in self.mlp_resnet_params:
                raise ValueError(
                    "mlp_resnet_params must have 'input_size' (flattened conv output size)"
                )
            self.mlp_resnet = MLPResNet(**self.mlp_resnet_params)
            ###DEV
            # from dlk.nets.mlp import LinearFiber
            # # create contraction layer for space
            # if isinstance(self.mlp_resnet_params["input_size"], int):
            #     in_fiber_size = self.mlp_resnet_params["input_size"] // out_channels
            # else:
            #     in_fiber_size = self.mlp_resnet_params["input_size"][0] // out_channels
            # self.mlp_space_l0 = LinearFiber(
            #     ndim=2,
            #     input_fiber_size=in_fiber_size,
            #     output_fiber_size=128,
            # )
            # self.mlp_activation = activation
            # self.mlp_space_l1 = LinearFiber(
            #     ndim=2,
            #     input_fiber_size=128,
            #     output_fiber_size=1,
            # )
            # if isinstance(self.mlp_resnet_params["input_size"], int):
            #     in_size = out_channels
            # else:
            #     in_size = (
            #         out_channels +
            #         self.mlp_resnet_params["residual_blocks_sizes"][0][0] -
            #         self.mlp_resnet_params["input_size"][1]
            #     )
            # out_size = self.mlp_resnet_params["output_size"]
            # self.mlp_output_l0 = nn.Linear(in_size, 128)
            # self.mlp_output_l1 = nn.Linear(128, out_size)
            ###/DEV
        else:
            self.mlp_resnet = None

        # initialize parameters
        self.init_parameters()

    def forward(self, x: torch.Tensor, **h_kwargs: torch.Tensor) -> torch.Tensor:
        """Apply the forward function: ``y = net(x, h0=..., h1=..., ...)``.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.
            **h_kwargs: Optional hidden-input tensors passed to ``MLPResNet``.

        Returns:
            Output tensor from convolutional stack or residual MLP head.
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"

        # apply input layer
        h = self.input_layer(x)

        # apply convolutional layers
        for layer in self.conv_resnet:
            h = cast(nn.Module, layer)(h)

        # return if nothing to do
        if self.mlp_resnet is None:
            return h

        # flatten for dense layers
        h = torch.flatten(h, 1)

        # apply dense residual network if configured
        y = self.mlp_resnet(h, **h_kwargs)

        ###DEV
        # h = self.mlp_space_l1(self.mlp_activation(self.mlp_space_l0(h)))
        # h = torch.flatten(h, 1)
        # h_in = h_kwargs.get(f"h{0}")
        # if h_in is None:
        #     h_in = h_kwargs.get("h_all")
        # if h_in is not None:
        #     h_in = torch.flatten(h_in, 1)
        #     h = torch.cat([h, h_in], dim=1)
        # y = self.mlp_output_l1(self.mlp_activation(self.mlp_output_l0(h)))
        ###/DEV

        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters of all active submodules."""
        # initialize input layer
        set_init_parameters(
            cast(WeightedLayer, self.input_layer), get_gain(None, default="conv1d")
        )
        # initialize convolutional block
        for layer in self.conv_resnet:
            cast(MultiLevelBlock, layer).init_parameters()
        # initialize dense block
        if self.mlp_resnet is not None:
            self.mlp_resnet.init_parameters()


# --------------------------------------
# UNet Components
# --------------------------------------


class UNetDownsample(nn.Module):
    """
    Build a downsampling layer based on strided 1D convolution.

    Args:
        input_channels: Channels in the input tensor.
        output_channels: Channels in the output tensor.
        kernel_size: Convolution kernel size.
        activation: Optional activation module after convolution.
        dropout: Optional dropout module after activation.
        scale_factor: Downsampling factor mapped to convolution stride.
        **layer_kwargs: Additional keyword arguments for ``nn.Conv1d``.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        activation: nn.Module | None = None,
        dropout: nn.Module | None = None,
        scale_factor: int = 2,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        # create convolutional layer with stride=scale_factor
        # add default values only if keys don't exist
        self.layer_kwargs = dict(layer_kwargs)  # copy to avoid modifying input
        self.layer_kwargs.setdefault("padding", 1)
        self.layer_kwargs.setdefault("padding_mode", "replicate")
        self.layer_kwargs.setdefault("stride", scale_factor)
        self.input_channels = input_channels
        # create layers
        block = OrderedDict()
        block["layer"] = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            **self.layer_kwargs,
        )
        if activation is not None:
            block["activation"] = activation
        if dropout is not None:
            block["dropout"] = dropout
        self.block = nn.Sequential(block)
        # initialize parameters
        self.init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the downsampling block to an input tensor.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.

        Returns:
            Downsampled output tensor.
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        y = self.block(x)
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters using the block activation gain."""
        activation = getattr(self.block, "activation", None)
        set_init_parameters(
            cast(nn.Conv1d, self.block.layer), get_gain(activation, default="conv1d")
        )


class UNetUpsample(nn.Module):
    """
    Build an upsampling layer using interpolation followed by 1D convolution.

    Args:
        input_channels: Channels in the input tensor.
        output_channels: Channels in the output tensor.
        kernel_size: Convolution kernel size.
        activation: Optional activation module after convolution.
        dropout: Optional dropout module after activation.
        scale_factor: Upsampling factor used in interpolation.
        interp_mode: Interpolation mode passed to ``torch.nn.functional.interpolate``.
        **layer_kwargs: Additional keyword arguments for ``nn.Conv1d``.

    Note:
        Mode ``nearest-exact`` matches Scikit-Image and PIL nearest-neighbor
        interpolation algorithms and fixes known issues with ``nearest``.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        activation: nn.Module | None = None,
        dropout: nn.Module | None = None,
        scale_factor: int = 2,
        interp_mode: str = "nearest-exact",
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        # create convolutional layer
        # add default values only if keys don't exist
        self.layer_kwargs = dict(layer_kwargs)  # copy to avoid modifying input
        self.layer_kwargs.setdefault("padding", 1)
        self.layer_kwargs.setdefault("padding_mode", "replicate")
        self.input_channels = input_channels
        self.scale_factor = scale_factor
        self.interp_mode = interp_mode
        # create layers
        block = OrderedDict()
        block["layer"] = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            **self.layer_kwargs,
        )
        if activation is not None:
            block["activation"] = activation
        if dropout is not None:
            block["dropout"] = dropout
        self.block = nn.Sequential(block)
        # initialize parameters
        self.init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply interpolation and convolution to upsample an input tensor.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.

        Returns:
            Upsampled output tensor.
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.interp_mode
        )
        y = self.block(h)
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters using the block activation gain."""
        activation = getattr(self.block, "activation", None)
        set_init_parameters(
            cast(nn.Conv1d, self.block.layer), get_gain(activation, default="conv1d")
        )


def Normalization(num_channels: int, num_groups: int = 1) -> nn.GroupNorm:
    """Build a group normalization layer for 1D feature maps.

    Args:
        num_channels: Number of channels in the normalized tensor.
        num_groups: Number of groups used by group normalization.

    Returns:
        Configured group normalization layer.
    """
    return nn.GroupNorm(num_groups, num_channels)


class UNetResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    Args:
        input_channels: Number of input channels.
        output_channels: Optional number of output channels.
        use_conv: If ``True``, use a spatial convolution in the skip branch.
        normalization: Factory that builds channel-aware normalization layers.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int | None = None,
        use_conv: bool = False,
        with_normalization: NormalizationFactory | None = None,
    ) -> None:
        super().__init__()
        if with_normalization is None:
            with_normalization = Normalization
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels
        # create input layers
        self.in_layers = nn.Sequential(
            with_normalization(input_channels),
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
            with_normalization(self.output_channels),
            nn.SiLU(),
            set_zero_parameters(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to an input tensor.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.

        Returns:
            Residual output tensor with updated channels.
        """
        assert x.size(1) == self.input_channels, f"{x.size(1)=}, {self.input_channels=}"
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


# --------------------------------------
# Universal-Design Multi-Level Components
# --------------------------------------


class MultiLevelBlock(nn.Module):
    """
    Build a universal-design multi-level convolutional block.

    Args:
        input_channels: Channels in input tensors.
        kernel_size: Convolution kernel size.
        normalization: Optional normalization module before projection layers.
        normalization_layer_channels: Intermediate channels for normalization path.
        activation: Optional activation module.
        activation_layer_channels: Intermediate channels for activation path.
        output_channels: Output channels of the block.
        dropout: Optional dropout module applied after activation.
        scale_factor: Relative scaling factor for sequence length.
        skip_connection: If ``True``, add a residual skip branch.
        interp_mode: Interpolation mode for up/down sampling.
        logger: Logger used for argument-coherence warnings.
        with_Conv: Convolution layer factory used in the block.
        **conv_kwargs: Extra keyword arguments passed to convolution layers.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        normalization: nn.Module | bool | None = None,
        normalization_layer_channels: int | bool | None = None,
        activation: nn.Module | bool | None = None,
        activation_layer_channels: int | bool | None = None,
        output_channels: int | None = None,
        dropout: nn.Module | bool | None = None,
        scale_factor: float | None = None,
        skip_connection: bool = False,
        interp_mode: str = "nearest-exact",
        logger: logging.Logger = logging.getLogger("dlk.nets.conv1d.MultiLevelBlock"),
        with_Conv: ModuleFactory = nn.Conv1d,
        **conv_kwargs: Any,
    ) -> None:
        super().__init__()
        self.conv_kwargs = dict(conv_kwargs)  # copy to avoid modifying input args
        # set default padding mode (if key does not exist)
        self.conv_kwargs.setdefault("padding_mode", "replicate")
        # set default padding size (if key does not exist)
        # Note: Conv1d expects an int padding; a 2-tuple causes channel padding
        #   when padding_mode != 'zeros'; use symmetric integer padding for the conv
        #   layer and keep the 2-tuple separately for size bookkeeping
        self.conv_kwargs.setdefault("padding", kernel_size // 2)
        assert isinstance(
            self.conv_kwargs["padding"], int
        ), f"Expected type int, got {type(self.conv_kwargs['padding'])}"
        # store a 2-tuple version of padding for internal calculations (left, right)
        self._padding = (self.conv_kwargs["padding"], self.conv_kwargs["padding"])
        # define the padding s.t. `input_size == output_size` after convolution
        self._padding_const_size = (
            (kernel_size - 1) // 2,
            (kernel_size - 1) // 2 + (1 - kernel_size % 2),
        )
        # set attributes from arguments
        self.input_channels = input_channels
        self.interp_mode = interp_mode
        if scale_factor is not None:
            assert (
                "stride" not in conv_kwargs or scale_factor == conv_kwargs["stride"]
            ), f"Invalid args: Cannot set both scale_factor={scale_factor} and stride={conv_kwargs['stride']}."
            if 1 <= scale_factor:
                self.conv_kwargs["stride"] = 1
            elif 0 < scale_factor:
                self.conv_kwargs["stride"] = int(1 / scale_factor)
            else:
                raise ValueError(
                    f"Invalid arg: scale_factor={scale_factor} cannot be non-positive."
                )
            self.scale_factor = float(scale_factor)
        elif "stride" in conv_kwargs:
            self.scale_factor = 1.0 / conv_kwargs["stride"]
        else:
            self.scale_factor = 1.0
        # init channels
        ch = [input_channels]
        # set up normalization
        if normalization_layer_channels or normalization_layer_channels is None:
            if normalization_layer_channels is None:
                normalization_layer_channels = ch[-1]
            ch.append(normalization_layer_channels)
        if normalization is None:
            normalization = nn.GroupNorm(1, ch[-1])
        if (not normalization) != (not normalization_layer_channels):
            logger.warning(
                f"Incoherent args: {normalization=}, {normalization_layer_channels=}"
            )
        # set up activation
        if activation_layer_channels or activation_layer_channels is None:
            if activation_layer_channels is None:
                activation_layer_channels = 4 * ch[-1]
            ch.append(activation_layer_channels)
        if activation is None:
            activation = nn.ReLU()
        # set output channels
        if output_channels is not None:
            ch.append(output_channels)
        else:
            ch.append(input_channels)
        # create layers
        i = 0
        block = OrderedDict()
        block["layer_0"] = with_Conv(ch[i], ch[i + 1], kernel_size, **self.conv_kwargs)
        i += 1
        if normalization:
            block["normalization"] = normalization
        if normalization_layer_channels:
            block["layer_1"] = with_Conv(ch[i], ch[i + 1], 1)
            i += 1
        if activation:
            block["activation"] = activation
        if dropout:
            block["dropout"] = dropout
        if activation_layer_channels:
            block["layer_2"] = with_Conv(ch[i], ch[i + 1], 1)
            i += 1
        self.block = nn.Sequential(block)
        # create skip connection
        if skip_connection:
            self.skip_interp_mode = interp_mode
            if ch[0] == ch[-1]:
                self.skip_connection = nn.Identity()
            else:
                self.skip_connection = with_Conv(ch[0], ch[-1], 1)
        else:
            self.skip_connection = None
        # initialize parameters
        self.init_parameters()

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Apply the block to one or more channel-compatible input tensors.

        Args:
            *x: One or more tensors concatenated along the channel dimension.

        Returns:
            Output tensor after optional scaling, block transform, and skip path.
        """
        channel_dim, size_dim = 1, 2
        # concatenate along channel dimension
        if 1 < len(x):
            h = torch.cat(x, dim=channel_dim)
        else:
            assert 1 == len(x)
            h = x[0]
        assert (
            h.size(channel_dim) == self.input_channels
        ), f"Expect {self.input_channels=}, got {h.size(channel_dim)=}"
        # scale up
        if 1.0 < self.scale_factor:
            h = nn.functional.interpolate(
                h, scale_factor=self.scale_factor, mode=self.interp_mode
            )
        # apply block
        hb = self.block(h)
        # apply skip connection (optional)
        if self.skip_connection is not None:
            # scale down
            if self.scale_factor < 1.0:
                hs_size = (
                    int(math.ceil(h.size(size_dim) * self.scale_factor))
                    + sum(self._padding)
                    - sum(self._padding_const_size)
                )
                h = nn.functional.interpolate(
                    h, size=hs_size, mode=self.skip_interp_mode
                )
            hs = self.skip_connection(h)
            y = 0.5 * hs + 0.5 * hb
        else:
            y = hb
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters for all active convolutional layers."""
        if hasattr(self.block, "activation"):
            gain = get_gain(self.block.activation, default="conv1d")
        else:
            gain = get_gain(None, default="conv1d")
        set_init_parameters(cast(WeightedLayer, self.block.layer_0), gain)
        if hasattr(self.block, "layer_1"):
            set_init_parameters(cast(WeightedLayer, self.block.layer_1), gain)
        if hasattr(self.block, "layer_2"):
            set_init_parameters(cast(WeightedLayer, self.block.layer_2), gain)
        if self.skip_connection is not None and not isinstance(
            self.skip_connection, nn.Identity
        ):
            set_init_parameters(cast(WeightedLayer, self.skip_connection))


class DownsampleBlock(MultiLevelBlock):
    """
    Downsampling based on the universal-design multi-level block.

    Args:
        input_channels: Channels in input tensors.
        kernel_size: Convolution kernel size.
        scale_factor: Relative downsampling factor for sequence length.
        skip_connection: If ``True``, add a residual skip branch.
        interp_mode: Interpolation mode used by the skip path.
        **mlb_kwargs: Additional parameters forwarded to ``MultiLevelBlock``.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        scale_factor: float = 0.5,
        skip_connection: bool = True,
        interp_mode: str = "nearest-exact",
        **mlb_kwargs: Any,
    ) -> None:
        super().__init__(
            input_channels,
            kernel_size,
            scale_factor=scale_factor,
            skip_connection=skip_connection,
            interp_mode=interp_mode,
            logger=logging.getLogger("dlk.nets.conv1d.DownsampleBlock"),
            **mlb_kwargs,
        )


class UpsampleBlock(MultiLevelBlock):
    """
    Upsampling based on the universal-design multi-level block.

    Args:
        input_channels: Channels in input tensors.
        kernel_size: Convolution kernel size.
        scale_factor: Relative upsampling factor for sequence length.
        skip_connection: If ``True``, add a residual skip branch.
        interp_mode: Interpolation mode used by the skip path.
        **mlb_kwargs: Additional parameters forwarded to ``MultiLevelBlock``.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        scale_factor: float = 2.0,
        skip_connection: bool = True,
        interp_mode: str = "nearest-exact",
        **mlb_kwargs: Any,
    ) -> None:
        super().__init__(
            input_channels,
            kernel_size,
            scale_factor=scale_factor,
            skip_connection=skip_connection,
            interp_mode=interp_mode,
            logger=logging.getLogger("dlk.nets.conv1d.UpsampleBlock"),
            **mlb_kwargs,
        )


class LevelBlock(MultiLevelBlock):
    """
    Residual block based on the universal-design multi-level block.

    Args:
        input_channels: Channels in input tensors.
        kernel_size: Convolution kernel size.
        skip_connection: If ``True``, add a residual skip branch.
        **mlb_kwargs: Additional parameters forwarded to ``MultiLevelBlock``.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        skip_connection: bool = True,
        **mlb_kwargs: Any,
    ) -> None:
        super().__init__(
            input_channels,
            kernel_size,
            skip_connection=skip_connection,
            logger=logging.getLogger("dlk.nets.conv1d.LevelBlock"),
            **mlb_kwargs,
        )
