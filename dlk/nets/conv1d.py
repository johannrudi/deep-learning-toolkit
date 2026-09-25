"""1D convolutional network architectures and reusable blocks."""

import math
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from dlk.nets.mlp import MLPResNet
from dlk.nets.utils import (
    ModuleFactory,
    NormalizationFactory,
    get_gain,
    set_init_parameters,
    set_zero_parameters,
)

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
        self.conv_output_channels = in_channels
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
            h = cast(nn.Module, layer)(h)
            if self.hidden_conv_layers_activation is not None:
                h = self.hidden_conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        h = torch.flatten(h, 1)
        # apply hidden dense layers
        for layer in self.hidden_dense_layers:
            h = cast(nn.Module, layer)(h)
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

    def resolve_input_shape(self) -> tuple[int | None, ...]:
        """Return the shape of one input sample, excluding the batch dimension.

        Returns:
            The 2D shape ``(input_channels, None)``; the network constrains no
            sequence length.
        """
        return (self.input_channels, None)

    def resolve_output_shape(self) -> tuple[int | None, ...]:
        """Return the shape of one output sample, excluding the batch dimension.

        `forward` flattens the convolution stack unconditionally, so the output
        is 1D in every configuration. Its width comes from the output layer,
        from the last hidden dense layer when there is no output layer, and is
        unknown when there is neither, the flattened width then following a
        sequence length the network does not constrain.

        Returns:
            The 1D output shape.
        """
        if self.output_layer is not None:
            return (self.output_layer.out_features,)
        if 0 < len(self.hidden_dense_layers):
            last_dense_layer = cast(nn.Linear, self.hidden_dense_layers[-1])
            return (last_dense_layer.out_features,)
        return (None,)

    def init_parameters(self) -> None:
        """Initialize trainable parameters with layer-aware gains."""
        # initialize hidden convolutional layers
        gain = get_gain(self.hidden_conv_layers_activation, default="conv1d")
        for layer in self.hidden_conv_layers:
            set_init_parameters(layer, gain)
        # initialize hidden dense layers
        gain = get_gain(self.hidden_dense_layers_activation, default="conv1d")
        for layer in self.hidden_dense_layers:
            set_init_parameters(layer, gain)
        # initialize output layer
        gain = get_gain(self.output_layer_activation, default="conv1d")
        if self.output_layer is not None:
            set_init_parameters(self.output_layer, gain, bias_scale=0.0)


class ConvResNet(nn.Module):
    """
    Build a residual 1D convolutional network with an optional residual MLP head.

    With ``input_length``, the network derives the sequence length after the
    convolutional layers, and from it the ``input_size`` of the MLP head, which
    ``mlp_resnet_params`` then may omit.

    Args:
        input_channels: Number of input channels.
        input_length: Expected sequence length, or ``None`` to disable checks;
          required by an MLP head without ``input_size``.
        conv_resnet_params: Configuration for convolutional residual layers.
        mlp_resnet_params: Parameters passed to :class:`dlk.nets.mlp.MLPResNet`.
        conv: Convolution layer factory used for 1D blocks.

    Implementation plan: docs/features/2025.005__ConvResNet__1-plan.md
    """

    def __init__(
        self,
        input_channels: int,
        input_length: int | None = None,
        conv_resnet_params: dict[str, Any] | None = None,
        mlp_resnet_params: dict[str, Any] | None = None,
        conv: ModuleFactory = nn.Conv1d,
    ) -> None:
        super().__init__()
        if input_length is not None and input_length <= 0:
            raise ValueError(f"input length must be positive, got {input_length=}")

        # set from arguments
        self.input_channels = input_channels
        self.input_length = input_length
        self.conv_resnet_params = dict(conv_resnet_params or {})
        self.mlp_resnet_params = dict(mlp_resnet_params or {})

        # set default convolution parameters
        self.conv_resnet_params.setdefault("channels_mult", [8, 16, 32])
        self.conv_resnet_params.setdefault("kernels", [5, 5, 5])
        self.conv_resnet_params.setdefault("use_dropout", False)
        self.conv_resnet_params.setdefault("block_kwargs", {})
        assert len(self.conv_resnet_params["channels_mult"]) == len(
            self.conv_resnet_params["kernels"]
        )

        # copy to avoid modifying input args
        block_kwargs = dict(self.conv_resnet_params["block_kwargs"])
        if "conv" in block_kwargs:
            raise ValueError("pass `conv` to ConvResNet, not in block_kwargs")
            # NOTE: this is likely temporary until a config dataclass is created
        block_kwargs["conv"] = conv
        block_kwargs.setdefault(
            "normalization", partial(Normalization, num_groups=input_channels)
        )
        self.conv_resnet_params["block_kwargs"] = block_kwargs

        # set scale factor
        conv_kwargs = block_kwargs.get("conv_kwargs") or {}
        if "stride" not in conv_kwargs:
            scale_factor = 0.5  # downsample by factor 1/2
        else:
            scale_factor = None

        # set dropout probability
        dropout = float(self.conv_resnet_params["use_dropout"])

        # create input layer
        in_channels = self.input_channels
        out_channels = self.conv_resnet_params["channels_mult"][0] * self.input_channels
        self.input_layer: nn.Module = conv(
            in_channels, out_channels, 1, groups=in_channels
        )
        in_channels = out_channels

        # create convolutional residual blocks
        layers = list()
        for mult, kernel_size in zip(
            self.conv_resnet_params["channels_mult"], self.conv_resnet_params["kernels"]
        ):
            # create convolution block
            out_channels = mult * self.input_channels
            layers.append(
                LevelBlock(
                    in_channels,
                    kernel_size,
                    output_channels=out_channels,
                    dropout=dropout,
                    scale_factor=scale_factor,
                    **self.conv_resnet_params["block_kwargs"],
                )
            )
            in_channels = out_channels
        self.conv_resnet = nn.Sequential(*layers)
        self.conv_output_channels = in_channels

        # resolve sequence length after convolutional layers
        self.conv_output_length: int | None = None
        if input_length is not None:
            length = input_length
            for layer in self.conv_resnet:
                length = cast(LevelBlock, layer).resolve_output_length(length)
            if length <= 0:
                raise ValueError(
                    f"convolutional layers reduce {input_length=} to {length=}"
                )
            self.conv_output_length = length

        # create dense layers using MLPResNet if parameters provided
        if self.mlp_resnet_params:
            # set or check input size of MLP head (flattened conv output size)
            if self.conv_output_length is not None:
                conv_output_size = self.conv_output_channels * self.conv_output_length
                input_size = self.mlp_resnet_params.setdefault(
                    "input_size", conv_output_size
                )
                if isinstance(input_size, Sequence):
                    input_size = input_size[0]
                if input_size != conv_output_size:
                    raise ValueError(
                        f"mlp_resnet_params has {input_size=}, but the flattened "
                        f"conv output has size {conv_output_size} for {input_length=}"
                    )
            elif "input_size" not in self.mlp_resnet_params:
                raise ValueError(
                    "mlp_resnet_params must have 'input_size' (flattened conv output "
                    "size), or ConvResNet must have 'input_length'"
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
        assert (
            x.size(1) == self.input_channels
        ), f"expected input channels {self.input_channels}, got {x.size(1)}"
        assert (
            self.input_length is None or x.size(2) == self.input_length
        ), f"expected input length {self.input_length}, got {x.size(2)}"

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

    def resolve_input_shape(self) -> tuple[int | None, ...]:
        """Return the shape of one input sample, excluding the batch dimension.

        Returns:
            The 2D shape ``(input_channels, input_length)``; the length entry is
            ``None`` when ``input_length`` disables the length check.
        """
        return (self.input_channels, self.input_length)

    def resolve_output_shape(self) -> tuple[int | None, ...]:
        """Return the shape of one output sample, excluding the batch dimension.

        Without the residual MLP head the network returns the convolution stack
        unflattened, which is a feature map and is reported as one.

        Returns:
            The output shape of the residual MLP head, or the 2D shape
            ``(conv_output_channels, conv_output_length)`` without that head;
            the length entry is ``None`` without ``input_length``.
        """
        if self.mlp_resnet is None:
            return (self.conv_output_channels, self.conv_output_length)
        return self.mlp_resnet.resolve_output_shape()

    def init_parameters(self) -> None:
        """Initialize trainable parameters of all active submodules."""
        # initialize input layer
        set_init_parameters(self.input_layer, get_gain(None, default="conv1d"))
        # initialize convolutional block
        for layer in self.conv_resnet:
            cast(LevelBlock, layer).init_parameters()
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
        set_init_parameters(self.block.layer, get_gain(activation, default="conv1d"))


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
        set_init_parameters(self.block.layer, get_gain(activation, default="conv1d"))


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
        normalization: NormalizationFactory | None = None,
    ) -> None:
        super().__init__()
        if normalization is None:
            normalization = Normalization
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
# Universal Multi-Level Components
# --------------------------------------


def Normalization(num_channels: int, num_groups: int = 1) -> nn.GroupNorm:
    """Build a group normalization layer for 1D feature maps.

    Args:
        num_channels: Number of channels in the normalized tensor.
        num_groups: Number of groups used by group normalization.

    Returns:
        Configured group normalization layer.
    """
    return nn.GroupNorm(num_groups, num_channels)


class UniversalMultiLevelBlock(nn.Module):
    """
    Build a universal multi-level convolutional block.

    The main branch chains a convolution with kernel size ``kernel_size`` and two
    optional stages, each of which ends with a 1x1 convolution::

        conv_k -> [normalization -> conv_1x1] -> [activation -> dropout -> conv_1x1]

    An optional skip branch adds the input to the output of the main branch,
    through a 1x1 convolution when the channels differ. During training, drop
    path (stochastic depth) then drops the main branch for each sample with
    probability ``drop_path``. The block scales the sequence length by
    ``scale_factor``: it upsamples by interpolation before both branches, and it
    downsamples through the stride of ``conv_k``, resizing the skip branch to
    match.

    Args:
        input_channels: Channels in input tensors; `forward` concatenates
          several input tensors along the channels.
        kernel_size: Kernel size of the convolution ``conv_k``.
        normalization: Factory that builds the normalization layer from its number
          of channels; ``True`` for group normalization with one group, ``False``
          to drop the normalization and the 1x1 convolution after it.
        normalization_channels: Channels of the normalization; defaults to
          ``input_channels``. Requires normalization.
        activation: Activation module; ``True`` for GELU, ``False`` to drop the
          activation and the 1x1 convolution after it.
        activation_channels: Channels of the activation; defaults to four times
          the channels of the preceding layer. Requires activation.
        output_channels: Output channels of the block; defaults to
          ``input_channels``.
        dropout: Dropout probability after the activation; ``0`` disables dropout.
        scale_factor: Relative scaling factor of the sequence length; a factor
          below one must be the inverse of an integer, which becomes the stride
          of ``conv_k``. Defaults to the inverse of the stride in ``conv_kwargs``.
        skip_connection: If ``True``, add a residual skip branch.
        skip_scale: Factor that scales the sum of both branches; ``1`` gives the
          standard residual sum, ``0.5`` the average, and ``1 / sqrt(2)``
          preserves the variance of two independent branches.
        drop_path: Probability of dropping the main branch of a sample during
          training; ``0`` disables drop path. Requires the skip branch.
        interp_mode: Interpolation mode for upsampling.
        skip_interp_mode: Interpolation mode that resizes the skip branch to the
          length of the main branch; ``area`` averages when downsampling.
        conv: Convolution layer factory used in the block.
        conv_kwargs: Optional keyword arguments for ``conv_k``; the 1x1
          convolutions do not receive them. Defaults to replicate padding that
          keeps the length at stride 1.

    References:
        Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
        https://arxiv.org/abs/1603.09382
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        normalization: NormalizationFactory | bool = True,
        normalization_channels: int | None = None,
        activation: nn.Module | bool = True,
        activation_channels: int | None = None,
        output_channels: int | None = None,
        dropout: float = 0.0,
        scale_factor: float | None = None,
        skip_connection: bool = False,
        skip_scale: float = 1.0,
        drop_path: float = 0.0,
        interp_mode: str = "nearest-exact",
        skip_interp_mode: str = "area",
        enable_spectral_norm: bool = False,  # TODO
        conv: ModuleFactory = nn.Conv1d,
        conv_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= drop_path < 1:
            raise ValueError(f"invalid arg: {drop_path=} must be in [0, 1)")
        if 0 < drop_path and not skip_connection:
            raise ValueError(f"invalid args: {drop_path=} requires skip_connection")

        # set attributes from arguments
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels
        self.kernel_size = kernel_size
        self.skip_scale = skip_scale
        self.drop_path = drop_path
        self.interp_mode = interp_mode
        self.skip_interp_mode = skip_interp_mode

        # copy to avoid modifying input args
        self.conv_kwargs = dict(conv_kwargs or {})

        # set scale factor and stride
        stride = self.conv_kwargs.get("stride", 1)
        if scale_factor is None:
            self.scale_factor = 1.0 / stride
        else:
            if scale_factor <= 0:
                raise ValueError(f"scale factor must be positive, got {scale_factor=}")
            if scale_factor < 1:
                expected_stride = round(1 / scale_factor)
                if not math.isclose(expected_stride * scale_factor, 1.0):
                    raise ValueError(
                        f"scale factor must be the inverse of an integer, got {scale_factor=}"
                    )
            else:
                expected_stride = 1
            if "stride" in self.conv_kwargs and stride != expected_stride:
                raise ValueError(
                    f"{scale_factor=} requires stride={expected_stride}, got {stride=}"
                )
            stride = expected_stride
            self.scale_factor = float(scale_factor)
        self.conv_kwargs["stride"] = stride

        # set default padding (if keys do not exist)
        # NOTE: PyTorch supports "same" padding only at stride 1
        dilation = self.conv_kwargs.get("dilation", 1)
        self.conv_kwargs.setdefault("padding_mode", "replicate")
        self.conv_kwargs.setdefault(
            "padding", "same" if 1 == stride else dilation * (kernel_size - 1) // 2
        )

        # resolve normalization
        if normalization is False:
            if normalization_channels is not None:
                raise ValueError(
                    f"invalid args: {normalization_channels=} requires normalization"
                )
        else:
            if normalization is True:
                normalization = Normalization
            if normalization_channels is None:
                normalization_channels = input_channels

        # resolve activation
        if activation is False:
            if activation_channels is not None:
                raise ValueError(
                    f"invalid args: {activation_channels=} requires activation"
                )
        else:
            if activation is True:
                activation = nn.GELU()
            if activation_channels is None:
                activation_channels = 4 * (normalization_channels or input_channels)

        # create main branch
        block = OrderedDict()
        channels = normalization_channels or activation_channels or self.output_channels
        block["conv_0"] = conv(
            input_channels, channels, kernel_size, **self.conv_kwargs
        )
        if normalization is not False:
            block["normalization"] = normalization(channels)
            next_channels = activation_channels or self.output_channels
            block["conv_1"] = conv(channels, next_channels, 1)
            channels = next_channels
        if activation is not False:
            block["activation"] = activation
        if 0 < dropout:
            block["dropout"] = nn.Dropout(dropout)
        if activation is not False:
            block["conv_2"] = conv(channels, self.output_channels, 1)
        self.block = nn.Sequential(block)

        # create skip branch
        self.skip_connection: nn.Module | None
        if not skip_connection:
            self.skip_connection = None
        elif self.input_channels == self.output_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv(input_channels, self.output_channels, 1)

        # initialize parameters
        self.init_parameters()

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Apply the block to one or more channel-compatible input tensors.

        Args:
            *x: One or more tensors with shape ``(batch, channels, length)``,
              concatenated along the channels.

        Returns:
            Output tensor with ``output_channels`` channels and the length scaled
            by ``scale_factor``.
        """
        channel_dim, size_dim = 1, 2

        # concatenate inputs along channel dimension
        h = torch.cat(x, dim=channel_dim) if 1 < len(x) else x[0]
        assert (
            h.size(channel_dim) == self.input_channels
        ), f"expected {self.input_channels=}, got {h.size(channel_dim)=}"

        # scale up
        if 1.0 < self.scale_factor:
            h = nn.functional.interpolate(
                h, scale_factor=self.scale_factor, mode=self.interp_mode
            )

        # apply main branch
        y = self.block(h)

        # add skip branch (optional)
        if self.skip_connection is not None:
            # drop main branch per sample (training only)
            if self.training and 0 < self.drop_path:
                keep_prob = 1.0 - self.drop_path
                mask = torch.bernoulli(y.new_full((y.size(0), 1, 1), keep_prob))
                y = y * mask / keep_prob
            # resize skip branch to the length of the main branch
            if h.size(size_dim) != y.size(size_dim):
                h = nn.functional.interpolate(
                    h, size=y.size(size_dim), mode=self.skip_interp_mode
                )
            y = self.skip_scale * (self.skip_connection(h) + y)
        return y

    def resolve_output_length(self, input_length: int) -> int:
        """Return the sequence length of the output for an input sequence length.

        Follows `forward` without running it: upsampling by interpolation, then
        the length arithmetic of ``nn.Conv1d`` for ``conv_k``. The skip branch
        is resized to the main branch and does not affect the length.

        Args:
            input_length: Sequence length of the input tensors.

        Returns:
            Sequence length of the output tensor.
        """

        def first(value: Any) -> Any:
            # unpack a one-element tuple, as `nn.Conv1d` accepts `(n,)` for `n`
            return value[0] if isinstance(value, (tuple, list)) else value

        # scale up as `interpolate` does
        length = input_length
        if 1.0 < self.scale_factor:
            length = math.floor(length * self.scale_factor)

        # apply convolution
        padding = self.conv_kwargs["padding"]
        if "same" == padding:
            return length
        if "valid" == padding:
            padding = 0
        padding = first(padding)
        dilation = first(self.conv_kwargs.get("dilation", 1))
        stride = self.conv_kwargs["stride"]
        return (
            length + 2 * padding - dilation * (self.kernel_size - 1) - 1
        ) // stride + 1

    def init_parameters(self) -> None:
        """Initialize trainable parameters for all active convolutional layers.

        A convolution that feeds the activation gets the gain of the activation,
        and every other convolution the linear gain. With a skip branch, the last
        convolution of the main branch starts at zero, so the block starts as its
        skip branch; spectral normalization cannot divide by a zero weight, so
        parametrized convolutions keep their initialization.
        """
        # initialize main branch
        children = list(self.block.named_children())
        init_modules = []
        for index, (name, module) in enumerate(children):
            if not name.startswith("conv_"):
                continue
            next_name = children[index + 1][0] if index + 1 < len(children) else None
            if "activation" == next_name:
                gain = get_gain(self.block.activation, default="conv1d")
            else:
                gain = get_gain(None, default="conv1d")
            set_init_parameters(module, gain)
            init_modules.append(module)

        # initialize skip branch
        if self.skip_connection is None:
            return
        if not parametrize.is_parametrized(init_modules[-1]):
            set_zero_parameters(init_modules[-1])
        if not isinstance(self.skip_connection, nn.Identity):
            set_init_parameters(self.skip_connection)


class DownsampleBlock(UniversalMultiLevelBlock):
    """
    Downsampling based on the universal multi-level block.

    Args:
        input_channels: Channels in input tensors.
        kernel_size: Convolution kernel size.
        scale_factor: Relative downsampling factor for sequence length.
        skip_connection: If ``True``, add a residual skip branch.
        skip_interp_mode: Interpolation mode that downsamples the skip branch.
        **block_kwargs: Additional parameters forwarded to
          ``UniversalMultiLevelBlock``.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        scale_factor: float = 0.5,
        skip_connection: bool = True,
        skip_interp_mode: str = "area",
        **block_kwargs: Any,
    ) -> None:
        super().__init__(
            input_channels,
            kernel_size,
            scale_factor=scale_factor,
            skip_connection=skip_connection,
            skip_interp_mode=skip_interp_mode,
            **block_kwargs,
        )


class UpsampleBlock(UniversalMultiLevelBlock):
    """
    Upsampling based on the universal multi-level block.

    Args:
        input_channels: Channels in input tensors.
        kernel_size: Convolution kernel size.
        scale_factor: Relative upsampling factor for sequence length.
        skip_connection: If ``True``, add a residual skip branch.
        interp_mode: Interpolation mode for upsampling.
        **block_kwargs: Additional parameters forwarded to
          ``UniversalMultiLevelBlock``.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        scale_factor: float = 2.0,
        skip_connection: bool = True,
        interp_mode: str = "nearest-exact",
        **block_kwargs: Any,
    ) -> None:
        super().__init__(
            input_channels,
            kernel_size,
            scale_factor=scale_factor,
            skip_connection=skip_connection,
            interp_mode=interp_mode,
            **block_kwargs,
        )


class LevelBlock(UniversalMultiLevelBlock):
    """
    Residual block based on the universal multi-level block.

    Args:
        input_channels: Channels in input tensors.
        kernel_size: Convolution kernel size.
        skip_connection: If ``True``, add a residual skip branch.
        **block_kwargs: Additional parameters forwarded to
          ``UniversalMultiLevelBlock``.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        skip_connection: bool = True,
        **block_kwargs: Any,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            kernel_size=kernel_size,
            skip_connection=skip_connection,
            **block_kwargs,
        )


# --------------------------------------
# ConvNeXt Components
# --------------------------------------


class ChannelLayerNorm(nn.LayerNorm):
    r"""
    Layer normalization over the channels of 1D feature maps.

    Normalizes each position of a tensor with shape ``(batch, channels, length)``
    over its channels, as ``nn.LayerNorm`` normalizes each token of a transformer
    over its features:

    .. math::
        y_{b,c,l} = \gamma_c \frac{x_{b,c,l} - \mu_{b,l}}{\sqrt{\sigma^2_{b,l}
        + \epsilon}} + \beta_c

    where :math:`\mu_{b,l}` and :math:`\sigma^2_{b,l}` are the mean and variance
    over the channels at position :math:`l`.

    Args:
        num_channels: Number of channels in the normalized tensor.
        eps: Value added to the variance for numerical stability.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize an input tensor over its channels.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.

        Returns:
            Normalized tensor with the shape of the input.
        """
        assert (
            3 == x.dim() and x.size(1) == self.normalized_shape[0]
        ), f"Expect shape (batch, {self.normalized_shape[0]}, length), got {tuple(x.shape)}"
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock(UniversalMultiLevelBlock):
    """
    ConvNeXt block based on the universal multi-level block.

    The block adds its main branch to its input::

        y = x + drop_path(conv_1x1(activation(conv_1x1(norm(dwconv_k(x))))))

    The depthwise convolution ``dwconv_k`` mixes along the length within each
    channel, and the two 1x1 convolutions mix across channels, widening to four
    times the channels in between. Unlike the original, the block has no layer
    scale; the last 1x1 convolution starts at zero instead, so the block starts
    as the identity.

    Args:
        input_channels: Channels in input tensors.
        kernel_size: Kernel size of the depthwise convolution.
        skip_connection: If ``True``, add a residual skip branch.
        **block_kwargs: Additional parameters forwarded to
          ``UniversalMultiLevelBlock``; entries of ``conv_kwargs`` override the
          defaults of one group per channel and zero padding.

    References:
        Liu et al., "A ConvNet for the 2020s", CVPR 2022.
        https://arxiv.org/abs/2201.03545
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int = 7,
        skip_connection: bool = True,
        **block_kwargs: Any,
    ) -> None:
        # set defaults of ConvNeXt (if keys do not exist)
        conv_kwargs = {"groups": input_channels, "padding_mode": "zeros"}
        conv_kwargs.update(block_kwargs.pop("conv_kwargs", None) or {})
        block_kwargs.setdefault("normalization", ChannelLayerNorm)
        block_kwargs.setdefault("activation", nn.GELU())
        super().__init__(
            input_channels,
            kernel_size,
            skip_connection=skip_connection,
            conv_kwargs=conv_kwargs,
            **block_kwargs,
        )
