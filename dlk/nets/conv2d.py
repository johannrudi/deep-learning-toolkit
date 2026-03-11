"""2D convolutional network architectures and reusable building blocks."""

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from dlk.nets.utils import (
    Activation,
    NormalizationFactory,
    get_gain,
    set_init_parameters,
    set_zero_parameters,
)

# --------------------------------------
# Convolutional Nets
# --------------------------------------


class ConvNet(nn.Module):
    """Build a stacked 2D convolutional network with optional dense layers and an optional output layer.

    Args:
        input_channels: Number of channels in the input tensor.
        hidden_conv_layers_channels_mult: Multipliers used to derive each convolution layer's
            output channels from `input_channels`.
        hidden_conv_layers_kernels: Kernel sizes for hidden convolution layers.
        hidden_conv_layers_activation: Activation applied after each hidden convolution layer.
        hidden_conv_layers_kwargs: Extra keyword arguments for hidden convolution layers.
        hidden_dense_input_size: Flattened feature size that feeds the first dense layer.
        hidden_dense_layers_sizes: Hidden dense layer sizes.
        hidden_dense_layers_activation: Activation applied after each hidden dense layer.
        hidden_dense_layers_kwargs: Extra keyword arguments for hidden dense layers.
        output_size: Output size for the optional output dense layer.
        output_layer_activation: Activation applied after the output layer.
        output_layer_kwargs: Extra keyword arguments for the output layer.
        use_dropout: Dropout probability. Falsy values disable dropout.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_conv_layers_channels_mult: Sequence[int] | None = None,
        hidden_conv_layers_kernels: Sequence[int] | None = None,
        hidden_conv_layers_activation: Activation | None = nn.ReLU(),
        hidden_conv_layers_kwargs: dict[str, Any] | None = None,
        hidden_dense_input_size: int | None = None,
        hidden_dense_layers_sizes: Sequence[int] | None = None,
        hidden_dense_layers_activation: Activation | None = nn.ReLU(),
        hidden_dense_layers_kwargs: dict[str, Any] | None = None,
        output_size: int | None = None,
        output_layer_activation: Activation | None = None,
        output_layer_kwargs: dict[str, Any] | None = None,
        use_dropout: float | bool = False,
    ) -> None:
        super().__init__()
        hidden_conv_layers_channels_mult = list(
            hidden_conv_layers_channels_mult or [8, 16, 32]
        )
        hidden_conv_layers_kernels = list(hidden_conv_layers_kernels or [3, 3, 3])
        hidden_conv_layers_kwargs = dict(hidden_conv_layers_kwargs or {})
        hidden_dense_layers_sizes = list(hidden_dense_layers_sizes or [])
        hidden_dense_layers_kwargs = dict(hidden_dense_layers_kwargs or {})
        output_layer_kwargs = dict(output_layer_kwargs or {})
        # set from arguments
        self.input_channels = input_channels
        self.hidden_conv_layers_activation = hidden_conv_layers_activation
        self.hidden_dense_layers_activation = hidden_dense_layers_activation
        self.output_layer_activation = output_layer_activation
        self.dropout = nn.Dropout(float(use_dropout)) if use_dropout else None
        # create hidden convolutional layers
        assert len(hidden_conv_layers_channels_mult) == len(
            hidden_conv_layers_kernels
        ), "hidden_conv_layers_channels_mult and hidden_conv_layers_kernels must match."
        in_channels = input_channels
        self.hidden_conv_layers = nn.ModuleList()
        for channel_mult, kernel_size in zip(
            hidden_conv_layers_channels_mult, hidden_conv_layers_kernels
        ):
            out_channels = channel_mult * input_channels
            layer = nn.Conv2d(
                in_channels, out_channels, kernel_size, **hidden_conv_layers_kwargs
            )
            self.hidden_conv_layers.append(layer)
            in_channels = out_channels
        # create hidden dense layers
        assert (
            hidden_dense_input_size is not None or len(hidden_dense_layers_sizes) == 0
        ), (
            "hidden_dense_input_size must be provided when hidden_dense_layers_sizes "
            "is non-empty."
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
        """Apply the model function to the input tensor.

        Args:
            x: Input tensor with shape `[batch, channels, height, width]`.

        Returns:
            Output tensor after convolutional, dense, and optional output layers.
        """
        assert (
            x.ndim == 4
        ), f"expected a 4D tensor [N, C, H, W], got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.input_channels
        ), f"input has {x.size(1)} channels but expected {self.input_channels}."
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

    def init_parameters(self) -> None:
        """Initialize trainable parameters with activation-aware gains."""
        # initialize hidden convolutional layers
        gain = get_gain(self.hidden_conv_layers_activation, default="conv2d")
        for layer in self.hidden_conv_layers:
            set_init_parameters(cast(nn.Conv2d, layer), gain)
        # initialize hidden dense layers
        gain = get_gain(self.hidden_dense_layers_activation, default="conv2d")
        for layer in self.hidden_dense_layers:
            set_init_parameters(cast(nn.Linear, layer), gain)
        # initialize output layer
        gain = get_gain(self.output_layer_activation, default="conv2d")
        if self.output_layer is not None:
            set_init_parameters(self.output_layer, gain, bias_scale=0.0)


# --------------------------------------
# Upsample Convolutional Nets
# --------------------------------------


class ConvUpsampleNet_Reshuffle(nn.Module):
    """Upsample feature maps with pixel reshuffling (depth-to-space) and convolutions.

    Args:
        input_channels: Number of channels in the input tensor.
        scale_factor: Spatial upsampling factor used by pixel shuffle.
        pre_up_layers_channels: Channels for hidden convolution layers before upsampling.
        pre_up_layers_kernels: Kernel sizes for hidden convolution layers before upsampling.
        pre_up_layers_activation: Activation applied before upsampling.
        pre_up_layers_kwargs: Extra keyword arguments for pre-upsampling convolutions.
        post_up_layers_channels: Channels for hidden convolution layers after upsampling.
        post_up_layers_kernels: Kernel sizes for hidden convolution layers after upsampling.
        post_up_layers_activation: Activation applied around post-upsampling layers.
        post_up_layers_kwargs: Extra keyword arguments for post-upsampling convolutions.
        output_activation: Activation applied to final network output.
        use_dropout: Dropout probability. Falsy values disable dropout.
    """

    def __init__(
        self,
        input_channels: int,
        scale_factor: int,
        pre_up_layers_channels: Sequence[int] | None = None,
        pre_up_layers_kernels: Sequence[int] | None = None,
        pre_up_layers_activation: Activation | None = nn.ReLU(),
        pre_up_layers_kwargs: dict[str, Any] | None = None,
        post_up_layers_channels: Sequence[int] | None = None,
        post_up_layers_kernels: Sequence[int] | None = None,
        post_up_layers_activation: Activation | None = nn.ReLU(),
        post_up_layers_kwargs: dict[str, Any] | None = None,
        output_activation: Activation | None = None,
        use_dropout: float | bool = False,
    ) -> None:
        super().__init__()
        pre_up_layers_channels = list(pre_up_layers_channels or [8, 16, 32])
        pre_up_layers_kernels = list(pre_up_layers_kernels or [3, 3, 3])
        pre_up_layers_kwargs = dict(pre_up_layers_kwargs or {})
        pre_up_layers_kwargs.setdefault("padding", 1)
        pre_up_layers_kwargs.setdefault("padding_mode", "replicate")
        post_up_layers_channels = list(post_up_layers_channels or [])
        post_up_layers_kernels = list(post_up_layers_kernels or [])
        post_up_layers_kwargs = dict(post_up_layers_kwargs or {})
        post_up_layers_kwargs.setdefault("padding", 1)
        post_up_layers_kwargs.setdefault("padding_mode", "replicate")
        # set from arguments
        self.input_channels = input_channels
        self.scale_factor = scale_factor
        self.pre_up_layers_activation = pre_up_layers_activation
        self.post_up_layers_activation = post_up_layers_activation
        self.output_activation = output_activation
        self.dropout = nn.Dropout(float(use_dropout)) if use_dropout else None
        # create hidden convolutional layers before upsampling
        assert len(pre_up_layers_channels) == len(
            pre_up_layers_kernels
        ), "pre_up_layers_channels and pre_up_layers_kernels must match."
        in_channels = input_channels
        self.pre_up_layers = nn.ModuleList()
        for channels, kernel_size in zip(pre_up_layers_channels, pre_up_layers_kernels):
            layer = nn.Conv2d(
                in_channels,
                channels,
                kernel_size,
                **pre_up_layers_kwargs,
            )
            self.pre_up_layers.append(layer)
            in_channels = channels
        # create upsample layer
        up_kernel_size = pre_up_layers_kernels[-1] if pre_up_layers_kernels else 3
        self.up_layer = nn.Conv2d(
            in_channels,
            input_channels * scale_factor**2,
            up_kernel_size,
            **pre_up_layers_kwargs,
        )
        # create hidden convolutional layers after upsampling
        assert len(post_up_layers_channels) == len(
            post_up_layers_kernels
        ), "post_up_layers_channels and post_up_layers_kernels must match."
        in_channels = input_channels
        self.post_up_layers = nn.ModuleList()
        for channels, kernel_size in zip(
            post_up_layers_channels, post_up_layers_kernels
        ):
            layer = nn.Conv2d(
                in_channels,
                channels,
                kernel_size,
                **post_up_layers_kwargs,
            )
            self.post_up_layers.append(layer)
            in_channels = channels
        # initialize parameters
        self.init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the model function to the input tensor.

        Args:
            x: Input tensor with shape `[batch, channels, height, width]`.

        Returns:
            Upsampled tensor with optional post-upsampling refinement.
        """
        assert (
            x.ndim == 4
        ), f"expected a 4D tensor [N, C, H, W], got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.input_channels
        ), f"input has {x.size(1)} channels but expected {self.input_channels}."
        h = x
        # apply hidden convolutional layers before upsampling
        for layer in self.pre_up_layers:
            h = layer(h)
            if self.pre_up_layers_activation is not None:
                h = self.pre_up_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply upsample layer
        h = self.up_layer(h)
        h = nn.functional.pixel_shuffle(h, self.scale_factor)
        # apply hidden convolutional layers after upsampling
        for layer in self.post_up_layers:
            if self.post_up_layers_activation is not None:
                h = self.post_up_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
            h = layer(h)
        # apply output activation
        if self.output_activation is not None:
            y = self.output_activation(h)
        else:
            y = h
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters with activation-aware gains."""
        # initialize hidden convolutional layers before upsampling
        gain = get_gain(self.pre_up_layers_activation, default="conv2d")
        for layer in self.pre_up_layers:
            set_init_parameters(cast(nn.Conv2d, layer), gain)
        # initialize hidden convolutional layers after upsampling
        if self.post_up_layers:
            gain = get_gain(self.post_up_layers_activation, default="conv2d")
            set_init_parameters(self.up_layer, gain)
            for layer in self.post_up_layers[:-1]:
                set_init_parameters(cast(nn.Conv2d, layer), gain)
            set_init_parameters(
                cast(nn.Conv2d, self.post_up_layers[-1]),
                get_gain(None, default="conv2d"),
            )
        else:
            set_init_parameters(self.up_layer, get_gain(None, default="conv2d"))


class ConvUpsampleNet_Interpolate(nn.Module):
    """Upsample feature maps by interpolation before selected convolution layers.

    Args:
        input_channels: Number of channels in the input tensor.
        conv_layers_channels: Channels for each convolution layer.
        conv_layers_kernels: Kernel sizes for each convolution layer.
        conv_layers_activation: Activation applied after each hidden convolution layer.
        conv_layers_kwargs: Extra keyword arguments for convolution layers.
        upsample_layer_indices: Layer indices before which interpolation is applied.
        scale_factor: Spatial upsampling factor passed to interpolation.
        interp_mode: Interpolation mode used by `torch.nn.functional.interpolate`.
        output_activation: Optional activation for the final layer output.
        use_dropout: Dropout probability. Falsy values disable dropout.
    """

    def __init__(
        self,
        input_channels: int,
        conv_layers_channels: Sequence[int] | None = None,
        conv_layers_kernels: Sequence[int] | None = None,
        conv_layers_activation: Activation | None = nn.ReLU(),
        conv_layers_kwargs: dict[str, Any] | None = None,
        upsample_layer_indices: Sequence[int] | None = None,
        scale_factor: int = 2,
        interp_mode: str = "nearest",
        output_activation: Activation | None = None,
        use_dropout: float | bool = False,
    ) -> None:
        super().__init__()
        conv_layers_channels = list(conv_layers_channels or [4, 4, 1])
        conv_layers_kernels = list(conv_layers_kernels or [3, 3, 3])
        conv_layers_kwargs = dict(conv_layers_kwargs or {})
        conv_layers_kwargs.setdefault("padding", 1)
        conv_layers_kwargs.setdefault("padding_mode", "replicate")
        upsample_layer_indices = list(upsample_layer_indices or [1])
        # set from arguments
        self.input_channels = input_channels
        self.upsample_layer_indices = upsample_layer_indices
        self.scale_factor = scale_factor
        self.interp_mode = interp_mode
        self.conv_layers_activation = conv_layers_activation
        self.output_activation = output_activation
        self.dropout = nn.Dropout(float(use_dropout)) if use_dropout else None
        # create hidden convolutional layers
        assert len(conv_layers_channels) == len(
            conv_layers_kernels
        ), "conv_layers_channels and conv_layers_kernels must match."
        in_channels = input_channels
        self.conv_layers = nn.ModuleList()
        for channels, kernel_size in zip(conv_layers_channels, conv_layers_kernels):
            layer = nn.Conv2d(
                in_channels,
                channels,
                kernel_size,
                **conv_layers_kwargs,
            )
            self.conv_layers.append(layer)
            in_channels = channels
        # initialize parameters
        self.init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the model function to the input tensor.

        Args:
            x: Input tensor with shape `[batch, channels, height, width]`.

        Returns:
            Upsampled tensor after interpolation and convolution stages.
        """
        assert (
            x.ndim == 4
        ), f"expected a 4D tensor [N, C, H, W], got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.input_channels
        ), f"input has {x.size(1)} channels but expected {self.input_channels}."
        h = x
        # apply hidden convolutional layers
        for i, layer in enumerate(self.conv_layers):
            if i in self.upsample_layer_indices:
                h = nn.functional.interpolate(
                    h, scale_factor=self.scale_factor, mode=self.interp_mode
                )
            h = layer(h)
            if (
                self.conv_layers_activation is not None
                and i < len(self.conv_layers) - 1
            ):
                h = self.conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply output activation
        if self.output_activation is not None:
            y = self.output_activation(h)
        else:
            y = h
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters with activation-aware gains."""
        gain = get_gain(self.conv_layers_activation, default="conv2d")
        for layer in self.conv_layers[:-1]:
            set_init_parameters(cast(nn.Conv2d, layer), gain)
        set_init_parameters(
            cast(nn.Conv2d, self.conv_layers[-1]),
            get_gain(None, default="conv2d"),
        )


# --------------------------------------
# UNet Components
# --------------------------------------


class Downsample(nn.Module):
    """Downsample a feature map with a stride convolution and optional post-processing.

    Args:
        input_channels: Number of channels in the input tensor.
        output_channels: Number of channels in the output tensor.
        kernel_size: Convolution kernel size.
        activation: Activation applied after convolution.
        dropout: Dropout module applied after activation.
        scale_factor: Downsampling factor applied through convolution stride.
        **conv_kwargs: Additional `nn.Conv2d` keyword arguments.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int],
        activation: Activation | None = None,
        dropout: nn.Module | None = None,
        scale_factor: int = 2,
        **conv_kwargs: Any,
    ) -> None:
        super().__init__()
        conv_kwargs = dict(conv_kwargs)
        conv_kwargs.setdefault("padding", 1)
        conv_kwargs.setdefault("padding_mode", "replicate")
        conv_kwargs["stride"] = scale_factor
        # create convolutional layer with stride=scale_factor
        self.layer = nn.Conv2d(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling to an input tensor.

        Args:
            x: Input tensor with shape `[batch, channels, height, width]`.

        Returns:
            Downsampled tensor.
        """
        assert (
            x.ndim == 4
        ), f"expected a 4D tensor [N, C, H, W], got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.input_channels
        ), f"input has {x.size(1)} channels but expected {self.input_channels}."
        h = self.layer(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters with activation-aware gains."""
        set_init_parameters(self.layer, get_gain(self.activation, default="conv2d"))


class Upsample(nn.Module):
    """Upsample a feature map by interpolation followed by convolution.

    Args:
        input_channels: Number of channels in the input tensor.
        output_channels: Number of channels in the output tensor.
        kernel_size: Convolution kernel size.
        activation: Activation applied after convolution.
        dropout: Dropout module applied after activation.
        scale_factor: Upsampling factor used by interpolation.
        interp_mode: Interpolation mode used by `torch.nn.functional.interpolate`.
        **conv_kwargs: Additional `nn.Conv2d` keyword arguments.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int],
        activation: Activation | None = None,
        dropout: nn.Module | None = None,
        scale_factor: int = 2,
        interp_mode: str = "nearest",
        **conv_kwargs: Any,
    ) -> None:
        super().__init__()
        conv_kwargs = dict(conv_kwargs)
        conv_kwargs.setdefault("padding", 1)
        conv_kwargs.setdefault("padding_mode", "replicate")
        # create convolutional layer
        self.layer = nn.Conv2d(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling to an input tensor.

        Args:
            x: Input tensor with shape `[batch, channels, height, width]`.

        Returns:
            Upsampled tensor.
        """
        assert (
            x.ndim == 4
        ), f"expected a 4D tensor [N, C, H, W], got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.input_channels
        ), f"input has {x.size(1)} channels but expected {self.input_channels}."
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

    def init_parameters(self) -> None:
        """Initialize trainable parameters with activation-aware gains."""
        set_init_parameters(self.layer, get_gain(self.activation, default="conv2d"))


class LevelBlock(nn.Module):
    """Apply a residual convolution block with optional normalization and dropout.

    Args:
        input_channels: Number of channels in the input tensor.
        output_channels: Number of channels in the output tensor.
        kernel_size: Convolution kernel size.
        normalization: Normalization module applied after residual addition.
        activation: Activation applied after normalization.
        dropout: Dropout module applied after activation.
        **conv_kwargs: Additional `nn.Conv2d` keyword arguments.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int],
        normalization: nn.Module | None = None,
        activation: Activation | None = None,
        dropout: nn.Module | None = None,
        **conv_kwargs: Any,
    ) -> None:
        super().__init__()
        conv_kwargs = dict(conv_kwargs)
        conv_kwargs.setdefault("padding", 1)
        conv_kwargs.setdefault("padding_mode", "replicate")
        # set from arguments
        self.input_channels = input_channels
        # create convolutional layer
        self.layer = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            **conv_kwargs,
        )
        # create residual/skip connection
        if input_channels == output_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_channels, output_channels, 1)
        # create additional layers
        self.normalization = normalization
        self.activation = activation
        self.dropout = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual level block to an input tensor.

        Args:
            x: Input tensor with shape `[batch, channels, height, width]`.

        Returns:
            Output tensor after residual addition and optional post-processing.
        """
        assert (
            x.ndim == 4
        ), f"expected a 4D tensor [N, C, H, W], got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.input_channels
        ), f"input has {x.size(1)} channels but expected {self.input_channels}."
        h = self.layer(x) + self.residual_layer(x)
        if self.normalization is not None:
            h = self.normalization(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters for residual behavior."""
        set_zero_parameters(self.layer)
        if not isinstance(self.residual_layer, nn.Identity):
            set_init_parameters(self.residual_layer, get_gain(None, default="conv2d"))


def Normalization(num_channels: int, num_groups: int = 1) -> nn.GroupNorm:
    """Construct group normalization for image-like tensors.

    Args:
        num_channels: Number of channels in the input tensor.
        num_groups: Number of groups used by group normalization.

    Returns:
        A `torch.nn.GroupNorm` layer.
    """
    return nn.GroupNorm(num_groups, num_channels)


# TODO replace by implementation of LevelBlock
class ResBlock(nn.Module):
    """Apply a residual block that can optionally change the channel dimension.

    Args:
        input_channels: Number of channels in the input tensor.
        output_channels: Number of channels in the output tensor. If `None`, uses
            `input_channels`.
        use_conv: If `True`, use a 3x3 convolution in the skip path when channel
            dimensions change. If `False`, use a 1x1 convolution.
        normalization: Factory that creates normalization modules from channel counts.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int | None = None,
        use_conv: bool = False,
        normalization: NormalizationFactory | None = None,
    ) -> None:
        super().__init__()
        normalization = normalization or Normalization
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels
        # create input layers
        self.in_layers = nn.Sequential(
            normalization(input_channels),
            nn.SiLU(),
            nn.Conv2d(
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
                nn.Conv2d(
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
            self.skip_connection = nn.Conv2d(
                input_channels,
                self.output_channels,
                3,
                padding=1,
                padding_mode="replicate",
            )
        else:
            self.skip_connection = nn.Conv2d(input_channels, self.output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to an input tensor.

        Args:
            x: Input tensor with shape `[batch, channels, height, width]`.

        Returns:
            Output tensor with residual connection applied.
        """
        assert (
            x.ndim == 4
        ), f"expected a 4D tensor [N, C, H, W], got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.input_channels
        ), f"input has {x.size(1)} channels but expected {self.input_channels}."
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h
