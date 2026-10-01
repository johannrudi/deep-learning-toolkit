"""EfficientNet-inspired 1D convolutional network for time-series classification.

TODO: write a meaningful overview about efficient net.

For implementation and usage details, see:
- docs/features/2026.004__efficient_net__1-explore.md
- docs/features/2026.004__efficient_net__2-plan.md
- docs/features/2026.009__efficientnet_spectral_norm__1-plan.md
- docs/features/2026.009__efficientnet_spectral_norm__2-usage.md
- docs/features/2026.010__efficientnet_arch_mod__2-plan.md
"""

import math
from dataclasses import dataclass, replace
from typing import Literal, Self, cast

import torch
import torch.nn as nn

from dlk.nets.spectral_norm import (
    get_depthwise_spectral_norm,
    get_spectral_norm,
    get_weight_parametrizations,
    set_depthwise_spectral_norm,
    set_spectral_norm,
)
from dlk.nets.utils import NormalizationFactory, set_init_parameters

# --------------------------------------
# Config
# --------------------------------------


@dataclass(frozen=True)
class MBConvConfig:
    """Store one MBConv or Fused-MBConv block configuration.

    See ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, decisions
    13 and 17.

    Attributes:
        kernel_size: Convolution kernel size for the block's spatial filter.
        stride: Stride of the block's spatial filter.
        expand_ratio: Channel expansion factor for the bottleneck.
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        se_ratio: Squeeze-and-excitation channel reduction ratio, or `None`
            to disable squeeze-and-excitation (required for Fused-MBConv).
        dropout: Element-wise dropout probability on the residual branch.
        drop_path: Probability of dropping the residual branch per sample.
    """

    kernel_size: int
    stride: int
    expand_ratio: int
    input_channels: int
    output_channels: int
    se_ratio: float | None
    dropout: float = 0.0
    drop_path: float = 0.0

    def __post_init__(self) -> None:
        """Validate the dropout and drop path rates."""
        assert (
            0.0 <= self.dropout < 1.0
        ), f"dropout must be in the range [0, 1), got {self.dropout}"
        assert (
            0.0 <= self.drop_path < 1.0
        ), f"drop_path must be in the range [0, 1), got {self.drop_path}"
        assert not (
            self.dropout > 0 and self.drop_path > 0
        ), f"at most one of dropout and drop_path may be positive, got {self.dropout=}, {self.drop_path=}"
        assert (
            self.has_residual or self.drop_path == 0
        ), f"drop_path needs a residual branch, got {self.drop_path=} with {self.stride=}, {self.input_channels=}, {self.output_channels=}"

    @property
    def has_residual(self) -> bool:
        """Whether the block adds its input to the branch output."""
        return self.stride == 1 and self.input_channels == self.output_channels

    def scaled(
        self,
        width_coefficient: float,
        depth_divisor: int = 8,
        min_depth: int | None = None,
    ) -> Self:
        """Return the config with both channel counts scaled by `round_filters`."""
        return replace(
            self,
            input_channels=round_filters(
                self.input_channels, width_coefficient, depth_divisor, min_depth
            ),
            output_channels=round_filters(
                self.output_channels, width_coefficient, depth_divisor, min_depth
            ),
        )


@dataclass(frozen=True)
class StageConfig:
    """Pair a block config with the block class and the number of blocks of a stage.

    Attributes:
        block_cls: Block module class (`MBConv1D` or `FusedMBConv1D`), called
            as `block_cls(config, enable_spectral_norm=..., style=...)`.
        config: Config of the stage's first block.
        num_blocks: Number of blocks in the stage.
    """

    block_cls: type[nn.Module]
    config: MBConvConfig
    num_blocks: int

    def __post_init__(self) -> None:
        """Validate the number of blocks."""
        assert (
            self.num_blocks >= 1
        ), f"num_blocks must be at least 1, got {self.num_blocks}"

    def scaled(
        self,
        width_coefficient: float,
        depth_coefficient: float,
        depth_divisor: int = 8,
        min_depth: int | None = None,
    ) -> Self:
        """Return the stage with scaled channels and `round_repeats` blocks."""
        return replace(
            self,
            config=self.config.scaled(width_coefficient, depth_divisor, min_depth),
            num_blocks=round_repeats(self.num_blocks, depth_coefficient),
        )

    def block_configs(self) -> list[MBConvConfig]:
        """Return the config of each block of the stage.

        The first block uses the stage config; the later blocks use stride 1 and
        keep the stage's output channels.
        """
        rest = replace(
            self.config, input_channels=self.config.output_channels, stride=1
        )
        return [self.config] + [rest] * (self.num_blocks - 1)


@dataclass(frozen=True)
class StemConfig:
    """Store the stem size.

    Attributes:
        channels: Number of output channels of the stem convolution.
    """

    channels: int

    def __post_init__(self) -> None:
        """Validate the channel count."""
        assert self.channels > 0, f"channels must be positive, got {self.channels}"

    def scaled(
        self,
        width_coefficient: float,
        depth_divisor: int = 8,
        min_depth: int | None = None,
    ) -> Self:
        """Return the config with `channels` scaled by `round_filters`."""
        return replace(
            self,
            channels=round_filters(
                self.channels, width_coefficient, depth_divisor, min_depth
            ),
        )


@dataclass(frozen=True)
class HeadConfig:
    """Store the head size and dropout.

    Attributes:
        channels: Number of output channels of the head convolution.
        dropout: Dropout probability before the final classifier.
    """

    channels: int = 1280
    dropout: float = 0.2

    def __post_init__(self) -> None:
        """Validate the channel count and the dropout probability."""
        assert self.channels > 0, f"channels must be positive, got {self.channels}"
        assert (
            0.0 <= self.dropout < 1.0
        ), f"dropout must be in [0, 1), got {self.dropout}"

    def scaled(
        self,
        width_coefficient: float,
        depth_divisor: int = 8,
        min_depth: int | None = None,
    ) -> Self:
        """Return the config with `channels` scaled by `round_filters`."""
        return replace(
            self,
            channels=round_filters(
                self.channels, width_coefficient, depth_divisor, min_depth
            ),
        )


# --------------------------------------
# Styles
# --------------------------------------


@dataclass(frozen=True)
class ConvNorm:
    """Store the bias of one convolution and the normalization after it.

    See ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, decision 4.

    Attributes:
        normalization: Normalization factory, or `None` for no normalization.
        bias: Whether the convolution has a bias.
    """

    normalization: NormalizationFactory | None = nn.BatchNorm1d
    bias: bool = False


BATCH_NORM = ConvNorm(nn.BatchNorm1d, bias=False)
NO_NORM = ConvNorm(None, bias=True)


@dataclass(frozen=True)
class MBConvStyle:
    """Store the design choices of every `MBConv1D` block of a network.

    See ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, decisions
    5, 6, and 11.

    Attributes:
        pre_normalization: Normalization of the block input inside the
            residual branch (P0), or `None`.
        expand: Expansion convolution (P1); ignored when `expand_ratio == 1`.
        depthwise: Depthwise convolution (P2).
        project: Projection convolution (P3).
        skip_scale: Factor on the residual sum.
        depthwise_spectral_norm: Spectral normalization of the depthwise
            convolution, through the kernel matrix (`"matrix"`, C0) or its
            exact operator norm (`"exact"`, C2).
    """

    pre_normalization: NormalizationFactory | None = None
    expand: ConvNorm = BATCH_NORM
    depthwise: ConvNorm = BATCH_NORM
    project: ConvNorm = BATCH_NORM
    skip_scale: float = 1.0
    depthwise_spectral_norm: Literal["matrix", "exact"] = "matrix"

    def __post_init__(self) -> None:
        """Validate the residual scale and the depthwise spectral norm."""
        assert (
            self.skip_scale > 0
        ), f"skip_scale must be positive, got {self.skip_scale}"
        assert self.depthwise_spectral_norm in (
            "matrix",
            "exact",
        ), f"depthwise_spectral_norm must be 'matrix' or 'exact', got {self.depthwise_spectral_norm!r}"


@dataclass(frozen=True)
class FusedMBConvStyle:
    """Store the design choices of every `FusedMBConv1D` block of a network.

    See ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, decisions
    5 and 6.

    Attributes:
        pre_normalization: Normalization of the block input inside the
            residual branch (F0), or `None`.
        fused: Fused convolution (F1).
        project: Projection convolution (F2); ignored when `expand_ratio == 1`.
        skip_scale: Factor on the residual sum.
    """

    pre_normalization: NormalizationFactory | None = None
    fused: ConvNorm = BATCH_NORM
    project: ConvNorm = BATCH_NORM
    skip_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the residual scale."""
        assert (
            self.skip_scale > 0
        ), f"skip_scale must be positive, got {self.skip_scale}"


@dataclass(frozen=True)
class StemStyle:
    """Store the design choices of the stem.

    Attributes:
        conv: Stem convolution.
    """

    conv: ConvNorm = BATCH_NORM


@dataclass(frozen=True)
class HeadStyle:
    """Store the design choices of the head.

    The whole head style, pre-norm included, is ignored when the network has
    no head (`head=None`).

    Attributes:
        pre_normalization: Normalization of the last block's output before the
            head convolution, or `None`.
        conv: Head convolution.
    """

    pre_normalization: NormalizationFactory | None = None
    conv: ConvNorm = BATCH_NORM


@dataclass(frozen=True)
class NetStyle:
    """Bundle the design choices that apply to the whole network.

    `NetStyle()` builds the paper's EfficientNet. See
    ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, decisions 8,
    10, and 17.

    Attributes:
        mbconv: Style of every `MBConv1D` block.
        fused: Style of every `FusedMBConv1D` block.
        stem: Style of the stem.
        head: Style of the head.
        enable_spectral_norm: Whether to spectrally normalize every convolution
            and linear layer.
        dropout_connect_mode: Whether the network's ramped `dropout_connect`
            rate sets the blocks' element-wise `dropout` or their `drop_path`.
    """

    mbconv: MBConvStyle = MBConvStyle()
    fused: FusedMBConvStyle = FusedMBConvStyle()
    stem: StemStyle = StemStyle()
    head: HeadStyle = HeadStyle()
    enable_spectral_norm: bool = False
    dropout_connect_mode: Literal["dropout", "drop_path"] = "dropout"

    def __post_init__(self) -> None:
        """Validate the dropout connect mode."""
        assert self.dropout_connect_mode in (
            "dropout",
            "drop_path",
        ), f"dropout_connect_mode must be 'dropout' or 'drop_path', got {self.dropout_connect_mode!r}"

    def for_block(self, block_cls: type[nn.Module]) -> MBConvStyle | FusedMBConvStyle:
        """Return the style of a block class.

        Args:
            block_cls: `MBConv1D`, `FusedMBConv1D`, or a subclass of either.

        Returns:
            `mbconv` for `MBConv1D` and `fused` for `FusedMBConv1D`.

        Raises:
            TypeError: If `block_cls` is neither.
        """
        if issubclass(block_cls, MBConv1D):
            return self.mbconv
        if issubclass(block_cls, FusedMBConv1D):
            return self.fused
        raise TypeError(
            f"block_cls must be MBConv1D or FusedMBConv1D, got {block_cls!r}"
        )


BASELINE = NetStyle()


def _conv_norm_activation(
    conv_norm: ConvNorm,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    activation: bool,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
) -> nn.Sequential:
    """Stack a convolution, its normalization if set, and SiLU if requested.

    Keeps the indices of today's layout: conv at 0, normalization at 1, SiLU at
    2; without normalization, SiLU moves to 1.

    Args:
        conv_norm: Bias of the convolution and the normalization after it.
        in_channels: Number of input channels of the convolution.
        out_channels: Number of output channels of the convolution.
        kernel_size: Kernel size of the convolution.
        activation: Whether to end with SiLU.
        stride: Stride of the convolution.
        padding: Padding of the convolution.
        groups: Number of groups of the convolution.

    Returns:
        nn.Sequential: The stacked layers.
    """
    layers: list[nn.Module] = []
    layers.append(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=conv_norm.bias,
        )
    )
    if conv_norm.normalization is not None:
        layers.append(conv_norm.normalization(out_channels))
    if activation:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _pre_norm(normalization: NormalizationFactory | None, channels: int) -> nn.Module:
    """Build a pre-norm for `channels` channels, or `nn.Identity()` if unset."""
    return nn.Identity() if normalization is None else normalization(channels)


def _add_residual(
    branch: torch.Tensor,
    identity: torch.Tensor,
    dropout: nn.Dropout | None,
    drop_path: float,
    skip_scale: float,
    training: bool,
) -> torch.Tensor:
    """Regularize the branch output, add the identity, and scale the sum.

    Args:
        branch: Output of the residual branch.
        identity: Block input.
        dropout: Element-wise dropout on the branch, or `None`.
        drop_path: Probability of dropping the branch per sample.
        skip_scale: Factor on the residual sum.
        training: Whether the block is in training mode.

    Returns:
        torch.Tensor: `skip_scale * (branch + identity)` after regularization.
    """
    if dropout is not None:
        branch = dropout(branch)
    # drop branch per sample (training only)
    if training and 0 < drop_path:
        keep_prob = 1.0 - drop_path
        mask = torch.bernoulli(branch.new_full((branch.size(0), 1, 1), keep_prob))
        branch = branch * mask / keep_prob
    return skip_scale * (branch + identity)


def round_filters(
    filters: int,
    width_coefficient: float,
    depth_divisor: int = 8,
    min_depth: int | None = None,
) -> int:
    """Scale and round a channel count under compound scaling.

    Direct port of upstream ``round_filters`` without the 0.9-safeguard branch.

    Args:
        filters: Base channel count before width scaling.
        width_coefficient: Multiplier applied to `filters`.
        depth_divisor: Rounding unit for the scaled channel count.
        min_depth: Lower bound on the rounded channel count, or `None` to use
            `depth_divisor`.

    Returns:
        int: Scaled channel count rounded to a multiple of `depth_divisor`.
    """
    if not width_coefficient:
        return filters

    filters_scaled = filters * width_coefficient
    min_depth = min_depth or depth_divisor
    new_filters = max(
        min_depth,
        int(filters_scaled + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    """Scale and round a stage repeat count under compound scaling.

    Direct port of upstream ``round_repeats``.

    Args:
        repeats: Base number of blocks in the stage.
        depth_coefficient: Multiplier applied to `repeats`.

    Returns:
        int: Scaled repeat count rounded up to the nearest integer.
    """
    if not depth_coefficient:
        return repeats
    return int(math.ceil(depth_coefficient * repeats))


# --------------------------------------
# Blocks
# --------------------------------------


class SqueezeExcitation1DConv(nn.Module):
    """Apply squeeze-and-excitation reweighting for 1D feature maps."""

    def __init__(self, channels: int, squeeze_channels: int) -> None:
        """Initialize the squeeze-and-excitation block.

        Args:
            channels: Number of channels in the input feature map.
            squeeze_channels: Bottleneck width for the squeeze path.
        """
        super().__init__()
        assert channels > 0, f"channels must be positive, got {channels}"
        assert (
            squeeze_channels > 0
        ), f"squeeze_channels must be positive, got {squeeze_channels}"

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, squeeze_channels, 1),
            nn.SiLU(),
            nn.Conv1d(squeeze_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweight channels of the input feature map.

        Args:
            x: Input tensor with shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Reweighted tensor with the same shape as `x`.
        """
        return x * self.se(x)


class SqueezeExcitation1DLinear(nn.Module):
    """Apply squeeze-and-excitation reweighting for 1D feature maps."""

    def __init__(
        self,
        channels: int,
        squeeze_channels: int,
        enable_spectral_norm: bool = False,
    ) -> None:
        """Initialize the squeeze-and-excitation block.

        Args:
            channels: Number of channels in the input feature map.
            squeeze_channels: Bottleneck width for the squeeze path.
            enable_spectral_norm: If `True`, wrap both projections with
                `parametrizations.spectral_norm`.
        """
        super().__init__()
        assert channels > 0, f"channels must be positive, got {channels}"
        assert (
            squeeze_channels > 0
        ), f"squeeze_channels must be positive, got {squeeze_channels}"

        # 1x1 convolutions on a length-1 squeeze are plain channel projections
        self.reduce = nn.Linear(channels, squeeze_channels)
        self.activation = nn.SiLU()
        self.expand = nn.Linear(squeeze_channels, channels)
        if enable_spectral_norm:
            # wrap with spectral norm in place so both attributes stay typed as nn.Linear
            set_spectral_norm(self.reduce)
            set_spectral_norm(self.expand)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweight channels of the input feature map.

        Args:
            x: Input tensor with shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Reweighted tensor with the same shape as `x`.
        """
        assert (
            x.dim() == 3
        ), f"SqueezeExcitation1DLinear expected a 3D tensor with shape (batch_size, channels, sequence_length), got {x.dim()}"
        # squeeze over the sequence axis, then gate channels
        scale = x.mean(dim=2)
        scale = self.gate(self.expand(self.activation(self.reduce(scale))))
        return x * scale.unsqueeze(-1)


class MBConv1D(nn.Module):
    """A Mobile Inverted Bottleneck Convolution block for 1D signals."""

    def __init__(
        self,
        config: MBConvConfig,
        enable_spectral_norm: bool = False,
        style: MBConvStyle = MBConvStyle(),
    ) -> None:
        """Initialize the MBConv block.

        Args:
            config: Layer and channel configuration for this block.
            enable_spectral_norm: If `True`, wrap every convolution and both
                squeeze-and-excitation projections with
                `parametrizations.spectral_norm`, or the depthwise convolution
                with `DepthwiseSpectralNorm` if `style` asks for it.
            style: Normalizations, biases, and residual scale of the block.
        """
        super().__init__()

        self.config = config
        self.style = style
        self.has_se = config.se_ratio is not None and config.se_ratio > 0

        # Pre-normalization of the residual branch
        self.pre_norm = _pre_norm(style.pre_normalization, config.input_channels)

        # Expansion phase
        expanded_channels = config.input_channels * config.expand_ratio
        if config.expand_ratio != 1:
            self.expand_conv = _conv_norm_activation(
                style.expand,
                config.input_channels,
                expanded_channels,
                1,
                activation=True,
            )
        else:
            self.expand_conv = nn.Identity()

        # Depthwise convolution
        self.depthwise_conv = _conv_norm_activation(
            style.depthwise,
            expanded_channels,
            expanded_channels,
            config.kernel_size,
            activation=True,
            stride=config.stride,
            padding=config.kernel_size // 2,
            groups=expanded_channels,
        )

        # Squeeze-and-Excitation (bottleneck sized from pre-expansion channels)
        self.se: SqueezeExcitation1DLinear | None = None
        if self.has_se:
            assert (
                config.se_ratio is not None
            ), "se_ratio must be set when has_se is True"
            squeeze_channels = max(1, int(config.input_channels * config.se_ratio))
            self.se = SqueezeExcitation1DLinear(
                expanded_channels,
                squeeze_channels,
                enable_spectral_norm=enable_spectral_norm,
            )

        # Output projection
        self.project_conv = _conv_norm_activation(
            style.project,
            expanded_channels,
            config.output_channels,
            1,
            activation=False,
        )

        # wrap every convolution with spectral norm in place
        if enable_spectral_norm:
            depthwise = self.depthwise_conv[0]
            for module in self.modules():
                if not isinstance(module, nn.Conv1d):
                    continue
                if module is depthwise and style.depthwise_spectral_norm == "exact":
                    set_depthwise_spectral_norm(module)
                else:
                    set_spectral_norm(module)

        # Dropout for residual connection
        self.dropout = nn.Dropout(config.dropout) if config.dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass through the MBConv block.

        Args:
            x: Input tensor with shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after MBConv transformations.
        """
        assert (
            x.dim() == 3
        ), f"MBConv1D expected a 3D tensor with shape (batch_size, channels, sequence_length), got {x.dim()}"
        assert (
            x.shape[1] == self.config.input_channels
        ), f"MBConv1D expected {self.config.input_channels} input channels, got {x.shape[1]}"
        identity = x
        x = self.pre_norm(x)

        # Expansion
        x = self.expand_conv(x)

        # Depthwise convolution
        x = self.depthwise_conv(x)

        # Squeeze-and-Excitation
        if self.se is not None:
            x = self.se(x)

        # Output projection
        x = self.project_conv(x)

        # Residual connection
        if self.config.has_residual:
            x = _add_residual(
                x,
                identity,
                self.dropout,
                self.config.drop_path,
                self.style.skip_scale,
                self.training,
            )

        return x


class FusedMBConv1D(nn.Module):
    """A Fused Mobile Inverted Bottleneck Convolution block for 1D signals.

    Introduced in EfficientNetV2, this block replaces the expansion and
    depthwise steps of `MBConv1D` with a single dense convolution, trading
    higher FLOPs for higher arithmetic intensity on modern accelerators.
    Unlike `MBConv1D`, it has no squeeze-and-excitation step.
    """

    def __init__(
        self,
        config: MBConvConfig,
        enable_spectral_norm: bool = False,
        style: FusedMBConvStyle = FusedMBConvStyle(),
    ) -> None:
        """Initialize the Fused-MBConv block.

        Args:
            config: Layer and channel configuration for this block.
                `config.se_ratio` must be `None`.
            enable_spectral_norm: If `True`, wrap every convolution with
                `parametrizations.spectral_norm`.
            style: Normalizations, biases, and residual scale of the block.
        """
        super().__init__()
        assert (
            config.se_ratio is None
        ), f"FusedMBConv1D does not support squeeze-and-excitation, got se_ratio={config.se_ratio}"

        self.config = config
        self.style = style

        # Pre-normalization of the residual branch
        self.pre_norm = _pre_norm(style.pre_normalization, config.input_channels)

        # Fused expansion and spatial filter; expansion ratio 1 collapses the
        # block to this single dense conv
        expanded_channels = config.input_channels * config.expand_ratio
        fused_channels = (
            config.output_channels if config.expand_ratio == 1 else expanded_channels
        )
        self.fused_conv = _conv_norm_activation(
            style.fused,
            config.input_channels,
            fused_channels,
            config.kernel_size,
            activation=True,
            stride=config.stride,
            padding=config.kernel_size // 2,
        )

        # Output projection
        self.project_conv: nn.Module
        if config.expand_ratio != 1:
            self.project_conv = _conv_norm_activation(
                style.project,
                expanded_channels,
                config.output_channels,
                1,
                activation=False,
            )
        else:
            self.project_conv = nn.Identity()

        # wrap every convolution with spectral norm in place
        if enable_spectral_norm:
            for module in self.modules():
                if isinstance(module, nn.Conv1d):
                    set_spectral_norm(module)

        # Dropout for residual connection
        self.dropout = nn.Dropout(config.dropout) if config.dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass through the Fused-MBConv block.

        Args:
            x: Input tensor with shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after Fused-MBConv transformations.
        """
        assert (
            x.dim() == 3
        ), f"FusedMBConv1D expected a 3D tensor with shape (batch_size, channels, sequence_length), got {x.dim()}"
        assert (
            x.shape[1] == self.config.input_channels
        ), f"FusedMBConv1D expected {self.config.input_channels} input channels, got {x.shape[1]}"
        identity = x
        x = self.pre_norm(x)

        # Fused expansion and spatial filter
        x = self.fused_conv(x)

        # Output projection
        x = self.project_conv(x)

        # Residual connection
        if self.config.has_residual:
            x = _add_residual(
                x,
                identity,
                self.dropout,
                self.config.drop_path,
                self.style.skip_scale,
                self.training,
            )

        return x


# --------------------------------------
# Scalable Base Network
# --------------------------------------


class ScalableEfficientNet1D(nn.Module):
    """Compound-scalable EfficientNet classifier for 1D time-series inputs.

    Builds stem, stages, and head from a list of ``StageConfig`` values,
    applying EfficientNet width and depth compound scaling.
    """

    def __init__(
        self,
        stage_configs: list[StageConfig],
        stem: StemConfig | None,
        head: HeadConfig | None = HeadConfig(),
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        depth_divisor: int = 8,
        min_depth: int | None = None,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        style: NetStyle = NetStyle(),
    ) -> None:
        """Initialize ScalableEfficientNet1D.

        Args:
            stage_configs: Per-stage block class, base configuration, and
                number of blocks.
            stem: Stem config before width scaling, or `None` to replace the
                stem by `nn.Identity()`; the network then expects input with
                `resolve_input_shape()[0]` channels instead of `input_channels`.
            head: Head config before width scaling, or `None` to replace the
                head by `nn.Identity()`; `forward` then returns the raw block
                output instead of class logits, and `style.head`, pre-norm
                included, is ignored.
            width_coefficient: Compound-scaling multiplier for channel counts.
            depth_coefficient: Compound-scaling multiplier for stage repeats.
            depth_divisor: Rounding unit for scaled channel counts.
            min_depth: Lower bound on rounded channel counts, or `None` to use
                `depth_divisor`.
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__()
        assert (
            input_channels > 0
        ), f"input_channels must be positive, got {input_channels}"
        assert (
            input_length is None or input_length > 0
        ), f"input_length must be positive when provided, got {input_length}"
        assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
        assert (
            0.0 <= dropout_connect < 1.0
        ), f"dropout_connect must be in [0, 1), got {dropout_connect}"
        assert (
            len(stage_configs) > 0
        ), f"stage_configs must be non-empty, got {repr(stage_configs)}"

        self.input_channels = input_channels
        self.input_length = input_length
        self.num_classes = num_classes
        self.enable_stem = stem is not None
        self.enable_head = head is not None
        self.style = style

        # store pre-scaling config for introspection (see `dlk/nets/cli_efficientnet.py`)
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.base_stem = stem
        self.base_head = head
        self.base_stage_configs = stage_configs

        # rescale channels and repeats under compound scaling
        self.stem_config = (
            None
            if stem is None
            else stem.scaled(width_coefficient, depth_divisor, min_depth)
        )
        self.head_config = (
            None
            if head is None
            else head.scaled(width_coefficient, depth_divisor, min_depth)
        )
        self.stage_configs = [
            stage.scaled(width_coefficient, depth_coefficient, depth_divisor, min_depth)
            for stage in stage_configs
        ]
        total_blocks = sum(stage.num_blocks for stage in self.stage_configs)

        # Stem
        self.stem: nn.Module
        if self.stem_config is not None:
            self.stem = _conv_norm_activation(
                style.stem.conv,
                input_channels,
                self.stem_config.channels,
                3,
                activation=True,
                stride=2,
                padding=1,
            )
        else:
            self.stem = nn.Identity()

        # Stage blocks (MBConv or Fused-MBConv per StageConfig)
        self.blocks = nn.ModuleList()
        for stage in self.stage_configs:
            for block_config in stage.block_configs():
                # ramp the rate linearly: element-wise dropout or drop path (stochastic depth)
                rate = dropout_connect * len(self.blocks) / total_blocks
                if style.dropout_connect_mode == "dropout":
                    block_config = replace(block_config, dropout=rate, drop_path=0.0)
                elif block_config.has_residual:
                    block_config = replace(block_config, dropout=0.0, drop_path=rate)
                else:
                    block_config = replace(block_config, dropout=0.0, drop_path=0.0)

                self.blocks.append(
                    stage.block_cls(
                        block_config,
                        enable_spectral_norm=style.enable_spectral_norm,
                        style=style.for_block(stage.block_cls),
                    )
                )

        # Head
        out_channels = self.stage_configs[-1].config.output_channels
        self.head_pre_norm = (
            nn.Identity()
            if self.head_config is None
            else _pre_norm(style.head.pre_normalization, out_channels)
        )
        self.head: nn.Module
        if self.head_config is not None:
            head_out = self.head_config.channels
            self.head = nn.Sequential(
                *_conv_norm_activation(
                    style.head.conv, out_channels, head_out, 1, activation=True
                ),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(self.head_config.dropout),
                nn.Linear(head_out, num_classes),
            )
        else:
            self.head = nn.Identity()

        # wrap stem and head layers with spectral norm in place
        if style.enable_spectral_norm:
            for module in (*self.stem.modules(), *self.head.modules()):
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    set_spectral_norm(module)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize module parameters with standard heuristics.

        Spectrally normalized layers get an orthogonal weight from
        `set_init_parameters` and a zero bias. Depthwise spectrally normalized
        layers get the Kaiming weight of the other convolutions in their
        unnormalized weight (plan 2026.010, decision 12).
        """
        # SE projections keep the fan-out scaling of the 1x1 convolutions they replaced
        se_linears = {
            layer
            for block in self.modules()
            if isinstance(block, SqueezeExcitation1DLinear)
            for layer in block.modules()
            if isinstance(layer, nn.Linear)
        }
        for m in self.modules():
            if get_spectral_norm(m) is not None:
                set_init_parameters(m)
                if isinstance(m.bias, torch.Tensor):
                    nn.init.zeros_(m.bias)
            elif get_depthwise_spectral_norm(m) is not None:
                # write into the unnormalized weight, since `weight` is recomputed
                weight_parametrizations = get_weight_parametrizations(m)
                assert weight_parametrizations is not None
                original = cast(torch.Tensor, weight_parametrizations.original)
                nn.init.kaiming_normal_(original, mode="fan_out", nonlinearity="relu")
                if isinstance(m.bias, torch.Tensor):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d) or (
                isinstance(m, nn.Linear) and m in se_linears
            ):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                assert m not in se_linears
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def resolve_input_shape(self) -> tuple[int | None, ...]:
        """Return the shape of one input sample, excluding the batch dimension.

        The channel entry is `input_channels` when the stem is enabled, and the
        (width-scaled) input channel count of the first block otherwise. The
        length entry is `None` when `input_length` disables the length check.

        Returns:
            The 2D shape `(channels, input_length)`.
        """
        if self.enable_stem:
            return (self.input_channels, self.input_length)
        return (self.stage_configs[0].config.input_channels, self.input_length)

    def resolve_output_shape(self) -> tuple[int | None, ...]:
        """Return the shape of one output sample, excluding the batch dimension.

        Without the head, `forward` returns the unpooled block output, which is
        a feature map and is reported as one. Its length is `None` because no
        code computes it without replicating the stem and stage strides.

        Returns:
            The 1D shape `(num_classes,)` when the head is enabled, or the 2D
            shape `(block output channels, None)` otherwise.
        """
        if self.enable_head:
            return (self.num_classes,)
        return (self.stage_configs[-1].config.output_channels, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of 1D time-series samples.

        Args:
            x: Input tensor with shape (batch_size, sequence_length) or
                (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Class logits with shape (batch_size, num_classes)
            when the head is enabled, or the raw block output with shape
            (batch_size, *resolve_output_shape()) otherwise, whose trailing
            entry is the sequence length this network does not compute.
        """
        # add a channel axis for univariate inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)
        else:
            assert x.dim() == 3, (
                "ScalableEfficientNet1D expected "
                "a 2D tensor (batch_size, sequence_length) or "
                "a 3D tensor (batch_size, channels, sequence_length), "
                f"got {x.dim()}"
            )

        # validate channel and sequence dimensions
        expected_input_channels = self.resolve_input_shape()[0]
        assert (
            x.shape[1] == expected_input_channels
        ), f"ScalableEfficientNet1D expected {expected_input_channels} input channels, got {x.shape[1]}"
        if self.input_length is not None:
            assert (
                x.shape[2] == self.input_length
            ), f"ScalableEfficientNet1D expected sequence length {self.input_length}, got {x.shape[2]}"

        # apply stem and stage blocks
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        # apply classification head
        x = self.head(self.head_pre_norm(x))
        return x


# --------------------------------------
# EfficientNet V1: B0-B8, L2
# --------------------------------------


def get_efficientnet_v1_b0_config() -> list[StageConfig]:
    """Return the canonical EfficientNetV1-B0 stage layout for 1D convolutions.

    Decoded from upstream ``v1_b0_block_str``. All stages use ``MBConv1D``.

    Returns:
        list[StageConfig]: Ordered stage configs (16 blocks total).
    """
    return [
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=32,
                output_channels=16,
                se_ratio=0.25,
            ),
            num_blocks=1,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=16,
                output_channels=24,
                se_ratio=0.25,
            ),
            num_blocks=2,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=5,
                stride=2,
                expand_ratio=6,
                input_channels=24,
                output_channels=40,
                se_ratio=0.25,
            ),
            num_blocks=2,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=40,
                output_channels=80,
                se_ratio=0.25,
            ),
            num_blocks=3,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=5,
                stride=1,
                expand_ratio=6,
                input_channels=80,
                output_channels=112,
                se_ratio=0.25,
            ),
            num_blocks=3,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=5,
                stride=2,
                expand_ratio=6,
                input_channels=112,
                output_channels=192,
                se_ratio=0.25,
            ),
            num_blocks=4,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=192,
                output_channels=320,
                se_ratio=0.25,
            ),
            num_blocks=1,
        ),
    ]


class EfficientNetV1BB0(ScalableEfficientNet1D):
    """Below the baseline 1D EfficientNetV1-B0 with under a ~100k-parameter budget."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.2),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1BB0.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=0.15,
            depth_coefficient=0.5,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B0(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B0 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.2),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B0.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B1(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B1 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.2),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B1.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=1.0,
            depth_coefficient=1.1,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B2(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B2 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.3),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B2.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=1.1,
            depth_coefficient=1.2,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B3(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B3 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.3),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B3.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=1.2,
            depth_coefficient=1.4,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B4(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B4 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.4),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B4.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=1.4,
            depth_coefficient=1.8,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B5(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B5 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.4),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B5.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=1.6,
            depth_coefficient=2.2,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B6(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B6 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.5),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B6.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=1.8,
            depth_coefficient=2.6,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B7(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B7 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.5),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B7.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=2.0,
            depth_coefficient=3.1,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1B8(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B8 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.5),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1B8.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=2.2,
            depth_coefficient=3.6,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV1L2(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-L2 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.5),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV1L2.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v1_b0_config(),
            stem=stem,
            head=head,
            width_coefficient=4.3,
            depth_coefficient=5.3,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


# --------------------------------------
# EfficientNet V2: B0-B3
# --------------------------------------


def get_efficientnet_v2_b_config() -> list[StageConfig]:
    """Return the canonical EfficientNetV2-B base stage layout for 1D convolutions.

    Decoded from upstream ``v2_base_block``. Stages 1-3 use ``FusedMBConv1D``
    without squeeze-and-excitation; stages 4-6 use ``MBConv1D`` with SE.
    Shared by V2-B0 through V2-B3 under compound scaling.

    Returns:
        list[StageConfig]: Ordered stage configs (21 blocks at 1.0 depth).
    """
    return [
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=32,
                output_channels=16,
                se_ratio=None,
            ),
            num_blocks=1,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=16,
                output_channels=32,
                se_ratio=None,
            ),
            num_blocks=2,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=32,
                output_channels=48,
                se_ratio=None,
            ),
            num_blocks=2,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=48,
                output_channels=96,
                se_ratio=0.25,
            ),
            num_blocks=3,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=96,
                output_channels=112,
                se_ratio=0.25,
            ),
            num_blocks=5,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=112,
                output_channels=192,
                se_ratio=0.25,
            ),
            num_blocks=8,
        ),
    ]


class EfficientNetV2BB0(ScalableEfficientNet1D):
    """Below the baseline 1D EfficientNetV2-B0 with under a ~150k-parameter budget."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.2),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2BB0.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_b_config(),
            stem=stem,
            head=head,
            width_coefficient=0.2,
            depth_coefficient=0.5,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV2B0(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-B0 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.2),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2B0.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_b_config(),
            stem=stem,
            head=head,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV2B1(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-B1 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.2),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2B1.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_b_config(),
            stem=stem,
            head=head,
            width_coefficient=1.0,
            depth_coefficient=1.1,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV2B2(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-B2 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.3),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2B2.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_b_config(),
            stem=stem,
            head=head,
            width_coefficient=1.1,
            depth_coefficient=1.2,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV2B3(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-B3 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.3),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2B3.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_b_config(),
            stem=stem,
            head=head,
            width_coefficient=1.2,
            depth_coefficient=1.4,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


# --------------------------------------
# EfficientNet V2: S, M, L, XL
# --------------------------------------


def get_efficientnet_v2_s_config() -> list[StageConfig]:
    """Return the canonical EfficientNetV2-S stage layout for 1D convolutions.

    Decoded from upstream ``v2_s_block``. Stages 1-3 use ``FusedMBConv1D``
    without squeeze-and-excitation; stages 4-6 use ``MBConv1D`` with SE.

    Returns:
        list[StageConfig]: Ordered stage configs (40 blocks total).
    """
    return [
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=24,
                output_channels=24,
                se_ratio=None,
            ),
            num_blocks=2,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=24,
                output_channels=48,
                se_ratio=None,
            ),
            num_blocks=4,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=48,
                output_channels=64,
                se_ratio=None,
            ),
            num_blocks=4,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=64,
                output_channels=128,
                se_ratio=0.25,
            ),
            num_blocks=6,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=128,
                output_channels=160,
                se_ratio=0.25,
            ),
            num_blocks=9,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=160,
                output_channels=256,
                se_ratio=0.25,
            ),
            num_blocks=15,
        ),
    ]


def get_efficientnet_v2_m_config() -> list[StageConfig]:
    """Return the canonical EfficientNetV2-M stage layout for 1D convolutions.

    Decoded from upstream ``v2_m_block``. Stages 1-3 use ``FusedMBConv1D``;
    stages 4-7 use ``MBConv1D`` with SE.

    Returns:
        list[StageConfig]: Ordered stage configs (57 blocks total).
    """
    return [
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=24,
                output_channels=24,
                se_ratio=None,
            ),
            num_blocks=3,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=24,
                output_channels=48,
                se_ratio=None,
            ),
            num_blocks=5,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=48,
                output_channels=80,
                se_ratio=None,
            ),
            num_blocks=5,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=80,
                output_channels=160,
                se_ratio=0.25,
            ),
            num_blocks=7,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=160,
                output_channels=176,
                se_ratio=0.25,
            ),
            num_blocks=14,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=176,
                output_channels=304,
                se_ratio=0.25,
            ),
            num_blocks=18,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=304,
                output_channels=512,
                se_ratio=0.25,
            ),
            num_blocks=5,
        ),
    ]


def get_efficientnet_v2_l_config() -> list[StageConfig]:
    """Return the canonical EfficientNetV2-L stage layout for 1D convolutions.

    Decoded from upstream ``v2_l_block``. Stages 1-3 use ``FusedMBConv1D``;
    stages 4-7 use ``MBConv1D`` with SE.

    Returns:
        list[StageConfig]: Ordered stage configs (79 blocks total).
    """
    return [
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=32,
                output_channels=32,
                se_ratio=None,
            ),
            num_blocks=4,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=32,
                output_channels=64,
                se_ratio=None,
            ),
            num_blocks=7,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=64,
                output_channels=96,
                se_ratio=None,
            ),
            num_blocks=7,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=96,
                output_channels=192,
                se_ratio=0.25,
            ),
            num_blocks=10,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=192,
                output_channels=224,
                se_ratio=0.25,
            ),
            num_blocks=19,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=224,
                output_channels=384,
                se_ratio=0.25,
            ),
            num_blocks=25,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=384,
                output_channels=640,
                se_ratio=0.25,
            ),
            num_blocks=7,
        ),
    ]


def get_efficientnet_v2_xl_config() -> list[StageConfig]:
    """Return the canonical EfficientNetV2-XL stage layout for 1D convolutions.

    Decoded from upstream ``v2_xl_block``. Stages 1-3 use ``FusedMBConv1D``;
    stages 4-7 use ``MBConv1D`` with SE.

    Returns:
        list[StageConfig]: Ordered stage configs (100 blocks total).
    """
    return [
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=32,
                output_channels=32,
                se_ratio=None,
            ),
            num_blocks=4,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=32,
                output_channels=64,
                se_ratio=None,
            ),
            num_blocks=8,
        ),
        StageConfig(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=64,
                output_channels=96,
                se_ratio=None,
            ),
            num_blocks=8,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=96,
                output_channels=192,
                se_ratio=0.25,
            ),
            num_blocks=16,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=192,
                output_channels=256,
                se_ratio=0.25,
            ),
            num_blocks=24,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=256,
                output_channels=512,
                se_ratio=0.25,
            ),
            num_blocks=32,
        ),
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=512,
                output_channels=640,
                se_ratio=0.25,
            ),
            num_blocks=8,
        ),
    ]


class EfficientNetV2S(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-S classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=24),
        head: HeadConfig | None = HeadConfig(dropout=0.2),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2S.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_s_config(),
            stem=stem,
            head=head,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV2M(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-M classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=24),
        head: HeadConfig | None = HeadConfig(dropout=0.3),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2M.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_m_config(),
            stem=stem,
            head=head,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV2L(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-L classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.4),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2L.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_l_config(),
            stem=stem,
            head=head,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )


class EfficientNetV2XL(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-XL classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        stem: StemConfig | None = StemConfig(channels=32),
        head: HeadConfig | None = HeadConfig(dropout=0.4),
        style: NetStyle = BASELINE,
    ) -> None:
        """Initialize EfficientNetV2XL.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Largest residual-branch dropout or drop path rate,
                ramped linearly over the blocks.
            stem: Stem config before width scaling, or `None` to disable the stem.
            head: Head config before width scaling, or `None` to disable the head.
            style: Network-wide design choices; see `NetStyle`.
        """
        super().__init__(
            stage_configs=get_efficientnet_v2_xl_config(),
            stem=stem,
            head=head,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            style=style,
        )
