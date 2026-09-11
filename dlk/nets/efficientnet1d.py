"""EfficientNet-inspired 1D convolutional network for time-series classification."""

import math
from typing import NamedTuple

import torch
import torch.nn as nn

# --------------------------------------
# Config
# --------------------------------------


class MBConvConfig(NamedTuple):
    """Store one MBConv or Fused-MBConv stage configuration.

    Attributes:
        kernel_size: Convolution kernel size for the block's spatial filter.
        stride: Stride for the first block in the stage.
        expand_ratio: Channel expansion factor for the bottleneck.
        input_channels: Number of input channels for the stage.
        output_channels: Number of output channels for the stage.
        num_layers: Number of blocks in the stage.
        se_ratio: Squeeze-and-excitation channel reduction ratio, or `None`
            to disable squeeze-and-excitation (required for Fused-MBConv).
    """

    kernel_size: int
    stride: int
    expand_ratio: int
    input_channels: int
    output_channels: int
    num_layers: int
    se_ratio: float | None


class StageSpec(NamedTuple):
    """Pair a stage config with the block class that implements it.

    Attributes:
        block_cls: Block module class (`MBConv1D` or `FusedMBConv1D`).
        config: Stage layout and channel configuration.
    """

    block_cls: type[nn.Module]
    config: MBConvConfig


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


class SqueezeExcitation1D(nn.Module):
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


class MBConv1D(nn.Module):
    """A Mobile Inverted Bottleneck Convolution block for 1D signals."""

    def __init__(self, config: MBConvConfig, dropout: float = 0.0) -> None:
        """Initialize the MBConv block.

        Args:
            config: Layer and channel configuration for this block.
            dropout: Dropout probability applied before residual addition.
        """
        super().__init__()
        assert (
            0.0 <= dropout < 1.0
        ), f"dropout must be in the range [0, 1), got {dropout}"

        self.config = config
        self.has_se = config.se_ratio is not None and config.se_ratio > 0
        self.use_residual = (
            config.stride == 1 and config.input_channels == config.output_channels
        )

        # Expansion phase
        expanded_channels = config.input_channels * config.expand_ratio
        if config.expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv1d(config.input_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm1d(expanded_channels),
                nn.SiLU(),
            )
        else:
            self.expand_conv = nn.Identity()

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(
                expanded_channels,
                expanded_channels,
                config.kernel_size,
                stride=config.stride,
                padding=config.kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm1d(expanded_channels),
            nn.SiLU(),
        )

        # Squeeze-and-Excitation (bottleneck sized from pre-expansion channels)
        self.se: SqueezeExcitation1D | None = None
        if self.has_se:
            assert (
                config.se_ratio is not None
            ), "se_ratio must be set when has_se is True"
            squeeze_channels = max(1, int(config.input_channels * config.se_ratio))
            self.se = SqueezeExcitation1D(expanded_channels, squeeze_channels)

        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv1d(expanded_channels, config.output_channels, 1, bias=False),
            nn.BatchNorm1d(config.output_channels),
        )

        # Dropout for residual connection
        self.dropout: nn.Dropout | None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

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
        if self.use_residual:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity

        return x


class FusedMBConv1D(nn.Module):
    """A Fused Mobile Inverted Bottleneck Convolution block for 1D signals.

    Introduced in EfficientNetV2, this block replaces the expansion and
    depthwise steps of `MBConv1D` with a single dense convolution, trading
    higher FLOPs for higher arithmetic intensity on modern accelerators.
    Unlike `MBConv1D`, it has no squeeze-and-excitation step.
    """

    def __init__(self, config: MBConvConfig, dropout: float = 0.0) -> None:
        """Initialize the Fused-MBConv block.

        Args:
            config: Layer and channel configuration for this block.
                `config.se_ratio` must be `None`.
            dropout: Dropout probability applied before residual addition.
        """
        super().__init__()
        assert (
            0.0 <= dropout < 1.0
        ), f"dropout must be in the range [0, 1), got {dropout}"
        assert (
            config.se_ratio is None
        ), f"FusedMBConv1D does not support squeeze-and-excitation, got se_ratio={config.se_ratio}"

        self.config = config
        self.use_residual = (
            config.stride == 1 and config.input_channels == config.output_channels
        )

        # Fused expansion and spatial filter, and output projection
        expanded_channels = config.input_channels * config.expand_ratio
        self.project_conv: nn.Module
        if config.expand_ratio != 1:
            self.fused_conv = nn.Sequential(
                nn.Conv1d(
                    config.input_channels,
                    expanded_channels,
                    config.kernel_size,
                    stride=config.stride,
                    padding=config.kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(expanded_channels),
                nn.SiLU(),
            )
            self.project_conv = nn.Sequential(
                nn.Conv1d(expanded_channels, config.output_channels, 1, bias=False),
                nn.BatchNorm1d(config.output_channels),
            )
        else:
            # Expansion ratio 1 collapses the block to a single dense conv
            self.fused_conv = nn.Sequential(
                nn.Conv1d(
                    config.input_channels,
                    config.output_channels,
                    config.kernel_size,
                    stride=config.stride,
                    padding=config.kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(config.output_channels),
                nn.SiLU(),
            )
            self.project_conv = nn.Identity()

        # Dropout for residual connection
        self.dropout: nn.Dropout | None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

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

        # Fused expansion and spatial filter
        x = self.fused_conv(x)

        # Output projection
        x = self.project_conv(x)

        # Residual connection
        if self.use_residual:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + identity

        return x


# --------------------------------------
# Scalable Base Network
# --------------------------------------


class ScalableEfficientNet1D(nn.Module):
    """Compound-scalable EfficientNet classifier for 1D time-series inputs.

    Builds stem, stages, and head from a list of ``StageSpec`` values, applying
    EfficientNet width and depth compound scaling.
    """

    def __init__(
        self,
        stage_specs: list[StageSpec],
        stem_channels: int,
        head_channels: int = 1280,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        depth_divisor: int = 8,
        min_depth: int | None = None,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize ScalableEfficientNet1D.

        Args:
            stage_specs: Per-stage block class and base configuration.
            stem_channels: Base stem width before width scaling.
            head_channels: Base head feature width before width scaling.
            width_coefficient: Compound-scaling multiplier for channel counts.
            depth_coefficient: Compound-scaling multiplier for stage repeats.
            depth_divisor: Rounding unit for scaled channel counts.
            min_depth: Lower bound on rounded channel counts, or `None` to use
                `depth_divisor`.
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
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
        ), f"dropout_connect must be in [0, 1], got {dropout_connect}"
        assert (
            0.0 <= dropout_head < 1.0
        ), f"dropout_head must be in [0, 1], got {dropout_head}"
        assert stem_channels > 0, f"stem_channels must be positive, got {stem_channels}"
        assert head_channels > 0, f"head_channels must be positive, got {head_channels}"
        assert (
            len(stage_specs) > 0
        ), f"stage_specs must be non-empty, got {repr(stage_specs)}"

        self.input_channels = input_channels
        self.input_length = input_length

        # rescale stage channels and repeats under compound scaling
        scaled_specs: list[StageSpec] = []
        for spec in stage_specs:
            cfg = spec.config
            scaled_config = cfg._replace(
                input_channels=round_filters(
                    cfg.input_channels, width_coefficient, depth_divisor, min_depth
                ),
                output_channels=round_filters(
                    cfg.output_channels, width_coefficient, depth_divisor, min_depth
                ),
                num_layers=round_repeats(cfg.num_layers, depth_coefficient),
            )
            scaled_specs.append(StageSpec(spec.block_cls, scaled_config))
        self.stage_specs = scaled_specs
        total_blocks = sum(spec.config.num_layers for spec in self.stage_specs)

        # Stem
        stem_out = round_filters(
            stem_channels, width_coefficient, depth_divisor, min_depth
        )
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_out),
            nn.SiLU(),
        )

        # Stage blocks (MBConv or Fused-MBConv per StageSpec)
        self.blocks = nn.ModuleList()
        for stage_spec in self.stage_specs:
            stage_config = stage_spec.config
            for i in range(stage_config.num_layers):
                # Only first block in each stage uses the specified stride
                if i == 0:
                    block_config = stage_config
                else:
                    block_config = stage_config._replace(
                        input_channels=stage_config.output_channels, stride=1
                    )

                # Stochastic depth (drop connect)
                dropout_block = dropout_connect * len(self.blocks) / total_blocks

                self.blocks.append(stage_spec.block_cls(block_config, dropout_block))

        # Head
        out_channels = self.stage_specs[-1].config.output_channels
        head_out = round_filters(
            head_channels, width_coefficient, depth_divisor, min_depth
        )
        self.head = nn.Sequential(
            nn.Conv1d(out_channels, head_out, 1, bias=False),
            nn.BatchNorm1d(head_out),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_head),
            nn.Linear(head_out, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize module parameters with standard heuristics."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of 1D time-series samples.

        Args:
            x: Input tensor with shape (batch_size, sequence_length) or
                (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Class logits with shape (batch_size, num_classes).
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
        assert (
            x.shape[1] == self.input_channels
        ), f"ScalableEfficientNet1D expected {self.input_channels} input channels, got {x.shape[1]}"
        if self.input_length is not None:
            assert (
                x.shape[2] == self.input_length
            ), f"ScalableEfficientNet1D expected sequence length {self.input_length}, got {x.shape[2]}"

        # apply stem and stage blocks
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        # apply classification head
        x = self.head(x)
        return x


# --------------------------------------
# EfficientNet V1: B0-B8, L2
# --------------------------------------


# TODO: outdated
def get_efficientnet_b0_config() -> list[MBConvConfig]:
    """Return a compact EfficientNet-B0 stage layout adapted for 1D convolutions.

    Returns:
        list[MBConvConfig]: Ordered MBConv stage configuration.
    """
    return [
        MBConvConfig(
            kernel_size=3,
            stride=1,
            expand_ratio=1,
            input_channels=16,
            output_channels=8,
            num_layers=1,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=3,
            stride=2,
            expand_ratio=4,
            input_channels=8,
            output_channels=12,
            num_layers=1,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=5,
            stride=2,
            expand_ratio=4,
            input_channels=12,
            output_channels=20,
            num_layers=1,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=3,
            stride=2,
            expand_ratio=4,
            input_channels=20,
            output_channels=40,
            num_layers=2,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=5,
            stride=1,
            expand_ratio=4,
            input_channels=40,
            output_channels=56,
            num_layers=2,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=5,
            stride=2,
            expand_ratio=4,
            input_channels=56,
            output_channels=96,
            num_layers=2,
            se_ratio=0.25,
        ),
        MBConvConfig(
            kernel_size=3,
            stride=1,
            expand_ratio=4,
            input_channels=96,
            output_channels=160,
            num_layers=1,
            se_ratio=0.25,
        ),
    ]


# TODO: outdated
class EfficientNet1D(nn.Module):
    """EfficientNet classifier for 1D time-series inputs."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize EfficientNet1D.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__()
        assert input_channels > 0, "input_channels must be positive."
        if input_length is not None:
            assert input_length > 0, "input_length must be positive when provided."
        assert num_classes > 0, "num_classes must be positive."
        assert 0.0 <= dropout_connect <= 1.0, "dropout_connect must be in [0, 1]."
        assert 0.0 <= dropout_head <= 1.0, "dropout_head must be in [0, 1]."

        self.input_channels = input_channels
        self.input_length = input_length
        self.config = get_efficientnet_b0_config()
        total_blocks = sum(config.num_layers for config in self.config)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )

        # Mobile Inverted Bottleneck blocks
        self.blocks = nn.ModuleList()
        for stage_config in self.config:
            for i in range(stage_config.num_layers):
                # Only first block in each stage uses the specified stride
                if i == 0:
                    block_config = stage_config
                else:
                    block_config = stage_config._replace(
                        input_channels=stage_config.output_channels, stride=1
                    )

                # Stochastic depth (drop connect)
                dropout_block = dropout_connect * len(self.blocks) / total_blocks

                self.blocks.append(MBConv1D(block_config, dropout_block))

        # Head
        out_channels = self.config[-1].output_channels
        self.head = nn.Sequential(
            nn.Conv1d(out_channels, 640, 1, bias=False),
            nn.BatchNorm1d(640),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_head),
            nn.Linear(640, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize module parameters with standard heuristics."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of 1D time-series samples.

        Args:
            x: Input tensor with shape (batch_size, sequence_length) or
                (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Class logits with shape (batch_size, num_classes).
        """
        # add a channel axis for univariate inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)
        else:
            assert x.dim() == 3, (
                "EfficientNet1D expected a 2D tensor "
                "(batch_size, sequence_length) or a 3D tensor "
                "(batch_size, channels, sequence_length)."
            )

        # validate channel and sequence dimensions
        assert x.shape[1] == self.input_channels, (
            f"EfficientNet1D expected {self.input_channels} input channels, "
            f"but received {x.shape[1]}."
        )
        if self.input_length is not None:
            assert x.shape[2] == self.input_length, (
                f"EfficientNet1D expected sequence length {self.input_length}, "
                f"but received {x.shape[2]}."
            )

        # apply stem and MBConv blocks
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        # apply classification head
        x = self.head(x)
        return x


def get_efficientnet_v1_b0_config() -> list[StageSpec]:
    """Return the canonical EfficientNetV1-B0 stage layout for 1D convolutions.

    Decoded from upstream ``v1_b0_block_str``. All stages use ``MBConv1D``.

    Returns:
        list[StageSpec]: Ordered stage specifications (16 blocks total).
    """
    return [
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=32,
                output_channels=16,
                num_layers=1,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=16,
                output_channels=24,
                num_layers=2,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=5,
                stride=2,
                expand_ratio=6,
                input_channels=24,
                output_channels=40,
                num_layers=2,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=40,
                output_channels=80,
                num_layers=3,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=5,
                stride=1,
                expand_ratio=6,
                input_channels=80,
                output_channels=112,
                num_layers=3,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=5,
                stride=2,
                expand_ratio=6,
                input_channels=112,
                output_channels=192,
                num_layers=4,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=192,
                output_channels=320,
                num_layers=1,
                se_ratio=0.25,
            ),
        ),
    ]


class EfficientNetV1B0Minimal(ScalableEfficientNet1D):
    """Minimal 1D EfficientNetV1-B0 classifier under a ~100k-parameter budget."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize EfficientNetV1B0Minimal.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=0.15,
            depth_coefficient=0.5,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B0(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B0 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize EfficientNetV1B0.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B1(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B1 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize EfficientNetV1B1.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.0,
            depth_coefficient=1.1,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B2(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B2 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.3,
    ) -> None:
        """Initialize EfficientNetV1B2.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.1,
            depth_coefficient=1.2,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B3(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B3 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.3,
    ) -> None:
        """Initialize EfficientNetV1B3.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.2,
            depth_coefficient=1.4,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B4(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B4 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.4,
    ) -> None:
        """Initialize EfficientNetV1B4.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.4,
            depth_coefficient=1.8,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B5(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B5 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.4,
    ) -> None:
        """Initialize EfficientNetV1B5.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.6,
            depth_coefficient=2.2,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B6(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B6 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.5,
    ) -> None:
        """Initialize EfficientNetV1B6.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.8,
            depth_coefficient=2.6,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B7(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B7 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.5,
    ) -> None:
        """Initialize EfficientNetV1B7.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=2.0,
            depth_coefficient=3.1,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1B8(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-B8 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.5,
    ) -> None:
        """Initialize EfficientNetV1B8.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=2.2,
            depth_coefficient=3.6,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV1L2(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV1-L2 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.5,
    ) -> None:
        """Initialize EfficientNetV1L2.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in MBConv blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v1_b0_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=4.3,
            depth_coefficient=5.3,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


# --------------------------------------
# EfficientNet V2: B0-B3
# --------------------------------------


def get_efficientnet_v2_b_config() -> list[StageSpec]:
    """Return the canonical EfficientNetV2-B base stage layout for 1D convolutions.

    Decoded from upstream ``v2_base_block``. Stages 1-3 use ``FusedMBConv1D``
    without squeeze-and-excitation; stages 4-6 use ``MBConv1D`` with SE.
    Shared by V2-B0 through V2-B3 under compound scaling.

    Returns:
        list[StageSpec]: Ordered stage specifications (21 blocks at 1.0 depth).
    """
    return [
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=32,
                output_channels=16,
                num_layers=1,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=16,
                output_channels=32,
                num_layers=2,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=32,
                output_channels=48,
                num_layers=2,
                se_ratio=None,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=48,
                output_channels=96,
                num_layers=3,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=96,
                output_channels=112,
                num_layers=5,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=112,
                output_channels=192,
                num_layers=8,
                se_ratio=0.25,
            ),
        ),
    ]


class EfficientNetV2B0Minimal(ScalableEfficientNet1D):
    """Minimal 1D EfficientNetV2-B0 classifier under a ~100k-parameter budget."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize EfficientNetV2B0Minimal.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_b_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=0.15,
            depth_coefficient=0.5,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV2B0(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-B0 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize EfficientNetV2B0.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_b_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV2B1(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-B1 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize EfficientNetV2B1.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_b_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.0,
            depth_coefficient=1.1,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV2B2(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-B2 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.3,
    ) -> None:
        """Initialize EfficientNetV2B2.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_b_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.1,
            depth_coefficient=1.2,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV2B3(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-B3 classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.3,
    ) -> None:
        """Initialize EfficientNetV2B3.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_b_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.2,
            depth_coefficient=1.4,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


# --------------------------------------
# EfficientNet V2: S, M, L, XL
# --------------------------------------


def get_efficientnet_v2_s_config() -> list[StageSpec]:
    """Return the canonical EfficientNetV2-S stage layout for 1D convolutions.

    Decoded from upstream ``v2_s_block``. Stages 1-3 use ``FusedMBConv1D``
    without squeeze-and-excitation; stages 4-6 use ``MBConv1D`` with SE.

    Returns:
        list[StageSpec]: Ordered stage specifications (40 blocks total).
    """
    return [
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=24,
                output_channels=24,
                num_layers=2,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=24,
                output_channels=48,
                num_layers=4,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=48,
                output_channels=64,
                num_layers=4,
                se_ratio=None,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=64,
                output_channels=128,
                num_layers=6,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=128,
                output_channels=160,
                num_layers=9,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=160,
                output_channels=256,
                num_layers=15,
                se_ratio=0.25,
            ),
        ),
    ]


def get_efficientnet_v2_m_config() -> list[StageSpec]:
    """Return the canonical EfficientNetV2-M stage layout for 1D convolutions.

    Decoded from upstream ``v2_m_block``. Stages 1-3 use ``FusedMBConv1D``;
    stages 4-7 use ``MBConv1D`` with SE.

    Returns:
        list[StageSpec]: Ordered stage specifications (57 blocks total).
    """
    return [
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=24,
                output_channels=24,
                num_layers=3,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=24,
                output_channels=48,
                num_layers=5,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=48,
                output_channels=80,
                num_layers=5,
                se_ratio=None,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=80,
                output_channels=160,
                num_layers=7,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=160,
                output_channels=176,
                num_layers=14,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=176,
                output_channels=304,
                num_layers=18,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=304,
                output_channels=512,
                num_layers=5,
                se_ratio=0.25,
            ),
        ),
    ]


def get_efficientnet_v2_l_config() -> list[StageSpec]:
    """Return the canonical EfficientNetV2-L stage layout for 1D convolutions.

    Decoded from upstream ``v2_l_block``. Stages 1-3 use ``FusedMBConv1D``;
    stages 4-7 use ``MBConv1D`` with SE.

    Returns:
        list[StageSpec]: Ordered stage specifications (79 blocks total).
    """
    return [
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=32,
                output_channels=32,
                num_layers=4,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=32,
                output_channels=64,
                num_layers=7,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=64,
                output_channels=96,
                num_layers=7,
                se_ratio=None,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=96,
                output_channels=192,
                num_layers=10,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=192,
                output_channels=224,
                num_layers=19,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=224,
                output_channels=384,
                num_layers=25,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=384,
                output_channels=640,
                num_layers=7,
                se_ratio=0.25,
            ),
        ),
    ]


def get_efficientnet_v2_xl_config() -> list[StageSpec]:
    """Return the canonical EfficientNetV2-XL stage layout for 1D convolutions.

    Decoded from upstream ``v2_xl_block``. Stages 1-3 use ``FusedMBConv1D``;
    stages 4-7 use ``MBConv1D`` with SE.

    Returns:
        list[StageSpec]: Ordered stage specifications (100 blocks total).
    """
    return [
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=32,
                output_channels=32,
                num_layers=4,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=32,
                output_channels=64,
                num_layers=8,
                se_ratio=None,
            ),
        ),
        StageSpec(
            FusedMBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=64,
                output_channels=96,
                num_layers=8,
                se_ratio=None,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=4,
                input_channels=96,
                output_channels=192,
                num_layers=16,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=192,
                output_channels=256,
                num_layers=24,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=2,
                expand_ratio=6,
                input_channels=256,
                output_channels=512,
                num_layers=32,
                se_ratio=0.25,
            ),
        ),
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=6,
                input_channels=512,
                output_channels=640,
                num_layers=8,
                se_ratio=0.25,
            ),
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
        dropout_head: float = 0.2,
    ) -> None:
        """Initialize EfficientNetV2S.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_s_config(),
            stem_channels=24,
            head_channels=1280,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV2M(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-M classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.3,
    ) -> None:
        """Initialize EfficientNetV2M.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_m_config(),
            stem_channels=24,
            head_channels=1280,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV2L(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-L classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.4,
    ) -> None:
        """Initialize EfficientNetV2L.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_l_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


class EfficientNetV2XL(ScalableEfficientNet1D):
    """Faithful 1D EfficientNetV2-XL classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int | None = 1000,
        num_classes: int = 2,
        dropout_connect: float = 0.2,
        dropout_head: float = 0.4,
    ) -> None:
        """Initialize EfficientNetV2XL.

        Args:
            input_channels: Number of channels in each input sample.
            input_length: Expected sequence length, or `None` to disable checks.
            num_classes: Number of output classes.
            dropout_connect: Residual-branch dropout probability in blocks.
            dropout_head: Dropout probability before the final classifier.
        """
        super().__init__(
            stage_specs=get_efficientnet_v2_xl_config(),
            stem_channels=32,
            head_channels=1280,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            input_channels=input_channels,
            input_length=input_length,
            num_classes=num_classes,
            dropout_connect=dropout_connect,
            dropout_head=dropout_head,
        )


# --------------------------------------
# Inspect configs
# --------------------------------------


def _format_mbconv_config(config: MBConvConfig, block_name: str | None = None) -> str:
    """Format one stage config as a compact one-line summary.

    Args:
        config: Stage channel and layout fields.
        block_name: Optional block class name to prepend.

    Returns:
        str: Compact stage description.
    """
    if block_name is None:
        block_name = "---"
    se = f"se{config.se_ratio:g}" if config.se_ratio is not None else "se---"
    body = (
        f"k{config.kernel_size}  s{config.stride}  e{config.expand_ratio}  "
        f"Cin {config.input_channels:3}  Cout {config.output_channels:3}  "
        f"r{config.num_layers} {se}"
    )
    return f"{block_name:<14} {body}"


def _format_param_count(num_params: int) -> str:
    """Format a parameter count with a compact magnitude suffix.

    Args:
        num_params: Total number of trainable parameters.

    Returns:
        str: Human-readable parameter count.
    """
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:7.2f}M  ({num_params:>11,})"
    if num_params >= 1_000:
        return f"{num_params / 1_000:7.1f}K  ({num_params:>11,})"
    return f"{num_params:7d}   ({num_params:>11,})"


def _count_model_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: Instantiated module.

    Returns:
        int: Sum of ``numel()`` over parameters with ``requires_grad``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_efficientnet_configs() -> None:
    """Discover and print every ``get_efficientnet_*`` stage configuration.

    Finds zero-argument config builders in this module whose names start with
    ``get_efficientnet_``, calls each, and prints stages in compact form.
    """
    import inspect
    import sys

    module = sys.modules[__name__]
    builders = sorted(
        (name, obj)
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("get_efficientnet_") and obj.__module__ == module.__name__
    )

    print("Stage configurations")
    print("=" * 72)
    print()

    for name, builder in builders:
        stages = builder()
        if not stages:
            print(f"{name}()\n  (empty)\n")
            continue

        # total block count depends on return type
        if isinstance(stages[0], StageSpec):
            total_blocks = sum(spec.config.num_layers for spec in stages)
            kind = "StageSpec"
        else:
            total_blocks = sum(cfg.num_layers for cfg in stages)
            kind = "MBConvConfig"

        print(f"{name}()  [{kind}, {len(stages)} stages, {total_blocks} blocks]")
        print("-" * 72)
        for i, stage in enumerate(stages, start=1):
            if isinstance(stage, StageSpec):
                line = _format_mbconv_config(stage.config, stage.block_cls.__name__)
            else:
                line = _format_mbconv_config(stage)
            print(f"  {i:02d}  {line}")
        print()


def print_efficientnet_parameter_counts(
    input_channels: int = 1,
    num_classes: int = 2,
) -> None:
    """Discover concrete EfficientNet models and print trainable parameter counts.

    Instantiates every ``EfficientNet*`` class in this module except
    ``ScalableEfficientNet1D``, using ``input_length=None``.

    Args:
        input_channels: Channel count passed to each model constructor.
        num_classes: Class count passed to each model constructor.
    """
    import inspect
    import sys

    module = sys.modules[__name__]
    model_classes = sorted(
        (name, obj)
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
        and issubclass(obj, nn.Module)
        and name.startswith("EfficientNet")
        and name != "ScalableEfficientNet1D"
    )

    print("Model parameter counts")
    print("=" * 72)
    print(
        f"settings: input_channels={input_channels}, "
        f"num_classes={num_classes}, input_length=None"
    )
    print("-" * 72)
    for name, model_cls in model_classes:
        model = model_cls(
            input_channels=input_channels,
            input_length=None,
            num_classes=num_classes,
        )
        num_params = _count_model_parameters(model)
        blocks = getattr(model, "blocks", None)
        num_blocks = len(blocks) if isinstance(blocks, nn.ModuleList) else 0
        print(
            f"  {name:<20}  blocks={num_blocks:3d}  params={_format_param_count(num_params)}"
        )
    print()


if __name__ == "__main__":
    print_efficientnet_configs()
    print_efficientnet_parameter_counts()
