"""EfficientNet-inspired 1D convolutional network for time-series classification."""

from typing import NamedTuple

import torch
import torch.nn as nn

# --------------------------------------
# Config
# --------------------------------------


class MBConvConfig(NamedTuple):
    """Store one MBConv stage configuration.

    Attributes:
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride for the first block in the stage.
        expand_ratio: Channel expansion factor for the bottleneck.
        input_channels: Number of input channels for the stage.
        output_channels: Number of output channels for the stage.
        num_layers: Number of MBConv blocks in the stage.
        se_ratio: Squeeze-and-excitation channel reduction ratio.
    """

    kernel_size: int
    stride: int
    expand_ratio: int
    input_channels: int
    output_channels: int
    num_layers: int
    se_ratio: float | None


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


# --------------------------------------
# Components
# --------------------------------------


class Swish(nn.Module):
    """Apply the swish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply swish nonlinearity.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Activated tensor with the same shape as `x`.
        """
        return x * torch.sigmoid(x)


class SqueezeExcitation1D(nn.Module):
    """Apply squeeze-and-excitation reweighting for 1D feature maps."""

    def __init__(self, input_channels: int, se_ratio: float = 0.25) -> None:
        """Initialize the squeeze-and-excitation block.

        Args:
            input_channels: Number of channels in the input feature map.
            se_ratio: Reduction ratio used to compute squeeze channels.
        """
        super().__init__()
        assert input_channels > 0, "input_channels must be positive."
        assert se_ratio > 0.0, "se_ratio must be greater than 0."
        squeeze_channels = max(1, int(input_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(input_channels, squeeze_channels, 1),
            Swish(),
            nn.Conv1d(squeeze_channels, input_channels, 1),
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
        assert 0.0 <= dropout <= 1.0, "dropout must be in the range [0, 1]."
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
                Swish(),
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
            Swish(),
        )

        # Squeeze-and-Excitation
        self.se: SqueezeExcitation1D | None = None
        if self.has_se:
            assert (
                config.se_ratio is not None
            ), "se_ratio must be set when has_se is True."
            self.se = SqueezeExcitation1D(expanded_channels, config.se_ratio)

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
        assert x.dim() == 3, (
            "MBConv1D expected a 3D tensor with shape "
            "(batch_size, channels, sequence_length)."
        )
        assert x.shape[1] == self.config.input_channels, (
            f"MBConv1D expected {self.config.input_channels} input channels, "
            f"but received {x.shape[1]}."
        )
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


# --------------------------------------
# EfficientNet
# --------------------------------------


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
            Swish(),
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
            Swish(),
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
