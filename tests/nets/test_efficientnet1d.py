import math
from typing import cast

import pytest
import torch

from dlk.nets.efficientnet1d import (
    EfficientNetV1B0,
    EfficientNetV1B0Minimal,
    EfficientNetV1B1,
    EfficientNetV1B2,
    EfficientNetV1B3,
    EfficientNetV1B4,
    EfficientNetV1B5,
    EfficientNetV1B6,
    EfficientNetV1B7,
    EfficientNetV1B8,
    EfficientNetV1L2,
    EfficientNetV2B0,
    EfficientNetV2B0Minimal,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
    EfficientNetV2XL,
    FusedMBConv1D,
    MBConv1D,
    MBConvConfig,
    ScalableEfficientNet1D,
    SqueezeExcitation1DLinear,
    StageSpec,
    round_filters,
    round_repeats,
)


def test_round_filters_rescales_channels_at_a_nontrivial_coefficient() -> None:
    """Round channel counts with EfficientNet-B2's width coefficient."""
    assert round_filters(40, 1.1) == 48
    assert round_filters(32, 1.1) == 32


def test_round_repeats_rounds_up_fractional_depth() -> None:
    """Round stage repeats with EfficientNet-B2's depth coefficient."""
    assert round_repeats(2, 1.2) == 3
    assert round_repeats(1, 1.2) == 2


def test_squeeze_excitation1d_bottleneck_matches_explicit_channels() -> None:
    """Pin ``SqueezeExcitation1DLinear`` projection widths to the explicit constructor args."""
    se = SqueezeExcitation1DLinear(channels=24, squeeze_channels=6)

    assert isinstance(se.reduce, torch.nn.Linear)
    assert isinstance(se.expand, torch.nn.Linear)
    assert se.reduce.in_features == 24
    assert se.reduce.out_features == 6
    assert se.expand.in_features == 6
    assert se.expand.out_features == 24


def test_squeeze_excitation1d_gates_each_channel_uniformly_along_sequence() -> None:
    """Scale every position of a channel by one shared gate value."""
    torch.manual_seed(0)
    se = SqueezeExcitation1DLinear(channels=8, squeeze_channels=2)
    x = torch.randn(4, 8, 16).abs() + 0.5

    y = se(x)

    assert y.shape == x.shape
    gate = y / x
    assert torch.allclose(gate, gate[:, :, :1].expand_as(gate), atol=1e-5)


def test_squeeze_excitation1d_projections_use_fan_out_initialization() -> None:
    """Keep SE projections at 1x1-convolution fan-out scale, not the head's 0.01."""
    model = EfficientNetV1B0(input_channels=1, input_length=None, num_classes=2)

    se_blocks = [m for m in model.modules() if isinstance(m, SqueezeExcitation1DLinear)]
    assert se_blocks

    for se in se_blocks:
        for layer in (se.reduce, se.expand):
            # skip narrow layers, whose sample standard deviation is too noisy
            if layer.weight.numel() < 200:
                continue
            expected_std = math.sqrt(2.0 / layer.out_features)
            assert layer.weight.std().item() == pytest.approx(expected_std, rel=0.25)


def test_mbconv1d_se_bottleneck_uses_pre_expansion_channels() -> None:
    """Size the SE bottleneck from pre-expansion channels, not expanded channels."""
    config = MBConvConfig(
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        input_channels=16,
        output_channels=24,
        num_layers=1,
        se_ratio=0.25,
    )
    block = MBConv1D(config)

    assert block.se is not None
    assert isinstance(block.se.reduce, torch.nn.Linear)
    assert block.se.reduce.out_features == 4


def test_mbconv1d_forward_output_shape_with_expansion_and_stride() -> None:
    """Run ``MBConv1D`` with expansion and stride and validate output shape."""
    config = MBConvConfig(
        kernel_size=3,
        stride=2,
        expand_ratio=4,
        input_channels=8,
        output_channels=16,
        num_layers=1,
        se_ratio=0.25,
    )
    block = MBConv1D(config)
    x = torch.randn(2, 8, 64)

    y = block(x)

    assert y.shape == (2, 16, 32)


def test_mbconv1d_forward_output_shape_with_residual_connection() -> None:
    """Run ``MBConv1D`` with a residual connection and validate output shape."""
    config = MBConvConfig(
        kernel_size=3,
        stride=1,
        expand_ratio=1,
        input_channels=8,
        output_channels=8,
        num_layers=1,
        se_ratio=0.25,
    )
    block = MBConv1D(config, dropout=0.1)
    x = torch.randn(2, 8, 64)

    y = block(x)

    assert y.shape == (2, 8, 64)


def test_fusedmbconv1d_forward_output_shape_with_expansion_and_stride() -> None:
    """Run ``FusedMBConv1D`` with expansion and stride and validate output shape."""
    config = MBConvConfig(
        kernel_size=3,
        stride=2,
        expand_ratio=4,
        input_channels=8,
        output_channels=16,
        num_layers=1,
        se_ratio=None,
    )
    block = FusedMBConv1D(config)
    x = torch.randn(2, 8, 64)

    y = block(x)

    assert y.shape == (2, 16, 32)


def test_fusedmbconv1d_forward_output_shape_with_residual_connection() -> None:
    """Run ``FusedMBConv1D`` with a residual connection and validate output shape."""
    config = MBConvConfig(
        kernel_size=3,
        stride=1,
        expand_ratio=1,
        input_channels=8,
        output_channels=8,
        num_layers=1,
        se_ratio=None,
    )
    block = FusedMBConv1D(config, dropout=0.1)
    x = torch.randn(2, 8, 64)

    y = block(x)

    assert y.shape == (2, 8, 64)


def test_fusedmbconv1d_raises_for_se_ratio() -> None:
    """Raise an assertion error when the config sets a squeeze-and-excitation ratio."""
    config = MBConvConfig(
        kernel_size=3,
        stride=1,
        expand_ratio=4,
        input_channels=8,
        output_channels=8,
        num_layers=1,
        se_ratio=0.25,
    )

    with pytest.raises(AssertionError, match="squeeze-and-excitation"):
        FusedMBConv1D(config)


def test_efficientnet_v1_b0_forward_output_shape() -> None:
    """Run ``EfficientNetV1B0`` and validate logits shape."""
    net = EfficientNetV1B0(input_length=320, num_classes=3)
    x = torch.randn(2, 320)

    y = net(x)

    assert y.shape == (2, 3)


def test_efficientnet_v1_b0_has_expected_block_count() -> None:
    """Pin ``EfficientNetV1B0`` to the published 16-block layout."""
    net = EfficientNetV1B0(input_length=320)

    assert len(net.blocks) == 16


def test_efficientnet_v2_s_forward_output_shape() -> None:
    """Run ``EfficientNetV2S`` and validate logits shape."""
    net = EfficientNetV2S(input_length=320, num_classes=3)
    x = torch.randn(2, 320)

    y = net(x)

    assert y.shape == (2, 3)


def test_efficientnet_v2_s_has_expected_block_count() -> None:
    """Pin ``EfficientNetV2S`` to the published 40-block layout."""
    net = EfficientNetV2S(input_length=320)

    assert len(net.blocks) == 40


def test_efficientnet_v2_s_uses_fused_blocks_in_early_stages_and_mbconv_in_late_stages() -> (
    None
):
    """Check V2-S places fused blocks early and SE-MBConv blocks late."""
    net = EfficientNetV2S(input_length=320)

    first_block = net.blocks[0]
    last_fused_block = net.blocks[9]
    first_mbconv_block = net.blocks[10]

    assert isinstance(first_block, FusedMBConv1D)
    assert isinstance(last_fused_block, FusedMBConv1D)
    assert isinstance(first_mbconv_block, MBConv1D)
    assert first_mbconv_block.has_se is True
    assert first_block.config.se_ratio is None


@pytest.mark.parametrize(
    ("model_cls", "expected_blocks"),
    [
        (EfficientNetV1B0Minimal, 10),
        (EfficientNetV1B1, 23),
        (EfficientNetV1B2, 23),
        (EfficientNetV1B3, 26),
        (EfficientNetV1B4, 32),
        (EfficientNetV1B5, 39),
        (EfficientNetV1B6, 45),
        (EfficientNetV1B7, 55),
        (EfficientNetV1B8, 61),
        (EfficientNetV1L2, 88),
        (EfficientNetV2B0, 21),
        (EfficientNetV2B0Minimal, 12),
        (EfficientNetV2B1, 27),
        (EfficientNetV2B2, 28),
        (EfficientNetV2B3, 32),
        (EfficientNetV2M, 57),
        (EfficientNetV2L, 79),
        (EfficientNetV2XL, 100),
    ],
)
def test_efficientnet_variant_forward_shape_and_block_count(
    model_cls: type, expected_blocks: int
) -> None:
    """Check logits shape and published block count for each remaining variant."""
    net = cast(ScalableEfficientNet1D, model_cls(input_length=320, num_classes=3))
    x = torch.randn(2, 320)

    y = net(x)

    assert y.shape == (2, 3)
    assert len(net.blocks) == expected_blocks


def _small_scalable_net(
    input_channels: int = 3,
    input_length: int | None = 64,
    num_classes: int = 5,
    enable_stem: bool = True,
    enable_head: bool = True,
) -> ScalableEfficientNet1D:
    """Build a two-stage network whose channel counts survive width scaling.

    Every channel count is a multiple of the depth divisor at the default width
    coefficient, so the stage specs report the values written here.

    Args:
        input_channels: Number of channels in each input sample.
        input_length: Expected sequence length, or `None` to disable checks.
        num_classes: Number of output classes.
        enable_stem: Whether to build the stem.
        enable_head: Whether to build the classification head.

    Returns:
        The configured network.
    """
    stage_specs = [
        StageSpec(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=8,
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
                expand_ratio=4,
                input_channels=16,
                output_channels=32,
                num_layers=1,
                se_ratio=0.25,
            ),
        ),
    ]
    return ScalableEfficientNet1D(
        stage_specs=stage_specs,
        stem_channels=8,
        head_channels=64,
        input_channels=input_channels,
        input_length=input_length,
        num_classes=num_classes,
        enable_stem=enable_stem,
        enable_head=enable_head,
    )


def test_headless_efficientnet_reports_block_channels_and_returns_a_feature_map() -> (
    None
):
    """Report the last stage's channels and return an unpooled feature map."""
    net = _small_scalable_net(enable_head=False)
    last_stage_channels = net.stage_specs[-1].config.output_channels
    x = torch.randn(2, 3, 64)

    y = net(x)

    assert net.resolve_output_shape() == (last_stage_channels, None)
    # the length entry is None because no code computes it
    assert y.ndim == 3
    assert y.shape[:2] == (2, last_stage_channels)


def test_stemless_efficientnet_expects_first_block_channels() -> None:
    """Expect the first block's channel count when the stem is disabled."""
    net = _small_scalable_net(enable_stem=False)
    first_block_channels = net.stage_specs[0].config.input_channels
    assert first_block_channels != net.input_channels

    y = net(torch.randn(2, first_block_channels, 64))

    assert net.resolve_input_shape() == (first_block_channels, 64)
    assert y.shape == (2, 5)


def test_resolve_input_shape_reports_the_configured_length() -> None:
    """Report the configured sequence length, and `None` when it is unconstrained."""
    constrained = _small_scalable_net(input_length=64)
    unconstrained = _small_scalable_net(input_length=None)

    assert constrained.resolve_input_shape() == (3, 64)
    assert unconstrained.resolve_input_shape() == (3, None)
    # a None length entry means forward accepts any length
    assert unconstrained(torch.randn(2, 3, 48)).shape == (2, 5)
