from typing import cast

import pytest
import torch

from dlk.nets.efficientnet1d import (
    EfficientNet1D,
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
    SqueezeExcitation1D,
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
    """Pin ``SqueezeExcitation1D`` Conv1d widths to the explicit constructor args."""
    se = SqueezeExcitation1D(channels=24, squeeze_channels=6)

    reduce_conv = se.se[1]
    expand_conv = se.se[3]

    assert isinstance(reduce_conv, torch.nn.Conv1d)
    assert isinstance(expand_conv, torch.nn.Conv1d)
    assert reduce_conv.in_channels == 24
    assert reduce_conv.out_channels == 6
    assert expand_conv.in_channels == 6
    assert expand_conv.out_channels == 24


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
    reduce_conv = block.se.se[1]
    assert isinstance(reduce_conv, torch.nn.Conv1d)
    assert reduce_conv.out_channels == 4


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


def test_efficientnet1d_forward_output_shape_for_univariate_input() -> None:
    """Run ``EfficientNet1D`` on 2D input and validate logits shape."""
    net = EfficientNet1D(input_length=256, num_classes=3)
    x = torch.randn(4, 256)

    y = net(x)

    assert y.shape == (4, 3)


def test_efficientnet1d_forward_output_shape_for_multichannel_input() -> None:
    """Run ``EfficientNet1D`` on 3D input and validate logits shape."""
    net = EfficientNet1D(input_channels=2, input_length=256, num_classes=5)
    x = torch.randn(4, 2, 256)

    y = net(x)

    assert y.shape == (4, 5)


def test_efficientnet1d_raises_for_invalid_sequence_length() -> None:
    """Raise an assertion error when the input sequence length is invalid."""
    net = EfficientNet1D(input_length=256)
    x = torch.randn(4, 255)

    with pytest.raises(AssertionError, match="expected sequence length"):
        net(x)


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
