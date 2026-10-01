import math
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from dlk.nets.efficientnet1d import (
    EfficientNetV1B0,
    EfficientNetV1B1,
    EfficientNetV1B2,
    EfficientNetV1B3,
    EfficientNetV1B4,
    EfficientNetV1B5,
    EfficientNetV1B6,
    EfficientNetV1B7,
    EfficientNetV1B8,
    EfficientNetV1BB0,
    EfficientNetV1L2,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2BB0,
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
from dlk.nets.spectral_norm import get_spectral_norm


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


# Keys of a BatchNorm1d module's state_dict, in their order.
_BATCH_NORM_KEYS = (
    "weight",
    "bias",
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


def _summarize_state_dict(net: nn.Module) -> list[str]:
    """Summarize the ``state_dict`` keys and shapes of a network, one line per module.

    A line reads ``"<module>: <name>(<shape>) ..."``, in ``state_dict`` order. A
    module whose entries are exactly those of a ``BatchNorm1d`` with ``C``
    channels reads ``"<module>: batch_norm(C)"``.

    Args:
        net: Network to summarize.

    Returns:
        The summary lines.
    """
    modules: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    for key, value in net.state_dict().items():
        module, _, name = key.rpartition(".")
        modules.setdefault(module, []).append((name, tuple(value.shape)))
    lines = []
    for module, entries in modules.items():
        names = tuple(name for name, _ in entries)
        shapes = [shape for _, shape in entries]
        if names == _BATCH_NORM_KEYS and len(set(shapes[:4])) == 1 and not shapes[4]:
            lines.append(f"{module}: batch_norm({shapes[0][0]})")
        else:
            fields = " ".join(f"{n}({','.join(map(str, s))})" for n, s in entries)
            lines.append(f"{module}: {fields}")
    return lines


BB0Variant = type[EfficientNetV1BB0] | type[EfficientNetV2BB0]


def _build_bb0(net_cls: BB0Variant) -> ScalableEfficientNet1D:
    """Build a BB0 variant with fixed sizes for the ``state_dict`` constants."""
    return net_cls(input_channels=1, input_length=32, num_classes=3)


# generated by running this file as a script; see _print_state_dict_summaries
STATE_DICT_SUMMARIES: dict[str, list[str]] = {
    "EfficientNetV1BB0": [
        "stem.0: weight(8,1,3)",
        "stem.1: batch_norm(8)",
        "blocks.0.depthwise_conv.0: weight(8,1,3)",
        "blocks.0.depthwise_conv.1: batch_norm(8)",
        "blocks.0.se.reduce: weight(2,8) bias(2)",
        "blocks.0.se.expand: weight(8,2) bias(8)",
        "blocks.0.project_conv.0: weight(8,8,1)",
        "blocks.0.project_conv.1: batch_norm(8)",
        "blocks.1.expand_conv.0: weight(48,8,1)",
        "blocks.1.expand_conv.1: batch_norm(48)",
        "blocks.1.depthwise_conv.0: weight(48,1,3)",
        "blocks.1.depthwise_conv.1: batch_norm(48)",
        "blocks.1.se.reduce: weight(2,48) bias(2)",
        "blocks.1.se.expand: weight(48,2) bias(48)",
        "blocks.1.project_conv.0: weight(8,48,1)",
        "blocks.1.project_conv.1: batch_norm(8)",
        "blocks.2.expand_conv.0: weight(48,8,1)",
        "blocks.2.expand_conv.1: batch_norm(48)",
        "blocks.2.depthwise_conv.0: weight(48,1,5)",
        "blocks.2.depthwise_conv.1: batch_norm(48)",
        "blocks.2.se.reduce: weight(2,48) bias(2)",
        "blocks.2.se.expand: weight(48,2) bias(48)",
        "blocks.2.project_conv.0: weight(8,48,1)",
        "blocks.2.project_conv.1: batch_norm(8)",
        "blocks.3.expand_conv.0: weight(48,8,1)",
        "blocks.3.expand_conv.1: batch_norm(48)",
        "blocks.3.depthwise_conv.0: weight(48,1,3)",
        "blocks.3.depthwise_conv.1: batch_norm(48)",
        "blocks.3.se.reduce: weight(2,48) bias(2)",
        "blocks.3.se.expand: weight(48,2) bias(48)",
        "blocks.3.project_conv.0: weight(16,48,1)",
        "blocks.3.project_conv.1: batch_norm(16)",
        "blocks.4.expand_conv.0: weight(96,16,1)",
        "blocks.4.expand_conv.1: batch_norm(96)",
        "blocks.4.depthwise_conv.0: weight(96,1,3)",
        "blocks.4.depthwise_conv.1: batch_norm(96)",
        "blocks.4.se.reduce: weight(4,96) bias(4)",
        "blocks.4.se.expand: weight(96,4) bias(96)",
        "blocks.4.project_conv.0: weight(16,96,1)",
        "blocks.4.project_conv.1: batch_norm(16)",
        "blocks.5.expand_conv.0: weight(96,16,1)",
        "blocks.5.expand_conv.1: batch_norm(96)",
        "blocks.5.depthwise_conv.0: weight(96,1,5)",
        "blocks.5.depthwise_conv.1: batch_norm(96)",
        "blocks.5.se.reduce: weight(4,96) bias(4)",
        "blocks.5.se.expand: weight(96,4) bias(96)",
        "blocks.5.project_conv.0: weight(16,96,1)",
        "blocks.5.project_conv.1: batch_norm(16)",
        "blocks.6.expand_conv.0: weight(96,16,1)",
        "blocks.6.expand_conv.1: batch_norm(96)",
        "blocks.6.depthwise_conv.0: weight(96,1,5)",
        "blocks.6.depthwise_conv.1: batch_norm(96)",
        "blocks.6.se.reduce: weight(4,96) bias(4)",
        "blocks.6.se.expand: weight(96,4) bias(96)",
        "blocks.6.project_conv.0: weight(16,96,1)",
        "blocks.6.project_conv.1: batch_norm(16)",
        "blocks.7.expand_conv.0: weight(96,16,1)",
        "blocks.7.expand_conv.1: batch_norm(96)",
        "blocks.7.depthwise_conv.0: weight(96,1,5)",
        "blocks.7.depthwise_conv.1: batch_norm(96)",
        "blocks.7.se.reduce: weight(4,96) bias(4)",
        "blocks.7.se.expand: weight(96,4) bias(96)",
        "blocks.7.project_conv.0: weight(32,96,1)",
        "blocks.7.project_conv.1: batch_norm(32)",
        "blocks.8.expand_conv.0: weight(192,32,1)",
        "blocks.8.expand_conv.1: batch_norm(192)",
        "blocks.8.depthwise_conv.0: weight(192,1,5)",
        "blocks.8.depthwise_conv.1: batch_norm(192)",
        "blocks.8.se.reduce: weight(8,192) bias(8)",
        "blocks.8.se.expand: weight(192,8) bias(192)",
        "blocks.8.project_conv.0: weight(32,192,1)",
        "blocks.8.project_conv.1: batch_norm(32)",
        "blocks.9.expand_conv.0: weight(192,32,1)",
        "blocks.9.expand_conv.1: batch_norm(192)",
        "blocks.9.depthwise_conv.0: weight(192,1,3)",
        "blocks.9.depthwise_conv.1: batch_norm(192)",
        "blocks.9.se.reduce: weight(8,192) bias(8)",
        "blocks.9.se.expand: weight(192,8) bias(192)",
        "blocks.9.project_conv.0: weight(48,192,1)",
        "blocks.9.project_conv.1: batch_norm(48)",
        "head.0: weight(192,48,1)",
        "head.1: batch_norm(192)",
        "head.6: weight(3,192) bias(3)",
    ],
    "EfficientNetV2BB0": [
        "stem.0: weight(8,1,3)",
        "stem.1: batch_norm(8)",
        "blocks.0.fused_conv.0: weight(8,8,3)",
        "blocks.0.fused_conv.1: batch_norm(8)",
        "blocks.1.fused_conv.0: weight(32,8,3)",
        "blocks.1.fused_conv.1: batch_norm(32)",
        "blocks.1.project_conv.0: weight(8,32,1)",
        "blocks.1.project_conv.1: batch_norm(8)",
        "blocks.2.fused_conv.0: weight(32,8,3)",
        "blocks.2.fused_conv.1: batch_norm(32)",
        "blocks.2.project_conv.0: weight(8,32,1)",
        "blocks.2.project_conv.1: batch_norm(8)",
        "blocks.3.expand_conv.0: weight(32,8,1)",
        "blocks.3.expand_conv.1: batch_norm(32)",
        "blocks.3.depthwise_conv.0: weight(32,1,3)",
        "blocks.3.depthwise_conv.1: batch_norm(32)",
        "blocks.3.se.reduce: weight(2,32) bias(2)",
        "blocks.3.se.expand: weight(32,2) bias(32)",
        "blocks.3.project_conv.0: weight(16,32,1)",
        "blocks.3.project_conv.1: batch_norm(16)",
        "blocks.4.expand_conv.0: weight(64,16,1)",
        "blocks.4.expand_conv.1: batch_norm(64)",
        "blocks.4.depthwise_conv.0: weight(64,1,3)",
        "blocks.4.depthwise_conv.1: batch_norm(64)",
        "blocks.4.se.reduce: weight(4,64) bias(4)",
        "blocks.4.se.expand: weight(64,4) bias(64)",
        "blocks.4.project_conv.0: weight(16,64,1)",
        "blocks.4.project_conv.1: batch_norm(16)",
        "blocks.5.expand_conv.0: weight(96,16,1)",
        "blocks.5.expand_conv.1: batch_norm(96)",
        "blocks.5.depthwise_conv.0: weight(96,1,3)",
        "blocks.5.depthwise_conv.1: batch_norm(96)",
        "blocks.5.se.reduce: weight(4,96) bias(4)",
        "blocks.5.se.expand: weight(96,4) bias(96)",
        "blocks.5.project_conv.0: weight(24,96,1)",
        "blocks.5.project_conv.1: batch_norm(24)",
        "blocks.6.expand_conv.0: weight(144,24,1)",
        "blocks.6.expand_conv.1: batch_norm(144)",
        "blocks.6.depthwise_conv.0: weight(144,1,3)",
        "blocks.6.depthwise_conv.1: batch_norm(144)",
        "blocks.6.se.reduce: weight(6,144) bias(6)",
        "blocks.6.se.expand: weight(144,6) bias(144)",
        "blocks.6.project_conv.0: weight(24,144,1)",
        "blocks.6.project_conv.1: batch_norm(24)",
        "blocks.7.expand_conv.0: weight(144,24,1)",
        "blocks.7.expand_conv.1: batch_norm(144)",
        "blocks.7.depthwise_conv.0: weight(144,1,3)",
        "blocks.7.depthwise_conv.1: batch_norm(144)",
        "blocks.7.se.reduce: weight(6,144) bias(6)",
        "blocks.7.se.expand: weight(144,6) bias(144)",
        "blocks.7.project_conv.0: weight(24,144,1)",
        "blocks.7.project_conv.1: batch_norm(24)",
        "blocks.8.expand_conv.0: weight(144,24,1)",
        "blocks.8.expand_conv.1: batch_norm(144)",
        "blocks.8.depthwise_conv.0: weight(144,1,3)",
        "blocks.8.depthwise_conv.1: batch_norm(144)",
        "blocks.8.se.reduce: weight(6,144) bias(6)",
        "blocks.8.se.expand: weight(144,6) bias(144)",
        "blocks.8.project_conv.0: weight(40,144,1)",
        "blocks.8.project_conv.1: batch_norm(40)",
        "blocks.9.expand_conv.0: weight(240,40,1)",
        "blocks.9.expand_conv.1: batch_norm(240)",
        "blocks.9.depthwise_conv.0: weight(240,1,3)",
        "blocks.9.depthwise_conv.1: batch_norm(240)",
        "blocks.9.se.reduce: weight(10,240) bias(10)",
        "blocks.9.se.expand: weight(240,10) bias(240)",
        "blocks.9.project_conv.0: weight(40,240,1)",
        "blocks.9.project_conv.1: batch_norm(40)",
        "blocks.10.expand_conv.0: weight(240,40,1)",
        "blocks.10.expand_conv.1: batch_norm(240)",
        "blocks.10.depthwise_conv.0: weight(240,1,3)",
        "blocks.10.depthwise_conv.1: batch_norm(240)",
        "blocks.10.se.reduce: weight(10,240) bias(10)",
        "blocks.10.se.expand: weight(240,10) bias(240)",
        "blocks.10.project_conv.0: weight(40,240,1)",
        "blocks.10.project_conv.1: batch_norm(40)",
        "blocks.11.expand_conv.0: weight(240,40,1)",
        "blocks.11.expand_conv.1: batch_norm(240)",
        "blocks.11.depthwise_conv.0: weight(240,1,3)",
        "blocks.11.depthwise_conv.1: batch_norm(240)",
        "blocks.11.se.reduce: weight(10,240) bias(10)",
        "blocks.11.se.expand: weight(240,10) bias(240)",
        "blocks.11.project_conv.0: weight(40,240,1)",
        "blocks.11.project_conv.1: batch_norm(40)",
        "head.0: weight(256,40,1)",
        "head.1: batch_norm(256)",
        "head.6: weight(3,256) bias(3)",
    ],
}


@pytest.mark.parametrize("net_cls", [EfficientNetV1BB0, EfficientNetV2BB0])
def test_efficientnet_state_dict_keys_and_shapes_unchanged(net_cls: BB0Variant) -> None:
    """Freeze the ``state_dict`` keys and shapes, so checkpoints keep loading."""
    expected = STATE_DICT_SUMMARIES[net_cls.__name__]

    summary = _summarize_state_dict(_build_bb0(net_cls))

    assert summary == expected


@pytest.mark.parametrize(
    ("model_cls", "expected_blocks"),
    [
        (EfficientNetV1BB0, 10),
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
        (EfficientNetV2BB0, 12),
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
    enable_spectral_norm: bool = False,
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
        enable_spectral_norm: Whether to spectrally normalize every convolution
            and linear layer.

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
        enable_spectral_norm=enable_spectral_norm,
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


def _weighted_layers(module: nn.Module) -> list[nn.Module]:
    """Return every convolution and linear layer in `module`."""
    return [m for m in module.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]


def test_efficientnet_spectral_norm_wraps_every_conv_and_linear() -> None:
    """Wrap, orthogonally initialize, and normalize every conv and linear layer."""
    net = EfficientNetV2BB0(input_length=64, num_classes=3, enable_spectral_norm=True)
    plain_net = EfficientNetV2BB0(input_length=64, num_classes=3)

    layers = _weighted_layers(net)
    assert len(layers) == len(_weighted_layers(plain_net))
    assert all(parametrize.is_parametrized(m, "weight") for m in layers)

    # every wrapped weight starts with orthonormal rows or columns; biases start at zero
    for layer in layers:
        spectral = get_spectral_norm(layer)
        assert spectral is not None
        _, original = spectral
        w = original.detach().flatten(1)
        # compare the wide orientation, whose rows are orthonormal
        if w.shape[0] > w.shape[1]:
            w = w.T
        torch.testing.assert_close(w @ w.T, torch.eye(w.shape[0]), atol=1e-5, rtol=0)
        bias = cast(torch.Tensor | None, layer.bias)
        if bias is not None:
            assert not bias.detach().any()
        # scale up so that sigma is 1 only if the normalization acts
        with torch.no_grad():
            original.mul_(3.0)

    # five training forwards let the power iteration converge sigma to 1
    x = torch.randn(2, 64)
    net.train()
    for _ in range(5):
        net(x)
    net.eval()
    y = net(x)

    assert y.shape == (2, 3)
    for layer in layers:
        weight = cast(torch.Tensor, layer.weight).detach()
        sigma = torch.linalg.matrix_norm(weight.flatten(1), ord=2)
        torch.testing.assert_close(sigma, torch.tensor(1.0), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("enable_spectral_norm", [True, False])
@pytest.mark.parametrize(
    "block_cls, expand_ratio, se_ratio",
    [
        pytest.param(MBConv1D, 4, 0.25, id="mbconv-expand-se"),
        pytest.param(MBConv1D, 1, None, id="mbconv-no-expand-no-se"),
        pytest.param(FusedMBConv1D, 1, None, id="fused-expand-1"),
        pytest.param(FusedMBConv1D, 4, None, id="fused-expand-4"),
    ],
)
def test_blocks_spectral_norm_wraps_every_weighted_layer(
    block_cls: type[MBConv1D] | type[FusedMBConv1D],
    expand_ratio: int,
    se_ratio: float | None,
    enable_spectral_norm: bool,
) -> None:
    """Wrap every weighted layer of a block exactly when the flag is on."""
    config = MBConvConfig(
        kernel_size=3,
        stride=1,
        expand_ratio=expand_ratio,
        input_channels=16,
        output_channels=16,
        num_layers=1,
        se_ratio=se_ratio,
    )
    block = block_cls(config, enable_spectral_norm=enable_spectral_norm)

    layers = _weighted_layers(block)
    assert layers
    wrapped = [parametrize.is_parametrized(m, "weight") for m in layers]
    assert all(w == enable_spectral_norm for w in wrapped)


@pytest.mark.parametrize("enable_stem, enable_head", [(False, True), (True, False)])
def test_efficientnet_spectral_norm_without_stem_or_head(
    enable_stem: bool, enable_head: bool
) -> None:
    """Wrap the remaining layers when the stem or the head is disabled."""
    net = _small_scalable_net(
        enable_stem=enable_stem, enable_head=enable_head, enable_spectral_norm=True
    )
    input_channels, input_length = net.resolve_input_shape()
    assert input_channels is not None and input_length is not None
    x = torch.randn(2, input_channels, input_length)

    y = net(x)

    layers = _weighted_layers(net)
    assert layers
    assert all(parametrize.is_parametrized(m, "weight") for m in layers)
    assert y.shape[:2] == (2, net.resolve_output_shape()[0])


def _print_state_dict_summaries() -> None:
    """Regenerate :data:`STATE_DICT_SUMMARIES` and print it as a dict literal.

    Run ``uv run python tests/nets/test_efficientnet1d.py``.
    """
    print("STATE_DICT_SUMMARIES: dict[str, list[str]] = {")
    for net_cls in (EfficientNetV1BB0, EfficientNetV2BB0):
        print(f'    "{net_cls.__name__}": [')
        for line in _summarize_state_dict(_build_bb0(net_cls)):
            print(f'        "{line}",')
        print("    ],")
    print("}")


if __name__ == "__main__":
    _print_state_dict_summaries()
