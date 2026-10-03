import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from functools import partial
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from dlk.nets.efficientnet1d import (
    CONV_NORM_NONE,
    NET_BASELINE,
    NET_EXACT_SN_FLOORED_PRE_GN,
    NET_EXACT_SN_PRE_GN,
    NET_SN_FLOORED_PRE_GN,
    NET_SN_PRE_GN,
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
    FusedMBConvStyle,
    HeadConfig,
    HeadStyle,
    MBConv1D,
    MBConvConfig,
    MBConvStyle,
    NetStyle,
    ScalableEfficientNet1D,
    SqueezeExcitation1DLinear,
    StageConfig,
    StemConfig,
    StemStyle,
    round_channels,
    round_repeats,
)
from dlk.nets.spectral_norm import (
    get_depthwise_spectral_norm,
    get_spectral_norm,
    get_weight_parametrizations,
)

# Style of the 2026.009 network, with every convolution and linear layer spectrally normalized.
SPECTRAL_NORM = replace(NET_BASELINE, enable_spectral_norm=True)


def test_round_channels_rescales_channels_at_a_nontrivial_coefficient() -> None:
    """Round channel counts with EfficientNet-B2's width coefficient."""
    assert round_channels(40, 1.1) == 48
    assert round_channels(32, 1.1) == 32


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
        se_ratio=0.25,
        dropout=0.1,
    )
    block = MBConv1D(config)
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
        se_ratio=None,
        dropout=0.1,
    )
    block = FusedMBConv1D(config)
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
        se_ratio=0.25,
    )

    with pytest.raises(ValueError, match="squeeze-and-excitation"):
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
    stem: StemConfig | None = StemConfig(channels=8),
    head: HeadConfig | None = HeadConfig(channels=64),
    style: NetStyle = NET_BASELINE,
) -> ScalableEfficientNet1D:
    """Build a two-stage network whose channel counts survive width scaling.

    Every channel count is a multiple of the depth divisor at the default width
    coefficient, so the stage configs report the values written here.

    Args:
        input_channels: Number of channels in each input sample.
        input_length: Expected sequence length, or `None` to disable checks.
        num_classes: Number of output classes.
        stem: Stem config, or `None` to disable the stem.
        head: Head config, or `None` to disable the head.
        style: Network-wide design choices.

    Returns:
        The configured network.
    """
    stage_configs = [
        StageConfig(
            MBConv1D,
            MBConvConfig(
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                input_channels=8,
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
                expand_ratio=4,
                input_channels=16,
                output_channels=32,
                se_ratio=0.25,
            ),
            num_blocks=1,
        ),
    ]
    return ScalableEfficientNet1D(
        stage_configs=stage_configs,
        stem=stem,
        head=head,
        input_channels=input_channels,
        input_length=input_length,
        num_classes=num_classes,
        style=style,
    )


def test_headless_efficientnet_reports_block_channels_and_returns_a_feature_map() -> (
    None
):
    """Report the last stage's channels and return an unpooled feature map."""
    net = _small_scalable_net(head=None)
    last_stage_channels = net.stage_configs[-1].config.output_channels
    x = torch.randn(2, 3, 64)

    y = net(x)

    assert net.resolve_output_shape() == (last_stage_channels, None)
    # the length entry is None because no code computes it
    assert y.ndim == 3
    assert y.shape[:2] == (2, last_stage_channels)


def test_stemless_efficientnet_expects_first_block_channels() -> None:
    """Expect the first block's channel count when the stem is disabled."""
    net = _small_scalable_net(stem=None)
    first_block_channels = net.stage_configs[0].config.input_channels
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
    net = EfficientNetV2BB0(input_length=64, num_classes=3, style=SPECTRAL_NORM)
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


BlockClass = type[MBConv1D] | type[FusedMBConv1D]

# Blocks with residual: stride 1 and 16 channels in and out.
BLOCK_CASES = [
    pytest.param(MBConv1D, 4, 0.25, id="mbconv-expand-se"),
    pytest.param(MBConv1D, 1, None, id="mbconv-no-expand-no-se"),
    pytest.param(FusedMBConv1D, 1, None, id="fused-expand-1"),
    pytest.param(FusedMBConv1D, 4, None, id="fused-expand-4"),
]

# Style fields that hold a ConvNorm, per block class.
CONV_NORM_FIELDS: dict[BlockClass, tuple[str, ...]] = {
    MBConv1D: ("expand_conv_norm", "depthwise_conv_norm", "project_conv_norm"),
    FusedMBConv1D: ("fused_conv_norm", "project_conv_norm"),
}


def _block_config(
    expand_ratio: int,
    se_ratio: float | None,
    dropout: float = 0.0,
    drop_path: float = 0.0,
) -> MBConvConfig:
    """Return the config of a residual block with 16 channels."""
    return MBConvConfig(
        kernel_size=3,
        stride=1,
        expand_ratio=expand_ratio,
        input_channels=16,
        output_channels=16,
        se_ratio=se_ratio,
        dropout=dropout,
        drop_path=drop_path,
    )


def _build_block(
    block_cls: BlockClass, config: MBConvConfig, **style_kwargs: object
) -> MBConv1D | FusedMBConv1D:
    """Build `block_cls` with the default style of its class changed by `style_kwargs`."""
    # `replace` validates the changed style like the constructor does
    if block_cls is MBConv1D:
        return MBConv1D(config, style=replace(MBConvStyle(), **style_kwargs))
    return FusedMBConv1D(config, style=replace(FusedMBConvStyle(), **style_kwargs))


@pytest.mark.parametrize("enable_spectral_norm", [True, False])
@pytest.mark.parametrize("block_cls, expand_ratio, se_ratio", BLOCK_CASES)
def test_blocks_spectral_norm_wraps_every_weighted_layer(
    block_cls: BlockClass,
    expand_ratio: int,
    se_ratio: float | None,
    enable_spectral_norm: bool,
) -> None:
    """Wrap every weighted layer of a block exactly when the flag is on."""
    config = _block_config(expand_ratio, se_ratio)
    block = block_cls(config, enable_spectral_norm=enable_spectral_norm)

    layers = _weighted_layers(block)
    assert layers
    wrapped = [parametrize.is_parametrized(m, "weight") for m in layers]
    assert all(w == enable_spectral_norm for w in wrapped)


@pytest.mark.parametrize(
    "stem, head",
    [(None, HeadConfig(channels=64)), (StemConfig(channels=8), None)],
    ids=["stemless", "headless"],
)
def test_efficientnet_spectral_norm_without_stem_or_head(
    stem: StemConfig | None, head: HeadConfig | None
) -> None:
    """Wrap the remaining layers when the stem or the head is disabled."""
    net = _small_scalable_net(stem=stem, head=head, style=SPECTRAL_NORM)
    input_channels, input_length = net.resolve_input_shape()
    assert input_channels is not None and input_length is not None
    x = torch.randn(2, input_channels, input_length)

    y = net(x)

    layers = _weighted_layers(net)
    assert layers
    assert all(parametrize.is_parametrized(m, "weight") for m in layers)
    assert y.shape[:2] == (2, net.resolve_output_shape()[0])


@pytest.mark.parametrize(
    "width_coefficient, depth_coefficient, depth_divisor, min_depth",
    [(1.1, 1.2, 8, None), (0.15, 0.5, 8, None), (0.5, 3.1, 4, 16)],
)
def test_scaled_configs_match_round_channels_and_round_repeats(
    width_coefficient: float,
    depth_coefficient: float,
    depth_divisor: int,
    min_depth: int | None,
) -> None:
    """Scale every config as `round_channels` and `round_repeats` do."""
    config = MBConvConfig(
        kernel_size=5,
        stride=2,
        expand_ratio=6,
        input_channels=40,
        output_channels=80,
        se_ratio=0.25,
        dropout=0.1,
    )
    stage = StageConfig(MBConv1D, config, num_blocks=3)

    def filters(channels: int) -> int:
        return round_channels(channels, width_coefficient, depth_divisor, min_depth)

    scaled = stage.scaled(
        width_coefficient, depth_coefficient, depth_divisor, min_depth
    )

    assert scaled == StageConfig(
        MBConv1D,
        replace(config, input_channels=filters(40), output_channels=filters(80)),
        num_blocks=round_repeats(3, depth_coefficient),
    )
    stem_args = (width_coefficient, depth_divisor, min_depth)
    assert StemConfig(32).scaled(*stem_args) == StemConfig(filters(32))
    assert HeadConfig(1280, 0.3).scaled(*stem_args) == HeadConfig(filters(1280), 0.3)


def test_stage_config_block_configs_follow_the_stage_layout() -> None:
    """Give the first block the stage config and the later ones stride 1 and a residual."""
    config = MBConvConfig(
        kernel_size=3,
        stride=2,
        expand_ratio=4,
        input_channels=16,
        output_channels=24,
        se_ratio=None,
    )

    block_configs = StageConfig(FusedMBConv1D, config, num_blocks=3).block_configs()

    assert len(block_configs) == 3
    assert block_configs[0] is config
    # a residual needs both stride 1 and unchanged channels
    assert not config.has_residual
    assert not replace(config, stride=1).has_residual
    assert not replace(config, output_channels=16).has_residual
    for block_config in block_configs[1:]:
        assert block_config == replace(config, stride=1, input_channels=24)
        assert block_config.has_residual


@pytest.mark.parametrize(
    "value, field",
    [(_block_config(1, None), "stride"), (NET_BASELINE, "enable_spectral_norm")],
    ids=["config", "style"],
)
def test_configs_and_styles_are_frozen(value: object, field: str) -> None:
    """Raise on assignment, so shared configs and styles cannot change."""
    with pytest.raises(FrozenInstanceError):
        setattr(value, field, 2)


@pytest.mark.parametrize(
    "build",
    [
        partial(StemConfig, 0),
        partial(StageConfig, MBConv1D, _block_config(1, None), 0),
        partial(HeadConfig, channels=0),
        partial(HeadConfig, dropout=-0.1),
        partial(HeadConfig, dropout=1.0),
        partial(MBConvStyle, skip_scale=0.0),
        partial(FusedMBConvStyle, skip_scale=-1.0),
        partial(_block_config, 1, None, dropout=1.0),
        partial(_block_config, 1, None, drop_path=-0.1),
        partial(_block_config, 1, None, dropout=0.1, drop_path=0.1),
        partial(MBConvConfig, 3, 2, 1, 16, 16, None, drop_path=0.1),
    ],
    ids=[
        "stem-channels",
        "stage-num-blocks",
        "head-channels",
        "head-dropout-negative",
        "head-dropout-one",
        "mbconv-skip-scale-zero",
        "fused-skip-scale-negative",
        "dropout-one",
        "drop-path-negative",
        "dropout-and-drop-path",
        "drop-path-without-residual",
    ],
)
def test_configs_and_styles_reject_invalid_values(build: Callable[[], object]) -> None:
    """Reject non-positive sizes and scales and out-of-range rates."""
    with pytest.raises(ValueError):
        build()


def test_net_style_for_block_picks_the_style_by_block_class() -> None:
    """Return `mbconv` for MBConv blocks and subclasses, `fused` for fused blocks."""

    class CustomMBConv1D(MBConv1D):
        pass

    style = NetStyle(mbconv=MBConvStyle(skip_scale=0.5))

    assert style.for_block(MBConv1D) is style.mbconv
    assert style.for_block(CustomMBConv1D) is style.mbconv
    assert style.for_block(FusedMBConv1D) is style.fused
    with pytest.raises(TypeError):
        style.for_block(nn.Conv1d)


@pytest.mark.parametrize("block_cls, expand_ratio, se_ratio", BLOCK_CASES)
def test_block_without_normalization_has_conv_biases(
    block_cls: BlockClass, expand_ratio: int, se_ratio: float | None
) -> None:
    """Leave out every normalization and give every convolution a bias with `CONV_NORM_NONE`."""
    fields = dict.fromkeys(CONV_NORM_FIELDS[block_cls], CONV_NORM_NONE)
    block = _build_block(block_cls, _block_config(expand_ratio, se_ratio), **fields)

    convs = [m for m in block.modules() if isinstance(m, nn.Conv1d)]
    assert convs and all(conv.bias is not None for conv in convs)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in block.modules())
    # SiLU follows the convolution directly
    for sequential in (m for m in block.modules() if isinstance(m, nn.Sequential)):
        assert len(sequential) == 1 or isinstance(sequential[1], nn.SiLU)


@pytest.mark.parametrize("block_cls, expand_ratio, se_ratio", BLOCK_CASES)
def test_block_pre_normalization_normalizes_the_branch_input_only(
    block_cls: BlockClass, expand_ratio: int, se_ratio: float | None
) -> None:
    """Normalize the branch input with the pre-norm and keep the identity path."""
    torch.manual_seed(0)
    config = _block_config(expand_ratio, se_ratio)

    default_block = block_cls(config)
    block = _build_block(block_cls, config, pre_normalization=partial(nn.GroupNorm, 1))

    assert isinstance(default_block.pre_norm, nn.Identity)
    assert isinstance(block.pre_norm, nn.GroupNorm)
    assert block.pre_norm.num_channels == 16
    # the scale-invariant pre-norm makes the branch f(N(x)) blind to the input scale
    block.eval()
    x = torch.randn(2, 16, 32)
    torch.testing.assert_close(block(3 * x) - 3 * x, block(x) - x, atol=1e-4, rtol=0)


@pytest.mark.parametrize("block_cls, expand_ratio, se_ratio", BLOCK_CASES)
def test_block_skip_scale_scales_the_residual_sum(
    block_cls: BlockClass, expand_ratio: int, se_ratio: float | None
) -> None:
    """Return `skip_scale` times the output of the same block without the scale."""
    torch.manual_seed(0)
    config = _block_config(expand_ratio, se_ratio)
    block = _build_block(block_cls, config, skip_scale=0.5)
    reference = _build_block(block_cls, config, skip_scale=1.0)
    reference.load_state_dict(block.state_dict())
    block.eval()
    reference.eval()
    x = torch.randn(2, 16, 32)

    y = reference(x)

    # the branch must not vanish, or the scale would act on x alone
    assert not torch.allclose(y, x)
    torch.testing.assert_close(block(x), 0.5 * y)


@pytest.mark.parametrize(
    "block_cls, style_kwargs",
    [
        (MBConv1D, {"expand_conv_norm": CONV_NORM_NONE}),
        (FusedMBConv1D, {"project_conv_norm": CONV_NORM_NONE}),
    ],
    ids=["mbconv-expand", "fused-project"],
)
def test_block_ignores_missing_convolution_for_expand_ratio_one(
    block_cls: BlockClass, style_kwargs: dict[str, object]
) -> None:
    """Ignore the style of a convolution that `expand_ratio == 1` leaves out."""
    config = _block_config(1, None)

    default_keys = list(block_cls(config).state_dict())
    keys = list(_build_block(block_cls, config, **style_kwargs).state_dict())

    assert keys == default_keys


def test_stem_and_head_styles_build_their_layers() -> None:
    """Build the stem and head from their styles and apply the head pre-norm first."""
    style = replace(
        NET_BASELINE,
        stem=StemStyle(conv_norm=CONV_NORM_NONE),
        head=HeadStyle(
            pre_normalization=partial(nn.GroupNorm, 1), conv_norm=CONV_NORM_NONE
        ),
    )
    net = _small_scalable_net(style=style)
    stem = cast(nn.Sequential, net.stem)
    head = cast(nn.Sequential, net.head)

    for sequential in (stem, head):
        assert cast(nn.Conv1d, sequential[0]).bias is not None
        assert isinstance(sequential[1], nn.SiLU)
    assert isinstance(net.head_pre_norm, nn.GroupNorm)
    assert net.head_pre_norm.num_channels == 32
    # a headless network ignores the head style
    headless = _small_scalable_net(head=None, style=style)
    assert isinstance(headless.head_pre_norm, nn.Identity)

    # compose stem, blocks, head pre-norm, and head in this order
    net.eval()
    x = torch.randn(2, 3, 64)
    features = stem(x)
    for block in net.blocks:
        features = block(features)
    torch.testing.assert_close(net(x), head(net.head_pre_norm(features)))


@pytest.mark.parametrize("block_cls, expand_ratio, se_ratio", BLOCK_CASES)
def test_block_drop_path_drops_whole_branches_in_training_only(
    block_cls: BlockClass, expand_ratio: int, se_ratio: float | None
) -> None:
    """Drop or rescale each sample's branch in training, and keep it in eval mode."""
    torch.manual_seed(0)
    config = _block_config(expand_ratio, se_ratio)
    block = _build_block(block_cls, replace(config, drop_path=0.5), skip_scale=0.5)
    reference = _build_block(block_cls, config, skip_scale=0.5)
    reference.load_state_dict(block.state_dict())
    x = torch.randn(16, 16, 32)

    # recover the branch output f(x) from the reference, skip_scale * (f(x) + x)
    branch = reference(x) / 0.5 - x
    y = block(x)

    dropped = torch.isclose(y, 0.5 * x, atol=1e-5).flatten(1).all(dim=1)
    kept = torch.isclose(y, 0.5 * (2 * branch + x), atol=1e-5).flatten(1).all(dim=1)
    assert (dropped | kept).all()
    assert dropped.any() and kept.any()
    block.eval()
    reference.eval()
    torch.testing.assert_close(block(x), reference(x))


@pytest.mark.parametrize("dropout_mode", ["dropout", "drop_path"])
def test_block_dropout_ramps_over_the_blocks(dropout_mode: str) -> None:
    """Ramp `block_dropout` into dropout, or into drop path of residual blocks."""
    style = replace(NET_BASELINE, dropout_mode=dropout_mode)
    net = EfficientNetV2BB0(input_length=None, block_dropout=0.2, style=style)
    configs = [cast(MBConvConfig, block.config) for block in net.blocks]
    assert any(c.has_residual for c in configs)
    assert not all(c.has_residual for c in configs)

    for i, config in enumerate(configs):
        rate = 0.2 * i / len(configs)
        if dropout_mode == "dropout":
            assert (config.dropout, config.drop_path) == (rate, 0.0)
        else:
            expected = rate if config.has_residual else 0.0
            assert (config.dropout, config.drop_path) == (0.0, expected)


def test_exact_depthwise_spectral_norm_wraps_and_initializes_depthwise_convs() -> None:
    """Wrap only MBConv depthwise convolutions with C2, at Kaiming scale."""
    exact = MBConvStyle(depthwise_spectral_norm="exact")
    net = EfficientNetV2BB0(input_length=64, style=replace(SPECTRAL_NORM, mbconv=exact))
    unnormalized = EfficientNetV2BB0(
        input_length=64, style=replace(NET_BASELINE, mbconv=exact)
    )

    depthwise = {
        cast(nn.Sequential, block.depthwise_conv)[0]
        for block in net.blocks
        if isinstance(block, MBConv1D)
    }
    assert depthwise
    for layer in _weighted_layers(net):
        is_exact = get_depthwise_spectral_norm(layer) is not None
        assert is_exact == (layer in depthwise)
        assert (get_spectral_norm(layer) is None) == is_exact
    assert not any(
        parametrize.is_parametrized(m, "weight") for m in _weighted_layers(unnormalized)
    )

    # pool the normalized originals, whose fan-out Kaiming std is sqrt(2 / (C k))
    normalized = []
    for layer in depthwise:
        weight_parametrizations = get_weight_parametrizations(layer)
        assert weight_parametrizations is not None
        original = cast(torch.Tensor, weight_parametrizations.original).detach()
        normalized.append(original.flatten() / math.sqrt(2.0 / original[:, 0].numel()))
    assert torch.cat(normalized).std().item() == pytest.approx(1.0, rel=0.1)
    assert net(torch.randn(2, 64)).shape == (2, 2)


PRESETS = {
    "NET_BASELINE": NET_BASELINE,
    "NET_SN_PRE_GN": NET_SN_PRE_GN,
    "NET_SN_FLOORED_PRE_GN": NET_SN_FLOORED_PRE_GN,
    "NET_EXACT_SN_PRE_GN": NET_EXACT_SN_PRE_GN,
    "NET_EXACT_SN_FLOORED_PRE_GN": NET_EXACT_SN_FLOORED_PRE_GN,
}
BATCH_INDEPENDENT_PRESETS = [name for name in PRESETS if name != "NET_BASELINE"]
# SN presets with their depthwise spectral norm and GroupNorm eps.
SN_PRESETS = [
    ("NET_SN_PRE_GN", False, 1e-5),
    ("NET_SN_FLOORED_PRE_GN", False, 1e-4),
    ("NET_EXACT_SN_PRE_GN", True, 1e-5),
    ("NET_EXACT_SN_FLOORED_PRE_GN", True, 1e-4),
]
BB0_VARIANTS = pytest.mark.parametrize(
    "net_cls", [EfficientNetV1BB0, EfficientNetV2BB0]
)


def _build_preset_net(net_cls: BB0Variant, preset: str) -> ScalableEfficientNet1D:
    """Build a deterministic BB0 variant with a named preset, as a critic would."""
    torch.manual_seed(0)
    return net_cls(
        input_length=32,
        num_classes=3,
        block_dropout=0.0,
        head=HeadConfig(dropout=0.0),
        style=PRESETS[preset],
    )


@BB0_VARIANTS
@pytest.mark.parametrize("preset", PRESETS)
def test_preset_forward_shape_and_batch_norm(net_cls: BB0Variant, preset: str) -> None:
    """Build every preset with the right output shape and BatchNorm only in NET_BASELINE."""
    net = _build_preset_net(net_cls, preset)

    assert net(torch.randn(2, 32)).shape == (2, 3)
    has_batch_norm = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
    assert has_batch_norm == (preset == "NET_BASELINE")


@BB0_VARIANTS
@pytest.mark.parametrize("preset", BATCH_INDEPENDENT_PRESETS)
def test_preset_is_batch_independent_in_train_mode(
    net_cls: BB0Variant, preset: str
) -> None:
    """Keep a sample's train-mode output when the other samples change."""
    net = _build_preset_net(net_cls, preset).train()
    x = torch.randn(4, 32)
    x_other = torch.cat([x[:1], torch.randn(3, 32)])

    # cache the weights so that both forwards share one power iteration step
    with torch.no_grad(), parametrize.cached():
        y, y_other = net(x), net(x_other)
    torch.testing.assert_close(y[0], y_other[0])


@BB0_VARIANTS
@pytest.mark.parametrize(
    "preset, exact_depthwise, eps", SN_PRESETS, ids=[case[0] for case in SN_PRESETS]
)
def test_sn_preset_normalizes_every_layer_and_sets_eps(
    net_cls: BB0Variant, preset: str, exact_depthwise: bool, eps: float
) -> None:
    """Normalize every layer with C0, or the depthwise ones with C2, and floor GroupNorm."""
    net = _build_preset_net(net_cls, preset)

    depthwise = {
        cast(nn.Sequential, block.depthwise_conv)[0]
        for block in net.blocks
        if isinstance(block, MBConv1D)
    }
    for layer in _weighted_layers(net):
        is_exact = exact_depthwise and layer in depthwise
        assert (get_depthwise_spectral_norm(layer) is not None) == is_exact
        assert (get_spectral_norm(layer) is not None) != is_exact

    group_norms = [m for m in net.modules() if isinstance(m, nn.GroupNorm)]
    assert group_norms
    assert all(m.num_groups == 1 and m.eps == eps for m in group_norms)


@BB0_VARIANTS
@pytest.mark.parametrize("preset", BATCH_INDEPENDENT_PRESETS)
def test_preset_output_depends_on_input_amplitude(
    net_cls: BB0Variant, preset: str
) -> None:
    """Change the eval-mode output when the input doubles (explore doc, Section 1.3)."""
    net = _build_preset_net(net_cls, preset).eval()
    # stand in for a trained stem bias, since all biases start at zero
    stem_conv = cast(nn.Sequential, net.stem)[0]
    assert isinstance(stem_conv, nn.Conv1d) and stem_conv.bias is not None
    with torch.no_grad():
        nn.init.normal_(stem_conv.bias)
        x = torch.randn(4, 32)
        y, y_doubled = net(x), net(2 * x)

    # compare relatively, since the output scale differs across presets
    assert (y_doubled - y).abs().max() > 0.1 * y.abs().max()


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
