"""Freeze the forward output of every composable network in ``dlk/nets``.

The networks here are the ones that gain shape accessors when composition lands
(see ``docs/features/2026.006__compose_nets__1-plan.md``). The accessors are pure
readers of values the constructors already receive, and these baselines are the
evidence that adding them moves no forward output. The file also freezes blocks
that inherit new constructor flags, such as ``enable_spectral_norm``, to pin that
the flag off leaves their forward output unchanged.

Each case builds its network under ``torch.manual_seed(0)``, keeps it small and
on the CPU, and evaluates it in ``eval()`` mode so dropout and batch-norm
statistics play no part. The cases in ``TRAIN_MODE_CASES`` also run in
``train()`` mode, under the key ``"<name>[train]"``: their batch norm uses the
batch statistics, and their builders turn dropout off. The inputs come from
their own generator, so the case order does not affect any result.

The constants were generated with ``torch 2.13.0+cu130`` on one thread and are
version and machine specific. Every forward pass runs on one thread, because the
reduction order of batch norm in train mode, and with it the last digits of the
output, varies with the thread count and the CPU. Run this file as a script to
regenerate the constants:

    uv run python tests/nets/test_net_baselines.py
"""

from collections.abc import Callable
from functools import partial
from typing import cast

import pytest
import torch
import torch.nn as nn

from dlk.nets.conv1d import ConvNet, ConvNeXtBlock, ConvResNet
from dlk.nets.efficientnet1d import EfficientNetV1BB0, EfficientNetV2BB0, HeadConfig
from dlk.nets.mlp import MLPNet, MLPResNet
from dlk.nets.transformer1d import ChannelWiseTransformerNet, TransformerNet
from dlk.nets.utils import get_gain, set_init_parameters

# Tolerances for the comparison against the stored constants:
# - ELEMENT_RTOL is the relative tolerance per entry
# - SCALE_RTOL is the relative tolerance for the largest magnitude
ELEMENT_RTOL = 1e-4
SCALE_RTOL = 1e-6

# Type for one case: tuple with a built network and the positional inputs of one forward pass
Case = tuple[nn.Module, tuple[torch.Tensor, ...]]


def _input(*shape: int, seed: int = 1) -> torch.Tensor:
    """Draw a deterministic input tensor from a dedicated generator.

    Args:
        *shape: Shape of the tensor to draw.
        seed: Seed of the generator drawing the values.

    Returns:
        A normally distributed tensor, independent of the global RNG state.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def _build_mlpnet() -> Case:
    """Build a small ``MLPNet`` and its input."""
    torch.manual_seed(0)
    net = MLPNet(4, 3, hidden_layers_sizes=(8, 8))
    return net, (_input(2, 4),)


def _build_mlpresnet() -> Case:
    """Build a small ``MLPResNet`` and its input."""
    torch.manual_seed(0)
    net = MLPResNet(
        6,
        3,
        residual_blocks_sizes=((8, 8, 16, 8), (8, 8, 16, 8)),
    )
    return net, (_input(2, 6),)


def _build_convnet() -> Case:
    """Build a small ``ConvNet`` with dense and output layers, and its input."""
    torch.manual_seed(0)
    # Two kernel-3 convolutions shorten length 16 to 12, at 4 channels.
    net = ConvNet(
        input_channels=1,
        hidden_conv_layers_channels_mult=(2, 4),
        hidden_conv_layers_kernels=(3, 3),
        hidden_dense_input_size=4 * 12,
        hidden_dense_layers_sizes=(8,),
        output_size=3,
    )
    return net, (_input(2, 1, 16),)


def _build_convresnet() -> Case:
    """Build a small ``ConvResNet`` with an ``MLPResNet`` head, and its input."""
    torch.manual_seed(0)
    # Two residual levels downsample length 16 to 4, at 4 channels.
    net = ConvResNet(
        input_channels=1,
        conv_resnet_params={
            "channels_mult": [2, 4],
            "kernels": [3, 3],
            "conv_kwargs": {"activation": nn.ReLU()},
        },
        mlp_resnet_params={
            "input_size": 4 * 4,
            "output_size": 3,
            "residual_blocks_sizes": [(8, 8, 16, 8)],
        },
    )
    return net, (_input(2, 1, 16),)


def _build_convnext_block() -> Case:
    """Build a small ``ConvNeXtBlock`` behind a linear head, and its input.

    The block's last convolution starts at zero, so a frozen forward would
    otherwise pin only the identity; re-initialize it under the fixed seed to
    freeze more than that. The head reduces the block's (2, 4, 16) output to
    (2, 3) like the other cases, and every output entry still depends on every
    entry of the block's output.
    """
    torch.manual_seed(0)
    block = ConvNeXtBlock(input_channels=4, kernel_size=3)
    conv_2 = cast(nn.Conv1d, block.block.conv_2)
    set_init_parameters(conv_2, get_gain(None, default="conv1d"))
    net = nn.Sequential(block, nn.Flatten(), nn.Linear(4 * 16, 3))
    return net, (_input(2, 4, 16),)


def _build_transformer_net() -> Case:
    """Build a small ``TransformerNet`` and its input."""
    torch.manual_seed(0)
    net = TransformerNet(
        input_seq_size=16,
        output_size=3,
        patch_size=4,
        embedding_size=8,
        attn_n_heads=(2, 2),
    )
    return net, (_input(2, 16),)


def _build_channel_wise_transformer_net() -> Case:
    """Build a small ``ChannelWiseTransformerNet`` and its input."""
    torch.manual_seed(0)
    net = ChannelWiseTransformerNet(
        input_channels=2,
        input_seq_size=16,
        output_size=3,
        patch_size=4,
        embedding_size=8,
        attn_n_heads=(2, 2),
    )
    return net, (_input(2, 2, 16),)


def _build_efficientnet(
    net_cls: type[EfficientNetV1BB0] | type[EfficientNetV2BB0],
) -> Case:
    """Build a stem-to-head EfficientNet without dropout, and its input.

    Dropout and the batch of four serve the train-mode case; see
    ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, Section 6a.

    Args:
        net_cls: EfficientNet variant to build.

    Returns:
        The network and its input.
    """
    torch.manual_seed(0)
    net = net_cls(
        input_channels=1,
        input_length=32,
        num_classes=3,
        block_dropout=0,
        head=HeadConfig(dropout=0),
    )
    return net, (_input(4, 1, 32),)


BUILDERS: dict[str, Callable[[], Case]] = {
    "channel_wise_transformer_net": _build_channel_wise_transformer_net,
    "convnet": _build_convnet,
    "convnext_block": _build_convnext_block,
    "convresnet": _build_convresnet,
    "efficientnet_v1_bb0": partial(_build_efficientnet, EfficientNetV1BB0),
    "efficientnet_v2_bb0": partial(_build_efficientnet, EfficientNetV2BB0),
    "mlpnet": _build_mlpnet,
    "mlpresnet": _build_mlpresnet,
    "transformer_net": _build_transformer_net,
}

# Cases that also run in train mode, where batch norm uses batch statistics.
TRAIN_MODE_CASES = ("efficientnet_v1_bb0", "efficientnet_v2_bb0")

# Every (name, train) pair the baselines cover.
MODES: list[tuple[str, bool]] = [(name, False) for name in sorted(BUILDERS)] + [
    (name, True) for name in TRAIN_MODE_CASES
]


def _key(name: str, train: bool) -> str:
    """Return the constants key: ``name``, or ``"<name>[train]"`` in train mode."""
    return f"{name}[train]" if train else name


# generated by running this file as a script; see the module docstring
BASELINE_OUTPUTS: dict[str, list[list[float]]] = {
    "channel_wise_transformer_net": [
        [-0.5168095827102661, 0.8953530788421631, -0.16747784614562988],
        [-0.5395035743713379, 0.9731947183609009, -0.1984843909740448],
    ],
    "convnet": [
        [3.0222561359405518, -0.6783685684204102, 1.6486051082611084],
        [0.4073147475719452, -1.6424623727798462, 1.8300310373306274],
    ],
    "convnext_block": [
        [0.12845756113529205, 1.4875056743621826, 0.4079121947288513],
        [0.5338109135627747, -1.279374122619629, -1.3488813638687134],
    ],
    "convresnet": [
        [-0.7722904682159424, -0.05231970548629761, 0.453189492225647],
        [0.10470473766326904, 0.17464125156402588, -0.24429607391357422],
    ],
    "efficientnet_v1_bb0": [
        [-3.638241652392793e-10, -1.6396653479411327e-10, -2.7084659515175247e-10],
        [-2.2792805837168473e-11, -1.0313079557011662e-10, -7.658809697552726e-11],
        [-1.6143933412315903e-10, -1.5931400643154348e-10, -3.9899633597073603e-10],
        [1.4552048455129807e-10, 2.4023122380256723e-10, 1.834270374478919e-10],
    ],
    "efficientnet_v1_bb0[train]": [
        [0.0280679389834404, 0.08117476105690002, -0.10261688381433487],
        [-0.03332015872001648, 0.04700576514005661, 0.21423295140266418],
        [0.019668180495500565, -0.05920999497175217, -0.08743955194950104],
        [-0.014621236361563206, 0.03641166910529137, -0.03254004567861557],
    ],
    "efficientnet_v2_bb0": [
        [-1.0591048038577355e-07, -1.0125273064431894e-08, -1.0978828868246637e-07],
        [-1.6397900992615178e-07, 1.9807069406851951e-07, 1.7407799646207422e-08],
        [-9.879225615350151e-08, 1.590814235896687e-07, -9.351736451890247e-08],
        [3.229208971333719e-07, -3.246358915021119e-07, 4.0907505649556697e-07],
    ],
    "efficientnet_v2_bb0[train]": [
        [-0.043014124035835266, -0.0033219438046216965, 0.011386148631572723],
        [0.014246227219700813, -0.038201428949832916, -0.1394394338130951],
        [-0.03531438112258911, 0.09348168224096298, 0.02560703083872795],
        [-0.16220994293689728, 0.04529629647731781, 0.010897468775510788],
    ],
    "mlpnet": [
        [-0.2424575388431549, 0.503169059753418, -0.3081715703010559],
        [-1.0519530773162842, 0.15276935696601868, -1.0358805656433105],
    ],
    "mlpresnet": [
        [-0.14274761080741882, 0.3249003291130066, 0.8066041469573975],
        [-1.1489803791046143, -1.6987935304641724, -0.1310654878616333],
    ],
    "transformer_net": [
        [0.4538363218307495, -0.943134069442749, -0.1351582407951355],
        [0.301982045173645, -0.9044573307037354, -0.2554270625114441],
    ],
}


def _forward(name: str, train: bool) -> torch.Tensor:
    """Build one case and run its forward pass in the given mode.

    Args:
        name: Key of the case in :data:`BUILDERS`.
        train: Whether to run in train mode rather than eval mode.

    Returns:
        The output tensor of the case's forward pass.
    """
    # one thread fixes the reduction order, which varies with the thread count
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        net, inputs = BUILDERS[name]()
        net.train(train)
        with torch.no_grad():
            return net(*inputs)
    finally:
        torch.set_num_threads(num_threads)


@pytest.mark.parametrize("name, train", MODES)
def test_forward_output_matches_the_frozen_baseline(name: str, train: bool) -> None:
    """Compare a network's forward output against its stored baseline."""
    key = _key(name, train)
    assert key in BASELINE_OUTPUTS, f"missing baseline constants for {key=}"
    expected = torch.tensor(BASELINE_OUTPUTS[key])

    y = _forward(name, train)

    assert y.shape == expected.shape, f"{key}: {y.shape=} != {expected.shape=}"
    # convert the scale-relative tolerance into the absolute one allclose takes
    scale = max(expected.abs().max().item(), torch.finfo(torch.float32).tiny)
    assert torch.allclose(
        y, expected, atol=SCALE_RTOL * scale, rtol=ELEMENT_RTOL
    ), f"{key}: {y.tolist()} != {expected.tolist()}"


def test_every_builder_has_baseline_constants() -> None:
    """Pin that no case is left without stored constants."""
    assert {_key(name, train) for name, train in MODES} == set(BASELINE_OUTPUTS)


def _print_baselines() -> None:
    """Regenerate the baseline constants and print them as a dict literal."""
    print(f"# generated with torch {torch.__version__}")
    print("BASELINE_OUTPUTS: dict[str, list[list[float]]] = {")
    for name, train in sorted(MODES, key=lambda mode: _key(*mode)):
        rows = _forward(name, train).tolist()
        print(f'    "{_key(name, train)}": [')
        for row in rows:
            values = ", ".join(repr(value) for value in row)
            print(f"        [{values}],")
        print("    ],")
    print("}")


if __name__ == "__main__":
    _print_baselines()
