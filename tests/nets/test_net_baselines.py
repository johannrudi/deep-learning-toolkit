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

The constants were generated with ``torch 2.13.0+cu130`` and are version and
machine specific; they were checked to be bit-identical under ``2.13.0+cpu``,
which is the build CI installs, train-mode cases included. Run this file as a
script to regenerate them:

    uv run python tests/nets/test_net_baselines.py
"""

from collections.abc import Callable
from functools import partial
from typing import cast

import pytest
import torch
import torch.nn as nn

from dlk.nets.conv1d import ConvNet, ConvNeXtBlock, ConvResNet
from dlk.nets.efficientnet1d import EfficientNetV1BB0, EfficientNetV2BB0
from dlk.nets.mlp import MLPNet, MLPResNet
from dlk.nets.transformer1d import ChannelWiseTransformerNet, TransformerNet
from dlk.nets.utils import get_gain, set_init_parameters

# Tolerances for the comparison against the stored constants. Both are
# relative, and they differ in what they are relative to: ELEMENT_RTOL is taken
# per entry, and SCALE_RTOL is taken against the largest magnitude of the case,
# which is what covers entries near zero. Nothing is absolute, because an
# untrained EfficientNet in eval mode decays its activations stage by stage and
# returns values near 1e-10, against which a fixed absolute tolerance would
# accept anything.
ELEMENT_RTOL = 1e-5
SCALE_RTOL = 1e-6

# one case: a built network and the positional inputs of one forward pass
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
        dropout_connect=0,
        dropout_head=0,
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
        [0.12845762073993683, 1.4875056743621826, 0.40791213512420654],
        [0.5338109135627747, -1.279374122619629, -1.3488813638687134],
    ],
    "convresnet": [
        [-0.7722904086112976, -0.05231968313455582, 0.4531893730163574],
        [0.10470473766326904, 0.17464129626750946, -0.24429607391357422],
    ],
    "efficientnet_v1_bb0": [
        [-3.638240264614012e-10, -1.6396646540517423e-10, -2.7084653964060124e-10],
        [-2.2792809306615425e-11, -1.0313081638679833e-10, -7.65879859532248e-11],
        [-1.6143925085643218e-10, -1.5931383989808978e-10, -3.989962527040092e-10],
        [1.4552042904014684e-10, 2.4023122380256723e-10, 1.8342670438098452e-10],
    ],
    "efficientnet_v1_bb0[train]": [
        [0.02806950733065605, 0.0811753123998642, -0.10261622071266174],
        [-0.03332018479704857, 0.04700588434934616, 0.21423229575157166],
        [0.01966693624854088, -0.05921027436852455, -0.0874396413564682],
        [-0.014621413312852383, 0.03641124442219734, -0.03254002332687378],
    ],
    "efficientnet_v2_bb0": [
        [-1.0591045196406412e-07, -1.0125177141162567e-08, -1.0978830999874845e-07],
        [-1.6397908098042535e-07, 1.9807063722510065e-07, 1.740789912219043e-08],
        [-9.879227746978358e-08, 1.5908146622223285e-07, -9.351737872975718e-08],
        [3.2292075502482476e-07, -3.246358915021119e-07, 4.0907502807385754e-07],
    ],
    "efficientnet_v2_bb0[train]": [
        [-0.04301675036549568, -0.0033205971121788025, 0.011387664824724197],
        [0.01424730196595192, -0.03820154070854187, -0.13943955302238464],
        [-0.03531705588102341, 0.09347786754369736, 0.025604240596294403],
        [-0.16220541298389435, 0.04529890418052673, 0.010898835957050323],
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
        [0.4538363218307495, -0.943134069442749, -0.13515830039978027],
        [0.3019818961620331, -0.9044573307037354, -0.25542715191841125],
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
    net, inputs = BUILDERS[name]()
    net.train(train)
    with torch.no_grad():
        return net(*inputs)


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
