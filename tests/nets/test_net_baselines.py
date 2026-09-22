"""Freeze the forward output of every composable network in ``dlk/nets``.

The networks here are the ones that gain shape accessors when composition lands
(see ``docs/features/2026.05__compose_nets__1-plan.md``). The accessors are pure
readers of values the constructors already receive, and these baselines are the
evidence that adding them moves no forward output.

Each case builds its network under ``torch.manual_seed(0)``, keeps it small and
on the CPU, and evaluates it in ``eval()`` mode so dropout and batch-norm
statistics play no part. The inputs come from their own generator, so the case
order does not affect any result.

The constants were generated with ``torch 2.13.0+cu130`` and are version and
machine specific; they were checked to be bit-identical under ``2.13.0+cpu``,
which is the build CI installs. Run this file as a script to regenerate them:

    uv run python tests/nets/test_net_baselines.py
"""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from dlk.nets.conv1d import ConvNet, ConvResNet
from dlk.nets.efficientnet1d import EfficientNetV1B0Minimal
from dlk.nets.mlp import MLPNet, MLPResNet
from dlk.nets.transformer1d import ChannelWiseTransformerNet, TransformerNet

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
            "activation": nn.ReLU(),
        },
        mlp_resnet_params={
            "input_size": 4 * 4,
            "output_size": 3,
            "residual_blocks_sizes": [(8, 8, 16, 8)],
        },
    )
    return net, (_input(2, 1, 16),)


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


def _build_efficientnet() -> Case:
    """Build an ``EfficientNetV1B0Minimal`` stem-to-head network and its input."""
    torch.manual_seed(0)
    net = EfficientNetV1B0Minimal(input_channels=1, input_length=32, num_classes=3)
    return net, (_input(2, 1, 32),)


BUILDERS: dict[str, Callable[[], Case]] = {
    "channel_wise_transformer_net": _build_channel_wise_transformer_net,
    "convnet": _build_convnet,
    "convresnet": _build_convresnet,
    "efficientnet_v1_b0_minimal": _build_efficientnet,
    "mlpnet": _build_mlpnet,
    "mlpresnet": _build_mlpresnet,
    "transformer_net": _build_transformer_net,
}

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
    "convresnet": [
        [0.08671721816062927, 0.15811944007873535, -0.3143555819988251],
        [0.23487216234207153, -0.41824793815612793, -1.0563223361968994],
    ],
    "efficientnet_v1_b0_minimal": [
        [-3.638240264614012e-10, -1.6396646540517423e-10, -2.7084653964060124e-10],
        [-2.2792809306615425e-11, -1.0313081638679833e-10, -7.65879859532248e-11],
    ],
    "mlpnet": [
        [-0.2424575388431549, 0.503169059753418, -0.3081715703010559],
        [-1.0519530773162842, 0.15276935696601868, -1.0358805656433105],
    ],
    "mlpresnet": [
        [-1.0894097089767456, -0.6791186332702637, 0.9551457762718201],
        [0.3882388472557068, -1.8481355905532837, -1.507836103439331],
    ],
    "transformer_net": [
        [0.4538363218307495, -0.943134069442749, -0.13515830039978027],
        [0.3019818961620331, -0.9044573307037354, -0.25542715191841125],
    ],
}


def _forward(name: str) -> torch.Tensor:
    """Build one case and run its forward pass in evaluation mode.

    Args:
        name: Key of the case in :data:`BUILDERS`.

    Returns:
        The output tensor of the case's forward pass.
    """
    net, inputs = BUILDERS[name]()
    net.eval()
    with torch.no_grad():
        return net(*inputs)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_forward_output_matches_the_frozen_baseline(name: str) -> None:
    """Compare a network's forward output against its stored baseline."""
    assert name in BASELINE_OUTPUTS, f"missing baseline constants for {name=}"
    expected = torch.tensor(BASELINE_OUTPUTS[name])

    y = _forward(name)

    assert y.shape == expected.shape, f"{name}: {y.shape=} != {expected.shape=}"
    # convert the scale-relative tolerance into the absolute one allclose takes
    scale = max(expected.abs().max().item(), torch.finfo(torch.float32).tiny)
    assert torch.allclose(
        y, expected, atol=SCALE_RTOL * scale, rtol=ELEMENT_RTOL
    ), f"{name}: {y.tolist()} != {expected.tolist()}"


def test_every_builder_has_baseline_constants() -> None:
    """Pin that no case is left without stored constants."""
    assert set(BUILDERS) == set(BASELINE_OUTPUTS)


def _print_baselines() -> None:
    """Regenerate the baseline constants and print them as a dict literal."""
    print(f"# generated with torch {torch.__version__}")
    print("BASELINE_OUTPUTS: dict[str, list[list[float]]] = {")
    for name in sorted(BUILDERS):
        rows = _forward(name).tolist()
        print(f'    "{name}": [')
        for row in rows:
            values = ", ".join(repr(value) for value in row)
            print(f"        [{values}],")
        print("    ],")
    print("}")


if __name__ == "__main__":
    _print_baselines()
