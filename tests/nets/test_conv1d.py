import math
from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm

from dlk.nets.conv1d import (
    ChannelLayerNorm,
    ConvNet,
    ConvNeXtBlock,
    ConvResNet,
    UniversalMultiLevelBlock,
)


def test_convnet_forward_output_shape() -> None:
    """Run a forward pass for ``ConvNet`` and validate output shape."""
    input_channels = 1
    input_size = 16
    hidden_conv_layers_channels_mult = [2, 4, 8]
    hidden_dense_input_size = (
        input_size - 2 * len(hidden_conv_layers_channels_mult)
    ) * hidden_conv_layers_channels_mult[-1]
    net = ConvNet(
        input_channels=input_channels,
        output_size=2,
        hidden_conv_layers_channels_mult=hidden_conv_layers_channels_mult,
        hidden_dense_input_size=hidden_dense_input_size,
        hidden_dense_layers_sizes=[32, 32],
    )
    x = torch.ones((1, input_channels, input_size))

    y = net(x)

    assert y.shape == (1, 2)


def test_convresnet_forward_without_mlp_head() -> None:
    """Validate ``ConvResNet`` output shape when only conv blocks are enabled."""
    batch_size = 4
    input_channels = 1
    input_length = 64
    channels_mult = [4, 8]
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": channels_mult,
            "kernels": [3, 3],
        },
    )
    x = torch.randn(batch_size, input_channels, input_length)
    expected_out_size = (
        batch_size,
        input_channels * channels_mult[-1],
        input_length // 4,
    )

    y = net(x)

    assert y.shape == expected_out_size


def test_convresnet_forward_with_mlp_head() -> None:
    """Validate ``ConvResNet`` output shape when an ``MLPResNet`` head is enabled."""
    batch_size = 4
    input_channels = 1
    input_length = 64
    output_size = 10
    channels_mult = [4, 8]
    conv_out_size = (
        batch_size,
        input_channels * channels_mult[-1],
        input_length // 4,
    )
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": channels_mult,
            "kernels": [3, 3],
        },
        mlp_resnet_params={
            "input_size": conv_out_size[1] * conv_out_size[2],
            "output_size": output_size,
            "residual_blocks_sizes": [(16, 16, 64, 16)],
        },
    )
    x = torch.randn(batch_size, input_channels, input_length)

    y = net(x)

    assert y.shape == (batch_size, output_size)


def test_convresnet_forward_with_hidden_inputs() -> None:
    """Validate ``ConvResNet`` output shape when hidden inputs are passed to MLP blocks."""
    batch_size = 4
    input_channels = 1
    input_length = 64
    output_size = 10
    hidden_input_size = 8
    channels_mult = [4, 8]
    conv_out_size = (
        batch_size,
        input_channels * channels_mult[-1],
        input_length // 4,
    )
    net = ConvResNet(
        input_channels=input_channels,
        conv_resnet_params={
            "channels_mult": channels_mult,
            "kernels": [3, 3],
        },
        mlp_resnet_params={
            "input_size": conv_out_size[1] * conv_out_size[2],
            "output_size": output_size,
            "residual_blocks_sizes": [
                (16, 16, 64, 16),
                (16 + hidden_input_size, 16, 64, 16),
            ],
        },
    )
    x = torch.randn(batch_size, input_channels, input_length)
    h1 = torch.randn(batch_size, 1, hidden_input_size)

    y = net(x, h1=h1)

    assert y.shape == (batch_size, output_size)


def test_multilevel_block_forward_shapes_without_scaling() -> None:
    """Validate ``UniversalMultiLevelBlock`` shapes for level blocks without resizing."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    x = torch.randn(batch_size, input_channels, input_length)

    net = UniversalMultiLevelBlock(input_channels=input_channels, kernel_size=3)
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=False,
        activation=False,
    )
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)

    output_channels = 32
    normalization_channels = 8
    activation_channels = 64
    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=lambda num_channels: nn.GroupNorm(2, num_channels),
        normalization_channels=normalization_channels,
        activation=nn.SiLU(),
        activation_channels=activation_channels,
        output_channels=output_channels,
    )
    y = net(x)
    assert y.shape == (batch_size, output_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        output_channels=output_channels,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == (batch_size, output_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        dropout=0.1,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == (batch_size, input_channels, input_length)

    # concatenate two inputs along the channels
    net = UniversalMultiLevelBlock(input_channels=input_channels, kernel_size=3)
    y = net(x[:, :4], x[:, 4:])
    assert y.shape == (batch_size, input_channels, input_length)


@pytest.mark.parametrize(
    "kernel_size, conv_kwargs, scale_factor, expected_length",
    [
        (3, {"padding": 0}, None, 62),
        (3, {"padding": 0}, 0.5, 31),
        (4, None, None, 64),
        (4, None, 0.5, 32),
        (3, {"dilation": 2}, 0.5, 32),
        (3, None, 1 / 3, 22),
        (3, None, 1.5, 96),
    ],
)
def test_multilevel_block_skip_matches_main_branch_length(
    kernel_size: int,
    conv_kwargs: dict | None,
    scale_factor: float | None,
    expected_length: int,
) -> None:
    """Validate that the skip branch follows the length of the main branch."""
    x = torch.randn(2, 8, 64)
    net = UniversalMultiLevelBlock(
        input_channels=8,
        kernel_size=kernel_size,
        output_channels=16,
        scale_factor=scale_factor,
        skip_connection=True,
        conv_kwargs=conv_kwargs,
    )
    y = net(x)
    assert y.shape == (2, 16, expected_length)


@pytest.mark.parametrize("input_length", [15, 16, 17])
@pytest.mark.parametrize(
    "kernel_size, conv_kwargs, scale_factor",
    [
        (3, None, None),
        (4, None, None),
        (3, None, 0.5),
        (4, None, 0.5),
        (5, {"dilation": 2}, 0.5),
        (3, None, 1 / 3),
        (3, {"padding": 0}, 0.5),
        (3, {"padding": "valid"}, None),
        (3, {"padding": (2,), "stride": 2}, None),
        (3, None, 1.5),
        (3, None, 2.0),
    ],
)
def test_multilevel_block_resolve_output_length_matches_forward(
    input_length: int,
    kernel_size: int,
    conv_kwargs: dict | None,
    scale_factor: float | None,
) -> None:
    """Validate that ``resolve_output_length`` predicts the output length."""
    net = UniversalMultiLevelBlock(
        input_channels=4,
        kernel_size=kernel_size,
        scale_factor=scale_factor,
        skip_connection=True,
        conv_kwargs=conv_kwargs,
    )
    y = net(torch.randn(2, 4, input_length))
    assert net.resolve_output_length(input_length) == y.size(2)


def test_convresnet_input_length_sets_mlp_input_size() -> None:
    """Validate that ``input_length`` derives the input size of the MLP head."""
    input_length = 30
    channels_mult = [4, 8]
    mlp_resnet_params = {
        "output_size": 10,
        "residual_blocks_sizes": [(16, 16, 64, 16)],
    }
    net = ConvResNet(
        input_channels=1,
        conv_resnet_params={"channels_mult": channels_mult, "kernels": [3, 3]},
        mlp_resnet_params=mlp_resnet_params,
        input_length=input_length,
    )
    assert "input_size" not in mlp_resnet_params

    y = net(torch.randn(4, 1, input_length))

    assert y.shape == (4, 10)
    assert net.conv_output_length == 8  # 30 -> 15 -> 8
    assert net.resolve_input_shape() == (1, input_length)
    assert net.mlp_resnet is not None
    assert net.mlp_resnet.input_size == 8 * 8


def test_convresnet_input_length_checks() -> None:
    """Validate the checks of ``input_length`` against sizes and inputs."""
    conv_resnet_params = {"channels_mult": [4, 8], "kernels": [3, 3]}
    net = ConvResNet(1, conv_resnet_params=conv_resnet_params, input_length=64)
    assert net.resolve_output_shape() == (8, 16)
    with pytest.raises(AssertionError, match="input length"):
        net(torch.randn(2, 1, 63))
    with pytest.raises(ValueError, match="flattened conv output"):
        ConvResNet(
            1,
            conv_resnet_params=conv_resnet_params,
            mlp_resnet_params={"input_size": 8 * 15, "output_size": 3},
            input_length=64,
        )
    with pytest.raises(ValueError, match="ConvResNet must have 'input_length'"):
        ConvResNet(
            1,
            conv_resnet_params=conv_resnet_params,
            mlp_resnet_params={"output_size": 3},
        )
    with pytest.raises(ValueError, match="must be positive"):
        ConvResNet(1, conv_resnet_params=conv_resnet_params, input_length=0)


def test_multilevel_block_scale_factor_and_stride() -> None:
    """Validate the coupling of ``scale_factor`` and the stride."""
    net = UniversalMultiLevelBlock(
        input_channels=8, kernel_size=3, scale_factor=0.5, conv_kwargs={"stride": 2}
    )
    assert net.scale_factor == 0.5
    net = UniversalMultiLevelBlock(
        input_channels=8, kernel_size=3, conv_kwargs={"stride": 4}
    )
    assert net.scale_factor == 0.25
    with pytest.raises(ValueError, match="requires stride"):
        UniversalMultiLevelBlock(
            input_channels=8, kernel_size=3, scale_factor=0.5, conv_kwargs={"stride": 3}
        )
    with pytest.raises(ValueError, match="inverse of an integer"):
        UniversalMultiLevelBlock(input_channels=8, kernel_size=3, scale_factor=0.3)
    with pytest.raises(ValueError, match="must be positive"):
        UniversalMultiLevelBlock(input_channels=8, kernel_size=3, scale_factor=0.0)


def test_multilevel_block_starts_as_skip_branch() -> None:
    """Validate that the main branch starts at zero with a skip branch."""
    x = torch.randn(2, 8, 64)
    net = UniversalMultiLevelBlock(
        input_channels=8, kernel_size=3, skip_connection=True, skip_scale=0.5
    )
    torch.testing.assert_close(net(x), 0.5 * x)

    net = UniversalMultiLevelBlock(
        input_channels=8, kernel_size=3, activation=False, skip_connection=True
    )
    torch.testing.assert_close(net(x), x)


def test_multilevel_block_rejects_channels_without_operation() -> None:
    """Validate that channels of a disabled operation raise an error."""
    with pytest.raises(ValueError, match="normalization_channels"):
        UniversalMultiLevelBlock(
            input_channels=16,
            kernel_size=3,
            normalization=False,
            normalization_channels=8,
        )
    with pytest.raises(ValueError, match="activation_channels"):
        UniversalMultiLevelBlock(
            input_channels=16,
            kernel_size=3,
            activation=False,
            activation_channels=64,
        )


def test_multilevel_block_forward_shapes_with_downsampling() -> None:
    """Validate ``UniversalMultiLevelBlock`` shapes when downsampling is enabled."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    expected_out_size = (batch_size, input_channels, input_length // 2)
    x = torch.randn(batch_size, input_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=0.5,
        normalization=False,
        activation=False,
    )
    y = net(x)
    assert y.shape == expected_out_size

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        skip_connection=True,
        conv_kwargs={"stride": 2},
    )
    y = net(x)
    assert y.shape == expected_out_size

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=0.5,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == expected_out_size


def test_multilevel_block_forward_shapes_with_upsampling() -> None:
    """Validate ``UniversalMultiLevelBlock`` shapes when upsampling is enabled."""
    batch_size = 4
    input_channels = 16
    input_length = 64
    expected_out_size = (batch_size, input_channels, input_length * 2)
    x = torch.randn(batch_size, input_channels, input_length)

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=2.0,
        normalization=False,
        activation=False,
    )
    y = net(x)
    assert y.shape == expected_out_size

    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        scale_factor=2.0,
        skip_connection=True,
    )
    y = net(x)
    assert y.shape == expected_out_size


def test_channel_layer_norm_normalizes_each_position_over_channels() -> None:
    """Validate ``ChannelLayerNorm`` against statistics over the channels."""
    x = torch.randn(4, 8, 32)
    norm = ChannelLayerNorm(8)
    y = norm(x)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    torch.testing.assert_close(y, (x - mean) / torch.sqrt(var + norm.eps))


@pytest.mark.filterwarnings("ignore:Using padding='same' with even kernel:UserWarning")
@pytest.mark.parametrize("kernel_size", [7, 4])
def test_convnext_block_keeps_input_shape(kernel_size: int) -> None:
    """Validate that ``ConvNeXtBlock`` keeps channels and length."""
    x = torch.randn(4, 16, 63)
    net = ConvNeXtBlock(input_channels=16, kernel_size=kernel_size)
    y = net(x)
    assert y.shape == x.shape


def test_convnext_block_layers() -> None:
    """Validate the layers of ``ConvNeXtBlock`` and that it starts as the identity."""
    x = torch.randn(4, 16, 64)
    net = ConvNeXtBlock(input_channels=16, conv_kwargs={"padding_mode": "replicate"})
    conv_0 = cast(nn.Conv1d, net.block.conv_0)
    assert (conv_0.groups, conv_0.padding_mode) == (16, "replicate")
    assert isinstance(net.block.normalization, ChannelLayerNorm)
    assert isinstance(net.block.activation, nn.GELU)
    assert cast(nn.Conv1d, net.block.conv_1).out_channels == 64
    assert isinstance(net.skip_connection, nn.Identity)
    torch.testing.assert_close(net(x), x)


def test_multilevel_block_drop_path() -> None:
    """Validate that drop path keeps or drops the main branch per sample."""
    torch.manual_seed(0)
    x = torch.randn(64, 16, 32)
    net = ConvNeXtBlock(input_channels=16, drop_path=0.5)
    nn.init.normal_(cast(nn.Conv1d, net.block.conv_2).weight)

    # evaluate without drop path
    net.eval()
    branch = net(x) - x

    # drop path either removes or rescales the branch of each sample
    net.train()
    y = net(x)
    dropped = torch.isclose(y, x).flatten(1).all(dim=1)
    assert 0 < dropped.sum() < x.size(0)
    torch.testing.assert_close(y[~dropped], x[~dropped] + 2.0 * branch[~dropped])

    with pytest.raises(ValueError, match="drop_path"):
        ConvNeXtBlock(input_channels=16, drop_path=1.0)
    with pytest.raises(ValueError, match="requires skip_connection"):
        UniversalMultiLevelBlock(input_channels=16, kernel_size=3, drop_path=0.1)


def test_convresnet_state_dict_keys_unchanged() -> None:
    """Validate that the flag off leaves the `state_dict` keys of `ConvResNet` untouched."""
    net = ConvResNet(
        input_channels=1,
        conv_resnet_params={"channels_mult": [2, 4], "kernels": [3, 3]},
        mlp_resnet_params={"input_size": 4 * 4, "output_size": 3},
        input_length=16,
    )
    assert not any("parametrizations" in key for key in net.state_dict())


def test_convresnet_spectral_norm_wraps_every_conv() -> None:
    """Wrap the input layer and every block convolution; leave the MLP head unwrapped."""
    net = ConvResNet(
        input_channels=1,
        input_length=64,
        conv_resnet_params={
            "channels_mult": [2, 4, 8],
            "kernels": [3, 3, 3],
            "enable_spectral_norm": True,
        },
        mlp_resnet_params={
            "output_size": 3,
            "residual_blocks_sizes": [(64, 32, 128, 16)],
        },
    )

    # count the convolutions the ship gate expects: input layer + 3 per level +
    # 1 skip per level that changes channels
    input_convs = [m for m in net.input_layer.modules() if isinstance(m, nn.Conv1d)]
    block_convs = [m for m in net.conv_resnet.modules() if isinstance(m, nn.Conv1d)]
    n_skip_convs = sum(
        isinstance(cast(UniversalMultiLevelBlock, level).skip_connection, nn.Conv1d)
        for level in net.conv_resnet
    )
    assert len(input_convs) == 1
    assert len(block_convs) == 3 * len(net.conv_resnet) + n_skip_convs
    assert all(
        parametrize.is_parametrized(m, "weight") for m in input_convs + block_convs
    )

    # the MLP head keeps its own flag and stays unwrapped
    assert net.mlp_resnet is not None
    mlp_linears = [m for m in net.mlp_resnet.modules() if isinstance(m, nn.Linear)]
    assert not any(parametrize.is_parametrized(m, "weight") for m in mlp_linears)

    # five training forwards let the power iteration converge sigma to 1
    net.train()
    x = torch.randn(2, 1, 64)
    for _ in range(5):
        net(x)
    net.eval()
    for conv in input_convs + block_convs:
        weight = cast(torch.Tensor, conv.weight).detach()
        sigma = torch.linalg.matrix_norm(weight.flatten(1), ord=2)
        torch.testing.assert_close(sigma, torch.tensor(1.0), rtol=1e-3, atol=1e-3)

    # the last convolution of every level stays nonzero after init_parameters
    for level in net.conv_resnet:
        level = cast(UniversalMultiLevelBlock, level)
        conv_names = [
            name for name, _ in level.block.named_children() if name.startswith("conv_")
        ]
        last_conv = cast(nn.Conv1d, getattr(level.block, conv_names[-1]))
        weight = cast(torch.Tensor, last_conv.weight).detach()
        assert not torch.allclose(weight, torch.zeros_like(weight))


@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("activation", [True, False])
@pytest.mark.parametrize("output_channels", [None, 16])
def test_multilevel_block_spectral_norm_wraps_active_convs(
    normalization: bool, activation: bool, output_channels: int | None
) -> None:
    """Wrap exactly the convolutions active for a normalization/activation combination."""
    input_channels = 8
    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=3,
        normalization=normalization,
        activation=activation,
        output_channels=output_channels,
        skip_connection=True,
        enable_spectral_norm=True,
    )

    conv_names = [
        name for name, _ in net.block.named_children() if name.startswith("conv_")
    ]
    assert len(conv_names) == 1 + int(normalization) + int(activation)
    for name in conv_names:
        module = getattr(net.block, name)
        assert parametrize.is_parametrized(module, "weight")

    if isinstance(net.skip_connection, nn.Identity):
        assert output_channels in (None, input_channels)
    else:
        assert isinstance(net.skip_connection, nn.Conv1d)
        assert parametrize.is_parametrized(net.skip_connection, "weight")


def test_multilevel_block_spectral_norm_is_idempotent() -> None:
    """A `conv` factory that already wraps with spectral_norm gets no second wrap."""

    def wrapped_conv(
        in_channels: int, out_channels: int, kernel_size: int, **kwargs: Any
    ) -> nn.Module:
        return spectral_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        )

    net = UniversalMultiLevelBlock(
        input_channels=8,
        kernel_size=3,
        output_channels=16,
        skip_connection=True,
        enable_spectral_norm=True,
        conv=wrapped_conv,
    )

    conv_modules = [
        module
        for name, module in net.block.named_children()
        if name.startswith("conv_")
    ]
    assert isinstance(net.skip_connection, nn.Conv1d)
    conv_modules.append(net.skip_connection)

    for module in conv_modules:
        parametrizations = cast(nn.ModuleDict, module.parametrizations)
        weight_parametrizations = cast(
            parametrize.ParametrizationList, parametrizations["weight"]
        )
        assert len(weight_parametrizations) == 1


def test_convnext_block_spectral_norm_wraps_depthwise_conv() -> None:
    """Wrap the depthwise convolution and bound each of its groups by 1."""
    net = ConvNeXtBlock(input_channels=8, kernel_size=3, enable_spectral_norm=True)
    conv_0 = cast(nn.Conv1d, net.block.conv_0)
    assert parametrize.is_parametrized(conv_0, "weight")

    # five training forwards let the power iteration converge sigma to 1
    net.train()
    x = torch.randn(2, 8, 32)
    for _ in range(5):
        net(x)
    net.eval()

    # depthwise: weight shape (channels, 1, kernel_size), one group per channel
    weight = cast(torch.Tensor, conv_0.weight).detach()
    for group_weight in weight:
        sigma = torch.linalg.matrix_norm(group_weight, ord=2)
        assert sigma <= 1.0 + 1e-3


def _operator_norm(
    conv: nn.Conv1d, input_shape: tuple[int, int, int], steps: int = 50
) -> torch.Tensor:
    """Estimate the operator norm of a bias-free convolution by power iteration.

    Alternates the convolution and its adjoint through `torch.func.vjp`, as
    `_SpectralNorm._power_method` does for the reshaped kernel matrix.

    Args:
        conv: Convolution whose bias-free operator norm is estimated.
        input_shape: Shape of the tensor the convolution is applied to.
        steps: Number of power-iteration steps.

    Returns:
        The estimated operator norm.
    """
    weight = cast(torch.Tensor, conv.weight).detach()

    def apply(x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv1d(
            x,
            weight,
            bias=None,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    v = torch.randn(input_shape)
    v = v / v.norm()
    for _ in range(steps):
        u, vjp_fn = cast(
            "tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor]]]",
            torch.func.vjp(apply, v),
        )
        u = u / u.norm()
        (v,) = vjp_fn(u)
        v = v / v.norm()
    return apply(v).norm()


@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("scale_factor", [1, 0.5])
def test_multilevel_block_spectral_norm_lipschitz_bound(
    kernel_size: int, scale_factor: float
) -> None:
    """Validate the Lipschitz bound of a strictly-configured spectrally normalized block."""
    torch.manual_seed(0)
    input_channels = 4
    output_channels = 8
    input_length = 32
    net = UniversalMultiLevelBlock(
        input_channels=input_channels,
        kernel_size=kernel_size,
        output_channels=output_channels,
        normalization=False,
        activation=nn.ReLU(),
        scale_factor=scale_factor,
        skip_connection=True,
        skip_scale=0.5,
        enable_spectral_norm=True,
        conv_kwargs={"padding_mode": "zeros"},
    )

    # converge the power-iteration vectors with training-mode forwards
    net.train()
    x = torch.randn(4, input_channels, input_length)
    for _ in range(5):
        net(x)
    net.eval()

    # estimate the operator norm of each bias-free convolution
    conv_0 = cast(nn.Conv1d, net.block.conv_0)
    sigma_0 = _operator_norm(conv_0, (1, input_channels, input_length))
    assert sigma_0 <= math.sqrt(kernel_size) + 1e-3

    conv_2 = cast(nn.Conv1d, net.block.conv_2)
    conv_2_input_length = net.resolve_output_length(input_length)
    sigma_2 = _operator_norm(conv_2, (1, conv_2.in_channels, conv_2_input_length))
    assert sigma_2 <= 1.0 + 1e-3

    skip_connection = cast(nn.Conv1d, net.skip_connection)
    sigma_skip = _operator_norm(skip_connection, (1, input_channels, input_length))
    assert sigma_skip <= 1.0 + 1e-3

    # empirical Lipschitz ratio over random pairs
    bound = 0.5 * (1.0 + math.sqrt(kernel_size))
    x1 = torch.randn(64, input_channels, input_length)
    x2 = torch.randn(64, input_channels, input_length)
    with torch.no_grad():
        y1 = net(x1)
        y2 = net(x2)
    ratios = (y1 - y2).flatten(1).norm(dim=1) / (x1 - x2).flatten(1).norm(dim=1)
    assert ratios.max() <= bound + 1e-3
