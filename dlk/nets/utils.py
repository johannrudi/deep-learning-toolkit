"""Provide utilities for layer initialization and parameter accounting."""

import math
from collections.abc import Callable
from typing import Any, Literal, Protocol, cast

import torch
import torch.nn as nn
from prettytable import PrettyTable
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _SpectralNorm

# --------------------------------------
# Types
# --------------------------------------

TensorTransform = Callable[[torch.Tensor], torch.Tensor]
Activation = TensorTransform | nn.Module
Nonlinearity = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]

SUPPORTED_NONLINEARITIES: set[str] = set(Nonlinearity.__args__)

ModuleFactory = Callable[..., nn.Module]
ConvFactory = Callable[[int, int], nn.Module]
LinearFactory = Callable[[int, int], nn.Module]
SampleFactory = Callable[[int, int, int], nn.Module]
NormalizationFactory = Callable[[int], nn.Module]


class WeightedLayer(Protocol):
    """Define the interface required for parameter initialization helpers."""

    @property
    def weight(self) -> torch.Tensor:
        """Return the trainable weight tensor."""
        ...

    @property
    def bias(self) -> torch.Tensor | None:
        """Return the optional trainable bias tensor."""
        ...


WEIGHTED_LAYER_COMPATIBLE_TYPES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

# --------------------------------------


def _resolve_nonlinearity(
    activation_name: str,
    default: Nonlinearity,
) -> Nonlinearity:
    """Resolve a string activation name into a supported nonlinearity literal.

    Args:
        activation_name: Raw name of the activation to resolve.
        default: Fallback nonlinearity when the name is unsupported.

    Returns:
        Valid nonlinearity literal accepted by ``torch.nn.init.calculate_gain``.
    """
    normalized_name = (
        "relu"
        if activation_name.lower() in {"silu", "gelu"}
        else activation_name.lower()
    )
    if normalized_name in SUPPORTED_NONLINEARITIES:
        return cast(Nonlinearity, normalized_name)
    return default


def get_gain(
    activation: object | None,
    default: Nonlinearity = "linear",
) -> float:
    """Calculate the gain for parameter initialization.

    Args:
        activation: Activation-like value used by the layer, or ``None``.
        default: Fallback gain mode when activation lookup fails.

    Returns:
        Gain value for initialization.
    """
    if isinstance(activation, nn.Module):
        activation_name = type(activation).__name__.lower()
        nonlinearity = _resolve_nonlinearity(activation_name, default)
        return nn.init.calculate_gain(nonlinearity)
    return nn.init.calculate_gain(default)


def _resolve_layer(module: Any, *, name: str = "module") -> WeightedLayer:
    # check the module class
    if not isinstance(module, WEIGHTED_LAYER_COMPATIBLE_TYPES):
        raise TypeError(
            f"{name} must be a supported Linear/Conv* layer, got {type(module).__name__}"
        )

    # validate protocol fields
    if not isinstance(module.weight, torch.Tensor):
        raise TypeError(f"{name}.weight must be a torch.Tensor")
    if module.bias is not None and not isinstance(module.bias, torch.Tensor):
        raise TypeError(f"{name}.bias must be torch.Tensor | None")

    return cast(WeightedLayer, module)


def get_spectral_norm(layer: nn.Module) -> tuple[_SpectralNorm, torch.Tensor] | None:
    """Return the spectral norm of `layer.weight` and the weight it normalizes.

    Args:
        layer: Layer that may be wrapped with `parametrizations.spectral_norm`.

    Returns:
        The spectral-norm parametrization and the trainable weight, or ``None``
        if `layer.weight` is not spectrally normalized.
    """
    if not parametrize.is_parametrized(layer, "weight"):
        return None
    parametrizations = cast(nn.ModuleDict, layer.parametrizations)
    weight_parametrizations = cast(
        parametrize.ParametrizationList, parametrizations["weight"]
    )
    for parametrization in weight_parametrizations:
        if isinstance(parametrization, _SpectralNorm):
            return parametrization, cast(torch.Tensor, weight_parametrizations.original)
    return None


def set_init_parameters(
    layer: Any,
    gain: float = 1.0,
    bias_scale: float = 0.1,
) -> None:
    """Initialize trainable parameters of a layer.

    Layers wrapped with `parametrizations.spectral_norm` get an orthogonal weight in
    place of Xavier's. The normalization divides out any weight scale, so
    `gain` has no effect on them, and an orthogonal weight keeps every singular
    value at 1, so the layer starts norm-preserving (Anil et al., 2019). For
    convolutions, the normalization bounds the reshaped kernel matrix, not the
    convolution operator, whose norm can be larger (Sedghi et al., 2019; Gouk
    et al., 2021; Farnia et al., 2019).

    References:
        Miyato et al., "Spectral Normalization for Generative Adversarial
        Networks", ICLR 2018. https://arxiv.org/abs/1802.05957

        Anil et al., "Sorting Out Lipschitz Function Approximation", ICML 2019.
        https://arxiv.org/abs/1811.05381

        Sedghi et al., "The Singular Values of Convolutional Layers", ICLR 2019.
        https://arxiv.org/abs/1805.10408

        Gouk et al., "Regularisation of Neural Networks by Enforcing Lipschitz
        Continuity", Machine Learning 2021. https://arxiv.org/abs/1804.04368

        Farnia et al., "Generalizable Adversarial Training via Spectral
        Normalization", ICLR 2019. https://arxiv.org/abs/1811.07457

    Args:
        layer: Layer to initialize.
        gain: Gain scaling for Xavier initialization and the bias range.
        bias_scale: Uniform scale factor used for bias initialization.

    Returns:
        None.
    """
    spectral_norm_and_weight = get_spectral_norm(layer)
    layer = _resolve_layer(layer)
    if spectral_norm_and_weight is None:
        nn.init.xavier_uniform_(layer.weight, gain=gain)
    else:
        # initialize the trainable weight; `layer.weight` is derived from it
        spectral_norm, weight_orig = spectral_norm_and_weight
        nn.init.orthogonal_(weight_orig)
        # re-sync power-iteration vectors, as registering the parametrization does
        weight_mat = spectral_norm._reshape_weight_to_matrix(weight_orig.detach())
        spectral_norm._power_method(weight_mat, 15)
    if layer.bias is not None:
        lim = bias_scale * gain / math.sqrt(layer.bias.size(0))
        nn.init.uniform_(layer.bias, a=-lim, b=+lim)


def set_zero_parameters(layer: nn.Module) -> nn.Module:
    """Set all layer parameters to zero.

    Args:
        layer: Layer whose parameters are zeroed.

    Returns:
        The same layer with zeroed parameters.
    """
    for parameter in layer.parameters():
        torch.nn.init.zeros_(parameter)
    return layer


def get_parameters(net: nn.Module) -> tuple[int, int, PrettyTable]:
    """Build a table with parameter counts and trainability flags for a network.

    Source:
        https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    Args:
        net: Network to inspect.

    Returns:
        Tuple containing trainable parameter count, non-trainable parameter
        count, and a formatted ``PrettyTable`` summary.
    """
    table = PrettyTable(["Module name", "Num. parameters", "Trainable"])
    table.align["Module name"] = "l"
    table.align["Num. parameters"] = "r"
    table.align["Trainable"] = "c"
    n_trainable_params = 0
    n_nontrainable_params = 0
    for name, parameter in net.named_parameters():
        n_params = parameter.numel()
        if parameter.requires_grad:
            table.add_row([name, n_params, True])
            n_trainable_params += n_params
        else:
            table.add_row([name, n_params, False])
            n_nontrainable_params += n_params
    table.add_divider()
    table.add_row(
        ["Total number of trainable parameters", f"{n_trainable_params}", True]
    )
    table.add_row(
        ["Total number of non-trainable parameters", f"{n_nontrainable_params}", False]
    )
    return n_trainable_params, n_nontrainable_params, table


def count_trainable_parameters(net: nn.Module) -> int:
    """Count the number of trainable parameters in a network.

    Args:
        net: Network to inspect.

    Returns:
        Number of parameters with ``requires_grad=True``.
    """
    assert isinstance(net, nn.Module), "Expected 'net' to be an nn.Module."
    return sum(
        parameter.numel() for parameter in net.parameters() if parameter.requires_grad
    )


def count_all_parameters(net: nn.Module) -> int:
    """Count the total number of parameters in a network.

    Args:
        net: Network to inspect.

    Returns:
        Total number of parameters, including non-trainable parameters.
    """
    assert isinstance(net, nn.Module), "Expected 'net' to be an nn.Module."
    return sum(parameter.numel() for parameter in net.parameters())
