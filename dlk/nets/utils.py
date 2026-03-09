"""Provide utilities for layer initialization and parameter accounting."""

import math
from typing import Literal, Protocol, cast

import torch
import torch.nn as nn
from prettytable import PrettyTable

# --------------------------------------
# Types
# --------------------------------------

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


# --------------------------------------

SUPPORTED_NONLINEARITIES: set[str] = set(Nonlinearity.__args__)


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


def set_init_parameters(
    layer: WeightedLayer,
    gain: float = 1.0,
    bias_scale: float = 0.1,
) -> None:
    """Initialize trainable parameters of a layer.

    Args:
        layer: Layer to initialize.
        gain: Gain scaling for Xavier initialization.
        bias_scale: Uniform scale factor used for bias initialization.

    Returns:
        None.
    """
    nn.init.xavier_uniform_(layer.weight, gain=gain)
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
