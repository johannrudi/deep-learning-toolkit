"""Provide spectral normalization helpers for weighted layers.

See ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md`` (decisions 11
and 15).
"""

from typing import cast

import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _SpectralNorm, spectral_norm


def get_weight_parametrizations(
    layer: nn.Module,
) -> parametrize.ParametrizationList | None:
    """Return the parametrizations registered on `layer.weight`.

    Args:
        layer: Layer whose weight may be parametrized.

    Returns:
        The weight's parametrization list, or ``None`` if `layer.weight` has none.
    """
    if not parametrize.is_parametrized(layer, "weight"):
        return None
    parametrizations = cast(nn.ModuleDict, layer.parametrizations)
    return cast(parametrize.ParametrizationList, parametrizations["weight"])


def get_spectral_norm(layer: nn.Module) -> tuple[_SpectralNorm, torch.Tensor] | None:
    """Return the spectral norm of `layer.weight` and the weight it normalizes.

    Args:
        layer: Layer that may be wrapped with `parametrizations.spectral_norm`.

    Returns:
        The spectral-norm parametrization and the trainable weight, or ``None``
        if `layer.weight` is not spectrally normalized.
    """
    weight_parametrizations = get_weight_parametrizations(layer)
    if weight_parametrizations is None:
        return None
    for parametrization in weight_parametrizations:
        if isinstance(parametrization, _SpectralNorm):
            return parametrization, cast(torch.Tensor, weight_parametrizations.original)
    return None


def set_spectral_norm(layer: nn.Module) -> nn.Module:
    """Wrap a layer with `parametrizations.spectral_norm`, unless already wrapped.

    Args:
        layer: Layer whose weight should be spectrally normalized.

    Returns:
        The wrapped layer, or `layer` itself if it is already wrapped.
    """
    if get_spectral_norm(layer) is not None:
        return layer
    return spectral_norm(layer)


class DepthwiseSpectralNorm(nn.Module):
    r"""Normalize a depthwise convolution weight by its operator norm (C2).

    Divide the whole weight by one scalar, $\max_{c, \omega} |\hat w_c(\omega)|$,
    where $\hat w_c$ is the DFT of channel $c$'s filter, zero-padded to $n$ =
    ``dft_length`` points; the channels keep their relative gains. With circular
    padding at input length $n$, the layer's operator norm is exactly 1.
    Otherwise it is at most $1 / (1 - (k - 1)\pi / n)$ for kernel size $k$
    (Bernstein's inequality), i.e., at most about 1.24 for $k = 5$ and $n = 64$; stride
    only lowers it.

    See ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, decision 11.

    Args:
        dft_length: Number of DFT points $n$; at least the kernel size.
    """

    def __init__(self, dft_length: int = 64) -> None:
        super().__init__()
        assert dft_length > 0, f"dft_length must be positive, got {dft_length=}"
        self.dft_length = dft_length

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the weight divided by its largest DFT magnitude.

        Args:
            weight: Depthwise convolution weight of shape ``(C, 1, k)``.

        Returns:
            The normalized weight, of the same shape.
        """
        assert (
            weight.ndim == 3 and weight.size(1) == 1
        ), f"expected a depthwise weight of shape (C, 1, k), got {tuple(weight.shape)}"
        assert (
            weight.size(2) <= self.dft_length
        ), f"kernel size {weight.size(2)} exceeds dft_length {self.dft_length}"
        spectrum = torch.fft.rfft(weight[:, 0, :], n=self.dft_length)
        return weight / spectrum.abs().amax()

    def extra_repr(self) -> str:
        """Return the DFT length for the module's printed representation."""
        return f"dft_length={self.dft_length}"


def get_depthwise_spectral_norm(layer: nn.Module) -> DepthwiseSpectralNorm | None:
    """Return the depthwise spectral norm of `layer.weight`, if any.

    Args:
        layer: Layer that may be wrapped with :class:`DepthwiseSpectralNorm`.

    Returns:
        The parametrization, or ``None`` if `layer.weight` has none.
    """
    weight_parametrizations = get_weight_parametrizations(layer)
    if weight_parametrizations is None:
        return None
    for parametrization in weight_parametrizations:
        if isinstance(parametrization, DepthwiseSpectralNorm):
            return parametrization
    return None


def set_depthwise_spectral_norm(layer: nn.Module, dft_length: int = 64) -> nn.Module:
    """Wrap a depthwise convolution with :class:`DepthwiseSpectralNorm`, once.

    Args:
        layer: Depthwise ``nn.Conv1d``, with ``groups == in_channels == out_channels``.
        dft_length: Number of DFT points; at least the kernel size.

    Returns:
        The wrapped layer, or `layer` itself if it is already wrapped.
    """
    assert isinstance(layer, nn.Conv1d) and (
        layer.groups == layer.in_channels == layer.out_channels
    ), (
        "layer must be a depthwise nn.Conv1d with groups == in_channels == "
        f"out_channels, got {layer!r}"
    )
    if get_depthwise_spectral_norm(layer) is not None:
        return layer
    parametrize.register_parametrization(
        layer, "weight", DepthwiseSpectralNorm(dft_length)
    )
    return layer
