"""Autoencoder network wrapper that composes encoder and decoder modules."""

from collections.abc import Callable

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        e_net: nn.Module,
        d_net: nn.Module,
        output_layer_transformation: (
            Callable[[torch.Tensor], torch.Tensor] | None
        ) = None,  # TODO: make this a type in utils.py
    ) -> None:
        """Initialize the autoencoder wrapper.

        Args:
            e_net: Encoder network that maps inputs to latent representations.
            d_net: Decoder network that maps latent representations to outputs.
            output_layer_transformation: Optional transformation applied to decoder outputs.
        """
        super().__init__()
        self.e_net = e_net
        self.d_net = d_net
        self.output_layer_transformation = output_layer_transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the autoencoder mapping ``y = d_net(e_net(x))``.

        Args:
            x: Input tensor.

        Returns:
            Reconstructed output tensor.
        """
        z = self.e_net(x)
        y = self.d_net(z)
        if self.output_layer_transformation is not None:
            y = self.output_layer_transformation(y)
        return y

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the encoder mapping ``z = e_net(x)``.

        Args:
            x: Input tensor.

        Returns:
            Encoded latent representation.
        """
        return self.e_net(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the decoder mapping ``y = d_net(z)``.

        Args:
            z: Latent representation tensor.

        Returns:
            Decoded output tensor.
        """
        y = self.d_net(z)
        if self.output_layer_transformation is not None:
            y = self.output_layer_transformation(y)
        return y
