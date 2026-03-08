"""Compute hinge-based adversarial losses for GAN discriminator and generator training."""

import torch
from torch.nn import functional as F


def hinge_loss_fn(
    d_outputs_gen: torch.Tensor | None,
    d_outputs_data: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute hinge losses for discriminator or generator training steps.

    .. math::
        \ell_D = E[ \max(0, 1 - D(x_{\mathrm{data}})) ] + E[ \max(0, 1 + D(x_{\mathrm{gen}})) ]
        \ell_G = -E[ D(x_{\mathrm{gen}})) ]

    Args:
        d_outputs_gen: Discriminator outputs for generated samples.
            Pass this tensor during discriminator updates.
        d_outputs_data: Discriminator outputs for real samples during
            discriminator updates. During generator-only updates, pass
            discriminator outputs for generated samples.

    Returns:
        tuple[Tensor, Tensor | None]: A tuple containing:
            - The total hinge loss for the active training step.
            - The generated-sample hinge term (`loss_gen`) when both inputs are
              provided, otherwise `None` for generator-only updates.

    Raises:
        ValueError: If both inputs are `None`.
    """
    if d_outputs_gen is not None and d_outputs_data is not None:
        loss_gen = torch.mean(F.relu(1.0 + d_outputs_gen))
        loss_data = torch.mean(F.relu(1.0 - d_outputs_data))
        hinge_loss = loss_data + loss_gen
        return hinge_loss, loss_gen

    if d_outputs_data is None:
        raise ValueError("d_outputs_data must be provided when d_outputs_gen is None.")

    hinge_loss = -torch.mean(d_outputs_data)
    return hinge_loss, None
