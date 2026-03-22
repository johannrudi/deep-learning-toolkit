"""Compute hinge-based adversarial losses for GAN discriminator and generator training."""

import torch
import torch.nn.functional as F


def hinge_loss_fn(
    d_outputs_gen: torch.Tensor,
    d_outputs_data: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute hinge loss functions for discriminator or generator training steps.

    .. math::
        \ell_D = E[ \max(0, 1 - D(x_{\mathrm{data}})) ] + E[ \max(0, 1 + D(x_{\mathrm{gen}})) ]
        \ell_G = -E[ D(x_{\mathrm{gen}}) ]

    Args:
        d_outputs_gen: Discriminator outputs for generated samples. Pass during
            discriminator and generator updates.
        d_outputs_data: Discriminator outputs for real samples. Pass during
            discriminator updates; set to ``None`` for generator-only updates.

    Returns:
        tuple[Tensor, Tensor | None]: A tuple containing:
            - The total hinge loss to minimize.
            - The generated-sample hinge term when both inputs are provided,
              otherwise ``None`` for generator-only updates.
    """
    assert d_outputs_gen is not None

    # loss for discriminator update
    if d_outputs_data is not None:
        loss_gen = torch.mean(F.relu(1.0 + d_outputs_gen))
        loss_data = torch.mean(F.relu(1.0 - d_outputs_data))
        hinge_loss = loss_data + loss_gen
        return hinge_loss, loss_gen

    # loss for generator update
    hinge_loss = -torch.mean(d_outputs_gen)
    return hinge_loss, None
