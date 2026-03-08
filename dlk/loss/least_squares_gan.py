"""Define least-squares GAN losses for discriminator and generator training."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def least_squares_loss_fn(
    d_outputs_gen: Optional[torch.Tensor],
    d_outputs_data: Optional[torch.Tensor],
    gen_label: float = 0.0,
    data_label: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the least-squares GAN losses for one adversarial update step.

    .. math::
        \ell_D = \frac{1}{2} E[ (a - D(x_{\mathrm{data}})^2 ] + \frac{1}{2} E[ (b + D(x_{\mathrm{gen}})^2 ]
        \ell_G = \frac{1}{2} E[ (a - D(x_{\mathrm{gen}})^2 ]

    Args:
        d_outputs_gen: Discriminator outputs for generated samples, or ``None``.
        d_outputs_data: Discriminator outputs for real data samples, or ``None``.
        gen_label: Target label assigned to generated data for discriminator loss.
        data_label: Target label assigned to real data for discriminator loss.

    Returns:
        A tuple ``(ls_loss, loss_gen)`` where:
            - ``ls_loss`` is the total least-squares loss term that sums available
              discriminator components.
            - ``loss_gen`` is the generated-sample least-squares term.

    Raises:
        ValueError: If both discriminator output tensors are ``None``.
    """
    reference_outputs = d_outputs_data if d_outputs_data is not None else d_outputs_gen
    if reference_outputs is None:
        raise ValueError(
            "At least one of d_outputs_data or d_outputs_gen must be provided."
        )

    device = reference_outputs.device
    dtype = reference_outputs.dtype
    zero_loss = torch.zeros((), device=device, dtype=dtype)

    if d_outputs_data is not None:
        loss_data = 0.5 * F.mse_loss(
            d_outputs_data, torch.full_like(d_outputs_data, data_label, device=device)
        )
    else:
        loss_data = zero_loss

    if d_outputs_gen is not None:
        loss_gen = 0.5 * F.mse_loss(
            d_outputs_gen, torch.full_like(d_outputs_gen, gen_label, device=device)
        )
    else:
        loss_gen = zero_loss

    ls_loss = loss_data + loss_gen
    return ls_loss, loss_gen
