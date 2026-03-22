"""Define least-squares GAN losses for discriminator and generator training."""

import torch
import torch.nn.functional as F


def least_squares_loss_fn(
    d_outputs_gen: torch.Tensor,
    d_outputs_data: torch.Tensor | None,
    gen_label: float = 0.0,
    data_label: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute the least-squares GAN loss functions for discriminator or generator training steps.

    .. math::
        \ell_D = \frac{1}{2} E[ (a - D(x_{\mathrm{data}}))^2 ] + \frac{1}{2} E[ (b - D(x_{\mathrm{gen}}))^2 ]
        \ell_G = \frac{1}{2} E[ (a - D(x_{\mathrm{gen}}))^2 ]

    Args:
        d_outputs_gen: Discriminator outputs for generated samples. Pass during
            discriminator and generator updates.
        d_outputs_data: Discriminator outputs for real samples. Pass during
            discriminator updates; set to ``None`` for generator-only updates.
        gen_label: Target label, b, assigned to generated data for discriminator loss.
        data_label: Target label, a, assigned to real data for discriminator loss.

    Returns:
        tuple[Tensor, Tensor | None]: A tuple containing:
            - The total least-squares loss to minimize.
            - The generated-sample least-squares term when both inputs are provided,
              otherwise ``None`` for generator-only updates.
    """
    assert d_outputs_gen is not None
    device = d_outputs_gen.device

    # loss for discriminator update
    if d_outputs_data is not None:
        loss_data = 0.5 * F.mse_loss(
            d_outputs_data, torch.full_like(d_outputs_data, data_label, device=device)
        )
        loss_gen = 0.5 * F.mse_loss(
            d_outputs_gen, torch.full_like(d_outputs_gen, gen_label, device=device)
        )
        ls_loss = loss_data + loss_gen
        return ls_loss, loss_gen

    # loss for generator update
    ls_loss = 0.5 * F.mse_loss(
        d_outputs_gen, torch.full_like(d_outputs_gen, data_label, device=device)
    )
    return ls_loss, None
