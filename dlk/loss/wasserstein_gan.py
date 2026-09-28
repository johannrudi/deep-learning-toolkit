"""Provide the Wasserstein GAN loss for critic and generator training."""

import torch


def wasserstein_loss_fn(
    d_outputs_gen: torch.Tensor,
    d_outputs_data: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute the Wasserstein value functions for critic or generator training steps.

    The critic `D` approximates the maximizer of the Kantorovich-Rubinstein dual
    form of the Wasserstein-1 distance between the data distribution
    :math:`P_r` and the generator distribution :math:`P_g` (Villani 2009;
    Arjovsky et al. 2017),

    .. math::
        W_1(P_r, P_g) = \sup_{\|f\|_L \le 1} \; E_{x \sim P_r}[f(x)] - E_{x \sim P_g}[f(x)],

    which this function estimates over a batch as

    .. math::
        \ell_D = -E[ D(x_\mathrm{data}) ] + E[ D(x_\mathrm{gen}) ]
        \ell_G = -E[ D(x_\mathrm{gen}) ]

    Any `K`-Lipschitz critic yields `K` times the true distance, so only `K`
    needs to stay bounded; the exact value does not matter. The penalties in
    `dlk.loss.gradient_penalty` and `dlk.loss.spectral_penalty` constrain `K`.

    Args:
        d_outputs_gen: Critic outputs for generated samples. Pass during
            critic and generator updates.
        d_outputs_data: Critic outputs for real samples. Pass during
            critic updates; set to ``None`` for generator-only updates.

    Returns:
        tuple[Tensor, Tensor | None]: A tuple containing:
            - The total critic loss term to minimize.
            - The generated-sample score term when both inputs are provided,
              otherwise ``None`` for generator-only updates.

    References:
        Villani, "Optimal Transport: Old and New", Springer 2009.

        Arjovsky, Chintala, Bottou, "Wasserstein GAN", 2017.
        https://arxiv.org/abs/1701.07875
    """
    assert d_outputs_gen is not None

    # loss for critic update
    if d_outputs_data is not None:
        score_data = torch.mean(d_outputs_data)
        score_gen = torch.mean(d_outputs_gen)
        w_score = score_data - score_gen  # value to be maximized
        w_loss = -w_score  # value to be minimized
        return w_loss, score_gen

    # loss for generator update
    w_loss = -torch.mean(d_outputs_gen)  # value to be minimized
    return w_loss, None
