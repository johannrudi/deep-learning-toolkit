"""Provide Wasserstein GAN loss and gradient-penalty utilities for critic training."""

from collections.abc import Callable

import torch


def wasserstein_loss_fn(
    d_outputs_gen: torch.Tensor | None,
    d_outputs_data: torch.Tensor | None,
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    """Calculate the Wasserstein critic loss and generator score term.

    D_loss = -E[D(data)] + E[D(gen)].

    Args:
        d_outputs_gen: Critic outputs evaluated on generated samples.
        d_outputs_data: Critic outputs evaluated on real samples.

    Returns:
        A tuple containing:
            - Critic loss value to minimize.
            - Generator score term E[D(gen)].
    """
    loss_gen = torch.mean(d_outputs_gen) if d_outputs_gen is not None else 0.0
    loss_data = torch.mean(d_outputs_data) if d_outputs_data is not None else 0.0
    w_loss = loss_data - loss_gen  # value to be maximized
    return -w_loss, loss_gen  # value to be minimized


def gradient_norm_sq(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the squared gradient norm used by Wasserstein penalties.

    Args:
        d_net: Critic network used to score interpolated samples.
        x_gen: Generated samples from the model.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic.
        device: Device used to sample interpolation coefficients.

    Returns:
        Per-sample squared L2 norm of critic gradients.
    """
    batch_size, *other_dims = x_data.size()
    epsilon = torch.rand([batch_size] + [1] * len(other_dims), device=device)
    epsilon = epsilon.expand(-1, *other_dims)
    x_hat = epsilon * x_data + (1.0 - epsilon) * x_gen
    x_hat.requires_grad = True
    if y_data is not None:
        y_data.requires_grad = True
        d_outputs_hat = d_net(x_hat, y_data)
        grad_inputs = (x_hat, y_data)
    else:
        d_outputs_hat = d_net(x_hat)
        grad_inputs = x_hat
    # compute gradient
    grad_outputs = torch.ones_like(d_outputs_hat, device=device)
    grad = torch.autograd.grad(
        outputs=d_outputs_hat,
        inputs=grad_inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    # compute the squared l2-norm of the gradient
    grad_x = grad[0].view(batch_size, -1)
    grad_x_norm = torch.sum(torch.square(grad_x), dim=1)
    if y_data is not None:
        grad_y = grad[1].view(batch_size, -1)
        grad_y_norm = torch.sum(torch.square(grad_y), dim=1)
        grad_norm_sq = torch.add(grad_x_norm, grad_y_norm)
    else:
        grad_norm_sq = grad_x_norm
    return grad_norm_sq


def gradient_penalty_lip(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    eps: float = 0.0,
    device: torch.device | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    """Computes the regularization term for the critic network.

    This penalizes gradients greater `k` to achieve k-Lipschitz continuity.

    Args:
        d_net: Critic network used to score interpolated samples.
        x_gen: Generated samples from the model.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic.
        lip: Target Lipschitz constant.
        eps: Numerical margin added before thresholding.
        device: Device used to sample interpolation coefficients.
        dlog: Optional dictionary for logging summary statistics.

    Returns:
        Scalar gradient penalty term.
    """
    grad_norm_sq = gradient_norm_sq(d_net, x_gen, x_data, y_data=y_data, device=device)
    grad_norm = torch.sqrt(grad_norm_sq.detach())  # only for logging purposes
    grad_penalty = torch.nn.functional.relu(grad_norm_sq + eps - lip * lip).mean()
    # log to dictionary
    if dlog is not None:
        assert isinstance(dlog, dict), type(dlog)
        dlog["grad_norm"] = grad_norm.detach().mean().item()
    return grad_penalty


def gradient_penalty_opt(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    device: torch.device | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    """Computes the regularization term for the critic network.

    This achieves the optimal Kantorovich potential in the Kantorovich–Rubinstein
    duality.

    Args:
        d_net: Critic network used to score interpolated samples.
        x_gen: Generated samples from the model.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic.
        device: Device used to sample interpolation coefficients.
        dlog: Optional dictionary for logging summary statistics.

    Returns:
        Scalar gradient penalty term.
    """
    grad_norm_sq = gradient_norm_sq(d_net, x_gen, x_data, y_data=y_data, device=device)
    grad_norm = torch.sqrt(grad_norm_sq)
    grad_penalty = ((grad_norm - 1.0) ** 2).mean()
    # log to dictionary
    if dlog is not None:
        assert isinstance(dlog, dict), type(dlog)
        dlog["grad_norm"] = grad_norm.detach().mean().item()
    return grad_penalty
