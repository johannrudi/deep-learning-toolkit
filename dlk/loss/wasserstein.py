import torch


def wasserstein_loss_fn(d_outputs_gen, d_outputs_data):
    """Calculates Wasserstein loss for adversarial training.

    D_loss = - E[ D(data) ] + E[ D(gen) ]
    """
    loss_gen = torch.mean(d_outputs_gen) if d_outputs_gen is not None else 0.0
    loss_data = torch.mean(d_outputs_data) if d_outputs_data is not None else 0.0
    w_loss = loss_data - loss_gen  # value to be maximized
    return -w_loss, loss_gen  # value to be minimized


def gradient_norm_sq(
    d_net,
    x_gen,
    x_data,
    y_data=None,
    device=None,
):
    """Computes the penalty term for Lipschitz continuity."""
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
    d_net, x_gen, x_data, y_data, lip=1.0, eps=0.0, device=None, dlog=None
):
    """Computes the regularization term for the critic network.

    This penalizes gradients greater `k` to achieve k-Lipschitz continuity.
    """
    grad_norm_sq = gradient_norm_sq(d_net, x_gen, x_data, y_data=y_data, device=device)
    grad_norm = torch.sqrt(grad_norm_sq.detach())  # only for logging purposes
    grad_penalty = torch.nn.functional.relu(grad_norm_sq + eps - lip * lip).mean()
    # log to dictionary
    if dlog is not None:
        assert isinstance(dlog, dict), type(dlog)
        dlog["grad_norm"] = grad_norm.detach().mean().item()
    return grad_penalty


def gradient_penalty_opt(d_net, x_gen, x_data, y_data, device=None, dlog=None):
    """Computes the regularization term for the critic network.

    This achieves the optimal Kantorovich potential in the Kantorovich–Rubinstein
    duality.
    """
    grad_norm_sq = gradient_norm_sq(d_net, x_gen, x_data, y_data=y_data, device=device)
    grad_norm = torch.sqrt(grad_norm_sq)
    grad_penalty = ((grad_norm - 1.0) ** 2).mean()
    # log to dictionary
    if dlog is not None:
        assert isinstance(dlog, dict), type(dlog)
        dlog["grad_norm"] = grad_norm.detach().mean().item()
    return grad_penalty
