import torch


def least_squares_loss_fn(d_outputs_gen, d_outputs_data, gen_label=0.0, data_label=1.0):
    """Calculates the Hinge loss for adversarial training.

    D_loss = 1/2 * E[(D(x) - a)^2] + 1/2 * E[(D(G(z)) - b)^2]
    G_loss = 1/2 * E[(D(G(z)) - a)^2]
    """
    device = d_outputs_data.device
    if d_outputs_data is not None:
        loss_data = 0.5 * torch.nn.functional.mse_loss(
            d_outputs_data, torch.full_like(d_outputs_data, data_label, device=device)
        )
    else:
        loss_data = 0.0
    if d_outputs_gen is not None:
        loss_gen = 0.5 * torch.nn.functional.mse_loss(
            d_outputs_gen, torch.full_like(d_outputs_gen, gen_label, device=device)
        )
    else:
        loss_gen = 0.0
    ls_loss = loss_data + loss_gen
    return ls_loss, loss_gen
