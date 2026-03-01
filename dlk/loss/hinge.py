import torch


def hinge_loss_fn(d_outputs_gen, d_outputs_data):
    """Calculates the Hinge loss for adversarial training.

    D_loss = E[ max(0, 1 - D(data)) ] + E[ max(0, 1 + D(gen)) ]
    G_loss = -E[ D(gen) ]
    """
    if d_outputs_gen is not None and d_outputs_data is not None:
        loss_gen = torch.mean(torch.nn.functional.relu(1.0 + d_outputs_gen))
        loss_data = torch.mean(torch.nn.functional.relu(1.0 - d_outputs_data))
        hinge_loss = loss_data + loss_gen
        return hinge_loss, loss_gen
    else:
        assert (
            d_outputs_data is not None
        )  # assume flipped gen <-> data for generator use
        hinge_loss = -torch.mean(d_outputs_data)
        return hinge_loss, None
