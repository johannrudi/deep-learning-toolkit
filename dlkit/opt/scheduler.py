"""Schedulers for learning rates.
"""
import torch

def create_learning_rate_scheduler(optimizer, n_epochs, learning_rate,
                                   linear_epochs=None, constant_epochs=None,
                                   init_learning_rate=None, final_learning_rate=None):
    """
    Creates a schedule for learning rate with these stages:

    1. `init_learning_rate .. learning_rate` for #epochs = `0 .. linear_epochs`
    2. `learning_rate` for #epochs = `linear_epochs .. constant_epochs`
    3. `learning_rate .. final_learning_rate` for #epochs = `constant_epochs .. n_epochs`
    """
    # set up defaults
    if linear_epochs is None:
        linear_epochs   = n_epochs // 10
    if constant_epochs is None:
        constant_epochs = n_epochs // 10
    if init_learning_rate is None:
        init_learning_rate  = learning_rate / 10.0
    if final_learning_rate is None:
        final_learning_rate = learning_rate / 100.0
    # set epoch counts for different stages in scheduling (list of length #stages - 1)
    milestone_epochs = [
        linear_epochs,
        linear_epochs + constant_epochs,
    ]
    # set list of schedules (list of length #stages)
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor = init_learning_rate/learning_rate,
                end_factor   = 1.0,
                total_iters  = linear_epochs
        ),
        torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor       = 1.0,
                total_iters  = constant_epochs
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max        = n_epochs - milestone_epochs[-1] - 1,
                eta_min      = final_learning_rate
        ),
    ]
    # create and return a sequential scheduler
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers = schedulers,
            milestones = milestone_epochs
    )
    return lr_scheduler
