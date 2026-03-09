"""Build learning-rate schedulers with warmup, hold, and cosine decay."""

import torch


def create_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    learning_rate: float,
    linear_epochs: int | None = None,
    constant_epochs: int | None = None,
    init_learning_rate: float | None = None,
    final_learning_rate: float | None = None,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Create a staged learning-rate schedule.

    The scheduler has three stages:
    1. linear ramp from `init_learning_rate` to `learning_rate`
    2. constant `learning_rate`
    3. cosine decay from `learning_rate` to `final_learning_rate`

    Args:
        optimizer: Optimizer to update with scheduled learning rates.
        n_epochs: Total number of training epochs.
        learning_rate: Target learning rate after warmup.
        linear_epochs: Number of warmup epochs for the linear ramp.
        constant_epochs: Number of epochs to keep a constant learning rate.
        init_learning_rate: Starting learning rate at epoch zero.
        final_learning_rate: Minimum learning rate reached by cosine decay.

    Returns:
        A sequential scheduler composed of linear, constant, and cosine stages.

    Raises:
        ValueError: If any scheduler configuration parameter is invalid.
    """
    if n_epochs <= 0:
        raise ValueError(f"Expected n_epochs > 0, got {n_epochs}.")
    if learning_rate <= 0.0:
        raise ValueError(f"Expected learning_rate > 0, got {learning_rate}.")

    # set stage defaults
    if linear_epochs is None:
        linear_epochs = n_epochs // 10
    if constant_epochs is None:
        constant_epochs = n_epochs // 10
    if init_learning_rate is None:
        init_learning_rate = learning_rate / 10.0
    if final_learning_rate is None:
        final_learning_rate = learning_rate / 100.0

    if linear_epochs < 0:
        raise ValueError(f"Expected linear_epochs >= 0, got {linear_epochs}.")
    if constant_epochs < 0:
        raise ValueError(f"Expected constant_epochs >= 0, got {constant_epochs}.")
    if init_learning_rate <= 0.0:
        raise ValueError(f"Expected init_learning_rate > 0, got {init_learning_rate}.")
    if final_learning_rate < 0.0:
        raise ValueError(
            f"Expected final_learning_rate >= 0, got {final_learning_rate}."
        )

    # calculate stage boundaries
    milestone_epochs = [
        linear_epochs,
        linear_epochs + constant_epochs,
    ]
    cosine_epochs = n_epochs - milestone_epochs[-1]
    if cosine_epochs <= 0:
        raise ValueError(
            "Expected n_epochs > linear_epochs + constant_epochs so cosine decay "
            "has at least one epoch."
        )

    # create stage schedulers
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=init_learning_rate / learning_rate,
            end_factor=1.0,
            total_iters=linear_epochs,
        ),
        torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=constant_epochs
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cosine_epochs - 1),
            eta_min=final_learning_rate,
        ),
    ]

    # create and return the combined scheduler
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=milestone_epochs
    )
