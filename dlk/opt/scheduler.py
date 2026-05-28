"""Build learning-rate schedulers with warmup, hold, and cosine decay."""

import torch


def create_linear_const_cosine_scheduler(
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


def create_learning_rate_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    opt_params: dict,
    n_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create a learning rate scheduler, or return None if not configured.

    Supports:
      - linear_cosine:       CosineAnnealingLR with optional warm-up
      - linear_const_cosine: Linear then constant then CosineAnnealingLR
      - step:                StepLR

    If opt_params has no "learning_rate_scheduler" key, returns None.
    """
    scheduler_params = opt_params.get("learning_rate_scheduler")
    if scheduler_params is None:
        return None

    scheduler_type = scheduler_params.get("type", "linear_cosine").casefold()

    if scheduler_type == "linear_cosine":
        warmup_epochs = scheduler_params.get("warmup_epochs", 0)
        if 0 < warmup_epochs:
            # start with linear warm-up then cosine decay
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
                return 0.5 * (
                    1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item()
                )

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    if scheduler_type == "linear_const_cosine":
        return create_linear_const_cosine_scheduler(
            optimizer,
            n_epochs,
            opt_params["learning_rate"],
            linear_epochs=scheduler_params.get("linear_epochs"),
            constant_epochs=scheduler_params.get("constant_epochs"),
            init_learning_rate=scheduler_params.get("init_learning_rate"),
            final_learning_rate=scheduler_params.get("final_learning_rate"),
        )

    if scheduler_type == "step":
        step_size = scheduler_params.get("step_size", n_epochs // 3)
        gamma = scheduler_params.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    raise ValueError(f"unknown scheduler type: {repr(scheduler_type)}")
