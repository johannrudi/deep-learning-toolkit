"""Build learning-rate schedulers with warmup, hold, and cosine decay."""

from dataclasses import InitVar, dataclass, field, fields
from typing import Any

import torch

from dlk.opt.utils import LRSchedulerType


@dataclass
class LinearConstCosineConfig:
    """Typed argument set for `create_linear_const_cosine_scheduler`.

    `linear_epochs`, `constant_epochs`, `init_learning_rate`, and
    `final_learning_rate` each default to an automatic heuristic derived from
    `n_epochs` or `learning_rate` when their configured value is `None`. The
    optimizer is not configured here; it is passed at the call site.

    Attributes:
        n_epochs: Total number of training epochs.
        learning_rate: Target learning rate after warmup.
        param_linear_epochs: Constructor-only input; `None` to derive
            `linear_epochs` from `n_epochs` (not stored on the instance).
        param_constant_epochs: Constructor-only input; `None` to derive
            `constant_epochs` from `n_epochs` (not stored on the instance).
        param_init_learning_rate: Constructor-only input; `None` to derive
            `init_learning_rate` from `learning_rate` (not stored on the
            instance).
        param_final_learning_rate: Constructor-only input; `None` to derive
            `final_learning_rate` from `learning_rate` (not stored on the
            instance).
        linear_epochs: Number of warmup epochs for the linear ramp.
        constant_epochs: Number of epochs to keep a constant learning rate.
        init_learning_rate: Starting learning rate at epoch zero.
        final_learning_rate: Minimum learning rate reached by cosine decay.
    """

    n_epochs: int
    learning_rate: float
    param_linear_epochs: InitVar[int | None] = None
    param_constant_epochs: InitVar[int | None] = None
    param_init_learning_rate: InitVar[float | None] = None
    param_final_learning_rate: InitVar[float | None] = None
    linear_epochs: int = field(init=False)
    constant_epochs: int = field(init=False)
    init_learning_rate: float = field(init=False)
    final_learning_rate: float = field(init=False)

    def __post_init__(
        self,
        param_linear_epochs: int | None,
        param_constant_epochs: int | None,
        param_init_learning_rate: float | None,
        param_final_learning_rate: float | None,
    ) -> None:
        if self.n_epochs <= 0:
            raise ValueError(f"expected n_epochs > 0, got {self.n_epochs}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"expected learning_rate > 0, got {self.learning_rate}.")

        self.linear_epochs = (
            param_linear_epochs
            if param_linear_epochs is not None
            else self.auto_stage_epochs(self.n_epochs)
        )
        self.constant_epochs = (
            param_constant_epochs
            if param_constant_epochs is not None
            else self.auto_stage_epochs(self.n_epochs)
        )
        self.init_learning_rate = (
            param_init_learning_rate
            if param_init_learning_rate is not None
            else self.auto_init_learning_rate(self.learning_rate)
        )
        self.final_learning_rate = (
            param_final_learning_rate
            if param_final_learning_rate is not None
            else self.auto_final_learning_rate(self.learning_rate)
        )

        if self.linear_epochs < 0:
            raise ValueError(f"expected linear_epochs >= 0, got {self.linear_epochs}.")
        if self.constant_epochs < 0:
            raise ValueError(
                f"expected constant_epochs >= 0, got {self.constant_epochs}."
            )
        if self.init_learning_rate <= 0.0:
            raise ValueError(
                f"expected init_learning_rate > 0, got {self.init_learning_rate}."
            )
        if self.final_learning_rate < 0.0:
            raise ValueError(
                f"expected final_learning_rate >= 0, got {self.final_learning_rate}."
            )
        if self.cosine_epochs <= 0:
            raise ValueError(
                "expected n_epochs > linear_epochs + constant_epochs so cosine decay "
                "has at least one epoch."
            )

    @staticmethod
    def auto_stage_epochs(n_epochs: int, epochs_divisor: int = 10) -> int:
        """Heuristic length of the linear and constant stages."""
        return n_epochs // epochs_divisor

    @staticmethod
    def auto_init_learning_rate(
        learning_rate: float, learning_rate_divisor: float = 10.0
    ) -> float:
        """Heuristic starting learning rate of the linear ramp."""
        return learning_rate / learning_rate_divisor

    @staticmethod
    def auto_final_learning_rate(
        learning_rate: float, learning_rate_divisor: float = 100.0
    ) -> float:
        """Heuristic final learning rate of the cosine decay."""
        return learning_rate / learning_rate_divisor

    @property
    def milestone_epochs(self) -> list[int]:
        """Epochs at which the schedule moves to its next stage."""
        return [self.linear_epochs, self.linear_epochs + self.constant_epochs]

    @property
    def cosine_epochs(self) -> int:
        """Number of epochs left for the cosine decay stage."""
        return self.n_epochs - self.milestone_epochs[-1]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LinearConstCosineConfig":
        """Build a `LinearConstCosineConfig` from a dict of constructor arguments.

        Args:
            config: Mapping of constructor argument names to values.
                `linear_epochs`, `constant_epochs`, `init_learning_rate`, and
                `final_learning_rate` map to their `param_*` constructor inputs;
                every other key must match a field name directly.

        Returns:
            A configured `LinearConstCosineConfig` instance.

        Raises:
            TypeError: If `config` contains a key that is not a constructor
                argument, or omits a required one.
        """
        key_aliases = {
            "linear_epochs": "param_linear_epochs",
            "constant_epochs": "param_constant_epochs",
            "init_learning_rate": "param_init_learning_rate",
            "final_learning_rate": "param_final_learning_rate",
        }
        kwargs = {key_aliases.get(key, key): value for key, value in config.items()}
        return cls(**kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        """Return this config as kwargs for `create_linear_const_cosine_scheduler`."""
        return {field_.name: getattr(self, field_.name) for field_ in fields(self)}


def create_linear_const_cosine_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: LinearConstCosineConfig,
) -> LRSchedulerType:
    """Create a staged learning-rate schedule from a config.

    The scheduler has three stages:
    1. linear ramp from `init_learning_rate` to `learning_rate`
    2. constant `learning_rate`
    3. cosine decay from `learning_rate` to `final_learning_rate`

    Args:
        optimizer: Optimizer to update with scheduled learning rates.
        config: Resolved stage lengths and learning rates.

    Returns:
        A sequential scheduler composed of linear, constant, and cosine stages.
    """
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.init_learning_rate / config.learning_rate,
            end_factor=1.0,
            total_iters=config.linear_epochs,
        ),
        torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=config.constant_epochs
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.cosine_epochs - 1),
            eta_min=config.final_learning_rate,
        ),
    ]

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=config.milestone_epochs
    )


def create_linear_const_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    learning_rate: float,
    linear_epochs: int | None = None,
    constant_epochs: int | None = None,
    init_learning_rate: float | None = None,
    final_learning_rate: float | None = None,
) -> LRSchedulerType:
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
    config = LinearConstCosineConfig(
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        param_linear_epochs=linear_epochs,
        param_constant_epochs=constant_epochs,
        param_init_learning_rate=init_learning_rate,
        param_final_learning_rate=final_learning_rate,
    )
    return create_linear_const_cosine_scheduler_from_config(optimizer, config)


def create_learning_rate_scheduler_from_params(
    optimizer: torch.optim.Optimizer,
    opt_params: dict[str, Any],
    n_epochs: int,
) -> LRSchedulerType | None:
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
        config = LinearConstCosineConfig(
            n_epochs=n_epochs,
            learning_rate=opt_params["learning_rate"],
            param_linear_epochs=scheduler_params.get("linear_epochs"),
            param_constant_epochs=scheduler_params.get("constant_epochs"),
            param_init_learning_rate=scheduler_params.get("init_learning_rate"),
            param_final_learning_rate=scheduler_params.get("final_learning_rate"),
        )
        return create_linear_const_cosine_scheduler_from_config(optimizer, config)

    if scheduler_type == "step":
        step_size = scheduler_params.get("step_size", n_epochs // 3)
        gamma = scheduler_params.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    raise ValueError(f"unknown scheduler type: {repr(scheduler_type)}")
