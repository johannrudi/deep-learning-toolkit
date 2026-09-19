"""Build optimizers from typed argument sets."""

from dataclasses import InitVar, dataclass, field, fields
from typing import Any, ClassVar

import torch
import torch.nn as nn


class _OptimizerConfigMixin:
    """Shared heuristics and conversions for the optimizer config dataclasses.

    The dataclasses below each mirror one `torch.optim` class, so they share no
    fields. What they share is how `fused` and `foreach` are resolved and how a
    config is converted to and from plain data.
    """

    # declare what every subclass supplies, so the type checker accepts `fields`
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @staticmethod
    def auto_fused() -> bool:
        """Heuristic `fused` used when it is not configured.

        `fused` requires every parameter to be a floating point tensor on the
        accelerator, which a config cannot verify. Build the optimizer after
        moving the model to its device, or pass `param_fused=False` to opt out.

        Returns:
            Whether an accelerator is available.
        """
        return torch.accelerator.is_available()

    @staticmethod
    def auto_foreach() -> bool:
        """Heuristic `foreach` used when it is not configured.

        Only reached when `fused` is disabled, because the two are mutually
        exclusive.

        Returns:
            Whether an accelerator is available.
        """
        return torch.accelerator.is_available()

    def _resolve_fused_foreach(
        self, param_fused: bool | None, param_foreach: bool | None
    ) -> tuple[bool, bool]:
        """Resolve the `fused` and `foreach` pair.

        Args:
            param_fused: Configured `fused`, or `None` to auto-detect.
            param_foreach: Configured `foreach`, or `None` to auto-detect.

        Returns:
            The resolved `fused` and `foreach`.

        Raises:
            ValueError: If both are explicitly enabled.
        """
        fused = param_fused if param_fused is not None else self.auto_fused()
        if fused:
            if param_foreach:
                raise ValueError(
                    "expected at most one of fused and foreach to be enabled, got both."
                )
            return fused, False
        foreach = param_foreach if param_foreach is not None else self.auto_foreach()
        return fused, foreach

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Any:
        """Build a config from a dict of constructor arguments.

        Args:
            config: Mapping of constructor argument names to values. `fused` and
                `foreach` map to their `param_*` constructor inputs; every other
                key must match a field name directly.

        Returns:
            A configured instance of the calling class.

        Raises:
            TypeError: If `config` contains a key that is not a constructor
                argument, or omits a required one.
        """
        key_aliases = {"fused": "param_fused", "foreach": "param_foreach"}
        kwargs = {key_aliases.get(key, key): value for key, value in config.items()}
        return cls(**kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        """Return this config as constructor kwargs for its optimizer."""
        return {field_.name: getattr(self, field_.name) for field_ in fields(self)}


@dataclass
class AdamConfig(_OptimizerConfigMixin):
    """Typed argument set for `torch.optim.Adam`.

    `fused` and `foreach` each default to an automatic heuristic when their
    configured value is `None`. `params` is not configured here; it is passed at
    the call site.

    Attributes:
        lr: Learning rate.
        param_fused: Constructor-only input; `None` to auto-detect `fused` from
            accelerator availability (not stored on the instance).
        param_foreach: Constructor-only input; `None` to auto-detect `foreach`
            from accelerator availability (not stored on the instance).
        fused: Whether to use the fused implementation.
        foreach: Whether to use the multi-tensor implementation.
        betas: Coefficients for the running averages of gradient and its square.
        eps: Term added to the denominator for numerical stability.
        weight_decay: L2 penalty.
        amsgrad: Whether to use the AMSGrad variant.
        maximize: Whether to maximize the objective instead of minimizing it.
        capturable: Whether this instance may be captured in a CUDA graph.
        differentiable: Whether autograd may run through the optimizer step.
        decoupled_weight_decay: Whether to decouple weight decay, as in AdamW.
    """

    lr: float
    param_fused: InitVar[bool | None] = None
    param_foreach: InitVar[bool | None] = None
    fused: bool = field(init=False, default=False)
    foreach: bool = field(init=False, default=False)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    decoupled_weight_decay: bool = False

    def __post_init__(
        self, param_fused: bool | None, param_foreach: bool | None
    ) -> None:
        self.fused, self.foreach = self._resolve_fused_foreach(
            param_fused, param_foreach
        )


@dataclass
class AdamWConfig(_OptimizerConfigMixin):
    """Typed argument set for `torch.optim.AdamW`.

    `fused` and `foreach` each default to an automatic heuristic when their
    configured value is `None`. `params` is not configured here; it is passed at
    the call site.

    Attributes:
        lr: Learning rate.
        param_fused: Constructor-only input; `None` to auto-detect `fused` from
            accelerator availability (not stored on the instance).
        param_foreach: Constructor-only input; `None` to auto-detect `foreach`
            from accelerator availability (not stored on the instance).
        fused: Whether to use the fused implementation.
        foreach: Whether to use the multi-tensor implementation.
        betas: Coefficients for the running averages of gradient and its square.
        eps: Term added to the denominator for numerical stability.
        weight_decay: Decoupled weight decay coefficient.
        amsgrad: Whether to use the AMSGrad variant.
        maximize: Whether to maximize the objective instead of minimizing it.
        capturable: Whether this instance may be captured in a CUDA graph.
        differentiable: Whether autograd may run through the optimizer step.
    """

    lr: float
    param_fused: InitVar[bool | None] = None
    param_foreach: InitVar[bool | None] = None
    fused: bool = field(init=False, default=False)
    foreach: bool = field(init=False, default=False)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False

    def __post_init__(
        self, param_fused: bool | None, param_foreach: bool | None
    ) -> None:
        self.fused, self.foreach = self._resolve_fused_foreach(
            param_fused, param_foreach
        )


@dataclass
class SGDConfig(_OptimizerConfigMixin):
    """Typed argument set for `torch.optim.SGD`.

    `fused` and `foreach` each default to an automatic heuristic when their
    configured value is `None`. `momentum` is required, because inheriting the
    momentum-free default by accident is a common training mistake. `params` is
    not configured here; it is passed at the call site.

    Attributes:
        lr: Learning rate.
        momentum: Momentum factor; 0 for plain gradient descent.
        param_fused: Constructor-only input; `None` to auto-detect `fused` from
            accelerator availability (not stored on the instance).
        param_foreach: Constructor-only input; `None` to auto-detect `foreach`
            from accelerator availability (not stored on the instance).
        fused: Whether to use the fused implementation.
        foreach: Whether to use the multi-tensor implementation.
        dampening: Dampening for momentum.
        weight_decay: L2 penalty.
        nesterov: Whether to use Nesterov momentum.
        maximize: Whether to maximize the objective instead of minimizing it.
        differentiable: Whether autograd may run through the optimizer step.
    """

    lr: float
    momentum: float
    param_fused: InitVar[bool | None] = None
    param_foreach: InitVar[bool | None] = None
    fused: bool = field(init=False, default=False)
    foreach: bool = field(init=False, default=False)
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    maximize: bool = False
    differentiable: bool = False

    def __post_init__(
        self, param_fused: bool | None, param_foreach: bool | None
    ) -> None:
        self.fused, self.foreach = self._resolve_fused_foreach(
            param_fused, param_foreach
        )


def create_optimizer_from_params(
    net: nn.Module,
    opt_params: dict[str, Any],
) -> torch.optim.Optimizer:
    """Create an Adam, AdamW, or SGD optimizer from a parameter dict.

    Translates the dict keys used by training scripts (`learning_rate`, `beta1`,
    `beta2`, `epsilon`) into one of the config dataclasses above, which resolve
    `fused` and `foreach` and supply the remaining defaults. `learning_rate` and
    `epsilon` may instead be given as `lr` and `eps`, the shorter names
    `torch.optim` constructors use.

    Args:
        net: Network whose parameters are optimized.
        opt_params: Mapping with a `type` key naming the optimizer and a
            `learning_rate` (or `lr`) key, plus optional per-optimizer keys.
            `epsilon` may be given as `eps`.

    Returns:
        The configured optimizer.

    Raises:
        ValueError: If `type` names an optimizer that is not supported.
    """
    key_aliases = {"lr": "learning_rate", "eps": "epsilon"}
    opt_params = {key_aliases.get(key, key): value for key, value in opt_params.items()}

    opt_type = opt_params["type"].casefold()
    lr = opt_params["learning_rate"]
    betas = (
        opt_params.get("beta1", 0.9),
        opt_params.get("beta2", 0.999),
    )
    eps = opt_params.get("epsilon", 1e-8)

    if opt_type == "adam":
        adam_config = AdamConfig(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=opt_params.get("weight_decay", 0.0),
        )
        return torch.optim.Adam(net.parameters(), **adam_config.to_kwargs())

    if opt_type == "adamw":
        adamw_config = AdamWConfig(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=opt_params.get("weight_decay", 1e-2),
        )
        return torch.optim.AdamW(net.parameters(), **adamw_config.to_kwargs())

    if opt_type == "sgd":
        sgd_config = SGDConfig(
            lr=lr,
            momentum=opt_params.get("momentum", 0.0),
            weight_decay=opt_params.get("weight_decay", 0.0),
            nesterov=opt_params.get("nesterov", False),
        )
        return torch.optim.SGD(net.parameters(), **sgd_config.to_kwargs())

    raise ValueError(f"unknown optimizer type: {opt_type}")
