"""Unit tests for the optimizer configs in `dlk.opt.optimizer`."""

import dataclasses
import inspect
from typing import Any

import pytest
import torch

from dlk.opt.optimizer import (
    AdamConfig,
    AdamWConfig,
    SGDConfig,
    create_optimizer_from_params,
)

CONFIG_CLASSES: list[tuple[Any, Any]] = [
    (AdamConfig, torch.optim.Adam),
    (AdamWConfig, torch.optim.AdamW),
    (SGDConfig, torch.optim.SGD),
]


def _make(config_class: Any, **kwargs: Any) -> Any:
    """Build a config, supplying the required fields of each class."""
    if config_class is SGDConfig:
        return config_class(lr=1e-3, momentum=0.9, **kwargs)
    return config_class(lr=1e-3, **kwargs)


@pytest.mark.parametrize(("config_class", "optimizer_class"), CONFIG_CLASSES)
def test_config_fields_are_valid_optimizer_arguments(
    config_class: Any, optimizer_class: Any
) -> None:
    """Every stored field of a config must be a parameter of its optimizer."""
    optimizer_parameters = inspect.signature(optimizer_class.__init__).parameters

    config_fields = {field.name for field in dataclasses.fields(config_class)}

    assert config_fields <= optimizer_parameters.keys()


@pytest.mark.parametrize(("config_class", "optimizer_class"), CONFIG_CLASSES)
@pytest.mark.parametrize("accelerator_available", [True, False])
def test_fused_follows_accelerator_availability(
    config_class: Any,
    optimizer_class: Any,
    accelerator_available: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`fused` defaults to whether an accelerator is available."""
    monkeypatch.setattr(
        torch.accelerator, "is_available", lambda: accelerator_available
    )

    config = _make(config_class)

    assert config.fused is accelerator_available


@pytest.mark.parametrize(("config_class", "optimizer_class"), CONFIG_CLASSES)
def test_foreach_is_disabled_while_fused_is_enabled(
    config_class: Any, optimizer_class: Any
) -> None:
    """The two implementations are mutually exclusive, so `fused` wins."""
    config = _make(config_class, param_fused=True)

    assert config.fused is True
    assert config.foreach is False


@pytest.mark.parametrize(("config_class", "optimizer_class"), CONFIG_CLASSES)
def test_enabling_both_implementations_raises(
    config_class: Any, optimizer_class: Any
) -> None:
    """Asking for both `fused` and `foreach` is a configuration error."""
    with pytest.raises(ValueError, match="at most one of fused and foreach"):
        _make(config_class, param_fused=True, param_foreach=True)


@pytest.mark.parametrize(("config_class", "optimizer_class"), CONFIG_CLASSES)
def test_to_kwargs_round_trips_through_from_dict(
    config_class: Any, optimizer_class: Any
) -> None:
    """`from_dict` accepts what `to_kwargs` produces, unchanged."""
    config = _make(config_class, param_fused=False, param_foreach=True)

    assert config_class.from_dict(config.to_kwargs()) == config


def test_create_optimizer_from_params_accepts_short_keys() -> None:
    """`lr` and `eps` are accepted as aliases for `learning_rate` and `epsilon`."""
    net = torch.nn.Linear(1, 1)

    long_optimizer = create_optimizer_from_params(
        net, {"type": "adam", "learning_rate": 1e-3, "epsilon": 1e-6}
    )
    short_optimizer = create_optimizer_from_params(
        net, {"type": "adam", "lr": 1e-3, "eps": 1e-6}
    )

    assert long_optimizer.defaults["lr"] == short_optimizer.defaults["lr"] == 1e-3
    assert long_optimizer.defaults["eps"] == short_optimizer.defaults["eps"] == 1e-6
