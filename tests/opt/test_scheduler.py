"""Unit tests for the scheduler config in `dlk.opt.scheduler`."""

import dataclasses
import inspect

import pytest

from dlk.opt.scheduler import (
    LinearConstCosineConfig,
    create_linear_const_cosine_scheduler,
)


def test_config_fields_are_valid_scheduler_arguments() -> None:
    """Every stored field must be a parameter of the scheduler builder."""
    scheduler_parameters = inspect.signature(
        create_linear_const_cosine_scheduler
    ).parameters

    config_fields = {
        field.name for field in dataclasses.fields(LinearConstCosineConfig)
    }

    assert config_fields <= scheduler_parameters.keys()


def test_stages_derive_from_n_epochs_and_learning_rate() -> None:
    """Unconfigured stage lengths and learning rates come from the heuristics."""
    config = LinearConstCosineConfig(n_epochs=100, learning_rate=1e-3)

    assert config.linear_epochs == 10
    assert config.constant_epochs == 10
    assert config.init_learning_rate == 1e-4
    assert config.final_learning_rate == 1e-5
    assert config.milestone_epochs == [10, 20]
    assert config.cosine_epochs == 80


def test_configured_stages_override_the_heuristics() -> None:
    """A configured stage length is kept and the rest stay derived."""
    config = LinearConstCosineConfig(
        n_epochs=100, learning_rate=1e-3, param_linear_epochs=25
    )

    assert config.linear_epochs == 25
    assert config.constant_epochs == 10
    assert config.cosine_epochs == 65


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_epochs": 0, "learning_rate": 1e-3}, "expected n_epochs > 0"),
        ({"n_epochs": 10, "learning_rate": 0.0}, "expected learning_rate > 0"),
        (
            {"n_epochs": 10, "learning_rate": 1e-3, "param_linear_epochs": -1},
            "expected linear_epochs >= 0",
        ),
        (
            {"n_epochs": 10, "learning_rate": 1e-3, "param_constant_epochs": 20},
            "cosine decay",
        ),
    ],
)
def test_invalid_configurations_raise(kwargs: dict, message: str) -> None:
    """Every stage constraint is checked while the config is built."""
    with pytest.raises(ValueError, match=message):
        LinearConstCosineConfig(**kwargs)


def test_to_kwargs_round_trips_through_from_dict() -> None:
    """`from_dict` accepts what `to_kwargs` produces, unchanged."""
    config = LinearConstCosineConfig(
        n_epochs=50, learning_rate=2e-3, param_final_learning_rate=0.0
    )

    assert LinearConstCosineConfig.from_dict(config.to_kwargs()) == config
