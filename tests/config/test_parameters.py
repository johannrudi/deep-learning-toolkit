import pathlib

import pytest

from dlk.config.parameters import (load, save, update_from_json,
                                   update_from_toml)


# TODO: add typing annotation here; and for all arguments "example_params" below
@pytest.fixture
def example_params():
    return {
        "train": {
            "lr": 1e-3,
            "optimizer": {
                "name": "Adam",
                "weight_decay": 0.0,
            },
        },
        "seed": 7,
    }


def test_save_with_default_format(
    tmp_path: pathlib.Path,
    example_params,
) -> None:
    """Save parameters when no format is provided."""
    save(params=example_params, save_dir=tmp_path)

    saved_path = tmp_path / "params.json"
    assert saved_path.exists()
    assert load(saved_path) == example_params


@pytest.mark.parametrize(
    ("parameter_format", "expected_filename"),
    [
        ("toml", "params.toml"),
        ("json", "params.json"),
        ("yaml", "params.yaml"),
        ("yml", "params.yaml"),
    ],
)
def test_save_with_requested_format(
    parameter_format: str,
    expected_filename: str,
    tmp_path: pathlib.Path,
    example_params,
) -> None:
    """Write parameters in the requested format and keep load compatibility."""
    save(
        params=example_params,
        save_dir=tmp_path,
        parameter_format=parameter_format,
    )

    saved_path = tmp_path / expected_filename
    assert saved_path.exists()
    assert load(saved_path) == example_params


def test_save_raises_for_unsupported_format(
    tmp_path: pathlib.Path,
) -> None:
    """Raise ValueError when an unsupported format is requested."""
    params = {"seed": 7}

    with pytest.raises(ValueError, match="Unsupported parameter format"):
        save(
            params=params,
            save_dir=tmp_path,
            parameter_format="ini",
        )


def test_save_raises_for_none_values_in_toml(
    tmp_path: pathlib.Path,
) -> None:
    """Raise ValueError when TOML output includes null values."""
    params = {"seed": None}

    with pytest.raises(ValueError, match="TOML does not support null values"):
        save(
            params=params,
            save_dir=tmp_path,
            parameter_format="toml",
        )


def test_update_parameters_from_json_updates_nested_keys_and_warns_for_unknown(
    capsys: pytest.CaptureFixture[str],
    example_params,
) -> None:
    """Update nested parameters in place and warn on unknown key paths."""
    json_params = """\
{
    "train": {
        "lr": 1e-2,
        "optimizer": {
            "weight_decay": 0.1,
            "beta1": 0.9
        },
        "missing": true
    },
    "seed": 11,
    "ghost": 1
}"""
    params = example_params.copy()
    update_from_json(params, json_params)

    # TODO: check if two dictionaries can be compared with "=="
    assert params == {
        "train": {
            "lr": 1e-2,
            "optimizer": {
                "name": "Adam",
                "weight_decay": 0.1,
            },
        },
        "seed": 11,
    }
    captured = capsys.readouterr()
    assert "Warning: parameter 'train.optimizer.beta1' does not exist" in captured.out
    assert "Warning: parameter 'train.missing' does not exist" in captured.out
    assert "Warning: parameter 'ghost' does not exist" in captured.out


def test_update_parameters_from_json_raises_for_invalid_json() -> None:
    """Raise ValueError when the input string is not valid JSON."""
    params = {"seed": 7}

    with pytest.raises(ValueError, match="json_params is not valid JSON."):
        update_from_json(params, "{invalid-json")


def test_update_parameters_from_json_raises_for_non_dict_json() -> None:
    """Raise ValueError when JSON does not decode to a dictionary."""
    params = {"seed": 7}

    with pytest.raises(ValueError, match="json_params must decode to a dictionary."):
        update_from_json(params, "[1, 2, 3]")


def test_update_parameters_from_toml_updates_nested_keys_and_warns_for_unknown(
    capsys: pytest.CaptureFixture[str],
    example_params,
) -> None:
    """Update nested parameters from TOML and warn on unknown key paths."""
    toml_params = """
seed = 11
ghost = 1

[train]
lr = 0.01
missing = true

[train.optimizer]
weight_decay = 0.1
beta1 = 0.9
"""
    params = example_params.copy()
    update_from_toml(params, toml_params)

    assert params == {
        "train": {
            "lr": 0.01,
            "optimizer": {
                "name": "Adam",
                "weight_decay": 0.1,
            },
        },
        "seed": 11,
    }
    captured = capsys.readouterr()
    assert "Warning: parameter 'train.optimizer.beta1' does not exist" in captured.out
    assert "Warning: parameter 'train.missing' does not exist" in captured.out
    assert "Warning: parameter 'ghost' does not exist" in captured.out


def test_update_parameters_from_toml_raises_for_invalid_toml() -> None:
    """Raise ValueError when the input string is not valid TOML."""
    params = {"seed": 7}

    with pytest.raises(ValueError, match="toml_params is not valid TOML."):
        update_from_toml(params, "seed = ")
