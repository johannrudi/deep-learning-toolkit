import argparse
import json
import pathlib
import re
import tomllib as toml
from typing import Any

import yaml

_SUPPORTED_PARAMETER_FORMATS = {"json", "toml", "yaml", "yml"}
_TOML_BARE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _normalize_parameter_format(parameter_format: str) -> str:
    """Normalize and validate a parameter serialization format.

    Args:
        parameter_format (str): Requested output format name.

    Returns:
        str: Normalized format name.

    Raises:
        ValueError: If ``parameter_format`` is unsupported.
    """
    normalized_format = parameter_format.strip().lower()
    if normalized_format not in _SUPPORTED_PARAMETER_FORMATS:
        raise ValueError(
            f"Unsupported parameter format `{parameter_format}`."
            + " Supported formats are: json, toml, yaml."
        )
    return "yaml" if normalized_format == "yml" else normalized_format


def _default_parameter_filename(parameter_format: str) -> str:
    """Build the default output filename for a serialization format.

    Args:
        parameter_format (str): Normalized output format.

    Returns:
        str: Default filename for the requested format.
    """
    return f"params.{parameter_format}"


def _format_toml_key(key: str) -> str:
    """Format a key for TOML serialization.

    Args:
        key (str): Key to encode in TOML.

    Returns:
        str: TOML-compatible key representation.

    Raises:
        ValueError: If ``key`` is not a non-empty string.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("TOML keys must be non-empty strings.")
    if _TOML_BARE_KEY_PATTERN.match(key):
        return key
    return json.dumps(key)


def _serialize_toml_value(value: Any) -> str:
    """Serialize a python value into TOML value syntax.

    Args:
        value (Any): Value to serialize.

    Returns:
        str: TOML-compatible value string.

    Raises:
        ValueError: If ``value`` has a type unsupported for TOML output.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        raise ValueError(
            "TOML does not support null values. Use json/yaml for parameters"
            + " containing None."
        )
    if isinstance(value, list):
        serialized_values = [_serialize_toml_value(item) for item in value]
        return "[" + ", ".join(serialized_values) + "]"
    if isinstance(value, dict):
        inline_fields = [
            f"{_format_toml_key(k)} = {_serialize_toml_value(v)}"
            for k, v in value.items()
        ]
        return "{ " + ", ".join(inline_fields) + " }"
    raise ValueError(
        f"Unsupported type `{type(value).__name__}` for TOML serialization."
    )


def _serialize_toml_dict(
    params: dict[str, Any],
    table_key_parts: list[str] | None = None,
) -> list[str]:
    """Serialize a nested dictionary into TOML lines.

    Args:
        params (dict[str, Any]): Dictionary to serialize.
        table_key_parts (list[str] | None, optional): Dot path of the current
            table. Defaults to ``None``.

    Returns:
        list[str]: TOML lines for ``params``.
    """
    table_key_parts = table_key_parts or []
    lines: list[str] = []
    nested_tables: list[tuple[str, dict[str, Any]]] = []

    # write non-table fields first
    for key, value in params.items():
        formatted_key = _format_toml_key(key)
        if isinstance(value, dict):
            nested_tables.append((formatted_key, value))
            continue
        lines.append(f"{formatted_key} = {_serialize_toml_value(value)}")

    # write nested tables after scalar fields
    for i, (formatted_key, nested_value) in enumerate(nested_tables):
        if lines or i > 0:
            lines.append("")
        nested_table_key_parts = [*table_key_parts, formatted_key]
        lines.append(f"[{'.'.join(nested_table_key_parts)}]")
        lines.extend(
            _serialize_toml_dict(
                params=nested_value,
                table_key_parts=nested_table_key_parts,
            )
        )
    return lines


def _serialize_toml(params: dict[str, Any]) -> str:
    """Serialize a parameter dictionary to TOML text.

    Args:
        params (dict[str, Any]): Parameters to serialize.

    Returns:
        str: TOML document string.
    """
    lines = _serialize_toml_dict(params)
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def load(filepath: str | pathlib.Path) -> dict[str, Any]:
    """Load parameters from a JSON, TOML, or YAML file.

    Args:
        filepath (str | pathlib.Path): Path to the parameter file.

    Returns:
        dict: Parameters parsed from the file.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    path = pathlib.Path(filepath)
    suffix = path.suffix.lower()

    # parse parameters by file extension
    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".json":
            params = json.load(f)
        elif suffix == ".toml":
            params = toml.loads(f.read())
        elif suffix in (".yaml", ".yml"):
            params = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported parameter file extension `{suffix}` for `{filepath}`."
            )
    return params if params is not None else {}


def save(
    params: dict[str, Any],
    save_dir: str | pathlib.Path,
    filename: str | None = None,
    parameter_format: str = "json",
) -> None:
    """Save parameters in JSON, TOML, or YAML format.

    Args:
        params (dict): Parameters to serialize.
        save_dir (str | pathlib.Path): Directory where the file is written.
        filename (str | None, optional): Output file name. If omitted, the
            default name is derived from ``parameter_format``.
        parameter_format (str, optional): Serialization format. Supported
            values are ``"json"``, ``"toml"``, ``"yaml"``, and ``"yml"``.
            Defaults to ``"json"``.

    Raises:
        ValueError: If ``save_dir`` is not provided or points to an invalid path.
        ValueError: If ``parameter_format`` is unsupported.
    """
    if not save_dir:
        raise ValueError(
            "save_dir is not provided. For saving params, a user-defined"
            + " save_dir must be passed through the yaml file"
        )
    # validate parameter format and assign output filename
    normalized_format = _normalize_parameter_format(parameter_format)
    output_filename = filename or _default_parameter_filename(normalized_format)

    # set path and create subdirectories
    path = pathlib.Path(save_dir) / output_filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(f"Invalid path {save_dir}") from error
    # save parameters in the requested format
    with open(path, "w+", encoding="utf-8") as f:
        if normalized_format == "json":
            json.dump(params, f, indent=2)
            f.write("\n")
        elif normalized_format == "toml":
            f.write(_serialize_toml(params))
        else:
            yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)


def _update_existing_nested_parameters(
    params: dict[str, Any],
    updates: dict[str, Any],
    key_prefix: str = "",
) -> None:
    """Recursively update only existing keys in a nested parameter dictionary.

    Args:
        params (dict[str, Any]): Mutable parameter dictionary to update in place.
        updates (dict[str, Any]): Nested dictionary with proposed updates.
        key_prefix (str, optional): Dot-delimited key path used for warnings.
            Defaults to ``""``.

    Returns:
        None: Updates ``params`` in place.
    """
    for key, value in updates.items():
        current_key_path = f"{key_prefix}.{key}" if key_prefix else key
        if key not in params:
            print(
                f"Warning: parameter {repr(current_key_path)} does not exist and is ignored"
            )
            continue

        current_value = params[key]
        if isinstance(current_value, dict) and not isinstance(value, dict):
            print(f"Warning: expected dict, but got value {repr(value)} and is ignored")
            continue

        if isinstance(current_value, dict) and isinstance(value, dict):
            # recurse into nested dictionaries when both values are dictionaries
            _update_existing_nested_parameters(
                params=current_value,
                updates=value,
                key_prefix=current_key_path,
            )
            continue
        params[key] = value


def update_from_json(
    params: dict[str, Any],
    json_params: str,
) -> None:
    """Update existing parameters from a JSON string.

    Args:
        json_params (str): JSON string with parameter updates.
        params (dict[str, Any]): Mutable parameter dictionary to update in place.

    Returns:
        None: Updates ``params`` in place.

    Raises:
        ValueError: If ``json_params`` cannot be parsed as a JSON dictionary.
    """
    try:
        updates = json.loads(json_params)
    except json.JSONDecodeError as error:
        raise ValueError("json_params is not valid JSON.") from error

    if not isinstance(updates, dict):
        raise ValueError("json_params must decode to a dictionary.")

    _update_existing_nested_parameters(params=params, updates=updates)


def update_from_toml(
    params: dict[str, Any],
    toml_params: str,
) -> None:
    """Update existing parameters from a TOML string.

    Args:
        toml_params (str): TOML string with parameter updates.
        params (dict[str, Any]): Mutable parameter dictionary to update in place.

    Returns:
        None: Updates ``params`` in place.

    Raises:
        ValueError: If ``toml_params`` cannot be parsed as TOML.
    """
    try:
        updates = toml.loads(toml_params)
    except toml.TOMLDecodeError as error:
        raise ValueError("toml_params is not valid TOML.") from error

    _update_existing_nested_parameters(params=params, updates=updates)


def update_runconfig_params_from_args(
    runconfig_params: dict[str, Any],
    args: argparse.Namespace | None,
) -> None:
    """Update run configuration parameters from parsed command line arguments.

    Args:
        runconfig_params (dict): Mutable run configuration dictionary to update.
        args (argparse.Namespace | None): Parsed command line arguments.

    Returns:
        None: Updates ``runconfig_params`` in place.
    """
    if not args:
        return

    # copy arguments to runconfig parameters
    for key, value in list(vars(args).items()):
        if value is None:
            continue
        runconfig_params[key] = value


def add_args_to_parser(
    parser: argparse.ArgumentParser,
    default_params_path: str | None = "params.toml",
    default_save_dir: str | None = "runs",
    default_mode: str = "train_eval",
) -> None:
    """Add command line arguments to a parser."""
    parser.add_argument(
        "-p",
        "--params",
        default=default_params_path,
        help="Path to a file with parameters (JSON, TOML, or, YAML)",
    )
    parser.add_argument(
        "-j",
        "--json_params",
        default=None,
        help="JSON string to override parameters from the `--params` file",
    )
    parser.add_argument(
        "-t",
        "--toml_params",
        default=None,
        help="TOML string to override parameters from the `--params` file",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        default=default_save_dir,
        help="Directory for saving the network and all outputs",
    )
    parser.add_argument(
        "-l",
        "--load_dir",
        default=None,
        help="Directory for loading a network",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=[
            "train",
            "predict",
            "eval",
            "train_eval",
            "train_predict",
            "train_profile",
        ],  # TODO: get choices from Mode class
        default=default_mode,
        help=(
            "Can train, predict, eval, and combine train_eval (default)."
            + "  eval runs on available checkpoints."
            + "  train_eval runs train, predict, and eval."
            + "  train_profile runs profiling of a few training steps."
        ),
    )
