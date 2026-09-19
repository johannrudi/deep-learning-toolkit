#!/usr/bin/env python3
"""Compare a config dataclass against the constructor it configures.

Run with one argument to read a constructor before writing its config:

    uv run .agents/skills/write-config-dataclass/scripts/check_config.py \
        torch.utils.data.DataLoader

Run with two arguments to check an existing config against its target:

    uv run .agents/skills/write-config-dataclass/scripts/check_config.py \
        torch.utils.data.DataLoader dlk.data.loader.DataLoaderConfig

Exits non-zero when any check fails.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import sys
from typing import Any, Optional

PARAM_PREFIX = "param_"


def resolve(dotted_path: str) -> Any:
    """Import a dotted path such as `torch.utils.data.DataLoader`.

    Args:
        dotted_path: Module path, optionally followed by attribute names.

    Returns:
        The imported object.

    Raises:
        SystemExit: If no prefix of `dotted_path` imports as a module, or the
            remaining attributes do not resolve.
    """
    parts = dotted_path.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for attribute in parts[split:]:
                obj = getattr(obj, attribute)
        except AttributeError:
            break
        return obj
    raise SystemExit(f"cannot import {dotted_path}")


def target_parameters(target: Any) -> dict[str, inspect.Parameter]:
    """Return the keyword-addressable parameters of a class or function.

    Args:
        target: Class whose `__init__` is inspected, or a function.

    Returns:
        Mapping of parameter name to parameter, without `self`, `*args`, and
        `**kwargs`.
    """
    function = target.__init__ if inspect.isclass(target) else target
    parameters = dict(inspect.signature(function).parameters)
    parameters.pop("self", None)
    return {
        name: parameter
        for name, parameter in parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def field_default(field: dataclasses.Field[Any]) -> Any:
    """Return a dataclass field's default, or `inspect.Parameter.empty`."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return inspect.Parameter.empty


def show(value: Any) -> str:
    """Render a default value for one column of output."""
    if value is inspect.Parameter.empty:
        return "required"
    return repr(value)


def report_target(target: Any, parameters: dict[str, inspect.Parameter]) -> None:
    """Print every parameter of the target with its annotation and default."""
    print(f"target: {target.__module__}.{target.__qualname__}")
    print(f"        {len(parameters)} parameters\n")
    for name, parameter in parameters.items():
        annotation = (
            ""
            if parameter.annotation is inspect.Parameter.empty
            else str(parameter.annotation).replace("typing.", "")
        )
        print(f"  {name:<28} {annotation:<34} {show(parameter.default)}")


def check_config(
    config: Any, parameters: dict[str, inspect.Parameter]
) -> tuple[list[str], list[str]]:
    """Check one config dataclass against its target's parameters.

    Args:
        config: The config dataclass to check.
        parameters: Parameters of the target, from `target_parameters`.

    Returns:
        A pair of failure lines and informational lines.
    """
    failures: list[str] = []
    infos: list[str] = []

    stored = {field.name: field for field in dataclasses.fields(config)}
    init_arguments = [
        name for name in inspect.signature(config.__init__).parameters if name != "self"
    ]
    init_only = [name for name in init_arguments if name not in stored]

    print(f"config: {config.__module__}.{config.__qualname__}")
    print(f"        {len(stored)} stored fields, {len(init_only)} constructor-only\n")

    for name, field in stored.items():
        role = "auto " if not field.init else "set  "
        source = f"<- {PARAM_PREFIX}{name}" if not field.init else ""
        print(f"  {role} {name:<28} {show(field_default(field)):<20} {source}")
    print()

    # Every stored field must survive to the target's constructor.
    for name in stored:
        if name not in parameters:
            failures.append(f"stored field `{name}` is not a parameter of the target")

    # Every constructor-only input must pair with a field it resolves.
    for name in init_only:
        if not name.startswith(PARAM_PREFIX):
            failures.append(
                f"constructor-only input `{name}` lacks the `{PARAM_PREFIX}` prefix"
            )
            continue
        paired = name[len(PARAM_PREFIX) :]
        if paired not in stored:
            failures.append(f"`{name}` pairs with no stored field `{paired}`")
        elif stored[paired].init:
            failures.append(f"`{paired}` is set by `{name}`, so it needs `init=False`")

    # An auto-resolved field with no constructor-only input is unreachable.
    for name, field in stored.items():
        if not field.init and f"{PARAM_PREFIX}{name}" not in init_only:
            failures.append(
                f"`{name}` has `init=False` and no `{PARAM_PREFIX}{name}` input"
            )

    # Uncovered parameters and deliberate default changes are choices, not bugs.
    uncovered = [name for name in parameters if name not in stored]
    if uncovered:
        infos.append(f"target parameters not covered: {', '.join(uncovered)}")
    for name, field in stored.items():
        if name not in parameters or not field.init:
            continue
        default, target_default = field_default(field), parameters[name].default
        if default != target_default:
            infos.append(
                f"`{name}` defaults to {show(default)}, "
                f"target defaults to {show(target_default)}"
            )

    return failures, infos


def main(argv: list[str]) -> int:
    """Read a target constructor, and check a config against it when given."""
    if not 1 <= len(argv) <= 2:
        raise SystemExit(__doc__)

    target = resolve(argv[0])
    parameters = target_parameters(target)
    report_target(target, parameters)

    if len(argv) == 1:
        return 0

    config: Optional[Any] = resolve(argv[1])
    if not dataclasses.is_dataclass(config):
        raise SystemExit(f"{argv[1]} is not a dataclass")

    print()
    failures, infos = check_config(config, parameters)
    for info in infos:
        print(f"INFO  {info}")
    for failure in failures:
        print(f"FAIL  {failure}")
    if not failures:
        print("PASS  every stored field and constructor-only input lines up")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
