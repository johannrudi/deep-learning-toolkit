"""Unit tests for the compile config in `dlk.opt.compile`."""

import dataclasses
import inspect

import pytest
import torch

from dlk.opt.compile import CompileConfig, CompileMode


def test_config_fields_are_valid_compile_arguments() -> None:
    """Every stored field of the config must be a parameter of `torch.compile`."""
    compile_parameters = inspect.signature(torch.compile).parameters

    config_fields = {field.name for field in dataclasses.fields(CompileConfig)}

    assert config_fields <= compile_parameters.keys()


def test_mode_and_options_default_to_none() -> None:
    """Leaving both unset defers to torch's own default (`mode="default"`)."""
    config = CompileConfig()

    assert config.mode is None
    assert config.options is None


@pytest.mark.parametrize(
    "mode",
    ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
)
def test_mode_accepts_every_documented_value(
    mode: CompileMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every mode named in `torch.compile`'s docs is a valid config value."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    config = CompileConfig(mode=mode)

    assert config.mode == mode


def test_invalid_mode_raises() -> None:
    """A mode outside the documented set is a configuration error."""
    with pytest.raises(ValueError, match="mode"):
        CompileConfig(mode="not-a-real-mode")  # type: ignore[arg-type]


def test_mode_and_options_are_mutually_exclusive() -> None:
    """Specifying both `mode` and `options` is a configuration error."""
    with pytest.raises(ValueError, match="at most one of mode and options"):
        CompileConfig(mode="default", options={"trace.enabled": True})


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_cuda_graph_modes_require_cuda(
    mode: CompileMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Modes that enable CUDA graphs are rejected without a CUDA accelerator."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA"):
        CompileConfig(mode=mode)


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_cuda_graph_modes_are_accepted_with_cuda(
    mode: CompileMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Modes that enable CUDA graphs are accepted when CUDA is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    config = CompileConfig(mode=mode)

    assert config.mode == mode


def test_max_autotune_no_cudagraphs_does_not_require_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hardware-independent variant is accepted without CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = CompileConfig(mode="max-autotune-no-cudagraphs")

    assert config.mode == "max-autotune-no-cudagraphs"


def test_to_kwargs_round_trips_through_from_dict() -> None:
    """`from_dict` accepts what `to_kwargs` produces, unchanged."""
    config = CompileConfig(
        fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs"
    )

    assert CompileConfig.from_dict(config.to_kwargs()) == config
