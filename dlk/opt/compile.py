"""Configure `torch.compile` for compiling networks."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch
import torch.nn as nn

CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]

_COMPILE_MODES = set(CompileMode.__args__)

# these modes enable triton.cudagraphs, per torch._inductor.list_mode_options
_CUDA_GRAPH_MODES = {"reduce-overhead", "max-autotune"}


@dataclass
class CompileConfig:
    """Typed argument set for `torch.compile`.

    `mode` and `options` are mutually exclusive, mirroring `torch.compile`
    itself: leaving both `None` lets torch fall back to `mode="default"`.
    `"reduce-overhead"` and `"max-autotune"` enable CUDA graphs and are
    rejected when no CUDA accelerator is available; `"max-autotune-no-cudagraphs"`
    is the hardware-independent alternative. The network or function to compile
    is not configured here; it is passed at the call site.

    Attributes:
        mode: Compilation mode, one of `"default"`, `"reduce-overhead"`,
            `"max-autotune"`, or `"max-autotune-no-cudagraphs"`, or `None` to
            defer to `options` (or to torch's own default if `options` is also
            `None`). `"reduce-overhead"` and `"max-autotune"` require CUDA.
        options: Backend-specific options, or `None` to use `mode` instead.
        fullgraph: Whether to require the entire function to be captured into
            a single graph, raising if a graph break occurs.
        dynamic: Whether to use dynamic shape tracing; `None` to let torch
            detect dynamism automatically.
        backend: Compiler backend to use, or `None` for torch's default
            (`"inductor"`).
        disable: Whether to turn compilation off, returning the original
            callable unchanged.
    """

    mode: CompileMode | None = None
    options: dict[str, str | int | bool | Callable[..., Any]] | None = None
    fullgraph: bool = False
    dynamic: bool | None = None
    backend: str | Callable[..., Any] | None = None
    disable: bool = False

    def __post_init__(self) -> None:
        if self.mode is not None and self.options is not None:
            raise ValueError(
                "expected at most one of mode and options to be specified, got both."
            )
        if self.mode is not None and self.mode not in _COMPILE_MODES:
            raise ValueError(
                f"expected mode to be one of {sorted(_COMPILE_MODES)} or None, "
                f"got {self.mode!r}."
            )
        if self.mode in _CUDA_GRAPH_MODES and not self.auto_cuda_available():
            raise ValueError(
                f"mode {self.mode!r} enables CUDA graphs, which requires a CUDA "
                "accelerator; none is available. Use 'max-autotune-no-cudagraphs' "
                "or 'default' instead."
            )

    @staticmethod
    def auto_cuda_available() -> bool:
        """Hardware probe backing the CUDA-graph mode check."""
        return torch.cuda.is_available()

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "CompileConfig":
        """Build a `CompileConfig` from a dict of constructor arguments.

        Args:
            config: Mapping of constructor argument names to values; every key
                must match a `CompileConfig` field name directly.

        Returns:
            A configured `CompileConfig` instance.

        Raises:
            TypeError: If `config` contains a key that is not a constructor
                argument.
        """
        return cls(**config)

    def to_kwargs(self) -> dict[str, Any]:
        """Return this config as constructor kwargs for `torch.compile`."""
        return {field_.name: getattr(self, field_.name) for field_ in fields(self)}


def compile_net_from_params(
    net: nn.Module,
    compile_params: dict[str, Any] | None,
    logger: logging.Logger | None = None,
) -> nn.Module:
    """Compile a network's forward method from a parameter dict.

    Builds a `CompileConfig` from `compile_params` and replaces `net.forward`
    with its compiled counterpart in place. `compile_params=None` leaves `net`
    unchanged, so compilation can be toggled off from a run configuration.

    Args:
        net: Network whose `forward` method is compiled.
        compile_params: Mapping of `CompileConfig` field names to values, or
            `None` to skip compilation; see `CompileConfig` for the accepted
            keys and their defaults.
        logger: Logger used to report the resolved `CompileConfig`; defaults
            to a logger named after this function.

    Returns:
        `net`, with `forward` replaced by its compiled version, or `net`
        unchanged if `compile_params` is `None`.

    Raises:
        ValueError: If `compile_params` requests an invalid or unsupported
            `CompileConfig`.
        TypeError: If `compile_params` contains a key that is not a
            `CompileConfig` field.
    """
    if compile_params is None:
        return net

    if logger is None:
        logger = logging.getLogger("dlk.opt.compile.compile_net_from_params")

    config = CompileConfig.from_dict(compile_params)
    logger.info("compiling network with %s", config)
    net.forward = torch.compile(net.forward, **config.to_kwargs())
    return net
