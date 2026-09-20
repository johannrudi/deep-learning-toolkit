"""Utility helpers for typing, checkpoint I/O, and training-loop performance helpers."""

import math
import pathlib
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, TypeAlias

import torch

from dlk.opt import distributed

# --------------------------------------
# Types
# --------------------------------------

EpochHookFn: TypeAlias = Callable[[int], None]
BatchHookFn: TypeAlias = Callable[[int], None]
LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorTransformFn: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
InputsTransformFn: TypeAlias = Callable[
    [torch.Tensor | tuple[torch.Tensor, ...]],
    torch.Tensor | tuple[torch.Tensor, ...],
]


class LRSchedulerType(Protocol):
    """Protocol for learning-rate schedulers used during training."""

    def get_last_lr(self) -> Sequence[float | torch.Tensor]:
        """Return learning rates for each optimizer parameter group.

        Mirrors `torch.optim.lr_scheduler.LRScheduler.get_last_lr`, which may
        return tensor learning rates. The return type is a covariant `Sequence`
        so schedulers returning `list[float]` also satisfy the protocol.
        """
        ...

    def step(self) -> None:
        """Advance the scheduler state by one step."""
        ...


class DataLoaderType(Protocol):
    """Protocol for objects that can serve as a dataloader in training loops.

    Covers `torch.utils.data.DataLoader` and custom wrappers.
    """

    @property
    def batch_size(self) -> int | None:
        """Number of samples per batch, or `None` if not fixed."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Yield batches."""
        ...

    def __len__(self) -> int:
        """Return the number of batches."""
        ...


class ValidationFn(Protocol):
    def __call__(self, epoch_idx: int, **kwargs: torch.nn.Module) -> None: ...


# --------------------------------------
# Checkpoints
# --------------------------------------


def checkpoint_path(
    checkpoint_dir: str | pathlib.Path,
    n_epochs: int,
    prefix: str,
    epoch: int,
) -> pathlib.Path:
    """Build the checkpoint path for a given training epoch.

    Args:
        checkpoint_dir: Directory where checkpoint files are stored.
        n_epochs: Total number of epochs in training.
        prefix: Prefix used in checkpoint filenames.
        epoch: Epoch index to encode in the filename.

    Returns:
        Path to the checkpoint file for the selected epoch.
    """
    if n_epochs <= 0:
        raise ValueError(f"expected n_epochs > 0, got {n_epochs}.")
    n_digits = int(math.ceil(math.log10(1.01 * n_epochs)))
    filename = f"{prefix}_e{epoch:0{n_digits}d}.pt"
    return pathlib.Path(checkpoint_dir) / filename


def checkpoint_save(
    model: torch.nn.Module,
    filepath: str | pathlib.Path,
    epoch: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Save model and optimizer state dictionaries to a checkpoint file.

    DDP-wrapped models are unwrapped before saving, so checkpoints never
    contain `module.`-prefixed keys.

    Args:
        model: Model whose parameters should be saved.
        filepath: Output checkpoint file path.
        epoch: Epoch index to store in the checkpoint metadata.
        optimizer: Optimizer whose state should be saved.

    Returns:
        None.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": distributed.unwrap_net(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filepath,
    )


def checkpoint_load(
    filepath: str | pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> int:
    """Load model and optimizer state dictionaries from a checkpoint file.

    DDP-wrapped models are unwrapped before loading, and a legacy `module.`
    key prefix (from checkpoints saved with a wrapped model) is stripped.
    Under DDP, every rank loads the same file; pass the process's device as
    `map_location` and load preferably before wrapping with DDP.

    Args:
        filepath: Checkpoint file path written by `checkpoint_save`.
        model: Model whose parameters are restored in place.
        optimizer: Optional optimizer whose state is restored in place.
        map_location: Device the stored tensors are mapped to; defaults to CPU.

    Returns:
        Epoch index stored in the checkpoint metadata.
    """
    checkpoint = torch.load(
        filepath,
        map_location=map_location if map_location is not None else "cpu",
        weights_only=True,
    )
    model_state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    distributed.unwrap_net(model).load_state_dict(model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


# --------------------------------------
# Performance Helpers
# --------------------------------------


def transfer_non_blocking(
    dataloader: DataLoaderType,
    device: torch.device | None = None,
) -> bool:
    """Decide whether host-to-device copies of batches may be asynchronous.

    Asynchronous copies pay off only when the source batch lives in pinned
    (page-locked) host memory and the destination is an accelerator that
    supports them. Copies from pageable memory are staged through a pinned
    buffer by the driver and stay synchronous regardless of the flag, so this
    returns True only when the dataloader pins its batches and `device` is a
    CUDA or XPU device.

    Dataloaders without a `pin_memory` attribute are treated as not pinning.

    Args:
        dataloader: Dataloader serving the batches.
        device: Device the batches are moved to; `None` means no transfer.

    Returns:
        True if `Tensor.to` may be called with `non_blocking=True`.
    """
    if device is None or device.type not in ("cuda", "xpu"):
        return False
    return bool(getattr(dataloader, "pin_memory", False))


def autocast_context(
    device: torch.device | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> torch.autocast:
    """Create an autocast context for mixed-precision forward passes.

    Autocast is enabled for `torch.bfloat16` only; `None` and `torch.float32`
    request full precision and yield a disabled context. `torch.float16` is
    rejected because it needs loss scaling (`torch.amp.GradScaler`), which the
    training loops do not apply.

    Args:
        device: Device holding the batch tensors; `None` selects the CPU.
        autocast_dtype: Compute dtype inside the context. Use `torch.bfloat16`
            for mixed precision, `None` or `torch.float32` for full precision.

    Returns:
        Configured `torch.autocast` context manager, disabled for full precision.

    Raises:
        ValueError: If `autocast_dtype` is neither `None`, `torch.float32`, nor
            `torch.bfloat16`.
    """
    if autocast_dtype is None:
        autocast_dtype = torch.float32
    if autocast_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "autocast_dtype must be None, torch.float32, or torch.bfloat16, got "
            f"{autocast_dtype}; torch.float16 requires loss scaling with "
            "torch.amp.GradScaler, which these training loops do not apply"
        )
    return torch.autocast(
        device_type=device.type if device is not None else "cpu",
        dtype=autocast_dtype,
        enabled=torch.bfloat16 == autocast_dtype,
    )


# --------------------------------------
# Print Helpers
# --------------------------------------


def format_seconds(seconds: float, precision: int = 2) -> str:
    """Format a duration in seconds as milliseconds or seconds, whichever reads better.

    Uses milliseconds below one second, seconds otherwise.

    Args:
        seconds: Duration in seconds.
        precision: Number of digits after the decimal point.

    Returns:
        Formatted duration with a unit suffix (`ms` or `s`).
    """
    if abs(seconds) < 1.0:
        return f"{seconds * 1e3:.{precision}f} ms"
    return f"{seconds:.{precision}f} s"


def tqdm_disable() -> bool:
    """Return True when tqdm output should be suppressed.

    Notebooks are detected via IPython and always show tqdm. Non-TTY
    environments (e.g. SLURM batch jobs) and non-main ranks of distributed
    runs suppress it.

    Returns:
        True to disable tqdm, False to enable it.
    """
    # suppress on non-main ranks of distributed runs
    if not distributed.is_main_process():
        return True
    # show in Jupyter notebooks regardless of TTY
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        if get_ipython() is not None:
            return False
    except ImportError:
        pass
    # suppress for non-interactive (e.g. SLURM) jobs
    return not sys.stdout.isatty()
