"""Utility helpers for typing, checkpoint I/O, and per-batch/per-epoch training logs."""

import math
import pathlib
import sys
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any, Protocol, TypeAlias

import torch

# --------------------------------------
# Types
# --------------------------------------

EpochHookFn: TypeAlias = Callable[[int], None]
BatchHookFn: TypeAlias = Callable[[int], None]
LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ValidationFn: TypeAlias = Callable[[int, torch.nn.Module], None]
TensorTransformFn: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
InputsTransformFn: TypeAlias = Callable[
    [torch.Tensor | tuple[torch.Tensor, ...]],
    torch.Tensor | tuple[torch.Tensor, ...],
]
TrainLog: TypeAlias = dict[str, Any]


class LRSchedulerType(Protocol):
    """Protocol for learning-rate schedulers used during training."""

    def get_last_lr(self) -> list[float]:
        """Return learning rates for each optimizer parameter group."""
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
        raise ValueError(f"Expected n_epochs > 0, got {n_epochs}.")
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filepath,
    )


@torch.no_grad()
def train_dlog_batch_initialize(
    n_batches: int,
    tags: Sequence[str],
    save_list: bool = False,
) -> dict[str, Any]:
    """Initialize batch-level training log storage and running statistics.

    Args:
        n_batches: Number of batches in the current epoch.
        tags: Metric names to track.
        save_list: Whether to store per-batch values for each metric.

    Returns:
        Mutable dictionary storing per-tag values and aggregate statistics.
    """
    dlog: dict[str, Any] = {"n_batches": n_batches}
    for tag in tags:
        if save_list:
            dlog[tag] = torch.empty((n_batches,), dtype=torch.float64)
        else:
            dlog[tag] = None
        dlog[f"{tag}_mean_n"] = 0
        dlog[f"{tag}_mean"] = 0.0
        dlog[f"{tag}_sq_mean"] = 0.0
        dlog[f"{tag}_std"] = None
    return dlog


@torch.no_grad()
def train_dlog_batch_update(
    dlog: MutableMapping[str, Any],
    batch_idx: int,
    values: Mapping[str, float | int | torch.Tensor],
) -> None:
    """Update batch-level logs with metric values from a single batch.

    Args:
        dlog: Training log dictionary from `train_dlog_batch_initialize`.
        batch_idx: Index of the current batch.
        values: Mapping from metric names to scalar values.

    Returns:
        None.
    """
    for tag, val in values.items():
        if isinstance(val, torch.Tensor):
            val = float(val.detach().item())
        else:
            val = float(val)
        if dlog[tag] is not None:
            dlog[tag][batch_idx] = val
        if not math.isnan(val):
            dlog[f"{tag}_mean_n"] += 1
            dlog[f"{tag}_mean"] += val
            dlog[f"{tag}_sq_mean"] += val * val


@torch.no_grad()
def train_dlog_batch_finalize(
    dlog: MutableMapping[str, Any],
    tags: Sequence[str],
) -> None:
    """Finalize batch-level running statistics for the requested metric tags.

    Args:
        dlog: Training log dictionary containing running aggregates.
        tags: Metric names to finalize.

    Returns:
        None.
    """
    for tag in tags:
        assert dlog[f"{tag}_std"] is None
        assert not math.isnan(dlog[f"{tag}_mean"])
        assert not math.isnan(dlog[f"{tag}_sq_mean"])
        if 0 < dlog[f"{tag}_mean_n"]:
            dlog[f"{tag}_mean"] *= 1.0 / dlog[f"{tag}_mean_n"]
            dlog[f"{tag}_sq_mean"] *= 1.0 / dlog[f"{tag}_mean_n"]
            variance = dlog[f"{tag}_sq_mean"] - dlog[f"{tag}_mean"] ** 2
            dlog[f"{tag}_std"] = torch.sqrt(
                torch.tensor(variance, dtype=torch.float64)
            ).item()
        else:
            dlog[f"{tag}_mean"] = 0.0
            dlog[f"{tag}_sq_mean"] = 0.0
            dlog[f"{tag}_std"] = 0.0


@torch.no_grad()
def train_dlog_epoch_initialize(
    n_epochs: int,
    tags: Sequence[str],
    save_list: bool = True,
) -> dict[str, Any]:
    """Initialize epoch-level training log storage.

    Args:
        n_epochs: Total number of epochs to store.
        tags: Metric names to track.
        save_list: Whether to store per-epoch values for each metric.

    Returns:
        Mutable dictionary storing per-epoch metrics and batch logs.
    """
    dlog: dict[str, Any] = {}
    for tag in tags:
        if save_list:
            dlog[tag] = torch.empty((n_epochs,), dtype=torch.float64)
        else:
            dlog[tag] = None
    dlog["batch_dlog"] = []
    return dlog


@torch.no_grad()
def train_dlog_epoch_update(
    dlog: MutableMapping[str, Any],
    epoch_idx: int,
    tags: Sequence[str],
    batch_dlog: Mapping[str, Any],
) -> None:
    """Update epoch-level logs using finalized metrics from one epoch.

    Args:
        dlog: Epoch-level training log dictionary.
        epoch_idx: Index of the epoch to update.
        tags: Metric names to copy from `batch_dlog`.
        batch_dlog: Finalized batch-level metrics for one epoch.

    Returns:
        None.
    """
    for tag in tags:
        if dlog[tag] is not None:
            dlog[tag][epoch_idx] = batch_dlog[tag]
    dlog["batch_dlog"].append(batch_dlog)


def train_dlog_epoch_finalize(
    dlog: MutableMapping[str, Any],
    time_train: float,
) -> None:
    """Attach total training wall-clock time to the epoch-level log.

    Args:
        dlog: Epoch-level training log dictionary.
        time_train: Total training time in seconds.

    Returns:
        None.
    """
    dlog["time_train"] = time_train


# --------------------------------------


def tqdm_disable() -> bool:
    """Return True when tqdm output should be suppressed.

    Notebooks are detected via IPython and always show tqdm. Non-TTY
    environments (e.g. SLURM batch jobs) suppress it.

    Returns:
        True to disable tqdm, False to enable it.
    """
    # show in Jupyter notebooks regardless of TTY
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        if get_ipython() is not None:
            return False
    except ImportError:
        pass
    # suppress for non-interactive (e.g. SLURM) jobs
    return not sys.stdout.isatty()
