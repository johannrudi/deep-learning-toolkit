"""Profile model training loops with PyTorch profiler and export step diagnostics."""

import logging
import os
from collections.abc import Mapping, Sized
from typing import Any, Literal, Protocol, TypeAlias

import torch

from dlk.opt.utils import (
    BatchHookFn,
    EpochHookFn,
    LossFn,
    LRScheduler,
    TensorTransformFn,
    TrainLog,
    ValidationFn,
)

# --------------------------------------
# Types
# --------------------------------------

ProfileDevice: TypeAlias = Literal["cuda", "xpu"]


class KeyAveragesLike(Protocol):
    """Protocol for profiler key-averages objects that format summary tables."""

    def table(self, sort_by: str, row_limit: int = 10) -> str:
        """Return a text table sorted by a selected metric."""
        ...


class ProfilerLike(Protocol):
    """Protocol for profiler objects used by this module."""

    step_num: int

    def key_averages(self) -> KeyAveragesLike:
        """Return aggregated profiling statistics."""
        ...

    def export_chrome_trace(self, path: str) -> None:
        """Export profiling data to a Chrome trace JSON file."""
        ...

    def step(self) -> None:
        """Advance the profiler to the next scheduled step."""
        ...


class TrainEpochsFn(Protocol):
    """Protocol for epoch-level training callables used by the profiler."""

    def __call__(
        self,
        n_epochs: int,
        net: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFn,
        *,
        validation_fn: ValidationFn | None = ...,
        lr_scheduler: LRScheduler | None = ...,
        device: torch.device | None = ...,
        inputs_transform_fn: TensorTransformFn | None = ...,
        targets_transform_fn: TensorTransformFn | None = ...,
        logger: logging.Logger = ...,
        checkpoint_epochs: int | None = ...,
        checkpoint_dir: str = ...,
        epoch_initialize_fn: EpochHookFn | None = ...,
        epoch_finalize_fn: EpochHookFn | None = ...,
    ) -> TrainLog:
        """Run training over epochs and return aggregate training diagnostics."""
        ...


class TrainBatchesFn(Protocol):
    """Protocol for batch-level training callables used by the profiler."""

    def __call__(
        self,
        epoch_idx: int,
        net: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFn,
        *,
        device: torch.device | None = ...,
        inputs_transform_fn: TensorTransformFn | None = ...,
        targets_transform_fn: TensorTransformFn | None = ...,
        logger: logging.Logger = ...,
        batch_initialize_fn: BatchHookFn | None = ...,
        batch_finalize_fn: BatchHookFn | None = ...,
        max_batches: int | None = ...,
    ) -> TrainLog:
        """Run training over batches for one epoch and return training diagnostics."""
        ...


# --------------------------------------


def get_table(prof: ProfilerLike, sort_by: str, row_limit: int = 10) -> str:
    """Format a profiler summary table for one sorting metric.

    Args:
        prof: Active profiler handle with captured step statistics.
        sort_by: Metric key passed to `prof.key_averages().table(...)`.
        row_limit: Maximum number of rows included in the summary table.

    Returns:
        XML-like wrapped string containing one formatted profiler table.
    """
    table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    table = f"<{sort_by}>\n{table.strip()}\n</{sort_by}>\n"
    return table


def trace_handler(
    prof: ProfilerLike,
    device: ProfileDevice | None = None,
    log_profile_dir: str = ".",
) -> None:
    """Write profiler tables and a Chrome trace file for one completed trace.

    Replaces the built-in handler: `torch.profiler.tensorboard_trace_handler(log_profile_dir)`

    Args:
        prof: Active profiler handle for the completed trace window.
        device: Optional accelerator name used for device-specific metrics.
        log_profile_dir: Directory where table and trace files are written.

    Returns:
        None.
    """
    os.makedirs(log_profile_dir, exist_ok=True)

    # generate profiler summary tables
    table = f"<profile_result step={prof.step_num}>\n"
    table += get_table(prof, "cpu_time_total")
    table += get_table(prof, "self_cpu_time_total")
    if device is not None:
        table += get_table(prof, f"{device}_time_total")
        table += get_table(prof, f"self_{device}_time_total")
    table += get_table(prof, "self_cpu_memory_usage")
    if device is not None:
        table += get_table(prof, f"self_{device}_memory_usage")
    table += "</profile_result>\n"

    # write summary table to file and stdout
    table_path = os.path.join(log_profile_dir, f"table_prof_step_{prof.step_num}.txt")
    with open(table_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(table)
    print(table)

    # write Chrome trace JSON
    trace_path = os.path.join(log_profile_dir, f"trace_prof_step_{prof.step_num}.json")
    prof.export_chrome_trace(trace_path)


def _select_profiler_activities() -> (
    tuple[list[torch.profiler.ProfilerActivity], ProfileDevice | None]
):
    """Select profiler activities and associated accelerator label.

    Uses the activities-based profiler API and avoids legacy flags.

    Returns:
        Tuple of selected profiler activities and optional accelerator label.
    """
    # select profiler activities for available hardware
    activities: list[torch.profiler.ProfilerActivity] = [
        torch.profiler.ProfilerActivity.CPU
    ]
    device: ProfileDevice | None = None
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        device = "cuda"
    elif torch.xpu.is_available():
        activities.append(torch.profiler.ProfilerActivity.XPU)
        device = "xpu"
    return activities, device


def profile_train_epochs(
    train_epochs_fn: TrainEpochsFn,
    train_epochs_fn_kwargs: Mapping[str, Any],
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    log_profile_dir: str = ".",
) -> ProfilerLike:
    """Profile training for multiple epochs using a fixed profiler schedule.

    Total number of profiling steps:
        (1 wait + 1 warmup + 3 active) * 2 repeats = 10 steps.

    Source:
        https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

    Args:
        train_epochs_fn: Training loop callable with epoch-level stepping hooks.
        train_epochs_fn_kwargs: Extra keyword arguments forwarded to `train_epochs_fn`.
        net: Model optimized by `train_epochs_fn`.
        dataloader: Data loader consumed by `train_epochs_fn`.
        optimizer: Optimizer forwarded to `train_epochs_fn`.
        loss_fn: Loss callable forwarded to `train_epochs_fn`.
        log_profile_dir: Output directory for profiler reports and traces.

    Returns:
        Profiler handle with captured profiling data.
    """
    # select profiler activities for available hardware
    activities, device = _select_profiler_activities()

    # configure a periodic profiling schedule
    schedule = torch.profiler.schedule(
        skip_first=0,  # ignore initial steps
        wait=1,  # keep first step inactive in each cycle
        warmup=1,  # warm up profiler for one step in each cycle
        active=3,  # record three active steps per cycle
        repeat=2,  # repeat the cycle twice
    )

    # run training with profiler stepping hooks
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda p: trace_handler(p, device, log_profile_dir),
        record_shapes=True,  # capture operator input shapes
        profile_memory=True,  # track model tensor memory usage
        with_stack=True,  # include Python stack traces
    ) as prof:

        def epoch_finalize_fn(_epoch_idx: int) -> None:
            """Signal the end of each training step to the profiler."""
            prof.step()

        n_epochs: int = 10  # keep above the total number of profiled steps
        train_epochs_fn(
            n_epochs,
            net,
            dataloader,
            optimizer,
            loss_fn,
            logger=logging.getLogger("dlk.opt.profile_train_epochs"),
            epoch_finalize_fn=epoch_finalize_fn,
            **train_epochs_fn_kwargs,
        )

        return prof


def _infer_profiled_batch_count(batch_dlog: TrainLog, default: int) -> int:
    """Infer the number of profiled batches from a training log dictionary.

    Args:
        batch_dlog: Batch-level training diagnostics returned by `train_batches_fn`.
        default: Fallback value used when no batch count can be inferred.

    Returns:
        Number of processed batches inferred from available log fields.
    """
    loss_mean_n = batch_dlog.get("loss_mean_n")
    if isinstance(loss_mean_n, (int, float)):
        return int(loss_mean_n)

    loss_values = batch_dlog.get("loss")
    if isinstance(loss_values, torch.Tensor):
        return int(loss_values.numel())
    if isinstance(loss_values, Sized):
        return len(loss_values)
    return default


def profile_train_batches(
    train_batches_fn: TrainBatchesFn,
    train_batches_fn_kwargs: Mapping[str, Any],
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    log_profile_dir: str = ".",
) -> ProfilerLike:
    """Profile training for one epoch by stepping at batch boundaries.

    Total number of profiling steps:
        (1 wait + 1 warmup + 3 active) * 2 repeats = 10 steps.

    Source:
        https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

    Args:
        train_batches_fn: Batch-level training callable with step-finalize hooks.
        train_batches_fn_kwargs: Extra keyword arguments forwarded to `train_batches_fn`.
        net: Model optimized by `train_batches_fn`.
        dataloader: Data loader consumed by `train_batches_fn`.
        optimizer: Optimizer forwarded to `train_batches_fn`.
        loss_fn: Loss callable forwarded to `train_batches_fn`.
        log_profile_dir: Output directory for profiler reports and traces.
        logger: Logger used for profiling diagnostics.

    Returns:
        Profiler handle with captured profiling data.
    """
    # select profiler activities for available hardware
    activities, device = _select_profiler_activities()

    # configure a periodic profiling schedule
    schedule = torch.profiler.schedule(
        skip_first=0,  # ignore initial steps
        wait=1,  # keep first step inactive in each cycle
        warmup=1,  # warm up profiler for one step
        active=3,  # record three active steps per cycle
        repeat=2,  # repeat the cycle twice
    )
    max_batches: int = 10  # keep above the total number of profiled steps

    # run training with profiler stepping hooks
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda p: trace_handler(p, device, log_profile_dir),
        record_shapes=True,  # capture operator input shapes
        profile_memory=True,  # track model tensor memory usage
        with_stack=True,  # include Python stack traces
    ) as prof:

        def batch_finalize_fn(_batch_idx: int) -> None:
            """Signal the end of each training step to the profiler."""
            prof.step()

        epoch_idx: int = 0
        logger: logging.Logger = logging.getLogger("dlk.opt.profile_train_batches")
        batch_dlog = train_batches_fn(
            epoch_idx,
            net,
            dataloader,
            optimizer,
            loss_fn,
            logger=logger,
            batch_finalize_fn=batch_finalize_fn,
            max_batches=max_batches,
            **train_batches_fn_kwargs,
        )
        n_batches = _infer_profiled_batch_count(batch_dlog, default=max_batches)

        if n_batches < max_batches:
            logger.warning(
                f"Expected {max_batches} batches for profiling, got {n_batches} batches"
            )

        return prof
