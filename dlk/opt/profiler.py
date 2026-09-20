"""Profile model training loops with PyTorch profiler and export step diagnostics."""

import logging
import pathlib
from collections.abc import Mapping, Sized
from typing import Any, Literal, Protocol, TypeAlias

import torch

from dlk.opt import distributed
from dlk.opt.monitor import TrainLog
from dlk.opt.utils import (
    BatchHookFn,
    EpochHookFn,
    InputsTransformFn,
    LossFn,
    LRSchedulerType,
    TensorTransformFn,
    ValidationFn,
)

# --------------------------------------
# Types
# --------------------------------------

ProfileDevice: TypeAlias = Literal["cpu", "cuda", "xpu"]

# map device names to profiler activities
_DEVICE_ACTIVITIES: dict[ProfileDevice, torch.profiler.ProfilerActivity] = {
    "cpu": torch.profiler.ProfilerActivity.CPU,
    "cuda": torch.profiler.ProfilerActivity.CUDA,
    "xpu": torch.profiler.ProfilerActivity.XPU,
}


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
    """Protocol for epoch-level training callables used by the profiler.

    Mirrors the signature of `dlk.opt.train.train_epochs`; update in lockstep
    with any signature change there.
    """

    def __call__(
        self,
        n_epochs: int,
        net: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFn,
        *,
        validation_fn: ValidationFn | None = ...,
        lr_scheduler: LRSchedulerType | None = ...,
        device: torch.device | None = ...,
        inputs_transform_fn: InputsTransformFn | None = ...,
        targets_transform_fn: TensorTransformFn | None = ...,
        logger: logging.Logger = ...,
        checkpoint_epochs: int | None = ...,
        checkpoint_dir: str = ...,
        epoch_initialize_fn: EpochHookFn | None = ...,
        epoch_finalize_fn: EpochHookFn | None = ...,
        autocast_dtype: torch.dtype | None = ...,
    ) -> TrainLog:
        """Run training over epochs and return aggregate training diagnostics."""
        ...


class TrainBatchesFn(Protocol):
    """Protocol for batch-level training callables used by the profiler.

    Mirrors the signature of `dlk.opt.train.train_batches`; update in lockstep
    with any signature change there.
    """

    def __call__(
        self,
        epoch_idx: int,
        net: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFn,
        *,
        device: torch.device | None = ...,
        inputs_transform_fn: InputsTransformFn | None = ...,
        targets_transform_fn: TensorTransformFn | None = ...,
        logger: logging.Logger = ...,
        batch_initialize_fn: BatchHookFn | None = ...,
        batch_finalize_fn: BatchHookFn | None = ...,
        max_batches: int | None = ...,
        autocast_dtype: torch.dtype | None = ...,
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
    logger: logging.Logger,
    device: ProfileDevice | None = None,
    profile_memory: bool = False,
    table_row_limit: int = 10,
    trace_dir: str | pathlib.Path = ".",
) -> None:
    """Write profiler tables and a Chrome trace file for one completed trace.

    Replaces the built-in handler: `torch.profiler.tensorboard_trace_handler(trace_dir)`

    Under distributed training, every rank writes its own rank-suffixed table
    and trace files (per-rank traces expose communication ops and stragglers),
    while the table is printed on the main process only.

    Args:
        prof: Active profiler handle for the completed trace window.
        logger: Logger used to report written table and trace file paths.
        device: Optional accelerator name used for device-specific metrics.
        profile_memory: Whether memory usage was profiled and should be reported.
        trace_dir: Directory where table and trace files are written.

    Returns:
        None.
    """
    profile_dir = pathlib.Path(trace_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    # suffix output files with the rank in distributed runs
    if distributed.is_distributed():
        rank_suffix = f"_rank{distributed.get_rank():04d}"
    else:
        rank_suffix = ""

    # generate profiler summary tables
    table = f"<profile_result step={prof.step_num}>\n"
    table += get_table(prof, "cpu_time_total", row_limit=table_row_limit)
    table += get_table(prof, "self_cpu_time_total", row_limit=table_row_limit)
    if device is not None:
        table += get_table(prof, f"{device}_time_total", row_limit=table_row_limit)
        table += get_table(prof, f"self_{device}_time_total", row_limit=table_row_limit)
    if profile_memory:
        table += get_table(prof, "self_cpu_memory_usage", row_limit=table_row_limit)
        if device is not None:
            table += get_table(
                prof, f"self_{device}_memory_usage", row_limit=table_row_limit
            )
    table += "</profile_result>\n"

    # write summary table to file on every rank; print on the main process only
    table_path = profile_dir / f"table_step_{prof.step_num}{rank_suffix}.txt"
    with open(table_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(table)
    logger.info(f"Wrote profiler table to {table_path}")
    if distributed.is_main_process():
        print(table)

    # write Chrome trace JSON
    trace_path = profile_dir / f"trace_step_{prof.step_num}{rank_suffix}.json"
    prof.export_chrome_trace(str(trace_path))
    logger.info(f"Wrote Chrome trace to {trace_path}")


def _select_profiler_activities() -> (
    tuple[list[torch.profiler.ProfilerActivity], ProfileDevice | None]
):
    """Select profiler activities for the available hardware.

    Uses the activities-based profiler API and avoids legacy flags.

    Returns:
        Tuple of selected profiler activities and optional accelerator label.
    """
    # always select the CPU
    activities: list[torch.profiler.ProfilerActivity] = [
        torch.profiler.ProfilerActivity.CPU
    ]

    # select the current accelerator, if any, among the supported devices
    device: ProfileDevice | None = None
    if torch.accelerator.is_available():
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is not None:
            for candidate, activity in _DEVICE_ACTIVITIES.items():
                if accelerator.type == candidate:
                    device = candidate
                    activities.append(activity)
                    break

    return activities, device


def profile_train_epochs(
    train_epochs_fn: TrainEpochsFn,
    train_epochs_fn_kwargs: Mapping[str, Any],
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    logger: logging.Logger | None = None,
    skip_first: int = 0,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 2,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    trace_dir: str | pathlib.Path = ".",
) -> ProfilerLike:
    """Profile training for multiple epochs using a periodic profiler schedule.

    Total number of profiling steps:
        2 skip_first + (1 wait + 1 warmup + 3 active) * 2 repeats = 12 steps.

    Under distributed training, call on every rank with a DDP-wrapped model;
    each rank writes its own rank-suffixed trace showing communication ops.

    Source:
        https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

    Args:
        train_epochs_fn: Training loop callable with epoch-level stepping hooks.
        train_epochs_fn_kwargs: Extra keyword arguments forwarded to `train_epochs_fn`.
        net: Model optimized by `train_epochs_fn`.
        dataloader: Data loader consumed by `train_epochs_fn`.
        optimizer: Optimizer forwarded to `train_epochs_fn`.
        loss_fn: Loss callable forwarded to `train_epochs_fn`.
        logger: Logger used for profiling diagnostics; defaults to a module logger.
        skip_first: Number of steps to skip before the start of profiling cycles.
        wait: Number of steps kept inactive at the start of each profiling cycle.
        warmup: Number of steps used to warm up the profiler in each cycle.
        active: Number of steps actively recorded in each profiling cycle.
        repeat: Number of times the wait/warmup/active cycle repeats.
        record_shapes: Whether to capture operator input shapes.
        profile_memory: Whether to track model tensor memory usage.
        with_stack: Whether to include Python stack traces.
        trace_dir: Output directory for profiler reports and traces.

    Returns:
        Profiler handle with captured profiling data.
    """
    if logger is None:
        logger = logging.getLogger("dlk.opt.profile.profile_train_epochs")

    reserved_kwargs = {"logger", "epoch_finalize_fn"}
    if reserved_kwargs & train_epochs_fn_kwargs.keys():
        raise ValueError(
            f"train_epochs_fn_kwargs must not set {reserved_kwargs}; "
            "the profiler sets these internally"
        )

    # select profiler activities for available hardware
    activities, device = _select_profiler_activities()

    # configure a periodic profiling schedule
    schedule = torch.profiler.schedule(
        skip_first=skip_first,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )
    n_epochs = skip_first + (wait + warmup + active) * repeat

    # run training with profiler stepping hooks
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda p: trace_handler(
            p,
            logger,
            device=device,
            profile_memory=profile_memory,
            table_row_limit=20,
            trace_dir=trace_dir,
        ),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:

        def epoch_finalize_fn(_epoch_idx: int) -> None:
            """Signal the end of each training step to the profiler."""
            prof.step()

        train_epochs_fn(
            n_epochs,
            net,
            dataloader,
            optimizer,
            loss_fn,
            logger=logger,
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
    train_batches_fn_args: tuple[Any, ...],
    train_batches_fn_kwargs: Mapping[str, Any],
    logger: logging.Logger | None = None,
    skip_first: int = 0,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 2,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    trace_dir: str | pathlib.Path = ".",
) -> ProfilerLike:
    """Profile training for one epoch by stepping at batch boundaries.

    Total number of profiling steps:
        4 skip_first + (1 wait + 1 warmup + 3 active) * 2 repeats = 14 steps.

    Under distributed training, call on every rank with a DDP-wrapped model;
    each rank writes its own rank-suffixed trace showing communication ops.
    For meaningful (unpadded) profiled batches, the dataset should hold at
    least `10 * batch_size * world_size` samples.

    Source:
        https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

    Args:
        train_batches_fn: Batch-level training callable with step-finalize hooks.
        train_batches_fn_args: Positional arguments forwarded to `train_batches_fn`
            after `epoch_idx` (typically `net, dataloader, optimizer, loss_fn`).
        train_batches_fn_kwargs: Extra keyword arguments forwarded to `train_batches_fn`.
        logger: Logger used for profiling diagnostics; defaults to a module logger.
        skip_first: Number of steps to skip before the start of profiling cycles.
        wait: Number of steps kept inactive at the start of each profiling cycle.
        warmup: Number of steps used to warm up the profiler in each cycle.
        active: Number of steps actively recorded in each profiling cycle.
        repeat: Number of times the wait/warmup/active cycle repeats.
        record_shapes: Whether to capture operator input shapes.
        profile_memory: Whether to track model tensor memory usage.
        with_stack: Whether to include Python stack traces.
        trace_dir: Output directory for profiler reports and traces.

    Returns:
        Profiler handle with captured profiling data.
    """
    if logger is None:
        logger = logging.getLogger("dlk.opt.profile.profile_train_batches")

    reserved_kwargs = {"logger", "batch_finalize_fn", "max_batches"}
    if reserved_kwargs & train_batches_fn_kwargs.keys():
        raise ValueError(
            f"train_batches_fn_kwargs must not set {reserved_kwargs}; "
            "the profiler sets these internally"
        )

    # select profiler activities for available hardware
    activities, device = _select_profiler_activities()

    # configure a periodic profiling schedule
    schedule = torch.profiler.schedule(
        skip_first=skip_first,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )
    max_batches = skip_first + (wait + warmup + active) * repeat

    # run training with profiler stepping hooks
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda p: trace_handler(
            p,
            logger,
            device=device,
            profile_memory=profile_memory,
            table_row_limit=20,
            trace_dir=trace_dir,
        ),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:

        def batch_finalize_fn(_batch_idx: int) -> None:
            """Signal the end of each training step to the profiler."""
            prof.step()

        epoch_idx = 0
        batch_dlog = train_batches_fn(
            epoch_idx,
            *train_batches_fn_args,
            **train_batches_fn_kwargs,
            logger=logger,
            batch_finalize_fn=batch_finalize_fn,
            max_batches=max_batches,
        )
        n_batches = _infer_profiled_batch_count(batch_dlog, default=max_batches)

        if n_batches < max_batches:
            logger.warning(
                f"Expected {max_batches} batches for profiling, got {n_batches} batches"
            )

        return prof
