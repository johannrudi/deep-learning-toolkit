"""Provide reusable epoch- and batch-level training loops for supervised models."""

import enum
import logging
import pathlib
import time
from datetime import datetime

import torch
from torch.profiler import record_function
from tqdm import tqdm

from dlk.opt import distributed
from dlk.opt.utils import (
    BatchHookFn,
    DataLoaderType,
    EpochHookFn,
    InputsTransformFn,
    LossFn,
    LRSchedulerType,
    TensorTransformFn,
    TrainLog,
    ValidationFn,
    autocast_context,
    checkpoint_path,
    checkpoint_save,
    format_seconds,
    tqdm_disable,
    train_dlog_batch_all_reduce,
    train_dlog_batch_finalize,
    train_dlog_batch_initialize,
    train_dlog_batch_update,
    train_dlog_epoch_finalize,
    train_dlog_epoch_initialize,
    train_dlog_epoch_update,
    transfer_non_blocking,
)

DLOG_BASENAMES = [
    "loss",
    "time_step",
]


class RecordFunctionName(enum.StrEnum):
    """Labels for `torch.profiler.record_function` regions."""

    DATA_H2D = "data_h2d"
    DATA_TRANSFORM = "data_transform"
    OPTIMIZER_ZERO = "optimizer_zero"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER_STEP = "optimizer_step"
    LOG = "log"


def train_epochs(
    n_epochs: int,
    net: torch.nn.Module,
    dataloader: DataLoaderType,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    validation_fn: ValidationFn | None = None,
    lr_scheduler: LRSchedulerType | None = None,
    device: torch.device | None = None,
    inputs_transform_fn: InputsTransformFn | None = None,
    targets_transform_fn: TensorTransformFn | None = None,
    logger: logging.Logger | None = None,
    checkpoint_epochs: int | None = None,
    checkpoint_dir: str = "checkpoints",
    epoch_initialize_fn: EpochHookFn | None = None,
    epoch_finalize_fn: EpochHookFn | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> TrainLog:
    """Run the training loop over epochs.

    Checkpointing saves model and optimizer states at every epoch divisible by
    `checkpoint_epochs`. Setting `checkpoint_epochs=None` disables checkpointing.
    When enabled, checkpoints are written to a run-specific directory under
    `checkpoint_dir`.

    Under distributed training (DDP), checkpointing and validation run on the
    main process only, `validation_fn` receives the unwrapped model, the
    distributed sampler's epoch is advanced automatically, and loss statistics
    are reduced exactly across all processes.

    Host-to-device copies are asynchronous when the dataloader pins its batches
    and `device` is an accelerator; see `train_batches`.

    Args:
        n_epochs: Number of epochs to train.
        net: Model to optimize.
        dataloader: Iterable of `(inputs, targets)` training batches.
        optimizer: Optimizer used to update model parameters.
        loss_fn: Callable that maps `(outputs, targets)` to a scalar loss tensor.
        validation_fn: Optional callback invoked as `validation_fn(epoch_idx, net=net)`
            before each epoch and once after training.
        lr_scheduler: Optional learning-rate scheduler with `get_last_lr` and `step`.
        device: Optional device used to move inputs and targets.
        inputs_transform_fn: Optional transform applied to each input batch.
        targets_transform_fn: Optional transform applied to each target batch.
        logger: Logger used for progress and metrics reporting.
        checkpoint_epochs: Checkpoint period in epochs; disable with `None`.
        checkpoint_dir: Root directory used for checkpoint files.
        epoch_initialize_fn: Optional callback invoked at the start of each epoch.
        epoch_finalize_fn: Optional callback invoked at the end of each epoch.
        autocast_dtype: Compute dtype for the autocast forward pass. Use
            `torch.bfloat16` for mixed precision, `None` or `torch.float32` for
            full precision.

    Returns:
        Training log dictionary with per-epoch metrics and run timing.

    Raises:
        ValueError: If `n_epochs < 1` or `checkpoint_epochs < 1` when provided.
    """
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
    if logger is None:
        logger = logging.getLogger("dlk.opt.train.train_epochs")

    dlog_tags = [f"{name}_mean" for name in DLOG_BASENAMES]
    dlog_tags += [f"{name}_std" for name in DLOG_BASENAMES]
    epoch_dlog = train_dlog_epoch_initialize(n_epochs, dlog_tags)

    # set checkpoint directory on the main process only; create if it doesn't exist
    checkpoint_dir_: pathlib.Path | None = None
    if checkpoint_epochs is not None:
        if checkpoint_epochs < 1:
            raise ValueError(f"checkpoint_epochs must be >= 1, got {checkpoint_epochs}")
        assert checkpoint_dir is not None
        if distributed.is_main_process():
            checkpoint_time = datetime.now().strftime("%Y-%m-%d_t%H%M%S")
            checkpoint_dir_ = pathlib.Path(checkpoint_dir) / checkpoint_time
            checkpoint_dir_.mkdir(parents=True, exist_ok=True)

    # <training_loop_over_epochs>
    time_train = time.perf_counter()
    with tqdm(range(n_epochs), desc="epochs", disable=tqdm_disable()) as pbar:
        for epoch_idx in pbar:
            # initialize epoch
            if epoch_initialize_fn:
                epoch_initialize_fn(epoch_idx)

            # advance the distributed sampler's epoch (no-op otherwise)
            distributed.sampler_set_epoch(dataloader, epoch_idx)

            # save checkpoint (main process only)
            if (
                checkpoint_epochs is not None
                and checkpoint_dir_ is not None
                and epoch_idx % checkpoint_epochs == 0
            ):
                path = checkpoint_path(
                    checkpoint_dir_, n_epochs, prefix="net", epoch=epoch_idx
                )
                logger.debug(f"epoch {epoch_idx:4d}, save checkpoint to '{path}'")
                checkpoint_save(net, path, epoch=epoch_idx, optimizer=optimizer)

            # call validation function (main process only, with unwrapped model)
            if validation_fn is not None and distributed.is_main_process():
                validation_fn(epoch_idx, net=distributed.unwrap_net(net))

            # train on batches
            batch_dlog = train_batches(
                epoch_idx,
                net,
                dataloader,
                optimizer,
                loss_fn,
                device=device,
                inputs_transform_fn=inputs_transform_fn,
                targets_transform_fn=targets_transform_fn,
                logger=logger,
                autocast_dtype=autocast_dtype,
            )

            # update the learning rate scheduler
            if lr_scheduler is not None:
                lr_current = lr_scheduler.get_last_lr()
                if 1 == len(lr_current):
                    lr_current = f"{lr_current[0]:.6e}"
                else:
                    lr_current = str(lr_current)
                logger.debug(f"epoch {epoch_idx:4d}, learning_rate {lr_current}")
                lr_scheduler.step()

            # log
            train_dlog_epoch_update(epoch_dlog, epoch_idx, dlog_tags, batch_dlog)
            logger.info(
                f"epoch {epoch_idx:4d}, "
                f"loss mean {batch_dlog['loss_mean']:.6e} "
                f"std {batch_dlog['loss_std']:.3e}, "
                f"time/step mean {format_seconds(batch_dlog['time_step_mean'])} "
                f"std {format_seconds(batch_dlog['time_step_std'])}"
            )

            # finalize epoch
            if epoch_finalize_fn:
                epoch_finalize_fn(epoch_idx)

    # save checkpoint---after training (main process only)
    if checkpoint_epochs is not None and checkpoint_dir_ is not None:
        path = checkpoint_path(checkpoint_dir_, n_epochs, prefix="net", epoch=n_epochs)
        logger.debug(f"epoch {n_epochs:4d}, save checkpoint to '{path}'")
        checkpoint_save(net, path, epoch=n_epochs, optimizer=optimizer)

    # call validation function---after training (main process only)
    if validation_fn is not None and distributed.is_main_process():
        validation_fn(n_epochs, net=distributed.unwrap_net(net))
    time_train = time.perf_counter() - time_train
    # </training_loop_over_epochs>

    # finalize log
    train_dlog_epoch_finalize(epoch_dlog, time_train)

    # print statistics; sample counts are global across all processes
    n_steps = n_epochs * len(dataloader)
    n_samples = (
        n_steps * dataloader.batch_size * distributed.get_world_size()
        if dataloader.batch_size is not None
        else 0
    )
    logger.info(
        f"number of epochs {n_epochs}, optimizer steps {n_steps}, samples processed {n_samples}"
    )
    time_per_epoch = time_train / n_epochs
    time_per_step = time_train / n_steps if n_steps > 0 else float("nan")
    samples_per_second = n_samples / time_train if time_train > 0 else float("nan")
    logger.info(f"training time {time_train:g} sec, time/epoch {time_per_epoch:g} sec")
    logger.info(
        f"time/step {time_per_step:g} sec, samples/sec {samples_per_second:g} sec"
    )

    # return log
    return epoch_dlog


def train_batches(
    epoch_idx: int,
    net: torch.nn.Module,
    dataloader: DataLoaderType,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device | None = None,
    inputs_transform_fn: InputsTransformFn | None = None,
    targets_transform_fn: TensorTransformFn | None = None,
    logger: logging.Logger | None = None,
    batch_initialize_fn: BatchHookFn | None = None,
    batch_finalize_fn: BatchHookFn | None = None,
    max_batches: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> TrainLog:
    """Run the training loop over batches for a single epoch.

    Host-to-device copies are asynchronous (`non_blocking=True`) when the
    dataloader pins its batches (`pin_memory=True`) and `device` is a CUDA or
    XPU device; otherwise they stay synchronous.

    Args:
        epoch_idx: Current epoch index used in logging.
        net: Model to optimize.
        dataloader: Iterable of `(inputs, targets)` training batches.
        optimizer: Optimizer used to update model parameters.
        loss_fn: Callable that maps `(outputs, targets)` to a scalar loss tensor.
        device: Optional device used to move inputs and targets.
        inputs_transform_fn: Optional transform applied to each input batch.
        targets_transform_fn: Optional transform applied to each target batch.
        logger: Logger used for per-batch debug metrics.
        batch_initialize_fn: Optional callback invoked before each batch step.
        batch_finalize_fn: Optional callback invoked after each batch step.
        max_batches: Optional maximum number of batches processed.
        autocast_dtype: Compute dtype for the autocast forward pass. Use
            `torch.bfloat16` for mixed precision, `None` or `torch.float32` for
            full precision.

    Returns:
        Batch-level training log dictionary with aggregate loss statistics.

    Raises:
        ValueError: If `autocast_dtype` is unsupported for autocast.
    """
    if logger is None:
        logger = logging.getLogger("dlk.opt.train.train_batches")
    if max_batches is None:
        max_batches = len(dataloader)

    dlog_tags = DLOG_BASENAMES
    batch_dlog = train_dlog_batch_initialize(max_batches, dlog_tags, save_list=False)

    # overlap host-to-device copies when the dataloader pins its batches
    non_blocking = transfer_non_blocking(dataloader, device)

    # <training_loop_over_batches>
    for batch_idx, data in enumerate(dataloader):
        if max_batches <= batch_idx:
            break

        # initialize batch
        if batch_initialize_fn:
            batch_initialize_fn(batch_idx)

        # set network to training mode
        net.train()

        # start iteration timer
        time_step = time.perf_counter()

        # get input and target tensors
        with record_function(RecordFunctionName.DATA_H2D):
            inputs, targets = data
            if device is not None:
                if isinstance(inputs, tuple):
                    inputs = tuple(
                        x.to(device, non_blocking=non_blocking) for x in inputs
                    )
                else:
                    inputs = inputs.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)

        # transform input and target tensors
        with record_function(RecordFunctionName.DATA_TRANSFORM):
            if inputs_transform_fn is not None:
                inputs = inputs_transform_fn(inputs)
            if targets_transform_fn is not None:
                targets = targets_transform_fn(targets)

        # zero the gradients (begin AD)
        with record_function(RecordFunctionName.OPTIMIZER_ZERO):
            optimizer.zero_grad()

        with record_function(RecordFunctionName.FORWARD):
            with autocast_context(device, autocast_dtype):
                # forward pass; unpack inputs tuple when applicable
                outputs = net(*inputs) if isinstance(inputs, tuple) else net(inputs)

                # calculate loss
                loss = loss_fn(outputs, targets)

        # calculate derivatives (end AD)
        with record_function(RecordFunctionName.BACKWARD):
            loss.backward()

        # update network parameters
        with record_function(RecordFunctionName.OPTIMIZER_STEP):
            optimizer.step()

        # log
        with record_function(RecordFunctionName.LOG):
            loss_v = loss.item()
            time_step = time.perf_counter() - time_step
            train_dlog_batch_update(
                batch_dlog, batch_idx, {"loss": loss_v, "time_step": time_step}
            )
            logger.debug(
                f"epoch {epoch_idx:4d}, batch {batch_idx:4d}, loss {loss_v:.6e}"
            )

        # finalize batch
        if batch_finalize_fn:
            batch_finalize_fn(batch_idx)
    # </training_loop_over_batches>

    # reduce running aggregates across all processes (no-op otherwise)
    train_dlog_batch_all_reduce(batch_dlog, dlog_tags, device=device)

    # finalize and return log
    train_dlog_batch_finalize(batch_dlog, dlog_tags)
    return batch_dlog
