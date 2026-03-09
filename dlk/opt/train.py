"""Provide reusable epoch- and batch-level training loops for supervised models."""

import logging
import pathlib
import timeit
from datetime import datetime

import torch
from tqdm import tqdm

from dlk.opt.utils import (
    BatchHookFn,
    EpochHookFn,
    LossFn,
    LRScheduler,
    TensorTransformFn,
    TrainLog,
    ValidationFn,
    checkpoint_path,
    checkpoint_save,
    train_dlog_batch_finalize,
    train_dlog_batch_initialize,
    train_dlog_batch_update,
    train_dlog_epoch_finalize,
    train_dlog_epoch_initialize,
    train_dlog_epoch_update,
)


def train_epochs(
    n_epochs: int,
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    validation_fn: ValidationFn | None = None,
    lr_scheduler: LRScheduler | None = None,
    device: torch.device | None = None,
    inputs_transform_fn: TensorTransformFn | None = None,
    targets_transform_fn: TensorTransformFn | None = None,
    logger: logging.Logger = logging.getLogger("dlk.opt.train_epochs"),
    checkpoint_epochs: int | None = None,
    checkpoint_dir: str = "checkpoints",
    epoch_initialize_fn: EpochHookFn | None = None,
    epoch_finalize_fn: EpochHookFn | None = None,
) -> TrainLog:
    """Run the training loop over epochs.

    Checkpointing saves model and optimizer states at every epoch divisible by
    `checkpoint_epochs`. Setting `checkpoint_epochs=None` disables checkpointing.
    When enabled, checkpoints are written to a run-specific directory under
    `checkpoint_dir`.

    Args:
        n_epochs: Number of epochs to train.
        net: Model to optimize.
        dataloader: Iterable of `(inputs, targets)` training batches.
        optimizer: Optimizer used to update model parameters.
        loss_fn: Callable that maps `(outputs, targets)` to a scalar loss tensor.
        validation_fn: Optional callback invoked as `validation_fn(epoch_idx, net)`.
        lr_scheduler: Optional learning-rate scheduler with `get_last_lr` and `step`.
        device: Optional device used to move inputs and targets.
        inputs_transform_fn: Optional transform applied to each input batch.
        targets_transform_fn: Optional transform applied to each target batch.
        logger: Logger used for progress and metrics reporting.
        checkpoint_epochs: Checkpoint period in epochs; disable with `None`.
        checkpoint_dir: Root directory used for checkpoint files.
        epoch_initialize_fn: Optional callback invoked at the start of each epoch.
        epoch_finalize_fn: Optional callback invoked at the end of each epoch.

    Returns:
        Training log dictionary with per-epoch metrics and run timing.

    Raises:
        ValueError: If `n_epochs < 1` or `checkpoint_epochs < 1` when provided.
    """
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
    epoch_dlog = train_dlog_epoch_initialize(n_epochs, ["loss_mean", "loss_std"])

    # set checkpoint directory; create if it doesn't exist
    if checkpoint_epochs is not None:
        if checkpoint_epochs < 1:
            raise ValueError(f"checkpoint_epochs must be >= 1, got {checkpoint_epochs}")
        assert checkpoint_dir is not None
        checkpoint_time = datetime.now().strftime("%Y-%m-%d_t%H%M%S")
        checkpoint_dir_ = pathlib.Path(checkpoint_dir) / checkpoint_time
        checkpoint_dir_.mkdir(parents=True, exist_ok=True)

    # <training_loop_over_epochs>
    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs), desc="epochs"):
        # initialize epoch
        if epoch_initialize_fn:
            epoch_initialize_fn(epoch_idx)

        # save checkpoint
        if checkpoint_epochs is not None and (epoch_idx % checkpoint_epochs == 0):
            path = checkpoint_path(
                checkpoint_dir_, n_epochs, prefix="net", epoch=epoch_idx
            )
            logger.debug(f"epoch {epoch_idx:4d}, save checkpoint to '{path}'")
            checkpoint_save(net, path, epoch=epoch_idx, optimizer=optimizer)

        # call validation function
        if validation_fn is not None:
            validation_fn(epoch_idx, net)

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
        train_dlog_epoch_update(
            epoch_dlog, epoch_idx, ["loss_mean", "loss_std"], batch_dlog
        )
        logger.info(
            f"epoch {epoch_idx:4d}, "
            f"loss mean {batch_dlog['loss_mean']:.6e} std {batch_dlog['loss_std']:.3e}"
        )

        # finalize epoch
        if epoch_finalize_fn:
            epoch_finalize_fn(epoch_idx)

    # save checkpoint---after training
    if checkpoint_epochs is not None:
        path = checkpoint_path(checkpoint_dir_, n_epochs, prefix="net", epoch=n_epochs)
        logger.debug(f"epoch {n_epochs:4d}, save checkpoint to '{path}'")
        checkpoint_save(net, path, epoch=n_epochs, optimizer=optimizer)

    # call validation function---after training
    if validation_fn is not None:
        validation_fn(n_epochs, net)
    time_train = timeit.default_timer() - time_train
    # </training_loop_over_epochs>

    # finalize log
    train_dlog_epoch_finalize(epoch_dlog, time_train)

    # print statistics
    n_steps = n_epochs * len(dataloader)
    n_samples = (
        n_steps * dataloader.batch_size if dataloader.batch_size is not None else 0
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
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device | None = None,
    inputs_transform_fn: TensorTransformFn | None = None,
    targets_transform_fn: TensorTransformFn | None = None,
    logger: logging.Logger = logging.getLogger("dlk.opt.train_batches"),
    batch_initialize_fn: BatchHookFn | None = None,
    batch_finalize_fn: BatchHookFn | None = None,
    max_batches: int | None = None,
) -> TrainLog:
    """Run the training loop over batches for a single epoch.

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

    Returns:
        Batch-level training log dictionary with aggregate loss statistics.
    """
    if max_batches is None:
        max_batches = len(dataloader)
    batch_dlog = train_dlog_batch_initialize(max_batches, ["loss"], save_list=False)

    # <training_loop_over_batches>
    for batch_idx, data in enumerate(dataloader):
        if max_batches <= batch_idx:
            break

        # initialize batch
        if batch_initialize_fn:
            batch_initialize_fn(batch_idx)

        # set network to training mode
        net.train()

        # get input and target tensors
        inputs, targets = data
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)
        if inputs_transform_fn is not None:
            inputs = inputs_transform_fn(inputs)
        if targets_transform_fn is not None:
            targets = targets_transform_fn(targets)

        # zero the gradients (begin AD)
        optimizer.zero_grad()
        # forward pass
        outputs = net(inputs)
        # calculate loss
        loss = loss_fn(outputs, targets)
        # calculate derivatives (end AD)
        loss.backward()
        # update network parameters
        optimizer.step()

        # log
        loss_v = loss.item()
        train_dlog_batch_update(batch_dlog, batch_idx, {"loss": loss_v})
        logger.debug(f"epoch {epoch_idx:4d}, batch {batch_idx:4d}, loss {loss_v:.6e}")

        # finalize batch
        if batch_finalize_fn:
            batch_finalize_fn(batch_idx)
    # </training_loop_over_batches>

    # finalize and return log
    train_dlog_batch_finalize(batch_dlog, ["loss"])
    return batch_dlog
