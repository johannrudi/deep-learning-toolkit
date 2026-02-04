"""Training loops for feed-forward networks."""

import logging
import pathlib
import timeit
from datetime import datetime

import torch
from tqdm import tqdm

from dlk.opt.train_utils import (checkpoint_path, checkpoint_save,
                                 train_dlog_batch_finalize,
                                 train_dlog_batch_initialize,
                                 train_dlog_batch_update,
                                 train_dlog_epoch_finalize,
                                 train_dlog_epoch_initialize,
                                 train_dlog_epoch_update)


def train_epochs(
    n_epochs,
    net,
    dataloader,
    optimizer,
    loss_fn,
    validation_fn=None,
    lr_scheduler=None,
    device=None,
    inputs_transform_fn=None,
    targets_transform_fn=None,
    logger=logging.getLogger("dlk.opt.train_epochs"),
    checkpoint_epochs=None,
    checkpoint_dir="checkpoints",
    epoch_initialize_fn=None,
    epoch_finalize_fn=None,
):
    """Runs training loop over epochs.

    Checkpointing saves model and optimizer states at every epoch divisible
    by `checkpoint_epochs`. Setting `checkpoint_epochs=None` turns off
    checkpointing. Checkpointing will create a new dir under `checkpoint_dir/`
    followed by a directory with date and time of training start.
    """
    epoch_dlog = train_dlog_epoch_initialize(n_epochs, ["loss_mean", "loss_std"])

    # set checkpoint directory; create if it doesn't exist
    if checkpoint_epochs is not None:
        assert 1 <= checkpoint_epochs, checkpoint_epochs
        assert checkpoint_dir is not None
        checkpoint_time = datetime.now().strftime("%Y-%m-%d_t%H%M%S")
        checkpoint_dir_ = pathlib.Path(checkpoint_dir) / checkpoint_time
        checkpoint_dir_.mkdir(parents=True, exist_ok=True)

    # <code id="training_loop_over_epochs">
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
    # </code>

    # finalize log
    train_dlog_epoch_finalize(epoch_dlog, time_train)

    # print statistics
    n_steps = n_epochs * len(dataloader)
    n_samples = n_steps * dataloader.batch_size
    logger.info(
        f"number of epochs {n_epochs}, optimizer steps {n_steps}, samples processed {n_samples}"
    )
    logger.info(
        f"training time {time_train:g} sec, time/epoch {time_train / n_epochs:g} sec"
    )
    logger.info(
        f"time/step {time_train / n_steps:g} sec, samples/sec {n_samples / time_train:g} sec"
    )

    # return log
    return epoch_dlog


def train_batches(
    epoch_idx,
    net,
    dataloader,
    optimizer,
    loss_fn,
    device=None,
    inputs_transform_fn=None,
    targets_transform_fn=None,
    logger=logging.getLogger("dlk.opt.train_batches"),
    batch_initialize_fn=None,
    batch_finalize_fn=None,
    max_batches=None,
):
    """Runs training loop over batches."""
    if max_batches is None:
        max_batches = len(dataloader)
    batch_dlog = train_dlog_batch_initialize(max_batches, ["loss"], save_list=False)

    # <code id="training_loop_over_batches">
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
        logger.debug(f"batch {batch_idx:4d}, loss {loss_v:.6e}")

        # finalize batch
        if batch_finalize_fn:
            batch_finalize_fn(batch_idx)
    # </code>

    # finalize and return log
    train_dlog_batch_finalize(batch_dlog, ["loss"])
    return batch_dlog
