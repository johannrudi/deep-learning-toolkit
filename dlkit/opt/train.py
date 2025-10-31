"""Training loops for feed-forward networks."""

import logging, math, os, timeit
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime


def _checkpoint_path(checkpoint_dir, n_epochs, epoch):
    n_digits = int(math.ceil(math.log10(1.01 * n_epochs)))
    fmt = "e{:0" + str(n_digits) + "d}.pt"
    filename = fmt.format(epoch)
    return os.path.join(checkpoint_dir, filename)


def _checkpoint_save(model, filepath, epoch, optimizer):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filepath,
    )


def _dlog_train_epoch_initialize(n_epochs):
    dlog = {}
    for key in ["loss_mean", "loss_std"]:
        dlog[key] = np.empty((n_epochs,))
    dlog["batch_dlog"] = []
    return dlog


def _dlog_train_epoch_update(dlog, epoch_idx, batch_dlog):
    for key in ["loss_mean", "loss_std"]:
        dlog[key][epoch_idx] = batch_dlog[key]
    dlog["batch_dlog"].append(batch_dlog)


def _dlog_train_epoch_finalize(dlog, time_train):
    dlog["time_train"] = time_train


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
    logger=logging.getLogger("dlkit.opt.train_epochs"),
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
    epoch_dlog = _dlog_train_epoch_initialize(n_epochs)
    # set checkpoint directory; create if it doesn't exist
    if checkpoint_epochs is not None:
        assert 1 <= checkpoint_epochs, checkpoint_epochs
        assert checkpoint_dir is not None
        checkpoint_time = datetime.now().strftime("%Y-%m-%d_t%H%M%S")
        checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    # <code id="training_loop_over_epochs">
    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs), desc="epochs"):
        # initialize epoch
        if epoch_initialize_fn:
            epoch_initialize_fn(epoch_idx)
        # save checkpoint
        if checkpoint_epochs is not None and (epoch_idx % checkpoint_epochs == 0):
            path = _checkpoint_path(checkpoint_dir, n_epochs, epoch=epoch_idx)
            logger.debug("epoch {:6d}, save checkpoint to {}".format(epoch_idx, path))
            _checkpoint_save(net, path, epoch=epoch_idx, optimizer=optimizer)
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
                lr_current = "{:.6e}".format(lr_current[0])
            else:
                lr_current = str(lr_current)
            logger.debug("epoch {:6d}, learning_rate {}".format(epoch_idx, lr_current))
            lr_scheduler.step()
        # log
        _dlog_train_epoch_update(epoch_dlog, epoch_idx, batch_dlog)
        logger.info(
            "epoch {:6d}, loss_mean {:.6e} std {:.3e}".format(
                epoch_idx, batch_dlog["loss_mean"], batch_dlog["loss_std"]
            )
        )
        # finalize epoch
        if epoch_finalize_fn:
            epoch_finalize_fn(epoch_idx)
    # save checkpoint---after training
    if checkpoint_epochs is not None:
        path = _checkpoint_path(checkpoint_dir, n_epochs, epoch=n_epochs)
        logger.debug("epoch {:6d}, save checkpoint to {}".format(n_epochs, path))
        _checkpoint_save(net, path, epoch=n_epochs, optimizer=optimizer)
    # call validation function---after training
    if validation_fn is not None:
        validation_fn(n_epochs, net)
    time_train = timeit.default_timer() - time_train
    # </code>
    # finalize and return log
    _dlog_train_epoch_finalize(epoch_dlog, time_train)
    return epoch_dlog


def _dlog_train_batch_initialize(n_batches):
    dlog = {}
    for key in ["loss"]:
        dlog[key] = np.empty((n_batches,))
    return dlog


def _dlog_train_batch_update(dlog, batch_idx, loss):
    dlog["loss"][batch_idx] = loss


def _dlog_train_batch_finalize(dlog):
    is_valid = np.logical_not(np.isnan(dlog["loss"]))
    dlog["loss_mean"] = np.mean(dlog["loss"][is_valid])
    dlog["loss_std"] = np.std(dlog["loss"][is_valid])


def train_batches(
    epoch_idx,
    net,
    dataloader,
    optimizer,
    loss_fn,
    device=None,
    inputs_transform_fn=None,
    targets_transform_fn=None,
    logger=logging.getLogger("dlkit.opt.train_batches"),
    batch_initialize_fn=None,
    batch_finalize_fn=None,
    max_batches=None,
):
    """Runs training loop over batches."""
    if max_batches is None:
        max_batches = len(dataloader)
    batch_dlog = _dlog_train_batch_initialize(max_batches)
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
        _dlog_train_batch_update(batch_dlog, batch_idx, loss_v)
        logger.debug("batch {:6d}, loss {:.6e}".format(batch_idx, loss_v))
        # finalize batch
        if batch_finalize_fn:
            batch_finalize_fn(batch_idx)
    # </code>
    # finalize and return log
    _dlog_train_batch_finalize(batch_dlog)
    return batch_dlog
