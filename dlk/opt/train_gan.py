"""Training loops for adversarial learning with Generative Adversarial Nets."""

import logging
import pathlib
import timeit
from collections.abc import Callable
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from dlk.opt.utils import (checkpoint_path, checkpoint_save,
                           train_dlog_batch_finalize,
                           train_dlog_batch_initialize,
                           train_dlog_batch_update, train_dlog_epoch_finalize,
                           train_dlog_epoch_initialize,
                           train_dlog_epoch_update)

DLOG_BASENAMES = [
    "g_loss",
    "d_pre_loss",
    "d_pre_loss_g",
    "d_pre_reg",
    "d_pre_grad_norm",
    "d_post_loss",
    "d_post_loss_g",
    "d_post_reg",
    "d_post_grad_norm",
]


def train_epochs(
    n_epochs: int,
    g_net: torch.nn.Module,
    d_net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    z_sample_fn: Callable,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    d_reg_fn: Callable | None = None,
    d_opt_pre: int = 1,
    d_opt_post: int = 1,
    g_opt_freq: int = 1,
    validation_fn: Callable | None = None,
    g_lr_scheduler=None,
    d_lr_scheduler=None,
    device: torch.device | None = None,
    logger: logging.Logger = logging.getLogger("dlk.opt.train_epochs"),
    checkpoint_epochs: int | None = None,
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """Runs training loop over epochs.

    Checkpointing saves networks `g_net`, `d_net` and both optimizer states at
    every epoch divisible by `checkpoint_epochs`. Setting `checkpoint_epochs=None`
    turns off checkpointing. Checkpointing will create a new dir under
    `checkpoint_dir/` followed by a directory with date and time of training start.
    """
    DLOG_TAGS = [f"{name}_mean" for name in DLOG_BASENAMES]
    DLOG_TAGS += [f"{name}_std" for name in DLOG_BASENAMES]
    epoch_dlog = train_dlog_epoch_initialize(n_epochs, DLOG_TAGS)

    # set checkpoint directory; create if it doesn't exist
    if checkpoint_epochs is not None:
        assert 1 <= checkpoint_epochs, checkpoint_epochs
        assert checkpoint_dir is not None
        checkpoint_time = datetime.now().strftime("%Y-%m-%d_t%H%M%S")
        checkpoint_dir_ = pathlib.Path(checkpoint_dir) / checkpoint_time
        checkpoint_dir_.mkdir(parents=True, exist_ok=True)

    # <training_loop_over_epochs>
    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs), desc="epochs"):
        # save checkpoint
        if checkpoint_epochs is not None and (epoch_idx % checkpoint_epochs == 0):
            for tag_, net_, opt_ in zip(
                ["g", "d"], [g_net, d_net], [g_optimizer, d_optimizer]
            ):
                path = checkpoint_path(
                    checkpoint_dir_, n_epochs, prefix=f"{tag_}-net", epoch=epoch_idx
                )
                logger.debug(f"epoch {epoch_idx:6d}, save checkpoint to '{path}'")
                checkpoint_save(net_, path, epoch=epoch_idx, optimizer=opt_)

        # call validation function
        if validation_fn is not None:
            validation_fn(epoch_idx, g_net, d_net)

        # train on batches
        batch_dlog = train_batches(
            epoch_idx,
            g_net,
            d_net,
            dataloader,
            z_sample_fn,
            g_optimizer,
            d_optimizer,
            loss_fn,
            d_reg_fn=d_reg_fn,
            d_opt_pre=d_opt_pre,
            d_opt_post=d_opt_post,
            g_opt_freq=g_opt_freq,
            device=device,
            logger=logger,
        )

        # update the learning rate schedulers
        if g_lr_scheduler is not None:
            g_lr_current = g_lr_scheduler.get_last_lr()
            if 1 == len(g_lr_current):
                g_lr_current = f"{g_lr_current[0]:.6e}"
            else:
                g_lr_current = str(g_lr_current)
            g_lr_scheduler.step()
        else:
            g_lr_current = "n/a"
        if d_lr_scheduler is not None:
            d_lr_current = d_lr_scheduler.get_last_lr()
            if 1 == len(d_lr_current):
                d_lr_current = f"{d_lr_current[0]:.6e}"
            else:
                d_lr_current = str(d_lr_current)
            d_lr_scheduler.step()
        else:
            d_lr_current = "n/a"
        if g_lr_scheduler is not None or d_lr_scheduler is not None:
            logger.debug(
                f"epoch {epoch_idx:6d}, g_learning_rate {g_lr_current}, d_learning_rate {d_lr_current}"
            )

        # log
        train_dlog_epoch_update(epoch_dlog, epoch_idx, DLOG_TAGS, batch_dlog)
        logger.info(
            f"epoch {epoch_idx:6d}, "
            f"d_loss pre mean {batch_dlog['d_pre_loss_mean']:.6e} std {batch_dlog['d_pre_loss_std']:.3e}, "
            f"g_loss mean {batch_dlog['g_loss_mean']:.6e} std {batch_dlog['g_loss_std']:.3e}, "
            f"d_loss post mean {batch_dlog['d_post_loss_mean']:.6e} std {batch_dlog['d_post_loss_std']:.3e}, "
        )

    # save checkpoint---after training
    if checkpoint_epochs is not None:
        for tag_, net_, opt_ in zip(
            ["g", "d"], [g_net, d_net], [g_optimizer, d_optimizer]
        ):
            path = checkpoint_path(
                checkpoint_dir_, n_epochs, prefix=f"{tag_}-net", epoch=n_epochs
            )
            logger.debug(f"epoch {n_epochs:6d}, save checkpoint to '{path}'")
            checkpoint_save(net_, path, epoch=n_epochs, optimizer=opt_)

    # call validation function---after training
    if validation_fn is not None:
        validation_fn(n_epochs, g_net, d_net)
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
    logger.info(
        f"training time {time_train:g} sec, time/epoch {time_train / n_epochs:g} sec"
    )
    logger.info(
        f"time/step {time_train / n_steps:g} sec, samples/sec {n_samples / time_train:g} sec"
    )

    # return log
    return epoch_dlog


def _train_step_discriminator(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    z_sample_fn: Callable,
    g_net: torch.nn.Module,
    d_net: torch.nn.Module,
    d_optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    d_reg_fn: Callable | None = None,
    dlog_item: dict | None = None,
) -> float:
    """Trains discriminator network."""
    # sample from latent space for generator
    batch_size = y_data.size(0)
    z = z_sample_fn(batch_size)

    # generate outputs with `g_net`
    x_gen = g_net(y_data, z).detach()

    # evalutate discriminator (begin AD)
    d_optimizer.zero_grad()
    d_outputs_gen = d_net(x_gen, y_data)
    d_outputs_data = d_net(x_data, y_data)

    # evaluate discriminator loss
    # Note: output must have correct sign for minimization
    d_loss, d_loss_g = loss_fn(d_outputs_gen, d_outputs_data)
    d_reg_dlog = dict()
    if d_reg_fn is not None:
        d_reg = d_reg_fn(
            d_net,
            x_gen[np.random.randint(0, x_gen.size(0))],
            x_data,
            y_data,
            dlog=d_reg_dlog,
        )
    else:
        d_reg = torch.tensor(0.0)
    loss = d_loss + d_reg

    # calculate derivatives (end AD) and update network parameters
    loss.backward()
    d_optimizer.step()

    # output values
    if dlog_item is not None:
        dlog_item["d_loss"] = d_loss.item()
        dlog_item["d_loss_g"] = d_loss_g.item()
        dlog_item["d_reg"] = d_reg.item()
        for key, val in d_reg_dlog.items():
            dlog_item["d_" + key] = val
    return loss.item()


def _train_step_generator(
    y_data: torch.Tensor,
    z_sample_fn: Callable,
    g_net: torch.nn.Module,
    d_net: torch.nn.Module,
    g_optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    dlog_item: dict | None = None,
) -> float:
    """Trains generator network."""
    # sample from latent space for generator
    batch_size = y_data.size(0)
    z = z_sample_fn(batch_size)

    # generate outputs with `g_net` (begin AD)
    g_optimizer.zero_grad()
    x_gen = g_net(y_data, z)

    # evalutate discriminator
    d_outputs_gen = d_net(x_gen, y_data)

    # evaluate discriminator loss
    g_loss, _ = loss_fn(None, d_outputs_gen)  # pass generated outputs as data/truth
    loss = g_loss

    # calculate derivatives (end AD) and update network parameters
    loss.backward()
    g_optimizer.step()

    # output values
    if dlog_item is not None:
        dlog_item["g_loss"] = g_loss.item()
    return loss.item()


def train_batches(
    epoch_idx: int,
    g_net: torch.nn.Module,
    d_net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    z_sample_fn: Callable,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    d_reg_fn: Callable | None = None,
    d_opt_pre: int = 1,
    d_opt_post: int = 1,
    g_opt_freq: int = 1,
    device: torch.device | None = None,
    logger=logging.getLogger("dlk.opt.train_batches"),
    batch_initialize_fn: Callable | None = None,
    batch_finalize_fn: Callable | None = None,
    max_batches: int | None = None,
) -> dict:
    """Runs training loop over batches."""
    DLOG_TAGS = DLOG_BASENAMES
    if max_batches is None:
        max_batches = len(dataloader)
    batch_dlog = train_dlog_batch_initialize(max_batches, DLOG_TAGS, save_list=False)

    # <training_loop_over_batches>
    for batch_idx, data in enumerate(dataloader):
        if max_batches <= batch_idx:
            break
        # initialize batch
        if batch_initialize_fn:
            batch_initialize_fn(batch_idx)

        # set networks to training mode
        g_net.train()
        d_net.train()

        # get input and target tensors
        x_data, y_data = data
        if device is not None:
            x_data = x_data.to(device)
            y_data = y_data.to(device)

        dlog_item = dict()

        # pre-train discriminator network
        dlog_buf = dict()
        for i in range(d_opt_pre):
            d_loss_reg_v = _train_step_discriminator(
                x_data,
                y_data,
                z_sample_fn,
                g_net,
                d_net,
                d_optimizer,
                loss_fn,
                d_reg_fn=d_reg_fn,
                dlog_item=dlog_buf,
            )
            logger.debug(
                f"epoch {epoch_idx:6d}, batch {batch_idx:6d}, pre {i:2d}, "
                f"d_loss_reg {d_loss_reg_v:.6e}, d_loss {dlog_buf['d_loss']:.6e}"
            )
        for k, v in dlog_buf.items():
            dlog_item[f"d_pre_{k.removeprefix('d_')}"] = v

        # train generator network
        if 0 == batch_idx % g_opt_freq:
            g_loss_v = _train_step_generator(
                y_data,
                z_sample_fn,
                g_net,
                d_net,
                g_optimizer,
                loss_fn,
                dlog_item=dlog_item,
            )
            logger.debug(
                f"epoch {epoch_idx:6d}, batch {batch_idx:6d}, g_loss {g_loss_v:.6e}"
            )

        # post-train discriminator network
        dlog_buf = dict()
        for _ in range(d_opt_post):
            d_loss_reg_v = _train_step_discriminator(
                x_data,
                y_data,
                z_sample_fn,
                g_net,
                d_net,
                d_optimizer,
                loss_fn,
                d_reg_fn=d_reg_fn,
                dlog_item=dlog_buf,
            )
            logger.debug(
                f"epoch {epoch_idx:6d}, batch {batch_idx:6d}, post {i}, "
                f"d_loss_reg {d_loss_reg_v:.6e}, d_loss {dlog_buf['d_loss']:.6e}"
            )
        for k, v in dlog_buf.items():
            dlog_item[f"d_post_{k.removeprefix('d_')}"] = v

        # log
        train_dlog_batch_update(batch_dlog, batch_idx, dlog_item)

        # finalize batch
        if batch_finalize_fn:
            batch_finalize_fn(batch_idx)
    # </training_loop_over_batches>

    # finalize and return log
    train_dlog_batch_finalize(batch_dlog, DLOG_TAGS)
    return batch_dlog
