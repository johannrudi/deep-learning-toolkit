"""Train GAN models across epochs and batches with checkpointing and structured logging."""

import logging
import math
import pathlib
import sys
import timeit
from collections.abc import Callable
from datetime import datetime
from typing import Protocol, TypeAlias

import torch
from tqdm import tqdm

from dlk.opt.utils import (
    BatchHookFn,
    EpochHookFn,
    LRScheduler,
    TrainLog,
    checkpoint_path,
    checkpoint_save,
    train_dlog_batch_finalize,
    train_dlog_batch_initialize,
    train_dlog_batch_update,
    train_dlog_epoch_finalize,
    train_dlog_epoch_initialize,
    train_dlog_epoch_update,
)

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


# --------------------------------------
# Types
# --------------------------------------

GANLossFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor | None]
]
GANValidationFn: TypeAlias = Callable[[int, torch.nn.Module, torch.nn.Module], None]
LatentSampleFn: TypeAlias = Callable[[int], torch.Tensor]


class DiscriminatorRegularizerFn(Protocol):
    """Protocol for discriminator regularizers that optionally emit logs."""

    def __call__(
        self,
        d_net: torch.nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        *,
        dlog: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Return a scalar regularization penalty for discriminator updates."""
        ...


# --------------------------------------


def train_epochs(
    n_epochs: int,
    g_net: torch.nn.Module,
    d_net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    z_sample_fn: LatentSampleFn,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    loss_fn: GANLossFn,
    d_reg_fn: DiscriminatorRegularizerFn | None = None,
    d_opt_pre: int = 1,
    d_opt_post: int = 1,
    g_opt_freq: int = 1,
    validation_fn: GANValidationFn | None = None,
    g_lr_scheduler: LRScheduler | None = None,
    d_lr_scheduler: LRScheduler | None = None,
    device: torch.device | None = None,
    logger: logging.Logger = logging.getLogger("dlk.opt.train_epochs"),
    checkpoint_epochs: int | None = None,
    checkpoint_dir: str = "checkpoints",
    epoch_initialize_fn: EpochHookFn | None = None,
    epoch_finalize_fn: EpochHookFn | None = None,
) -> TrainLog:
    """Run the GAN training loop over epochs.

    Checkpointing saves `g_net`, `d_net`, and optimizer states at epochs divisible
    by `checkpoint_epochs`. If checkpointing is enabled, a timestamped directory is
    created under `checkpoint_dir`.

    Args:
        n_epochs: Number of epochs to train.
        g_net: Generator network.
        d_net: Discriminator network.
        dataloader: Iterable of training batches `(x_data, y_data)`.
        z_sample_fn: Function that samples latent vectors for a batch size.
        g_optimizer: Optimizer for generator parameters.
        d_optimizer: Optimizer for discriminator parameters.
        loss_fn: Adversarial loss function used by both networks.
        d_reg_fn: Optional discriminator regularizer.
        d_opt_pre: Number of discriminator updates before generator update.
        d_opt_post: Number of discriminator updates after generator update.
        g_opt_freq: Frequency of generator updates in batch steps.
        validation_fn: Optional callback invoked before each epoch and once after
            training.
        g_lr_scheduler: Optional generator learning-rate scheduler.
        d_lr_scheduler: Optional discriminator learning-rate scheduler.
        device: Optional device to move batch tensors to.
        logger: Logger instance for training progress.
        checkpoint_epochs: Epoch interval for checkpoint saves. `None` disables
            checkpointing.
        checkpoint_dir: Parent directory where checkpoint runs are stored.
        epoch_initialize_fn: Optional callback invoked at the start of each epoch.
        epoch_finalize_fn: Optional callback invoked at the end of each epoch.

    Returns:
        Aggregated epoch-level training diagnostics.

    Raises:
        ValueError: If `n_epochs < 1` or `checkpoint_epochs < 1` when provided.
    """
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
    DLOG_TAGS = [f"{name}_mean" for name in DLOG_BASENAMES]
    DLOG_TAGS += [f"{name}_std" for name in DLOG_BASENAMES]
    epoch_dlog = train_dlog_epoch_initialize(n_epochs, DLOG_TAGS)

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
    with tqdm(range(n_epochs), desc="epochs", disable=not sys.stdout.isatty()) as pbar:
        for epoch_idx in pbar:
            # initialize epoch
            if epoch_initialize_fn:
                epoch_initialize_fn(epoch_idx)

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
                epoch_idx=epoch_idx,
                g_net=g_net,
                d_net=d_net,
                dataloader=dataloader,
                z_sample_fn=z_sample_fn,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                loss_fn=loss_fn,
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

            # finalize epoch
            if epoch_finalize_fn:
                epoch_finalize_fn(epoch_idx)

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
    z_sample_fn: LatentSampleFn,
    g_net: torch.nn.Module,
    d_net: torch.nn.Module,
    d_optimizer: torch.optim.Optimizer,
    loss_fn: GANLossFn,
    d_reg_fn: DiscriminatorRegularizerFn | None = None,
    dlog_item: dict[str, float] | None = None,
) -> float:
    """Run one discriminator optimization step.

    Args:
        x_data: Real input samples.
        y_data: Conditioning inputs paired with `x_data`.
        z_sample_fn: Function that samples latent vectors for the generator.
        g_net: Generator network.
        d_net: Discriminator network.
        d_optimizer: Optimizer for discriminator parameters.
        loss_fn: Adversarial loss callable for discriminator training.
        d_reg_fn: Optional discriminator regularizer.
        dlog_item: Optional dictionary updated with scalar diagnostics.

    Returns:
        Total discriminator loss value after regularization.
    """
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
    # NOTE: output must have correct sign for minimization
    d_loss, d_loss_g = loss_fn(d_outputs_gen, d_outputs_data)
    d_reg_dlog: dict[str, float] = {}
    if d_reg_fn is not None:
        random_idx = int(torch.randint(0, x_gen.size(0), size=()).item())
        d_reg = d_reg_fn(
            d_net,
            x_gen[random_idx],
            x_data,
            y_data,
            dlog=d_reg_dlog,
        )
    else:
        d_reg = d_loss.new_tensor(0.0)
    loss = d_loss + d_reg

    # calculate derivatives (end AD) and update network parameters
    loss.backward()
    d_optimizer.step()

    # output values
    if dlog_item is not None:
        dlog_item["d_loss"] = d_loss.item()
        dlog_item["d_loss_g"] = d_loss_g.item() if d_loss_g is not None else math.nan
        dlog_item["d_reg"] = d_reg.item()
        for key, val in d_reg_dlog.items():
            dlog_item["d_" + key] = val
    return loss.item()


def _train_step_generator(
    y_data: torch.Tensor,
    z_sample_fn: LatentSampleFn,
    g_net: torch.nn.Module,
    d_net: torch.nn.Module,
    g_optimizer: torch.optim.Optimizer,
    loss_fn: GANLossFn,
    dlog_item: dict[str, float] | None = None,
) -> float:
    """Run one generator optimization step.

    Args:
        y_data: Conditioning inputs for generation.
        z_sample_fn: Function that samples latent vectors for the generator.
        g_net: Generator network.
        d_net: Discriminator network.
        g_optimizer: Optimizer for generator parameters.
        loss_fn: Adversarial loss callable for generator training.
        dlog_item: Optional dictionary updated with scalar diagnostics.

    Returns:
        Generator loss value for this step.
    """
    # sample from latent space for generator
    batch_size = y_data.size(0)
    z = z_sample_fn(batch_size)

    # generate outputs with `g_net` (begin AD)
    g_optimizer.zero_grad()
    x_gen = g_net(y_data, z)

    # evalutate discriminator
    d_outputs_gen = d_net(x_gen, y_data)

    # evaluate discriminator loss
    # NOTE: pass only generated outputs for generator steps
    g_loss, _ = loss_fn(d_outputs_gen, None)
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
    z_sample_fn: LatentSampleFn,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    loss_fn: GANLossFn,
    d_reg_fn: DiscriminatorRegularizerFn | None = None,
    d_opt_pre: int = 1,
    d_opt_post: int = 1,
    g_opt_freq: int = 1,
    device: torch.device | None = None,
    logger: logging.Logger = logging.getLogger("dlk.opt.train_batches"),
    batch_initialize_fn: BatchHookFn | None = None,
    batch_finalize_fn: BatchHookFn | None = None,
    max_batches: int | None = None,
) -> TrainLog:
    """Run the GAN training loop over batches for one epoch.

    Args:
        epoch_idx: Index of the current epoch.
        g_net: Generator network.
        d_net: Discriminator network.
        dataloader: Iterable of training batches `(x_data, y_data)`.
        z_sample_fn: Function that samples latent vectors for a batch size.
        g_optimizer: Optimizer for generator parameters.
        d_optimizer: Optimizer for discriminator parameters.
        loss_fn: Adversarial loss function used by both networks.
        d_reg_fn: Optional discriminator regularizer.
        d_opt_pre: Number of discriminator updates before generator update.
        d_opt_post: Number of discriminator updates after generator update.
        g_opt_freq: Frequency of generator updates in batch steps.
        device: Optional device to move batch tensors to.
        logger: Logger instance for batch progress.
        batch_initialize_fn: Optional callback run before each batch.
        batch_finalize_fn: Optional callback run after each batch.
        max_batches: Optional cap on number of processed batches.

    Returns:
        Batch-level diagnostics aggregated across the epoch.
    """
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

        dlog_item: dict[str, float] = {}

        # pre-train discriminator network
        dlog_pre_buf: dict[str, float] = {}
        for i in range(d_opt_pre):
            d_loss_reg_v = _train_step_discriminator(
                x_data=x_data,
                y_data=y_data,
                z_sample_fn=z_sample_fn,
                g_net=g_net,
                d_net=d_net,
                d_optimizer=d_optimizer,
                loss_fn=loss_fn,
                d_reg_fn=d_reg_fn,
                dlog_item=dlog_pre_buf,
            )
            logger.debug(
                f"epoch {epoch_idx:6d}, batch {batch_idx:6d}, pre {i:2d}, "
                f"d_loss_reg {d_loss_reg_v:.6e}, d_loss {dlog_pre_buf['d_loss']:.6e}"
            )
        for k, v in dlog_pre_buf.items():
            dlog_item[f"d_pre_{k.removeprefix('d_')}"] = v

        # train generator network
        if 0 == batch_idx % g_opt_freq:
            g_loss_v = _train_step_generator(
                y_data=y_data,
                z_sample_fn=z_sample_fn,
                g_net=g_net,
                d_net=d_net,
                g_optimizer=g_optimizer,
                loss_fn=loss_fn,
                dlog_item=dlog_item,
            )
            logger.debug(
                f"epoch {epoch_idx:6d}, batch {batch_idx:6d}, g_loss {g_loss_v:.6e}"
            )

        # post-train discriminator network
        dlog_post_buf: dict[str, float] = {}
        for j in range(d_opt_post):
            d_loss_reg_v = _train_step_discriminator(
                x_data=x_data,
                y_data=y_data,
                z_sample_fn=z_sample_fn,
                g_net=g_net,
                d_net=d_net,
                d_optimizer=d_optimizer,
                loss_fn=loss_fn,
                d_reg_fn=d_reg_fn,
                dlog_item=dlog_post_buf,
            )
            logger.debug(
                f"epoch {epoch_idx:6d}, batch {batch_idx:6d}, post {j:2d}, "
                f"d_loss_reg {d_loss_reg_v:.6e}, d_loss {dlog_post_buf['d_loss']:.6e}"
            )
        for k, v in dlog_post_buf.items():
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
