"""Reusable epoch- and batch-level training loops for diffusion models."""

import logging
from collections.abc import Iterator

import torch
from flow_matching.path import AffineProbPath, ProbPath
from flow_matching.path.scheduler import CondOTScheduler

from dlk.opt import train
from dlk.opt.utils import (
    DataLoaderType,
    EpochHookFn,
    InputsTransformFn,
    LossFn,
    LRSchedulerType,
    TensorTransformFn,
    TrainLog,
    ValidationFn,
)


class DataLoaderWrapper:
    """Wraps a DataLoader to produce flow-matching training pairs per batch.

    Each batch from the underlying dataloader is converted into:

    - inputs `(x_t, t)` or `(x_t, t, y)` when `conditional=True`:
      interpolated sample at time `t` and optional conditioning tensor.
    - target `dx_t`: conditional velocity, the regression target for the network.

    All tensors are moved to `device` before path sampling.

    Args:
        dataloader: Source dataloader. Yields `(x, y)` when
            `conditional=True`, or plain `x` tensors otherwise.
        diffusion_path: Probability path used to sample `x_t` and `dx_t`.
            Any `ProbPath` subclass is accepted. Defaults to
            `AffineProbPath(scheduler=CondOTScheduler())`.
        device: Device to move all tensors to. `None` leaves tensors on
            their current device.
        conditional: When `True`, the dataloader is expected to yield
            `(x, y)` pairs and `y` is appended to the model inputs.
            Defaults to `False`.
    """

    def __init__(
        self,
        dataloader: DataLoaderType,
        diffusion_path: ProbPath | None = None,
        device: torch.device | None = None,
        conditional: bool = False,
    ):
        self.dataloader = dataloader
        self.diffusion_path = (
            diffusion_path
            if diffusion_path is not None
            else AffineProbPath(scheduler=CondOTScheduler())
        )
        self.device = device
        self.conditional = conditional

    def transform(
        self, x_final: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform a target batch into a flow-matching path sample.

        Args:
            x_final: Target data batch `x_1`, shape `(batch_size, ...)`.

        Returns:
            Tuple `(x_t, t, dx_t)` where `x_t` is the interpolated sample,
            `t` is the sampled time vector of shape `(batch_size,)`, and
            `dx_t` is the conditional velocity target matching the shape of
            `x_final`.
        """
        # sample the distribution at initial diffusion time
        x_init = torch.randn_like(x_final)

        # sample time
        t = torch.rand(x_final.shape[0], device=x_final.device)

        # compute sample along the probability path
        path_sample = self.diffusion_path.sample(x_0=x_init, x_1=x_final, t=t)

        return path_sample.x_t, path_sample.t, path_sample.dx_t

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ]
    ]:
        """Iterate over batches, yielding `(inputs, targets)` pairs. Moves all tensors
        to the device.

        Yields:
            Tuple `(inputs, targets)` where `inputs` is `(x_t, t, y)`
            when `conditional=True` and `(x_t, t)` otherwise, and
            `targets` is `dx_t`.
        """
        for batch in self.dataloader:
            if self.conditional:
                # get the target and conditional tensors; move to device
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                # transform target and compute velocity
                x_t, t, dx_t = self.transform(x)

                # set input tensors
                inputs = x_t, t, y
            else:
                # get the target tensor; move to device
                x = batch
                x = x.to(self.device)

                # transform target and compute velocity
                x_t, t, dx_t = self.transform(x)

                # set input tensors
                inputs = x_t, t

            # set targets to be the velocity
            targets = dx_t

            yield inputs, targets

    @property
    def batch_size(self) -> int | None:
        """Number of samples per batch from the underlying dataloader."""
        return self.dataloader.batch_size

    def __len__(self) -> int:
        """Return the number of batches in the underlying dataloader."""
        return len(self.dataloader)


def train_epochs(
    n_epochs: int,
    net: torch.nn.Module,
    dataloader: DataLoaderType,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    diffusion_path: ProbPath | None = None,
    diffusion_conditional: bool = False,
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
) -> TrainLog:
    """Run the diffusion model training loop over epochs.

    Wraps `dataloader` in a `DataLoaderWrapper` to produce flow-matching
    training pairs `(x_t, t, dx_t)` per batch, then delegates to
    `train.train_epochs`. See `train.train_epochs` for full details on the
    shared arguments.

    Checkpointing saves model and optimizer states at every epoch divisible by
    `checkpoint_epochs`. Setting `checkpoint_epochs=None` disables checkpointing.
    When enabled, checkpoints are written to a run-specific directory under
    `checkpoint_dir`.

    Args:
        n_epochs: Number of epochs to train.
        net: Model to optimize.
        dataloader: Source dataloader. Yields `(x, y)` pairs when
            `diffusion_conditional=True`, or plain `x` tensors otherwise.
        optimizer: Optimizer used to update model parameters.
        loss_fn: Callable that maps `(outputs, targets)` to a scalar loss tensor.
        diffusion_path: Probability path used to sample `(x_t, t, dx_t)` per
            batch. Defaults to `AffineProbPath(scheduler=CondOTScheduler())`.
        diffusion_conditional: When `True`, the dataloader is expected to yield
            `(x, y)` pairs and `y` is appended to the model inputs as a
            conditioning tensor. Defaults to `False`.
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
    if logger is None:
        logger = logging.getLogger("dlk.opt.train_diffusion.train_epochs")

    # wrap the dataloader
    dataloader_wrapper = DataLoaderWrapper(
        dataloader,
        diffusion_path=diffusion_path,
        device=device,
        conditional=diffusion_conditional,
    )

    # run training
    return train.train_epochs(
        n_epochs=n_epochs,
        net=net,
        dataloader=dataloader_wrapper,
        optimizer=optimizer,
        loss_fn=loss_fn,
        validation_fn=validation_fn,
        lr_scheduler=lr_scheduler,
        device=device,
        inputs_transform_fn=inputs_transform_fn,
        targets_transform_fn=targets_transform_fn,
        logger=logger,
        checkpoint_epochs=checkpoint_epochs,
        checkpoint_dir=checkpoint_dir,
        epoch_initialize_fn=epoch_initialize_fn,
        epoch_finalize_fn=epoch_finalize_fn,
    )
