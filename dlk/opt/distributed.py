"""Building blocks for distributed data-parallel (DDP) training with torchrun.

Launch with torchrun (primary path), which sets the environment variables
`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`:

    torchrun --standalone --nproc-per-node=4 run.py

Alternatively, launch one task per process with Slurm (fallback path), which
requires exporting `MASTER_ADDR` and `MASTER_PORT` in the job script:

    srun python run.py

Every function degrades to a no-op or passthrough when the process group is not
initialized, so single-process runs need no code changes.
"""

import dataclasses
import datetime
import inspect
import logging
import os
import random

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

# newer torch deprecates DDP's `broadcast_buffers` in favor of `forward_sync_buffers`
_DDP_SUPPORTS_FORWARD_SYNC_BUFFERS = (
    "forward_sync_buffers"
    in inspect.signature(DistributedDataParallel.__init__).parameters
)


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    """Describe the role of this process in a (possibly single-process) run.

    Attributes:
        rank: Global process index; 0 in single-process runs.
        local_rank: Process index within the node; 0 in single-process runs.
        world_size: Total number of processes; 1 in single-process runs.
        device: Device assigned to this process.
        is_main: Whether this is the main process (rank 0).
        is_distributed: Whether a process group is initialized.
    """

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main: bool
    is_distributed: bool


def is_distributed() -> bool:
    """Return True when a distributed process group is initialized.

    Returns:
        True if `torch.distributed` is available and initialized.
    """
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    """Return the global rank of this process.

    Returns:
        Global rank, or 0 when not distributed.
    """
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    """Return the total number of processes.

    Returns:
        World size, or 1 when not distributed.
    """
    if not is_distributed():
        return 1
    return torch.distributed.get_world_size()


def is_main_process() -> bool:
    """Return True when this process is the main process (rank 0).

    Returns:
        True on rank 0 and in single-process runs.
    """
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes; no-op when not distributed.

    Returns:
        None.
    """
    if is_distributed():
        torch.distributed.barrier()


def all_reduce_sum_(values: torch.Tensor) -> None:
    """Sum a tensor elementwise across all ranks, in place.

    No-op when not distributed. With the NCCL backend the tensor must reside
    on this process's GPU; with gloo it must reside on the CPU.

    Args:
        values: Tensor to reduce; overwritten with the global sum.

    Returns:
        None.
    """
    if not is_distributed():
        return
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)


def _read_launcher_environment() -> tuple[int, int, int] | None:
    """Read rank information from torchrun or Slurm environment variables.

    Detection order:
    1. torchrun: `RANK`, `LOCAL_RANK`, `WORLD_SIZE` are set.
    2. Slurm: `SLURM_PROCID` is set and `MASTER_ADDR`/`MASTER_PORT` are
       exported by the job script; ranks derive from `SLURM_PROCID`,
       `SLURM_LOCALID`, `SLURM_NTASKS`. The variables are written back to the
       environment so that `init_process_group("env://")` finds them.

    Returns:
        Tuple `(rank, local_rank, world_size)`, or `None` for single-process.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, local_rank, world_size
    if (
        "SLURM_PROCID" in os.environ
        and "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
    ):
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        return rank, local_rank, world_size
    return None


def initialize(
    backend: str | None = None,
    timeout_seconds: float = 1800.0,
    logger: logging.Logger | None = None,
) -> DistributedContext:
    """Initialize distributed training from launcher environment variables.

    Detects torchrun environment variables (primary) or Slurm variables
    (fallback, see `_read_launcher_environment`). When neither is present,
    returns a single-process context without creating a process group.

    On CUDA systems, the device `cuda:{local_rank}` is selected via
    `torch.cuda.set_device` before the process group is initialized.

    Args:
        backend: Process group backend; `None` selects `"nccl"` when CUDA is
            available and `"gloo"` otherwise.
        timeout_seconds: Timeout for collective operations.
        logger: Logger used for initialization reporting.

    Returns:
        Context describing this process's rank, device, and distributed state.
    """
    if logger is None:
        logger = logging.getLogger("dlk.opt.distributed.initialize")

    launcher_env = _read_launcher_environment()
    if launcher_env is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"single-process run, device {device}")
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
            is_main=True,
            is_distributed=False,
        )

    rank, local_rank, world_size = launcher_env
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        # pin this process to its GPU before creating the process group
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    if world_size <= 8:
        logger.info(
            f"distributed run, rank {rank}/{world_size}, local_rank {local_rank}, "
            f"device {device}, backend {backend}"
        )
    elif is_main_process():
        logger.info(
            f"distributed run, world_size {world_size}, "
            f"device {device}, backend {backend}"
        )

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=(rank == 0),
        is_distributed=True,
    )


def finalize() -> None:
    """Synchronize and destroy the process group; no-op when not distributed.

    Returns:
        None.
    """
    if is_distributed():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def wrap_net(
    net: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
    static_graph: bool = False,
) -> torch.nn.Module:
    """Wrap a model with `DistributedDataParallel` when distributed.

    The model must already reside on `device`. When not distributed, the model
    is returned unchanged, so callers can wrap unconditionally.

    For GAN training, wrap generator and discriminator independently;
    `broadcast_buffers=False` is recommended for the discriminator to avoid
    buffer broadcasts on its multiple forward passes per batch.

    Args:
        net: Model to wrap; must be on `device` already.
        device: Device assigned to this process.
        find_unused_parameters: Whether DDP tracks parameters unused in the
            forward pass; keep False when all parameters contribute.
        broadcast_buffers: Whether buffers are synchronized from rank 0 at
            each forward pass (on newer torch, buffers still synchronize once
            at initialization when False).
        static_graph: Whether the autograd graph is identical in every
            iteration; keep False for loops with varying graphs.

    Returns:
        The DDP-wrapped model, or the unchanged model when not distributed.
    """
    if not is_distributed():
        return net
    device_ids = [device.index] if device.type == "cuda" else None
    if _DDP_SUPPORTS_FORWARD_SYNC_BUFFERS:
        return DistributedDataParallel(
            net,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph,
            forward_sync_buffers=broadcast_buffers,
        )
    return DistributedDataParallel(
        net,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        broadcast_buffers=broadcast_buffers,
    )


def unwrap_net(net: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model of a DDP-wrapped model.

    Args:
        net: Possibly DDP-wrapped model.

    Returns:
        The inner model when wrapped, otherwise the model itself.
    """
    if isinstance(net, DistributedDataParallel):
        return net.module
    return net


def sampler_create(
    dataset: "torch.utils.data.Dataset[object]",
    shuffle: bool = True,
    seed: int | None = None,
    drop_last: bool = False,
) -> torch.utils.data.DistributedSampler | None:
    """Create a `DistributedSampler` when distributed, otherwise `None`.

    Pass the returned sampler to `torch.utils.data.DataLoader(sampler=...)`
    with `shuffle=False` (sampler and shuffle are mutually exclusive). When
    `None` is returned, keep the plain `shuffle=` path.

    Args:
        dataset: Dataset to shard across processes.
        shuffle: Whether the sampler shuffles the shard each epoch.
        seed: Base random seed for shuffling; must be identical on all ranks
            (do not pass a rank-offset seed).
        drop_last: Whether to drop trailing samples instead of padding, so
            that shards have equal length without repeated samples.

    Returns:
        Sampler for this process's shard, or `None` when not distributed.
    """
    if not is_distributed():
        return None
    if seed is None:
        seed = 0
    return torch.utils.data.DistributedSampler(
        dataset,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def sampler_set_epoch(dataloader: object, epoch_idx: int) -> None:
    """Advance a dataloader's `DistributedSampler` to a new epoch.

    Duck-typed: acts when the dataloader has a `sampler` attribute whose value
    has a `set_epoch` method; safe no-op otherwise (plain dataloaders,
    single-process runs). Without `set_epoch`, a `DistributedSampler` repeats
    the same shuffle order every epoch.

    Args:
        dataloader: Dataloader possibly holding a `DistributedSampler`.
        epoch_idx: Current epoch index.

    Returns:
        None.
    """
    sampler = getattr(dataloader, "sampler", None)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch_idx)


def seed_random_generators(base_seed: int, rank_offset: bool = True) -> int:
    """Seed `random`, numpy, and torch generators with a rank-offset seed.

    Each rank seeds with `base_seed + rank` so that per-rank random draws
    (e.g. latent samples, data augmentation) differ across processes. Note
    that `sampler_create` must receive `base_seed`, not the
    returned rank-offset seed.

    Args:
        base_seed: Seed shared by all ranks before the rank offset.
        rank_offset: Whether to add the global rank to the seed.

    Returns:
        The effective seed used by this process.
    """
    seed = base_seed + get_rank() if rank_offset else base_seed
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
