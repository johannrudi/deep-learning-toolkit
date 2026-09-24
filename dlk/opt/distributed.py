"""Building blocks for distributed data-parallel (DDP) training with torchrun.

Launch with torchrun (primary path), which sets the environment variables
`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`:

    torchrun --standalone --nproc-per-node=4 run.py

Alternatively, launch one task per process with Slurm (fallback path), which
requires exporting `MASTER_ADDR` and `MASTER_PORT` in the job script:

    srun python run.py

Wrap the run in `session`, which initializes the process group and destroys it
on every path out of the block:

    with distributed.session() as ctx:
        train(ctx)

Every function degrades to a no-op or passthrough when the process group is not
initialized, so single-process runs need no code changes.
"""

import contextlib
import datetime
import inspect
import logging
import os
import random
import re
from collections.abc import Generator
from dataclasses import dataclass

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

# newer torch deprecates DDP's `broadcast_buffers` in favor of `forward_sync_buffers`
_DDP_SUPPORTS_FORWARD_SYNC_BUFFERS = (
    "forward_sync_buffers"
    in inspect.signature(DistributedDataParallel.__init__).parameters
)

# device types whose backends bind a communicator to one device per process
# (cuda -> nccl, xpu -> xccl); cpu and mps run collectives through gloo, which
# takes neither a `device_id` nor DDP `device_ids`
_COMMUNICATOR_DEVICE_TYPES = ("cuda", "xpu")


@dataclass(frozen=True)
class DistributedContext:
    """Describe the role of this process in a (possibly single-process) run.

    Attributes:
        rank: Global process index; 0 in single-process runs.
        local_rank: Process index within the node; 0 in single-process runs.
        world_size: Total number of processes; 1 in single-process runs.
        local_world_size: Number of processes on this node; 1 in single-process
            runs.
        device: Device assigned to this process.
        num_threads: Intra-op thread count set for this process.
        is_main: Whether this is the main process (rank 0).
        is_distributed: Whether a process group is initialized.
    """

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device
    num_threads: int
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


def all_reduce_min_(values: torch.Tensor) -> None:
    """Take the elementwise minimum of a tensor across all ranks, in place.

    No-op when not distributed. With the NCCL backend the tensor must reside
    on this process's GPU; with gloo it must reside on the CPU.

    Args:
        values: Tensor to reduce; overwritten with the global minimum.

    Returns:
        None.
    """
    if not is_distributed():
        return
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MIN)


def all_reduce_max_(values: torch.Tensor) -> None:
    """Take the elementwise maximum of a tensor across all ranks, in place.

    No-op when not distributed. With the NCCL backend the tensor must reside
    on this process's GPU; with gloo it must reside on the CPU.

    Args:
        values: Tensor to reduce; overwritten with the global maximum.

    Returns:
        None.
    """
    if not is_distributed():
        return
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)


def _expand_slurm_tasks_per_node(value: str) -> list[int]:
    r"""Expand Slurm's compressed per-node task counts into one entry per node.

    Slurm writes `SLURM_TASKS_PER_NODE` as a comma-separated list in which a
    repeated count collapses into `count(xnodes)`, for example `"2(x3),1"` for
    four nodes running 2, 2, 2, and 1 tasks.

    Args:
        value: Raw value of `SLURM_TASKS_PER_NODE`.

    Returns:
        Task count per node, or an empty list when the value does not parse.
    """
    counts: list[int] = []
    for field in value.split(","):
        match = re.fullmatch(r"(\d+)(?:\(x(\d+)\))?", field.strip())
        if match is None:
            return []
        counts.extend([int(match.group(1))] * int(match.group(2) or 1))
    return counts


def _read_slurm_local_world_size(world_size: int) -> int:
    """Determine how many Slurm tasks share this node.

    Prefers `SLURM_NTASKS_PER_NODE`, which Slurm sets only when the job
    requested `--ntasks-per-node`. Otherwise reads the actual allocation from
    `SLURM_TASKS_PER_NODE` at the index given by `SLURM_NODEID`, which covers
    heterogeneous allocations.

    Args:
        world_size: Total number of tasks, used as the fallback.

    Returns:
        Number of tasks on this node; `world_size` when Slurm reports neither.
    """
    ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
    if ntasks_per_node is not None:
        return max(1, int(ntasks_per_node))
    tasks_per_node = os.environ.get("SLURM_TASKS_PER_NODE")
    if tasks_per_node is not None:
        counts = _expand_slurm_tasks_per_node(tasks_per_node)
        node_idx = int(os.environ.get("SLURM_NODEID", 0))
        if 0 <= node_idx < len(counts):
            return max(1, counts[node_idx])
    return world_size


def _read_launcher_environment() -> tuple[int, int, int, int] | None:
    """Read rank information from torchrun or Slurm environment variables.

    Detection order:
    1. torchrun: `RANK`, `LOCAL_RANK`, `WORLD_SIZE` are set. `LOCAL_WORLD_SIZE`
       is set by torchrun too, but defaults to `WORLD_SIZE` so that a manual
       `env://` launch works as well.
    2. Slurm: `SLURM_PROCID` is set and `MASTER_ADDR`/`MASTER_PORT` are
       exported by the job script; ranks derive from `SLURM_PROCID`,
       `SLURM_LOCALID`, `SLURM_NTASKS`, and the per-node task count from
       `SLURM_NTASKS_PER_NODE` or `SLURM_TASKS_PER_NODE`. The variables are
       written back to the environment so that `init_process_group("env://")`
       finds them.

    Returns:
        Tuple `(rank, local_rank, world_size, local_world_size)`, or `None` for
        single-process.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        return rank, local_rank, world_size, local_world_size
    if (
        "SLURM_PROCID" in os.environ
        and "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
    ):
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ["SLURM_NTASKS"])
        local_world_size = _read_slurm_local_world_size(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
        return rank, local_rank, world_size, local_world_size
    return None


def _select_device(local_rank: int) -> torch.device:
    """Bind this process to one accelerator, chosen by local rank.

    Args:
        local_rank: Process index within this node.

    Returns:
        The accelerator device for this process, or CPU when no accelerator is
        available.

    Raises:
        RuntimeError: If fewer accelerators are visible than the local rank
            requires.
    """
    accel = torch.accelerator.current_accelerator(check_available=True)
    if accel is None:
        return torch.device("cpu")

    device_count = torch.accelerator.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"local rank {local_rank} needs at least {local_rank + 1} "
            f"{accel.type} devices, but only {device_count} are visible; "
            "start one process per device or widen the device mask "
            "(e.g. CUDA_VISIBLE_DEVICES)"
        )

    # NOTE: This must run before init_process_group that builds the communicator on it.
    torch.accelerator.set_device_index(local_rank)
    return torch.device(accel.type, local_rank)


def initialize(
    backend: str | None = None,
    timeout_seconds: float = 1800.0,
    logger: logging.Logger | None = None,
) -> DistributedContext:
    """Initialize distributed training from launcher environment variables.

    Detects torchrun environment variables (primary) or Slurm variables
    (fallback, see `_read_launcher_environment`). When neither is present,
    returns a single-process context without creating a process group.

    The process is bound to the accelerator at index `local_rank` (see
    `_select_device`) before the process group is created, because it must
    happen before the first tensor operation.

    Args:
        backend: Process group backend; `None` selects the default backend for
            the device (cuda -> nccl, xpu -> xccl, cpu/mps -> gloo).
        timeout_seconds: Timeout for collective operations.
        logger: Logger used for initialization reporting.

    Returns:
        Context describing this process's rank, device, and distributed state.

    Raises:
        RuntimeError: If fewer accelerators are visible than the local rank
            requires.
    """
    if logger is None:
        logger = logging.getLogger("dlk.opt.distributed.initialize")

    # get the multi-process context
    launcher_env = _read_launcher_environment()
    if launcher_env is not None:
        rank, local_rank, world_size, local_world_size = launcher_env
    else:
        rank, local_rank, world_size, local_world_size = 0, 0, 1, 1

    # get the number of threads
    num_threads = torch.get_num_threads()

    # bind this process to one accelerator, chosen by LOCAL rank
    device = _select_device(local_rank)

    # exit with a new single-process context
    if launcher_env is None:
        logger.info(f"single-process run, device {device}, num_threads {num_threads}")
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            device=device,
            num_threads=num_threads,
            is_main=True,
            is_distributed=False,
        )

    # set device-specific backend (e.g., cuda -> nccl, xpu -> xccl, cpu/mps -> gloo)
    if backend is None:
        backend = torch.distributed.get_default_backend_for_device(device)

    # create the process group
    # NOTE: The following arguments are not passed here because they come from the
    #       torchrun environment: init_method, world_size, rank
    #       Omitting init_method defaults it to "env://", which reads MASTER_ADDR,
    #       MASTER_PORT, RANK and WORLD_SIZE from the environment torchrun set.
    torch.distributed.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(seconds=timeout_seconds),
        device_id=device if device.type in _COMMUNICATOR_DEVICE_TYPES else None,
    )
    if world_size <= 8:
        logger.info(
            f"distributed run, "
            f"rank {rank}/{world_size}, local_rank {local_rank}/{local_world_size}, "
            f"device {device}, backend {backend}, num_threads {num_threads}"
        )
    elif is_main_process():
        logger.info(
            f"distributed run, "
            f"world_size {world_size}, local_world_size {local_world_size}, "
            f"device {device}, backend {backend}, num_threads {num_threads}"
        )

    # create the multi-process context
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        device=device,
        num_threads=num_threads,
        is_main=(rank == 0),
        is_distributed=True,
    )


def finalize(synchronize: bool = True) -> None:
    """Destroy the process group; no-op when not distributed.

    Args:
        synchronize: Whether to barrier before destroying the group. Pass False
            while unwinding an error: a rank that has already died never
            reaches the barrier, so the survivors would block until the
            collective timeout expires.

    Returns:
        None.
    """
    if not is_distributed():
        return
    if synchronize:
        torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@contextlib.contextmanager
def session(
    backend: str | None = None,
    timeout_seconds: float = 1800.0,
    logger: logging.Logger | None = None,
) -> Generator[DistributedContext]:
    """Initialize distributed training and tear it down on exit.

    Wraps `initialize` and `finalize` so that the process group is destroyed on
    every path out of the block:

        with distributed.session() as ctx:
            train(ctx)

    On the normal path the ranks synchronize before the group is destroyed; on
    the error path they do not, so a surviving rank tears down immediately
    instead of waiting on a rank that has already failed.

    Args:
        backend: Process group backend; see `initialize`.
        timeout_seconds: Timeout for collective operations.
        logger: Logger used for initialization reporting.

    Yields:
        Context describing this process's rank, device, and distributed state.
    """
    ctx = initialize(
        backend=backend,
        timeout_seconds=timeout_seconds,
        logger=logger,
    )
    try:
        yield ctx
    except BaseException:
        finalize(synchronize=False)
        raise
    else:
        finalize()


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

    # set the device the module resides on
    # NOTE: DDP takes device_ids only for backends that bind one device per
    #       process; a gloo run (cpu, mps) must leave it unset.
    device_ids = [device] if device.type in _COMMUNICATOR_DEVICE_TYPES else None

    # wrap the network
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


def check_net_replica_drift(
    net: torch.nn.Module,
    include_buffers: bool = False,
) -> float:
    r"""Measure how far this rank's model has drifted from rank 0's.

    Broadcasts rank 0's parameters and reduces the largest elementwise
    difference over all ranks, so every rank returns the same number. A
    correctly synchronized DDP run returns exactly 0.0; a non-zero value means
    the replicas diverged, typically from a rank-dependent initialization or an
    optimizer step taken outside DDP's gradient synchronization.

    This is a collective: every rank must call it, or the run deadlocks. It is
    a debugging aid, not part of the training loop.

    Args:
        net: Model to compare; DDP wrappers are unwrapped automatically.
        include_buffers: Whether to compare buffers (e.g. batch-norm running
            statistics) in addition to parameters.

    Returns:
        Largest absolute difference from rank 0 over all ranks and tensors;
        0.0 when not distributed or when the model has no tensors.
    """
    if not is_distributed():
        return 0.0

    module = unwrap_net(net)
    tensors = [parameter.detach() for parameter in module.parameters()]
    if include_buffers:
        tensors += [buffer.detach() for buffer in module.buffers()]
    if not tensors:
        return 0.0

    # compare in float64 so that low-precision dtypes do not mask the drift
    drift = torch.zeros((), dtype=torch.float64, device=tensors[0].device)
    for tensor in tensors:
        values = tensor.to(torch.float64)
        reference = values.clone()
        torch.distributed.broadcast(reference, src=0)
        drift = torch.maximum(drift, (values - reference).abs().max())

    torch.distributed.all_reduce(drift, op=torch.distributed.ReduceOp.MAX)
    return float(drift.item())


def sampler_create(
    dataset: torch.utils.data.Dataset[object],
    ctx: DistributedContext | None = None,
    shuffle: bool = True,
    base_seed: int = 0,
    drop_last: bool = False,
) -> torch.utils.data.DistributedSampler | None:
    """Create a `DistributedSampler` when distributed, otherwise `None`.

    Pass the returned sampler to `torch.utils.data.DataLoader(sampler=...)`
    with `shuffle=False` (sampler and shuffle are mutually exclusive). When
    `None` is returned, keep the plain `shuffle=` path.

    With `ctx`, the sharding follows the context and a sampler is always
    returned, including for a single-process context, where it shards into one
    piece. Without `ctx`, the sampler reads the process group directly and
    single-process runs get `None`.

    Args:
        dataset: Dataset to shard across processes.
        ctx: Context whose rank and world size define the sharding; `None`
            reads the process group instead.
        shuffle: Whether the sampler shuffles the shard each epoch.
        base_seed: Base random seed for shuffling; must be identical on all ranks
            (do not pass a rank-offset seed).
        drop_last: Whether to drop trailing samples instead of padding, so
            that shards have equal length without repeated samples.

    Returns:
        Sampler for this process's shard, or `None` when no context is given
        and no process group is initialized.

    Raises:
        RuntimeError: If `ctx` describes a multi-process run but no process
            group is initialized, which would shard the dataset among ranks
            that never exchange anything.
    """
    if ctx is None:
        if not is_distributed():
            return None
        num_replicas, rank = None, None
    else:
        if ctx.world_size > 1 and not is_distributed():
            raise RuntimeError(
                f"context reports world_size {ctx.world_size}, but no process "
                "group is initialized; call initialize() before sampler_create()"
            )
        num_replicas, rank = ctx.world_size, ctx.rank

    return torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=base_seed,
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


def check_batch_ranks_differ(batch: torch.Tensor) -> bool:
    r"""Check that every rank drew a different batch.

    Gathers one checksum per rank and reports whether all of them are distinct.
    Identical checksums mean the ranks train on the same samples, which turns a
    `world_size`-way run into `world_size` copies of the same gradient; the
    usual cause is a dataloader built without a `DistributedSampler`.

    The checksum collides in principle, so a False is conclusive while a True
    is strong evidence rather than proof.

    This is a collective: every rank must call it, or the run deadlocks. It is
    a debugging aid, not part of the training loop.

    Args:
        batch: Batch drawn on this rank; must live on this process's device.

    Returns:
        True when all ranks report distinct batches, or when not distributed.
    """
    if not is_distributed():
        return True

    checksum = batch.detach().to(torch.float64).sum().reshape(1)
    gathered = [torch.zeros_like(checksum) for _ in range(get_world_size())]
    torch.distributed.all_gather(gathered, checksum)

    checksums = [float(value.item()) for value in gathered]
    return len(set(checksums)) == len(checksums)


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
    torch.manual_seed(seed)  # sets the seed for all device backends
    return seed
