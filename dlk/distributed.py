"""Small helpers for torchrun-based distributed training."""

import os
from collections.abc import Mapping
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler


def torchrun_env(env: Mapping[str, str] | None = None) -> dict[str, int] | None:
    """Parse the environment variables set by ``torchrun``.

    Args:
        env: Environment mapping to parse. Defaults to ``os.environ``.

    Returns:
        A dictionary with ``rank``, ``local_rank``, and ``world_size`` when all
        required variables are present; otherwise ``None``.
    """
    if env is None:
        env = os.environ
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if not all(name in env for name in required):
        return None
    values = {
        "rank": int(env["RANK"]),
        "local_rank": int(env["LOCAL_RANK"]),
        "world_size": int(env["WORLD_SIZE"]),
    }
    if values["rank"] < 0:
        raise ValueError(f"RANK must be >= 0, got {values['rank']}")
    if values["local_rank"] < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {values['local_rank']}")
    if values["world_size"] < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {values['world_size']}")
    return values


def init_process_group(timeout: timedelta = timedelta(minutes=30)) -> torch.device:
    """Initialize torch distributed from ``torchrun`` variables when present.

    Args:
        timeout: Collective-operation timeout. Defaults to 30 minutes, matching
            the NCCL default. Increase when rank 0 performs long setup work
            (e.g., dataset preparation) before the first collective.

    Returns:
        The process-local compute device. In distributed CUDA runs, this is
        selected from ``LOCAL_RANK``.
    """
    env = torchrun_env()
    if env is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(env["local_rank"])
        device = torch.device("cuda", env["local_rank"])
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timeout)
    return device


def is_distributed() -> bool:
    """Return whether the current process is part of a distributed group."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current distributed rank, or zero outside distributed runs."""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Return the local rank from ``torchrun``, or zero when unset."""
    env = torchrun_env()
    if env is None:
        return 0
    return env["local_rank"]


def get_world_size() -> int:
    """Return the distributed world size, or one outside distributed runs."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Return whether the current process should perform user-visible output."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks when distributed is initialized."""
    if not is_distributed():
        return
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[get_local_rank()])
    else:
        dist.barrier()


def cleanup() -> None:
    """Destroy the distributed process group when initialized."""
    if is_distributed():
        dist.destroy_process_group()


def create_distributed_sampler(
    dataset: Dataset,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DistributedSampler | None:
    """Create a ``DistributedSampler`` only when distributed mode is active."""
    if not is_distributed():
        return None
    return DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)


def wrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap a model with ``DistributedDataParallel`` when distributed is active."""
    if not is_distributed():
        return model
    if torch.cuda.is_available():
        return DistributedDataParallel(
            model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
        )
    return DistributedDataParallel(model)


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for DDP-wrapped models."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model
