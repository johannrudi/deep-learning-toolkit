"""Helpers to run test workers in a 2-process gloo group via `mp.spawn`."""

import os
import socket
from collections.abc import Callable
from typing import Any

from torch.multiprocessing.spawn import spawn

from dlk.opt import distributed


def find_free_port() -> int:
    """Return a free TCP port for the process-group rendezvous.

    Returns:
        Port number currently not in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def init_worker(
    rank: int,
    world_size: int,
    port: int,
    slurm_style: bool = False,
) -> distributed.DistributedContext:
    """Initialize a spawned test worker through `initialize`.

    Sets the launcher environment variables (torchrun-style by default,
    Slurm-style when requested) and hides CUDA devices so that tests run on
    CPU with the gloo backend everywhere.

    Args:
        rank: Global rank of this worker.
        world_size: Total number of workers.
        port: Rendezvous port shared by all workers.
        slurm_style: Whether to exercise the Slurm environment fallback.

    Returns:
        Context returned by `initialize`.
    """
    # hide GPUs before the first CUDA query in this process
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    if slurm_style:
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ["SLURM_PROCID"] = str(rank)
        os.environ["SLURM_LOCALID"] = str(rank)
        os.environ["SLURM_NTASKS"] = str(world_size)
    else:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
    return distributed.initialize()


def run_distributed(
    worker_fn: Callable[..., None],
    args: tuple[Any, ...] = (),
    world_size: int = 2,
) -> None:
    """Spawn `world_size` processes running `worker_fn(rank, world_size, port, *args)`.

    Worker failures (assertions included) propagate to the caller as
    `ProcessRaisedException`.

    Args:
        worker_fn: Module-level function executed by each worker.
        args: Extra arguments appended to the worker call.
        world_size: Number of processes to spawn.

    Returns:
        None.
    """
    port = find_free_port()
    spawn(
        worker_fn,
        args=(world_size, port) + args,
        nprocs=world_size,
        join=True,
    )
