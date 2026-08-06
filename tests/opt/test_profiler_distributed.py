"""Tests for `dlk.opt.profiler` output files under a 2-process gloo group."""

import pathlib

import torch
from ddp_test_utils import init_worker, run_distributed

from dlk.opt import distributed
from dlk.opt.profiler import profile_train_batches
from dlk.opt.train import train_batches


def _make_net() -> torch.nn.Module:
    """Build the small model used by the profiler tests."""
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    )


def _make_dataset(n_samples: int) -> torch.utils.data.TensorDataset:
    """Build a deterministic dataset with `n_samples` samples."""
    generator = torch.Generator().manual_seed(1234)
    features = torch.randn((n_samples, 4), generator=generator, dtype=torch.float32)
    targets = torch.randn((n_samples, 2), generator=generator, dtype=torch.float32)
    return torch.utils.data.TensorDataset(features, targets)


def _profile_worker(rank: int, world_size: int, port: int, tmp_dir: str) -> None:
    """Profile a DDP training epoch and verify rank-suffixed output files."""
    ctx = init_worker(rank, world_size, port)

    # 10 profiled batches per rank at batch_size 4 across 2 ranks
    dataset = _make_dataset(10 * 4 * world_size)
    sampler = distributed.sampler_create(dataset, shuffle=False, seed=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)

    net = distributed.wrap_net(_make_net(), ctx.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    profile_train_batches(
        train_batches,
        {"device": ctx.device},
        net,
        dataloader,
        optimizer,
        torch.nn.MSELoss(),
        log_profile_dir=tmp_dir,
    )

    # every rank writes its own rank-suffixed table and trace files
    distributed.barrier()
    if rank == 0:
        for suffix in ["rank0", "rank1"]:
            tables = list(pathlib.Path(tmp_dir).glob(f"table_prof_step_*_{suffix}.txt"))
            traces = list(
                pathlib.Path(tmp_dir).glob(f"trace_prof_step_*_{suffix}.json")
            )
            assert len(tables) > 0, f"missing profiler tables for {suffix}"
            assert len(traces) > 0, f"missing profiler traces for {suffix}"

    distributed.finalize()


def test_two_process_profiling_writes_rank_suffixed_files(
    tmp_path: pathlib.Path,
) -> None:
    """Profile under DDP and check per-rank table and trace files."""
    run_distributed(_profile_worker, args=(str(tmp_path),))


def test_single_process_profiling_keeps_unsuffixed_filenames(
    tmp_path: pathlib.Path,
) -> None:
    """Profile single-process and check filenames are unchanged (no rank suffix)."""
    dataset = _make_dataset(40)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    net = _make_net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    profile_train_batches(
        train_batches,
        {},
        net,
        dataloader,
        optimizer,
        torch.nn.MSELoss(),
        log_profile_dir=str(tmp_path),
    )

    tables = list(tmp_path.glob("table_prof_step_*.txt"))
    traces = list(tmp_path.glob("trace_prof_step_*.json"))
    assert len(tables) > 0
    assert len(traces) > 0
    assert all("rank" not in path.name for path in tables + traces)
