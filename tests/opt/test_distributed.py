"""Tests for `dlk.opt.distributed` helpers and dlog cross-rank reduction."""

import math

import pytest
import torch
from ddp_test_utils import init_worker, run_distributed

from dlk.opt import distributed
from dlk.opt.utils import (
    train_dlog_batch_all_reduce,
    train_dlog_batch_finalize,
    train_dlog_batch_initialize,
    train_dlog_batch_update,
)

# --------------------------------------
# Single-process degradation
# --------------------------------------


def test_single_process_context_and_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degrade every helper gracefully without launcher environment variables."""
    for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID"]:
        monkeypatch.delenv(name, raising=False)

    ctx = distributed.initialize()

    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert ctx.world_size == 1
    assert ctx.is_main
    assert not ctx.is_distributed
    assert not distributed.is_distributed()
    assert distributed.get_rank() == 0
    assert distributed.get_world_size() == 1
    assert distributed.is_main_process()
    # no-ops must not raise
    distributed.barrier()
    distributed.finalize()


def test_single_process_wrap_and_sampler_passthrough() -> None:
    """Return models and samplers unchanged when not distributed."""
    net = torch.nn.Linear(2, 2)
    dataset = torch.utils.data.TensorDataset(torch.zeros((4, 2)))

    assert distributed.wrap_net(net, torch.device("cpu")) is net
    assert distributed.unwrap_net(net) is net
    assert distributed.sampler_create(dataset) is None


def test_single_process_seed_returns_base_seed() -> None:
    """Return the base seed on rank 0 with and without rank offset."""
    assert distributed.seed_random_generators(123) == 123
    assert distributed.seed_random_generators(123, rank_offset=False) == 123


def test_single_process_all_reduce_is_noop() -> None:
    """Leave tensors unchanged by `all_reduce_sum_` when not distributed."""
    values = torch.tensor([1.0, 2.0])

    distributed.all_reduce_sum_(values)

    assert values.tolist() == [1.0, 2.0]


def test_sampler_set_epoch_duck_typing() -> None:
    """Call `set_epoch` when present and ignore plain dataloaders."""

    class _FakeSampler:
        def __init__(self) -> None:
            self.epoch: int | None = None

        def set_epoch(self, epoch_idx: int) -> None:
            self.epoch = epoch_idx

    class _FakeDataLoader:
        def __init__(self) -> None:
            self.sampler = _FakeSampler()

    dataloader = _FakeDataLoader()
    distributed.sampler_set_epoch(dataloader, 7)
    assert dataloader.sampler.epoch == 7

    # plain objects without a sampler must not raise
    distributed.sampler_set_epoch(object(), 7)


# --------------------------------------
# 2-process gloo group
# --------------------------------------


def _context_worker(rank: int, world_size: int, port: int) -> None:
    """Verify context fields, rank helpers, and all-reduce in a 2-proc group."""
    ctx = init_worker(rank, world_size, port)

    assert ctx.rank == rank
    assert ctx.local_rank == rank
    assert ctx.world_size == world_size
    assert ctx.device == torch.device("cpu")
    assert ctx.is_main == (rank == 0)
    assert ctx.is_distributed
    assert distributed.is_distributed()
    assert distributed.get_rank() == rank
    assert distributed.get_world_size() == world_size
    assert distributed.is_main_process() == (rank == 0)
    assert distributed.seed_random_generators(100) == 100 + rank

    values = torch.tensor([float(rank + 1), 10.0])
    distributed.all_reduce_sum_(values)
    assert values.tolist() == [3.0, 20.0]

    distributed.finalize()
    assert not distributed.is_distributed()


def test_two_process_context_and_all_reduce() -> None:
    """Initialize a 2-proc gloo group from torchrun-style environment variables."""
    run_distributed(_context_worker)


def _slurm_fallback_worker(rank: int, world_size: int, port: int) -> None:
    """Verify the Slurm environment fallback of `initialize`."""
    ctx = init_worker(rank, world_size, port, slurm_style=True)

    assert ctx.rank == rank
    assert ctx.world_size == world_size
    assert ctx.is_distributed

    distributed.finalize()


def test_two_process_slurm_environment_fallback() -> None:
    """Initialize a 2-proc gloo group from Slurm-style environment variables."""
    run_distributed(_slurm_fallback_worker)


def _dlog_all_reduce_worker(rank: int, world_size: int, port: int) -> None:
    """Verify exact global mean/std from the dlog cross-rank reduction."""
    init_worker(rank, world_size, port)

    # per-rank loss values; rank 1 includes a NaN that must be excluded
    values_per_rank = {
        0: [1.0, 2.0, 3.0],
        1: [4.0, 5.0, math.nan],
    }
    values = values_per_rank[rank]
    dlog = train_dlog_batch_initialize(len(values), ["loss"])
    for batch_idx, value in enumerate(values):
        train_dlog_batch_update(dlog, batch_idx, {"loss": value})

    train_dlog_batch_all_reduce(dlog, ["loss"])
    train_dlog_batch_finalize(dlog, ["loss"])

    global_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    expected_mean = global_values.mean().item()
    expected_std = global_values.std(correction=0).item()
    assert dlog["loss_mean"] == pytest.approx(expected_mean)
    assert dlog["loss_std"] == pytest.approx(expected_std)
    assert dlog["loss_mean_n"] == 5

    distributed.finalize()


def test_two_process_dlog_all_reduce_exactness() -> None:
    """Reduce dlog aggregates to the exact global mean/std across ranks."""
    run_distributed(_dlog_all_reduce_worker)
