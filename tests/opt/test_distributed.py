"""Tests for `dlk.opt.distributed` helpers and dlog cross-rank reduction."""

import math
import os

import pytest
import torch
from ddp_test_utils import init_worker, run_distributed, set_launcher_environment

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
# Slurm and launcher-environment helpers
# --------------------------------------


def test_expand_slurm_tasks_per_node_compressed_and_plain() -> None:
    """Expand a compressed run-length value and pass through a plain single count."""
    assert distributed._expand_slurm_tasks_per_node("2(x3),1") == [2, 2, 2, 1]
    assert distributed._expand_slurm_tasks_per_node("4") == [4]


def test_expand_slurm_tasks_per_node_unparseable_returns_empty() -> None:
    """Return an empty list for a value that does not match the Slurm format."""
    assert distributed._expand_slurm_tasks_per_node("not-a-count") == []


def test_read_slurm_local_world_size_prefers_ntasks_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer `SLURM_NTASKS_PER_NODE` over `SLURM_TASKS_PER_NODE`."""
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "3")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "1(x4)")
    assert distributed._read_slurm_local_world_size(world_size=4) == 3


def test_read_slurm_local_world_size_falls_back_to_tasks_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Index `SLURM_TASKS_PER_NODE` by `SLURM_NODEID` when the per-node count is unset."""
    monkeypatch.delenv("SLURM_NTASKS_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "2(x3),1")
    monkeypatch.setenv("SLURM_NODEID", "3")
    assert distributed._read_slurm_local_world_size(world_size=7) == 1


def test_read_slurm_local_world_size_falls_back_to_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to `world_size` when neither Slurm variable is set."""
    monkeypatch.delenv("SLURM_NTASKS_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_TASKS_PER_NODE", raising=False)
    assert distributed._read_slurm_local_world_size(world_size=4) == 4


def test_read_launcher_environment_returns_none_without_launcher_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return None with no torchrun or Slurm environment variables set."""
    for name in [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "SLURM_PROCID",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]:
        monkeypatch.delenv(name, raising=False)
    assert distributed._read_launcher_environment() is None


def test_read_launcher_environment_torchrun_defaults_local_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default LOCAL_WORLD_SIZE to WORLD_SIZE when torchrun leaves it unset."""
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert distributed._read_launcher_environment() == (1, 1, 4, 4)


def test_read_launcher_environment_torchrun_respects_local_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read an explicit LOCAL_WORLD_SIZE from torchrun instead of defaulting."""
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert distributed._read_launcher_environment() == (1, 1, 4, 2)


def test_read_launcher_environment_slurm_writes_back_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive ranks from Slurm variables and write them back as integers."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "2")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12345")

    # the function writes RANK/LOCAL_RANK/WORLD_SIZE/LOCAL_WORLD_SIZE directly into
    # os.environ, bypassing monkeypatch's tracking (which never saw them as "set"
    # since they were absent at the delenv calls above); pop them explicitly so no
    # state survives into later tests
    try:
        assert distributed._read_launcher_environment() == (1, 1, 4, 2)
        assert os.environ["RANK"] == "1"
        assert os.environ["LOCAL_RANK"] == "1"
        assert os.environ["WORLD_SIZE"] == "4"
        assert os.environ["LOCAL_WORLD_SIZE"] == "2"
    finally:
        for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"]:
            os.environ.pop(name, None)


def test_initialize_single_process_observes_num_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record the ambient thread count in the context without changing it."""
    for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID"]:
        monkeypatch.delenv(name, raising=False)
    ambient_num_threads = torch.get_num_threads()

    ctx = distributed.initialize()

    assert ctx.num_threads == ambient_num_threads
    assert ctx.local_world_size == 1
    assert torch.get_num_threads() == ambient_num_threads


# --------------------------------------
# session() single-process
# --------------------------------------


def test_session_single_process_yields_and_leaves_no_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Yield a usable single-process context and leave no process group behind."""
    for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID"]:
        monkeypatch.delenv(name, raising=False)

    with distributed.session() as ctx:
        assert ctx.world_size == 1
        assert not ctx.is_distributed
        assert not distributed.is_distributed()

    assert not distributed.is_distributed()


def test_session_single_process_reraises_and_leaves_no_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-raise the original exception and leave no process group behind."""
    for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID"]:
        monkeypatch.delenv(name, raising=False)

    class _Boom(Exception):
        """Marker exception raised inside the session block."""

    with pytest.raises(_Boom):
        with distributed.session():
            raise _Boom("boom")

    assert not distributed.is_distributed()


# --------------------------------------
# sampler_create single-process
# --------------------------------------


def test_sampler_create_with_single_process_ctx_always_returns_sampler() -> None:
    """Return a one-replica sampler for a single-process context."""
    dataset = torch.utils.data.TensorDataset(torch.zeros((4, 2)))
    ctx = distributed.initialize()

    sampler = distributed.sampler_create(dataset, ctx=ctx)

    assert sampler is not None
    assert sampler.num_replicas == 1
    assert sampler.rank == 0


def test_sampler_create_raises_for_multi_process_ctx_without_group() -> None:
    """Raise when a hand-built multi-process context outruns actual initialization."""
    dataset = torch.utils.data.TensorDataset(torch.zeros((4, 2)))
    ctx = distributed.DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
        num_threads=1,
        is_main=True,
        is_distributed=False,
    )

    with pytest.raises(RuntimeError):
        distributed.sampler_create(dataset, ctx=ctx)


# --------------------------------------
# check_net_replica_drift / check_batch_ranks_differ single-process
# --------------------------------------


def test_single_process_debugging_collectives_return_defaults() -> None:
    """Return the not-distributed defaults for both debugging collectives."""
    net = torch.nn.Linear(2, 2)

    assert distributed.check_net_replica_drift(net) == 0.0
    assert distributed.check_batch_ranks_differ(torch.zeros(4)) is True


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


def _session_worker(rank: int, world_size: int, port: int) -> None:
    """Yield a usable context under `session` and destroy the group after the block."""
    set_launcher_environment(rank, world_size, port)

    with distributed.session() as ctx:
        assert ctx.rank == rank
        assert ctx.world_size == world_size
        assert ctx.is_distributed
        assert distributed.is_distributed()

    assert not distributed.is_distributed()


def test_two_process_session_yields_and_destroys_group() -> None:
    """Initialize and tear down a 2-proc gloo group through `session`."""
    run_distributed(_session_worker)


def _session_exception_worker(rank: int, world_size: int, port: int) -> None:
    """Re-raise the original exception without a barrier and leave the group destroyed."""
    set_launcher_environment(rank, world_size, port)

    class _Boom(Exception):
        """Marker exception raised inside the session block."""

    with pytest.raises(_Boom):
        with distributed.session():
            raise _Boom("boom")

    assert not distributed.is_distributed()


def test_two_process_session_reraises_and_destroys_group_on_error() -> None:
    """Re-raise and leave no process group behind when every rank fails."""
    run_distributed(_session_exception_worker)


def _num_threads_worker(rank: int, world_size: int, port: int) -> None:
    """Verify `initialize` observes the thread count without setting it."""
    set_launcher_environment(rank, world_size, port)
    # stand-in for the thread count OMP_NUM_THREADS established at startup
    torch.set_num_threads(2)

    ctx = distributed.initialize()

    assert ctx.num_threads == 2
    assert ctx.local_world_size == world_size
    assert torch.get_num_threads() == 2

    distributed.finalize()


def test_two_process_initialize_observes_num_threads() -> None:
    """Report the ambient thread count and the local world size in the context."""
    run_distributed(_num_threads_worker)


def _sampler_create_shards_disjointly_worker(
    rank: int, world_size: int, port: int
) -> None:
    """Shard a dataset disjointly across ranks via a ctx-driven sampler."""
    ctx = init_worker(rank, world_size, port)

    dataset = torch.utils.data.TensorDataset(torch.arange(10))
    sampler = distributed.sampler_create(
        dataset, ctx=ctx, shuffle=False, base_seed=0, drop_last=False
    )
    assert sampler is not None
    indices = list(sampler)

    gathered: list[list[int]] = [[] for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, indices)
    all_indices = [index for shard in gathered for index in shard]
    assert len(all_indices) == len(set(all_indices))

    distributed.finalize()


def test_two_process_sampler_create_shards_disjointly() -> None:
    """Shard disjointly and accept `base_seed` identically on both ranks."""
    run_distributed(_sampler_create_shards_disjointly_worker)


def _replica_drift_worker(rank: int, world_size: int, port: int) -> None:
    """Detect an injected parameter perturbation via `check_net_replica_drift`."""
    ctx = init_worker(rank, world_size, port)
    torch.manual_seed(rank)  # differ per-rank before DDP broadcasts on wrap
    linear = torch.nn.Linear(4, 2)
    net = distributed.wrap_net(linear, ctx.device)

    assert distributed.check_net_replica_drift(net) == pytest.approx(0.0)

    if rank == 1:
        with torch.no_grad():
            assert linear.bias is not None
            linear.bias.add_(0.25)

    assert distributed.check_net_replica_drift(net) == pytest.approx(0.25)

    distributed.finalize()


def test_two_process_check_net_replica_drift_detects_perturbation() -> None:
    """Report 0.0 for synchronized replicas and the exact injected perturbation."""
    run_distributed(_replica_drift_worker)


def _replica_drift_buffers_worker(rank: int, world_size: int, port: int) -> None:
    """Detect buffer drift only when `include_buffers=True`."""
    ctx = init_worker(rank, world_size, port)
    batch_norm = torch.nn.BatchNorm1d(4)
    net = distributed.wrap_net(batch_norm, ctx.device, broadcast_buffers=False)

    if rank == 1:
        with torch.no_grad():
            assert batch_norm.running_mean is not None
            batch_norm.running_mean.add_(0.5)

    assert distributed.check_net_replica_drift(
        net, include_buffers=False
    ) == pytest.approx(0.0)
    assert distributed.check_net_replica_drift(
        net, include_buffers=True
    ) == pytest.approx(0.5)

    distributed.finalize()


def test_two_process_check_net_replica_drift_include_buffers() -> None:
    """Detect a buffer-only perturbation exclusively when `include_buffers=True`."""
    run_distributed(_replica_drift_buffers_worker)


def _batch_ranks_differ_worker(rank: int, world_size: int, port: int) -> None:
    """Detect differing sharded batches and identical hand-built batches."""
    init_worker(rank, world_size, port)

    dataset = torch.utils.data.TensorDataset(torch.arange(8, dtype=torch.float32))
    sampler = distributed.sampler_create(dataset, shuffle=False)
    assert sampler is not None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)
    (batch,) = next(iter(dataloader))
    assert distributed.check_batch_ranks_differ(batch)

    identical_batch = torch.ones(4)
    assert not distributed.check_batch_ranks_differ(identical_batch)

    distributed.finalize()


def test_two_process_check_batch_ranks_differ() -> None:
    """Report True for sampler-sharded batches and False for identical batches."""
    run_distributed(_batch_ranks_differ_worker)
