"""Unit tests for `dlk.opt.monitor`: Welford mean/std and summary statistics."""

import math
from typing import Any

import pytest
import torch
from ddp_test_utils import init_worker, run_distributed

from dlk.opt import distributed, monitor

# --------------------------------------
# Single-process: Welford mean/std, extended stats
# --------------------------------------


def test_batch_update_welford_matches_naive_reference() -> None:
    """Match a naive mean/std reference on random per-batch data."""
    torch.manual_seed(0)
    values = torch.randn(200, dtype=torch.float64) * 3.0 + 1000.0

    dlog = monitor.batch_initialize(len(values), ["x"])
    for batch_idx, value in enumerate(values):
        monitor.batch_update(dlog, batch_idx, {"x": value.item()})
    monitor.batch_finalize(dlog, ["x"])

    assert dlog["x_mean"] == pytest.approx(values.mean().item())
    assert dlog["x_std"] == pytest.approx(values.std(correction=0).item())
    assert dlog["x_mean_n"] == len(values)


def test_batch_update_excludes_nan_from_running_stats() -> None:
    """Skip NaN values in the running mean/std but still record them raw."""
    values = [1.0, math.nan, 3.0]
    dlog = monitor.batch_initialize(len(values), ["x"], extended_stats_tags=["x"])
    for batch_idx, value in enumerate(values):
        monitor.batch_update(dlog, batch_idx, {"x": value})
    monitor.batch_finalize(dlog, ["x"])

    assert dlog["x_mean"] == pytest.approx(2.0)
    assert dlog["x_mean_n"] == 2
    assert math.isnan(dlog["x"][1].item())


def test_batch_finalize_extended_stats_match_torch_reference() -> None:
    """Match `torch.median`/`.min()`/`.max()` for an `extended_stats_tags` tag."""
    torch.manual_seed(1)
    values = torch.randn(50, dtype=torch.float64)

    dlog = monitor.batch_initialize(
        len(values), ["time_step"], extended_stats_tags=["time_step"]
    )
    for batch_idx, value in enumerate(values):
        monitor.batch_update(dlog, batch_idx, {"time_step": value.item()})
    monitor.batch_finalize(dlog, ["time_step"])

    assert dlog["time_step_median"] == pytest.approx(values.median().item())
    assert dlog["time_step_min"] == pytest.approx(values.min().item())
    assert dlog["time_step_max"] == pytest.approx(values.max().item())


def test_batch_finalize_extended_stats_drop_nan() -> None:
    """Drop NaNs from the raw per-batch tensor before computing median/min/max."""
    values = [1.0, math.nan, 3.0, 2.0]
    dlog = monitor.batch_initialize(len(values), ["x"], extended_stats_tags=["x"])
    for batch_idx, value in enumerate(values):
        monitor.batch_update(dlog, batch_idx, {"x": value})
    monitor.batch_finalize(dlog, ["x"])

    assert dlog["x_median"] == pytest.approx(2.0)
    assert dlog["x_min"] == pytest.approx(1.0)
    assert dlog["x_max"] == pytest.approx(3.0)


# --------------------------------------
# Per-epoch: extended stats tags, raw tags
# --------------------------------------


def test_epoch_initialize_extended_stats_tags_selects_materialized_tags() -> None:
    """Materialize a per-epoch tensor only for tags named in `extended_stats_tags`."""
    n_epochs = 4
    epoch_dlog = monitor.epoch_initialize(
        n_epochs, ["loss_mean", "loss_std"], extended_stats_tags=["loss_mean"]
    )

    assert epoch_dlog["loss_mean"] is not None
    assert epoch_dlog["loss_std"] is None

    for epoch_idx in range(n_epochs):
        batch_dlog = {"loss_mean": float(epoch_idx), "loss_std": float(epoch_idx) * 0.1}
        monitor.epoch_update(
            epoch_dlog, epoch_idx, ["loss_mean", "loss_std"], batch_dlog
        )

    assert torch.equal(
        epoch_dlog["loss_mean"], torch.arange(n_epochs, dtype=torch.float64)
    )


def test_epoch_update_raw_tags_pool_matches_summary_stats_reference() -> None:
    """Collect a raw per-epoch tensor via `raw_tags`, then pool it for `summary_stats`."""
    torch.manual_seed(3)
    n_epochs = 3
    n_batches = 10
    epoch_dlog = monitor.epoch_initialize(n_epochs, [], raw_tags=["time_step"])

    per_epoch_values = []
    for epoch_idx in range(n_epochs):
        values = torch.rand(n_batches, dtype=torch.float64)
        per_epoch_values.append(values)

        batch_dlog = monitor.batch_initialize(
            n_batches, ["time_step"], extended_stats_tags=["time_step"]
        )
        for batch_idx, value in enumerate(values):
            monitor.batch_update(batch_dlog, batch_idx, {"time_step": value.item()})
        monitor.batch_finalize(batch_dlog, ["time_step"])

        monitor.epoch_update(
            epoch_dlog, epoch_idx, [], batch_dlog, raw_tags=["time_step"]
        )

    assert len(epoch_dlog["time_step"]) == n_epochs
    for epoch_idx in range(n_epochs):
        assert torch.equal(
            epoch_dlog["time_step"][epoch_idx], per_epoch_values[epoch_idx]
        )

    pooled = torch.cat(epoch_dlog["time_step"][1:])
    expected = torch.cat(per_epoch_values[1:])
    stats = monitor.summary_stats(pooled)

    assert stats["n"] == len(expected)
    assert stats["mean"] == pytest.approx(expected.mean().item())
    assert stats["std"] == pytest.approx(expected.std(correction=0).item())
    assert stats["median"] == pytest.approx(expected.median().item())
    assert stats["min"] == pytest.approx(expected.min().item())
    assert stats["max"] == pytest.approx(expected.max().item())


def _epoch_dlog_for_finalize(
    n_epochs: int, n_batches: int, with_time_step: bool
) -> tuple[monitor.TrainLog, list[torch.Tensor]]:
    """Build an `epoch_dlog` with `time_epoch` and, optionally, raw `time_step`."""
    raw_tags = ["time_step"] if with_time_step else []
    epoch_dlog = monitor.epoch_initialize(
        n_epochs, ["time_epoch"], extended_stats_tags=["time_epoch"], raw_tags=raw_tags
    )

    per_epoch_time_step = []
    for epoch_idx in range(n_epochs):
        batch_dlog: dict[str, Any] = {"time_epoch": 1.0 + epoch_idx}
        if with_time_step:
            values = torch.rand(n_batches, dtype=torch.float64) + epoch_idx
            per_epoch_time_step.append(values)
            batch_dlog["time_step"] = values
        monitor.epoch_update(
            epoch_dlog, epoch_idx, ["time_epoch"], batch_dlog, raw_tags=raw_tags
        )
    return epoch_dlog, per_epoch_time_step


def test_epoch_finalize_single_epoch_omits_rest_and_step_stats() -> None:
    """Report only `time_epoch_first` and `time_train` when `n_epochs == 1`."""
    epoch_dlog, _ = _epoch_dlog_for_finalize(
        n_epochs=1, n_batches=5, with_time_step=True
    )

    monitor.epoch_finalize(epoch_dlog, time_train=12.5, n_epochs=1)

    assert epoch_dlog["time_train"] == 12.5
    summary = epoch_dlog["summary"]
    assert summary["time_epoch_first"]["mean"] == pytest.approx(1.0)
    assert "time_epoch_rest" not in summary
    assert "time_step" not in summary
    assert "samples_per_sec" not in summary


def test_epoch_finalize_multi_epoch_without_time_step_omits_step_stats() -> None:
    """Skip `time_step`/`samples_per_sec` when `dlog` has no raw `time_step` list."""
    n_epochs = 4
    epoch_dlog, _ = _epoch_dlog_for_finalize(
        n_epochs=n_epochs, n_batches=5, with_time_step=False
    )

    monitor.epoch_finalize(
        epoch_dlog, time_train=1.0, n_epochs=n_epochs, global_batch_size=8
    )

    summary = epoch_dlog["summary"]
    assert summary["time_epoch_first"]["mean"] == pytest.approx(1.0)
    assert summary["time_epoch_rest"]["mean"] == pytest.approx(3.0)
    assert "time_step" not in summary
    assert "samples_per_sec" not in summary


def test_epoch_finalize_multi_epoch_pools_time_step_and_samples_per_sec() -> None:
    """Match hand-computed references for pooled `time_step`/`samples_per_sec`."""
    torch.manual_seed(4)
    n_epochs = 3
    n_batches = 10
    global_batch_size = 16
    epoch_dlog, per_epoch_time_step = _epoch_dlog_for_finalize(
        n_epochs=n_epochs, n_batches=n_batches, with_time_step=True
    )

    monitor.epoch_finalize(
        epoch_dlog,
        time_train=99.0,
        n_epochs=n_epochs,
        global_batch_size=global_batch_size,
    )

    summary = epoch_dlog["summary"]
    time_epoch_all = epoch_dlog["time_epoch"]
    expected_first = monitor.summary_stats(time_epoch_all[:1])
    expected_rest = monitor.summary_stats(time_epoch_all[1:])
    step_pool = torch.cat(per_epoch_time_step[1:])
    expected_step = monitor.summary_stats(step_pool)
    expected_sps = monitor.summary_stats(global_batch_size / step_pool)

    assert summary["time_epoch_first"] == expected_first
    assert summary["time_epoch_rest"] == expected_rest
    assert summary["time_step"] == expected_step
    assert summary["samples_per_sec"] == expected_sps


def test_summary_stats_single_process_matches_torch_reference() -> None:
    """Return exact torch reference stats when not distributed."""
    torch.manual_seed(2)
    values = torch.randn(500, dtype=torch.float64) * 2.0 + 500.0

    stats = monitor.summary_stats(values)

    assert stats["n"] == len(values)
    assert stats["mean"] == pytest.approx(values.mean().item())
    assert stats["std"] == pytest.approx(values.std(correction=0).item())
    assert stats["median"] == pytest.approx(values.median().item())
    assert stats["min"] == pytest.approx(values.min().item())
    assert stats["max"] == pytest.approx(values.max().item())


# --------------------------------------
# 2-process gloo: ship gate
# --------------------------------------


def _batch_all_reduce_exactness_worker(rank: int, world_size: int, port: int) -> None:
    """Prove `batch_all_reduce` matches a hand-computed exact global mean/std.

    Uses adversarial data (mean large relative to std, the case the naive
    `E[X^2] - E[X]^2` formula gets wrong) and unequal per-rank batch counts,
    so the exact combine cannot degenerate into an unweighted average.
    """
    init_worker(rank, world_size, port)

    torch.manual_seed(100 + rank)
    n = 37 if rank == 0 else 41
    local_values = torch.randn(n, dtype=torch.float64) * 1e-3 + 1e6

    dlog = monitor.batch_initialize(n, ["x"])
    for batch_idx, value in enumerate(local_values):
        monitor.batch_update(dlog, batch_idx, {"x": value.item()})
    monitor.batch_all_reduce(dlog, ["x"])
    monitor.batch_finalize(dlog, ["x"])

    # hand-compute the exact global reference out-of-band (test-only all_gather)
    gathered: list[list[float]] = [[] for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_values.tolist())
    all_values = torch.tensor(
        [v for shard in gathered for v in shard], dtype=torch.float64
    )

    assert dlog["x_mean_n"] == len(all_values)
    assert dlog["x_mean"] == pytest.approx(all_values.mean().item(), rel=1e-10)
    assert dlog["x_std"] == pytest.approx(all_values.std(correction=0).item(), rel=1e-6)

    distributed.finalize()


def test_two_process_batch_all_reduce_exactness_adversarial() -> None:
    """Combine adversarial per-rank running stats into the exact global mean/std."""
    run_distributed(_batch_all_reduce_exactness_worker)


def _summary_stats_worker(rank: int, world_size: int, port: int) -> None:
    """Verify `summary_stats`'s cross-rank combine against hand-computed references.

    Unequal per-rank pool sizes exercise the same exact-combine requirement as
    `_batch_all_reduce_exactness_worker`: passing a placeholder count instead
    of the real local size would silently break the mean/std combine.
    """
    init_worker(rank, world_size, port)

    torch.manual_seed(200 + rank)
    n = 23 if rank == 0 else 29
    local_values = torch.randn(n, dtype=torch.float64) * 1e-3 + 1e6

    stats = monitor.summary_stats(local_values)

    gathered: list[list[float]] = [[] for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_values.tolist())
    all_values = torch.tensor(
        [v for shard in gathered for v in shard], dtype=torch.float64
    )
    local_medians = [0.0 for _ in range(world_size)]
    torch.distributed.all_gather_object(local_medians, local_values.median().item())
    expected_median = sum(local_medians) / world_size

    assert stats["n"] == len(all_values)
    assert stats["mean"] == pytest.approx(all_values.mean().item(), rel=1e-10)
    assert stats["std"] == pytest.approx(all_values.std(correction=0).item(), rel=1e-6)
    assert stats["median"] == pytest.approx(expected_median)
    assert stats["min"] == pytest.approx(all_values.min().item())
    assert stats["max"] == pytest.approx(all_values.max().item())

    distributed.finalize()


def test_two_process_summary_stats_matches_hand_computed_references() -> None:
    """Combine mean/std/min/max exactly and median as the mean of per-rank medians."""
    run_distributed(_summary_stats_worker)
