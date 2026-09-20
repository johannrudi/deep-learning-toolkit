"""Per-batch/per-epoch training logs: Welford mean/std, cross-rank combining, and
end-of-run summary statistics.
"""

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, TypeAlias

import torch

from dlk.opt import distributed

TrainLog: TypeAlias = dict[str, Any]


# --------------------------------------
# Per-batch: Welford mean/std, optional extended stats
# --------------------------------------


@torch.no_grad()
def batch_initialize(
    n_batches: int,
    tags: Sequence[str],
    extended_stats_tags: Sequence[str] = (),
) -> TrainLog:
    """Initialize batch-level training log storage and running statistics.

    Args:
        n_batches: Number of batches in the current epoch.
        tags: Metric names to track.
        extended_stats_tags: Subset of `tags` that additionally get
            `median`/`min`/`max` at `batch_finalize`, computed from a fully
            materialized per-batch tensor.

    Returns:
        Mutable dictionary storing per-tag values and aggregate statistics.
    """
    dlog: dict[str, Any] = {"n_batches": n_batches}
    for tag in tags:
        if tag in extended_stats_tags:
            dlog[tag] = torch.empty((n_batches,), dtype=torch.float64)
        else:
            dlog[tag] = None
        dlog[f"{tag}_mean_n"] = 0
        dlog[f"{tag}_mean"] = 0.0
        dlog[f"{tag}_m2"] = 0.0
        dlog[f"{tag}_std"] = None
    return dlog


@torch.no_grad()
def batch_update(
    dlog: MutableMapping[str, Any],
    batch_idx: int,
    values: Mapping[str, float | int | torch.Tensor],
) -> None:
    """Update batch-level logs with metric values from a single batch.

    Uses Welford's online algorithm for the running mean and `M2` (sum of
    squared deviations from the running mean), which avoids the catastrophic
    cancellation.

    Args:
        dlog: Training log dictionary from `batch_initialize`.
        batch_idx: Index of the current batch.
        values: Mapping from metric names to scalar values.

    Returns:
        None.
    """
    for tag, val in values.items():
        if isinstance(val, torch.Tensor):
            val = float(val.detach().item())
        else:
            val = float(val)
        if dlog[tag] is not None:  # if this is an extended tag
            dlog[tag][batch_idx] = val
        if not math.isnan(val):
            n = dlog[f"{tag}_mean_n"] + 1
            delta = val - dlog[f"{tag}_mean"]
            dlog[f"{tag}_mean"] += delta / n
            delta2 = val - dlog[f"{tag}_mean"]
            dlog[f"{tag}_m2"] += delta * delta2
            dlog[f"{tag}_mean_n"] = n


def _distributed_combine_mean_m2(
    n_local: torch.Tensor,
    mean_local: torch.Tensor,
    m2_local: torch.Tensor,
    device: torch.device | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Combine per-group `(n, mean, M2)` into an exact global triple across ranks.

    Welford's `M2` does not combine with a single all-reduce; it needs the
    between-group cross term. This uses the parallel-axis theorem,
    `sum((x_i - c)^2) = M2 + n * (mean - c)^2`, in two sequential all-reduces
    instead of gathering every raw value:

        round 1: combine `n` and `n * mean` into the exact global `n`, `mean`.
        round 2: add each rank's between-group contribution,
            `M2_local + n_local * (mean_local - mean_global)^2`, then sum.

    `n_local` must be this rank's actual element count for the group, since
    the round-2 correction term scales with it; passing anything else breaks
    the exactness of the combined `M2`.

    No-op (inputs returned as-is) when not distributed.

    Args:
        n_local: Per-group element counts on this rank.
        mean_local: Per-group means on this rank.
        m2_local: Per-group sums of squared deviations on this rank.
        device: Device holding the reduction buffers; required to be this
            process's GPU for the NCCL backend, CPU (`None`) for gloo.

    Returns:
        Tuple `(n_global, mean_global, m2_global)`, exact across ranks.
    """
    if not distributed.is_distributed():
        return n_local, mean_local, m2_local
    # round 1: combine n and n * mean into the exact global n and mean
    packed = torch.stack([n_local, n_local * mean_local]).to(device, torch.float64)
    distributed.all_reduce_sum_(packed)
    n_global, mean_global = packed[0], packed[1] / packed[0].clamp_min(1)
    # round 2: each rank's contribution to global M2 via the parallel-axis theorem
    delta = mean_local.to(device, torch.float64) - mean_global
    correction = (
        m2_local.to(device, torch.float64)
        + n_local.to(device, torch.float64) * delta**2
    )
    distributed.all_reduce_sum_(correction)
    return n_global, mean_global, correction


@torch.no_grad()
def batch_all_reduce(
    dlog: MutableMapping[str, Any],
    tags: Sequence[str],
    device: torch.device | None = None,
) -> None:
    """Combine batch-level running aggregates across all ranks, in place.

    Combines `{tag}_mean_n`, `{tag}_mean`, and `{tag}_m2` for every tag with
    two all-reduces (see `_distributed_combine_mean_m2`), so that a subsequent
    `batch_finalize` yields the exact global mean and standard deviation over
    all ranks. Call before `batch_finalize` (which divides by the counts).
    No-op when not distributed.

    Args:
        dlog: Training log dictionary from `batch_initialize`.
        tags: Metric names to reduce.
        device: Device holding the reduction buffer; required to be this
            process's GPU for the NCCL backend, CPU (`None`) for gloo.

    Returns:
        None.
    """
    if not distributed.is_distributed():
        return
    n_local = torch.tensor(
        [dlog[f"{tag}_mean_n"] for tag in tags], dtype=torch.float64, device=device
    )
    mean_local = torch.tensor(
        [dlog[f"{tag}_mean"] for tag in tags], dtype=torch.float64, device=device
    )
    m2_local = torch.tensor(
        [dlog[f"{tag}_m2"] for tag in tags], dtype=torch.float64, device=device
    )
    n_global, mean_global, m2_global = _distributed_combine_mean_m2(
        n_local, mean_local, m2_local, device
    )
    n_list = n_global.cpu().tolist()
    mean_list = mean_global.cpu().tolist()
    m2_list = m2_global.cpu().tolist()
    for tag_idx, tag in enumerate(tags):
        dlog[f"{tag}_mean_n"] = int(n_list[tag_idx])
        dlog[f"{tag}_mean"] = mean_list[tag_idx]
        dlog[f"{tag}_m2"] = m2_list[tag_idx]


@torch.no_grad()
def batch_finalize(
    dlog: MutableMapping[str, Any],
    tags: Sequence[str],
) -> None:
    """Finalize batch-level running statistics for the requested metric tags.

    Derives `std` from `M2` (population standard deviation, `sqrt(M2 / n)`).
    A tag whose raw per-batch values were materialized at `batch_initialize`
    (an `extended_stats_tags` tag) also gets `{tag}_median/min/max`, computed
    locally on this rank with NaNs dropped; combining these across ranks is
    the job of `summary_stats`, not this function.

    Args:
        dlog: Training log dictionary containing running aggregates.
        tags: Metric names to finalize.

    Returns:
        None.
    """
    for tag in tags:
        assert dlog[f"{tag}_std"] is None
        assert not math.isnan(dlog[f"{tag}_mean"])
        assert not math.isnan(dlog[f"{tag}_m2"])
        if 0 < dlog[f"{tag}_mean_n"]:
            variance = dlog[f"{tag}_m2"] / dlog[f"{tag}_mean_n"]
            dlog[f"{tag}_std"] = torch.sqrt(
                torch.tensor(variance, dtype=torch.float64)
            ).item()
        else:
            dlog[f"{tag}_mean"] = 0.0
            dlog[f"{tag}_m2"] = 0.0
            dlog[f"{tag}_std"] = 0.0

        if dlog[tag] is not None:
            raw = dlog[tag]
            raw = raw[~torch.isnan(raw)]
            if raw.numel() > 0:
                dlog[f"{tag}_median"] = raw.median().item()
                dlog[f"{tag}_min"] = raw.min().item()
                dlog[f"{tag}_max"] = raw.max().item()
            else:
                dlog[f"{tag}_median"] = 0.0
                dlog[f"{tag}_min"] = 0.0
                dlog[f"{tag}_max"] = 0.0


# --------------------------------------
# Per-epoch
# --------------------------------------


@torch.no_grad()
def epoch_initialize(
    n_epochs: int,
    tags: Sequence[str],
    extended_stats_tags: Sequence[str] = (),
    raw_tags: Sequence[str] = (),
) -> TrainLog:
    """Initialize epoch-level training log storage.

    Args:
        n_epochs: Total number of epochs to store.
        tags: Metric names to track.
        extended_stats_tags: Subset of `tags` that get a per-epoch tensor of
            values, filled in by `epoch_update`. A tag not listed here stays
            `None` and is not tracked across epochs.
        raw_tags: Metric names, distinct from `tags`, whose raw per-batch
            tensor (from a `batch_initialize` `extended_stats_tags` tag) is
            collected into a plain list, one entry per epoch, by
            `epoch_update`. Meant for pooling across epochs later (e.g. with
            `torch.cat` and `summary_stats`), without retaining the rest of
            each epoch's batch-level log.

    Returns:
        Mutable dictionary storing per-epoch metrics and raw per-epoch tensors.
    """
    dlog: dict[str, Any] = {}
    for tag in tags:
        if tag in extended_stats_tags:
            dlog[tag] = torch.empty((n_epochs,), dtype=torch.float64)
        else:
            dlog[tag] = None
    for tag in raw_tags:
        dlog[tag] = []
    return dlog


@torch.no_grad()
def epoch_update(
    dlog: MutableMapping[str, Any],
    epoch_idx: int,
    tags: Sequence[str],
    batch_dlog: Mapping[str, Any],
    raw_tags: Sequence[str] = (),
) -> None:
    """Update epoch-level logs using finalized metrics from one epoch.

    Args:
        dlog: Epoch-level training log dictionary.
        epoch_idx: Index of the epoch to update.
        tags: Metric names to copy from `batch_dlog`.
        batch_dlog: Finalized batch-level metrics for one epoch.
        raw_tags: Metric names whose raw per-batch tensor from `batch_dlog`
            is appended to `dlog[tag]` (a list, one entry per epoch); see
            `epoch_initialize`.

    Returns:
        None.
    """
    for tag in tags:
        if dlog[tag] is not None:
            dlog[tag][epoch_idx] = batch_dlog[tag]
    for tag in raw_tags:
        dlog[tag].append(batch_dlog[tag])


def epoch_finalize(
    dlog: MutableMapping[str, Any],
    time_train: float,
    n_epochs: int,
    global_batch_size: int | None = None,
    device: torch.device | None = None,
) -> None:
    """Finalize the epoch-level log with total time and an end-of-run summary.

    Splits `dlog["time_epoch"]` into the first epoch (one-time overhead:
    dataloader worker spin-up, cudnn autotune, `torch.compile` warm-up) and the
    rest, combining each pool's stats across ranks via `summary_stats`. When
    `dlog["time_step"]` (the `raw_tags` per-epoch list from `epoch_update`) is
    present, also pools epochs `1..n_epochs-1`'s per-step time and, when
    `global_batch_size` is given, derives a matching samples/sec pool.

    Args:
        dlog: Epoch-level training log dictionary, from `epoch_initialize`,
            with `dlog["time_epoch"]` populated by `epoch_update` for every
            epoch.
        time_train: Total training wall-clock time in seconds.
        n_epochs: Number of epochs trained.
        global_batch_size: Batch size summed across all processes; when
            given, adds a `samples_per_sec` entry to `dlog["summary"]`
            (requires `dlog["time_step"]`).
        device: Device holding the reduction buffers for `summary_stats`.

    Returns:
        None.
    """
    dlog["time_train"] = time_train

    time_epoch_all = dlog["time_epoch"]
    summary: dict[str, Any] = {
        "time_epoch_first": summary_stats(time_epoch_all[:1], device=device)
    }

    if n_epochs > 1:
        summary["time_epoch_rest"] = summary_stats(time_epoch_all[1:], device=device)

        if dlog.get("time_step"):
            step_pool = torch.cat(dlog["time_step"][1:])
            if step_pool.numel() > 0:
                summary["time_step"] = summary_stats(step_pool, device=device)
                if global_batch_size is not None:
                    sps_pool = global_batch_size / step_pool
                    summary["samples_per_sec"] = summary_stats(sps_pool, device=device)

    dlog["summary"] = summary


# --------------------------------------
# End-of-run summary
# --------------------------------------


def summary_stats(
    values: torch.Tensor,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute mean/std/median/min/max for a local pool of values, combined across ranks.

    `mean`/`std` combine exactly across ranks via the parallel-axis theorem
    (see `_distributed_combine_mean_m2`), using this rank's actual element
    count, so the combine is exact regardless of whether pool sizes differ
    across ranks. `median` combines as the mean of each rank's local median,
    an approximation under DDP the caller should note when logging it (exact
    when single-process). `min`/`max` combine exactly via
    `distributed.all_reduce_min_`/`all_reduce_max_`. No-op combine
    (single-rank stats returned as-is) when not distributed.

    Operates on an already fully materialized local tensor with plain
    `torch.mean`/`.var`/`.median`/`.min`/`.max`, not a streaming accumulator;
    meant for a single end-of-training call over a bounded pool (e.g. one
    training run's per-epoch or per-step values), not per-batch use.

    Args:
        values: Local pool of values on this rank, one dimension, non-empty.
        device: Device holding the reduction buffers; required to be this
            process's GPU for the NCCL backend, CPU (`None`) for gloo.

    Returns:
        Dictionary with keys `n` (global count), `mean`, `std`, `median`,
        `min`, `max`.
    """
    values = values.to(torch.float64)
    n_local = torch.tensor([float(values.numel())], dtype=torch.float64)
    mean_local = torch.tensor([values.mean().item()], dtype=torch.float64)
    m2_local = torch.tensor(
        [values.var(unbiased=False).item() * values.numel()], dtype=torch.float64
    )

    n_global, mean_global, m2_global = _distributed_combine_mean_m2(
        n_local, mean_local, m2_local, device
    )
    std_global = torch.sqrt(m2_global.clamp_min(0) / n_global.clamp_min(1))

    median_global = values.median().to(device, torch.float64).reshape(1)
    min_global = values.min().to(device, torch.float64).reshape(1)
    max_global = values.max().to(device, torch.float64).reshape(1)
    if distributed.is_distributed():
        distributed.all_reduce_sum_(median_global)
        median_global = median_global / distributed.get_world_size()
        distributed.all_reduce_min_(min_global)
        distributed.all_reduce_max_(max_global)

    return {
        "n": float(n_global.item()),
        "mean": float(mean_global.item()),
        "std": float(std_global.item()),
        "median": float(median_global.item()),
        "min": float(min_global.item()),
        "max": float(max_global.item()),
    }
