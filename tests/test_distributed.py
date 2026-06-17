import pathlib

import pytest
import torch
import torch.distributed as torch_dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, TensorDataset

from dlk import distributed
from dlk.opt import train as train_mod
from dlk.opt.train import _distributed_reduce_batch_dlog
from dlk.opt.utils import checkpoint_load, checkpoint_save


class _FakeDDP(DistributedDataParallel):
    """DDP subclass that skips process-group init for isinstance-based tests."""

    def __init__(self, module: torch.nn.Module) -> None:
        torch.nn.Module.__init__(self)
        self.module = module


@pytest.fixture
def fake_dist(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Pretend a 2-rank gloo process group is up (this rank is rank 1)."""
    state: dict = {"barrier_kwargs": None, "destroy_calls": 0}
    for name, value in [
        ("is_available", lambda: True),
        ("is_initialized", lambda: True),
        ("get_rank", lambda: 1),
        ("get_world_size", lambda: 2),
        ("get_backend", lambda: "gloo"),
        ("barrier", lambda **kw: state.update(barrier_kwargs=kw)),
        ("destroy_process_group", lambda: state.update(destroy_calls=state["destroy_calls"] + 1)),
    ]:
        monkeypatch.setattr(distributed.dist, name, value)
    return state


def test_torchrun_env_returns_none_when_required_variables_are_missing() -> None:
    """Return None outside a torchrun environment."""
    assert distributed.torchrun_env({}) is None
    assert distributed.torchrun_env({"RANK": "0", "WORLD_SIZE": "1"}) is None
    assert distributed.torchrun_env({"RANK": "0", "LOCAL_RANK": "0"}) is None


def test_torchrun_env_parses_rank_values() -> None:
    """Parse rank metadata from torchrun environment variables."""
    env = {"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}

    assert distributed.torchrun_env(env) == {
        "rank": 3,
        "local_rank": 1,
        "world_size": 8,
    }


def test_torchrun_env_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read from os.environ when no mapping is supplied."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    assert distributed.torchrun_env() == {
        "rank": 0,
        "local_rank": 0,
        "world_size": 2,
    }


@pytest.mark.parametrize(
    "env",
    [
        {"RANK": "-1", "LOCAL_RANK": "0", "WORLD_SIZE": "1"},
        {"RANK": "0", "LOCAL_RANK": "-1", "WORLD_SIZE": "1"},
        {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "0"},
    ],
)
def test_torchrun_env_rejects_invalid_values(env: dict[str, str]) -> None:
    """Reject negative ranks and non-positive world sizes."""
    with pytest.raises(ValueError):
        distributed.torchrun_env(env)


def test_helpers_return_single_process_defaults() -> None:
    """Return safe defaults when no process group is initialized."""
    assert not distributed.is_distributed()
    assert distributed.get_rank() == 0
    assert distributed.get_local_rank() == 0
    assert distributed.get_world_size() == 1
    assert distributed.is_main_process()


def test_get_local_rank_reads_torchrun_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return LOCAL_RANK from environment even when no process group is up."""
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert distributed.get_local_rank() == 1


def test_init_process_group_returns_device_without_torchrun() -> None:
    """Return a usable device and skip group init outside torchrun."""
    device = distributed.init_process_group()

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected
    assert not distributed.is_distributed()


def test_barrier_and_cleanup_are_noops_outside_distributed() -> None:
    """Synchronization helpers must be safe in single-process runs."""
    distributed.barrier()
    distributed.cleanup()
    assert not distributed.is_distributed()


def test_distributed_sampler_is_none_outside_distributed_mode() -> None:
    """Do not create a DistributedSampler in ordinary Python runs."""
    dataset = TensorDataset(torch.arange(4))

    assert distributed.create_distributed_sampler(dataset) is None


def test_wrap_ddp_is_identity_outside_distributed_mode() -> None:
    """Return the unwrapped model when distributed is not active."""
    model = torch.nn.Linear(2, 2)

    assert distributed.wrap_ddp(model) is model


def test_unwrap_ddp_returns_module_unchanged_for_plain_model() -> None:
    """Return the same module when it is not DDP-wrapped."""
    model = torch.nn.Linear(2, 2)

    assert distributed.unwrap_ddp(model) is model


def test_reduce_batch_dlog_is_noop_outside_distributed() -> None:
    """Skip the all-reduce path when no process group is initialized."""
    dlog = {
        "loss_mean_n": 5,
        "loss_mean": 1.25,
        "loss_sq_mean": 2.5,
        "loss_std": 0.5,
    }
    snapshot = dict(dlog)

    _distributed_reduce_batch_dlog(dlog, ["loss"])

    assert dlog == snapshot


def test_checkpoint_load_restores_model_and_optimizer(tmp_path: pathlib.Path) -> None:
    """Round-trip a checkpoint and verify weights and optimizer state restore."""
    torch.manual_seed(0)
    model_a = torch.nn.Linear(3, 2)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.1)

    # take an optimizer step so state is non-empty
    inputs = torch.randn(4, 3)
    targets = torch.randn(4, 2)
    loss = ((model_a(inputs) - targets) ** 2).mean()
    loss.backward()
    optimizer_a.step()

    path = tmp_path / "ckpt.pt"
    checkpoint_save(model_a, path, epoch=7, optimizer=optimizer_a)

    model_b = torch.nn.Linear(3, 2)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.1)

    epoch = checkpoint_load(path, model_b, optimizer=optimizer_b, map_location="cpu")

    assert epoch == 7
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)
    assert optimizer_b.state_dict()["state"] == optimizer_a.state_dict()["state"]


def test_checkpoint_load_without_optimizer(tmp_path: pathlib.Path) -> None:
    """Allow loading model weights without restoring optimizer state."""
    model_a = torch.nn.Linear(2, 2)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
    path = tmp_path / "ckpt.pt"
    checkpoint_save(model_a, path, epoch=3, optimizer=optimizer_a)

    model_b = torch.nn.Linear(2, 2)

    epoch = checkpoint_load(path, model_b, map_location="cpu")

    assert epoch == 3
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_helpers_and_sampler_use_distributed_state(fake_dist: dict) -> None:
    """Rank, world size, main-process gate, and sampler shard from fake group."""
    assert distributed.is_distributed()
    assert distributed.get_rank() == 1
    assert distributed.get_world_size() == 2
    assert not distributed.is_main_process()

    sampler = distributed.create_distributed_sampler(
        TensorDataset(torch.arange(8)), shuffle=False
    )
    assert isinstance(sampler, DistributedSampler)
    assert sampler.num_replicas == 2
    assert sampler.rank == 1


def test_barrier_and_cleanup_dispatch_to_dist(fake_dist: dict) -> None:
    """barrier() goes to dist (gloo: no device_ids); cleanup() destroys group."""
    distributed.barrier()
    distributed.cleanup()

    assert fake_dist["barrier_kwargs"] == {}
    assert fake_dist["destroy_calls"] == 1


def test_unwrap_and_checkpoint_handle_ddp_wrapper(tmp_path: pathlib.Path) -> None:
    """unwrap_ddp peels module; checkpoint_save stores plain state dict."""
    inner = torch.nn.Linear(2, 2)
    wrapped = _FakeDDP(inner)
    optimizer = torch.optim.SGD(inner.parameters(), lr=0.1)

    assert distributed.unwrap_ddp(wrapped) is inner

    path = tmp_path / "ddp.pt"
    checkpoint_save(wrapped, path, epoch=1, optimizer=optimizer)
    checkpoint = torch.load(path, weights_only=False)
    assert set(checkpoint["model_state_dict"]) == set(inner.state_dict())


def test_reduce_batch_dlog_aggregates_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-reduce sums per-rank stats; reduce rebuilds mean, sq_mean, std."""
    monkeypatch.setattr(train_mod.dist_utils, "is_distributed", lambda: True)
    monkeypatch.setattr(train_mod.dist_utils, "get_local_rank", lambda: 0)

    def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
        assert op is torch_dist.ReduceOp.SUM
        tensor.mul_(2)  # simulate 2 ranks with identical local stats

    monkeypatch.setattr(train_mod.torch_dist, "all_reduce", fake_all_reduce)

    # Per-rank n=4, mean=2.0, sq_mean=5.0 → variance=1.0, std=1.0 after reduce.
    dlog = {"loss_mean_n": 4, "loss_mean": 2.0, "loss_sq_mean": 5.0, "loss_std": None}
    _distributed_reduce_batch_dlog(dlog, ["loss"])

    assert dlog["loss_mean_n"] == 8
    assert dlog["loss_mean"] == pytest.approx(2.0)
    assert dlog["loss_sq_mean"] == pytest.approx(5.0)
    assert dlog["loss_std"] == pytest.approx(1.0)
