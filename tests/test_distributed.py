import pytest
import torch
from torch.utils.data import TensorDataset
from dlk import distributed


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
