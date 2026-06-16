import torch
from torch.utils.data import TensorDataset

from dlk import distributed


def test_torchrun_env_returns_none_when_required_variables_are_missing() -> None:
    """Return None outside a torchrun environment."""
    assert distributed.torchrun_env({}) is None
    assert distributed.torchrun_env({"RANK": "0", "WORLD_SIZE": "1"}) is None


def test_torchrun_env_parses_rank_values() -> None:
    """Parse rank metadata from torchrun environment variables."""
    env = {"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}

    assert distributed.torchrun_env(env) == {
        "rank": 3,
        "local_rank": 1,
        "world_size": 8,
    }


def test_helpers_return_single_process_defaults() -> None:
    """Return safe defaults when no process group is initialized."""
    assert not distributed.is_distributed()
    assert distributed.get_rank() == 0
    assert distributed.get_local_rank() == 0
    assert distributed.get_world_size() == 1
    assert distributed.is_main_process()


def test_distributed_sampler_is_none_outside_distributed_mode() -> None:
    """Do not create a DistributedSampler in ordinary Python runs."""
    dataset = TensorDataset(torch.arange(4))

    assert distributed.create_distributed_sampler(dataset) is None
