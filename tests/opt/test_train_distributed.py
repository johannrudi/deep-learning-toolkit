"""Tests for `dlk.opt.train.train_epochs` under a 2-process gloo group."""

import pathlib

import torch
from ddp_test_utils import init_worker, run_distributed

from dlk.opt import distributed
from dlk.opt.train import train_epochs
from dlk.opt.utils import checkpoint_load, checkpoint_save


def _make_dataset() -> torch.utils.data.TensorDataset:
    """Build the deterministic dataset shared by all ranks."""
    generator = torch.Generator().manual_seed(1234)
    features = torch.randn((64, 4), generator=generator, dtype=torch.float32)
    targets = torch.randn((64, 2), generator=generator, dtype=torch.float32)
    return torch.utils.data.TensorDataset(features, targets)


def _make_net() -> torch.nn.Module:
    """Build the small model used by the distributed training tests."""
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    )


def _train_worker(rank: int, world_size: int, port: int, tmp_dir: str) -> None:
    """Train with DDP and verify synchronization, checkpoints, and validation."""
    ctx = init_worker(rank, world_size, port)
    distributed.seed_random_generators(42)

    dataset = _make_dataset()
    sampler = distributed.create_distributed_sampler(dataset, shuffle=True, seed=0)
    assert sampler is not None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=sampler)

    # per-rank initialization differs; DDP broadcasts rank 0's parameters on wrap
    net = distributed.wrap_ddp(_make_net(), ctx.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    validation_calls: list[int] = []

    def validation_fn(epoch_idx: int, **kwargs: torch.nn.Module) -> None:
        """Record validation calls and check the model is unwrapped."""
        validation_calls.append(epoch_idx)
        assert not isinstance(
            kwargs["net"], torch.nn.parallel.DistributedDataParallel
        ), "validation_fn must receive the unwrapped model"

    n_epochs = 2
    epoch_dlog = train_epochs(
        n_epochs=n_epochs,
        net=net,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=torch.nn.MSELoss(),
        validation_fn=validation_fn,
        device=ctx.device,
        checkpoint_epochs=1,
        checkpoint_dir=tmp_dir,
    )

    # validation runs on the main process only (per epoch and once after)
    if rank == 0:
        assert validation_calls == [0, 1, n_epochs]
    else:
        assert validation_calls == []

    # parameters must be identical across ranks after training
    params = torch.nn.utils.parameters_to_vector(net.parameters())
    params_list = [torch.empty_like(params) for _ in range(world_size)]
    torch.distributed.all_gather(params_list, params)
    assert torch.equal(params_list[0], params_list[1])

    # reduced loss statistics must be identical across ranks
    loss_mean = epoch_dlog["loss_mean"].clone()
    loss_mean_list = [torch.empty_like(loss_mean) for _ in range(world_size)]
    torch.distributed.all_gather(loss_mean_list, loss_mean)
    assert torch.equal(loss_mean_list[0], loss_mean_list[1])

    # exactly one timestamped checkpoint directory, written by the main process
    distributed.barrier()
    checkpoint_dirs = sorted(pathlib.Path(tmp_dir).iterdir())
    assert len(checkpoint_dirs) == 1
    checkpoint_files = sorted(path.name for path in checkpoint_dirs[0].iterdir())
    assert checkpoint_files == ["net_e0.pt", "net_e1.pt", "net_e2.pt"]

    # checkpoint round-trip restores the trained parameters
    net_restored = _make_net()
    optimizer_restored = torch.optim.SGD(net_restored.parameters(), lr=0.1)
    epoch = checkpoint_load(
        checkpoint_dirs[0] / "net_e2.pt",
        net_restored,
        optimizer=optimizer_restored,
        map_location=ctx.device,
    )
    assert epoch == n_epochs
    params_restored = torch.nn.utils.parameters_to_vector(net_restored.parameters())
    assert torch.equal(params_restored, params)

    distributed.finalize_distributed()


def test_two_process_training_synchronizes_and_checkpoints(
    tmp_path: pathlib.Path,
) -> None:
    """Train under DDP with sharded data, rank-0 checkpoints, and validation."""
    run_distributed(_train_worker, args=(str(tmp_path),))


def test_checkpoint_load_strips_legacy_module_prefix(
    tmp_path: pathlib.Path,
) -> None:
    """Load a legacy checkpoint whose keys carry a `module.` prefix."""
    net_source = _make_net()
    optimizer_source = torch.optim.SGD(net_source.parameters(), lr=0.1)
    filepath = tmp_path / "net_legacy.pt"
    checkpoint_save(net_source, filepath, epoch=3, optimizer=optimizer_source)

    # rewrite the checkpoint with `module.`-prefixed keys
    checkpoint = torch.load(filepath, weights_only=True)
    checkpoint["model_state_dict"] = {
        f"module.{key}": value for key, value in checkpoint["model_state_dict"].items()
    }
    torch.save(checkpoint, filepath)

    net_restored = _make_net()
    epoch = checkpoint_load(filepath, net_restored)

    assert epoch == 3
    params_source = torch.nn.utils.parameters_to_vector(net_source.parameters())
    params_restored = torch.nn.utils.parameters_to_vector(net_restored.parameters())
    assert torch.equal(params_restored, params_source)


def test_checkpoint_save_load_round_trip(tmp_path: pathlib.Path) -> None:
    """Round-trip model and optimizer states through save and load."""
    net_source = _make_net()
    optimizer_source = torch.optim.SGD(net_source.parameters(), lr=0.1)
    filepath = tmp_path / "net_e1.pt"
    checkpoint_save(net_source, filepath, epoch=1, optimizer=optimizer_source)

    net_restored = _make_net()
    optimizer_restored = torch.optim.SGD(net_restored.parameters(), lr=0.1)
    epoch = checkpoint_load(filepath, net_restored, optimizer=optimizer_restored)

    assert epoch == 1
    params_source = torch.nn.utils.parameters_to_vector(net_source.parameters())
    params_restored = torch.nn.utils.parameters_to_vector(net_restored.parameters())
    assert torch.equal(params_restored, params_source)
    assert optimizer_restored.state_dict() == optimizer_source.state_dict()
