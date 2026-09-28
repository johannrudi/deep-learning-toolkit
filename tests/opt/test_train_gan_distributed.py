"""Tests for `dlk.opt.train_gan.train_epochs` under a 2-process gloo group."""

import torch
from ddp_test_utils import init_worker, run_distributed

from dlk.loss.gradient_penalty import gradient_penalty_fd_adversarial
from dlk.opt import distributed
from dlk.opt.compile import compile_net_from_params
from dlk.opt.train_gan import train_epochs


class _Generator(torch.nn.Module):
    """Tiny conditional generator mapping `(y, z)` to samples."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3 + 2, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 4),
        )

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((y, z), dim=1))


class _Discriminator(torch.nn.Module):
    """Tiny conditional discriminator mapping `(x, y)` to a score."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4 + 3, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((x, y), dim=1))


def _gan_loss_fn(
    d_outputs_gen: torch.Tensor,
    d_outputs_data: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute a Wasserstein-style loss for discriminator and generator steps."""
    loss_g = -d_outputs_gen.mean()
    if d_outputs_data is None:
        return loss_g, None
    return d_outputs_gen.mean() - d_outputs_data.mean(), loss_g


def _gradient_penalty_fn(
    d_net: torch.nn.Module,
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    *,
    dlog: dict[str, float] | None = None,
    d_outputs_gen: torch.Tensor | None = None,
    d_outputs_data: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a gradient penalty with double backward through `d_net`.

    Exercises the riskiest DDP path: an extra discriminator forward and an
    inner `autograd.grad` w.r.t. inputs with `create_graph=True`.
    `d_outputs_gen`/`d_outputs_data` are unused; declared to conform to
    `DiscriminatorRegularizerFn`.
    """
    del d_outputs_gen, d_outputs_data
    alpha = torch.rand((x_data.size(0), 1))
    x_hat = (alpha * x_data + (1.0 - alpha) * x_gen).requires_grad_(True)
    d_outputs = d_net(x_hat, y_data)
    (grad,) = torch.autograd.grad(d_outputs.sum(), x_hat, create_graph=True)
    penalty = (grad.norm(2, dim=1) - 1.0).square().mean()
    if dlog is not None:
        dlog["reg"] = penalty.item()
        dlog["grad_norm"] = grad.norm(2, dim=1).mean().item()
    return penalty


def _make_dataloader() -> torch.utils.data.DataLoader:
    """Build the shared toy dataset's dataloader, sharded by the distributed sampler."""
    generator = torch.Generator().manual_seed(5678)
    x_data = torch.randn((64, 4), generator=generator, dtype=torch.float32)
    y_data = torch.randn((64, 3), generator=generator, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    sampler = distributed.sampler_create(dataset, shuffle=True, base_seed=0)
    assert sampler is not None
    return torch.utils.data.DataLoader(dataset, batch_size=8, sampler=sampler)


def _assert_params_synced(nets: list[torch.nn.Module], world_size: int) -> None:
    """Assert that every network's parameters are identical across ranks."""
    for net in nets:
        params = torch.nn.utils.parameters_to_vector(net.parameters())
        params_list = [torch.empty_like(params) for _ in range(world_size)]
        torch.distributed.all_gather(params_list, params)
        assert torch.equal(params_list[0], params_list[1])


def _assert_tags_synced(
    epoch_dlog: dict[str, torch.Tensor],
    tags: list[str],
    world_size: int,
) -> None:
    """Assert that reduced epoch statistics are identical across ranks."""
    for tag in tags:
        values = epoch_dlog[tag].clone()
        values_list = [torch.empty_like(values) for _ in range(world_size)]
        torch.distributed.all_gather(values_list, values)
        assert torch.equal(values_list[0], values_list[1])


def _train_gan_worker(rank: int, world_size: int, port: int) -> None:
    """Train a DDP-wrapped GAN with gradient penalty and verify synchronization."""
    ctx = init_worker(rank, world_size, port)
    distributed.seed_random_generators(42)
    dataloader = _make_dataloader()

    # per-rank initialization differs; DDP broadcasts rank 0's parameters on wrap
    g_net = distributed.wrap_net(_Generator(), ctx.device)
    d_net = distributed.wrap_net(_Discriminator(), ctx.device, broadcast_buffers=False)
    g_optimizer = torch.optim.SGD(g_net.parameters(), lr=0.05)
    d_optimizer = torch.optim.SGD(d_net.parameters(), lr=0.05)

    epoch_dlog = train_epochs(
        n_epochs=2,
        g_net=g_net,
        d_net=d_net,
        dataloader=dataloader,
        z_sample_fn=lambda batch_size: torch.randn((batch_size, 2)),
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        loss_fn=_gan_loss_fn,
        d_reg_fn=_gradient_penalty_fn,
        d_opt_pre=2,
        d_opt_post=1,
        device=ctx.device,
    )

    # generator and discriminator parameters must be identical across ranks
    _assert_params_synced([g_net, d_net], world_size)

    # reduced loss statistics must be identical across ranks
    _assert_tags_synced(
        epoch_dlog, ["g_loss_mean", "d_pre_loss_mean", "d_pre_reg_mean"], world_size
    )

    distributed.finalize()


def test_two_process_gan_training_with_gradient_penalty() -> None:
    """Train a DDP GAN with double-backward gradient penalty across 2 ranks."""
    run_distributed(_train_gan_worker)


def _gradient_penalty_fd_adversarial_fn(
    d_net: torch.nn.Module,
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    *,
    dlog: dict[str, float] | None = None,
    d_outputs_gen: torch.Tensor | None = None,
    d_outputs_data: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the adversarial finite-difference Lipschitz penalty.

    Exercises the riskiest DDP path: an extra input-gradient backward through
    the compiled critic, unwrapped from DDP, inside a DDP training step.
    `d_outputs_gen`/`d_outputs_data` are unused; declared to conform to
    `DiscriminatorRegularizerFn`.
    """
    del d_outputs_gen, d_outputs_data
    return gradient_penalty_fd_adversarial(d_net, x_gen, x_data, y_data, dlog=dlog)


def _train_gan_fd_adversarial_compiled_worker(
    rank: int, world_size: int, port: int
) -> None:
    """Train a DDP critic, compiled with aot_eager, under the adversarial FD penalty."""
    # start every worker from a clean dynamo state
    torch._dynamo.reset()

    ctx = init_worker(rank, world_size, port)
    distributed.seed_random_generators(42)
    dataloader = _make_dataloader()

    # compile the critic's forward the way applications do, then DDP-wrap it
    d_net_local = compile_net_from_params(_Discriminator(), {"backend": "aot_eager"})

    # per-rank initialization differs; DDP broadcasts rank 0's parameters on wrap
    g_net = distributed.wrap_net(_Generator(), ctx.device)
    d_net = distributed.wrap_net(d_net_local, ctx.device, broadcast_buffers=False)
    d_params_init = torch.nn.utils.parameters_to_vector(d_net.parameters()).clone()

    g_optimizer = torch.optim.SGD(g_net.parameters(), lr=0.05)
    d_optimizer = torch.optim.SGD(d_net.parameters(), lr=0.05)

    epoch_dlog = train_epochs(
        n_epochs=2,
        g_net=g_net,
        d_net=d_net,
        dataloader=dataloader,
        z_sample_fn=lambda batch_size: torch.randn((batch_size, 2)),
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        loss_fn=_gan_loss_fn,
        d_reg_fn=_gradient_penalty_fd_adversarial_fn,
        d_opt_pre=2,
        d_opt_post=1,
        device=ctx.device,
    )

    # confirm the critic actually compiled instead of silently falling back to eager
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0

    # generator and discriminator parameters must be identical across ranks
    _assert_params_synced([g_net, d_net], world_size)

    # reduced loss and regularizer statistics must be identical across ranks
    _assert_tags_synced(
        epoch_dlog,
        ["g_loss_mean", "d_pre_loss_mean", "d_pre_reg_mean", "d_pre_grad_fd_norm_mean"],
        world_size,
    )
    assert torch.all(epoch_dlog["d_pre_grad_fd_norm_mean"] > 0.0)

    # confirm the critic's parameters moved from their post-wrap initial values
    d_params_final = torch.nn.utils.parameters_to_vector(d_net.parameters())
    assert not torch.equal(d_params_init, d_params_final)

    distributed.finalize()


def test_gradient_penalty_fd_adversarial_compiled_ddp() -> None:
    """Train a DDP GAN with a compiled critic under the adversarial FD penalty."""
    run_distributed(_train_gan_fd_adversarial_compiled_worker)
