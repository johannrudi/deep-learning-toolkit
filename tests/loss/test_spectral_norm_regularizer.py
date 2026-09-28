"""Unit tests for `SpectralNormRegularizer` in `dlk.loss.spectral_penalty`."""

from collections.abc import Callable
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from dlk.loss.spectral_penalty import SpectralNormRegularizer
from dlk.opt import distributed
from dlk.opt.train_gan import train_epochs

_UNUSED_BATCH = torch.zeros(1)


def _matrix_with_gap(
    rows: int, cols: int, singular_values: list[float]
) -> torch.Tensor:
    """Build a `rows x cols` matrix with the given singular values, for a clear spectral gap."""
    u, _ = torch.linalg.qr(torch.randn(rows, rows))
    v, _ = torch.linalg.qr(torch.randn(cols, cols))
    s = torch.zeros(rows, cols)
    for i, value in enumerate(singular_values):
        s[i, i] = value
    return u @ s @ v.T


# --------------------------------------
# Sigma matches `torch.linalg.matrix_norm`
# --------------------------------------


def _expected_reshape(layer: nn.Module) -> torch.Tensor:
    """Reshape `layer.weight` the way `SpectralNormRegularizer` does, for the expected `matrix_norm`."""
    weight = cast(torch.Tensor, layer.weight).detach()
    if isinstance(layer, nn.ConvTranspose2d):
        weight = weight.transpose(0, 1)
    return weight.reshape(weight.size(0), -1)


@pytest.mark.parametrize(
    "make_layer",
    [
        lambda: nn.Linear(6, 4, bias=False),
        lambda: nn.Conv2d(3, 5, kernel_size=3, bias=False),
        lambda: nn.ConvTranspose2d(3, 5, kernel_size=3, bias=False),
        lambda: weight_norm(nn.Linear(5, 3, bias=False)),
    ],
    ids=["linear", "conv2d", "convtranspose2d", "weight_norm_linear"],
)
def test_sigma_matches_matrix_norm_after_warmup(
    make_layer: Callable[[], nn.Module],
) -> None:
    """After warm-up, a layer's sigma matches its (correctly reshaped) `matrix_norm`."""
    torch.manual_seed(20)
    layer = make_layer()
    net = nn.Sequential(layer)
    reg = SpectralNormRegularizer(n_warmup_iterations=300)
    dlog: dict[str, float] = {}

    reg(net, _UNUSED_BATCH, _UNUSED_BATCH, _UNUSED_BATCH, dlog=dlog)

    expected = torch.linalg.matrix_norm(_expected_reshape(layer), ord=2).item()
    assert dlog["spectral_norm"] == pytest.approx(expected, rel=1e-3)


# --------------------------------------
# Rank-one gradient
# --------------------------------------


@pytest.mark.parametrize("max_norm", [None, 1.0], ids=["published", "hinge"])
def test_gradient_matches_rank_one_svd_formula(max_norm: float | None) -> None:
    """The parameter gradient equals the rank-one formula built from SVD's top singular vectors."""
    torch.manual_seed(10)
    weight = _matrix_with_gap(4, 5, [3.0, 1.0, 0.4])
    layer = nn.Linear(5, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    net = nn.Sequential(layer)
    reg = SpectralNormRegularizer(
        penalty_weight=2.0, max_norm=max_norm, n_warmup_iterations=200
    )

    penalty = reg.penalty(net)
    penalty.backward()

    u_svd, s_svd, vh_svd = torch.linalg.svd(weight, full_matrices=False)
    sigma = s_svd[0]
    excess = (
        sigma if max_norm is None else sigma - max_norm
    )  # sigma (3.0) exceeds k (1.0)
    expected_grad = 2.0 * excess * torch.outer(u_svd[:, 0], vh_svd[0, :])

    assert layer.weight.grad is not None
    torch.testing.assert_close(layer.weight.grad, expected_grad, atol=1e-3, rtol=1e-3)


def test_gradient_is_rank_one_with_single_non_converged_power_step() -> None:
    """A single power step still gives an exactly rank-one gradient with Frobenius norm `lambda * sigma`.

    Guards against running the power iteration with grad enabled on the
    non-detached weight, which would leak extra gradient terms into the
    iteration itself and break the exact rank-one structure.
    """
    torch.manual_seed(40)
    layer = nn.Linear(6, 5, bias=False)
    net = nn.Sequential(layer)
    reg = SpectralNormRegularizer(penalty_weight=3.0, n_warmup_iterations=1)

    penalty = reg.penalty(net)
    penalty.backward()

    assert layer.weight.grad is not None
    grad_svdvals = torch.linalg.svdvals(layer.weight.grad)
    assert grad_svdvals[1] < 1e-5 * grad_svdvals[0]

    sigma = torch.sqrt(2.0 * penalty / reg.penalty_weight)
    expected_frobenius_norm = reg.penalty_weight * sigma
    torch.testing.assert_close(
        torch.linalg.matrix_norm(layer.weight.grad, ord="fro"),
        expected_frobenius_norm,
        atol=1e-4,
        rtol=1e-4,
    )


def test_hinge_form_zero_penalty_and_zero_grad_when_sigma_below_target() -> None:
    """The hinge form gives 0 and zero gradients when sigma is below the target."""
    torch.manual_seed(12)
    layer = nn.Linear(4, 3, bias=False)
    net = nn.Sequential(layer)
    reg = SpectralNormRegularizer(max_norm=10.0, n_warmup_iterations=50)

    penalty = reg.penalty(net)
    penalty.backward()

    assert penalty.item() == 0.0
    assert layer.weight.grad is not None
    torch.testing.assert_close(layer.weight.grad, torch.zeros_like(layer.weight))


# --------------------------------------
# Skipping and unwrapping
# --------------------------------------


def test_spectral_norm_wrapped_layer_is_skipped() -> None:
    """A layer wrapped with `parametrizations.spectral_norm` is skipped."""
    torch.manual_seed(24)
    layer = spectral_norm(nn.Linear(5, 4, bias=False))
    net = nn.Sequential(layer)
    reg = SpectralNormRegularizer(n_warmup_iterations=50)
    dlog: dict[str, float] = {}

    penalty = reg(net, _UNUSED_BATCH, _UNUSED_BATCH, _UNUSED_BATCH, dlog=dlog)

    assert torch.equal(penalty, torch.zeros(()))
    assert dlog == {}


def test_ddp_wrapped_network_is_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DDP wrapper is unwrapped through `dlk.opt.distributed.unwrap_net` (monkeypatched: no process group here)."""
    torch.manual_seed(25)
    inner = nn.Sequential(nn.Linear(5, 4, bias=False))
    wrapper = object()
    monkeypatch.setattr(
        distributed, "unwrap_net", lambda net: inner if net is wrapper else net
    )
    reg = SpectralNormRegularizer(n_warmup_iterations=50)
    dlog: dict[str, float] = {}

    penalty = reg(wrapper, _UNUSED_BATCH, _UNUSED_BATCH, _UNUSED_BATCH, dlog=dlog)  # type: ignore[arg-type]

    assert penalty.item() > 0.0
    assert "spectral_norm" in dlog


def test_no_qualifying_layer_returns_zero_and_no_dlog_keys() -> None:
    """A network with no qualifying layer returns a 0-dim zero tensor and logs nothing."""
    net = nn.Sequential(nn.ReLU())
    reg = SpectralNormRegularizer()
    dlog: dict[str, float] = {}

    penalty = reg(net, _UNUSED_BATCH, _UNUSED_BATCH, _UNUSED_BATCH, dlog=dlog)

    assert torch.equal(penalty, torch.zeros(()))
    assert dlog == {}


# --------------------------------------
# State
# --------------------------------------


def test_state_recreated_after_dtype_change() -> None:
    """The state is recreated after `net.double()`, matching the new dtype's `matrix_norm`."""
    torch.manual_seed(26)
    layer = nn.Linear(5, 4, bias=False)
    net = nn.Sequential(layer)
    reg = SpectralNormRegularizer(n_warmup_iterations=200)

    reg.penalty(net)  # warm up in float32
    penalty = reg.penalty(net.double())

    assert penalty.dtype == torch.float64
    expected_sigma = torch.linalg.matrix_norm(layer.weight.detach(), ord=2)
    sigma = torch.sqrt(2.0 * penalty)
    torch.testing.assert_close(sigma, expected_sigma, atol=1e-3, rtol=1e-3)


def test_two_single_steps_equal_one_two_step_warmup() -> None:
    """Two calls with one warm-up and one power step match one call with a two-step warm-up."""
    torch.manual_seed(41)
    layer = nn.Linear(5, 4, bias=False)
    net = nn.Sequential(layer)

    reg_two_calls = SpectralNormRegularizer(
        seed=3, n_warmup_iterations=1, n_power_iterations=1
    )
    reg_two_calls.penalty(net)
    penalty_two_calls = reg_two_calls.penalty(net)

    reg_one_call = SpectralNormRegularizer(seed=3, n_warmup_iterations=2)
    penalty_one_call = reg_one_call.penalty(net)

    torch.testing.assert_close(penalty_two_calls, penalty_one_call)


def test_pruned_layer_reconverges_to_the_same_penalty() -> None:
    """A layer dropped from the state (net not seen) reconverges to the same penalty when seen again."""
    torch.manual_seed(42)
    layer = nn.Linear(5, 4, bias=False)
    net = nn.Sequential(layer)
    reg = SpectralNormRegularizer(n_warmup_iterations=1)

    first = reg.penalty(net)
    reg.penalty(nn.Sequential(nn.ReLU()))
    second = reg.penalty(net)

    torch.testing.assert_close(first, second)


def test_same_seed_gives_identical_penalty_and_gradient_after_warmup() -> None:
    """Two regularizers with the same seed give identical penalties and gradients."""
    torch.manual_seed(30)
    layer = nn.Linear(5, 4, bias=False)
    net = nn.Sequential(layer)
    reg_a = SpectralNormRegularizer(seed=7, n_warmup_iterations=50)
    reg_b = SpectralNormRegularizer(seed=7, n_warmup_iterations=50)

    penalty_a = reg_a.penalty(net)
    penalty_a.backward()
    assert layer.weight.grad is not None
    grad_a = layer.weight.grad.clone()
    layer.weight.grad = None

    penalty_b = reg_b.penalty(net)
    penalty_b.backward()
    assert layer.weight.grad is not None
    grad_b = layer.weight.grad

    torch.testing.assert_close(penalty_a, penalty_b)
    torch.testing.assert_close(grad_a, grad_b)


def test_different_seeds_give_different_penalty_after_single_call() -> None:
    """Two regularizers with different seeds give different penalties after one call."""
    torch.manual_seed(31)
    layer = nn.Linear(5, 4, bias=False)
    net = nn.Sequential(layer)
    reg_a = SpectralNormRegularizer(seed=1, n_warmup_iterations=1)
    reg_b = SpectralNormRegularizer(seed=2, n_warmup_iterations=1)

    penalty_a = reg_a.penalty(net)
    penalty_b = reg_b.penalty(net)

    assert penalty_a.item() != pytest.approx(penalty_b.item())


# --------------------------------------
# Loop test: compiled critic through `train_epochs`
# --------------------------------------

_BATCH_SIZE = 4
_N_BATCHES = 2
_X_SIZE = 3
_Y_SIZE = 2
_Z_SIZE = 2


class _Generator(nn.Module):
    """Tiny conditional generator mapping `(y, z)` to samples."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(_Y_SIZE + _Z_SIZE, 8),
            nn.Tanh(),
            nn.Linear(8, _X_SIZE),
        )

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((y, z), dim=1))


class _Discriminator(nn.Module):
    """Tiny conditional discriminator mapping `(x, y)` to a score."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(_X_SIZE + _Y_SIZE, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((x, y), dim=1))


def _make_dataloader() -> torch.utils.data.DataLoader:
    """Build a small deterministic dataloader for the loop test."""
    generator = torch.Generator().manual_seed(0)
    x_data = torch.randn(
        (_BATCH_SIZE * _N_BATCHES, _X_SIZE), generator=generator, dtype=torch.float32
    )
    y_data = torch.randn(
        (_BATCH_SIZE * _N_BATCHES, _Y_SIZE), generator=generator, dtype=torch.float32
    )
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    return torch.utils.data.DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=False)


def _z_sample_fn(batch_size: int) -> torch.Tensor:
    """Sample latent vectors for the generator."""
    return torch.randn((batch_size, _Z_SIZE))


def _loss_fn(
    d_outputs_gen: torch.Tensor, d_outputs_data: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute a Wasserstein-style loss for discriminator and generator steps."""
    if d_outputs_data is None:
        return -d_outputs_gen.mean(), None
    return d_outputs_gen.mean() - d_outputs_data.mean(), -d_outputs_gen.mean()


def test_compiled_critic_trains_and_logs_spectral_norm(
    compile_aot_eager: Callable[[torch.nn.Module], CompileCounterWithBackend],
) -> None:
    """`SpectralNormRegularizer` as `d_reg_fn` trains a compiled critic and logs
    `spectral_norm`."""
    torch.manual_seed(0)
    g_net = _Generator()
    d_net = _Discriminator()
    g_optimizer = torch.optim.SGD(g_net.parameters(), lr=0.05)
    d_optimizer = torch.optim.SGD(d_net.parameters(), lr=0.05)
    compile_aot_eager(d_net)

    initial_params = [parameter.detach().clone() for parameter in d_net.parameters()]

    epoch_dlog = train_epochs(
        n_epochs=2,
        g_net=g_net,
        d_net=d_net,
        dataloader=_make_dataloader(),
        z_sample_fn=_z_sample_fn,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        loss_fn=_loss_fn,
        d_reg_fn=SpectralNormRegularizer(max_norm=0.5),
    )

    assert (epoch_dlog["d_pre_spectral_norm_mean"] > 0.0).all()
    changed = [
        not torch.equal(before, after)
        for before, after in zip(initial_params, d_net.parameters(), strict=True)
    ]
    assert any(changed)
