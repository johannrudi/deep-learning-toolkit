"""Unit tests for the finite-difference Lipschitz penalties in `dlk.loss.wasserstein_gan`."""

import inspect
import math
from collections.abc import Callable

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from dlk.loss.wasserstein_gan import (
    gradient_penalty_lip_fd_random,
    gradient_penalty_lip_fd_segment,
)

BATCH_SIZE = 8
X_SIZE = 4
Y_SIZE = 3

PenaltyFn = Callable[..., torch.Tensor]
CompileFn = Callable[[torch.nn.Module], CompileCounterWithBackend]

# Geometry tests: exact quotient identities against `|w^T u|`; segment now,
# endpoint later.
SEGMENT_PENALTY_FNS: tuple[PenaltyFn, ...] = (gradient_penalty_lip_fd_segment,)

# General penalty tests: identical pairs, broadcasting, conditional and
# unconditional critic, compiled critic; segment and random now, endpoint and
# adversarial later.
FD_PENALTY_FNS: tuple[PenaltyFn, ...] = (
    gradient_penalty_lip_fd_segment,
    gradient_penalty_lip_fd_random,
)


def _two_sided_kwargs(penalty_fn: PenaltyFn) -> dict[str, bool]:
    """Request the two-sided penalty from variants that support it."""
    if "one_sided" in inspect.signature(penalty_fn).parameters:
        return {"one_sided": False}
    return {}


class _LinearCritic(torch.nn.Module):
    """Linear critic `D(x, y) = x.flatten(1) @ w`, ignoring `y`."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(input_size, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Score `x` with `w`, ignoring `y`."""
        del y
        return x.flatten(1) @ self.w


class _MLPCritic(torch.nn.Module):
    """Small nonlinear critic scoring `(x, y)` pairs, `y` optional."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(X_SIZE + Y_SIZE, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Score samples, padding with zeros when `y` is absent."""
        if y is None:
            y = torch.zeros(x.size(0), Y_SIZE)
        return self.net(torch.cat([x.flatten(1), y], dim=1))


class _SpyCritic(torch.nn.Module):
    """Critic wrapper that records every input tensor it is called with."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Record `x`, then delegate to the wrapped critic."""
        self.inputs.append(x)
        return self.inner(x, y)


def _linear_critic_with_norm(input_size: int, w_norm: float) -> _LinearCritic:
    """Build a linear critic whose weight has norm `w_norm`, in a random direction."""
    d_net = _LinearCritic(input_size)
    d_net.w.data.copy_(d_net.w.data / d_net.w.data.norm() * w_norm)
    return d_net


def _aligned_linear_critic(
    w_norm: float,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a linear critic and pairs whose difference is aligned with `w`.

    `x_data - x_gen` is parallel to `w` for every sample, so the segment
    quotient equals `w_norm` exactly, independent of the random `t_a`, `t_b`.

    Args:
        w_norm: Norm to give the critic's weight `w`.

    Returns:
        The critic, its weight `w`, `x_gen`, and `x_data`.
    """
    d_net = _linear_critic_with_norm(X_SIZE, w_norm)
    w = d_net.w.detach().clone()
    w_unit = (w / w_norm).squeeze(1)

    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    alpha = torch.rand(BATCH_SIZE, 1).clamp(min=0.1)  # keep pairs distinguishable
    x_data = x_gen + alpha * w_unit
    return d_net, w, x_gen, x_data


def _assert_on_segment(
    x: torch.Tensor, x_gen: torch.Tensor, x_data: torch.Tensor
) -> None:
    """Assert `x` sits on the `x_gen`-`x_data` segment, at a per-sample `t` in `[0, 1]` that varies."""
    t = (x - x_gen) / (x_data - x_gen)
    torch.testing.assert_close(t, t[:, :1].expand_as(t))
    assert (t >= 0).all() and (t <= 1).all()
    assert (t[:, 0].max() - t[:, 0].min()) > 1e-3  # differs across samples


@pytest.mark.parametrize("penalty_fn", SEGMENT_PENALTY_FNS)
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_dlog_matches_directional_derivative(
    penalty_fn: PenaltyFn, non_contiguous: bool
) -> None:
    """`dlog["lip_quotient"]` and its max equal the exact directional slope."""
    torch.manual_seed(0)
    if non_contiguous:
        channels, length = 2, 3
        d_net = _LinearCritic(channels * length)
        x_gen = torch.randn(BATCH_SIZE, length, channels).transpose(1, 2)
        x_data = torch.randn(BATCH_SIZE, length, channels).transpose(1, 2)
        assert not x_gen.is_contiguous()
    else:
        d_net = _LinearCritic(X_SIZE)
        x_gen = torch.randn(BATCH_SIZE, X_SIZE)
        x_data = torch.randn(BATCH_SIZE, X_SIZE)

    dlog: dict[str, float] = {}
    penalty_fn(d_net, x_gen, x_data, None, dlog=dlog)

    diff = (x_data - x_gen).flatten(1)
    u = diff / diff.norm(dim=1, keepdim=True)
    expected_q = (u @ d_net.w).squeeze(1).abs()

    torch.testing.assert_close(
        torch.tensor(dlog["lip_quotient"]), expected_q.mean(), rtol=1e-5, atol=1e-8
    )
    torch.testing.assert_close(
        torch.tensor(dlog["lip_quotient_max"]), expected_q.max(), rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("penalty_fn", SEGMENT_PENALTY_FNS)
def test_penalty_zero_below_target(penalty_fn: PenaltyFn) -> None:
    """The one-sided penalty is exactly 0 when `‖w‖ <= lip`."""
    torch.manual_seed(0)
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)

    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=(w.norm() + 1.0).item())
    torch.testing.assert_close(penalty, torch.zeros_like(penalty))


@pytest.mark.parametrize("penalty_fn", SEGMENT_PENALTY_FNS)
def test_penalty_positive_above_target(penalty_fn: PenaltyFn) -> None:
    """The one-sided penalty equals `(‖w‖ - lip)^2` when `‖w‖ > lip`."""
    torch.manual_seed(0)
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)

    lip = (w.norm() - 0.5).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip)
    expected = (w.norm() - lip) ** 2
    torch.testing.assert_close(penalty, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("penalty_fn", SEGMENT_PENALTY_FNS)
def test_penalty_two_sided_below_target(penalty_fn: PenaltyFn) -> None:
    """`one_sided=False` also penalizes `‖w‖ < lip`, as `(lip - ‖w‖)^2`."""
    torch.manual_seed(0)
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)

    lip = (w.norm() + 0.5).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip, one_sided=False)
    expected = (lip - w.norm()) ** 2
    torch.testing.assert_close(penalty, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("penalty_fn", SEGMENT_PENALTY_FNS)
def test_gradient_matches_analytic_value(penalty_fn: PenaltyFn) -> None:
    """The parameter gradient equals `2 (q - lip) w / ‖w‖` at the aligned pairs."""
    torch.manual_seed(0)

    # one-sided, ‖w‖ > lip: the relu is active
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)
    lip = (w.norm() - 0.5).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip)
    penalty.backward()
    expected_grad = 2 * (w.norm() - lip) * w / w.norm()
    assert d_net.w.grad is not None
    torch.testing.assert_close(d_net.w.grad, expected_grad, rtol=1e-4, atol=1e-6)

    # two-sided, ‖w‖ < lip: the same formula, with a negative factor
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)
    lip = (w.norm() + 0.5).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip, one_sided=False)
    penalty.backward()
    expected_grad = 2 * (w.norm() - lip) * w / w.norm()
    assert d_net.w.grad is not None
    torch.testing.assert_close(d_net.w.grad, expected_grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("penalty_fn", SEGMENT_PENALTY_FNS)
def test_pairs_lie_on_segment_with_matching_dtype(penalty_fn: PenaltyFn) -> None:
    """Both points lie on the segment, at a bounded, per-sample `t`, in `x_data`'s dtype."""
    torch.manual_seed(0)
    d_net = _SpyCritic(_LinearCritic(X_SIZE)).double()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE, dtype=torch.float64)
    x_data = torch.randn(BATCH_SIZE, X_SIZE, dtype=torch.float64)

    penalty_fn(d_net, x_gen, x_data, None)

    assert len(d_net.inputs) == 2
    for x_pair in d_net.inputs:
        assert x_pair.dtype == torch.float64
        _assert_on_segment(x_pair, x_gen, x_data)


@pytest.mark.parametrize("penalty_fn", FD_PENALTY_FNS)
def test_identical_pairs_finite_gradients(penalty_fn: PenaltyFn) -> None:
    """Identical `x_gen`/`x_data` give a finite penalty and a nonzero, finite gradient."""
    torch.manual_seed(0)
    d_net = _MLPCritic()
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    x_gen = x_data.clone()
    y_data = torch.randn(BATCH_SIZE, Y_SIZE)

    # Segment: two-sided, nonzero lip, forces a nonzero gradient through the
    # (exactly zero) zero-difference norm. Random: lip=0.0 already gives a
    # generically nonzero `q`, since its pair is always `radius` apart.
    two_sided_kwargs = _two_sided_kwargs(penalty_fn)
    lip = 1.0 if two_sided_kwargs else 0.0
    penalty = penalty_fn(d_net, x_gen, x_data, y_data, lip=lip, **two_sided_kwargs)
    assert torch.isfinite(penalty)

    penalty.backward()
    grad_nonzero = False
    for p in d_net.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        grad_nonzero = grad_nonzero or bool((p.grad != 0).any())
    assert grad_nonzero


@pytest.mark.parametrize("penalty_fn", FD_PENALTY_FNS)
def test_broadcasts_single_x_gen_sample(penalty_fn: PenaltyFn) -> None:
    """A single broadcast `x_gen` sample matches an explicitly expanded batch."""
    torch.manual_seed(0)
    d_net = _LinearCritic(X_SIZE)
    x_gen_single = torch.randn(X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)

    torch.manual_seed(1)
    penalty_single = penalty_fn(d_net, x_gen_single, x_data, None)

    torch.manual_seed(1)
    penalty_batch = penalty_fn(
        d_net, x_gen_single.expand(BATCH_SIZE, X_SIZE), x_data, None
    )

    assert torch.isfinite(penalty_single)
    torch.testing.assert_close(penalty_single, penalty_batch)


@pytest.mark.parametrize("penalty_fn", FD_PENALTY_FNS)
@pytest.mark.parametrize("conditional", [True, False])
def test_runs_conditional_and_unconditional(
    penalty_fn: PenaltyFn, conditional: bool
) -> None:
    """The penalty runs with a conditional critic and `y_data`, and without."""
    torch.manual_seed(0)
    d_net = _MLPCritic()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    y_data = torch.randn(BATCH_SIZE, Y_SIZE) if conditional else None

    penalty = penalty_fn(d_net, x_gen, x_data, y_data)
    assert torch.isfinite(penalty)


@pytest.mark.parametrize("penalty_fn", FD_PENALTY_FNS)
def test_compiled_critic_trains_without_recompile(
    penalty_fn: PenaltyFn, compile_aot_eager: CompileFn
) -> None:
    """A compiled critic trains for a few steps with no recompile after the first."""
    torch.manual_seed(0)
    d_net = _MLPCritic()
    counter = compile_aot_eager(d_net)
    opt = torch.optim.SGD(d_net.parameters(), lr=0.1)

    params_before = [p.detach().clone() for p in d_net.parameters()]
    for _ in range(3):
        x_gen = torch.randn(BATCH_SIZE, X_SIZE)
        x_data = torch.randn(BATCH_SIZE, X_SIZE)
        y_data = torch.randn(BATCH_SIZE, Y_SIZE)

        d_net(x_data, y_data)  # Loss-term forward pass, as the training loop does.

        opt.zero_grad()
        # `lip=0.0` keeps the penalty active regardless of scale; two-sided
        # where supported, one-sided otherwise (the random quotient is >= 0).
        penalty = penalty_fn(
            d_net, x_gen, x_data, y_data, lip=0.0, **_two_sided_kwargs(penalty_fn)
        )
        penalty.backward()
        opt.step()

    assert counter.frame_count == 1
    # The final layer's bias cancels out of every difference quotient, so it
    # gets no gradient and never changes; the other parameters do.
    changed = [
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(params_before, d_net.parameters(), strict=True)
    ]
    assert any(changed)


# Random-variant-specific tests, beyond the generic FD_PENALTY_FNS ones above.


def test_random_quotient_matches_sphere_moment() -> None:
    """`dlog["lip_quotient"]` matches the exact mean `‖w‖ E|u_1|` of a linear critic."""
    torch.manual_seed(0)
    d = 16
    batch_size = 4096
    w_norm = 2.0
    radius = 0.1
    # align `w` with the first coordinate axis, so `q = |w^T u| = w_norm |u_1|`
    # compares directly against the sphere moment `E|u_1|` below.
    inner = _LinearCritic(d)
    inner.w.data.zero_()
    inner.w.data[0, 0] = w_norm
    d_net = _SpyCritic(inner)
    x_gen = torch.randn(batch_size, d)
    x_data = torch.randn(batch_size, d)

    dlog: dict[str, float] = {}
    gradient_penalty_lip_fd_random(d_net, x_gen, x_data, None, radius=radius, dlog=dlog)

    x_hat, x_pert = d_net.inputs
    u = (x_pert - x_hat) / radius
    # each coordinate of `u` has mean 0, variance 1/d; bound with 4 standard errors
    assert u.mean(0).abs().max() < 4 / math.sqrt(d * batch_size)

    # `u_1` is one coordinate of a point uniform on the unit sphere in R^d;
    # its exact first absolute moment is Gamma(d/2) / (sqrt(pi) Gamma((d+1)/2)).
    log_e_abs_u1 = (
        math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d + 1) / 2)
    )
    expected_mean_q = w_norm * math.exp(log_e_abs_u1)

    # The crude bound `Var(|u_1|) <= E[u_1^2] = 1/d` gives a relative standard
    # error of about 1.9% for this batch size; the exact `Var(|u_1|)` gives
    # about 1.1%. `rel=0.03` covers the looser bound with room to spare.
    assert dlog["lip_quotient"] == pytest.approx(expected_mean_q, rel=0.03)
    # `q = |w^T u| <= ‖w‖ ‖u‖ = ‖w‖` exactly, up to floating-point error.
    assert dlog["lip_quotient_max"] <= w_norm * (1 + 1e-4)


def test_random_penalty_zero_below_target() -> None:
    """The one-sided penalty is exactly 0 when `‖w‖ <= lip` for a linear critic."""
    torch.manual_seed(0)
    d_net = _LinearCritic(X_SIZE)
    w_norm = d_net.w.norm().item()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)

    # `q = |w^T u| <= ‖w‖ <= lip` for every sample, so the relu never fires.
    penalty = gradient_penalty_lip_fd_random(
        d_net, x_gen, x_data, None, lip=w_norm + 1.0
    )
    torch.testing.assert_close(penalty, torch.zeros_like(penalty))


def test_random_penalty_positive_above_target() -> None:
    """The one-sided penalty is positive when `‖w‖` is well above `lip`."""
    torch.manual_seed(0)
    d = 16
    batch_size = 256
    lip = 1.0
    w_norm = 4.0 * lip * math.sqrt(d)
    d_net = _linear_critic_with_norm(d, w_norm)
    x_gen = torch.randn(batch_size, d)
    x_data = torch.randn(batch_size, d)

    penalty = gradient_penalty_lip_fd_random(d_net, x_gen, x_data, None, lip=lip)
    assert penalty.item() > 0.0


def test_random_pair_geometry() -> None:
    """`x_hat` lies on the segment; `x_pert` is exactly `radius` away, in varying directions."""
    torch.manual_seed(0)
    d_net = _SpyCritic(_LinearCritic(X_SIZE))
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    radius = 0.1

    gradient_penalty_lip_fd_random(d_net, x_gen, x_data, None, radius=radius)

    assert len(d_net.inputs) == 2
    x_hat, x_pert = d_net.inputs
    _assert_on_segment(x_hat, x_gen, x_data)

    # `x_pert` sits exactly `radius` away from `x_hat`, flattened norm
    diff = x_pert - x_hat
    diff_norm = torch.linalg.vector_norm(diff.flatten(1), dim=1)
    torch.testing.assert_close(diff_norm, torch.full_like(diff_norm, radius))

    # confirm directions differ: same norm (just asserted), different vectors
    assert (diff[0] - diff[1]).abs().max() > 1e-3


def test_random_gradient_matches_analytic_value() -> None:
    """`w.grad` equals `mean 2 relu(|w^T u| - lip) sign(w^T u) u`, from a linear critic."""
    torch.manual_seed(0)
    batch_size = 64
    d = X_SIZE
    lip = 1.0
    w_norm = 2.0 * lip * math.sqrt(d)  # some, not all, samples end up active
    radius = 0.1
    inner = _linear_critic_with_norm(d, w_norm)
    d_net = _SpyCritic(inner)
    x_gen = torch.randn(batch_size, d)
    x_data = torch.randn(batch_size, d)

    penalty = gradient_penalty_lip_fd_random(
        d_net, x_gen, x_data, None, lip=lip, radius=radius
    )
    penalty.backward()

    x_hat, x_pert = d_net.inputs
    u = (x_pert - x_hat) / radius  # reconstruct the (already unit-norm) direction
    w = inner.w.detach().squeeze(1)
    w_dot_u = u @ w
    active = torch.relu(w_dot_u.abs() - lip)
    # require some but not all samples active, or the test would not exercise
    # both branches of the relu
    assert 0 < (active > 0).sum().item() < batch_size

    expected_grad = (2 * active * torch.sign(w_dot_u)).unsqueeze(1) * u
    expected_grad = expected_grad.mean(dim=0, keepdim=True).t()

    assert inner.w.grad is not None
    torch.testing.assert_close(inner.w.grad, expected_grad, rtol=1e-4, atol=1e-6)
