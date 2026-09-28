"""Unit tests for the finite-difference gradient penalties in `dlk.loss.gradient_penalty`."""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

from dlk.loss.gradient_penalty import (
    gradient_penalty_fd_adversarial,
    gradient_penalty_fd_endpoint,
    gradient_penalty_fd_random,
    gradient_penalty_fd_segment,
)

BATCH_SIZE = 8
X_SIZE = 4
Y_SIZE = 3
EPS = 1e-6  # the penalties' default `eps`

# one-sided penalty `relu(q^2 - lip^2)`, exactly zero below the target
ONE_SIDED_RELU_KWARGS: dict[str, Any] = {"eps": 0.0, "one_sided_nonlinearity": F.relu}

PenaltyFn = Callable[..., torch.Tensor]
CompileFn = Callable[[torch.nn.Module], CompileCounterWithBackend]


def _endpoint_from_critic(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor:
    """Adapt the endpoint penalty to score its own endpoints, like the other variants.

    Scores `x_gen.expand_as(x_data)` and `x_data` with `d_net`, `_critic`-style
    (`d_net(x, y)` or `d_net(x)`), then calls `gradient_penalty_fd_endpoint`
    on the resulting outputs, as the loop's keywords would.
    """
    x_gen = x_gen.expand_as(x_data)
    d_outputs_gen = d_net(x_gen, y_data) if y_data is not None else d_net(x_gen)
    d_outputs_data = d_net(x_data, y_data) if y_data is not None else d_net(x_data)
    return gradient_penalty_fd_endpoint(
        d_net,
        x_gen,
        x_data,
        y_data,
        d_outputs_gen=d_outputs_gen,
        d_outputs_data=d_outputs_data,
        **kwargs,
    )


# Geometry tests: exact quotient identities against `|w^T u|`; holds for
# segment (any pair, by the mean value theorem argument in B.1) and endpoint
# (the pair is exactly `x_data`, `x_gen`).
SEGMENT_PENALTY_FNS: tuple[PenaltyFn, ...] = (
    gradient_penalty_fd_segment,
    _endpoint_from_critic,
)

# Zero/positive-above-target tests: variants whose quotient is exact for any
# linear critic, independent of the pair drawn (segment: `|w^T u|` from an
# aligned pair; adversarial: `‖w‖` after one power step; endpoint: `|w^T u|`,
# same as segment, since the pair is fixed at the endpoints).
LINEAR_EXACT_PENALTY_FNS: tuple[PenaltyFn, ...] = (
    gradient_penalty_fd_segment,
    gradient_penalty_fd_adversarial,
    _endpoint_from_critic,
)

# General penalty tests: broadcasting, conditional and unconditional critic,
# compiled critic; segment, random, adversarial, and endpoint.
FD_PENALTY_FNS: tuple[PenaltyFn, ...] = (
    gradient_penalty_fd_segment,
    gradient_penalty_fd_random,
    gradient_penalty_fd_adversarial,
    _endpoint_from_critic,
)


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


class _FlatCritic(torch.nn.Module):
    """Critic whose output is constant in `x`, to exercise the flat-direction fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Return `bias`, ignoring `y`.

        `x.sum(1, keepdim=True) * 0` is load-bearing: it keeps `x` in the
        autograd graph with an exactly zero gradient. Without it, the
        output would not depend on `x` at all, and `torch.autograd.grad`
        would find no path to differentiate, rather than a valid zero
        gradient.
        """
        del y
        return x.sum(1, keepdim=True) * 0 + self.bias


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
    """`dlog["grad_fd_norm"]` equals the exact directional slope."""
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
        torch.tensor(dlog["grad_fd_norm"]), expected_q.mean(), rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("penalty_fn", LINEAR_EXACT_PENALTY_FNS)
def test_penalty_zero_below_target(penalty_fn: PenaltyFn) -> None:
    """The one-sided `relu` penalty is exactly 0 when `‖w‖ <= lip`."""
    torch.manual_seed(0)
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)

    penalty = penalty_fn(
        d_net, x_gen, x_data, None, lip=(w.norm() + 1.0).item(), **ONE_SIDED_RELU_KWARGS
    )
    torch.testing.assert_close(penalty, torch.zeros_like(penalty))


@pytest.mark.parametrize("penalty_fn", LINEAR_EXACT_PENALTY_FNS)
def test_penalty_positive_above_target(penalty_fn: PenaltyFn) -> None:
    """The one-sided `relu` penalty equals `‖w‖^2 - lip^2` when `‖w‖ > lip`."""
    torch.manual_seed(0)
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)

    lip = (w.norm() - 0.5).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip, **ONE_SIDED_RELU_KWARGS)
    expected = w.norm() ** 2 - lip**2
    torch.testing.assert_close(penalty, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("penalty_fn", LINEAR_EXACT_PENALTY_FNS)
def test_penalty_default_one_sided_is_small_below_target(
    penalty_fn: PenaltyFn,
) -> None:
    """The default one-sided penalty, a sharp softplus, is tiny well below the target."""
    torch.manual_seed(0)
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)

    lip = (w.norm() + 1.0).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip)
    expected = F.softplus(w.norm() ** 2 + EPS - lip**2, beta=10.0)
    assert 0.0 < penalty.item() < 1e-10
    torch.testing.assert_close(penalty, expected, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize("penalty_fn", SEGMENT_PENALTY_FNS)
def test_penalty_two_sided_below_target(penalty_fn: PenaltyFn) -> None:
    """`one_sided=False` also penalizes `‖w‖ < lip`, as `(sqrt(‖w‖^2 + eps) - lip)^2`."""
    torch.manual_seed(0)
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)

    lip = (w.norm() + 0.5).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip, one_sided=False)
    expected = (torch.sqrt(w.norm() ** 2 + EPS) - lip) ** 2
    torch.testing.assert_close(penalty, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("penalty_fn", SEGMENT_PENALTY_FNS)
def test_gradient_matches_analytic_value(penalty_fn: PenaltyFn) -> None:
    """The parameter gradient matches the analytic value at the aligned pairs.

    With `q = ‖w‖` at every aligned pair, `grad_w q^2 = 2 w`, so the one-sided
    `relu` penalty has gradient `2 w`, and the two-sided penalty, with
    `s = sqrt(‖w‖^2 + eps)`, has gradient `2 (s - lip) w / s`.
    """
    torch.manual_seed(0)

    # one-sided, ‖w‖ > lip: the relu is active
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)
    lip = (w.norm() - 0.5).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip, **ONE_SIDED_RELU_KWARGS)
    penalty.backward()
    expected_grad = 2 * w
    assert d_net.w.grad is not None
    torch.testing.assert_close(d_net.w.grad, expected_grad, rtol=1e-4, atol=1e-6)

    # two-sided, ‖w‖ < lip: a negative factor
    d_net, w, x_gen, x_data = _aligned_linear_critic(w_norm=1.0)
    lip = (w.norm() + 0.5).item()
    penalty = penalty_fn(d_net, x_gen, x_data, None, lip=lip, one_sided=False)
    penalty.backward()
    s = torch.sqrt(w.norm() ** 2 + EPS)
    expected_grad = 2 * (s - lip) * w / s
    assert d_net.w.grad is not None
    torch.testing.assert_close(d_net.w.grad, expected_grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("penalty_fn", [gradient_penalty_fd_segment])
def test_pairs_lie_on_segment_with_matching_dtype(penalty_fn: PenaltyFn) -> None:
    """Both points lie on the segment, at a bounded, per-sample `t`, in `x_data`'s dtype.

    The endpoint variant is not parametrized here: its pair is fixed at
    `t in {0, 1}`, which `_assert_on_segment` rejects (it requires `t` to
    vary across samples); see `test_endpoint_quotient_matches_formula` for
    its geometry check instead.
    """
    torch.manual_seed(0)
    d_net = _SpyCritic(_LinearCritic(X_SIZE)).double()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE, dtype=torch.float64)
    x_data = torch.randn(BATCH_SIZE, X_SIZE, dtype=torch.float64)

    penalty_fn(d_net, x_gen, x_data, None)

    assert len(d_net.inputs) == 2
    for x_pair in d_net.inputs:
        assert x_pair.dtype == torch.float64
        _assert_on_segment(x_pair, x_gen, x_data)


def test_endpoint_quotient_matches_formula() -> None:
    """`dlog["grad_fd_norm"]` matches the B.2 formula, including the `min_dist` clamp.

    Calls `gradient_penalty_fd_endpoint` directly on hand-made critic outputs,
    so the pair is exactly `(x_data, x_gen)`, unlike the interpolated segment
    pair. The critic passed as `d_net` is unused.
    """
    torch.manual_seed(0)
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    x_gen[0] = x_data[0]  # one identical pair, to exercise the `min_dist` clamp
    d_outputs_gen = torch.randn(BATCH_SIZE, 1)
    d_outputs_data = torch.randn(BATCH_SIZE, 1)
    min_dist = 1e-6

    dlog: dict[str, float] = {}
    gradient_penalty_fd_endpoint(
        _LinearCritic(X_SIZE),
        x_gen,
        x_data,
        None,
        min_dist=min_dist,
        d_outputs_gen=d_outputs_gen,
        d_outputs_data=d_outputs_data,
        dlog=dlog,
    )

    x_diff = torch.linalg.vector_norm((x_data - x_gen).flatten(1), dim=1)
    d_diff = torch.linalg.vector_norm(
        (d_outputs_data - d_outputs_gen).flatten(1), dim=1
    )
    expected_q = d_diff / x_diff.clamp(min=min_dist)

    torch.testing.assert_close(
        torch.tensor(dlog["grad_fd_norm"]), expected_q.mean(), rtol=1e-5, atol=1e-8
    )


def test_endpoint_requires_critic_outputs() -> None:
    """The endpoint variant raises without `d_outputs_gen` and `d_outputs_data`."""
    d_net = _LinearCritic(X_SIZE)
    x = torch.randn(BATCH_SIZE, X_SIZE)
    with pytest.raises(ValueError, match="requires `d_outputs_gen`"):
        gradient_penalty_fd_endpoint(d_net, x, x, None)


@pytest.mark.parametrize(
    ("penalty_kwargs", "match"),
    [
        ({"eps": -1e-6}, "eps must be >= 0"),
        ({"eps": 0.0, "one_sided": False}, "eps must be > 0"),
        ({"min_dist": 0.0}, "min_dist must be > 0"),
    ],
)
@pytest.mark.parametrize("penalty_fn", FD_PENALTY_FNS)
def test_invalid_eps_or_min_dist_raises(
    penalty_fn: PenaltyFn, penalty_kwargs: dict[str, Any], match: str
) -> None:
    """`eps < 0`, `eps == 0` for the two-sided penalty, and `min_dist <= 0` raise."""
    torch.manual_seed(0)
    d_net = _LinearCritic(X_SIZE)
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    with pytest.raises(ValueError, match=match):
        penalty_fn(d_net, x_gen, x_data, None, **penalty_kwargs)


# Excludes the endpoint variant: with `x_gen = x_data.clone()`, its pair is
# bit-identical (unlike the other variants' interpolation/perturbation
# arithmetic), so its gradient is exactly, not just numerically, zero; see
# `test_endpoint_identical_pairs_give_exactly_zero_gradient`.
@pytest.mark.parametrize(
    "penalty_fn",
    [
        gradient_penalty_fd_segment,
        gradient_penalty_fd_random,
        gradient_penalty_fd_adversarial,
    ],
)
def test_identical_pairs_finite_gradients(penalty_fn: PenaltyFn) -> None:
    """Identical `x_gen`/`x_data` give a finite penalty and a nonzero, finite gradient."""
    torch.manual_seed(0)
    d_net = _MLPCritic()
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    x_gen = x_data.clone()
    y_data = torch.randn(BATCH_SIZE, Y_SIZE)

    # two-sided with a nonzero lip, so the gradient passes through the square
    # root at a (near) zero quotient, which `eps` keeps finite
    penalty = penalty_fn(d_net, x_gen, x_data, y_data, lip=1.0, one_sided=False)
    assert torch.isfinite(penalty)

    penalty.backward()
    grad_nonzero = False
    for p in d_net.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        grad_nonzero = grad_nonzero or bool((p.grad != 0).any())
    assert grad_nonzero


def test_endpoint_identical_pairs_give_exactly_zero_gradient() -> None:
    """Identical `x_gen`/`x_data` give a penalty of exactly `eps` and a zero gradient (7(b): finite gradients).

    With `q = 0` and `lip=0`, the two-sided penalty is `(sqrt(eps) - 0)^2 = eps`.
    """
    torch.manual_seed(0)
    d_net = _MLPCritic()
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    x_gen = x_data.clone()
    y_data = torch.randn(BATCH_SIZE, Y_SIZE)

    penalty = _endpoint_from_critic(
        d_net, x_gen, x_data, y_data, lip=0.0, one_sided=False
    )
    torch.testing.assert_close(penalty, torch.full_like(penalty, EPS))

    penalty.backward()
    for p in d_net.parameters():
        assert p.grad is not None
        assert (p.grad == 0).all()


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
    """A compiled critic trains for a few steps with no recompile after the first step.

    The adversarial variant's direction search adds a second graph, for its
    grad-requiring input; the pinned frame count below accounts for that.
    """
    torch.manual_seed(0)
    d_net = _MLPCritic()
    counter = compile_aot_eager(d_net)
    opt = torch.optim.SGD(d_net.parameters(), lr=0.1)
    expected_frame_count = 2 if penalty_fn is gradient_penalty_fd_adversarial else 1

    params_before = [p.detach().clone() for p in d_net.parameters()]
    for _ in range(3):
        x_gen = torch.randn(BATCH_SIZE, X_SIZE)
        x_data = torch.randn(BATCH_SIZE, X_SIZE)
        y_data = torch.randn(BATCH_SIZE, Y_SIZE)

        d_net(x_data, y_data)  # Loss-term forward pass, as the training loop does.

        opt.zero_grad()
        # `lip=0.0` with the two-sided penalty keeps it active regardless of scale
        penalty = penalty_fn(d_net, x_gen, x_data, y_data, lip=0.0, one_sided=False)
        penalty.backward()
        opt.step()

        assert counter.frame_count == expected_frame_count
    # The final layer's bias cancels out of every difference quotient, so it
    # gets no gradient and never changes; the other parameters do.
    changed = [
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(params_before, d_net.parameters(), strict=True)
    ]
    assert any(changed)


# Random-variant-specific tests, beyond the generic FD_PENALTY_FNS ones above.


def test_random_quotient_matches_sphere_moment() -> None:
    """`dlog["grad_fd_norm"]` matches the exact mean `‖w‖ E|u_1|` of a linear critic."""
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
    gradient_penalty_fd_random(d_net, x_gen, x_data, None, radius=radius, dlog=dlog)

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
    assert dlog["grad_fd_norm"] == pytest.approx(expected_mean_q, rel=0.03)


def test_random_penalty_zero_below_target() -> None:
    """The one-sided `relu` penalty is exactly 0 when `‖w‖ <= lip` for a linear critic."""
    torch.manual_seed(0)
    d_net = _LinearCritic(X_SIZE)
    w_norm = d_net.w.norm().item()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)

    # `q = |w^T u| <= ‖w‖ <= lip` for every sample, so the relu never fires.
    penalty = gradient_penalty_fd_random(
        d_net, x_gen, x_data, None, lip=w_norm + 1.0, **ONE_SIDED_RELU_KWARGS
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

    penalty = gradient_penalty_fd_random(
        d_net, x_gen, x_data, None, lip=lip, **ONE_SIDED_RELU_KWARGS
    )
    assert penalty.item() > 0.0


def test_random_pair_geometry() -> None:
    """`x_hat` lies on the segment; `x_pert` is exactly `radius` away, in varying directions."""
    torch.manual_seed(0)
    d_net = _SpyCritic(_LinearCritic(X_SIZE))
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    radius = 0.1

    gradient_penalty_fd_random(d_net, x_gen, x_data, None, radius=radius)

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
    """`w.grad` equals `mean 1[|w^T u| > lip] 2 (w^T u) u`, from a linear critic."""
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

    penalty = gradient_penalty_fd_random(
        d_net, x_gen, x_data, None, lip=lip, radius=radius, **ONE_SIDED_RELU_KWARGS
    )
    penalty.backward()

    x_hat, x_pert = d_net.inputs
    u = (x_pert - x_hat) / radius  # reconstruct the (already unit-norm) direction
    w = inner.w.detach().squeeze(1)
    w_dot_u = u @ w
    # `q^2 = (w^T u)^2`, so the relu is active where `|w^T u| > lip`
    active = (w_dot_u.abs() > lip).to(w_dot_u.dtype)
    # require some but not all samples active, or the test would not exercise
    # both branches of the relu
    assert 0 < active.sum().item() < batch_size

    expected_grad = (2 * active * w_dot_u).unsqueeze(1) * u
    expected_grad = expected_grad.mean(dim=0, keepdim=True).t()

    assert inner.w.grad is not None
    torch.testing.assert_close(inner.w.grad, expected_grad, rtol=1e-4, atol=1e-6)


# Adversarial-variant-specific tests, beyond the generic FD_PENALTY_FNS ones above.


def test_adversarial_quotient_equals_weight_norm_for_linear_critic() -> None:
    """`dlog["grad_fd_norm"]` equals `‖w‖` exactly, after one power step."""
    torch.manual_seed(0)
    d_net = _LinearCritic(X_SIZE)
    w_norm = d_net.w.norm().item()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)

    dlog: dict[str, float] = {}
    gradient_penalty_fd_adversarial(d_net, x_gen, x_data, None, dlog=dlog)

    assert dlog["grad_fd_norm"] == pytest.approx(w_norm, rel=1e-5, abs=1e-6)


def test_adversarial_pair_geometry() -> None:
    """The three recorded critic calls are `x_hat`, `x_hat + r0`, and `x_adv`, `xi`/`radius` apart."""
    torch.manual_seed(0)
    d_net = _SpyCritic(_MLPCritic())
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    y_data = torch.randn(BATCH_SIZE, Y_SIZE)
    xi, radius = 0.03, 0.2

    gradient_penalty_fd_adversarial(d_net, x_gen, x_data, y_data, xi=xi, radius=radius)

    assert len(d_net.inputs) == 3
    x_hat, x_r0, x_adv = d_net.inputs
    r0_norm = torch.linalg.vector_norm((x_r0 - x_hat).flatten(1), dim=1)
    torch.testing.assert_close(r0_norm, torch.full_like(r0_norm, xi))
    r_adv_norm = torch.linalg.vector_norm((x_adv - x_hat).flatten(1), dim=1)
    torch.testing.assert_close(r_adv_norm, torch.full_like(r_adv_norm, radius))


def test_adversarial_quotient_approximates_gradient_norm_and_beats_random() -> None:
    """For small `xi`/`radius`, `q` approximates `‖grad_x D(x_hat)‖` and exceeds the random quotient."""
    torch.manual_seed(0)
    small = 1e-4
    inner = _MLPCritic().double()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE, dtype=torch.float64)
    x_data = torch.randn(BATCH_SIZE, X_SIZE, dtype=torch.float64)
    y_data = torch.randn(BATCH_SIZE, Y_SIZE, dtype=torch.float64)

    seed = 42
    spy = _SpyCritic(inner)
    torch.manual_seed(seed)
    dlog_adv: dict[str, float] = {}
    gradient_penalty_fd_adversarial(
        spy, x_gen, x_data, y_data, xi=small, radius=small, dlog=dlog_adv
    )

    torch.manual_seed(seed)
    dlog_random: dict[str, float] = {}
    gradient_penalty_fd_random(
        inner, x_gen, x_data, y_data, radius=small, dlog=dlog_random
    )

    # recompute the reference gradient norm at `x_hat` (the spy's first call)
    # independently, with an ordinary `torch.autograd.grad` call
    x_hat = spy.inputs[0].detach().requires_grad_()
    d_ref = inner(x_hat, y_data)
    (grad_ref,) = torch.autograd.grad(d_ref.sum(), x_hat)
    grad_norm_ref = torch.linalg.vector_norm(grad_ref.flatten(1), dim=1)

    torch.testing.assert_close(
        torch.tensor(dlog_adv["grad_fd_norm"], dtype=torch.float64),
        grad_norm_ref.mean(),
        rtol=2e-2,
        atol=1e-6,
    )
    # the random direction only sees a `1/sqrt(d)`-ish fraction of the slope
    assert dlog_adv["grad_fd_norm"] >= dlog_random["grad_fd_norm"]


def test_adversarial_gradient_matches_analytic_value() -> None:
    """`w.grad` equals `2 w` for the active `relu`; fails if `d_hat`/`d_adv` were detached."""
    torch.manual_seed(0)
    w_norm = 2.0
    d_net = _linear_critic_with_norm(X_SIZE, w_norm)
    w = d_net.w.detach().clone()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)

    lip = w_norm - 0.5
    penalty = gradient_penalty_fd_adversarial(
        d_net, x_gen, x_data, None, lip=lip, **ONE_SIDED_RELU_KWARGS
    )
    penalty.backward()

    # `q^2 = ‖w‖^2`, whose gradient in `w` is `2 w`
    expected_grad = 2 * w
    assert d_net.w.grad is not None
    torch.testing.assert_close(d_net.w.grad, expected_grad, rtol=1e-4, atol=1e-6)


def test_adversarial_search_does_not_touch_parameter_gradients() -> None:
    """The direction search leaves every critic parameter's `.grad` at `None`."""
    torch.manual_seed(0)
    d_net = _MLPCritic()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    y_data = torch.randn(BATCH_SIZE, Y_SIZE)

    gradient_penalty_fd_adversarial(d_net, x_gen, x_data, y_data)

    for p in d_net.parameters():
        assert p.grad is None


def test_adversarial_direction_search_uses_unwrap_net(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search runs on `unwrap_net(d_net)`'s return value, called once with `d_net`."""
    torch.manual_seed(0)
    d_net = _MLPCritic()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    y_data = torch.randn(BATCH_SIZE, Y_SIZE)

    calls: list[torch.nn.Module] = []
    spies: list[_SpyCritic] = []

    def fake_unwrap_net(net: torch.nn.Module) -> torch.nn.Module:
        calls.append(net)
        spy = _SpyCritic(net)
        spies.append(spy)
        return spy

    monkeypatch.setattr("dlk.opt.distributed.unwrap_net", fake_unwrap_net)

    gradient_penalty_fd_adversarial(d_net, x_gen, x_data, y_data)

    assert calls == [d_net]
    assert len(spies) == 1
    assert len(spies[0].inputs) == 1
    assert spies[0].inputs[0].requires_grad


def test_adversarial_flat_critic_direction_fallback_is_finite() -> None:
    """A critic flat in `x` falls back to the random direction and gives a finite 0 penalty."""
    torch.manual_seed(0)
    d_net = _FlatCritic()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)

    penalty = gradient_penalty_fd_adversarial(
        d_net, x_gen, x_data, None, **ONE_SIDED_RELU_KWARGS
    )
    assert torch.isfinite(penalty)
    torch.testing.assert_close(penalty, torch.zeros_like(penalty))
