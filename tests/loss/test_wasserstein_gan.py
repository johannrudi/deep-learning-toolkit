"""Unit tests for the gradient penalties in `dlk.loss.wasserstein_gan`."""

import copy
from collections.abc import Callable, Iterator

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from dlk.loss.wasserstein_gan import gradient_penalty_lip, gradient_penalty_opt

BATCH_SIZE = 8
X_SIZE = 4
Y_SIZE = 3

PenaltyFn = Callable[..., torch.Tensor]


class _Critic(torch.nn.Module):
    """Small conditional critic scoring `(x, y)` pairs."""

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
        return self.net(torch.cat([x, y], dim=1))


@pytest.fixture(autouse=True)
def _reset_dynamo() -> Iterator[None]:
    """Clear compile caches so tests do not share compiled graphs."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _compile(net: torch.nn.Module) -> CompileCounterWithBackend:
    """Compile `net.forward` in place, as `compile_net_from_params` does.

    The `aot_eager` backend runs AOTAutograd, the source of the double-backward
    limitation, without the cost of inductor code generation.
    """
    counter = CompileCounterWithBackend("aot_eager")
    net.forward = torch.compile(net.forward, backend=counter)
    return counter


def _penalty_grads(
    penalty_fn: PenaltyFn, d_net: torch.nn.Module, conditional: bool, eager: bool
) -> list[torch.Tensor | None]:
    """Backpropagate the penalty and return the critic's parameter gradients."""
    torch.manual_seed(0)  # fix the interpolation coefficients
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    y_data = torch.randn(BATCH_SIZE, Y_SIZE) if conditional else None
    d_net.zero_grad()
    penalty = penalty_fn(d_net, x_gen, x_data, y_data, eager=eager)
    penalty.backward()
    return [None if p.grad is None else p.grad.clone() for p in d_net.parameters()]


@pytest.mark.parametrize("penalty_fn", [gradient_penalty_lip, gradient_penalty_opt])
def test_compiled_critic_fails_without_eager(penalty_fn: PenaltyFn) -> None:
    """A compiled critic cannot take the double backward the penalty needs."""
    d_net = _Critic()
    _compile(d_net)

    with pytest.raises(RuntimeError, match="double backward|donated buffer"):
        _penalty_grads(penalty_fn, d_net, conditional=True, eager=False)


@pytest.mark.parametrize("conditional", [True, False])
@pytest.mark.parametrize("penalty_fn", [gradient_penalty_lip, gradient_penalty_opt])
def test_compiled_critic_with_eager_matches_uncompiled(
    penalty_fn: PenaltyFn, conditional: bool
) -> None:
    """With `eager=True`, a compiled critic yields the uncompiled gradients."""
    torch.manual_seed(1)
    d_net_ref = _Critic()
    d_net = copy.deepcopy(d_net_ref)
    _compile(d_net)

    grads_ref = _penalty_grads(penalty_fn, d_net_ref, conditional, eager=False)
    grads = _penalty_grads(penalty_fn, d_net, conditional, eager=True)

    assert any(g is not None for g in grads)
    for grad, grad_ref in zip(grads, grads_ref, strict=True):
        if grad_ref is None:
            assert grad is None
        else:
            assert grad is not None
            torch.testing.assert_close(grad, grad_ref)


def test_eager_does_not_leak_into_later_calls() -> None:
    """The penalty skips compilation, and later critic calls compile again."""
    d_net = _Critic()
    counter = _compile(d_net)

    _penalty_grads(gradient_penalty_opt, d_net, conditional=True, eager=True)
    assert counter.frame_count == 0

    d_net(torch.randn(BATCH_SIZE, X_SIZE), torch.randn(BATCH_SIZE, Y_SIZE))
    assert counter.frame_count == 1
