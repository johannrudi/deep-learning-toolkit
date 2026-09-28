"""Unit tests for the autograd gradient penalty in `dlk.loss.gradient_penalty`."""

import copy
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

from dlk.loss.gradient_penalty import gradient_penalty

BATCH_SIZE = 8
X_SIZE = 4
Y_SIZE = 3

CompileFn = Callable[[torch.nn.Module], CompileCounterWithBackend]


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


def _penalty_grads(
    one_sided: bool, d_net: torch.nn.Module, conditional: bool, eager: bool
) -> list[torch.Tensor | None]:
    """Backpropagate the penalty and return the critic's parameter gradients."""
    torch.manual_seed(0)  # fix the interpolation coefficients
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)
    y_data = torch.randn(BATCH_SIZE, Y_SIZE) if conditional else None
    d_net.zero_grad()
    penalty = gradient_penalty(
        d_net, x_gen, x_data, y_data, one_sided=one_sided, eager=eager
    )
    penalty.backward()
    return [None if p.grad is None else p.grad.clone() for p in d_net.parameters()]


@pytest.mark.parametrize("one_sided", [True, False])
def test_compiled_critic_fails_without_eager(
    one_sided: bool, compile_aot_eager: CompileFn
) -> None:
    """A compiled critic cannot take the double backward the penalty needs."""
    d_net = _Critic()
    compile_aot_eager(d_net)

    with pytest.raises(RuntimeError, match="double backward|donated buffer"):
        _penalty_grads(one_sided, d_net, conditional=True, eager=False)


@pytest.mark.parametrize("conditional", [True, False])
@pytest.mark.parametrize("one_sided", [True, False])
def test_compiled_critic_with_eager_matches_uncompiled(
    one_sided: bool, conditional: bool, compile_aot_eager: CompileFn
) -> None:
    """With `eager=True`, a compiled critic yields the uncompiled gradients."""
    torch.manual_seed(1)
    d_net_ref = _Critic()
    d_net = copy.deepcopy(d_net_ref)
    compile_aot_eager(d_net)

    grads_ref = _penalty_grads(one_sided, d_net_ref, conditional, eager=False)
    grads = _penalty_grads(one_sided, d_net, conditional, eager=True)

    assert any(g is not None for g in grads)
    for grad, grad_ref in zip(grads, grads_ref, strict=True):
        if grad_ref is None:
            assert grad is None
        else:
            assert grad is not None
            torch.testing.assert_close(grad, grad_ref)


def test_eager_does_not_leak_into_later_calls(compile_aot_eager: CompileFn) -> None:
    """The penalty skips compilation, and later critic calls compile again."""
    d_net = _Critic()
    counter = compile_aot_eager(d_net)

    _penalty_grads(False, d_net, conditional=True, eager=True)
    assert counter.frame_count == 0

    d_net(torch.randn(BATCH_SIZE, X_SIZE), torch.randn(BATCH_SIZE, Y_SIZE))
    assert counter.frame_count == 1


@pytest.mark.parametrize(
    ("one_sided", "eps"), [(True, -1e-6), (False, -1e-6), (False, 0.0)]
)
def test_invalid_eps_raises(one_sided: bool, eps: float) -> None:
    """`eps < 0` raises, and so does `eps == 0` for the two-sided penalty."""
    x = torch.randn(BATCH_SIZE, X_SIZE)
    with pytest.raises(ValueError, match="eps must be"):
        gradient_penalty(_Critic(), x, x, None, eps=eps, one_sided=one_sided)


def test_one_sided_default_nonlinearity_is_sharp_softplus() -> None:
    """The default one-sided nonlinearity is `softplus` with `beta=10`."""
    torch.manual_seed(0)
    d_net = _Critic()
    x_gen = torch.randn(BATCH_SIZE, X_SIZE)
    x_data = torch.randn(BATCH_SIZE, X_SIZE)

    torch.manual_seed(1)
    penalty = gradient_penalty(d_net, x_gen, x_data, None)
    torch.manual_seed(1)
    penalty_ref = gradient_penalty(
        d_net,
        x_gen,
        x_data,
        None,
        one_sided_nonlinearity=lambda z: F.softplus(z, beta=10.0),
    )

    torch.testing.assert_close(penalty, penalty_ref)
