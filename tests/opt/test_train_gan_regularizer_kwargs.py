"""Tests for the opt-in `d_outputs_gen`/`d_outputs_data` keywords in `dlk.opt.train_gan`."""

import functools
from collections.abc import Callable
from typing import Any

import pytest
import torch

from dlk.loss.wasserstein_gan import (
    gradient_penalty_lip_fd_adversarial,
    gradient_penalty_lip_fd_endpoint,
    gradient_penalty_lip_fd_random,
    gradient_penalty_lip_fd_segment,
)
from dlk.opt.monitor import TrainLog
from dlk.opt.train_gan import DiscriminatorRegularizerFn, GANLossFn, train_epochs

BATCH_SIZE = 4
N_BATCHES = 2
X_SIZE = 3
Y_SIZE = 2
Z_SIZE = 2


class _Generator(torch.nn.Module):
    """Tiny conditional generator mapping `(y, z)` to samples."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(Y_SIZE + Z_SIZE, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, X_SIZE),
        )

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((y, z), dim=1))


class _Discriminator(torch.nn.Module):
    """Tiny conditional discriminator mapping `(x, y)` to a score."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(X_SIZE + Y_SIZE, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((x, y), dim=1))


def _build_gan() -> (
    tuple[
        torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.optim.Optimizer
    ]
):
    """Build a tiny deterministic generator/critic pair with SGD optimizers."""
    g_net = _Generator()
    d_net = _Discriminator()
    g_optimizer = torch.optim.SGD(g_net.parameters(), lr=0.05)
    d_optimizer = torch.optim.SGD(d_net.parameters(), lr=0.05)
    return g_net, d_net, g_optimizer, d_optimizer


def _make_dataloader() -> torch.utils.data.DataLoader:
    """Build a small deterministic dataloader for the GAN loop tests."""
    generator = torch.Generator().manual_seed(0)
    x_data = torch.randn(
        (BATCH_SIZE * N_BATCHES, X_SIZE), generator=generator, dtype=torch.float32
    )
    y_data = torch.randn(
        (BATCH_SIZE * N_BATCHES, Y_SIZE), generator=generator, dtype=torch.float32
    )
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def _z_sample_fn(batch_size: int) -> torch.Tensor:
    """Sample latent vectors for the generator."""
    return torch.randn((batch_size, Z_SIZE))


def _loss_fn(
    d_outputs_gen: torch.Tensor, d_outputs_data: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute a Wasserstein-style loss for discriminator and generator steps."""
    if d_outputs_data is None:
        return -d_outputs_gen.mean(), None
    return d_outputs_gen.mean() - d_outputs_data.mean(), -d_outputs_gen.mean()


def _train(
    d_reg_fn: DiscriminatorRegularizerFn, **overrides: Any
) -> tuple[TrainLog, torch.nn.Module]:
    """Seed, build a tiny GAN and dataloader, and run `train_epochs` with shared defaults.

    `overrides` merges into the `train_epochs` keyword arguments (`loss_fn`,
    `n_epochs`, `d_opt_pre`, `d_opt_post`, `autocast_dtype`, ...). Returns the
    epoch-level dlog and the trained `d_net`.
    """
    torch.manual_seed(0)
    g_net, d_net, g_optimizer, d_optimizer = _build_gan()
    kwargs: dict[str, Any] = {
        "n_epochs": 1,
        "g_net": g_net,
        "d_net": d_net,
        "dataloader": _make_dataloader(),
        "z_sample_fn": _z_sample_fn,
        "g_optimizer": g_optimizer,
        "d_optimizer": d_optimizer,
        "loss_fn": _loss_fn,
        "d_reg_fn": d_reg_fn,
    }
    kwargs.update(overrides)
    epoch_dlog: TrainLog = train_epochs(**kwargs)
    return epoch_dlog, d_net


def _make_recording_loss_fn() -> (
    tuple[GANLossFn, list[tuple[torch.Tensor, torch.Tensor]]]
):
    """Build `_loss_fn`, plus the list of `(d_outputs_gen, d_outputs_data)` from discriminator steps.

    Generator-step calls (`d_outputs_data is None`) are not recorded, so the
    recorded list lines up one-to-one, in order, with discriminator-step
    regularizer calls in the same run.
    """
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def loss_fn(
        d_outputs_gen: torch.Tensor, d_outputs_data: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if d_outputs_data is not None:
            calls.append((d_outputs_gen, d_outputs_data))
        return _loss_fn(d_outputs_gen, d_outputs_data)

    return loss_fn, calls


def test_regularizer_declaring_only_gen_receives_only_that_keyword() -> None:
    """A regularizer declaring only `d_outputs_gen` is called with only that keyword."""
    received: list[torch.Tensor] = []

    def d_reg_fn(
        d_net: torch.nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        *,
        d_outputs_gen: torch.Tensor | None = None,
        dlog: dict[str, float] | None = None,
    ) -> torch.Tensor:
        assert d_outputs_gen is not None
        received.append(d_outputs_gen)
        return d_outputs_gen.new_tensor(0.0)

    # `d_reg_fn` has no `d_outputs_data` parameter and no `**kwargs`; if the
    # loop tried to pass it anyway, this call would raise `TypeError`.
    _train(d_reg_fn, d_opt_pre=1, d_opt_post=0)

    # exactly one discriminator step per batch, each calling `d_reg_fn` once
    assert len(received) == N_BATCHES


def test_fixed_signature_closure_runs_with_no_unexpected_keyword() -> None:
    """A closure with the fixed `fhn_gan`/`mops_gan` signature still runs."""

    def d_reg_fn(
        d_net: torch.nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        *,
        dlog: dict[str, float] | None = None,
    ) -> torch.Tensor:
        penalty = 0.1 * d_net(x_data, y_data).square().mean()
        if dlog is not None:
            dlog["reg"] = penalty.item()
        return penalty

    epoch_dlog, _ = _train(d_reg_fn)

    assert epoch_dlog["d_pre_reg_mean"].shape == (1,)


def _both_keywords_regularizer() -> (
    tuple[DiscriminatorRegularizerFn, list[dict[str, torch.Tensor]]]
):
    """Build a regularizer that declares both opt-in keywords, recording every call."""
    calls: list[dict[str, torch.Tensor]] = []

    def d_reg_fn(
        d_net: torch.nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        *,
        d_outputs_gen: torch.Tensor | None = None,
        d_outputs_data: torch.Tensor | None = None,
        dlog: dict[str, float] | None = None,
    ) -> torch.Tensor:
        assert d_outputs_gen is not None and d_outputs_data is not None
        calls.append({"d_outputs_gen": d_outputs_gen, "d_outputs_data": d_outputs_data})
        return d_outputs_gen.new_tensor(0.0)

    return d_reg_fn, calls


@pytest.mark.parametrize("autocast_dtype", [None, torch.bfloat16])
def test_regularizer_declaring_both_keywords_receives_matching_full_batch_float32(
    autocast_dtype: torch.dtype | None,
) -> None:
    """Both keywords carry full-batch, `float32`, graph-attached copies of the loss's outputs."""
    loss_fn, loss_calls = _make_recording_loss_fn()
    d_reg_fn, reg_calls = _both_keywords_regularizer()

    _train(
        d_reg_fn,
        loss_fn=loss_fn,
        d_opt_pre=1,
        d_opt_post=0,
        autocast_dtype=autocast_dtype,
    )

    assert len(reg_calls) == len(loss_calls) == N_BATCHES
    for reg_call, (loss_gen, loss_data) in zip(reg_calls, loss_calls, strict=True):
        d_outputs_gen = reg_call["d_outputs_gen"]
        d_outputs_data = reg_call["d_outputs_data"]
        assert d_outputs_gen.shape[0] == BATCH_SIZE
        assert d_outputs_data.shape[0] == BATCH_SIZE
        assert d_outputs_gen.dtype == torch.float32
        assert d_outputs_data.dtype == torch.float32
        assert d_outputs_gen.grad_fn is not None
        assert d_outputs_data.grad_fn is not None
        torch.testing.assert_close(d_outputs_gen, loss_gen.float())
        torch.testing.assert_close(d_outputs_data, loss_data.float())


def _scenario_functools_partial() -> None:
    """`functools.partial` of a function declaring both opt-in keywords, with an extra bound arg."""
    received: list[dict[str, torch.Tensor]] = []

    def reg_fn(
        weight: float,
        d_net: torch.nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        *,
        d_outputs_gen: torch.Tensor | None = None,
        d_outputs_data: torch.Tensor | None = None,
        dlog: dict[str, float] | None = None,
    ) -> torch.Tensor:
        assert d_outputs_gen is not None and d_outputs_data is not None
        received.append(
            {"d_outputs_gen": d_outputs_gen, "d_outputs_data": d_outputs_data}
        )
        return weight * d_outputs_gen.new_tensor(0.0)

    _train(functools.partial(reg_fn, 0.5), d_opt_pre=1, d_opt_post=0)

    assert len(received) == N_BATCHES
    assert all({"d_outputs_gen", "d_outputs_data"} == set(c) for c in received)


def _scenario_kwargs_closure() -> None:
    """A regularizer accepting `**kwargs` receives both opt-in keywords through it."""
    received: list[dict[str, torch.Tensor]] = []

    def reg_fn(
        d_net: torch.nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        dlog: dict[str, float] | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        received.append(dict(kwargs))
        return x_data.new_tensor(0.0)

    _train(reg_fn, d_opt_pre=1, d_opt_post=0)

    assert len(received) == N_BATCHES
    assert all({"d_outputs_gen", "d_outputs_data"} == set(c) for c in received)


def _scenario_callable_class() -> None:
    """A callable class instance, with the opt-in keywords declared on `__call__`."""

    class _Regularizer:
        def __init__(self) -> None:
            self.calls: list[dict[str, torch.Tensor]] = []

        def __call__(
            self,
            d_net: torch.nn.Module,
            x_gen: torch.Tensor,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            *,
            d_outputs_gen: torch.Tensor | None = None,
            d_outputs_data: torch.Tensor | None = None,
            dlog: dict[str, float] | None = None,
        ) -> torch.Tensor:
            assert d_outputs_gen is not None and d_outputs_data is not None
            self.calls.append(
                {"d_outputs_gen": d_outputs_gen, "d_outputs_data": d_outputs_data}
            )
            return d_outputs_gen.new_tensor(0.0)

    reg_fn = _Regularizer()
    _train(reg_fn, d_opt_pre=1, d_opt_post=0)

    assert len(reg_fn.calls) == N_BATCHES
    assert all({"d_outputs_gen", "d_outputs_data"} == set(c) for c in reg_fn.calls)


def _scenario_module_both_keywords() -> None:
    """An `nn.Module` regularizer, with the opt-in keywords declared on `forward`."""

    class _Regularizer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, torch.Tensor]] = []

        def forward(
            self,
            d_net: torch.nn.Module,
            x_gen: torch.Tensor,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            *,
            d_outputs_gen: torch.Tensor | None = None,
            d_outputs_data: torch.Tensor | None = None,
            dlog: dict[str, float] | None = None,
        ) -> torch.Tensor:
            assert d_outputs_gen is not None and d_outputs_data is not None
            self.calls.append(
                {"d_outputs_gen": d_outputs_gen, "d_outputs_data": d_outputs_data}
            )
            return d_outputs_gen.new_tensor(0.0)

    reg_fn = _Regularizer()
    _train(reg_fn, d_opt_pre=1, d_opt_post=0)

    assert len(reg_fn.calls) == N_BATCHES
    assert all({"d_outputs_gen", "d_outputs_data"} == set(c) for c in reg_fn.calls)


def _scenario_module_fixed_signature() -> None:
    """An `nn.Module` regularizer with the fixed signature (no opt-in keywords) trains.

    Regression: `inspect.signature` on the module instance itself reports
    `Module.__call__`'s `(*args, **kwargs)`, not `forward`'s actual
    parameters. Before `_critic_output_kwargs` inspected `forward` directly,
    this made the loop think the module accepted every opt-in keyword and
    pass them, raising `TypeError` from `forward`.
    """

    class _Regularizer(torch.nn.Module):
        def forward(
            self,
            d_net: torch.nn.Module,
            x_gen: torch.Tensor,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            *,
            dlog: dict[str, float] | None = None,
        ) -> torch.Tensor:
            return 0.1 * d_net(x_data, y_data).square().mean()

    epoch_dlog, _ = _train(_Regularizer(), d_opt_pre=1, d_opt_post=0)

    assert epoch_dlog["d_pre_reg_mean"].shape == (1,)


@pytest.mark.parametrize(
    "scenario",
    [
        _scenario_functools_partial,
        _scenario_kwargs_closure,
        _scenario_callable_class,
        _scenario_module_both_keywords,
        _scenario_module_fixed_signature,
    ],
    ids=[
        "functools_partial",
        "kwargs_closure",
        "callable_class",
        "module_both_keywords",
        "module_fixed_signature",
    ],
)
def test_regularizer_forms_receive_declared_keywords(
    scenario: Callable[[], None],
) -> None:
    """Each regularizer form trains and, if it declares opt-in keywords, receives them."""
    scenario()


def test_endpoint_closure_trains_and_logs_positive_lip_quotient() -> None:
    """The Section 6 endpoint closure trains for a few steps and logs a positive `lip_quotient`."""
    reg_param = 0.1

    def d_reg_fn(
        d_net: torch.nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        *,
        d_outputs_gen: torch.Tensor | None = None,
        d_outputs_data: torch.Tensor | None = None,
        dlog: dict[str, float] | None = None,
    ) -> torch.Tensor:
        assert d_outputs_gen is not None and d_outputs_data is not None
        return reg_param * gradient_penalty_lip_fd_endpoint(
            x_gen=x_gen,
            x_data=x_data,
            d_outputs_gen=d_outputs_gen,
            d_outputs_data=d_outputs_data,
            dlog=dlog,
        )

    epoch_dlog, d_net = _train(d_reg_fn, n_epochs=2)

    # rebuild the untrained critic deterministically (same seed, same
    # construction order as inside `_train`) to compare parameters against
    torch.manual_seed(0)
    _, d_net_init, _, _ = _build_gan()

    assert (epoch_dlog["d_pre_lip_quotient_mean"] > 0.0).all()
    assert (
        epoch_dlog["d_post_lip_quotient_mean"] > 0.0
    ).all()  # d_opt_post defaults to 1
    changed = [
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(
            d_net_init.parameters(), d_net.parameters(), strict=True
        )
    ]
    assert any(changed)


@pytest.mark.parametrize(
    "penalty_fn",
    [
        gradient_penalty_lip_fd_segment,
        gradient_penalty_lip_fd_random,
        gradient_penalty_lip_fd_adversarial,
    ],
)
def test_fixed_signature_penalty_closures_train_without_keyerror(
    penalty_fn: Callable[..., torch.Tensor],
) -> None:
    """Segment, random, and adversarial closures, called with the fixed signature, train one epoch.

    Regression guard: these penalties log `lip_quotient`/`lip_quotient_max`
    through the same `dlog` path as the endpoint variant, which
    `monitor.batch_update` only accepts for tags in `MONITOR_BASENAMES`.
    """

    def d_reg_fn(
        d_net: torch.nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        *,
        dlog: dict[str, float] | None = None,
    ) -> torch.Tensor:
        return 0.1 * penalty_fn(d_net, x_gen, x_data, y_data, dlog=dlog)

    _train(d_reg_fn)
