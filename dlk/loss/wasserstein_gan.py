"""Provide Wasserstein GAN loss and Lipschitz penalties for critic training."""

from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from dlk.opt import distributed


def wasserstein_loss_fn(
    d_outputs_gen: torch.Tensor,
    d_outputs_data: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute the Wasserstein value functions for critic or generator training steps.

    The critic `D` approximates the maximizer of the Kantorovich-Rubinstein dual
    form of the Wasserstein-1 distance between the data distribution
    :math:`P_r` and the generator distribution :math:`P_g` (Villani 2009;
    Arjovsky et al. 2017),

    .. math::
        W_1(P_r, P_g) = \sup_{\|f\|_L \le 1} \; E_{x \sim P_r}[f(x)] - E_{x \sim P_g}[f(x)],

    which this function estimates over a batch as

    .. math::
        \ell_D = -E[ D(x_\mathrm{data}) ] + E[ D(x_\mathrm{gen}) ]
        \ell_G = -E[ D(x_\mathrm{gen}) ]

    Any `K`-Lipschitz critic yields `K` times the true distance, so only `K`
    needs to stay bounded; the exact value does not matter. `gradient_penalty_lip`,
    `gradient_penalty_opt`, and the finite-difference penalties below constrain `K`.

    Args:
        d_outputs_gen: Critic outputs for generated samples. Pass during
            critic and generator updates.
        d_outputs_data: Critic outputs for real samples. Pass during
            critic updates; set to ``None`` for generator-only updates.

    Returns:
        tuple[Tensor, Tensor | None]: A tuple containing:
            - The total critic loss term to minimize.
            - The generated-sample score term when both inputs are provided,
              otherwise ``None`` for generator-only updates.

    References:
        Villani, "Optimal Transport: Old and New", Springer 2009.

        Arjovsky, Chintala, Bottou, "Wasserstein GAN", 2017.
        https://arxiv.org/abs/1701.07875
    """
    assert d_outputs_gen is not None

    # loss for critic update
    if d_outputs_data is not None:
        score_data = torch.mean(d_outputs_data)
        score_gen = torch.mean(d_outputs_gen)
        w_score = score_data - score_gen  # value to be maximized
        w_loss = -w_score  # value to be minimized
        return w_loss, score_gen

    # loss for generator update
    w_loss = -torch.mean(d_outputs_gen)  # value to be minimized
    return w_loss, None


def _critic(
    d_net: Callable[..., torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None,
) -> torch.Tensor:
    """Score `x` with the critic, passing `y` only when it is given."""
    return d_net(x, y) if y is not None else d_net(x)


def gradient_norm_sq(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None = None,
    device: torch.device | None = None,
    eager: bool = False,
) -> torch.Tensor:
    """Compute the squared gradient norm used by Wasserstein penalties.

    Args:
        d_net: Critic network used to score interpolated samples.
        x_gen: Generated samples from the model.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic.
        device: Device used to sample interpolation coefficients.
        eager: Whether to run the critic in eager mode, ignoring `torch.compile`.
            Set to ``True`` for a compiled critic, because the penalty needs a
            double backward, which compiled graphs do not support.

    Returns:
        Per-sample squared L2 norm of critic gradients.
    """
    batch_size, *other_dims = x_data.size()
    epsilon = torch.rand([batch_size] + [1] * len(other_dims), device=device)
    epsilon = epsilon.expand(-1, *other_dims)
    x_hat = epsilon * x_data + (1.0 - epsilon) * x_gen
    x_hat.requires_grad = True
    stance = torch.compiler.set_stance("force_eager") if eager else nullcontext()
    with stance:
        if y_data is not None:
            # detach and re-enable grad on a copy so the caller's tensor is left alone
            y_data = y_data.detach().requires_grad_(True)
            grad_inputs = (x_hat, y_data)
        else:
            grad_inputs = x_hat
        d_outputs_hat = _critic(d_net, x_hat, y_data)
    # compute gradient
    grad_outputs = torch.ones_like(d_outputs_hat)
    grad = torch.autograd.grad(
        outputs=d_outputs_hat,
        inputs=grad_inputs,
        grad_outputs=grad_outputs,
        create_graph=True,  # needed for the gradient wrt. parameters during training
    )
    # compute the squared l2-norm of the gradient; `flatten(1)`, unlike
    # `.view(batch_size, -1)`, also works on non-contiguous gradients
    grad_x = grad[0].flatten(1)
    grad_x_norm_sq = torch.sum(torch.square(grad_x), dim=1)
    if y_data is not None:
        grad_y = grad[1].flatten(1)
        grad_y_norm_sq = torch.sum(torch.square(grad_y), dim=1)
        grad_norm_sq = torch.add(grad_x_norm_sq, grad_y_norm_sq)
    else:
        grad_norm_sq = grad_x_norm_sq
    return grad_norm_sq


# tiny margin added before `sqrt` in the two-sided penalty; a zero gradient
# norm otherwise yields an infinite derivative and NaN parameter gradients
_GRAD_NORM_SQRT_EPS = 1e-12
# TODO: delete after fixing below


# TODO: delete after putting into new function gradient_penalty
def _autograd_gradient_penalty(
    grad_norm_sq: torch.Tensor,
    lip: float,
    one_sided: bool,
    eps: float,
    dlog: dict[str, float] | None,
) -> torch.Tensor:
    """Compute the autograd gradient penalty shared by the two variants below.

    Args:
        grad_norm_sq: Per-sample squared gradient norm, from `gradient_norm_sq`.
        lip: Target Lipschitz constant `k`; `gradient_penalty_opt` fixes it at 1.
        one_sided: ``True`` for the one-sided penalty on the squared norm used
            by `gradient_penalty_lip`; ``False`` for the two-sided penalty on
            the norm itself used by `gradient_penalty_opt`.
        eps: One-sided margin added before thresholding at ``lip * lip``.
            Ignored when `one_sided` is ``False``.
        dlog: Optional dictionary for logging summary statistics.

    Returns:
        Scalar gradient penalty term.
    """
    if one_sided:
        grad_norm = torch.sqrt(grad_norm_sq.detach())  # only for logging purposes
        grad_penalty = F.relu(grad_norm_sq + eps - lip * lip).mean()
    else:
        # TODO: this does not make sense; it should not take a sqrt, too; see ^
        grad_norm = torch.sqrt(grad_norm_sq + _GRAD_NORM_SQRT_EPS)
        grad_penalty = ((grad_norm - lip) ** 2).mean()
    # log to dictionary
    if dlog is not None:
        dlog["grad_norm"] = grad_norm.detach().mean().item()
    return grad_penalty


# TODO: this function should be just gradient_penalty, and a kwarg one_sided sets the
# two versions
def gradient_penalty_lip(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    eps: float = 0.0,
    device: torch.device | None = None,
    eager: bool = False,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a one-sided Lipschitz penalty on the squared gradient norm.

    Draws a random interpolate between `x_gen` and `x_data`,

    .. math::
        \hat{x} = \epsilon \, x_\mathrm{data} + (1 - \epsilon) \, x_\mathrm{gen},
        \qquad \epsilon \sim U(0, 1),

    and penalizes the squared norm of the critic's gradient at :math:`\hat{x}`
    (and, for a conditional critic, at `y_data`; see `gradient_norm_sq`)
    whenever it exceeds the target Lipschitz constant `lip`,

    .. math::
        g = \nabla_{(x, y)} D(\hat{x}, y_\mathrm{data}), \qquad
        \mathcal{R} = E[\mathrm{relu}(\|g\|^2 + \varepsilon - \mathtt{lip}^2)],

    with a margin :math:`\varepsilon` (argument `eps`). This is a one-sided
    variant in the spirit of WGAN-LP (Petzka et al. 2018), applied to the
    squared norm rather than the norm itself: WGAN-LP penalizes
    :math:`E[\mathrm{relu}(\|g\| - \mathtt{lip})^2]`; this function's form
    penalizes the same set of points, with a different shape of penalty.

    Args:
        d_net: Critic network used to score interpolated samples.
        x_gen: Generated samples from the model.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic.
        lip: Target Lipschitz constant `k`. The penalty is zero while the
            gradient norm stays at or below `lip` (up to the margin `eps`).
        eps: Margin added to the squared gradient norm before comparing to
            ``lip * lip``.
        device: Device used to sample interpolation coefficients.
        eager: Whether to run the critic in eager mode; see `gradient_norm_sq`.
        dlog: Optional dictionary for logging summary statistics.

    Returns:
        Scalar gradient penalty term.

    References:
        Gulrajani, Ahmed, Arjovsky, Dumoulin, Courville, "Improved Training of
        Wasserstein GANs", NeurIPS 2017. https://arxiv.org/abs/1704.00028

        Petzka, Fischer, Lukovnikov, "On the Regularization of Wasserstein
        GANs", ICLR 2018. https://arxiv.org/abs/1709.08894
    """
    grad_norm_sq = gradient_norm_sq(
        d_net, x_gen, x_data, y_data=y_data, device=device, eager=eager
    )
    return _autograd_gradient_penalty(
        grad_norm_sq, lip=lip, one_sided=True, eps=eps, dlog=dlog
    )


def gradient_penalty_opt(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    device: torch.device | None = None,
    eager: bool = False,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute the two-sided WGAN-GP gradient penalty.

    Draws a random interpolate between `x_gen` and `x_data`,

    .. math::
        \hat{x} = \epsilon \, x_\mathrm{data} + (1 - \epsilon) \, x_\mathrm{gen},
        \qquad \epsilon \sim U(0, 1),

    and penalizes any deviation of the critic's gradient norm at
    :math:`\hat{x}` (and, for a conditional critic, at `y_data`; see
    `gradient_norm_sq`) from 1, the norm attained by the optimal critic on
    segments between coupled real and generated points (Gulrajani et al.
    2017, Proposition 1),

    .. math::
        g = \nabla_{(x, y)} D(\hat{x}, y_\mathrm{data}), \qquad
        \mathcal{R} = E[(\|g\| - 1)^2].

    This achieves the optimal Kantorovich potential in the
    Kantorovich–Rubinstein duality (see `wasserstein_loss_fn`).

    Args:
        d_net: Critic network used to score interpolated samples.
        x_gen: Generated samples from the model.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic.
        device: Device used to sample interpolation coefficients.
        eager: Whether to run the critic in eager mode; see `gradient_norm_sq`.
        dlog: Optional dictionary for logging summary statistics.

    Returns:
        Scalar gradient penalty term.

    References:
        Gulrajani, Ahmed, Arjovsky, Dumoulin, Courville, "Improved Training of
        Wasserstein GANs", NeurIPS 2017. https://arxiv.org/abs/1704.00028
    """
    grad_norm_sq = gradient_norm_sq(
        d_net, x_gen, x_data, y_data=y_data, device=device, eager=eager
    )
    return _autograd_gradient_penalty(
        grad_norm_sq, lip=1.0, one_sided=False, eps=0.0, dlog=dlog
    )


def _segment_coefficients(
    x_data: torch.Tensor, device: torch.device | None
) -> torch.Tensor:
    """Draw one `U(0, 1)` coefficient per sample, shaped to broadcast over `x_data`."""
    batch_size, *other_dims = x_data.size()
    sample_device = device if device is not None else x_data.device
    coefficient_shape = [batch_size] + [1] * len(other_dims)
    return torch.rand(coefficient_shape, device=sample_device, dtype=x_data.dtype)


def _random_unit_directions(
    x: torch.Tensor, device: torch.device | None
) -> torch.Tensor:
    """Draw one direction per sample, uniform on the unit sphere of the flattened sample."""
    sample_device = device if device is not None else x.device
    v = torch.randn(x.shape, device=sample_device, dtype=x.dtype)
    v_norm = torch.linalg.vector_norm(v, dim=tuple(range(1, v.dim())), keepdim=True)
    return v / v_norm


def _finite_difference_penalty(
    d_a: torch.Tensor,
    d_b: torch.Tensor,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    lip: float,
    one_sided: bool,
    min_dist: float,
    dlog: dict[str, float] | None,
    dlog_prefix: str,
) -> torch.Tensor:
    """Compute the finite-difference Lipschitz penalty shared by the FD variants.

    See `docs/features/2026.008__gradient_penalties__1-plan.md`, Section B.0,
    for the difference-quotient formula and the one- and two-sided penalties.

    Args:
        d_a: Critic outputs at the first point of each pair.
        d_b: Critic outputs at the second point of each pair.
        x_a: First point of each pair.
        x_b: Second point of each pair.
        lip: Target Lipschitz constant.
        one_sided: Whether to penalize only quotients above `lip`.
        min_dist: Minimum denominator, guarding against division by zero.
        dlog: Optional dictionary for logging summary statistics.
        dlog_prefix: Key for the quotient's mean.

    Returns:
        Scalar finite-difference Lipschitz penalty term.
    """
    # TODO: taking a norm and then squaring makes no sense
    d_diff = torch.linalg.vector_norm((d_a - d_b).flatten(1), dim=1)
    x_diff = torch.linalg.vector_norm((x_a - x_b).flatten(1), dim=1).clamp(min=min_dist)
    q = d_diff / x_diff
    if one_sided:
        penalty = F.relu(q - lip).square().mean()
    else:
        penalty = (q - lip).square().mean()
    # log to dictionary
    if dlog is not None:
        dlog[dlog_prefix] = q.detach().mean().item()
    return penalty


def gradient_penalty_lip_fd_segment(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    one_sided: bool = True,
    min_dist: float = 1e-6,
    device: torch.device | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a finite-difference Lipschitz penalty on real-to-generated segments.

    Draws two points per sample on the segment between `x_gen` and `x_data`,

    .. math::
        x_a = t_a x_\mathrm{data} + (1 - t_a) x_\mathrm{gen}, \qquad
        x_b = t_b x_\mathrm{data} + (1 - t_b) x_\mathrm{gen},

    with :math:`t_a, t_b \sim U(0, 1)` drawn independently per sample, and
    penalizes their difference quotient

    .. math::
        q = \frac{|D(x_a) - D(x_b)|}{\max(|t_a - t_b| \, \|x_\mathrm{data} - x_\mathrm{gen}\|, \delta)},

    with a small :math:`\delta` (argument `min_dist`) that guards against
    division by zero, and with the one- and two-sided penalties

    .. math::
        \mathcal{R}_\mathrm{one} = E[\mathrm{relu}(q - k)^2] \quad (\texttt{one\_sided=True}),
        \qquad
        \mathcal{R}_\mathrm{two} = E[(q - k)^2] \quad (\texttt{one\_sided=False}).

    By the mean value theorem, :math:`q` is a lower bound on the local
    Lipschitz constant along the segment, and constrains the slope of `D` in
    `x` only, not in `y`. See
    `docs/features/2026.008__gradient_penalties__1-plan.md`, Section B.1, for
    the derivation.

    Args:
        d_net: Critic network used to score the segment points.
        x_gen: Generated samples from the model. A single sample is
            broadcast to `x_data`'s batch shape.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic, shared by
            both points of each pair.
        lip: Target Lipschitz constant `k`.
        one_sided: Whether to penalize only quotients above `lip`.
        min_dist: Minimum denominator, guarding against division by zero.
        device: Device used to sample the segment coefficients. Defaults to
            `x_data`'s device.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_norm_fd` (mean of `q`).

    Returns:
        Scalar finite-difference Lipschitz penalty term.

    References:
        Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
        https://arxiv.org/abs/1704.00028

        Petzka, Fischer, Lukovnikov, "On the Regularization of Wasserstein GANs",
        ICLR 2018. https://arxiv.org/abs/1709.08894

        Wei et al., "Improving the Improved Training of Wasserstein GANs: A
        Consistency Term and Its Dual Effect", ICLR 2018.
        https://arxiv.org/abs/1803.01541
    """
    x_gen = x_gen.expand_as(x_data)
    t_a = _segment_coefficients(x_data, device)
    t_b = _segment_coefficients(x_data, device)
    x_a = t_a * x_data + (1 - t_a) * x_gen
    x_b = t_b * x_data + (1 - t_b) * x_gen
    d_a = _critic(d_net, x_a, y_data)
    d_b = _critic(d_net, x_b, y_data)
    return _finite_difference_penalty(
        d_a,
        d_b,
        x_a,
        x_b,
        lip=lip,
        one_sided=one_sided,
        min_dist=min_dist,
        dlog=dlog,
        dlog_prefix="grad_norm_fd",
    )


def gradient_penalty_lip_fd_random(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    radius: float = 1e-1,
    min_dist: float = 1e-6,
    device: torch.device | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a finite-difference Lipschitz penalty at randomly perturbed interpolates.

    Draws one interpolate per sample on the real-to-generated segment,

    .. math::
        \hat{x} = t x_\mathrm{data} + (1 - t) x_\mathrm{gen},
        \qquad t \sim U(0, 1),

    then perturbs it by a random direction :math:`u`, uniform on the unit
    sphere in :math:`\mathbb{R}^d`, at a fixed radius :math:`\rho` (argument
    `radius`),

    .. math::
        x' = \hat{x} + \rho \, u,

    and penalizes the one-sided difference quotient

    .. math::
        \mathcal{R} = E[\mathrm{relu}(q(\hat{x}, x') - k)^2].

    The penalty sees only about a :math:`1 / \sqrt{d}` fraction of the slope,
    so it barely constrains the critic in high dimension. It is kept as a
    baseline for ablations only, not a recommended penalty. Call it in full
    precision (outside autocast), as the training loop does; small
    perturbations fall within a few low-precision ulps of the outputs.
    `radius` is an absolute distance in input units; the defaults assume
    roughly standardized data. See
    `docs/features/2026.008__gradient_penalties__1-plan.md`, Section B.4, for
    the derivation.

    Args:
        d_net: Critic network used to score the interpolate and its
            perturbation.
        x_gen: Generated samples from the model. A single sample is
            broadcast to `x_data`'s batch shape.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic, shared by
            both points of each pair.
        lip: Target Lipschitz constant `k`.
        radius: Perturbation radius `rho`, an absolute distance in input
            units.
        min_dist: Minimum denominator, guarding against division by zero.
        device: Device used to sample the segment coefficient and the
            direction. Defaults to `x_data`'s device.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_norm_fd` (mean of `q`).

    Returns:
        Scalar finite-difference Lipschitz penalty term.

    References:
        Kodali et al., "On Convergence and Stability of GANs", 2017.
        https://arxiv.org/abs/1705.07215

        Miyato et al., "Virtual Adversarial Training: A Regularization
        Method for Supervised and Semi-Supervised Learning", TPAMI 2018.
        https://arxiv.org/abs/1704.03976
    """
    x_gen = x_gen.expand_as(x_data)
    t = _segment_coefficients(x_data, device)
    x_hat = t * x_data + (1 - t) * x_gen
    u = _random_unit_directions(x_hat, device)
    x_pert = x_hat + radius * u
    d_hat = _critic(d_net, x_hat, y_data)
    d_pert = _critic(d_net, x_pert, y_data)
    return _finite_difference_penalty(
        d_hat,
        d_pert,
        x_hat,
        x_pert,
        lip=lip,
        one_sided=True,
        min_dist=min_dist,
        dlog=dlog,
        dlog_prefix="grad_norm_fd",
    )


def gradient_penalty_lip_fd_adversarial(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    xi: float = 1e-2,
    radius: float = 1e-1,
    min_dist: float = 1e-6,
    device: torch.device | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a finite-difference Lipschitz penalty at an adversarially searched direction.

    Draws one interpolate per sample on the real-to-generated segment,

    .. math::
        \hat{x} = t x_\mathrm{data} + (1 - t) x_\mathrm{gen},
        \qquad t \sim U(0, 1),

    then searches for the perturbation that most violates the Lipschitz
    constraint,

    .. math::
        r^* = \arg\max_{\|r\| \le \rho} |D(\hat{x}) - D(\hat{x} + r)|,

    approximated by one power-iteration step (Miyato et al. 2018, VAT).
    Starting from a random direction :math:`d \sim \mathcal{N}(0, I)` of size
    :math:`\xi` (argument `xi`),

    .. math::
        r_0 = \xi \, \frac{d}{\|d\|}, \qquad
        g = \nabla_r \, |D(\hat{x}) - D(\hat{x} + r)| \big|_{r = r_0}, \qquad
        r_\mathrm{adv} = \rho \, \frac{g}{\|g\|},

    with :math:`\rho` (argument `radius`), and penalizes the one-sided
    difference quotient at :math:`x' = \hat{x} + r_\mathrm{adv}`,

    .. math::
        \mathcal{R} = E[\mathrm{relu}(q(\hat{x}, x') - k)^2].

    For small `xi`, `g` is parallel to :math:`\nabla_x D(\hat{x})`, so
    :math:`q(\hat{x}, x') \approx \|\nabla_x D(\hat{x})\|`. For a linear
    critic this is exact after one step. The search takes a first-order
    input gradient without `create_graph`, so it works with a compiled
    critic (no `eager` needed). It runs on
    `dlk.opt.distributed.unwrap_net(d_net)`, so it does not arm DDP's
    gradient reducer. Call it in full precision (outside autocast), as the
    training loop does; small perturbations fall within a few low-precision
    ulps of the outputs. `xi` and `radius` are absolute distances in input
    units; the defaults assume roughly standardized data. See
    `docs/features/2026.008__gradient_penalties__1-plan.md`, Section B.3.

    Args:
        d_net: Critic network used to score the interpolate and the
            adversarial perturbation.
        x_gen: Generated samples from the model. A single sample is
            broadcast to `x_data`'s batch shape.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic, shared by
            both points of each pair.
        lip: Target Lipschitz constant `k`.
        xi: Size of the initial random direction used to start the power
            iteration, an absolute distance in input units.
        radius: Perturbation radius `rho` of the adversarial direction, an
            absolute distance in input units.
        min_dist: Minimum denominator, guarding against division by zero.
        device: Device used to sample the segment coefficient and the
            initial direction. Defaults to `x_data`'s device.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_norm_fd` (mean of `q`).

    Returns:
        Scalar finite-difference Lipschitz penalty term.

    References:
        Terjék, "Adversarial Lipschitz Regularization", ICLR 2020.
        https://arxiv.org/abs/1907.05681

        Miyato et al., "Virtual Adversarial Training: A Regularization
        Method for Supervised and Semi-Supervised Learning", TPAMI 2018.
        https://arxiv.org/abs/1704.03976
    """
    x_gen = x_gen.expand_as(x_data)
    t = _segment_coefficients(x_data, device)
    x_hat = t * x_data + (1 - t) * x_gen
    d_hat = _critic(d_net, x_hat, y_data)

    # search on the unwrapped critic; see the docstring. Needed under
    # `static_graph=True` DDP, where a search through the wrapper would
    # desynchronize the critic; the toolkit's DDP training loop does not
    # support that setting, so no test currently observes the difference.
    search_net = (
        distributed.unwrap_net(d_net) if isinstance(d_net, torch.nn.Module) else d_net
    )
    d = _random_unit_directions(x_hat, device)
    r = (xi * d).requires_grad_()
    with torch.enable_grad():
        d_r = _critic(search_net, x_hat.detach() + r, y_data)
        # sum over the batch: samples are independent, so each sample's
        # gradient is its own
        distance = torch.linalg.vector_norm(
            (d_hat.detach() - d_r).flatten(1), dim=1
        ).sum()
    (g,) = torch.autograd.grad(distance, r)

    g_norm = torch.linalg.vector_norm(g, dim=tuple(range(1, g.dim())), keepdim=True)
    # fall back to the random direction where the critic is flat in every
    # direction at that point
    direction = torch.where(
        g_norm > 0, g / g_norm.clamp(min=torch.finfo(g.dtype).tiny), d
    )
    r_adv = (radius * direction).detach()

    x_adv = x_hat + r_adv
    d_adv = _critic(d_net, x_adv, y_data)
    return _finite_difference_penalty(
        d_hat,
        d_adv,
        x_hat,
        x_adv,
        lip=lip,
        one_sided=True,
        min_dist=min_dist,
        dlog=dlog,
        dlog_prefix="grad_norm_fd",
    )


def gradient_penalty_lip_fd_endpoint(
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    d_outputs_gen: torch.Tensor,
    d_outputs_data: torch.Tensor,
    lip: float = 1.0,
    one_sided: bool = True,
    min_dist: float = 1e-6,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a finite-difference Lipschitz penalty at the real and generated endpoints.

    Penalizes the difference quotient of each real-to-generated pair itself,

    .. math::
        q_i = \frac{|D(x_{\mathrm{data},i}) - D(x_{\mathrm{gen},i})|}
                   {\max(\|x_{\mathrm{data},i} - x_{\mathrm{gen},i}\|, \delta)},

    with a small :math:`\delta` (argument `min_dist`) that guards against
    division by zero, and with the one- and two-sided penalties

    .. math::
        \mathcal{R}_\mathrm{one} = E[\mathrm{relu}(q - k)^2] \quad (\texttt{one\_sided=True}),
        \qquad
        \mathcal{R}_\mathrm{two} = E[(q - k)^2] \quad (\texttt{one\_sided=False}).

    :math:`q` is the average slope of `D` over the *whole* segment, the
    loosest of the segment-based bounds. Unlike the other variants, this
    function takes no `d_net`: its numerator reuses critic outputs already
    computed by the Wasserstein loss, so it costs no extra critic evaluation,
    and it takes those outputs directly, obtained through the training
    loop's `d_outputs_gen`/`d_outputs_data` keywords (see
    `dlk.opt.train_gan.DiscriminatorRegularizerFn`, which every regularizer
    must accept). A consumer closure narrows them with an `assert`:

    .. code-block:: python

        def d_reg_fn(d_net, x_gen, x_data, y_data, *, d_outputs_gen=None, d_outputs_data=None, dlog=None):
            assert d_outputs_gen is not None and d_outputs_data is not None
            return reg_param * gradient_penalty_lip_fd_endpoint(
                x_gen=x_gen,
                x_data=x_data,
                d_outputs_gen=d_outputs_gen,
                d_outputs_data=d_outputs_data,
                dlog=dlog,
            )

    See `docs/features/2026.008__gradient_penalties__1-plan.md`, Section B.2,
    for the derivation.

    Args:
        x_gen: Generated samples from the model. A single sample is
            broadcast to `x_data`'s batch shape.
        x_data: Real data samples.
        d_outputs_gen: Critic outputs for `x_gen`, from the loss's forward pass.
        d_outputs_data: Critic outputs for `x_data`, from the loss's forward pass.
        lip: Target Lipschitz constant `k`.
        one_sided: Whether to penalize only quotients above `lip`.
        min_dist: Minimum denominator, guarding against division by zero.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_norm_fd` (mean of `q`).

    Returns:
        Scalar finite-difference Lipschitz penalty term.

    References:
        Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
        https://arxiv.org/abs/1704.00028

        Wei et al., "Improving the Improved Training of Wasserstein GANs: A
        Consistency Term and Its Dual Effect", ICLR 2018.
        https://arxiv.org/abs/1803.01541
    """
    x_gen = x_gen.expand_as(x_data)
    return _finite_difference_penalty(
        d_outputs_data,
        d_outputs_gen,
        x_data,
        x_gen,
        lip=lip,
        one_sided=one_sided,
        min_dist=min_dist,
        dlog=dlog,
        dlog_prefix="grad_norm_fd",
    )
