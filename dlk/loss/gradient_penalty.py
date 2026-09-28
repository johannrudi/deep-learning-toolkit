"""Provide gradient and Lipschitz penalties for network training."""

from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import dlk.opt.distributed as distributed


def _critic(
    d_net: Callable[..., torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor | None,
) -> torch.Tensor:
    """Score `x` with the critic, passing `y` only when it is given."""
    return d_net(x, y) if y is not None else d_net(x)


def _sharp_softplus(z: torch.Tensor) -> torch.Tensor:
    """Approximate `relu` by a softplus with `beta=10`, the default one-sided nonlinearity."""
    return F.softplus(z, beta=10.0)


def _check_eps(eps: float, one_sided: bool) -> None:
    """Raise `ValueError` unless `eps >= 0`, or `eps > 0` for the two-sided penalty."""
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if not one_sided and eps == 0:
        raise ValueError(
            "eps must be > 0 for the two-sided penalty, whose square root has an "
            "infinite derivative at a zero norm"
        )


def _segment_coefficients(
    x_data: torch.Tensor, device: torch.device | None
) -> torch.Tensor:
    """Draw one `U(0, 1)` coefficient per sample, shaped to broadcast over `x_data`."""
    batch_size, *other_dims = x_data.size()
    sample_device = device if device is not None else x_data.device
    coefficient_shape = [batch_size] + [1] * len(other_dims)
    return torch.rand(coefficient_shape, device=sample_device, dtype=x_data.dtype)


# --------------------------------------
# Gradient Penalties with Autograd
# --------------------------------------


def _gradient_norm_sq(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None = None,
    device: torch.device | None = None,
    eager: bool = False,
) -> torch.Tensor:
    """Compute the squared gradient norm of the critic at random interpolates.

    Args:
        d_net: Critic network used to score interpolated samples.
        x_gen: Generated samples from the model.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic.
        device: Device used to sample interpolation coefficients. Defaults to
            `x_data`'s device.
        eager: Whether to run the critic in eager mode, ignoring `torch.compile`.
            Set to ``True`` for a compiled critic, because the penalty needs a
            double backward, which compiled graphs do not support.

    Returns:
        Per-sample squared l2-norm of critic gradients.
    """
    t = _segment_coefficients(x_data, device)
    x_hat = t * x_data + (1 - t) * x_gen
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


def gradient_penalty(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    eps: float = 1e-6,
    one_sided: bool = True,
    one_sided_nonlinearity: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | None = None,
    eager: bool = False,
    d_outputs_gen: torch.Tensor | None = None,
    d_outputs_data: torch.Tensor | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a one- or two-sided gradient penalty at random interpolates.

    Draws one interpolate per sample on the real-to-generated segment,

    .. math::
        \hat{x} = t \, x_\mathrm{data} + (1 - t) \, x_\mathrm{gen},
        \qquad t \sim U(0, 1),

    takes the critic's gradient there, with respect to `x` and, for a
    conditional critic, `y_data` (see `_gradient_norm_sq`),

    .. math::
        g = \nabla_{(x, y)} D(\hat{x}, y_\mathrm{data}),

    and penalizes its norm with the one- and two-sided penalties

    .. math::
        \mathcal{R}_\mathrm{one} = E[\phi(\|g\|^2 + \varepsilon - k^2)] \quad (\texttt{one\_sided=True}),
        \qquad
        \mathcal{R}_\mathrm{two} = E[(\sqrt{\|g\|^2 + \varepsilon} - k)^2] \quad (\texttt{one\_sided=False}),

    with the target Lipschitz constant :math:`k` (argument `lip`), a small
    :math:`\varepsilon` (argument `eps`), and a nonlinearity :math:`\phi`
    (argument `one_sided_nonlinearity`).

    The one-sided penalty follows WGAN-LP (Petzka et al. 2018), applied to
    the squared norm rather than the norm: WGAN-LP penalizes
    :math:`E[\mathrm{relu}(\|g\| - k)^2]`. With :math:`\phi = \mathrm{relu}`,
    both vanish on the same set of points. The default
    :math:`\phi(z) = \mathrm{softplus}(10 z) / 10` is a smooth relu; it stays
    slightly positive below the threshold. Here :math:`\varepsilon \ge 0` is
    a margin that tightens the threshold.

    The two-sided penalty is WGAN-GP (Gulrajani et al. 2017) for
    :math:`k = 1`: the optimal critic of the Kantorovich–Rubinstein dual (see
    `dlk.loss.wasserstein_gan.wasserstein_loss_fn`) has unit gradient norm on
    segments between coupled real and generated points. Here
    :math:`\varepsilon > 0` guards the square root, whose derivative is
    infinite at a zero norm.

    Args:
        d_net: Critic network used to score interpolated samples.
        x_gen: Generated samples from the model.
        x_data: Real data samples.
        y_data: Optional conditional inputs passed to the critic.
        lip: Target Lipschitz constant `k`.
        eps: Small constant `varepsilon`: a margin `>= 0` for the one-sided
            penalty, a guard `> 0` of the square root for the two-sided
            penalty.
        one_sided: Whether to use the one-sided penalty instead of the
            two-sided one.
        one_sided_nonlinearity: Nonlinearity `phi` of the one-sided penalty.
            Defaults to `F.softplus` with `beta=10`.
        device: Device used to sample interpolation coefficients. Defaults to
            `x_data`'s device.
        eager: Whether to run the critic in eager mode; see `_gradient_norm_sq`.
        d_outputs_gen: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        d_outputs_data: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_norm` (mean of `||g||`).

    Returns:
        Scalar gradient penalty term.

    Raises:
        ValueError: If `eps < 0`, or `eps == 0` for the two-sided penalty.

    References:
        Gulrajani, Ahmed, Arjovsky, Dumoulin, Courville, "Improved Training of
        Wasserstein GANs", NeurIPS 2017. https://arxiv.org/abs/1704.00028

        Petzka, Fischer, Lukovnikov, "On the Regularization of Wasserstein
        GANs", ICLR 2018. https://arxiv.org/abs/1709.08894
    """
    del d_outputs_gen, d_outputs_data  # unused

    _check_eps(eps, one_sided)

    # compute the squared-norm of the input-output gradient
    grad_norm_sq = _gradient_norm_sq(
        d_net, x_gen, x_data, y_data=y_data, device=device, eager=eager
    )

    # compute the penalty
    if one_sided:
        nl = one_sided_nonlinearity or _sharp_softplus
        grad_penalty = nl(grad_norm_sq + eps - lip * lip).mean()
    else:
        grad_penalty = ((torch.sqrt(grad_norm_sq + eps) - lip) ** 2).mean()

    # log the gradient norm
    if dlog is not None:
        grad_norm = torch.sqrt(grad_norm_sq.detach())
        dlog["grad_norm"] = grad_norm.mean().item()

    return grad_penalty


# --------------------------------------
# Gradient Penalties with Finite Difference
# --------------------------------------


def _gradient_fd_norm_sq(
    d_a: torch.Tensor,
    d_b: torch.Tensor,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    min_dist: float,
) -> torch.Tensor:
    r"""Compute the squared difference quotient of each pair of points.

    .. math::
        q^2 = \frac{\|D(x_a) - D(x_b)\|^2}{\max(\|x_a - x_b\|^2, \delta^2)},

    with :math:`\delta` (argument `min_dist`).

    Args:
        d_a: Critic outputs at the first point of each pair.
        d_b: Critic outputs at the second point of each pair.
        x_a: First point of each pair.
        x_b: Second point of each pair.
        min_dist: Minimum distance between the points of a pair, guarding
            against division by zero.

    Returns:
        Per-sample squared difference quotient `q**2`.
    """
    d_diff = torch.sum(torch.square((d_a - d_b).flatten(1)), dim=1)
    x_diff = torch.sum(torch.square((x_a - x_b).flatten(1)), dim=1)
    return d_diff / x_diff.clamp(min=min_dist**2)


def _gradient_fd_penalty(
    d_a: torch.Tensor,
    d_b: torch.Tensor,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    lip: float,
    eps: float,
    min_dist: float,
    one_sided: bool,
    one_sided_nonlinearity: Callable[[torch.Tensor], torch.Tensor] | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute the finite-difference penalty shared by the FD variants.

    Penalizes the difference quotient :math:`q` of each pair (see
    `_gradient_fd_norm_sq`) like `gradient_penalty` penalizes the gradient
    norm,

    .. math::
        \mathcal{R}_\mathrm{one} = E[\phi(q^2 + \varepsilon - k^2)] \quad (\texttt{one\_sided=True}),
        \qquad
        \mathcal{R}_\mathrm{two} = E[(\sqrt{q^2 + \varepsilon} - k)^2] \quad (\texttt{one\_sided=False}).

    Args:
        d_a: Critic outputs at the first point of each pair.
        d_b: Critic outputs at the second point of each pair.
        x_a: First point of each pair.
        x_b: Second point of each pair.
        lip: Target Lipschitz constant `k`.
        eps: Small constant `varepsilon`; see `gradient_penalty`.
        min_dist: Minimum distance between the points of a pair, guarding
            against division by zero.
        one_sided: Whether to use the one-sided penalty instead of the
            two-sided one.
        one_sided_nonlinearity: Nonlinearity `phi` of the one-sided penalty.
            Defaults to `F.softplus` with `beta=10`.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_fd_norm` (mean of `q`).

    Returns:
        Scalar finite-difference gradient penalty term.

    Raises:
        ValueError: If `eps < 0`, `eps == 0` for the two-sided penalty, or
            `min_dist <= 0`.
    """
    _check_eps(eps, one_sided)
    if min_dist <= 0:
        raise ValueError(f"min_dist must be > 0, got {min_dist}")

    # compute the squared-norm of the finite difference gradient
    grad_fd_norm_sq = _gradient_fd_norm_sq(d_a, d_b, x_a, x_b, min_dist)

    # compute the penalty
    if one_sided:
        nl = one_sided_nonlinearity or _sharp_softplus
        penalty = nl(grad_fd_norm_sq + eps - lip * lip).mean()
    else:
        penalty = ((torch.sqrt(grad_fd_norm_sq + eps) - lip) ** 2).mean()

    # log the FD-gradient norm
    if dlog is not None:
        grad_fd_norm = torch.sqrt(grad_fd_norm_sq.detach())
        dlog["grad_fd_norm"] = grad_fd_norm.mean().item()

    return penalty


def _random_unit_directions(
    x: torch.Tensor, device: torch.device | None
) -> torch.Tensor:
    """Draw one direction per sample, uniform on the unit sphere of the flattened sample."""
    sample_device = device if device is not None else x.device
    v = torch.randn(x.shape, device=sample_device, dtype=x.dtype)
    v_norm = torch.linalg.vector_norm(v, dim=tuple(range(1, v.dim())), keepdim=True)
    return v / v_norm


def gradient_penalty_fd_segment(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    eps: float = 1e-6,
    min_dist: float = 1e-6,
    one_sided: bool = True,
    one_sided_nonlinearity: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | None = None,
    d_outputs_gen: torch.Tensor | None = None,
    d_outputs_data: torch.Tensor | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a finite-difference gradient penalty on real-to-generated segments.

    Draws two points per sample on the segment between `x_gen` and `x_data`,

    .. math::
        x_a = t_a x_\mathrm{data} + (1 - t_a) x_\mathrm{gen}, \qquad
        x_b = t_b x_\mathrm{data} + (1 - t_b) x_\mathrm{gen},

    with :math:`t_a, t_b \sim U(0, 1)` drawn independently per sample, and
    penalizes their difference quotient

    .. math::
        q = \frac{|D(x_a) - D(x_b)|}{\max(\|x_a - x_b\|, \delta)},

    with a small :math:`\delta` (argument `min_dist`) that guards against
    division by zero, with the one- or two-sided penalty of
    `gradient_penalty`, :math:`q` in place of :math:`\|g\|`.

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
        eps: Small constant `varepsilon`; see `gradient_penalty`.
        min_dist: Minimum distance between the points of a pair, guarding
            against division by zero.
        one_sided: Whether to use the one-sided penalty instead of the
            two-sided one.
        one_sided_nonlinearity: Nonlinearity `phi` of the one-sided penalty.
            Defaults to `F.softplus` with `beta=10`.
        device: Device used to sample the segment coefficients. Defaults to
            `x_data`'s device.
        d_outputs_gen: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        d_outputs_data: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_fd_norm` (mean of `q`).

    Returns:
        Scalar finite-difference gradient penalty term.

    Raises:
        ValueError: If `eps < 0`, `eps == 0` for the two-sided penalty, or
            `min_dist <= 0`.

    References:
        Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
        https://arxiv.org/abs/1704.00028

        Petzka, Fischer, Lukovnikov, "On the Regularization of Wasserstein GANs",
        ICLR 2018. https://arxiv.org/abs/1709.08894

        Wei et al., "Improving the Improved Training of Wasserstein GANs: A
        Consistency Term and Its Dual Effect", ICLR 2018.
        https://arxiv.org/abs/1803.01541
    """
    del d_outputs_gen, d_outputs_data  # unused

    # compute the FD points
    x_gen = x_gen.expand_as(x_data)
    t_a = _segment_coefficients(x_data, device)
    t_b = _segment_coefficients(x_data, device)
    x_a = t_a * x_data + (1 - t_a) * x_gen
    x_b = t_b * x_data + (1 - t_b) * x_gen
    d_a = _critic(d_net, x_a, y_data)
    d_b = _critic(d_net, x_b, y_data)

    # compute the penalty
    return _gradient_fd_penalty(
        d_a,
        d_b,
        x_a,
        x_b,
        lip=lip,
        eps=eps,
        min_dist=min_dist,
        one_sided=one_sided,
        one_sided_nonlinearity=one_sided_nonlinearity,
        dlog=dlog,
    )


def gradient_penalty_fd_adversarial(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    eps: float = 1e-6,
    xi: float = 1e-2,
    radius: float = 1e-1,
    min_dist: float = 1e-6,
    one_sided: bool = True,
    one_sided_nonlinearity: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | None = None,
    d_outputs_gen: torch.Tensor | None = None,
    d_outputs_data: torch.Tensor | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a finite-difference gradient penalty at an adversarially searched direction.

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

    with :math:`\rho` (argument `radius`), and penalizes the difference
    quotient :math:`q(\hat{x}, x')` at :math:`x' = \hat{x} + r_\mathrm{adv}`
    (see `gradient_penalty_fd_segment`) with the one- or two-sided penalty
    of `gradient_penalty`, :math:`q` in place of :math:`\|g\|`.

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
        eps: Small constant `varepsilon`; see `gradient_penalty`.
        xi: Size of the initial random direction used to start the power
            iteration, an absolute distance in input units.
        radius: Perturbation radius `rho` of the adversarial direction, an
            absolute distance in input units.
        min_dist: Minimum distance between the points of a pair, guarding
            against division by zero.
        one_sided: Whether to use the one-sided penalty instead of the
            two-sided one.
        one_sided_nonlinearity: Nonlinearity `phi` of the one-sided penalty.
            Defaults to `F.softplus` with `beta=10`.
        device: Device used to sample the segment coefficient and the
            initial direction. Defaults to `x_data`'s device.
        d_outputs_gen: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        d_outputs_data: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_fd_norm` (mean of `q`).

    Returns:
        Scalar finite-difference gradient penalty term.

    Raises:
        ValueError: If `eps < 0`, `eps == 0` for the two-sided penalty, or
            `min_dist <= 0`.

    References:
        Terjék, "Adversarial Lipschitz Regularization", ICLR 2020.
        https://arxiv.org/abs/1907.05681

        Miyato et al., "Virtual Adversarial Training: A Regularization
        Method for Supervised and Semi-Supervised Learning", TPAMI 2018.
        https://arxiv.org/abs/1704.03976
    """
    del d_outputs_gen, d_outputs_data  # unused

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

    # compute the penalty
    return _gradient_fd_penalty(
        d_hat,
        d_adv,
        x_hat,
        x_adv,
        lip=lip,
        eps=eps,
        min_dist=min_dist,
        one_sided=one_sided,
        one_sided_nonlinearity=one_sided_nonlinearity,
        dlog=dlog,
    )


def gradient_penalty_fd_endpoint(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    eps: float = 1e-6,
    min_dist: float = 1e-6,
    one_sided: bool = True,
    one_sided_nonlinearity: Callable[[torch.Tensor], torch.Tensor] | None = None,
    d_outputs_gen: torch.Tensor | None = None,
    d_outputs_data: torch.Tensor | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a finite-difference gradient penalty at the real and generated endpoints.

    Penalizes the difference quotient of each real-to-generated pair itself,

    .. math::
        q_i = \frac{|D(x_{\mathrm{data},i}) - D(x_{\mathrm{gen},i})|}
                   {\max(\|x_{\mathrm{data},i} - x_{\mathrm{gen},i}\|, \delta)},

    with a small :math:`\delta` (argument `min_dist`) that guards against
    division by zero, with the one- or two-sided penalty of
    `gradient_penalty`, :math:`q` in place of :math:`\|g\|`.

    :math:`q` is the average slope of `D` over the *whole* segment, the
    loosest of the segment-based bounds. The numerator reuses the critic
    outputs already computed by the Wasserstein loss, which the training
    loop passes as `d_outputs_gen` and `d_outputs_data` (see
    `dlk.opt.train_gan.DiscriminatorRegularizerFn`), so it costs no extra
    critic evaluation; `d_net` and `y_data` are unused. A consumer closure
    forwards those outputs:

    .. code-block:: python

        def d_reg_fn(d_net, x_gen, x_data, y_data, *, d_outputs_gen=None, d_outputs_data=None, dlog=None):
            return reg_param * gradient_penalty_fd_endpoint(
                d_net,
                x_gen,
                x_data,
                y_data,
                d_outputs_gen=d_outputs_gen,
                d_outputs_data=d_outputs_data,
                dlog=dlog,
            )

    See `docs/features/2026.008__gradient_penalties__1-plan.md`, Section B.2,
    for the derivation.

    Args:
        d_net: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        x_gen: Generated samples from the model. A single sample is
            broadcast to `x_data`'s batch shape.
        x_data: Real data samples.
        y_data: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        lip: Target Lipschitz constant `k`.
        eps: Small constant `varepsilon`; see `gradient_penalty`.
        min_dist: Minimum distance between the points of a pair, guarding
            against division by zero.
        one_sided: Whether to use the one-sided penalty instead of the
            two-sided one.
        one_sided_nonlinearity: Nonlinearity `phi` of the one-sided penalty.
            Defaults to `F.softplus` with `beta=10`.
        d_outputs_gen: Critic outputs for `x_gen`, from the loss's forward
            pass. Required.
        d_outputs_data: Critic outputs for `x_data`, from the loss's forward
            pass. Required.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_fd_norm` (mean of `q`).

    Returns:
        Scalar finite-difference gradient penalty term.

    Raises:
        ValueError: If `d_outputs_gen` or `d_outputs_data` is ``None``, if
            `eps < 0`, if `eps == 0` for the two-sided penalty, or if
            `min_dist <= 0`.

    References:
        Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
        https://arxiv.org/abs/1704.00028

        Wei et al., "Improving the Improved Training of Wasserstein GANs: A
        Consistency Term and Its Dual Effect", ICLR 2018.
        https://arxiv.org/abs/1803.01541
    """
    del d_net, y_data  # unused

    if d_outputs_gen is None or d_outputs_data is None:
        raise ValueError(
            "gradient_penalty_fd_endpoint requires `d_outputs_gen` and "
            "`d_outputs_data`, the critic outputs from the loss's forward pass"
        )

    # compute the penalty
    x_gen = x_gen.expand_as(x_data)
    return _gradient_fd_penalty(
        d_outputs_data,
        d_outputs_gen,
        x_data,
        x_gen,
        lip=lip,
        eps=eps,
        min_dist=min_dist,
        one_sided=one_sided,
        one_sided_nonlinearity=one_sided_nonlinearity,
        dlog=dlog,
    )


def gradient_penalty_fd_random(
    d_net: Callable[..., torch.Tensor],
    x_gen: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor | None,
    lip: float = 1.0,
    eps: float = 1e-6,
    radius: float = 1e-1,
    min_dist: float = 1e-6,
    one_sided: bool = True,
    one_sided_nonlinearity: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | None = None,
    d_outputs_gen: torch.Tensor | None = None,
    d_outputs_data: torch.Tensor | None = None,
    dlog: dict[str, float] | None = None,
) -> torch.Tensor:
    r"""Compute a finite-difference gradient penalty at randomly perturbed interpolates.

    This is a baseline for ablations only, not a recommended penalty!

    Draws one interpolate per sample on the real-to-generated segment,

    .. math::
        \hat{x} = t x_\mathrm{data} + (1 - t) x_\mathrm{gen},
        \qquad t \sim U(0, 1),

    then perturbs it by a random direction :math:`u`, uniform on the unit
    sphere in :math:`\mathbb{R}^d`, at a fixed radius :math:`\rho` (argument
    `radius`),

    .. math::
        x' = \hat{x} + \rho \, u,

    and penalizes the difference quotient :math:`q(\hat{x}, x')` (see
    `gradient_penalty_fd_segment`) with the one- or two-sided penalty of
    `gradient_penalty`, :math:`q` in place of :math:`\|g\|`.

    The penalty sees only about a :math:`1 / \sqrt{d}` fraction of the slope,
    so it barely constrains the critic in high dimension. Call it in full
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
        eps: Small constant `varepsilon`; see `gradient_penalty`.
        radius: Perturbation radius `rho`, an absolute distance in input
            units.
        min_dist: Minimum distance between the points of a pair, guarding
            against division by zero.
        one_sided: Whether to use the one-sided penalty instead of the
            two-sided one.
        one_sided_nonlinearity: Nonlinearity `phi` of the one-sided penalty.
            Defaults to `F.softplus` with `beta=10`.
        device: Device used to sample the segment coefficient and the
            direction. Defaults to `x_data`'s device.
        d_outputs_gen: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        d_outputs_data: Unused; accepted to match
            `dlk.opt.train_gan.DiscriminatorRegularizerFn`.
        dlog: Optional dictionary for logging summary statistics. Logs
            `grad_fd_norm` (mean of `q`).

    Returns:
        Scalar finite-difference gradient penalty term.

    Raises:
        ValueError: If `eps < 0`, `eps == 0` for the two-sided penalty, or
            `min_dist <= 0`.

    References:
        Kodali et al., "On Convergence and Stability of GANs", 2017.
        https://arxiv.org/abs/1705.07215

        Miyato et al., "Virtual Adversarial Training: A Regularization
        Method for Supervised and Semi-Supervised Learning", TPAMI 2018.
        https://arxiv.org/abs/1704.03976
    """
    del d_outputs_gen, d_outputs_data  # unused

    # compute the FD points
    x_gen = x_gen.expand_as(x_data)
    t = _segment_coefficients(x_data, device)
    x_hat = t * x_data + (1 - t) * x_gen
    u = _random_unit_directions(x_hat, device)
    x_pert = x_hat + radius * u
    d_hat = _critic(d_net, x_hat, y_data)
    d_pert = _critic(d_net, x_pert, y_data)

    # compute the penalty
    return _gradient_fd_penalty(
        d_hat,
        d_pert,
        x_hat,
        x_pert,
        lip=lip,
        eps=eps,
        min_dist=min_dist,
        one_sided=one_sided,
        one_sided_nonlinearity=one_sided_nonlinearity,
        dlog=dlog,
    )
