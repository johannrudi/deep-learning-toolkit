r"""Provide spectral norm regularization of network weights.

For every weighted layer :math:`l` with weight :math:`W_l` reshaped to a
matrix :math:`\tilde{W}_l` (out features x everything else, as
`parametrizations.spectral_norm` does for convolutions), `SpectralNormPenalty`
computes

.. math::
    \mathcal{R}_\mathrm{SN} = \frac{1}{2} \sum_l \sigma(\tilde{W}_l)^2
    \quad (\texttt{max\_norm=None}),
    \qquad
    \mathcal{R}_\mathrm{SN} =
        \frac{1}{2} \sum_l \phi\bigl(\sigma(\tilde{W}_l) - k\bigr)^2
    \quad (\texttt{max\_norm}=k),

with a nonlinearity :math:`\phi` (argument `max_norm_nonlinearity`). The
first form is spectral norm regularization as published (Yoshida and Miyato
2017); the second is its hinge variant, which acts on layers above the target
`k`. With :math:`\phi = \mathrm{relu}`, the hinge is exact. The default
:math:`\phi(z) = \mathrm{softplus}(10 z) / 10` is a smooth relu; it stays
slightly positive below the target. Each :math:`\sigma_l` is estimated by power iteration,
with a persistent left singular vector :math:`u_l`:

.. math::
    v_l \leftarrow \frac{\tilde{W}_l^\top u_l}{\|\tilde{W}_l^\top u_l\|},
    \qquad
    u_l \leftarrow \frac{\tilde{W}_l v_l}{\|\tilde{W}_l v_l\|},
    \qquad
    \sigma_l = u_l^\top \tilde{W}_l v_l.

The iteration runs under `torch.no_grad()`, so :math:`u_l, v_l` enter
:math:`\sigma_l` as constants. Autograd then differentiates
:math:`\sigma_l = u_l^\top \tilde{W}_l v_l` directly and produces the rank-one
parameter gradient

.. math::
    \frac{\partial \mathcal{R}_\mathrm{SN}}{\partial \tilde{W}_l} =
        \sigma_l \, u_l v_l^\top
    \quad \text{or} \quad
        \phi(\sigma_l - k) \, \phi'(\sigma_l - k) \, u_l v_l^\top,

with first-order backward only.

See the implementation plans:
`docs/features/2026.008__gradient_penalties__1-plan.md`,
`docs/features/2026.008__gradient_penalties__2-revision.md`.

References:
    Yoshida, Miyato, "Spectral Norm Regularization for Improving the
    Generalizability of Deep Learning", 2017. https://arxiv.org/abs/1705.10941

    Miyato et al., "Spectral Normalization for Generative Adversarial
    Networks", ICLR 2018. https://arxiv.org/abs/1802.05957

    Liu et al., "Spectral Regularization for Combating Mode Collapse in
    GANs", ICCV 2019. https://arxiv.org/abs/1908.10999
"""

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlk.nets.utils import WEIGHTED_LAYER_COMPATIBLE_TYPES, get_spectral_norm
from dlk.opt import distributed


def _default_max_norm_nonlinearity(z: torch.Tensor) -> torch.Tensor:
    """Approximate `relu` by a softplus with `beta=10`."""
    return F.softplus(z, beta=10.0)


def _weight_matrix(module: nn.Module) -> torch.Tensor:
    """Reshape `module.weight` to (out features, rest), dimension 1 first for `ConvTranspose*`."""
    weight = cast(torch.Tensor, module.weight)
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        weight = weight.transpose(0, 1)
    return weight.reshape(weight.size(0), -1)


class SpectralNormPenalty:
    r"""Penalize network weights' spectral norms, as an alternative to spectral normalization.

    Unlike `torch.nn.utils.parametrizations.spectral_norm`, this leaves the
    network's forward pass unchanged; it only penalizes large spectral norms through
    the loss. `__call__` matches `dlk.opt.train_gan.DiscriminatorRegularizerFn`,
    so an instance can be passed directly as `d_reg_fn`:

    .. code-block:: python

        from dlk.loss.spectral_penalty import SpectralNormPenalty

        d_reg_fn = SpectralNormPenalty(max_norm=1.0)

    For Lipschitz control, prefer the hinge form (`max_norm=1.0`): the
    published form (`max_norm=None`) also shrinks layers already below 1,
    which Liu et al. 2019 tie to a spectrum collapse that precedes mode
    collapse. The default smooth hinge still shrinks such layers slightly;
    pass `max_norm_nonlinearity=F.relu` to leave them untouched, as Liu et
    al. suggest.

    An instance holds per-layer power-iteration state keyed by qualified
    module name; use one instance per network.

    Args:
        max_norm: Target spectral norm `k` for the hinge form; `None` uses
            the published form of Yoshida and Miyato 2017.
        max_norm_nonlinearity: Nonlinearity `phi` of the hinge form; unused
            when `max_norm` is `None`. Defaults to `F.softplus` with
            `beta=10`.
        n_power_iterations: Power-iteration steps per call, once a layer's
            :math:`u_l` is warmed up.
        n_warmup_iterations: Power-iteration steps used the first time a
            layer is seen, or after its shape, device, or dtype changes.
        seed: Seed for the local generator that draws each layer's initial
            :math:`u_l`, so that all DDP ranks start from the same vector.

    Raises:
        ValueError: If `max_norm < 0`, `n_power_iterations < 1`, or
            `n_warmup_iterations < 1`.
    """

    def __init__(
        self,
        max_norm: float | None = None,
        max_norm_nonlinearity: Callable[[torch.Tensor], torch.Tensor] | None = None,
        n_power_iterations: int = 1,
        n_warmup_iterations: int = 15,
        seed: int = 0,
    ) -> None:
        if max_norm is not None and max_norm < 0:
            raise ValueError(f"max_norm must be None or >= 0, got {max_norm}")
        if n_power_iterations < 1:
            raise ValueError(
                f"n_power_iterations must be >= 1, got {n_power_iterations}"
            )
        if n_warmup_iterations < 1:
            raise ValueError(
                f"n_warmup_iterations must be >= 1, got {n_warmup_iterations}"
            )
        self.max_norm = max_norm
        self.max_norm_nonlinearity = (
            max_norm_nonlinearity or _default_max_norm_nonlinearity
        )
        self.n_power_iterations = n_power_iterations
        self.n_warmup_iterations = n_warmup_iterations
        self.seed = seed
        self._u: dict[str, torch.Tensor] = {}

    def _estimate_spectral_norm(self, name: str, module: nn.Module) -> torch.Tensor:
        """Estimate `module`'s spectral norm by power iteration and update its `u_l` entry.

        See the module docstring for the iteration and the rank-one gradient
        it produces.
        """
        weight_mat = _weight_matrix(module)
        u = self._u.get(name)
        if (
            u is None
            or u.shape != (weight_mat.size(0),)
            or u.device != weight_mat.device
            or u.dtype != weight_mat.dtype
        ):
            u = torch.randn(
                weight_mat.size(0),
                generator=torch.Generator().manual_seed(self.seed),
            ).to(device=weight_mat.device, dtype=weight_mat.dtype)
            u = F.normalize(u, dim=0)
            n_steps = self.n_warmup_iterations
        else:
            n_steps = self.n_power_iterations

        with torch.no_grad():
            w = weight_mat.detach()
            for _ in range(n_steps):
                v = F.normalize(w.T @ u, dim=0)
                u = F.normalize(w @ v, dim=0)
            self._u[name] = u

        return u @ (weight_mat @ v)

    def _penalty_and_spectral_norms(
        self, net: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the penalty and the stacked, detached sigmas, shared by `penalty` and `__call__`.

        Replaces `self._u` by the entries seen in this call, so a renamed or
        removed layer drops its state (plan Section 11).
        """
        module = distributed.unwrap_net(net)
        sigmas: list[torch.Tensor] = []
        seen_u: dict[str, torch.Tensor] = {}
        for name, layer in module.named_modules():
            if not isinstance(layer, WEIGHTED_LAYER_COMPATIBLE_TYPES):
                continue
            if get_spectral_norm(layer) is not None:
                continue
            sigmas.append(self._estimate_spectral_norm(name, layer))
            seen_u[name] = self._u[name]
        self._u = seen_u

        if not sigmas:
            first_param = next(module.parameters(), None)
            device = (
                first_param.device if first_param is not None else torch.device("cpu")
            )
            return torch.zeros((), device=device), torch.zeros(0, device=device)

        sigmas_stacked = torch.stack(sigmas)
        if self.max_norm is None:
            excess = sigmas_stacked
        else:
            excess = self.max_norm_nonlinearity(sigmas_stacked - self.max_norm)
        penalty = 0.5 * excess.square().sum()
        return penalty, sigmas_stacked.detach()

    def penalty(self, net: nn.Module) -> torch.Tensor:
        """Return the spectral norm penalty of `net`'s qualifying layers; a zero tensor on `net`'s device if none qualify."""
        penalty, _ = self._penalty_and_spectral_norms(net)
        return penalty

    def __call__(
        self,
        d_net: nn.Module,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        *,
        d_outputs_gen: torch.Tensor | None = None,
        d_outputs_data: torch.Tensor | None = None,
        dlog: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Return `self.penalty(d_net)`, matching `dlk.opt.train_gan.DiscriminatorRegularizerFn`.

        `x_gen`, `x_data`, `y_data`, `d_outputs_gen`, and `d_outputs_data` are
        unused: the regularizer only depends on the critic's weights. Logs
        `spectral_norm`, the mean of the layers' sigmas when `dlog` is given
        and at least one layer qualified.
        """
        del x_gen, x_data, y_data, d_outputs_gen, d_outputs_data  # unused

        # compute the penalty
        penalty, sigmas = self._penalty_and_spectral_norms(d_net)

        # log the spectral norm
        if dlog is not None and sigmas.numel() > 0:
            dlog["spectral_norm"] = sigmas.mean().item()

        return penalty
