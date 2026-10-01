r"""Estimate Lipschitz constants of the 1D EfficientNet presets from below.

The inputs are sums of radial basis functions in time,
$x(t) = \sum_j a_j \exp(-(t - t_j)^2 / (2 w_j^2))$, with random amplitudes,
centers, and widths. For the whole network and for each block, two lower
bounds of the Lipschitz constant are estimated: the largest Jacobian norm
(power iteration on $J^\top J$) at the data points and random interpolations,
and the largest finite-difference quotient over random pairs and over pairs on
the segments between them. A block is measured on the residual stream that the
network feeds it.

Per block, the estimates are compared with the rough bound
$\text{skip\_scale} \cdot (1 + L_\text{branch} \cdot \max\lvert\gamma\rvert / \tau)$
of plan 2026.010, Section 4, with $\tau = \sqrt{\text{eps}}$ of the GroupNorm
pre-norm and $L_\text{branch}$ the product of the per-layer bounds (see
`conv_bound` and `SILU_LIPSCHITZ`). The bound is plotted only for the SE-free
blocks of the floored presets ($\tau = 0.01$); post-conv normalizations, the
unfloored $\tau \approx 0.003$, and the SE gate make it uninformative or void.
SE is compared on and off (`se_ratio=None` in every stage config).

The networks run in eval mode, so that the Jacobian is per sample, and in
double precision. Dropout is off. In eval mode, `BASELINE`'s `BatchNorm1d` uses
untrained running statistics that hardly normalize, so its outputs (about
$10^{-10}$) and whole-network estimates are tiny and say nothing about a
trained network.

See ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, Section 4.

Run it as a script; the figures go to the output directory (default: the
current directory), and `--quick` runs a smaller, faster version:

    uv run python examples/nets/efficientnet_empirical_lipschitz.py [output_dir] [--quick]
"""

import argparse
import logging
import math
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from dlk.nets.efficientnet1d import (  # noqa: E402
    BASELINE,
    BASELINE_GN,
    SN_EXACT_DW_FLOORED_PRE_GN,
    SN_EXACT_DW_PRE_GN,
    SN_FLOORED_PRE_GN,
    SN_PRE_GN,
    EfficientNetV1BB0,
    EfficientNetV2BB0,
    FusedMBConv1D,
    HeadConfig,
    MBConv1D,
    NetStyle,
    ScalableEfficientNet1D,
)
from dlk.nets.spectral_norm import get_depthwise_spectral_norm  # noqa: E402

PRESETS: dict[str, NetStyle] = {
    "BASELINE": BASELINE,
    "BASELINE_GN": BASELINE_GN,
    "SN_PRE_GN": SN_PRE_GN,
    "SN_FLOORED_PRE_GN": SN_FLOORED_PRE_GN,
    "SN_EXACT_DW_PRE_GN": SN_EXACT_DW_PRE_GN,
    "SN_EXACT_DW_FLOORED_PRE_GN": SN_EXACT_DW_FLOORED_PRE_GN,
}
BB0Variant = type[EfficientNetV1BB0] | type[EfficientNetV2BB0]
NETWORKS: tuple[BB0Variant, ...] = (EfficientNetV1BB0, EfficientNetV2BB0)
INPUT_LENGTH = 128
NUM_BUMPS = 6
# Smallest GroupNorm eps of a floored preset; below it the bound is not plotted.
MIN_EPS = 1e-4
# Maximum slope of SiLU, max_x SiLU'(x), at x of about 2.4.
SILU_LIPSCHITZ = 1.0998
# Double precision keeps the power iteration finite at tiny Jacobian norms.
DTYPE = torch.float64

logger = logging.getLogger("examples.nets.efficientnet_empirical_lipschitz")


@dataclass
class Estimates:
    """Store Lipschitz estimates, the whole network first, then one entry per block.

    Attributes:
        gradient: Largest Jacobian norm.
        finite_difference: Largest difference quotient.
        bound: Rough upper bound, or `nan` where it does not apply (always for
            the whole network).
    """

    gradient: list[float] = field(default_factory=list)
    finite_difference: list[float] = field(default_factory=list)
    bound: list[float] = field(default_factory=list)


def rbf_signals(num_signals: int, generator: torch.Generator) -> torch.Tensor:
    """Draw sums of Gaussian bumps with random amplitudes, centers, and widths.

    Args:
        num_signals: Number of signals.
        generator: Random number generator.

    Returns:
        Signals of shape `(num_signals, 1, INPUT_LENGTH)`.
    """
    t = torch.linspace(0.0, 1.0, INPUT_LENGTH)
    amplitudes = torch.randn(num_signals, NUM_BUMPS, 1, generator=generator)
    centers = torch.rand(num_signals, NUM_BUMPS, 1, generator=generator)
    widths = 0.01 + 0.09 * torch.rand(num_signals, NUM_BUMPS, 1, generator=generator)
    bumps = amplitudes * torch.exp(-((t - centers) ** 2) / (2 * widths**2))
    return bumps.sum(dim=1, keepdim=True).to(DTYPE)


def per_sample_norm(x: torch.Tensor) -> torch.Tensor:
    """Return the Euclidean norm of each sample of a batch."""
    return x.flatten(1).norm(dim=1)


def jacobian_norms(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    power_iterations: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Estimate each sample's Jacobian operator norm by power iteration.

    `f` must act on each sample independently, so that the batch Jacobian is
    block diagonal. The estimate never exceeds the true norm.

    Args:
        f: Function of a batch of 3D samples.
        x: Batch of points.
        power_iterations: Number of power iteration steps.
        generator: Random number generator for the start vectors.

    Returns:
        The estimated norms, one per sample.
    """
    vjp_fn = torch.func.vjp(f, x)[1]
    v = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    for _ in range(power_iterations):
        v = v / per_sample_norm(v)[:, None, None]
        v = vjp_fn(torch.func.jvp(f, (x,), (v,))[1])[0]
    v = v / per_sample_norm(v)[:, None, None]
    return per_sample_norm(torch.func.jvp(f, (x,), (v,))[1])


def difference_quotients(
    f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Return `|f(x) - f(y)| / |x - y|` for each pair of samples."""
    with torch.no_grad():
        return per_sample_norm(f(x) - f(y)) / per_sample_norm(x - y)


def conv_bound(conv: nn.Conv1d) -> float:
    r"""Return the operator-norm bound of a spectrally normalized convolution.

    - C2 (depthwise): the Bernstein cap $1 / (1 - (k - 1)\pi / n)$ with $n$
      the parametrization's `dft_length`.
    - C0 with kernel size $k$: $\sqrt{k}$. Spectral normalization makes the
      reshaped kernel matrix's norm 1, exactly at initialization, where the
      power iteration is converged; a zero-padded convolution has operator
      norm at most $\sqrt{k}$ times that norm, since each input entry enters
      at most $k$ patches. A $1 \times 1$ convolution gets 1.
    """
    kernel_size = conv.kernel_size[0]
    depthwise = get_depthwise_spectral_norm(conv)
    if depthwise is not None:
        return 1.0 / (1.0 - (kernel_size - 1) * math.pi / depthwise.dft_length)
    return math.sqrt(kernel_size)


def block_bound(block: nn.Module) -> float:
    """Return the rough Lipschitz bound of a block, or `nan` if it does not apply.

    The bound applies to SE-free blocks with a floored GroupNorm pre-norm; a
    block without residual drops the `skip_scale * (1 + ...)` around the branch.
    """
    assert isinstance(block, (MBConv1D, FusedMBConv1D))
    pre_norm = block.pre_norm
    if isinstance(block, MBConv1D) and block.se is not None:
        return math.nan
    if not isinstance(pre_norm, nn.GroupNorm) or pre_norm.eps < MIN_EPS:
        return math.nan
    convs = [m for m in block.modules() if isinstance(m, nn.Conv1d)]
    branch = math.prod(conv_bound(conv) for conv in convs)
    branch *= SILU_LIPSCHITZ ** sum(isinstance(m, nn.SiLU) for m in block.modules())
    gamma = pre_norm.weight.detach().abs().max().item() if pre_norm.affine else 1.0
    branch_with_norm = branch * gamma / math.sqrt(pre_norm.eps)
    if block.config.has_residual:
        return block.style.skip_scale * (1.0 + branch_with_norm)
    return branch_with_norm


def build_network(
    net_cls: BB0Variant, style: NetStyle, se: bool, seed: int
) -> ScalableEfficientNet1D:
    """Build a BB0 network in eval mode, with or without squeeze-and-excitation.

    Args:
        net_cls: `EfficientNetV1BB0` or `EfficientNetV2BB0`.
        style: Preset of the network.
        se: Whether to keep the squeeze-and-excitation layers.
        seed: Seed for the parameters.

    Returns:
        The network in eval mode.
    """
    torch.manual_seed(seed)
    net: ScalableEfficientNet1D = net_cls(
        input_length=INPUT_LENGTH,
        num_classes=1,
        dropout_connect=0.0,
        head=HeadConfig(dropout=0.0),
        style=style,
    )
    if not se:
        # rebuild with the same sizes and every stage config without SE
        stage_configs = [
            replace(stage, config=replace(stage.config, se_ratio=None))
            for stage in net.base_stage_configs
        ]
        torch.manual_seed(seed)
        net = ScalableEfficientNet1D(
            stage_configs=stage_configs,
            stem=net.base_stem,
            head=net.base_head,
            width_coefficient=net.width_coefficient,
            depth_coefficient=net.depth_coefficient,
            input_length=INPUT_LENGTH,
            num_classes=1,
            dropout_connect=0.0,
            style=style,
        )
    return net.to(DTYPE).eval()


def estimate_network(
    net: ScalableEfficientNet1D,
    x: torch.Tensor,
    y: torch.Tensor,
    power_iterations: int,
    generator: torch.Generator,
    case: str,
) -> Estimates:
    """Estimate the Lipschitz constants of a network and of its blocks.

    Logs a warning for every estimate above its bound, which would mean a bug
    or a wrong bound.

    Args:
        net: Network in eval mode.
        x: First signals of the pairs.
        y: Second signals of the pairs.
        power_iterations: Number of power iteration steps per Jacobian norm.
        generator: Random number generator for interpolations and power iteration.
        case: Name of the network for the log.

    Returns:
        The estimates and bounds of the whole network and of each block.
    """
    # interpolate at two random points per pair: Jacobians at both, quotients between them
    alpha = torch.rand(2, x.size(0), 1, 1, generator=generator, dtype=x.dtype)
    a = alpha[0] * x + (1 - alpha[0]) * y
    b = alpha[1] * x + (1 - alpha[1]) * y
    inputs = torch.cat([x, y, a, b])

    # record each block's input at the same points
    block_inputs: list[torch.Tensor] = []
    handles = [
        block.register_forward_pre_hook(
            lambda module, args: block_inputs.append(args[0])
        )
        for block in net.blocks
    ]
    with torch.no_grad():
        net(inputs)
    for handle in handles:
        handle.remove()

    estimates = Estimates()
    functions = [net, *net.blocks]
    points = [inputs, *block_inputs]
    for index, (f, z) in enumerate(zip(functions, points, strict=True)):
        zx, zy, za, zb = z.chunk(4)
        gradient = jacobian_norms(f, z, power_iterations, generator).max().item()
        finite_difference = (
            difference_quotients(f, torch.cat([zx, za]), torch.cat([zy, zb]))
            .max()
            .item()
        )
        bound = math.nan if f is net else block_bound(f)
        # compare with the bound; a `nan` bound never warns
        if max(gradient, finite_difference) > bound:
            logger.warning(
                f"{case}, block {index}: estimate "
                f"{max(gradient, finite_difference):.3g} > bound {bound:.3g}"
            )
        estimates.gradient.append(gradient)
        estimates.finite_difference.append(finite_difference)
        estimates.bound.append(bound)
    return estimates


def plot_blocks(
    net_cls: BB0Variant,
    estimates: dict[tuple[str, bool], Estimates],
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Plot the per-block estimates and bounds of one network, one panel per preset."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, PRESETS, strict=True):
        for se, color in ((True, "tab:blue"), (False, "tab:orange")):
            result = estimates[(name, se)]
            label = "SE on" if se else "SE off"
            blocks = range(1, len(result.gradient))
            ax.plot(
                blocks,
                result.gradient[1:],
                color=color,
                marker="o",
                markersize=3,
                label=f"gradient, {label}",
            )
            ax.plot(
                blocks,
                result.finite_difference[1:],
                color=color,
                linestyle="--",
                marker="x",
                markersize=4,
                label=f"finite difference, {label}",
            )
            # draw the bound as markers; `nan` leaves a block without one
            ax.plot(
                blocks,
                result.bound[1:],
                color=color,
                linestyle="none",
                marker="_",
                markersize=12,
                markeredgewidth=2,
                label=f"bound, {label}",
            )
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.grid(True, which="major", alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Block")
    for ax in axes[:, 0]:
        ax.set_ylabel("Lipschitz estimate (log scale)")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=6)
    fig.suptitle(
        f"{net_cls.__name__}: per-block Lipschitz lower bounds and the rough upper "
        "bound, eval mode, at initialization"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = output_dir / f"lipschitz_blocks_{net_cls.__name__}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_networks(
    estimates: dict[BB0Variant, dict[tuple[str, bool], Estimates]],
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Plot the whole-network estimates of every preset as grouped bars."""
    fig, axes = plt.subplots(1, len(estimates), figsize=(14, 4.5), sharey=True)
    names = list(PRESETS)
    width = 0.2
    bars = [
        (se, kind) for se in (True, False) for kind in ("gradient", "finite_difference")
    ]
    for ax, (net_cls, net_estimates) in zip(axes, estimates.items(), strict=True):
        for offset, (se, kind) in enumerate(bars):
            values = [getattr(net_estimates[(name, se)], kind)[0] for name in names]
            label = f"{kind.replace('_', ' ')}, SE {'on' if se else 'off'}"
            positions = [index + (offset - 1.5) * width for index in range(len(names))]
            ax.bar(positions, values, width=width, label=label)
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_title(net_cls.__name__)
        ax.grid(True, axis="y", which="major", alpha=0.3)
    axes[0].set_ylabel("Lipschitz estimate (log scale)")
    # place the legend below the panels, clear of the bars
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4)
    fig.suptitle("Whole-network Lipschitz lower bounds, eval mode, at initialization")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = output_dir / "lipschitz_networks.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(
    output_dir: pathlib.Path = pathlib.Path("."),
    seed: int = 0,
    num_pairs: int = 16,
    power_iterations: int = 8,
) -> list[pathlib.Path]:
    """Estimate and plot the Lipschitz constants of every preset on both networks.

    Args:
        output_dir: Directory for the figures.
        seed: Seed for the networks, the signals, and the power iteration.
        num_pairs: Number of random input pairs.
        power_iterations: Number of power iteration steps per Jacobian norm.

    Returns:
        The paths of the saved figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    x = rbf_signals(num_pairs, generator)
    y = rbf_signals(num_pairs, generator)

    all_estimates: dict[BB0Variant, dict[tuple[str, bool], Estimates]] = {}
    for net_cls in NETWORKS:
        net_estimates: dict[tuple[str, bool], Estimates] = {}
        for name, style in PRESETS.items():
            for se in (True, False):
                case = f"{net_cls.__name__} {name} SE {'on' if se else 'off'}"
                net = build_network(net_cls, style, se, seed)
                result = estimate_network(net, x, y, power_iterations, generator, case)
                net_estimates[(name, se)] = result
                logger.info(
                    f"{case}: network gradient {result.gradient[0]:.3g}, "
                    f"finite difference {result.finite_difference[0]:.3g}"
                )
        all_estimates[net_cls] = net_estimates

    paths = [
        plot_blocks(net_cls, all_estimates[net_cls], output_dir) for net_cls in NETWORKS
    ]
    paths.append(plot_networks(all_estimates, output_dir))
    for path in paths:
        logger.info(f"saved {path}")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Estimate Lipschitz constants of the 1D EfficientNet presets."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="directory for the figures (default: the current directory)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use fewer pairs and power iteration steps, e.g., for smoke tests",
    )
    args = parser.parse_args()
    if args.quick:
        main(args.output_dir, num_pairs=2, power_iterations=2)
    else:
        main(args.output_dir)
