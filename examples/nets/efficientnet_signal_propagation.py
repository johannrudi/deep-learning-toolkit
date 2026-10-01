"""Plot signal propagation through the 1D EfficientNet presets at initialization.

A signal propagation plot [Brock et al., 2021a] follows the residual stream
through a network: for random inputs, it shows per block the mean squared
activation and the channel-mean squared, i.e., the squared mean of each
channel over the batch and positions, averaged over channels. A stable network
keeps both of order one from block to block; growth or decay means that the
blocks rescale the stream.

The example compares the six `NetStyle` presets of plan 2026.010 on
`EfficientNetV1BB0` and `EfficientNetV2BB0`. It shows how `skip_scale = 0.5`
and the GroupNorm pre-norm of the `SN_*` presets shape the stream: a residual
block averages its input with its branch, and a block without residual (stride
2 or a channel change) replaces the stream by its branch output.

The networks run in train mode, as at the start of training: `BatchNorm1d` in
`BASELINE` normalizes with the batch statistics. In eval mode it would use its
untrained running statistics (mean 0, variance 1) and hardly normalize at all.
GroupNorm computes the same in both modes. Dropout is off.

See ``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, Section 4.

Run it as a script; the figures go to the output directory (default: the
current directory):

    uv run python examples/nets/efficientnet_signal_propagation.py [output_dir]

[Brock et al., 2021a]: https://arxiv.org/abs/2101.08692
"""

import argparse
import logging
import pathlib

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

PRESETS: dict[str, NetStyle] = {
    "BASELINE": BASELINE,
    "BASELINE_GN": BASELINE_GN,
    "SN_PRE_GN": SN_PRE_GN,
    "SN_FLOORED_PRE_GN": SN_FLOORED_PRE_GN,
    "SN_EXACT_DW_PRE_GN": SN_EXACT_DW_PRE_GN,
    "SN_EXACT_DW_FLOORED_PRE_GN": SN_EXACT_DW_FLOORED_PRE_GN,
}
# Line style and marker per preset, so that overlapping curves stay visible.
LINE_STYLES = {
    "BASELINE": ("--", "s"),
    "BASELINE_GN": (":", "o"),
    "SN_PRE_GN": ("-", "^"),
    "SN_FLOORED_PRE_GN": ("-.", "v"),
    "SN_EXACT_DW_PRE_GN": ("-", "D"),
    "SN_EXACT_DW_FLOORED_PRE_GN": ("-.", "x"),
}
NETWORKS = (EfficientNetV1BB0, EfficientNetV2BB0)
BATCH_SIZE = 64
INPUT_LENGTH = 256


def stream_statistics(
    net: ScalableEfficientNet1D, x: torch.Tensor
) -> tuple[list[float], list[float]]:
    """Record the residual stream statistics after the stem and after each block.

    Args:
        net: Network to probe.
        x: Input batch of shape `(batch_size, channels, length)`.

    Returns:
        The mean squared activation and the channel-mean squared, one entry for
        the stem output and one per block.
    """
    mean_squares: list[float] = []
    channel_mean_squares: list[float] = []

    def record(module: nn.Module, inputs: object, output: torch.Tensor) -> None:
        # average over the batch and positions per channel, then over channels
        mean_squares.append(output.pow(2).mean().item())
        channel_mean_squares.append(output.mean(dim=(0, 2)).pow(2).mean().item())

    handles = [net.stem.register_forward_hook(record)]
    handles += [block.register_forward_hook(record) for block in net.blocks]
    with torch.no_grad():
        net(x)
    for handle in handles:
        handle.remove()
    return mean_squares, channel_mean_squares


def plot_network(
    net_cls: type[EfficientNetV1BB0] | type[EfficientNetV2BB0],
    output_dir: pathlib.Path,
    seed: int,
) -> pathlib.Path:
    """Plot the stream statistics of every preset for one network.

    Args:
        net_cls: `EfficientNetV1BB0` or `EfficientNetV2BB0`.
        output_dir: Directory for the figure.
        seed: Seed for the networks and the inputs.

    Returns:
        The path of the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    net: ScalableEfficientNet1D | None = None
    for name, style in PRESETS.items():
        torch.manual_seed(seed)
        net = net_cls(
            input_length=INPUT_LENGTH,
            num_classes=1,
            dropout_connect=0.0,
            head=HeadConfig(dropout=0.0),
            style=style,
        ).train()
        x = torch.randn(BATCH_SIZE, 1, INPUT_LENGTH)
        mean_squares, channel_mean_squares = stream_statistics(net, x)
        positions = range(len(mean_squares))
        linestyle, marker = LINE_STYLES[name]
        line_style = dict(linestyle=linestyle, marker=marker, markersize=4)
        axes[0].plot(positions, mean_squares, label=name, **line_style)
        axes[1].plot(positions, channel_mean_squares, **line_style)

    # mark the blocks without residual, which replace the stream
    assert net is not None
    transitions = [
        index + 1
        for index, block in enumerate(net.blocks)
        if isinstance(block, (MBConv1D, FusedMBConv1D))
        and not block.config.has_residual
    ]
    for ax, title in zip(
        axes, ("Mean squared activation", "Channel-mean squared"), strict=True
    ):
        for position in transitions:
            ax.axvline(position, color="0.85", linewidth=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Residual stream after stem (0) and block i")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("Value (log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{net_cls.__name__} at initialization, train mode, N(0, 1) inputs; "
        "gray lines: blocks without residual"
    )
    fig.tight_layout()

    path = output_dir / f"signal_propagation_{net_cls.__name__}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(
    output_dir: pathlib.Path = pathlib.Path("."), seed: int = 0
) -> list[pathlib.Path]:
    """Plot the signal propagation of every preset on both networks.

    Args:
        output_dir: Directory for the figures.
        seed: Seed for the networks and the inputs.

    Returns:
        The paths of the saved figures.
    """
    logger = logging.getLogger("examples.nets.efficientnet_signal_propagation")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [plot_network(net_cls, output_dir, seed) for net_cls in NETWORKS]
    for path in paths:
        logger.info(f"saved {path}")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Plot signal propagation through the 1D EfficientNet presets."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="directory for the figures (default: the current directory)",
    )
    main(parser.parse_args().output_dir)
