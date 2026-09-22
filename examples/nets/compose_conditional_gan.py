"""Train a conditional GAN whose generator is a composed network.

The generator is the motivating case of `dlk.nets.compose`: a convolutional
feature stem over a time series, a separate stem over the latent vector, and a
residual trunk over both stems' concatenated outputs. Nothing about that shape
is written as a bespoke module here; the graph is declared, and the widths that
cross an edge are derived from the networks themselves.

The example shows three things:

1. a `ComposedNetConfig` describing the graph, and `create_composed_net`
   building it from already-constructed `dlk` networks;
2. the composed generator dropping into `dlk.opt.train_gan.train_epochs`
   unchanged, at its `g_net(y_data, z)` call contract, beside an ordinary
   `MLPResNet` discriminator;
3. a checkpoint reloading into a freshly built composed net, which works
   because the `state_dict` keys are `nodes.<node name>.<the network's keys>`.

Run it as a script:

    uv run python examples/nets/compose_conditional_gan.py
"""

import logging
import pathlib
import tempfile

import torch

from dlk.loss.least_squares_gan import least_squares_loss_fn
from dlk.nets.compose import (
    ComposedNet,
    ComposedNetConfig,
    NodeConfig,
    create_composed_net,
)
from dlk.nets.efficientnet1d import EfficientNetV1B0Minimal
from dlk.nets.mlp import MLPResNet
from dlk.opt import train_gan
from dlk.opt.utils import checkpoint_load

# shape of the problem: a 3-channel series conditions a 3-dimensional sample
INPUT_CHANNELS = 3
INPUT_LENGTH = 32
SAMPLE_SIZE = 3
LATENT_SIZE = 4
STEM_OUTPUT_SIZE = 8


def generator_config() -> ComposedNetConfig:
    """Describe the generator's graph.

    The external inputs are stated rather than derived, because their order is
    the positional order `train_gan` calls the generator with.

    Returns:
        The configuration of the two-stem generator.
    """
    return ComposedNetConfig(
        nodes=(
            NodeConfig("stem_y", inputs=("features",)),
            NodeConfig("stem_z", inputs=("latent",)),
            NodeConfig("trunk", inputs=("stem_y", "stem_z")),
        ),
        param_external_inputs=("features", "latent"),  # g_net(y_data, z)
    )  # the output resolves to "trunk", the single sink node


def build_generator(logger: logging.Logger | None = None) -> ComposedNet:
    """Build the composed generator from its config and its modules.

    The trunk's input width is the sum of the two stems' output widths, and
    `create_composed_net` checks that at construction rather than at the first
    batch.

    Args:
        logger: Logger for the per-node summary, or `None` for the default.

    Returns:
        The composed generator.
    """
    return create_composed_net(
        **generator_config().to_kwargs(),
        modules={
            "stem_y": EfficientNetV1B0Minimal(
                input_channels=INPUT_CHANNELS,
                input_length=INPUT_LENGTH,
                num_classes=STEM_OUTPUT_SIZE,
            ),
            "stem_z": MLPResNet(
                LATENT_SIZE,
                LATENT_SIZE,
                residual_blocks_sizes=((8, 8, 16, 8),),
            ),
            "trunk": MLPResNet(
                STEM_OUTPUT_SIZE + LATENT_SIZE,
                SAMPLE_SIZE,
                residual_blocks_sizes=((16, 16, 32, 16), (16, 16, 32, 16)),
            ),
        },
        logger=logger,
    )


def build_discriminator() -> MLPResNet:
    """Build an ordinary residual discriminator.

    It flattens and concatenates its two positional inputs, which is the
    `d_net(x_gen, y_data)` call contract of `dlk.opt.train_gan`.

    Returns:
        The discriminator network.
    """
    return MLPResNet(
        SAMPLE_SIZE + INPUT_CHANNELS * INPUT_LENGTH,
        1,
        residual_blocks_sizes=((16, 16, 32, 16),),
    )


def synthetic_dataloader(
    n_samples: int = 32, batch_size: int = 8
) -> torch.utils.data.DataLoader:
    """Draw a synthetic dataset of conditioned samples.

    Each sample is the mean of its own conditioning series over time, plus
    noise, so the conditioning carries signal the generator could use.

    Args:
        n_samples: Number of samples to draw.
        batch_size: Number of samples per batch.

    Returns:
        A dataloader over `(x_data, y_data)` pairs.
    """
    y_data = torch.randn(n_samples, INPUT_CHANNELS, INPUT_LENGTH)
    x_data = y_data.mean(dim=2) + 0.1 * torch.randn(n_samples, SAMPLE_SIZE)
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(n_epochs: int = 2, seed: int = 0) -> dict[str, float]:
    """Train the composed generator, then reload it from its checkpoint.

    Args:
        n_epochs: Number of epochs to train.
        seed: Seed for the networks and the synthetic data.

    Returns:
        The final generator and discriminator losses.
    """
    logger = logging.getLogger("examples.nets.compose_conditional_gan")
    torch.manual_seed(seed)

    g_net = build_generator(logger=logger)
    d_net = build_discriminator()
    logger.info(
        f"generator takes {g_net.resolve_input_shape()} and "
        f"returns {g_net.resolve_output_shape()}"
    )

    dataloader = synthetic_dataloader()
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        train_log = train_gan.train_epochs(
            n_epochs=n_epochs,
            g_net=g_net,
            d_net=d_net,
            dataloader=dataloader,
            z_sample_fn=lambda batch_size: torch.randn(batch_size, LATENT_SIZE),
            g_optimizer=torch.optim.AdamW(g_net.parameters(), lr=1e-3),
            d_optimizer=torch.optim.AdamW(d_net.parameters(), lr=1e-3),
            loss_fn=least_squares_loss_fn,
            logger=logger,
            checkpoint_epochs=n_epochs,
            checkpoint_dir=checkpoint_dir,
        )

        # the composed net reloads into a freshly built one of the same graph
        checkpoint = next(pathlib.Path(checkpoint_dir).glob("**/g-net_e*.pt"))
        reloaded = build_generator(logger=logger)
        assert set(reloaded.state_dict()) == set(g_net.state_dict())
        checkpoint_load(checkpoint, reloaded)

    g_net.eval()
    reloaded.eval()
    y_data = torch.randn(4, INPUT_CHANNELS, INPUT_LENGTH)
    z = torch.randn(4, LATENT_SIZE)
    with torch.no_grad():
        assert torch.equal(g_net(y_data, z), reloaded(y_data, z))

    losses = {
        "g_loss": float(train_log["g_loss_mean"][-1]),
        "d_pre_loss": float(train_log["d_pre_loss_mean"][-1]),
        "d_post_loss": float(train_log["d_post_loss_mean"][-1]),
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    logger.info(f"final losses {losses}, checkpoint reloaded and outputs match")
    return losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
