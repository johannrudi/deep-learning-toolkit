"""Draw several latent samples per observation from a composed generator.

An application that draws replicas evaluates the generator once per latent
sample, typically with `torch.vmap` over the replica dimension. Two things go
wrong if the whole graph is vmapped. The feature stem carries batch
normalization, whose running-statistics update is a write into an unbatched
buffer that `vmap` rejects in training mode. And the stem does not depend on the
latent at all, so evaluating it inside the vmapped region recomputes the same
tensor once per replica.

`ComposedNet.forward_partial` splits the graph at the dependency boundary: the
latent-independent nodes are evaluated once, and only the rest is vmapped.

The example shows:

1. `dependents_of` reporting which nodes the latent reaches;
2. `modules_with_running_stats` asserting the region about to be vmapped is
   free of batch normalization, checked at setup rather than at the first step;
3. `forward_partial` plus `torch.vmap`, verified against evaluating each
   replica in a loop, with a hook proving the stem ran once.

Run it as a script:

    uv run python examples/nets/compose_replica_latents.py
"""

import logging

import torch

from dlk.nets.compose import (
    ComposedNet,
    NodeConfig,
    create_composed_net,
    modules_with_running_stats,
)
from dlk.nets.efficientnet1d import EfficientNetV1B0Minimal
from dlk.nets.mlp import MLPResNet

INPUT_CHANNELS = 3
INPUT_LENGTH = 32
SAMPLE_SIZE = 3
LATENT_SIZE = 4
STEM_OUTPUT_SIZE = 8


def build_generator(logger: logging.Logger | None = None) -> ComposedNet:
    """Build a two-stem generator with a batch-normalized feature stem.

    Args:
        logger: Logger for the per-node summary, or `None` for the default.

    Returns:
        The composed generator.
    """
    return create_composed_net(
        nodes=(
            NodeConfig("stem_y", inputs=("features",)),
            NodeConfig("stem_z", inputs=("latent",)),
            NodeConfig("trunk", inputs=("stem_y", "stem_z")),
        ),
        external_inputs=("features", "latent"),
        output="trunk",
        modules={
            "stem_y": EfficientNetV1B0Minimal(
                input_channels=INPUT_CHANNELS,
                input_length=INPUT_LENGTH,
                num_classes=STEM_OUTPUT_SIZE,
            ),
            "stem_z": MLPResNet(
                LATENT_SIZE, LATENT_SIZE, residual_blocks_sizes=((8, 8, 16, 8),)
            ),
            "trunk": MLPResNet(
                STEM_OUTPUT_SIZE + LATENT_SIZE,
                SAMPLE_SIZE,
                residual_blocks_sizes=((16, 16, 32, 16),),
            ),
        },
        logger=logger,
    )


def main(batch_size: int = 4, replicas: int = 8, seed: int = 0) -> torch.Tensor:
    """Generate several replicas per observation, splitting the graph once.

    Args:
        batch_size: Number of observations per batch.
        replicas: Number of latent samples drawn per observation.
        seed: Seed for the network and the inputs.

    Returns:
        The generated samples, of shape `(replicas, batch_size, sample size)`.
    """
    logger = logging.getLogger("examples.nets.compose_replica_latents")
    torch.manual_seed(seed)
    net = build_generator(logger=logger)
    net.train()

    # which nodes the latent reaches, and what stays outside the vmapped region
    latent_region = net.dependents_of("latent")
    independent = [name for name in net.topological_order if name not in latent_region]
    logger.info(
        f"latent reaches {sorted(latent_region)}, evaluated once: {independent}"
    )

    # the region about to be vmapped must hold no running statistics
    blocking = modules_with_running_stats(net, latent_region)
    assert not blocking, f"batch normalization inside the vmapped region: {blocking}"
    logger.info(
        f"nodes with running statistics: "
        f"{sorted(modules_with_running_stats(net, net.topological_order))}"
    )

    y_data = torch.randn(batch_size, INPUT_CHANNELS, INPUT_LENGTH)
    z = torch.randn(replicas, batch_size, LATENT_SIZE)

    # count the feature stem's evaluations
    stem_calls: list[int] = []
    handle = net.nodes["stem_y"].register_forward_hook(
        lambda module, inputs, output: stem_calls.append(1)
    )

    cache = net.forward_partial(exclude_dependents_of="latent", features=y_data)
    x_gen = torch.vmap(lambda z_: net(y_data, z_, cache=cache))(z)
    handle.remove()

    assert x_gen.shape == (replicas, batch_size, SAMPLE_SIZE)
    assert len(stem_calls) == 1, f"the stem ran {len(stem_calls)} times"
    logger.info(
        f"generated {tuple(x_gen.shape)} with {len(stem_calls)} stem evaluation "
        f"for {replicas} replicas"
    )

    # the split changes what is computed, not what comes out
    net.eval()
    cache_eval = net.forward_partial(exclude_dependents_of="latent", features=y_data)
    with torch.no_grad():
        split = torch.vmap(lambda z_: net(y_data, z_, cache=cache_eval))(z)
        looped = torch.stack([net(y_data, z[index]) for index in range(replicas)])
    assert torch.allclose(split, looped, atol=1e-6)
    logger.info("the vmapped split matches evaluating each replica in a loop")

    # the cache belongs to one forward and its backward pass; a new step needs
    # a new cache, since reusing it backpropagates through a freed graph
    return x_gen


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
