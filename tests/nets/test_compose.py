import inspect
import os
import subprocess
import sys
from dataclasses import fields
from typing import Any, Literal

import pytest
import torch
import torch.nn as nn

from dlk.nets.compose import (
    ComposedNet,
    ComposedNetConfig,
    HiddenEdge,
    NetShape,
    NodeConfig,
    Shape,
    create_composed_net,
    modules_with_running_stats,
    resolve_shape,
)
from dlk.nets.conv1d import ConvNet, ConvResNet
from dlk.nets.efficientnet1d import EfficientNetV1B0Minimal
from dlk.nets.mlp import MLPResNet


class _ShapedNet(nn.Module):
    """Answer the shape protocol with the shapes it was built with."""

    def __init__(self, input_shape: Shape, output_shape: Shape) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def resolve_input_shape(self) -> Shape:
        """Return the per-sample input shape."""
        return self.input_shape

    def resolve_output_shape(self) -> Shape:
        """Return the per-sample output shape."""
        return self.output_shape


def _chain_net(modules: dict[str, nn.Module]) -> ComposedNet:
    """Build a composed net chaining the given modules, one after the other.

    Shapes are left unchecked and no forward pass is run; the tests using this
    inspect the node set rather than the tensors.

    Args:
        modules: One module per node, keyed by node name, in chain order.

    Returns:
        The composed net over those nodes.
    """
    names = list(modules)
    nodes = tuple(
        NodeConfig(name, inputs=(names[index - 1] if index else "x",))
        for index, name in enumerate(names)
    )
    return ComposedNet(
        nodes=nodes,
        external_inputs=("x",),
        output=names[-1],
        modules=modules,
        validate_shapes=False,
    )


def test_net_shape_accepts_a_module_implementing_both_accessors() -> None:
    """Recognize a module that answers both shape questions."""
    net = _ShapedNet((3, 500), (64,))

    assert isinstance(net, NetShape)


def test_net_shape_rejects_a_module_missing_an_accessor() -> None:
    """Reject a module that answers neither shape question."""
    assert not isinstance(nn.Linear(4, 3), NetShape)


def test_resolve_shape_reads_the_protocol_when_no_override_is_set() -> None:
    """Take both shapes from the module when the config overrides neither."""
    net = _ShapedNet((3, 500), (64,))

    config = NodeConfig("stem", inputs=("features",))

    input_shape, output_shape = resolve_shape(net, config)

    assert input_shape == (3, 500)
    assert output_shape == (64,)


def test_resolve_shape_prefers_the_explicit_overrides() -> None:
    """Take a shape from the config when it is set, and the rest from the module."""
    net = _ShapedNet((3, 500), (64,))
    config = NodeConfig("stem", inputs=("features",), output_shape=(32,))

    input_shape, output_shape = resolve_shape(net, config)

    assert input_shape == (3, 500)
    assert output_shape == (32,)


def test_resolve_shape_accepts_a_plain_module_with_both_overrides() -> None:
    """Resolve a module outside the protocol when the config states both shapes."""
    adapter = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
    config = NodeConfig(
        "pool", inputs=("stem",), input_shape=(16, None), output_shape=(16,)
    )

    shapes = resolve_shape(adapter, config)

    assert shapes == ((16, None), (16,))


def test_resolve_shape_rejects_a_module_without_shapes_naming_the_node() -> None:
    """Raise and name the node when neither the module nor the config answers."""
    config = NodeConfig("pool", inputs=("stem",), input_shape=(16, None))

    with pytest.raises(TypeError, match="pool"):
        resolve_shape(nn.Flatten(), config)


def test_modules_with_running_stats_finds_the_batch_norm_nodes() -> None:
    """Report the nodes holding batch normalization and no others."""
    net = _chain_net(
        {
            "stem": nn.Sequential(nn.Conv1d(1, 4, 3), nn.BatchNorm1d(4)),
            "trunk": nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)),
            "head": nn.Linear(4, 2),
        }
    )

    found = modules_with_running_stats(net, ("stem", "trunk", "head"))

    assert found == frozenset({"stem"})


def test_modules_with_running_stats_inspects_only_the_named_sub_graph() -> None:
    """Leave a batch-normalization node out of the report when it is not named."""
    net = _chain_net(
        {
            "stem": nn.BatchNorm1d(4),
            "trunk": nn.Linear(4, 2),
        }
    )

    assert modules_with_running_stats(net, ("trunk",)) == frozenset()


def test_modules_with_running_stats_reports_untracked_batch_norm() -> None:
    """Report batch normalization conservatively, without tracked statistics."""
    net = _chain_net({"stem": nn.BatchNorm1d(4, track_running_stats=False)})

    assert modules_with_running_stats(net, ("stem",)) == frozenset({"stem"})


def test_modules_with_running_stats_rejects_an_unknown_node_name() -> None:
    """Raise and name the unknown node rather than reporting it as clean."""
    net = _chain_net({"trunk": nn.Linear(4, 2)})

    with pytest.raises(ValueError, match="stem"):
        modules_with_running_stats(net, ("stem",))


def test_batch_norm_in_a_node_fails_under_vmap_in_training_mode() -> None:
    """Pin the ``torch`` premise that ``modules_with_running_stats`` rests on.

    The helper is only worth having while `vmap` rejects a running-statistics
    update, so a failure here is a reason to revisit the premise, not a bug.
    """
    net = _chain_net({"stem": nn.BatchNorm1d(4)})
    module = net.nodes["stem"]
    module.train()
    x = torch.randn(3, 8, 4)

    assert modules_with_running_stats(net, ("stem",)) == frozenset({"stem"})
    with pytest.raises(RuntimeError):
        torch.vmap(module)(x)


# --------------------------------------
# Graph configuration
# --------------------------------------


def _two_stem_nodes() -> tuple[NodeConfig, ...]:
    """Build the node configs of the two-stem generator used across these tests."""
    return (
        NodeConfig("stem_y", inputs=("features",)),
        NodeConfig("stem_z", inputs=("latent",)),
        NodeConfig("trunk", inputs=("stem_y", "stem_z")),
    )


@pytest.mark.parametrize(
    ("block", "expected"), [(0, "h0"), (3, "h3"), ("all", "h_all")]
)
def test_hidden_edge_maps_its_block_to_a_keyword(
    block: int | Literal["all"], expected: str
) -> None:
    """Map a block index to `h{index}` and the every-block case to `h_all`."""
    assert HiddenEdge("stem_z", block).keyword == expected


def test_node_config_lists_positional_sources_before_hidden_ones() -> None:
    """Report both edge kinds as sources, positional first."""
    node = NodeConfig(
        "trunk",
        inputs=("stem_y",),
        hidden_inputs=(HiddenEdge("stem_z", 0), HiddenEdge("bias", "all")),
    )

    assert node.sources == ("stem_y", "stem_z", "bias")


def test_auto_external_inputs_orders_by_first_reference() -> None:
    """Take every source that is not a node, in order of first reference."""
    config = ComposedNetConfig(nodes=_two_stem_nodes())

    assert config.external_inputs == ("features", "latent")


def test_auto_external_inputs_keeps_one_entry_per_name() -> None:
    """Report a name once, at its first reference, however often it is consumed."""
    nodes = (
        NodeConfig("stem_a", inputs=("features",)),
        NodeConfig("stem_b", inputs=("latent",)),
        NodeConfig("stem_c", inputs=("features",)),
        NodeConfig("trunk", inputs=("stem_a", "stem_b", "stem_c", "latent")),
    )

    assert ComposedNetConfig.auto_external_inputs(nodes) == ("features", "latent")


def test_auto_external_inputs_does_not_depend_on_string_hashing() -> None:
    """Pin the order against the hash seed, since it binds `forward` positionally.

    A set difference would dedupe just as well and reorder between processes,
    which would silently swap the arguments at a `g_net(y, z)` call site.
    """
    nodes = tuple(
        NodeConfig(f"stem_{name}", inputs=(name,))
        for name in ("features", "latent", "mask", "time", "weights")
    )
    nodes += (NodeConfig("trunk", inputs=tuple(node.name for node in nodes)),)
    expected = ("features", "latent", "mask", "time", "weights")

    script = (
        "from dlk.nets.compose import ComposedNetConfig, NodeConfig\n"
        f"nodes = {nodes!r}\n"
        "print(ComposedNetConfig.auto_external_inputs(nodes))"
    )
    orders = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).stdout.strip()
        for seed in (1, 2)
    }

    assert ComposedNetConfig.auto_external_inputs(nodes) == expected
    assert orders == {repr(expected)}


def test_auto_external_inputs_includes_hidden_edge_sources() -> None:
    """Count a hidden edge's source as external when no node produces it."""
    nodes = (
        NodeConfig("stem_y", inputs=("features",)),
        NodeConfig(
            "trunk", inputs=("stem_y",), hidden_inputs=(HiddenEdge("latent", 0),)
        ),
    )

    assert ComposedNetConfig.auto_external_inputs(nodes) == ("features", "latent")


def test_external_inputs_can_be_stated_explicitly() -> None:
    """Keep the stated order rather than the order of first reference."""
    config = ComposedNetConfig(
        nodes=_two_stem_nodes(), param_external_inputs=("latent", "features")
    )

    assert config.external_inputs == ("latent", "features")


def test_auto_output_resolves_the_single_sink_and_rejects_two() -> None:
    """Resolve the only sink node, and name the candidates when there are two."""
    config = ComposedNetConfig(nodes=_two_stem_nodes())
    assert config.output == "trunk"

    two_sinks = (
        NodeConfig("stem_y", inputs=("features",)),
        NodeConfig("stem_z", inputs=("latent",)),
    )
    with pytest.raises(ValueError, match="stem_y"):
        ComposedNetConfig(nodes=two_sinks)


def test_config_round_trips_through_from_dict() -> None:
    """Rebuild an identical config from its own kwargs."""
    config = ComposedNetConfig(
        nodes=_two_stem_nodes(), param_external_inputs=("features", "latent")
    )

    assert ComposedNetConfig.from_dict(config.to_kwargs()) == config


def test_config_round_trips_from_plain_dicts_and_pairs() -> None:
    """Build the same config from a graph written without any dlk class."""
    config = ComposedNetConfig(
        nodes=(
            NodeConfig("stem_y", inputs=("features",), output_shape=(64,)),
            NodeConfig(
                "trunk",
                inputs=("stem_y",),
                hidden_inputs=(HiddenEdge("latent", 0),),
                input_shape=(64,),
            ),
        ),
        param_external_inputs=("features", "latent"),
    )

    plain = ComposedNetConfig.from_dict(
        {
            "nodes": [
                {"name": "stem_y", "inputs": ["features"], "output_shape": [64]},
                {
                    "name": "trunk",
                    "inputs": ["stem_y"],
                    "hidden_inputs": [("latent", 0)],
                    "input_shape": [64],
                },
            ],
            "external_inputs": ["features", "latent"],
        }
    )

    assert plain == config
    assert ComposedNetConfig.from_dict(plain.to_kwargs()) == config


def test_from_dict_accepts_hidden_edges_written_as_mappings() -> None:
    """Convert a hidden edge given as a mapping of its fields."""
    config = ComposedNetConfig.from_dict(
        {
            "nodes": [
                {"name": "stem_z", "inputs": ["latent"]},
                {
                    "name": "trunk",
                    "inputs": ["features"],
                    "hidden_inputs": [{"source": "stem_z", "block": "all"}],
                },
            ]
        }
    )

    assert config.nodes[1].hidden_inputs == (HiddenEdge("stem_z", "all"),)
    assert config.output == "trunk"


def test_from_dict_rejects_an_unknown_key() -> None:
    """Raise rather than silently dropping a key that is no constructor argument."""
    with pytest.raises(TypeError):
        ComposedNetConfig.from_dict(
            {"nodes": [{"name": "trunk", "inputs": ["features"]}], "modules": {}}
        )


def test_config_rejects_an_empty_graph() -> None:
    """Raise for a graph with no nodes rather than resolving nothing."""
    with pytest.raises(ValueError, match="at least one node"):
        ComposedNetConfig(nodes=())


def test_config_stores_nodes_as_a_tuple() -> None:
    """Normalize the node container so equality and round trips hold."""
    # a caller passing a list is exactly what the normalization is for
    nodes_as_list: Any = list(_two_stem_nodes())
    config = ComposedNetConfig(nodes=nodes_as_list)

    assert isinstance(config.nodes, tuple)
    assert ComposedNetConfig.from_dict(config.to_kwargs()) == config


def test_config_does_not_store_its_param_inputs() -> None:
    """Keep the constructor-only inputs off the instance, as the InitVars intend."""
    config = ComposedNetConfig(nodes=_two_stem_nodes())
    field_names = {field_.name for field_ in fields(config)}

    # the InitVar defaults live on the class, so the instance is what to check
    assert field_names == {"nodes", "external_inputs", "output"}
    assert "param_external_inputs" not in vars(config)
    assert "param_output" not in vars(config)


def test_to_kwargs_covers_every_stored_field() -> None:
    """Pass every stored field on, which is what makes it a call-site splat."""
    config = ComposedNetConfig(nodes=_two_stem_nodes())

    assert config.to_kwargs() == {
        "nodes": config.nodes,
        "external_inputs": ("features", "latent"),
        "output": "trunk",
    }
    assert set(config.to_kwargs()) == {
        field_.name for field_ in fields(ComposedNetConfig)
    }


def test_config_field_names_are_free_of_the_param_prefix() -> None:
    """Pin that no stored field carries the constructor-only spelling."""
    assert not any(
        field_.name.startswith("param_") for field_ in fields(ComposedNetConfig)
    )
    assert "modules" not in {field_.name for field_ in fields(ComposedNetConfig)}
    assert inspect.signature(ComposedNetConfig).parameters.keys() == {
        "nodes",
        "param_external_inputs",
        "param_output",
    }


def test_config_fields_are_valid_create_composed_net_arguments() -> None:
    """Fail when a stored field stops being a parameter of the target.

    This does not test `create_composed_net`. It catches the drift that would
    otherwise surface as a `TypeError` at a `**config.to_kwargs()` call site.
    """
    parameters = inspect.signature(create_composed_net).parameters
    config_fields = {field_.name for field_ in fields(ComposedNetConfig)}

    assert config_fields <= parameters.keys()


# --------------------------------------
# Composed network
# --------------------------------------


def _mlp_resnet(input_size: int, output_size: int, width: int = 8) -> MLPResNet:
    """Build a small ``MLPResNet`` with one residual block."""
    return MLPResNet(
        input_size,
        output_size,
        residual_blocks_sizes=((width, width, 2 * width, width),),
    )


def _two_stem_generator() -> ComposedNet:
    """Build the section 11 generator at small widths, with an MLP feature stem."""
    return create_composed_net(
        nodes=_two_stem_nodes(),
        external_inputs=("features", "latent"),
        output="trunk",
        modules={
            "stem_y": _mlp_resnet(12, 6),
            "stem_z": _mlp_resnet(4, 4),
            "trunk": _mlp_resnet(10, 3),
        },
    )


def test_two_stem_graph_forward_output_shape() -> None:
    """Run the two-stem generator and validate the output shape."""
    net = _two_stem_generator()
    y = torch.randn(5, 12)
    z = torch.randn(5, 4)

    x_gen = net(y, z)

    assert x_gen.shape == (5, 3)


def test_single_node_graph_matches_the_bare_network() -> None:
    """Return exactly what the wrapped network returns, with no graph effect."""
    torch.manual_seed(0)
    bare = _mlp_resnet(6, 3)
    net = create_composed_net(
        nodes=(NodeConfig("only", inputs=("x",)),),
        external_inputs=("x",),
        output="only",
        modules={"only": bare},
    )
    x = torch.randn(4, 6)

    composed_output = net(x)

    assert torch.equal(composed_output, bare(x))


def test_hidden_edge_reaches_the_configured_block() -> None:
    """Route a hidden edge into one residual block, where it changes the output."""
    torch.manual_seed(0)
    nodes = (
        NodeConfig("stem_z", inputs=("latent",)),
        NodeConfig(
            "trunk", inputs=("features",), hidden_inputs=(HiddenEdge("stem_z", 0),)
        ),
    )
    # the input layer emits width 8, so block 0 takes 8 + the hidden edge's 4
    trunk = MLPResNet((6, 8), 3, residual_blocks_sizes=((12, 12, 24, 8), (8, 8, 16, 8)))
    net = create_composed_net(
        nodes=nodes,
        external_inputs=("features", "latent"),
        output="trunk",
        modules={"stem_z": _mlp_resnet(4, 4), "trunk": trunk},
    )
    y = torch.randn(4, 6)

    first = net(y, torch.randn(4, 4))
    second = net(y, torch.randn(4, 4))

    assert first.shape == (4, 3)
    assert not torch.allclose(first, second)


def test_node_configs_resolve_edge_widths_from_the_shape_protocol() -> None:
    """Reject a trunk built at the wrong width, and accept the derived one."""
    nodes = _two_stem_nodes()
    modules = {"stem_y": _mlp_resnet(12, 6), "stem_z": _mlp_resnet(4, 4)}

    with pytest.raises(ValueError, match="trunk"):
        create_composed_net(
            nodes=nodes,
            external_inputs=("features", "latent"),
            output="trunk",
            modules={**modules, "trunk": _mlp_resnet(9, 3)},
        )

    net = create_composed_net(
        nodes=nodes,
        external_inputs=("features", "latent"),
        output="trunk",
        modules={**modules, "trunk": _mlp_resnet(6 + 4, 3)},
    )

    assert net.resolve_output_shape() == (3,)


def test_edge_from_a_feature_map_source_is_rejected() -> None:
    """Reject a feature-map source, and skip the width check for an unknown width."""
    nodes = (
        NodeConfig("stem", inputs=("features",)),
        NodeConfig("trunk", inputs=("stem",)),
    )
    conv_params = {"channels_mult": [2, 4], "kernels": [3, 3]}

    # a headless ConvResNet returns (batch, channels, length), which is 2D here
    feature_map_stem = ConvResNet(input_channels=1, conv_resnet_params=conv_params)
    with pytest.raises(ValueError, match="trunk"):
        create_composed_net(
            nodes=nodes,
            external_inputs=("features",),
            output="trunk",
            modules={"stem": feature_map_stem, "trunk": _mlp_resnet(16, 3)},
        )

    # a ConvNet without dense layers flattens, so it is 1D of unknown width
    flat_stem = ConvNet(
        input_channels=1,
        hidden_conv_layers_channels_mult=(2, 4),
        hidden_conv_layers_kernels=(3, 3),
    )
    assert flat_stem.resolve_output_shape() == (None,)

    net = create_composed_net(
        nodes=nodes,
        external_inputs=("features",),
        output="trunk",
        modules={"stem": flat_stem, "trunk": _mlp_resnet(48, 3)},
    )

    assert net.resolve_output_shape() == (3,)


@pytest.mark.parametrize(
    ("nodes", "external_inputs", "output", "message"),
    [
        (
            (
                NodeConfig("trunk", inputs=("x",)),
                NodeConfig("trunk", inputs=("x",)),
            ),
            ("x",),
            "trunk",
            "trunk",
        ),
        (
            (NodeConfig("x", inputs=("x",)),),
            ("x",),
            "x",
            "collide",
        ),
        (
            (NodeConfig("trunk", inputs=("typo",)),),
            ("x",),
            "trunk",
            "typo",
        ),
        (
            (
                NodeConfig("a", inputs=("b",)),
                NodeConfig("b", inputs=("a",)),
            ),
            ("x",),
            "a",
            "cycle",
        ),
        (
            (NodeConfig("trunk", inputs=("x",)),),
            ("x",),
            "missing",
            "not a node",
        ),
    ],
    ids=["duplicate", "name-collision", "unknown-input", "cycle", "unknown-output"],
)
def test_graph_validation_rejects_cycles_duplicates_and_unknown_inputs(
    nodes: tuple[NodeConfig, ...],
    external_inputs: tuple[str, ...],
    output: str,
    message: str,
) -> None:
    """Reject each malformed graph, naming what is wrong with it."""
    modules = {node.name: nn.Linear(4, 4) for node in nodes}

    with pytest.raises(ValueError, match=message):
        ComposedNet(
            nodes=nodes,
            external_inputs=external_inputs,
            output=output,
            modules=modules,
            validate_shapes=False,
        )


def test_graph_validation_rejects_a_node_the_output_cannot_reach() -> None:
    """Reject a node wired into nothing, which DDP would fail on much later."""
    nodes = (
        NodeConfig("trunk", inputs=("x",)),
        NodeConfig("orphan", inputs=("x",)),
    )

    with pytest.raises(ValueError, match="orphan"):
        ComposedNet(
            nodes=nodes,
            external_inputs=("x",),
            output="trunk",
            modules={"trunk": nn.Linear(4, 4), "orphan": nn.Linear(4, 4)},
            validate_shapes=False,
        )


def test_graph_validation_rejects_a_module_taking_too_few_inputs() -> None:
    """Reject a single-input module wired to several sources."""
    nodes = (
        NodeConfig("stem_y", inputs=("features",)),
        NodeConfig("stem_z", inputs=("latent",)),
        NodeConfig("trunk", inputs=("stem_y", "stem_z")),
    )
    modules = {
        "stem_y": _mlp_resnet(12, 6),
        "stem_z": _mlp_resnet(4, 4),
        "trunk": ConvNet(input_channels=1),
    }

    with pytest.raises(ValueError, match="trunk"):
        ComposedNet(
            nodes=nodes,
            external_inputs=("features", "latent"),
            output="trunk",
            modules=modules,
            validate_shapes=False,
        )


def test_graph_validation_rejects_a_surplus_or_missing_module() -> None:
    """Reject a module set that does not pair one to one with the nodes."""
    nodes = (NodeConfig("trunk", inputs=("x",)),)

    with pytest.raises(ValueError, match="trunk"):
        ComposedNet(
            nodes=nodes,
            external_inputs=("x",),
            output="trunk",
            modules={},
            validate_shapes=False,
        )

    with pytest.raises(ValueError, match="typo"):
        ComposedNet(
            nodes=nodes,
            external_inputs=("x",),
            output="trunk",
            modules={"trunk": nn.Linear(4, 4), "typo": nn.Linear(4, 4)},
            validate_shapes=False,
        )


def test_topological_order_follows_the_edges_not_the_config_order() -> None:
    """Order the nodes by their edges, whatever order they were configured in."""
    nodes = (
        NodeConfig("trunk", inputs=("stem",)),
        NodeConfig("stem", inputs=("x",)),
    )
    net = ComposedNet(
        nodes=nodes,
        external_inputs=("x",),
        output="trunk",
        modules={"stem": nn.Linear(4, 4), "trunk": nn.Linear(4, 4)},
        validate_shapes=False,
    )

    assert net.topological_order == ("stem", "trunk")


def test_forward_rejects_a_wrong_number_of_inputs() -> None:
    """Fail when the call does not supply one tensor per external input."""
    net = _two_stem_generator()

    with pytest.raises(AssertionError, match="expected 2 inputs"):
        net(torch.randn(5, 12))


def test_composed_net_is_itself_a_net_shape() -> None:
    """Use a composed net as a node inside another composed net."""
    inner = _two_stem_generator()

    assert isinstance(inner, NetShape)
    assert inner.resolve_input_shape() == (12,)
    assert inner.resolve_output_shape() == (3,)

    outer = create_composed_net(
        nodes=(
            NodeConfig("inner", inputs=("features", "latent")),
            NodeConfig("head", inputs=("inner",)),
        ),
        external_inputs=("features", "latent"),
        output="head",
        modules={"inner": inner, "head": _mlp_resnet(3, 2)},
    )
    y = torch.randn(5, 12)
    z = torch.randn(5, 4)

    assert outer(y, z).shape == (5, 2)
    assert outer.resolve_output_shape() == (2,)


def test_state_dict_keys_are_prefixed_by_node_name() -> None:
    """Pin the checkpoint layout, which node names are part of."""
    net = _two_stem_generator()

    keys = list(net.state_dict())

    assert keys
    assert all(
        key.startswith(("nodes.stem_y.", "nodes.stem_z.", "nodes.trunk."))
        for key in keys
    )


def test_parameter_count_is_the_sum_of_its_parts() -> None:
    """Add no parameters of its own, which is what makes composition wiring."""
    stem_y = _mlp_resnet(12, 6)
    stem_z = _mlp_resnet(4, 4)
    trunk = _mlp_resnet(10, 3)
    net = create_composed_net(
        nodes=_two_stem_nodes(),
        external_inputs=("features", "latent"),
        output="trunk",
        modules={"stem_y": stem_y, "stem_z": stem_z, "trunk": trunk},
    )

    composed = sum(parameter.numel() for parameter in net.parameters())
    parts = sum(
        parameter.numel()
        for module in (stem_y, stem_z, trunk)
        for parameter in module.parameters()
    )

    assert composed == parts


def test_config_kwargs_build_the_net_at_the_call_site() -> None:
    """Splat a config into the factory, which is how the two are meant to meet."""
    config = ComposedNetConfig(
        nodes=_two_stem_nodes(), param_external_inputs=("features", "latent")
    )

    net = create_composed_net(
        **config.to_kwargs(),
        modules={
            "stem_y": _mlp_resnet(12, 6),
            "stem_z": _mlp_resnet(4, 4),
            "trunk": _mlp_resnet(10, 3),
        },
    )

    assert net.external_inputs == ("features", "latent")
    assert net.output == "trunk"
    assert net(torch.randn(5, 12), torch.randn(5, 4)).shape == (5, 3)


# --------------------------------------
# Replica partitioning
# --------------------------------------


def test_dependents_of_reports_forward_reachability() -> None:
    """Report every node needing a name, transitively and over both edge kinds."""
    net = _two_stem_generator()

    assert net.dependents_of("latent") == frozenset({"stem_z", "trunk"})
    assert net.dependents_of("features") == frozenset({"stem_y", "trunk"})
    # a node is not a dependent of itself
    assert net.dependents_of("stem_z") == frozenset({"trunk"})
    assert net.dependents_of("trunk") == frozenset()


def test_dependents_of_follows_hidden_edges() -> None:
    """Count a hidden edge as a dependency, like a positional one."""
    net = create_composed_net(
        nodes=(
            NodeConfig("stem_z", inputs=("latent",)),
            NodeConfig(
                "trunk", inputs=("features",), hidden_inputs=(HiddenEdge("stem_z", 0),)
            ),
        ),
        external_inputs=("features", "latent"),
        output="trunk",
        modules={
            "stem_z": _mlp_resnet(4, 4),
            "trunk": MLPResNet((6, 8), 3, residual_blocks_sizes=((12, 12, 24, 8),)),
        },
    )

    assert net.dependents_of("latent") == frozenset({"stem_z", "trunk"})


def test_dependents_of_rejects_an_unknown_name() -> None:
    """Raise rather than reporting an empty set for a name the graph has not got."""
    net = _two_stem_generator()

    with pytest.raises(ValueError, match="typo"):
        net.dependents_of("typo")


def test_forward_partial_evaluates_the_independent_nodes_only() -> None:
    """Return the nodes that do not need the excluded input, and no others."""
    net = _two_stem_generator()

    cache = net.forward_partial(
        exclude_dependents_of="latent", features=torch.randn(5, 12)
    )

    assert set(cache) == {"stem_y"}
    assert cache["stem_y"].shape == (5, 6)


def test_forward_partial_excludes_the_named_node_itself() -> None:
    """Leave out the named node too, since its output is what will vary."""
    net = _two_stem_generator()

    cache = net.forward_partial(
        exclude_dependents_of="stem_z", features=torch.randn(5, 12)
    )

    assert set(cache) == {"stem_y"}


def test_forward_partial_rejects_a_missing_external_input() -> None:
    """Name the external input an evaluated node needs and did not get."""
    net = _two_stem_generator()

    with pytest.raises(ValueError, match="features"):
        net.forward_partial(exclude_dependents_of="latent")


def test_forward_partial_rejects_an_unknown_keyword() -> None:
    """Name a keyword that is no external input rather than ignoring it."""
    net = _two_stem_generator()

    with pytest.raises(ValueError, match="feature"):
        net.forward_partial(exclude_dependents_of="latent", feature=torch.randn(5, 12))


def test_partial_forward_matches_the_full_forward() -> None:
    """Compare the vmapped replica pattern against evaluating each replica.

    A wrong cache, a wrong topological order or a stale edge shows up here and
    nowhere else.
    """
    torch.manual_seed(0)
    net = _two_stem_generator()
    net.eval()
    replicas = 8
    y = torch.randn(5, 12)
    z = torch.randn(replicas, 5, 4)

    cache = net.forward_partial(exclude_dependents_of="latent", features=y)
    vmapped = torch.vmap(lambda z_: net(y, z_, cache=cache))(z)
    stacked = torch.stack([net(y, z[index]) for index in range(replicas)])

    assert vmapped.shape == (replicas, 5, 3)
    assert torch.allclose(vmapped, stacked, atol=1e-6)


def test_partial_forward_evaluates_latent_independent_nodes_once() -> None:
    """Keep a batch-normalized stem out of the vmapped region, and run it once.

    The stem is left in training mode, which is where its running statistics
    would be updated and `vmap` would reject the write.
    """
    torch.manual_seed(0)
    batch_size = 4
    replicas = 8
    net = create_composed_net(
        nodes=_two_stem_nodes(),
        external_inputs=("features", "latent"),
        output="trunk",
        modules={
            "stem_y": EfficientNetV1B0Minimal(
                input_channels=3, input_length=32, num_classes=6
            ),
            "stem_z": _mlp_resnet(4, 4),
            "trunk": _mlp_resnet(10, 3),
        },
    )
    net.train()

    # the region about to be vmapped holds no running statistics
    latent_region = net.dependents_of("latent")
    assert latent_region == frozenset({"stem_z", "trunk"})
    assert modules_with_running_stats(net, latent_region) == frozenset()
    assert modules_with_running_stats(net, ("stem_y",)) == frozenset({"stem_y"})

    calls: list[int] = []
    handle = net.nodes["stem_y"].register_forward_hook(
        lambda module, inputs, output: calls.append(1)
    )
    y = torch.randn(batch_size, 3, 32)
    z = torch.randn(replicas, batch_size, 4)

    cache = net.forward_partial(exclude_dependents_of="latent", features=y)
    outputs = torch.vmap(lambda z_: net(y, z_, cache=cache))(z)
    handle.remove()

    assert calls == [1]
    assert outputs.shape == (replicas, batch_size, 3)
