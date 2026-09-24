"""Compose already-built networks into a directed acyclic graph.

Composition here is wiring and nothing else: a node is a network the caller has
already built, the graph adds no layers, no parameters and no initialization
policy of its own, and every tensor crossing an edge is a value one node
returned and another node consumes unchanged.

This module starts with the shape interface the wiring is written against.
Edge widths are derived by asking each node what it takes and what it returns,
never by running a probe batch.

Specs:
- docs/features/2026.006__compose_nets__1-plan.md
"""

import inspect
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

# --------------------------------------
# Types
# --------------------------------------

# A per-sample shape: the shape of one sample with the batch dimension
# excluded, one entry per remaining dimension, and `None` for a dimension the
# network accepts at any extent.
Shape = tuple[int | None, ...]


@runtime_checkable
class NetShape(Protocol):
    """Define the shape interface a network satisfies to become a graph node.

    Both methods answer with a per-sample shape, so `MLPResNet(32, 3, ...)`
    answers `(32,)` and `(3,)`. EfficientNet taking three channels of length 500
    and returning 64 classes answers `(3, 500)` and `(64,)`; and when keeping
    the length unset (i.e., arbitrary) it answers `(3, None)` for the input
    shape.

    The ndim of an answer is the ndim the network's own `forward` assertions are
    written against, minus the batch dimension. A network documenting a flat
    input therefore answers with ndim 1 even when its `forward` also accepts a
    `(batch_size, 1, size)` tensor that it flattens, and a network with an
    explicit channel argument answers with ndim 2.

    The element count per sample is the product of the entries, and is unknown
    when any entry is `None`.
    """

    def resolve_input_shape(self) -> Shape:
        """Return the per-sample shape of the input this network accepts."""
        ...

    def resolve_output_shape(self) -> Shape:
        """Return the per-sample shape of the output this network returns."""
        ...


# --------------------------------------
# Graph configuration
# --------------------------------------


class HiddenEdge(NamedTuple):
    """Pair a source with the residual block its tensor is injected into.

    `MLPResNet` and `ConvResNet` accept tensors named `h{index}` or `h_all` and
    route them into one residual block, or into every block, rather than into
    the input layer. A hidden edge is what wires a node's output there.

    Attributes:
        source: Name of the node or external input producing the tensor.
        block: Index of the residual block receiving the tensor, or `"all"` for
            every block.
    """

    source: str
    block: int | Literal["all"]

    @property
    def keyword(self) -> str:
        """Return the keyword argument this edge is passed as.

        Returns:
            `"h_all"` when the edge feeds every block, `"h{index}"` otherwise.
        """
        return "h_all" if self.block == "all" else f"h{self.block}"


@dataclass(frozen=True)
class NodeConfig:
    """Describe one node of a composed graph, its name and where it reads from.

    A node pairs a name with the names its inputs come from. The module itself
    is not configured here; it is passed at the call site, keyed by this name.

    Attributes:
        name: Name of the node. It keys the module mapping and prefixes the
            node's parameters in the composed net's `state_dict`.
        inputs: Names the positional inputs come from, each a node name or an
            external input name, in the order the module is called with.
        hidden_inputs: Edges delivered as keyword arguments to residual blocks
            instead of positionally.
        input_shape: Per-sample input shape, or `None` to ask the module
            through `NetShape`.
        output_shape: Per-sample output shape, or `None` to ask the module
            through `NetShape`.
    """

    name: str
    inputs: tuple[str, ...]
    hidden_inputs: tuple[HiddenEdge, ...] = ()
    input_shape: Shape | None = None
    output_shape: Shape | None = None

    @property
    def sources(self) -> tuple[str, ...]:
        """Return every name this node depends on, positional before hidden.

        Returns:
            The sources of both edge kinds, in the order they are referenced.
        """
        return self.inputs + tuple(edge.source for edge in self.hidden_inputs)


def _as_shape(shape: Sequence[int | None] | None) -> Shape | None:
    """Convert a shape given as any sequence into a tuple.

    Args:
        shape: Shape as a sequence, or `None` to ask the module.

    Returns:
        The shape as a tuple, or `None`.
    """
    return None if shape is None else tuple(shape)


def _as_hidden_edge(edge: HiddenEdge | Mapping[str, Any] | Sequence[Any]) -> HiddenEdge:
    """Convert a hidden edge given as a mapping or a pair into a `HiddenEdge`.

    Args:
        edge: Edge as a `HiddenEdge`, a mapping of its fields, or a
            `(source, block)` pair.

    Returns:
        The edge as a `HiddenEdge`, unchanged when it already is one.
    """
    if isinstance(edge, HiddenEdge):
        return edge
    if isinstance(edge, Mapping):
        return HiddenEdge(**edge)
    source, block = edge
    return HiddenEdge(source, block)


def _as_node_config(node: NodeConfig | Mapping[str, Any]) -> NodeConfig:
    """Convert a node given as a mapping into a `NodeConfig`.

    Args:
        node: Node as a `NodeConfig` or as a mapping of its fields.

    Returns:
        The node as a `NodeConfig`, unchanged when it already is one.

    Raises:
        TypeError: If the mapping holds a key that is not a field, or omits a
            required one.
    """
    if isinstance(node, NodeConfig):
        return node
    node_fields = dict(node)
    node_fields["inputs"] = tuple(node_fields.get("inputs", ()))
    node_fields["hidden_inputs"] = tuple(
        _as_hidden_edge(edge) for edge in node_fields.get("hidden_inputs", ())
    )
    node_fields["input_shape"] = _as_shape(node_fields.get("input_shape"))
    node_fields["output_shape"] = _as_shape(node_fields.get("output_shape"))
    return NodeConfig(**node_fields)


@dataclass
class ComposedNetConfig:
    """Typed argument set for `create_composed_net`.

    `external_inputs` and `output` each default to an automatic heuristic
    derived from `nodes` when their configured value is `None`. The modules are
    not configured here; they are passed at the call site.

    Attributes:
        nodes: Configurations of the graph's nodes.
        param_external_inputs: Constructor-only input; `None` to derive
            `external_inputs` from `nodes` (not stored on the instance).
        param_output: Constructor-only input; `None` to derive `output` from
            `nodes` (not stored on the instance).
        external_inputs: Names of the tensors the caller supplies, in the
            order `ComposedNet.forward` binds its positional arguments.
        output: Name of the node whose output the graph returns.
    """

    nodes: tuple[NodeConfig, ...]
    param_external_inputs: InitVar[Sequence[str] | None] = None
    param_output: InitVar[str | None] = None
    external_inputs: tuple[str, ...] = field(init=False)
    output: str = field(init=False)

    def __post_init__(
        self,
        param_external_inputs: Sequence[str] | None,
        param_output: str | None,
    ) -> None:
        self.nodes = tuple(self.nodes)
        if not self.nodes:
            raise ValueError("expected at least one node, got none.")

        self.external_inputs = (
            tuple(param_external_inputs)
            if param_external_inputs is not None
            else self.auto_external_inputs(self.nodes)
        )
        self.output = (
            param_output if param_output is not None else self.auto_output(self.nodes)
        )

    @staticmethod
    def auto_external_inputs(nodes: Sequence[NodeConfig]) -> tuple[str, ...]:
        """Heuristic external inputs: every source that is not a node.

        Sources are visited node by node in the order given, positional inputs
        before hidden ones, and a name is appended on its first reference. The
        result is therefore deterministic, and it is the order `forward` binds
        positionally, so a caller who cares about that order states it instead.

        Args:
            nodes: Configurations of the graph's nodes.

        Returns:
            The external input names, in order of first reference.
        """
        node_names = {node.name for node in nodes}
        # dict.fromkeys dedupes by hash and keeps first-reference order; a set
        # difference would dedupe just as well and lose that order, which is
        # the positional binding order of `ComposedNet.forward`
        return tuple(
            dict.fromkeys(
                source
                for node in nodes
                for source in node.sources
                if source not in node_names
            )
        )

    @staticmethod
    def auto_output(nodes: Sequence[NodeConfig]) -> str:
        """Heuristic output node: the single node no other node consumes.

        Args:
            nodes: Configurations of the graph's nodes.

        Returns:
            The name of the only sink node.

        Raises:
            ValueError: If the graph has no sink node, or more than one.
        """
        consumed = {source for node in nodes for source in node.sources}
        sinks = [node.name for node in nodes if node.name not in consumed]
        if len(sinks) != 1:
            raise ValueError(
                f"expected exactly one sink node to use as the output, got {sinks}; "
                "name the output node explicitly."
            )
        return sinks[0]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ComposedNetConfig":
        """Build a `ComposedNetConfig` from a dict of constructor arguments.

        `external_inputs` and `output` map to their `param_*` constructor
        inputs; every other key must match a field name directly. Beyond that
        alias map, an entry of `nodes` given as a mapping is converted to
        `NodeConfig`, an entry of its `hidden_inputs` given as a mapping or a
        pair to `HiddenEdge`, and a shape given as a list to a tuple, so a graph
        can be written in YAML or JSON. Entries that already are `NodeConfig` or
        `HiddenEdge` pass through unchanged.

        Args:
            config: Mapping of constructor argument names to values.

        Returns:
            A configured `ComposedNetConfig` instance.

        Raises:
            TypeError: If `config` contains a key that is not a constructor
                argument, or omits a required one.
        """
        key_aliases = {
            "external_inputs": "param_external_inputs",
            "output": "param_output",
        }
        kwargs = {key_aliases.get(key, key): value for key, value in config.items()}
        if "nodes" in kwargs:
            kwargs["nodes"] = tuple(_as_node_config(node) for node in kwargs["nodes"])
        return cls(**kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        """Return this config as kwargs for `create_composed_net`."""
        return {field_.name: getattr(self, field_.name) for field_ in fields(self)}


# --------------------------------------


def resolve_shape(module: nn.Module, config: NodeConfig) -> tuple[Shape, Shape]:
    """Resolve the per-sample input and output shapes of one node.

    Each shape is taken from the explicit field of the node's configuration when
    that field is set, and from the `NetShape` protocol otherwise. The explicit
    fields are the escape hatch for a module that does not satisfy the protocol,
    such as a pooling adapter or a module from outside `dlk`.

    Args:
        module: Network of the node.
        config: Configuration of the node, carrying the optional overrides.

    Returns:
        The input shape and the output shape of the node, in that order.

    Raises:
        TypeError: If a shape is neither overridden by the configuration nor
            answered by the module.
    """
    input_shape = config.input_shape
    output_shape = config.output_shape
    if input_shape is not None and output_shape is not None:
        return input_shape, output_shape

    if not isinstance(module, NetShape):
        raise TypeError(
            f"cannot resolve the shapes of node '{config.name}': its module "
            f"'{type(module).__name__}' does not implement NetShape, and its "
            "config sets no explicit input_shape and output_shape"
        )
    if input_shape is None:
        input_shape = module.resolve_input_shape()
    if output_shape is None:
        output_shape = module.resolve_output_shape()
    return input_shape, output_shape


def modules_with_running_stats(
    net: "ComposedNet", names: Iterable[str]
) -> frozenset[str]:
    """Find the nodes of a sub-graph that carry batch-normalization modules.

    A module that updates running statistics writes into an unbatched buffer,
    which `torch.vmap` rejects in training mode. Asserting that this function
    returns an empty set over the region about to be vmapped catches that at
    setup time rather than at the first step.

    The report is conservative: a batch-normalization module constructed with
    `track_running_stats=False` holds no running state, and is reported anyway.

    Args:
        net: Composed network owning the nodes.
        names: Names of the nodes forming the sub-graph to inspect.

    Returns:
        The names among `names` whose module contains a batch-normalization
        module.

    Raises:
        ValueError: If a name is not a node of `net`.
    """
    node_names = tuple(names)
    unknown = [name for name in node_names if name not in net.nodes]
    if unknown:
        raise ValueError(f"unknown node names: {unknown}, expected {list(net.nodes)}")
    return frozenset(
        name
        for name in node_names
        if any(
            isinstance(submodule, _BatchNorm) for submodule in net.nodes[name].modules()
        )
    )


def _positional_capacity(module: nn.Module) -> int | None:
    """Count the positional inputs a module's `forward` accepts.

    Args:
        module: Network whose `forward` is inspected.

    Returns:
        The number of positional parameters, or `None` when the count is
        unbounded because `forward` takes a variadic positional parameter or
        cannot be inspected.
    """
    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        # an uninspectable forward is treated as unbounded rather than rejected
        return None
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    capacity = 0
    for parameter in parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if parameter.kind in positional_kinds:
            capacity += 1
    return capacity


def _element_count(shape: Shape) -> int | None:
    """Count the elements of one sample.

    Args:
        shape: Per-sample shape.

    Returns:
        The product of the entries, or `None` when any entry is unknown.
    """
    if any(entry is None for entry in shape):
        return None
    return math.prod(entry for entry in shape if entry is not None)


# --------------------------------------
# Composed network
# --------------------------------------


class ComposedNet(nn.Module):
    """Wire already-built networks into a directed acyclic graph.

    The graph adds no layers, no parameters and no initialization policy of its
    own. Modules are held in an `nn.ModuleDict` named `nodes`, so parameter
    names, `state_dict` keys and `get_parameters` output all read as
    `nodes.<node name>.<the network's own keys>`, and any node can be lifted out
    and used alone.

    `ComposedNet` itself satisfies `NetShape`, so a composed net can be a node
    inside another composed net.

    Attributes:
        node_configs: Configurations of the graph's nodes, in the order given.
        external_inputs: Names of the tensors the caller supplies, in the order
            `forward` binds its positional arguments.
        output: Name of the node whose output `forward` returns.
        topological_order: Node names ordered so every node follows its inputs.
        nodes: Modules of the graph, keyed by node name.
    """

    def __init__(
        self,
        nodes: Sequence[NodeConfig],
        external_inputs: Sequence[str],
        output: str,
        modules: Mapping[str, nn.Module],
        validate_shapes: bool = True,
    ) -> None:
        """Initialize ComposedNet and validate the graph it describes.

        Args:
            nodes: Configurations of the graph's nodes.
            external_inputs: Names of the tensors the caller supplies, in the
                order `forward` binds its positional arguments.
            output: Name of the node whose output `forward` returns.
            modules: One already-built module per node, keyed by node name.
            validate_shapes: Whether to check edge shapes at construction.

        Raises:
            ValueError: If the graph is malformed, naming the offending node.
                The graph is rejected for a duplicate node name, a node name
                colliding with an external input, a missing or surplus module,
                an input naming neither a node nor an external input, a node
                whose module takes fewer positional inputs than it declares, a
                cycle, an output that is not a node, a node the output cannot
                reach, and, under `validate_shapes`, an edge that is not flat
                or an input width that does not match its sources.
        """
        super().__init__()
        self.node_configs = tuple(nodes)
        self.external_inputs = tuple(external_inputs)
        self.output = output
        self._config_by_name = {node.name: node for node in self.node_configs}

        node_names = [node.name for node in self.node_configs]
        self._validate_names(node_names, modules)
        self._validate_edges(modules)
        self.topological_order = self._resolve_topological_order(node_names)
        self._validate_output(node_names)

        self._consumers: dict[str, list[str]] = {}
        for node in self.node_configs:
            for source in node.sources:
                self._consumers.setdefault(source, []).append(node.name)
        self._dependents_cache: dict[str, frozenset[str]] = {}

        self.nodes = nn.ModuleDict({name: modules[name] for name in node_names})
        if validate_shapes:
            self._validate_shapes()

    def _validate_names(
        self, node_names: Sequence[str], modules: Mapping[str, nn.Module]
    ) -> None:
        """Check that node names are unique, distinct and backed by a module.

        Args:
            node_names: Names of the graph's nodes, in the order given.
            modules: One already-built module per node, keyed by node name.

        Raises:
            ValueError: If a name is duplicated, collides with an external
                input, or does not pair with exactly one module.
        """
        duplicates = sorted(
            name for name, count in Counter(node_names).items() if 1 < count
        )
        if duplicates:
            raise ValueError(f"duplicate node names: {duplicates}")

        collisions = sorted(set(node_names) & set(self.external_inputs))
        if collisions:
            raise ValueError(
                f"node names collide with external input names: {collisions}"
            )

        missing = [name for name in node_names if name not in modules]
        if missing:
            raise ValueError(f"no module given for nodes: {missing}")
        surplus = sorted(set(modules) - set(node_names))
        if surplus:
            # a surplus module is a misspelled node name more often than not
            raise ValueError(f"modules given for names that are not nodes: {surplus}")

    def _validate_edges(self, modules: Mapping[str, nn.Module]) -> None:
        """Check that every edge names a known source its consumer can take.

        Args:
            modules: One already-built module per node, keyed by node name.

        Raises:
            ValueError: If a source names neither a node nor an external input,
                or a node declares more positional inputs than its module takes.
        """
        known = set(self._config_by_name) | set(self.external_inputs)
        for node in self.node_configs:
            unknown = [source for source in node.sources if source not in known]
            if unknown:
                raise ValueError(
                    f"node '{node.name}' reads from {unknown}, which name neither "
                    "a node nor an external input"
                )

            capacity = _positional_capacity(modules[node.name])
            if capacity is not None and capacity < len(node.inputs):
                raise ValueError(
                    f"node '{node.name}' declares {len(node.inputs)} inputs, but its "
                    f"module '{type(modules[node.name]).__name__}' takes {capacity} "
                    "positional inputs"
                )

    def _resolve_topological_order(self, node_names: Sequence[str]) -> tuple[str, ...]:
        """Order the nodes so every node follows all of its inputs.

        Uses Kahn's algorithm over both edge kinds, taking ready nodes in the
        order they were configured so the result is deterministic.

        Args:
            node_names: Names of the graph's nodes, in the order given.

        Returns:
            The node names in topological order.

        Raises:
            ValueError: If the graph has a cycle, naming the nodes on it.
        """
        dependents: dict[str, list[str]] = {name: [] for name in node_names}
        unmet: dict[str, int] = {}
        for node in self.node_configs:
            dependencies = {
                source for source in node.sources if source in self._config_by_name
            }
            unmet[node.name] = len(dependencies)
            for dependency in dependencies:
                dependents[dependency].append(node.name)

        ready = [name for name in node_names if unmet[name] == 0]
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for dependent in dependents[name]:
                unmet[dependent] -= 1
                if unmet[dependent] == 0:
                    ready.append(dependent)

        if len(order) != len(node_names):
            cyclic = sorted(set(node_names) - set(order))
            raise ValueError(f"the graph has a cycle through nodes: {cyclic}")
        return tuple(order)

    def _validate_output(self, node_names: Sequence[str]) -> None:
        """Check that the output is a node that reaches every other node.

        A node wired into nothing has to fail here rather than at the first
        distributed step, where DDP raises for a parameter that got no gradient.

        Args:
            node_names: Names of the graph's nodes, in the order given.

        Raises:
            ValueError: If the output is not a node, or a node cannot reach it.
        """
        if self.output not in self._config_by_name:
            raise ValueError(
                f"the output '{self.output}' is not a node, expected one of "
                f"{list(node_names)}"
            )

        reached: set[str] = set()
        pending = [self.output]
        while pending:
            name = pending.pop()
            if name in reached:
                continue
            reached.add(name)
            config = self._config_by_name.get(name)
            if config is not None:
                pending.extend(config.sources)

        unreachable = [name for name in node_names if name not in reached]
        if unreachable:
            raise ValueError(
                f"the output '{self.output}' does not reach nodes {unreachable}; "
                "their parameters would receive no gradient"
            )

    def _output_shape_of(self, name: str) -> Shape:
        """Resolve one node's output shape.

        Args:
            name: Name of the node.

        Returns:
            The per-sample output shape of that node.
        """
        return resolve_shape(self.nodes[name], self._config_by_name[name])[1]

    def _validate_shapes(self) -> None:
        """Check the edges between nodes, which is where widths are derivable.

        Two rules are enforced, and this is the only place either is. Every
        source of an internal edge answers with a flat shape, so a source that
        emits a feature map is named and rejected here. And the positional
        sources' widths sum to the consuming node's element count; hidden edges
        are left out of that sum because they enter a residual block rather than
        the input layer. Either check is skipped where a `None` entry makes the
        arithmetic unknown.

        Edges from external inputs are not checked, because external shapes are
        names only; a mismatch there surfaces from the consuming network's own
        `forward` assertion on the first batch.

        Raises:
            ValueError: If an edge is not flat, or a width does not match.
        """
        for node in self.node_configs:
            if not node.sources or any(
                source not in self._config_by_name for source in node.sources
            ):
                continue

            for source in node.sources:
                source_shape = self._output_shape_of(source)
                if len(source_shape) != 1:
                    raise ValueError(
                        f"node '{node.name}' reads from node '{source}', whose output "
                        f"shape {source_shape} is not flat; an edge carries a tensor "
                        "of shape (batch, size)"
                    )

            if not node.inputs:
                continue
            widths = [self._output_shape_of(source)[0] for source in node.inputs]
            expected = _element_count(resolve_shape(self.nodes[node.name], node)[0])
            if expected is None or any(width is None for width in widths):
                continue
            total = sum(width for width in widths if width is not None)
            if total != expected:
                raise ValueError(
                    f"node '{node.name}' takes {expected} elements per sample, but "
                    f"its inputs {list(node.inputs)} supply {widths}, summing to "
                    f"{total}"
                )

    def forward(
        self,
        *inputs: torch.Tensor,
        cache: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Evaluate the graph and return the output node's tensor.

        Args:
            *inputs: One tensor per external input, in the declared order.
            cache: Node outputs to reuse instead of recomputing, as returned by
                `forward_partial`. It is valid for one forward and its backward
                pass, since it holds activations of one autograd graph.

        Returns:
            The output node's tensor.

        Raises:
            ValueError: If a node is neither cached nor computable from what
                was supplied.
        """
        assert len(inputs) == len(self.external_inputs), (
            f"expected {len(self.external_inputs)} inputs "
            f"{list(self.external_inputs)}, got {len(inputs)}"
        )

        values: dict[str, torch.Tensor] = dict(zip(self.external_inputs, inputs))
        if cache is not None:
            values.update(cache)

        for name in self.topological_order:
            if name in values:
                continue
            config = self._config_by_name[name]
            missing = [source for source in config.sources if source not in values]
            if missing:
                raise ValueError(
                    f"node '{name}' needs {missing}, which is neither supplied nor "
                    "computed"
                )
            positional = [values[source] for source in config.inputs]
            hidden = {
                edge.keyword: values[edge.source] for edge in config.hidden_inputs
            }
            values[name] = self.nodes[name](*positional, **hidden)

        return values[self.output]

    def dependents_of(self, name: str) -> frozenset[str]:
        """Find every node that transitively depends on a name.

        Reachability runs forward over both edge kinds. The name itself is not
        a dependent of itself, so a node name is absent from its own set; what
        `forward_partial` excludes is this set together with that node.

        The result is cached per name, since the graph is fixed at construction.

        Args:
            name: Name of an external input or of a node.

        Returns:
            The names of the nodes that need `name`, directly or indirectly.

        Raises:
            ValueError: If `name` is neither a node nor an external input.
        """
        cached = self._dependents_cache.get(name)
        if cached is not None:
            return cached
        if name not in self._config_by_name and name not in self.external_inputs:
            raise ValueError(
                f"'{name}' is neither a node nor an external input, expected one of "
                f"{list(self._config_by_name) + list(self.external_inputs)}"
            )

        dependents: set[str] = set()
        pending = list(self._consumers.get(name, ()))
        while pending:
            current = pending.pop()
            if current in dependents:
                continue
            dependents.add(current)
            pending.extend(self._consumers.get(current, ()))

        self._dependents_cache[name] = frozenset(dependents)
        return self._dependents_cache[name]

    def forward_partial(
        self, *, exclude_dependents_of: str, **externals: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Evaluate the part of the graph that does not need one input.

        The returned cache is what makes the replica pattern both correct and
        cheap: the latent-independent nodes are evaluated once, outside the
        vmapped region, and passed back into `forward` as unbatched constants.

            cache = net.forward_partial(exclude_dependents_of="latent", features=y)
            outputs = torch.vmap(lambda z_: net(y, z_, cache=cache))(z)

        Two limits are not enforceable in code. The cache holds activations of
        one autograd graph, so it is valid for one forward and its backward
        pass; reusing it across optimizer steps backpropagates through a freed
        graph. And the split only avoids what the graph says is independent, so
        a module with running statistics left inside the latent-dependent
        region still fails under `vmap`; assert on `modules_with_running_stats`
        over that region to catch it at setup rather than at the first step.

        Args:
            exclude_dependents_of: Name of the external input or node whose
                dependents are left out, along with that node itself.
            **externals: One tensor per external input the evaluated nodes
                need, by name rather than by position, because the excluded
                input is the one the caller has no single value for yet.

        Returns:
            The computed node outputs, keyed by node name.

        Raises:
            ValueError: If a keyword names no external input, or an external
                input an evaluated node needs was not supplied.
        """
        unknown = sorted(set(externals) - set(self.external_inputs))
        if unknown:
            raise ValueError(
                f"{unknown} name no external input, expected any of "
                f"{list(self.external_inputs)}"
            )

        excluded = set(self.dependents_of(exclude_dependents_of))
        if exclude_dependents_of in self._config_by_name:
            excluded.add(exclude_dependents_of)

        values: dict[str, torch.Tensor] = dict(externals)
        cache: dict[str, torch.Tensor] = {}
        for name in self.topological_order:
            if name in excluded:
                continue
            config = self._config_by_name[name]
            missing = [source for source in config.sources if source not in values]
            if missing:
                raise ValueError(
                    f"node '{name}' needs {missing}, which was not supplied as a "
                    "keyword argument"
                )
            positional = [values[source] for source in config.inputs]
            hidden = {
                edge.keyword: values[edge.source] for edge in config.hidden_inputs
            }
            values[name] = self.nodes[name](*positional, **hidden)
            cache[name] = values[name]
        return cache

    def resolve_input_shape(self) -> Shape:
        """Return the shape of one input sample, excluding the batch dimension.

        The answer is the input shape of the node consuming the first external
        input, which is the tensor `forward` binds first.

        Returns:
            The per-sample input shape of that node.

        Raises:
            ValueError: If the graph declares no external input, and so takes
                nothing a consumer could name.
        """
        if not self.external_inputs:
            raise ValueError("the graph declares no external input")
        first = self.external_inputs[0]
        for name in self.topological_order:
            config = self._config_by_name[name]
            if first in config.sources:
                return resolve_shape(self.nodes[name], config)[0]
        raise ValueError(f"no node consumes the external input '{first}'")

    def resolve_output_shape(self) -> Shape:
        """Return the shape of one output sample, excluding the batch dimension.

        Returns:
            The per-sample output shape of the output node.
        """
        return self._output_shape_of(self.output)


def create_composed_net(
    nodes: Sequence[NodeConfig],
    external_inputs: Sequence[str],
    output: str,
    modules: Mapping[str, nn.Module],
    *,
    validate_shapes: bool = True,
    logger: logging.Logger | None = None,
) -> ComposedNet:
    """Create a composed network from a graph description and its modules.

    Args:
        nodes: Configurations of the graph's nodes.
        external_inputs: Names of the tensors the caller supplies, in the order
            `ComposedNet.forward` binds its positional arguments.
        output: Name of the node whose output the graph returns.
        modules: One already-built module per node, keyed by node name.
        validate_shapes: Whether to check edge shapes at construction.
        logger: Logger for the per-node summary, or `None` for the module's own.

    Returns:
        The validated composed network.
    """
    if logger is None:
        logger = logging.getLogger("dlk.nets.compose.create_composed_net")

    net = ComposedNet(
        nodes=nodes,
        external_inputs=external_inputs,
        output=output,
        modules=modules,
        validate_shapes=validate_shapes,
    )

    logger.info(
        f"composed net over {len(net.node_configs)} nodes, "
        f"inputs {list(net.external_inputs)}, output '{net.output}'"
    )
    config_by_name = {node.name: node for node in net.node_configs}
    for name in net.topological_order:
        config = config_by_name[name]
        input_shape, output_shape = resolve_shape(net.nodes[name], config)
        logger.info(
            f"  node '{name}': {type(net.nodes[name]).__name__}, "
            f"in {input_shape} from {list(config.sources)}, out {output_shape}"
        )
    return net
