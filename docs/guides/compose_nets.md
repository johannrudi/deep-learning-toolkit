---
Title: Composing Existing Networks into a Directed Acyclic Graph
Author: Johann Rudi
Co-Authored-By: Claude Sonnet 5
Date: 2026-09-22
tags:
  - composition
  - architecture
  - nets
link: "[[2026.006__compose_nets__1-plan]]"
---

# Composing Existing Networks into a Directed Acyclic Graph

`dlk.nets.compose` wires already-built `dlk` networks into a **composed net**: a directed acyclic graph of named nodes, where each node is one network and each edge is a tensor one node returns and another consumes unchanged. Building a stem-and-trunk architecture by hand means writing a bespoke `nn.Module` and recomputing every concatenation width whenever a stem changes size. `ComposedNet` derives those widths instead, from the networks themselves, and checks them once at construction rather than on the first batch.

This guide assumes you already build `dlk` networks from `dlk/nets/` (e.g., `dlk/nets/mlp.py` or `dlk/nets/efficientnet1d.py`) and that you know what a directed acyclic graph is. It walks through declaring a graph, building it, and using it as a drop-in for an existing training loop, then covers the two things that are not obvious from the interface: [routing a tensor into a residual block](#routing-a-tensor-into-a-residual-block) instead of concatenating it, and [splitting a graph so a batch-normalized stem is evaluated once per batch](#splitting-off-vmap-safe-replicas) instead of once per `torch.vmap` replica.

> [!NOTE]
> Every node's module must answer `resolve_input_shape()` and `resolve_output_shape()`, the two-method `NetShape` protocol, or the node's `NodeConfig` must set `input_shape` and `output_shape` explicitly. `MLPNet`, `MLPResNet`, `ConvNet`, `ConvResNet`, `TransformerNet`, `ChannelWiseTransformerNet`, and every `ScalableEfficientNet1D` subclass already implement it. A module that does neither raises `TypeError` naming the node at construction, not at the first forward call.

## Declaring the graph

### Step 1: Name the nodes and where their inputs come from

A `NodeConfig` pairs a name with the names its inputs come from, each either another node's name or an external input, a tensor the caller supplies. Build the motivating case, a convolutional feature stem and a latent stem feeding a residual trunk:

```python
from dlk.nets.compose import NodeConfig

nodes = (
    NodeConfig("stem_y", inputs=("features",)),
    NodeConfig("stem_z", inputs=("latent",)),
    NodeConfig("trunk", inputs=("stem_y", "stem_z")),
)
```

`stem_y` and `stem_z` each read one external input; `trunk` reads both stems' outputs. This snippet does not build a module or define widths, because `NodeConfig` is a wiring description.

### Step 2: Let the external inputs and the output resolve automatically

`ComposedNetConfig` wraps the nodes and derives two things you would otherwise have to restate: `external_inputs`, every source name that is not a node, in the order first referenced; and `output`, the single node no other node consumes.

```python
from dlk.nets.compose import ComposedNetConfig

config = ComposedNetConfig(nodes=nodes)
print(config.external_inputs, config.output)
```

```text
('features', 'latent') trunk
```

That order matters beyond cosmetics: it is the positional order `ComposedNet.forward` binds its arguments in, so `net(y, z)` and `net(z, y)` are not interchangeable once the graph resolves it for you. State it explicitly with `param_external_inputs=("features", "latent")` whenever the call site is fixed elsewhere.

## Building and validating the composed network

### Step 3: Build the modules and call `create_composed_net`

The modules themselves are not part of the config, so build them separately. Key the modules by the node name in the config, and hand them to `create_composed_net` alongside the config:

```python
from dlk.nets.compose import create_composed_net
from dlk.nets.efficientnet1d import EfficientNetV1B0Minimal
from dlk.nets.mlp import MLPResNet

g_net = create_composed_net(
    **config.to_kwargs(),
    modules={
        "stem_y": EfficientNetV1B0Minimal(input_channels=3, input_length=32, num_classes=8),
        "stem_z": MLPResNet(4, 4, residual_blocks_sizes=((4, 4, 8, 4),)),
        "trunk": MLPResNet(12, 3, residual_blocks_sizes=((16, 16, 32, 16), (16, 16, 32, 16))),
    },
)
```

Above, `trunk` takes 12 elements because `stem_y` returns 8 and `stem_z` returns 4; that sum is checked here, not on the first batch. `create_composed_net` logs the resolved graph so a mismatch is visible before training starts:

```text
composed net over 3 nodes, inputs ['features', 'latent'], output 'trunk'
  node 'stem_y': EfficientNetV1B0Minimal, in (3, 32) from ['features'], out (8,)
  node 'stem_z': MLPResNet, in (4,) from ['latent'], out (4,)
  node 'trunk': MLPResNet, in (12,) from ['stem_y', 'stem_z'], out (3,)
```

### Step 4: Read the errors when a graph is malformed

`ComposedNet.__init__` validates the whole graph before it stores a single module, and every error names the offending node. Get the trunk's width wrong and construction fails immediately:

```text
ValueError: node 'trunk' takes 9 elements per sample, but its inputs ['stem'] supply [4], summing to 4
```

Wire a node whose output is a feature map, for example a `ConvResNet` with no residual MLP head, into another node's positional input, and construction rejects the edge rather than letting `MLPResNet.forward` fail confusingly on a 3D tensor:

```text
ValueError: node 'trunk' reads from node 'stem', whose output shape (96, None) is not flat; an edge carries a tensor of shape (batch, size)
```

A node that no path from the output reaches also fails here, because DDP raises at the first backward pass when a wrapped module holds parameters that got no gradient; catching it at construction:

```text
ValueError: the output 'trunk' does not reach nodes ['orphan']; their parameters would receive no gradient
```

## Using the composed network

### Step 5: Call it like any other `dlk` network

`ComposedNet.forward(*inputs)` binds its positional arguments to `external_inputs` in order and returns the output node's tensor, so a composed net drops into an existing call site unchanged:

```python
x_gen = g_net(y_data, z)  # the g_net(y_data, z) contract of dlk/opt/train_gan.py
```

Training it is nothing special either. `examples/nets/compose_conditional_gan.py` trains this exact generator against an ordinary `MLPResNet` discriminator through `dlk.opt.train_gan.train_epochs`, unmodified:

```text
epoch      0, d_loss pre mean 1.989e+00 std 9.09e-01, g_loss mean 9.003e-01 std 3.40e-01, d_loss post mean 1.757e+00 std 8.82e-01, time/step mean 29.50 ms std 9.82 ms
epoch      1, d_loss pre mean 9.676e-01 std 4.65e-01, g_loss mean 4.359e-01 std 2.13e-01, d_loss post mean 8.859e-01 std 4.27e-01, time/step mean 23.16 ms std 0.42 ms
```

Run it yourself:

```sh
uv run python examples/nets/compose_conditional_gan.py
```

### Step 6: Checkpoint and reload it

`ComposedNet` adds no parameters of its own, so its `state_dict` keys are exactly `nodes.<node name>.<the network's own keys>`, and `dlk.opt.utils.checkpoint_load` needs no special case:

```python
from dlk.opt.utils import checkpoint_load

reloaded = create_composed_net(**config.to_kwargs(), modules={...})  # same graph, fresh modules
checkpoint_load(checkpoint_path, reloaded)
assert set(reloaded.state_dict()) == set(g_net.state_dict())
```

Renaming a node in the config changes those keys, so a checkpoint written before the rename loads silently wrong under `torch.load(..., strict=False)`; treat node names as part of the checkpoint format.

---

## Routing a tensor into a residual block

Optional: read this when an existing conditional generator injects a tensor into one residual block of its trunk rather than concatenating it at the input, and you want to re-express that graph without redesigning it.

`MLPResNet` and `ConvResNet` accept keyword tensors named `h{index}` or `h_all`, routed into one residual block or into every block instead of the input layer. A `HiddenEdge` is what wires a node's output there:

```python
from dlk.nets.compose import HiddenEdge, NodeConfig

nodes = (
    NodeConfig("stem_y", inputs=("features",)),
    NodeConfig("stem_z", inputs=("latent",)),
    NodeConfig("trunk", inputs=("stem_y",), hidden_inputs=(HiddenEdge("stem_z", block=0),)),
)
```

The trunk now takes only `stem_y`'s width at its input layer, since `stem_z`'s output arrives at block `0` as `h0` instead. Build the trunk at that narrower width, with the first block sized to accept the extra channels. The width check from Step 4 covers positional edges only, since a hidden edge enters one block rather than the input layer, so a hidden edge's width still has to agree with the receiving block by hand, which `dlk.nets.compose` does not check for you.

## Satisfying `NetShape` for a module the protocol does not cover

Optional: read this when a node's module is a plain `nn.Sequential`, a pooling adapter, or anything from outside `dlk`.

`NodeConfig.input_shape` and `NodeConfig.output_shape` are the escape hatch. Set either to a concrete per-sample shape and `resolve_shape` uses it instead of asking the module:

```python
NodeConfig(
    "pool",
    inputs=("features",),
    output_shape=(64,),  # e.g. nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
)
```

A shape is a tuple with one entry per dimension after the batch dimension, and `None` for a dimension the module accepts at any extent; a node reporting `(64,)` is flat and can feed another node's positional input, while `(64, None)` reports a feature map and is rejected there. Set both fields to skip the `NetShape` check entirely for that node.

## Splitting off vmap-safe replicas

Optional: read this when an application draws several latent samples per observation and evaluates a generator once per replica with `torch.vmap`.

Vmapping the whole graph is both wrong and wasteful when a stem does not depend on the latent at all. `MLPResNet` and `ConvResNet` tolerate `vmap` because their normalization is `nn.LayerNorm`, which holds no running state, but every `ScalableEfficientNet1D` block carries `nn.BatchNorm1d`, whose running-statistics update writes into an unbatched buffer and raises under `vmap` in training mode. Evaluating that stem inside the vmapped region also recomputes the same tensor once per replica, since it never uses the latent.

`ComposedNet.dependents_of` reports which nodes a name reaches, and `modules_with_running_stats` reports which of those carry batch normalization, so the mismatch is checked at setup instead of surfacing as a `vmap` error at the first step:

```python
from dlk.nets.compose import modules_with_running_stats

latent_region = net.dependents_of("latent")
blocking = modules_with_running_stats(net, latent_region)
assert not blocking, f"batch normalization inside the vmapped region: {blocking}"
```

`forward_partial` then evaluates everything outside that region once, and `forward` takes the result back as a `cache` of unbatched constants the vmapped callable closes over:

```python
cache = net.forward_partial(exclude_dependents_of="latent", features=y_data)
x_gen = torch.vmap(lambda z_: net(y_data, z_, cache=cache))(z)
```

`examples/nets/compose_replica_latents.py` runs this against eight replicas with a forward hook counting the stem's calls, and confirms the vmapped split matches evaluating each replica in a loop:

```text
latent reaches ['stem_z', 'trunk'], evaluated once: ['stem_y']
nodes with running statistics: ['stem_y']
generated (8, 4, 3) with 1 stem evaluation for 8 replicas
the vmapped split matches evaluating each replica in a loop
```

> [!WARNING]
> The cache holds activations of one autograd graph, so it is valid for one `forward` call and its backward pass. Reusing it across optimizer steps backpropagates through a graph PyTorch already freed and raises; compute a fresh cache every step.

---

## Things worth knowing

**A one-node graph changes nothing.** Wrapping a single existing network in a one-node `ComposedNet` produces the same forward output as calling that network directly; composition adds no layers, parameters, or initialization of its own.

**Edges are flat tensors, always.** Every internal edge carries a `(batch, size)` tensor, and `resolve_output_shape()` with more than one entry is a feature map, rejected wherever it appears on an edge. A network that reports `(channels, None)`, such as `ConvResNet` without its residual MLP head, is still rejected on the ndim check even though the width comparison is skipped for the `None` entry.

**A `ComposedNet` can be a node inside another `ComposedNet`.** It implements `NetShape` itself: `resolve_output_shape()` delegates to its own output node, and `resolve_input_shape()` delegates to whichever node consumes its first external input. Nesting a graph therefore needs no special case anywhere.

**External input shapes are names only.** A composed net does not check the shape of a tensor supplied from outside the graph; a wrong observation shape surfaces from the consuming network's own `forward` assertion on the first batch, one step later than an internal edge mismatch, which is caught at construction.

**Surplus modules are rejected too.** Passing a module keyed by a name that is not a node raises `ValueError`, on the theory that an extra key is a misspelled node name more often than it is a module nobody uses.

## Fixing common issues

**`TypeError: cannot resolve the shapes of node '<name>' ...`.** The node's module implements neither `resolve_input_shape` nor `resolve_output_shape`, and its `NodeConfig` sets no explicit override. Add both to the module (see the `NetShape` accessors already on `MLPNet`, `MLPResNet`, `ConvNet`, `ConvResNet`, `TransformerNet`, `ChannelWiseTransformerNet`, and `ScalableEfficientNet1D`) or set `input_shape`/`output_shape` on the `NodeConfig` directly.

**`ValueError: node '<name>' takes N elements per sample, but its inputs [...] supply [...], summing to M`.** A source's output width does not sum to what the consuming node's input layer expects. Fix the consuming network's declared input size, or recheck which nodes feed it; the message names both sides.

**`ValueError: node '<name>' reads from node '<source>', whose output shape (...) is not flat`.** The source node returns a feature map, not a flat vector, most commonly a `ConvNet` or `ConvResNet` built without its dense head. Give the source an output layer, or route it through an adapter node with an explicit `output_shape` override before wiring it into another node's positional input.

**`ValueError: the output '<name>' does not reach nodes [...]`.** A node in the mapping is wired into nothing the output depends on. Either wire its output into another node, or remove it from `modules` and `nodes`; a genuinely unused node would otherwise fail later, inside DDP, with an unused-parameter error that names no node at all.

**`ValueError: modules given for names that are not nodes: [...]`.** A key in `modules` does not match any `NodeConfig.name`. Check for a typo in one of the two; `dlk` treats this direction as an error rather than silently ignoring the extra module.

## Learn more

- [2026.006__compose_nets__1-plan](2026.006__compose_nets__1-plan.md) for the design decisions behind the shape protocol, the two edge kinds, and the deviations found during implementation.
- [torch.vmap documentation] for what vectorized mapping traces and why it rejects writes into unbatched state.
- `examples/nets/compose_conditional_gan.py` and `examples/nets/compose_replica_latents.py`, runnable end to end, for the two patterns this guide walks through.

<!-- REFERENCES -->

[torch.vmap documentation]: https://docs.pytorch.org/docs/stable/generated/torch.vmap.html
