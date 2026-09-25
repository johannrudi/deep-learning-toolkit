---
Title: Choosing a Normalization Layer for 1D Convolutional Networks
Author: Johann Rudi
Co-Authored-By: Claude Opus 5.5
Date: 2026-09-25
tags:
  - normalization
  - architecture
  - nets
  - training
links:
  - "[[composed_nets]]"
  - "[[2026.008__gradient_penalties__1-plan]]"
---

# Choosing a Normalization Layer for 1D Convolutional Networks

A **normalization layer** rescales activations to zero mean and unit variance using statistics it computes from the activations themselves, then applies a learned per-channel scale and shift. Every normalization layer in common use is this one operation; they differ only in which entries they average over. That choice decides what the network can no longer see: a normalization that averages over the channels at each position erases how loud that position was, and one that averages along the sequence erases how loud each channel was.

Picking a normalization by what a famous architecture uses therefore carries that architecture's assumptions about its data into yours. ConvNeXt normalizes each position over its channels because, for natural images, the brightness of a patch is mostly a nuisance. For a 1D signal where the energy of a segment is the information, the same choice throws the signal away. This guide gives the math of each normalization in one notation, shows how to measure what each one keeps, and walks through a controlled comparison on your own dataset.

This guide assumes you have trained convolutional networks in PyTorch and build blocks from `dlk/nets/conv1d.py`. The examples use `UniversalMultiLevelBlock`, whose `normalization` argument takes any factory that builds a layer from its number of channels.

> [!WARNING]
> `nn.LayerNorm(C)` does not normalize over the channels of a `(batch, channels, length)` tensor. It normalizes the trailing dimensions, which for a 1D feature map is the length, and it fails outright when the length differs from `C`: `RuntimeError: Given normalized_shape=[8], expected input with shape [*, 8], but got input of size[4, 8, 64]`. Use `ChannelLayerNorm` from `dlk/nets/conv1d.py` for layer normalization over channels.

## The common form

Write an activation tensor as $x \in \mathbb{R}^{B \times C \times L}$, with batch index $b$, channel index $c$, and position index $l$. A normalization layer picks a set $S$ of entries, computes their mean and variance,

$$
\mu_S = \frac{1}{|S|} \sum_{(b,c,l) \in S} x_{b,c,l},
\qquad
\sigma_S^2 = \frac{1}{|S|} \sum_{(b,c,l) \in S} \left( x_{b,c,l} - \mu_S \right)^2,
$$

and maps every entry of $S$ to

$$
y_{b,c,l} = \gamma_c \, \frac{x_{b,c,l} - \mu_S}{\sqrt{\sigma_S^2 + \epsilon}} + \beta_c,
$$

where $\gamma_c$ and $\beta_c$ are the learned **affine parameters**, one scale and one shift per channel, and $\epsilon$ is a small constant that keeps the division finite. The affine parameters let the network undo the normalization where it hurts, so the layer constrains the training dynamics more than it constrains what the network can express.

The output is invariant to whatever the statistics absorb: scale every entry of $S$ by the same factor, and $y$ does not change. That invariance is the whole design question.

## Normalizing activations

The methods below differ only in $S$. The table contrasts them for a `(B, C, L)` tensor; the subsections give the formulas.

| Method | One set $S$ per | $S$ spans | Keeps loud positions | Keeps loud channels | Depends on the batch |
|---|---|---|---|---|---|
| `nn.GroupNorm(1, C)` | sample | all channels and positions | yes | yes | no |
| `nn.GroupNorm(G, C)` | sample and group | $C/G$ channels, all positions | yes | across groups | no |
| `nn.GroupNorm(C, C)`, instance norm | sample and channel | all positions | yes | no | no |
| `ChannelLayerNorm(C)` | sample and position | all channels | no | partly | no |
| `nn.BatchNorm1d(C)` | channel | all samples and positions | yes | no | yes |

### Group normalization

**Group normalization** splits the $C$ channels into $G$ groups $\mathcal{G}_1, \dots, \mathcal{G}_G$ of $C/G$ consecutive channels and normalizes each group of each sample over its channels and all positions:

$$
\mu_{b,g} = \frac{G}{C L} \sum_{c \in \mathcal{G}_g} \sum_{l=1}^{L} x_{b,c,l},
\qquad
\sigma_{b,g}^2 = \frac{G}{C L} \sum_{c \in \mathcal{G}_g} \sum_{l=1}^{L} \left( x_{b,c,l} - \mu_{b,g} \right)^2.
$$

Two settings of $G$ have their own names. With $G = 1$, one mean and one variance per sample cover the whole feature map, so the layer removes a single global offset and gain and keeps every relative magnitude, between channels and between positions. This is the default of `UniversalMultiLevelBlock` (the `Normalization` helper in `dlk/nets/conv1d.py`) and of `UNetResBlock`. With $G = C$, each channel gets its own statistics along the sequence, which is **instance normalization**: it removes each channel's level and gain.

`nn.GroupNorm` requires $C$ to be divisible by $G$, and it checks this at construction.

### Layer normalization over channels

**Layer normalization** in the transformer sense normalizes each token over its features. For a 1D feature map, a token is a position and its features are the channels, so the statistics are per sample and position:

$$
\mu_{b,l} = \frac{1}{C} \sum_{c=1}^{C} x_{b,c,l},
\qquad
\sigma_{b,l}^2 = \frac{1}{C} \sum_{c=1}^{C} \left( x_{b,c,l} - \mu_{b,l} \right)^2.
$$

`ChannelLayerNorm` in `dlk/nets/conv1d.py` implements exactly this by transposing to channels-last, applying `nn.LayerNorm(C)`, and transposing back; it is the normalization of `ConvNeXtBlock`. After it, the channel vector at every position has zero mean and unit variance, so only its **direction** survives. A silent stretch of the signal and a loud one produce outputs of the same size.

That per-position view needs enough channels to estimate a variance from. With $C = 1$ the output is $\beta$ at every position, whatever the input, and with $C = 2$ every normalized entry is $\pm 1$. Both happen in practice in early blocks of networks whose input has few channels.

`nn.LayerNorm([C, L])` is the other reading of "layer norm": it averages over all channels and positions, which gives the statistics of `GroupNorm(1, C)`, but it learns $\gamma$ and $\beta$ per entry, shape `(C, L)`, and so ties the layer to one sequence length. `GroupNorm(1, C)` is the length-independent version.

### Batch normalization

**Batch normalization** computes one mean and variance per channel over the whole batch and all positions:

$$
\mu_{c} = \frac{1}{B L} \sum_{b=1}^{B} \sum_{l=1}^{L} x_{b,c,l},
\qquad
\sigma_{c}^2 = \frac{1}{B L} \sum_{b=1}^{B} \sum_{l=1}^{L} \left( x_{b,c,l} - \mu_{c} \right)^2.
$$

During training it also tracks exponential moving averages of $\mu_c$ and $\sigma_c^2$, the **running statistics**, and uses them in evaluation mode instead of batch statistics. It is the only method here whose output for one sample depends on the other samples in its batch, which is its strength (regularizing noise, no per-sample cost at inference) and the source of every one of its problems: a gap between training and evaluation behavior, poor estimates at small per-device batch sizes, and cross-process synchronization under DDP. `dlk/nets/efficientnet1d.py` uses it throughout.

### RMS normalization

**RMS normalization** drops the mean subtraction and the shift, and divides by the root mean square over the channels at each position:

$$
y_{b,c,l} = \gamma_c \, \frac{x_{b,c,l}}{\sqrt{\frac{1}{C} \sum_{c'=1}^{C} x_{b,c',l}^2 + \epsilon}}.
$$

It is cheaper than layer normalization and matches it in quality for large transformers, which is why current language models use it. For 1D feature maps it shares the per-position invariance of `ChannelLayerNorm`. PyTorch ships `nn.RMSNorm`, which normalizes trailing dimensions and needs the same transpose as `ChannelLayerNorm` (see Step 3).

## Choosing a normalization for your dataset

### Step 1: Decide which amplitudes carry information

Before training anything, answer two questions about your data. Does the energy of a segment of the sequence carry information, as the loudness of an event in an audio or seismic trace does? Does the gain of each channel carry information, or is it a nuisance, as the calibration of a sensor often is? A normalization that removes an amplitude your task depends on forces the network to recover it through the skip branch or not at all; one that keeps an amplitude that is a nuisance leaves the network to learn the invariance from data.

Measure what each candidate keeps with two probes: make half of the sequence ten times louder, and make one channel ten times louder, then compare standard deviations after normalization.

```python
import torch
import torch.nn as nn

from dlk.nets.conv1d import ChannelLayerNorm

torch.manual_seed(0)
x = torch.randn(4, 8, 64)
x_pos = x.clone()
x_pos[..., 32:] *= 10  # second half of the sequence 10x louder
x_ch = x.clone()
x_ch[:, 0] *= 10  # channel 0 10x louder

candidates = {
    "GroupNorm(1, C)": nn.GroupNorm(1, 8),
    "GroupNorm(C, C)": nn.GroupNorm(8, 8),
    "ChannelLayerNorm(C)": ChannelLayerNorm(8),
}
for name, norm in candidates.items():
    y_pos, y_ch = norm(x_pos), norm(x_ch)
    pos_ratio = y_pos[..., 32:].std() / y_pos[..., :32].std()
    ch_ratio = y_ch[:, 0].std() / y_ch[:, 1:].std()
    print(f"{name:20s} positions {pos_ratio:5.2f}  channels {ch_ratio:5.2f}")
```

A ratio near 10 means the layer kept the contrast; a ratio of 1.00 means it erased it.

```text
GroupNorm(1, C)      positions  9.41  channels 10.38
GroupNorm(C, C)      positions  6.99  channels  1.00
ChannelLayerNorm(C)  positions  1.00  channels  3.62
```

`GroupNorm(1, C)` keeps both contrasts. Instance normalization erases the loud channel and keeps the loud half of the sequence, since each channel's statistics span the whole sequence. `ChannelLayerNorm` erases the loud half exactly and compresses the loud channel: that channel dominates the variance at every position, so dividing by it shrinks the contrast from 10 to 3.6. The table in "Normalizing activations" is the summary of this output.

### Step 2: Rule out candidates by your training setup

Several constraints eliminate candidates before any comparison, and they are cheaper to check than to discover in a failed run.

- **Small per-device batches.** Batch statistics over two or four samples are noise. Prefer a per-sample method when each GPU sees fewer than roughly 16 samples.
- **DDP.** `nn.BatchNorm1d` computes statistics per process. Converting it with `nn.SyncBatchNorm.convert_sync_batchnorm` restores full-batch statistics at the cost of a synchronization in every normalization layer of every forward pass.
- **GAN critics with a gradient penalty.** `dlk/loss/wasserstein_gan.py` penalizes the gradient norm of the critic per sample (`gradient_penalty_lip`, `gradient_penalty_opt`). Batch normalization couples the samples of a batch, so the gradient with respect to one sample depends on the others and the penalty no longer constrains the function it is meant to constrain. Gulrajani et al. (2017) recommend layer normalization in the critic for this reason; any batch-independent method here qualifies.
- **`torch.vmap`.** The running-statistics update of batch normalization writes into an unbatched buffer and fails under `vmap` in training mode; [the composed networks guide](composed_nets.md) covers splitting such a module out of the vmapped region.
- **Few channels.** `ChannelLayerNorm` and RMS normalization over one or two channels degenerate as described above.

Batch dependence is easy to confirm directly: normalize one sample alone and together with three louder companions.

```python
companions = torch.randn(3, 8, 64) * 5
for norm in (nn.BatchNorm1d(8).train(), nn.GroupNorm(1, 8)):
    alone = norm(x[:1])
    in_batch = norm(torch.cat([x[:1], companions]))[:1]
    print(f"{type(norm).__name__:12s} {(alone - in_batch).abs().max():.3f}")
```

```text
BatchNorm1d  2.583
GroupNorm    0.000
```

The same sample comes out differently depending on what it shares a batch with, by more than two standard deviations.

### Step 3: Build the candidates as factories

`UniversalMultiLevelBlock` and its subclasses take `normalization` as a factory called with the number of channels to normalize, so every candidate is one expression. RMS normalization over channels needs the same transpose as `ChannelLayerNorm`, written once as a small subclass:

```python
from functools import partial

import torch
import torch.nn as nn

from dlk.nets.conv1d import ChannelLayerNorm, UniversalMultiLevelBlock


class ChannelRMSNorm(nn.RMSNorm):
    """RMS normalization over the channels of ``(batch, channels, length)`` tensors."""

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


candidates = {
    "group_1": True,  # the block's default, GroupNorm(1, C)
    "group_4": partial(nn.GroupNorm, 4),
    "instance": lambda c: nn.GroupNorm(c, c),
    "channel_layer": ChannelLayerNorm,
    "channel_rms": ChannelRMSNorm,
    "batch": nn.BatchNorm1d,
}
block = UniversalMultiLevelBlock(8, kernel_size=5, normalization=candidates["instance"], skip_connection=True)
```

Every candidate in this mapping builds and runs a forward pass on a `(4, 8, 64)` input. Keep the rest of the architecture fixed across candidates, so the comparison measures the normalization and nothing else.

### Step 4: Run a controlled comparison

Train one model per candidate and seed, under the same data split, budget, and schedule, and compare a validation metric:

```python
results = {}
for name, normalization in candidates.items():
    results[name] = [train_and_validate(normalization, seed=seed) for seed in range(3)]
```

Here `train_and_validate` stands for your own training script, parameterized by the factory and a seed, returning the validation metric. Three rules make the numbers mean something:

- **Use at least three seeds** and report the mean and standard deviation. Normalization changes are small architectural changes, and differences between them are often smaller than the spread across seeds; a difference inside that spread is a tie.
- **Tune the learning rate per candidate**, with a short sweep over a factor of ten in each direction. Normalization changes the effective step size of the layers before it, so one shared learning rate favors whichever candidate it happens to suit.
- **Break ties toward the batch-independent, cheaper method.** It removes a class of DDP and evaluation problems for free.

### Step 5: Check what the trained model learned

A validation score says which candidate fits this split. Two more checks say whether it will hold up when the data shifts.

Rescale the validation inputs by a global gain and see how the metric moves. A model whose normalization absorbed the gain barely changes; one that relies on absolute amplitude degrades, which is correct if amplitude is the information and a warning if it is a nuisance.

```python
@torch.no_grad()
def gain_sensitivity(net, loader, metric, gains=(0.1, 1.0, 10.0)):
    net.eval()
    return {
        gain: torch.stack([metric(net(gain * x), y) for x, y in loader]).mean().item()
        for gain in gains
    }
```

Repeat with a gain on one input channel to test the per-channel invariance. For a batch-normalized candidate, also compare the validation metric in training mode against evaluation mode: a large gap means the running statistics do not describe the batches the model trained on, typically because the per-device batch is too small.

---

## Other normalizations worth knowing

Optional: read this when an architecture outside the activation normalizations above comes up, or when a paper you are reproducing uses one.

### Global response normalization

ConvNeXt V2 (Woo et al., 2023) adds **global response normalization** (GRN) after the activation of each block, to counter channels that collapse to near-constant outputs. It computes each channel's $L_2$ norm along the sequence, divides by the mean norm over channels, and feeds the result back as a gate:

$$
G_{b,c} = \Big( \sum_{l=1}^{L} x_{b,c,l}^2 \Big)^{1/2},
\qquad
N_{b,c} = \frac{G_{b,c}}{\frac{1}{C} \sum_{c'=1}^{C} G_{b,c'} + \epsilon},
\qquad
y_{b,c,l} = \gamma_c \, x_{b,c,l} \, N_{b,c} + \beta_c + x_{b,c,l}.
$$

Channels compete for response: one whose norm exceeds the average is amplified relative to the rest. The trailing $+ x$ makes the layer start as the identity when $\gamma = \beta = 0$.

### Conditional normalization

**Conditional normalization** predicts the affine parameters from a conditioning vector $e$, such as a diffusion timestep embedding or a class label, instead of learning them as constants:

$$
y = \gamma(e) \odot \hat{x} + \beta(e),
$$

where $\hat{x}$ is the normalized activation and $\gamma(e)$, $\beta(e)$ come from a small network. FiLM (Perez et al., 2018) introduced the pattern, and diffusion U-Nets apply it after group normalization as **AdaGN** (Dhariwal and Nichol, 2021). DiT's **adaLN-Zero** (Peebles and Xie, 2023) also predicts a gate $\alpha(e)$ for the residual branch, $x + \alpha(e) \odot F(x)$, initialized to zero so every block starts as the identity. The embedding blocks in `dlk/nets/unet.py` are where this pattern belongs.

### Normalizing weights

Some methods normalize the weights instead of the activations. **Spectral normalization** divides a weight matrix by its largest singular value,

$$
W_{\mathrm{SN}} = \frac{W}{\sigma_{\max}(W)},
$$

which bounds each layer's Lipschitz constant by one; `MLPResNet` offers it through `use_spectral_norm` for GAN critics. **Weight standardization** gives each output filter zero mean and unit variance over its fan-in, $\hat{W}_{o,:} = (W_{o,:} - \mu_o)/\sigma_o$, and pairs well with group normalization at small batch sizes.

### Normalizing queries and keys

**QK-norm** applies layer or RMS normalization to the queries and keys of attention before their dot product,

$$
A = \operatorname{softmax}\!\Big( \frac{N(Q) \, N(K)^{\top}}{\sqrt{d}} \Big),
$$

which bounds the attention logits and prevents the logit growth that destabilizes large transformers. It is relevant to `dlk/nets/transformer1d.py`, which normalizes only the residual stream.

### Networks without normalization

**Dynamic Tanh** (DyT; Zhu, Chen, He, and LeCun, 2025) replaces layer normalization in transformers with an elementwise squashing function and computes no statistics at all:

$$
y = \gamma \odot \tanh(\alpha x) + \beta,
$$

with a learned scalar $\alpha$. It matched layer-normalized transformers in their experiments, and it remains a research direction. NFNets (Brock et al., 2021) reached the same goal for convolutional networks with careful initialization and adaptive gradient clipping.

---

## Things worth knowing

**Normalization and Lipschitz bounds.** A normalization layer divides by a standard deviation computed from the data, so its Lipschitz constant is bounded by nothing the weights control. Spectrally normalizing the convolutions of a network that also normalizes activations does not make the network 1-Lipschitz; a critic that relies on that bound needs its normalization layers checked too.

**Checkpoints record the choice.** Group and layer normalization store `weight` and `bias`; batch normalization adds the buffers `running_mean`, `running_var`, and `num_batches_tracked`. Switching the normalization of a trained model changes its `state_dict` keys, and loading an old checkpoint with `strict=False` silently skips the mismatched entries.

**Epsilon differs between implementations.** `ChannelLayerNorm` defaults to $\epsilon = 10^{-6}$, as in ConvNeXt, while PyTorch's `nn.LayerNorm`, `nn.GroupNorm`, and `nn.BatchNorm1d` default to $10^{-5}$. The difference matters only for activations whose variance approaches $\epsilon$, which is exactly the regime of low-precision training.

**The skip branch sees the unnormalized input.** In `UniversalMultiLevelBlock` and `ConvNeXtBlock`, normalization sits on the main branch only, so an amplitude it erases still reaches the output through the skip connection. That softens every invariance in the table: a residual network with `ChannelLayerNorm` can still use the loudness of a segment, only through a narrower path.

## Fixing common issues

**`RuntimeError: Given normalized_shape=[8], expected input with shape [*, 8], but got input of size[4, 8, 64]`.** An `nn.LayerNorm(C)` received a channels-first tensor and tried to normalize the length. Replace it with `ChannelLayerNorm(C)`, or transpose to `(batch, length, channels)` around it.

**`ValueError: num_channels (8) must be divisible by num_groups (3)`.** `nn.GroupNorm` needs equal-size groups. Pick $G$ as a divisor of every channel count the factory will see; in `UniversalMultiLevelBlock` that is `normalization_channels`, which defaults to `input_channels`.

**`ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 8, 1])`.** Batch normalization in training mode received one value per channel, so it cannot estimate a variance. It happens with a batch of one at the end of a sequence that pooling has shortened to length one. Drop the last incomplete batch with `drop_last=True`, or switch to a batch-independent method.

**`AssertionError: Expect shape (batch, 8, length), got (4, 16, 64)`.** `ChannelLayerNorm` was built for a different number of channels than it received. Check the channel count passed to the factory against the layer before it.

## Learn more

- [Group Normalization] (Wu and He, 2018) for the group-size study and the comparison against batch normalization at small batch sizes.
- [ConvNeXt] and [ConvNeXt V2] for why a modern convolutional network uses layer normalization per position, and what GRN adds.
- [Improved Training of Wasserstein GANs] for the argument against batch normalization in a gradient-penalized critic.
- [PyTorch normalization layers] for the exact signatures and defaults of the built-in layers.

<!-- REFERENCES -->

[Group Normalization]: https://arxiv.org/abs/1803.08494
[ConvNeXt]: https://arxiv.org/abs/2201.03545
[ConvNeXt V2]: https://arxiv.org/abs/2301.00808
[Improved Training of Wasserstein GANs]: https://arxiv.org/abs/1704.00028
[PyTorch normalization layers]: https://docs.pytorch.org/docs/stable/nn.html#normalization-layers
