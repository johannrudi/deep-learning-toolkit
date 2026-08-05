# Deep Learning Toolkit

<!-- NOTE: github badges work with private repos -->
<!--
[![CI](https://github.com/johannrudi/deep-learning-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/johannrudi/deep-learning-toolkit/actions/workflows/ci.yml)
-->
<!-- NOTE: shields.io badges work *only* with public repos -->
[![CI](https://img.shields.io/github/actions/workflow/status/johannrudi/deep-learning-toolkit/ci.yml?style=for-the-badge&label=CI)](https://github.com/johannrudi/deep-learning-toolkit/actions/workflows/ci.yml)

Reusable [PyTorch](https://pytorch.org/) building blocks for artificial intelligence & scientific machine learning: networks, losses, training loops, and utilities.

The *Deep Learning Toolkit* is a Python library of reusable [PyTorch](https://pytorch.org/) components for artificial intelligence and scientific machine learning. It is designed to be composed into research codes (not a standalone application) and accelerate development of such codes.

The package provides:

- **Network architectures**: multilayer perceptrons with residual and attention blocks, 1D and 2D convolutional networks, UNets, 1D transformers with patch embeddings, autoencoders, and more.
- **Training and optimization**: epoch- and batch-level training loops with checkpointing and validation hooks, distributed training, GAN training loops, and multi-stage learning rate schedulers.
- **Supporting components**: loss functions, evaluation metrics, plotting helpers, and configuration management.

Network modules share a consistent activation-aware parameter initialization scheme, and training routines return structured logs of per-epoch and per-batch loss statistics.

---

## Installing the `deep-learning-toolkit`

### Requirements

- Python version `>=3.11`

### Runtime dependencies

- `matplotlib` version `>=3,<4`
- `prettytable` version `>=3,<4`
- `pyyaml` version `>=6,<7`
- **`torch` version `>=2,<3`**
- `tqdm` version `>=4,<5`

### Install in regular mode

```sh
pip install deep-learning-toolkit
```

### Install the package in editable mode

When using a clone of the [Git repository](https://github.com/johannrudi/deep-learning-toolkit/), run this command from inside the cloned directory:

```sh
pip install -e .
```

#### Install with optional extras

Dependencies for generative diffusion models:

```sh
pip install -e ".[diffusion]"
```

Dependencies for kernel density estimation:

```sh
pip install -e ".[kde]"
```

### Set up a development environment

This project is managed with [uv](https://docs.astral.sh/uv/). The development
dependencies are declared as a dependency group rather than as extras, so they
are installed by `uv` instead of `pip`:

```sh
uv sync
```

This creates `.venv` from the pinned versions in `uv.lock` and installs the
default `dev` group, which covers formatting, linting, and testing.

To additionally install the published extras:

```sh
uv sync --all-extras
```

After changing dependencies in `pyproject.toml`, refresh and commit the lock file:

```sh
uv lock
```

Continuous integration runs with `UV_LOCKED=1`, so a stale `uv.lock` fails the build.

---

## Importing and using `dlk`

To use the toolkit, import its modules in your Python code like this:

```py
from dlk.nets.mlp import MLPNet
from dlk.opt.train import train_epochs

# load your data
...

# create the model
net = MLPNet(input_size=784, output_size=10)

# train the model
train_epochs(n_epochs=100, net=net, dataloader=..., optimizer=..., loss_fn=...)

# evaluate
...
```

---

## Architecture

### Neural network architectures &rarr; `dlk/nets/`

- `mlp.py`: Multilayer Perceptron (MLPNet, MLPNet_MultIn, MLPResNet with residual and attention blocks)
- `autoencoder.py`: Generic autoencoder wrapper for encoder/decoder pairs
- `conv1d.py`, `conv2d.py`: 1D/2D convolutional networks and UNet components (Downsample, Upsample)
- `unet.py`: Complete UNet implementations (older UNet1D/UNet2D and newer UNetXd_2025 architecture)
- `transformer1d.py`: 1D transformer networks with patch embeddings and multi-head attention
- `efficientnet.py`: EfficientNet architecture

#### Network initialization

All network modules follow a consistent pattern:

- Constructor calls `self.init_parameters()` at the end
- `init_parameters()` uses Xavier initialization with gain calculated from activation functions
- Utility functions `_get_gain()` and `_set_init_parameters()` handle activation-aware initialization

### Training and optimization &rarr; `dlk/opt/`

- `train.py`: Training loops (`train_epochs`, `train_batches`) with checkpointing and validation hooks
- `train_gan.py`: GAN-specific training loops
- `scheduler.py`: Learning rate schedulers (multi-stage: linear warmup, constant, cosine annealing)

#### Logging of the training progress

Training functions return detailed logging dictionaries (`dlog`) containing:

- Per-epoch loss statistics (`loss_mean`, `loss_std`)
- Batch-level logs nested in `batch_dlog`
- Total training time in `time_train`
- Checkpointing saves model and optimizer states at specified intervals

### Additional components of the package

- `dlk/mgmt/`: Management of configuration parameter loading/saving, logging, etc.
- `dlk/loss/`: Loss functions
- `dlk/metrics/`: Metrics for evaluating trained nets

---

## Development

### Commands for development

- `make format`: run `isort` and `black` on `dlk/` and `tests/`
- `make format-check`: check `isort` and `black` formatting without modifying files
- `make compile`: run `python -m compileall -q -f` on `dlk/` and `tests/`
- `make lint`: run `basedpyright` on `dlk/` and `tests/`
- `make test`: run `pytest` (after `make compile`)
- `make testq`: run `pytest -q` (after `make compile`)
- `make testv`: run `pytest -v` (after `make compile`)
- `make testvv`: run `pytest -sv` (after `make compile`)

All targets run their tools through `uv run`, so `uv` must be installed.

### Building a distribution

```sh
uv build --no-sources
```

---

## Citing

Citation metadata is provided in [`CITATION.cff`](CITATION.cff), which GitHub
renders under "Cite this repository". Releases are archived on
[Zenodo](https://zenodo.org/), which mints a DOI for each version.

---

## License

Licensed under the [Apache License 2.0](LICENSE).
