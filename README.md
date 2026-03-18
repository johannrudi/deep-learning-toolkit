# Deep Learning Toolkit

[![Format](https://img.shields.io/github/actions/workflow/status/johannrudi/deep-learning-toolkit/format.yml?style=for-the-badge&label=Format)](https://github.com/johannrudi/deep-learning-toolkit/actions/workflows/format.yml)
[![Compile](https://img.shields.io/github/actions/workflow/status/johannrudi/deep-learning-toolkit/compile.yml?style=for-the-badge&label=Compile)](https://github.com/johannrudi/deep-learning-toolkit/actions/workflows/compile.yml)
[![Test](https://img.shields.io/github/actions/workflow/status/johannrudi/deep-learning-toolkit/test.yml?style=for-the-badge&label=Test)](https://github.com/johannrudi/deep-learning-toolkit/actions/workflows/test.yml)
[![Lint](https://img.shields.io/github/actions/workflow/status/johannrudi/deep-learning-toolkit/lint.yml?style=for-the-badge&label=Lint)](https://github.com/johannrudi/deep-learning-toolkit/actions/workflows/lint.yml)

Reusable [PyTorch](https://pytorch.org/) building blocks for artificial intelligence & scientific machine learning:
networks, losses, training loops, and utilities.

## Installing the `deep-learning-toolkit`

### Requirements

- Python `>=3.11`

### Runtime dependencies

- `torch>=2,<3`
- `prettytable>=3,<4`
- `tqdm>=4,<5`

<!-- TODO: uncomment this after a public release
### Install in regular mode

```sh
pip install deep-learning-toolkit
```
-->

### Install the package in editable mode

Using a clone of the Git repository:

```sh
pip install -e .
```

#### Install with optional extras

Kernel density estimation features:

```sh
pip install -e ".[kde]"
```

Running tests:

```sh
pip install -e ".[test]"
```

#### Install with optional extras for development

```sh
pip install -e ".[dev]"
```

This includes extras from `[test]`.

## Development commands

- `make format`: run `isort` and `black` on `dlk/` and `tests/`
- `make format-check`: check `isort` and `black` formatting without modifying files
- `make compile`: run `python -m compileall -q -f` on `dlk/` and `tests/`
- `make lint`: run `basedpyright` on `dlk/` and `tests/`
- `make test`: run `pytest` (after `make compile`)
- `make testq`: run `pytest -q` (after `make compile`)
- `make testv`: run `pytest -v` (after `make compile`)
- `make testvv`: run `pytest -sv` (after `make compile`)

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
