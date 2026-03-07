# DL-Kit: Deep Learning Toolkit

This is a PyTorch-based toolkit for building and training deep learning models, with a
focus on scientific computing and inverse problems. The package provides reusable neural
network architectures, loss functions, training loops, and utilities.

## Installation

### Install the package in editable mode

```sh
pip install -e
```

### Install with development dependencies (`[dev]`)

```sh
pip install -e ".[dev]"
```

### Install in regular mode

```sh
pip install .
```

## Architecture

### Neural network architectures - `dlk/nets/`

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

### Training and optimization - `dlk/opt/`

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

- `dlk/log/`: Logging
- `dlk/loss/`: Loss functions
- `dlk/metrics/`: Metrics for evaluating trained nets
