# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

**dl-kit** (Deep Learning tool-Kit) is a PyTorch-based toolkit for building and training deep learning models, with a focus on scientific computing and inverse problems. The package provides reusable neural network architectures, loss functions, training loops, and utilities.

## Installation and Development

### Install the package in editable mode
```bash
pip install -e .
```

### Install with development dependencies
```bash
pip install -e ".[dev]"
```

### Run tests for specific network modules
```bash
python -m dlk.nets.mlp
python -m dlk.nets.transformer1d
python -m dlk.nets.unet
python -m dlk.nets.conv1d
python -m dlk.nets.conv2d
python -m dlk.nets.efficientnet
```

Note: Tests are embedded in network modules as `test_*` functions called from `if __name__ == '__main__'` blocks. There is no separate test directory or pytest framework configured.

## Architecture

### Package Structure

- **`dlk/nets/`**: Neural network architectures
  - `mlp.py`: Multilayer Perceptron (MLPNet, MLPNet_MultIn, MLPResNet with residual and attention blocks)
  - `autoencoder.py`: Generic autoencoder wrapper for encoder/decoder pairs
  - `conv1d.py`, `conv2d.py`: 1D/2D convolutional networks and UNet components (Downsample, Upsample)
  - `unet.py`: Complete UNet implementations (older UNet1D/UNet2D and newer UNetXd_2025 architecture)
  - `transformer1d.py`: 1D transformer networks with patch embeddings and multi-head attention
  - `efficientnet.py`: EfficientNet architecture
  - `util.py`: Shared utilities (parameter initialization, parameter counting, printing)

- **`dlk/opt/`**: Training and optimization
  - `train.py`: Training loops (`train_epochs`, `train_batches`) with checkpointing and validation hooks
  - `train_gan.py`: GAN-specific training loops
  - `scheduler.py`: Learning rate schedulers (multi-stage: linear warmup, constant, cosine annealing)

- **`dlk/loss/`**: Loss functions
  - `gaussian.py`: Gaussian loss with covariance (using low-rank SVD approximation)
  - `wasserstein.py`: Wasserstein loss and gradient penalty for GANs

- **`dlk/log/`**: Logging utilities
  - `log_util.py`: Structured logging setup with file handlers, console output, and library filtering (e.g., suppress matplotlib debug logs)

### Key Design Patterns

#### Network Initialization
All network modules follow a consistent pattern:
- Constructor calls `self.init_parameters()` at the end
- `init_parameters()` uses Xavier initialization with gain calculated from activation functions
- Utility functions `_get_gain()` and `_set_init_parameters()` handle activation-aware initialization

#### Multi-Input Networks
Several architectures support multiple input tensors and hidden layer inputs:
- `MLPNet_MultIn`: concatenates multiple inputs, accepts hidden inputs via `h0=`, `h1=`, etc. kwargs
- `MLPResNet`: supports hidden inputs `h0=`, `h1=`, ... to inject information at intermediate residual blocks

#### Training Loops
Training functions return detailed logging dictionaries (`dlog`) containing:
- Per-epoch loss statistics (`loss_mean`, `loss_std`)
- Batch-level logs nested in `batch_dlog`
- Total training time in `time_train`
- Checkpointing saves model and optimizer states at specified intervals

## Coding Conventions

### Documentation Style
- Use docstrings for all classes and functions
- Document Args with type and description
- For mathematical notation, use `r"""` raw docstrings to support LaTeX-like notation

### Parameter Naming
- Use descriptive names: `input_size`, `output_size`, `hidden_layers_sizes`
- Activation layers: `*_activation` (e.g., `hidden_layers_activation=nn.ReLU()`)
- Layer-specific kwargs: `*_kwargs` (e.g., `input_layer_kwargs={}`)

### Network Forward Methods
- Always validate input dimensions with assertions showing actual vs expected values
- Use informative f-strings: `assert h.size(1) == self.input_size, f"{h.size(1)=}, {self.input_size=}"`

### Comments
- start with a verb in lowercase
- if starting with a noun, capitalize the first letter (e.g., "# Vector norm" or "# calculate norm")
- keep comments concise and similar to paragraph titles

## Dependencies

- Core: `torch>=2`, `numpy>=2`, `tqdm>=4`, `prettytable>=3`
- Dev: `black>=25`, `pytest>=8`, `build`, `twine`, `bumpver`, `pip-tools`
- Python: `>=3.8` (tested on 3.8-3.12)
