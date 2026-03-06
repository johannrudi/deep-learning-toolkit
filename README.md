# DL-Kit: Deep Learning Toolkit

## Install

### Install in develop mode (`-e`) with developer dependencies (`[dev]`)

```bash
pip install -e ".[dev]"
```

### Install in regular mode

```bash
pip install .
```

## Architecture

### Main components of the package

- `dlk/nets/`: Neural network architectures
  - `mlp.py`: Multilayer Perceptron (MLPNet, MLPNet_MultIn, MLPResNet with residual and attention blocks)
  - `autoencoder.py`: Generic autoencoder wrapper for encoder/decoder pairs
  - `conv1d.py`, `conv2d.py`: 1D/2D convolutional networks and UNet components (Downsample, Upsample)
  - `unet.py`: Complete UNet implementations (older UNet1D/UNet2D and newer UNetXd_2025 architecture)
  - `transformer1d.py`: 1D transformer networks with patch embeddings and multi-head attention
  - `efficientnet.py`: EfficientNet architecture

- `dlk/opt/`: Training and optimization
  - `train.py`: Training loops (`train_epochs`, `train_batches`) with checkpointing and validation hooks
  - `train_gan.py`: GAN-specific training loops
  - `scheduler.py`: Learning rate schedulers (multi-stage: linear warmup, constant, cosine annealing)

### Additional components of the package

- `dlk/log/`: Logging
- `dlk/loss/`: Loss functions
- `dlk/metrics/`: Metrics for evaluating trained nets
