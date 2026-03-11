import torch
import torch.nn as nn

from dlk.nets.transformer_experiment import (
    DataEmbedding,
    PositionalEmbedding,
    Transformer1d0dModel,
)


def _build_model() -> Transformer1d0dModel:
    """Build a small transformer model for smoke tests."""
    return Transformer1d0dModel(
        1,
        3,
        32,
        8,
        n_encoder_layers=2,
        n_decoder_layers=2,
        feedforward_size=128,
        transformer_activation=nn.GELU(),
    )


def test_data_embedding_forward_shape() -> None:
    """Run ``DataEmbedding`` and validate the output shape."""
    emb = DataEmbedding(1, 2)
    x = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0]]])

    y = emb(x)

    assert y.shape == (1, 4, 2)


def test_positional_embedding_forward_shape() -> None:
    """Run ``PositionalEmbedding`` and validate the output shape."""
    pos = PositionalEmbedding(2)
    x = torch.tensor([[[1.0, 2.0], [-1.0, -2.0], [1.0, 2.0], [-1.0, -2.0]]])

    y = pos(x)

    assert y.shape == (1, 4, 2)


def test_positional_embedding_on_data_embedding_output_shape() -> None:
    """Compose ``DataEmbedding`` and ``PositionalEmbedding`` and validate shape."""
    emb = DataEmbedding(1, 2)
    pos = PositionalEmbedding(2)
    x = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0]]])

    y = pos(emb(x))

    assert y.shape == (1, 4, 2)


def test_transformer1d0d_model_forward_with_target_shape() -> None:
    """Run ``Transformer1d0dModel`` with explicit target input."""
    model = _build_model()
    x_src = torch.randn(1, 30, 1)
    x_trg = torch.randn(1, 1, 3)

    y = model(x_src, x_trg)

    assert y.shape == (1, 3)


def test_transformer1d0d_model_forward_without_target_shape() -> None:
    """Run ``Transformer1d0dModel`` with internally-created target input."""
    model = _build_model()
    x_src = torch.randn(1, 6, 1)

    y = model(x_src)

    assert y.shape == (1, 3)
