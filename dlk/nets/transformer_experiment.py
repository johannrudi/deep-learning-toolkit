"""Define experimental transformer components for sequence-to-vector prediction."""

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from dlk.nets.mlp import MLPNet


class DataEmbedding(nn.Module):
    """Project feature vectors into a transformer embedding space."""

    def __init__(
        self, input_size: int, output_size: int, scale_outputs: bool = False
    ) -> None:
        """Initialize the data embedding layer.

        Args:
            input_size: Size of input feature vectors.
            output_size: Size of output embedding vectors.
            scale_outputs: Whether to scale outputs by ``sqrt(output_size)``.

        Returns:
            None.
        """
        super().__init__()
        self.embedding = nn.Linear(input_size, output_size, bias=False)
        self.output_size = output_size
        self.scale_outputs = scale_outputs

        # initialize parameters
        self.init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input features into embedding coordinates.

        Args:
            x: Input tensor with shape ``(..., input_size)``.

        Returns:
            Output tensor with shape ``(..., output_size)``.
        """
        assert 1 <= x.dim(), f"{x.dim()=}"
        assert (
            x.size(-1) == self.embedding.in_features
        ), f"{x.size(-1)=}, {self.embedding.in_features=}"
        in_size = x.size()
        x = torch.reshape(x, (-1, in_size[-1]))
        y = self.embedding(x)
        if self.scale_outputs:
            y *= math.sqrt(self.output_size)
        y = torch.reshape(y, (*in_size[:-1], self.output_size))
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters.

        Returns:
            None.
        """
        nn.init.orthogonal_(self.embedding.weight)


class PositionalEmbedding(nn.Module):
    """Add sinusoidal positional information to token embeddings."""

    pos_embedding: torch.Tensor

    def __init__(
        self,
        embedding_size: int,
        position_max_size: int = 5000,
        use_dropout: float | bool = False,
    ) -> None:
        """Initialize sinusoidal positional embeddings.

        Args:
            embedding_size: Size of each embedding vector.
            position_max_size: Maximum sequence length supported.
            use_dropout: Dropout probability applied after adding positions.

        Returns:
            None.
        """
        super().__init__()
        assert 0 < embedding_size, f"{embedding_size=}"
        assert 0 < position_max_size <= int(1.0e4), f"{position_max_size=}"

        # compute positional encodings once in log space
        den = torch.exp(
            -torch.arange(0, embedding_size, 2, dtype=torch.float32)
            * math.log(1.0e4)
            / embedding_size
        )
        pos = torch.arange(0, position_max_size, dtype=torch.float32).unsqueeze(1)
        pos_embedding = torch.zeros(
            (position_max_size, embedding_size), dtype=torch.float32
        )
        sin_values = torch.sin(pos * den)
        cos_values = torch.cos(pos * den)
        pos_embedding[:, 0::2] = sin_values
        pos_embedding[:, 1::2] = cos_values[:, : pos_embedding[:, 1::2].shape[1]]
        pos_embedding = pos_embedding.unsqueeze(0)

        # set dropout
        self.dropout: nn.Dropout | None
        if use_dropout:
            self.dropout = nn.Dropout(float(use_dropout))
        else:
            self.dropout = None

        # register buffer
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input embeddings.

        Args:
            x: Tensor with shape ``(batch_size, sequence_length, embedding_size)``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        assert 3 == x.dim(), f"{x.dim()=}"
        assert x.size(-1) == self.pos_embedding.size(
            -1
        ), f"{x.size(-1)=}, {self.pos_embedding.size(-1)=}"
        y = x + self.pos_embedding[:, : x.size(1)]
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class Transformer1d0dModel(nn.Module):
    """Map source sequences to fixed-size targets with a transformer model."""

    def __init__(
        self,
        src_size: int,
        trg_size: int,
        embedding_size: int,
        n_head: int,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        feedforward_size: int = 2048,
        transformer_activation: str | Callable[[torch.Tensor], torch.Tensor] = "relu",
        transformer_kwargs: dict[str, Any] | None = None,
        output_layer_activation: nn.Module | None = None,
        use_dropout: float | bool = False,
    ) -> None:
        """Initialize the transformer model.

        Args:
            src_size: Number of source features per token.
            trg_size: Number of target features per token.
            embedding_size: Size of transformer embeddings.
            n_head: Number of transformer attention heads.
            n_encoder_layers: Number of transformer encoder layers.
            n_decoder_layers: Number of transformer decoder layers.
            feedforward_size: Size of transformer feedforward layers.
            transformer_activation: Activation in transformer feedforward blocks.
            transformer_kwargs: Additional keyword arguments for ``nn.Transformer``.
            output_layer_activation: Optional activation applied at model output.
            use_dropout: Dropout probability used by positional and transformer blocks.

        Returns:
            None.
        """
        super().__init__()
        assert 0 < src_size, f"{src_size=}"
        assert 0 < trg_size, f"{trg_size=}"
        assert 0 < embedding_size, f"{embedding_size=}"
        assert 0 < n_head, f"{n_head=}"
        transformer_kwargs = dict(transformer_kwargs or {})
        self.trg_size = trg_size

        # create embeddings for input data
        self.encoder_emb = DataEmbedding(src_size, embedding_size)
        self.decoder_emb = DataEmbedding(trg_size, embedding_size)

        # create positional embedding for encoder data
        self.encoder_pos_emb = PositionalEmbedding(
            embedding_size, use_dropout=use_dropout
        )

        # create transformer model
        dropout = float(use_dropout) if use_dropout else 0.0
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation=transformer_activation,
            batch_first=True,  # tensors provided as (batch, seq, feature)
            **transformer_kwargs,
        )

        # create output projection model
        self.generator = MLPNet(
            embedding_size,
            trg_size,
            hidden_layers_sizes=(embedding_size,),
            hidden_layers_activation=None,
            output_layer_activation=output_layer_activation,
        )

    def forward(
        self, x_src: torch.Tensor, x_trg: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """Apply the transformer model.

        Args:
            x_src: Source tensor with shape ``(batch_size, src_sequence, src_size)``.
            x_trg: Target tensor with shape ``(batch_size, 1, trg_size)``.
            **kwargs: Additional keyword arguments passed to ``nn.Transformer``.

        Returns:
            Model output with shape ``(batch_size, trg_size)``.
        """
        # create a zero target during inference if target input is omitted
        if x_trg is None:
            x_trg = torch.zeros(
                x_src.size(0),
                1,
                self.trg_size,
                device=x_src.device,
                dtype=x_src.dtype,
            )

        # check input sizes
        assert 3 == x_src.dim(), f"{x_src.dim()=}"
        assert 3 == x_trg.dim(), f"{x_trg.dim()=}"
        assert x_src.size(0) == x_trg.size(0), f"{x_src.size(0)=}, {x_trg.size(0)=}"
        assert 1 == x_trg.size(1), f"{x_trg.size(1)=}"
        assert self.trg_size == x_trg.size(2), f"{self.trg_size=}, {x_trg.size(2)=}"

        # map inputs to transformer embedding space
        x_src_emb = self.encoder_pos_emb(self.encoder_emb(x_src))
        x_trg_emb = self.decoder_emb(x_trg)

        # apply transformer
        h = self.transformer(x_src_emb, x_trg_emb, **kwargs)

        # apply output model
        y = self.generator(h)
        return y

    def encode(self, x_src: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode source inputs with the transformer encoder.

        Args:
            x_src: Source tensor with shape ``(batch_size, src_sequence, src_size)``.
            **kwargs: Additional keyword arguments for ``TransformerEncoder``.

        Returns:
            Encoded source representation.
        """
        assert 3 == x_src.dim(), f"{x_src.dim()=}"
        x_src_emb = self.encoder_pos_emb(self.encoder_emb(x_src))
        return self.transformer.encoder(x_src_emb, **kwargs)

    def decode(
        self, x_trg: torch.Tensor, h_src: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Decode target inputs conditioned on encoded source states.

        Args:
            x_trg: Target tensor with shape ``(batch_size, trg_sequence, trg_size)``.
            h_src: Encoded source representation from ``encode``.
            **kwargs: Additional keyword arguments for ``TransformerDecoder``.

        Returns:
            Decoded target representation.
        """
        assert 3 == x_trg.dim(), f"{x_trg.dim()=}"
        assert 3 == h_src.dim(), f"{h_src.dim()=}"
        x_trg_emb = self.decoder_emb(x_trg)
        return self.transformer.decoder(x_trg_emb, h_src, **kwargs)
