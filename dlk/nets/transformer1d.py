"""1D transformer networks for sequence modeling with optional channel-wise attention."""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

# --------------------------------------
# Embeddings
# --------------------------------------


class PatchEmbedding1D(nn.Module):
    """Embed non-overlapping patches from a 1D sequence.

    Args:
        patch_size: Number of time steps in each patch.
        embedding_size: Output embedding size for each patch.
    """

    def __init__(self, patch_size: int, embedding_size: int) -> None:
        super().__init__()
        assert patch_size > 0, f"expected a positive patch size, got {patch_size=}"
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project sequence patches into embeddings.

        Args:
            x: Input tensor with shape `(batch_size, input_seq_size)`.

        Returns:
            Tensor with shape `(batch_size, num_patches, embedding_size)`.
        """
        assert x.ndim == 2, f"expected a 2D input tensor, got shape {tuple(x.shape)}"
        assert (
            x.size(-1) % self.patch_size == 0
        ), f"expected input length divisible by patch_size, got {x.size(-1)=}, {self.patch_size=}"

        batch_size = x.size(0)
        # reshape flat sequences into non-overlapping patches
        patches = x.reshape(batch_size, -1, self.patch_size)
        # project each patch into the shared embedding space
        embeddings = self.projection(patches)
        return embeddings


class ChannelWisePatchEmbedding1D(nn.Module):
    """Embed non-overlapping patches for each input channel independently.

    Args:
        patch_size: Number of time steps in each patch.
        embedding_size: Total embedding size across all channels.
        n_channels: Number of channels in the input tensor.
        combine_channels: Whether to apply a final linear projection after concatenation.
    """

    def __init__(
        self,
        patch_size: int,
        embedding_size: int,
        n_channels: int = 1,
        combine_channels: bool = True,
    ) -> None:
        super().__init__()
        assert patch_size > 0, f"expected a positive patch size, got {patch_size=}"
        assert n_channels > 0, f"expected a positive channel count, got {n_channels=}"
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.combine_channels = combine_channels

        assert (
            embedding_size % n_channels == 0
        ), f"embedding size is not divisible by the number of channels: {embedding_size=}, {n_channels=}"

        self.channel_projections = nn.ModuleList(
            [
                nn.Linear(patch_size, embedding_size // n_channels)
                for _ in range(n_channels)
            ]
        )

        if self.combine_channels:
            self.combine_projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project channel-wise sequence patches into embeddings.

        Args:
            x: Input tensor with shape `(batch_size, n_channels, input_seq_size)`.

        Returns:
            Tensor with shape `(batch_size, num_patches, embedding_size)`.
        """
        assert x.ndim == 3, f"expected a 3D input tensor, got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.n_channels
        ), f"expected {self.n_channels} channels, got {x.size(1)}"
        assert (
            x.size(-1) % self.patch_size == 0
        ), f"expected input length divisible by patch_size, got {x.size(-1)=}, {self.patch_size=}"

        batch_size = x.size(0)
        channel_embeddings = []

        # embed each channel independently before concatenation
        for channel_idx in range(self.n_channels):
            # slice one channel sequence from the input tensor
            channel_data = x[:, channel_idx, :]
            # split channel sequence into non-overlapping patches
            patches = channel_data.reshape(batch_size, -1, self.patch_size)
            # project channel patches with the channel-specific layer
            embedded = self.channel_projections[channel_idx](patches)
            channel_embeddings.append(embedded)

        # concatenate channel embeddings along the feature dimension
        output = torch.cat(channel_embeddings, dim=-1)
        if self.combine_channels:
            # mix features across channels after concatenation
            output = self.combine_projection(output)

        return output


# --------------------------------------
# Attention Blocks
# --------------------------------------


class MultiHeadAttention(nn.Module):
    """Apply scaled dot-product multi-head self-attention.

    Args:
        embedding_size: Input and output embedding size.
        n_heads: Number of attention heads.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(self, embedding_size: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert (
            embedding_size > 0
        ), f"expected positive embedding size, got {embedding_size=}"
        assert n_heads > 0, f"expected a positive number of heads, got {n_heads=}"
        self.embedding_size = embedding_size
        self.head_size = embedding_size // n_heads
        self.n_heads = n_heads

        assert (
            embedding_size % n_heads == 0
        ), f"embedding size is not divisible by the number of heads: {embedding_size=}, {n_heads=}"

        self.qkv = nn.Linear(embedding_size, embedding_size * 3)
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run multi-head self-attention.

        Args:
            x: Input tensor with shape `(batch_size, seq_len, embedding_size)`.

        Returns:
            Tensor with shape `(batch_size, seq_len, embedding_size)`.
        """
        assert x.ndim == 3, f"expected a 3D input tensor, got shape {tuple(x.shape)}"
        assert (
            x.size(-1) == self.embedding_size
        ), f"expected embedding size {self.embedding_size}, got {x.size(-1)}"

        batch_size, seq_len, embedding_size = x.shape

        # compute QKV once, then split into heads for attention
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # compute scaled dot-product attention weights
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_size))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # merge heads back into the original embedding layout
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, embedding_size
        )
        return self.projection(attn_output)


class ChannelWiseMultiHeadAttention(nn.Module):
    """Apply multi-head self-attention separately for each channel embedding block.

    Args:
        embedding_size: Input and output embedding size.
        n_heads_per_channel: Number of attention heads for each channel block.
        n_channels: Number of channel blocks.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        embedding_size: int,
        n_heads_per_channel: int,
        n_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert (
            embedding_size > 0
        ), f"expected positive embedding size, got {embedding_size=}"
        assert (
            n_heads_per_channel > 0
        ), f"expected a positive number of heads, got {n_heads_per_channel=}"
        assert n_channels > 0, f"expected a positive channel count, got {n_channels=}"

        self.embedding_size = embedding_size
        self.channel_embed_size = embedding_size // n_channels
        self.n_heads_per_channel = n_heads_per_channel
        self.head_size_per_channel = self.channel_embed_size // n_heads_per_channel
        self.n_channels = n_channels

        assert (
            embedding_size % n_channels == 0
        ), f"embedding size is not divisible by the number of channels: {embedding_size=}, {n_channels=}"
        assert (
            self.channel_embed_size % n_heads_per_channel == 0
        ), f"embedding size is not divisible by the number of heads: {self.channel_embed_size=}, {n_heads_per_channel=}"

        self.channel_qkv = nn.ModuleList(
            [
                nn.Linear(self.channel_embed_size, self.channel_embed_size * 3)
                for _ in range(n_channels)
            ]
        )
        self.channel_projections = nn.ModuleList(
            [
                nn.Linear(self.channel_embed_size, self.channel_embed_size)
                for _ in range(n_channels)
            ]
        )
        self.combine_projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run channel-wise multi-head self-attention.

        Args:
            x: Input tensor with shape `(batch_size, seq_len, embedding_size)`.

        Returns:
            Tensor with shape `(batch_size, seq_len, embedding_size)`.
        """
        assert x.ndim == 3, f"expected a 3D input tensor, got shape {tuple(x.shape)}"
        assert (
            x.size(-1) == self.embedding_size
        ), f"expected embedding size {self.embedding_size}, got {x.size(-1)}"

        batch_size, seq_len, _ = x.shape
        channel_outputs = []

        # run attention independently on each channel embedding block
        for channel_idx in range(self.n_channels):
            start_idx = channel_idx * self.channel_embed_size
            end_idx = (channel_idx + 1) * self.channel_embed_size
            # isolate the feature slice for the current channel block
            x_channel = x[:, :, start_idx:end_idx]

            # build QKV tensors for the current channel block
            qkv_channel = self.channel_qkv[channel_idx](x_channel)
            qkv_channel = qkv_channel.reshape(
                batch_size,
                seq_len,
                3,
                self.n_heads_per_channel,
                self.head_size_per_channel,
            )
            qkv_channel = qkv_channel.permute(2, 0, 3, 1, 4)
            q_channel, k_channel, v_channel = (
                qkv_channel[0],
                qkv_channel[1],
                qkv_channel[2],
            )

            # compute scaled dot-product attention inside this channel block
            scores_channel = torch.matmul(
                q_channel, k_channel.transpose(-2, -1)
            ) / math.sqrt(float(self.head_size_per_channel))
            attn_weights_channel = torch.softmax(scores_channel, dim=-1)
            attn_weights_channel = self.dropout(attn_weights_channel)
            attn_output_channel = torch.matmul(attn_weights_channel, v_channel)

            # merge channel heads and project to channel embedding size
            attn_output_channel = attn_output_channel.transpose(1, 2).reshape(
                batch_size, seq_len, self.channel_embed_size
            )
            attn_output_channel = self.channel_projections[channel_idx](
                attn_output_channel
            )
            channel_outputs.append(attn_output_channel)

        # concatenate channel outputs and mix them with a final projection
        combined_output = torch.cat(channel_outputs, dim=-1)
        output = self.combine_projection(combined_output)
        return output


# --------------------------------------
# Transformer Nets
# --------------------------------------


class TransformerBlock(nn.Module):
    """Compose pre-norm attention and MLP residual layers.

    Args:
        embedding_size: Input and output embedding size.
        attn_n_heads: Number of heads for the attention block.
        mlp_ratio: Width multiplier for the MLP hidden dimension.
        dropout: Dropout probability for attention and MLP layers.
    """

    def __init__(
        self,
        embedding_size: int,
        attn_n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            MultiHeadAttention(embedding_size, attn_n_heads, dropout),
        )

        activation_size = int(embedding_size * mlp_ratio)
        self.activation_block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, activation_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(activation_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP residual blocks.

        Args:
            x: Input tensor with shape `(batch_size, seq_len, embedding_size)`.

        Returns:
            Tensor with shape `(batch_size, seq_len, embedding_size)`.
        """
        assert x.ndim == 3, f"expected a 3D input tensor, got shape {tuple(x.shape)}"
        # apply the attention residual branch
        hidden = x + self.attention_block(x)
        # apply the MLP residual branch
        output = hidden + self.activation_block(hidden)
        return output


class ChannelWiseTransformerBlock(nn.Module):
    """Compose pre-norm channel-wise attention and MLP residual layers.

    Args:
        embedding_size: Input and output embedding size.
        attn_n_heads: Number of attention heads per channel block.
        n_channels: Number of channel blocks.
        mlp_ratio: Width multiplier for the MLP hidden dimension.
        dropout: Dropout probability for attention and MLP layers.
    """

    def __init__(
        self,
        embedding_size: int,
        attn_n_heads: int,
        n_channels: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            ChannelWiseMultiHeadAttention(
                embedding_size, attn_n_heads, n_channels, dropout
            ),
        )

        activation_size = int(embedding_size * mlp_ratio)
        self.activation_block = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, activation_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(activation_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise attention and MLP residual blocks.

        Args:
            x: Input tensor with shape `(batch_size, seq_len, embedding_size)`.

        Returns:
            Tensor with shape `(batch_size, seq_len, embedding_size)`.
        """
        assert x.ndim == 3, f"expected a 3D input tensor, got shape {tuple(x.shape)}"
        # apply the channel-wise attention residual branch
        hidden = x + self.attention_block(x)
        # apply the MLP residual branch
        output = hidden + self.activation_block(hidden)
        return output


class TransformerNet(nn.Module):
    """Apply a ViT-style transformer to single-channel 1D sequences.

    Args:
        input_seq_size: Length of each input sequence.
        output_size: Output feature dimension.
        patch_size: Number of time steps in each patch.
        embedding_size: Embedding size used throughout the network.
        attn_n_heads: Attention head counts for each transformer block.
        dropout: Dropout probability used across transformer layers.
    """

    def __init__(
        self,
        input_seq_size: int,
        output_size: int,
        patch_size: int = 50,
        embedding_size: int = 256,
        attn_n_heads: Sequence[int] = (8, 8, 8, 8, 8, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert patch_size > 0, f"expected a positive patch size, got {patch_size=}"
        assert (
            input_seq_size % patch_size == 0
        ), f"expected input_seq_size divisible by patch_size, got {input_seq_size=}, {patch_size=}"
        assert attn_n_heads, "expected at least one transformer block"

        self.input_seq_size = input_seq_size

        self.patch_embed = PatchEmbedding1D(patch_size, embedding_size)
        num_patches = input_seq_size // patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_size) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(embedding_size, n_heads, dropout=dropout)
                for n_heads in attn_n_heads
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.head = nn.Linear(embedding_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer and return sequence-level predictions.

        Args:
            x: Input tensor with shape `(batch_size, input_seq_size)` or
                `(batch_size, 1, input_seq_size)`.

        Returns:
            Tensor with shape `(batch_size, output_size)`.
        """
        if x.ndim == 3:
            assert x.size(1) == 1, f"expected one channel in 3D input, got {x.size(1)=}"
            # squeeze singleton channel inputs to the 2D sequence format
            x = x.squeeze(1)

        assert x.ndim == 2, f"expected a 2D input tensor, got shape {tuple(x.shape)}"
        assert (
            x.size(-1) == self.input_seq_size
        ), f"expected input sequence length {self.input_seq_size}, got {x.size(-1)}"

        batch_size = x.shape[0]
        # convert the sequence into patch tokens
        x = self.patch_embed(x)
        # prepend the class token used for sequence-level prediction
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        assert x.size(1) == self.pos_embed.size(
            1
        ), f"positional embedding length mismatch: {x.size(1)=}, {self.pos_embed.size(1)=}"

        # add positional context before transformer processing
        x = x + self.pos_embed
        x = self.dropout(x)
        # process tokens through stacked transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)
        x = self.norm(x)

        # use the final class token embedding for prediction
        cls_token_final = x[:, 0]
        output = self.head(cls_token_final)
        return output


class ChannelWiseTransformerNet(nn.Module):
    """Apply a transformer with channel-wise embedding and attention.

    Args:
        input_channels: Number of input channels.
        input_seq_size: Length of each input sequence.
        output_size: Output feature dimension.
        patch_size: Number of time steps in each patch.
        embedding_size: Embedding size used throughout the network.
        attn_n_heads: Attention head counts for each transformer block.
        dropout: Dropout probability used across transformer layers.
    """

    def __init__(
        self,
        input_channels: int,
        input_seq_size: int,
        output_size: int,
        patch_size: int = 50,
        embedding_size: int = 256,
        attn_n_heads: Sequence[int] = (8, 8, 8, 8, 8, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert (
            input_channels > 0
        ), f"expected a positive channel count, got {input_channels=}"
        assert patch_size > 0, f"expected a positive patch size, got {patch_size=}"
        assert (
            input_seq_size % patch_size == 0
        ), f"expected input_seq_size divisible by patch_size, got {input_seq_size=}, {patch_size=}"
        assert attn_n_heads, "expected at least one transformer block"
        assert (
            embedding_size % input_channels == 0
        ), f"embedding size is not divisible by the number of channels: {embedding_size=}, {input_channels=}"

        self.input_channels = input_channels
        self.input_seq_size = input_seq_size

        self.patch_embed = ChannelWisePatchEmbedding1D(
            patch_size, embedding_size, input_channels
        )
        num_patches = input_seq_size // patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_size) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

        self.transformer = nn.ModuleList(
            [
                ChannelWiseTransformerBlock(
                    embedding_size, n_heads, input_channels, dropout=dropout
                )
                for n_heads in attn_n_heads
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.head = nn.Linear(embedding_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the channel-wise transformer and return sequence-level predictions.

        Args:
            x: Input tensor with shape `(batch_size, input_channels, input_seq_size)`.

        Returns:
            Tensor with shape `(batch_size, output_size)`.
        """
        assert x.ndim == 3, f"expected a 3D input tensor, got shape {tuple(x.shape)}"
        assert (
            x.size(1) == self.input_channels
        ), f"expected {self.input_channels} channels, got {x.size(1)}"
        assert (
            x.size(-1) == self.input_seq_size
        ), f"expected input sequence length {self.input_seq_size}, got {x.size(-1)}"

        batch_size = x.shape[0]
        # convert each channel into patch embeddings, then concatenate
        x = self.patch_embed(x)
        # prepend the class token used for sequence-level prediction
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        assert x.size(1) == self.pos_embed.size(
            1
        ), f"positional embedding length mismatch: {x.size(1)=}, {self.pos_embed.size(1)=}"

        # add positional context before transformer processing
        x = x + self.pos_embed
        x = self.dropout(x)
        # process tokens through stacked channel-wise transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)
        x = self.norm(x)

        # use the final class token embedding for prediction
        cls_token_final = x[:, 0]
        output = self.head(cls_token_final)
        return output
