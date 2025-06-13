"""
Networks with 1D transformer layers.
"""

import math
import torch
import torch.nn as nn

# --------------------------------------
# Embeddings
# --------------------------------------

class PatchEmbedding1D(nn.Module):
    """Convert 1D time series into patches and embed them"""
    def __init__(self, input_seq_size, patch_size, embedding_size):
        super().__init__()
        self.input_seq_size = input_seq_size
        self.num_patches    = input_seq_size // patch_size
        self.patch_size     = patch_size

        # Linear projection to embed each patch
        self.projection = nn.Linear(patch_size, embedding_size)

    def forward(self, x):
        # x size: (batch_size, input_seq_size)
        batch_size = x.size(0)

        # Reshape into patches: (batch_size, num_patches, patch_size)
        x = x.view(batch_size, self.num_patches, self.patch_size)

        # Project each patch to embedding dimension
        x = self.projection(x)  # -> (batch_size, num_patches, embedding_size)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, n_heads, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.head_size      = embedding_size // n_heads
        self.n_heads        = n_heads
        assert embedding_size % n_heads == 0, \
               f"embedding size is not divisible by the number of heads: {embedding_size=}, {n_heads=}"

        self.qkv = nn.Linear(embedding_size, embedding_size * 3)
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, input_seq_size, embedding_size = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, input_seq_size, 3, self.n_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, input_seq_size, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, input_seq_size, embedding_size)

        return self.projection(attn_output)

# --------------------------------------
# Transformer Nets
# --------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, attn_n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.attn  = MultiHeadAttention(embedding_size, attn_n_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_size)

        mlp_dim = int(embedding_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        h = x + self.attn(self.norm1(x))

        # MLP with residual connection
        y = h + self.mlp(self.norm2(h))
        return y

class TransformerNet(nn.Module):
    """ Transformer network derived from ViT architecture. """

    def __init__(
        self,
        input_seq_size=1000,    # Length of input time series
        patch_size=50,          # Size of each patch
        embedding_size=256,     # Embedding dimension
        attn_n_heads=8,         # Number of attention heads
        num_layers=6,           # Number of transformer blocks
        output_size=2,          # Output dimension
        dropout=0.1
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding1D(input_seq_size, patch_size, embedding_size)
        num_patches = input_seq_size // patch_size

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embedding_size) * 0.02)

        # Class token (like in original ViT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(embedding_size, attn_n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embedding_size)

        # Classification head - outputs 2D vector
        self.head = nn.Linear(embedding_size, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_seq_size)
        batch_size = x.shape[0]

        # Create patches and embed
        x = self.patch_embed(x)  # (batch_size, num_patches, embedding_size)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches + 1, embedding_size)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)

        # Layer norm
        x = self.norm(x)

        # Use class token for final prediction
        cls_token_final = x[:, 0]  # (batch_size, embedding_size)

        # Output 2D vector
        output = self.head(cls_token_final)  # (batch_size, 2)

        return output

# --------------------------------------
# Tests
# --------------------------------------

# TODO use doxygen for these test

def test_TransformerNet():
    print('---------------------------------------^')
    batch_size     = 4
    input_channels = 1
    input_size     = 1000
    output_size    = 2

    net = TransformerNet(
            input_seq_size = input_size,
            patch_size     = 50,
            embedding_size = 256,
            attn_n_heads   = 8,
            num_layers     = 6,
            output_size    = output_size
    )
    print(net)
    print('Total number of parameters =', sum(p.numel() for p in net.parameters()))

    print('Test 1:')
    x = torch.ones((batch_size, input_channels, input_size))
    y = net(x)
    print('- input  x.size() =', x.size())
    print('- output y.size() =', y.size())
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')

if __name__ == "__main__":
    r"""Runs tests."""
    test_TransformerNet()
