"""Define multilayer perceptron architectures, including residual and attention-based variants."""

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from dlk.nets.utils import get_gain, set_init_parameters

# --------------------------------------
# MLP Nets
# --------------------------------------


class MLPNet(nn.Module):
    """Build a multilayer perceptron with optional activations and dropout.

    Args:
        input_size: Length of flattened input vectors.
        output_size: Length of output vectors.
        input_layer_activation: Optional activation after the input layer.
        input_layer_kwargs: Optional keyword arguments passed to input ``nn.Linear``.
        hidden_layers_sizes: Width of each hidden layer.
        hidden_layers_activation: Optional activation for hidden layers.
        hidden_layers_kwargs: Optional keyword arguments passed to hidden ``nn.Linear``.
        use_dropout: Dropout probability for hidden layers, or ``False`` to disable.
        output_layer_activation: Optional activation after the output layer.
        output_layer_kwargs: Optional keyword arguments passed to output ``nn.Linear``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        input_layer_activation: nn.Module | None = None,
        input_layer_kwargs: dict[str, Any] | None = None,
        hidden_layers_sizes: Sequence[int] = (32, 32, 32, 32),
        hidden_layers_activation: nn.Module | None = nn.ReLU(),
        hidden_layers_kwargs: dict[str, Any] | None = None,
        use_dropout: float | bool = False,
        output_layer_activation: nn.Module | None = None,
        output_layer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # set default layer kwargs
        input_layer_kwargs = dict(input_layer_kwargs or {})
        hidden_layers_kwargs = dict(hidden_layers_kwargs or {})
        output_layer_kwargs = dict(output_layer_kwargs or {})

        # check arguments
        assert 0 < len(hidden_layers_sizes), hidden_layers_sizes

        # create input layer
        self.input_size = input_size
        layer = nn.Linear(input_size, hidden_layers_sizes[0], **input_layer_kwargs)
        if input_layer_activation is not None:
            self.input_layer = nn.Sequential(
                OrderedDict([("layer", layer), ("activation", input_layer_activation)])
            )
        else:
            self.input_layer = layer

        # create hidden layers
        blocks: list[nn.Sequential] = []
        for in_size, out_size in zip(hidden_layers_sizes[:-1], hidden_layers_sizes[1:]):
            block = OrderedDict()
            block["layer"] = nn.Linear(in_size, out_size, **hidden_layers_kwargs)
            if hidden_layers_activation is not None:
                block["activation"] = hidden_layers_activation
            if use_dropout:
                block["dropout"] = nn.Dropout(use_dropout)
            blocks.append(nn.Sequential(block))
        self.hidden_blocks = nn.Sequential(*blocks)

        # create output layer
        layer = nn.Linear(hidden_layers_sizes[-1], output_size, **output_layer_kwargs)
        if output_layer_activation is not None:
            self.output_layer = nn.Sequential(
                OrderedDict([("layer", layer), ("activation", output_layer_activation)])
            )
        else:
            self.output_layer = layer

        # initialize parameters
        self.init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network to an input tensor.

        Args:
            x: Input tensor with shape ``(batch, input_size)`` or higher-rank shape
                that can be flattened over non-batch dimensions.

        Returns:
            Output tensor with shape ``(batch, output_size)``.
        """
        # flatten inputs
        if 2 < x.dim():
            h = torch.flatten(x, 1)
        else:
            h = x
        assert h.size(1) == self.input_size, f"{h.size(1)=}, {self.input_size=}"

        # apply layers
        h = self.input_layer(h)
        h = self.hidden_blocks(h)
        y = self.output_layer(h)
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters of MLPNet with layer-aware gains.

        Returns:
            None.
        """
        # initialize input layer
        if isinstance(self.input_layer, nn.Sequential):
            gain = get_gain(self.input_layer.activation)
            set_init_parameters(cast(nn.Linear, self.input_layer.layer), gain)
        else:
            set_init_parameters(self.input_layer, get_gain(None))

        # initialize hidden layers
        for block in self.hidden_blocks:
            gain = get_gain(getattr(block, "activation", None))
            set_init_parameters(cast(nn.Linear, block.layer), gain)

        # initialize output layer
        if isinstance(self.output_layer, nn.Sequential):
            gain = get_gain(self.output_layer.activation)
            set_init_parameters(
                cast(nn.Linear, self.output_layer.layer), gain, bias_scale=0.0
            )
        else:
            set_init_parameters(self.output_layer, get_gain(None), bias_scale=0.0)


class MLPNet_MultIn(MLPNet):
    """Build an MLP that accepts multiple inputs and optional per-block hidden inputs.

    Args:
        input_size: Total size of flattened input vectors after concatenation.
        output_size: Length of output vectors.
        input_layer_activation: Optional activation after the input layer.
        input_layer_kwargs: Optional keyword arguments passed to input ``nn.Linear``.
        hidden_input_sizes: Extra features concatenated before each hidden block.
        hidden_layers_sizes: Width of each hidden layer.
        hidden_layers_activation: Optional activation for hidden layers.
        hidden_layers_kwargs: Optional keyword arguments passed to hidden ``nn.Linear``.
        use_dropout: Dropout probability for hidden layers, or ``False`` to disable.
        output_layer_activation: Optional activation after the output layer.
        output_layer_kwargs: Optional keyword arguments passed to output ``nn.Linear``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        input_layer_activation: nn.Module | None = None,
        input_layer_kwargs: dict[str, Any] | None = None,
        hidden_input_sizes: Sequence[int] | None = None,
        hidden_layers_sizes: Sequence[int] = (32, 32, 32, 32),
        hidden_layers_activation: nn.Module | None = nn.ReLU(),
        hidden_layers_kwargs: dict[str, Any] | None = None,
        use_dropout: float | bool = False,
        output_layer_activation: nn.Module | None = None,
        output_layer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # set default layer kwargs
        input_layer_kwargs = dict(input_layer_kwargs or {})
        hidden_layers_kwargs = dict(hidden_layers_kwargs or {})
        output_layer_kwargs = dict(output_layer_kwargs or {})

        # set sizes of the hidden layers
        hidden_layers_sizes = list(hidden_layers_sizes)
        if hidden_input_sizes is None:
            hidden_input_sizes = [0] * len(hidden_layers_sizes)
        else:
            hidden_input_sizes = list(hidden_input_sizes)
        if len(hidden_input_sizes) == (len(hidden_layers_sizes) - 1):
            hidden_input_sizes += [0]
        assert len(hidden_input_sizes) == len(hidden_layers_sizes)

        # create regular MLP net with only input and output layers
        super().__init__(
            input_size,
            output_size,
            input_layer_activation=input_layer_activation,
            input_layer_kwargs=input_layer_kwargs,
            hidden_layers_sizes=[
                hidden_layers_sizes[0],
                hidden_layers_sizes[-1] + hidden_input_sizes[-1],
            ],
            hidden_layers_activation=hidden_layers_activation,
            hidden_layers_kwargs=hidden_layers_kwargs,
            use_dropout=use_dropout,
            output_layer_activation=output_layer_activation,
            output_layer_kwargs=output_layer_kwargs,
        )

        # create hidden layer with additional input sizes
        blocks: list[nn.Sequential] = []
        for k, (in_size, out_size) in enumerate(
            zip(hidden_layers_sizes[:-1], hidden_layers_sizes[1:])
        ):
            in_size += hidden_input_sizes[k]
            block = OrderedDict()
            block["layer"] = nn.Linear(in_size, out_size, **hidden_layers_kwargs)
            if hidden_layers_activation is not None:
                block["activation"] = hidden_layers_activation
            if use_dropout:
                block["dropout"] = nn.Dropout(use_dropout)
            blocks.append(nn.Sequential(block))
        self.hidden_blocks = nn.Sequential(*blocks)

        # initialize parameters
        self.init_parameters()

    def forward(self, *x: torch.Tensor, **h_kwargs: torch.Tensor) -> torch.Tensor:
        r"""Apply the network to one or more input tensors.

        Args:
            *x: Positional input tensors concatenated after flattening.
            **h_kwargs: Optional tensors named ``h0``, ``h1``, ... concatenated
                before matching hidden block indices.

        Returns:
            Output tensor with shape ``(batch, output_size)``.
        """
        # concatenate inputs; flatten inputs
        assert 0 < len(x)
        if 1 < len(x):
            h = torch.cat([torch.flatten(x_, 1) for x_ in x], dim=1)
        else:
            h = torch.flatten(x[0], 1)

        # apply input layer
        h = self.input_layer(h)

        # apply hidden layers
        for block_idx, block in enumerate(self.hidden_blocks):
            block_layer = cast(nn.Linear, block.layer)
            h_in = h_kwargs.get(f"h{block_idx}")
            if h_in is not None:
                h = torch.cat((h, torch.flatten(h_in, 1)), dim=1)
            assert (
                h.size(1) == block_layer.in_features
            ), f"{block_idx=}, {h.size(1)=}, {block_layer.in_features=}"
            h = block(h)

        # apply output layer
        y = self.output_layer(h)
        return y


# --------------------------------------
# Residual MLP Net
# --------------------------------------


class SelfAttentionLayer(nn.MultiheadAttention):
    """Wrap multi-head self-attention with a simplified forward interface."""

    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        use_dropout: float | bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a self-attention layer.

        Args:
            embedding_size: Embedding dimension expected by attention.
            n_heads: Number of attention heads.
            use_dropout: Dropout probability for attention weights, or ``False``.
            **kwargs: Additional ``nn.MultiheadAttention`` keyword arguments.

        Returns:
            None.
        """
        if use_dropout:
            dropout = use_dropout
        else:
            dropout = 0.0
        super().__init__(
            embedding_size, n_heads, dropout=dropout, batch_first=True, **kwargs
        )
        # NOTE: initialization of parameters is done in constructor of
        # MultiheadAttention, which is calling _reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to an input sequence.

        Args:
            x: Input tensor with shape ``(batch, sequence, embedding_size)``.

        Returns:
            Attention output tensor with the same shape as ``x``.
        """
        y, _ = super().forward(x, x, x, need_weights=False)
        return y


class AttentionBlock(nn.Module):
    """Build a residual self-attention block with a feed-forward projection.

    Args:
        embedding_size: Input embedding dimension.
        attention_layer_n_heads: Number of heads in self-attention.
        output_embedding_size: Output embedding dimension; defaults to ``embedding_size``.
        activation_layer_size: Width of the feed-forward hidden layer.
        activation: Activation used in the feed-forward layer.
        block_kwargs: Optional keyword arguments passed to ``nn.Linear`` layers.
        use_dropout: Dropout probability used in the feed-forward path, or ``False``.
        use_spectral_norm: Whether to wrap linear layers with spectral normalization.
    """

    def __init__(
        self,
        embedding_size: int,
        attention_layer_n_heads: int,
        output_embedding_size: int | None = None,
        activation_layer_size: int = 128,
        activation: nn.Module | None = nn.ReLU(),
        block_kwargs: dict[str, Any] | None = None,
        use_dropout: float | bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        # set default layer kwargs
        block_kwargs = dict(block_kwargs or {})

        # set from arguments
        self.embedding_size = embedding_size

        # create layers
        in_size = embedding_size
        al_size = activation_layer_size
        out_size = (
            output_embedding_size
            if output_embedding_size is not None
            else embedding_size
        )
        self.output_embedding_size = out_size
        self.attention_layer = SelfAttentionLayer(
            in_size, attention_layer_n_heads, use_dropout=use_dropout, **block_kwargs
        )
        block = OrderedDict()
        block["normalization"] = nn.LayerNorm(in_size)
        block["layer_0"] = nn.Linear(in_size, al_size, **block_kwargs)
        block["activation"] = activation
        if use_dropout:
            block["dropout"] = nn.Dropout(use_dropout)
        block["layer_1"] = nn.Linear(al_size, out_size, **block_kwargs)
        if use_spectral_norm:
            for i in range(2):
                block[f"layer_{i}"] = nn.utils.spectral_norm(block[f"layer_{i}"])
        self.attention_block = nn.Sequential(block)

        # create skip connection
        if in_size != out_size:
            self.skip_connection = nn.Linear(in_size, out_size, **block_kwargs)
            if use_spectral_norm:
                self.skip_connection = nn.utils.spectral_norm(self.skip_connection)
        else:
            self.skip_connection = None

        # initialize parameters
        self.init_parameters()

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        r"""Apply the attention block to one or more tensors.

        Args:
            *x: Input tensors concatenated along feature dimensions.

        Returns:
            Tensor with shape ``(batch, output_embedding_size, input_size)``.
        """
        embed_dim, input_dim = 1, 2

        # flatten and concatenate along input dimension
        if 1 < len(x):
            h_in = torch.cat([torch.flatten(x_, input_dim) for x_ in x], dim=input_dim)
        else:
            assert 1 == len(x)
            h_in = torch.flatten(x[0], input_dim)
        input_size = h_in.size(input_dim)

        # swap embedding and input dimensions
        ht = h_in.transpose(embed_dim, input_dim)

        # apply skip connection
        h_pre_skip = ht.reshape(-1, self.embedding_size)
        if self.skip_connection is not None:
            hs = self.skip_connection(h_pre_skip)
        else:
            hs = h_pre_skip

        # apply attention layer followed by the post-attention block
        ha = self.attention_layer(ht)
        h_pre_block = ha.reshape(-1, self.embedding_size)
        hb = self.attention_block(h_pre_block)

        # compute output
        y = 0.5 * hs + 0.5 * hb
        y = y.reshape(-1, input_size, self.output_embedding_size)

        # swap embedding and input dimensions back
        y = y.transpose(embed_dim, input_dim)
        return y

    def init_parameters(self) -> None:
        r"""Initialize the trainable parameters of AttentionBlock in the feed-forward sub-block.

        Returns:
            None.
        """
        set_init_parameters(
            cast(nn.Linear, self.attention_block.layer_0),
            get_gain(self.attention_block.activation),
        )
        set_init_parameters(
            cast(nn.Linear, self.attention_block.layer_1), get_gain(None)
        )
        if self.skip_connection is not None:
            set_init_parameters(self.skip_connection, get_gain(None))


class ResidualBlock(nn.Module):
    r"""Build a residual dense block with normalization, activation, and optional dropout.

    Args:
        input_size: Length of flattened input vectors.
        output_size: Length of output vectors; defaults to ``input_size``.
        normalization_layer_size: Width of the normalization projection layer.
        activation_layer_size: Width of the feed-forward activation layer.
        activation: Activation applied in the residual branch.
        layer_kwargs: Optional keyword arguments passed to ``nn.Linear`` layers.
        use_dropout: Dropout probability used in the residual branch, or ``False``.
        use_spectral_norm: Whether to wrap linear layers with spectral normalization.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        normalization_layer_size: int = 32,
        activation_layer_size: int = 128,
        activation: nn.Module | None = nn.ReLU(),
        layer_kwargs: dict[str, Any] | None = None,
        use_dropout: float | bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        # set default layer kwargs
        layer_kwargs = dict(layer_kwargs or {})

        # set from arguments
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size

        # create layers
        in_size = self.input_size
        nl_size = normalization_layer_size
        al_size = activation_layer_size
        out_size = self.output_size
        block = OrderedDict()
        block["layer_0"] = nn.Linear(in_size, nl_size, **layer_kwargs)
        block["normalization"] = nn.LayerNorm(nl_size)
        block["layer_1"] = nn.Linear(nl_size, al_size, **layer_kwargs)
        block["activation"] = activation
        if use_dropout:
            block["dropout"] = nn.Dropout(use_dropout)
        block["layer_2"] = nn.Linear(al_size, out_size, **layer_kwargs)
        if use_spectral_norm:
            for i in range(3):
                block[f"layer_{i}"] = nn.utils.spectral_norm(block[f"layer_{i}"])
        self.residual_block = nn.Sequential(block)

        # create skip connection
        if in_size != out_size:
            self.skip_connection = nn.Linear(in_size, out_size, **layer_kwargs)
            if use_spectral_norm:
                self.skip_connection = nn.utils.spectral_norm(self.skip_connection)
        else:
            self.skip_connection = None

        # initialize parameters
        self.init_parameters()

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        r"""Apply the residual block to one or more tensors.

        Args:
            *x: Input tensors concatenated along feature dimensions.

        Returns:
            Tensor with shape ``(batch, embedding, output_size)``.
        """
        # flatten and concatenate along input dimension
        # embed_dim = 1
        input_dim = 2
        if 1 < len(x):
            h = torch.cat([torch.flatten(x_, input_dim) for x_ in x], dim=input_dim)
        else:
            assert 1 == len(x)
            h = torch.flatten(x[0], input_dim)
        assert (
            h.size(input_dim) == self.input_size
        ), f"{h.size(input_dim)=}, {self.input_size=}"

        # combine batch and embedding dimensions for subsequent layers
        batch_size = h.size(0)
        h = h.reshape(-1, self.input_size)

        # apply skip connection
        if self.skip_connection is not None:
            hs = self.skip_connection(h)
        else:
            hs = h

        # apply block
        hb = self.residual_block(h)

        # compute output
        y = 0.5 * hs + 0.5 * hb

        # separate batch and embedding dimensions
        y = y.reshape(batch_size, -1, self.output_size)
        return y

    def init_parameters(self) -> None:
        r"""Initialize trainable parameters of ResidualBlock in the residual and skip layers.

        Returns:
            None.
        """
        set_init_parameters(
            cast(nn.Linear, self.residual_block.layer_0), get_gain(None)
        )
        set_init_parameters(
            cast(nn.Linear, self.residual_block.layer_1),
            get_gain(self.residual_block.activation),
        )
        set_init_parameters(
            cast(nn.Linear, self.residual_block.layer_2), get_gain(None)
        )
        if self.skip_connection is not None:
            set_init_parameters(self.skip_connection, get_gain(None))


class MLPResNet(nn.Module):
    r"""Build an MLP-based residual network with optional attention blocks.

    Args:
        input_size: Input feature size, or ``(input_size, input_layer_out_size)``.
        output_size: Length of output vectors.
        embedding_size: Number of embeddings used across residual stages.
        input_layer_activation: Optional activation after the input layer.
        input_layer_kwargs: Optional keyword arguments passed to input ``nn.Linear``.
        attention_blocks_n_heads: Number of attention heads per residual stage.
        attention_blocks_activation_size: Feed-forward hidden width inside attention blocks.
        attention_blocks_activation: Activation in attention feed-forward layers.
        attention_blocks_kwargs: Optional keyword arguments for attention block linears.
        residual_blocks_sizes: Per-block sizes ``(in_size, norm_size, act_size, [out_size])``.
        residual_blocks_activation: Activation in residual block feed-forward layers.
        residual_blocks_kwargs: Optional keyword arguments for residual block linears.
        use_dropout: Dropout probability, or ``False`` to disable.
        use_spectral_norm: Whether to wrap linear layers with spectral normalization.
        output_layer_activation: Optional activation after the output layer.
        output_layer_kwargs: Optional keyword arguments passed to output ``nn.Linear``.
    """

    def __init__(
        self,
        input_size: int | tuple[int, int],
        output_size: int,
        embedding_size: int = 1,
        input_layer_activation: nn.Module | None = None,
        input_layer_kwargs: dict[str, Any] | None = None,
        attention_blocks_n_heads: Sequence[int] | None = None,
        attention_blocks_activation_size: int = 128,
        attention_blocks_activation: nn.Module | None = nn.ReLU(),
        attention_blocks_kwargs: dict[str, Any] | None = None,
        residual_blocks_sizes: Sequence[Sequence[int]] = (
            (32, 32, 128, 32),
            (32, 32, 128, 32),
            (32, 32, 128, 32),
            (32, 32, 128, 32),
        ),
        residual_blocks_activation: nn.Module | None = nn.ReLU(),
        residual_blocks_kwargs: dict[str, Any] | None = None,
        use_dropout: float | bool = False,
        use_spectral_norm: bool = False,
        output_layer_activation: nn.Module | None = None,
        output_layer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # set default layer kwargs
        input_layer_kwargs = dict(input_layer_kwargs or {})
        attention_blocks_kwargs = dict(attention_blocks_kwargs or {})
        residual_blocks_kwargs = dict(residual_blocks_kwargs or {})
        output_layer_kwargs = dict(output_layer_kwargs or {})
        residual_blocks_sizes = [tuple(sizes) for sizes in residual_blocks_sizes]
        assert 1 <= embedding_size
        assert 0 < len(residual_blocks_sizes)
        assert 0 < len(residual_blocks_sizes[0])

        # set up parameters for [optional] attention layers
        self.embedding_size = embedding_size
        if attention_blocks_n_heads is None:
            attention_blocks_n_heads = [0] * len(residual_blocks_sizes)
        else:
            attention_blocks_n_heads = list(attention_blocks_n_heads)
        assert len(attention_blocks_n_heads) == len(residual_blocks_sizes)

        # create input layer
        if isinstance(input_size, tuple):
            assert len(input_size) == 2, input_size
            self.input_size = input_size[0]
            out_size_ = input_size[1]
        else:
            self.input_size = input_size
            out_size_ = residual_blocks_sizes[0][0]
        layer = nn.Linear(self.input_size, out_size_, **input_layer_kwargs)
        if input_layer_activation is not None:
            self.input_layer = nn.Sequential(
                OrderedDict([("layer", layer), ("activation", input_layer_activation)])
            )
        else:
            self.input_layer = layer
        if 1 < embedding_size:
            self.input_embedding_layer = nn.Linear(
                1, embedding_size, **input_layer_kwargs
            )
        else:
            self.input_embedding_layer = None

        # create residual blocks
        blocks: list[nn.Module] = []
        for i, sizes in enumerate(residual_blocks_sizes):
            assert 3 <= len(sizes)
            block_in_size = sizes[0]
            block_nl_size = sizes[1]
            block_al_size = sizes[2]
            block_out_size = sizes[3] if 3 < len(sizes) else sizes[0]
            if attention_blocks_n_heads[i]:
                blocks.append(
                    AttentionBlock(
                        embedding_size,
                        attention_blocks_n_heads[i],
                        activation_layer_size=attention_blocks_activation_size,
                        activation=attention_blocks_activation,
                        block_kwargs=attention_blocks_kwargs,
                        use_dropout=use_dropout,
                        use_spectral_norm=use_spectral_norm,
                    )
                )
            blocks.append(
                ResidualBlock(
                    block_in_size,
                    output_size=block_out_size,
                    normalization_layer_size=block_nl_size,
                    activation_layer_size=block_al_size,
                    activation=residual_blocks_activation,
                    layer_kwargs=residual_blocks_kwargs,
                    use_dropout=use_dropout,
                    use_spectral_norm=use_spectral_norm,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # create output layer
        if 1 < embedding_size:
            self.output_embedding_layer = nn.Linear(
                embedding_size, 1, **output_layer_kwargs
            )
        else:
            self.output_embedding_layer = None
        in_size_ = block_out_size
        layer = nn.Linear(in_size_, output_size, **output_layer_kwargs)
        if output_layer_activation is not None:
            self.output_layer = nn.Sequential(
                OrderedDict([("layer", layer), ("activation", output_layer_activation)])
            )
        else:
            self.output_layer = layer

        # initialize parameters
        self.init_parameters()

    def forward(self, *x: torch.Tensor, **h_kwargs: torch.Tensor) -> torch.Tensor:
        r"""Apply the residual network to one or more input tensors.

        Args:
            *x: Positional input tensors concatenated after flattening.
            **h_kwargs: Optional tensors named ``h{idx}`` or ``h_all`` supplied
                to residual/attention blocks.

        Returns:
            Output tensor with shape ``(batch, output_size)``.
        """
        # concatenate inputs; flatten inputs
        assert 0 < len(x)
        if 1 < len(x):
            h = torch.cat([torch.flatten(x_, 1) for x_ in x], dim=1)
        else:
            h = torch.flatten(x[0], 1)

        # apply input layer
        batch_size = h.size(0)
        embed_dim, input_dim = 1, 2
        assert h.size(1) == self.input_size, f"{h.size(1)=}, {self.input_size=}"
        h = self.input_layer(h)
        if self.input_embedding_layer is not None:
            h = h.reshape(-1, 1)
            h = self.input_embedding_layer(h)
            h = h.reshape(batch_size, -1, self.embedding_size)
            h = h.transpose(embed_dim, input_dim)
        h = h.reshape(batch_size, self.embedding_size, -1)

        # apply [optional attention and] residual blocks
        for block_idx, block in enumerate(self.blocks):
            block_embed_size = getattr(block, f"embedding_size", None)
            block_input_size = getattr(block, f"input_size", None)
            assert (block_embed_size is None) ^ (block_input_size is None)

            # get hidden input
            h_in = h_kwargs.get(f"h{block_idx}")
            if h_in is None:
                h_in = h_kwargs.get("h_all")

            # apply residual block
            if h_in is not None:
                assert h_in.dim() in [2, 3], f"{h_in.dim()=}"
                assert h_in.size(0) == batch_size, f"{h_in.size(0)=}, {batch_size}"
                if 2 == h_in.dim():
                    h_in = h_in.reshape(batch_size, self.embedding_size, -1)
                assert block_embed_size is None or (
                    block_embed_size == h.size(embed_dim)
                    and block_embed_size == h_in.size(embed_dim)
                ), f"{block_idx=}, {block_embed_size=}, {h.size(input_dim)=}, {sum(h_in.size()[input_dim:])=}"
                assert block_input_size is None or block_input_size == (
                    h.size(input_dim) + sum(h_in.size()[input_dim:])
                ), f"{block_idx=}, {block_input_size=}, {h.size(input_dim)=}, {sum(h_in.size()[input_dim:])=}"
                h = block(h, h_in)
            else:
                assert block_embed_size is None or block_embed_size == h.size(
                    embed_dim
                ), f"{block_idx=}, {block_embed_size=}, {h.size(embed_dim)=}"
                assert block_input_size is None or block_input_size == h.size(
                    input_dim
                ), f"{block_idx=}, {block_input_size=}, {h.size(input_dim)=}"
                h = block(h)

        # apply output layer
        if self.output_embedding_layer is not None:
            h = h.transpose(embed_dim, input_dim)
            h = self.output_embedding_layer(h)
        h = h.reshape(batch_size, -1)
        y = self.output_layer(h)
        return y

    def init_parameters(self) -> None:
        """Initialize trainable parameters of MLPResNet for input and output projections.

        Returns:
            None.
        """
        # initialize input layer
        if isinstance(self.input_layer, nn.Sequential):
            gain = get_gain(self.input_layer.activation)
            set_init_parameters(cast(nn.Linear, self.input_layer.layer), gain)
        else:
            set_init_parameters(self.input_layer, get_gain(None))

        # initialize output layer
        if isinstance(self.output_layer, nn.Sequential):
            gain = get_gain(self.output_layer.activation)
            set_init_parameters(
                cast(nn.Linear, self.output_layer.layer), gain, bias_scale=0.0
            )
        else:
            set_init_parameters(self.output_layer, get_gain(None), bias_scale=0.0)
