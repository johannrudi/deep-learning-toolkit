"""
UNet
"""

from abc import abstractmethod

import math
import torch
import torch.nn as nn

import dlkit.nets.conv1d
import dlkit.nets.conv2d

###############################################################################
# UNet (2025)
###############################################################################

def _create_level_blocks(input_channels_all, output_channels_all, kernel_size,
                         activation=None, dropout=None,
                         with_LevelBlock=None, with_Normalization=None, **conv_kwargs):
    block = list()
    for ch_in, ch_out in zip(input_channels_all, output_channels_all):
        block.append(
            with_LevelBlock(ch_in, ch_out, kernel_size, **conv_kwargs,
                            normalization=with_Normalization(ch_out),
                            activation=activation,
                            dropout=dropout)
        )
    return block


class UNetXd_2025(nn.Module):
    r"""
    UNet.

    Args:
        input_channels:  number of channels of inputs
        output_channels: number of channels of outputs
    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 input_conv_kernels=3,
                 input_conv_kwargs={},
                 down_levels_conv_channels=[2, 4, 8],
                 down_levels_conv_kernels=3,
                 down_levels_conv_kwargs={},
                 coarse_level_conv_channels=[16, 16],
                 coarse_level_conv_kernels=3,
                 coarse_level_conv_kwargs={},
                 up_levels_conv_channels=[8, 4, 2],
                 up_levels_conv_kernels=3,
                 up_levels_conv_kwargs={},
                 output_conv_kernels=3,
                 output_conv_kwargs={},
                 hidden_layers_activation=nn.ReLU(),
                 output_activation=None,
                 use_dropout=False,
                 # dimension dependent classes
                 with_Downsample=None,
                 with_Upsample=None,
                 with_LevelBlock=None,
                 with_Normalization=None):
        super().__init__()
        # check dimension dependent classes
        assert with_Downsample is not None
        assert with_Upsample is not None
        assert with_LevelBlock is not None
        assert with_Normalization is not None
        # check channels
        assert isinstance(down_levels_conv_channels , list), type(down_levels_conv_channels)
        assert isinstance(up_levels_conv_channels   , list), type(up_levels_conv_channels)
        assert isinstance(coarse_level_conv_channels, list), type(coarse_level_conv_channels)
        for l, channels in enumerate(down_levels_conv_channels):
            if not isinstance(channels, list):
                down_levels_conv_channels[l] = [channels]
        for l, channels in enumerate(up_levels_conv_channels):
            if not isinstance(channels, list):
                up_levels_conv_channels[l] = [channels]
        # set number of layers
        assert len(down_levels_conv_channels) == len(up_levels_conv_channels)
        n_levels = len(down_levels_conv_channels)
        # check kernels
        if isinstance(down_levels_conv_kernels, list):
            assert len(down_levels_conv_kernels) == n_levels
        else: # otherwise assume single integer
            down_levels_conv_kernels = [down_levels_conv_kernels] * n_levels
        if isinstance(up_levels_conv_kernels, list):
            assert len(up_levels_conv_kernels) == n_levels
        else: # otherwise assume single integer
            up_levels_conv_kernels = [up_levels_conv_kernels] * n_levels
        assert isinstance(coarse_level_conv_kernels, int), type(coarse_level_conv_kernels)
        # set from arguments
        self.input_channels  = input_channels
        self.output_channels = output_channels
        if use_dropout:
            dropout = nn.Dropout(use_dropout)
        else:
            dropout = None
        #TODO manage print statements
        print(f"### {down_levels_conv_channels=}")
        print(f"### {coarse_level_conv_channels=}")
        print(f"### {up_levels_conv_channels=}")
        _indent = ''
        #
        # create input block
        #
        ch_in  = input_channels
        ch_out = down_levels_conv_channels[0][0]
        print(f"###{_indent} input {ch_in=}, {ch_out=}")
        self.input_block = with_LevelBlock(ch_in, ch_out, input_conv_kernels, **input_conv_kwargs,
                                           normalization=with_Normalization(ch_out),
                                           activation=hidden_layers_activation,
                                           dropout=dropout)
        ch_in = ch_out
        #
        # create downsample levels
        #
        ch_down_all = list()  # initialize channels of all layers (incl. downsample)
        self.down_levels = nn.ModuleList()
        for l, (channels, kernel_size) in enumerate(zip(down_levels_conv_channels, down_levels_conv_kernels)):
            _indent = l*'  '
            level = list()
            # add sequence of layers
            ch_in_all  = [ch_in] + channels[:-1]   # input channels of all layers
            ch_out_all = channels                  # output channels of all layers
            for ci_, co_ in zip(ch_in_all, ch_out_all):
                print(f"###{_indent} down       level={l}, ch_in={ci_}, ch_out={co_}")
            level.extend(
                _create_level_blocks(
                    ch_in_all, ch_out_all, kernel_size,
                    activation=hidden_layers_activation,
                    dropout=dropout,
                    with_LevelBlock=with_LevelBlock,
                    with_Normalization=with_Normalization,
                    **down_levels_conv_kwargs)
            )
            ch_in = ch_out_all[-1]
            # add downsample layer
            # <code id="v1">
            #if l < n_levels - 1:  # if not the last level
            # </code
            # <code id="v2">
            if True:
            # </code
                ch_out = ch_in
                print(f"###{_indent} downsample level={l}, {ch_in=}, {ch_out=}")
                layer = with_Downsample(ch_in, ch_out, kernel_size)
                level.append(layer)
            # add this down level
            self.down_levels.append(nn.ModuleList(level))
        #
        # create coarse level
        #
        _indent = (l+1)*'  '
        ch_in_all  = [ch_in] + coarse_level_conv_channels[:-1]  # input channels of all layers
        ch_out_all = coarse_level_conv_channels                 # output channels of all layers
        for ci_, co_ in zip(ch_in_all, ch_out_all):
            print(f"###{_indent} coarse     level={l+1}, ch_in={ci_}, ch_out={co_}")
        kernel_size = coarse_level_conv_kernels
        level = _create_level_blocks(
                ch_in_all, ch_out_all, kernel_size,
                activation=hidden_layers_activation,
                dropout=dropout,
                with_LevelBlock=with_LevelBlock,
                with_Normalization=with_Normalization,
                **coarse_level_conv_kwargs)
        self.coarse_level = nn.Sequential(*level)
        ch_in = ch_out_all[-1]
        #
        # create upsample levels
        #
        self.up_levels = nn.ModuleList()
        for l_inv, (channels, kernel_size) in enumerate(zip(up_levels_conv_channels, up_levels_conv_kernels)):
            l = n_levels - 1 - l_inv
            _indent = l*'  '
            level = list()
            # set channels of corresponding down level `l`
            ch_down_inv = list(reversed(down_levels_conv_channels[l]))
            # add upsample layer
            # <code id="v1">
            #if l < n_levels - 1:  # if not the last level
            # </code
            # <code id="v2">
            if True:
            # </code
                ch_in += ch_down_inv[0]
                ch_out = channels[0]
                print(f"###{_indent} upsample   level={l}, {ch_in=} {ch_out=}")
                layer = nn.Sequential(
                        with_LevelBlock(ch_in, ch_out, kernel_size, **up_levels_conv_kwargs,
                                        normalization=with_Normalization(ch_out),
                                        activation=hidden_layers_activation,
                                        dropout=dropout),
                        with_Upsample(ch_out, ch_out, kernel_size)
                )
                level.append(layer)
                ch_in = ch_out
            # add sequence of layers
            ch_in_all  = [ch_in] + channels[:-1]                        # input channels of all layers
            ch_in_all  = [sum(c) for c in zip(ch_in_all, ch_down_inv)]  # add channels of down level
            ch_out_all = channels                                       # output channels of all layers
            for ci_, co_ in zip(ch_in_all, ch_out_all):
                print(f"###{_indent} up         level={l}, ch_in={ci_}, ch_out={co_}")
            level.extend(
                _create_level_blocks(
                    ch_in_all, ch_out_all, kernel_size,
                    activation=hidden_layers_activation,
                    dropout=dropout,
                    with_LevelBlock=with_LevelBlock,
                    with_Normalization=with_Normalization,
                    **up_levels_conv_kwargs)
            )
            ch_in = ch_out_all[-1]
            # add this up level
            self.up_levels.append(nn.ModuleList(level))
        #
        # create output block
        #
        ch_out = output_channels
        print(f"###{_indent} output {ch_in=}, {ch_out=}")
        self.output_block = with_LevelBlock(ch_in, ch_out, output_conv_kernels, **output_conv_kwargs,
                                            normalization=with_Normalization(ch_out),
                                            activation=output_activation,
                                            dropout=dropout)
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the network's forward function: y = net(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels
        # input layer
        h = self.input_block(x)
        # downsample levels
        h_down = list()
        for level in self.down_levels:
            for block in level:
                h = block(h)
                h_down.append(h)
        # coarse level
        h = self.coarse_level(h)
        # upsample levels
        for level in self.up_levels:
            for block in level:
                h_cat = torch.cat([h, h_down.pop()], dim=1)  # concatenate along channel dimension
                h = block(h_cat)
        # output layer
        y = self.output_block(h)
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # input layer
        self.input_block.init_parameters()
        # downsample levels
        for level in self.down_levels:
            for block in level:
                try:
                    block.init_parameters()
                except AttributeError:
                    for layer in block:
                        layer.init_parameters()
        # coarse level
        try:
            self.coarse_level.init_parameters()
        except AttributeError:
            for block in self.coarse_level:
                block.init_parameters()
        # upsample levels
        for level in self.up_levels:
            for block in level:
                try:
                    block.init_parameters()
                except AttributeError:
                    for layer in block:
                        layer.init_parameters()
        # output layer
        self.output_block.init_parameters()

########################################

class UNet2d_2025(UNetXd_2025):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_Downsample    = dlkit.nets.conv2d.Downsample,
                         with_Upsample      = dlkit.nets.conv2d.Upsample,
                         with_LevelBlock    = dlkit.nets.conv2d.LevelBlock,
                         with_Normalization = dlkit.nets.conv2d.Normalization)

###############################################################################
# UNet (2021) <https://github.com/openai/improved-diffusion>
###############################################################################

def _zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# TODO use Downsample in conv2d.py
class Downsample(nn.Module):
    """
    A downsampling layer with a convolution.

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        stride = 2
        self.op = nn.Conv2d(channels, channels, 3, stride=stride,
                            padding=1, padding_mode='replicate')

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# TODO use Upsample in conv2d.py
class Upsample(nn.Module):
    """
    An upsampling layer with a convolution.

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3,
                              padding=1, padding_mode='replicate')

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """


class EmbedBlockSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                assert emb is not None
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# TODO use LevelBlock in conv2d.py
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param output_channels: if specified, the number of out channels.
    :param use_conv: if True and output_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    """

    def __init__(
        self,
        channels,
        output_channels=None,
        use_conv=False,
        normalization=None,
    ):
        super().__init__()
        self.channels = channels
        self.output_channels = output_channels or channels
        self.use_conv = use_conv
        # create input layers
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.output_channels, 3,
                      padding=1, padding_mode='replicate')
        )
        # create output layers
        self.out_layers = nn.Sequential(
            normalization(self.output_channels),
            nn.SiLU(),
            _zero_module(nn.Conv2d(self.output_channels, self.output_channels, 3,
                                   padding=1, padding_mode='replicate')),
        )
        # create skip connection
        if self.output_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.output_channels, 3,
                                             padding=1, padding_mode='replicate')
        else:
            self.skip_connection = nn.Conv2d(channels, self.output_channels, 3,
                                             padding=1, padding_mode='replicate')

    def forward(self, x):
        """
        Apply the block to a Tensor.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class ResBlock_EmbedBlock(ResBlock, EmbedBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param output_channels: if specified, the number of out channels.
    :param use_conv: if True and output_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    """
    def __init__(
        self,
        channels,
        emb_channels=None,
        output_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        normalization=None,
    ):
        super().__init__(
            channels,
            output_channels=output_channels,
            use_conv=use_conv,
            normalization=normalization
        )
        self.emb_channels = emb_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        # create embedding layers
        if use_scale_shift_norm:
            emb_out_channels = 2 * self.output_channels
        else:
            emb_out_channels = self.output_channels
        if emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, emb_out_channels),
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # apply input layers
        h = self.in_layers(x)
        # apply embedding layers
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # apply output layers
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        # apply skip connection
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(self, channels, num_heads=1, normalization=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = _zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])


class UNetXd_2021_idd(nn.Module):
    r"""
    UNet.

    Args:
        input_channels:  number of channels of inputs
        output_channels: number of channels of outputs
        model_channels: base channel count for the model.
        num_res_blocks: number of residual blocks per downsample.
        attention_resolutions: a collection of downsample rates at which
            attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling, attention
            will be used.

        channel_mult: channel multiplier for each level of the UNet.
        num_classes: if specified (as an int), then this model will be
            class-conditional with `num_classes` classes.
        num_heads: the number of attention heads in each attention layer.
    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 model_channels,
                 num_res_blocks=1,
                 attention_resolutions=[],
                 channel_mult=(1, 2, 4, 8),
                 num_classes=None,
                 num_heads=1,
                 num_heads_upsample=-1,
                 time_embed_dim=None, # e.g., model_channels*4
                 # dimension dependent classes
                 with_Downsample=None,
                 with_Upsample=None,
                 with_LevelBlock=None,
                 with_Normalization=None):
        super().__init__()
        # check dimension dependent classes
        assert with_Downsample is not None
        assert with_Upsample is not None
        assert with_LevelBlock is not None
        assert with_Normalization is not None
        # set from arguments
        self.input_channels  = input_channels
        self.output_channels = output_channels
        self.model_channels  = model_channels
        self.num_res_blocks        = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_classes  = num_classes
        self.num_heads    = num_heads
        if num_heads_upsample == -1:
            self.num_heads_upsample = num_heads
        else:
            self.num_heads_upsample = num_heads_upsample
        # create time embedding layers
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        # create label embedding layers
        if self.num_classes is not None and time_embed_dim is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        #
        # create downsample blocks
        #
        input_layer = EmbedBlockSequential(
                nn.Conv2d(input_channels, model_channels, 3,
                          padding=1, padding_mode='replicate')
        )
        self.input_blocks = nn.ModuleList([input_layer])
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch,
                             output_channels=mult * model_channels,
                             normalization=with_Normalization)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=self.num_heads,
                                       normalization=with_Normalization)
                    )
                self.input_blocks.append(EmbedBlockSequential(*layers))
                input_block_channels.append(ch)
            if level != len(channel_mult) - 1:
                down_layer = EmbedBlockSequential(Downsample(ch))
                self.input_blocks.append(down_layer)
                input_block_channels.append(ch)
                ds *= 2
        #
        # create middle block
        #
        layers = [
            ResBlock(ch, normalization=with_Normalization),
            AttentionBlock(ch, num_heads=self.num_heads,
                           normalization=with_Normalization),
            ResBlock(ch, normalization=with_Normalization),
        ]
        self.middle_block = EmbedBlockSequential(*layers)
        #
        # create upsample blocks
        #
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(ch + input_block_channels.pop(),
                             output_channels=model_channels * mult,
                             normalization=with_Normalization)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=self.num_heads_upsample,
                                       normalization=with_Normalization)
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(EmbedBlockSequential(*layers))
        #
        # create output layer
        #
        self.output_layer = nn.Sequential(
            with_Normalization(ch),
            nn.SiLU(),
            _zero_module(nn.Conv2d(model_channels, output_channels, 3,
                                   padding=1, padding_mode='replicate')),
        )

    def forward(self, x, timesteps=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (timesteps is not None) == (
            self.time_embed_dim is not None
        ), "must specify timestep if and only if the model has time embedding"
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        # apply embedding layers
        if self.time_embed_dim is not None:
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
            if self.num_classes is not None:
                assert y.shape == (x.shape[0],)
                emb = emb + self.label_emb(y)
        else:
            emb = None
        # apply layers
        hs = []
        h = x.type(self.inner_dtype)
        for block in self.input_blocks:
            h = block(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for block in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = block(cat_in, emb)
        h = h.type(x.dtype)
        return self.output_layer(h)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

########################################

class UNet2d_2021_idd(UNetXd_2021_idd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_Downsample    = dlkit.nets.conv2d.Downsample,
                         with_Upsample      = dlkit.nets.conv2d.Upsample,
                         with_LevelBlock    = dlkit.nets.conv2d.LevelBlock,
                         with_Normalization = dlkit.nets.conv2d.Normalization)

###############################################################################

# TODO use doxygen for these test

def test_UNet_2021_idd():
    print('---------------------------------------^')
    net = UNet2d_2021_idd(1, 1, 4, channel_mult=(1, 2, 4))
    print(net)

    print('Test 1:')
    row = 16*[1.]
    x = torch.tensor([[ [row for _ in range(16)] ]])
    y = net(x)
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')


def test_UNet_2025():
    print('---------------------------------------^')
    net = UNet2d_2025(1, 1,
                      down_levels_conv_channels = [[2,2], [4,4], [8,8]],
                      up_levels_conv_channels   = [[8,8], [4,4], [2,2]],
                      with_Normalization=dlkit.nets.conv2d.Normalization,
    )
    print(net)

    print('Test 1:')
    row = 16*[1.]
    x = torch.tensor([[ [row for _ in range(16)] ]])
    y = net(x)
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')


if __name__ == '__main__':
    r"""Runs tests."""
    test_UNet_2021_idd()
    test_UNet_2025()
