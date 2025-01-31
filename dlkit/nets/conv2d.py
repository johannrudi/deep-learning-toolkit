"""
Networks with 2D convolutional layers.
"""

import math
import torch
import torch.nn as nn

#TODO rename "upscale/upscaling" -> "upsample/upsampling"

class ConvUpsampleNet_Reshuffle(nn.Module):
    r"""
    Convolutional network for upsampling via reshuffling (or depth-to-space).

    Args:
        input_channels: number of channels
        scale_factor:   factor by which the coarse input gets scaled to produce fine outputs
    """
    def __init__(self,
                 input_channels,
                 scale_factor,
                 pre_up_layers_channels=[8, 16, 32],
                 pre_up_layers_kernels=[3, 3, 3],
                 pre_up_layers_activation=nn.ReLU(),
                 pre_up_layers_kwargs={},
                 post_up_layers_channels=[],
                 post_up_layers_kernels=[],
                 post_up_layers_activation=nn.ReLU(),
                 post_up_layers_kwargs={},
                 output_activation=None,
                 use_dropout=False):
        super().__init__()
        # set from arguments
        self.input_channels            = input_channels
        self.scale_factor              = scale_factor
        self.pre_up_layers_activation  = pre_up_layers_activation
        self.post_up_layers_activation = post_up_layers_activation
        self.output_activation         = output_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers before upsampling
        assert len(pre_up_layers_channels) == len(pre_up_layers_kernels)
        in_channels = input_channels
        self.pre_up_layers = nn.ModuleList()
        for channels, kernel_size in zip(pre_up_layers_channels, pre_up_layers_kernels):
            out_channels = channels
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, **pre_up_layers_kwargs,
                              padding=1, padding_mode='replicate')
            self.pre_up_layers.append(layer)
            in_channels = out_channels
        # create upsample layer
        self.up_layer = nn.Conv2d(in_channels, scale_factor**2, kernel_size, **pre_up_layers_kwargs,
                                  padding=1, padding_mode='replicate')
        # create hidden convolutional layers after upsampling
        assert len(post_up_layers_channels) == len(post_up_layers_kernels)
        in_channels = input_channels
        self.post_up_layers = nn.ModuleList()
        for channels, kernel_size in zip(post_up_layers_channels, post_up_layers_kernels):
            out_channels = channels
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, **post_up_layers_kwargs,
                              padding=1, padding_mode='replicate')
            self.post_up_layers.append(layer)
            in_channels = out_channels
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the network's forward function: y = net(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels
        h = x
        # apply hidden convolutional layers before upsampling
        for layer in self.pre_up_layers:
            h = layer(h)
            if self.pre_up_layers_activation is not None:
                h = self.pre_up_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply upsample layer
        h = self.up_layer(h)
        h = nn.functional.pixel_shuffle(h, self.scale_factor)
        # apply hidden convolutional layers after upsampling
        for layer in self.post_up_layers:
            if self.post_up_layers_activation is not None:
                h = self.post_up_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
            h = layer(h)
        # apply output activation
        if self.output_activation is not None:
            y = self.output_activation(h)
        else:
            y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden convolutional layers before upsampling
        gain = get_gain(self.pre_up_layers_activation)
        for layer in self.pre_up_layers:
            set_init_parameters(layer, gain)
        # initialize hidden convolutional layers after upsampling
        if self.post_up_layers:
            gain = get_gain(self.post_up_layers_activation)
            set_init_parameters(self.up_layer, gain)
            for layer in self.post_up_layers[:-1]:
                set_init_parameters(layer, gain)
            set_init_parameters(self.post_up_layers[-1], get_gain(None))
        else:
            set_init_parameters(self.up_layer, get_gain(None))


class ConvUpsampleNet_Interpolate(nn.Module):
    r"""
    Convolutional network for upsampling via interpolation.

    Args:
        input_channels: number of channels
    """
    def __init__(self,
                 input_channels,
                 conv_layers_channels=[4, 4, 1],
                 conv_layers_kernels=[3, 3, 3],
                 conv_layers_activation=nn.ReLU(),
                 conv_layers_kwargs={},
                 upsample_layer_indices=[1],
                 scale_factor=2,
                 interp_mode='nearest',
                 output_activation=None,
                 use_dropout=False):
        super().__init__()
        # set from arguments
        self.input_channels          = input_channels
        self.upsample_layer_indices  = upsample_layer_indices
        self.scale_factor            = scale_factor
        self.interp_mode             = interp_mode
        self.conv_layers_activation  = conv_layers_activation
        self.output_activation       = output_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers
        assert len(conv_layers_channels) == len(conv_layers_kernels)
        in_channels = input_channels
        self.conv_layers = nn.ModuleList()
        for channels, kernel_size in zip(conv_layers_channels, conv_layers_kernels):
            out_channels = channels
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_layers_kwargs,
                              padding=1, padding_mode='replicate')
            self.conv_layers.append(layer)
            in_channels = out_channels
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the network's forward function: y = net(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels
        h = x
        # apply hidden convolutional layers
        for i, layer in enumerate(self.conv_layers):
            if i in self.upsample_layer_indices:
                h = nn.functional.interpolate(h, scale_factor=self.scale_factor, mode=self.interp_mode)
            h = layer(h)
            if self.conv_layers_activation is not None and i < len(self.conv_layers) - 1:
                h = self.conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply output activation
        if self.output_activation is not None:
            y = self.output_activation(h)
        else:
            y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        gain = get_gain(self.conv_layers_activation)
        for layer in self.conv_layers[:-1]:
            set_init_parameters(layer, gain)
        set_init_parameters(self.conv_layers[-1], get_gain(None))

###############################################################################
# UNet
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


class UNetXd(nn.Module):
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


class Downsample2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 activation=None,
                 dropout=None,
                 scale_factor=2,
                 **conv_kwargs):
        super().__init__()
        # create convolutional layer with stride=2
        self.layer = nn.Conv2d(input_channels, output_channels, kernel_size, **conv_kwargs,
                               padding=1, padding_mode='replicate', stride=scale_factor)
        # set from arguments
        self.input_channels = input_channels
        self.scale_factor   = scale_factor
        self.activation     = activation
        self.dropout        = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels
        h = self.layer(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        gain = get_gain(self.activation)
        set_init_parameters(self.layer, gain)


class Upsample2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 activation=None,
                 dropout=None,
                 scale_factor=2,
                 interp_mode='nearest',
                 **conv_kwargs):
        super().__init__()
        # create convolutional layer
        self.layer = nn.Conv2d(input_channels, output_channels, kernel_size, **conv_kwargs,
                               padding=1, padding_mode='replicate')
        # set from arguments
        self.input_channels = input_channels
        self.scale_factor   = scale_factor
        self.interp_mode    = interp_mode
        self.activation     = activation
        self.dropout        = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels
        h = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.interp_mode)
        h = self.layer(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        gain = get_gain(self.activation)
        set_init_parameters(self.layer, gain)


class LevelBlock2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 normalization=None,
                 activation=None,
                 dropout=None,
                 **conv_kwargs):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        # create convolutional layer
        self.layer = nn.Conv2d(input_channels, output_channels, kernel_size, **conv_kwargs,
                               padding=1, padding_mode='replicate')
        # create residual/skip connection
        if input_channels == output_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_channels, output_channels, 1)
        # create additional layers
        self.normalization = normalization
        self.activation    = activation
        self.dropout       = dropout
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        assert x.size(1) == self.input_channels
        h = self.layer(x) + self.residual_layer(x)
        if self.normalization is not None:
            h = self.normalization(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        zero_parameters(self.layer)
        if not isinstance(self.residual_layer, nn.Identity):
            set_init_parameters(self.residual_layer, get_gain(None))


def Normalization(num_channels, num_groups=1):
    return nn.GroupNorm(num_groups, num_channels)


class UNet2d(UNetXd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         with_Downsample = Downsample2d,
                         with_Upsample   = Upsample2d,
                         with_LevelBlock = LevelBlock2d)

###############################################################################

def get_gain(activation):
    r"""Calculates the gain to be used as an argument for initializing parameter values."""
    if activation is not None:
        activation_name = type(activation).__name__.lower()
        if activation_name in ['silu', 'gelu']:
            activation_name = 'relu'
        gain = nn.init.calculate_gain(activation_name)
    else:
        gain = nn.init.calculate_gain('conv2d')
    return gain

def set_init_parameters(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        lim = 0.1*gain/math.sqrt(layer.bias.size(0))
        nn.init.uniform_(layer.bias, a=-lim, b=+lim)

def zero_parameters(layer):
    torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)

###############################################################################

# TODO use doxygen for these test

def test_ConvUpsampleNet_Reshuffle():
    print('---------------------------------------^')
    net = ConvUpsampleNet_Reshuffle(1, 2)
    print(net)

    print('Test 1:')
    x = torch.tensor([[ [[1., -1.], [1., -1.]] ]])
    y = net(x)
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')

def test_ConvUpsampleNet_Interpolate():
    print('---------------------------------------^')
    net = ConvUpsampleNet_Interpolate(1)
    print(net)

    print('Test 1:')
    x = torch.tensor([[ [[1., -1.], [1., -1.]] ]])
    y = net(x)
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')

def test_Upsample2d():
    print('---------------------------------------^')
    layer = Upsample2d(1, 1, 3)
    print(layer)

    print('Test 1:')
    row = 4*[1.]
    x = torch.tensor([[ [row for _ in range(4)] ]])
    y = layer(x)
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')

def test_Downsample2d():
    print('---------------------------------------^')
    layer = Downsample2d(1, 1, 3)
    print(layer)

    print('Test 1:')
    row = 8*[1.]
    x = torch.tensor([[ [row for _ in range(8)] ]])
    y = layer(x)
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')

def test_LevelBlock2d():
    print('---------------------------------------^')
    layer = LevelBlock2d(1, 1, 3, activation=nn.ReLU())
    print(layer)

    print('Test 1:')
    row = 8*[1.]
    x = torch.tensor([[ [row for _ in range(8)] ]])
    y = layer(x)
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')

def test_UNet():
    print('---------------------------------------^')
    net = UNet2d(1, 1,
                 down_levels_conv_channels = [[2,2], [4,4], [8,8]],
                 up_levels_conv_channels   = [[8,8], [4,4], [2,2]],
                 with_Normalization=Normalization,
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
    test_ConvUpsampleNet_Reshuffle()
    test_ConvUpsampleNet_Interpolate()
    test_Upsample2d()
    test_Downsample2d()
    test_LevelBlock2d()
    test_UNet()
