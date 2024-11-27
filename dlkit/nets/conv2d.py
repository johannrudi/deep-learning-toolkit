"""
Models with 2D convolutional layers.
"""

import math
import torch
import torch.nn as nn

class Conv2dUpscaleModelReshuffle(nn.Module):
    def __init__(self,
                 input_channels,
                 upscale_factor,
                 hidden_pre_up_layers_channels=[8, 16, 32],
                 hidden_pre_up_layers_kernels=[3, 3, 3],
                 hidden_pre_up_layers_activation=nn.ReLU(),
                 hidden_pre_up_layers_kwargs={},
                 hidden_post_up_layers_channels=[],
                 hidden_post_up_layers_kernels=[],
                 hidden_post_up_layers_activation=nn.ReLU(),
                 hidden_post_up_layers_kwargs={},
                 output_activation=None,
                 use_dropout=False):
        r"""Creates the model.

        Args:
            input_channels: number of channels
            upscale_factor: factor by which the coarse input gets scaled to produce fine outputs
        """
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.upscale_factor = upscale_factor
        self.hidden_pre_up_layers_activation  = hidden_pre_up_layers_activation
        self.hidden_post_up_layers_activation = hidden_post_up_layers_activation
        self.output_activation                = output_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers before upscaling
        assert len(hidden_pre_up_layers_channels) == len(hidden_pre_up_layers_kernels)
        in_channels = input_channels
        self.hidden_pre_up_layers = nn.ModuleList()
        for channels, kernel_size in zip(hidden_pre_up_layers_channels, hidden_pre_up_layers_kernels):
            out_channels = channels
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, **hidden_pre_up_layers_kwargs, padding='same')
            self.hidden_pre_up_layers.append(layer)
            in_channels = out_channels
        # create upscale layer
        self.upscale_layer = nn.Conv2d(in_channels, upscale_factor**2, kernel_size, **hidden_pre_up_layers_kwargs, padding='same')
        # create hidden convolutional layers after upscaling
        assert len(hidden_post_up_layers_channels) == len(hidden_post_up_layers_kernels)
        in_channels = input_channels
        self.hidden_post_up_layers = nn.ModuleList()
        for channels, kernel_size in zip(hidden_post_up_layers_channels, hidden_post_up_layers_kernels):
            out_channels = channels
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, **hidden_post_up_layers_kwargs, padding='same')
            self.hidden_post_up_layers.append(layer)
            in_channels = out_channels
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the model function: y = model(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels
        h = x
        # apply hidden convolutional layers before upscaling
        for layer in self.hidden_pre_up_layers:
            h = layer(h)
            if self.hidden_pre_up_layers_activation is not None:
                h = self.hidden_pre_up_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply upscale layer
        h = self.upscale_layer(h)
        h = nn.functional.pixel_shuffle(h, self.upscale_factor)
        # apply hidden convolutional layers after upscaling
        for layer in self.hidden_post_up_layers:
            if self.hidden_post_up_layers_activation is not None:
                h = self.hidden_post_up_layers_activation(h)
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
        # initialize hidden convolutional layers before upscaling
        gain = get_gain(self.hidden_pre_up_layers_activation)
        for layer in self.hidden_pre_up_layers:
            set_init_parameters(layer, gain)
        # initialize hidden convolutional layers after upscaling
        if self.hidden_post_up_layers:
            gain = get_gain(self.hidden_post_up_layers_activation)
            set_init_parameters(self.upscale_layer, gain)
            for layer in self.hidden_post_up_layers[:-1]:
                set_init_parameters(layer, gain)
            set_init_parameters(self.hidden_post_up_layers[-1], get_gain(None))
        else:
            set_init_parameters(self.upscale_layer, get_gain(None))


class Conv2dUpscaleModelInterpolate(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_conv_layers_channels=[4, 4, 1],
                 hidden_conv_layers_kernels=[3, 3, 3],
                 hidden_conv_layers_activation=nn.ReLU(),
                 hidden_conv_layers_kwargs={},
                 upscale_layer_indices=[1],
                 output_activation=None,
                 use_dropout=False):
        r"""Creates the model.

        Args:
            input_channels: number of channels
        """
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.upscale_layer_indices = upscale_layer_indices
        self.hidden_conv_layers_activation = hidden_conv_layers_activation
        self.output_activation             = output_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers before upscaling
        assert len(hidden_conv_layers_channels) == len(hidden_conv_layers_kernels)
        in_channels = input_channels
        self.hidden_conv_layers = nn.ModuleList()
        for channels, kernel_size in zip(hidden_conv_layers_channels, hidden_conv_layers_kernels):
            out_channels = channels
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, **hidden_conv_layers_kwargs, padding='same')
            self.hidden_conv_layers.append(layer)
            in_channels = out_channels
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the model function: y = model(x)

        Args:
            x: input tensor
        """
        assert x.size(1) == self.input_channels
        h = x
        # apply hidden convolutional layers before upscaling
        for i, layer in enumerate(self.hidden_conv_layers):
            if i in self.upscale_layer_indices:
                h = nn.functional.interpolate(h, scale_factor=2, mode='nearest')
            h = layer(h)
            if self.hidden_conv_layers_activation is not None and i < len(self.hidden_conv_layers) - 1:
                h = self.hidden_conv_layers_activation(h)
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
        # initialize hidden convolutional layers before upscaling
        gain = get_gain(self.hidden_conv_layers_activation)
        for layer in self.hidden_conv_layers[:-1]:
            set_init_parameters(layer, gain)
        set_init_parameters(self.hidden_conv_layers[-1], get_gain(None))

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

###############################################################################

# TODO use doxygen for these test

def test_Conv2dUpscaleModelReshuffle():
    model = Conv2dUpscaleModelReshuffle(1, 2)
    print(model)

    print('Test 1:')
    x = torch.tensor([[ [[1., -1.], [1., -1.]] ]])
    y = model(x)
    print('- input  x =', x)
    print('- output y =', y)

    print('----------------------------------------')

def test_Conv2dUpscaleModelInterpolate():
    model = Conv2dUpscaleModelInterpolate(1)
    print(model)

    print('Test 1:')
    x = torch.tensor([[ [[1., -1.], [1., -1.]] ]])
    y = model(x)
    print('- input  x =', x)
    print('- output y =', y)

    print('----------------------------------------')

if __name__ == '__main__':
    r"""Runs tests."""
    test_Conv2dUpscaleModelReshuffle()
    test_Conv2dUpscaleModelInterpolate()
