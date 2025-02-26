"""
Networks with 1D convolutional layers.
"""

import math
import torch
import torch.nn as nn

# --------------------------------------
# Convolutional Nets
# --------------------------------------

class ConvNet(nn.Module):
    r"""
    Network with convolutional layers followed by dense layers.

    Args:
        input_channels: number of input channels
    """
    def __init__(self,
                 input_channels,
                 hidden_conv_layers_channels_mult=[8, 16, 32],
                 hidden_conv_layers_kernels=[3, 3, 3],
                 hidden_conv_layers_activation=nn.ReLU(),
                 hidden_conv_layers_kwargs={},
                 hidden_dense_input_size=None,
                 hidden_dense_layers_sizes=[],
                 hidden_dense_layers_activation=nn.ReLU(),
                 hidden_dense_layers_kwargs={},
                 output_size=None,
                 output_layer_activation=None,
                 output_layer_kwargs={},
                 use_dropout=False):
        super().__init__()
        # set from arguments
        self.input_channels = input_channels
        self.hidden_conv_layers_activation  = hidden_conv_layers_activation
        self.hidden_dense_layers_activation = hidden_dense_layers_activation
        self.output_layer_activation        = output_layer_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden convolutional layers
        assert len(hidden_conv_layers_channels_mult) == len(hidden_conv_layers_kernels)
        in_channels = input_channels
        self.hidden_conv_layers = nn.ModuleList()
        for channel_mult, kernel_size in zip(hidden_conv_layers_channels_mult, hidden_conv_layers_kernels):
            out_channels = channel_mult * input_channels
            layer = nn.Conv1d(in_channels, out_channels, kernel_size, **hidden_conv_layers_kwargs)
            self.hidden_conv_layers.append(layer)
            in_channels = out_channels
        # create hidden dense layers
        assert hidden_dense_input_size is not None or 0 == len(hidden_dense_layers_sizes)
        in_size = hidden_dense_input_size
        self.hidden_dense_layers = nn.ModuleList()
        for layer_size in hidden_dense_layers_sizes:
            layer = nn.Linear(in_size, layer_size, **hidden_dense_layers_kwargs)
            self.hidden_dense_layers.append(layer)
            in_size = layer_size
        # create output layer
        if in_size is not None and output_size is not None:
            self.output_layer = nn.Linear(in_size, output_size, **output_layer_kwargs)
        else:
            self.output_layer = None
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the model function: y = model(x)

        Args:
            x: input tensor
        """
        assert x.shape[1] == self.input_channels
        h = x
        # apply hidden convolutional layers
        for layer in self.hidden_conv_layers:
            h = layer(h)
            if self.hidden_conv_layers_activation is not None:
                h = self.hidden_conv_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        h = torch.flatten(h, 1)
        # apply hidden dense layers
        for layer in self.hidden_dense_layers:
            h = layer(h)
            if self.hidden_dense_layers_activation is not None:
                h = self.hidden_dense_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply output layer
        if self.output_layer is not None:
            y = self.output_layer(h)
            if self.output_layer_activation is not None:
                y = self.output_layer_activation(y)
        else:
            y = h
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden convolutional layers
        gain = _get_gain(self.hidden_conv_layers_activation)
        for layer in self.hidden_conv_layers:
            _set_init_parameters(layer, gain)
        # initialize hidden dense layers
        gain = _get_gain(self.hidden_dense_layers_activation)
        for layer in self.hidden_dense_layers:
            _set_init_parameters(layer, gain)
        # initialize output layer
        gain = _get_gain(self.output_layer_activation)
        if self.output_layer is not None:
            _set_init_parameters(self.output_layer, gain, bias_scale=0.0)

# --------------------------------------
# Utility Functions
# --------------------------------------

def _get_gain(activation):
    r"""
    Calculates the gain to be used as an argument for initializing parameter values.

    Args:
        activation: Object of activation function or None
    """
    if activation is not None:
        activation_name = type(activation).__name__.lower()
        if activation_name in ['silu', 'gelu']:
            activation_name = 'relu'
        gain = nn.init.calculate_gain(activation_name)
    else:
        gain = nn.init.calculate_gain('conv1d')
    return gain

def _set_init_parameters(layer, gain=1.0, bias_scale=0.1):
    r"""
    Initializes the trainable parameters of a layer.

    Args:
        layer:      Layer of a network to be initialized (of type nn.Module)
        gain:       Gain to use for sampling initial parameters
        bias_scale: Scaling of uniform distribution for initializing the bias
    """
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        lim = bias_scale * gain/math.sqrt(layer.bias.size(0))
        nn.init.uniform_(layer.bias, a=-lim, b=+lim)

def _set_zero_parameters(layer):
    r"""
    Zeros the parameters of a layer.
    """
    for p in layer.parameters():
        torch.nn.init.zeros_(p)
    return layer

# --------------------------------------
# Tests
# --------------------------------------

# TODO use doxygen for these test

def test_ConvNet():
    print('---------------------------------------^')
    input_channels = 1
    input_size     = 16
    hidden_conv_layers_channels_mult = [2, 4, 8]
    hidden_dense_input_size = (16 - 2*len(hidden_conv_layers_channels_mult)) * hidden_conv_layers_channels_mult[-1]
    net = ConvNet(input_channels, output_size=2,
                  hidden_conv_layers_channels_mult=hidden_conv_layers_channels_mult,
                  hidden_dense_input_size=hidden_dense_input_size,
                  hidden_dense_layers_sizes=[32, 32])
    print(net)

    print('Test 1:')
    x = torch.ones((1, input_channels, input_size))
    y = net(x)
    print('- input  x =', x, sep='\n')
    print('- output y =', y, sep='\n')
    print('---------------------------------------$')

if __name__ == '__main__':
    r"""Runs tests."""
    test_ConvNet()
