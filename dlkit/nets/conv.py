"""Models with convolutional layers.
"""

import torch
import torch.nn as nn

class Conv1dModel(nn.Module):
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
        r"""Creates the model.

        Args:
            input_channels: number of channels
        """
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

    @staticmethod
    def get_gain(activation):
        r"""Calculates the gain to be used as an argument for initializing parameter values."""
        if activation is not None:
            activation_name = type(activation).__name__.lower()
            if activation_name in ['silu']:
                activation_name = 'relu'
            gain = nn.init.calculate_gain(activation_name)
        else:
            gain = nn.init.calculate_gain('conv1d')
        return gain

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden convolutional layers
        gain = self.get_gain(self.hidden_conv_layers_activation)
        for layer in self.hidden_conv_layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1)
        # initialize hidden dense layers
        gain = self.get_gain(self.hidden_dense_layers_activation)
        for layer in self.hidden_dense_layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1)
        # initialize output layer
        gain = self.get_gain(self.output_layer_activation)
        if self.output_layer is not None:
            nn.init.xavier_uniform_(self.output_layer.weight, gain=gain)
            if self.output_layer.bias is not None:
                nn.init.constant_(self.output_layer.bias, 0.0)

