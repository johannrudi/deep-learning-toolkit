"""
Models for Multilayer Perceptron.

- wiki: <https://en.wikipedia.org/wiki/Multilayer_perceptron>
"""

import math
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_layers_sizes=4*[32],
                 hidden_layers_activation=nn.ReLU(),
                 hidden_layers_kwargs={},
                 output_layer_activation=None,
                 output_layer_kwargs={},
                 use_dropout=False):
        r"""Creates the model.

        Args:
            input_size (int): length of input features
            output_size (int): length of outputs
        """
        super().__init__()
        # set from arguments
        self.hidden_layers_activation = hidden_layers_activation
        self.output_layer_activation  = output_layer_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden layers
        in_size = input_size
        self.hidden_layers = nn.ModuleList()
        for layer_size in hidden_layers_sizes:
            layer = nn.Linear(in_size, layer_size, **hidden_layers_kwargs)
            self.hidden_layers.append(layer)
            in_size = layer_size
        # create output layer
        self.output_layer = nn.Linear(in_size, output_size, **output_layer_kwargs)
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the model function: y = model(x)

        Args:
            x (tensor): input tensor
        """
        # flatten inputs
        h = torch.flatten(x, 1)
        # apply hidden layers
        for layer in self.hidden_layers:
            h = layer(h)
            if self.hidden_layers_activation is not None:
                h = self.hidden_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        # apply output layer
        y = self.output_layer(h)
        if self.output_layer_activation is not None:
            y = self.output_layer_activation(y)
        return y

    @staticmethod
    def get_gain(activation):
        r"""Calculates the gain to be used as an argument for initializing parameter values."""
        if activation is not None:
            activation_name = type(activation).__name__.lower()
            if activation_name in ['silu', 'gelu']:
                activation_name = 'relu'
            gain = nn.init.calculate_gain(activation_name)
        else:
            gain = nn.init.calculate_gain('linear')
        return gain

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize hidden layers
        gain = self.get_gain(self.hidden_layers_activation)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None:
                lim = 0.1*gain/math.sqrt(layer.bias.size(0))
                nn.init.uniform_(layer.bias, a=-lim, b=+lim)
        # initialize output layer
        gain = self.get_gain(self.output_layer_activation)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=gain)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)


class MLPModelMultIn(MLPModel):
    def __init__(self,
                 input_sizes,
                 output_size,
                 hidden_input_sizes=None,
                 hidden_layers_sizes=4*[32],
                 hidden_layers_activation=nn.ReLU(),
                 hidden_layers_kwargs={},
                 output_layer_activation=None,
                 output_layer_kwargs={},
                 use_dropout=False):
        r"""Creates the model.

        Args:
            input_sizes (list): list of lengths for each input feature vector
            output_size (int): length of outputs
        """
        # set input size
        input_size = sum(input_sizes)
        # set up input sizes of hidden layers
        if hidden_input_sizes is None:
            hidden_input_sizes = [0]*len(hidden_layers_sizes)
        assert len(hidden_input_sizes) == len(hidden_layers_sizes)
        # create regular MLP model with only one hidden layer and the output layer,
        # where the hidden layer will be replaced below
        super().__init__(
            input_size, output_size,
            hidden_layers_sizes=[hidden_layers_sizes[-1] + hidden_input_sizes[-1]],
            hidden_layers_activation=hidden_layers_activation,
            hidden_layers_kwargs=hidden_layers_kwargs,
            output_layer_activation=output_layer_activation,
            output_layer_kwargs=output_layer_kwargs,
            use_dropout=use_dropout
        )
        # create hidden layers
        in_size = input_size
        self.hidden_layers = nn.ModuleList()
        for layer_size, hidden_input_size in zip(hidden_layers_sizes, hidden_input_sizes):
            layer = nn.Linear(in_size, layer_size, **hidden_layers_kwargs)
            self.hidden_layers.append(layer)
            in_size = layer_size + hidden_input_size
        # initialize parameters
        self.init_parameters()

    def forward(self, *x_args, **h_kwargs):
        r"""Applies the model function: y = model(x0, x1, ..., h1=hidden_input_1, h2=hidden_input2, ...)

        Args:
            x0, x1, ... (tensor): input tensors
            h1, h2, ... (tensor, optional): input tensors to hidden layers
        """
        # concatenate inputs; flatten inputs
        h = torch.cat([torch.flatten(x_, 1) for x_ in x_args], dim=1)
        # apply hidden layers
        for layer_idx, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if self.hidden_layers_activation is not None:
                h = self.hidden_layers_activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
            # concatenate input to hidden layer (if it exists)
            h_in = h_kwargs.get('h'+str(layer_idx+1), None)
            if h_in is not None:
                h = torch.cat((h, torch.flatten(h_in, 1)), dim=1)
        # apply output layer
        y = self.output_layer(h)
        if self.output_layer_activation is not None:
            y = self.output_layer_activation(y)
        return y

###############################################################################

# TODO use doxygen for these test

def test_MLPModel():
    model = MLPModel(4, 3, hidden_layers_activation=nn.SiLU())
    print(model)

    print('Test 1:')
    x = torch.tensor([[1., -1., 1., -1.]])
    y = model(x)
    print('- input  x =', x)
    print('- output y =', y)

    print('Test 2:')
    x = torch.randn(8, 4)
    y = model(x)
    print('- input  x ='); print(x)
    print('- output y ='); print(y)

    print('Test 3:')
    x = torch.randn(10000, 4)
    y = model(x)
    print('- input  mean = %.6e, std = %.6e' % (x.mean().item(), x.std().item()))
    print('- output mean = %.6e, std = %.6e' % (y.mean().item(), y.std().item()))

    print('----------------------------------------')

def test_MLPModelMultIn():
    print('Test 1:')
    model = MLPModelMultIn([4,2], 3)
    print(model)
    x0 = torch.tensor([[1., -1., 1., -1.]])
    x1 = torch.tensor([[2., -2.]])
    y  = model(x0, x1)
    print('- input  x0 =', x0)
    print('- input  x1 =', x1)
    print('- output y  =', y)

    print('Test 2:')
    model = MLPModelMultIn([4,2], 3, hidden_input_sizes=[0,3,0,5])
    print(model)
    x0 = torch.randn(8, 4)
    x1 = torch.randn(8, 2)
    h2 = torch.randn(8, 3)
    h4 = torch.randn(8, 5)
    y  = model(x0, x1, h2=h2, h4=h4)
    print('- input  x0 ='); print(x0)
    print('- input  x1 ='); print(x1)
    print('- input  h2 ='); print(h2)
    print('- input  h4 ='); print(h4)
    print('- output y  ='); print(y)

    print('----------------------------------------')

if __name__ == '__main__':
    r"""Runs tests."""
    test_MLPModel()
    test_MLPModelMultIn()
