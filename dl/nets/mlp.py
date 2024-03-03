"""Model for Multilayer Perceptron.
wiki: <https://en.wikipedia.org/wiki/Multilayer_perceptron>
"""

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
        super().__init__()
        # set from arguments
        self.hidden_layers_activation = hidden_layers_activation
        self.output_layer_activation  = output_layer_activation
        if use_dropout:
            self.dropout = nn.Dropout(use_dropout)
        else:
            self.dropout = None
        # create hidden layers
        self.hidden_layers = nn.ModuleList()
        for layer_size in hidden_layers_sizes:
            layer = nn.Linear(input_size, layer_size, **hidden_layers_kwargs)
            self.hidden_layers.append(layer)
            input_size = layer_size
        # create output layer
        self.output_layer = nn.Linear(input_size, output_size, **output_layer_kwargs)
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the model function: y = f(x)

        Args:
            x: input tensor
        """
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
            if activation_name in ['silu']:
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
                nn.init.constant_(layer.bias, 0.1)
        # initialize output layer
        gain = self.get_gain(self.output_layer_activation)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=gain)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)

###############################################################################

def test_model():
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


if __name__ == '__main__':
    r"""Runs tests."""
    test_model()
