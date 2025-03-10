"""
Multilayer Perceptron Networks.

- wiki: <https://en.wikipedia.org/wiki/Multilayer_perceptron>
"""

from collections import OrderedDict
import math
import torch
import torch.nn as nn

# --------------------------------------
# MLP Nets
# --------------------------------------

class MLPNet(nn.Module):
    r"""MLPNet.

    Args:
        input_size (int): length of input vectors
        output_size (int): length of output vectors
    """
    def __init__(self,
                 input_size,
                 output_size,
                 input_layer_activation   = None,
                 input_layer_kwargs       = {},
                 hidden_layers_sizes      = 4*[32],
                 hidden_layers_activation = nn.ReLU(),
                 hidden_layers_kwargs     = {},
                 use_dropout              = False,
                 output_layer_activation  = None,
                 output_layer_kwargs      = {}
        ):
        super().__init__()
        # check arguments
        assert 0 < len(hidden_layers_sizes), hidden_layers_sizes
        # create input layer
        self.input_size = input_size
        layer = nn.Linear(input_size, hidden_layers_sizes[0], **input_layer_kwargs)
        if input_layer_activation is not None:
            self.input_layer = nn.Sequential(OrderedDict([
                    ('layer', layer),
                    ('activation', input_layer_activation)
            ]))
        else:
            self.input_layer = layer
        # create hidden layers
        blocks = list()
        for k, (in_size, out_size) in enumerate(zip(hidden_layers_sizes[:-1], hidden_layers_sizes[1:])):
            block = OrderedDict()
            block['layer'] = nn.Linear(in_size, out_size, **hidden_layers_kwargs)
            if hidden_layers_activation is not None:
                block['activation'] = hidden_layers_activation
            if use_dropout:
                block['dropout'] = nn.Dropout(use_dropout)
            blocks.append(nn.Sequential(block))
        self.hidden_blocks = nn.Sequential(*blocks)
        # create output layer
        layer = nn.Linear(hidden_layers_sizes[-1], output_size, **output_layer_kwargs)
        if output_layer_activation is not None:
            self.output_layer = nn.Sequential(OrderedDict([
                    ('layer', layer),
                    ('activation', output_layer_activation)
            ]))
        else:
            self.output_layer = layer
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the forward function: y = net(x)

        Args:
            x (tensor): input tensor
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

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize input layer
        if isinstance(self.input_layer, nn.Sequential):
            gain = _get_gain(self.input_layer.activation)
            _set_init_parameters(self.input_layer.layer, gain)
        else:
            _set_init_parameters(self.input_layer, _get_gain(None))
        # initialize hidden layers
        for block in self.hidden_blocks:
            try:
                gain = _get_gain(block.activation)
            except:
                gain = _get_gain(None)
            _set_init_parameters(block.layer, gain)
        # initialize output layer
        if isinstance(self.output_layer, nn.Sequential):
            gain = _get_gain(self.output_layer.activation)
            _set_init_parameters(self.output_layer.layer, gain, bias_scale=0.0)
        else:
            _set_init_parameters(self.output_layer, _get_gain(None), bias_scale=0.0)


class MLPNet_MultIn(MLPNet):
    def __init__(self,
                 input_sizes,
                 output_size,
                 input_layer_activation   = None,
                 input_layer_kwargs       = {},
                 hidden_input_sizes       = None,
                 hidden_layers_sizes      = 4*[32],
                 hidden_layers_activation = nn.ReLU(),
                 hidden_layers_kwargs     = {},
                 use_dropout              = False,
                 output_layer_activation  = None,
                 output_layer_kwargs      = {}
        ):
        r"""MLPNet_MultIn.

        Args:
            input_sizes (list): list of lengths for each input feature vector
            output_size (int): length of outputs
        """
        # set size of the input layer
        input_size = sum(input_sizes)
        # set sizes of the hidden layers
        if hidden_input_sizes is None:
            hidden_input_sizes = [0]*len(hidden_layers_sizes)
        assert len(hidden_input_sizes) == len(hidden_layers_sizes)
        # create regular MLP net with only input and output layers
        super().__init__(
            input_size,
            output_size,
            input_layer_activation   = input_layer_activation,
            input_layer_kwargs       = input_layer_kwargs,
            hidden_layers_sizes      = [hidden_layers_sizes[0],
                                        hidden_layers_sizes[-1] + hidden_input_sizes[-1]],
            hidden_layers_activation = hidden_layers_activation,
            hidden_layers_kwargs     = hidden_layers_kwargs,
            use_dropout              = use_dropout,
            output_layer_activation  = output_layer_activation,
            output_layer_kwargs      = output_layer_kwargs
        )
        # create hidden layer with additional input sizes
        blocks = list()
        for k, (in_size, out_size) in enumerate(zip(hidden_layers_sizes[:-1], hidden_layers_sizes[1:])):
            in_size += hidden_input_sizes[k]
            block = OrderedDict()
            block['layer'] = nn.Linear(in_size, out_size, **hidden_layers_kwargs)
            if hidden_layers_activation is not None:
                block['activation'] = hidden_layers_activation
            if use_dropout:
                block['dropout'] = nn.Dropout(use_dropout)
            blocks.append(nn.Sequential(block))
        self.hidden_blocks = nn.Sequential(*blocks)
        # initialize parameters
        self.init_parameters()

    def forward(self, *x_args, **h_kwargs):
        r"""Applies the forward function: y = net(x0, x1, ..., h1=hidden_input_1, h2=hidden_input2, ...)

        Args:
            x0, x1, ... (tensor): input tensors
            h1, h2, ... (tensor, optional): input tensors to hidden layers
        """
        # concatenate inputs; flatten inputs
        h = torch.cat([torch.flatten(x_, 1) for x_ in x_args], dim=1)
        # apply input layer
        h = self.input_layer(h)
        # apply hidden layers
        for block_idx, block in enumerate(self.hidden_blocks):
            h_in = h_kwargs.get(f"h{block_idx}", None)
            if h_in is not None:
                h = torch.cat((h, torch.flatten(h_in, 1)), dim=1)
            assert h.size(1) == block.layer.in_features, f"{h.size(1)=}, {block.layer.in_features=}"
            h = block(h)
        # apply output layer
        h_in = h_kwargs.get(f"h{len(self.hidden_blocks)}", None)
        if h_in is not None:
            h = torch.cat((h, torch.flatten(h_in, 1)), dim=1)
        if isinstance(self.output_layer, nn.Sequential):
            assert h.size(1) == self.output_layer.layer.in_features, f"{h.size(1)=}, {self.output_layer.layer.in_features=}"
        else:
            assert h.size(1) == self.output_layer.in_features, f"{h.size(1)=}, {self.output_layer.in_features=}"
        y = self.output_layer(h)
        return y

# --------------------------------------
# Residual MLP Net
# --------------------------------------

class ResidualBlock(nn.Module):
    r"""ResidualBlock.

    Args:
        input_output_size (int): length of input and output vectors
    """
    def __init__(self,
                 input_output_size,
                 normalization_layer_size = 32,
                 activation_layer_size    = 128,
                 activation               = nn.ReLU(),
                 layers_kwargs            = {},
                 use_dropout              = False
        ):
        super().__init__()
        # set from arguments
        self.input_output_size = input_output_size
        # create layers
        io_size = input_output_size
        nl_size = normalization_layer_size
        al_size = activation_layer_size
        block = OrderedDict()
        block['layer_0']       = nn.Linear(io_size, nl_size, **layers_kwargs)
        block['normalization'] = nn.LayerNorm(nl_size)
        block['layer_1']       = nn.Linear(nl_size, al_size, **layers_kwargs)
        block['activation']    = activation
        if use_dropout:
            block['dropout']   = nn.Dropout(use_dropout)
        block['layer_2']       = nn.Linear(al_size, io_size, **layers_kwargs)
        self.block = nn.Sequential(block)
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the forward function: y = net(x)

        Args:
            x (tensor): input tensor
        """
        # flatten inputs
        if 2 < x.dim():
            h = h0 = torch.flatten(x, 1)
        else:
            h = h0 = x
        assert h.size(1) == self.input_output_size, f"{h.size(1)=}, {self.input_output_size=}"
        # apply layers
        h = self.block(h)
        # compute output
        y = 0.5*h + 0.5*h0
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize layers
        _set_init_parameters(self.block.layer_0, _get_gain(None))
        _set_init_parameters(self.block.layer_1, _get_gain(self.block.activation))
        _set_init_parameters(self.block.layer_2, _get_gain(None))


class MLPResNet(nn.Module):
    r"""MLPResNet.

    Args:
        input_size (int): length of input vectors
        output_size (int): length of output vectors
    """
    def __init__(self,
                 input_size,
                 output_size,
                 input_layer_activation             = None,
                 input_layer_kwargs                 = {},
                 n_residual_blocks                  = 4,
                 residual_blocks_size               = 32,
                 residual_blocks_normalization_size = 32,
                 residual_blocks_activation_size    = 128,
                 residual_blocks_activation         = nn.ReLU(),
                 residual_blocks_kwargs             = {},
                 use_dropout                        = False,
                 output_layer_activation            = None,
                 output_layer_kwargs                = {}
        ):
        super().__init__()
        # create input layer
        self.input_size = input_size
        layer = nn.Linear(input_size, residual_blocks_size, **input_layer_kwargs)
        if input_layer_activation is not None:
            self.input_layer = nn.Sequential(OrderedDict([
                    ('layer', layer),
                    ('activation', input_layer_activation)
            ]))
        else:
            self.input_layer = layer
        # create residual blocks
        blocks = list()
        for k in range(n_residual_blocks):
            blocks.append(ResidualBlock(
                    residual_blocks_size,
                    residual_blocks_normalization_size,
                    residual_blocks_activation_size,
                    activation=residual_blocks_activation,
                    layers_kwargs=residual_blocks_kwargs,
                    use_dropout=use_dropout
            ))
        self.residual_blocks = nn.Sequential(*blocks)
        # create output layer
        layer = nn.Linear(residual_blocks_size, output_size, **output_layer_kwargs)
        if output_layer_activation is not None:
            self.output_layer = nn.Sequential(OrderedDict([
                    ('layer', layer),
                    ('activation', output_layer_activation)
            ]))
        else:
            self.output_layer = layer
        # initialize parameters
        self.init_parameters()

    def forward(self, x):
        r"""Applies the forward function: y = net(x)

        Args:
            x (tensor): input tensor
        """
        # flatten inputs
        if 2 < x.dim():
            h = torch.flatten(x, 1)
        else:
            h = x
        assert h.size(1) == self.input_size, f"{h.size(1)=}, {self.input_size=}"
        # apply layers
        h = self.input_layer(h)
        h = self.residual_blocks(h)
        y = self.output_layer(h)
        return y

    def init_parameters(self):
        r"""Initializes the values of trainable parameters."""
        # initialize input layer
        if isinstance(self.input_layer, nn.Sequential):
            gain = _get_gain(self.input_layer.activation)
            _set_init_parameters(self.input_layer.layer, gain)
        else:
            _set_init_parameters(self.input_layer, _get_gain(None))
        # initialize output layer
        if isinstance(self.output_layer, nn.Sequential):
            gain = _get_gain(self.output_layer.activation)
            _set_init_parameters(self.output_layer.layer, gain, bias_scale=0.0)
        else:
            _set_init_parameters(self.output_layer, _get_gain(None), bias_scale=0.0)

# --------------------------------------
# Utility Functions
# --------------------------------------

def _get_gain(activation):
    r"""Calculates the gain to be used as an argument for initializing parameter values."""
    if activation is not None:
        activation_name = type(activation).__name__.lower()
        if activation_name in ['silu', 'gelu']:
            activation_name = 'relu'
        gain = nn.init.calculate_gain(activation_name)
    else:
        gain = nn.init.calculate_gain('linear')
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

def test_MLPNet():
    print('---------------------------------------^')
    print('Test 1:')
    net = MLPNet(4, 3, hidden_layers_activation=nn.SiLU())
    print(net)
    x = torch.tensor([[1., -1., 1., -1.]])
    y = net(x)
    print('- input  x =', x)
    print('- output y =', y)

    print('Test 2:')
    net = MLPNet(4, 3, input_layer_activation=nn.SiLU())
    print(net)
    x = torch.randn(8, 4)
    y = net(x)
    print('- input  x ='); print(x)
    print('- output y ='); print(y)

    print('Test 3:')
    net = MLPNet(4, 3, output_layer_activation=nn.Sigmoid())
    print(net)
    x = torch.randn(10000, 4)
    y = net(x)
    print('- input  mean = %.6e, std = %.6e' % (x.mean().item(), x.std().item()))
    print('- output mean = %.6e, std = %.6e' % (y.mean().item(), y.std().item()))
    print('---------------------------------------$')

def test_MLPNet_MultIn():
    print('---------------------------------------^')
    print('Test 1:')
    net = MLPNet_MultIn([4,2], 3)
    print(net)
    x0 = torch.tensor([[1., -1., 1., -1.]])
    x1 = torch.tensor([[2., -2.]])
    y  = net(x0, x1)
    print('- input  x0 =', x0)
    print('- input  x1 =', x1)
    print('- output y  =', y)

    print('Test 2:')
    net = MLPNet_MultIn([4,2], 3, hidden_input_sizes=[0,3,0,5])
    print(net)
    x0 = torch.randn(8, 4)
    x1 = torch.randn(8, 2)
    h1 = torch.randn(8, 3)
    h3 = torch.randn(8, 5)
    y  = net(x0, x1, h1=h1, h3=h3)
    print('- input  x0 ='); print(x0)
    print('- input  x1 ='); print(x1)
    print('- input  h1 ='); print(h1)
    print('- input  h3 ='); print(h3)
    print('- output y  ='); print(y)
    print('---------------------------------------$')

def test_ResidualBlock():
    print('---------------------------------------^')
    net = ResidualBlock(4)
    print(net)

    print('Test 1:')
    x = torch.tensor([[1., -1., 1., -1.]])
    y = net(x)
    print('- input  x =', x)
    print('- output y =', y)
    print('---------------------------------------$')

def test_MLPResNet():
    print('---------------------------------------^')
    net = MLPResNet(4, 2)
    print(net)

    print('Test 1:')
    x = torch.tensor([[1., -1., 1., -1.]])
    y = net(x)
    print('- input  x =', x)
    print('- output y =', y)
    print('---------------------------------------$')

if __name__ == '__main__':
    r"""Runs tests."""
    test_MLPNet()
    test_MLPNet_MultIn()
    test_ResidualBlock()
    test_MLPResNet()
