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
                 input_size,
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
            input_size (int): sum of lengths of each input vector
            output_size (int): length of outputs
        """
        # set sizes of the hidden layers
        if hidden_input_sizes is None:
            hidden_input_sizes = [0]*len(hidden_layers_sizes)
        if len(hidden_input_sizes) == (len(hidden_layers_sizes) - 1):
            hidden_input_sizes += [0]
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

    def forward(self, *x, **h_kwargs):
        r"""Applies the forward function: y = net(x0, x1, ..., h0=hidden_input_0, h1=hidden_input1, ...)

        Args:
            x0, x1, ... (tensor): input tensors
            h0, h1, ... (tensor, optional): input tensors to hidden layers
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
            h_in = h_kwargs.get(f"h{block_idx}", None)
            if h_in is not None:
                h = torch.cat((h, torch.flatten(h_in, 1)), dim=1)
            assert h.size(1) == block.layer.in_features, f"{block_idx=}, {h.size(1)=}, {block.layer.in_features=}"
            h = block(h)
        # apply output layer
        y = self.output_layer(h)
        return y

# --------------------------------------
# Residual MLP Net
# --------------------------------------

class SelfAttentionLayer(nn.MultiheadAttention):
    def __init__(self, embedding_size, n_heads, use_dropout=False, **kwargs):
        if use_dropout:
            dropout = use_dropout
        else:
            dropout = 0.0
        super().__init__(embedding_size, n_heads, dropout=dropout, batch_first=True, **kwargs)
        # initialization of parameters is done in constructor of MultiheadAttention calling _reset_parameters()

    def forward(self, x):
        y, _ = super().forward(x, x, x, need_weights=False)
        return y


class ResidualBlock(nn.Module):
    r"""ResidualBlock.

    Args:
        input_size (int): length of input vectors
        output_size (int): length of output vectors (if None, `output_size = input_size`)
    """
    def __init__(self,
                 input_size,
                 output_size              = None,
                 embedding_size           = 0,
                 attention_layer_n_heads  = 0,
                 normalization_layer_size = 32,
                 activation_layer_size    = 128,
                 activation               = nn.ReLU(),
                 layers_kwargs            = {},
                 attention_layer_kwargs   = {},
                 use_dropout              = False
        ):
        super().__init__()
        # set from arguments
        self.input_size     = input_size
        self.embedding_size = embedding_size
        # create attention layer
        if attention_layer_n_heads:
            assert 0 < self.embedding_size
            self.attention_layer = SelfAttentionLayer(embedding_size, attention_layer_n_heads,
                                                      use_dropout=use_dropout, **attention_layer_kwargs)
        else:
            self.attention_layer = None
        # create layers of the residual block
        in_size = input_size
        nl_size = normalization_layer_size
        al_size = activation_layer_size
        out_size = output_size if output_size is not None else input_size
        block = OrderedDict()
        block['layer_0']             = nn.Linear(in_size, nl_size, **layers_kwargs)
        block['normalization']       = nn.LayerNorm(nl_size)
        block['layer_1']             = nn.Linear(nl_size, al_size, **layers_kwargs)
        block['activation']          = activation
        if use_dropout:
            block['dropout']         = nn.Dropout(use_dropout)
        block['layer_2']             = nn.Linear(al_size, out_size, **layers_kwargs)
        self.block = nn.Sequential(block)
        # create skip connection
        if in_size != out_size:
            self.skip_connection = nn.Linear(in_size, out_size, **layers_kwargs)
        else:
            self.skip_connection = None
        # initialize parameters
        self.init_parameters()

    def forward(self, *x):
        r"""Applies the forward function: y = net(x0, x1, ...)

        Args:
            x0, x1, ... (tensor): input tensors
        """
        # flatten and concatenate inputs
        input_dim = 1 + int(0 < self.embedding_size)
        if 1 < len(x):
            h_in = torch.cat([torch.flatten(x_, input_dim) for x_ in x], dim=input_dim)
        else:
            assert 1 == len(x)
            h_in = torch.flatten(x[0], input_dim)
        assert h_in.size(input_dim) == self.input_size, f"{h_in.size(input_dim)=}, {self.input_size=}"
        # combine batch and embedding dimensions for subsequent layers
        h = h_in
        if 0 < self.embedding_size:
            h = h.reshape(-1, h.size(input_dim))
        # apply skip connection
        if self.skip_connection is not None:
            hs = self.skip_connection(h)
        else:
            hs = h
        # apply block
        if self.attention_layer is not None:
            ha = self.attention_layer(h_in.transpose(1, 2)).transpose(1, 2)
            ha = ha.reshape(-1, ha.size(input_dim))
            hb = self.block(ha)
        else:
            hb = self.block(h)
        # compute output
        y = 0.5*hs + 0.5*hb
        # separate batch and embedding dimensions
        if 0 < self.embedding_size:
            y = y.reshape(-1, self.embedding_size, self.input_size)
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
                 embedding_size             = 0,
                 input_layer_activation     = None,
                 input_layer_kwargs         = {},
                 residual_blocks_sizes      = 4*[(32, 32, 128, 32)],
                 residual_blocks_activation = nn.ReLU(),
                 residual_blocks_kwargs     = {},
                 attention_layers_n_heads   = None,
                 attention_layers_kwargs    = {},
                 use_dropout                = False,
                 output_layer_activation    = None,
                 output_layer_kwargs        = {}
        ):
        super().__init__()
        assert 0 < len(residual_blocks_sizes)
        assert 0 < len(residual_blocks_sizes[0])
        # set up parameters for attention layers
        self.embedding_size = embedding_size
        assert attention_layers_n_heads is None or 0 < embedding_size
        if attention_layers_n_heads is None:
            attention_layers_n_heads = len(residual_blocks_sizes)*[0]
        assert len(attention_layers_n_heads) == len(residual_blocks_sizes)
        # create input layer
        try:
            self.input_size = input_size[0]
            out_size_       = input_size[1]
        except:
            self.input_size = input_size
            out_size_       = residual_blocks_sizes[0][0]
        if 0 < embedding_size:
            out_size_ *= embedding_size
        layer = nn.Linear(self.input_size, out_size_, **input_layer_kwargs)
        if input_layer_activation is not None:
            self.input_layer = nn.Sequential(OrderedDict([
                    ('layer', layer),
                    ('activation', input_layer_activation)
            ]))
        else:
            self.input_layer = layer
        # create residual blocks
        blocks = list()
        for i, sizes in enumerate(residual_blocks_sizes):
            assert 3 <= len(sizes)
            block_in_size = sizes[0]
            block_nl_size = sizes[1]
            block_al_size = sizes[2]
            block_out_size = sizes[3] if 3 < len(sizes) else sizes[0]
            blocks.append(ResidualBlock(
                    block_in_size,
                    output_size              = block_out_size,
                    embedding_size           = embedding_size,
                    attention_layer_n_heads  = attention_layers_n_heads[i],
                    normalization_layer_size = block_nl_size,
                    activation_layer_size    = block_al_size,
                    activation               = residual_blocks_activation,
                    layers_kwargs            = residual_blocks_kwargs,
                    attention_layer_kwargs   = attention_layers_kwargs,
                    use_dropout              = use_dropout
            ))
        self.residual_blocks = nn.Sequential(*blocks)
        # create output layer
        in_size_ = block_out_size
        if 0 < embedding_size:
            in_size_ *= embedding_size
        layer = nn.Linear(in_size_, output_size, **output_layer_kwargs)
        if output_layer_activation is not None:
            self.output_layer = nn.Sequential(OrderedDict([
                    ('layer', layer),
                    ('activation', output_layer_activation)
            ]))
        else:
            self.output_layer = layer
        # initialize parameters
        self.init_parameters()

    def forward(self, *x, **h_kwargs):
        r"""Applies the forward function: y = net(x0, x1, ..., h0=hidden_input_0, h1=hidden_input1, ...)

        Args:
            x0, x1, ... (tensor): input tensors
            h0, h1, ... (tensor, optional): input tensors to hidden layers
        """
        # concatenate inputs; flatten inputs
        assert 0 < len(x)
        if 1 < len(x):
            h = torch.cat([torch.flatten(x_, 1) for x_ in x], dim=1)
        else:
            h = torch.flatten(x[0], 1)
        batch_size = h.size(0)
        input_dim  = 1 + int(0 < self.embedding_size)
        # apply input layer
        h = self.input_layer(h)
        if 0 < self.embedding_size:
            h = h.reshape(batch_size, self.embedding_size, -1)
        # apply residual blocks
        for block_idx, block in enumerate(self.residual_blocks):
            h_in = h_kwargs.get(f"h{block_idx}", None)
            if h_in is not None:
                assert (h.size(input_dim) + sum(h_in.size()[input_dim:])) == block.input_size, \
                       f"{block_idx=}, {h.size(input_dim)=}, {sum(h_in.size()[input_dim:])=}, {block.input_size=}"
                h = block(h, h_in)
            else:
                assert h.size(input_dim) == block.input_size, f"{block_idx=}, {h.size(input_dim)=}, {block.input_size=}"
                h = block(h)
        # apply output layer
        if 0 < self.embedding_size:
            h = h.reshape(batch_size, -1)
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
    net = MLPNet_MultIn(4+2, 3)
    print(net)
    x0 = torch.tensor([[1., -1., 1., -1.]])
    x1 = torch.tensor([[2., -2.]])
    y  = net(x0, x1)
    print('- input  x0 =', x0)
    print('- input  x1 =', x1)
    print('- output y  =', y)

    print('Test 2:')
    net = MLPNet_MultIn(4+2, 3, hidden_input_sizes=[0,3,0,5], hidden_layers_sizes=[10,20,30,40,50])
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
    print('Test 1:')
    net = ResidualBlock(4)
    print(net)
    x = torch.tensor([[1., -1., 1., -1.]])
    y = net(x)
    print('- input  x =', x)
    print('- output y =', y)

    print('Test 2:')
    net = ResidualBlock(4+4, 3)
    print(net)
    x0 = torch.tensor([[1., -1., 1., -1.]])
    x1 = torch.tensor([[5., -5., 5., -5.]])
    y = net(x0, x1)
    print('- input  x0 =', x0)
    print('- input  x1 =', x1)
    print('- output y  =', y)

    print('Test 3:')
    net = ResidualBlock(4, embedding_size=2)
    print(net)
    x = torch.randn(1, 2, 4)
    y = net(x)
    print('- input  x =', x)
    print('- output y =', y)

    print('Test 4:')
    net = ResidualBlock(4, embedding_size=6, attention_layer_n_heads=3)
    print(net)
    x = torch.randn(1, 6, 4)
    y = net(x)
    print('- input  x =', x)
    print('- output y =', y)
    print('---------------------------------------$')

def test_MLPResNet():
    print('---------------------------------------^')
    print('Test 1:')
    net = MLPResNet(4, 10, residual_blocks_sizes=2*[(16, 32, 128, 16)] + [(16, 16, 64, 8)])
    print(net)
    x = torch.tensor([[1., -1., 1., -1.]])
    y = net(x)
    print('- input  x =', x)
    print('- output y =', y)

    print('Test 2:')
    net = MLPResNet((4+2, 8), 10, residual_blocks_sizes=2*[(20, 30, 60, 10)])
    print(net)
    x0 = torch.tensor([[1., -1., 1., -1.]])
    x1 = torch.tensor([[2., -2.]])
    h0 = torch.randn(1, 12)
    h1 = torch.randn(1, 10)
    y = net(x0, x1, h0=h0, h1=h1)
    print('- input  x0 =', x0)
    print('- input  x1 =', x1)
    print('- input  h0 =', h0)
    print('- input  h1 =', h1)
    print('- output y  =', y)

    print('Test 3:')
    net = MLPResNet(4, 10, embedding_size=6, residual_blocks_sizes=2*[(16, 32, 128, 16)], attention_layers_n_heads=2*[3])
    print(net)
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
