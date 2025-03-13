"""
Utility Functions
"""

import math
from prettytable import PrettyTable
import torch
import torch.nn as nn

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

def set_init_parameters(layer, gain=1.0, bias_scale=0.1):
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

def set_zero_parameters(layer):
    r"""
    Zeros the parameters of a layer.
    """
    for p in layer.parameters():
        torch.nn.init.zeros_(p)
    return layer

def count_trainable_parameters(net):
    """
    Original source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        n_params = parameter.numel()
        table.add_row([name, n_params])
        total_params += n_params
    print(table)
    print(f"Total trainable parameters: {total_params}")
    return total_params
