import torch
import torch.nn as nn

from dlk.nets.mlp import AttentionBlock, MLPNet, MLPNet_MultIn, MLPResNet, ResidualBlock


def test_mlpnet_forward_with_hidden_activation() -> None:
    """Run a forward pass for ``MLPNet`` with hidden-layer activation."""
    net = MLPNet(4, 3, hidden_layers_activation=nn.SiLU())
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])

    y = net(x)

    assert y.shape == (1, 3)


def test_mlpnet_forward_with_input_activation() -> None:
    """Run a forward pass for ``MLPNet`` with an input-layer activation."""
    net = MLPNet(4, 3, input_layer_activation=nn.SiLU())
    x = torch.randn(8, 4)

    y = net(x)

    assert y.shape == (8, 3)


def test_mlpnet_output_sigmoid_maps_values_to_unit_interval() -> None:
    """Map MLP outputs to the unit interval when using a sigmoid output layer."""
    net = MLPNet(4, 3, output_layer_activation=nn.Sigmoid())
    x = torch.randn(128, 4)

    y = net(x)

    assert y.shape == (128, 3)
    assert torch.all(y >= 0.0).item()
    assert torch.all(y <= 1.0).item()


def test_mlpnet_multin_forward_with_two_inputs() -> None:
    """Run a forward pass for ``MLPNet_MultIn`` with two input tensors."""
    net = MLPNet_MultIn(4 + 2, 3)
    x0 = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    x1 = torch.tensor([[2.0, -2.0]])

    y = net(x0, x1)

    assert y.shape == (1, 3)


def test_mlpnet_multin_forward_with_hidden_inputs() -> None:
    """Run ``MLPNet_MultIn`` with hidden inputs injected into selected blocks."""
    net = MLPNet_MultIn(
        4 + 2,
        3,
        hidden_input_sizes=[0, 3, 0, 5],
        hidden_layers_sizes=[10, 20, 30, 40, 50],
    )
    x0 = torch.randn(8, 4)
    x1 = torch.randn(8, 2)
    h1 = torch.randn(8, 3)
    h3 = torch.randn(8, 5)

    y = net(x0, x1, h1=h1, h3=h3)

    assert y.shape == (8, 3)


def test_residual_block_forward_single_embedding() -> None:
    """Run ``ResidualBlock`` for a single embedding channel."""
    net = ResidualBlock(4)
    x = torch.tensor([[[1.0, -1.0, 1.0, -1.0]]])

    y = net(x)

    assert y.shape == (1, 1, 4)


def test_residual_block_forward_multiple_embeddings() -> None:
    """Run ``ResidualBlock`` for multiple embedding channels."""
    net = ResidualBlock(4)
    x = torch.randn(1, 2, 4)

    y = net(x)

    assert y.shape == (1, 2, 4)


def test_residual_block_forward_with_skip_connection_and_concat_input() -> None:
    """Run ``ResidualBlock`` with concatenated inputs and output-size projection."""
    net = ResidualBlock(4 + 4, 3)
    x0 = torch.tensor([[[1.0, -1.0, 1.0, -1.0]]])
    x1 = torch.tensor([[[5.0, -5.0, 5.0, -5.0]]])

    y = net(x0, x1)

    assert y.shape == (1, 1, 3)


def test_attention_block_forward() -> None:
    """Run ``AttentionBlock`` with valid head/embedding dimensions."""
    net = AttentionBlock(embedding_size=6, attention_layer_n_heads=3)
    x = torch.randn(1, 6, 4)

    y = net(x)

    assert y.shape == (1, 6, 4)


def test_mlpresnet_single_input_forward() -> None:
    """Run ``MLPResNet`` for a single-input residual network."""
    net = MLPResNet(
        4,
        10,
        residual_blocks_sizes=2 * [(16, 32, 128, 16)] + [(16, 16, 64, 8)],
    )
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])

    y = net(x)

    assert y.shape == (1, 10)


def test_mlpresnet_multi_input_and_hidden_input_forward() -> None:
    """Run ``MLPResNet`` with multiple inputs and block-specific hidden tensors."""
    net = MLPResNet(
        (4 + 2, 8),
        10,
        residual_blocks_sizes=[(8 + 12, 30, 60, 10), (10 + 10, 30, 60, 10)],
    )
    x0 = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    x1 = torch.tensor([[2.0, -2.0]])
    h0 = torch.randn(1, 12)
    h1 = torch.randn(1, 10)

    y = net(x0, x1, h0=h0, h1=h1)

    assert y.shape == (1, 10)


def test_mlpresnet_forward_with_attention_blocks() -> None:
    """Run ``MLPResNet`` when attention blocks are enabled."""
    net = MLPResNet(
        4,
        10,
        embedding_size=6,
        residual_blocks_sizes=2 * [(16, 32, 128, 16)],
        attention_blocks_n_heads=2 * [3],
    )
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])

    y = net(x)

    assert y.shape == (1, 10)
