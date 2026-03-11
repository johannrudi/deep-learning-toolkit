import torch

from dlk.nets.transformer1d import ChannelWiseTransformerNet, TransformerNet


def test_transformer_net_forward_output_shape() -> None:
    """Run a forward pass for ``TransformerNet`` and validate output shape."""
    batch_size = 4
    input_size = 1000
    output_size = 2
    net = TransformerNet(
        input_seq_size=input_size,
        output_size=output_size,
        patch_size=50,
        embedding_size=128,
        attn_n_heads=(8, 8, 8, 8, 8, 8),
    )
    x = torch.ones((batch_size, input_size))

    y = net(x)

    assert y.shape == (batch_size, output_size)


def test_channel_wise_transformer_net_forward_output_shape() -> None:
    """Run a forward pass for ``ChannelWiseTransformerNet`` and validate output shape."""
    batch_size = 4
    input_channels = 3
    input_size = 1000
    output_size = 2
    net = ChannelWiseTransformerNet(
        input_channels=input_channels,
        input_seq_size=input_size,
        output_size=output_size,
        patch_size=50,
        embedding_size=192,
        attn_n_heads=(8, 8, 8, 8, 8, 8),
    )
    x = torch.ones((batch_size, input_channels, input_size))

    y = net(x)

    assert y.shape == (batch_size, output_size)
