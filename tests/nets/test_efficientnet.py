import pytest
import torch

from dlk.nets.efficientnet import EfficientNet1D


def test_efficientnet1d_forward_output_shape_for_univariate_input() -> None:
    """Run ``EfficientNet1D`` on 2D input and validate logits shape."""
    net = EfficientNet1D(input_length=256, num_classes=3)
    x = torch.randn(4, 256)

    y = net(x)

    assert y.shape == (4, 3)


def test_efficientnet1d_forward_output_shape_for_multichannel_input() -> None:
    """Run ``EfficientNet1D`` on 3D input and validate logits shape."""
    net = EfficientNet1D(input_channels=2, input_length=256, num_classes=5)
    x = torch.randn(4, 2, 256)

    y = net(x)

    assert y.shape == (4, 5)


def test_efficientnet1d_raises_for_invalid_sequence_length() -> None:
    """Raise an assertion error when the input sequence length is invalid."""
    net = EfficientNet1D(input_length=256)
    x = torch.randn(4, 255)

    with pytest.raises(AssertionError, match="expected sequence length"):
        net(x)
