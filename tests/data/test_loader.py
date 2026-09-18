import dataclasses
import inspect

import torch

from dlk.data.loader import DataLoaderConfig


def test_config_fields_are_valid_dataloader_arguments() -> None:
    """Every stored field of `DataLoaderConfig` must be a `DataLoader` parameter."""
    dataloader_parameters = inspect.signature(
        torch.utils.data.DataLoader.__init__
    ).parameters

    config_fields = {field.name for field in dataclasses.fields(DataLoaderConfig)}

    assert config_fields <= dataloader_parameters.keys()
