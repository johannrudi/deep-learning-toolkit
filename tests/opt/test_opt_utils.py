"""Unit tests for the training utilities in `dlk.opt.utils`."""

from collections.abc import Iterator
from typing import Any

import pytest
import torch

from dlk.opt.utils import transfer_non_blocking


class _Dataloader:
    """Minimal dataloader stub exposing `pin_memory`."""

    def __init__(self, pin_memory: bool) -> None:
        self.pin_memory = pin_memory

    @property
    def batch_size(self) -> int | None:
        return 1

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def __len__(self) -> int:
        return 0


class _DataloaderWithoutPinMemory(_Dataloader):
    """Dataloader stub without a `pin_memory` attribute."""

    def __init__(self) -> None:
        super().__init__(pin_memory=False)
        del self.pin_memory


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_transfer_non_blocking_enabled_for_pinned_accelerator(device_type: str) -> None:
    dataloader = _Dataloader(pin_memory=True)
    assert transfer_non_blocking(dataloader, torch.device(device_type))


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_transfer_non_blocking_disabled_without_pinning(device_type: str) -> None:
    dataloader = _Dataloader(pin_memory=False)
    assert not transfer_non_blocking(dataloader, torch.device(device_type))


def test_transfer_non_blocking_disabled_on_cpu() -> None:
    dataloader = _Dataloader(pin_memory=True)
    assert not transfer_non_blocking(dataloader, torch.device("cpu"))


def test_transfer_non_blocking_disabled_without_device() -> None:
    dataloader = _Dataloader(pin_memory=True)
    assert not transfer_non_blocking(dataloader, None)


def test_transfer_non_blocking_disabled_without_pin_memory_attribute() -> None:
    dataloader = _DataloaderWithoutPinMemory()
    assert not transfer_non_blocking(dataloader, torch.device("cuda"))


def test_transfer_non_blocking_matches_torch_dataloader() -> None:
    dataset = torch.utils.data.TensorDataset(torch.zeros(4, 2), torch.zeros(4, 1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, pin_memory=True)
    assert transfer_non_blocking(dataloader, torch.device("cuda"))
    assert not transfer_non_blocking(dataloader, torch.device("cpu"))
