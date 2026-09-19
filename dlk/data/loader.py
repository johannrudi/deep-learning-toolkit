import os
from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Optional

import torch


@dataclass
class DataLoaderConfig:
    """Typed argument set for `torch.utils.data.DataLoader`.

    `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`, and
    `multiprocessing_context` each default to an automatic heuristic when their
    configured value is `None`.

    Attributes:
        shuffle: Whether to shuffle samples every epoch.
        drop_last: Whether to drop the last incomplete batch.
        batch_size: Number of samples per batch.
        param_num_workers: Constructor-only input; `None` to auto-detect
            `num_workers` from CPU core count (not stored on the instance).
        param_prefetch_factor: Constructor-only input; `None` to use the default
            `prefetch_factor` when `num_workers > 0` (not stored on the instance).
        param_pin_memory: Constructor-only input; `None` to auto-detect
            `pin_memory` from accelerator availability (not stored on the
            instance).
        param_persistent_workers: Constructor-only input; `None` to use the
            default `persistent_workers` when `num_workers > 0` (not stored on
            the instance).
        param_multiprocessing_context: Constructor-only input; `None` to use the
            default `multiprocessing_context` when `num_workers > 0` (not stored
            on the instance).
        num_workers: CPU subprocesses for data loading.
        prefetch_factor: Batches to prefetch per worker.
        pin_memory: Faster CPU->GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        multiprocessing_context: How to create workers (fork/spawn).
        timeout: Seconds to wait for a batch from a worker before raising.
        pin_memory_device: Device that pinned memory is copied to, if
            `pin_memory` is enabled.
        in_order: Whether to return batches in the order workers produce them.
    """

    shuffle: bool
    drop_last: bool
    batch_size: int
    param_num_workers: InitVar[Optional[int]]
    param_prefetch_factor: InitVar[Optional[int]]
    param_pin_memory: InitVar[Optional[bool]]
    param_persistent_workers: InitVar[Optional[bool]]
    param_multiprocessing_context: InitVar[Optional[str]]
    num_workers: int = field(init=False)
    prefetch_factor: Optional[int] = field(init=False, default=None)
    pin_memory: bool = field(init=False, default=False)
    persistent_workers: bool = field(init=False, default=False)
    multiprocessing_context: Optional[str] = field(init=False, default=None)
    timeout: float = 0
    pin_memory_device: str = ""
    in_order: bool = True

    def __post_init__(
        self,
        param_num_workers: Optional[int],
        param_prefetch_factor: Optional[int],
        param_pin_memory: Optional[bool],
        param_persistent_workers: Optional[bool],
        param_multiprocessing_context: Optional[str],
    ) -> None:
        self.num_workers = (
            param_num_workers
            if param_num_workers is not None
            else self.auto_num_workers()
        )
        self.pin_memory = (
            param_pin_memory
            if param_pin_memory is not None
            else torch.accelerator.is_available()
        )
        if self.num_workers > 0:
            self.prefetch_factor = (
                param_prefetch_factor if param_prefetch_factor is not None else 2
            )
            self.persistent_workers = (
                param_persistent_workers
                if param_persistent_workers is not None
                else True
            )
            self.multiprocessing_context = (
                param_multiprocessing_context
                if param_multiprocessing_context is not None
                else "spawn"
            )

    @staticmethod
    def auto_num_workers(
        min_num_workers: int = 1,
        cpu_cores_ratio: float = 0.2,
        cpu_cores_ratio_device: float = 0.5,
    ) -> int:
        """Heuristic worker count used when `num_workers` is not configured."""
        logical_cpu_cores = os.cpu_count()
        if logical_cpu_cores is None:
            return min_num_workers

        num_workers = logical_cpu_cores
        if torch.accelerator.is_available():
            num_workers = int(num_workers * cpu_cores_ratio_device)
        else:
            num_workers = int(num_workers * cpu_cores_ratio)
        return max(num_workers, min_num_workers)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DataLoaderConfig":
        """Build a `DataLoaderConfig` from a dict of constructor arguments.

        Args:
            config: Mapping of constructor argument names to values.
                `num_workers`, `prefetch_factor`, `pin_memory`,
                `persistent_workers`, and `multiprocessing_context` map to their
                `param_*` constructor inputs; every other key must match a
                `DataLoaderConfig` field name directly.

        Returns:
            A configured `DataLoaderConfig` instance.

        Raises:
            TypeError: If `config` contains a key that is not a constructor
                argument, or omits a required one.
        """
        key_aliases = {
            "num_workers": "param_num_workers",
            "prefetch_factor": "param_prefetch_factor",
            "pin_memory": "param_pin_memory",
            "persistent_workers": "param_persistent_workers",
            "multiprocessing_context": "param_multiprocessing_context",
        }
        kwargs = {key_aliases.get(key, key): value for key, value in config.items()}
        return cls(**kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        """Return this config as constructor kwargs for `DataLoader`."""
        return {field_.name: getattr(self, field_.name) for field_ in fields(self)}
