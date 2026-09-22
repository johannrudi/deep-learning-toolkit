"""Run the composition examples as smoke tests.

The examples in `examples/nets/` are the end-to-end exercise of
`dlk.nets.compose`: they reach real entry points that the unit tests do not,
`dlk.opt.train_gan.train_epochs` and `torch.vmap`, and they assert their own
results. Running them here keeps them from rotting.

Each example is run as a subprocess, the way a reader runs it, so a broken
import or a missing `__main__` guard fails the suite rather than passing
unnoticed.
"""

import pathlib
import subprocess
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "nets"

EXAMPLES = [
    "compose_conditional_gan.py",
    "compose_replica_latents.py",
]


@pytest.mark.parametrize("filename", EXAMPLES)
def test_example_runs_as_a_script(filename: str) -> None:
    """Run one example end to end and require it to succeed."""
    path = EXAMPLES_DIR / filename
    assert path.is_file(), f"missing example {path}"

    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, f"{filename} failed:\n{result.stdout}{result.stderr}"


def test_conditional_gan_example_reports_finite_losses() -> None:
    """Train the composed generator for one epoch and check the losses."""
    example = _load("compose_conditional_gan.py")

    losses = example.main(n_epochs=1)

    assert set(losses) == {"g_loss", "d_pre_loss", "d_post_loss"}
    assert all(value == value for value in losses.values())  # not NaN
    assert all(abs(value) < float("inf") for value in losses.values())


def test_replica_example_returns_one_sample_per_replica() -> None:
    """Generate replicas through the split forward and check the shape."""
    example = _load("compose_replica_latents.py")

    x_gen = example.main(batch_size=2, replicas=3)

    assert x_gen.shape == (3, 2, example.SAMPLE_SIZE)


def _load(filename: str):  # type: ignore[no-untyped-def]
    """Import one example module by path.

    Args:
        filename: Name of the example file in `examples/nets/`.

    Returns:
        The imported module.
    """
    import importlib.util

    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
