"""Run the EfficientNet examples as smoke tests.

Each example is run as a subprocess, the way a reader runs it, with a
temporary output directory, and must save its figures there. See
``docs/features/2026.010__efficientnet_arch_mod__2-plan.md``, Section 4.
"""

import pathlib
import subprocess
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "nets"

# Example, extra command-line arguments, and the figures it must save.
EXAMPLES = {
    "efficientnet_signal_propagation.py": (
        [],
        [
            "signal_propagation_EfficientNetV1BB0.png",
            "signal_propagation_EfficientNetV2BB0.png",
        ],
    ),
    "efficientnet_empirical_lipschitz.py": (
        ["--quick"],
        [
            "lipschitz_blocks_EfficientNetV1BB0.png",
            "lipschitz_blocks_EfficientNetV2BB0.png",
            "lipschitz_networks.png",
        ],
    ),
}


@pytest.mark.parametrize("filename", EXAMPLES)
def test_example_saves_its_figures(filename: str, tmp_path: pathlib.Path) -> None:
    """Run one example end to end and require its figures."""
    path = EXAMPLES_DIR / filename
    assert path.is_file(), f"missing example {path}"
    arguments, figures = EXAMPLES[filename]

    result = subprocess.run(
        [sys.executable, str(path), str(tmp_path), *arguments],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, f"{filename} failed:\n{result.stdout}{result.stderr}"
    for figure in figures:
        assert (tmp_path / figure).is_file(), f"{filename} did not save {figure}"
