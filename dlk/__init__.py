"""Deep Learning Toolkit.

Reusable PyTorch building blocks for artificial intelligence and scientific
machine learning: networks, losses, training loops, and utilities.

Attributes:
    __version__: Version of the installed package, read from its metadata.
        Falls back to "unknown" when the package is not installed.
"""

from importlib.metadata import PackageNotFoundError, version

# read the version from the installed package metadata
try:
    __version__ = version("deep-learning-toolkit")
except PackageNotFoundError:
    __version__ = "unknown"
