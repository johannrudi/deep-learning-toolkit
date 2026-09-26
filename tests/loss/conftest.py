"""Share compile infrastructure across `tests/loss` test modules."""

from collections.abc import Callable, Iterator

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend


@pytest.fixture(autouse=True)
def _reset_dynamo() -> Iterator[None]:
    """Clear compile caches so tests do not share compiled graphs."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def compile_aot_eager() -> Callable[[torch.nn.Module], CompileCounterWithBackend]:
    """Return a callable that compiles a module's `forward` in place.

    The `aot_eager` backend runs AOTAutograd, the source of the double-backward
    limitation, without the cost of inductor code generation.
    """

    def _compile(net: torch.nn.Module) -> CompileCounterWithBackend:
        counter = CompileCounterWithBackend("aot_eager")
        net.forward = torch.compile(net.forward, backend=counter)
        return counter

    return _compile
