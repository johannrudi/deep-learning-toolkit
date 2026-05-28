"""Forward solvers to sample from flow matching diffusion models."""

from typing import Any, Callable, Sequence, Tuple, Union

import torch
from flow_matching.solver import ODESolver
from torch import nn


def _to_tensor(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """Convert a tensor or sequence of tensors to a single tensor.

    Args:
        x: A tensor or sequence of tensors to normalize.

    Returns:
        A tensor. If x is already a tensor, returns it unchanged.
        If x is a sequence, stacks it along a new leading dimension.
    """
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack(list(x))


class Sampler:
    """Sample from a flow matching model using an ODE solver.

    Wraps ``ODESolver`` from the ``flow_matching`` package to provide
    generation and likelihood estimation for a learned velocity field.

    Attributes:
        forward_solver: ODE solver driven by the velocity network.
    """

    def __init__(
        self,
        velocity_net: nn.Module,
    ):
        """Initialize the sampler.

        Args:
            velocity_net: Neural network that predicts the velocity field
                used by the ODE solver.
        """
        self.forward_solver = ODESolver(velocity_model=velocity_net)

    def time_steps(
        self,
        num_steps: int = 10,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build a uniform time grid from 0 to 1.

        Args:
            num_steps: Number of evenly spaced time points.
            dtype: Floating-point dtype of the output tensor.
            device: Device on which to allocate the tensor.

        Returns:
            1-D tensor of shape ``(num_steps,)`` with values in ``[0, 1]``.
        """
        return torch.linspace(0.0, 1.0, num_steps, dtype=dtype, device=device)

    def initial_condition(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Draw the initial noise sample from the standard normal distribution.

        Args:
            shape: Shape of the output tensor, typically ``(batch_size, ...)``,
                where the trailing dimensions match the data dimensionality.
            dtype: Floating-point dtype of the output tensor.
            device: Device on which to allocate the tensor.

        Returns:
            Tensor of the given shape sampled from ``N(0, I)``.
        """
        return torch.randn(shape, dtype=dtype, device=device)

    def generate(
        self,
        step_size: float | None,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        time_steps: torch.Tensor | None = None,
        model_kwargs: dict[str, Any] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Generate samples by integrating the velocity field from t=0 to t=1.

        Draws an initial condition from ``N(0, I)`` and solves the ODE forward
        in time. When ``time_steps`` is provided, the solver returns the
        trajectory at every requested time point; otherwise only the final
        sample at t=1 is returned.

        Args:
            step_size: Fixed step size for the ODE solver. Pass ``None`` to
                use an adaptive step size solver.
            shape: Shape of the sample tensor, typically ``(batch_size, ...)``,
                where the trailing dimensions match the data dimensionality.
            dtype: Floating-point dtype used for the initial condition and
                the time grid.
            device: Device on which to run the solver.
            time_steps: Custom 1-D time grid in ``[0, 1]``. When provided,
                the returned tensor contains the ODE state at each time point
                and has shape ``(len(time_steps), *shape)``. When ``None``,
                only the final state is returned with shape ``shape``.
            model_kwargs: Optional keyword arguments forwarded directly to
                the velocity network at every ODE step (e.g.
                ``{"y": condition}`` for conditional generation).
            solver_kwargs: Optional keyword arguments forwarded to
                ``ODESolver.sample`` (e.g. ``method``, ``atol``, ``rtol``).

        Returns:
            Generated samples. Shape is ``shape`` when ``time_steps`` is
            ``None``, or ``(len(time_steps), *shape)`` otherwise.
        """
        # set up time steps
        if time_steps is None:
            time_steps = self.time_steps(2, dtype=dtype, device=device)
            return_intermediate_time_steps = False
        else:
            return_intermediate_time_steps = True

        # create the initial condition
        init_cond = self.initial_condition(shape, dtype=dtype, device=device)

        # merge model kwargs into a fresh dict to avoid mutating the default
        _solver_kwargs: dict[str, Any] = dict(solver_kwargs or {})
        if model_kwargs is not None:
            _solver_kwargs.update(model_kwargs)

        # generate samples using the forward ODE solver
        samples = self.forward_solver.sample(
            x_init=init_cond,
            step_size=step_size,
            time_grid=time_steps,
            return_intermediates=return_intermediate_time_steps,
            **_solver_kwargs,
        )

        return _to_tensor(samples)

    def compute_likelihood(
        self,
        step_size: float | None,
        target_sample: torch.Tensor,
        source_log_prob: Callable[[torch.Tensor], torch.Tensor],
        model_kwargs: dict[str, Any] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the log-likelihood of a target sample under the model.

        Solves the ODE in reverse (from t=1 to t=0) to map the target sample
        back to the source distribution, then computes the log-likelihood via
        the instantaneous change-of-variables formula.

        Args:
            step_size: Fixed step size for the ODE solver. Pass ``None`` to
                use an adaptive step size solver.
            target_sample: Observed data tensor of shape ``(batch_size, ...)``
                representing samples from the target distribution at t=1.
            source_log_prob: Log-probability function of the source
                distribution. Accepts a tensor of shape ``(batch_size, ...)``
                and returns a tensor of shape ``(batch_size,)``.
            model_kwargs: Optional keyword arguments forwarded directly to
                the velocity network at every ODE step (e.g.
                ``{"y": condition}`` for conditional likelihood estimation).
            solver_kwargs: Optional keyword arguments forwarded to
                ``ODESolver.compute_likelihood`` (e.g. ``method``, ``atol``,
                ``rtol``, ``exact_divergence``).

        Returns:
            A tuple ``(source_samples, log_likelihood)`` where:

            - ``source_samples``: tensor of shape ``(batch_size, ...)``
              containing the mapped source points at t=0.
            - ``log_likelihood``: tensor of shape ``(batch_size,)`` with the
              estimated log-likelihood of each target sample.
        """
        # merge model kwargs into a fresh dict to avoid mutating the default
        _solver_kwargs: dict[str, Any] = dict(solver_kwargs or {})
        if model_kwargs is not None:
            _solver_kwargs.update(model_kwargs)

        # compute the likelihood log probability at the target samples
        samples, log_prob = self.forward_solver.compute_likelihood(
            x_1=target_sample,
            log_p0=source_log_prob,
            step_size=step_size,
            **_solver_kwargs,
        )

        return _to_tensor(samples), log_prob
