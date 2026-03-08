"""Provide shared utilities for metric computations."""

from typing import Literal

import torch


def pairwise_distances(
    x: torch.Tensor,
    y: torch.Tensor,
    p: Literal[1, 2] = 2,
) -> torch.Tensor:
    """Compute pairwise distances between rows of two matrices.

    Args:
        x: Input tensor with shape ``[N, D]``.
        y: Input tensor with shape ``[M, D]``.
        p: Distance mode selector. Use ``1`` for Euclidean distance and ``2`` for
            squared Euclidean distance.

    Returns:
        A tensor with shape ``[N, M]`` where entry ``(i, j)`` is the distance
        between ``x[i]`` and ``y[j]``.

    Raises:
        ValueError: If ``x`` or ``y`` is not 2D.
        ValueError: If ``x`` and ``y`` do not share the same feature dimension.
        ValueError: If ``p`` is not one of ``{1, 2}``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D with shape [N, D], got shape {tuple(x.shape)}")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D with shape [M, D], got shape {tuple(y.shape)}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "x and y must have the same feature dimension, "
            f"got x.shape[1]={x.shape[1]} and y.shape[1]={y.shape[1]}"
        )
    if not p in [1, 2]:
        raise ValueError(f"p must be 1 (Euclidean) or 2 (squared Euclidean), got {p=}")

    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    sq = torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)
    if p == 2:
        return sq
    if p == 1:
        return torch.sqrt(sq + 1e-12)
