"""Build optimizers."""

import torch
import torch.nn as nn


def create_optimizer_from_config(
    net: nn.Module,
    opt_params: dict,
) -> torch.optim.Optimizer:
    """Create Adam or AdamW optimizer from config."""
    opt_type = opt_params["type"].casefold()
    lr = opt_params["learning_rate"]
    betas = (
        opt_params.get("beta1", 0.9),
        opt_params.get("beta2", 0.999),
    )
    eps = opt_params.get("epsilon", 1e-8)

    if opt_type == "adam":
        return torch.optim.Adam(net.parameters(), lr=lr, betas=betas, eps=eps)

    if opt_type == "adamw":
        weight_decay = opt_params.get("weight_decay", 1e-2)
        return torch.optim.AdamW(
            net.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    raise ValueError(f"unknown optimizer type: {opt_type}")
