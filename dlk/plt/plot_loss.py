"""
Plotting loss functions.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(
    loss: list | np.ndarray,
    path,
    step_offset: int = 1,
    loss_std: list | np.ndarray | None = None,
    step_label: str = "logged step",
    loss_label: str = "loss",
    title: str = "Training loss",
    x_twin_limits: tuple[int, int] | None = None,
    x_twin_label: str = "",
    y_scale: str = "log",
):
    """Save loss values and plot the training curve."""
    loss = np.array(loss)
    steps = np.arange(step_offset, step_offset + len(loss))

    # save raw numbers
    if loss_std is not None:
        np.savetxt(f"{path}.txt", np.stack((loss, np.array(loss_std)), axis=-1))
    else:
        np.savetxt(f"{path}.txt", loss)

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    if loss_std is not None:
        ax.errorbar(
            steps,
            loss,
            yerr=loss_std,
            fmt="k-",
            ecolor="tab:gray",
            elinewidth=1.0,
        )
    else:
        ax.plot(steps, loss, "k-")

    # set labels
    ax.set_xlabel(step_label)
    ax.set_ylabel(loss_label)
    ax.set_title(title)

    # create a twin x-axis at the top
    if x_twin_limits is not None:
        ax_twin = ax.twiny()
        ax_twin.set_xlim(x_twin_limits)
        ax_twin.set_xlabel(x_twin_label)

    # format and save
    ax.set_xlim((0, steps[-1]))
    ax.set_yscale(y_scale)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f"{path}.pdf", dpi=300)
