import math
import pathlib

import torch


def checkpoint_path(checkpoint_dir, n_epochs, prefix, epoch):
    n_digits = int(math.ceil(math.log10(1.01 * n_epochs)))
    fmt = "{}_e{:0" + str(n_digits) + "d}.pt"
    filename = fmt.format(prefix, epoch)
    filepath = pathlib.Path(checkpoint_dir) / filename
    return filepath


def checkpoint_save(model, filepath, epoch, optimizer):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filepath,
    )


@torch.no_grad()
def train_dlog_batch_initialize(n_batches, tags, save_list=False):
    dlog = {"n_batches": n_batches}
    for tag in tags:
        if save_list:
            dlog[tag] = torch.empty((n_batches,), dtype=torch.float64)
        else:
            dlog[tag] = None
        dlog[f"{tag}_mean_n"] = 0
        dlog[f"{tag}_mean"] = 0.0
        dlog[f"{tag}_sq_mean"] = 0.0
        dlog[f"{tag}_std"] = None
    return dlog


@torch.no_grad()
def train_dlog_batch_update(dlog, batch_idx, values):
    for tag in values.keys():
        if tag in values:
            val = values[tag]
        else:
            val = math.nan
        if isinstance(val, torch.Tensor):
            val = float(val.detach().item())
        if dlog[tag] is not None:
            dlog[tag][batch_idx] = val
        if not math.isnan(val):
            dlog[f"{tag}_mean_n"] += 1
            dlog[f"{tag}_mean"] += val
            dlog[f"{tag}_sq_mean"] += val * val


@torch.no_grad()
def train_dlog_batch_finalize(dlog, tags):
    for tag in tags:
        assert dlog[f"{tag}_std"] is None
        assert not math.isnan(dlog[f"{tag}_mean"])
        assert not math.isnan(dlog[f"{tag}_sq_mean"])
        if 0 < dlog[f"{tag}_mean_n"]:
            dlog[f"{tag}_mean"] *= 1.0 / dlog[f"{tag}_mean_n"]
            dlog[f"{tag}_sq_mean"] *= 1.0 / dlog[f"{tag}_mean_n"]
            variance = dlog[f"{tag}_sq_mean"] - dlog[f"{tag}_mean"] ** 2
            dlog[f"{tag}_std"] = torch.sqrt(
                torch.tensor(variance, dtype=torch.float64)
            ).item()
        else:
            dlog[f"{tag}_mean"] = 0.0
            dlog[f"{tag}_sq_mean"] = 0.0
            dlog[f"{tag}_std"] = 0.0


@torch.no_grad()
def train_dlog_epoch_initialize(n_epochs, tags, save_list=True):
    dlog = {}
    for tag in tags:
        if save_list:
            dlog[tag] = torch.empty((n_epochs,), dtype=torch.float64)
        else:
            dlog[tag] = None
    dlog["batch_dlog"] = []
    return dlog


@torch.no_grad()
def train_dlog_epoch_update(dlog, epoch_idx, tags, batch_dlog):
    for tag in tags:
        if dlog[tag] is not None:
            dlog[tag][epoch_idx] = batch_dlog[tag]
    dlog["batch_dlog"].append(batch_dlog)


def train_dlog_epoch_finalize(dlog, time_train):
    dlog["time_train"] = time_train
