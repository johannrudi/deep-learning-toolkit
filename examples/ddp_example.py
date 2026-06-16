"""Tiny supervised-training test for torchrun/DDP.

Launch examples:

    python examples/ddp_example.py
    torchrun --standalone --nproc_per_node=1 examples/ddp_example.py
    torchrun --standalone --nproc_per_node=2 examples/ddp_example.py
"""

import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from dlk import distributed
from dlk.opt.train import train_epochs


def main() -> None:
    device = distributed.init_process_group()
    try:
        logging.basicConfig(
            level=logging.INFO if distributed.is_main_process() else logging.WARNING,
            format="[rank %(rank)s] %(name)s: %(message)s",
        )
        logging.setLogRecordFactory(
            _rank_record_factory(distributed.get_rank(), logging.getLogRecordFactory())
        )

        torch.manual_seed(0)
        x = torch.linspace(-1.0, 1.0, 128).unsqueeze(1)
        y = 2.0 * x - 0.25
        dataset = TensorDataset(x, y)
        sampler = distributed.create_distributed_sampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=sampler is None,
            sampler=sampler,
        )

        model = torch.nn.Linear(1, 1).to(device)
        model = distributed.wrap_ddp(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        train_epochs(
            n_epochs=30,
            net=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=torch.nn.MSELoss(),
            device=device,
            checkpoint_epochs=None,
        )

        # Self-check: linear regression should approximately recover y = 2x - 0.25.
        trained = distributed.unwrap_ddp(model)
        weight = trained.weight.detach().item()
        bias = trained.bias.detach().item()
        if distributed.is_main_process():
            logging.getLogger("ddp_example").info(
                f"final weight {weight:.3f} (target 2.000), "
                f"bias {bias:.3f} (target -0.250)"
            )
        assert abs(weight - 2.0) < 0.25, f"weight {weight} far from 2.0"
        assert abs(bias - (-0.25)) < 0.1, f"bias {bias} far from -0.25"

        distributed.barrier()
    finally:
        distributed.cleanup()


def _rank_record_factory(rank: int, base):
    def factory(*args, **kwargs):
        record = base(*args, **kwargs)
        record.rank = rank
        return record

    return factory


if __name__ == "__main__":
    main()
