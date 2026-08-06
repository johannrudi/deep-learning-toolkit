# Voice Examples

Paired passages showing the same content written weakly and written well. The strong versions are grounded in `docs/features/2026.02__ddp__2-usage.md` and the code in `dlk/opt/distributed.py`, `dlk/opt/train.py`, and `dlk/opt/train_gan.py`. Read a pair, then look at the diagnosis under it.

## 1. Opening a guide

**Weak**

> In this guide, we will walk you through the process of running distributed training with the dlk toolkit using PyTorch's DDP. We will cover initialization, data loading, and launching. By the end, you will be able to train your models on multiple GPUs.

**Strong**

> dlk trains with **DDP** (`DistributedDataParallel`): a launcher starts one process per GPU, each process holds a full replica of the model, and gradients are averaged across processes during the backward pass. Going distributed asks more of an application than a bigger batch size does. Each process must discover its rank and device, see a distinct shard of the data, and agree with every other process on the loss statistics it reports.

The weak version spends four sentences describing the document. The strong version spends them describing the subject, defines `DDP` in its first clause, and by the end the reader already knows the three things that make this task harder than it looks. A table of contents can list sections; prose should not.

## 2. Justifying an instruction

**Weak**

> Make sure to pass `shuffle=False` to the DataLoader when using a distributed sampler.

**Strong**

> `shuffle=False` in the DataLoader matters. Shuffling now belongs to the sampler: each epoch, all ranks compute the same permutation of the dataset and slice their own shard from it. Passing `shuffle=True` alongside a sampler makes PyTorch raise `ValueError: sampler option is mutually exclusive with shuffle`, which is the friendly failure; forgetting the sampler entirely fails silently, with every rank training on the full dataset.

The weak version produces obedience. The strong version produces understanding, which is what the reader needs when their situation differs from the guide's. It also names both failure shapes, so the reader can recognize the silent one.

## 3. Describing behavior you had to read the source to learn

**Weak**

> Note that the generator output is detached during the discriminator step.

**Strong**

> Inside a discriminator step, `train_gan` runs the generator under `torch.no_grad()` instead of detaching its output. The distinction matters once the generator is DDP-wrapped: a forward pass through a DDP module arms its gradient reducer for a backward pass, and `.detach()` on the output does nothing to disarm it. The reducer then waits for gradients that never come, which is a miserable bug to chase. `no_grad` keeps the reducer out of the picture entirely.

The weak version is the answer you get by assuming a DDP-wrapped module behaves like a plain module. It is wrong, and a reader following it loses an afternoon. The strong version came out of reading `dlk/opt/train_gan.py`. It names the mechanism, gives the design reason, and warns about the exact trap.

## 4. Reassuring the reader about risk

**Weak**

> Note that these helpers should not affect single-process runs.

**Strong**

> Every helper checks whether a process group is initialized before doing anything. Without a launcher, `wrap_ddp` returns the model unchanged, `create_distributed_sampler` returns `None`, and `barrier` returns immediately. The same script runs single-process exactly as before, and `tests/opt/test_train_single_process_regression.py` pins this down with frozen loss trajectories that must match to the last digit.

"Should not" invites doubt. The strong version explains the mechanism that makes the guarantee true and names a test the reader can run, so the guarantee becomes a checkable fact rather than a promise.

## 5. Handling a genuine uncertainty

**Weak**

> Slurm forwards the environment to all tasks.

**Strong**

> Under the Slurm fallback, `MASTER_ADDR` and `MASTER_PORT` must reach every task. `srun` forwards the exported environment by default with `--export=ALL`. (Check your site's default!)

The weak version is true at most sites and false at some, and the reader has no way to tell which they are at. The parenthetical costs four words and converts a possible dead end into a thing the reader knows to check.

## 6. Rewarding attention

**Strong**

> That reduction is worth appreciating. Each rank contributes its running sums to the all-reduce, its finished means stay local, so the reported statistics are exactly what a single process would have computed over the whole dataset. Averaging per-rank means instead would quietly bias the result whenever shards differ in length.

Nothing here is strictly necessary. It is three sentences that tell the reader why a piece of engineering was done a particular way, and they are the sentences that make a guide feel personable. Budget roughly one such passage per major section, and only where there is something actually interesting to say.

## 7. Sentences that move forward

**Weak**

> `validation_fn` does not receive the DDP-wrapped model, but rather the underlying module.

**Strong**

> `validation_fn` receives the unwrapped module, the same object the application constructed, so existing validation and plotting code runs untouched.

The weak version makes the reader load a false statement, hold it, and then discard it. The strong version states the fact and immediately says why the design is that way. The same fix applies to any "not X, but Y" you catch in a draft.
