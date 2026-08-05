---
Title: Running Distributed Training with DDP
Author: Johann Rudi
Co-Authored-By: Claude Fable 5
Date: 2026-08-04
tags:
  - training
---

# Running Distributed Training with DDP

`dlk` trains with **DDP** (`DistributedDataParallel`): a launcher starts one process per GPU, each process holds a full replica of the model, and gradients are averaged across processes during the backward pass. Going distributed asks more of an application than a bigger batch size does. Each process must discover its **rank** (its index in the run) and device, see a distinct shard of the data, and agree with every other process on the loss statistics it reports.

`dlk.opt.distributed` makes this opt-in. Every helper checks whether a process group is initialized before doing anything: without a launcher, `wrap_net` returns the model unchanged, `sampler_create` returns `None`, and `barrier` returns immediately, so the same script keeps running single-process exactly as before (`tests/opt/test_train_single_process_regression.py` pins this down with frozen loss trajectories). Going distributed is the six code changes below plus a launch command.

!!! note

    The train loops (`dlk.opt.train`, `dlk.opt.train_gan`, `dlk.opt.train_diffusion`) are rank-aware internally and take no new arguments. Do not add rank checks around them: checkpointing and validation already run on the main process only, `validation_fn` receives the unwrapped model(s), the sampler's epoch advances automatically, and the returned loss statistics are reduced exactly across all processes.

## Preparing the application

### Step 1: Initialize the process group

`initialize` replaces manual device selection. It reads the launcher's environment variables and returns a `DistributedContext` with `rank`, `local_rank`, `world_size`, `device`, `is_main`, and `is_distributed`.

```python
from dlk.opt import distributed

ctx = distributed.initialize()
device = ctx.device
```

Detection runs in order: torchrun variables (`RANK`, `WORLD_SIZE`) first, then the Slurm fallback (`SLURM_PROCID` with exported `MASTER_ADDR`/`MASTER_PORT`), and with neither present the function returns a single-process context without creating a process group. Each process logs what it found, so the first lines of a log tell you which path was taken: `distributed run, rank 0/2, local_rank 0, device cpu, backend gloo` or `single-process run, device cuda`. On GPU nodes the device is `cuda:{local_rank}` and the process is pinned to it before the process group starts.

### Step 2: Seed the random generators

Replace manual seeding with the rank-offset helper, keeping the base seed in a variable because Step 5 needs it.

```python
distributed.seed_random_generators(base_seed)
```

The helper seeds `random`, numpy, and torch with `base_seed + rank`, so per-rank random draws (latent samples, augmentation noise) differ across processes. Identical seeds on every rank would make the processes generate the same "random" data, which defeats data parallelism silently.

### Step 3: Rank-suffix the log files

`dlk/mgmt/log.py` opens log files with `filemode="w"`, so all ranks writing to one file name clobber each other. Give each rank its own file:

```python
log_name = name if not ctx.is_distributed else f"{name}_rank{ctx.rank}"
```

Single-process runs keep their unsuffixed file names, so nothing changes for existing workflows.

### Step 4: Wrap the models

Wrap each model after moving it to the device; `wrap_net` requires the model to reside on `device` already.

```python
net = distributed.wrap_net(net.to(device), device)
```

For GAN training, wrap the generator and the discriminator independently, and pass `broadcast_buffers=False` for the discriminator: it runs several forward passes per batch, and each one would otherwise pay a buffer synchronization.

```python
g_net = distributed.wrap_net(g_net, device)
d_net = distributed.wrap_net(d_net, device, broadcast_buffers=False)
```

Keep `find_unused_parameters=False` and `static_graph=False` (the defaults); the train loops are written for them.

### Step 5: Shard the data

Create the sampler with the **base** seed and thread it into the DataLoader:

```python
sampler = distributed.sampler_create(dataset, shuffle=shuffle, seed=base_seed)
if sampler is not None:
    dataloader_kwargs["sampler"] = sampler
    dataloader_kwargs["shuffle"] = False
```

The base seed matters. All ranks must compute the same shuffle permutation before slicing their own shard from it; a rank-offset seed makes ranks shuffle differently, so some samples are trained twice per epoch and others never. `shuffle=False` matters too. Shuffling now belongs to the sampler, and PyTorch rejects the combination with `ValueError: sampler option is mutually exclusive with shuffle`, which is the friendly failure; forgetting the sampler entirely fails silently, with every rank training on the full dataset.

!!! note

    The train loops call `set_epoch` on the sampler each epoch, so the classic silent DDP bug (the same shuffle order every epoch) cannot happen here.

### Step 6: Finish on the main process alone

After training, tear down the process group and let only the main process continue to predict, evaluate, and plot:

```python
distributed.finalize()
if not ctx.is_main:
    return
```

`finalize` runs a barrier before destroying the process group, so no rank exits while another still trains. Everything after this point is ordinary single-process code operating on the unwrapped model (`distributed.unwrap_net(net)` returns it if you kept a wrapped reference).

## Launching on the cluster

### Step 7: Launch with torchrun

torchrun is PyTorch's maintained launcher; it spawns the processes and sets the environment variables that Step 1 reads. On a single node with four GPUs (or four CPU processes when no GPU is present):

```sh
torchrun --standalone --nproc-per-node=4 run.py
```

Every rank logs its own lines, so expect each message multiplied by the world size. The identical loss values across ranks are the reduction from the note above at work:

```text
INFO:dlk.opt.distributed.initialize:distributed run, rank 0/2, local_rank 0, device cpu, backend gloo
INFO:dlk.opt.distributed.initialize:distributed run, rank 1/2, local_rank 1, device cpu, backend gloo
INFO:dlk.opt.train.train_epochs:epoch    0, loss mean 1.184511e+00 std 5.070e-01
INFO:dlk.opt.train.train_epochs:epoch    0, loss mean 1.184511e+00 std 5.070e-01
```

See the [torchrun documentation] for options beyond the ones used here.

### Step 8: Launch under Slurm

For a single-node Slurm job, run torchrun inside the allocation and derive the process count from the allocation itself, so the two cannot drift apart:

```sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

srun torchrun --standalone --nproc-per-node=$SLURM_GPUS_PER_NODE run.py
```

Multi-node jobs need the ranks to find each other across nodes, called the **rendezvous**; point every node at a port on the first node of the allocation:

```sh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$head_node:29500 \
    run.py
```

Where torchrun is unavailable or unwanted, the Slurm fallback runs one Slurm task per process. Its contract is that the job script exports the rendezvous address:

```sh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
srun python run.py
```

`srun` forwards the exported environment by default with `--export=ALL`. (Check your site's default!)

---

## Resuming from a checkpoint

Optional; read this when a run must survive job time limits. The train loops write checkpoints (files `net_e{epoch}.pt` in a timestamped directory under `checkpoint_dir`) from the main process only, and `checkpoint_save` stores the unwrapped model's weights, so checkpoints never contain DDP-prefixed keys.

To resume, every rank loads the same file onto its own device, preferably before wrapping:

```python
from dlk.opt.utils import checkpoint_load

epoch = checkpoint_load(path, net, optimizer=optimizer, map_location=ctx.device)
net = distributed.wrap_net(net, ctx.device)
```

`checkpoint_load` returns the stored epoch, restores the optimizer state when one is passed, and strips a leading `module.` from parameter keys, so checkpoints written before this feature (or by other DDP code) load as well.

---

## Profiling a distributed run

Optional; read this when a distributed run is slower than the world size promises. `dlk.opt.profiler` works per rank: call `profile_train_epochs` or `profile_train_batches` on every rank with the DDP-wrapped model, exactly like the train loops. Each rank writes its own `table_prof_step_{N}_rank{r}.txt` and `trace_prof_step_{N}_rank{r}.json`; the summary table prints on the main process only, and single-process file names stay unsuffixed.

The per-rank traces are the point. Open one in [Perfetto] to see the communication ops (`ncclDevKernel_AllReduce...` kernels under NCCL, CPU-side collectives under gloo) and check that the gradient all-reduce overlaps the backward pass; comparing traces across ranks reveals stragglers. `record_shapes=True` (the module's default) annotates the all-reduce bucket sizes, which is what to inspect when tuning DDP's `bucket_cap_mb`.

`profile_train_batches` profiles 10 batches, so give the dataset at least `10 * batch_size * world_size` samples for unpadded profiled batches.

---

## Things worth knowing

**Shard padding.** `DistributedSampler` pads shards to equal length by repeating samples, so no rank runs out of batches early (a rank that stops while others continue deadlocks the collectives). The repeats bias epoch metrics slightly; pass `drop_last=True` to `sampler_create` to drop trailing samples instead.

**DataLoader workers.** `multiprocessing_context="fork"` is unsafe once CUDA is initialized; under DDP use `"spawn"` or `num_workers=0`. `num_workers` counts per rank, so 4 ranks with 8 workers each start 32 loader processes.

**CPU clusters.** The same commands work without GPUs; the backend auto-selects gloo when CUDA is unavailable, and `initialize(backend="gloo")` forces it.

**NCCL debugging.** `export NCCL_DEBUG=INFO` makes NCCL log its topology and transport decisions; `NCCL_SOCKET_IFNAME=<iface>` pins the network interface when rendezvous picks the wrong one.

**Changing your mind.** There is nothing to turn off. Launch the same script without torchrun (or without the exported `MASTER_ADDR`) and it runs single-process, with unsuffixed logs and no process group.


[torchrun documentation]: https://docs.pytorch.org/docs/stable/elastic/run.html
[Perfetto]: https://ui.perfetto.dev
