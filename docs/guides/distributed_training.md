---
Title: Running Distributed Training with DDP
Author: Johann Rudi
Co-Authored-By: Claude Fable 5 and Claude Opus 5
Date: 2026-08-29
tags:
  - training
  - profiling
  - performance
  - distributed
---

# Running Distributed Training with DDP

`dlk` trains with **DDP** (`DistributedDataParallel`): a launcher starts one process per device, each process holds a full replica of the model, and gradients are averaged across processes during the backward pass. Going distributed asks more of an application than a bigger batch size does. Each process must discover its **rank** (its index in the run) and device, see a distinct shard of the data, and agree with every other process on the loss statistics it reports.

`dlk.opt.distributed` makes this opt-in. Every helper checks whether a process group is initialized before doing anything: without a launcher, `wrap_net` returns the model unchanged, `sampler_create` returns `None`, and `barrier` returns immediately, so the same script keeps running single-process exactly as before (`tests/opt/test_train_single_process_regression.py` pins this down with frozen loss trajectories). Going distributed is the six code changes below plus a launch command.

> [!NOTE]
> The train loops (`dlk.opt.train`, `dlk.opt.train_gan`, `dlk.opt.train_diffusion`) are rank-aware internally and take no new arguments. Do not add rank checks around them: checkpointing and validation already run on the main process only, `validation_fn` receives the unwrapped model(s), the sampler's epoch advances automatically, and the returned loss statistics are reduced exactly across all processes.

## Preparing the application

### Step 1: Open a distributed session

`session` replaces manual device selection and process-group management. It reads the launcher's environment variables, creates the process group, and destroys it again on every path out of the `with` block, including an exception.

```python
from dlk.opt import distributed

with distributed.session() as ctx:
    device = ctx.device
    ...  # Steps 2 to 5 happen inside this block
```

`ctx` carries `rank`, `local_rank`, `world_size`, `local_world_size`, `device`, `num_threads`, `is_main`, and `is_distributed`. Detection runs in order: torchrun variables (`RANK`, `WORLD_SIZE`) first, then the Slurm fallback (`SLURM_PROCID` with exported `MASTER_ADDR`/`MASTER_PORT`), and with neither present the function returns a single-process context without creating a process group. Each process logs what it found, so the first lines of a log tell you which path was taken:

```text
INFO:dlk.opt.distributed.initialize:distributed run, rank 0/2, local_rank 0/2, device cpu, backend gloo, num_threads 8
INFO:dlk.opt.distributed.initialize:single-process run, device cpu, num_threads 16
```

On GPU nodes the device is `cuda:{local_rank}` with the nccl backend, and the process is pinned to it before the process group starts. Prefer the `with` block over calling `initialize()` and `finalize()` yourself: when one rank crashes, `session` tears the others down immediately instead of leaving them in a barrier until the 1800-second timeout expires.

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
g_net = distributed.wrap_net(g_net.to(device), device)
d_net = distributed.wrap_net(d_net.to(device), device, broadcast_buffers=False)
```

Keep `find_unused_parameters=False` and `static_graph=False` (the defaults); the train loops are written for them.

### Step 5: Shard the data

Create the sampler with the context and the **base** seed, then thread it into the DataLoader:

```python
sampler = distributed.sampler_create(dataset, ctx=ctx, shuffle=shuffle, base_seed=base_seed)
if sampler is not None:
    dataloader_kwargs["sampler"] = sampler
    dataloader_kwargs["shuffle"] = False
```

The base seed matters. All ranks must compute the same shuffle permutation before slicing their own shard from it; a rank-offset seed makes ranks shuffle differently, so some samples are trained twice per epoch and others never. `shuffle=False` matters too. Shuffling now belongs to the sampler, and PyTorch rejects the combination with `ValueError: sampler option is mutually exclusive with shuffle`, which is the friendly failure; forgetting the sampler entirely fails silently, with every rank training on the full dataset.

> [!NOTE]
> The train loops call `set_epoch` on the sampler each epoch, so the classic silent DDP bug (the same shuffle order every epoch) cannot happen here.

### Step 6: Finish on the main process alone

Prediction, evaluation, and plotting belong outside the session, on one rank:

```python
with distributed.session() as ctx:
    ...  # training

if ctx.is_main:
    evaluate(distributed.unwrap_net(net))
```

Leaving the `with` block synchronizes the ranks and destroys the process group, so no rank exits while another still trains. Everything after it is ordinary single-process code; `distributed.unwrap_net(net)` returns the model inside a DDP wrapper, and any other model unchanged.

## Launching on the cluster

### Step 7: Launch with torchrun

torchrun is PyTorch's maintained launcher; it spawns the processes and sets the environment variables that Step 1 reads. On a single node with four GPUs (or four CPU processes when no GPU is present):

```sh
uv run torchrun --standalone --nproc-per-node=4 run.py
```

Every rank logs its own lines, so expect each message multiplied by the world size. The identical loss values across ranks are the reduction from the note above at work:

```text
INFO:dlk.opt.distributed.initialize:distributed run, rank 0/2, local_rank 0/2, device cpu, backend gloo, num_threads 8
INFO:dlk.opt.distributed.initialize:distributed run, rank 1/2, local_rank 1/2, device cpu, backend gloo, num_threads 8
INFO:dlk.opt.train.train_epochs:epoch    0, loss mean 3.434223e+00 std 1.560e+00
INFO:dlk.opt.train.train_epochs:epoch    0, loss mean 3.434223e+00 std 1.560e+00
```

See the [torchrun documentation] for options beyond the ones used here.

### Step 8: Launch under Slurm

For a single-node Slurm job, run torchrun inside the allocation and derive the process count from the allocation itself, so the two cannot drift apart:

```sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4

srun torchrun --standalone --nproc-per-node=$SLURM_GPUS_PER_NODE run.py
```

Give each rank at least one core, as above. Every rank is a separate process driving one GPU: it runs the training loop, issues the CUDA kernel launches, and does NCCL's host-side coordination, all of it on the CPU. Allocate fewer cores than ranks and the rank processes time-slice onto the cores you did allocate, so a descheduled rank stops launching kernels while its GPU drains its queue and idles. Models built from many small kernels feel this first, since launch overhead already dominates their step time.

One core per GPU is the floor rather than the target. Where the node is yours anyway, divide its cores evenly among the GPUs. Add the DataLoader workers on top of that: `num_workers` counts per rank and each worker is its own process, so four ranks at `num_workers=3` want 4 x (1 + 3) = 16 cores.

```sh
#SBATCH --cpus-per-task=16    # 4 ranks x (1 training process + 3 DataLoader workers)
#SBATCH --gpus-per-node=4
```

> [!WARNING]
> **`--cpus-per-task` counts per Slurm task, and the two layouts here differ.** With `--ntasks-per-node=1` and `torchrun --nproc-per-node=4`, all four ranks share one task's cores, so `--cpus-per-task=16` gives each rank four. With `--ntasks-per-node=4` and `srun python run.py`, each rank is its own task, so `--cpus-per-task=4` already gives each rank four. Size `OMP_NUM_THREADS` to the per-rank share: `$SLURM_CPUS_PER_TASK` in the second layout, `$SLURM_CPUS_PER_TASK` divided by the ranks per task in the first; confirm with `torch.get_num_threads()`.

Multi-node jobs need the ranks to find each other across nodes, called the **rendezvous**; point every node at a port on the first node of the allocation:

```sh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
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
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
srun python run.py
```

`srun` forwards the exported environment by default with `--export=ALL`. (Check your site's default!)

---

## Confirming that the run is really distributed

Optional; read this the first time an application goes distributed, or when the run is no faster than a single process. Two checks answer the two questions that matter, and both are **collectives**: every rank must call them, or the run hangs.

```python
batch = next(iter(dataloader))[0].to(ctx.device)
print(f"[rank {ctx.rank}] batches differ: {distributed.check_batch_ranks_differ(batch)}")
print(f"[rank {ctx.rank}] replica drift: {distributed.check_net_replica_drift(net)}")
```

```text
[rank 0] batches differ: True
[rank 1] batches differ: True
[rank 0] replica drift: 0.0
[rank 1] replica drift: 0.0
```

`check_batch_ranks_differ` reports whether the ranks drew different samples; a `False` means every rank is training on the same data, which turns a four-way run into four copies of one gradient and points at a missing sampler in Step 5. `check_net_replica_drift` reports the largest parameter difference from rank 0 over all ranks; `0.0` is a healthy DDP run, and anything else means the replicas diverged, typically from a rank-dependent initialization or an optimizer step taken outside DDP. Pass `include_buffers=True` to compare batch-norm running statistics as well. Delete both calls once the run checks out; they are debugging aids, not training-loop code.

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

Open a per-rank trace in [Perfetto] to see the communication ops (`ncclDevKernel_AllReduce...` kernels under NCCL, CPU-side collectives under gloo) and to check that the gradient all-reduce overlaps the backward pass; comparing traces across ranks reveals stragglers. `record_shapes=True` (the module's default) annotates the all-reduce bucket sizes, which is what to inspect when tuning DDP's `bucket_cap_mb`.

`profile_train_batches` profiles 10 batches, so give the dataset at least `10 * batch_size * world_size` samples for unpadded profiled batches.

---

## Things worth knowing

**Thread counts are reported, never set.** `OMP_NUM_THREADS` stays the only way to control intra-op threads; the session never calls `torch.set_num_threads`, it only records what it finds as `ctx.num_threads` and at the end of the initialization log line. Mind that torchrun exports `OMP_NUM_THREADS=1` when the variable is unset, so set it explicitly to the per-rank core share (see the caveat in Step 8), and see [Diagnosing PyTorch CPU Multithreading Behavior](multithreading_cpu.md) when CPU throughput still disappoints.

**Shard padding.** `DistributedSampler` pads shards to equal length by repeating samples, so no rank runs out of batches early (a rank that stops while others continue deadlocks the collectives). The repeats bias epoch metrics slightly; pass `drop_last=True` to `sampler_create` to drop trailing samples instead.

**DataLoader workers.** `multiprocessing_context="fork"` is unsafe once CUDA is initialized; under DDP use `"spawn"` or `num_workers=0`. `num_workers` counts per rank, so 4 ranks with 8 workers each start 32 loader processes, and every one of them wants a core from the allocation in Step 8. See the [torch multiprocessing documentation].

**CPU clusters and other accelerators.** The same commands work without GPUs; the backend follows the device (cuda to nccl, Intel GPUs to xccl, CPU and MPS to gloo), and `session(backend="gloo")` forces it.

**NCCL debugging.** `export NCCL_DEBUG=INFO` makes NCCL log its topology and transport decisions; `NCCL_SOCKET_IFNAME=<iface>` pins the network interface when rendezvous picks the wrong one.

**Changing your mind.** There is nothing to turn off. Launch the same script without torchrun (or without the exported `MASTER_ADDR`) and it runs single-process, with unsuffixed logs and no process group.

## Fixing common issues

**`ValueError: sampler option is mutually exclusive with shuffle`.** The DataLoader received both a sampler and `shuffle=True`. Set `shuffle=False` there and pass the shuffling intent to `sampler_create` (Step 5).

**`RuntimeError: local rank 2 needs at least 3 cuda devices, but only 2 are visible`.** More processes were started than devices are visible to them, usually `--nproc-per-node` exceeding the GPUs per node, or a stale `CUDA_VISIBLE_DEVICES`. Match the process count to the devices, or widen the mask.

**`RuntimeError: context reports world_size 4, but no process group is initialized`.** `sampler_create` received a multi-process context outside a live session. Move the call inside the `with distributed.session()` block (Steps 1 and 5).

**The job hangs instead of exiting.** One rank left a collective the others are still waiting in. Check that every rank reaches every collective, including the two checks above, and use `session` rather than a hand-written `initialize`/`finalize` pair so a crash does not strand the survivors in a barrier.

**Every rank writes the same log file.** Step 3 was skipped; the ranks truncate each other's output because `filemode="w"`.

## Learn more

- [torchrun documentation] for elasticity, rendezvous backends, and the launcher's own flags.
- [DDP documentation] for the wrapper's semantics, `bucket_cap_mb`, and gradient accumulation with `no_sync`.
- [Slurm srun documentation] for `--export`, task layout, and the variables the fallback path reads.
- [Perfetto] for opening the traces from the profiling section.


[torchrun documentation]: https://docs.pytorch.org/docs/stable/elastic/run.html
[DDP documentation]: https://docs.pytorch.org/docs/stable/notes/ddp.html
[Slurm srun documentation]: https://slurm.schedmd.com/srun.html
[torch multiprocessing documentation]: https://docs.pytorch.org/docs/main/notes/multiprocessing.html
[Perfetto]: https://ui.perfetto.dev
