---
Title: Diagnosing PyTorch CPU Multithreading Behavior
Author: Johann Rudi
Co-Authored-By: Claude Sonnet 5
Date: 2026-08-06
tags:
  - cpu
  - multithreading
  - performance
  - numa
  - training
---

# Diagnosing PyTorch CPU Multithreading Behavior

A CPU-only PyTorch training run decides how many threads to use for its tensor math from a chain of defaults:

- the OS-reported core count,
- the CPU affinity mask the process was launched with,
- a handful of environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`), and,
- on machines with an Intel MKL build, a runtime heuristic called `MKL_DYNAMIC` that can override all of the above per call.

Nothing in the `deep-learning-toolkit` sets a thread count explicitly, so the same training run can use every core on *one* machine and a single core on *another* without any code difference between them, purely because the surrounding environment differs.

This guide walks through that chain of defaults in the order to check them, from the cheapest read of PyTorch's own state down to NUMA topology and CPU pinning. Each step narrows the possible causes, so it's recommended to run them in order. The guide closes with how PyTorch's DataLoader workers fit into the same thread budget, since they are a separate process pool that most of the diagnostics above do not see.

!!! note "Before you start"

    Everything here is Linux-specific: `/proc`, `taskset`, `numactl`, and cgroup files have no equivalent on macOS. If you are diagnosing a Mac, PyTorch's own reporting (Step 1) still works, but affinity and NUMA pinning (Steps 3 through 8) do not apply. The guide also assumes a CPU-only PyTorch build (installed with `uv sync --group cpu` in this repository); a CUDA build adds `torch.cuda` device transfers to the picture that this guide does not cover.

    The tooling used here is [uv] and [numactl].

## Reading what PyTorch already reports

### Step 1: Dump PyTorch's live thread configuration

Before touching the environment, ask PyTorch what it already thinks it should do. Run this from the project's synced environment:

```sh
uv run python3 -c "
import os, torch
print('torch intraop threads:', torch.get_num_threads())
print('torch interop threads:', torch.get_num_interop_threads())
print(torch.__config__.parallel_info())  # shows backend & thread count
print(torch.__config__.show())           # shows build info
"
```

- `torch.get_num_threads()` is the intra-op thread count, the number of threads a single matmul or convolution can spread across.
- `torch.get_num_interop_threads()` governs a different pool, the one used to run independent ops concurrently (autograd task parallelism, mostly irrelevant to a straight-line training loop).

`parallel_info()` prints both numbers again alongside the backend that produced them, for example:

```text
ATen/Parallel:
    at::get_num_threads() : 48
    at::get_num_interop_threads() : 48
OpenMP 201511 (a.k.a. OpenMP 4.5)
    omp_get_max_threads() : 48
Intel(R) oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R) 64 architecture applications
    mkl_get_max_threads() : 48
Environment variables:
    OMP_NUM_THREADS : 48
    MKL_NUM_THREADS : [not set]
ATen parallel backend: OpenMP
```

A `mkl_get_max_threads()` line means the build links Intel MKL for its BLAS routines, which matters later: MKL has its own thread-count heuristic that a plain OpenMP-only build does not. If this number already matches what you expect (roughly your physical core count, or an explicit value you exported), PyTorch's own configuration is not the problem, and the cause lives further down this list, in the OS or in MKL's runtime decisions rather than in torch's reported settings.

### Step 2: Check for thread-limiting environment variables

Several libraries in the dependency stack read their own thread-count variable, independently of PyTorch:

```sh
env | grep -Ei 'OMP_|MKL_|OPENBLAS|NUMEXPR|KMP_|GOMP|VECLIB|SLURM_CPUS'
```

- `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are the two that matter for PyTorch's own ops; if both are unset, PyTorch picks a default based on the visible core count.
- `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` matter only if NumPy or another library in the pipeline was built against OpenBLAS or uses `numexpr`, which this project's dependency stack does not by default.
- `KMP_AFFINITY` and `KMP_BLOCKTIME` configure Intel's OpenMP runtime (`libiomp5`) specifically and are worth noting if present, since they can pin threads to cores independently of the process's own affinity mask (see Step 7).
- `SLURM_CPUS_PER_TASK` is worth checking on a cluster node: batch job wrapper scripts commonly do `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, and that value silently becomes `1` if the job was submitted without requesting multiple CPUs per task.

If an interactive shell and the actual training invocation see different values here, for instance because a login shell exports one thing and a batch scheduler exports another, that mismatch is the more likely explanation than anything in PyTorch. Confirm by running Step 1 inside the exact invocation used for training, not in a separate diagnostic shell.

## Checking whether the OS grants the cores you expect

### Step 3: Compare affinity-aware core counts against the raw core count

PyTorch's default intra-op thread count is derived from the CPU affinity mask visible to the process (`sched_getaffinity` on Linux), not from the total core count of the box. A process confined to one core by a container, a cluster allocation, or a `numactl`/`taskset` wrapper further up the launch chain will default to one thread even on a 96-core machine:

```sh
uv run python3 -c "
import os
print('os.cpu_count():      ', os.cpu_count())
print('affinity-aware cores:', len(os.sched_getaffinity(0)))
"
nproc          # affinity-aware
nproc --all    # ignores affinity, total cores on the box
```

If `nproc` reports far fewer cores than `nproc --all`, the process is affinity-restricted and that restriction, explains a low thread count. If the two numbers match, affinity is not the cause and the next step is to watch the real training process.

### Step 4: Watch the real training process, not a diagnostic script

A diagnostic run in an interactive shell can see a different affinity mask than the actual training invocation, particularly if training is launched through a batch scheduler, a container, or a wrapper script that the interactive shell does not go through. Find the running process and read its affinity directly from `/proc`:

```sh
pgrep -fa run.py  # `run.py` is the script name used in this guide
cat /proc/<PID>/status | grep Cpus_allowed_list
taskset -pc <PID>
```

A CPU-only DataLoader forks additional worker processes (see "Accounting for DataLoader workers" below), so expect more than one PID per training run; check the main process, the one running `python3 run.py` directly rather than the `uv run` or `bash` wrapper around it. A restricted `Cpus_allowed_list`, for example a single core when the host has many, points straight at the launcher; an unrestricted list matching `nproc --all` rules affinity out entirely and moves the investigation to MKL's own thread heuristic in Step 6.

### Step 5: Check for a cgroup quota

Containers and some cluster schedulers restrict CPU access through a cgroup quota rather than (or in addition to) an affinity mask, which `sched_getaffinity` does not always reflect:

```sh
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null    # cgroup v2
cat /sys/fs/cgroup/cpu/cpuset/cpuset.cpus 2>/dev/null   # cgroup v1
cat /sys/fs/cgroup/cpu.max 2>/dev/null                  # cgroup v2 quota, "<quota> <period>"
```

A narrow `cpuset.cpus.effective` list or a low `cpu.max` quota relative to the period (a quota of `100000 100000` grants one full core; `800000 100000` grants eight) caps real parallelism regardless of what `OMP_NUM_THREADS` requests. This step is usually only relevant inside a container or a Kubernetes pod; skip it on a bare-metal or VM host where Steps 3 and 4 already showed full affinity.

## Accounting for MKL's own thread heuristic

If Steps 1 through 5 show correct affinity and a correct `OMP_NUM_THREADS`, but the training process still visibly uses far fewer cores than requested, the remaining cause on an MKL-linked build is `MKL_DYNAMIC`. It defaults to `TRUE`, and with it enabled MKL does not simply honor `MKL_NUM_THREADS`/`OMP_NUM_THREADS` as a fixed count; for each matmul it runs an internal cost estimate weighing the problem size against the overhead of coordinating that many threads, and can fall back to a single thread when it judges the overhead is not worth it. That estimate is sensitive to the total core count and the socket topology of the machine, so the identical batch size and identical thread request can parallelize on an eight-core machine and silently serialize on a ninety-six-core one, where MKL's overhead estimate for spinning up threads across sockets is higher.

### Step 6: Force MKL to honor the requested thread count

```sh
MKL_DYNAMIC=FALSE OMP_NUM_THREADS=<N> ./uv run python3 run.py <args>
```

Watch per-core utilization (`htop`, pressing `t` for the tree view or looking at the per-core meters directly) while this runs. If cores light up that did not before, `MKL_DYNAMIC` was overriding the requested thread count; the number of cores that light up should match `<N>` exactly, since `MKL_DYNAMIC=FALSE` removes MKL's discretion entirely. A run that used one core at `OMP_NUM_THREADS=48` and correctly used all forty-eight once `MKL_DYNAMIC=FALSE` was added confirms this mechanism.

!!! tip

    Getting MKL to use the requested thread count is not the same as getting the fastest training run. More threads add coordination overhead per call, and on a multi-socket machine, threads split across sockets add cross-socket memory latency on top of that. A configuration that correctly uses every requested core can still be slower than a smaller thread count that stays within one socket. Compare wall-clock time across a few values before picking one; see "Sweeping thread counts to find the true optimum" below.

### Step 7: Read the machine's NUMA topology

A multi-socket machine splits its physical cores and memory across NUMA nodes, groups of cores with fast local memory access and slower access to another node's memory. `lscpu` gives a quick summary; `numactl --hardware` gives the full picture including per-node memory:

```sh
lscpu | grep -i "socket\|numa"
numactl --hardware
```

Example output from a two-socket, ninety-six-core machine:

```text
Core(s) per socket:    24
Socket(s):             2
NUMA node(s):          2
NUMA node0 CPU(s):     0-23,48-71
NUMA node1 CPU(s):     24-47,72-95

available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
node 0 size: 257622 MB
node 0 free: 204946 MB
node 1 cpus: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
node 1 size: 257981 MB
node 1 free: 82298 MB
```

Read the CPU list per node carefully: `node0 CPU(s): 0-23,48-71` means logical CPUs 0-23 are the node's twenty-four physical cores, and 48-71 are their hyperthread siblings, not twenty-four additional physical cores. Requesting more threads than the physical core count on one node schedules two threads onto the same physical execution unit, which contends for shared cache and floating-point hardware rather than adding real throughput; for compute-bound matmuls, size a thread count to the physical core count, not the logical one.

The free-memory line is worth reading too. A large imbalance, here 205 GB free on node 0 against 82 GB free on node 1, means something else already holds memory on the busier node, whether another job on a shared host or a prior process that has not released it. A process whose threads land on that node, or whose memory allocations cross into it, competes with whatever is already there on top of the ordinary cross-socket latency.

### Step 8: Pin the process to one NUMA node and compare

With the topology from Step 7 in hand, bind both CPU execution and memory allocation to a single node's physical cores:

```sh
numactl --physcpubind=0-23 --membind=0 env OMP_NUM_THREADS=24 ./uv_run_localdev run.py <args>
```

- `--physcpubind` takes the physical-core range read from `numactl --hardware`, deliberately excluding the hyperthread siblings from Step 7's caveat.
- `--membind=0` forces every memory allocation onto node 0, so a large free-memory node from Step 7 actually gets used rather than only being available in principle.

Where `numactl` is not installed, `taskset -c 0-23` pins CPU execution alone, without the memory guarantee. `taskset` is however *not* able to provide anything analogous to `numactl`'s `--membind`.

Confirm the binding took effect while training runs:

```sh
numastat -p <PID>
```

This reports how a process's resident memory splits across nodes; a successful `--membind=0` should show it almost entirely on node 0. Compare wall-clock training time for this configuration against your best result from Step 6. If pinning beats an unpinned run at the same thread count, the earlier run was paying for cross-socket scheduling; if the two are close, the workload was already staying within one socket on its own and pinning mainly buys reproducibility rather than speed.

---

## Accounting for DataLoader workers

This section is optional background for anyone whose `DataLoader` uses `num_workers > 0`. The steps above tune the thread pool of the main training process; DataLoader workers are separate forked processes with their own thread pool, and it is worth knowing how PyTorch already constrains them before adding manual limits of your own.

### Workers already run single-threaded, by design

Every DataLoader worker calls `torch.set_num_threads(1)` at startup (as of torch version 2.13), unconditionally, inside PyTorch's own `torch/utils/data/_utils/worker.py`. This exists specifically so that `num_workers` worker processes do not each spin up their own full-size OMP or MKL thread pool on top of the main process's; without it, `num_workers=4` on a machine tuned for forty-eight main-process threads would try to run 4 × 48 threads concurrently. None of the tuning in Steps 1 through 8 needs to account for worker processes multiplying the requested thread count, because PyTorch has already ruled that out.

### Workers still compete for cores, and they inherit your pinning

You can increase the workers, for instance, `num_workers = 2` for a CPU-only run, and having two separate DataLoaders, one for training and one for evaluation, both with `persistent_workers=True`. In practice that is four worker processes alive alongside the main process for most of a run, not two; `pgrep -fa run.py` during training shows one PID for the main process and four more for the workers. Each worker is single-threaded internally, per the point above, but each is still a separate OS process wanting scheduler time, so the true core budget for a pinned run is the intra-op thread count from Step 8 plus the worker count, not the thread count alone.

DataLoaders created with `multiprocessing_context = "fork"` inherit CPU affinity across `fork()` on Linux. A `numactl --physcpubind=0-23` wrapped around the main process invocation therefore confines the forked workers to the same range automatically; they do not need, and cannot easily be given, separate pinning. The practical effect is that a Step 8 configuration binding twenty-four physical cores actually has twenty-four intra-op threads and four worker processes contending for those same twenty-four cores. This is usually mild, since the workers mostly alternate between light NumPy preprocessing and blocking on the prefetch queue rather than saturating a core continuously, but where the training-time comparison is close, try the intra-op thread count a few steps below the physical core count (e.g., `OMP_NUM_THREADS=20` instead of `24` on a twenty-four-core node) to leave the workers explicit headroom, and compare.

## Sweeping thread counts to find the true optimum

This section is optional for anyone who has ruled out affinity, cgroup, and `MKL_DYNAMIC` restrictions and now wants the fastest configuration rather than just a correct one. The goal is lowest wall-clock time per epoch. A small timed sweep finds the actual optimum for a given batch size and machine:

```sh
for n in 2 4 8 16 24 32 48; do
  echo "OMP_NUM_THREADS=$n"
  /usr/bin/time -f '%e s' env OMP_NUM_THREADS=$n ./uv run python run.py <args>
done
```

Leave `MKL_DYNAMIC` at its default `TRUE` for this sweep; Step 6 already showed it only downgrades a thread count it judges not worth the overhead, so it will not interfere with finding a genuine optimum and removes the risk of forcing a worse configuration to "work" at full core count. Repeat the sweep, with `numactl --physcpubind`/`--membind` added, once Step 7 has identified the node boundaries, since the useful range of thread counts to try is bounded by the physical core count of one socket, not by the total core count of the machine.

---

## Things worth knowing

**MKL_DYNAMIC is the setting most likely to surprise you.** By default it is on, so a thread count that works correctly on a small machine and silently collapses to one thread on a larger one is expected MKL behavior.

**Grain size limits parallelism independent of thread count.** ATen only splits an op across threads once its element count crosses an internal threshold (on the order of tens of thousands of elements by default); a small batch size or narrow hidden layer can leave every intra-op call below that threshold, in which case no amount of `OMP_NUM_THREADS` tuning adds real parallelism, because there was never enough work to split.

**Hyperthread siblings are not extra physical cores.** A NUMA node reporting forty-eight logical CPUs from `numactl --hardware` typically means twenty-four physical cores and their hyperthread siblings; for compute-bound matmuls, size thread counts to the physical count, since scheduling two MKL threads onto sibling logical CPUs mostly creates contention rather than throughput.

**A shared host can have per-node memory pressure that looks unrelated to threading.** The free-memory imbalance in Step 7's example output, one NUMA node with less than half the free memory of the other, is worth checking on any multi-tenant machine before assuming a performance difference is purely about thread counts; `--membind` in Step 8 is also a way to sidestep a busy node, not only a NUMA-locality optimization.

**Changing your mind.** Undo `MKL_DYNAMIC=FALSE` by simply omitting it from the invocation; the default (`TRUE`) resumes on the next unset run. Undo `numactl` pinning the same way, by dropping the wrapper; nothing here writes persistent state.


[uv]: https://github.com/astral-sh/uv
[numactl]: https://github.com/numactl/numactl
