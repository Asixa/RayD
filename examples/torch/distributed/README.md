# Process-per-GPU recipes

Two runnable recipes for driving RayD with **one process per GPU**, the shape
that scales from a single node to a cluster:

- [`ddp_intersect_train.py`](ddp_intersect_train.py) — a rank-sharded global ray
  batch, a differentiable `scene.intersect` loss, and one `all_reduce(SUM)` of
  the vertex gradient per step. The optimizer applies the identical update on
  every rank, and the script asserts zero cross-rank parameter drift and prints
  a hash of the final parameter.
- [`ddp_accum_grids.py`](ddp_accum_grids.py) — a rank-sharded Monte-Carlo
  accumulation: each rank runs `accum_dfr_direct` over its own lane window of
  one global sample space, and the partial grids are `all_reduce(SUM)`-ed into
  the grid the single launch would have produced.

This is the **only** multi-GPU route for the Dr.Jit backend (Dr.Jit is
single-device per process), and it is the cluster-scale route for both backends.
For single-node Torch work there is also the in-process layer,
`Scene(devices=[...])` — see
[`docs/dev/multi_gpu_operations.md`](../../../docs/dev/multi_gpu_operations.md)
for which one to reach for.

## Run it on one node

```bash
torchrun --nnodes=1 --nproc_per_node=2 ddp_intersect_train.py
torchrun --nnodes=1 --nproc_per_node=2 ddp_accum_grids.py
```

`--nproc_per_node` is the GPU count: one rank per GPU, and each rank uses
`LOCAL_RANK` as its CUDA device. Useful flags:

```bash
torchrun --nnodes=1 --nproc_per_node=2 ddp_intersect_train.py \
    --steps=64 --rays=1048576 --lr=0.02 --check-every=8
torchrun --nnodes=1 --nproc_per_node=2 ddp_accum_grids.py \
    --samples=67108864 --resolution=32 --out=merged.pt
```

## Run it on several nodes

Nothing in the scripts changes; only the rendezvous does. On **every** node:

```bash
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --rdzv_id=rayd-ddp \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HEAD_NODE_ADDR:29500 \
    ddp_intersect_train.py
```

`RANK` becomes global and `LOCAL_RANK` stays node-local, which is exactly what
the scripts already use: the device comes from `LOCAL_RANK`, the shard and the
lane window come from `RANK` and `WORLD_SIZE`.

**Traffic is scene-sized and independent of the batch.** A step of
`ddp_intersect_train.py` sends one `[V, 3]` float32 gradient (7.5 kB for its
default 625-vertex mesh) and one scalar loss, whether the global batch is 2^18
rays or 2^30. `ddp_accum_grids.py` sends 14 grids of `resolution^2` elements,
whether it drew a million samples or a billion. Nothing per-ray and nothing
per-sample crosses a rank boundary, because each rank holds a full scene replica
and consumes its own share of the work locally. That is what makes the recipe
embarrassingly parallel across nodes: the interconnect never sees the axis that
grows.

## Give every process its own OptiX cache

OptiX keeps an on-disk cache of JIT-compiled modules, and by default every
process on the machine shares one cache database. Several processes compiling
into it concurrently is a known corruption hazard
(`OPTIX_ERROR_DISK_CACHE_INVALID_DATA`). Point each rank at its own path:

```bash
# in the launcher, before torchrun starts the workers
export RAYD_EXAMPLE_OPTIX_CACHE_ROOT=/var/tmp/optix-cache
```

Both scripts read that root and set `OPTIX_CACHE_PATH=$ROOT/rank-$RANK` for
themselves before their first RayD query, which is when the OptiX context (and
therefore the cache) is opened. In your own code the same job is usually done by
the launcher:

```bash
export OPTIX_CACHE_PATH=/var/tmp/optix-cache/rank-$RANK
```

Use a real per-rank path on local disk; a shared network path re-introduces the
contention. `OPTIX_CACHE_MAXSIZE=0` removes the hazard by disabling the cache
entirely, at the cost of a full cold JIT in every process.

## The Dr.Jit route: one visible GPU per rank

`rayd.drjit` is single-device per process, so a rank must see exactly one GPU:

```python
# before importing rayd.drjit, and before anything else initializes CUDA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
```

or, equivalently, from a launcher wrapper:

```bash
#!/usr/bin/env bash
# torchrun --nnodes=1 --nproc_per_node=2 ./pin_gpu.sh my_drjit_job.py
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
export OPTIX_CACHE_PATH=/var/tmp/optix-cache/rank-$RANK
exec python "$@"
```

Each rank then sees its GPU as device `0`, and everything else in these scripts
carries over minus the Torch specifics:

- **Ray sharding** (`ddp_intersect_train.py`) transfers directly. Replace the
  `Scene`/`Ray`/`intersect` calls with their `rayd.drjit` equivalents and keep
  the shard arithmetic. The gradient reduction stays a `torch.distributed`
  all-reduce over a plain tensor view of the Dr.Jit gradient buffer, or any
  other collective library you already use — RayD has no opinion about it.
- **Lane-window sharding** (`ddp_accum_grids.py`) does *not* transfer as-is:
  `lane_offset` / `lane_count` are Torch-backend parameters, and `DfrOptions`
  has no equivalent. Split a Dr.Jit accumulation by giving each rank the full
  `direct_samples` count with a **decorrelated seed** and averaging the merged
  grids (`all_reduce(SUM)` then divide by `WORLD_SIZE`, since the accumulation
  normalizes by its sample count). That is a different estimator from the
  Torch recipe — `world_size` independent runs averaged, rather than one
  launch's sample space partitioned — so it is unbiased and `world_size` times
  the samples, but it does not reproduce the single-launch grid.

## Failure behavior

Both scripts are written not to hang when a peer dies:

- the process group is created with an explicit `timeout` (`--timeout`, default
  600 s), so a collective waiting on a rank that will never arrive fails instead
  of blocking forever;
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` is set before the group is created, so a
  NCCL error surfaces as an exception rather than a stalled stream;
- `destroy_process_group()` runs in a `finally`, so a rank that raises tears its
  own group down instead of leaving its peers waiting;
- a closing `barrier()` means "every rank finished" is a fact, not an inference
  from rank 0's output.

`torchrun` does the rest: when one worker exits non-zero or is killed, it
terminates the surviving workers on the node and reports the root cause.
