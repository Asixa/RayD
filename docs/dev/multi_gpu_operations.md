# Multi-Device Operation (Torch Backend)

Date: 2026-07-27

This note is the operational contract for running RayD on more than one GPU
today. It corresponds to Phase 1 of
[`multi_gpu_plan.md`](multi_gpu_plan.md): per-device correctness plus manual
orchestration. Automatic batch sharding, replicated scenes, and
`Scene(devices=[...])` are Phase 2 and do not exist yet; everything below is
caller-driven.

## 1. The per-device contract

- **One `Scene` per device.** A scene owns device-resident acceleration
  structures (GAS/IAS, edge BVH, compact BVH, SoA buffers). It is not
  shareable across devices, and there is no implicit migration. To use two
  GPUs, build the same mesh twice, once per device.
- **Every tensor in one query lives on the scene's device.** This has always
  been the rule; it is now checked at every operation entry, so a mismatch
  raises a loud error instead of producing undefined behavior. Mesh updates
  (`update_mesh_vertices`, edge masks) follow the same rule.
- **Operations are ambient-device independent.** Each entry point takes a
  device guard derived from the scene (or from the validated input tensors),
  so a scene on `cuda:1` answers queries correctly while `cuda:0` is the
  current device, and the ambient device is unchanged on return. Wrapping
  calls in `torch.cuda.device(...)` is allowed but not required. Scene
  construction is guarded the same way; the tested convention is still to
  build a scene with its own device current, which a per-device worker thread
  does naturally.
- **Work is launched on the current Torch CUDA stream of the scene's device.**
  Callers that want overlap set their own stream per thread; public operations
  follow it rather than a stream of their own.
- **Outputs and gradients stay on the scene's device.** Cross-device
  concatenation or reduction of results is the caller's job in this phase.

Coverage for these properties lives in
[`backends/torch/tests/torch_backend/test_multi_device_smoke.py`](../../backends/torch/tests/torch_backend/test_multi_device_smoke.py)
(skipped when fewer than two CUDA devices are visible).

## 2. Driving several devices from one process

One host thread per device is the intended in-process shape, and the per-device
results it produces are correct — but read
[the known issue below](#known-issue-concurrent-driving-can-hang) before relying
on it unattended. The typical shape is:

```python
import threading
import torch
import rayd.torch as rt

results = {}

def worker(index, vertices, faces, ray_o, ray_d):
    device = torch.device("cuda", index)
    torch.cuda.set_device(index)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices.to(device), faces.to(device)))
    scene.build()
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        ray = rt.Ray(ray_o.to(device), ray_d.to(device))
        hit = scene.intersect(ray)
        torch.cuda.current_stream(device).synchronize()
    results[index] = hit

threads = [
    threading.Thread(target=worker, args=(k, vertices, faces, ray_o, ray_d))
    for k in range(torch.cuda.device_count())
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

Per-device results are bitwise equal to the same work run alone on that
device; there is no cross-device state.

### Known issue: concurrent driving can hang

The bits are right; the concurrency itself is not yet a guarantee. On this
repository's verification machine (2x RTX A6000) the pattern above
intermittently deadlocks in the native layer instead of finishing:

| Shape | Hangs |
| --- | ---: |
| Snippet above, cold module JIT (`OPTIX_CACHE_MAXSIZE=0`) | 4 / 30 |
| Snippet above, warm OptiX disk cache | 0 / 40 |
| Same work serially, one device after another, cold JIT | 0 / 23 |
| Threaded variant where one worker returns while the other is still in a RayD op | 6–11 per batch of 8–12 |
| Same variant, per-device scenes kept alive past the workers | 0 / 12 |

`faulthandler` dumps of the hangs show both worker threads stuck in native code
with no Python frames below `threading.run` — one inside `scene.intersect`, the
other past its last statement — while the main thread blocks in `join`. The
fault is in the native layer, not in caller code. Concurrency is necessary (the
serial control never hung) and cold OptiX work makes it far more likely, but a
warm cache did not remove it from every threaded shape. It is intermittent and
timing-sensitive: long clean batches prove nothing.

Until the native fix lands, treat one thread per device as usable but
provisional:

- For unattended or production runs, prefer one process per GPU (section 3).
  Each rank then drives its device from a single thread, which is the shape
  that has never hung here.
- If you do use threads, build the per-device scenes once and keep them alive
  for the life of the process instead of creating and destroying them inside
  short-lived workers.
- A warm OptiX disk cache helps but is not a fix: a serial in-process warm-up
  followed by threaded work still hung 3 times in 15 with the disk cache
  disabled and 2 times in 15 with it warm.
- Give long threaded jobs a watchdog. The failure mode is a hang, not an
  exception, so a timeout is the only thing that will notice it.

### OptiX creation is serialized, launches are not

RayD serializes OptiX resource creation with internal locks
([`backends/torch/src/torch_ext/scene/optix_context.cpp`](../../backends/torch/src/torch_ext/scene/optix_context.cpp),
[`backends/torch/src/torch_ext/common/optix_pipeline.cpp`](../../backends/torch/src/torch_ext/common/optix_pipeline.cpp)):

- OptiX device-context creation is serialized process-wide and memoized per
  `(device index, CUDA context)`.
- Scene/edge module and pipeline builds are serialized per device context;
  the multipath launch-pipeline cache is serialized process-wide and keyed by
  OptiX context, so each device gets its own pipeline objects.

Consequences worth planning around:

- **First touch of a device pays its module JIT cost once.** With N devices the
  process pays N pipeline builds, all on the critical path of the first real
  query. How much of that cost overlaps depends on which lock the build takes:
  device-context creation and the multipath launch-pipeline builds hold
  process-wide locks and therefore serialize across devices, while scene and
  edge module/pipeline builds hold only their device context's own lock and do
  run concurrently. Threads that arrive during another thread's build wait for
  it; they never race. Measured on the verification machine, a cold first
  `intersect` on two devices takes ~0.47 s from two threads against ~0.52 s
  device-after-device, so the overlap is real but partial.
- **After warm-up, launches on distinct devices do not serialize against each
  other.** Launch-side locking is per pipeline object, and pipeline objects
  are per device context.
- Therefore: warm every device before any timed or latency-sensitive region,
  and never measure multi-device scaling on a cold process.

### Warm-up helper

`rayd.torch` ships a private warm-up helper — `_warmup.warm_up_devices` — that
pre-builds the per-device OptiX contexts and pipelines on worker threads, so
the build cost is paid once at startup instead of inside the first real query
on each device. Call it once, with the devices you intend to use, before the
timed region.

It is private (leading underscore), not public API: the exact name and
signature may change, so check the module for the current form instead of
pinning it downstream. Code that wants a stable equivalent gets the same
effect by issuing one tiny throwaway query per device at startup, which
exercises the same creation paths.

Because it drives one worker thread per device, the helper is concurrent
cold-start work by construction, which is exactly the shape of the
[known issue above](#known-issue-concurrent-driving-can-hang) — on a machine
with no OptiX disk cache it hung 6 times in 12 runs here. Warming devices one
after another on the calling thread is slower but has not reproduced the hang.

## 3. One process per GPU

Process-per-GPU works for both backends and needs no RayD API change: each
rank builds a rank-local scene on its own device. For the Dr.Jit backend this
is the *only* supported multi-GPU route — Dr.Jit is single-device per process,
so pin each rank with `CUDA_VISIBLE_DEVICES` and let it see one GPU as device
`0`.

### Set a per-process `OPTIX_CACHE_PATH`

OptiX keeps an on-disk cache of JIT-compiled modules, and by default every
process on the machine uses the *same* per-user cache database. Several
processes compiling into that shared database concurrently is a known hazard:
it is the Blender-class failure mode that surfaces as

```text
OPTIX_ERROR_DISK_CACHE_INVALID_DATA
```

(also reported as a corrupt or locked cache when the cache lives on a network
or read-only location). RayD does not override the cache location, so give
every rank its own:

```bash
# one process per GPU; each rank gets a private OptiX cache
export CUDA_VISIBLE_DEVICES=$RANK
export OPTIX_CACHE_PATH=/var/tmp/optix-cache/rank-$RANK
python train.py
```

Notes:

- The location is read when the OptiX device context is created, i.e. on the
  first RayD query in the process. Set it before launching, not after the
  first query.
- Use a real per-rank path on local disk. A shared network path re-introduces
  the same contention.
- `OPTIX_CACHE_MAXSIZE=0` disables the disk cache entirely. It removes the
  hazard but makes every process pay a full cold JIT, so prefer per-process
  paths.

Threads inside one process do not need this: one process opens the cache
database once, and RayD's own locks keep two threads from building the same
OptiX resource twice. The in-process hazard is a different one — see
[the known issue](#known-issue-concurrent-driving-can-hang).

## 4. Not covered in this phase

- No automatic sharding: splitting a batch across devices, gathering results,
  and reducing gradients is caller code today.
- No replicated `Scene`: keeping several per-device scenes consistent under
  vertex updates is the caller's responsibility (update and `sync()` each
  replica).
- Cross-device gradient flow works if replica vertices are produced as
  `master.to(device_k)` — autograd then reduces per-replica gradients onto the
  master leaf — but RayD does not set this up for you.
- Batch-coupled semantics (dedup, `Compact` exporter counts, ADR-0033 failure
  bits, Monte-Carlo lane assignment) are defined per launch. Splitting a batch
  by hand changes them; see D5/D6 in [`multi_gpu_plan.md`](multi_gpu_plan.md)
  before doing so.
- No guarantee that concurrent in-process driving always completes: the
  intermittent native hang described in section 2 is open, so
  process-per-GPU is the route to pick when a stall would be expensive.

## 5. Checklist

- One `Scene` per device, every query tensor on that device.
- No `torch.cuda.device(...)` wrapper needed; ambient device is irrelevant.
- One host thread per device, each on its own stream — with a watchdog, and
  with long-lived scenes; the concurrent-hang issue in section 2 is open.
- Warm every device before timing (`_warmup.warm_up_devices`, or one throwaway
  query per device).
- Process-parallel: `CUDA_VISIBLE_DEVICES` per rank **and** a private
  `OPTIX_CACHE_PATH` per rank.
