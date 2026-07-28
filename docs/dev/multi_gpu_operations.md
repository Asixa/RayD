# Multi-Device Operation (Torch Backend)

Date: 2026-07-27

This note is the operational contract for running RayD on more than one GPU
today. Sections 1-4 are Phase 1 of [`multi_gpu_plan.md`](multi_gpu_plan.md):
per-device correctness plus the manual route, where the caller owns one `Scene`
per device and every split, gather and reduction between them. Section 5 is the
measured performance of the Phase 2 layer that does the same thing for you
(`Scene(devices=[...])`) and of the configurations where it does not pay --
read it before assuming a second GPU is worth engaging.

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

One host thread per device is the intended in-process shape. The typical shape
is:

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

### Resolved: the concurrent-driving hang (2026-07-27)

Concurrent in-process driving used to deadlock intermittently in the native
layer. It is fixed; the shape above is supported, and nothing in caller code
has to work around it any more.

**Root cause.** The GIL is the outermost lock in this process: Torch drops it
before entering a boxed op, and every RayD op wrapper in
[`library.cpp`](../../backends/torch/src/torch_ext/library.cpp) re-acquires it
for the duration of the call, then takes RayD's own mutexes — first the scene
registry, through `get_scene()`. `destroy_scene()` broke that order. It held
the registry mutex across the whole `SceneCache` destructor, and a `SceneCache`
owns the caller's mesh tensors: releasing an `at::Tensor` that carries a Python
object re-enters Python, and Torch's `THPVariable_clear` gives the GIL up
around the release and takes it back afterwards. So a thread dropping a scene
held the registry mutex and then waited for the GIL, while any thread inside an
op held the GIL and waited for the registry mutex — a textbook ABBA deadlock,
caught in the act by `gdb` on a hung process. The fix detaches the map entry
under the mutex and destroys the scene after releasing it, so no RayD lock is
ever held across code that can acquire the GIL. Nothing about the op wrappers,
op semantics, or single-thread behavior changed.

That explains every measurement in the original report: concurrency was
necessary (a serial control never hung), scenes being created and destroyed
inside short-lived workers was necessary (keeping them alive never hung), and
cold OptiX JIT only widened the window by making the op that holds the GIL take
longer.

The history, measured on this repository's verification machine (2x RTX A6000)
before and after the fix. Every "after" batch ran against the same repro
scripts, with a 120 s watchdog per trial:

| Shape | Before | After |
| --- | ---: | ---: |
| Snippet above, cold module JIT (`OPTIX_CACHE_MAXSIZE=0`) | 4 / 30 | 0 / 42 |
| Snippet above, warm OptiX disk cache | 0 / 40 | 0 / 42 |
| Threaded variant where one worker returns while the other is still in a RayD op | 6–11 per batch of 8–12 | 0 / 42 |
| The same variant, private OptiX cache per trial | 4 / 12 | 0 / 42 |
| Serial in-process warm-up, then the threaded pattern, cold JIT | 3 / 15 | 0 / 42 |
| First touch of both devices from two threads, cold JIT | — | 0 / 42 |
| Threaded cold warm-up helper (`_warmup.warm_up_devices`) | 6 / 12 | 0 / 20 |
| Same work serially, one device after another, cold JIT (control) | 0 / 23 | — |
| Per-device scenes kept alive past the workers | 0 / 12 | — |

The regression guard is
`ConcurrentHostThreadTests.test_building_and_dropping_scenes_concurrently_completes`
in
[`backends/torch/tests/torch_backend/test_warmup.py`](../../backends/torch/tests/torch_backend/test_warmup.py):
two host threads build, query and drop scenes at once, which is the shape that
reproduced the deadlock. It needs only one CUDA device — the defect was a
host-thread defect, not a multi-device one.

One property to keep in mind when changing native code: **no RayD native lock
may be held across anything that can acquire the GIL**, tensor releases
included. The rule is written into the wrappers' header comment in
`library.cpp` and into `destroy_scene`.

Long threaded jobs still deserve a watchdog, for the same reason any long
GPU job does — but not because of this issue.

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
- **The GIL is a second serializer, above those locks.** Op wrappers hold the
  GIL for the whole native call (see the header comment in `library.cpp` for
  why), so whatever build or launch work happens inside an op does not overlap
  with another Python thread's op. Scene construction — where the OptiX
  context and the acceleration structures are built — runs GIL-free, which is
  why per-device warm-up still overlaps well in practice.
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

It drives one worker thread per device and the workers run concurrently, so
the per-device build cost overlaps instead of summing. That is the whole point
of the helper; it was serialized while the concurrent-driving deadlock was open
and is not any more.

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
OptiX resource twice.

## 4. Not covered by the manual route

These are the limits of the caller-driven shape above. `Scene(devices=[...])`
(section 5) removes the first three of them; the rest still hold under it.

- No automatic sharding: splitting a batch across devices, gathering results,
  and reducing gradients is caller code on this route.
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
- No cross-thread parallelism inside a single op: op wrappers hold the GIL for
  the native call, so per-device threads overlap on scene construction and on
  their own stream waits, not on the op bodies themselves. Removing that needs
  refcounted scene ownership and is a Phase 2 question, not a correctness one.

## 5. Multi-GPU performance

`Scene(devices=[...])` shards a batch across replicas for you. Whether that is
faster than one GPU is a property of the *workload*, not of the layer: a second
device can only win when a row's compute costs more than its bytes cost to
move. This section is the measured version of that sentence, and the benchmark
that produced it is
[`backends/torch/tests/benchmark_multi_device.py`](../../backends/torch/tests/benchmark_multi_device.py):

```bash
python -m backends.torch.tests.benchmark_multi_device            # both configurations
python -m backends.torch.tests.benchmark_multi_device --config light
python -m backends.torch.tests.benchmark_multi_device --devices 0   # single-GPU baseline
```

It runs on one GPU as well as on two: with a single device visible it prints
the single-device column and nothing else, which is the baseline the scaling
numbers below are ratios of.

### 5.1 Measured, 2026-07-27

Machine: this repository's verification host, 2x NVIDIA RTX A6000 (48 GB each),
peer access enabled, measured device-to-device copy bandwidth **49.1 GB/s in
one direction** (256 MiB copy, min of 5; 47.6-49.1 GB/s across the runs
recorded here, 49.1 in the run of record). Torch 2.13.0+cu130, conda env
`maxwell`. Batch: 4,194,304 rays; accumulation: the sample count in the table.
Timing is interleaved (single and multi-device runs alternate inside one round)
and reduced with a minimum over 7 rounds after 2 warm-up rounds.

> **Contention caveat.** This is a shared machine. Another tenant's job held
> ~5 GB and 10-40% utilization on both devices throughout these runs. The
> minimum-of-interleaved-rounds reduction is what keeps that from landing on
> one variant rather than on both, but treat every number below as ±5% and
> re-measure before making a deployment decision. Twenty full runs were taken;
> the min/median/max of every row is tabulated after the run of record, and it
> is the calibrated rows -- which are decisions rather than measurements --
> that move.

The table below is one run of the twenty (2026-07-27), printed verbatim by the
benchmark:

| Configuration | Operation | Batch | 1 GPU | 2 GPUs | speedup | dispatch | chunks | weights |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| light | intersect | 4194304 rays | 1.27 ms | 4.63 ms | 0.27x | pipelined | 5 | -- |
| light | trace_reflections | 4194304 rays | 1.14 ms | 2.26 ms | 0.50x | pipelined | 5 | -- |
| light | intersect (calibrated) | 4194304 rays | 1.22 ms | 1.22 ms | 1.00x | master | -- | 0.487, 0.000 |
| light | trace_reflections (calibrated) | 4194304 rays | 1.09 ms | 1.10 ms | 1.00x | master | -- | 1.186, 0.000 |
| light | accum_dfr_direct | 8192 samples | 0.31 ms | 1.34 ms | 0.23x | lane-sharded | 2 | -- |
| light | intersect (chunked + offload) | 4194304 rays | 1.27 ms | 4.48 ms | 0.28x | chunked | 8 | -- |
| compute | intersect | 4194304 rays | 19.09 ms | 11.83 ms | 1.61x | pipelined | 5 | -- |
| compute | trace_reflections | 4194304 rays | 53.33 ms | 28.38 ms | 1.88x | pipelined | 5 | -- |
| compute | intersect (calibrated) | 4194304 rays | 19.08 ms | 11.83 ms | 1.61x | pipelined | 5 | 1.002, 0.998 |
| compute | trace_reflections (calibrated) | 4194304 rays | 53.43 ms | 28.45 ms | 1.88x | pipelined | 5 | 1.000, 1.000 |
| compute | accum_dfr_direct | 67108864 samples | 34.76 ms | 18.83 ms | 1.85x | lane-sharded | 2 | -- |
| compute | intersect (chunked + offload) | 4194304 rays | 19.18 ms | 12.54 ms | 1.53x | chunked | 8 | -- |

with the benchmark's own account of where each chunked row sits between "no
overlap" and "perfect overlap", and of how close each calibration was to
deciding the other way:

```text
- overlap, light intersect: 5 chunks, 100.0 B per row round trip = 1.07 ms per
  chunk; no overlap would cost 4.91 ms, perfect overlap 1.70 ms, measured
  4.63 ms (residual +2.93 ms = fixed cost + unhidden copies)
- overlap, light trace_reflections: 5 chunks, 33.0 B per row round trip =
  0.35 ms per chunk; no overlap 1.98 ms, perfect overlap 0.92 ms, measured
  2.26 ms (residual +1.34 ms)
- overlap, compute intersect: 5 chunks, 100.0 B per row round trip = 1.07 ms
  per chunk; no overlap 13.82 ms, perfect overlap 10.61 ms, measured 11.83 ms
  (residual +1.22 ms)
- overlap, compute trace_reflections: 5 chunks, 60.0 B per row round trip =
  0.64 ms per chunk; no overlap 29.23 ms, perfect overlap 27.30 ms, measured
  28.38 ms (residual +1.07 ms)
- overlap, compute intersect (chunked + offload): 8 chunks, 100.0 B per row
  round trip = 1.07 ms per chunk; no overlap 13.86 ms, perfect overlap
  10.66 ms, measured 12.54 ms (residual +1.88 ms)
- calibration, light intersect: chose the master alone; best sharded rung
  1.08 ms vs master-only 0.43 ms (-154.2% for sharding), re-timed at 1.00x
- calibration, light trace_reflections: chose the master alone; best sharded
  rung 1.37 ms vs master-only 1.29 ms (-6.0% for sharding), re-timed at 1.00x
- calibration, compute intersect: sharded with a measured margin; best sharded
  rung 9.54 ms vs master-only 16.99 ms (+43.9%), re-timed at 1.61x
- calibration, compute trace_reflections: sharded with a measured margin; best
  sharded rung 28.42 ms vs master-only 53.14 ms (+46.5%), re-timed at 1.88x
```

The overlap lines are arithmetic over the chunk plan and this run's measured
link speed, not instrumented timestamps (the executor does not expose
per-chunk timings), and their per-row byte counts are the same ones 5.2
derives by hand: 100 B for a full `Intersection`, 33 B and 60 B for 1- and
4-bounce reflections. Read them as bounds: the compute rows land 1.1-1.2 ms
above perfect overlap, the light rows land near their *no-overlap* bound
because there is no compute to hide a copy behind.

**What is stable and what is not.** The whole benchmark takes about 11 s, so
it was run 20 times back to back. Min / median / max of every row over those
20 runs:

| Row | min | median | max | comment |
| --- | ---: | ---: | ---: | --- |
| light `intersect` | 0.26x | 0.27x | 0.28x | stable |
| light `trace_reflections` | 0.48x | 0.49x | 0.50x | stable |
| light `intersect` (calibrated) | 0.69x | 1.00x | 1.01x | master-only chosen 20/20; the 0.69x is one contended round, not a split |
| light `trace_reflections` (calibrated) | **0.38x** | 1.00x | 1.00x | master-only chosen 17/20; the other 3 kept a 1/4 remote share |
| light `accum_dfr_direct` (8k samples) | 0.23x | 0.23x | 0.24x | stable; far below the accumulation crossover |
| light `intersect` (chunked + offload) | 0.18x | 0.28x | 0.28x | stable apart from one contended round |
| compute `intersect` | 1.60x | 1.62x | 1.63x | stable |
| compute `trace_reflections` | 1.85x | 1.87x | 1.90x | stable |
| compute `intersect` (calibrated) | 1.42x | 1.61x | 1.70x | sharded 20/20; two runs weighted a busy device down |
| compute `trace_reflections` (calibrated) | 1.47x | 1.88x | 1.89x | sharded 20/20; one run demoted the remote share to 1/2 |
| compute `accum_dfr_direct` (67M samples) | 1.81x | 1.85x | 1.89x | stable |
| compute `intersect` (chunked + offload) | 1.44x | 1.65x | 1.66x | the hook runs on the master's stream, between chunks |

The uncalibrated `per_ray` rows reproduce to ±0.02x of their median. The
calibrated rows do not, and the two ways they fail are worth separating:

- **A contended throughput stage weights a device wrong.** The 1.42x
  `compute intersect` run measured `cuda:1` at 30.1 ms against `cuda:0`'s
  17.0 ms on identical hardware, weighted it 0.72, and ran a split that no
  longer matched the devices; the 1.47x `compute trace_reflections` run
  demoted the remote share to 1/2 the same way. The dispatch was fine in both;
  the measurement it was configured from was not. Both stayed *faster* than one
  GPU, because the operation is compute-bound by a wide margin.
- **A near-crossover ladder keeps a split that loses.** Light 1-bounce
  `trace_reflections` is the case: its sharded and master-only rungs are within
  1-6% of each other, so the 3% tie-break decides. In 3 of these 20 runs it
  kept a 1/4 remote share and ran at 0.86x, 0.85x and 0.38x; in 5 of 6
  back-to-back repeats measured separately it did the same, at 0.79-0.89x
  (5.4). Here the loss is unbounded by anything the calibrator knows, because
  the operation has no compute margin to fall back on.

Calibrate on a quiet machine, pin the weights you measured, and pin
`[1.0, 0.0]` outright where the benchmark prints `NEAR-CROSSOVER`.

`light` is the 192-vertex grid of `benchmark_torch_native.py` (72,962
triangles) with one bounce and a small sample count. `compute` is a
2.1M-triangle *cloud* with incoherent rays and four bounces, the configuration
the pipelined dispatch was validated on. Triangle count is not what separates
them: a 2.1M-triangle plane grid answers a ray in 0.6 ns, the same triangles
scattered through a cube take 4.5 ns, because a ray has to descend a deep
overlapping BVH instead of hitting the first leaf it touches.

Every sharded `per_ray` result in that table was **bitwise identical** to the
single-device result (the benchmark checks it and reports the agreement
fraction, which was 1.0 everywhere). The merged accumulation grids matched the
single-launch grids to a relative deviation of 6.5e-08 (`light`) and 2.9e-07
(`compute`); the guarantee is only equality up to float32 summation order, so a
different split or chunk count differs in the last ULPs, as those two numbers
are.

### 5.2 Why: the transfer-bound / compute-bound crossover

A sharded row travels twice: its inputs go out, its outputs come back. Both
copies run over the same interconnect the benchmark measured at 49.1 GB/s in
one direction, so a remote row's transfer cost is arithmetic:

| Operation | bytes out (input) | bytes back (output) | total | at 49.1 GB/s | compute, `light` | compute, `compute` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `intersect` (full `Intersection`) | 24 B | 76 B | 100 B | **2.04 ns/ray** | 0.31 ns/ray | 4.55 ns/ray |
| `trace_reflections`, 1 bounce | 24 B | 9 B | 33 B | **0.67 ns/ray** | 0.27 ns/ray | -- |
| `trace_reflections`, 4 bounces | 24 B | 36 B | 60 B | **1.22 ns/ray** | -- | 12.71 ns/ray |

The output sizes are the executor's own per-row schema costs, which the
benchmark cross-checks against what the first retired chunk actually produced
(`measured_row_bytes`: 76, 9 and 36 respectively).

The crossover is the direct comparison of the last three columns. Send half the
batch to a second device and that half costs `bytes/bandwidth` per row to move
and saves `compute` per row; the pipelined executor overlaps the two, so the
ideal is

```text
T(2 GPUs) ~= max( T(1 GPU) / 2, remote_rows * bytes_per_row / bandwidth ) + fixed
```

and the whole question is which term of the `max` wins:

| Configuration / operation | `T/2` | transfer of the remote half | predicted | measured | fixed cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| light `intersect` | 0.64 ms | 2.10M x 100 B = 4.27 ms | 4.27 ms | 4.63 ms | 0.4 ms |
| light `trace_reflections` (1) | 0.57 ms | 2.10M x 33 B = 1.41 ms | 1.41 ms | 2.26 ms | 0.9 ms |
| compute `intersect` | 9.55 ms | 2.10M x 100 B = 4.27 ms | 9.55 ms | 11.83 ms | 2.3 ms |
| compute `trace_reflections` (4) | 26.67 ms | 2.10M x 60 B = 2.56 ms | 26.67 ms | 28.38 ms | 1.7 ms |

The model is worth trusting to about a millisecond, and it says the useful
thing: on the `light` configuration the interconnect term is 2.5x (1-bounce
reflections) to 6.6x (`intersect`) the compute term, so the second device
*cannot* help however well the copies overlap --
0.27x is not a defect in the pipeline, it is 210 MB crossing a 49 GB/s link
while one GPU was already finished in 1.27 ms. On the `compute` configuration
the compute term dominates by 2.2x (`intersect`) and 10x (`trace_reflections`),
and the measured speedups (1.61x, 1.88x) are exactly `2 / (1 + fixed/(T/2))`.

The residual fixed cost is the pipeline's first scatter, its last gather, the
master's copy of its own shard into the output, and roughly 0.3 ms of host time
per chunk (one native launch plus one copy per output field). It is why more
chunks are not always better and why `pipeline_chunks_per_device` defaults to
4.

The benchmark computes this model for you, per chunked row, from the chunk plan
and the link speed it measured that run -- those are the `overlap, ...` lines
in 5.1. Use them on your own configuration rather than re-deriving the table
above by hand.

### 5.3 When multi-GPU pays off

- **`grid_reduce` (accumulation): almost always, once the launch is real
  work.** `accum_dfr_direct` shards the Monte-Carlo *lane* space and moves no
  per-row data at all -- each device returns one grid, and the grids are summed
  on the master. There is no bandwidth ceiling, only a fixed 1.1-1.5 ms of
  scatter, launch and merge, so the speedup is `T / (T/2 + ~1.4 ms)` and
  approaches 2x as the sample count grows. Measured on this machine with
  `--accum-only --accum-samples N`:

  | samples | 1 GPU | 2 GPUs | speedup |
  | ---: | ---: | ---: | ---: |
  | 65,536 | 0.34 ms | 1.45 ms | 0.23x |
  | 1,048,576 | 0.78 ms | 1.48 ms | 0.53x |
  | 4,194,304 | 2.37 ms | 2.20 ms | 1.08x |
  | 16,777,216 | 8.84 ms | 5.52 ms | 1.60x |
  | 33,554,432 | 17.44 ms | 9.98 ms | 1.75x |
  | 67,108,864 | 34.83 ms | 18.98 ms | 1.84x |

  The crossover is a single-device launch of roughly 3 ms (a few million
  samples). Above it, sharding wins and keeps winning; there is no size at
  which it turns back into a loss, which is the difference from `per_ray`.

- **`per_ray` (intersect, reflections, visibility, edges): only with enough
  compute per ray.** Compare the operation's per-row bytes at your
  interconnect's bandwidth (the table in 5.2) with its single-device time
  divided by the batch. A scene whose rays are answered in well under 2 ns --
  a plane, a small mesh, a shallow BVH, coherent rays -- is transfer-bound at
  any batch size, and the right answer for it is one GPU. Deep BVHs,
  incoherent rays, several bounces, and heavy per-hit work move the other way.
  Reducing the output helps directly: `flags=0` brings a row back in 4 bytes
  instead of 76, which takes `intersect` from 100 B to 28 B per sharded row
  and its crossover from 2.04 ns/ray to 0.57 ns/ray. (The flag is spelled `0`
  or `getattr(rt.RayFlags, "None")` -- `RayFlags.None` is a Python syntax
  error, because `None` is a keyword; the member is reachable as
  `rt.RayFlags["None"]`. Verified: `flags=0` reports a 4-byte measured row.)

- **Either way, the work has to be big enough to amortize 1-2 ms of
  orchestration.** For `per_ray` that is a hard floor the layer enforces (5.4);
  for accumulation there is no floor, and a small launch simply loses.

### 5.4 The small-batch fallback

A multi-device `per_ray` batch with fewer than `min_rays_per_device *
len(devices)` rows (default 262,144 per device, so 524,288 rows on two) does
not shard at all. It runs on the master replica, with the caller's own tensors,
through the same code a single-device `Scene` runs -- so the result is bitwise
the single-device result and the layer costs one Python comparison. The floor
is measured, not guessed: the pipelined dispatch spends ~3 ms of host time
before any device work, so a batch whose single-device time is below that
cannot win however the copies are arranged.

Two consequences worth knowing:

- An explicit chunking knob (`chunk_rays`, `tape_memory_budget_bytes`,
  `offload`) **outranks** the floor. Those are memory contracts, and a memory
  bound has to hold at every batch size; the floor is only a throughput
  heuristic.
- `weights=[1.0, 0.0]` -- which is what `calibrate_devices()` answers with when
  it *measures* an operation as transfer-bound -- is also dispatched as the
  single-device call it is, at any batch size (unless a chunking knob asked for
  chunked execution, which again is a memory contract). That is what puts the
  calibrated `light` rows at ~1.00x rather than at 0.27x in the runs above.

  The strength of that second sentence is worth being exact about, because the
  ladder is a tie-break, not a proof. It keeps the **largest** remote share
  within `_REFINE_TOLERANCE` (3%) of the fastest rung, so the honest statement
  is:

  > Calibration will not knowingly keep a split that it measured as more than
  > the refinement tolerance (3%) slower than the master alone -- and on an
  > operation whose sharded and master-only rungs are that close, run-time
  > noise puts you on either side of one GPU.

  Not: "calibration cannot leave you slower than one GPU". Measured here, six
  back-to-back calibrations of the near-crossover case (light
  `trace_reflections`, 1 bounce: 0.67 ns/ray of transfer against 0.27 ns/ray of
  compute, 4.19M rays):

  | calibration | chosen weights | ladder: 1/4 share vs master-only | re-timed |
  | ---: | --- | ---: | ---: |
  | 1 | `[1.00, 0.25]` | 1.311 vs 1.281 ms (-2.3%) | **0.88x** |
  | 2 | `[0.92, 0.00]` | 1.420 vs 1.293 ms (-9.8%) | 1.00x |
  | 3 | `[1.00, 0.25]` | 1.302 vs 1.284 ms (-1.4%) | **0.89x** |
  | 4 | `[1.00, 0.25]` | 1.311 vs 1.277 ms (-2.7%) | **0.79x** |
  | 5 | `[1.00, 0.25]` | 1.315 vs 1.283 ms (-2.5%) | **0.88x** |
  | 6 | `[1.00, 0.25]` | 1.297 vs 1.277 ms (-1.6%) | **0.88x** |

  Five of six kept a quarter of the batch on the remote device -- each time
  after timing that rung as *slower* than the master alone, by less than the
  3% the ladder forgives -- and each time ran the workload 11-21% slower than
  one GPU. The same six calibrations of light `intersect`, whose rungs are far
  apart (0.42-0.63 ms master-only against 0.71-0.80 ms for the best split, i.e.
  sharding 27-71% slower every time), chose `[1.0, 0.0]` six times out of six
  and ran at 0.99-1.00x. The difference is
  the margin, not the operation class: a decision taken inside the tolerance
  band is a coin flip, and the benchmark now labels those rows
  `NEAR-CROSSOVER, decision flips between runs`.

  So: read `record.describe()`, and where the best sharded rung is within a
  few percent of the master-only rung, do not ship the ladder's answer --
  construct the scene with `MultiDeviceOptions(weights=[1.0, 0.0])` (or run a
  single-device `Scene`, which is cheaper still). The floor in the first half
  of this section, unlike calibration, *is* a guarantee: below
  `min_rays_per_device * len(devices)` rows the batch is bitwise the
  single-device call, whatever the weights say.

### 5.5 Using `calibrate_devices()`

```python
scene = rt.Scene(devices=[0, 1])
scene.add_mesh(rt.Mesh(vertices, faces))
scene.build()

record = scene.calibrate_devices(rays=4_194_304, max_bounces=4)
print(record.describe())
print(scene.device_weights)
```

It measures in two stages. The *throughput* stage runs the same probe on every
replica with resident inputs, so what is timed is the device rather than the
interconnect, and makes the weights inversely proportional to the times. The
*refinement* stage then times the real multi-device dispatch of that probe
while scaling every non-master weight through `1, 1/2, 1/4, 1/10, 0`, and keeps
the largest rung within 3% of the fastest one. The second stage is the one that
knows about the interconnect, and its last rung is the master alone. Both
stages are single measurements of a machine you are sharing, which is what the
rules below are about.

Practical rules, all learned the hard way on this machine:

- **Calibrate at the batch size you will run.** The refinement stage times a
  real dispatch, and the dispatch's fixed cost is a different fraction of a 1M-
  row batch than of a 4M-row one, so the probe size moves where the crossover
  sits. Measured on the `light` configuration against its 4.19M-row workload:

  | operation | probe | ladder: best sharded vs master-only | chose | re-timed |
  | --- | ---: | ---: | --- | ---: |
  | `intersect` | 1.05M | 0.72 vs 0.20 ms | `[1, 0]` 3/3 | 0.99-1.00x |
  | `trace_reflections` (1) | 1.05M | 0.87 vs 0.42 ms | `[1, 0]` 3/3 | 1.00x |
  | `trace_reflections` (1) | 4.19M | 1.32 vs 1.30 ms | `[1, 0.25]` 2/3 | 0.71-0.84x |

  A probe smaller than the workload spreads the pipeline's fixed cost over
  fewer rows and so makes sharding look worse than it will be; a larger one
  makes it look better. The benchmark defaults its probe to the configuration's
  own batch for that reason. But note what the third row says: probing at the
  right size does not make the answer *reliable* on a near-crossover operation
  -- it only measures the tie honestly, and the ladder then resolves the tie
  towards sharding. It is also slightly optimistic about the split it keeps:
  the rung timed at 1.32 ms during calibration ran the workload at ~1.33 and
  ~1.56 ms (0.84x and 0.71x of the ~1.11 ms one-GPU time in the same rounds). Where the two rungs are within a few percent,
  pin the weights (5.4) rather than re-deriving them.
- **Pass your own probe when the default shape is not your workload.**
  `calibrate_devices(probe=...)` is handed `(scene_like, device)` and should
  build its inputs on the device it is given and call the operation
  positionally; the default probe is `intersect` (or `trace_reflections` with
  `max_bounces`) over rays drawn in the scene's bounding box, which is an op
  *shape*, not your workload.
- Calibration only chooses weights. At fixed weights, execution is as
  reproducible as it ever was: same devices, same weights, same chunking, same
  inputs give the same results run to run.
- **Read `record.describe()` and check the margin, not just the answer.** It
  prints the per-device probe times and the full candidate ladder. Two failure
  shapes are visible there and nowhere else:
  - *A near-crossover ladder.* The best sharded rung is within a few percent
    of the master-only rung, so the 3% tie-break decides, and it decides
    towards sharding. Measured here: 5 of 6 calibrations of light 1-bounce
    `trace_reflections` kept a 1/4 remote share and ran at 0.79-0.89x (5.4).
    Pin `MultiDeviceOptions(weights=[1.0, 0.0])` instead of shipping the
    ladder's answer.
  - *A contended throughput stage.* If a neighbour is on one device while the
    first stage times it, the weights inherit the neighbour. Two of the 20
    recorded runs weighted `cuda:1` at 0.72 and 0.81 on a compute-bound
    `intersect` whose devices are identical, and ran at 1.42x and 1.54x where
    the same workload at `[1.0, 1.0]` runs at 1.61x. The tell is a throughput stage whose
    two identical devices disagree by more than a few percent.

  More `repeats` narrows both; neither disappears. Calibration is cheap: re-run
  it when the machine is quiet, log the weights you shipped with, and treat
  them as a configuration value rather than as something to re-derive at
  startup.

### 5.6 Keeping results out of the master's memory: the `offload` hook

A `per_ray` batch whose output does not fit on the master is executed as a
stream instead of a concatenation:

```python
hits = torch.zeros((), device="cuda", dtype=torch.long)

def consume(start_row, chunk):
    # `chunk` holds this chunk's rows, already on the master device.
    # Reduce it, write it out, or backpropagate it here -- do not keep it.
    hits.add_(torch.isfinite(chunk.t).sum())

scene = rt.Scene(
    devices=[0, 1],
    options=rt.MultiDeviceOptions(chunk_rays=1 << 19, offload=consume),
)
...
result = scene.intersect(ray)      # -> None; every row went through consume()
```

The hook is called once per chunk as `offload(chunk_start_row, chunk_result)`
with the chunk's fields already on the master, and the operation itself returns
`None`. Chunks arrive in the order they were issued: ascending rows per device,
interleaved across devices (which is what keeps both busy). Use
`chunk_start_row` rather than assuming the hook walks the batch front to back.
Measured above at 4,194,304 rows and 524,288
rows per chunk: the streamed run peaked at 2.10 GB on the master against
2.29 GB for the concatenated one (the 319 MB full-`Intersection` output never
exists at once), at 1.53x against the pipelined path's 1.61x in the same run
(1.48-1.66x over seven runs, the widest spread in the table) -- streaming is
cheap here but not free, and it is the row a busy master hurts most. Nor
will it stay cheap for a heavy consumer: the hook runs on the master's
stream, between chunks.

Training at extreme batch sizes pairs this with a per-chunk backward: reduce
and backpropagate inside the hook rather than holding the whole batch's graph.
That is ordinary gradient accumulation and is exact for RayD geometry
gradients, which land in `grad_vertices` by summation; only float32 summation
order differs from the unchunked backward.

## 6. Distributed: one rank per GPU, one node or many

Section 3 gives the per-process rules; this section is the worked recipe built
on them. Two runnable examples live in
[`backends/torch/examples/distributed`](../../backends/torch/examples/distributed):

- [`ddp_intersect_train.py`](../../backends/torch/examples/distributed/ddp_intersect_train.py)
  — one rank per GPU, a rank-local `Scene` built from the same mesh, a global
  ray batch sharded by rank, a differentiable `intersect` loss, and one
  `all_reduce(SUM)` of `vertices.grad` per step. The optimizer then applies the
  same update to the same replicated parameter on every rank; the script
  asserts zero cross-rank drift every `--check-every` steps and prints a hash
  of the final parameter.
- [`ddp_accum_grids.py`](../../backends/torch/examples/distributed/ddp_accum_grids.py)
  — rank-sharded Monte-Carlo accumulation. Each rank calls `accum_dfr_direct`
  with the *same* `direct_samples` and its own `lane_offset` / `lane_count`
  window, so the ranks' windows partition one global lane space (§ D5 of
  [`multi_gpu_plan.md`](multi_gpu_plan.md)); `all_reduce(SUM)` on the grids
  reproduces the single launch's grid up to summation order.
- [`README.md`](../../backends/torch/examples/distributed/README.md) — the
  launcher commands, the `OPTIX_CACHE_PATH` requirement, the Dr.Jit variant,
  and the failure-behavior notes.

Both are exercised by
[`backends/torch/tests/torch_backend/test_distributed_recipe.py`](../../backends/torch/tests/torch_backend/test_distributed_recipe.py),
which launches them under `torchrun --nproc_per_node=2` in a subprocess and
checks that the ranks' final parameters are bitwise equal and that the merged
grid matches a single-process, single-device launch of the full sample count.

### Which route for which problem

| | `Scene(devices=[...])` (§5) | process per GPU (this section) |
|---|---|---|
| Backends | Torch only | Torch **and** Dr.Jit |
| Scope | one node | one node or a cluster |
| What crosses the link | sharded rows, both ways | grids and gradients only |
| Who writes the split | RayD | you |

The in-process layer is the better answer whenever it applies, because it
shards work RayD understands. The process-per-GPU recipe is what you use when
the layer does not apply — the Dr.Jit backend, or more GPUs than one node has —
and it scales further, because the only thing on the wire is scene-sized.

### Multi-node invocation

Nothing in the scripts changes across nodes; only the rendezvous does. On every
node:

```bash
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --rdzv_id=rayd-ddp \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HEAD_NODE_ADDR:29500 \
    ddp_intersect_train.py
```

`RANK` becomes global and `LOCAL_RANK` stays node-local, which is what the
examples already consume: the CUDA device comes from `LOCAL_RANK`, the ray shard
and lane window from `RANK` / `WORLD_SIZE`.

**Honest scope of what has been run.** The verification machine for this
repository is a single node with two RTX A6000s. The single-node,
two-rank form is executed by the test above on every run. The multi-node form
is **documented but not executed here** — there is no second node to run it on.
What can be said without a cluster is structural and is worth stating plainly:
the examples contain no node-count-dependent code path, they read only `RANK`,
`LOCAL_RANK` and `WORLD_SIZE`, and per-step traffic is one `[V, 3]` gradient (or
a fixed set of grids), independent of the ray or sample count. The claim that
needs a cluster to confirm is a throughput claim, and none is made here.

### NCCL environment notes

Generic, and deliberately not tuned for a fabric this repository cannot see:

- `NCCL_DEBUG=INFO` prints the topology, the transport NCCL picked per pair, and
  the ring/tree it built. It is the first thing to set when a run is slower than
  the interconnect should allow or when initialization hangs. `NCCL_DEBUG=WARN`
  is a reasonable thing to leave on in production.
- `NCCL_SOCKET_IFNAME` selects the interface used for bootstrap and for
  ethernet transport. Set it when a node has several NICs (or docker/virtual
  interfaces) and NCCL picks one that cannot reach the peers, which shows up as
  a rendezvous that connects but a first collective that never completes.
- `NCCL_IB_DISABLE=1` forces the ethernet path. Use it to *diagnose* an
  InfiniBand configuration problem, not as a fix — it usually costs bandwidth.
- `NCCL_IB_HCA` restricts which HCAs are used on nodes with more adapters than
  the job should touch.
- `NCCL_P2P_DISABLE=1` turns off intra-node peer-to-peer. Same role: a
  diagnostic that separates "the fabric is wrong" from "the topology is wrong".
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (set by both examples before the process
  group is created) turns a NCCL failure into an exception instead of a stalled
  stream. Pair it with an explicit `timeout=` on `init_process_group` so a rank
  waiting on a dead peer fails rather than hangs.

These interact with RayD in exactly one place: none of them. RayD launches no
collectives and holds no communicator — the reductions are the caller's, on the
caller's tensors. What RayD does contribute to a distributed run is the
per-process OptiX cache requirement of §3, which is a filesystem concern rather
than a network one and is easy to forget when a job scales from one node to
many: `$ROOT/rank-$RANK` has to be per-rank *and* on local disk.

## 7. Checklist

- One `Scene` per device, every query tensor on that device.
- No `torch.cuda.device(...)` wrapper needed; ambient device is irrelevant.
- One host thread per device, each on its own stream.
- Warm every device before timing (`_warmup.warm_up_devices`, or one throwaway
  query per device).
- Process-parallel: `CUDA_VISIBLE_DEVICES` per rank **and** a private
  `OPTIX_CACHE_PATH` per rank.
- Distributed: shard the batch (or the Monte-Carlo lane space) by `RANK`,
  reduce only grids and gradients, give the process group an explicit
  `timeout=`, and `destroy_process_group()` in a `finally` so a dead peer fails
  the run instead of hanging it (§6).
