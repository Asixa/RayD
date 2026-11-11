# RayD

[![PyPI](https://img.shields.io/pypi/v/rayd)](https://pypi.org/project/rayd/)
[![Downloads](https://img.shields.io/pypi/dm/rayd)](https://pypi.org/project/rayd/)
![Code Size](https://img.shields.io/github/languages/code-size/Asixa/RayD)
![Total Lines](https://tokei.tvj.one/b1/github/Asixa/RayD?style=flat)
[![License](https://img.shields.io/github/license/Asixa/RayD)](LICENSE)

RayD is a CUDA/OptiX library for differentiable ray geometry, edge queries,
visibility, and RF-style multipath primitives. RayD provides two independent,
backend-native Python APIs:

```python
import rayd.drjit as rd
import rayd.torch as rt
```

RayD is not a full renderer. It exposes low-level geometry and wave-propagation
primitives for custom renderers, RF simulators, acoustics, sensing, visibility
analysis, and inverse-design systems without imposing a material-light-
integrator framework.

## Installation

Install the backend you use. For the Torch backend, install the CUDA 12.8 build
of PyTorch first, from PyTorch's own index: PyPI's default `torch` wheel is
CPU-only on Windows, and `rayd-torch` cannot load against a CPU-only or a
different Torch release.

```bash
# Torch backend
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install rayd-torch

# Dr.Jit backend (installs drjit==1.3.1)
python -m pip install rayd-drjit
```

Then check the installation, the driver, and OptiX on this machine:

```bash
python -m rayd.torch.doctor
python -m rayd.drjit.doctor
```

`pip install rayd` installs the meta-distribution, which depends on both
backends at exactly its own version: it pulls in `rayd-drjit` with its pinned
`drjit==1.3.1` and `rayd-torch` with `torch>=2.10,<2.11`. Use it only when you
want both runtimes in one environment, and install the CUDA Torch build first
as above so pip keeps it instead of resolving `torch` from PyPI. The exact
`drjit` pin can conflict with an environment that already holds another Dr.Jit
release, such as one installed with Mitsuba; install `rayd-torch` alone there.

All three distributions share version `0.9.0`. The `rayd` meta-distribution
pins `rayd-drjit` and `rayd-torch` to exactly the same version.

Release artifacts cover CPython 3.10 through 3.14 on Windows x86-64 and
`manylinux_2_28_x86_64`. The native backend wheels are CPython-specific while
the Torch backend still contains the transitional `_C` extension. The
`_stable_ops` library inside `rayd-torch` is untagged and uses the LibTorch 2.10
Stable ABI boundary. The complete `rayd-torch` wheel requires PyTorch 2.10
(`torch>=2.10,<2.11`) while that separately audited Stable ABI library is
validated through PyTorch 2.13. The `rayd` meta-distribution is a universal
pure-Python wheel and is the only distribution that also publishes an sdist.

> [!IMPORTANT]
> Since 0.6.0, RayD uses explicit backend namespaces. The parent `rayd` namespace does
> not select or re-export a default backend. Replace legacy `import rayd as rd`
> with `import rayd.drjit as rd` or `import rayd.torch as rt`.

For downstream migration details, see
[`docs/downstream-migration.md`](docs/downstream-migration.md).

## Scope

RayD focuses on geometry and wave-propagation primitives:

- differentiable ray-mesh intersection
- scene-level GPU acceleration through OptiX
- nearest-edge point and ray queries
- primary-edge and secondary-edge sampling support
- segment visibility and reflection-path tracing
- equivalent-path correction (EPC) primitives
- reflection and diffraction field accumulation
- shared RF device primitives for complex media, Fresnel terms, resident
  layer stacks, and Jones field transport
- solver-neutral RF scattering table, sampling, ensemble, and phase-screen
  evaluation primitives over caller-owned resident tensors
- differentiable sphere-traced intersection of a caller-owned dense signed
  distance field (Torch backend)
- Dr.Jit and PyTorch reverse/forward automatic differentiation
- source-level integration for native downstream CMake projects

RayD intentionally does not provide:

- a high-level BSDF or material-model framework, or emitters
- rendering integrators
- scene loaders
- bitmap or image I/O
- a material-light-integrator framework
- implicit conversions between Dr.Jit and Torch objects

The exclusion of a high-level BSDF framework does not exclude low-level,
solver-neutral RF scattering primitives. RayD evaluates caller-owned resident
tensors; it does not acquire table builders, material policy, solver state, or
resource lifecycle through those operations.

## Why RayD?

Mitsuba is an excellent physically based renderer, but it can be too high-level
when the main workload is RF propagation, acoustics, sonar, visibility analysis,
or a custom wave simulator. Those applications often need direct control over
ray-scene queries, edges, reflection chains, diffraction state, and geometry
gradients instead of a complete rendering runtime.

RayD keeps that API surface focused: meshes, scenes, rays, intersections,
edges, visibility, and multipath query results.

## Backend Capabilities

| Capability | Dr.Jit | Torch |
| --- | --- | --- |
| Ray-mesh intersection | Yes | Yes |
| Point/ray nearest edge | Yes | Yes |
| Top-k nearest edges | Yes | Yes |
| Segment visibility | Yes | Yes |
| Pair/chain/edge visibility helpers | Yes | Yes |
| Reflection tracing and accumulation | Yes | Yes |
| EPC path and field queries | Yes | Yes |
| Direct and chained diffraction | Yes | Yes |
| Surfel primitives | Yes | Yes |
| SDF grid intersection | Yes | Yes |
| Unified mesh/SDF/surfel scene | Yes | Yes |
| Reverse-mode AD | Operation/variant dependent | Operation/variant dependent |
| Forward-mode AD | Operation/variant dependent | Operation/variant dependent |
| `torch.compile` integration | No | Yes |
| Replicated multi-device execution | No | Yes |

Use `backend_capabilities()` on either backend for the machine-readable
backward-compatible flat capability mapping. Its `reverse_ad` and `forward_ad`
booleans mean that the backend has AD support, not that every operation variant
and input domain supports both modes. Use
`api_manifest()["derivatives"][operation][variant]["input_domains"]` for the
resolved `supported`, `unsupported`, or `not_applicable` VJP/JVP status.
Unsupported functionality does not silently cross into the other runtime.

Each backend owns its scene objects, GPU allocations, current stream, OptiX
pipelines, acceleration structures, and AD graph. A `rayd.drjit.Scene` cannot
be passed to `rayd.torch`, and a `rayd.torch.Scene` cannot be passed to
`rayd.drjit`.

## Core API

The two backends use the same high-level vocabulary where their capabilities
overlap:

- `Mesh`: triangle geometry, transforms, UVs, and edge topology
- `Scene`: mesh container plus OptiX acceleration structures
- `MixedScene`: unified differentiable mesh, bounded-SDF, and surfel ray queries
- `Ray`: batched origins, directions, and optional `tmax`; Dr.Jit adds
  `RayAD` for its differentiable overloads, while Torch has only `Ray` and
  selects AD from `requires_grad`
- `scene.intersect(ray)`: closest differentiable ray-mesh hit
- `scene.nearest_edge(query)`: nearest-edge point or ray query
- `scene.trace_reflections(...)`: specular reflection chains
- `scene.trace_refl_epc_field(...)`: complex reflected EPC fields
- `scene.accum_dfr_direct(...)` / `scene.accum_dfr(...)`: diffraction
  accumulation
- `scene.trace_dfr_paths(...)`: diffraction path export with required
  capacity-shaped CUDA validity; single-device calls default to `Compact`,
  while replicated scenes require deterministic `SourceLane` rows
- `scene.nearest_edges(point, k)` for `k <= 16`
- `scene.visible(...)`, `visible_pair(...)`, `visible_chain(...)`, and
  `visible_edge(...)`
- `scene.set_edge_mask(mask)` / `scene.edge_mask()`

Both backends expose `scene.trace_refl_epc(...)` for EPC path geometry and
`scene.accum_refl(...)` for reflected field/power accumulation.
They also expose
`scene.add_instance(geometry_id, transform, dynamic=True)` for an independent
transform, instance id, and scene-global primitive range over a shared OptiX
GAS. This first slice retains per-instance world-space geometry/edge caches;
the CUDA trace fallback refreshes those caches instead of an IAS, and Torch
rejects every `Scene(devices=...)` orchestrator, including one-device chunking.
Torch wraps its
`torch.ops.rayd_torch.reflection_epc_paths_forward` (source positions),
`reflection_epc_ray_paths_forward` (a `Ray`), and
`reflection_accumulation_forward` dispatcher operations with backend-native
options and result records.

The Dr.Jit backend additionally exposes:

- `scene.shadow_test(ray)`
- surfel rendering primitives (intersection, LOS, reflection, and alpha
  compositing are cross-backend)

Both backends expose standalone differentiable signed-distance-field queries:

- `SdfGrid`: a caller-owned dense grid of vertex-centred world-metric
  distances placed by an oriented box. Torch takes
  `SdfGrid(values, position, rotation, scale)` with a shaped `values` tensor;
  Dr.Jit takes `SdfGrid(values, nx, ny, nz, position, rotation, scale)` with a
  flat `values` array and explicit grid dimensions
- Torch `SdfGridBatch(values, position, rotation, scale)`: two or more
  shape-compatible grids packed under one native primal query owner
- `grid.intersect(ray, options=SdfTraceOptions(...), active)`: sphere-traced
  intersection returning `t`, a validity mask `valid`, `position`, `normal`, and a `steps` diagnostic
- Torch-only `sdf_intersect(grid, origins, directions, ...)`: the functional
  form of the same query
- `grid.visible(...)` and `grid.trace_reflections(...)`: SDF-only LOS and
  specular reflection

Gradients and tangents reach the grid values, the box transform, and the rays
through the frozen-winner implicit function theorem. Torch uses the existing
CUDA dispatcher and Dr.Jit launches the shared sphere march on its current CUDA
stream. The primitive remains directly usable and can also join a `MixedScene`.

Both backends also expose standalone surfels. Surfel LOS and reflection use the
accepted analytic Gaussian hit; `composite_alpha(...)` (and Torch's
`transmittance(...)` convenience method) provides transmission. The closed
scope is intentional: SDF participates only in LOS/reflection, surfel in
LOS/reflection/transmission, and neither participates in diffraction. See
[`ADR-0037`](docs/adr/0037-differentiable-sdf-intersection.md) and
[`ADR-0042`](docs/adr/0042-cross-backend-implicit-geometry.md).

`MixedScene` owns one public `intersect`, `visible`, `trace_reflections`, and
`transmittance` surface for any simultaneous mix of meshes, oriented-bbox SDFs,
and surfels. It keeps each geometry owner's specialized GPU accelerator, merges
closest hits on the device, and differentiates through the fixed winning hit.
Mesh hits are opaque for transmission, SDFs are ignored, and surfels contribute
alpha transmission. A mesh-only `MixedScene` forwards directly to `Scene`, and
the class intentionally exposes no diffraction method.
Torch `MixedScene.add_sdf_batch()` reduces an untracked packed group's
closest-hit/reflection candidate work to one CUDA launch. Tracked inputs retain
the existing per-grid frozen-winner VJP/JVP path; LOS still applies each grid's
own bias and SDF remains outside transmission and diffraction.
See [`ADR-0043`](docs/adr/0043-unified-mixed-geometry-scene.md).

## Differentiation Contract

RayD differentiates continuous geometry and field quantities while treating
the discrete winner selected during the forward pass as fixed:

- primitive, edge, visibility, and path selection are discrete
- hit distance, position, normals, transforms, ray parameters, materials, and
  supported field inputs retain gradients
- native reflection and diffraction operators provide explicit JVP/VJP paths
- unsupported AD strategies fail explicitly instead of silently copying data
  through the other backend

For Dr.Jit, `RayAD` selects the differentiable intersection overload. Torch
selects AD from tensors with `requires_grad=True` and supports both backward
VJP and forward JVP for implemented operators.

## Recorded Backend Differences

> [!WARNING]
> Two numeric conventions differ between the backends on purpose and are
> recorded in `contracts/operations.json`:
>
> - **Intersection `ray_tmin`.** Dr.Jit `Scene.intersect` and `shadow_test`
>   start rays at `t = 1e-3`; Torch `Scene.intersect` starts them at
>   `t = 1e-6`. A hit closer than `1e-3` to the origin is found by Torch only.
> - **Reflection miss `t`.** For an invalid bounce, Torch `ReflectionChain.t`
>   reports the trace kernel's finite `tmax` (`1e8`), while Dr.Jit
>   `ReflectionChain.t` and `ReflectionBounce.t` report `+inf`. Detect a missing
>   bounce from validity (`valid` / `is_valid()`), never from `t`.

## Dr.Jit Quick Start

The following example traces one ray against a triangle and differentiates the
hit distance with respect to the mesh vertices:

```python
import drjit as dr
import rayd.drjit as rd


mesh = rd.Mesh(
    dr.cuda.Array3f(
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ),
    dr.cuda.Array3i([0], [1], [2]),
)

vertices = dr.cuda.ad.Array3f(
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
)
dr.enable_grad(vertices)
mesh.vertex_positions = vertices

scene = rd.Scene()
scene.add_mesh(mesh)
scene.build()

ray = rd.RayAD(
    dr.cuda.ad.Array3f([0.25], [0.25], [-1.0]),
    dr.cuda.ad.Array3f([0.0], [0.0], [1.0]),
)

hit = scene.intersect(ray)
dr.backward(dr.sum(hit.t))

print("t =", hit.t)
print("vertex z gradients =", dr.grad(vertices).z)
```

## Torch Quick Start

The equivalent Torch example stays entirely in Torch tensors and autograd:

```python
import torch
import rayd.torch as rt


vertices = torch.tensor(
    [[0.0, 0.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    device="cuda",
    dtype=torch.float32,
    requires_grad=True,
)
faces = torch.tensor(
    [[0, 1, 2]],
    device="cuda",
    dtype=torch.int32,
)

scene = rt.Scene()
scene.add_mesh(rt.Mesh(vertices, faces))
scene.build()

ray = rt.Ray(
    torch.tensor([[0.25, 0.25, -1.0]], device="cuda"),
    torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
)

hit = scene.intersect(ray)
hit.t.sum().backward()

print("t =", hit.t)
print("vertex z gradients =", vertices.grad[:, 2])
```

Torch vector inputs use contiguous CUDA `float32` tensors with shape `(N, 3)`;
index tensors use CUDA `int32`, and masks use `torch.bool`. CUDA operations run
on the current Torch stream.

## Scene Lifecycle and Input Checks

Both backends check their inputs up front and fail with a specific error rather
than reading out of bounds or answering for stale geometry:

- Face indices must address existing vertices: Torch `Scene.build()` rejects
  negative or out-of-range indices and meshes without vertices, and Dr.Jit
  `Mesh` checks its face indices when they are set and again at build.
- Vertices must be finite: a NaN or infinite vertex position is rejected at
  `build()` and by every vertex update, in both backends.
- Every ray, point, and other query tensor must live on the scene's CUDA device;
  a query from another device raises instead of reading foreign memory.
- A face with a repeated vertex index has no edges and contributes none. An edge
  shared by more than 256 faces, which usually means duplicated or coincident
  faces, fails the build unless edges are disabled for that mesh
  (`edges_enabled=False`).
- Torch queries refuse a scene with pending updates: after
  `update_mesh_vertices()` or a transform change, call `sync()` before querying,
  as Dr.Jit already required. A backward or forward-mode pass, a lazily read
  result, or a `torch.compile` graph recorded before a geometry-changing
  `sync()` raises instead of differentiating against the new geometry.
  Double backward (`create_graph=True` followed by another backward) raises a
  clear error.
- Torch `visible()` reports `False` for segments with a non-finite endpoint,
  `trace_reflections()` requires `max_bounces >= 1`, and
  `update_mesh_vertices()` with an unknown mesh id raises `ValueError`.
- Torch `Scene.build()` works from any thread, including one that has not used
  CUDA yet, and one scene can be queried from several threads while another
  thread updates it: each scene holds a reader/writer lock.
- Dr.Jit symbolic `trace_reflections()` accepts at most 65,536 bounces and
  evaluates every 8 bounces; `nearest_edge()` and `shadow_test()` return empty
  results for empty batches, while `intersect()` raises for one.

Nearest-edge queries default to a compact CUDA edge BVH in both backends.
Torch's `edge_bvh_backend="auto"` resolves to `cuda`, measured 20-80x faster
than the OptiX edge backend; pass `edge_bvh_backend="optix"` to select OptiX
explicitly. Torch builds the edge acceleration structures on the first edge
query after `build()` or a geometry-changing `sync()`, so scenes that never
query edges never pay for them. Dr.Jit's `"auto"` resolves to `drjit`, a native
traversal of the same compact BVH launched on Dr.Jit's stream, which was faster
than the OptiX edge backend for point, ray, and top-k queries at every measured
size; its edge BVH is built in `build()`, and `"optix"` and `"optix_drjit"`
(OptiX for rays only) remain available. Backend selectors are keyword-only
(`Scene(trace_backend=..., edge_bvh_backend=...)`) and match their values
exactly and case-sensitively; there are no aliases, and an unknown value raises
`ValueError`. The two default backends break exact
distance ties toward the lower global edge id (`contracts/operations.json`
records where the OptiX backends differ). On every edge backend,
`nearest_edge(ray)` searches a row only when its origin is finite, its
direction is finite and non-zero, and its `tmax` is positive (`+inf` or no
`tmax` is the half-ray); any other row returns the miss sentinels (distance
`inf`, ids `-1`).
A dynamic mesh's OptiX acceleration structure is refit on each `sync()` and
rebuilt every 16th commit to bound refit quality loss.

## Edge Queries

RayD provides scene-level GPU acceleration for point-to-edge and ray-to-edge
queries. Typical uses include:

- diffraction edge selection
- closest-feature geometry terms
- visibility-boundary estimators
- differentiable geometric constraints

For finite ray queries, nearest-edge distance uses segment semantics on
`[0, tmax]`. Dr.Jit's `Scene.set_edge_mask(mask)` filters the secondary-edge
BVH in scene-global edge index space without changing the underlying edge
topology or mesh offsets.

## Multipath Queries

RayD includes low-level reflection, visibility, and diffraction primitives for
custom wave simulators:

- reflection chains with image sources and scene-global primitive IDs
- batched segment visibility
- equivalent-path correction geometry and complex reflected fields
- reflection field/power accumulation onto receiver grids
- direct, Keller-cone, suffix-reflection, and order-2/3 diffraction
- coherent deterministic diffraction accumulation
- compact path export for downstream channel/CIR processing

`trace_*` APIs return geometric or field records per ray/path. `accum_*` APIs
reduce contributions into aggregate outputs such as receiver-grid cells. RayD
does not choose the source model, receiver model, material policy, objective,
or final channel representation for the caller.

The accepted source-level RF ownership boundary also places the complete
layer-stack and row-fused transmission primal/backward/JVP families in RayD.
Downstream consumers keep their material encodings, topology, estimators,
solver policy, accumulation, and result schemas. See
[`ADR-0002`](docs/adr/0002-shared-rf-transmission-ownership.md) for the precise
header, fusion, stream, failure, and cross-repository activation contracts.

The accepted diffraction boundary places the complete fixed-winner pure-wedge
field primal/backward/JVP family in RayD. Channel calls its primal, backward,
and JVP entries (checked 2026-11-09 against Channel's `dependencies/rayd.lock.json` pin `c7a9997`). Monte Carlo Sionna,
coupled RD/DD operations, and BDPT estimator policy remain in Channel, and
pure-wedge fast-math remains isolated to its family. See
[`ADR-0025`](docs/adr/0025-diffraction-family-ownership.md) for the typed API,
AD, fusion, activation, legacy-cleanup, and stop contracts.

The accepted generic-scattering boundary will place seventeen operations in
six complete table-evaluation, table-sampling, single-bounce, patch-integral,
chain-ensemble, and chain-realization families in RayD. A candidate becomes
active when Channel pins, validates, atomically switches, and deletes its local
implementation. RayD consumes caller-owned resident table and
phase-screen tensors; Channel retains their construction/lifecycle plus
topology, estimator, RNG/MIS, accumulation, and result policy. The move also
preserves family-specific AD (including loud chain-ensemble geometry-VJP
rejection and supported chain-realization geometry VJP/JVP), row fusion,
atomics, tape lifetime, and per-translation-unit compile flags. See
[`ADR-0026`](docs/adr/0026-generic-scattering-runtime-ownership.md).

The source tree contains all seventeen operations, and Channel calls all
seventeen, including the six chain operations (checked 2026-11-09 against Channel's `dependencies/rayd.lock.json` pin `c7a9997`).

Naming follows the public API standard in
[`docs/drjit/api_naming_standard.md`](docs/drjit/api_naming_standard.md):
`Dfr` denotes diffraction, `Refl` denotes reflection, `Epc` denotes
equivalent-path correction, and `AD` is reserved for automatic differentiation.

## Examples

Dr.Jit examples are kept as runnable applications under
[`examples/drjit`](examples/drjit):

- [`ray_mesh_intersection.py`](examples/drjit/basics/ray_mesh_intersection.py):
  differentiable triangle intersection
- [`nearest_edge_query.py`](examples/drjit/basics/nearest_edge_query.py):
  scene-level nearest-edge queries
- [`surfel_intersection.py`](examples/drjit/basics/surfel_intersection.py):
  differentiable surfel hits
- [`surfel_multiview_color_fit.py`](examples/drjit/basics/surfel_multiview_color_fit.py):
  multiview surfel optimization
- [`cornell_box.py`](examples/drjit/renderer/cornell_box.py):
  a compact renderer built from RayD primitives

Process-per-GPU examples, runnable under `torchrun` with one rank per GPU, are
under
[`examples/torch/distributed`](examples/torch/distributed):

- [`ddp_intersect_train.py`](examples/torch/distributed/ddp_intersect_train.py):
  rank-sharded differentiable `intersect` with an all-reduced vertex gradient
- [`ddp_accum_grids.py`](examples/torch/distributed/ddp_accum_grids.py):
  rank-sharded Monte-Carlo accumulation merged by all-reduce

The Torch test and benchmark suite also serves as executable usage coverage:

- [`test_intersect_grad.py`](tests/scene/test_intersect_grad.py):
  reverse-mode geometry gradients
- [`test_multipath.py`](tests/native/test_multipath.py):
  reflection, EPC, visibility, and diffraction APIs
- [`benchmark_rayd_backends.py`](benchmarks/torch/benchmark_rayd_backends.py):
  same-process Torch/Dr.Jit comparison

## Performance

The historical RayD-versus-Mitsuba benchmark was measured on an NVIDIA RTX
5080 and AMD Ryzen 7 9800X3D using a `192 x 192` mesh and `384 x 384` rays.
RayD stayed aligned with Mitsuba while improving static forward and gradient
latency. Representative average latencies in milliseconds were:

| Workload | RayD | Mitsuba |
| --- | ---: | ---: |
| Static full intersection | 0.162 | 0.190 |
| Static reduced intersection | 0.124 | 0.224 |
| Dynamic full intersection | 0.741 | 0.740 |
| Dynamic reduced intersection | 0.689 | 0.714 |
| Static gradient | 0.411 | 0.757 |
| Dynamic gradient | 1.324 | 1.413 |

Forward mismatch counts were zero in that sweep, and the largest static
gradient discrepancy was `9.54e-7`. Current backend-to-backend benchmarks live
under [`benchmarks/torch`](benchmarks/torch) and should be rerun for the
target GPU, CUDA toolkit, and workload before making deployment decisions.

## Device and Stream Selection

The Dr.Jit backend follows Dr.Jit's current-thread CUDA device:

```python
import rayd.drjit as rd

rd.set_device(0)
```

Existing scenes and OptiX resources should not be reused across device switches
in the same process. Multi-GPU use of this backend is one process per GPU, with
each rank pinned through `CUDA_VISIBLE_DEVICES`.

The Torch backend follows the device of its CUDA tensors and launches work on
the current Torch CUDA stream. Keep every tensor participating in one query on
the same device. Operations are independent of the ambient CUDA device: a scene
built on `cuda:1` answers queries correctly while `cuda:0` is current, and the
ambient device is unchanged on return.

The Torch backend runs a batch across several GPUs transparently:
`Scene(devices=[0, 1])` builds one full scene replica per device, shards the
batch across them, and returns results and gradients on the first device.
Sharded per-ray results are bitwise the single-device results; merged
accumulation grids match the single launch up to float32 summation order. The
regime, the merge semantics, the Monte-Carlo lane window and the invariance of
single-device execution are decided in
[`ADR-0038`](docs/adr/0038-replicated-multi-device-execution.md).

Multi-device execution also has a manual route: one `Scene` per device, driven
either from one process per GPU or from one host thread per device. The
operational contract, the per-device OptiX warm-up cost, the per-process
`OPTIX_CACHE_PATH` requirement for process-parallel launches, and the
GIL/native-lock ordering rule that concurrent in-process driving depends on
are documented in
[`docs/dev/multi_gpu_operations.md`](docs/dev/multi_gpu_operations.md). The
process-per-GPU recipes -- the only Dr.Jit multi-GPU route and the multi-node
route for both backends -- are runnable under
[`examples/torch/distributed`](examples/torch/distributed).

Whether a second GPU is worth engaging is a property of the workload: sharded
rays travel twice, so multi-GPU pays off for compute-heavy per-ray work and
large accumulations and loses for cheap queries with wide results. The measured
scaling on 2x RTX A6000, the transfer-bound/compute-bound crossover arithmetic,
and the benchmark that reproduces both
([`benchmarks/torch/benchmark_multi_device.py`](benchmarks/torch/benchmark_multi_device.py))
are in the multi-GPU performance section of the same note.

## Troubleshooting

Start with the doctor of the backend you use. It prints the OS, Python, driver,
each GPU with its compute capability and free memory, the installed build's
GPU targets and the framework version it was compiled against, the OptiX
library and the OptiX ABI the build needs versus what the driver offers, and
the trace and edge backends `"auto"` resolves to, with one verdict and fix per
check. It exits non-zero when a check fails, so it also works as a CI or
container health check.

```bash
python -m rayd.torch.doctor
python -m rayd.drjit.doctor
```

- **`rayd-torch` does not load.** The native libraries are built against one
  CUDA Torch release. With a CPU-only or a different Torch, `import rayd.torch`
  warns with the installed Torch, the Torch the wheel was built for, the loader
  error, and the fix, and the first native call raises the same explanation.
  The fix is the CUDA build from PyTorch's index:
  `python -m pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128`.
  The import itself still succeeds, so the doctor and `api_manifest()` work.
- **Slower than expected / `OptiX is unavailable`.** Without the driver's OptiX
  library (`libnvoptix.so.1`, `nvoptix.dll`) or with a driver older than the
  build's OptiX ABI, `trace_backend="auto"` falls back to the CUDA trace
  backend and prints one warning per process with the reason, in both
  backends. An explicit
  `trace_backend="optix"` raises the same reason. Set `RAYD_DISABLE_OPTIX=1` to
  choose the CUDA backend deliberately.
- **Docker.** The NVIDIA Container Toolkit mounts the OptiX library only with
  the `graphics` driver capability: run the container with
  `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` (or
  `all`). The default `compute,utility` silently loses OptiX.
- **WSL2.** Install the NVIDIA Windows driver (R555 or newer) and no Linux
  driver inside WSL; the Windows driver provides the GPU to WSL. Whether its
  OptiX library is visible inside WSL depends on the driver release, so run the
  doctor: without it RayD runs on the CUDA trace backend. RayD has not been
  tested under WSL2.
- **Headless servers** need no display or X server; the driver and a visible
  GPU are enough.
- **Driver floor.** The wheels need R555 or newer (OptiX 8.1 ABI). The doctor
  reports `driver version` and `OptiX ABI` failures with the required branch.
- **Unsupported GPU.** On a GPU the installed build has no code for, the first
  native call (Torch) or the import (Dr.Jit) raises `UnsupportedDeviceError`
  naming the GPU, its compute capability, the GPUs the build covers, and the
  command that builds RayD for it. GPUs below compute capability 7.0 (Maxwell,
  Pascal) cannot run RayD at all. `RAYD_SKIP_DEVICE_CHECK=1` skips the check.
- **Editable install raises `ImportError: ... no longer match the checkout`.**
  The build copies `python/rayd/_impl/*.py`; rebuild the backend after a
  `git pull` or an edit (`scripts/build_local.sh torch` or
  `.\scripts\build_local.cmd -Backend torch`).

## Building from Source

The `rayd-torch` wheel includes a lock-verifiable source bundle for native
consumers that compile RayD in their own CMake/LibTorch graph. Its fixed passive
metadata entry is `rayd/torch/_source/rayd-source.json`; it does not load a RayD
extension or select a runtime backend. Consumers must discover it through the
active Python distribution metadata, validate commit/repository/integration
ABI, and recompute the complete source manifest before using it. Prefix scans
and unvalidated global CMake package searches are intentionally unsupported.
See
[`ADR-0034`](docs/adr/0034-validated-package-source-discovery.md).

### Building from Source / older GPUs

The prebuilt wheels contain code for NVIDIA compute capability 7.0 (Volta) and
newer discrete GPUs:

| GPU | Prebuilt wheels | From source |
| --- | --- | --- |
| Maxwell, Pascal, and older (`sm_6x` and below) | no | no: not supported, because RayD's device code uses Volta-only warp intrinsics |
| Volta (`sm_70`, Tesla V100) | yes | yes |
| Turing through Blackwell (`sm_75`-`sm_12x`) | yes | yes |
| Jetson (`sm_72`, `sm_87`, `sm_110`) | no: wheels are x86-64 only | untested |

"Yes" means the build contains code for that family, not that RayD has run on
every card in it.

**What has actually run (2026-11-01):** one RTX 4090 Laptop GPU on driver 572.16
and Windows.

| Target | Verified by |
| --- | --- |
| `sm_89` | Native run, Torch and Dr.Jit |
| `sm_80`, `sm_86` | Cubin ran on the RTX 4090, Torch only |
| `sm_70`, `sm_75` | PTX translated for the RTX 4090, Torch only; not real old-GPU machine code |
| `sm_87`, `sm_90`, `sm_100`, `sm_101`, `sm_120` | Compile-checked only |

No other physical GPU, driver, or Linux has run it. Details are in the
[verification record](docs/dev/gpu_support.md#verification-record-2026-11-01).

**Driver minimum:** everything needs R555 or newer for the OptiX 8.1 ABI the
wheels use (a source build against newer OptiX headers needs the driver that
ships that ABI). Torch builds made with CUDA 12.9 or newer need R575 or newer;
on an older driver they fail with `OptiX error in
optixModuleCreate(multipath)`, and the OptiX log names the missing intrinsic
`optix.ptx.copysign.f32`. `python -m rayd.torch.doctor` and
`python -m rayd.drjit.doctor` compare the driver with the build's OptiX ABI.

On a GPU the installed build cannot run, RayD fails at first use with an
`UnsupportedDeviceError` that names the GPU, its compute capability, the
supported range, and the command below. To run on such a GPU, to build for
only the GPU you have, or to install on a platform without wheels, build from
a Git checkout. The backend distributions publish no sdist; see
[`docs/dev/gpu_support.md`](docs/dev/gpu_support.md) for the reasons, the
detailed compatibility findings, and verification steps.

Requirements: Python 3.10-3.14, CMake 3.26+ for `rayd-torch` (3.17+ for
`rayd-drjit`), a C++17 compiler (Visual Studio 2022 Desktop C++ on Windows, GCC
11+ on Linux), CUDA Toolkit 12.x (Volta cannot be targeted by CUDA 13; a conda
CUDA toolkit works), and for `rayd-torch` the OptiX SDK 8.1+ headers via
`OPTIX_INCLUDE_DIR` or `OPTIX_PATH` (for example a checkout of
[`NVIDIA/optix-dev`](https://github.com/NVIDIA/optix-dev) at tag `v8.1.0`, which
is what release CI uses, or configure with `-DRAYD_FETCH_OPTIX_HEADERS=ON` to
let CMake download those headers). Install the backend's framework and build
tools into the target environment first:

```bash
python -m pip install cmake ninja scikit-build-core
python -m pip install "drjit==1.3.1" "nanobind==2.11.0"
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Then build each backend you need. Linux:

```bash
git clone https://github.com/Asixa/RayD
cd RayD
RAYD_CUDA_ARCHS=native python -m pip install --no-build-isolation ./drjit
RAYD_CUDA_ARCHS=native python -m pip install --no-build-isolation ./torch
```

or, as an editable install with one build directory per target list:

```bash
scripts/build_local.sh drjit --cuda-arch native
scripts/build_local.sh torch --cuda-arch native
```

Windows:

```powershell
git clone https://github.com/Asixa/RayD
cd RayD
.\scripts\build_local.cmd -Backend drjit -CudaArch native
.\scripts\build_local.cmd -Backend torch -CudaArch native
```

`RAYD_CUDA_ARCHS` (and `-CudaArch`) accepts `native` (every GPU on this
machine), one compute capability such as `70` or `7.0`, a list such as
`70,86,89`, and a `+PTX` suffix such as `8.9+PTX` for forward-compatible PTX.
It drives every CUDA target of both backends; the OptiX PTX is `compute_70` in
every build.
`RAYD_CUDA_ARCHS` cannot be combined with the release-only variables
`CMAKE_CUDA_ARCHITECTURES`, `TORCH_CUDA_ARCH_LIST`, `RAYD_CUDA_GENCODE_ARCHES`,
or `RAYD_CUDA_PTX_ARCH`; configuring with both fails and names the conflicting
variable. A default build without `RAYD_CUDA_ARCHS` targets the local GPU, so on
a machine without one it fails and asks for `RAYD_CUDA_ARCHS`.
`build_local.cmd` initializes MSVC, installs editable into the current `python`
(pass `-PythonExe <path>` otherwise), and keeps one build directory per target
list; without `-CudaArch` it targets the first local GPU. `build_local.sh` does
the same on Linux (`--python <path>`), defaults to `RAYD_CUDA_ARCHS=native`,
clears the release-only variables, and stops with instructions when the OptiX
headers or a CUDA build of Torch are missing. Both pass `--no-build-isolation`:
an isolated build would download its own pinned framework instead of using the
installed one, and for Torch that is a CPU-only wheel that cannot build the
extension. An editable install copies `python/rayd/_impl/*.py` into
site-packages when it builds, so rebuild after pulling or editing those files;
`import rayd.torch` and `import rayd.drjit` raise an `ImportError` with the
rebuild command when the copies no longer match the checkout. The
multi-architecture release matrix is reserved for release CI.

Native downstream projects can add the Torch backend with CMake and link
against `rayd_torch_native_core`. The source-level integration declarations
are provided by
[`include/rayd/integration.h`](include/rayd/integration.h).
This interface is intended for projects built in the same CMake/libtorch graph;
it is not a stable binary ABI across unrelated libtorch builds.

Shared RF device headers are source-level contracts under
`src/transmission_device.cuh` and
`include/rayd/scattering_table.cuh`. RF families enter RayD as candidates and
become active through an atomic downstream pin and switch; a candidate never
creates a second production owner or authorizes runtime fallback dispatch. The
transmission-sequence, segment-penetration, and generic scattering families
are active: Channel calls them through `rayd/integration.h` at its current
RayD pin.

## Repository Layout

- [`drjit`](drjit): thin Dr.Jit distribution/build frontend
- [`torch`](torch): thin Torch distribution/build frontend
- [`python/rayd/drjit`](python/rayd/drjit) and [`python/rayd/torch`](python/rayd/torch): public backend frontends
- [`python/rayd/_impl`](python/rayd/_impl): private manifest-owned backend implementations
- [`include/rayd`](include/rayd): flat default C++ API, flat `jit/` Dr.Jit API, and backend-neutral `detail/` headers
- [`src`](src): concept-major native implementation with adjacent unsuffixed
  Torch, `*_jit.*` Dr.Jit, and `*_shared.*` backend-neutral variants
- [`contracts`](contracts): machine-readable public API,
  operation, and path-exchange manifests
- [`benchmarks`](benchmarks): benchmark schemas and recorded
  baselines
- [`scripts`](scripts): local build helpers (`build_local.sh` on Linux, `build_local.cmd` / `.ps1` on Windows)
- [`tests/packaging`](tests/packaging): distribution, namespace, and wheel-layout
  checks
- [`docs`](docs): migration, validation, and OptiX pipeline notes
- [`CHANGELOG.md`](CHANGELOG.md): release history

## Testing

Run packaging and namespace checks from the repository root (the commands are
the same in bash and PowerShell):

```bash
python -m unittest tests.packaging.test_project_metadata -v
python -m unittest tests.test_namespace_isolation -v
```

Run the governance and packaging suite that needs no GPU and no built backend,
as the `Governance` workflow does on every push and pull request:

```bash
python -m pip install "ruff==0.14.10" "clang-format==21.1.8"
python scripts/format_code.py --check
python -m tests.support.hosted_governance
```

Run representative Dr.Jit suites:

```bash
python -m unittest tests.scene.test_geometry_jit -v
python -m unittest tests.visibility.test_visibility_topk_jit -v
python -m unittest tests.reflection.test_epc_jit -v
python -m unittest tests.reflection.test_accumulation_jit -v
python -m unittest tests.diffraction.test_accumulation_jit -v
```

Run representative Torch suites:

```bash
python -m unittest tests.scene.test_intersect_forward -v
python -m unittest tests.scene.test_intersect_grad -v
python -m unittest tests.edge.test_edge_queries -v
python -m unittest tests.native.test_multipath -v
```

Behaviour tests skip cleanly without CUDA; the full suite runs on a machine
with an NVIDIA GPU (two for the multi-device tests).

## Credits

RayD is developed with reference to:

- [psdr-jit](https://github.com/andyyankai/psdr-jit)
- [redner](https://github.com/BachiLi/redner)
- [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3)

## Citation

```bibtex
@inproceedings{chen2026rfdt,
  title     = {Physically Accurate Differentiable Inverse Rendering
               for Radio Frequency Digital Twin},
  author    = {Chen, Xingyu and Zhang, Xinyu and Zheng, Kai and
               Fang, Xinmin and Li, Tzu-Mao and Lu, Chris Xiaoxuan
               and Li, Zhengxiong},
  booktitle = {Proceedings of the 32nd Annual International Conference
               on Mobile Computing and Networking (MobiCom)},
  year      = {2026},
  doi       = {10.1145/3795866.3796686},
  publisher = {ACM},
  address   = {Austin, TX, USA},
}
```

## License

RayD is released under the BSD 3-Clause License. See [LICENSE](LICENSE).
