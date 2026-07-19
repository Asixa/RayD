# RayD

[![PyPI](https://img.shields.io/pypi/v/rayd)](https://pypi.org/project/rayd/)
[![Downloads](https://img.shields.io/pypi/dm/rayd)](https://pypi.org/project/rayd/)
![Code Size](https://img.shields.io/github/languages/code-size/Asixa/RayD)
![Total Lines](https://tokei.tvj.one/b1/github/Asixa/RayD?style=flat)
[![License](https://img.shields.io/github/license/Asixa/RayD)](LICENSE)

RayD is a CUDA/OptiX library for differentiable ray geometry, edge queries,
visibility, and RF-style multipath primitives. Version 0.6 provides two
independent, backend-native Python APIs:

```python
import rayd.drjit as rd
import rayd.torch as rt
```

RayD is not a full renderer. It exposes low-level geometry and wave-propagation
primitives for custom renderers, RF simulators, acoustics, sensing, visibility
analysis, and inverse-design systems without imposing a material-light-
integrator framework.

## Installation

Install both backends through the meta-distribution:

```bash
pip install rayd
```

Install only the backend you use when the other runtime is not needed:

```bash
pip install rayd-drjit
pip install rayd-torch
```

All three distributions share version `0.6.0`. The `rayd` meta-distribution
pins `rayd-drjit` and `rayd-torch` to exactly the same version.

Release artifacts cover CPython 3.10 through 3.14 on Windows x86-64 and
`manylinux_2_28_x86_64`. The native backend wheels are CPython-specific while
the Torch backend still contains the transitional `_C` extension. The
`_stable_ops` library inside `rayd-torch` is untagged and uses the LibTorch 2.10
Stable ABI boundary. The `rayd` meta-distribution is a universal pure-Python
wheel and is the only distribution that also publishes an sdist.

> [!IMPORTANT]
> RayD 0.6 uses explicit backend namespaces. The parent `rayd` namespace does
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
- Dr.Jit and PyTorch reverse/forward automatic differentiation
- source-level integration for native downstream CMake projects

RayD intentionally does not provide:

- BSDFs or emitters
- rendering integrators
- scene loaders
- bitmap or image I/O
- a material-light-integrator framework
- implicit conversions between Dr.Jit and Torch objects

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
| Surfel primitives | Yes | No |
| Reverse-mode AD | Yes | Yes |
| Forward-mode AD | Yes | Yes |
| `torch.compile` integration | No | Yes |

Use `backend_capabilities()` on either backend for the machine-readable
capability manifest. Unsupported functionality does not silently cross into
the other runtime.

Each backend owns its scene objects, GPU allocations, current stream, OptiX
pipelines, acceleration structures, and AD graph. A `rayd.drjit.Scene` cannot
be passed to `rayd.torch`, and a `rayd.torch.Scene` cannot be passed to
`rayd.drjit`.

## Core API

The two backends use the same high-level vocabulary where their capabilities
overlap:

- `Mesh`: triangle geometry, transforms, UVs, and edge topology
- `Scene`: mesh container plus OptiX acceleration structures
- `Ray` / `RayAD`: batched origins, directions, and optional `tmax`
- `scene.intersect(ray)`: closest differentiable ray-mesh hit
- `scene.nearest_edge(query)`: nearest-edge point or ray query
- `scene.trace_reflections(...)`: specular reflection chains
- `scene.trace_refl_epc_field(...)`: complex reflected EPC fields
- `scene.accum_dfr_direct(...)` / `scene.accum_dfr(...)`: diffraction
  accumulation
- `scene.trace_dfr_paths(...)`: compact diffraction path export
- `scene.nearest_edges(point, k)` for `k <= 16`
- `scene.visible(...)`, `visible_pair(...)`, `visible_chain(...)`, and
  `visible_edge(...)`
- `scene.set_edge_mask(mask)` / `scene.edge_mask()`

The Dr.Jit backend additionally exposes:

- `scene.shadow_test(ray)`
- `scene.trace_refl_epc(...)`: EPC path geometry
- `scene.accumulate_reflections(...)`: reflected field/power accumulation
- surfel intersection, compositing, and rendering primitives

Torch reaches reflection accumulation and EPC path geometry through the
dispatcher ops `torch.ops.rayd_torch.reflection_accumulation_forward` and
`reflection_epc_paths_forward` rather than through `Scene` methods.

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

The accepted diffraction boundary will place the complete fixed-winner
pure-wedge field primal/backward/JVP family in RayD after the dormant typed
candidate is merged, pinned, validated, and activated by Channel. Until that
switch, Channel remains the production numerical owner. Monte Carlo Sionna,
coupled RD/DD operations, and BDPT estimator policy remain in Channel, and
pure-wedge fast-math remains isolated to its family. See
[`ADR-0025`](docs/adr/0025-diffraction-family-ownership.md) for the typed API,
AD, fusion, activation, legacy-cleanup, and stop contracts.

Naming follows the public API standard in
[`backends/drjit/API_NAMING_STANDARD.md`](backends/drjit/API_NAMING_STANDARD.md):
`Dfr` denotes diffraction, `Refl` denotes reflection, `Epc` denotes
equivalent-path correction, and `AD` is reserved for automatic differentiation.

## Examples

Dr.Jit examples are kept as runnable applications under
[`backends/drjit/examples`](backends/drjit/examples):

- [`ray_mesh_intersection.py`](backends/drjit/examples/basics/ray_mesh_intersection.py):
  differentiable triangle intersection
- [`nearest_edge_query.py`](backends/drjit/examples/basics/nearest_edge_query.py):
  scene-level nearest-edge queries
- [`surfel_intersection.py`](backends/drjit/examples/basics/surfel_intersection.py):
  differentiable surfel hits
- [`surfel_multiview_color_fit.py`](backends/drjit/examples/basics/surfel_multiview_color_fit.py):
  multiview surfel optimization
- [`cornell_box.py`](backends/drjit/examples/renderer/cornell_box.py):
  a compact renderer built from RayD primitives

The Torch test and benchmark suite also serves as executable usage coverage:

- [`test_intersect_grad.py`](backends/torch/tests/torch_backend/test_intersect_grad.py):
  reverse-mode geometry gradients
- [`test_multipath.py`](backends/torch/tests/torch_backend/test_multipath.py):
  reflection, EPC, visibility, and diffraction APIs
- [`benchmark_rayd_backends.py`](backends/torch/tests/benchmark_rayd_backends.py):
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
under [`backends/torch/tests`](backends/torch/tests) and should be rerun for the
target GPU, CUDA toolkit, and workload before making deployment decisions.

## Device and Stream Selection

The Dr.Jit backend follows Dr.Jit's current-thread CUDA device:

```python
import rayd.drjit as rd

rd.set_device(0)
```

Existing scenes and OptiX resources should not be reused across device switches
in the same process.

The Torch backend follows the device of its CUDA tensors and launches work on
the current Torch CUDA stream. Keep every tensor participating in one query on
the same device.

## Building from Source

RayD requires Python 3.10-3.14, CMake 3.22+, a C++17 compiler, CUDA, and the OptiX
SDK. On Windows, use Visual Studio 2022 with Desktop C++ tools.

Create an environment and install common build tools:

```powershell
conda create -n rayd python=3.11 -y
conda activate rayd
python -m pip install -U pip setuptools wheel
python -m pip install cmake ninja scikit-build-core
```

Build the Dr.Jit backend:

```powershell
python -m pip install "drjit==1.3.1" "nanobind==2.9.2"
.\scripts\build_local.cmd -Backend drjit
```

Build the Torch backend:

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\scripts\build_local.cmd -Backend torch
```

The helper detects the current GPU, compiles only its CUDA architecture, uses a
persistent per-architecture build directory, and enables parallel Ninja builds.
The multi-architecture CUDA matrix is reserved for release CI. Pass
`-PythonExe <path>` when `python` does not resolve to the intended environment.

Native downstream projects can add the Torch backend with CMake and link
against `rayd_torch_native_core`. The source-level integration declarations
are provided by
[`backends/torch/include/rayd/torch/integration_v2.h`](backends/torch/include/rayd/torch/integration_v2.h).
This interface is intended for projects built in the same CMake/libtorch graph;
it is not a stable binary ABI across unrelated libtorch builds.

Shared RF device headers are public source-level contracts under
`shared/include/rayd/shared/rf`. Transmission families are introduced as
dormant RayD candidates before a downstream pin and switch; dormancy does not
create a second production owner or authorize runtime fallback dispatch.

## Repository Layout

- [`backends/drjit`](backends/drjit): Dr.Jit Python bindings, C++/CUDA/OptiX
  implementation, examples, and tests
- [`backends/torch`](backends/torch): Torch Python API, dispatcher/autograd
  bindings, C++/CUDA/OptiX implementation, and tests
- [`shared/include`](shared/include): backend-neutral device contracts and UTD
  math
- [`shared/src`](shared/src): shared edge BVH, multipath, and scene-packing CUDA
  cores
- [`shared/contracts`](shared/contracts): machine-readable public API,
  operation, and path-exchange manifests
- [`shared/benchmarks`](shared/benchmarks): benchmark schemas and recorded
  baselines
- [`scripts`](scripts): local build helpers (`build_local.cmd` / `.ps1`)
- [`tests/packaging`](tests/packaging): distribution, namespace, and wheel-layout
  checks
- [`docs`](docs): migration, validation, and OptiX pipeline notes
- [`CHANGELOG.md`](CHANGELOG.md): release history

## Testing

Run packaging and namespace checks from the repository root:

```powershell
python -m unittest tests.packaging.test_project_metadata -v
python -m unittest tests.test_namespace_isolation -v
```

Run representative Dr.Jit suites:

```powershell
python -m unittest backends.drjit.tests.drjit.test_geometry -v
python -m unittest backends.drjit.tests.drjit.test_visibility_topk -v
python -m unittest backends.drjit.tests.drjit.test_reflection_epc -v
python -m unittest backends.drjit.tests.drjit.test_reflection_accumulation -v
python -m unittest backends.drjit.tests.drjit.test_diffraction_accumulation -v
```

Run representative Torch suites:

```powershell
python -m unittest backends.torch.tests.torch_backend.test_intersect_forward -v
python -m unittest backends.torch.tests.torch_backend.test_intersect_grad -v
python -m unittest backends.torch.tests.torch_backend.test_edge_queries -v
python -m unittest backends.torch.tests.torch_backend.test_multipath -v
```

The default local development environment used by this repository is
`witwin3`; downstream migration and release validation in the 0.6 cycle were
also run in `witwin2`.

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
