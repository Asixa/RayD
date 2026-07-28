# RayD Concept-Axis Layout and Backend Thinning Plan

Status: **executed — ADR-0039 and ADR-0040 govern the final layout.**

Date: 2026-07-28

Reference: `E:\Code\witwin-platform\channel\docs\dev\plans\15-concept-axis-layout-and-module-consolidation-plan.md`

This plan makes three owner decisions explicit:

1. RayD has **no maximum source-file line count**. A concept may occupy a large
   file when that is the clearest ownership boundary.
2. `drjit/` and `torch/` become root-level, thin distribution/build frontends.
   The current `backends/` container disappears.
3. Real implementation code is organized by **domain concept first**. Dr.Jit,
   Torch, and shared variants live next to one another, using names such as
   `scene.cpp`, `scene_jit.cpp`, and `scene_shared.cuh` where a concept
   has several owners.

This is a source-layout decision, not a proposal to merge the two runtimes.
`rayd.drjit` and `rayd.torch` remain independent, backend-native public APIs.

---

## 1. Context and diagnosis

The 0.6.0 dual-backend migration solved product naming, independent packaging,
and explicit backend selection. It deliberately used a repository layout that
mirrored the distributions:

```text
backends/drjit/{python,include,src,tests,...}
backends/torch/{python,include,src,tests,...}
shared/{include,src,contracts,...}
```

That layout is good at answering “which wheel owns this file?” but poor at
answering “where is the complete implementation of this concept?”

Examples:

- reflection work is spread across Dr.Jit `scene/`, Dr.Jit `multipath/`,
  Torch `reflection/`, Torch `scene/`, and several `shared/` subtrees;
- edge work is spread across both backend trees plus shared BVH, edge, OptiX,
  math, and RT headers;
- diffraction work is split by backend, host/device role, PTX role, AD role,
  and header/source artifact category;
- the Torch Python `Scene` surface is split across `scene.py`, `mesh.py`,
  `types.py`, `autograd.py`, and `_multi.py`;
- `backends/torch/src/torch_ext/` encodes both a backend axis and an extension
  implementation detail, while its `rf/` child adds another umbrella over the
  actual concepts `transmission`, `scattering`, and `diffraction`.

The result materializes a matrix:

```text
domain concept × backend × artifact kind × compiler role
```

Most changes, reviews, parity investigations, and ownership transfers move
along the **domain concept** axis. The current top-level backend split is
orthogonal to that change axis.

### 1.1 Measured baseline

Measured at the working tree on 2026-07-28, excluding generated `*_ptx.h`
headers, tests, examples, documentation, and build scripts:

| Area | Files | Lines |
| --- | ---: | ---: |
| Dr.Jit package Python | 5 | 1,559 |
| Torch package Python | 14 | 8,009 |
| Dr.Jit native source | 39 | 28,052 |
| Torch native source | 72 | 36,481 |
| Shared native source | 8 | 2,056 |
| Dr.Jit headers | 40 | 6,115 |
| Torch headers | 45 | 6,974 |
| Shared headers | 63 | 16,286 |
| **Total core Python/native source** | **286** | **95,532** |

The large file count is not caused by a single-file-size problem. The largest
files are already substantial:

- `backends/torch/src/torch_ext/diffraction/ops.cpp`: 3,512 lines;
- `backends/torch/python/rayd/torch/_multi.py`: 2,982 lines;
- `backends/torch/src/torch_ext/reflection/ops.cpp`: 2,957 lines;
- `backends/drjit/src/rayd.cpp`: 2,682 lines;
- `backends/torch/python/rayd/torch/autograd.py`: 2,137 lines.

The problem is that a large concept is accompanied by many sibling files in
other trees. A path/name inventory finds:

| Concept token | Hand-written native files containing the concept |
| --- | ---: |
| reflection | 52 |
| scene | 40 |
| diffraction | 34 |
| edge | 33 |
| current `rf` umbrella | 27 |
| bvh | 25 |

These counts overlap, but that overlap is itself the point: one file often sits
at the intersection of several artifact axes and is difficult to discover from
the concept alone.

Path movement is also a governed migration rather than a simple `git mv`:

- 66 repository files directly mention a `backends/drjit` or
  `backends/torch` path;
- 35 files directly mention `shared/include` or `shared/src`;
- `shared/contracts/compile_policy.json` owns exact translation-unit paths;
- `backends/drjit/ptx_sources.json` owns exact committed-PTX source closures;
- the Torch source bundle and its tests currently pin
  `backends/torch/{CMakeLists.txt,include,src}` plus `shared/{include,src}`;
- multiple ADRs and downstream Channel pins treat these paths as reviewed
  identity.

---

## 2. Decisions

### D1 — Retire file-line limits as an architectural rule

RayD currently has no machine-enforced source-file line budget. This plan makes
that absence intentional:

- do not introduce a Python, C++, CUDA, header, or CMake file-line ceiling;
- do not split a concept merely because a file is “too long”;
- line count remains a measurement, never an acceptance gate.

This does **not** remove other quality controls. Function complexity,
testability, compile time, generated-code size, binary size, register pressure,
public ABI size, and import-time side effects are separate concerns and may
justify a real boundary.

### D2 — The concept axis is primary

Production implementation code is located by concept:

- geometry/scene;
- BVH and edge queries;
- visibility and penetration;
- reflection;
- diffraction;
- transmission;
- scattering;
- cross-concept field transport primitives;
- SDF;
- surfel;
- runtime/binding infrastructure.

Backend and compiler role are expressed inside that concept by a deliberately
asymmetric filename convention:

- an unsuffixed implementation file is Torch-owned;
- `*_jit.*` is the adjacent Dr.Jit implementation;
- `*_shared.*` is a genuinely backend-neutral numerical owner;
- `*_stable.*` is a separate stable-ABI target, not another backend.

The target production tree therefore contains no `*_torch.*` or `*_drjit.*`
names. `jit` is reserved in filenames for Dr.Jit; a framework-neutral JIT
mechanism uses a purpose name such as `compile`, `ptx`, or `optix` instead.
Public API names remain symmetric: `rayd.torch` and `rayd.drjit`.

`common/`, `torch_ext/`, and artifact-category directories do not remain
primary ownership locations.

### D3 — Dr.Jit and Torch are thin frontends, not duplicate source universes

The root `drjit/` and `torch/` directories own only:

- distribution metadata;
- backend entry-point CMake;
- public Python package entry points and typing markers;
- backend-specific build helpers that cannot be shared;
- release/readme material whose subject is that distribution.

They do not own complete copies of `scene`, `edge`, `reflection`,
`diffraction`, `transmission`, or `scattering` implementation trees.

### D4 — Co-location does not mean runtime abstraction

The repository may place these files next to one another:

```text
scene.cpp
scene_jit.cpp
scene_shared.cuh
```

It must not create a host-side abstract `Scene` base class or a fake
framework-neutral tensor/allocator/stream abstraction solely to remove
duplication.

The following stay backend-specific:

- framework tensors and object models;
- allocator and lifetime ownership;
- current-stream acquisition;
- AD graph/tape integration;
- Scene/Mesh host objects;
- OptiX context, module, pipeline, SBT, GAS/IAS, and cache ownership;
- nanobind versus Torch dispatcher/LibTorch registration;
- backend-specific result containers and packed layouts.

### D5 — Shared code means one numerical owner

Code is named `*_shared.*` or placed under `include/rayd/shared/` only when both
backends compile or consume the same implementation as its single numerical
owner.

Two similar implementations do not become shared merely because they are now
adjacent. Extraction requires:

1. identical semantics and failure behavior;
2. explicit raw-pointer/value contracts that do not leak framework objects;
3. preserved stream and allocation ownership;
4. parity and generated-code evidence where numerical compilation matters;
5. removal of both former duplicate numerical bodies in the same change.

### D6 — File boundaries need a concrete reason

A new production file is justified only by at least one of:

- a public ABI/include boundary;
- a different compiler or language toolchain;
- a different CUDA numeric compile profile;
- a distinct PTX module/pipeline identity;
- an independently optional or lazy-loaded runtime component;
- an independently owned operation family with its own activation/rollback
  contract;
- generated code;
- a test oracle that must remain independent.

“The file became long”, “this is a config/result/helper”, or “one class per
file” is not a sufficient reason.

---

## 3. Target repository layout

```text
RayD/
├── pyproject.toml                 # file-free rayd meta-distribution
├── CMakeLists.txt                 # workspace entry: add_subdirectory(drjit|torch)
│
├── drjit/                         # thin rayd-drjit distribution frontend
│   ├── pyproject.toml
│   ├── CMakeLists.txt
│   ├── cmake/                     # Dr.Jit/nanobind-only build helpers
│   ├── scripts/
│   └── python/rayd/drjit/
│       ├── __init__.py            # public import/export layer only
│       ├── __init__.pyi
│       ├── _C.pyi
│       ├── path_exchange.py       # thin public submodule if retained
│       └── py.typed
│
├── torch/                         # thin rayd-torch distribution frontend
│   ├── pyproject.toml
│   ├── CMakeLists.txt
│   ├── scripts/                   # source-bundle/ABI build tools
│   └── python/rayd/torch/
│       ├── __init__.py            # public import/export layer only
│       ├── path_exchange.py       # thin public submodule if retained
│       └── py.typed
│
├── python/rayd/_impl/             # no __init__.py; private PEP 420 subtree
│   ├── runtime.py                 # Torch implementation; unsuffixed by convention
│   ├── runtime_jit.py             # adjacent Dr.Jit implementation
│   ├── capabilities.py
│   ├── capabilities_jit.py
│   ├── path_exchange.py
│   ├── path_exchange_jit.py
│   ├── geometry.py                # Torch Ray/intersection/edge records + AD facade
│   ├── scene.py                   # Torch Mesh + single-device Scene
│   ├── multi.py                   # genuinely lazy replicated execution
│   ├── multipath.py               # Torch reflection/diffraction records + AD facade
│   ├── camera.py
│   └── sdf.py
│
├── include/rayd/
│   ├── shared/                    # one backend-neutral numerical owner
│   ├── torch/                     # stable typed Torch integration ABI
│   └── ...                        # installed Dr.Jit public headers; see §4.3
│
├── src/                           # concept-major implementation
│   ├── runtime/
│   ├── camera/
│   ├── scene/
│   ├── bvh/
│   ├── edge/
│   ├── visibility/
│   ├── penetration/
│   ├── reflection/
│   ├── diffraction/
│   ├── transmission/
│   ├── scattering/
│   ├── sdf/
│   ├── surfel/
│   └── bindings/
│
├── generated/
│   └── drjit/ptx/                 # eight committed generated PTX headers
│
├── contracts/                     # public API, operation, compile policy
├── benchmarks/
├── examples/
├── tests/
│   ├── scene/                     # unsuffixed Torch, *_jit.py Dr.Jit, parity together
│   ├── camera/
│   ├── edge/
│   ├── visibility/
│   ├── penetration/
│   ├── reflection/
│   ├── diffraction/
│   ├── transmission/
│   ├── scattering/
│   ├── sdf/
│   ├── surfel/
│   ├── native/
│   ├── packaging/
│   └── governance/
└── docs/
```

Expected result: approximately **160–190** hand-written core source files
instead of 286, with most implementation paths at depth 2–3. This is an
estimate, not a quota. A correct CUDA translation-unit boundary takes priority
over the target count.

### 3.1 Concrete native file plan

This is the proposed **final** `src/` tree, not an illustrative naming sample.
It deliberately keeps large host facades and preserves only justified CUDA,
PTX, stable-ABI, and private cross-TU boundaries.

```text
src/
├── runtime/
│   ├── runtime_jit.cpp                 # OptiX setup, pipeline registry, launch audit, multipath config
│   ├── multipath_internal_jit.h        # one private contract shared by Dr.Jit concept facades
│   ├── optix.cpp                   # Torch OptiX context + pipeline cache
│   ├── diagnostics.cpp             # stats/diagnostic operator facade
│   └── diagnostics.cu              # stats kernels
│
├── camera/
│   ├── camera.cpp                  # Torch camera operator facade
│   ├── camera.cu                   # eager camera kernels
│   └── camera_stable.cu            # stable-ABI camera library; separate target
│
├── scene/
│   ├── scene_internal_jit.h            # private Scene state shared by Dr.Jit concept facades
│   ├── scene_jit.cpp                   # Mesh + Scene + intersect + OptiX scene + Dr.Jit custom op
│   ├── trace_backend_jit.cpp           # CUDA/OptiX eager trace backend selection
│   ├── triangle_bvh_jit.cu             # Dr.Jit triangle-BVH adapter
│   ├── multipath_jit.cu                # fused CUDA multipath executor [fast_math]
│   ├── packing_shared.cu                 # backend-neutral scene packing
│   ├── scene.cpp                   # Torch Scene lifecycle/cache/operator facade
│   ├── intersection.cpp            # Torch intersection typed/dispatcher facade
│   ├── cache.cu                    # Torch persistent scene-cache kernels
│   ├── intersection.cu             # intersection forward + backward kernels
│   ├── intersection_optix.cu       # Torch intersection OptiX module
│   ├── triangle_bvh.cu             # Torch triangle-BVH adapter
│   ├── multipath.cu                # fused Torch CUDA multipath executor
│   └── intersection_stable.cu      # stable-ABI intersection validity op
│
├── bvh/
│   ├── build_shared.cu                   # generic GPU BVH build/refit implementation
│   └── triangle_query_shared.cu          # generic triangle-query implementation
│
├── edge/
│   ├── edge_shared.cu                    # shared edge BVH build/query + AABB/distance kernels
│   ├── edge_jit.cpp                    # complete Dr.Jit edge host facade + OptiX ownership
│   ├── edge_bvh_jit.cu                 # Dr.Jit compact edge-BVH adapter
│   ├── edge_optix_jit.cu               # committed Dr.Jit edge OptiX source
│   ├── edge.cpp                    # complete Torch edge typed/dispatcher facade
│   ├── edge_bvh.cu                 # Torch compact edge-BVH adapter
│   ├── edge_queries.cu             # point/ray/top-k forward and backward kernels
│   └── edge_optix.cu               # Torch point/ray/top-k OptiX source
│
├── visibility/
│   ├── visibility_jit.cpp              # Dr.Jit segment visibility host facade
│   ├── visibility_optix_jit.cu         # committed Dr.Jit segment-visibility OptiX source
│   ├── visibility.cpp              # Torch segment/pair/chain visibility facade
│   ├── visibility_ad.cu            # visibility VJP/JVP object kernels
│   ├── visibility_optix.cu         # Torch segment visibility OptiX source
│   └── axial_edge_visibility_optix.cu  # distinct ADR-0029 PTX identity
│
├── penetration/
│   ├── penetration.cpp             # typed request/result facade + pipeline ownership
│   ├── penetration.cu              # primal/backward/JVP [precise_no_ftz]
│   └── penetration_optix.cu        # ADR-0033 OptiX module [precise_no_ftz]
│
├── reflection/
│   ├── reflection_jit.cpp              # trace + accumulation + EPC Dr.Jit host facade
│   ├── trace_optix_jit.cu              # committed reflection-trace OptiX source
│   ├── epc_optix_jit.cu                # committed EPC OptiX source
│   ├── accumulation_optix_jit.cu       # committed reflection-accumulation OptiX source
│   ├── reflection_kernels_jit.cu       # dedup + EPC-field object kernels
│   ├── dedup_shared.cu                   # backend-neutral dedup implementation
│   ├── reflection.cpp              # complete Torch reflection typed/dispatcher facade
│   ├── trace_optix.cu              # Torch reflection-trace OptiX source
│   ├── epc_optix.cu                # Torch EPC OptiX source
│   ├── accumulation_optix.cu       # Torch accumulation OptiX source
│   └── reflection_kernels.cu       # AD + dedup + EPC field/geometry + reduction kernels
│
├── diffraction/
│   ├── diffraction_jit.cpp             # paths + direct/chain/coherent accumulation facade
│   ├── paths_optix_jit.cu              # committed diffraction-path OptiX source
│   ├── accumulation_optix_jit.cu       # committed diffraction-accumulation OptiX source
│   ├── diffraction_ad_jit.cu           # Dr.Jit accumulation AD object kernels
│   ├── diffraction.cpp             # complete Torch diffraction typed/dispatcher facade
│   ├── paths_optix.cu              # Torch diffraction-path OptiX source
│   ├── accumulation_optix.cu       # Torch diffraction-accumulation OptiX source
│   ├── wedge.cu                    # pure wedge field family [fast_math]
│   └── diffraction_kernels.cu      # path init + accumulation reduction/AD
│
├── transmission/
│   └── transmission.cu             # layer stack + sequence primal/backward/JVP
│
├── scattering/
│   ├── scattering_internal.cuh     # one private validation + chain-AD helper contract
│   ├── table.cu                    # table primal/sample/PDF [nvcc_default]
│   ├── table_ad.cu                 # table backward/JVP [no_fmad]
│   ├── ensemble.cu                 # ensemble primal/backward/JVP [no_fmad]
│   ├── patch.cu                    # patch integral primal/backward/JVP [no_fmad]
│   ├── chain_ensemble.cu           # chain-ensemble primal/backward/JVP [no_fmad]
│   └── chain_realization.cu        # chain-realization primal/backward/JVP [no_fmad]
│
├── sdf/
│   ├── sdf.cpp                     # typed/dispatcher facade
│   └── sdf.cu                      # forward + backward/JVP kernels
│
├── surfel/
│   ├── surfel_jit.cpp                  # Surfel object + OptiX host ownership
│   └── surfel_optix_jit.cu             # committed surfel OptiX source
│
└── bindings/
    ├── module_jit.cpp                  # sole nanobind module; current rayd.cpp
    ├── library.cpp                 # sole Torch library/dispatcher registration owner
    ├── module.cpp                  # small _C metadata compatibility module
    ├── legacy_anchor.cpp           # distinct legacy shared-library export anchor
    ├── tensor_contract.cpp         # shared Torch tensor-contract checks
    └── integration_internal.h      # private typed-operation access contract
```

This plan has **79 native implementation/private-header files under `src/`**,
down from the current 119. Generated PTX headers and public headers are counted
separately. The number is a consequence of the ownership boundaries, not a new
budget.

Host consolidation is intentional:

| Final file | Current files absorbed |
| --- | --- |
| `runtime/runtime_jit.cpp` | `optix.cpp`, `native_launch_audit.cpp`, `multipath/pipelines.cpp`, `scene_multipath_config.cpp` |
| `scene/scene_jit.cpp` | `mesh.cpp`, `scene.cpp`, `scene_intersect.cpp`, `scene_optix.cpp`, `scene_custom_op.cpp` |
| `scene/trace_backend_jit.cpp` | `cuda_trace_backend.cpp`, `optix_trace_backend.cpp` |
| `reflection/reflection_jit.cpp` | `scene_multipath_reflection.cpp`, `scene_multipath_reflection_accum.cpp`, `scene_multipath_epc.cpp` |
| `diffraction/diffraction_jit.cpp` | the three Dr.Jit scene diffraction facade files |
| `runtime/optix.cpp` | `common/optix_pipeline.cpp`, `scene/optix_context.cpp` |
| `scene/scene.cpp` | `ops_scene.cpp`, `scene_cache.cpp` |
| `reflection/reflection.cpp` | reflection-owned portions of `reflection/ops.cpp` plus `reflection/pipeline.cpp` |
| `visibility/visibility.cpp` | visibility-owned portions currently mixed into `reflection/ops.cpp` |
| `diffraction/diffraction.cpp` | `diffraction/ops.cpp`, `diffraction/pipeline.cpp` |
| `penetration/penetration.cpp` | `penetration/ops.cpp`, `penetration/pipeline.cpp` |
| `surfel/surfel_jit.cpp` | `surfel.cpp`, `surfel_optix.cpp` |

CUDA consolidation is narrower:

- `edge_shared.cu` absorbs the four shared edge build/query/distance/AABB units;
- `edge_queries.cu` absorbs Torch edge forward/backward/top-k units;
- `reflection_kernels.cu` absorbs only `nvcc_default` reflection object
  units; the three OptiX module sources remain separate;
- `diffraction_kernels.cu` absorbs only `nvcc_default` init/reduction/AD
  units; wedge and OptiX sources remain separate;
- `transmission.cu` absorbs the three `nvcc_default` transmission units;
- each no-FMAD scattering family merges its own primal and AD companions, but
  families do not merge with one another;
- `sdf.cu` absorbs SDF forward and backward/JVP.

`embed_ptx.py` is build tooling, not implementation, and moves to
`torch/scripts/embed_ptx.py`. No generated `*_ptx.h` file lives under `src/`.
### 3.2 Python concept pattern

The public packages stay intentionally small:

```python
# rayd/torch/__init__.py
from rayd._impl.camera import Camera
from rayd._impl.scene import Mesh, Scene
...
```

Each wheel installs only its backend-owned implementation files from an explicit manifest:

| Distribution | Public files | Private implementation files |
| --- | --- | --- |
| `rayd-drjit` | `rayd/drjit/**` | `rayd/_impl/*_jit.py` |
| `rayd-torch` | `rayd/torch/**` | explicit unsuffixed `rayd/_impl/*.py` members |

`rayd/_impl/` has no `__init__.py`. The Torch wheel uses an exact file manifest,
not a broad `*.py` package-data glob, so it cannot accidentally collect adjacent
`*_jit.py` files. The two wheel member sets remain disjoint,
so installing, upgrading, or uninstalling one backend cannot overwrite or
remove the other backend's files.

This intentionally changes the old stronger rule “every wheel member must be
under its public backend subtree” into the more precise rule:

> Every installed file has exactly one distribution owner, and the two
> backend wheel member sets are disjoint.

The uninstall/coexistence requirement remains unchanged.

### 3.3 Why `_multi` remains a separate file

`_multi.py` is not retained because it is 2,982 lines or because “multi-device
deserves a package”. It stays as `multi.py` because ADR-0038 requires a
single-device `Scene` not to import the orchestration layer.

That is a real import-time and runtime-lifetime boundary. Merging it into
`scene.py` would change behavior even if every function body were copied
unchanged.

The same rule applies to any source-bundle loader or optional runtime component
whose absence from `sys.modules` is part of a tested contract.

---

## 4. Concrete consolidation map

### 4.1 Python

| Current files | Target owner | Notes |
| --- | --- | --- |
| Torch `mesh.py` + single-device parts of `scene.py` | `scene.py` | Mesh and Scene lifecycle change together |
| geometry/edge portion of Torch `types.py` + matching part of `autograd.py` | `geometry.py` | result records live with their operation facade |
| reflection/diffraction portion of `types.py` + matching part of `autograd.py` | `multipath.py` | one multipath Python owner, not one AD file plus one types file |
| `_stable.py` + `_legacy.py` + `_compile.py` + import/load part of `__init__.py` | `runtime.py` | preserve exact load and registration order |
| `_warmup.py` | `runtime.py` or `warmup.py` | decide from measured import/lazy behavior, not size |
| `_multi.py` | `multi.py` | remains lazy by contract |
| Dr.Jit `__init__.py` runtime logic | `runtime_jit.py` | public `__init__` becomes exports only |
| mirrored `_capabilities.py` files | adjacent `capabilities.py` / `capabilities_jit.py` | generated/validated from one manifest; backend values stay explicit |
| mirrored `path_exchange.py` files | adjacent unsuffixed/`*_jit.py` files plus thin public submodules | independent wheels retained; no shared installed file |

`camera.py` and `sdf.py` remain small because they are genuine
domain concepts, not because they satisfy a preferred line range.

Before moving Python definitions, audit:

- documented direct submodule imports;
- `__module__` and pickle-qualified identities;
- `isinstance` behavior;
- monkeypatch targets in tests/downstreams;
- import order and module-level registrations;
- the ADR-0038 lazy-import property;
- `__all__` and PEP 561 resolution.

Only `rayd.drjit` and `rayd.torch` root surfaces are assumed stable by default.
Any direct submodule path found to be public gets an explicit thin module or a
documented breaking migration; no accidental compatibility shell is added.

### 4.2 Native implementation

| Current owner | Target concept owner |
| --- | --- |
| `backends/*/src/scene/*` intersection/cache/lifecycle | unsuffixed Torch files beside `src/scene/*_jit.*` Dr.Jit variants |
| Dr.Jit scene multipath wrapper files | `src/{visibility,reflection,diffraction}/*_jit.cpp` |
| `backends/*/.../edge/*` + `shared/{bvh,edge}` | `src/{bvh,edge}/` with adjacent shared/backend variants |
| backend reflection trees + shared reflection/multipath/OptiX pieces | `src/reflection/` and `include/rayd/shared/reflection/` |
| backend diffraction trees + shared UTD/diffraction pieces | `src/diffraction/` and `include/rayd/shared/diffraction/` |
| Torch `torch_ext/rf/layer_stack*` and `transmission_sequence*` | `src/transmission/` with unsuffixed Torch files; passive-complex/medium/Fresnel/layer-stack helpers move to `include/rayd/shared/transmission/` |
| Torch `torch_ext/rf/scattering*` | `src/scattering/` with unsuffixed Torch files; table math moves to `include/rayd/shared/scattering/` |
| Torch `torch_ext/rf/diffraction_wedge.cu` | existing `src/diffraction/` owner beside the other diffraction implementations |
| cross-concept `field_transport.cuh` and `field_transport_ad.cuh` | flat, explicitly named shared primitives; they do not justify an `rf/` directory |
| Torch penetration | `src/visibility/penetration*.*` or `src/penetration/` if its compile profile justifies a separate concept |
| Torch SDF + shared SDF math | `src/sdf/` plus `include/rayd/shared/sdf/` |
| Dr.Jit surfel | `src/surfel/*_jit.*` |
| nanobind/Torch module/dispatcher entry files | unsuffixed Torch entry files plus `src/bindings/*_jit.*` Dr.Jit entries |
| backend/common OptiX and tensor helpers | concept owner first; truly cross-concept runtime code in `src/runtime/` |

### 4.3 Public headers

The stable aggregate boundary and the leaf source headers are treated
differently:

- `rayd/integration/torch.h` keeps identity `rayd.torch.integration` and uses
  API version 7 for this source-path break;
- the current `rayd/torch/rf/*` and `rayd/shared/rf/*` leaf paths are removed,
  because retaining them would preserve the umbrella this plan is eliminating;
- transmission headers move under `rayd/shared/transmission/` and `rayd/transmission/torch.h`;
- scattering headers move under `rayd/shared/scattering/` and `rayd/scattering/torch.h`;
- UTD and wedge-field headers move under the existing diffraction owner;
- genuinely cross-concept field transport uses the explicit flat names
  `rayd/shared/field_transport.cuh` and
  `rayd/field_transport/torch_ad.cuh`, not another category directory;
- ADR-0040 approves the installed Dr.Jit shared/multipath include hard break; a
  forwarding header or compatibility include root is forbidden.

ADR-0002 and ADR-0026 make several current `rf/` leaf headers public
source-level contracts for Channel. Their path and ownership-namespace clauses
must therefore be superseded and activated by an atomic Channel pin/include
update. Their numerical, fusion, stream, derivative, and failure contracts stay
unchanged.

No forwarding `rf/` headers or namespace aliases are added. Private headers may
move beside their concept implementation; public headers remain under
`include/rayd/` and are installed from their canonical concept owner.

### 4.4 CMake

Do not replace the current large backend CMake files with dozens of
one-concept CMake fragments.

Target:

```text
CMakeLists.txt
cmake/RayDCompilePolicy.cmake
cmake/RayDOptix.cmake
drjit/CMakeLists.txt
torch/CMakeLists.txt
```

The backend CMake files may be long. They should declare concept-major source
sets and attach per-source CUDA policy explicitly. A new CMake fragment is
justified only when it owns reusable build behavior, not merely a list of five
files.

---

## 5. Prerequisites and ADR impact

### P1 — Accept this source ownership axis

This plan reverses the repository-layout part of the 2026-07-09 dual-backend
plan while preserving its public runtime and packaging goals. Approval must
state:

- public backend namespaces remain `rayd.drjit` and `rayd.torch`;
- distribution set remains `rayd`, `rayd-drjit`, `rayd-torch`;
- runtime objects never cross backends;
- source ownership is now concept-major rather than distribution-major.

### P2 — Replace ADR-0036

ADR-0036 currently concludes that mirrored Python files cannot be deduplicated
because every installed backend file must live below its own public subtree.

The new decision is different:

- canonical implementation pairs are co-located in source;
- each wheel still owns distinct manifest-listed files;
- no installed file is shared by both wheels;
- uninstall independence is preserved;
- identical semantics are governed by a source/manifest test rather than two
  distant source trees.

ADR-0036 should be superseded, not edited in place.

### P3 — Amend ADR-0034 source-bundle layout

The passive resource remains fixed at `rayd/torch/_source`, but its internal
source tree changes from:

```text
backends/torch/{CMakeLists.txt,include,src}
shared/{include,src}
```

to the canonical root layout needed by the typed integration target:

```text
torch/CMakeLists.txt
include/
src/
cmake/
```

`rayd-source.json`, `source-files.json`, their digests, package tests, and every
downstream pin must change atomically. The bundle remains passive, relocatable,
fully manifested, and compiled in the consumer's graph.

### P4 — Preserve ADR-0028 identity through the ADR-0040 path break

Moving the source file does not authorize a new include identity. The bundle
must still expose:

```cpp
#include <rayd/integration/torch.h>
```

with exact identity `rayd.torch.integration` and current
`kIntegrationApiVersion = 7`, as approved by ADR-0040; no forwarding include is provided.

### P5 — Freeze all path-indexed governance before movement

Capture and classify every exact source path in:

- `contracts/compile_policy.json`;
- Dr.Jit `ptx_sources.json`;
- `contracts/operations.json`;
- public API and path-exchange contracts;
- source bundle manifests;
- ABI audit data;
- CMake source lists and scripts;
- packaging tests;
- downstream Channel path/hash pins;
- ADR evidence and historical documents.

Historical evidence remains historical and should not be mechanically rewritten.
Only live governance and current documentation move.

### P6 — Supersede the `rf/` source-API umbrella

ADR-0002 and ADR-0026 currently name these public source families under
`rayd/{shared,torch}/rf/`. The new layout makes the operation concept primary:

```text
transmission/       passive complex, medium, Fresnel, layer stack, sequence
scattering/         resident-table, ensemble, patch, and chain families
diffraction/        UTD, wedge field, paths, and accumulation
field_transport.*   only the helpers genuinely consumed across those concepts
```

The corresponding C++ ownership namespaces move away from a generic `rf`
owner in the same source-ABI cut. Channel updates its direct includes,
qualified names, RayD pin, and source-manifest digest atomically. No legacy
header, namespace alias, copied helper, or runtime owner selection remains.

This prerequisite changes names and paths only. It does not authorize merging
translation units, changing compile profiles, or altering any operation-family
numerics.

---

## 6. Migration phases

Every phase is independently buildable and reviewable. Layout motion does not
authorize numerical, launch, reduction, synchronization, API, or fallback
changes.

### Phase 0 — Baseline and governance

1. Record the tracked file set, test node-id set, public symbols, wheel member
   sets, source-bundle manifest, PTX headers/hashes, compile-policy assignment,
   native exported symbols, and benchmark baselines.
2. Add a written policy that file line count is not a gate.
3. Add an architecture check that rejects new production paths under
   `backends/` after the migration starts.
4. Add a one-owner check for backend wheel files and concept implementation
   variants.

Gate:

- working tree baseline is green;
- all three wheels build;
- Dr.Jit-only, Torch-only, coexistence, and uninstall matrix pass;
- compile-policy and PTX-source audits pass before any move.

### Phase 1 — Promote backend frontends to the root

Move:

```text
backends/drjit -> drjit
backends/torch -> torch
```

Do not consolidate implementation files yet. Update only:

- root workspace members and `add_subdirectory`;
- `scripts/build_local.*`;
- backend-local relative paths;
- test discovery and documentation that describes current paths.

Gate:

- both direct backend builds and the root meta build work;
- no public Python/dispatcher/include identity changes;
- test node-id set is identical;
- wheel contents are byte-equivalent except allowed metadata/source-path
  records.

### Phase 2 — Establish root canonical include/source trees

Move shared and backend implementation files to `include/`, `src/`, and
`generated/`, giving Dr.Jit variants a `_jit` suffix, leaving Torch variants
unsuffixed, and using `_shared` only where one implementation is genuinely
shared.

This phase is **move-only**:

- no function bodies merged;
- no CUDA translation units merged;
- no header bodies consolidated;
- no registration order changed;
- no compile option changed.

The temporary result may have more files than the final tree. Its purpose is to
make complete concepts visible in one location before deciding which file
boundaries are artificial.

Gate:

- `contracts/compile_policy.json` re-derives the same profile per logical TU;
- frozen divergences D1–D10 remain frozen;
- Torch same-graph integration builds from the new source bundle;
- public installed include spellings remain unchanged;
- native symbols and operation manifests are unchanged.

### Phase 3 — Regenerate and prove committed PTX identity

Moving a source or transitive header changes the path-indexed PTX closure even
when file bytes are unchanged. Do not “repair” `ptx_sources.json` by writing new
paths alone.

For all eight committed Dr.Jit PTX modules:

1. regenerate with the recorded toolchain/options;
2. byte-compare each regenerated `*_ptx.h` against the committed header;
3. investigate any difference before proceeding;
4. copy only genuinely regenerated headers;
5. run the PTX source audit to write the new closure;
6. mark verification only after byte comparison.

This phase is a hard gate. A source move that cannot prove the committed device
artifact remains valid does not land.

### Phase 4 — Build the Python thin frontends

Create `python/rayd/_impl`, move Torch modules into unsuffixed files and adjacent
Dr.Jit variants into `*_jit.py` files, and reduce public backend packages to
export/typing layers.

Preserve:

- root public imports and signatures;
- capability manifest values;
- path-exchange values and validation;
- native-load and dispatcher registration order;
- ADR-0038 lazy multi-device import;
- Torch-only/Dr.Jit-only dependency isolation;
- PEP 561 typing behavior.

Gate:

- public API snapshot/manifest is semantically identical;
- `from rayd.drjit import *` and `from rayd.torch import *` are unchanged;
- documented direct `path_exchange` imports still work;
- no backend imports the other framework;
- wheel file sets are disjoint;
- install/uninstall matrix passes in clean environments.

### Phase 5 — Consolidate Python by concept

Merge `types.py`, `autograd.py`, `mesh.py`, `scene.py`, and runtime helpers
according to §4.1.

This is the first phase that merges module objects, so it requires focused
adversarial review for:

- lost validation;
- changed module-level execution order;
- changed lazy imports;
- changed class identity;
- monkeypatch target changes;
- narrowed public exports;
- duplicate native registration;
- test set shrinkage.

Gate:

- exact public behavior and exception tests pass;
- forward/reverse AD suites pass;
- `torch.compile` tests pass;
- single-device execution still does not import `multi`;
- parity and full backend suites pass.

### Phase 6 — Consolidate native host facades

Within each concept, merge small host `.cpp` files and private headers when they
share:

- owner;
- compiler;
- compile options;
- static-registration lifetime;
- public ABI boundary;
- activation/rollback boundary.

Good candidates include several Dr.Jit `scene_multipath_*.cpp` wrappers after
they have moved to their actual reflection/diffraction/visibility owners.

Do not merge CUDA or OptiX units merely to reduce count. Such a merge is allowed
only with compile-policy, generated-code, numerical, and performance evidence.

Gate:

- symbols, launch count, stream behavior, and failure behavior unchanged;
- CMake compile commands show the same numeric profiles;
- cold OptiX pipeline creation tests pass in fresh subprocesses;
- no source-bundle manifest omissions;
- targeted performance does not regress beyond the recorded thresholds.

### Phase 7 — Reorganize tests and remove empty structure

Move backend tests next to their concept siblings:

```text
tests/reflection/test_trace.py
tests/reflection/test_trace_jit.py
tests/reflection/test_trace_parity.py
```

Delete empty `backends/`, `torch_ext/`, and artifact-category packages only
after all callers and governance manifests point at the canonical owners.

Gate:

- test node-id set is intentionally mapped with zero silent loss;
- full Dr.Jit, Torch, parity, packaging, governance, and native suites pass;
- repository search finds no live production reference to `backends/`;
- source bundle installs and builds from a clean wheel.

---

## 7. Verification matrix

### 7.1 Layout invariants

- no production implementation lives below `backends/`;
- `drjit/` and `torch/` contain only approved thin-frontend categories;
- unsuffixed implementation files are Torch-owned and only Dr.Jit variants use
  the `_jit` suffix;
- no target production filename uses `_torch` or `_drjit`;
- every `*_shared.*` file has one implementation owner and both consumers use
  it directly;
- no artifact-category package is introduced solely to shorten a file;
- no production target contains an `rf/` directory or generic `rf` ownership
  namespace; transmission, scattering, and diffraction are direct owners;
- no file-line maximum is enforced.

### 7.2 Packaging invariants

- `rayd` remains a file-free meta wheel;
- no wheel installs `rayd/__init__.py`;
- backend wheel file sets are disjoint;
- every `rayd/_impl/*_jit.py` file belongs only to `rayd-drjit`;
- every manifest-listed unsuffixed `rayd/_impl/*.py` file belongs only to
  `rayd-torch`;
- either backend can be installed, upgraded, and uninstalled independently;
- import order remains irrelevant;
- the Torch source bundle is complete, passive, relocatable, and hash-locked.

### 7.3 Numerical/native invariants

- no kernel body, launch geometry, launch count, reduction order, or atomic
  order changes during move-only phases;
- every CUDA TU retains its compile-policy profile;
- all committed Dr.Jit PTX modules are regenerated and verified after their
  source closures move;
- stable typed integration keeps its name, identity, API version, and operation
  signatures;
- cross-backend parity remains within the existing per-operation contracts;
- fresh-subprocess OptiX cold-create coverage stays green.

### 7.4 Python invariants

- public root names, signatures, defaults, errors, and capabilities are
  unchanged unless separately approved;
- no Torch expression or CPU fallback is introduced;
- unsupported derivatives continue to fail loudly;
- lazy multi-device import remains lazy;
- class/result identity changes are either prevented or explicitly migrated;
- PEP 561 behavior and stub ownership remain complete.

---

## 8. Governance changes

| Artifact | Required change |
| --- | --- |
| `pyproject.toml` workspace members | `backends/drjit`, `backends/torch` → `drjit`, `torch` |
| root `CMakeLists.txt` | root backend subdirectories |
| `scripts/build_local.*` | root backend discovery |
| `contracts/compile_policy.json` | exact TU path migration with unchanged logical profiles |
| compile-policy schema/test | parse canonical root CMake and reject global numeric flags |
| Dr.Jit `ptx_sources.json` + audit script | canonical closure paths only after real regeneration |
| Torch source-bundle generator/tests | canonical `torch/`, `include/`, `src/`, `cmake/` tree |
| wheel-layout tests | file-level single ownership, not public-subtree-only ownership |
| wheel install matrix | unchanged and mandatory |
| public API/path-exchange manifests | validate thin frontend exports and adjacent implementations |
| ABI audit/source manifest | new physical paths, unchanged public identities |
| ADR-0002 and ADR-0026 | supersede `rf/` path/namespace clauses; preserve numerical and ownership contracts |
| ADR-0036 | superseded |
| ADR-0034 | amended/superseded for internal bundle layout |
| ADR-0028 | unchanged stable identity; tests repointed only |
| ADR-0035 | unchanged profiles/divergences; path records repointed only |
| Channel RayD pins | atomic source bundle commit/manifest/path update |

Historical ADR evidence should retain old paths with a “historical layout”
annotation. Rewriting old evidence to look current destroys audit value.

---

## 9. Non-goals

- No combined Dr.Jit/Torch native extension.
- No requirement that one backend wheel depend on the other.
- No framework-neutral Scene, tensor, allocator, stream, or AD abstraction.
- No automatic Torch/Dr.Jit tensor conversion.
- No public default backend at `import rayd`.
- No compatibility alias for removed private source paths.
- No numerical-policy alignment.
- No compile-flag change.
- No PTX pipeline merge or launch fallback.
- No CUDA performance refactor disguised as file movement.
- No forced feature parity: Torch-only and Dr.Jit-only concepts remain valid.
- No target file-count gate.

---

## 10. Open decisions before execution

Only three decisions should remain open:

1. **Python implementation install path.** Accept the proposed private
   `rayd/_impl` PEP 420 subtree with unsuffixed Torch files and adjacent
   `*_jit.py` Dr.Jit files, or require build-time
   materialization into `rayd/drjit` and `rayd/torch`. The former keeps source
   identity simple; the latter preserves internal module-qualified names but
   makes source path differ from installed path. Recommendation:
   `rayd/_impl`.
2. **Direct Python submodule compatibility.** Inventory downstream imports and
   retain thin public modules only for paths that are actually documented or
   used. Recommendation: guarantee the two root packages and documented
   `path_exchange`; treat current `_multi`, `_stable`, `_legacy`, `types`,
   `autograd`, `mesh`, and `scene` paths as internal unless evidence says
   otherwise.
3. **Installed Dr.Jit C++ headers.** Keep current include spellings during this
   layout migration, as proposed, or authorize a separate hard C++ namespace
   cleanup. Recommendation: keep them; source layout does not justify an API
   break.

---

## 11. Completion criteria

The migration is complete when:

- `backends/` is absent;
- root `drjit/` and `torch/` are thin build/distribution frontends;
- real implementation code is discoverable by concept under root `src/`,
  `include/`, and `python/rayd/_impl/`;
- Dr.Jit/Torch variants are adjacent and explicitly named;
- shared numerical code has one owner;
- the hand-written core source count is materially reduced without a file-size
  ceiling or arbitrary target;
- every retained small file has a stated ABI, compiler, numeric, PTX, lazy-load,
  generated, or ownership reason;
- public Python APIs, independent packaging, source integration, CUDA compile
  profiles, committed PTX identity, and numerical behavior satisfy their
  existing contracts;
- full Dr.Jit, Torch, parity, packaging, coexistence, governance, native, and
  cold-create suites pass from clean builds.
