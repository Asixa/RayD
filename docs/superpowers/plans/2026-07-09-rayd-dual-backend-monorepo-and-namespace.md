# RayD Dual-Backend Monorepo and Hard-Cut Namespace Plan

Status: Implemented and locally accepted; package publication and coordinated downstream rollout remain release operations

Last reviewed: 2026-07-09

## Executive Decision

Move the current RayDN Torch implementation into the RayD repository and make
RayD a dual-backend CUDA/OptiX project with two explicit public namespaces:

```python
import rayd.drjit as rd
import rayd.torch as rt
```

This is a deliberate hard cut. There is no backward-compatibility layer:

- `import rayd` no longer exposes Dr.Jit classes such as `Scene` and `Mesh`;
- `import raydn` is removed;
- no `raydn` compatibility shim is published;
- no deprecation window is required;
- downstream projects must migrate in the same release transition.

`rayd` is a PEP 420 namespace package, not a default backend. Backend choice is
always explicit.

Do **not** use `rayd.native` as the canonical Torch namespace. Both backends run
native CUDA/OptiX code; the meaningful distinction is the tensor runtime and AD
system. Use `rayd.torch`. Reserve `rayd.native` for a future framework-neutral
C/C++ runtime if one is ever implemented.

## Final Naming

### Python imports

```python
import rayd.drjit as rd
import rayd.torch as rt
```

### Python distributions

- `rayd` (file-free meta-distribution; installs both backends)
- `rayd-drjit`
- `rayd-torch`

The file-free `rayd` meta-distribution pins and installs both backend
distributions at the same version. Installing `rayd-drjit` or `rayd-torch`
directly remains the supported single-backend path and must not install the
other runtime.

### Native extension names

- Dr.Jit: `rayd.drjit._C`
- Torch: `rayd.torch._C`

### Torch dispatcher names

- operators: `torch.ops.rayd_torch.*`
- custom classes: `torch.classes.rayd_torch.*`

The current `_raydn`, `torch.ops.raydn`, and `torch.classes.raydn` names are
renamed as part of the hard cut. There are no aliases or duplicate
registrations.

### C++ namespaces and include roots

- shared runtime-independent code: `rayd::shared`, `<rayd/shared/...>`;
- Dr.Jit backend: existing `rayd` C++ namespace may remain internally;
- Torch backend: migrate `raydn` C++ namespace to `rayd::torch_backend`;
- Torch-private headers: `<rayd/torch/...>`.

The C++ namespace rename may be implemented after the monorepo import, but must
finish before the first dual-backend release so the repository does not expose
two competing project names.

## Why Explicit Subpackages Are Better

The proposed form:

```python
import rayd       # Dr.Jit backend
import rayd.native
```

has two problems:

1. `rayd` and `rayd.native` are not symmetrical, so one backend appears to be
   the default implementation while the other appears secondary.
2. `native` is technically inaccurate because the Dr.Jit backend also owns and
   launches native CUDA/OptiX pipelines.

Without a compatibility requirement, the clean design is symmetric and
explicit:

```python
import rayd.drjit
import rayd.torch
```

It also prevents a library from silently changing behavior according to which
optional runtime happens to be installed.

## Goals

- Preserve the Git history of both repositories in one monorepo.
- Make backend selection explicit in every user program.
- Allow Dr.Jit-only installation without Torch.
- Allow Torch-only installation without Dr.Jit.
- Allow both backends to coexist in one environment and process.
- Keep Scene, allocator, stream, OptiX context, acceleration structure, and AD
  ownership backend-specific.
- Centralize semantic contracts, parity tests, benchmarks, documentation, and
  genuinely runtime-independent device code.
- Remove the RayDN product name from public APIs, source namespaces, tests, and
  documentation before the first combined release.

## Non-Goals

- Do not combine both backends into one native extension.
- Do not require both Torch and Dr.Jit in either backend wheel.
- Do not provide automatic Torch/Dr.Jit tensor conversion.
- Do not allow Scene objects or OptiX handles to switch backends.
- Do not retain `raydn`, `rayd-native`, `torch.ops.raydn`, or `_raydn` aliases.
- Do not provide top-level `rayd.Scene`, `rayd.Mesh`, or other default-backend
  exports.
- Do not force identical feature coverage before the monorepo is released.
- Do not refactor every duplicated CUDA kernel during the repository move.

## Target User Experience

### Dr.Jit

```bash
pip install rayd-drjit
```

```python
import rayd.drjit as rd

mesh = rd.Mesh(...)
scene = rd.Scene()
scene.add_mesh(mesh)
scene.build()
result = scene.intersect(ray)
```

### Torch

```bash
pip install rayd-torch
```

```python
import rayd.torch as rt

mesh = rt.Mesh(...)
scene = rt.Scene()
scene.add_mesh(mesh)
scene.build()
result = scene.intersect(ray)
```

### Both backends

```bash
pip install rayd
```

```python
import rayd.drjit as rdd
import rayd.torch as rdt

scene_d = rdd.Scene()
scene_t = rdt.Scene()
```

The two Scene objects are independent. They do not share GPU buffers, OptiX
handles, streams, or AD graphs.

### Invalid old imports

These imports intentionally fail or provide no backend API:

```python
import raydn          # ModuleNotFoundError
import rayd           # Namespace only; no Scene or Mesh
import rayd.native    # ModuleNotFoundError
```

## Namespace Packaging Model

Use a PEP 420 implicit namespace. No distribution installs
`rayd/__init__.py`.

Each backend wheel owns a disjoint subtree:

| Distribution | Installed files | Dependencies |
| --- | --- | --- |
| `rayd` | none outside distribution metadata | `rayd-drjit`, `rayd-torch` at the identical version |
| `rayd-drjit` | `rayd/drjit/**` | Dr.Jit, nanobind |
| `rayd-torch` | `rayd/torch/**` | Torch |

Required properties:

- `rayd/` has no `__init__.py` in source or wheels.
- `rayd/drjit/__init__.py` imports only the Dr.Jit runtime.
- `rayd/torch/__init__.py` imports only Torch.
- the wheels own no common installed path;
- uninstalling one backend leaves the other fully functional;
- import order does not change device, stream, or module state.

The required `rayd` meta-distribution owns no Python package files. It declares
exact-version dependencies on both backend distributions and is published only
after both backend artifacts are available.

## Target Repository Layout

```text
RayD/
├── pyproject.toml                    # Workspace/build orchestration only
├── CMakeLists.txt                    # Optional top-level convenience project
├── backends/
│   ├── drjit/
│   │   ├── pyproject.toml            # distribution: rayd-drjit
│   │   ├── CMakeLists.txt
│   │   ├── python/rayd/drjit/
│   │   │   ├── __init__.py
│   │   │   └── _C.*
│   │   ├── include/rayd/drjit/
│   │   ├── src/
│   │   └── tests/
│   └── torch/
│       ├── pyproject.toml            # distribution: rayd-torch
│       ├── CMakeLists.txt
│       ├── python/rayd/torch/
│       │   ├── __init__.py
│       │   ├── autograd.py
│       │   ├── camera.py
│       │   ├── mesh.py
│       │   ├── scene.py
│       │   ├── types.py
│       │   └── _C.*
│       ├── include/rayd/torch/
│       ├── src/
│       ├── tests/
│       └── scripts/
├── shared/
│   ├── include/rayd/shared/
│   ├── contracts/
│   └── testdata/
├── tests/
│   ├── parity/
│   ├── packaging/
│   └── coexistence/
├── benchmarks/
│   └── cross_backend/
└── docs/
```

## Backend Boundary

The two backends share semantics, not runtime objects:

```text
Shared test case / semantic contract
             |
             +-- rayd.drjit -> Dr.Jit arrays -> Dr.Jit AD/allocator/stream/OptiX
             |
             +-- rayd.torch -> Torch tensors -> ATen AD/allocator/stream/OptiX
```

Never share these objects across the boundary:

- Scene or Mesh runtime instances;
- OptiX context, module, pipeline, SBT, GAS, IAS, or traversable handles;
- framework-owned CUDA allocations;
- current-stream state or CUDA events;
- autograd graphs or saved tapes;
- backend-specific result containers.

If explicit detached tensor interoperability is added later, it must be a
separate DLPack utility with documented lifetime and stream synchronization. It
must not imply cross-framework AD preservation.

## Shared-Code Policy

### Share immediately

- the byte-identical UTD headers;
- pure CUDA vector, complex, and numerical math;
- constants and fixed-winner derivative formulas;
- sampling and hash utilities;
- semantic schemas, golden inputs, and expected outputs;
- benchmark case generation and reporting.

### Keep backend-specific

- Scene and Mesh host implementations;
- CUDA allocation and ownership;
- current stream selection;
- OptiX context and cache management;
- build/update/synchronization policy;
- TorchBind, dispatcher, nanobind, and Dr.Jit bindings;
- AD orchestration;
- backend-tuned launch parameter layouts;
- packed/AoS/SoA layouts and backend-specific fast paths.

Do not create an abstract host-side Scene base class solely to remove code
duplication. It would either leak framework types into shared code or rebuild
allocator and stream abstractions already supplied by each runtime.

## Capability Model

Both public subpackages expose:

```python
rayd.drjit.backend_capabilities()
rayd.torch.backend_capabilities()
```

Minimum capability keys:

```python
{
    "backend": "drjit" | "torch",
    "intersect": True,
    "nearest_edge_point": True,
    "nearest_edge_ray": True,
    "nearest_edges_topk": bool,
    "visibility": True,
    "visibility_pair": bool,
    "reflection_trace": True,
    "reflection_accumulation": bool,
    "diffraction_direct": True,
    "diffraction_chain": True,
    "surfel": bool,
    "reverse_ad": True,
    "forward_ad": True,
    "torch_compile": bool,
}
```

Unsupported features must not silently fall back to slow Python tensor code.
They are absent or raise `NotImplementedError` with the backend name.

## Implementation Plan

### Task 0: Freeze authoritative baselines

**Repositories:** current RayD and current RayDN worktrees

- [ ] Preserve all current uncommitted RayD work before migration.
- [ ] Remove generated stale extension files from the migration set and add
      ignore rules.
- [ ] Tag the exact RayDN commit used for import.
- [ ] Record both commit ids, branches, remotes, untracked files, toolchain
      versions, CUDA architectures, and GPU model.
- [ ] Save the latest numeric, parity, and performance JSON results.
- [ ] Run the authoritative focused test suites before moving files.

### Task 1: Import RayDN history under `backends/torch`

- [ ] Rewrite a temporary RayDN history with
      `git filter-repo --to-subdirectory-filter backends/torch` or an equivalent
      history-preserving method.
- [ ] Merge it into a dedicated RayD integration branch.
- [ ] Keep the existing RayD root layout unchanged in this task.
- [ ] Keep RayDN's original build and import paths unchanged until its tests pass
      from the new directory.
- [ ] Fix only path-dependent build scripts and test discovery.
- [ ] Verify file history reaches pre-merge RayDN commits.

Acceptance:

- both backends build independently from the monorepo;
- all pre-migration tests still pass;
- no namespace or CUDA refactor is mixed into the history import.

### Task 2: Move the existing RayD implementation to `backends/drjit`

- [ ] Move the existing Dr.Jit CMake, Python, native source, examples, and tests
      under `backends/drjit/` with history-preserving Git moves.
- [ ] Create `backends/drjit/pyproject.toml` for `rayd-drjit`.
- [ ] Install Python files under `rayd/drjit/`.
- [ ] Rename the extension import to `rayd.drjit._C`.
- [ ] Update internal relative imports, examples, and tests.
- [ ] Remove the old root `rayd/__init__.py`.
- [ ] Assert that the wheel contains no `rayd/__init__.py`.

### Task 3: Hard-rename RayDN Python API to `rayd.torch`

- [ ] Move `raydn/*.py` to `backends/torch/python/rayd/torch/`.
- [ ] Rename the distribution from `rayd-native` to `rayd-torch`.
- [ ] Rename the extension module from `_raydn` to
      `rayd.torch._C`.
- [ ] Rename all imports, exception strings, build metadata, tests, examples,
      documentation, and public backend labels.
- [ ] Do not create a `raydn` package or forwarding module.
- [ ] Add metadata tests that fail if public Python source contains `raydn`.
- [ ] Preserve tensor ABI, result classes, fixed-winner semantics, VJP/JVP, and
      `torch.compile` behavior.

### Task 4: Rename Torch dispatcher and custom-class namespaces

- [ ] Change `TORCH_LIBRARY(raydn, ...)` to
      `TORCH_LIBRARY(rayd_torch, ...)`.
- [ ] Change all fragments and implementation registrations consistently.
- [ ] Change `torch.classes.raydn.Scene` to
      `torch.classes.rayd_torch.Scene`.
- [ ] Change every Python `torch.ops.raydn.*` call to
      `torch.ops.rayd_torch.*`.
- [ ] Update fake/meta registrations and registered autograd kernels.
- [ ] Add a subprocess test proving the old dispatcher namespace is absent.
- [ ] Verify extension import registers each operator exactly once.

### Task 5: Rename Torch native include and C++ namespaces

- [ ] Move `include/raydn/**` to `include/rayd/torch/**`.
- [ ] Rename the C++ namespace from `raydn` to `rayd::torch_backend`.
- [ ] Update PTX embedding, include paths, generated headers, CMake dependencies,
      and error messages.
- [ ] Keep source changes mechanical; do not tune kernels in this task.
- [ ] Build incrementally using the persistent native build directory.
- [ ] Run the full Torch-native suite after the rename.

### Task 6: Establish independent packaging

- [ ] Configure the Dr.Jit wheel to install only `rayd/drjit/**`.
- [ ] Configure the Torch wheel to install only `rayd/torch/**`.
- [ ] Ensure neither wheel generates or installs `rayd/__init__.py`.
- [ ] Inspect wheel RECORD files and fail on overlapping installed paths.
- [ ] Verify each backend installs without the other runtime present.
- [ ] Verify uninstalling either wheel leaves the other usable.
- [x] Add a file-free `rayd` meta-distribution that depends on both backends at
      the identical version; neither backend depends on it.

Required packaging scenarios:

1. Dr.Jit only;
2. Torch only;
3. both backends, Dr.Jit installed first;
4. both backends, Torch installed first;
5. uninstall Dr.Jit while Torch remains;
6. uninstall Torch while Dr.Jit remains;
7. upgrade one backend without modifying the other backend's files;
8. install `rayd` and resolve both exact-version backend wheels.

### Task 7: Add namespace and import isolation tests

- [ ] `import rayd.drjit` succeeds without Torch installed.
- [ ] `import rayd.torch` succeeds without Dr.Jit installed.
- [ ] `import rayd` exposes no `Scene`, `Mesh`, or default backend.
- [ ] `import raydn` raises `ModuleNotFoundError`.
- [ ] `import rayd.native` raises `ModuleNotFoundError`.
- [ ] Importing the backends in either order produces the same results.
- [ ] `sys.modules` confirms each backend does not import the other runtime.
- [ ] `dir(rayd)` does not advertise backend-specific classes.

### Task 8: Build shared semantic contracts and parity adapters

- [ ] Define common shape, dtype, layout, id-space, invalid-value, active-mask,
      and fixed-winner contracts.
- [ ] Move common case generation out of backend-specific benchmark scripts.
- [ ] Convert the current opt-in parity tests to backend adapters plus shared
      assertions.
- [ ] Test equivalent fields without requiring identical result class types.
- [ ] Define per-operation numeric tolerances.
- [ ] Add capability manifests and verify them against public methods.

### Task 9: Extract the first shared device code

- [ ] Move the byte-identical UTD headers to
      `shared/include/rayd/shared/utd/`.
- [ ] Add temporary forwarding headers only inside source builds if required;
      do not publish legacy include paths.
- [ ] Update both CMake projects to compile against the shared headers.
- [ ] Add a repository check preventing private duplicated copies.
- [ ] Run all numeric and performance gates.
- [ ] Extract additional math only one subsystem at a time.
- [ ] Do not extract Scene, allocator, stream, or OptiX lifetime code.

### Task 10: Create independent CI and release pipelines

- [ ] Create `drjit`, `torch`, `parity`, `packaging`, and `coexistence` jobs.
- [ ] Give each backend an independent dependency environment and build cache.
- [ ] Use path filters only for PR latency; scheduled and release runs execute
      the full matrix.
- [ ] Publish `rayd-drjit` and `rayd-torch` independently, then publish the
      file-free `rayd` meta-distribution.
- [ ] Require both backend test jobs and packaging isolation before tagging the
      combined project release.
- [ ] Retire the old `rayd` and `rayd-native` distributions according to the
      chosen package-index policy; no compatibility packages are published.

### Task 11: Update downstream repositories in one coordinated cut

- [ ] Find every `import rayd`, `from rayd`, `import raydn`,
      `torch.ops.raydn`, and `torch.classes.raydn` reference.
- [ ] Update Dr.Jit consumers to `rayd.drjit`.
- [ ] Update Torch consumers to `rayd.torch` and `rayd_torch` dispatcher names.
- [ ] Update build-system package names to `rayd-drjit` or `rayd-torch`.
- [ ] Update documentation, notebooks, benchmarks, CI images, and examples.
- [ ] Land downstream changes only after new backend wheels are available in the
      selected release channel.

### Task 12: Rewrite project documentation

- [ ] Present RayD as one project with two explicit runtime backends.
- [ ] Document installation and imports separately for both backends.
- [ ] Publish a capability matrix rather than claiming full feature equality.
- [ ] Document independent Scene/allocator/stream/OptiX ownership.
- [ ] Explain that `rayd.torch` is Torch-native, not framework-neutral.
- [ ] Remove all instructions for `import rayd`, `import raydn`, and
      `rayd.native` as backend entry points.
- [ ] Document the breaking release prominently; do not include shim-based
      migration instructions.

## Verification Matrix

### Dr.Jit backend

- geometry/intersection;
- transforms and dynamic updates;
- nearest edge and Top-K;
- visibility variants;
- reflection/EPC/accumulation;
- diffraction;
- Surfel;
- Dr.Jit forward/reverse AD.

### Torch backend

- tensor contract and import isolation;
- Scene cache and dynamic updates;
- intersection forward/VJP/JVP;
- point/ray nearest edge;
- visibility and reflection;
- EPC and diffraction forward/VJP/JVP;
- dispatcher/custom-class bindings;
- `torch.compile` supported paths;
- no Dr.Jit import.

### Cross-backend parity

- scene intersection;
- multi-mesh global ids;
- nearest edge;
- visibility;
- reflection tracing;
- EPC field;
- diffraction paths;
- direct/Keller/suffix accumulation;
- order-2/order-3 chains;
- coherent direct accumulation.

### Coexistence

- create, query, and destroy both Scene types in one process;
- test both import orders;
- test current device selection independently;
- test current-stream behavior independently;
- verify no Scene or OptiX handle crosses runtimes;
- verify no duplicate Torch dispatcher registration.

### Packaging

- wheel contents are disjoint;
- neither wheel owns `rayd/__init__.py`;
- Torch-only installation has no Dr.Jit dependency;
- Dr.Jit-only installation has no Torch dependency;
- install/uninstall/upgrade order is safe;
- obsolete packages and imports are absent.

### Performance

- re-run maintained static and dynamic comparisons;
- re-run release-size intersection workloads;
- re-run reflection/diffraction workloads;
- report public AD time separately from direct kernel time;
- reject shared-code changes with a repeatable regression greater than 5%
  unless a different workload-specific threshold is recorded;
- require Nsight-backed evidence before broad performance claims.

## Release Strategy

Because compatibility is explicitly out of scope, use one coordinated breaking
release rather than staged aliases.

### Pre-release

- merge histories;
- complete hard renames;
- publish pre-release wheels for `rayd-drjit`, `rayd-torch`, and the `rayd`
  meta-distribution;
- migrate controlled downstream projects;
- run packaging and coexistence tests from clean environments.

### Release

- publish both backend wheels with the same project version, then publish the
  exact-version `rayd` meta-distribution;
- publish new documentation and capability matrix;
- tag the monorepo once both backend artifacts are verified;
- stop publishing old `rayd-native` artifacts;
- do not publish shim releases.

### Post-release

- fix only defects in the new namespaces;
- do not restore old aliases in response to incidental import failures;
- track feature parity separately from namespace migration;
- continue extracting shared code only with numeric and performance evidence.

## Rollback Strategy

The hard cut should still be implemented as separable commits:

1. history import;
2. Dr.Jit directory move;
3. Torch Python rename;
4. dispatcher rename;
5. C++ include/namespace rename;
6. packaging split;
7. shared contracts;
8. shared device code.

If the namespace migration fails before release, roll back to the monorepo with
the old packages internally and delay publication. After the breaking release
is published, fixes target only the new namespaces; old aliases are not restored.

Do not combine packaging ownership changes with CUDA performance refactors. Do
not combine shared-code extraction with dispatcher renaming.

## Estimated Effort

Single-developer, full-time rough estimate:

| Workstream | Estimate |
| --- | ---: |
| Preserve/import both histories | 3-5 days |
| Move Dr.Jit backend and hard-rename Python package | 1-2 weeks |
| Rename Torch Python/dispatcher/C++ namespaces | 1-2 weeks |
| Independent PEP 420 packaging and wheel tests | 1-2 weeks |
| CI, coexistence, and coordinated downstream cut | 1-2 weeks |
| Shared contracts and parity migration | 1-2 weeks |
| First shared device-code extraction | 1-3 weeks |

The hard-cut monorepo and namespace release is approximately 4-7 weeks. Shared
device-code extraction can continue afterward and should not block the first
dual-backend release.

## Completion Criteria

The migration is complete when:

- both repository histories live in the RayD monorepo;
- `rayd-drjit` and `rayd-torch` build and publish independently;
- the parent `rayd` directory is a PEP 420 namespace with no `__init__.py`;
- `rayd.drjit` works without Torch;
- `rayd.torch` works without Dr.Jit;
- `raydn`, `rayd.native`, `_raydn`, `torch.ops.raydn`, and
  `torch.classes.raydn` are absent;
- no top-level `rayd.Scene` or default backend exists;
- wheel contents do not overlap;
- both Scene/runtime implementations coexist safely;
- parity, AD, packaging, performance, and coexistence gates pass;
- all controlled downstream repositories use the new explicit namespaces;
- documentation describes only the new hard-cut API.

## Final Recommendation

With backward compatibility explicitly excluded, use the simplest and cleanest
public design:

```python
import rayd.drjit as rd
import rayd.torch as rt
```

Do not keep `import rayd` as a Dr.Jit default, do not create a `raydn` shim, and
do not use `rayd.native` for the Torch implementation. The symmetric explicit
namespaces remove default-backend ambiguity, allow fully independent wheels,
and leave `native` available for a genuinely framework-neutral backend later.
