# RayD Torch

RayD Torch is a Torch-native CUDA/OptiX package for RayD geometry primitives and
RayD-style multipath/diffraction kernels.

```python
import rayd.torch as rt
```

Install the `rayd-torch` distribution. It owns only `rayd/torch/**` and can
coexist with the independently installed `rayd-drjit` backend.

## Tensor ABI

RayD Torch APIs accept CUDA `torch.float32` tensors for vector data and CUDA `torch.int32` tensors for index data. Vector tensors are row-major `(N, 3)` unless otherwise documented, masks are `torch.bool`, and tensors should be contiguous. Outputs and AD tapes are Torch-owned tensors.

## Gradient Contract

Intersection, edge, reflection, EPC, and diffraction operators use a fixed-winner gradient contract where explicit native kernels exist. The discrete primitive, edge, visibility, or path decision selected in the forward pass is treated as non-differentiable; VJP and JVP propagate through the continuous values recomputed from the saved winner and live Torch tensors.

## Autograd

The native operators support Torch reverse-mode VJP and forward-mode JVP for the supported continuous inputs where explicit kernels have been implemented. CUDA work is launched on the current Torch CUDA stream.

## Native Source Integration

Native downstream projects built in the same CMake/LibTorch graph use the
versioned typed C++ surface in `rayd/torch/integration.h`; they do not load a
second RayD Python extension or use a dynamic symbol registry. Solver-neutral
RF device math is exposed through `rayd/shared/rf/*.cuh`. Torch-specific field
AD helpers remain under `rayd/torch/rf/` because they use Torch complex types.

The accepted transmission surface consists of complete primal/backward/JVP
families for resident CSR layer-stack evaluation and complete-row Jones field
transport. These operations preserve precise-math compilation, row fusion,
atomic layer-gradient order, and the no-persistent-tape contract. Inputs are
validated before launch, work runs on the caller's current Torch CUDA stream,
and invalid shape/dtype/device/ABI state or CUDA failure raises immediately;
there is no CPU, Torch-expression, finite-difference, or legacy-dispatch
fallback.

RayD owns the numerical primitives and typed native operations, not a
downstream application's material encoding, topology selection, solver
estimator policy, RNG/MIS, accumulation, metadata, or result schema. A newly
merged transmission implementation may remain a dormant candidate until the
consumer pins it, switches all callers, proves parity, and deletes its local
implementation. See
[`docs/adr/0002-shared-rf-transmission-ownership.md`](../../docs/adr/0002-shared-rf-transmission-ownership.md).

The accepted diffraction surface will place the complete fixed-winner
pure-wedge field primal/backward/JVP family behind the typed integration header
once the dormant RayD candidate is implemented and direct-tested. Channel
remains the production numerical owner until it pins, validates, activates, and
deletes its local CUDA implementation. The future typed family must preserve
optional winner vertices, three separate native entry launches, current-stream
execution, output schemas, and the family-local `--use_fast_math` contract
required for order-1 exporter parity. Monte Carlo Sionna accumulation, coupled
RD/DD operations, and BDPT estimator policy remain downstream-owned. See
[`docs/adr/0025-diffraction-family-ownership.md`](../../docs/adr/0025-diffraction-family-ownership.md).

The accepted generic-scattering surface consists of exactly seventeen typed
operations in six complete families: resident table evaluation AD, resident
table sampling/PDF, single-bounce ensemble, phase-screen patch integral,
chain ensemble, and chain realization. RayD evaluates caller-owned resident
CUDA tensors but does not own table construction, cache/version policy,
phase-screen seed/lifecycle, topology, estimator, RNG/MIS, accumulation, or
result policy. A high-level BSDF/material framework remains out of scope;
solver-neutral RF scattering primitives are in scope.

The migration preserves the existing AD asymmetry: chain-ensemble continuous
geometry supports JVP but reverse-mode requests fail loudly, while
chain-realization supports its existing continuous-geometry VJP and JVP.
Complete row fusion, launch count, recomputation/tape lifetime, backward
atomics, output schemas, and the source-TU compile split are frozen: table
primal/sample/PDF uses default CUDA flags, while the audited table-AD,
ensemble, patch, and chain lockstep TUs retain `--fmad=false`. A merged RayD
implementation remains dormant until Channel pins it, switches a complete
family with parity evidence, and deletes the local implementation. See
[`docs/adr/0026-generic-scattering-runtime-ownership.md`](../../docs/adr/0026-generic-scattering-runtime-ownership.md).

The stable source-level boundary is named `rayd/torch/integration.h`, its exact
identity is `rayd.torch.integration`, and its numeric API version remains `2`.
No `integration_v2` forwarding header, CMake target alias, or alternate
identity is supported. See
[`docs/adr/0028-stable-typed-integration-naming.md`](../../docs/adr/0028-stable-typed-integration-naming.md).

The typed boundary also carries a dormant axial-edge visibility candidate for
device-resident diffraction state selection. It consumes broadcast TX and
contiguous AoS edge tensors, evaluates the four exact binary32 fractions in one
separate OptiX launch on the caller's current CUDA stream, and returns a
resident boolean state mask. Its traversal inherits the legacy OptiX compile
policy while inline PTX locks
only point construction to non-FTZ, non-FMA round-to-nearest operations. The
legacy visibility dispatcher is unchanged and cannot select this entry. The
operation adds no synchronization beyond the existing common launch-parameter
staging path; removing that staging path's host event wait is a separate
optimization.
See [`docs/adr/0029-typed-axial-edge-visibility.md`](../../docs/adr/0029-typed-axial-edge-visibility.md).

RayD Torch carries all seventeen typed operations. Channel has activated the
Phase 10A table, sampling, single-bounce ensemble, and patch-integral entries.
The six Phase 10B chain entries are source-linked into the native core and
direct-tested but remain dormant: no RayD Python binding dispatches them, and
Channel remains their production numerical owner until its atomic
pin/switch/delete commit.

## Current Status

RayD Torch now builds separate native scene, edge, reflection, and diffraction
Torch extension bindings. The native build includes OptiX PTX pipelines for
scene intersection, edge queries, reflection tracing/EPC/visibility/
accumulation, and diffraction path/accumulation/coherent direct execution.

Current opt-in RayD parity tests cover forward cases for scene intersection,
multi-mesh global ids, nearest-edge, visibility, reflection tracing,
diffraction paths, direct/Keller/suffix diffraction accumulation, order-2 and
order-3 diffraction chains, and coherent direct accumulation. Torch VJP/JVP
coverage exists for geometry, edge, reflection trace, EPC, and diffraction
accumulation under the fixed-winner contract.

On the recorded same-script benchmark shape (grid 64, 4,096 queries, warm
caches), RayD Torch currently measures faster than RayD for scene build,
intersect, nearest edge, reflection trace, diffraction paths, and direct
diffraction accumulation. Far-from-surface nearest-edge queries use a tiled
exact fallback scan instead of the scene-diagonal OptiX tier. Release-size and
Nsight-counter-backed runs remain the broader performance gate. See
`docs/torch_gap_analysis.md` and `docs/torch_performance.md`.

## Dependencies

RayD Torch depends on PyTorch, CUDA, and OptiX for native execution. The RayD Torch package path has no Dr.Jit dependency.
