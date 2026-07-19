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
versioned typed C++ surface in `rayd/torch/integration_v2.h`; they do not load a
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

The accepted diffraction surface also places the complete fixed-winner
pure-wedge field primal/backward/JVP family behind the typed integration header.
It preserves optional winner vertices, three separate native entry launches,
current-stream execution, output schemas, and the family-local
`--use_fast_math` contract required for order-1 exporter parity. Monte Carlo
Sionna accumulation, coupled RD/DD operations, and BDPT estimator policy remain
downstream-owned. See
[`docs/adr/0025-diffraction-family-ownership.md`](../../docs/adr/0025-diffraction-family-ownership.md).

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
