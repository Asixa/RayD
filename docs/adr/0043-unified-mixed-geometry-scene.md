# ADR-0043: Unified mixed-geometry scene

- Status: Accepted
- Date: 2026-07-29
- Decision ID: `unified-mixed-geometry-scene`
- Scope: single-device mixed mesh, bounded-SDF, and surfel ray queries in the Dr.Jit and Torch backends

## Context

ADR-0042 made dense SDFs and surfels independently available in both backends,
but left composition to the caller. Applications need one scene that can hold
meshes, SDFs, and surfels simultaneously, select their closest hit, and preserve
geometry derivatives without imposing a material or renderer framework.

Putting every geometry type into the triangle OptiX pipeline would couple the
SDF custom-intersection program and surfel candidate representation to the
mesh and diffraction pipelines. It would also alter existing mesh-only launch,
SBT, and pipeline-creation behavior. The existing geometry owners already have
specialized resident accelerators and differentiable exact-hit re-evaluation.

## Decision

Both backends expose `MixedScene`. It accepts any number of meshes, dense SDF
grids, and surfel clouds and owns these operations:

| Geometry | Closest hit | LOS | Reflection | Transmission | Diffraction |
| --- | --- | --- | --- | --- | --- |
| mesh | yes | yes | yes | opaque | no mixed-scene path |
| dense SDF | yes | yes | yes | ignored | no |
| surfel | yes | yes | yes | alpha | no |

The public query is unified while acceleration remains specialized. Each
non-empty acceleration owner executes on the caller's current CUDA stream.
Closest-hit candidates are merged with resident device operations; there is no
host count read, stream synchronization, CPU fallback, or detached geometry
fallback. This is intentionally not a single fused OptiX launch.

Torch additionally accepts a caller-owned packed `SdfGridBatch`: contiguous
`[G,Nx,Ny,Nz]` values and row-major position, rotation, and scale tensors. It is
one specialized SDF owner, so an untracked closest-hit query submits one CUDA
launch over its `G*N` grid-ray lanes. Packing is explicit because caching copies
or uploading temporary host pointer tables would break caller-owned tensor
lifetime and current-stream guarantees. A batch requires `G >= 2`; a single
`SdfGrid` retains its original launch unchanged.

Tracked packed tensors or tracked rays use the established per-grid frozen-tape
VJP/JVP operations. This conditional fallback preserves winner-fixed AD without
adding a second derivative implementation. Closest-hit and reflection inherit
the packed primal path. LOS retains each grid's individual bias and query,
transmission continues to ignore SDFs, and diffraction remains absent.

A mesh-only `MixedScene` delegates directly to the existing `Scene` operation.
It submits no additional CUDA or OptiX launch and preserves the native result,
flags, numerical order, and AD implementation.

### Placement, ordering, and identifiers

Every SDF's `position`, scalar-first quaternion `rotation`, and full side-length
`scale` define an oriented bounding box. Sphere tracing is clipped to the ray's
box interval and `tmax`; a ray that misses the box does no SDF march.

Exact-depth ties use this stable priority: mesh, SDF insertion order, surfel
scene insertion order, then the surfel owner's ID order. Shape IDs are laid out
as mesh shapes, SDF grids, then surfel scenes. Global primitive IDs are laid out
as mesh faces, SDF grids, then surfels. Selection and identifiers are discrete.

### Differentiation

The winning primitive chosen by the primal query is frozen for differentiation.
Reverse-mode gradients and forward-mode tangents flow through the selected
continuous mesh, SDF, or surfel exact-hit fields. Non-winning geometry receives
zero contribution from that lane. SDF derivatives retain ADR-0037's frozen-cell
implicit-function rule; surfels retain ADR-0042's analytic re-evaluation.
Surfel opacity remains differentiable through transmission. Topology, visibility,
candidate ordering, IDs, and the winner mask are not differentiated.

Transmission preserves the closed effects matrix of ADR-0042: mesh hits are
opaque, SDF grids are not queried and have no effect, and surfel alpha
transmittance is multiplied in scene insertion order.

Reflection calls the same mixed closest-hit operation at every bounce, records
the unified global primitive ID, and uses the selected geometry's existing ray
bias rule. SDF and surfel reflection therefore retain both VJP and JVP support.

### Diffraction boundary

`MixedScene` has no diffraction, edge-query, or diffraction-accumulation method.
SDF and surfel geometry can neither generate a diffraction state nor block a
mixed diffraction path because no such path exists. The existing triangle
`Scene` remains the only diffraction owner. This explicit type boundary avoids
silently applying SDF or surfel geometry to diffraction.

### Performance contract

The benchmark is `benchmarks/benchmark_mixed_geometry.py`. It compares the
unified closest-hit query with the equivalent manually submitted resident mesh,
SDF, and surfel queries. The accepted overhead is at most 10 percent at the
median or 0.1 ms absolute. The mesh-only forwarding path must be within 5
percent or 0.01 ms absolute of `Scene.intersect`. Tests include batches through
1,048,576 rays. Measured evidence is committed under
`benchmarks/baselines/mixed_geometry_20260729.json`.

These gates bound composition overhead, not the inherent cost of tracing more
geometry. Adding an independent SDF or surfel owner performs that owner's
candidate query. A packed Torch SDF batch is deliberately one owner and performs
one primal query launch; its tracked AD fallback remains per-grid.

## API and contract changes

- `contracts/public_api.json` adds the cross-backend `mixed_scene` capability.
- `contracts/operations.json` records the operation matrix, tie order, ID
  namespace, oriented SDF bounds, execution rule, and fixed-winner AD contract.
- Torch exports `MixedScene` and immutable `SdfTraceOptions`.
- Torch exports `SdfGridBatch` and `MixedScene.add_sdf_batch()` for explicitly
  packed, shape-compatible dense grids.
- Dr.Jit installs `rayd/jit/mixed_scene.h` and binds the same scene operations.
- The operation is single-device. It is not silently routed through Torch's
  replicated multi-device `Scene` layer.

## Consequences

- Applications can trace a simultaneous mesh/SDF/surfel scene through one API.
- Mesh-only behavior and every existing diffraction pipeline remain unchanged.
- Independent owners submit the same specialized candidate work as equivalent
  manual composition; a packed Torch SDF batch submits one primal SDF query.
- A caller needing diffraction continues to use the triangle `Scene` and must
  not infer that mixed geometry participates.

## Superseded decisions

This record supersedes ADR-0042's standalone-only composition conclusion and
its rejection of a RayD-owned mixed query. ADR-0042 continues to own standalone
SDF/surfel numerics and the closed effects matrix. ADR-0037 continues to own the
SDF representation, march, derivative, sentinel, and failure contracts.

## Rejected alternatives

- Adding SDF or surfel branches to diffraction is outside the required effects
  matrix and would weaken the explicit type boundary.
- Replacing all specialized accelerators with one new generic kernel would
  regress mature mesh and surfel traversal and couple unrelated pipelines.
- Host-side candidate comparison is rejected because it would synchronize the
  caller's stream and detach the winner from both AD frontends.
- Treating an SDF's bounds as axis-aligned is rejected; its existing rotation is
  part of both broad-phase placement and differentiation.
