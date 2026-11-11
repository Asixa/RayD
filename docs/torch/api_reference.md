# RayD Torch API Reference

Use the RayD Torch backend as:

```python
import rayd.torch as rt
```

## Core Types

- `rt.Mesh(vertices, faces, ...)`: CUDA mesh input. `vertices` must be contiguous `torch.float32` with shape `(N, 3)`, and `faces` must be contiguous `torch.int32` with shape `(M, 3)`.
- `rt.Scene(*, trace_backend="auto", edge_bvh_backend="auto", devices=None, options=None)`: Native CUDA/OptiX
  scene. Build with `add_mesh()` and `build()`. Every constructor argument is keyword-only. `trace_backend` takes
  `rt.TraceBackend` (`AUTO`, `OPTIX`, `CUDA`) and `edge_bvh_backend` takes `rt.EdgeBVHBackend` (`AUTO`, `OPTIX`,
  `CUDA`); both are string enums, so the plain strings `"auto"`, `"optix"`, and `"cuda"` are accepted too. Matching
  is exact, and any other value raises `ValueError`. `rt.MixedScene` takes the same two keyword-only selectors.
- `rt.Ray(o, d, tmax=None)`: Ray batch with CUDA `torch.float32` origin and direction tensors of shape `(N, 3)`.
- `rt.Camera(width, height, fov_x)`: Torch Python camera helper for primary ray generation. Exposes the `aspect` property plus `sample_to_world(sample, depth=1.0)`, `world_to_sample(point)`, and `sample_ray(sample)`.

## Result Records

Every result record (`Intersection`, `NearestPointEdge`, `NearestRayEdge`, `NearestEdgesTopK`, the visibility
records, `ReflectionChain`, `ReflEpc`, `ReflEpcField`, `AccumResult`, `WedgeEvents`, `DfrAccum`, `DfrCoherentAccum`,
`DfrPaths`, `SdfIntersection`, `SceneGeometry`, `SurfelIntersection`, `SurfelComposite`) shares one helper set:

- a compact `repr` that shows each tensor field as `dtype[shape] device` and each scalar by value, and never reads
  tensor data back to the host. Lazy results (`Intersection` from a reduced or AD query, `ReflectionChain`) show only
  the fields already computed;
- `to(device)`, which returns a new record with every tensor moved;
- `detach()`, which returns a new record with every tensor detached.

`to()` and `detach()` on a lazy result compute its remaining fields first and return the plain record.

Every `Scene.intersect()` and `MixedScene.intersect()` result is an `Intersection` instance, including the lazy
results of reduced (`flags=RayFlags.Empty`) and AD queries, and each provides `instance_id`.

Records with per-row validity expose a boolean tensor `valid` and a method `is_valid()` that returns it:
`Intersection`, `NearestEdgesTopK`, `ReflectionChain`, `ReflEpc`, `ReflEpcField`, `DfrPaths`, `SdfIntersection`,
`SurfelIntersection`, and `SurfelComposite`. `SegmentChainVisibility.all_visible` and
`AxialEdgeVisibility.any_visible` are query results, not validity, and keep their names.

Complex fields stored as `<name>_re`/`<name>_im` tensor pairs also expose a complex property `<name>`:
`ReflEpcField.field`, `AccumResult.reflection_field_{x,y,z}`, `DfrAccum.field_{x,y,z}`, `DfrPaths.field_{x,y,z}`, and
`DfrCoherentAccum.{direct,multi}_field_{x,y,z}`.

`RayFlags` has the members `Empty` (t only), `Geometric`, `ShadingN`, `UV`, and `All`.

## Introspection

- `rt.backend_capabilities()`: backward-compatible flat capability mapping.
- `rt.api_manifest()`: public API surface and operation-level capability
  manifest. The `derivatives` tree is indexed by operation, public variant, and
  input domain; each domain reports `vjp` and `jvp` as `supported`,
  `unsupported`, or `not_applicable`.

The `rayd.torch.path_exchange` submodule provides the backend-neutral path
record exchange format.

## Scene Lifecycle

- `Scene.add_mesh(mesh, dynamic=False)` returns the new mesh id.
- `Scene.add_instance(geometry_id, transform, dynamic=True)` returns a new
  instance id. `geometry_id` must name an existing static owner mesh; this
  first slice intentionally rejects dynamic owners, instance-of-instance
  nesting, and every `Scene(devices=...)` orchestrator, including a one-device
  chunked scene.
- `Scene.build()` builds the acceleration structures; `Scene.is_ready()` reports build state.
- `Scene.num_meshes` / `Scene.num_geometries` / `Scene.version`: instance
  count, distinct GAS-owning geometry count, and scene revision.
- `Scene.update_mesh_vertices(mesh_id, positions)` then `Scene.sync()` refits without a full rebuild; `Scene.has_pending_updates()` reports outstanding edits.
- `Scene.set_instance_transform(instance_id, transform)` then `Scene.sync()`
  updates the OptiX IAS placement without rebuilding the shared GAS. The CUDA
  trace fallback has no IAS and instead refreshes the instance's world-space
  triangle and edge caches plus its backend-specific acceleration data.

## Geometry

- `Scene.intersect(ray, active=None, flags=RayFlags.All)` returns `Intersection`.
- `Intersection.instance_id` is the read-only semantic alias of `shape_id`.
  Every owner mesh and instance has an independent id and an independent
  scene-global primitive range. Face ranges are concatenated in mesh-id order;
  `global_geometry()` exposes the corresponding `shape_id`, `local_prim_id`,
  and `global_prim_id` tensors.
- `Scene.nearest_edge(query, active=None)` returns `NearestPointEdge` for a point tensor and `NearestRayEdge` for a
  `Ray`. An inactive row returns the miss sentinels (distance `inf`, every id `-1`, other fields `0`), as in
  `nearest_edges`.
- `Scene.nearest_edges(point, k, active=None)` returns `NearestEdgesTopK` for `1 <= k <= 16`.
- `Scene.global_geometry()` returns `SceneGeometry`.
- `Scene.edge_mask()` / `Scene.set_edge_mask(mask)` filter the secondary-edge BVH in scene-global edge index space.

## Visibility

- `Scene.visible(start, end, active=None)` returns a `torch.bool` visibility tensor.
- `Scene.visible_pair(start, end_a, end_b, ignore_prim_ids=None, active=None)`,
  `Scene.visible_edge(source, edge_position, edge_direction, edge_t_min, edge_t_max, sample_fractions, active)`, and
  `Scene.visible_chain(points, chain_length, ignore_prim_per_segment=None, active=None)`: segment-pair, edge, and
  chain visibility helpers.

## Multipath

- `Scene.trace_reflections(ray, max_bounces, active=None)` returns `ReflectionChain`; the forward path uses a RayD-source-ported single OptiX launch with the bounce loop inside raygen.
- `Scene.trace_refl_epc(source, receiver, max_bounces, options=None, active=None)` returns forward-only `ReflEpc`
  path geometry configured by `ReflEpcOptions`. `source` is either a `(N, 3)` source-position tensor or a `Ray`,
  dispatched on type:
  - Source positions reflect each source across the planes in `options`, which must then carry
    `expected_prim_ids`, `direct_plane_points`, `direct_plane_normals`, and the three surface-group tables.
  - A `Ray` gives Dr.Jit's semantics: the trace follows the ray from its origin, `tmax` bounds the first segment, and
    each hit must match the slot's `expected_prim_ids` entry when one is given (`-1` leaves a slot free). The direct
    planes must stay `None`, the surface-group tables are optional, and `options=None` traces without constraints. A
    row whose origin or direction is not finite, whose direction is zero, or whose finite `tmax` is not positive is
    inactive, as in Dr.Jit.
- `Scene.trace_refl_epc_field(source, receiver, max_bounces, options=None, active=None)` returns `ReflEpcField`,
  whose `field_re`/`field_im` are the x component of the reflected field. The forward path uses RayD-source-ported
  reflection EPC plus EPC field kernels. With source positions the path is traced from each source toward its
  receiver and no further; with a `Ray` it is traced along the ray up to its `tmax`, as in Dr.Jit. `options` is an
  `rt.ReflEpcFieldOptions(slot_eta_r=None, slot_mu_r=None, slot_sigma=None, slot_gain=None, tx_polarization=None,
  omega=2*pi*299792458, wavelength=1.0)`. The per-slot material tensors have shape `(N, max_bounces)` and
  `tx_polarization` has shape `(1, 3)` or `(N, 3)`. A tensor left at `None` keeps the fixed material of earlier
  releases (`eta_r=4`, `mu_r=1`, `sigma=0`, `gain=1`, x-polarized source), so `options=None` returns bit-for-bit the
  earlier result. This Torch method is forward-only: it rejects reverse-mode or forward-mode AD on the geometry, the
  source or ray, the receiver, and every option tensor, because its native derivatives do not cover the Fresnel,
  polarization, and multi-bounce computation of the primal.
- `Scene.accum_refl(ray, tx_position, grid, material, max_bounces, options=None, active=None,
  tx_polarization=None)` returns forward-only `AccumResult`; `AccumGrid`, `ReflMaterial`, and `AccumOptions` mirror
  the native accumulation contract. `AccumOptions.accumulation_strategy` takes `rt.AccumStrategy`: `AUTO` (stage and
  reduce when the batch is large enough for the grid, otherwise atomics), `ATOMIC`, or `STAGED`.
- `Scene.trace_dfr_paths(tx_positions, rx_positions, states, material=None, options=None, *, active)` returns
  `DfrPaths`. `active` is required (ADR-0031). `rt.DfrPathOptions(wavelength=1.0, max_paths=None, layout=None)`:
  `max_paths=None` exports every state, and `layout=None` selects `DfrPathLayout.Compact` on one device and
  transmitter-aligned `SourceLane` rows on a multi-device scene. Torch rejects tracked or forward-dual endpoint,
  state, material, and scene-geometry inputs because this exporter is forward-only.
- `Scene.accum_dfr_direct(states, grid, material=None, options=None, active=None)` returns `DfrAccum`.
- `Scene.accum_dfr(initial_states, recursive_states, grid, material=None, options=None, active=None,
  recursive_active=None)` returns `DfrAccum` for order-2/order-3 chain accumulation. Both state sets and the grid are
  required.
- Both take `rt.DfrOptions(wavelength=1.0, seed=0, max_order=1, direct_samples=0, keller_samples=0,
  suffix_samples=0, lane_offset=0, lane_count=-1)`. `max_order` is read only by `accum_dfr`; it defaults to `1`, as
  in Dr.Jit, so chain accumulation must ask for `max_order=2` or `3`. `lane_offset`/`lane_count` select the
  Monte-Carlo lane window.
- `Scene.accum_dfr_coherent_direct(states, grid, material=None, options=None, active=None)` returns
  `DfrCoherentAccum`, configured by `rt.DfrCoherentOptions(wavelength=1.0, select_diffraction_point=True,
  prefilter_visibility=True, lane_offset=0, lane_count=-1)`. Replicated scenes shard the deterministic
  `(state, grid-cell)` lane space and sum partial grids on the master. Torch rejects tracked or forward-dual state,
  material, and scene-geometry inputs because this variant is forward-only.
- A material of `None` selects the default per-face diffraction material.

### Options differences from Dr.Jit

The Torch options records use the Dr.Jit field name for every shared concept. They omit the Dr.Jit fields that the
Torch kernels cannot honour:

- `DfrOptions`: no `k`, `samples`, `strategy_mask`, `sample_sequence`, `receiver_model`, `collect_edge_use`, or
  `collect_debug_counts`. Torch chooses the strategies from the non-zero `*_samples` counts, always uses the hash
  sample sequence and the matched isotropic receiver, and always returns its counters. For that reason the Torch
  backend has no `DfrStrategy`, `DfrSampleSequence`, or `DfrReceiverModel` enum.
- `DfrCoherentOptions`: no `k`, `max_order`, `receiver_model`, `collect_debug_counts`, `omega`, `tx_pol_*`, or
  `higher_*` fields; Torch runs first-order coherent accumulation with an x-polarized source.
- `DfrPathOptions`: no `k`, `max_order`, `max_rx`, `strategy_mask`, or `receiver_model`. `max_paths` defaults to
  `None` (every state) in both backends; `layout` is Torch-only.
- `AccumOptions`: the shared fields `wavelength`, `solid_angle_per_ray`, `collect_wedges`, `collect_wedge_prefixes`,
  `wedge_capacity`, and `wedge_sample_stride` have the same names in both backends. Torch-only:
  `accumulation_strategy`, `compact_min_samples`, `staged_min_samples_per_cell`, `procedural_sample_count`, and
  `include_los`. Dr.Jit-only: `k`, `cell_area`, `seed`, `rr_depth`, `rr_prob`, and `stop_threshold`.
- `ReflEpcOptions`: `expected_prim_ids`, `surface_group_id`, `surface_group_size`, `surface_group_members`,
  `visibility_ignore_mode`, and `plane_tolerance` are shared. Torch-only: `direct_plane_points` and
  `direct_plane_normals`, which a source-position trace requires and a `Ray` trace rejects. Dr.Jit-only:
  `surface_max_group_size` and `final_ignore_group_ids`.
- `ReflEpcFieldOptions`: Torch has `slot_eta_r`, `slot_mu_r`, `slot_sigma`, `slot_gain`, `tx_polarization`, `omega`,
  and `wavelength`, with Dr.Jit's names and default `omega` and `wavelength`; its slot tensors are `(N,
  max_bounces)` where Dr.Jit takes flat `N * max_bounces` arrays. A material tensor left at `None` selects the earlier
  fixed material, where Dr.Jit requires every slot array. The Torch record is not a `ReflEpcOptions` and omits:
  - the `ReflEpcOptions` fields Dr.Jit inherits (`expected_prim_ids`, the surface-group tables,
    `surface_max_group_size`, `visibility_ignore_mode`, `plane_tolerance`, `final_ignore_group_ids`): the Torch field
    traces an unconstrained path; constrained paths belong to `trace_refl_epc`.
  - `slot_plane_point` and `slot_plane_normal`: Dr.Jit's position overload reflects across these planes and its `Ray`
    overload evaluates the field with these normals; the Torch field takes both the path and the normals from the
    triangles it hits, in both overloads.
  - `return_geom`, `return_endpoints`, `return_hit_points`, `return_normals`, `return_resolved_prim_ids`, and
    `return_surface_group_ids`: `ReflEpcField` has no geometry or endpoint outputs to switch on.
- Torch has no reflection export mode, so it has no `ReflectionExportMode` enum.

### Known naming mismatches

These names still differ between the backends and are left for a later round:

- Torch `ReflEpcOptions.direct_plane_points`/`direct_plane_normals` (per-path `[N, max_bounces, 3]` tensors)
  correspond to Dr.Jit `ReflEpcFieldOptions.slot_plane_point`/`slot_plane_normal`, which live on the field options
  record, not on `ReflEpcOptions`, and use a singular name. Renaming either side changes a native record field (the
  Torch tensors are `include/rayd/reflection.h` request fields).
- Dr.Jit `ReflEpcOptions.surface_max_group_size` is an integer bound with no Torch field; Torch derives the bound
  from the padded `surface_group_members` length divided by the group count. It is listed above as Dr.Jit-only and
  keeps its name.

## SDF

- `rt.SdfGrid(values, position, rotation, scale)` with `values` of shape `(Nx, Ny, Nz)`.
- `SdfGrid.intersect(ray, options=None, active=None)` returns `SdfIntersection`; `SdfGrid.visible(start, end,
  options=None, active=None)` and `SdfGrid.trace_reflections(ray, max_bounces, options=None, active=None)` follow the
  same order, as does `SdfGridBatch.intersect(ray, options=None, active=None)`. `options` is an
  `rt.SdfTraceOptions(max_steps=64, relaxation=0.9, eps_hit=None)`, the record `MixedScene.add_sdf()` takes.
- `rt.sdf_intersect(grid, origins, directions, *, tmax=inf, max_steps=64, relaxation=0.9, eps_hit=None)` is the
  low-level Torch-only entry point.

## Instance Transforms

`Scene.set_instance_transform(instance_id, transform)` (Torch) and `Scene.set_mesh_transform(mesh_id, mat,
set_left=True)` (Dr.Jit) are deliberately different operations. The Torch call places an instance of a shared static
owner mesh and refreshes only its IAS entry; the Dr.Jit call rewrites a mesh's own left or right world transform.

## Offload

Under `MultiDeviceOptions(offload=...)`, the per-ray queries (`intersect`, `nearest_edge`, `nearest_edges`,
`visible`, `visible_pair`, `visible_edge`, `visible_chain`, `trace_reflections`, and `trace_refl_epc_field`) stream
their chunks to the offload hook and return `None`. Their return annotations are `X | None`.

These APIs use native CUDA/OptiX source ports for the reflection and diffraction
multipath execution paths. Current completion risk is performance parity, not
placeholder kernel coverage.

The Torch EPC path wrapper intentionally reflects the fields available from its
dispatcher operations: it accepts source positions or a `Ray` plus `receiver`
tensors and returns `valid`, `path_length`, resolved IDs, surface-group IDs, hit
points, and normals. Dr.Jit's same-named method accepts only a `Ray` and exposes
additional diagnostic fields with backend-native names. The two APIs share the
operation vocabulary but are not record-shape aliases. The first parameter of
both Torch EPC methods is named `source` whatever its type; Dr.Jit names it `ray`
in its `Ray` overloads and `source` in its position overloads.

## Tensor ABI

All native geometry and multipath inputs are CUDA tensors. Continuous vector inputs use contiguous `torch.float32`; topology and primitive ids use `torch.int32`; masks and validity outputs use `torch.bool`. Native tapes are Torch-owned tensors and remain on the same CUDA device as the inputs.

## AD Contract

AD support is operation- and variant-specific. Query
`rt.api_manifest()["derivatives"]` instead of interpreting the flat
`forward_ad` or `reverse_ad` capability as universal support. In particular,
Torch reflection accumulation, diffraction path export, coherent diffraction
accumulation, and multi-bounce EPC path-discovery derivatives are currently
forward-only or unsupported as recorded in the manifest. Torch EPC-field
VJP/JVP are also unsupported and fail loudly because the available native
derivatives do not match the physical primal. `not_applicable` is reserved for
operations such as boolean visibility whose result has no derivative semantics.

For supported variants, discrete choices such as primitive id, edge id,
visibility, and fixed path sequence are non-differentiable and held fixed from
the forward pass. Continuous outputs are recomputed from the fixed winner and
live geometry tensors during AD.

Instance transforms participate in fixed-winner intersection and nearest-edge
VJP/JVP.
Topology and GAS storage remain shared with the static owner; the per-instance
world-space view used by scene-global identifiers and AD is refreshed when the
transform changes.

The Torch backend does not import or depend on Dr.Jit in the `rayd.torch` package path.

See [`torch_gap_analysis.md`](torch_gap_analysis.md) for the acceptance status
recorded on 2026-06-12.
