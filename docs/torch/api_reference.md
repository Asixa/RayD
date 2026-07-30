# RayD Torch API Reference

Use the RayD Torch backend as:

```python
import rayd.torch as rt
```

## Core Types

- `rt.Mesh(vertices, faces, ...)`: CUDA mesh input. `vertices` must be contiguous `torch.float32` with shape `(N, 3)`, and `faces` must be contiguous `torch.int32` with shape `(M, 3)`.
- `rt.Scene()`: Native CUDA/OptiX scene. Build with `add_mesh()` and `build()`.
- `rt.Ray(o, d, tmax=None)`: Ray batch with CUDA `torch.float32` origin and direction tensors of shape `(N, 3)`.
- `rt.Camera(width, height, fov_x)`: Torch Python camera helper for primary ray generation. Exposes the `aspect` property plus `sample_to_world(sample, depth=1.0)`, `world_to_sample(point)`, and `sample_ray(sample)`.

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
- `Scene.nearest_edge(point)` returns `NearestPointEdge`.
- `Scene.nearest_edge(ray)` returns `NearestRayEdge`.
- `Scene.nearest_edges(point, k, active=None)` returns `NearestEdgesTopK` for `1 <= k <= 16`.
- `Scene.global_geometry()` returns `SceneGlobalGeometry`.
- `Scene.edge_mask()` / `Scene.set_edge_mask(mask)` filter the secondary-edge BVH in scene-global edge index space.

## Visibility

- `Scene.visible(start, end, active=None)` returns a `torch.bool` visibility tensor.
- `Scene.visible_pair(...)`, `Scene.visible_edge(...)`, `Scene.visible_chain(...)`: segment-pair, edge, and chain visibility helpers.

## Multipath

- `Scene.trace_reflections(ray, max_bounces, active=None)` returns `ReflectionChain`; the forward path uses a RayD-source-ported single OptiX launch with the bounce loop inside raygen.
- `Scene.trace_refl_epc(source, receiver, max_bounces, options, active=None)` returns
  forward-only `ReflEpc` path geometry configured by `ReflEpcOptions`.
- `Scene.trace_refl_epc_field(source, receiver, max_bounces, active=None)` returns
  `ReflEpcField`; the forward path uses RayD-source-ported reflection EPC plus
  EPC field kernels. Its simplified default material is fixed to `eta_r=4`,
  `mu_r=1`, `sigma=0`, and `gain=1`, with an x-polarized source, so a valid
  reflected path produces a non-zero field without a second material API.
  This Torch method is forward-only and rejects reverse-mode or forward-mode AD
  inputs until its native derivatives cover the same Fresnel, polarization, and
  multi-bounce computation as the primal.
- `Scene.accumulate_reflections(ray, tx_position, grid, material, max_bounces,
  options=None, active=None, tx_polarization=None)` returns forward-only
  `AccumResult`; `AccumGrid`, `ReflMaterial`, and `AccumOptions` mirror the
  native accumulation contract.
- `Scene.trace_dfr_paths(tx_positions=..., rx_positions=..., states=..., material=..., active=..., max_paths=..., wavelength=..., layout=...)` returns `DfrPaths`. Single-device calls default to `DfrPathLayout.Compact`; multi-device scenes use transmitter-aligned `SourceLane` rows. Torch rejects tracked or forward-dual endpoint, state, material, and scene-geometry inputs because this exporter is forward-only.
- `Scene.accum_dfr_direct(states=..., grid=..., material=..., active=..., wavelength=..., direct_samples=..., keller_samples=..., suffix_samples=..., seed=..., lane_offset=..., lane_count=...)` returns `DfrAccum`.
- `Scene.accum_dfr(initial_states=..., recursive_states=..., grid=..., material=..., active=..., recursive_active=..., wavelength=..., direct_samples=..., keller_samples=..., suffix_samples=..., seed=..., max_order=..., lane_offset=..., lane_count=...)` returns `DfrAccum` for order-2/order-3 chain accumulation.
- `Scene.accum_dfr_coherent_direct(states=..., grid=..., material=..., active=..., wavelength=..., select_diffraction_point=..., prefilter_visibility=..., lane_offset=..., lane_count=...)` returns `DfrCoherentAccum`. Replicated scenes shard the deterministic `(state, grid-cell)` lane space and sum partial grids on the master. Torch rejects tracked or forward-dual state, material, and scene-geometry inputs because this variant is forward-only.

These APIs use native CUDA/OptiX source ports for the reflection and diffraction
multipath execution paths. Current completion risk is performance parity, not
placeholder kernel coverage.

The Torch EPC path wrapper intentionally reflects the fields available from its
dispatcher operation: it accepts `source` and `receiver` tensors and returns
`valid`, `path_length`, resolved IDs, surface-group IDs, hit points, and
normals. Dr.Jit's same-named method accepts a `Ray` and exposes additional
diagnostic fields with backend-native names. The two APIs share the operation
vocabulary but are not record-shape aliases.

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
