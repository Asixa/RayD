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

- `rt.backend_capabilities()`: machine-readable capability manifest.
- `rt.api_manifest()`: public API surface manifest.

The `rayd.torch.path_exchange` submodule provides the backend-neutral path
record exchange format.

## Scene Lifecycle

- `Scene.add_mesh(mesh, dynamic=False)` returns the new mesh id.
- `Scene.build()` builds the acceleration structures; `Scene.is_ready()` reports build state.
- `Scene.num_meshes` / `Scene.version`: mesh count and scene revision.
- `Scene.update_mesh_vertices(mesh_id, positions)` then `Scene.sync()` refits without a full rebuild; `Scene.has_pending_updates()` reports outstanding edits.

## Geometry

- `Scene.intersect(ray, active=None, flags=RayFlags.All)` returns `Intersection`.
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
- `Scene.trace_refl_epc_field(source, receiver, max_bounces, active=None)` returns `ReflEpcField`; the forward path uses RayD-source-ported reflection EPC plus EPC field kernels with simplified default material/options at the Python API boundary.
- `Scene.trace_dfr_paths(tx_positions=..., rx_positions=..., states=..., material=..., active=..., max_paths=..., wavelength=...)` returns `DfrPaths`.
- `Scene.accum_dfr_direct(states=..., grid=..., material=..., active=..., wavelength=..., direct_samples=..., keller_samples=..., suffix_samples=..., seed=...)` returns `DfrAccum`.
- `Scene.accum_dfr(initial_states=..., recursive_states=..., grid=..., material=..., active=..., recursive_active=..., wavelength=..., direct_samples=..., keller_samples=..., suffix_samples=..., seed=..., max_order=...)` returns `DfrAccum` for order-2/order-3 chain accumulation.
- `Scene.accum_dfr_coherent_direct(states=..., grid=..., material=..., active=..., wavelength=..., select_diffraction_point=..., prefilter_visibility=...)` returns `DfrCoherentAccum`.

These APIs use native CUDA/OptiX source ports for the reflection and diffraction
multipath execution paths. Current completion risk is performance parity, not
placeholder kernel coverage.

## Tensor ABI

All native geometry and multipath inputs are CUDA tensors. Continuous vector inputs use contiguous `torch.float32`; topology and primitive ids use `torch.int32`; masks and validity outputs use `torch.bool`. Native tapes are Torch-owned tensors and remain on the same CUDA device as the inputs.

## AD Contract

Native operators support VJP and JVP for continuous Torch inputs. Discrete choices such as primitive id, edge id, visibility, and fixed path sequence are non-differentiable and are held fixed from the forward pass. Continuous outputs are recomputed from the fixed winner and live geometry tensors during AD.

The Torch backend does not import or depend on Dr.Jit in the `rayd.torch` package path.

See [`torch_gap_analysis.md`](torch_gap_analysis.md) for the acceptance status
recorded on 2026-06-12.
