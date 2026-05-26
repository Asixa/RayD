# AGENTS.md

This repository contains **RayD**, a Dr.Jit-native GPU package for differentiable ray geometry, edge queries, and multipath simulation primitives built on OptiX.

## Environment

Default conda environment:

```bash
conda activate witwin3
```

## Build

```bash
pip install .
pip install --no-build-isolation -ve .
```

## Architecture

- `include/rayd/`, `src/`: C++/CUDA geometry, edge, and multipath kernels
- `Scene`: mesh container plus OptiX acceleration structure
- `Mesh`: raw triangle mesh input, transforms, edge topology, secondary edge query data
- `Camera`: primary-ray sampling plus primary-edge preprocessing/sampling
- `scene.intersect(ray)`: differentiable intersection query
- `scene.nearest_edge(point)` / `scene.nearest_edge(ray)`: scene-level nearest-edge query over a GPU BVH
- `scene.trace_reflections(...)`: specular reflection-path tracing
- `scene.trace_segment_visibility(...)`: batched segment visibility queries
- `scene.trace_reflection_epc(...)`: equivalent-path correction primitives for reflection paths
- `include/rayd/multipath/`, `src/multipath/`: multipath result types, OptiX launch wrappers, and CUDA/OptiX kernels
- `rayd/__init__.py`: Python package re-export

## OptiX Pipeline Guardrail

- If a native multipath call fails with `OptiX error in optixPipelineCreate(multipath)`, treat it first as a multipath OptiX pipeline configuration issue, not as an input/API issue.
- The verified 2026-05-26 fix keeps scene/edge OptiX production flags separate from multipath flags: multipath uses production module optimization plus `RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS=11`.
- Trace-call count and instruction count are useful diagnostics, but they are not proof of root cause; do not split reflection tracing or add fallback launches unless tests prove the pipeline shape itself is the failing variable.
- Always verify in a fresh subprocess with the actually loaded conda `.pyd`, and run `tests.drjit.test_optix_pipeline_cold_create` for public API cold-create coverage.
- See `docs/optix_pipeline_create_failures.md` for the root-cause writeup and regression checklist.

## Edge BVH Status

Current edge-query acceleration design:

- triangle intersections still use OptiX through `scene.intersect(ray)`
- nearest-edge queries default to the OptiX custom-AABB edge backend; a custom Dr.Jit/CUDA BVH over scene-global `SecondaryEdgeInfo` is available as an alternate backend (`edge_bvh_backend="drjit"`)
- the Dr.Jit backend build path is:
  - GPU LBVH construction with Morton codes
  - host-side treelet optimization for large-scene query quality
  - GPU traversal for point and ray closest-edge queries
- `nearest_edge(ray)` uses segment semantics on `[0, tmax]` when `tmax` is finite
- the broad phase is detached; after the winning edge is found, the implementation re-gathers AD geometry and recomputes the exact result so gradients remain available while the winning edge stays fixed
- dynamic vertex and transform updates use BVH refit after `sync()`

Current status notes:

- the treelet path is the best current tradeoff for actual nearest-edge query throughput
- an `LBVH + top-level SAH` HLBVH-style path exists as an experiment, but it is not the default because it made large-scene queries much slower
- a GPU treelet optimization prototype also exists, but it is disabled by default because it increased `build()` cost without improving end-to-end results enough

Performance snapshot used for the current decision, measured on the verified Windows machine in this repository (`RTX 5080`, `Ryzen 7 9800X3D`) with a `192x192` grid mesh, `110,976` edges, and `65,536` batched queries:

| Path | `build()` | point query | finite ray query | infinite ray query | `sync()` |
| --- | ---: | ---: | ---: | ---: | ---: |
| default treelet path | `138.43 ms` | `9.99 ms avg` | `14.34 ms avg` | `15.43 ms avg` | `3.69 ms` |
| HLBVH top-level SAH experiment | `13.45 ms` | `102.91 ms avg` | `136.59 ms avg` | `143.16 ms avg` | `3.09 ms` |

RayD is not a full renderer and intentionally does not include:

- BSDFs
- emitters
- integrators
- scene loaders
- bitmap/image I/O
- a material-light-integrator framework

## Tests

```bash
python -m unittest tests.drjit.test_geometry -v
```
