# API Reference

This document describes the current public Python API exposed by `raydi`.

## Module Overview

RayDi is a geometry-only differentiable ray-intersection package built on Dr.Jit and OptiX.

Public top-level exports:

- `Mesh`
- `Scene`
- `SceneCommitProfile`
- `Camera`
- `Ray`
- `RayDetached`
- `Intersection`
- `IntersectionDetached`
- `NearestPointEdge`
- `NearestPointEdgeDetached`
- `NearestRayEdge`
- `NearestRayEdgeDetached`
- `PrimaryEdgeSample`
- `SecondaryEdgeInfo`

All array-valued inputs and outputs use Dr.Jit CUDA arrays or tensors.

## Type Conventions

RayDi follows a simple naming rule:

- `Detached` suffix: query runs on detached values
- no suffix: query participates in Dr.Jit AD

Typical input types:

- detached scalar/array types:
  - `drjit.cuda.Float`
  - `drjit.cuda.Array2f`
  - `drjit.cuda.Array3f`
  - `drjit.cuda.Array3i`
  - `drjit.cuda.Matrix4f`
- differentiable types:
  - `drjit.cuda.ad.Float`
  - `drjit.cuda.ad.Array2f`
  - `drjit.cuda.ad.Array3f`
  - `drjit.cuda.ad.Matrix4f`
- differentiable tensor output:
  - `drjit.cuda.ad.TensorXf`

Batched queries are represented by dynamic Dr.Jit arrays.

## Mesh

`Mesh` stores triangle geometry, optional UVs, transforms, and precomputed edge data.

Construction:

```python
mesh = rd.Mesh(v, f)
mesh = rd.Mesh(v, f, uv)
mesh = rd.Mesh(v, f, uv, f_uv)
mesh = rd.Mesh(v, f, uv, f_uv, verbose=True)
```

Parameters:

- `v`: vertex positions, usually `Array3f` or `ad.Array3f`
- `f`: triangle vertex indices, `Array3i`
- `uv`: optional vertex UVs, `Array2f`
- `f_uv`: optional UV indices, `Array3i`
- `verbose`: print basic mesh loading statistics

Methods:

- `configure()`
  - builds derived geometry caches and GPU buffers
- `set_transform(mat, set_left=True)`
- `append_transform(mat, append_left=True)`
- `edge_indices()`
  - returns a 5-tuple of integer arrays describing mesh edges
- `secondary_edges()`
  - returns `SecondaryEdgeInfo`

Properties:

- `num_vertices`
- `num_faces`
- `to_world`
- `to_world_left`
- `to_world_right`
- `vertex_positions`
- `vertex_positions_world`
- `vertex_normals`
- `vertex_uv`
- `face_indices`
- `face_uv_indices`
- `use_face_normals`
- `edges_enabled`

Notes:

- call `configure()` before adding the mesh to a scene if you want mesh-local caches immediately
- `Scene.configure()` will also configure meshes added to that scene
- updating geometry or transforms marks derived data dirty until reconfigured or committed through `Scene`

## Scene

`Scene` owns meshes plus the OptiX and edge acceleration data used by queries.

Construction:

```python
scene = rd.Scene()
```

Methods:

- `add_mesh(mesh, dynamic=False) -> int`
  - adds a mesh and returns its scene-local mesh id
- `configure()`
  - builds scene-wide triangle and edge acceleration data
- `update_mesh_vertices(mesh_id, positions)`
  - only valid for meshes added with `dynamic=True`
- `set_mesh_transform(mesh_id, mat, set_left=True)`
- `append_mesh_transform(mesh_id, mat, append_left=True)`
- `commit_updates()`
  - pushes pending dynamic mesh updates into scene acceleration structures
- `is_ready() -> bool`
- `has_pending_updates() -> bool`
- `intersect(ray, active=True)`
  - overloads:
    - `RayDetached -> IntersectionDetached`
    - `Ray -> Intersection`
- `nearest_edge(point, active=True)`
  - overloads:
    - `Array3fDetached -> NearestPointEdgeDetached`
    - `Array3f -> NearestPointEdge`
- `nearest_edge(ray, active=True)`
  - overloads:
    - `RayDetached -> NearestRayEdgeDetached`
    - `Ray -> NearestRayEdge`

Properties:

- `num_meshes`
- `last_commit_profile`

Notes:

- a scene must be configured before queries
- if `has_pending_updates()` is true, queries will raise until `commit_updates()` is called
- camera primary-edge caches are invalidated when the scene is reconfigured or committed

## SceneCommitProfile

Timing and update counters from the last `Scene.commit_updates()` call.

Fields:

- `mesh_update_ms`
- `triangle_scatter_ms`
- `triangle_eval_ms`
- `optix_commit_ms`
- `total_ms`
- `optix_gas_update_ms`
- `optix_ias_update_ms`
- `updated_meshes`
- `updated_vertex_meshes`
- `updated_transform_meshes`

## Camera

`Camera` provides primary-ray sampling, primary-edge preparation, point-sampled depth rendering, and primary-edge gradient rendering.

Construction with horizontal FOV:

```python
camera = rd.Camera(45.0, 1e-4, 1e4)
```

Construction with calibrated intrinsics:

```python
camera = rd.Camera(fx, fy, cx, cy, 1e-4, 1e4)
```

Methods:

- `configure(cache=True)`
  - rebuilds projection and world/sample transform caches
- `prepare_edges(scene)`
  - preprocesses primary image-space edges for the current scene
- `sample_ray(sample)`
  - overloads:
    - `Array2fDetached -> RayDetached`
    - `Array2f -> Ray`
- `sample_edge(sample1) -> PrimaryEdgeSample`
- `render(scene, background=0.0) -> TensorXf`
  - point-sampled depth image
- `render_grad(scene, spp=4, background=0.0) -> TensorXf`
  - primary-edge visibility gradient image
- `set_transform(mat, set_left=True)`
- `append_transform(mat, append_left=True)`

Properties:

- `width`
- `height`
- `to_world`
- `to_world_left`
- `to_world_right`
- `camera_to_sample`
- `sample_to_camera`
- `world_to_sample`
- `sample_to_world`

Notes:

- call `configure()` after changing camera parameters or transforms
- `render_grad()` depends on the camera’s primary-edge pipeline and is connected directly to the Dr.Jit AD graph
- `prepare_edges(scene)` must be rerun after scene updates that affect visibility

## Ray Types

### `RayDetached`

Detached ray container.

Fields:

- `o`
- `d`
- `tmax`

Methods:

- `reversed()`

### `Ray`

Differentiable ray container with the same field layout as `RayDetached`.

## Intersection Types

### `IntersectionDetached`

Detached result of `Scene.intersect(RayDetached, ...)`.

Fields:

- `t`
- `p`
- `n`
- `geo_n`
- `uv`
- `barycentric`
- `shape_id`
- `prim_id`

Methods:

- `is_valid()`

### `Intersection`

Differentiable result of `Scene.intersect(Ray, ...)` with the same field layout as `IntersectionDetached`.

## Nearest-Edge Types

### `NearestPointEdgeDetached`

Detached result of `Scene.nearest_edge(point, ...)`.

Fields:

- `distance`
- `point`
- `edge_t`
- `edge_point`
- `shape_id`
- `edge_id`
- `is_boundary`

Methods:

- `is_valid()`

Notes:

- `point` is the original query point
- `edge_t` is the closest-point parameter along the edge segment

### `NearestPointEdge`

Differentiable point-query nearest-edge result with the same field layout as `NearestPointEdgeDetached`.

### `NearestRayEdgeDetached`

Detached result of `Scene.nearest_edge(ray, ...)`.

Fields:

- `distance`
- `ray_t`
- `point`
- `edge_t`
- `edge_point`
- `shape_id`
- `edge_id`
- `is_boundary`

Methods:

- `is_valid()`

Notes:

- `ray_t` is the closest-point parameter along the query ray or finite ray segment
- `point` is the closest point on the query ray

### `NearestRayEdge`

Differentiable ray-query nearest-edge result with the same field layout as `NearestRayEdgeDetached`.

## Edge Sampling Types

### `PrimaryEdgeSample`

Result of `Camera.sample_edge(sample1)`.

Fields:

- `x_dot_n`
- `idx`
- `ray_n`
- `ray_p`
- `pdf`

Interpretation:

- `idx`: flattened pixel index
- `ray_n` and `ray_p`: primary rays on opposite sides of the sampled image-space edge
- `x_dot_n`: signed boundary term
- `pdf`: sampling density at the sampled edge point

### `SecondaryEdgeInfo`

Per-edge geometric information precomputed on a mesh.

Fields:

- `start`
- `edge`
- `normal0`
- `normal1`
- `opposite`
- `is_boundary`

Methods:

- `size()`

Interpretation:

- `start`: first endpoint in world space
- `edge`: edge vector, so the second endpoint is `start + edge`
- `normal0`: face normal on one side
- `normal1`: face normal on the opposite side
- `opposite`: third vertex of the `normal0` face

## Minimal Examples

### Ray Intersection

```python
import raydi as rd
import drjit.cuda as cuda

mesh = rd.Mesh(
    cuda.Array3f([0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]),
    cuda.Array3i([0], [1], [2]),
)

scene = rd.Scene()
scene.add_mesh(mesh)
scene.configure()

ray = rd.RayDetached(
    cuda.Array3f([0.25], [0.25], [-1.0]),
    cuda.Array3f([0.0], [0.0], [1.0]),
)

its = scene.intersect(ray)
```

### Point Nearest-Edge Query

```python
import raydi as rd
import drjit.cuda as cuda

points = cuda.Array3f([0.25], [0.1], [0.0])
edge = scene.nearest_edge(points)
```

### Ray Nearest-Edge Query

```python
import raydi as rd
import drjit.cuda as cuda

ray = rd.RayDetached(
    cuda.Array3f([0.2], [0.4], [1.0]),
    cuda.Array3f([0.0], [0.0], [-1.0]),
)
ray.tmax = cuda.Float([2.0])
edge = scene.nearest_edge(ray)
```

### Depth Rendering

```python
import raydi as rd

camera = rd.Camera(45.0, 1e-4, 1e4)
camera.width = 128
camera.height = 128
camera.configure()

depth = camera.render(scene)
```

### Primary-Edge Gradient Rendering

```python
image = camera.render_grad(scene, spp=8)
```

This returns a `TensorXf` directly from the C++ extension.
