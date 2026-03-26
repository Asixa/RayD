from __future__ import annotations

from typing import Any

from ._env import dr, _native
from ._util import (
    _batch_size_from_vector,
    _device_from_values,
    _has_diff_fields,
    _identity_matrix,
    _infer_diff,
)
from ._convert import (
    _float_scalar_type,
    _matrix4_to_tensor,
    _scalar_array_to_tensor,
    _tensor_to_mask,
    _tensor_to_matrix4,
    _tensor_to_scalar_array,
    _tensor_to_vec2,
    _tensor_to_vec3,
    _tensor_to_vec3i,
    _to_torch_struct,
    _vec2_to_tensor,
    _vec3_to_tensor,
)
from .types import (
    Intersection,
    NearestPointEdge,
    NearestRayEdge,
    PrimaryEdgeSample,
    Ray,
    SceneEdgeInfo,
    SceneEdgeTopology,
    SecondaryEdgeInfo,
)
from ._state import _MeshState, _CameraState

from ._env import _torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ray_batch_size(ray: Ray) -> int:
    return _batch_size_from_vector(ray.o, 3, "ray.o")


def _mesh_device(state: _MeshState) -> _torch.device:
    return _device_from_values(
        state.vertex_positions,
        state.face_indices,
        state.vertex_uv,
        state.face_uv_indices,
        state.to_world,
        state.to_world_left,
        state.to_world_right,
    )


def _mesh_to_world_tensor(state: _MeshState, attr: str) -> _torch.Tensor:
    value = getattr(state, attr)
    if value is not None:
        return value
    return _identity_matrix(_mesh_device(state))


# ---------------------------------------------------------------------------
# Native ↔ public conversion
# ---------------------------------------------------------------------------

def _public_ray_from_native(ray: Any) -> Ray:
    return Ray(
        o=_vec3_to_tensor(ray.o),
        d=_vec3_to_tensor(ray.d),
        tmax=_scalar_array_to_tensor(ray.tmax),
    )


def _intersection_from_native(its: Any) -> Intersection:
    return Intersection(
        t=_scalar_array_to_tensor(its.t),
        p=_vec3_to_tensor(its.p),
        n=_vec3_to_tensor(its.n),
        geo_n=_vec3_to_tensor(its.geo_n),
        uv=_vec2_to_tensor(its.uv),
        barycentric=_vec3_to_tensor(its.barycentric),
        shape_id=_scalar_array_to_tensor(its.shape_id),
        prim_id=_scalar_array_to_tensor(its.prim_id),
    )


def _nearest_point_from_native(result: Any) -> NearestPointEdge:
    return NearestPointEdge(
        distance=_scalar_array_to_tensor(result.distance),
        point=_vec3_to_tensor(result.point),
        edge_t=_scalar_array_to_tensor(result.edge_t),
        edge_point=_vec3_to_tensor(result.edge_point),
        shape_id=_scalar_array_to_tensor(result.shape_id),
        edge_id=_scalar_array_to_tensor(result.edge_id),
        is_boundary=_scalar_array_to_tensor(result.is_boundary),
    )


def _nearest_ray_from_native(result: Any) -> NearestRayEdge:
    return NearestRayEdge(
        distance=_scalar_array_to_tensor(result.distance),
        ray_t=_scalar_array_to_tensor(result.ray_t),
        point=_vec3_to_tensor(result.point),
        edge_t=_scalar_array_to_tensor(result.edge_t),
        edge_point=_vec3_to_tensor(result.edge_point),
        shape_id=_scalar_array_to_tensor(result.shape_id),
        edge_id=_scalar_array_to_tensor(result.edge_id),
        is_boundary=_scalar_array_to_tensor(result.is_boundary),
    )


def _secondary_edges_from_native(info: Any) -> SecondaryEdgeInfo:
    return SecondaryEdgeInfo(
        start=_vec3_to_tensor(info.start),
        edge=_vec3_to_tensor(info.edge),
        normal0=_vec3_to_tensor(info.normal0),
        normal1=_vec3_to_tensor(info.normal1),
        opposite=_vec3_to_tensor(info.opposite),
        is_boundary=_scalar_array_to_tensor(info.is_boundary),
    )


def _scene_edge_info_from_native(info: Any) -> SceneEdgeInfo:
    return SceneEdgeInfo(
        start=_vec3_to_tensor(info.start),
        edge=_vec3_to_tensor(info.edge),
        end=_vec3_to_tensor(info.end),
        length=_scalar_array_to_tensor(info.length),
        normal0=_vec3_to_tensor(info.normal0),
        normal1=_vec3_to_tensor(info.normal1),
        is_boundary=_scalar_array_to_tensor(info.is_boundary),
        shape_id=_scalar_array_to_tensor(info.shape_id),
        local_edge_id=_scalar_array_to_tensor(info.local_edge_id),
        global_edge_id=_scalar_array_to_tensor(info.global_edge_id),
    )


def _scene_edge_topology_from_native(info: Any) -> SceneEdgeTopology:
    return SceneEdgeTopology(
        v0=_scalar_array_to_tensor(info.v0),
        v1=_scalar_array_to_tensor(info.v1),
        face0_local=_scalar_array_to_tensor(info.face0_local),
        face1_local=_scalar_array_to_tensor(info.face1_local),
        face0_global=_scalar_array_to_tensor(info.face0_global),
        face1_global=_scalar_array_to_tensor(info.face1_global),
        opposite_vertex0=_scalar_array_to_tensor(info.opposite_vertex0),
        opposite_vertex1=_scalar_array_to_tensor(info.opposite_vertex1),
    )


def _primary_edge_sample_from_native(sample: Any) -> PrimaryEdgeSample:
    return PrimaryEdgeSample(
        x_dot_n=_scalar_array_to_tensor(sample.x_dot_n),
        idx=_scalar_array_to_tensor(sample.idx),
        ray_n=_public_ray_from_native(sample.ray_n),
        ray_p=_public_ray_from_native(sample.ray_p),
        pdf=_scalar_array_to_tensor(sample.pdf),
    )


# ---------------------------------------------------------------------------
# Public → native conversion
# ---------------------------------------------------------------------------

def _native_ray_from_public(ray: Ray, *, diff: bool) -> Any:
    batch = _ray_batch_size(ray)
    default_tmax = dr.full(_float_scalar_type(diff), dr.inf, batch)
    native_ray = _native.Ray() if diff else _native.RayDetached()
    native_ray.o = _tensor_to_vec3(ray.o, diff=diff, name="ray.o")
    native_ray.d = _tensor_to_vec3(ray.d, diff=diff, name="ray.d")
    native_ray.tmax = _tensor_to_scalar_array(ray.tmax, diff=diff, default=default_tmax, name="ray.tmax")
    return native_ray


def _build_native_mesh(state: _MeshState, *, preserve_gradients: bool) -> Any:
    mesh = _native.Mesh(
        _tensor_to_vec3(state.vertex_positions, diff=False, name="mesh.vertex_positions"),
        _tensor_to_vec3i(state.face_indices, name="mesh.face_indices"),
        _tensor_to_vec2(state.vertex_uv, diff=False, allow_none=True, name="mesh.vertex_uv"),
        _tensor_to_vec3i(state.face_uv_indices, allow_none=True, name="mesh.face_uv_indices"),
        bool(state.verbose),
    )
    mesh.use_face_normals = bool(state.use_face_normals)
    mesh.edges_enabled = bool(state.edges_enabled)

    if state.to_world is not None:
        mesh.to_world = _tensor_to_matrix4(state.to_world, diff=False, name="mesh.to_world")
    if state.to_world_left is not None:
        mesh.to_world_left = _tensor_to_matrix4(state.to_world_left, diff=False, name="mesh.to_world_left")
    if state.to_world_right is not None:
        mesh.to_world_right = _tensor_to_matrix4(state.to_world_right, diff=False, name="mesh.to_world_right")

    if preserve_gradients:
        if _infer_diff(state.vertex_positions):
            mesh.vertex_positions = _tensor_to_vec3(state.vertex_positions, diff=True, name="mesh.vertex_positions")
        if state.vertex_uv is not None and _infer_diff(state.vertex_uv):
            mesh.vertex_uv = _tensor_to_vec2(state.vertex_uv, diff=True, name="mesh.vertex_uv")
        if state.to_world is not None and _infer_diff(state.to_world):
            mesh.to_world = _tensor_to_matrix4(state.to_world, diff=True, name="mesh.to_world")
        if state.to_world_left is not None and _infer_diff(state.to_world_left):
            mesh.to_world_left = _tensor_to_matrix4(state.to_world_left, diff=True, name="mesh.to_world_left")
        if state.to_world_right is not None and _infer_diff(state.to_world_right):
            mesh.to_world_right = _tensor_to_matrix4(state.to_world_right, diff=True, name="mesh.to_world_right")

    return mesh


def _build_native_scene(mesh_states: list[_MeshState], *, preserve_gradients: bool) -> Any:
    scene = _native.Scene()
    for state in mesh_states:
        scene.add_mesh(_build_native_mesh(state, preserve_gradients=preserve_gradients))
    scene.build()
    return scene


def _build_native_camera(state: _CameraState, *, preserve_gradients: bool) -> Any:
    if state.mode == "intrinsics":
        camera = _native.Camera(state.fx, state.fy, state.cx, state.cy, state.near_clip, state.far_clip)
    else:
        camera = _native.Camera(state.fov_x, state.near_clip, state.far_clip)
    camera.width = int(state.width)
    camera.height = int(state.height)

    if state.to_world is not None:
        camera.to_world = _tensor_to_matrix4(state.to_world, diff=False, name="camera.to_world")
    if state.to_world_left is not None:
        camera.to_world_left = _tensor_to_matrix4(state.to_world_left, diff=False, name="camera.to_world_left")
    if state.to_world_right is not None:
        camera.to_world_right = _tensor_to_matrix4(state.to_world_right, diff=False, name="camera.to_world_right")

    if preserve_gradients:
        if state.to_world is not None and _infer_diff(state.to_world):
            camera.to_world = _tensor_to_matrix4(state.to_world, diff=True, name="camera.to_world")
        if state.to_world_left is not None and _infer_diff(state.to_world_left):
            camera.to_world_left = _tensor_to_matrix4(state.to_world_left, diff=True, name="camera.to_world_left")
        if state.to_world_right is not None and _infer_diff(state.to_world_right):
            camera.to_world_right = _tensor_to_matrix4(state.to_world_right, diff=True, name="camera.to_world_right")

    camera.build(bool(state.cache))
    return camera


# ---------------------------------------------------------------------------
# @dr.wrap impl functions
# ---------------------------------------------------------------------------

@dr.wrap(source="torch", target="drjit")
def _scene_intersect_impl(mesh_states: list[_MeshState], ray: Ray, active: Any) -> Any:
    diff = _has_diff_fields(ray) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    its = scene.intersect(_native_ray_from_public(ray, diff=diff), _tensor_to_mask(active, diff=diff))
    return _intersection_from_native(its)


@dr.wrap(source="torch", target="drjit")
def _scene_shadow_test_impl(mesh_states: list[_MeshState], ray: Ray, active: Any) -> Any:
    diff = _has_diff_fields(ray) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    return _scalar_array_to_tensor(scene.shadow_test(_native_ray_from_public(ray, diff=diff), _tensor_to_mask(active, diff=diff)))


@dr.wrap(source="torch", target="drjit")
def _scene_nearest_point_impl(mesh_states: list[_MeshState], point: Any, active: Any) -> Any:
    diff = _infer_diff(point) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    result = scene.nearest_edge(_tensor_to_vec3(point, diff=diff, name="point"), _tensor_to_mask(active, diff=diff))
    return _nearest_point_from_native(result)


@dr.wrap(source="torch", target="drjit")
def _scene_nearest_ray_impl(mesh_states: list[_MeshState], ray: Ray, active: Any) -> Any:
    diff = _has_diff_fields(ray) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    result = scene.nearest_edge(_native_ray_from_public(ray, diff=diff), _tensor_to_mask(active, diff=diff))
    return _nearest_ray_from_native(result)


@dr.wrap(source="torch", target="drjit")
def _camera_sample_ray_impl(state: _CameraState, sample: Any) -> Any:
    diff = _infer_diff(sample) or _has_diff_fields(state)
    camera = _build_native_camera(state, preserve_gradients=True)
    ray = camera.sample_ray(_tensor_to_vec2(sample, diff=diff, name="sample"))
    return _public_ray_from_native(ray)


@dr.wrap(source="torch", target="drjit")
def _camera_sample_edge_impl(state: _CameraState, mesh_states: list[_MeshState], sample1: Any) -> Any:
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    camera = _build_native_camera(state, preserve_gradients=True)
    camera.prepare_edges(scene)
    return _primary_edge_sample_from_native(camera.sample_edge(_tensor_to_scalar_array(sample1, diff=False, name="sample1")))


@dr.wrap(source="torch", target="drjit")
def _camera_render_impl(state: _CameraState, mesh_states: list[_MeshState], background: float) -> Any:
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    camera = _build_native_camera(state, preserve_gradients=True)
    return camera.render(scene, background)


@dr.wrap(source="torch", target="drjit")
def _camera_render_grad_impl(state: _CameraState, mesh_states: list[_MeshState], spp: int, background: float) -> Any:
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    camera = _build_native_camera(state, preserve_gradients=True)
    return camera.render_grad(scene, spp, background)
