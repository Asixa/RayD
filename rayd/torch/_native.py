from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any

from ._env import dr, _native
from ._util import (
    _batch_size_from_vector,
    _device_from_values,
    _has_diff_fields,
    _identity_matrix,
    _infer_diff,
    _is_torch_tensor,
)
from ._convert import (
    _float_scalar_type,
    _matrix4_to_tensor,
    _scalar_array_to_tensor,
    _tensor_to_int_array,
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
    ReflectionChain,
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


@dataclass
class _NativeSceneCacheEntry:
    scene: Any
    topology_token: tuple[Any, ...]
    rebuild_token: tuple[Any, ...]
    vertex_tokens: tuple[Any, ...]
    left_tokens: tuple[Any, ...]
    right_tokens: tuple[Any, ...]
    build_count: int = 0
    sync_count: int = 0


_SCENE_QUERY_CACHE: dict[int, _NativeSceneCacheEntry] = {}
_SCENE_QUERY_CACHE_IDS = itertools.count(1)


def _allocate_native_scene_cache_id() -> int:
    return next(_SCENE_QUERY_CACHE_IDS)


def _reset_native_scene_cache(cache_id: int) -> None:
    _SCENE_QUERY_CACHE.pop(cache_id, None)


def _release_native_scene_cache(cache_id: int) -> None:
    _reset_native_scene_cache(cache_id)


def _tensor_layout_token(value: Any) -> Any:
    if value is None:
        return None
    if _is_torch_tensor(value):
        return (
            "torch-layout",
            tuple(int(v) for v in value.shape),
            tuple(int(v) for v in value.stride()),
            str(value.dtype),
            value.device.type,
            value.device.index,
        )
    return ("value-layout", type(value))


def _tensor_state_token(value: Any) -> Any:
    if value is None:
        return None
    if _is_torch_tensor(value):
        return (
            "torch-state",
            id(value),
            int(getattr(value, "_version", 0)),
            int(value.data_ptr()) if value.numel() > 0 else 0,
            tuple(int(v) for v in value.shape),
            tuple(int(v) for v in value.stride()),
            str(value.dtype),
            value.device.type,
            value.device.index,
            bool(value.requires_grad),
        )
    return ("value-state", type(value), value)


def _scene_topology_token(mesh_states: list[_MeshState]) -> tuple[Any, ...]:
    return tuple(
        (
            _tensor_layout_token(state.vertex_positions),
            _tensor_state_token(state.face_indices),
            _tensor_layout_token(state.vertex_uv),
            _tensor_state_token(state.face_uv_indices),
            bool(state.use_face_normals),
            bool(state.edges_enabled),
            bool(state.verbose),
        )
        for state in mesh_states
    )


def _scene_rebuild_token(mesh_states: list[_MeshState]) -> tuple[Any, ...]:
    return tuple(
        (
            _tensor_state_token(state.vertex_uv),
            _tensor_state_token(state.to_world),
        )
        for state in mesh_states
    )


def _scene_vertex_tokens(mesh_states: list[_MeshState]) -> tuple[Any, ...]:
    return tuple(_tensor_state_token(state.vertex_positions) for state in mesh_states)


def _scene_transform_tokens(mesh_states: list[_MeshState], attr: str) -> tuple[Any, ...]:
    return tuple(_tensor_state_token(getattr(state, attr)) for state in mesh_states)


def _scene_cache_tokens(
    mesh_states: list[_MeshState],
    edge_mask: Any | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    return (
        _scene_topology_token(mesh_states),
        _scene_rebuild_token(mesh_states) + (_tensor_state_token(edge_mask),),
        _scene_vertex_tokens(mesh_states),
        _scene_transform_tokens(mesh_states, "to_world_left"),
        _scene_transform_tokens(mesh_states, "to_world_right"),
    )


def _scene_cache_refresh_policy(
    mesh_states: list[_MeshState],
) -> tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]]:
    return (
        any(_infer_diff(state.vertex_uv) or _infer_diff(state.to_world) for state in mesh_states),
        tuple(_infer_diff(state.vertex_positions) for state in mesh_states),
        tuple(_infer_diff(state.to_world_left) for state in mesh_states),
        tuple(_infer_diff(state.to_world_right) for state in mesh_states),
    )


def _build_query_native_scene(mesh_states: list[_MeshState], edge_mask: Any | None = None) -> Any:
    scene = _native.Scene()
    for state in mesh_states:
        scene.add_mesh(_build_native_mesh(state, preserve_gradients=True), True)
    scene.build()
    if edge_mask is not None:
        native_edge_mask = _tensor_to_mask(edge_mask, diff=False)
        if not isinstance(native_edge_mask, bool) and dr.width(native_edge_mask) > 0 and not bool(dr.all(native_edge_mask)):
            scene.set_edge_mask(native_edge_mask)
            scene.sync()
    return scene


def _sync_query_native_scene(
    entry: _NativeSceneCacheEntry,
    mesh_states: list[_MeshState],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    force_vertex_refresh: tuple[bool, ...],
    force_left_refresh: tuple[bool, ...],
    force_right_refresh: tuple[bool, ...],
) -> None:
    dirty = False

    for mesh_id, state in enumerate(mesh_states):
        if entry.vertex_tokens[mesh_id] != vertex_tokens[mesh_id] or force_vertex_refresh[mesh_id]:
            entry.scene.update_mesh_vertices(
                mesh_id,
                _tensor_to_vec3(state.vertex_positions, diff=True, name=f"mesh_states[{mesh_id}].vertex_positions"),
            )
            dirty = True

        if entry.left_tokens[mesh_id] != left_tokens[mesh_id] or force_left_refresh[mesh_id]:
            entry.scene.set_mesh_transform(
                mesh_id,
                _tensor_to_matrix4(
                    state.to_world_left if state.to_world_left is not None else _identity_matrix(_mesh_device(state)),
                    diff=True,
                    name=f"mesh_states[{mesh_id}].to_world_left",
                ),
                True,
            )
            dirty = True

        if entry.right_tokens[mesh_id] != right_tokens[mesh_id] or force_right_refresh[mesh_id]:
            entry.scene.set_mesh_transform(
                mesh_id,
                _tensor_to_matrix4(
                    state.to_world_right if state.to_world_right is not None else _identity_matrix(_mesh_device(state)),
                    diff=True,
                    name=f"mesh_states[{mesh_id}].to_world_right",
                ),
                False,
            )
            dirty = True

    if dirty:
        entry.scene.sync()
        entry.sync_count += 1


def _prepare_native_scene_cache(
    cache_id: int,
    mesh_states: list[_MeshState],
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    edge_mask: Any | None = None,
) -> Any:
    force_rebuild, force_vertex_refresh, force_left_refresh, force_right_refresh = refresh_policy
    entry = _SCENE_QUERY_CACHE.get(cache_id)
    if force_rebuild or entry is None or entry.topology_token != topology_token or entry.rebuild_token != rebuild_token:
        build_count = 1 if entry is None else entry.build_count + 1
        sync_count = 0 if entry is None else entry.sync_count
        entry = _NativeSceneCacheEntry(
            scene=_build_query_native_scene(mesh_states, edge_mask),
            topology_token=topology_token,
            rebuild_token=rebuild_token,
            vertex_tokens=vertex_tokens,
            left_tokens=left_tokens,
            right_tokens=right_tokens,
            build_count=build_count,
            sync_count=sync_count,
        )
        _SCENE_QUERY_CACHE[cache_id] = entry
        return entry.scene

    _sync_query_native_scene(
        entry,
        mesh_states,
        vertex_tokens,
        left_tokens,
        right_tokens,
        force_vertex_refresh,
        force_left_refresh,
        force_right_refresh,
    )
    entry.vertex_tokens = vertex_tokens
    entry.left_tokens = left_tokens
    entry.right_tokens = right_tokens
    return entry.scene


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


def _reflection_chain_from_native(chain: Any) -> ReflectionChain:
    max_bounces = int(getattr(chain, "max_bounces", 0))
    ray_count = int(getattr(chain, "ray_count", 0))

    bounce_count = _scalar_array_to_tensor(chain.bounce_count)
    discovery_count = _scalar_array_to_tensor(chain.discovery_count)
    representative_ray_index = _scalar_array_to_tensor(chain.representative_ray_index)
    t = _scalar_array_to_tensor(chain.t)
    hit_points = _vec3_to_tensor(chain.hit_points)
    geo_normals = _vec3_to_tensor(chain.geo_normals)
    image_sources = _vec3_to_tensor(chain.image_sources)
    plane_points = _vec3_to_tensor(chain.plane_points)
    plane_normals = _vec3_to_tensor(chain.plane_normals)
    shape_ids = _scalar_array_to_tensor(chain.shape_ids)
    prim_ids = _scalar_array_to_tensor(chain.prim_ids)

    if ray_count <= 0 and hasattr(bounce_count, "numel"):
        ray_count = int(bounce_count.numel())
    if max_bounces <= 0 and ray_count > 0 and hasattr(t, "numel"):
        max_bounces = int(t.numel() // ray_count)

    bounce_count = dr.reshape(type(bounce_count), bounce_count, shape=(ray_count,))
    discovery_count = dr.reshape(type(discovery_count), discovery_count, shape=(ray_count,))
    representative_ray_index = dr.reshape(
        type(representative_ray_index),
        representative_ray_index,
        shape=(ray_count,),
    )
    t = dr.reshape(type(t), t, shape=(ray_count, max_bounces))
    hit_points = dr.reshape(type(hit_points), hit_points, shape=(ray_count, max_bounces, 3))
    geo_normals = dr.reshape(type(geo_normals), geo_normals, shape=(ray_count, max_bounces, 3))
    image_sources = dr.reshape(type(image_sources), image_sources, shape=(ray_count, max_bounces, 3))
    plane_points = dr.reshape(type(plane_points), plane_points, shape=(ray_count, max_bounces, 3))
    plane_normals = dr.reshape(type(plane_normals), plane_normals, shape=(ray_count, max_bounces, 3))
    shape_ids = dr.reshape(type(shape_ids), shape_ids, shape=(ray_count, max_bounces))
    prim_ids = dr.reshape(type(prim_ids), prim_ids, shape=(ray_count, max_bounces))

    return ReflectionChain(
        bounce_count=bounce_count,
        discovery_count=discovery_count,
        representative_ray_index=representative_ray_index,
        t=t,
        hit_points=hit_points,
        geo_normals=geo_normals,
        image_sources=image_sources,
        plane_points=plane_points,
        plane_normals=plane_normals,
        shape_ids=shape_ids,
        prim_ids=prim_ids,
        max_bounces=max_bounces,
        ray_count=ray_count,
    )


def _nearest_point_from_native(result: Any) -> NearestPointEdge:
    return NearestPointEdge(
        distance=_scalar_array_to_tensor(result.distance),
        point=_vec3_to_tensor(result.point),
        edge_t=_scalar_array_to_tensor(result.edge_t),
        edge_point=_vec3_to_tensor(result.edge_point),
        shape_id=_scalar_array_to_tensor(result.shape_id),
        edge_id=_scalar_array_to_tensor(result.edge_id),
        global_edge_id=_scalar_array_to_tensor(result.global_edge_id),
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
        global_edge_id=_scalar_array_to_tensor(result.global_edge_id),
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
def _scene_intersect_impl(
    cache_id: int,
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    mesh_states: list[_MeshState],
    edge_mask: Any,
    ray: Ray,
    active: Any,
) -> Any:
    diff = _has_diff_fields(ray) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _prepare_native_scene_cache(
        cache_id,
        mesh_states,
        topology_token,
        rebuild_token,
        vertex_tokens,
        left_tokens,
        right_tokens,
        refresh_policy,
        edge_mask,
    )
    its = scene.intersect(_native_ray_from_public(ray, diff=diff), _tensor_to_mask(active, diff=diff))
    return _intersection_from_native(its)


@dr.wrap(source="torch", target="drjit")
def _scene_trace_reflections_impl(
    cache_id: int,
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    mesh_states: list[_MeshState],
    edge_mask: Any,
    ray: Ray,
    max_bounces: int,
    deduplicate: bool,
    canonical_prim_table: Any,
    image_source_tolerance: float,
    active: Any,
) -> Any:
    diff = _has_diff_fields(ray) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _prepare_native_scene_cache(
        cache_id,
        mesh_states,
        topology_token,
        rebuild_token,
        vertex_tokens,
        left_tokens,
        right_tokens,
        refresh_policy,
        edge_mask,
    )
    options = _native.ReflectionTraceOptions()
    options.deduplicate = bool(deduplicate)
    options.canonical_prim_table = _tensor_to_int_array(
        canonical_prim_table,
        allow_none=True,
        name="canonical_prim_table",
    )
    options.image_source_tolerance = float(image_source_tolerance)
    chain = scene.trace_reflections(
        _native_ray_from_public(ray, diff=diff),
        int(max_bounces),
        options,
        _tensor_to_mask(active, diff=diff),
    )
    return _reflection_chain_from_native(chain)


@dr.wrap(source="torch", target="drjit")
def _scene_shadow_test_impl(
    cache_id: int,
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    mesh_states: list[_MeshState],
    edge_mask: Any,
    ray: Ray,
    active: Any,
) -> Any:
    diff = _has_diff_fields(ray) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _prepare_native_scene_cache(
        cache_id,
        mesh_states,
        topology_token,
        rebuild_token,
        vertex_tokens,
        left_tokens,
        right_tokens,
        refresh_policy,
        edge_mask,
    )
    return _scalar_array_to_tensor(scene.shadow_test(_native_ray_from_public(ray, diff=diff), _tensor_to_mask(active, diff=diff)))


@dr.wrap(source="torch", target="drjit")
def _scene_nearest_point_impl(
    cache_id: int,
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    mesh_states: list[_MeshState],
    edge_mask: Any,
    point: Any,
    active: Any,
) -> Any:
    diff = _infer_diff(point) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _prepare_native_scene_cache(
        cache_id,
        mesh_states,
        topology_token,
        rebuild_token,
        vertex_tokens,
        left_tokens,
        right_tokens,
        refresh_policy,
        edge_mask,
    )
    result = scene.nearest_edge(_tensor_to_vec3(point, diff=diff, name="point"), _tensor_to_mask(active, diff=diff))
    return _nearest_point_from_native(result)


@dr.wrap(source="torch", target="drjit")
def _scene_nearest_ray_impl(
    cache_id: int,
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    mesh_states: list[_MeshState],
    edge_mask: Any,
    ray: Ray,
    active: Any,
) -> Any:
    diff = _has_diff_fields(ray) or any(_has_diff_fields(s) for s in mesh_states)
    scene = _prepare_native_scene_cache(
        cache_id,
        mesh_states,
        topology_token,
        rebuild_token,
        vertex_tokens,
        left_tokens,
        right_tokens,
        refresh_policy,
        edge_mask,
    )
    result = scene.nearest_edge(_native_ray_from_public(ray, diff=diff), _tensor_to_mask(active, diff=diff))
    return _nearest_ray_from_native(result)


@dr.wrap(source="torch", target="drjit")
def _camera_sample_ray_impl(state: _CameraState, sample: Any) -> Any:
    diff = _infer_diff(sample) or _has_diff_fields(state)
    camera = _build_native_camera(state, preserve_gradients=True)
    ray = camera.sample_ray(_tensor_to_vec2(sample, diff=diff, name="sample"))
    return _public_ray_from_native(ray)


@dr.wrap(source="torch", target="drjit")
def _camera_sample_edge_impl(
    state: _CameraState,
    cache_id: int,
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    mesh_states: list[_MeshState],
    edge_mask: Any,
    sample1: Any,
) -> Any:
    scene = _prepare_native_scene_cache(
        cache_id,
        mesh_states,
        topology_token,
        rebuild_token,
        vertex_tokens,
        left_tokens,
        right_tokens,
        refresh_policy,
        edge_mask,
    )
    camera = _build_native_camera(state, preserve_gradients=True)
    camera.prepare_edges(scene)
    return _primary_edge_sample_from_native(camera.sample_edge(_tensor_to_scalar_array(sample1, diff=False, name="sample1")))


@dr.wrap(source="torch", target="drjit")
def _camera_render_impl(
    state: _CameraState,
    cache_id: int,
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    mesh_states: list[_MeshState],
    edge_mask: Any,
    background: float,
) -> Any:
    scene = _prepare_native_scene_cache(
        cache_id,
        mesh_states,
        topology_token,
        rebuild_token,
        vertex_tokens,
        left_tokens,
        right_tokens,
        refresh_policy,
        edge_mask,
    )
    camera = _build_native_camera(state, preserve_gradients=True)
    return camera.render(scene, background)


@dr.wrap(source="torch", target="drjit")
def _camera_render_grad_impl(
    state: _CameraState,
    cache_id: int,
    topology_token: tuple[Any, ...],
    rebuild_token: tuple[Any, ...],
    vertex_tokens: tuple[Any, ...],
    left_tokens: tuple[Any, ...],
    right_tokens: tuple[Any, ...],
    refresh_policy: tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
    mesh_states: list[_MeshState],
    edge_mask: Any,
    spp: int,
    background: float,
) -> Any:
    scene = _prepare_native_scene_cache(
        cache_id,
        mesh_states,
        topology_token,
        rebuild_token,
        vertex_tokens,
        left_tokens,
        right_tokens,
        refresh_policy,
        edge_mask,
    )
    camera = _build_native_camera(state, preserve_gradients=True)
    return camera.render_grad(scene, spp, background)
