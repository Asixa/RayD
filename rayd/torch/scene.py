from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import weakref

from ._env import _torch
from ._util import (
    _normalize_active_tensor,
    _normalize_matrix_tensor,
    _normalize_scalar_tensor,
    _normalize_vector_tensor,
)
from ._convert import _scalar_array_to_tensor, _tensor_to_mask, _tensor_to_matrix4, _tensor_to_vec3, _to_torch_struct
from .types import ReflectionChain, Intersection, Ray, SceneSyncProfile, SceneEdgeInfo, SceneEdgeTopology
from ._state import _MeshState
from ._native import (
    _allocate_native_scene_cache_id,
    _build_native_mesh,
    _scene_cache_refresh_policy,
    _mesh_to_world_tensor,
    _ray_batch_size,
    _release_native_scene_cache,
    _reset_native_scene_cache,
    _scene_cache_tokens,
    _scene_edge_info_from_native,
    _scene_edge_topology_from_native,
    _scene_intersect_impl,
    _scene_trace_reflections_impl,
    _scene_nearest_point_impl,
    _scene_nearest_ray_impl,
    _scene_shadow_test_impl,
)
from .mesh import Mesh


@dataclass
class _SceneMeshRecord:
    state: _MeshState
    dynamic: bool


class Scene:
    def __init__(self):
        self._records: list[_SceneMeshRecord] = []
        self._ready = False
        self._pending_updates = False
        self._version = 0
        self._edge_version = 0
        self._native_scene: Any | None = None
        self._edge_mask: _torch.Tensor | None = None
        self._last_sync_profile = SceneSyncProfile()
        self._query_cache_id = _allocate_native_scene_cache_id()
        self._query_cache_finalizer = weakref.finalize(self, _release_native_scene_cache, self._query_cache_id)

    def _mesh_states(self) -> list[_MeshState]:
        return [record.state for record in self._records]

    def _query_cache_inputs(
        self,
    ) -> tuple[
        list[_MeshState],
        tuple[Any, ...],
        tuple[Any, ...],
        tuple[Any, ...],
        tuple[Any, ...],
        tuple[Any, ...],
        tuple[bool, tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]],
        Any,
    ]:
        mesh_states = self._mesh_states()
        topology_token, rebuild_token, vertex_tokens, left_tokens, right_tokens = _scene_cache_tokens(
            mesh_states, self._edge_mask
        )
        refresh_policy = _scene_cache_refresh_policy(mesh_states)
        return (
            mesh_states,
            topology_token,
            rebuild_token,
            vertex_tokens,
            left_tokens,
            right_tokens,
            refresh_policy,
            self._edge_mask,
        )

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Scene is not built. Call build() before querying.")

    def _require_query_ready(self) -> None:
        self._require_ready()
        if self._pending_updates:
            raise RuntimeError("Scene has pending updates. Call sync() before querying.")

    def _validate_mesh_id(self, mesh_id: int) -> _SceneMeshRecord:
        if mesh_id < 0 or mesh_id >= len(self._records):
            raise IndexError(f"Invalid mesh_id: {mesh_id}")
        return self._records[mesh_id]

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("Scene.add_mesh() expects a rayd.torch.Mesh.")
        self._records.append(_SceneMeshRecord(mesh._state.copy(), bool(dynamic)))
        self._ready = False
        self._pending_updates = False
        self._native_scene = None
        self._edge_mask = None
        _reset_native_scene_cache(self._query_cache_id)
        return len(self._records) - 1

    def build(self) -> None:
        from ._env import _native
        ns = _native.Scene()
        for record in self._records:
            ns.add_mesh(_build_native_mesh(record.state, preserve_gradients=False), record.dynamic)
        ns.build()
        _reset_native_scene_cache(self._query_cache_id)
        self._native_scene = ns
        self._edge_mask = _scalar_array_to_tensor(ns.edge_mask()).torch()
        self._ready = True
        self._pending_updates = False
        self._version = int(ns.version)
        self._edge_version = int(ns.edge_version)

    def update_mesh_vertices(self, mesh_id: int, positions: Any) -> None:
        record = self._validate_mesh_id(mesh_id)
        if not record.dynamic:
            raise RuntimeError("Scene.update_mesh_vertices(): target mesh is not dynamic.")
        self._require_ready()
        new_positions = _normalize_vector_tensor(positions, "positions", 3, _torch.float32)
        if record.state.vertex_positions is not None and new_positions.shape[0] != record.state.vertex_positions.shape[0]:
            raise RuntimeError("Scene.update_mesh_vertices(): vertex count must remain unchanged.")
        record.state.vertex_positions = new_positions
        if self._native_scene is not None:
            self._native_scene.update_mesh_vertices(mesh_id, _tensor_to_vec3(new_positions, diff=False, name="positions"))
        self._pending_updates = True

    def set_mesh_transform(self, mesh_id: int, mat: Any, set_left: bool = True) -> None:
        record = self._validate_mesh_id(mesh_id)
        if not record.dynamic:
            raise RuntimeError("Scene.set_mesh_transform(): target mesh is not dynamic.")
        self._require_ready()
        matrix = _normalize_matrix_tensor(mat, "mat")
        if set_left:
            record.state.to_world_left = matrix
        else:
            record.state.to_world_right = matrix
        if self._native_scene is not None:
            self._native_scene.set_mesh_transform(mesh_id, _tensor_to_matrix4(matrix, diff=False, name="mat"), set_left)
        self._pending_updates = True

    def append_mesh_transform(self, mesh_id: int, mat: Any, append_left: bool = True) -> None:
        record = self._validate_mesh_id(mesh_id)
        if not record.dynamic:
            raise RuntimeError("Scene.append_mesh_transform(): target mesh is not dynamic.")
        self._require_ready()
        matrix = _normalize_matrix_tensor(mat, "mat")
        current = _mesh_to_world_tensor(record.state, "to_world_left" if append_left else "to_world_right")
        if append_left:
            record.state.to_world_left = matrix @ current
        else:
            record.state.to_world_right = current @ matrix
        if self._native_scene is not None:
            self._native_scene.append_mesh_transform(mesh_id, _tensor_to_matrix4(matrix, diff=False, name="mat"), append_left)
        self._pending_updates = True

    def set_edge_mask(self, mask: Any) -> None:
        self._require_ready()
        if self._native_scene is None:
            raise RuntimeError("Scene.set_edge_mask(): internal detached scene is unavailable.")

        mask_tensor = _normalize_scalar_tensor(mask, "mask", _torch.bool).clone()
        expected_size = 0 if self._edge_mask is None else int(self._edge_mask.shape[0])
        if int(mask_tensor.shape[0]) != expected_size:
            raise RuntimeError("Scene.set_edge_mask(): mask size must match the scene edge count.")
        if self._edge_mask is not None and _torch.equal(mask_tensor, self._edge_mask):
            return

        self._native_scene.set_edge_mask(_tensor_to_mask(mask_tensor, diff=False))
        self._edge_mask = mask_tensor
        self._pending_updates = True

    def sync(self) -> None:
        self._require_ready()
        if self._native_scene is None:
            raise RuntimeError("Scene.sync(): internal detached scene is unavailable.")
        self._native_scene.sync()
        self._last_sync_profile = SceneSyncProfile(self._native_scene.last_sync_profile)
        edge_mask = _scalar_array_to_tensor(self._native_scene.edge_mask()).torch()
        if self._edge_mask is None or not _torch.equal(edge_mask, self._edge_mask):
            self._edge_mask = edge_mask
        self._pending_updates = False
        self._version = int(self._native_scene.version)
        self._edge_version = int(self._native_scene.edge_version)

    def is_ready(self) -> bool:
        return self._ready

    def has_pending_updates(self) -> bool:
        return self._pending_updates

    @property
    def last_sync_profile(self) -> SceneSyncProfile:
        return SceneSyncProfile(self._last_sync_profile)

    @property
    def num_meshes(self) -> int:
        return len(self._records)

    @property
    def version(self) -> int:
        if self._native_scene is not None:
            return int(self._native_scene.version)
        return self._version

    @property
    def edge_version(self) -> int:
        if self._native_scene is not None:
            return int(self._native_scene.edge_version)
        return self._edge_version

    def edge_info(self) -> SceneEdgeInfo:
        self._require_query_ready()
        if self._native_scene is None:
            raise RuntimeError("Scene.edge_info(): internal detached scene is unavailable.")
        return _to_torch_struct(_scene_edge_info_from_native(self._native_scene.edge_info()))

    def edge_topology(self) -> SceneEdgeTopology:
        self._require_ready()
        if self._native_scene is None:
            raise RuntimeError("Scene.edge_topology(): internal detached scene is unavailable.")
        return _to_torch_struct(_scene_edge_topology_from_native(self._native_scene.edge_topology()))

    def edge_mask(self) -> _torch.Tensor:
        self._require_ready()
        if self._edge_mask is None:
            return _torch.empty((0,), device=_torch.device("cuda"), dtype=_torch.bool)
        return self._edge_mask.clone()

    def mesh_face_offsets(self) -> _torch.Tensor:
        self._require_ready()
        if self._native_scene is None:
            raise RuntimeError("Scene.mesh_face_offsets(): internal detached scene is unavailable.")
        return _scalar_array_to_tensor(self._native_scene.mesh_face_offsets()).torch()

    def mesh_edge_offsets(self) -> _torch.Tensor:
        self._require_ready()
        if self._native_scene is None:
            raise RuntimeError("Scene.mesh_edge_offsets(): internal detached scene is unavailable.")
        return _scalar_array_to_tensor(self._native_scene.mesh_edge_offsets()).torch()

    def triangle_edge_indices(self, prim_id: Any, global_: bool = True, **kwargs: Any) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        self._require_ready()
        if "global" in kwargs:
            global_ = kwargs["global"]
        if self._native_scene is None:
            raise RuntimeError("Scene.triangle_edge_indices(): internal detached scene is unavailable.")
        edge_ids = self._native_scene.triangle_edge_indices(prim_id, bool(global_))
        return tuple(_scalar_array_to_tensor(value).torch() for value in edge_ids)

    def edge_adjacent_faces(self, edge_id: Any, global_: bool = True, **kwargs: Any) -> tuple[_torch.Tensor, _torch.Tensor]:
        self._require_ready()
        if "global" in kwargs:
            global_ = kwargs["global"]
        if self._native_scene is None:
            raise RuntimeError("Scene.edge_adjacent_faces(): internal detached scene is unavailable.")
        face_ids = self._native_scene.edge_adjacent_faces(edge_id, bool(global_))
        return tuple(_scalar_array_to_tensor(value).torch() for value in face_ids)

    def intersect(self, ray: Ray, active: Any = True) -> Intersection:
        self._require_query_ready()
        if not isinstance(ray, Ray):
            raise TypeError("Scene.intersect() expects a rayd.torch.Ray.")
        mesh_states, topology_token, rebuild_token, vertex_tokens, left_tokens, right_tokens, refresh_policy, edge_mask = self._query_cache_inputs()
        return _scene_intersect_impl(
            self._query_cache_id,
            topology_token,
            rebuild_token,
            vertex_tokens,
            left_tokens,
            right_tokens,
            refresh_policy,
            mesh_states,
            edge_mask,
            ray,
            _normalize_active_tensor(active, _ray_batch_size(ray)),
        )

    def trace_reflections(self, ray: Ray, max_bounces: int, active: Any = True) -> ReflectionChain:
        self._require_query_ready()
        if not isinstance(ray, Ray):
            raise TypeError("Scene.trace_reflections() expects a rayd.torch.Ray.")
        mesh_states, topology_token, rebuild_token, vertex_tokens, left_tokens, right_tokens, refresh_policy, edge_mask = self._query_cache_inputs()
        return _scene_trace_reflections_impl(
            self._query_cache_id,
            topology_token,
            rebuild_token,
            vertex_tokens,
            left_tokens,
            right_tokens,
            refresh_policy,
            mesh_states,
            edge_mask,
            ray,
            int(max_bounces),
            _normalize_active_tensor(active, _ray_batch_size(ray)),
        )

    def shadow_test(self, ray: Ray, active: Any = True) -> _torch.Tensor:
        self._require_query_ready()
        if not isinstance(ray, Ray):
            raise TypeError("Scene.shadow_test() expects a rayd.torch.Ray.")
        mesh_states, topology_token, rebuild_token, vertex_tokens, left_tokens, right_tokens, refresh_policy, edge_mask = self._query_cache_inputs()
        return _scene_shadow_test_impl(
            self._query_cache_id,
            topology_token,
            rebuild_token,
            vertex_tokens,
            left_tokens,
            right_tokens,
            refresh_policy,
            mesh_states,
            edge_mask,
            ray,
            _normalize_active_tensor(active, _ray_batch_size(ray)),
        )

    def nearest_edge(self, query: Any, active: Any = True) -> Any:
        self._require_query_ready()
        mesh_states, topology_token, rebuild_token, vertex_tokens, left_tokens, right_tokens, refresh_policy, edge_mask = self._query_cache_inputs()
        if isinstance(query, Ray):
            return _scene_nearest_ray_impl(
                self._query_cache_id,
                topology_token,
                rebuild_token,
                vertex_tokens,
                left_tokens,
                right_tokens,
                refresh_policy,
                mesh_states,
                edge_mask,
                query,
                _normalize_active_tensor(active, _ray_batch_size(query)),
            )
        point = _normalize_vector_tensor(query, "point", 3, _torch.float32)
        return _scene_nearest_point_impl(
            self._query_cache_id,
            topology_token,
            rebuild_token,
            vertex_tokens,
            left_tokens,
            right_tokens,
            refresh_policy,
            mesh_states,
            edge_mask,
            point,
            _normalize_active_tensor(active, point.shape[0]),
        )

    def __repr__(self) -> str:
        return f"Scene(num_meshes={self.num_meshes}, ready={self._ready}, pending_updates={self._pending_updates})"
