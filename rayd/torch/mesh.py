from __future__ import annotations

from typing import Any

from ._env import _torch
from ._util import (
    _empty_idx3,
    _empty_vec2,
    _empty_vec3,
    _normalize_matrix_tensor,
    _normalize_vector_tensor,
    _torch_or_default,
)
from ._convert import _scalar_array_to_tensor, _to_torch_struct, _vec3_to_tensor
from .types import SecondaryEdgeInfo
from ._state import _MeshState
from ._native import (
    _build_native_mesh,
    _mesh_device,
    _mesh_to_world_tensor,
    _secondary_edges_from_native,
)


class Mesh:
    def __init__(
        self,
        v: Any = None,
        f: Any = None,
        uv: Any = None,
        f_uv: Any = None,
        verbose: bool = False,
    ):
        self._built = False
        if v is None and f is None:
            self._state = _MeshState(verbose=bool(verbose))
            return
        if v is None or f is None:
            raise TypeError("Mesh() expects both v and f, or neither.")
        vertices = _normalize_vector_tensor(v, "v", 3, _torch.float32)
        faces = _normalize_vector_tensor(f, "f", 3, _torch.int32)
        device = vertices.device
        self._state = _MeshState(
            vertex_positions=vertices,
            face_indices=faces,
            vertex_uv=_normalize_vector_tensor(uv, "uv", 2, _torch.float32) if uv is not None else _empty_vec2(device),
            face_uv_indices=_normalize_vector_tensor(f_uv, "f_uv", 3, _torch.int32) if f_uv is not None else _empty_idx3(device),
            verbose=bool(verbose),
        )

    def _invalidate(self) -> None:
        self._built = False

    def _native_detached(self) -> Any:
        mesh = _build_native_mesh(self._state, preserve_gradients=False)
        mesh.build()
        return mesh

    def build(self) -> None:
        self._native_detached()
        self._built = True

    def set_transform(self, mat: Any, set_left: bool = True) -> None:
        if set_left:
            self.to_world_left = mat
        else:
            self.to_world_right = mat

    def append_transform(self, mat: Any, append_left: bool = True) -> None:
        matrix = _normalize_matrix_tensor(mat, "mat")
        if append_left:
            self.to_world_left = matrix @ self.to_world_left
        else:
            self.to_world_right = self.to_world_right @ matrix

    def edge_indices(self) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
        edge_indices = self._native_detached().edge_indices()
        return tuple(_scalar_array_to_tensor(value).torch() for value in edge_indices)

    def secondary_edges(self) -> SecondaryEdgeInfo:
        return _to_torch_struct(_secondary_edges_from_native(self._native_detached().secondary_edges()))

    @property
    def num_vertices(self) -> int:
        return 0 if self._state.vertex_positions is None else int(self._state.vertex_positions.shape[0])

    @property
    def num_faces(self) -> int:
        return 0 if self._state.face_indices is None else int(self._state.face_indices.shape[0])

    @property
    def to_world(self) -> _torch.Tensor:
        return _mesh_to_world_tensor(self._state, "to_world")

    @to_world.setter
    def to_world(self, value: Any) -> None:
        self._state.to_world = _normalize_matrix_tensor(value, "to_world")
        self._invalidate()

    @property
    def to_world_left(self) -> _torch.Tensor:
        return _mesh_to_world_tensor(self._state, "to_world_left")

    @to_world_left.setter
    def to_world_left(self, value: Any) -> None:
        self._state.to_world_left = _normalize_matrix_tensor(value, "to_world_left")
        self._invalidate()

    @property
    def to_world_right(self) -> _torch.Tensor:
        return _mesh_to_world_tensor(self._state, "to_world_right")

    @to_world_right.setter
    def to_world_right(self, value: Any) -> None:
        self._state.to_world_right = _normalize_matrix_tensor(value, "to_world_right")
        self._invalidate()

    @property
    def vertex_positions(self) -> _torch.Tensor:
        return _torch_or_default(self._state.vertex_positions, _empty_vec3(_mesh_device(self._state)))

    @vertex_positions.setter
    def vertex_positions(self, value: Any) -> None:
        self._state.vertex_positions = _normalize_vector_tensor(value, "vertex_positions", 3, _torch.float32)
        self._invalidate()

    @property
    def vertex_positions_world(self) -> _torch.Tensor:
        return _vec3_to_tensor(self._native_detached().vertex_positions_world).torch()

    @property
    def vertex_normals(self) -> _torch.Tensor:
        return _vec3_to_tensor(self._native_detached().vertex_normals).torch()

    @property
    def vertex_uv(self) -> _torch.Tensor:
        return _torch_or_default(self._state.vertex_uv, _empty_vec2(_mesh_device(self._state)))

    @vertex_uv.setter
    def vertex_uv(self, value: Any) -> None:
        self._state.vertex_uv = _normalize_vector_tensor(value, "vertex_uv", 2, _torch.float32)
        self._invalidate()

    @property
    def face_indices(self) -> _torch.Tensor:
        return _torch_or_default(self._state.face_indices, _empty_idx3(_mesh_device(self._state)))

    @face_indices.setter
    def face_indices(self, value: Any) -> None:
        self._state.face_indices = _normalize_vector_tensor(value, "face_indices", 3, _torch.int32)
        self._invalidate()

    @property
    def face_uv_indices(self) -> _torch.Tensor:
        return _torch_or_default(self._state.face_uv_indices, _empty_idx3(_mesh_device(self._state)))

    @face_uv_indices.setter
    def face_uv_indices(self, value: Any) -> None:
        self._state.face_uv_indices = _normalize_vector_tensor(value, "face_uv_indices", 3, _torch.int32)
        self._invalidate()

    @property
    def use_face_normals(self) -> bool:
        return bool(self._state.use_face_normals)

    @use_face_normals.setter
    def use_face_normals(self, value: bool) -> None:
        self._state.use_face_normals = bool(value)
        self._invalidate()

    @property
    def edges_enabled(self) -> bool:
        return bool(self._state.edges_enabled)

    @edges_enabled.setter
    def edges_enabled(self, value: bool) -> None:
        self._state.edges_enabled = bool(value)
        self._invalidate()

    def __repr__(self) -> str:
        return (
            f"Mesh(num_vertices={self.num_vertices}, num_faces={self.num_faces}, "
            f"use_face_normals={self.use_face_normals}, edges_enabled={self.edges_enabled})"
        )
