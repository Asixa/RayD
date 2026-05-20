from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ._util import _normalize_scalar_tensor, _normalize_vector_tensor, _expand_1d_tensor, _shape_tuple


def _register_drjit_struct(cls):
    cls.DRJIT_STRUCT = {f.name: object for f in fields(cls)}
    return cls


@_register_drjit_struct
@dataclass
class Ray:
    o: Any = None
    d: Any = None
    tmax: Any = None

    def __post_init__(self) -> None:
        import torch
        if not (isinstance(self.o, torch.Tensor) and isinstance(self.d, torch.Tensor)):
            return
        self.o = _normalize_vector_tensor(self.o, "o", 3, torch.float32)
        self.d = _normalize_vector_tensor(self.d, "d", 3, torch.float32)
        if self.o.shape[0] != self.d.shape[0]:
            raise ValueError("o and d must have the same batch size.")
        if self.tmax is not None:
            limit = _normalize_scalar_tensor(self.tmax, "tmax", torch.float32)
            self.tmax = _expand_1d_tensor(limit, self.o.shape[0], "tmax")

    def reversed(self) -> "Ray":
        return Ray(self.o, -self.d, self.tmax)


@_register_drjit_struct
@dataclass
class Intersection:
    t: Any = None
    p: Any = None
    n: Any = None
    geo_n: Any = None
    uv: Any = None
    barycentric: Any = None
    shape_id: Any = None
    prim_id: Any = None
    local_prim_id: Any = None
    global_prim_id: Any = None

    def is_valid(self) -> Any:
        return self.prim_id >= 0


@_register_drjit_struct
@dataclass
class ReflectionChain:
    bounce_count: Any = None
    discovery_count: Any = None
    representative_ray_index: Any = None
    t: Any = None
    hit_points: Any = None
    geo_normals: Any = None
    image_sources: Any = None
    plane_points: Any = None
    plane_normals: Any = None
    shape_ids: Any = None
    prim_ids: Any = None
    local_prim_ids: Any = None
    global_prim_ids: Any = None
    trailing_t: Any = None
    trailing_prim: Any = None
    trailing_dir: Any = None
    trailing_origin: Any = None

    @property
    def max_bounces(self) -> int:
        shape = _shape_tuple(self.t)
        return int(shape[1]) if len(shape) >= 2 else 0

    @property
    def ray_count(self) -> int:
        shape = _shape_tuple(self.bounce_count)
        return int(shape[0]) if len(shape) >= 1 else 0

    def is_valid(self) -> Any:
        return self.prim_ids >= 0


@_register_drjit_struct
@dataclass
class NearestPointEdge:
    distance: Any = None
    point: Any = None
    edge_t: Any = None
    edge_point: Any = None
    shape_id: Any = None
    edge_id: Any = None
    global_edge_id: Any = None
    is_boundary: Any = None

    def is_valid(self) -> Any:
        return self.edge_id >= 0


@_register_drjit_struct
@dataclass
class NearestRayEdge:
    distance: Any = None
    ray_t: Any = None
    point: Any = None
    edge_t: Any = None
    edge_point: Any = None
    shape_id: Any = None
    edge_id: Any = None
    global_edge_id: Any = None
    is_boundary: Any = None

    def is_valid(self) -> Any:
        return self.edge_id >= 0


@_register_drjit_struct
@dataclass
class PrimaryEdgeSample:
    x_dot_n: Any = None
    idx: Any = None
    ray_n: Any = None
    ray_p: Any = None
    pdf: Any = None


@_register_drjit_struct
@dataclass
class SecondaryEdgeInfo:
    start: Any = None
    edge: Any = None
    normal0: Any = None
    normal1: Any = None
    opposite: Any = None
    is_boundary: Any = None

    def size(self) -> int:
        shape = _shape_tuple(self.is_boundary)
        return int(shape[0]) if len(shape) >= 1 else 0


@_register_drjit_struct
@dataclass
class SceneEdgeInfo:
    start: Any = None
    edge: Any = None
    end: Any = None
    length: Any = None
    normal0: Any = None
    normal1: Any = None
    is_boundary: Any = None
    shape_id: Any = None
    local_edge_id: Any = None
    global_edge_id: Any = None

    def size(self) -> int:
        shape = _shape_tuple(self.global_edge_id)
        return int(shape[0]) if len(shape) >= 1 else 0


@_register_drjit_struct
@dataclass
class SceneEdgeTopology:
    v0: Any = None
    v1: Any = None
    v0_global: Any = None
    v1_global: Any = None
    face0_local: Any = None
    face1_local: Any = None
    face0_global: Any = None
    face1_global: Any = None
    opposite_vertex0: Any = None
    opposite_vertex1: Any = None
    opposite_vertex0_global: Any = None
    opposite_vertex1_global: Any = None

    def size(self) -> int:
        shape = _shape_tuple(self.v0)
        return int(shape[0]) if len(shape) >= 1 else 0


@_register_drjit_struct
@dataclass
class SceneGlobalGeometry:
    vertices: Any = None
    faces: Any = None
    face_normal: Any = None
    shape_id: Any = None
    local_prim_id: Any = None
    global_prim_id: Any = None

    def vertex_count(self) -> int:
        shape = _shape_tuple(self.vertices)
        return int(shape[0]) if len(shape) >= 1 else 0

    def face_count(self) -> int:
        shape = _shape_tuple(self.global_prim_id)
        return int(shape[0]) if len(shape) >= 1 else 0


@dataclass
class SceneSyncProfile:
    mesh_update_ms: float = 0.0
    triangle_scatter_ms: float = 0.0
    triangle_eval_ms: float = 0.0
    edge_scatter_ms: float = 0.0
    edge_refit_ms: float = 0.0
    optix_sync_ms: float = 0.0
    total_ms: float = 0.0
    optix_gas_update_ms: float = 0.0
    optix_ias_update_ms: float = 0.0
    updated_meshes: int = 0
    updated_vertex_meshes: int = 0
    updated_transform_meshes: int = 0
    updated_edge_meshes: int = 0
    updated_edges: int = 0

    @classmethod
    def from_native(cls, native: Any) -> "SceneSyncProfile":
        return cls(**{f.name: getattr(native, f.name) for f in fields(cls)})
