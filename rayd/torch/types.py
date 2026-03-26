from __future__ import annotations

from typing import Any

from ._util import _normalize_public_ray_fields


class _StructRepr:
    DRJIT_STRUCT: dict[str, object] = {}

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.DRJIT_STRUCT)
        return f"{type(self).__name__}({parts})"


class Ray(_StructRepr):
    DRJIT_STRUCT = {"o": object, "d": object, "tmax": object}

    def __init__(self, o: Any = None, d: Any = None, tmax: Any = None):
        self.o, self.d, self.tmax = _normalize_public_ray_fields(o, d, tmax)

    def reversed(self) -> "Ray":
        return Ray(self.o, -self.d, self.tmax)


class Intersection(_StructRepr):
    DRJIT_STRUCT = {
        "t": object,
        "p": object,
        "n": object,
        "geo_n": object,
        "uv": object,
        "barycentric": object,
        "shape_id": object,
        "prim_id": object,
    }

    def __init__(
        self,
        t: Any = None,
        p: Any = None,
        n: Any = None,
        geo_n: Any = None,
        uv: Any = None,
        barycentric: Any = None,
        shape_id: Any = None,
        prim_id: Any = None,
    ):
        self.t = t
        self.p = p
        self.n = n
        self.geo_n = geo_n
        self.uv = uv
        self.barycentric = barycentric
        self.shape_id = shape_id
        self.prim_id = prim_id

    def is_valid(self) -> Any:
        return self.prim_id >= 0


class NearestPointEdge(_StructRepr):
    DRJIT_STRUCT = {
        "distance": object,
        "point": object,
        "edge_t": object,
        "edge_point": object,
        "shape_id": object,
        "edge_id": object,
        "is_boundary": object,
    }

    def __init__(
        self,
        distance: Any = None,
        point: Any = None,
        edge_t: Any = None,
        edge_point: Any = None,
        shape_id: Any = None,
        edge_id: Any = None,
        is_boundary: Any = None,
    ):
        self.distance = distance
        self.point = point
        self.edge_t = edge_t
        self.edge_point = edge_point
        self.shape_id = shape_id
        self.edge_id = edge_id
        self.is_boundary = is_boundary

    def is_valid(self) -> Any:
        return self.edge_id >= 0


class NearestRayEdge(_StructRepr):
    DRJIT_STRUCT = {
        "distance": object,
        "ray_t": object,
        "point": object,
        "edge_t": object,
        "edge_point": object,
        "shape_id": object,
        "edge_id": object,
        "is_boundary": object,
    }

    def __init__(
        self,
        distance: Any = None,
        ray_t: Any = None,
        point: Any = None,
        edge_t: Any = None,
        edge_point: Any = None,
        shape_id: Any = None,
        edge_id: Any = None,
        is_boundary: Any = None,
    ):
        self.distance = distance
        self.ray_t = ray_t
        self.point = point
        self.edge_t = edge_t
        self.edge_point = edge_point
        self.shape_id = shape_id
        self.edge_id = edge_id
        self.is_boundary = is_boundary

    def is_valid(self) -> Any:
        return self.edge_id >= 0


class PrimaryEdgeSample(_StructRepr):
    DRJIT_STRUCT = {
        "x_dot_n": object,
        "idx": object,
        "ray_n": object,
        "ray_p": object,
        "pdf": object,
    }

    def __init__(
        self,
        x_dot_n: Any = None,
        idx: Any = None,
        ray_n: Any = None,
        ray_p: Any = None,
        pdf: Any = None,
    ):
        self.x_dot_n = x_dot_n
        self.idx = idx
        self.ray_n = ray_n
        self.ray_p = ray_p
        self.pdf = pdf


class SecondaryEdgeInfo(_StructRepr):
    DRJIT_STRUCT = {
        "start": object,
        "edge": object,
        "normal0": object,
        "normal1": object,
        "opposite": object,
        "is_boundary": object,
    }

    def __init__(
        self,
        start: Any = None,
        edge: Any = None,
        normal0: Any = None,
        normal1: Any = None,
        opposite: Any = None,
        is_boundary: Any = None,
    ):
        self.start = start
        self.edge = edge
        self.normal0 = normal0
        self.normal1 = normal1
        self.opposite = opposite
        self.is_boundary = is_boundary

    def size(self) -> int:
        if self.is_boundary is None:
            return 0
        from ._util import _shape_tuple
        return int(_shape_tuple(self.is_boundary)[0])


class SceneEdgeInfo(_StructRepr):
    DRJIT_STRUCT = {
        "start": object,
        "edge": object,
        "end": object,
        "length": object,
        "normal0": object,
        "normal1": object,
        "is_boundary": object,
        "shape_id": object,
        "local_edge_id": object,
        "global_edge_id": object,
    }

    def __init__(
        self,
        start: Any = None,
        edge: Any = None,
        end: Any = None,
        length: Any = None,
        normal0: Any = None,
        normal1: Any = None,
        is_boundary: Any = None,
        shape_id: Any = None,
        local_edge_id: Any = None,
        global_edge_id: Any = None,
    ):
        self.start = start
        self.edge = edge
        self.end = end
        self.length = length
        self.normal0 = normal0
        self.normal1 = normal1
        self.is_boundary = is_boundary
        self.shape_id = shape_id
        self.local_edge_id = local_edge_id
        self.global_edge_id = global_edge_id

    def size(self) -> int:
        if self.global_edge_id is None:
            return 0
        from ._util import _shape_tuple
        return int(_shape_tuple(self.global_edge_id)[0])


class SceneEdgeTopology(_StructRepr):
    DRJIT_STRUCT = {
        "v0": object,
        "v1": object,
        "face0_local": object,
        "face1_local": object,
        "face0_global": object,
        "face1_global": object,
        "opposite_vertex0": object,
        "opposite_vertex1": object,
    }

    def __init__(
        self,
        v0: Any = None,
        v1: Any = None,
        face0_local: Any = None,
        face1_local: Any = None,
        face0_global: Any = None,
        face1_global: Any = None,
        opposite_vertex0: Any = None,
        opposite_vertex1: Any = None,
    ):
        self.v0 = v0
        self.v1 = v1
        self.face0_local = face0_local
        self.face1_local = face1_local
        self.face0_global = face0_global
        self.face1_global = face1_global
        self.opposite_vertex0 = opposite_vertex0
        self.opposite_vertex1 = opposite_vertex1

    def size(self) -> int:
        if self.v0 is None:
            return 0
        from ._util import _shape_tuple
        return int(_shape_tuple(self.v0)[0])


class SceneCommitProfile:
    _FIELDS = (
        "mesh_update_ms",
        "triangle_scatter_ms",
        "triangle_eval_ms",
        "edge_scatter_ms",
        "edge_refit_ms",
        "optix_commit_ms",
        "total_ms",
        "optix_gas_update_ms",
        "optix_ias_update_ms",
        "updated_meshes",
        "updated_vertex_meshes",
        "updated_transform_meshes",
        "updated_edge_meshes",
        "updated_edges",
    )

    def __init__(self, native_profile: Any | None = None):
        for field in self._FIELDS:
            setattr(self, field, getattr(native_profile, field, 0))

    def __repr__(self) -> str:
        parts = ", ".join(f"{field}={getattr(self, field)!r}" for field in self._FIELDS)
        return f"SceneCommitProfile({parts})"
