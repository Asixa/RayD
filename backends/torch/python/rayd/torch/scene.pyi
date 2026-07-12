from collections.abc import Iterable
from typing import overload
import torch
from .mesh import Mesh
from .types import (
    AxialEdgeVisibility,
    Intersection,
    NearestEdgesTopK,
    NearestPointEdge,
    NearestRayEdge,
    Ray,
    RayFlags,
    ReflectionChain,
    ReflEpcField,
    SceneGlobalGeometry,
    SegmentChainVisibility,
    SegmentPairVisibility,
)

class Scene:
    def __init__(self) -> None: ...
    def add_mesh(self, mesh: Mesh, dynamic: bool = ...) -> int: ...
    def build(self) -> None: ...
    def is_ready(self) -> bool: ...
    @property
    def num_meshes(self) -> int: ...
    @property
    def version(self) -> int: ...
    def intersect(
        self,
        ray: Ray,
        active: torch.Tensor | None = ...,
        flags: RayFlags = ...,
    ) -> Intersection: ...
    @overload
    def nearest_edge(self, point: torch.Tensor) -> NearestPointEdge: ...
    @overload
    def nearest_edge(self, point: Ray) -> NearestRayEdge: ...
    def nearest_edges(
        self,
        point: torch.Tensor,
        k: int,
        active: torch.Tensor | None = ...,
    ) -> NearestEdgesTopK: ...
    def edge_mask(self) -> torch.Tensor: ...
    def set_edge_mask(self, mask: torch.Tensor) -> None: ...
    def global_geometry(self) -> SceneGlobalGeometry: ...
    def visible(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        active: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
    def visible_pair(
        self,
        start: torch.Tensor,
        end_a: torch.Tensor,
        end_b: torch.Tensor,
        ignore_prim_ids: torch.Tensor | None = ...,
        active: torch.Tensor | None = ...,
    ) -> SegmentPairVisibility: ...
    def visible_edge(
        self,
        source: torch.Tensor,
        edge_position: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_t_min: torch.Tensor,
        edge_t_max: torch.Tensor,
        sample_fractions: Iterable[float] = ...,
        active: torch.Tensor | None = ...,
    ) -> AxialEdgeVisibility: ...
    def visible_chain(
        self,
        points: torch.Tensor,
        chain_length: torch.Tensor,
        ignore_prim_per_segment: torch.Tensor | None = ...,
        active: torch.Tensor | None = ...,
    ) -> SegmentChainVisibility: ...
    def trace_reflections(
        self,
        ray: Ray,
        max_bounces: int,
        active: torch.Tensor | None = ...,
    ) -> ReflectionChain: ...
    def trace_refl_epc_field(
        self,
        source: torch.Tensor,
        receiver: torch.Tensor,
        max_bounces: int,
        active: torch.Tensor | None = ...,
    ) -> ReflEpcField: ...
    def update_mesh_vertices(self, mesh_id: int, positions: torch.Tensor) -> None: ...
    def sync(self) -> None: ...
    def has_pending_updates(self) -> bool: ...
