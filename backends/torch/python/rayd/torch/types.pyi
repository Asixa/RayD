from dataclasses import dataclass
from enum import IntFlag
from typing import Any, Callable
import torch

class RayFlags(IntFlag):
    Geometric: RayFlags
    ShadingN: RayFlags
    UV: RayFlags
    All: RayFlags

@dataclass(frozen=True)
class Ray:
    o: torch.Tensor
    d: torch.Tensor
    tmax: torch.Tensor | None = ...

@dataclass(frozen=True)
class Intersection:
    t: torch.Tensor
    p: torch.Tensor
    n: torch.Tensor
    geo_n: torch.Tensor
    uv: torch.Tensor
    barycentric: torch.Tensor
    shape_id: torch.Tensor
    prim_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor
    def is_valid(self) -> torch.Tensor: ...

@dataclass(frozen=True)
class NearestPointEdge:
    distance: torch.Tensor
    edge_point: torch.Tensor
    edge_t: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor

@dataclass(frozen=True)
class NearestEdgesTopK:
    query_count: int
    k: int
    is_valid: torch.Tensor
    distances: torch.Tensor
    points: torch.Tensor
    edge_t: torch.Tensor
    edge_points: torch.Tensor
    shape_ids: torch.Tensor
    edge_ids: torch.Tensor
    global_edge_ids: torch.Tensor
    is_boundary: torch.Tensor

@dataclass(frozen=True)
class NearestRayEdge:
    distance: torch.Tensor
    ray_t: torch.Tensor
    point: torch.Tensor
    edge_t: torch.Tensor
    edge_point: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor

@dataclass(frozen=True)
class SegmentPairVisibility:
    ray_count: int
    visible_a: torch.Tensor
    visible_b: torch.Tensor

@dataclass(frozen=True)
class AxialEdgeVisibility:
    state_count: int
    any_visible: torch.Tensor

@dataclass(frozen=True)
class SegmentChainVisibility:
    chain_count: int
    max_segments: int
    all_visible: torch.Tensor
    first_blocked_segment: torch.Tensor
    first_blocked_prim: torch.Tensor

class ReflectionChain:
    def __init__(
        self,
        valid: torch.Tensor | None = ...,
        t: torch.Tensor | None = ...,
        image_sources: torch.Tensor | None = ...,
        prim_ids: torch.Tensor | None = ...,
        *,
        loader: Callable[[bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]] | None = ...,
    ) -> None: ...
    @property
    def valid(self) -> torch.Tensor: ...
    @property
    def t(self) -> torch.Tensor: ...
    @property
    def image_sources(self) -> torch.Tensor: ...
    @property
    def prim_ids(self) -> torch.Tensor: ...

@dataclass(frozen=True)
class ReflEpcField:
    field_real: torch.Tensor
    field_imag: torch.Tensor
    path_length: torch.Tensor
    valid: torch.Tensor
    resolved_prim_ids: torch.Tensor

@dataclass(frozen=True)
class DfrGrid:
    axis: int = ...
    position: float = ...
    coord0_min: float = ...
    coord0_max: float = ...
    coord1_min: float = ...
    coord1_max: float = ...
    resolution0: int = ...
    resolution1: int = ...
    cell_area: float | None = ...
    def resolved_cell_area(self) -> float: ...

@dataclass(frozen=True)
class DfrMaterial:
    eta_r: torch.Tensor
    sigma: torch.Tensor
    mu_r: torch.Tensor
    gain: torch.Tensor
    valid: torch.Tensor
    @staticmethod
    def default(
        count: int, *, device: torch.device, dtype: torch.dtype = ...
    ) -> DfrMaterial: ...

@dataclass(frozen=True)
class DfrStates:
    edge_index: torch.Tensor
    edge_pos: torch.Tensor
    edge_dir: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    prim0: torch.Tensor
    prim1: torch.Tensor
    exterior_angle: torch.Tensor
    src: torch.Tensor
    src_power: torch.Tensor
    wi: torch.Tensor | None = ...
    d0: torch.Tensor | None = ...
    count: int | None = ...
    @property
    def state_count(self) -> int: ...
    def with_default_vectors(self) -> DfrStates: ...

@dataclass(frozen=True)
class DfrAccum:
    grid_cell_count: int
    power: torch.Tensor
    field_x_re: torch.Tensor
    field_x_im: torch.Tensor
    field_y_re: torch.Tensor
    field_y_im: torch.Tensor
    field_z_re: torch.Tensor
    field_z_im: torch.Tensor
    direct_count: torch.Tensor
    keller_count: torch.Tensor
    suffix_count: torch.Tensor
    vis_rejects: torch.Tensor
    edge_vis_rejects: torch.Tensor
    utd_rejects: torch.Tensor
    edge_uses: torch.Tensor

@dataclass(frozen=True)
class DfrCoherentAccum:
    grid_cell_count: int
    direct_field_x_re: torch.Tensor
    direct_field_x_im: torch.Tensor
    direct_field_y_re: torch.Tensor
    direct_field_y_im: torch.Tensor
    direct_field_z_re: torch.Tensor
    direct_field_z_im: torch.Tensor
    multi_field_x_re: torch.Tensor
    multi_field_x_im: torch.Tensor
    multi_field_y_re: torch.Tensor
    multi_field_y_im: torch.Tensor
    multi_field_z_re: torch.Tensor
    multi_field_z_im: torch.Tensor
    direct_count: torch.Tensor
    multi_count: torch.Tensor
    visibility_reject_count: torch.Tensor
    utd_reject_count: torch.Tensor

@dataclass(frozen=True)
class DfrPaths:
    capacity: int
    count: torch.Tensor
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    order: torch.Tensor
    edge0: torch.Tensor
    edge1: torch.Tensor
    edge2: torch.Tensor
    delay: torch.Tensor
    field_x_re: torch.Tensor
    field_x_im: torch.Tensor
    field_y_re: torch.Tensor
    field_y_im: torch.Tensor
    field_z_re: torch.Tensor
    field_z_im: torch.Tensor
    p0: torch.Tensor
    p1: torch.Tensor
    p2: torch.Tensor

@dataclass(frozen=True)
class SceneGlobalGeometry:
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normal: torch.Tensor
    shape_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor
