# Copyright Xingyu Chen.
# Exposes the Dr.Jit C Python API.

from collections.abc import Sequence
from enum import Enum, IntFlag
from typing import Any, TypeAlias, overload

Array: TypeAlias = Any
Mask: TypeAlias = Any
Matrix4f: TypeAlias = Any

REFLECTION_EXPORT_FULL: int
REFLECTION_EXPORT_MINIMAL: int
REFLECTION_EXPORT_COUNT_ONLY: int
RAYD_DFR_DIRECT: int
RAYD_DFR_KELLER: int
RAYD_DFR_SUFFIX_REFL: int
RAYD_DFR_HASH: int
RAYD_DFR_SOBOL: int
RAYD_DFR_MATCHED_ISO: int

def device_count() -> int: ...
def current_device() -> int: ...
def set_device(device: int, initialize_optix: bool = ...) -> None: ...
def optix_available() -> bool: ...
def native_launch_audit_clear() -> None: ...
def native_launch_audit() -> dict[str, dict[str, Any]]: ...

class RayFlags(IntFlag):
    None_: RayFlags
    Geometric: RayFlags
    ShadingN: RayFlags
    UV: RayFlags
    All: RayFlags

class SurfelPrimitiveMode(Enum):
    Icosahedron20: SurfelPrimitiveMode
    QuadTriangles: SurfelPrimitiveMode
    SingleTriangle: SurfelPrimitiveMode

class SurfelColorModel(Enum):
    ConstantRGB: SurfelColorModel
    FeatureChannels: SurfelColorModel
    SH: SurfelColorModel

class SurfelRenderMode(Enum):
    RGB: SurfelRenderMode
    Alpha: SurfelRenderMode
    Depth: SurfelRenderMode
    RGBDepth: SurfelRenderMode
    Feature: SurfelRenderMode

class _RayBase:
    o: Array
    d: Array
    tmax: Array
    def __init__(self, o: Array = ..., d: Array = ...) -> None: ...
    def reversed(self) -> Any: ...

class Ray(_RayBase):
    def reversed(self) -> Ray: ...

class RayAD(_RayBase):
    def reversed(self) -> RayAD: ...

class SecondaryEdgeInfo:
    start: Array
    edge: Array
    normal0: Array
    normal1: Array
    opposite: Array
    is_boundary: Array
    def size(self) -> int: ...

class SceneEdgeInfo:
    start: Array
    edge: Array
    end: Array
    length: Array
    normal0: Array
    normal1: Array
    is_boundary: Array
    shape_id: Array
    local_edge_id: Array
    global_edge_id: Array
    def size(self) -> int: ...

class SceneEdgeTopology:
    v0: Array
    v1: Array
    v0_global: Array
    v1_global: Array
    face0_local: Array
    face1_local: Array
    face0_global: Array
    face1_global: Array
    opposite_vertex0: Array
    opposite_vertex1: Array
    opposite_vertex0_global: Array
    opposite_vertex1_global: Array
    def size(self) -> int: ...

class SceneGeometry:
    vertices: Array
    faces: Array
    face_normal: Array
    shape_id: Array
    local_prim_id: Array
    global_prim_id: Array
    def vertex_count(self) -> int: ...
    def face_count(self) -> int: ...

class SurfelTraceOptions:
    alpha_min: float
    cutoff: float
    alpha_cap: float
    proxy_epsilon: float
    max_candidate_hits: int
    primitive_mode: SurfelPrimitiveMode
    face_forward: bool
    single_launch: bool
    collect_candidate_stats: bool
    opacity_aware_proxy_bounds: bool
    continue_after_full_buffer: bool
    transmittance_min: float
    max_trace_segments: int
    def __init__(self) -> None: ...

class SdfTraceOptions:
    max_steps: int
    relaxation: float
    eps_hit: float
    def __init__(self) -> None: ...

class SurfelRenderOptions:
    mode: SurfelRenderMode
    color_model: SurfelColorModel
    normal: bool
    sh_degree: int
    channel_count: int
    channel_chunk: int
    background_rgb: Any
    def __init__(self) -> None: ...
    @staticmethod
    def rgb(sh_degree: int = ..., background_rgb: Any = ..., normal: bool = ...) -> SurfelRenderOptions: ...
    @staticmethod
    def feature(channel_count: int) -> SurfelRenderOptions: ...

class ReflectionTraceOptions:
    deduplicate: bool
    canonical_prim_table: Array
    image_source_tolerance: float
    export_mode: int
    return_trailing: bool
    def __init__(self) -> None: ...

class ReflEpcOptions:
    expected_prim_ids: Array
    surface_group_id: Array
    surface_group_size: Array
    surface_group_members: Array
    surface_max_group_size: int
    visibility_ignore_mode: str
    plane_tolerance: float
    final_ignore_group_ids: Array
    def __init__(self) -> None: ...

class _ReflEpcFieldOptionsBase(ReflEpcOptions):
    slot_plane_point: Array
    slot_plane_normal: Array
    slot_eta_r: Array
    slot_mu_r: Array
    slot_sigma: Array
    slot_gain: Array
    tx_polarization: Array
    omega: float
    wavelength: float
    return_geom: bool
    return_endpoints: bool
    return_hit_points: bool
    return_normals: bool
    return_resolved_prim_ids: bool
    return_surface_group_ids: bool

class ReflEpcFieldOptions(_ReflEpcFieldOptionsBase): ...
class ReflEpcFieldOptionsAD(_ReflEpcFieldOptionsBase): ...

class AccumGrid:
    axis: int
    position: float
    coord0_min: float
    coord0_max: float
    coord1_min: float
    coord1_max: float
    resolution0: int
    resolution1: int
    def __init__(self) -> None: ...

class AccumOptions:
    wavelength: float
    k: float
    solid_angle_per_ray: float
    cell_area: float
    seed: int
    rr_depth: int
    rr_prob: float
    stop_threshold: float
    collect_wedges: bool
    collect_wedge_prefixes: bool
    wedge_capacity: int
    wedge_sample_stride: int
    def __init__(self) -> None: ...

class _MaterialBase:
    eta_r: Array
    sigma: Array
    gain: Array
    mu_r: Array
    valid: Array
    def __init__(self) -> None: ...

class Material(_MaterialBase): ...
class MaterialAD(_MaterialBase): ...

class DfrGrid(AccumGrid):
    cell_area: float

class DfrOptions:
    wavelength: float
    k: float
    seed: int
    samples: int
    max_order: int
    direct_samples: int
    keller_samples: int
    suffix_samples: int
    strategy_mask: int
    sample_sequence: int
    receiver_model: int
    collect_edge_use: bool
    collect_debug_counts: bool
    def __init__(self) -> None: ...

class DfrCoherentOptions:
    wavelength: float
    k: float
    max_order: int
    receiver_model: int
    select_diffraction_point: bool
    prefilter_visibility: bool
    collect_debug_counts: bool
    omega: float
    tx_pol_x: float
    tx_pol_y: float
    tx_pol_z: float
    higher_probe_radius_scale: float
    higher_probe_radius_min: float
    higher_probe_radius_max: float
    higher_filter_visibility: bool
    def __init__(self) -> None: ...

class DfrPathOptions:
    wavelength: float
    k: float
    seed: int
    sample_count: int
    max_order: int
    max_paths: int
    max_rx: int
    strategy_mask: int
    receiver_model: int
    return_geom: int
    def __init__(self) -> None: ...

class DfrMaterial(_MaterialBase): ...
class DfrMaterialAD(_MaterialBase): ...

class _IntersectionBase:
    t: Array
    p: Array
    n: Array
    geo_n: Array
    uv: Array
    barycentric: Array
    shape_id: Array
    prim_id: Array
    local_prim_id: Array
    global_prim_id: Array
    def is_valid(self) -> Array: ...

class Intersection(_IntersectionBase): ...
class IntersectionAD(_IntersectionBase): ...

class _NearestPointEdgeBase:
    distance: Array
    point: Array
    edge_t: Array
    edge_point: Array
    shape_id: Array
    edge_id: Array
    global_edge_id: Array
    is_boundary: Array
    def is_valid(self) -> Array: ...

class NearestPointEdge(_NearestPointEdgeBase): ...
class NearestPointEdgeAD(_NearestPointEdgeBase): ...

class _NearestRayEdgeBase(_NearestPointEdgeBase):
    ray_t: Array

class NearestRayEdge(_NearestRayEdgeBase): ...
class NearestRayEdgeAD(_NearestRayEdgeBase): ...

class _NearestEdgesTopKBase:
    query_count: int
    k: int
    is_valid: Array
    distances: Array
    points: Array
    edge_t: Array
    edge_points: Array
    shape_ids: Array
    edge_ids: Array
    global_edge_ids: Array
    is_boundary: Array

class NearestEdgesTopK(_NearestEdgesTopKBase): ...
class NearestEdgesTopKAD(_NearestEdgesTopKBase): ...

class _SegmentVisibilityBase:
    ray_count: int
    visible: Array

class SegmentVisibility(_SegmentVisibilityBase): ...
class SegmentVisibilityAD(_SegmentVisibilityBase): ...

class _SegmentPairVisibilityBase:
    ray_count: int
    visible_a: Array
    visible_b: Array

class SegmentPairVisibility(_SegmentPairVisibilityBase): ...
class SegmentPairVisibilityAD(_SegmentPairVisibilityBase): ...

class _AxialEdgeVisibilityBase:
    state_count: int
    any_visible: Array

class AxialEdgeVisibility(_AxialEdgeVisibilityBase): ...
class AxialEdgeVisibilityAD(_AxialEdgeVisibilityBase): ...

class _SegmentChainVisibilityBase:
    chain_count: int
    max_segments: int
    all_visible: Array
    first_blocked_segment: Array
    first_blocked_prim: Array

class SegmentChainVisibility(_SegmentChainVisibilityBase): ...
class SegmentChainVisibilityAD(_SegmentChainVisibilityBase): ...

class _ReflectionChainBase:
    max_bounces: int
    ray_count: int
    bounce_count: Array
    discovery_count: Array
    representative_ray_index: Array
    t: Array
    hit_points: Array
    geo_normals: Array
    image_sources: Array
    plane_points: Array
    plane_normals: Array
    shape_ids: Array
    prim_ids: Array
    local_prim_ids: Array
    global_prim_ids: Array
    trailing_t: Array
    trailing_prim: Array
    trailing_dir: Array
    trailing_origin: Array
    def is_valid(self) -> Array: ...

class ReflectionChain(_ReflectionChainBase): ...
class ReflectionChainAD(_ReflectionChainBase): ...

class _ReflectionBounceBase:
    t: Array
    hit_points: Array
    geo_normals: Array
    image_sources: Array
    plane_points: Array
    plane_normals: Array
    shape_ids: Array
    prim_ids: Array
    local_prim_ids: Array
    global_prim_ids: Array
    def is_valid(self) -> Array: ...

class ReflectionBounce(_ReflectionBounceBase): ...
class ReflectionBounceAD(_ReflectionBounceBase): ...

class _ReflectionTraceBase:
    max_bounces: int
    ray_count: int
    deduplicate_requested: bool
    deduplicate_applied: bool
    bounce_count: Array
    discovery_count: Array
    representative_ray_index: Array
    dedup_keep_mask: Array
    bounces: Sequence[Any]
    def is_valid(self) -> Array: ...
    def bounce(self, index: int) -> Any: ...

class ReflectionTrace(_ReflectionTraceBase):
    def bounce(self, index: int) -> ReflectionBounce: ...

class ReflectionTraceAD(_ReflectionTraceBase):
    def bounce(self, index: int) -> ReflectionBounceAD: ...

class _ReflEpcBase:
    ray_count: int
    max_bounces: int
    bounce_count: Array
    valid: Array
    path_length: Array
    reflection_points: Array
    plane_normals: Array
    prim_ids: Array
    trace_prim_ids: Array
    resolved_prim_ids: Array
    surface_group_ids: Array
    first_blocked_segment: Array
    first_blocked_prim: Array
    first_blocked_group: Array

class ReflEpc(_ReflEpcBase): ...
class ReflEpcAD(_ReflEpcBase): ...

class _ReflEpcFieldBase:
    ray_count: int
    max_bounces: int
    bounce_count: Array
    valid: Array
    field_x_re: Array
    field_x_im: Array
    field_y_re: Array
    field_y_im: Array
    field_z_re: Array
    field_z_im: Array
    path_length: Array
    tx_pos: Array
    first_hit: Array
    last_hit: Array
    hit_points: Array
    normals: Array
    resolved_prim_ids: Array
    surface_group_ids: Array

class ReflEpcField(_ReflEpcFieldBase): ...
class ReflEpcFieldAD(_ReflEpcFieldBase): ...

class _WedgeEventsBase:
    capacity: int
    count: Array
    ray_index: Array
    hit_points: Array
    normals: Array
    prim_id: Array
    directions: Array
    source_points: Array
    src_power: Array
    initial_directions: Array
    bounce_depth: Array

class WedgeEvents(_WedgeEventsBase): ...
class WedgeEventsAD(_WedgeEventsBase): ...

class _AccumResultBase:
    ray_count: int
    max_bounces: int
    grid_cell_count: int
    reflection_power: Array
    reflection_field_x: Array
    reflection_field_y: Array
    reflection_field_z: Array
    reflection_count: Array
    wedge_events: Any

class AccumResult(_AccumResultBase): ...
class AccumResultAD(_AccumResultBase): ...

class _DfrStatesBase:
    edge_index: Array
    edge_pos: Array
    edge_dir: Array
    edge_t_min: Array
    edge_t_max: Array
    n0: Array
    n1: Array
    prim0: Array
    prim1: Array
    exterior_angle: Array
    src: Array
    src_power: Array
    wi: Array
    d0: Array
    count: int
    prefix_depth: int

class DfrStates(_DfrStatesBase): ...
class DfrStatesAD(_DfrStatesBase): ...

class _DfrAccumBase:
    grid_cell_count: int
    power: Array
    field_x: Array
    field_y: Array
    field_z: Array
    direct_count: Array
    keller_count: Array
    suffix_count: Array
    vis_rejects: Array
    edge_vis_rejects: Array
    utd_rejects: Array
    edge_uses: Array

class DfrAccum(_DfrAccumBase): ...
class DfrAccumAD(_DfrAccumBase): ...

class _DfrCoherentAccumBase:
    grid_cell_count: int
    direct_field_x: Array
    direct_field_y: Array
    direct_field_z: Array
    multi_field_x: Array
    multi_field_y: Array
    multi_field_z: Array
    direct_count: Array
    multi_count: Array
    visibility_reject_count: Array
    utd_reject_count: Array

class DfrCoherentAccum(_DfrCoherentAccumBase): ...
class DfrCoherentAccumAD(_DfrCoherentAccumBase): ...

class _DfrPathsBase:
    capacity: int
    count: Array
    valid: Array
    tx_id: Array
    rx_id: Array
    order: Array
    edge0: Array
    edge1: Array
    edge2: Array
    delay: Array
    field_x: Array
    field_y: Array
    field_z: Array
    p0: Array
    p1: Array
    p2: Array

class DfrPaths(_DfrPathsBase): ...
class DfrPathsAD(_DfrPathsBase): ...

class _DfrCoherentEdgeBase:
    count: int
    edge_index: Array
    edge_pos: Array
    edge_dir: Array
    edge_line_min: Array
    edge_line_max: Array
    n0: Array
    n_face_n: Array
    wedge_n: Array
    adjacent_face0: Array
    adjacent_face1: Array
    ignore_prim_ids: Array
    ignore_k: int

class DfrCoherentEdge(_DfrCoherentEdgeBase): ...
class DfrCoherentEdgeAD(_DfrCoherentEdgeBase): ...

class _DfrCoherentCandidatePairsBase:
    count: int
    prev_index: Array
    edge_index: Array
    visibility_filtered: Array

class DfrCoherentCandidatePairs(_DfrCoherentCandidatePairsBase): ...
class DfrCoherentCandidatePairsAD(_DfrCoherentCandidatePairsBase): ...

class _DfrCoherentUtdStatesBase:
    count: int
    edge_index: Array
    edge_pos: Array
    edge_dir: Array
    edge_line_min: Array
    edge_line_max: Array
    n0: Array
    n_face_n: Array
    wedge_n: Array
    source_pos: Array
    first_interaction_pos: Array
    incident_field: Array
    incident_jones_u: Array
    incident_jones_v: Array
    path_length_prefix: Array
    order: Array
    owner_code: Array
    adjacent_face0: Array
    adjacent_face1: Array
    approximation_mode_code: Array
    face0_eta_r: Array
    face0_gain: Array
    face0_mu_r: Array
    face0_operator_m00: Array
    face0_operator_m01: Array
    face0_operator_m10: Array
    face0_operator_m11: Array
    face0_sigma: Array
    face0_use_fresnel: Array
    face1_eta_r: Array
    face1_gain: Array
    face1_mu_r: Array
    face1_operator_m00: Array
    face1_operator_m01: Array
    face1_operator_m10: Array
    face1_operator_m11: Array
    face1_sigma: Array
    face1_use_fresnel: Array
    incident_basis_k: Array
    incident_basis_u: Array
    incident_basis_v: Array
    incident_derivative_jones_u: Array
    incident_derivative_jones_v: Array
    incident_normal_derivative: Array
    incident_normal_derivative_vector_x: Array
    incident_normal_derivative_vector_y: Array
    incident_normal_derivative_vector_z: Array
    incident_vector_x: Array
    incident_vector_y: Array
    incident_vector_z: Array
    intermediate_reflection_depth: Array
    prefix_reflection_depth: Array
    r_face0: Array
    r_face_n: Array
    select_stationary_point: Array
    source_type_code: Array
    suffix_reflection_depth: Array

class DfrCoherentUtdStates(_DfrCoherentUtdStatesBase): ...
class DfrCoherentUtdStatesAD(_DfrCoherentUtdStatesBase): ...

class SceneSyncProfile:
    mesh_update_ms: float
    triangle_scatter_ms: float
    triangle_eval_ms: float
    edge_scatter_ms: float
    edge_refit_ms: float
    optix_sync_ms: float
    optix_gas_update_ms: float
    optix_ias_update_ms: float
    total_ms: float
    updated_meshes: int
    updated_vertex_meshes: int
    updated_transform_meshes: int
    updated_edge_meshes: int
    updated_edges: int

class SceneEdgeBVHStats:
    primitive_count: int
    node_count: int
    internal_node_count: int
    leaf_node_count: int
    min_leaf_size: int
    max_leaf_size: int
    avg_leaf_size: float
    max_height: int
    refit_level_count: int
    root_surface_area: float
    internal_surface_area_sum: float
    sibling_overlap_surface_area_sum: float
    sibling_overlap_surface_area_avg: float
    normalized_sibling_overlap: float
    leaf_size_histogram: dict[int, int]

class SurfelGeometry:
    center: Array
    tangent_u: Array
    tangent_v: Array
    surfel_count: int
    def __init__(self, center: Array, tangent_u: Array, tangent_v: Array) -> None: ...

class SurfelAppearance:
    opacity: Array
    values: Array
    color_model: SurfelColorModel
    sh_degree: int
    channel_count: int
    surfel_count: int
    @staticmethod
    def rgb(opacity: Array, rgb: Array) -> SurfelAppearance: ...
    @staticmethod
    def features(opacity: Array, values: Array, channel_count: int) -> SurfelAppearance: ...
    @staticmethod
    def sh(opacity: Array, coeffs: Array, sh_degree: int) -> SurfelAppearance: ...

class SurfelCloud:
    center: Array
    tangent_u: Array
    tangent_v: Array
    opacity: Array
    value: Array
    surfel_count: int
    def __init__(
        self, center: Array, tangent_u: Array, tangent_v: Array, opacity: Array, value: Array = ...
    ) -> None: ...

class _SurfelIntersectionBase:
    t: Array
    p: Array
    n: Array
    local_uv: Array
    gaussian_weight: Array
    opacity: Array
    alpha: Array
    value: Array
    surfel_id: Array
    triangle_id: Array
    def is_valid(self) -> Array: ...

class SurfelIntersection(_SurfelIntersectionBase): ...
class SurfelIntersectionAD(_SurfelIntersectionBase): ...

class _SdfIntersectionBase:
    t: Array
    hit_mask: Array
    position: Array
    normal: Array
    steps: Array
    def is_valid(self) -> Array: ...

class SdfIntersection(_SdfIntersectionBase): ...
class SdfIntersectionAD(_SdfIntersectionBase): ...

class _SurfelCompositeBase:
    intensity: Array
    alpha: Array
    transmittance: Array
    depth: Array
    candidate_count: Array
    candidate_buffer_full: Array
    def is_valid(self) -> Array: ...

class SurfelComposite(_SurfelCompositeBase): ...
class SurfelCompositeAD(_SurfelCompositeBase): ...

class _SurfelRenderBase:
    channels: Array
    rgb: Array
    normal: Array
    alpha: Array
    transmittance: Array
    depth: Array
    candidate_count: Array
    candidate_buffer_full: Array
    channel_count: int
    def is_valid(self) -> Array: ...

class SurfelRender(_SurfelRenderBase): ...
class SurfelRenderAD(_SurfelRenderBase): ...

class SurfelScene:
    def __init__(self, cloud: SurfelCloud | SurfelGeometry, options: SurfelTraceOptions = ...) -> None: ...
    def build(self) -> None: ...
    def is_ready(self) -> bool: ...
    @property
    def build_count(self) -> int: ...
    @property
    def surfel_count(self) -> int: ...
    @property
    def triangle_count(self) -> int: ...
    @overload
    def intersect(self, ray: Ray, active: Mask = ...) -> SurfelIntersection: ...
    @overload
    def intersect(self, ray: RayAD, active: Mask = ...) -> SurfelIntersectionAD: ...
    def shadow_test(self, ray: Ray | RayAD, active: Mask = ...) -> Array: ...
    def visible(self, start: Array, end: Array, active: Mask = ...) -> Array: ...
    def trace_reflections(
        self, ray: Ray | RayAD, max_bounces: int, active: Mask = ...
    ) -> ReflectionChain | ReflectionChainAD: ...
    def composite_alpha(self, ray: Ray | RayAD, active: Mask = ...) -> SurfelComposite | SurfelCompositeAD: ...
    def composite_alpha_reference(
        self, ray: Ray | RayAD, active: Mask = ...
    ) -> SurfelComposite | SurfelCompositeAD: ...
    def render(
        self, ray: Ray | RayAD, render_options: SurfelRenderOptions = ..., active: Mask = ...
    ) -> SurfelRender | SurfelRenderAD: ...
    def update_geometry(self, geometry: SurfelGeometry) -> None: ...
    def update_appearance(self, appearance: SurfelAppearance) -> None: ...

class SdfGrid:
    def __init__(
        self, values: Array, nx: int, ny: int, nz: int, position: Array, rotation: Array, scale: Array
    ) -> None: ...
    nx: int
    ny: int
    nz: int
    value_count: int
    values: Array
    position: Array
    rotation: Array
    scale: Array
    @overload
    def intersect(self, ray: Ray, options: SdfTraceOptions = ..., active: Mask = ...) -> SdfIntersection: ...
    @overload
    def intersect(self, ray: RayAD, options: SdfTraceOptions = ..., active: Mask = ...) -> SdfIntersectionAD: ...
    def visible(self, start: Array, end: Array, options: SdfTraceOptions = ..., active: Mask = ...) -> Array: ...
    @overload
    def trace_reflections(
        self, ray: Ray, max_bounces: int, options: SdfTraceOptions = ..., active: Mask = ...
    ) -> ReflectionChain: ...
    @overload
    def trace_reflections(
        self, ray: RayAD, max_bounces: int, options: SdfTraceOptions = ..., active: Mask = ...
    ) -> ReflectionChainAD: ...

class Mesh:
    def __init__(
        self, v: Array = ..., f: Array = ..., uv: Array = ..., f_uv: Array = ..., verbose: bool = ...
    ) -> None: ...
    def build(self) -> None: ...
    def set_transform(self, mat: Matrix4f, set_left: bool = ...) -> None: ...
    def append_transform(self, mat: Matrix4f, append_left: bool = ...) -> None: ...
    def edge_indices(self) -> tuple[Array, Array, Array, Array, Array]: ...
    def secondary_edges(self) -> SecondaryEdgeInfo: ...
    @property
    def num_vertices(self) -> int: ...
    @property
    def num_faces(self) -> int: ...
    to_world: Matrix4f
    to_world_left: Matrix4f
    to_world_right: Matrix4f
    vertex_positions: Array
    @property
    def vertex_positions_world(self) -> Array: ...
    @property
    def vertex_normals(self) -> Array: ...
    vertex_uv: Array
    face_indices: Array
    face_uv_indices: Array
    use_face_normals: bool
    edges_enabled: bool

class MixedScene:
    def __init__(self, trace_backend: str = ..., edge_bvh_backend: str = ...) -> None: ...
    def add_mesh(self, mesh: Mesh, dynamic: bool = ...) -> int: ...
    def add_sdf(self, grid: SdfGrid, options: SdfTraceOptions = ...) -> int: ...
    def add_surfel(self, cloud: SurfelCloud, options: SurfelTraceOptions = ...) -> int: ...
    def build(self) -> None: ...
    def is_ready(self) -> bool: ...
    @property
    def num_meshes(self) -> int: ...
    @property
    def num_sdfs(self) -> int: ...
    @property
    def num_surfel_scenes(self) -> int: ...
    @overload
    def intersect(self, ray: Ray, active: Mask = ..., flags: RayFlags = ...) -> Intersection: ...
    @overload
    def intersect(self, ray: RayAD, active: Mask = ..., flags: RayFlags = ...) -> IntersectionAD: ...
    @overload
    def visible(self, start: Array, end: Array, active: Mask = ...) -> SegmentVisibility: ...
    @overload
    def visible(self, start: Array, end: Array, active: Mask = ...) -> SegmentVisibilityAD: ...
    @overload
    def trace_reflections(self, ray: Ray, max_bounces: int, active: Mask = ...) -> ReflectionChain: ...
    @overload
    def trace_reflections(self, ray: RayAD, max_bounces: int, active: Mask = ...) -> ReflectionChainAD: ...
    @overload
    def transmittance(self, ray: Ray, active: Mask = ...) -> Array: ...
    @overload
    def transmittance(self, ray: RayAD, active: Mask = ...) -> Array: ...

class Scene:
    def __init__(self, edge_bvh_backend: str = ..., trace_backend: str = ...) -> None: ...
    def add_mesh(self, mesh: Mesh, dynamic: bool = ...) -> int: ...
    def build(self) -> None: ...
    def update_mesh_vertices(self, mesh_id: int, positions: Array) -> None: ...
    def set_mesh_transform(self, mesh_id: int, mat: Matrix4f, set_left: bool = ...) -> None: ...
    def append_mesh_transform(self, mesh_id: int, mat: Matrix4f, append_left: bool = ...) -> None: ...
    def set_edge_mask(self, mask: Mask) -> None: ...
    def sync(self) -> None: ...
    def is_ready(self) -> bool: ...
    def has_pending_updates(self) -> bool: ...
    @property
    def last_sync_profile(self) -> SceneSyncProfile: ...
    @property
    def edge_bvh_backend(self) -> str: ...
    def trace_backend_name(self) -> str: ...
    def capabilities(self) -> dict[str, Any]: ...
    @property
    def num_meshes(self) -> int: ...
    @property
    def version(self) -> int: ...
    @property
    def edge_version(self) -> int: ...
    def edge_info(self) -> SceneEdgeInfo: ...
    def edge_bvh_stats(self) -> SceneEdgeBVHStats: ...
    def edge_topology(self) -> SceneEdgeTopology: ...
    def edge_mask(self) -> Array: ...
    def mesh_face_offsets(self) -> Array: ...
    def mesh_edge_offsets(self) -> Array: ...
    def mesh_vertex_offsets(self) -> Array: ...
    def global_geometry(self) -> SceneGeometry: ...
    def triangle_edge_indices(self, prim_id: Array, global_: bool = ...) -> tuple[Array, Array, Array]: ...
    def edge_adjacent_faces(self, edge_id: Array, global_: bool = ...) -> tuple[Array, Array]: ...
    @overload
    def intersect(self, ray: Ray, active: Mask = ..., flags: RayFlags = ...) -> Intersection: ...
    @overload
    def intersect(self, ray: RayAD, active: Mask = ..., flags: RayFlags = ...) -> IntersectionAD: ...
    def _cuda_first_blocker_selftest(
        self, origin: Array, direction: Array, tmax: Array, ignore: list[int] = ...
    ) -> list[int]: ...
    def trace_reflections(
        self, ray: Ray | RayAD, max_bounces: int, *args: Any, **kwargs: Any
    ) -> ReflectionTrace | ReflectionTraceAD | ReflectionChain | ReflectionChainAD: ...
    def trace_refl_epc(
        self, ray: Ray, receiver: Array, max_bounces: int, options: ReflEpcOptions | None = ..., active: Mask = ...
    ) -> ReflEpc: ...
    def trace_refl_epc_field(
        self,
        source: Ray | RayAD | Array,
        receiver: Array,
        max_bounces: int,
        options: ReflEpcFieldOptions | ReflEpcFieldOptionsAD,
        active: Mask = ...,
    ) -> ReflEpcField | ReflEpcFieldAD: ...
    def trace_dfr_paths(
        self,
        tx_positions: Array,
        rx_positions: Array,
        states: DfrStates | DfrStatesAD,
        material: DfrMaterial | DfrMaterialAD,
        options: DfrPathOptions = ...,
        active: Mask = ...,
    ) -> DfrPaths | DfrPathsAD: ...
    def accumulate_reflections(
        self,
        ray: Ray | RayAD,
        tx_position: Array,
        grid: AccumGrid,
        material: Material | MaterialAD,
        max_bounces: int,
        options: AccumOptions = ...,
        active: Mask = ...,
        tx_polarization: Array = ...,
    ) -> AccumResult | AccumResultAD: ...
    def accum_dfr_direct(
        self,
        states: DfrStates | DfrStatesAD,
        grid: DfrGrid,
        material: DfrMaterial | DfrMaterialAD,
        options: DfrOptions = ...,
        active: Mask = ...,
    ) -> DfrAccum | DfrAccumAD: ...
    def accum_dfr(
        self,
        initial_states: DfrStates | DfrStatesAD,
        recursive_states: DfrStates | DfrStatesAD,
        grid: DfrGrid,
        material: DfrMaterial | DfrMaterialAD,
        options: DfrOptions = ...,
        active: Mask = ...,
    ) -> DfrAccum | DfrAccumAD: ...
    def accum_dfr_coherent_direct(
        self,
        states: DfrStates | DfrStatesAD,
        grid: DfrGrid,
        material: DfrMaterial | DfrMaterialAD,
        options: DfrCoherentOptions = ...,
        active: Mask = ...,
    ) -> DfrCoherentAccum | DfrCoherentAccumAD: ...
    def build_dfr_coherent_tx_states(
        self,
        edges: DfrCoherentEdge | DfrCoherentEdgeAD,
        tx_position: Array,
        material: DfrMaterial | DfrMaterialAD,
        options: DfrCoherentOptions = ...,
        active: Mask = ...,
    ) -> DfrCoherentUtdStates | DfrCoherentUtdStatesAD: ...
    def build_dfr_coherent_higher_candidates(
        self,
        prev_states: DfrCoherentUtdStates | DfrCoherentUtdStatesAD,
        edges: DfrCoherentEdge | DfrCoherentEdgeAD,
        global_to_local_edge_index: Array,
        options: DfrCoherentOptions = ...,
        active: Mask = ...,
    ) -> DfrCoherentCandidatePairs | DfrCoherentCandidatePairsAD: ...
    @overload
    def nearest_edge(self, point: Array, active: Mask = ...) -> NearestPointEdge | NearestPointEdgeAD: ...
    @overload
    def nearest_edge(self, point: Ray, active: Mask = ...) -> NearestRayEdge: ...
    @overload
    def nearest_edge(self, point: RayAD, active: Mask = ...) -> NearestRayEdgeAD: ...
    def nearest_edges(self, point: Array, k: int, active: Mask = ...) -> NearestEdgesTopK | NearestEdgesTopKAD: ...
    def shadow_test(self, ray: Ray | RayAD, active: Mask = ...) -> Array: ...
    def visible(
        self, start: Array, end: Array, ignore_prim_ids: Array = ..., active: Mask = ...
    ) -> SegmentVisibility | SegmentVisibilityAD: ...
    def visible_pair(
        self, start: Array, end_a: Array, end_b: Array, ignore_prim_ids: Array = ..., active: Mask = ...
    ) -> SegmentPairVisibility | SegmentPairVisibilityAD: ...
    def visible_edge(
        self,
        src: Array,
        edge_pos: Array,
        edge_dir: Array,
        edge_t_min: Array,
        edge_t_max: Array,
        sample_fractions: Sequence[float] = ...,
        active: Mask = ...,
    ) -> AxialEdgeVisibility | AxialEdgeVisibilityAD: ...
    def visible_chain(
        self, points: Array, chain_length: Array, ignore_prim_per_segment: Array = ..., active: Mask = ...
    ) -> SegmentChainVisibility | SegmentChainVisibilityAD: ...
