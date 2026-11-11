# Copyright Xingyu Chen.
# Exposes the Dr.Jit C Python API.

"""Differentiable geometry queries built on Dr.Jit and OptiX."""

import enum
from typing import overload

import drjit.cuda
import drjit.cuda.ad
import drjit.scalar

def device_count() -> int:
    """
    Return the number of Dr.Jit-compatible CUDA devices visible to RayD.

    Raises RuntimeError when there is none.
    """

def current_device() -> int:
    """
    Return the current thread's active Dr.Jit CUDA device index.

    Raises RuntimeError when no Dr.Jit-compatible CUDA device is available.
    """

def native_launch_audit_clear() -> None:
    """Reset every native launch audit counter to zero."""

def native_launch_audit() -> dict[str, dict[str, object]]:
    """
    Return the native launch audit counters grouped by operation stage.

    The result maps each stage ('unknown', 'build', 'sync', 'intersect', 'trace_reflections', 'accum_refl', 'accum_dfr', 'surfel_trace') to a dict of CUDA, CUB, Dr.Jit, and OptiX launch and copy counts plus a 'kernels' list with per-kernel launch, thread, and item totals. OptiX launch times are recorded only when the environment variable RAYD_NATIVE_LAUNCH_AUDIT_TIMING is set to a value other than 0. Counters accumulate until native_launch_audit_clear().
    """

def set_device(device: int, initialize_optix: bool = True) -> int:
    """
    Set the current thread's active CUDA device for RayD.

    Call this before constructing RayD meshes, scenes, or Dr.Jit arrays that you intend to use with them. When initialize_optix=True, RayD also initializes the OptiX device context when OptiX is available; otherwise CUDA-only operation remains available. Returns the new active device index. Raises RuntimeError when no device is available or device is out of range.
    """

def optix_available() -> bool:
    """
    Return True when a usable OptiX driver is present on this system.

    This is a non-throwing capability probe (safe on machines without OptiX). Set the environment variable RAYD_DISABLE_OPTIX=1 to force a False result, which lets an OptiX-capable machine exercise the OptiX-less code paths.
    """

class Ray:
    """
    Batch of rays with origins o, directions d, and maximum distances tmax.

    Input record. Ray(o, d) sets tmax to infinity. Directions need not be normalized. Use RayAD for differentiable queries.
    """

    @overload
    def __init__(self) -> None:
        """Create a ray batch. With o and d, tmax is infinity."""

    @overload
    def __init__(self, arg0: drjit.cuda.Array3f, arg1: drjit.cuda.Array3f, /) -> None: ...
    def reversed(self) -> Ray:
        """Return the same rays with flipped directions."""

    @property
    def o(self) -> drjit.cuda.Array3f:
        """Ray origins."""

    @o.setter
    def o(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def d(self) -> drjit.cuda.Array3f:
        """Ray directions; they need not be normalized."""

    @d.setter
    def d(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def tmax(self) -> drjit.cuda.Float:
        """Maximum ray distance; hits beyond it are ignored."""

    @tmax.setter
    def tmax(self, arg: drjit.cuda.Float, /) -> None: ...

class RayAD:
    """
    Batch of rays with Dr.Jit AD arrays for o, d, and tmax.

    Input record. Passing a RayAD selects the differentiable overload of a query.
    """

    @overload
    def __init__(self) -> None:
        """Create an AD ray batch. With o and d, tmax is infinity."""

    @overload
    def __init__(self, arg0: drjit.cuda.ad.Array3f, arg1: drjit.cuda.ad.Array3f, /) -> None: ...
    def reversed(self) -> RayAD:
        """Return the same rays with flipped directions."""

    @property
    def o(self) -> drjit.cuda.ad.Array3f:
        """Ray origins."""

    @o.setter
    def o(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def d(self) -> drjit.cuda.ad.Array3f:
        """Ray directions; they need not be normalized."""

    @d.setter
    def d(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def tmax(self) -> drjit.cuda.ad.Float:
        """Maximum ray distance; hits beyond it are ignored."""

    @tmax.setter
    def tmax(self, arg: drjit.cuda.ad.Float, /) -> None: ...

class SecondaryEdgeInfo:
    """
    World-space per-edge data of one mesh, as used by the edge queries.

    Result record returned by Mesh.secondary_edges().
    """

    def __init__(self) -> None: ...
    def size(self) -> int:
        """Return the number of edges."""

    @property
    def start(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def edge(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def normal0(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def normal1(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def opposite(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def is_boundary(self) -> drjit.cuda.ad.Bool: ...

class SceneEdgeInfo:
    """
    Scene-global per-edge geometry and ids.

    Result record returned by Scene.edge_info().
    """

    def __init__(self) -> None: ...
    def size(self) -> int:
        """Return the number of edges."""

    @property
    def start(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def edge(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def end(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def length(self) -> drjit.cuda.ad.Float: ...
    @property
    def normal0(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def normal1(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def is_boundary(self) -> drjit.cuda.ad.Bool: ...
    @property
    def shape_id(self) -> drjit.cuda.Int: ...
    @property
    def local_edge_id(self) -> drjit.cuda.Int: ...
    @property
    def global_edge_id(self) -> drjit.cuda.Int: ...

class SceneEdgeTopology:
    """
    Scene-global edge connectivity, one entry per edge.

    Result record returned by Scene.edge_topology(). Fields ending in _global index the scene-global vertex and face buffers; the others are per-mesh. face1, opposite_vertex1, and their global forms are -1 for boundary edges.
    """

    def __init__(self) -> None: ...
    def size(self) -> int:
        """Return the number of edges."""

    @property
    def v0(self) -> drjit.cuda.Int: ...
    @property
    def v1(self) -> drjit.cuda.Int: ...
    @property
    def v0_global(self) -> drjit.cuda.Int: ...
    @property
    def v1_global(self) -> drjit.cuda.Int: ...
    @property
    def face0_local(self) -> drjit.cuda.Int: ...
    @property
    def face1_local(self) -> drjit.cuda.Int: ...
    @property
    def face0_global(self) -> drjit.cuda.Int: ...
    @property
    def face1_global(self) -> drjit.cuda.Int: ...
    @property
    def opposite_vertex0(self) -> drjit.cuda.Int: ...
    @property
    def opposite_vertex1(self) -> drjit.cuda.Int: ...
    @property
    def opposite_vertex0_global(self) -> drjit.cuda.Int: ...
    @property
    def opposite_vertex1_global(self) -> drjit.cuda.Int: ...

class SceneGeometry:
    """
    All meshes' world-space vertices and faces flattened into one scene-global index space.

    Result record returned by Scene.global_geometry().
    """

    def __init__(self) -> None: ...
    def vertex_count(self) -> int:
        """Return the number of scene-global vertices."""

    def face_count(self) -> int:
        """Return the number of scene-global faces."""

    @property
    def vertices(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def faces(self) -> drjit.cuda.Array3i: ...
    @property
    def face_normal(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def shape_id(self) -> drjit.cuda.Int: ...
    @property
    def local_prim_id(self) -> drjit.cuda.Int: ...
    @property
    def global_prim_id(self) -> drjit.cuda.Int: ...

class RayFlags(enum.IntEnum):
    """
    Flags that select which Intersection fields Scene.intersect() computes.

    Geometric selects the geometric normal geo_n, ShadingN the interpolated shading normal n, and UV the texture coordinates uv. All sets every flag and Empty none. t and the ids are always filled.
    """

    Empty = 0

    Geometric = 1

    ShadingN = 2

    UV = 4

    All = 7

class SurfelPrimitiveMode(enum.Enum):
    """
    Proxy triangle geometry used to find surfel candidates.

    Icosahedron20 builds a thin 20-triangle shell per surfel, QuadTriangles two triangles, and SingleTriangle one.
    """

    Icosahedron20 = 0

    QuadTriangles = 1

    SingleTriangle = 2

class SurfelColorModel(enum.Enum):
    """
    How SurfelAppearance values are interpreted.

    ConstantRGB stores three values per surfel, SH stores spherical-harmonic RGB coefficients, and FeatureChannels stores channel_count arbitrary values.
    """

    ConstantRGB = 0

    SH = 1

    FeatureChannels = 2

class SurfelRenderMode(enum.Enum):
    """
    Output of SurfelScene.render(): Alpha, Depth, RGB, RGBDepth, or Feature channels.
    """

    Alpha = 0

    Depth = 1

    RGB = 2

    RGBDepth = 3

    Feature = 4

class SurfelTraceOptions:
    """
    Options for surfel tracing and alpha compositing.

    Options record for SurfelScene and MixedScene.add_surfel(). Values are checked when the SurfelScene is created or built, and an invalid value raises RuntimeError.
    """

    def __init__(self) -> None: ...
    @property
    def alpha_min(self) -> float:
        """Minimum Gaussian alpha a surfel contributes; must lie in (0, 1)."""

    @alpha_min.setter
    def alpha_min(self, arg: float, /) -> None: ...
    @property
    def cutoff(self) -> float:
        """Cap on the proxy radius in local Gaussian units; inf disables it."""

    @cutoff.setter
    def cutoff(self, arg: float, /) -> None: ...
    @property
    def alpha_cap(self) -> float:
        """Upper bound on a surfel's alpha; must lie in (0, 1]."""

    @alpha_cap.setter
    def alpha_cap(self, arg: float, /) -> None: ...
    @property
    def proxy_epsilon(self) -> float:
        """
        Normal-axis thickness of the Icosahedron20 proxy relative to the Gaussian radius; must be positive.
        """

    @proxy_epsilon.setter
    def proxy_epsilon(self, arg: float, /) -> None: ...
    @property
    def max_candidate_hits(self) -> int:
        """Per-ray candidate buffer size for alpha compositing."""

    @max_candidate_hits.setter
    def max_candidate_hits(self, arg: int, /) -> None: ...
    @property
    def primitive_mode(self) -> SurfelPrimitiveMode:
        """Proxy geometry per surfel."""

    @primitive_mode.setter
    def primitive_mode(self, arg: SurfelPrimitiveMode, /) -> None: ...
    @property
    def face_forward(self) -> bool:
        """Flip each surfel normal to face the incoming ray."""

    @face_forward.setter
    def face_forward(self, arg: bool, /) -> None: ...
    @property
    def single_launch(self) -> bool:
        """
        Use the native OptiX pipeline for detached intersect and composite calls.
        """

    @single_launch.setter
    def single_launch(self, arg: bool, /) -> None: ...
    @property
    def collect_candidate_stats(self) -> bool:
        """Collect per-ray candidate-buffer occupancy diagnostics."""

    @collect_candidate_stats.setter
    def collect_candidate_stats(self, arg: bool, /) -> None: ...
    @property
    def opacity_aware_proxy_bounds(self) -> bool:
        """Size each proxy from the surfel's opacity."""

    @opacity_aware_proxy_bounds.setter
    def opacity_aware_proxy_bounds(self, arg: bool, /) -> None: ...
    @property
    def continue_after_full_buffer(self) -> bool:
        """
        Keep tracing after the candidate buffer fills, until transmittance drops below transmittance_min.
        """

    @continue_after_full_buffer.setter
    def continue_after_full_buffer(self, arg: bool, /) -> None: ...
    @property
    def transmittance_min(self) -> float:
        """Transmittance at which compositing stops; must lie in [0, 1)."""

    @transmittance_min.setter
    def transmittance_min(self, arg: float, /) -> None: ...
    @property
    def max_trace_segments(self) -> int:
        """
        Maximum trace segments per ray when continue_after_full_buffer is set; must be positive.
        """

    @max_trace_segments.setter
    def max_trace_segments(self, arg: int, /) -> None: ...

class SdfTraceOptions:
    """
    Sphere-tracing options for SdfGrid queries.

    Options record for SdfGrid.intersect(), visible(), trace_reflections(), and MixedScene.add_sdf().
    """

    def __init__(self) -> None: ...
    @property
    def max_steps(self) -> int:
        """Maximum sphere-tracing steps; must be at least 1."""

    @max_steps.setter
    def max_steps(self, arg: int, /) -> None: ...
    @property
    def relaxation(self) -> float:
        """Step relaxation factor in (0, 1]."""

    @relaxation.setter
    def relaxation(self, arg: float, /) -> None: ...
    @property
    def eps_hit(self) -> float:
        """
        Hit threshold in world units. A non-positive value selects 1e-3 of the smallest voxel size.
        """

    @eps_hit.setter
    def eps_hit(self, arg: float, /) -> None: ...

class SurfelRenderOptions:
    """
    Options for SurfelScene.render().

    Options record. Use the rgb() or feature() factory for the common configurations.
    """

    def __init__(self) -> None: ...
    @staticmethod
    def rgb(
        sh_degree: int = 0, background_rgb: drjit.scalar.Array3f = ..., normal: bool = False
    ) -> SurfelRenderOptions:
        """
        Return options for an RGB render.

        sh_degree > 0 selects the SH color model, otherwise ConstantRGB. background_rgb is the background color and normal requests the alpha-weighted normal.
        """

    @staticmethod
    def feature(channel_count: int) -> SurfelRenderOptions:
        """Return options for a render of channel_count feature channels."""

    @property
    def mode(self) -> SurfelRenderMode:
        """What to render."""

    @mode.setter
    def mode(self, arg: SurfelRenderMode, /) -> None: ...
    @property
    def color_model(self) -> SurfelColorModel:
        """Color model; it must match the scene's SurfelAppearance."""

    @color_model.setter
    def color_model(self, arg: SurfelColorModel, /) -> None: ...
    @property
    def normal(self) -> bool:
        """Also output the alpha-weighted surfel normal."""

    @normal.setter
    def normal(self, arg: bool, /) -> None: ...
    @property
    def sh_degree(self) -> int:
        """Spherical-harmonic degree to evaluate, in [0, 3]."""

    @sh_degree.setter
    def sh_degree(self, arg: int, /) -> None: ...
    @property
    def channel_count(self) -> int:
        """Number of feature channels for the Feature mode."""

    @channel_count.setter
    def channel_count(self, arg: int, /) -> None: ...
    @property
    def background_rgb(self) -> drjit.scalar.Array3f:
        """Background color for the RGB modes."""

    @background_rgb.setter
    def background_rgb(self, arg: drjit.scalar.Array3f, /) -> None: ...

class ReflectionExportMode(enum.IntEnum):
    """
    Payload exported by Scene.trace_reflections() with symbolic=False.

    FULL exports every field, MINIMAL exports bounce counts plus per-slot ids and t, and COUNT_ONLY exports bounce counts only.
    """

    FULL = 0

    MINIMAL = 1

    COUNT_ONLY = 2

class ReflectionTraceOptions:
    """
    Options for Scene.trace_reflections().

    Options record.
    """

    def __init__(self) -> None: ...
    @property
    def deduplicate(self) -> bool:
        """
        Merge paths with the same reflector sequence. Needs symbolic=False and the FULL export mode.
        """

    @deduplicate.setter
    def deduplicate(self, arg: bool, /) -> None: ...
    @property
    def canonical_prim_table(self) -> drjit.cuda.Int:
        """
        Optional map from primitive id to a canonical id used when deduplicating.
        """

    @canonical_prim_table.setter
    def canonical_prim_table(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def image_source_tolerance(self) -> float:
        """Distance tolerance when comparing image sources during deduplication."""

    @image_source_tolerance.setter
    def image_source_tolerance(self, arg: float, /) -> None: ...
    @property
    def export_mode(self) -> ReflectionExportMode:
        """Exported payload; AD traces always export FULL."""

    @export_mode.setter
    def export_mode(self, arg: ReflectionExportMode, /) -> None: ...
    @property
    def return_trailing(self) -> bool:
        """Trace and export the segment after the last reflection."""

    @return_trailing.setter
    def return_trailing(self, arg: bool, /) -> None: ...

class ReflEpcOptions:
    """
    Options for Scene.trace_refl_epc(): the expected reflector sequence and the surface-group tables used for visibility.

    Options record.
    """

    def __init__(self) -> None: ...
    @property
    def expected_prim_ids(self) -> drjit.cuda.Int:
        """
        Expected reflector per slot, n_rays * max_bounces entries. trace_refl_epc() also accepts an empty array.
        """

    @expected_prim_ids.setter
    def expected_prim_ids(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def surface_group_id(self) -> drjit.cuda.Int:
        """Surface-group id per scene triangle."""

    @surface_group_id.setter
    def surface_group_id(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def surface_group_size(self) -> drjit.cuda.Int:
        """Member count of each surface group."""

    @surface_group_size.setter
    def surface_group_size(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def surface_group_members(self) -> drjit.cuda.Int:
        """
        Flattened member table with at least surface_group_count * surface_max_group_size entries.
        """

    @surface_group_members.setter
    def surface_group_members(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def surface_max_group_size(self) -> int:
        """Row width of surface_group_members."""

    @surface_max_group_size.setter
    def surface_max_group_size(self, arg: int, /) -> None: ...
    @property
    def visibility_ignore_mode(self) -> str:
        """
        How visibility checks ignore the reflecting surfaces: 'primitive' or 'surface_group'. 'surface_group' needs the surface-group tables.
        """

    @visibility_ignore_mode.setter
    def visibility_ignore_mode(self, arg: str, /) -> None: ...
    @property
    def plane_tolerance(self) -> float:
        """Relative tolerance for a reflection point to lie on its plane."""

    @plane_tolerance.setter
    def plane_tolerance(self, arg: float, /) -> None: ...
    @property
    def final_ignore_group_ids(self) -> drjit.cuda.Int:
        """
        Surface groups ignored on the last segment to the receiver, one or n_rays entries. Needs visibility_ignore_mode='surface_group'.
        """

    @final_ignore_group_ids.setter
    def final_ignore_group_ids(self, arg: drjit.cuda.Int, /) -> None: ...

class ReflEpcFieldOptions(ReflEpcOptions):
    """
    Options for Scene.trace_refl_epc_field() with detached arrays.

    Options record. Extends ReflEpcOptions with the per-slot reflecting planes and materials, the source polarization and frequency, and the output selectors. Per-slot arrays have n_rays * max_bounces entries.
    """

    def __init__(self) -> None: ...
    @property
    def slot_plane_point(self) -> drjit.cuda.Array3f:
        """Point on the reflecting plane of each slot."""

    @slot_plane_point.setter
    def slot_plane_point(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def slot_plane_normal(self) -> drjit.cuda.Array3f:
        """Normal of the reflecting plane of each slot."""

    @slot_plane_normal.setter
    def slot_plane_normal(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def slot_eta_r(self) -> drjit.cuda.Float:
        """Relative permittivity of each slot."""

    @slot_eta_r.setter
    def slot_eta_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def slot_mu_r(self) -> drjit.cuda.Float:
        """Relative permeability of each slot."""

    @slot_mu_r.setter
    def slot_mu_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def slot_sigma(self) -> drjit.cuda.Float:
        """Conductivity of each slot."""

    @slot_sigma.setter
    def slot_sigma(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def slot_gain(self) -> drjit.cuda.Float:
        """Extra gain factor of each slot."""

    @slot_gain.setter
    def slot_gain(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def tx_polarization(self) -> drjit.cuda.Array3f:
        """Transmitter polarization vector, one or n_rays entries."""

    @tx_polarization.setter
    def tx_polarization(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def omega(self) -> float:
        """Angular frequency in rad/s; must be positive."""

    @omega.setter
    def omega(self, arg: float, /) -> None: ...
    @property
    def wavelength(self) -> float:
        """Wavelength in world units; must be positive."""

    @wavelength.setter
    def wavelength(self, arg: float, /) -> None: ...
    @property
    def return_geom(self) -> bool:
        """Master switch for the per-slot geometry outputs."""

    @return_geom.setter
    def return_geom(self, arg: bool, /) -> None: ...
    @property
    def return_endpoints(self) -> bool:
        """Output tx_pos, first_hit, and last_hit."""

    @return_endpoints.setter
    def return_endpoints(self, arg: bool, /) -> None: ...
    @property
    def return_hit_points(self) -> bool:
        """Output per-slot hit points when return_geom is set."""

    @return_hit_points.setter
    def return_hit_points(self, arg: bool, /) -> None: ...
    @property
    def return_normals(self) -> bool:
        """Output per-slot normals when return_geom is set."""

    @return_normals.setter
    def return_normals(self, arg: bool, /) -> None: ...
    @property
    def return_resolved_prim_ids(self) -> bool:
        """Output resolved primitive ids when return_geom is set."""

    @return_resolved_prim_ids.setter
    def return_resolved_prim_ids(self, arg: bool, /) -> None: ...
    @property
    def return_surface_group_ids(self) -> bool:
        """Output surface-group ids when return_geom is set."""

    @return_surface_group_ids.setter
    def return_surface_group_ids(self, arg: bool, /) -> None: ...

class ReflEpcFieldOptionsAD(ReflEpcOptions):
    """
    Options for Scene.trace_refl_epc_field() whose per-slot planes, materials, and polarization are AD arrays.

    Options record. Same fields as ReflEpcFieldOptions; use it to differentiate with respect to those values.
    """

    def __init__(self) -> None: ...
    @property
    def slot_plane_point(self) -> drjit.cuda.ad.Array3f:
        """Point on the reflecting plane of each slot."""

    @slot_plane_point.setter
    def slot_plane_point(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def slot_plane_normal(self) -> drjit.cuda.ad.Array3f:
        """Normal of the reflecting plane of each slot."""

    @slot_plane_normal.setter
    def slot_plane_normal(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def slot_eta_r(self) -> drjit.cuda.ad.Float:
        """Relative permittivity of each slot."""

    @slot_eta_r.setter
    def slot_eta_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def slot_mu_r(self) -> drjit.cuda.ad.Float:
        """Relative permeability of each slot."""

    @slot_mu_r.setter
    def slot_mu_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def slot_sigma(self) -> drjit.cuda.ad.Float:
        """Conductivity of each slot."""

    @slot_sigma.setter
    def slot_sigma(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def slot_gain(self) -> drjit.cuda.ad.Float:
        """Extra gain factor of each slot."""

    @slot_gain.setter
    def slot_gain(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def tx_polarization(self) -> drjit.cuda.ad.Array3f:
        """Transmitter polarization vector, one or n_rays entries."""

    @tx_polarization.setter
    def tx_polarization(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def omega(self) -> float:
        """Angular frequency in rad/s; must be positive."""

    @omega.setter
    def omega(self, arg: float, /) -> None: ...
    @property
    def wavelength(self) -> float:
        """Wavelength in world units; must be positive."""

    @wavelength.setter
    def wavelength(self, arg: float, /) -> None: ...
    @property
    def return_geom(self) -> bool:
        """Master switch for the per-slot geometry outputs."""

    @return_geom.setter
    def return_geom(self, arg: bool, /) -> None: ...
    @property
    def return_endpoints(self) -> bool:
        """Output tx_pos, first_hit, and last_hit."""

    @return_endpoints.setter
    def return_endpoints(self, arg: bool, /) -> None: ...
    @property
    def return_hit_points(self) -> bool:
        """Output per-slot hit points when return_geom is set."""

    @return_hit_points.setter
    def return_hit_points(self, arg: bool, /) -> None: ...
    @property
    def return_normals(self) -> bool:
        """Output per-slot normals when return_geom is set."""

    @return_normals.setter
    def return_normals(self, arg: bool, /) -> None: ...
    @property
    def return_resolved_prim_ids(self) -> bool:
        """Output resolved primitive ids when return_geom is set."""

    @return_resolved_prim_ids.setter
    def return_resolved_prim_ids(self, arg: bool, /) -> None: ...
    @property
    def return_surface_group_ids(self) -> bool:
        """Output surface-group ids when return_geom is set."""

    @return_surface_group_ids.setter
    def return_surface_group_ids(self, arg: bool, /) -> None: ...

class Intersection:
    """
    Closest ray-triangle hit per ray.

    Result record of Scene.intersect() and MixedScene.intersect(). Missed rays have t = inf and ids = -1; check is_valid() before reading other fields.
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def t(self) -> drjit.cuda.Float: ...
    @property
    def p(self) -> drjit.cuda.Array3f: ...
    @property
    def n(self) -> drjit.cuda.Array3f: ...
    @property
    def geo_n(self) -> drjit.cuda.Array3f: ...
    @property
    def uv(self) -> drjit.cuda.Array2f: ...
    @property
    def barycentric(self) -> drjit.cuda.Array3f: ...
    @property
    def shape_id(self) -> drjit.cuda.Int: ...
    @property
    def prim_id(self) -> drjit.cuda.Int: ...
    @property
    def local_prim_id(self) -> drjit.cuda.Int: ...
    @property
    def global_prim_id(self) -> drjit.cuda.Int: ...

class IntersectionAD:
    """AD variant of Intersection, returned when intersect() gets a RayAD."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def t(self) -> drjit.cuda.ad.Float: ...
    @property
    def p(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def n(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def geo_n(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def uv(self) -> drjit.cuda.ad.Array2f: ...
    @property
    def barycentric(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def shape_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def prim_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def local_prim_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def global_prim_id(self) -> drjit.cuda.ad.Int: ...

class SurfelIntersection:
    """
    Closest surfel hit per ray.

    Result record of SurfelScene.intersect(). Missed rays have surfel_id = -1.
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def t(self) -> drjit.cuda.Float: ...
    @property
    def p(self) -> drjit.cuda.Array3f: ...
    @property
    def n(self) -> drjit.cuda.Array3f: ...
    @property
    def local_uv(self) -> drjit.cuda.Array2f: ...
    @property
    def gaussian_weight(self) -> drjit.cuda.Float: ...
    @property
    def opacity(self) -> drjit.cuda.Float: ...
    @property
    def alpha(self) -> drjit.cuda.Float: ...
    @property
    def value(self) -> drjit.cuda.Float: ...
    @property
    def surfel_id(self) -> drjit.cuda.Int: ...
    @property
    def triangle_id(self) -> drjit.cuda.Int: ...

class SurfelIntersectionAD:
    """AD variant of SurfelIntersection."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def t(self) -> drjit.cuda.ad.Float: ...
    @property
    def p(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def n(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def local_uv(self) -> drjit.cuda.ad.Array2f: ...
    @property
    def gaussian_weight(self) -> drjit.cuda.ad.Float: ...
    @property
    def opacity(self) -> drjit.cuda.ad.Float: ...
    @property
    def alpha(self) -> drjit.cuda.ad.Float: ...
    @property
    def value(self) -> drjit.cuda.ad.Float: ...
    @property
    def surfel_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def triangle_id(self) -> drjit.cuda.ad.Int: ...

class SdfIntersection:
    """
    Closest SDF surface hit per ray.

    Result record of SdfGrid.intersect(). valid marks hits.
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def t(self) -> drjit.cuda.Float: ...
    @property
    def valid(self) -> drjit.cuda.Bool: ...
    @property
    def position(self) -> drjit.cuda.Array3f: ...
    @property
    def normal(self) -> drjit.cuda.Array3f: ...
    @property
    def steps(self) -> drjit.cuda.Int: ...

class SdfIntersectionAD:
    """
    AD variant of SdfIntersection, returned when SdfGrid.intersect() gets a RayAD.
    """

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def t(self) -> drjit.cuda.ad.Float: ...
    @property
    def valid(self) -> drjit.cuda.ad.Bool: ...
    @property
    def position(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def normal(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def steps(self) -> drjit.cuda.ad.Int: ...

class SurfelComposite:
    """
    Front-to-back alpha composite of the surfels along each ray.

    Result record of SurfelScene.composite_alpha() and composite_alpha_reference().
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def intensity(self) -> drjit.cuda.Float: ...
    @property
    def alpha(self) -> drjit.cuda.Float: ...
    @property
    def transmittance(self) -> drjit.cuda.Float: ...
    @property
    def depth(self) -> drjit.cuda.Float: ...
    @property
    def candidate_count(self) -> drjit.cuda.Int: ...
    @property
    def candidate_buffer_full(self) -> drjit.cuda.Bool: ...

class SurfelCompositeAD:
    """AD variant of SurfelComposite."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def intensity(self) -> drjit.cuda.ad.Float: ...
    @property
    def alpha(self) -> drjit.cuda.ad.Float: ...
    @property
    def transmittance(self) -> drjit.cuda.ad.Float: ...
    @property
    def depth(self) -> drjit.cuda.ad.Float: ...
    @property
    def candidate_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def candidate_buffer_full(self) -> drjit.cuda.ad.Bool: ...

class SurfelRender:
    """
    Alpha-composited surfel render per ray.

    Result record of SurfelScene.render(). channels is a flat [ray_count, channel_count] buffer.
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def channels(self) -> drjit.cuda.Float: ...
    @property
    def rgb(self) -> drjit.cuda.Array3f: ...
    @property
    def normal(self) -> drjit.cuda.Array3f: ...
    @property
    def alpha(self) -> drjit.cuda.Float: ...
    @property
    def transmittance(self) -> drjit.cuda.Float: ...
    @property
    def depth(self) -> drjit.cuda.Float: ...
    @property
    def candidate_count(self) -> drjit.cuda.Int: ...
    @property
    def candidate_buffer_full(self) -> drjit.cuda.Bool: ...
    @property
    def channel_count(self) -> int: ...

class SurfelRenderAD:
    """AD variant of SurfelRender."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def channels(self) -> drjit.cuda.ad.Float: ...
    @property
    def rgb(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def normal(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def alpha(self) -> drjit.cuda.ad.Float: ...
    @property
    def transmittance(self) -> drjit.cuda.ad.Float: ...
    @property
    def depth(self) -> drjit.cuda.ad.Float: ...
    @property
    def candidate_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def candidate_buffer_full(self) -> drjit.cuda.ad.Bool: ...
    @property
    def channel_count(self) -> int: ...

class ReflectionChain:
    """
    Reflection paths with per-bounce arrays laid out ray-major as ray_count * max_bounces slots.

    Result record of Scene.trace_reflections() with symbolic=False, and of the SdfGrid, SurfelScene, and MixedScene reflection traces. bounce_count gives the valid slots per ray.
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def max_bounces(self) -> int: ...
    @property
    def ray_count(self) -> int: ...
    @property
    def bounce_count(self) -> drjit.cuda.Int: ...
    @property
    def discovery_count(self) -> drjit.cuda.Int: ...
    @property
    def representative_ray_index(self) -> drjit.cuda.Int: ...
    @property
    def t(self) -> drjit.cuda.Float: ...
    @property
    def hit_points(self) -> drjit.cuda.Array3f: ...
    @property
    def geo_normals(self) -> drjit.cuda.Array3f: ...
    @property
    def image_sources(self) -> drjit.cuda.Array3f: ...
    @property
    def plane_points(self) -> drjit.cuda.Array3f: ...
    @property
    def plane_normals(self) -> drjit.cuda.Array3f: ...
    @property
    def shape_ids(self) -> drjit.cuda.Int: ...
    @property
    def prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def local_prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def global_prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def trailing_t(self) -> drjit.cuda.Float: ...
    @property
    def trailing_prim(self) -> drjit.cuda.Int: ...
    @property
    def trailing_dir(self) -> drjit.cuda.Array3f: ...
    @property
    def trailing_origin(self) -> drjit.cuda.Array3f: ...

class ReflectionChainAD:
    """AD variant of ReflectionChain."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def max_bounces(self) -> int: ...
    @property
    def ray_count(self) -> int: ...
    @property
    def bounce_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def discovery_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def representative_ray_index(self) -> drjit.cuda.ad.Int: ...
    @property
    def t(self) -> drjit.cuda.ad.Float: ...
    @property
    def hit_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def geo_normals(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def image_sources(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def plane_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def plane_normals(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def shape_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def local_prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def global_prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def trailing_t(self) -> drjit.cuda.ad.Float: ...
    @property
    def trailing_prim(self) -> drjit.cuda.ad.Int: ...
    @property
    def trailing_dir(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def trailing_origin(self) -> drjit.cuda.ad.Array3f: ...

class AccumGrid:
    """
    Axis-aligned accumulation plane for Scene.accum_refl().

    Input record. The plane is normal to axis at offset position. Its in-plane coordinates are (y, z) for axis 0, (x, z) for axis 1, and (x, y) for axis 2, and cell (i, j) has the flat index i + j * resolution0.
    """

    def __init__(self) -> None: ...
    @property
    def axis(self) -> int:
        """Plane normal axis: 0 = x, 1 = y, 2 = z."""

    @axis.setter
    def axis(self, arg: int, /) -> None: ...
    @property
    def position(self) -> float:
        """Plane offset along axis."""

    @position.setter
    def position(self, arg: float, /) -> None: ...
    @property
    def coord0_min(self) -> float:
        """Lower bound of the first in-plane coordinate."""

    @coord0_min.setter
    def coord0_min(self, arg: float, /) -> None: ...
    @property
    def coord0_max(self) -> float:
        """Upper bound of the first in-plane coordinate; must exceed coord0_min."""

    @coord0_max.setter
    def coord0_max(self, arg: float, /) -> None: ...
    @property
    def coord1_min(self) -> float:
        """Lower bound of the second in-plane coordinate."""

    @coord1_min.setter
    def coord1_min(self, arg: float, /) -> None: ...
    @property
    def coord1_max(self) -> float:
        """Upper bound of the second in-plane coordinate; must exceed coord1_min."""

    @coord1_max.setter
    def coord1_max(self, arg: float, /) -> None: ...
    @property
    def resolution0(self) -> int:
        """Cell count along the first in-plane coordinate; must be positive."""

    @resolution0.setter
    def resolution0(self, arg: int, /) -> None: ...
    @property
    def resolution1(self) -> int:
        """Cell count along the second in-plane coordinate; must be positive."""

    @resolution1.setter
    def resolution1(self, arg: int, /) -> None: ...

class AccumOptions:
    """
    Options for Scene.accum_refl().

    Options record.
    """

    def __init__(self) -> None: ...
    @property
    def wavelength(self) -> float:
        """Wavelength in world units; must be positive."""

    @wavelength.setter
    def wavelength(self, arg: float, /) -> None: ...
    @property
    def k(self) -> float:
        """Wavenumber; 0 uses 2 * pi / wavelength."""

    @k.setter
    def k(self, arg: float, /) -> None: ...
    @property
    def solid_angle_per_ray(self) -> float:
        """Solid angle each ray represents; must be non-negative."""

    @solid_angle_per_ray.setter
    def solid_angle_per_ray(self, arg: float, /) -> None: ...
    @property
    def cell_area(self) -> float:
        """Area of one grid cell; must be positive."""

    @cell_area.setter
    def cell_area(self, arg: float, /) -> None: ...
    @property
    def seed(self) -> int:
        """Random seed."""

    @seed.setter
    def seed(self, arg: int, /) -> None: ...
    @property
    def rr_depth(self) -> int:
        """Bounce depth at which Russian roulette starts."""

    @rr_depth.setter
    def rr_depth(self, arg: int, /) -> None: ...
    @property
    def rr_prob(self) -> float:
        """Survival probability of each roulette test."""

    @rr_prob.setter
    def rr_prob(self, arg: float, /) -> None: ...
    @property
    def stop_threshold(self) -> float:
        """Stop a path once its contribution falls below this value."""

    @stop_threshold.setter
    def stop_threshold(self, arg: float, /) -> None: ...
    @property
    def collect_wedges(self) -> bool:
        """Record diffraction-wedge events in AccumResult.wedge_events."""

    @collect_wedges.setter
    def collect_wedges(self, arg: bool, /) -> None: ...
    @property
    def collect_wedge_prefixes(self) -> bool:
        """Also record the reflection prefix of each wedge event."""

    @collect_wedge_prefixes.setter
    def collect_wedge_prefixes(self, arg: bool, /) -> None: ...
    @property
    def wedge_capacity(self) -> int:
        """Maximum number of wedge events to store; must be non-negative."""

    @wedge_capacity.setter
    def wedge_capacity(self, arg: int, /) -> None: ...
    @property
    def wedge_sample_stride(self) -> int:
        """Sampling stride for prefix wedge events; must be at least 1."""

    @wedge_sample_stride.setter
    def wedge_sample_stride(self, arg: int, /) -> None: ...

class ReflMaterial:
    """
    Electromagnetic surface parameters for Scene.accum_refl().

    Input record with one entry per scene-global primitive. Use ReflMaterialAD to differentiate with respect to the material.
    """

    def __init__(self) -> None: ...
    @property
    def eta_r(self) -> drjit.cuda.Float:
        """Relative permittivity."""

    @eta_r.setter
    def eta_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def sigma(self) -> drjit.cuda.Float:
        """Conductivity."""

    @sigma.setter
    def sigma(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def gain(self) -> drjit.cuda.Float:
        """Extra gain factor."""

    @gain.setter
    def gain(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def mu_r(self) -> drjit.cuda.Float:
        """Relative permeability."""

    @mu_r.setter
    def mu_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def valid(self) -> drjit.cuda.Bool:
        """Whether the entry holds a real material."""

    @valid.setter
    def valid(self, arg: drjit.cuda.Bool, /) -> None: ...

class ReflMaterialAD:
    """
    ReflMaterial with AD arrays.

    Input record for the differentiable Scene.accum_refl() overload.
    """

    def __init__(self) -> None: ...
    @property
    def eta_r(self) -> drjit.cuda.ad.Float:
        """Relative permittivity."""

    @eta_r.setter
    def eta_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def sigma(self) -> drjit.cuda.ad.Float:
        """Conductivity."""

    @sigma.setter
    def sigma(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def gain(self) -> drjit.cuda.ad.Float:
        """Extra gain factor."""

    @gain.setter
    def gain(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def mu_r(self) -> drjit.cuda.ad.Float:
        """Relative permeability."""

    @mu_r.setter
    def mu_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def valid(self) -> drjit.cuda.ad.Bool:
        """Whether the entry holds a real material."""

    @valid.setter
    def valid(self, arg: drjit.cuda.ad.Bool, /) -> None: ...

class WedgeEvents:
    """
    Diffraction-wedge events recorded by Scene.accum_refl().

    Result record. The first count entries of the per-event arrays are valid, up to capacity.
    """

    @property
    def capacity(self) -> int: ...
    @property
    def count(self) -> drjit.cuda.Int: ...
    @property
    def ray_index(self) -> drjit.cuda.Int: ...
    @property
    def hit_points(self) -> drjit.cuda.Array3f: ...
    @property
    def normals(self) -> drjit.cuda.Array3f: ...
    @property
    def prim_id(self) -> drjit.cuda.Int: ...
    @property
    def directions(self) -> drjit.cuda.Array3f: ...
    @property
    def source_points(self) -> drjit.cuda.Array3f: ...
    @property
    def src_power(self) -> drjit.cuda.Float: ...
    @property
    def initial_directions(self) -> drjit.cuda.Array3f: ...
    @property
    def bounce_depth(self) -> drjit.cuda.Int: ...

class WedgeEventsAD:
    """AD variant of WedgeEvents."""

    @property
    def capacity(self) -> int: ...
    @property
    def count(self) -> drjit.cuda.ad.Int: ...
    @property
    def ray_index(self) -> drjit.cuda.ad.Int: ...
    @property
    def hit_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def normals(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def prim_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def directions(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def source_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def src_power(self) -> drjit.cuda.ad.Float: ...
    @property
    def initial_directions(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def bounce_depth(self) -> drjit.cuda.ad.Int: ...

class AccumResult:
    """
    Reflected power and complex field accumulated per grid cell.

    Result record of Scene.accum_refl(). Grid arrays have grid_cell_count entries.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def max_bounces(self) -> int: ...
    @property
    def grid_cell_count(self) -> int: ...
    @property
    def reflection_power(self) -> drjit.cuda.Float: ...
    @property
    def reflection_field_x(self) -> drjit.cuda.Complex2f: ...
    @property
    def reflection_field_y(self) -> drjit.cuda.Complex2f: ...
    @property
    def reflection_field_z(self) -> drjit.cuda.Complex2f: ...
    @property
    def reflection_count(self) -> drjit.cuda.Int: ...
    @property
    def wedge_events(self) -> WedgeEvents: ...

class AccumResultAD:
    """
    AD variant of AccumResult, returned when Scene.accum_refl() gets a RayAD.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def max_bounces(self) -> int: ...
    @property
    def grid_cell_count(self) -> int: ...
    @property
    def reflection_power(self) -> drjit.cuda.ad.Float: ...
    @property
    def reflection_field_x(self) -> drjit.cuda.ad.Complex2f: ...
    @property
    def reflection_field_y(self) -> drjit.cuda.ad.Complex2f: ...
    @property
    def reflection_field_z(self) -> drjit.cuda.ad.Complex2f: ...
    @property
    def reflection_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def wedge_events(self) -> WedgeEventsAD: ...

class DfrStrategy(enum.IntFlag):
    """
    Flags that select the Monte-Carlo strategies of the diffraction accumulators.

    DIRECT samples direct edge diffraction, KELLER samples the Keller cone, and SUFFIX_REFL adds a reflection after the diffraction event.
    """

    DIRECT = 1

    KELLER = 2

    SUFFIX_REFL = 4

class DfrSampleSequence(enum.IntEnum):
    """Sample sequence of the diffraction accumulators: HASH or SOBOL."""

    HASH = 0

    SOBOL = 1

class DfrReceiverModel(enum.IntEnum):
    """
    Receiver model of the diffraction kernels. MATCHED_ISO, a matched isotropic receiver, is the only model.
    """

    MATCHED_ISO = 0

class DfrGrid:
    """
    Axis-aligned accumulation plane for the diffraction accumulators.

    Input record. The plane is normal to axis at offset position. Its in-plane coordinates are (y, z) for axis 0, (x, z) for axis 1, and (x, y) for axis 2, and cell (i, j) has the flat index i + j * resolution0.
    """

    def __init__(self) -> None: ...
    @property
    def axis(self) -> int:
        """Plane normal axis: 0 = x, 1 = y, 2 = z."""

    @axis.setter
    def axis(self, arg: int, /) -> None: ...
    @property
    def position(self) -> float:
        """Plane offset along axis."""

    @position.setter
    def position(self, arg: float, /) -> None: ...
    @property
    def coord0_min(self) -> float:
        """Lower bound of the first in-plane coordinate."""

    @coord0_min.setter
    def coord0_min(self, arg: float, /) -> None: ...
    @property
    def coord0_max(self) -> float:
        """Upper bound of the first in-plane coordinate; must exceed coord0_min."""

    @coord0_max.setter
    def coord0_max(self, arg: float, /) -> None: ...
    @property
    def coord1_min(self) -> float:
        """Lower bound of the second in-plane coordinate."""

    @coord1_min.setter
    def coord1_min(self, arg: float, /) -> None: ...
    @property
    def coord1_max(self) -> float:
        """Upper bound of the second in-plane coordinate; must exceed coord1_min."""

    @coord1_max.setter
    def coord1_max(self, arg: float, /) -> None: ...
    @property
    def resolution0(self) -> int:
        """Cell count along the first in-plane coordinate; must be positive."""

    @resolution0.setter
    def resolution0(self, arg: int, /) -> None: ...
    @property
    def resolution1(self) -> int:
        """Cell count along the second in-plane coordinate; must be positive."""

    @resolution1.setter
    def resolution1(self, arg: int, /) -> None: ...
    @property
    def cell_area(self) -> float:
        """Area of one grid cell; must be positive."""

    @cell_area.setter
    def cell_area(self, arg: float, /) -> None: ...

class DfrOptions:
    """
    Options for Scene.accum_dfr_direct() and Scene.accum_dfr().

    Options record.
    """

    def __init__(self) -> None: ...
    @property
    def wavelength(self) -> float:
        """Wavelength in world units; must be positive."""

    @wavelength.setter
    def wavelength(self, arg: float, /) -> None: ...
    @property
    def k(self) -> float:
        """
        Wavenumber used for phases. It is used as given, not derived from wavelength.
        """

    @k.setter
    def k(self, arg: float, /) -> None: ...
    @property
    def seed(self) -> int:
        """Seed of the sample sequence."""

    @seed.setter
    def seed(self, arg: int, /) -> None: ...
    @property
    def samples(self) -> int:
        """Direct-strategy sample count used when direct_samples is 0."""

    @samples.setter
    def samples(self, arg: int, /) -> None: ...
    @property
    def max_order(self) -> int:
        """Diffraction order: 1 for accum_dfr_direct(), 2 or 3 for accum_dfr()."""

    @max_order.setter
    def max_order(self, arg: int, /) -> None: ...
    @property
    def direct_samples(self) -> int:
        """Samples for the DIRECT strategy; 0 falls back to samples."""

    @direct_samples.setter
    def direct_samples(self, arg: int, /) -> None: ...
    @property
    def keller_samples(self) -> int:
        """Samples for the KELLER strategy."""

    @keller_samples.setter
    def keller_samples(self, arg: int, /) -> None: ...
    @property
    def suffix_samples(self) -> int:
        """Samples for the SUFFIX_REFL strategy."""

    @suffix_samples.setter
    def suffix_samples(self, arg: int, /) -> None: ...
    @property
    def strategy_mask(self) -> DfrStrategy:
        """Enabled strategies; a strategy runs only when its flag is set."""

    @strategy_mask.setter
    def strategy_mask(self, arg: DfrStrategy, /) -> None: ...
    @property
    def sample_sequence(self) -> DfrSampleSequence:
        """Sample sequence."""

    @sample_sequence.setter
    def sample_sequence(self, arg: DfrSampleSequence, /) -> None: ...
    @property
    def receiver_model(self) -> DfrReceiverModel:
        """Receiver model."""

    @receiver_model.setter
    def receiver_model(self, arg: DfrReceiverModel, /) -> None: ...
    @property
    def collect_edge_use(self) -> bool:
        """Fill DfrAccum.edge_uses."""

    @collect_edge_use.setter
    def collect_edge_use(self, arg: bool, /) -> None: ...
    @property
    def collect_debug_counts(self) -> bool:
        """Fill the rejection counters of DfrAccum."""

    @collect_debug_counts.setter
    def collect_debug_counts(self, arg: bool, /) -> None: ...

class DfrCoherentOptions:
    """
    Options for deterministic coherent diffraction: Scene.accum_dfr_coherent_direct(), build_dfr_coherent_tx_states(), and build_dfr_coherent_higher_candidates().

    Options record.
    """

    def __init__(self) -> None: ...
    @property
    def wavelength(self) -> float:
        """Wavelength in world units; must be positive."""

    @wavelength.setter
    def wavelength(self, arg: float, /) -> None: ...
    @property
    def k(self) -> float:
        """
        Wavenumber used for phases. It is used as given; build_dfr_coherent_tx_states() requires k > 0.
        """

    @k.setter
    def k(self, arg: float, /) -> None: ...
    @property
    def max_order(self) -> int:
        """Diffraction order; accum_dfr_coherent_direct() supports only 1."""

    @max_order.setter
    def max_order(self, arg: int, /) -> None: ...
    @property
    def receiver_model(self) -> DfrReceiverModel:
        """Receiver model; only MATCHED_ISO is supported."""

    @receiver_model.setter
    def receiver_model(self, arg: DfrReceiverModel, /) -> None: ...
    @property
    def select_diffraction_point(self) -> bool:
        """
        For DfrStates input, diffract at the first-order stationary point for each cell instead of the edge midpoint.
        """

    @select_diffraction_point.setter
    def select_diffraction_point(self, arg: bool, /) -> None: ...
    @property
    def prefilter_visibility(self) -> bool:
        """Skip lanes whose source-to-edge or edge-to-cell segment is blocked."""

    @prefilter_visibility.setter
    def prefilter_visibility(self, arg: bool, /) -> None: ...
    @property
    def collect_debug_counts(self) -> bool:
        """Fill vis_rejects and utd_rejects."""

    @collect_debug_counts.setter
    def collect_debug_counts(self, arg: bool, /) -> None: ...
    @property
    def omega(self) -> float:
        """
        Angular frequency in rad/s for the face Fresnel coefficients. A value <= 0 uses the face operators stored in the UTD states.
        """

    @omega.setter
    def omega(self, arg: float, /) -> None: ...
    @property
    def tx_pol_x(self) -> float:
        """x component of the transmitter polarization."""

    @tx_pol_x.setter
    def tx_pol_x(self, arg: float, /) -> None: ...
    @property
    def tx_pol_y(self) -> float:
        """y component of the transmitter polarization."""

    @tx_pol_y.setter
    def tx_pol_y(self, arg: float, /) -> None: ...
    @property
    def tx_pol_z(self) -> float:
        """z component of the transmitter polarization."""

    @tx_pol_z.setter
    def tx_pol_z(self, arg: float, /) -> None: ...
    @property
    def higher_probe_radius_scale(self) -> float:
        """
        Probe radius as a fraction of the source-to-edge distance; must be positive.
        """

    @higher_probe_radius_scale.setter
    def higher_probe_radius_scale(self, arg: float, /) -> None: ...
    @property
    def higher_probe_radius_min(self) -> float:
        """Lower clamp of the probe radius."""

    @higher_probe_radius_min.setter
    def higher_probe_radius_min(self, arg: float, /) -> None: ...
    @property
    def higher_probe_radius_max(self) -> float:
        """Upper clamp of the probe radius."""

    @higher_probe_radius_max.setter
    def higher_probe_radius_max(self, arg: float, /) -> None: ...
    @property
    def higher_filter_visibility(self) -> bool:
        """Drop candidate pairs whose edge-to-edge segment is blocked."""

    @higher_filter_visibility.setter
    def higher_filter_visibility(self, arg: bool, /) -> None: ...

class DfrMaterial:
    """
    Electromagnetic surface parameters for the diffraction kernels.

    Input record. Use DfrMaterialAD to differentiate with respect to the material.
    """

    def __init__(self) -> None: ...
    @property
    def eta_r(self) -> drjit.cuda.Float:
        """Relative permittivity."""

    @eta_r.setter
    def eta_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def sigma(self) -> drjit.cuda.Float:
        """Conductivity."""

    @sigma.setter
    def sigma(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def mu_r(self) -> drjit.cuda.Float:
        """Relative permeability."""

    @mu_r.setter
    def mu_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def gain(self) -> drjit.cuda.Float:
        """Extra gain factor."""

    @gain.setter
    def gain(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def valid(self) -> drjit.cuda.Bool:
        """Whether the entry holds a real material."""

    @valid.setter
    def valid(self, arg: drjit.cuda.Bool, /) -> None: ...

class DfrMaterialAD:
    """
    DfrMaterial with AD arrays.

    Input record for the AD diffraction overloads.
    """

    def __init__(self) -> None: ...
    @property
    def eta_r(self) -> drjit.cuda.ad.Float:
        """Relative permittivity."""

    @eta_r.setter
    def eta_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def sigma(self) -> drjit.cuda.ad.Float:
        """Conductivity."""

    @sigma.setter
    def sigma(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def mu_r(self) -> drjit.cuda.ad.Float:
        """Relative permeability."""

    @mu_r.setter
    def mu_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def gain(self) -> drjit.cuda.ad.Float:
        """Extra gain factor."""

    @gain.setter
    def gain(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def valid(self) -> drjit.cuda.ad.Bool:
        """Whether the entry holds a real material."""

    @valid.setter
    def valid(self, arg: drjit.cuda.ad.Bool, /) -> None: ...

class DfrCoherentUtdStates:
    """
    Deterministic UTD states for coherent diffraction accumulation, one per (source, edge) pair.

    Input and result record. Produced by Scene.build_dfr_coherent_tx_states() and consumed by Scene.accum_dfr_coherent_direct(). count is the number of valid states.
    """

    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg: int, /) -> None: ...
    @property
    def edge_index(self) -> drjit.cuda.Int: ...
    @edge_index.setter
    def edge_index(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge_pos(self) -> drjit.cuda.Array3f: ...
    @edge_pos.setter
    def edge_pos(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def edge_dir(self) -> drjit.cuda.Array3f: ...
    @edge_dir.setter
    def edge_dir(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def n0(self) -> drjit.cuda.Array3f: ...
    @n0.setter
    def n0(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def n_face_n(self) -> drjit.cuda.Array3f: ...
    @n_face_n.setter
    def n_face_n(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def wedge_n(self) -> drjit.cuda.Float: ...
    @wedge_n.setter
    def wedge_n(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def edge_line_min(self) -> drjit.cuda.Float: ...
    @edge_line_min.setter
    def edge_line_min(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def edge_line_max(self) -> drjit.cuda.Float: ...
    @edge_line_max.setter
    def edge_line_max(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def source_pos(self) -> drjit.cuda.Array3f: ...
    @source_pos.setter
    def source_pos(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def incident_field(self) -> drjit.cuda.Complex2f: ...
    @incident_field.setter
    def incident_field(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_normal_derivative(self) -> drjit.cuda.Complex2f: ...
    @incident_normal_derivative.setter
    def incident_normal_derivative(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def r_face0(self) -> drjit.cuda.Complex2f: ...
    @r_face0.setter
    def r_face0(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def r_face_n(self) -> drjit.cuda.Complex2f: ...
    @r_face_n.setter
    def r_face_n(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_vector_x(self) -> drjit.cuda.Complex2f: ...
    @incident_vector_x.setter
    def incident_vector_x(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_vector_y(self) -> drjit.cuda.Complex2f: ...
    @incident_vector_y.setter
    def incident_vector_y(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_vector_z(self) -> drjit.cuda.Complex2f: ...
    @incident_vector_z.setter
    def incident_vector_z(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_normal_derivative_vector_x(self) -> drjit.cuda.Complex2f: ...
    @incident_normal_derivative_vector_x.setter
    def incident_normal_derivative_vector_x(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_normal_derivative_vector_y(self) -> drjit.cuda.Complex2f: ...
    @incident_normal_derivative_vector_y.setter
    def incident_normal_derivative_vector_y(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_normal_derivative_vector_z(self) -> drjit.cuda.Complex2f: ...
    @incident_normal_derivative_vector_z.setter
    def incident_normal_derivative_vector_z(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_jones_u(self) -> drjit.cuda.Complex2f: ...
    @incident_jones_u.setter
    def incident_jones_u(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_jones_v(self) -> drjit.cuda.Complex2f: ...
    @incident_jones_v.setter
    def incident_jones_v(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_derivative_jones_u(self) -> drjit.cuda.Complex2f: ...
    @incident_derivative_jones_u.setter
    def incident_derivative_jones_u(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_derivative_jones_v(self) -> drjit.cuda.Complex2f: ...
    @incident_derivative_jones_v.setter
    def incident_derivative_jones_v(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def incident_basis_u(self) -> drjit.cuda.Array3f: ...
    @incident_basis_u.setter
    def incident_basis_u(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def incident_basis_v(self) -> drjit.cuda.Array3f: ...
    @incident_basis_v.setter
    def incident_basis_v(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def incident_basis_k(self) -> drjit.cuda.Array3f: ...
    @incident_basis_k.setter
    def incident_basis_k(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def face0_operator_m00(self) -> drjit.cuda.Complex2f: ...
    @face0_operator_m00.setter
    def face0_operator_m00(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def face0_operator_m01(self) -> drjit.cuda.Complex2f: ...
    @face0_operator_m01.setter
    def face0_operator_m01(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def face0_operator_m10(self) -> drjit.cuda.Complex2f: ...
    @face0_operator_m10.setter
    def face0_operator_m10(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def face0_operator_m11(self) -> drjit.cuda.Complex2f: ...
    @face0_operator_m11.setter
    def face0_operator_m11(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def face1_operator_m00(self) -> drjit.cuda.Complex2f: ...
    @face1_operator_m00.setter
    def face1_operator_m00(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def face1_operator_m01(self) -> drjit.cuda.Complex2f: ...
    @face1_operator_m01.setter
    def face1_operator_m01(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def face1_operator_m10(self) -> drjit.cuda.Complex2f: ...
    @face1_operator_m10.setter
    def face1_operator_m10(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def face1_operator_m11(self) -> drjit.cuda.Complex2f: ...
    @face1_operator_m11.setter
    def face1_operator_m11(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def face0_eta_r(self) -> drjit.cuda.Float: ...
    @face0_eta_r.setter
    def face0_eta_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face0_mu_r(self) -> drjit.cuda.Float: ...
    @face0_mu_r.setter
    def face0_mu_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face0_sigma(self) -> drjit.cuda.Float: ...
    @face0_sigma.setter
    def face0_sigma(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face0_gain(self) -> drjit.cuda.Float: ...
    @face0_gain.setter
    def face0_gain(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face0_use_fresnel(self) -> drjit.cuda.Float: ...
    @face0_use_fresnel.setter
    def face0_use_fresnel(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face1_eta_r(self) -> drjit.cuda.Float: ...
    @face1_eta_r.setter
    def face1_eta_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face1_mu_r(self) -> drjit.cuda.Float: ...
    @face1_mu_r.setter
    def face1_mu_r(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face1_sigma(self) -> drjit.cuda.Float: ...
    @face1_sigma.setter
    def face1_sigma(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face1_gain(self) -> drjit.cuda.Float: ...
    @face1_gain.setter
    def face1_gain(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def face1_use_fresnel(self) -> drjit.cuda.Float: ...
    @face1_use_fresnel.setter
    def face1_use_fresnel(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def select_stationary_point(self) -> drjit.cuda.Float: ...
    @select_stationary_point.setter
    def select_stationary_point(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def owner_code(self) -> drjit.cuda.Int: ...
    @owner_code.setter
    def owner_code(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def adjacent_face0(self) -> drjit.cuda.Int: ...
    @adjacent_face0.setter
    def adjacent_face0(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def adjacent_face1(self) -> drjit.cuda.Int: ...
    @adjacent_face1.setter
    def adjacent_face1(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def path_length_prefix(self) -> drjit.cuda.Float: ...
    @path_length_prefix.setter
    def path_length_prefix(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def first_interaction_pos(self) -> drjit.cuda.Array3f: ...
    @first_interaction_pos.setter
    def first_interaction_pos(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def source_type_code(self) -> drjit.cuda.Int: ...
    @source_type_code.setter
    def source_type_code(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def prefix_reflection_depth(self) -> drjit.cuda.Int: ...
    @prefix_reflection_depth.setter
    def prefix_reflection_depth(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def intermediate_reflection_depth(self) -> drjit.cuda.Int: ...
    @intermediate_reflection_depth.setter
    def intermediate_reflection_depth(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def suffix_reflection_depth(self) -> drjit.cuda.Int: ...
    @suffix_reflection_depth.setter
    def suffix_reflection_depth(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def approximation_mode_code(self) -> drjit.cuda.Int: ...
    @approximation_mode_code.setter
    def approximation_mode_code(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def order(self) -> drjit.cuda.Int: ...
    @order.setter
    def order(self, arg: drjit.cuda.Int, /) -> None: ...

class DfrCoherentUtdStatesAD:
    """
    DfrCoherentUtdStates with AD arrays. The coherent diffraction operations reject AD inputs.
    """

    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg: int, /) -> None: ...
    @property
    def edge_index(self) -> drjit.cuda.ad.Int: ...
    @edge_index.setter
    def edge_index(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge_pos(self) -> drjit.cuda.ad.Array3f: ...
    @edge_pos.setter
    def edge_pos(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def edge_dir(self) -> drjit.cuda.ad.Array3f: ...
    @edge_dir.setter
    def edge_dir(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def n0(self) -> drjit.cuda.ad.Array3f: ...
    @n0.setter
    def n0(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def n_face_n(self) -> drjit.cuda.ad.Array3f: ...
    @n_face_n.setter
    def n_face_n(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def wedge_n(self) -> drjit.cuda.ad.Float: ...
    @wedge_n.setter
    def wedge_n(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def edge_line_min(self) -> drjit.cuda.ad.Float: ...
    @edge_line_min.setter
    def edge_line_min(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def edge_line_max(self) -> drjit.cuda.ad.Float: ...
    @edge_line_max.setter
    def edge_line_max(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def source_pos(self) -> drjit.cuda.ad.Array3f: ...
    @source_pos.setter
    def source_pos(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def incident_field(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_field.setter
    def incident_field(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_normal_derivative(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_normal_derivative.setter
    def incident_normal_derivative(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def r_face0(self) -> drjit.cuda.ad.Complex2f: ...
    @r_face0.setter
    def r_face0(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def r_face_n(self) -> drjit.cuda.ad.Complex2f: ...
    @r_face_n.setter
    def r_face_n(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_vector_x(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_vector_x.setter
    def incident_vector_x(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_vector_y(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_vector_y.setter
    def incident_vector_y(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_vector_z(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_vector_z.setter
    def incident_vector_z(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_normal_derivative_vector_x(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_normal_derivative_vector_x.setter
    def incident_normal_derivative_vector_x(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_normal_derivative_vector_y(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_normal_derivative_vector_y.setter
    def incident_normal_derivative_vector_y(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_normal_derivative_vector_z(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_normal_derivative_vector_z.setter
    def incident_normal_derivative_vector_z(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_jones_u(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_jones_u.setter
    def incident_jones_u(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_jones_v(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_jones_v.setter
    def incident_jones_v(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_derivative_jones_u(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_derivative_jones_u.setter
    def incident_derivative_jones_u(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_derivative_jones_v(self) -> drjit.cuda.ad.Complex2f: ...
    @incident_derivative_jones_v.setter
    def incident_derivative_jones_v(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def incident_basis_u(self) -> drjit.cuda.ad.Array3f: ...
    @incident_basis_u.setter
    def incident_basis_u(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def incident_basis_v(self) -> drjit.cuda.ad.Array3f: ...
    @incident_basis_v.setter
    def incident_basis_v(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def incident_basis_k(self) -> drjit.cuda.ad.Array3f: ...
    @incident_basis_k.setter
    def incident_basis_k(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def face0_operator_m00(self) -> drjit.cuda.ad.Complex2f: ...
    @face0_operator_m00.setter
    def face0_operator_m00(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def face0_operator_m01(self) -> drjit.cuda.ad.Complex2f: ...
    @face0_operator_m01.setter
    def face0_operator_m01(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def face0_operator_m10(self) -> drjit.cuda.ad.Complex2f: ...
    @face0_operator_m10.setter
    def face0_operator_m10(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def face0_operator_m11(self) -> drjit.cuda.ad.Complex2f: ...
    @face0_operator_m11.setter
    def face0_operator_m11(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def face1_operator_m00(self) -> drjit.cuda.ad.Complex2f: ...
    @face1_operator_m00.setter
    def face1_operator_m00(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def face1_operator_m01(self) -> drjit.cuda.ad.Complex2f: ...
    @face1_operator_m01.setter
    def face1_operator_m01(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def face1_operator_m10(self) -> drjit.cuda.ad.Complex2f: ...
    @face1_operator_m10.setter
    def face1_operator_m10(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def face1_operator_m11(self) -> drjit.cuda.ad.Complex2f: ...
    @face1_operator_m11.setter
    def face1_operator_m11(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def face0_eta_r(self) -> drjit.cuda.ad.Float: ...
    @face0_eta_r.setter
    def face0_eta_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face0_mu_r(self) -> drjit.cuda.ad.Float: ...
    @face0_mu_r.setter
    def face0_mu_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face0_sigma(self) -> drjit.cuda.ad.Float: ...
    @face0_sigma.setter
    def face0_sigma(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face0_gain(self) -> drjit.cuda.ad.Float: ...
    @face0_gain.setter
    def face0_gain(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face0_use_fresnel(self) -> drjit.cuda.ad.Float: ...
    @face0_use_fresnel.setter
    def face0_use_fresnel(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face1_eta_r(self) -> drjit.cuda.ad.Float: ...
    @face1_eta_r.setter
    def face1_eta_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face1_mu_r(self) -> drjit.cuda.ad.Float: ...
    @face1_mu_r.setter
    def face1_mu_r(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face1_sigma(self) -> drjit.cuda.ad.Float: ...
    @face1_sigma.setter
    def face1_sigma(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face1_gain(self) -> drjit.cuda.ad.Float: ...
    @face1_gain.setter
    def face1_gain(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def face1_use_fresnel(self) -> drjit.cuda.ad.Float: ...
    @face1_use_fresnel.setter
    def face1_use_fresnel(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def select_stationary_point(self) -> drjit.cuda.ad.Float: ...
    @select_stationary_point.setter
    def select_stationary_point(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def owner_code(self) -> drjit.cuda.ad.Int: ...
    @owner_code.setter
    def owner_code(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def adjacent_face0(self) -> drjit.cuda.ad.Int: ...
    @adjacent_face0.setter
    def adjacent_face0(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def adjacent_face1(self) -> drjit.cuda.ad.Int: ...
    @adjacent_face1.setter
    def adjacent_face1(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def path_length_prefix(self) -> drjit.cuda.ad.Float: ...
    @path_length_prefix.setter
    def path_length_prefix(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def first_interaction_pos(self) -> drjit.cuda.ad.Array3f: ...
    @first_interaction_pos.setter
    def first_interaction_pos(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def source_type_code(self) -> drjit.cuda.ad.Int: ...
    @source_type_code.setter
    def source_type_code(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def prefix_reflection_depth(self) -> drjit.cuda.ad.Int: ...
    @prefix_reflection_depth.setter
    def prefix_reflection_depth(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def intermediate_reflection_depth(self) -> drjit.cuda.ad.Int: ...
    @intermediate_reflection_depth.setter
    def intermediate_reflection_depth(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def suffix_reflection_depth(self) -> drjit.cuda.ad.Int: ...
    @suffix_reflection_depth.setter
    def suffix_reflection_depth(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def approximation_mode_code(self) -> drjit.cuda.ad.Int: ...
    @approximation_mode_code.setter
    def approximation_mode_code(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def order(self) -> drjit.cuda.ad.Int: ...
    @order.setter
    def order(self, arg: drjit.cuda.ad.Int, /) -> None: ...

class DfrCoherentEdge:
    """
    Selected diffraction edges used to build coherent UTD states.

    Input record for Scene.build_dfr_coherent_tx_states() and build_dfr_coherent_higher_candidates(). count is the number of valid edges.
    """

    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg: int, /) -> None: ...
    @property
    def edge_index(self) -> drjit.cuda.Int: ...
    @edge_index.setter
    def edge_index(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge_pos(self) -> drjit.cuda.Array3f: ...
    @edge_pos.setter
    def edge_pos(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def edge_dir(self) -> drjit.cuda.Array3f: ...
    @edge_dir.setter
    def edge_dir(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def n0(self) -> drjit.cuda.Array3f: ...
    @n0.setter
    def n0(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def n_face_n(self) -> drjit.cuda.Array3f: ...
    @n_face_n.setter
    def n_face_n(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def wedge_n(self) -> drjit.cuda.Float: ...
    @wedge_n.setter
    def wedge_n(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def edge_line_min(self) -> drjit.cuda.Float: ...
    @edge_line_min.setter
    def edge_line_min(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def edge_line_max(self) -> drjit.cuda.Float: ...
    @edge_line_max.setter
    def edge_line_max(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def adjacent_face0(self) -> drjit.cuda.Int: ...
    @adjacent_face0.setter
    def adjacent_face0(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def adjacent_face1(self) -> drjit.cuda.Int: ...
    @adjacent_face1.setter
    def adjacent_face1(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def ignore_prim_ids(self) -> drjit.cuda.Int: ...
    @ignore_prim_ids.setter
    def ignore_prim_ids(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def ignore_k(self) -> int: ...
    @ignore_k.setter
    def ignore_k(self, arg: int, /) -> None: ...

class DfrCoherentEdgeAD:
    """
    DfrCoherentEdge with AD arrays. The coherent diffraction operations reject AD inputs.
    """

    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg: int, /) -> None: ...
    @property
    def edge_index(self) -> drjit.cuda.ad.Int: ...
    @edge_index.setter
    def edge_index(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge_pos(self) -> drjit.cuda.ad.Array3f: ...
    @edge_pos.setter
    def edge_pos(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def edge_dir(self) -> drjit.cuda.ad.Array3f: ...
    @edge_dir.setter
    def edge_dir(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def n0(self) -> drjit.cuda.ad.Array3f: ...
    @n0.setter
    def n0(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def n_face_n(self) -> drjit.cuda.ad.Array3f: ...
    @n_face_n.setter
    def n_face_n(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def wedge_n(self) -> drjit.cuda.ad.Float: ...
    @wedge_n.setter
    def wedge_n(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def edge_line_min(self) -> drjit.cuda.ad.Float: ...
    @edge_line_min.setter
    def edge_line_min(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def edge_line_max(self) -> drjit.cuda.ad.Float: ...
    @edge_line_max.setter
    def edge_line_max(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def adjacent_face0(self) -> drjit.cuda.ad.Int: ...
    @adjacent_face0.setter
    def adjacent_face0(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def adjacent_face1(self) -> drjit.cuda.ad.Int: ...
    @adjacent_face1.setter
    def adjacent_face1(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def ignore_prim_ids(self) -> drjit.cuda.ad.Int: ...
    @ignore_prim_ids.setter
    def ignore_prim_ids(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def ignore_k(self) -> int: ...
    @ignore_k.setter
    def ignore_k(self, arg: int, /) -> None: ...

class DfrCoherentCandidatePairs:
    """
    Candidate (previous state, next edge) pairs for higher-order coherent diffraction.

    Result record of Scene.build_dfr_coherent_higher_candidates(). The first count entries of prev_index and edge_index are valid; visibility_filtered is 1 when the pairs were visibility filtered.
    """

    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg: int, /) -> None: ...
    @property
    def visibility_filtered(self) -> int: ...
    @visibility_filtered.setter
    def visibility_filtered(self, arg: int, /) -> None: ...
    @property
    def prev_index(self) -> drjit.cuda.Int: ...
    @prev_index.setter
    def prev_index(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge_index(self) -> drjit.cuda.Int: ...
    @edge_index.setter
    def edge_index(self, arg: drjit.cuda.Int, /) -> None: ...

class DfrCoherentCandidatePairsAD:
    """AD variant of DfrCoherentCandidatePairs."""

    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg: int, /) -> None: ...
    @property
    def visibility_filtered(self) -> int: ...
    @visibility_filtered.setter
    def visibility_filtered(self, arg: int, /) -> None: ...
    @property
    def prev_index(self) -> drjit.cuda.ad.Int: ...
    @prev_index.setter
    def prev_index(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge_index(self) -> drjit.cuda.ad.Int: ...
    @edge_index.setter
    def edge_index(self, arg: drjit.cuda.ad.Int, /) -> None: ...

class DfrStates:
    """
    Sampled diffraction states: an edge segment plus the source that illuminates it.

    Input record for the diffraction accumulators and Scene.trace_dfr_paths().
    """

    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg: int, /) -> None: ...
    @property
    def edge_index(self) -> drjit.cuda.Int: ...
    @edge_index.setter
    def edge_index(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge_pos(self) -> drjit.cuda.Array3f: ...
    @edge_pos.setter
    def edge_pos(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def edge_dir(self) -> drjit.cuda.Array3f: ...
    @edge_dir.setter
    def edge_dir(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def edge_t_min(self) -> drjit.cuda.Float: ...
    @edge_t_min.setter
    def edge_t_min(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def edge_t_max(self) -> drjit.cuda.Float: ...
    @edge_t_max.setter
    def edge_t_max(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def n0(self) -> drjit.cuda.Array3f: ...
    @n0.setter
    def n0(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def n1(self) -> drjit.cuda.Array3f: ...
    @n1.setter
    def n1(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def prim0(self) -> drjit.cuda.Int: ...
    @prim0.setter
    def prim0(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def prim1(self) -> drjit.cuda.Int: ...
    @prim1.setter
    def prim1(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def exterior_angle(self) -> drjit.cuda.Float: ...
    @exterior_angle.setter
    def exterior_angle(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def src(self) -> drjit.cuda.Array3f: ...
    @src.setter
    def src(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def src_power(self) -> drjit.cuda.Float: ...
    @src_power.setter
    def src_power(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def wi(self) -> drjit.cuda.Array3f: ...
    @wi.setter
    def wi(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def d0(self) -> drjit.cuda.Array3f: ...
    @d0.setter
    def d0(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def prefix_depth(self) -> drjit.cuda.Int: ...
    @prefix_depth.setter
    def prefix_depth(self, arg: drjit.cuda.Int, /) -> None: ...

class DfrStatesAD:
    """
    DfrStates with AD arrays.

    Input record for the differentiable diffraction overloads.
    """

    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg: int, /) -> None: ...
    @property
    def edge_index(self) -> drjit.cuda.ad.Int: ...
    @edge_index.setter
    def edge_index(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge_pos(self) -> drjit.cuda.ad.Array3f: ...
    @edge_pos.setter
    def edge_pos(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def edge_dir(self) -> drjit.cuda.ad.Array3f: ...
    @edge_dir.setter
    def edge_dir(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def edge_t_min(self) -> drjit.cuda.ad.Float: ...
    @edge_t_min.setter
    def edge_t_min(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def edge_t_max(self) -> drjit.cuda.ad.Float: ...
    @edge_t_max.setter
    def edge_t_max(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def n0(self) -> drjit.cuda.ad.Array3f: ...
    @n0.setter
    def n0(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def n1(self) -> drjit.cuda.ad.Array3f: ...
    @n1.setter
    def n1(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def prim0(self) -> drjit.cuda.ad.Int: ...
    @prim0.setter
    def prim0(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def prim1(self) -> drjit.cuda.ad.Int: ...
    @prim1.setter
    def prim1(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def exterior_angle(self) -> drjit.cuda.ad.Float: ...
    @exterior_angle.setter
    def exterior_angle(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def src(self) -> drjit.cuda.ad.Array3f: ...
    @src.setter
    def src(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def src_power(self) -> drjit.cuda.ad.Float: ...
    @src_power.setter
    def src_power(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def wi(self) -> drjit.cuda.ad.Array3f: ...
    @wi.setter
    def wi(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def d0(self) -> drjit.cuda.ad.Array3f: ...
    @d0.setter
    def d0(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def prefix_depth(self) -> drjit.cuda.ad.Int: ...
    @prefix_depth.setter
    def prefix_depth(self, arg: drjit.cuda.ad.Int, /) -> None: ...

class DfrAccum:
    """
    Diffraction power and complex field accumulated per grid cell, plus strategy and rejection counters.

    Result record of Scene.accum_dfr_direct() and Scene.accum_dfr(). Grid arrays have grid_cell_count entries.
    """

    def __init__(self) -> None: ...
    @property
    def grid_cell_count(self) -> int: ...
    @grid_cell_count.setter
    def grid_cell_count(self, arg: int, /) -> None: ...
    @property
    def power(self) -> drjit.cuda.Float: ...
    @power.setter
    def power(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def field_x(self) -> drjit.cuda.Complex2f: ...
    @field_x.setter
    def field_x(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def field_y(self) -> drjit.cuda.Complex2f: ...
    @field_y.setter
    def field_y(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def field_z(self) -> drjit.cuda.Complex2f: ...
    @field_z.setter
    def field_z(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def direct_count(self) -> drjit.cuda.Int: ...
    @direct_count.setter
    def direct_count(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def keller_count(self) -> drjit.cuda.Int: ...
    @keller_count.setter
    def keller_count(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def suffix_count(self) -> drjit.cuda.Int: ...
    @suffix_count.setter
    def suffix_count(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def vis_rejects(self) -> drjit.cuda.Int: ...
    @vis_rejects.setter
    def vis_rejects(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge_vis_rejects(self) -> drjit.cuda.Int: ...
    @edge_vis_rejects.setter
    def edge_vis_rejects(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def utd_rejects(self) -> drjit.cuda.Int: ...
    @utd_rejects.setter
    def utd_rejects(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge_uses(self) -> drjit.cuda.Int: ...
    @edge_uses.setter
    def edge_uses(self, arg: drjit.cuda.Int, /) -> None: ...

class DfrAccumAD:
    """AD variant of DfrAccum, returned by the AD overloads."""

    def __init__(self) -> None: ...
    @property
    def grid_cell_count(self) -> int: ...
    @grid_cell_count.setter
    def grid_cell_count(self, arg: int, /) -> None: ...
    @property
    def power(self) -> drjit.cuda.ad.Float: ...
    @power.setter
    def power(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def field_x(self) -> drjit.cuda.ad.Complex2f: ...
    @field_x.setter
    def field_x(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def field_y(self) -> drjit.cuda.ad.Complex2f: ...
    @field_y.setter
    def field_y(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def field_z(self) -> drjit.cuda.ad.Complex2f: ...
    @field_z.setter
    def field_z(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def direct_count(self) -> drjit.cuda.ad.Int: ...
    @direct_count.setter
    def direct_count(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def keller_count(self) -> drjit.cuda.ad.Int: ...
    @keller_count.setter
    def keller_count(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def suffix_count(self) -> drjit.cuda.ad.Int: ...
    @suffix_count.setter
    def suffix_count(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def vis_rejects(self) -> drjit.cuda.ad.Int: ...
    @vis_rejects.setter
    def vis_rejects(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge_vis_rejects(self) -> drjit.cuda.ad.Int: ...
    @edge_vis_rejects.setter
    def edge_vis_rejects(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def utd_rejects(self) -> drjit.cuda.ad.Int: ...
    @utd_rejects.setter
    def utd_rejects(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge_uses(self) -> drjit.cuda.ad.Int: ...
    @edge_uses.setter
    def edge_uses(self, arg: drjit.cuda.ad.Int, /) -> None: ...

class DfrCoherentAccum:
    """
    Coherent diffraction field accumulated per grid cell.

    Result record of Scene.accum_dfr_coherent_direct(). Grid arrays have grid_cell_count entries.
    """

    def __init__(self) -> None: ...
    @property
    def grid_cell_count(self) -> int: ...
    @grid_cell_count.setter
    def grid_cell_count(self, arg: int, /) -> None: ...
    @property
    def direct_field_x(self) -> drjit.cuda.Complex2f: ...
    @direct_field_x.setter
    def direct_field_x(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def direct_field_y(self) -> drjit.cuda.Complex2f: ...
    @direct_field_y.setter
    def direct_field_y(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def direct_field_z(self) -> drjit.cuda.Complex2f: ...
    @direct_field_z.setter
    def direct_field_z(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def multi_field_x(self) -> drjit.cuda.Complex2f: ...
    @multi_field_x.setter
    def multi_field_x(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def multi_field_y(self) -> drjit.cuda.Complex2f: ...
    @multi_field_y.setter
    def multi_field_y(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def multi_field_z(self) -> drjit.cuda.Complex2f: ...
    @multi_field_z.setter
    def multi_field_z(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def direct_count(self) -> drjit.cuda.Int: ...
    @direct_count.setter
    def direct_count(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def multi_count(self) -> drjit.cuda.Int: ...
    @multi_count.setter
    def multi_count(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def vis_rejects(self) -> drjit.cuda.Int: ...
    @vis_rejects.setter
    def vis_rejects(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def utd_rejects(self) -> drjit.cuda.Int: ...
    @utd_rejects.setter
    def utd_rejects(self, arg: drjit.cuda.Int, /) -> None: ...

class DfrCoherentAccumAD:
    """AD variant of DfrCoherentAccum."""

    def __init__(self) -> None: ...
    @property
    def grid_cell_count(self) -> int: ...
    @grid_cell_count.setter
    def grid_cell_count(self, arg: int, /) -> None: ...
    @property
    def direct_field_x(self) -> drjit.cuda.ad.Complex2f: ...
    @direct_field_x.setter
    def direct_field_x(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def direct_field_y(self) -> drjit.cuda.ad.Complex2f: ...
    @direct_field_y.setter
    def direct_field_y(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def direct_field_z(self) -> drjit.cuda.ad.Complex2f: ...
    @direct_field_z.setter
    def direct_field_z(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def multi_field_x(self) -> drjit.cuda.ad.Complex2f: ...
    @multi_field_x.setter
    def multi_field_x(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def multi_field_y(self) -> drjit.cuda.ad.Complex2f: ...
    @multi_field_y.setter
    def multi_field_y(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def multi_field_z(self) -> drjit.cuda.ad.Complex2f: ...
    @multi_field_z.setter
    def multi_field_z(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def direct_count(self) -> drjit.cuda.ad.Int: ...
    @direct_count.setter
    def direct_count(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def multi_count(self) -> drjit.cuda.ad.Int: ...
    @multi_count.setter
    def multi_count(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def vis_rejects(self) -> drjit.cuda.ad.Int: ...
    @vis_rejects.setter
    def vis_rejects(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def utd_rejects(self) -> drjit.cuda.ad.Int: ...
    @utd_rejects.setter
    def utd_rejects(self, arg: drjit.cuda.ad.Int, /) -> None: ...

class DfrPathOptions:
    """
    Options for Scene.trace_dfr_paths().

    Options record. The defaults export every state for every receiver.
    """

    def __init__(self) -> None: ...
    @property
    def wavelength(self) -> float:
        """Wavelength in world units; must be positive."""

    @wavelength.setter
    def wavelength(self, arg: float, /) -> None: ...
    @property
    def k(self) -> float:
        """
        Wavenumber used for path phases. It is used as given, not derived from wavelength.
        """

    @k.setter
    def k(self, arg: float, /) -> None: ...
    @property
    def max_order(self) -> int:
        """Diffraction order; only 1 is supported."""

    @max_order.setter
    def max_order(self, arg: int, /) -> None: ...
    @property
    def max_paths(self) -> int | None:
        """
        Maximum states exported per (transmitter, receiver) pair. None (the default) or 0 exports every state; reading it back gives None for both.
        """

    @max_paths.setter
    def max_paths(self, value: int | None) -> None: ...
    @property
    def max_rx(self) -> int:
        """Use at most this many receivers; 0 uses all."""

    @max_rx.setter
    def max_rx(self, arg: int, /) -> None: ...
    @property
    def strategy_mask(self) -> DfrStrategy:
        """Enabled strategies; must include DIRECT."""

    @strategy_mask.setter
    def strategy_mask(self, arg: DfrStrategy, /) -> None: ...
    @property
    def receiver_model(self) -> DfrReceiverModel:
        """Receiver model; only MATCHED_ISO is supported."""

    @receiver_model.setter
    def receiver_model(self, arg: DfrReceiverModel, /) -> None: ...

class DfrPaths:
    """
    Exported first-order diffraction paths.

    Result record of Scene.trace_dfr_paths(). Arrays are allocated to capacity, and the first count[0] entries are valid paths.
    """

    def __init__(self) -> None: ...
    @property
    def capacity(self) -> int: ...
    @capacity.setter
    def capacity(self, arg: int, /) -> None: ...
    @property
    def count(self) -> drjit.cuda.Int: ...
    @count.setter
    def count(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def valid(self) -> drjit.cuda.Bool: ...
    @valid.setter
    def valid(self, arg: drjit.cuda.Bool, /) -> None: ...
    @property
    def tx_id(self) -> drjit.cuda.Int: ...
    @tx_id.setter
    def tx_id(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def rx_id(self) -> drjit.cuda.Int: ...
    @rx_id.setter
    def rx_id(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def order(self) -> drjit.cuda.Int: ...
    @order.setter
    def order(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge0(self) -> drjit.cuda.Int: ...
    @edge0.setter
    def edge0(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge1(self) -> drjit.cuda.Int: ...
    @edge1.setter
    def edge1(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def edge2(self) -> drjit.cuda.Int: ...
    @edge2.setter
    def edge2(self, arg: drjit.cuda.Int, /) -> None: ...
    @property
    def delay(self) -> drjit.cuda.Float: ...
    @delay.setter
    def delay(self, arg: drjit.cuda.Float, /) -> None: ...
    @property
    def field_x(self) -> drjit.cuda.Complex2f: ...
    @field_x.setter
    def field_x(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def field_y(self) -> drjit.cuda.Complex2f: ...
    @field_y.setter
    def field_y(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def field_z(self) -> drjit.cuda.Complex2f: ...
    @field_z.setter
    def field_z(self, arg: drjit.cuda.Complex2f, /) -> None: ...
    @property
    def p0(self) -> drjit.cuda.Array3f: ...
    @p0.setter
    def p0(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def p1(self) -> drjit.cuda.Array3f: ...
    @p1.setter
    def p1(self, arg: drjit.cuda.Array3f, /) -> None: ...
    @property
    def p2(self) -> drjit.cuda.Array3f: ...
    @p2.setter
    def p2(self, arg: drjit.cuda.Array3f, /) -> None: ...

class DfrPathsAD:
    """
    AD variant of DfrPaths, returned by the AD overload of Scene.trace_dfr_paths().
    """

    def __init__(self) -> None: ...
    @property
    def capacity(self) -> int: ...
    @capacity.setter
    def capacity(self, arg: int, /) -> None: ...
    @property
    def count(self) -> drjit.cuda.ad.Int: ...
    @count.setter
    def count(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def valid(self) -> drjit.cuda.ad.Bool: ...
    @valid.setter
    def valid(self, arg: drjit.cuda.ad.Bool, /) -> None: ...
    @property
    def tx_id(self) -> drjit.cuda.ad.Int: ...
    @tx_id.setter
    def tx_id(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def rx_id(self) -> drjit.cuda.ad.Int: ...
    @rx_id.setter
    def rx_id(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def order(self) -> drjit.cuda.ad.Int: ...
    @order.setter
    def order(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge0(self) -> drjit.cuda.ad.Int: ...
    @edge0.setter
    def edge0(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge1(self) -> drjit.cuda.ad.Int: ...
    @edge1.setter
    def edge1(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def edge2(self) -> drjit.cuda.ad.Int: ...
    @edge2.setter
    def edge2(self, arg: drjit.cuda.ad.Int, /) -> None: ...
    @property
    def delay(self) -> drjit.cuda.ad.Float: ...
    @delay.setter
    def delay(self, arg: drjit.cuda.ad.Float, /) -> None: ...
    @property
    def field_x(self) -> drjit.cuda.ad.Complex2f: ...
    @field_x.setter
    def field_x(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def field_y(self) -> drjit.cuda.ad.Complex2f: ...
    @field_y.setter
    def field_y(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def field_z(self) -> drjit.cuda.ad.Complex2f: ...
    @field_z.setter
    def field_z(self, arg: drjit.cuda.ad.Complex2f, /) -> None: ...
    @property
    def p0(self) -> drjit.cuda.ad.Array3f: ...
    @p0.setter
    def p0(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def p1(self) -> drjit.cuda.ad.Array3f: ...
    @p1.setter
    def p1(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def p2(self) -> drjit.cuda.ad.Array3f: ...
    @p2.setter
    def p2(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...

class ReflEpc:
    """
    Reflection paths corrected to end at the receiver.

    Result record of Scene.trace_refl_epc(). Per-ray arrays have ray_count entries; per-slot arrays have ray_count * max_bounces entries.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def max_bounces(self) -> int: ...
    @property
    def valid(self) -> drjit.cuda.Bool: ...
    @property
    def bounce_count(self) -> drjit.cuda.Int: ...
    @property
    def path_length(self) -> drjit.cuda.Float: ...
    @property
    def reflection_points(self) -> drjit.cuda.Array3f: ...
    @property
    def prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def trace_prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def resolved_prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def surface_group_ids(self) -> drjit.cuda.Int: ...
    @property
    def plane_normals(self) -> drjit.cuda.Array3f: ...
    @property
    def first_blocked_segment(self) -> drjit.cuda.Int: ...
    @property
    def first_blocked_prim(self) -> drjit.cuda.Int: ...
    @property
    def first_blocked_group(self) -> drjit.cuda.Int: ...

class ReflEpcAD:
    """
    AD variant of ReflEpc. Scene.trace_refl_epc() never returns it because it rejects AD inputs.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def max_bounces(self) -> int: ...
    @property
    def valid(self) -> drjit.cuda.ad.Bool: ...
    @property
    def bounce_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def path_length(self) -> drjit.cuda.ad.Float: ...
    @property
    def reflection_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def trace_prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def resolved_prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def surface_group_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def plane_normals(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def first_blocked_segment(self) -> drjit.cuda.ad.Int: ...
    @property
    def first_blocked_prim(self) -> drjit.cuda.ad.Int: ...
    @property
    def first_blocked_group(self) -> drjit.cuda.ad.Int: ...

class ReflEpcField:
    """
    Reflection paths corrected to end at the receiver, with the complex field at the receiver.

    Result record of Scene.trace_refl_epc_field(). The field components are split into _re and _im arrays. Geometry and endpoint arrays are filled only when the matching ReflEpcFieldOptions flags are set.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def max_bounces(self) -> int: ...
    @property
    def valid(self) -> drjit.cuda.Bool: ...
    @property
    def bounce_count(self) -> drjit.cuda.Int: ...
    @property
    def path_length(self) -> drjit.cuda.Float: ...
    @property
    def field_x_re(self) -> drjit.cuda.Float: ...
    @property
    def field_x_im(self) -> drjit.cuda.Float: ...
    @property
    def field_y_re(self) -> drjit.cuda.Float: ...
    @property
    def field_y_im(self) -> drjit.cuda.Float: ...
    @property
    def field_z_re(self) -> drjit.cuda.Float: ...
    @property
    def field_z_im(self) -> drjit.cuda.Float: ...
    @property
    def tx_pos(self) -> drjit.cuda.Array3f: ...
    @property
    def first_hit(self) -> drjit.cuda.Array3f: ...
    @property
    def last_hit(self) -> drjit.cuda.Array3f: ...
    @property
    def hit_points(self) -> drjit.cuda.Array3f: ...
    @property
    def normals(self) -> drjit.cuda.Array3f: ...
    @property
    def resolved_prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def surface_group_ids(self) -> drjit.cuda.Int: ...

class ReflEpcFieldAD:
    """
    AD variant of ReflEpcField, returned by the differentiable Scene.trace_refl_epc_field() overloads.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def max_bounces(self) -> int: ...
    @property
    def valid(self) -> drjit.cuda.ad.Bool: ...
    @property
    def bounce_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def path_length(self) -> drjit.cuda.ad.Float: ...
    @property
    def field_x_re(self) -> drjit.cuda.ad.Float: ...
    @property
    def field_x_im(self) -> drjit.cuda.ad.Float: ...
    @property
    def field_y_re(self) -> drjit.cuda.ad.Float: ...
    @property
    def field_y_im(self) -> drjit.cuda.ad.Float: ...
    @property
    def field_z_re(self) -> drjit.cuda.ad.Float: ...
    @property
    def field_z_im(self) -> drjit.cuda.ad.Float: ...
    @property
    def tx_pos(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def first_hit(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def last_hit(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def hit_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def normals(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def resolved_prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def surface_group_ids(self) -> drjit.cuda.ad.Int: ...

class ReflectionBounce:
    """
    One reflection bounce for a batch of rays.

    Result record returned by ReflectionTrace.bounce().
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def t(self) -> drjit.cuda.Float: ...
    @property
    def hit_points(self) -> drjit.cuda.Array3f: ...
    @property
    def geo_normals(self) -> drjit.cuda.Array3f: ...
    @property
    def image_sources(self) -> drjit.cuda.Array3f: ...
    @property
    def plane_points(self) -> drjit.cuda.Array3f: ...
    @property
    def plane_normals(self) -> drjit.cuda.Array3f: ...
    @property
    def shape_ids(self) -> drjit.cuda.Int: ...
    @property
    def prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def local_prim_ids(self) -> drjit.cuda.Int: ...
    @property
    def global_prim_ids(self) -> drjit.cuda.Int: ...

class ReflectionBounceAD:
    """AD variant of ReflectionBounce."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def t(self) -> drjit.cuda.ad.Float: ...
    @property
    def hit_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def geo_normals(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def image_sources(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def plane_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def plane_normals(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def shape_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def local_prim_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def global_prim_ids(self) -> drjit.cuda.ad.Int: ...

class ReflectionTrace:
    """
    Reflection paths stored as one ReflectionBounce per bounce level.

    Result record of Scene.trace_reflections() with symbolic=True.
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    def bounce(self, index: int) -> ReflectionBounce:
        """
        Return the ReflectionBounce at bounce level index. An index outside [0, max_bounces) raises IndexError.
        """

    @property
    def max_bounces(self) -> int: ...
    @property
    def ray_count(self) -> int: ...
    @property
    def deduplicate_requested(self) -> bool: ...
    @property
    def deduplicate_applied(self) -> bool: ...
    @property
    def bounce_count(self) -> drjit.cuda.Int: ...
    @property
    def discovery_count(self) -> drjit.cuda.Int: ...
    @property
    def representative_ray_index(self) -> drjit.cuda.Int: ...
    @property
    def dedup_keep_mask(self) -> drjit.cuda.Bool: ...
    @property
    def bounces(self) -> list[ReflectionBounce]: ...

class ReflectionTraceAD:
    """AD variant of ReflectionTrace."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    def bounce(self, index: int) -> ReflectionBounceAD:
        """
        Return the ReflectionBounce at bounce level index. An index outside [0, max_bounces) raises IndexError.
        """

    @property
    def max_bounces(self) -> int: ...
    @property
    def ray_count(self) -> int: ...
    @property
    def deduplicate_requested(self) -> bool: ...
    @property
    def deduplicate_applied(self) -> bool: ...
    @property
    def bounce_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def discovery_count(self) -> drjit.cuda.ad.Int: ...
    @property
    def representative_ray_index(self) -> drjit.cuda.ad.Int: ...
    @property
    def dedup_keep_mask(self) -> drjit.cuda.ad.Bool: ...
    @property
    def bounces(self) -> list[ReflectionBounceAD]: ...

class NearestPointEdge:
    """
    Nearest scene edge to each query point.

    Result record of Scene.nearest_edge() with a point batch. Rows with no edge hold distance = inf and ids = -1.
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def distance(self) -> drjit.cuda.Float: ...
    @property
    def point(self) -> drjit.cuda.Array3f: ...
    @property
    def edge_t(self) -> drjit.cuda.Float: ...
    @property
    def edge_point(self) -> drjit.cuda.Array3f: ...
    @property
    def shape_id(self) -> drjit.cuda.Int: ...
    @property
    def edge_id(self) -> drjit.cuda.Int: ...
    @property
    def global_edge_id(self) -> drjit.cuda.Int: ...
    @property
    def is_boundary(self) -> drjit.cuda.Bool: ...

class NearestPointEdgeAD:
    """AD variant of NearestPointEdge."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def distance(self) -> drjit.cuda.ad.Float: ...
    @property
    def point(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def edge_t(self) -> drjit.cuda.ad.Float: ...
    @property
    def edge_point(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def shape_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def edge_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def global_edge_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def is_boundary(self) -> drjit.cuda.ad.Bool: ...

class NearestRayEdge:
    """
    Nearest scene edge to each query ray or segment.

    Result record of Scene.nearest_edge() with a Ray. Rows with no edge hold distance = inf and ids = -1.
    """

    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def distance(self) -> drjit.cuda.Float: ...
    @property
    def ray_t(self) -> drjit.cuda.Float: ...
    @property
    def point(self) -> drjit.cuda.Array3f: ...
    @property
    def edge_t(self) -> drjit.cuda.Float: ...
    @property
    def edge_point(self) -> drjit.cuda.Array3f: ...
    @property
    def shape_id(self) -> drjit.cuda.Int: ...
    @property
    def edge_id(self) -> drjit.cuda.Int: ...
    @property
    def global_edge_id(self) -> drjit.cuda.Int: ...
    @property
    def is_boundary(self) -> drjit.cuda.Bool: ...

class NearestRayEdgeAD:
    """AD variant of NearestRayEdge."""

    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def distance(self) -> drjit.cuda.ad.Float: ...
    @property
    def ray_t(self) -> drjit.cuda.ad.Float: ...
    @property
    def point(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def edge_t(self) -> drjit.cuda.ad.Float: ...
    @property
    def edge_point(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def shape_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def edge_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def global_edge_id(self) -> drjit.cuda.ad.Int: ...
    @property
    def is_boundary(self) -> drjit.cuda.ad.Bool: ...

class NearestEdgesTopK:
    """
    The k nearest scene edges to each query point.

    Result record of Scene.nearest_edges(). Arrays have query_count * k slots, each query's k results contiguous and ordered nearest first; valid marks filled slots.
    """

    @property
    def query_count(self) -> int: ...
    @property
    def k(self) -> int: ...
    @property
    def valid(self) -> drjit.cuda.Bool: ...
    def is_valid(self) -> drjit.cuda.Bool:
        """Return the per-row validity mask."""

    @property
    def distances(self) -> drjit.cuda.Float: ...
    @property
    def points(self) -> drjit.cuda.Array3f: ...
    @property
    def edge_t(self) -> drjit.cuda.Float: ...
    @property
    def edge_points(self) -> drjit.cuda.Array3f: ...
    @property
    def shape_ids(self) -> drjit.cuda.Int: ...
    @property
    def edge_ids(self) -> drjit.cuda.Int: ...
    @property
    def global_edge_ids(self) -> drjit.cuda.Int: ...
    @property
    def is_boundary(self) -> drjit.cuda.Bool: ...

class NearestEdgesTopKAD:
    """AD variant of NearestEdgesTopK."""

    @property
    def query_count(self) -> int: ...
    @property
    def k(self) -> int: ...
    @property
    def valid(self) -> drjit.cuda.ad.Bool: ...
    def is_valid(self) -> drjit.cuda.ad.Bool:
        """Return the per-row validity mask."""

    @property
    def distances(self) -> drjit.cuda.ad.Float: ...
    @property
    def points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def edge_t(self) -> drjit.cuda.ad.Float: ...
    @property
    def edge_points(self) -> drjit.cuda.ad.Array3f: ...
    @property
    def shape_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def edge_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def global_edge_ids(self) -> drjit.cuda.ad.Int: ...
    @property
    def is_boundary(self) -> drjit.cuda.ad.Bool: ...

class SegmentVisibility:
    """
    Per-segment visibility mask.

    Result record of Scene.visible() and MixedScene.visible(). visible is True where the endpoints see each other.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def visible(self) -> drjit.cuda.Bool: ...

class SegmentVisibilityAD:
    """
    AD variant of SegmentVisibility. Visibility is discrete and has no derivative.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def visible(self) -> drjit.cuda.ad.Bool: ...

class SegmentPairVisibility:
    """
    Visibility of two endpoints from a shared start point.

    Result record of Scene.visible_pair().
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def visible_a(self) -> drjit.cuda.Bool: ...
    @property
    def visible_b(self) -> drjit.cuda.Bool: ...

class SegmentPairVisibilityAD:
    """
    AD variant of SegmentPairVisibility. Visibility is discrete and has no derivative.
    """

    @property
    def ray_count(self) -> int: ...
    @property
    def visible_a(self) -> drjit.cuda.ad.Bool: ...
    @property
    def visible_b(self) -> drjit.cuda.ad.Bool: ...

class AxialEdgeVisibility:
    """
    Whether the source sees any sampled point of each edge.

    Result record of Scene.visible_edge().
    """

    @property
    def state_count(self) -> int: ...
    @property
    def any_visible(self) -> drjit.cuda.Bool: ...

class AxialEdgeVisibilityAD:
    """
    AD variant of AxialEdgeVisibility. Visibility is discrete and has no derivative.
    """

    @property
    def state_count(self) -> int: ...
    @property
    def any_visible(self) -> drjit.cuda.ad.Bool: ...

class SegmentChainVisibility:
    """
    Visibility of every segment of each polyline, with the first blocker.

    Result record of Scene.visible_chain(). Blocker ids are scene-global primitive ids.
    """

    @property
    def chain_count(self) -> int: ...
    @property
    def max_segments(self) -> int: ...
    @property
    def all_visible(self) -> drjit.cuda.Bool: ...
    @property
    def first_blocked_segment(self) -> drjit.cuda.Int: ...
    @property
    def first_blocked_prim(self) -> drjit.cuda.Int: ...

class SegmentChainVisibilityAD:
    """
    AD variant of SegmentChainVisibility. Visibility is discrete and has no derivative.
    """

    @property
    def chain_count(self) -> int: ...
    @property
    def max_segments(self) -> int: ...
    @property
    def all_visible(self) -> drjit.cuda.ad.Bool: ...
    @property
    def first_blocked_segment(self) -> drjit.cuda.ad.Int: ...
    @property
    def first_blocked_prim(self) -> drjit.cuda.ad.Int: ...

class SceneSyncProfile:
    """
    Timings (milliseconds) and update counts of the last Scene.sync().

    Result record returned by Scene.last_sync_profile.
    """

    @property
    def mesh_update_ms(self) -> float: ...
    @property
    def triangle_scatter_ms(self) -> float: ...
    @property
    def triangle_eval_ms(self) -> float: ...
    @property
    def edge_scatter_ms(self) -> float: ...
    @property
    def edge_refit_ms(self) -> float: ...
    @property
    def optix_sync_ms(self) -> float: ...
    @property
    def total_ms(self) -> float: ...
    @property
    def optix_gas_update_ms(self) -> float: ...
    @property
    def optix_ias_update_ms(self) -> float: ...
    @property
    def updated_meshes(self) -> int: ...
    @property
    def updated_vertex_meshes(self) -> int: ...
    @property
    def updated_transform_meshes(self) -> int: ...
    @property
    def updated_edge_meshes(self) -> int: ...
    @property
    def updated_edges(self) -> int: ...

class SceneEdgeBVHStats:
    """
    Structure and quality metrics of the built edge BVH, for diagnostics.

    Result record returned by Scene.edge_bvh_stats().
    """

    @property
    def primitive_count(self) -> int: ...
    @property
    def node_count(self) -> int: ...
    @property
    def internal_node_count(self) -> int: ...
    @property
    def leaf_node_count(self) -> int: ...
    @property
    def max_height(self) -> int: ...
    @property
    def refit_level_count(self) -> int: ...
    @property
    def min_leaf_size(self) -> int: ...
    @property
    def max_leaf_size(self) -> int: ...
    @property
    def avg_leaf_size(self) -> float: ...
    @property
    def root_surface_area(self) -> float: ...
    @property
    def internal_surface_area_sum(self) -> float: ...
    @property
    def sibling_overlap_surface_area_sum(self) -> float: ...
    @property
    def sibling_overlap_surface_area_avg(self) -> float: ...
    @property
    def normalized_sibling_overlap(self) -> float: ...
    @property
    def leaf_size_histogram(self) -> list[int]: ...

class SurfelGeometry:
    """
    Surfel positions and tangent axes.

    Input record for SurfelScene. Each surfel is a 2D Gaussian centered at center whose local axes are tangent_u and tangent_v. All three arrays have one entry per surfel.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self, center: drjit.cuda.Array3f, tangent_u: drjit.cuda.Array3f, tangent_v: drjit.cuda.Array3f
    ) -> None:
        """
        Create surfel geometry from per-surfel center, tangent_u, and tangent_v arrays.
        """

    @overload
    def __init__(
        self, center: drjit.cuda.ad.Array3f, tangent_u: drjit.cuda.ad.Array3f, tangent_v: drjit.cuda.ad.Array3f
    ) -> None: ...
    @property
    def surfel_count(self) -> int:
        """Number of surfels."""

    @property
    def center(self) -> drjit.cuda.ad.Array3f:
        """Surfel centers."""

    @property
    def tangent_u(self) -> drjit.cuda.ad.Array3f:
        """
        First tangent axis of each surfel; its length is the Gaussian scale along it.
        """

    @property
    def tangent_v(self) -> drjit.cuda.ad.Array3f:
        """
        Second tangent axis of each surfel; its length is the Gaussian scale along it.
        """

class SurfelAppearance:
    """
    Per-surfel opacity and color values.

    Input record for SurfelScene.update_appearance(). values holds surfel_count * values_per_surfel entries, where values_per_surfel is channel_count, or 3 * (sh_degree + 1)^2 for the SH model.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        opacity: drjit.cuda.Float,
        values: drjit.cuda.Float,
        color_model: SurfelColorModel,
        channel_count: int,
        sh_degree: int = 0,
    ) -> None:
        """
        Create an appearance from opacity (one per surfel) and values laid out for color_model.
        """

    @overload
    def __init__(
        self,
        opacity: drjit.cuda.ad.Float,
        values: drjit.cuda.ad.Float,
        color_model: SurfelColorModel,
        channel_count: int,
        sh_degree: int = 0,
    ) -> None: ...
    @overload
    @staticmethod
    def rgb(opacity: drjit.cuda.Float, rgb: drjit.cuda.Array3f) -> SurfelAppearance:
        """Return a ConstantRGB appearance from per-surfel opacity and RGB color."""

    @overload
    @staticmethod
    def rgb(opacity: drjit.cuda.ad.Float, rgb: drjit.cuda.ad.Array3f) -> SurfelAppearance: ...
    @overload
    @staticmethod
    def features(opacity: drjit.cuda.Float, values: drjit.cuda.Float, channel_count: int) -> SurfelAppearance:
        """
        Return a FeatureChannels appearance with channel_count values per surfel.
        """

    @overload
    @staticmethod
    def features(opacity: drjit.cuda.ad.Float, values: drjit.cuda.ad.Float, channel_count: int) -> SurfelAppearance: ...
    @overload
    @staticmethod
    def sh(opacity: drjit.cuda.Float, coeffs: drjit.cuda.Float, sh_degree: int) -> SurfelAppearance:
        """
        Return an SH appearance from RGB spherical-harmonic coefficients of degree sh_degree in [0, 3].
        """

    @overload
    @staticmethod
    def sh(opacity: drjit.cuda.ad.Float, coeffs: drjit.cuda.ad.Float, sh_degree: int) -> SurfelAppearance: ...
    @property
    def surfel_count(self) -> int:
        """Number of surfels."""

    @property
    def channel_count(self) -> int:
        """Number of color channels."""

    @property
    def sh_degree(self) -> int:
        """Spherical-harmonic degree; 0 for the non-SH models."""

    @property
    def color_model(self) -> SurfelColorModel:
        """Color model of values."""

    @property
    def opacity(self) -> drjit.cuda.ad.Float:
        """Per-surfel opacity."""

    @property
    def values(self) -> drjit.cuda.ad.Float:
        """Flat per-surfel color values."""

class SurfelCloud:
    """
    Surfel geometry with one opacity and one scalar value per surfel.

    Input record for SurfelScene and MixedScene.add_surfel().
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        center: drjit.cuda.Array3f,
        tangent_u: drjit.cuda.Array3f,
        tangent_v: drjit.cuda.Array3f,
        opacity: drjit.cuda.Float = ...,
        value: drjit.cuda.Float = ...,
    ) -> None:
        """
        Create a surfel cloud. opacity and value default to 1 for every surfel; when given they need one entry per surfel.
        """

    @overload
    def __init__(
        self,
        center: drjit.cuda.ad.Array3f,
        tangent_u: drjit.cuda.ad.Array3f,
        tangent_v: drjit.cuda.ad.Array3f,
        opacity: drjit.cuda.ad.Float = ...,
        value: drjit.cuda.ad.Float = ...,
    ) -> None: ...
    @property
    def surfel_count(self) -> int:
        """Number of surfels."""

    @property
    def center(self) -> drjit.cuda.ad.Array3f:
        """Surfel centers."""

    @property
    def tangent_u(self) -> drjit.cuda.ad.Array3f:
        """First tangent axis of each surfel."""

    @property
    def tangent_v(self) -> drjit.cuda.ad.Array3f:
        """Second tangent axis of each surfel."""

    @property
    def opacity(self) -> drjit.cuda.ad.Float:
        """Per-surfel opacity."""

    @property
    def value(self) -> drjit.cuda.ad.Float:
        """Per-surfel scalar value."""

class SurfelScene:
    """
    Ray-tracing scene of 2D Gaussian surfels, traced through proxy triangles with OptiX.

    Build it with build() before querying. Queries must run on the CUDA device it was built on.
    """

    @overload
    def __init__(self, cloud: SurfelCloud, options: SurfelTraceOptions = ...) -> None:
        """
        Create a surfel scene from a SurfelCloud or a SurfelGeometry. A SurfelGeometry gets opacity and value 1. Invalid options raise RuntimeError.
        """

    @overload
    def __init__(self, geometry: SurfelGeometry, options: SurfelTraceOptions = ...) -> None: ...
    def build(self) -> None:
        """
        Build the proxy triangles and the OptiX acceleration structure on the current CUDA device.
        """

    def is_ready(self) -> bool:
        """Return True once build() has run."""

    def update_geometry(self, geometry: SurfelGeometry) -> None:
        """
        Replace the surfel geometry and rebuild the acceleration structure. The scene must be built and the surfel count must not change.
        """

    def update_appearance(self, appearance: SurfelAppearance) -> None:
        """Replace the surfel appearance; the surfel count must not change."""

    @property
    def surfel_count(self) -> int:
        """Number of surfels."""

    @property
    def triangle_count(self) -> int:
        """Number of proxy triangles."""

    @property
    def build_count(self) -> int:
        """Number of builds so far, update_geometry() included."""

    @overload
    def intersect(self, ray: Ray, active: drjit.cuda.Bool = True) -> SurfelIntersection:
        """
        Return the closest surfel hit per ray as a SurfelIntersection.

        A RayAD returns SurfelIntersectionAD with the hit attributes recomputed from the AD surfel parameters.
        """

    @overload
    def intersect(self, ray: RayAD, active: drjit.cuda.ad.Bool = True) -> SurfelIntersectionAD: ...
    @overload
    def composite_alpha(self, ray: Ray, active: drjit.cuda.Bool = True) -> SurfelComposite:
        """
        Alpha-composite the surfels along each ray, front to back, and return a SurfelComposite.
        """

    @overload
    def composite_alpha(self, ray: RayAD, active: drjit.cuda.ad.Bool = True) -> SurfelCompositeAD: ...
    @overload
    def composite_alpha_reference(self, ray: Ray, active: drjit.cuda.Bool = True) -> SurfelComposite:
        """
        Compute the same composite as composite_alpha() with the reference implementation instead of the native pipeline.
        """

    @overload
    def composite_alpha_reference(self, ray: RayAD, active: drjit.cuda.ad.Bool = True) -> SurfelCompositeAD: ...
    @overload
    def render(
        self, ray: Ray, render_options: SurfelRenderOptions = ..., active: drjit.cuda.Bool = True
    ) -> SurfelRender:
        """
        Render each ray with render_options and return a SurfelRender.

        render_options must match the appearance: its color model, an SH degree no higher than the appearance's, and no more feature channels than it has. A mismatch raises RuntimeError.
        """

    @overload
    def render(
        self, ray: RayAD, render_options: SurfelRenderOptions = ..., active: drjit.cuda.ad.Bool = True
    ) -> SurfelRenderAD: ...
    @overload
    def shadow_test(self, ray: Ray, active: drjit.cuda.Bool = True) -> drjit.cuda.Bool:
        """Return a mask that is True where the ray hits any surfel."""

    @overload
    def shadow_test(self, ray: RayAD, active: drjit.cuda.ad.Bool = True) -> drjit.cuda.ad.Bool: ...
    @overload
    def visible(
        self, start: drjit.cuda.Array3f, end: drjit.cuda.Array3f, active: drjit.cuda.Bool = True
    ) -> drjit.cuda.Bool:
        """
        Return a mask that is True where no surfel blocks the segment from start to end. Degenerate segments are not visible.
        """

    @overload
    def visible(
        self, start: drjit.cuda.ad.Array3f, end: drjit.cuda.ad.Array3f, active: drjit.cuda.ad.Bool = True
    ) -> drjit.cuda.ad.Bool: ...
    @overload
    def trace_reflections(self, ray: Ray, max_bounces: int, active: drjit.cuda.Bool = True) -> ReflectionChain:
        """
        Trace specular reflections off the surfels for up to max_bounces bounces and return a ReflectionChain. max_bounces must be non-negative.
        """

    @overload
    def trace_reflections(
        self, ray: RayAD, max_bounces: int, active: drjit.cuda.ad.Bool = True
    ) -> ReflectionChainAD: ...

class SdfGrid:
    """
    Signed distance field sampled on a regular grid inside an oriented box.

    Standalone geometry for ray queries; it can also be added to a MixedScene.
    """

    @overload
    def __init__(
        self,
        values: drjit.cuda.Float,
        nx: int,
        ny: int,
        nz: int,
        position: drjit.cuda.Array3f,
        rotation: drjit.cuda.Float,
        scale: drjit.cuda.Array3f,
    ) -> None:
        """
        Create a grid from vertex-centered samples.

        Args:
            values: nx * ny * nz signed distances in world units, negative inside, flattened with z fastest: index (i * ny + j) * nz + k.
            nx, ny, nz: Sample counts per axis, each at least 2.
            position: World-space center of the box (one vector).
            rotation: Box rotation as a unit quaternion of four entries in (w, x, y, z) order.
            scale: Full side lengths of the box (one vector).

        AD arrays make the grid differentiable with respect to values and placement. Invalid shapes raise RuntimeError.
        """

    @overload
    def __init__(
        self,
        values: drjit.cuda.ad.Float,
        nx: int,
        ny: int,
        nz: int,
        position: drjit.cuda.ad.Array3f,
        rotation: drjit.cuda.ad.Float,
        scale: drjit.cuda.ad.Array3f,
    ) -> None: ...
    @property
    def nx(self) -> int:
        """Sample count along x."""

    @property
    def ny(self) -> int:
        """Sample count along y."""

    @property
    def nz(self) -> int:
        """Sample count along z."""

    @property
    def value_count(self) -> int:
        """Total sample count, nx * ny * nz."""

    @property
    def values(self) -> drjit.cuda.ad.Float:
        """Flat signed-distance samples."""

    @property
    def position(self) -> drjit.cuda.ad.Array3f:
        """World-space box center."""

    @property
    def rotation(self) -> drjit.cuda.ad.Float:
        """Box rotation quaternion in (w, x, y, z) order."""

    @property
    def scale(self) -> drjit.cuda.ad.Array3f:
        """Box side lengths."""

    @overload
    def intersect(self, ray: Ray, options: SdfTraceOptions = ..., active: drjit.cuda.Bool = True) -> SdfIntersection:
        """
        Sphere-trace each ray against the zero level set and return an SdfIntersection.

        Tracing is limited to the oriented box and ray.tmax. With a RayAD the result is differentiable with respect to the ray, the values, and the placement (position, rotation, scale); the hit is fixed by the forward pass.
        """

    @overload
    def intersect(
        self, ray: RayAD, options: SdfTraceOptions = ..., active: drjit.cuda.ad.Bool = True
    ) -> SdfIntersectionAD: ...
    @overload
    def visible(
        self,
        start: drjit.cuda.Array3f,
        end: drjit.cuda.Array3f,
        options: SdfTraceOptions = ...,
        active: drjit.cuda.Bool = True,
    ) -> drjit.cuda.Bool:
        """
        Return a mask that is True where the SDF surface does not block the segment from start to end.
        """

    @overload
    def visible(
        self,
        start: drjit.cuda.ad.Array3f,
        end: drjit.cuda.ad.Array3f,
        options: SdfTraceOptions = ...,
        active: drjit.cuda.ad.Bool = True,
    ) -> drjit.cuda.ad.Bool: ...
    @overload
    def trace_reflections(
        self, ray: Ray, max_bounces: int, options: SdfTraceOptions = ..., active: drjit.cuda.Bool = True
    ) -> ReflectionChain:
        """
        Trace specular reflections off the SDF surface for up to max_bounces bounces and return a ReflectionChain. max_bounces must be non-negative.
        """

    @overload
    def trace_reflections(
        self, ray: RayAD, max_bounces: int, options: SdfTraceOptions = ..., active: drjit.cuda.ad.Bool = True
    ) -> ReflectionChainAD: ...

class Mesh:
    """
    Triangle mesh with optional UVs and a world transform.

    Input geometry for Scene.add_mesh() and MixedScene.add_mesh(). The world transform is to_world_left @ to_world @ to_world_right.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        v: drjit.cuda.Array3f | drjit.cuda.ad.Array3f,
        f: drjit.cuda.Array3i,
        uv: drjit.cuda.Array2f | drjit.cuda.ad.Array2f = ...,
        f_uv: drjit.cuda.Array3i = ...,
        verbose: bool = False,
    ) -> None:
        """
        Create a mesh.

        Args:
            v: Object-space vertex positions, one entry per vertex.
            f: Vertex indices of each triangle.
            uv: Optional texture coordinates, indexed by f_uv when given and by vertex otherwise.
            f_uv: Optional UV indices of each triangle; when omitted, uv needs one entry per vertex.
            verbose: Print the vertex, face, and edge counts after loading.

        Grad-enabled drjit.cuda.ad positions or UVs are kept as AD arrays so gradients reach them.
        """

    def build(self) -> None:
        """
        Build the mesh's derived geometry and GPU buffers from its current state.
        """

    def set_transform(self, mat: drjit.cuda.ad.Matrix4f, set_left: bool = True) -> None:
        """
        Replace the left (set_left=True) or right factor of the world transform with mat.
        """

    def append_transform(self, mat: drjit.cuda.ad.Matrix4f, append_left: bool = True) -> None:
        """
        Compose mat onto the left (append_left=True, pre-multiply) or right (post-multiply) transform factor.
        """

    def edge_indices(self) -> tuple:
        """
        Return the per-edge connectivity as five Int arrays: (vertex0, vertex1, face0, face1, opposite_vertex).

        face1 is -1 for boundary edges. Raises RuntimeError if the faces changed since build() or admit no supported edge topology.
        """

    def secondary_edges(self) -> SecondaryEdgeInfo:
        """
        Return the world-space per-edge data as SecondaryEdgeInfo, or an empty record before it is computed.
        """

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""

    @property
    def num_faces(self) -> int:
        """Number of triangles."""

    @property
    def to_world(self) -> drjit.cuda.ad.Matrix4f:
        """Middle factor of the world transform."""

    @to_world.setter
    def to_world(self, arg: drjit.cuda.ad.Matrix4f, /) -> None: ...
    @property
    def to_world_left(self) -> drjit.cuda.ad.Matrix4f:
        """Left factor of the world transform."""

    @to_world_left.setter
    def to_world_left(self, arg: drjit.cuda.ad.Matrix4f, /) -> None: ...
    @property
    def to_world_right(self) -> drjit.cuda.ad.Matrix4f:
        """Right factor of the world transform."""

    @to_world_right.setter
    def to_world_right(self, arg: drjit.cuda.ad.Matrix4f, /) -> None: ...
    @property
    def vertex_positions(self) -> drjit.cuda.ad.Array3f:
        """Object-space vertex positions."""

    @vertex_positions.setter
    def vertex_positions(self, arg: drjit.cuda.ad.Array3f, /) -> None: ...
    @property
    def vertex_positions_world(self) -> drjit.cuda.ad.Array3f:
        """World-space vertex positions."""

    @property
    def vertex_normals(self) -> drjit.cuda.ad.Array3f:
        """Object-space area-weighted vertex normals, filled by build()."""

    @property
    def vertex_uv(self) -> drjit.cuda.ad.Array2f:
        """Texture coordinates."""

    @vertex_uv.setter
    def vertex_uv(self, arg: drjit.cuda.ad.Array2f, /) -> None: ...
    @property
    def face_indices(self) -> drjit.cuda.ad.Array3i:
        """Vertex indices of each triangle."""

    @face_indices.setter
    def face_indices(self, arg: drjit.cuda.ad.Array3i, /) -> None: ...
    @property
    def face_uv_indices(self) -> drjit.cuda.ad.Array3i:
        """UV indices of each triangle."""

    @face_uv_indices.setter
    def face_uv_indices(self, arg: drjit.cuda.ad.Array3i, /) -> None: ...
    @property
    def use_face_normals(self) -> bool:
        """Use the geometric face normal as the shading normal."""

    @use_face_normals.setter
    def use_face_normals(self, arg: bool, /) -> None: ...
    @property
    def edges_enabled(self) -> bool:
        """Include this mesh's edges in the scene edge set used by edge queries."""

    @edges_enabled.setter
    def edges_enabled(self, arg: bool, /) -> None: ...
    def __repr__(self) -> str: ...

class Scene:
    """
    Triangle-mesh scene with its ray-tracing and edge acceleration structures.

    Add meshes with add_mesh() or add_instance(), call build(), then run queries. Dynamic meshes can be edited and refitted with sync().
    """

    def __init__(self, *, trace_backend: str = "auto", edge_bvh_backend: str = "auto") -> None:
        """
        Create an empty scene.

        Args:
            trace_backend: Triangle trace backend. 'auto' selects 'optix' when OptiX is available and 'cuda' otherwise; 'optix' requires OptiX at build(); 'cuda' uses a CUDA BVH; 'none' builds no triangle backend, which leaves only the edge queries.
            edge_bvh_backend: Nearest-edge backend: 'auto' (same as 'drjit'), 'drjit', 'optix', or 'optix_drjit'.

        Backend strings match exactly and case-sensitively; any other value raises ValueError.
        """

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        """
        Add a copy of mesh and return its mesh id.

        Adding a mesh invalidates a previous build(). Pass dynamic=True to allow later update_mesh_vertices() and transform edits applied by sync().
        """

    def add_instance(self, geometry_id: int, transform: drjit.cuda.ad.Matrix4f, dynamic: bool = True) -> int:
        """
        Add a transformed instance of an existing static mesh and return its instance id.

        The instance shares the acceleration structure of mesh geometry_id but has its own 4x4 transform and its own scene-global primitive range. A dynamic instance can be moved with set_mesh_transform() or append_mesh_transform() followed by sync(). Instancing a dynamic mesh raises RuntimeError.
        """

    def build(self) -> None:
        """
        Build the triangle and edge acceleration structures. Call it before any query.

        Raises RuntimeError when the scene has no meshes or no triangles, or when a world-space vertex coordinate is NaN or infinite.
        """

    def update_mesh_vertices(self, mesh_id: int, positions: drjit.cuda.ad.Array3f) -> None:
        """
        Queue new object-space vertex positions for dynamic mesh mesh_id; sync() applies them.

        positions needs one entry per mesh vertex. Raises RuntimeError if the scene is not built or the mesh is not dynamic.
        """

    def set_mesh_transform(self, mesh_id: int, mat: drjit.cuda.ad.Matrix4f, set_left: bool = True) -> None:
        """
        Queue a new left (set_left=True) or right transform factor for dynamic mesh mesh_id; sync() applies it.

        Raises RuntimeError if the scene is not built or the mesh is not dynamic.
        """

    def append_mesh_transform(self, mesh_id: int, mat: drjit.cuda.ad.Matrix4f, append_left: bool = True) -> None:
        """
        Queue mat composed onto the left (append_left=True) or right transform factor of dynamic mesh mesh_id; sync() applies it.

        Raises RuntimeError if the scene is not built or the mesh is not dynamic.
        """

    @overload
    def set_edge_mask(self, mask: drjit.cuda.Bool) -> None:
        """
        Restrict the edge queries to the edges where mask is True.

        mask has one entry per scene-global edge. Raises RuntimeError if the scene is not built or the size does not match.
        """

    @overload
    def set_edge_mask(self, mask: drjit.cuda.ad.Bool) -> None: ...
    def sync(self) -> None:
        """
        Apply the queued dynamic-mesh edits and refit the acceleration structures in place.

        Raises RuntimeError if the scene is not built or an updated vertex is NaN or infinite; a failed sync keeps its updates pending.
        """

    def is_ready(self) -> bool:
        """Return True once the scene is built and needs no rebuild."""

    def has_pending_updates(self) -> bool:
        """Return True when edits are queued that sync() has not applied yet."""

    @property
    def last_sync_profile(self) -> SceneSyncProfile:
        """Timings and update counts of the last sync()."""

    def edge_info(self) -> SceneEdgeInfo:
        """
        Return the scene-global edge geometry and ids as SceneEdgeInfo. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """

    @property
    def edge_bvh_backend(self) -> str:
        """Resolved edge backend name: 'drjit', 'optix', or 'optix_drjit'."""

    @property
    def trace_backend(self) -> str:
        """
        Canonical name of the resolved triangle trace backend ('optix', 'cuda', or 'none').
        """

    def capabilities(self) -> dict[str, object]:
        """
        Machine-readable capability map for this scene's resolved backend plan (trace backend, edge backend, integration modes, and per-operation availability).
        """

    def edge_bvh_stats(self) -> SceneEdgeBVHStats:
        """
        Return structure and quality metrics of the edge BVH. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """

    def edge_topology(self) -> SceneEdgeTopology:
        """Return the scene-global edge connectivity as SceneEdgeTopology."""

    def edge_mask(self) -> drjit.cuda.Bool:
        """Return the per-edge mask set by set_edge_mask()."""

    def mesh_face_offsets(self) -> drjit.cuda.Int:
        """
        Return the prefix sum of face counts: mesh m owns faces [offset[m], offset[m + 1]).
        """

    def mesh_edge_offsets(self) -> drjit.cuda.Int:
        """Return the prefix sum of per-mesh edge counts."""

    def mesh_vertex_offsets(self) -> drjit.cuda.Int:
        """Return the prefix sum of per-mesh vertex counts."""

    def global_geometry(self) -> SceneGeometry:
        """
        Return the vertices, faces, normals, and ids of all meshes in one scene-global index space. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """

    def triangle_edge_indices(
        self, prim_id: drjit.cuda.Int, global_: bool = True
    ) -> tuple[drjit.cuda.Int, drjit.cuda.Int, drjit.cuda.Int]:
        """
        Return the three edge ids of each scene-global triangle prim_id.

        global_ selects scene-global edge ids (True) or per-mesh edge ids. Out-of-range triangles return -1.
        """

    def edge_adjacent_faces(
        self, edge_id: drjit.cuda.Int, global_: bool = True
    ) -> tuple[drjit.cuda.Int, drjit.cuda.Int]:
        """
        Return the two faces adjacent to each scene-global edge edge_id.

        global_ selects scene-global face ids (True) or per-mesh face ids. The second face is -1 on a boundary edge, and out-of-range edges return -1.
        """

    @overload
    def intersect(self, ray: Ray, active: drjit.cuda.Bool = True, flags: RayFlags = RayFlags.All) -> Intersection:
        """
        Find the closest triangle hit of each ray.

        Args:
            ray: Ray or RayAD batch.
            active: Per-ray mask; inactive rays return a miss.
            flags: RayFlags selecting which hit fields to compute.

        Returns:
            Intersection, or IntersectionAD for a RayAD.

        Differentiable: with a RayAD, the hit fields carry derivatives (VJP and JVP) with respect to the ray and the scene geometry, with the hit triangle fixed by the forward pass. Raises RuntimeError if the scene is not built or has pending updates (call sync() first). An empty ray batch also raises RuntimeError.
        """

    @overload
    def intersect(
        self, ray: RayAD, active: drjit.cuda.ad.Bool = True, flags: RayFlags = RayFlags.All
    ) -> IntersectionAD: ...
    @overload
    def trace_reflections(
        self, ray: Ray, max_bounces: int, active: drjit.cuda.Bool | bool = True, symbolic: bool = True
    ) -> ReflectionTrace | ReflectionChain:
        """
        Trace specular reflection paths for up to max_bounces bounces per ray.

        Args:
            ray: Ray or RayAD batch.
            max_bounces: Maximum number of reflections; must be positive.
            options: Optional ReflectionTraceOptions. The deduplicate, canonical_prim_table, and image_source_tolerance keywords are a shorthand for it.
            active: Per-ray mask.
            symbolic: True returns a ReflectionTrace with one record per bounce; False returns a flat ReflectionChain.

        Returns:
            ReflectionTrace or ReflectionChain, or their AD variants for a RayAD.

        Differentiable: with a RayAD, the path geometry carries derivatives (VJP and JVP) with respect to the ray and the scene geometry, with the reflector sequence fixed by the forward pass. Deduplication requires symbolic=False. A non-positive max_bounces raises ValueError; raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def trace_reflections(
        self, ray: RayAD, max_bounces: int, active: drjit.cuda.ad.Bool | bool = True, symbolic: bool = True
    ) -> ReflectionTraceAD | ReflectionChainAD: ...
    @overload
    def trace_reflections(
        self,
        ray: Ray,
        max_bounces: int,
        options: ReflectionTraceOptions,
        active: drjit.cuda.Bool | bool = True,
        symbolic: bool = True,
    ) -> ReflectionTrace | ReflectionChain: ...
    @overload
    def trace_reflections(
        self,
        ray: RayAD,
        max_bounces: int,
        options: ReflectionTraceOptions,
        active: drjit.cuda.ad.Bool | bool = True,
        symbolic: bool = True,
    ) -> ReflectionTraceAD | ReflectionChainAD: ...
    @overload
    def trace_reflections(
        self,
        ray: Ray,
        max_bounces: int,
        deduplicate: bool = False,
        canonical_prim_table: drjit.cuda.Int = ...,
        image_source_tolerance: float = 1e-05,
        active: drjit.cuda.Bool | bool = True,
        symbolic: bool = True,
    ) -> ReflectionTrace | ReflectionChain: ...
    @overload
    def trace_reflections(
        self,
        ray: RayAD,
        max_bounces: int,
        deduplicate: bool = False,
        canonical_prim_table: drjit.cuda.Int = ...,
        image_source_tolerance: float = 1e-05,
        active: drjit.cuda.ad.Bool | bool = True,
        symbolic: bool = True,
    ) -> ReflectionTraceAD | ReflectionChainAD: ...
    def trace_refl_epc(
        self,
        ray: Ray,
        receiver: drjit.cuda.Array3f,
        max_bounces: int,
        options: ReflEpcOptions | None = None,
        active: drjit.cuda.Bool | bool = True,
    ) -> ReflEpc:
        """
        Trace reflection paths from each ray and correct them into paths that end exactly at the receiver.

        Args:
            ray: Detached Ray batch.
            receiver: Receiver positions, one or one per ray.
            max_bounces: Maximum number of reflections, from 1 to 8.
            options: Optional ReflEpcOptions.
            active: Per-ray mask.

        Returns:
            ReflEpc.

        Not differentiable: a grad-enabled receiver or scene geometry raises RuntimeError. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """

    @overload
    def trace_refl_epc_field(
        self,
        ray: RayAD,
        receiver: drjit.cuda.Array3f | drjit.cuda.ad.Array3f,
        max_bounces: int,
        options: ReflEpcFieldOptions | ReflEpcFieldOptionsAD,
        active: drjit.cuda.Bool | drjit.cuda.ad.Bool | bool = True,
    ) -> ReflEpcFieldAD:
        """
        Trace corrected reflection paths to the receiver and evaluate the complex field there.

        Args:
            ray or source: A Ray, or transmitter positions (one per path).
            receiver: Receiver positions, one or one per path.
            max_bounces: Maximum number of reflections, from 1 to 8.
            options: ReflEpcFieldOptions or ReflEpcFieldOptionsAD; per-slot arrays need n_paths * max_bounces entries.
            active: Per-path mask.

        Returns:
            ReflEpcField, or ReflEpcFieldAD when any input is an AD array.

        Differentiable through the source overload: with AD arrays, the result carries derivatives (VJP and JVP) with respect to the source, the receiver, the scene geometry, and the per-slot planes and materials. The ray overload is forward-only, and a RayAD raises RuntimeError. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def trace_refl_epc_field(
        self,
        ray: Ray,
        receiver: drjit.cuda.Array3f,
        max_bounces: int,
        options: ReflEpcFieldOptions,
        active: drjit.cuda.Bool | bool = True,
    ) -> ReflEpcField: ...
    @overload
    def trace_refl_epc_field(
        self,
        source: drjit.cuda.ad.Array3f,
        receiver: drjit.cuda.Array3f | drjit.cuda.ad.Array3f,
        max_bounces: int,
        options: ReflEpcFieldOptions | ReflEpcFieldOptionsAD,
        active: drjit.cuda.Bool | drjit.cuda.ad.Bool | bool = True,
    ) -> ReflEpcFieldAD: ...
    @overload
    def trace_refl_epc_field(
        self,
        source: drjit.cuda.Array3f,
        receiver: drjit.cuda.Array3f,
        max_bounces: int,
        options: ReflEpcFieldOptions,
        active: drjit.cuda.Bool | bool = True,
    ) -> ReflEpcField: ...
    @overload
    def accum_refl(
        self,
        ray: Ray,
        tx_position: drjit.cuda.Array3f,
        grid: AccumGrid,
        material: ReflMaterial,
        max_bounces: int,
        options: AccumOptions = ...,
        active: drjit.cuda.Bool | bool = True,
        tx_polarization: drjit.cuda.Array3f | tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> AccumResult:
        """
        Trace reflection paths and accumulate the reflected field and power on a grid.

        Args:
            ray: Ray or RayAD batch leaving the transmitter.
            tx_position: Transmitter position, one or one per ray.
            grid: AccumGrid receiving plane.
            material: ReflMaterial (ReflMaterialAD with a RayAD) with one entry per scene-global primitive.
            max_bounces: Maximum number of reflections; must be positive.
            options: AccumOptions.
            active: Per-ray mask.
            tx_polarization: Transmitter polarization, one or one per ray.

        Returns:
            AccumResult, or AccumResultAD for a RayAD.

        Differentiable with a RayAD: VJP and JVP with respect to the scene geometry, the transmitter, and the material, with the paths fixed by the forward pass. With a detached Ray, a grad-enabled tx_position or tx_polarization raises RuntimeError. Invalid grid or options values raise RuntimeError, as does a scene that is not built or has pending updates.
        """
    @overload
    def accum_refl(
        self,
        ray: RayAD,
        tx_position: drjit.cuda.ad.Array3f,
        grid: AccumGrid,
        material: ReflMaterialAD,
        max_bounces: int,
        options: AccumOptions = ...,
        active: drjit.cuda.ad.Bool | bool = True,
        tx_polarization: drjit.cuda.ad.Array3f | tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> AccumResultAD: ...
    @overload
    def accum_dfr_direct(
        self,
        states: DfrStates,
        grid: DfrGrid,
        material: DfrMaterial,
        options: DfrOptions = ...,
        active: drjit.cuda.Bool | bool = True,
    ) -> DfrAccum:
        """
        Accumulate first-order edge diffraction from sampled states on a grid by Monte-Carlo sampling.

        Args:
            states: DfrStates, or DfrStatesAD for the AD overload.
            grid: DfrGrid receiving plane.
            material: DfrMaterial, or DfrMaterialAD for the AD overload.
            options: DfrOptions; max_order must be 1.
            active: Per-state mask, one or one per state.

        Returns:
            DfrAccum, or DfrAccumAD for the AD overload.

        Differentiable with AD states: VJP and JVP with respect to the scene geometry and the material. Invalid grid or options values raise RuntimeError, as does a scene that is not built or has pending updates.
        """
    @overload
    def accum_dfr_direct(
        self,
        states: DfrStatesAD,
        grid: DfrGrid,
        material: DfrMaterialAD,
        options: DfrOptions = ...,
        active: drjit.cuda.ad.Bool | bool = True,
    ) -> DfrAccumAD: ...
    @overload
    def accum_dfr_coherent_direct(
        self,
        states: DfrCoherentUtdStates | DfrStates,
        grid: DfrGrid,
        material: DfrMaterial | None,
        options: DfrCoherentOptions = ...,
        active: drjit.cuda.Bool | bool = True,
    ) -> DfrCoherentAccum:
        """
        Accumulate the first-order diffracted field on a grid deterministically, one lane per (state, cell) pair.

        Args:
            states: DfrCoherentUtdStates (material is then ignored and may be None) or DfrStates.
            grid: DfrGrid receiving plane.
            material: DfrMaterial for DfrStates input.
            options: DfrCoherentOptions; max_order must be 1 and receiver_model MATCHED_ISO.
            active: Per-state mask, one or one per state.

        Returns:
            DfrCoherentAccum.

        Forward-only: the AD overloads raise RuntimeError. Invalid grid or options values raise RuntimeError, as does a scene that is not built or has pending updates.
        """
    @overload
    def accum_dfr_coherent_direct(
        self,
        states: DfrCoherentUtdStatesAD | DfrStatesAD,
        grid: DfrGrid,
        material: DfrMaterialAD | None,
        options: DfrCoherentOptions = ...,
        active: drjit.cuda.ad.Bool | bool = True,
    ) -> DfrCoherentAccumAD: ...
    @overload
    def build_dfr_coherent_tx_states(
        self,
        edges: DfrCoherentEdge,
        tx_position: drjit.cuda.Array3f,
        material: DfrMaterial,
        options: DfrCoherentOptions = ...,
        active: drjit.cuda.Bool | bool = True,
    ) -> DfrCoherentUtdStates:
        """
        Build first-order coherent UTD states that illuminate the given edges from a transmitter.

        Args:
            edges: DfrCoherentEdge selection.
            tx_position: Transmitter position, one or one per edge.
            material: DfrMaterial.
            options: DfrCoherentOptions; wavelength and k must be positive.
            active: Per-edge mask, one or one per edge.

        Returns:
            DfrCoherentUtdStates for Scene.accum_dfr_coherent_direct().

        Forward-only: the AD overload and a grad-enabled tx_position raise RuntimeError. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def build_dfr_coherent_tx_states(
        self,
        edges: DfrCoherentEdgeAD,
        tx_position: drjit.cuda.ad.Array3f,
        material: DfrMaterialAD,
        options: DfrCoherentOptions = ...,
        active: drjit.cuda.ad.Bool | bool = True,
    ) -> DfrCoherentUtdStatesAD: ...
    @overload
    def build_dfr_coherent_higher_candidates(
        self,
        prev_states: DfrCoherentUtdStates,
        edges: DfrCoherentEdge,
        global_to_local_edge_index: drjit.cuda.Int,
        options: DfrCoherentOptions = ...,
        active: drjit.cuda.Bool | bool = True,
    ) -> DfrCoherentCandidatePairs:
        """
        Find candidate next edges for higher-order coherent diffraction from a set of UTD states.

        Probe rays around each previous state's edge find nearby edges; the probe radius is the source-to-edge distance times options.higher_probe_radius_scale, clamped to [higher_probe_radius_min, higher_probe_radius_max].

        Args:
            prev_states: DfrCoherentUtdStates to expand.
            edges: DfrCoherentEdge candidates.
            global_to_local_edge_index: Map from scene-global edge id to an index into edges, or -1.
            options: DfrCoherentOptions.
            active: Per-state mask, one or one per previous state.

        Returns:
            DfrCoherentCandidatePairs.

        Forward-only: the AD overload raises RuntimeError. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def build_dfr_coherent_higher_candidates(
        self,
        prev_states: DfrCoherentUtdStatesAD,
        edges: DfrCoherentEdgeAD,
        global_to_local_edge_index: drjit.cuda.ad.Int,
        options: DfrCoherentOptions = ...,
        active: drjit.cuda.ad.Bool | bool = True,
    ) -> DfrCoherentCandidatePairsAD: ...
    @overload
    def accum_dfr(
        self,
        initial_states: DfrStates,
        recursive_states: DfrStates,
        grid: DfrGrid,
        material: DfrMaterial,
        options: DfrOptions = ...,
        active: drjit.cuda.Bool | bool = True,
    ) -> DfrAccum:
        """
        Accumulate second- or third-order edge-diffraction chains on a grid by Monte-Carlo sampling.

        Args:
            initial_states: DfrStates of the first diffraction event.
            recursive_states: DfrStates drawn for each further event.
            grid: DfrGrid receiving plane.
            material: DfrMaterial.
            options: DfrOptions; max_order must be 2 or 3.
            active: Per-state mask, one or one per initial state.

        Returns:
            DfrAccum, or DfrAccumAD for AD states.

        Differentiable with AD states: VJP and JVP with respect to the scene geometry and the material. Invalid grid or options values raise RuntimeError, as does a scene that is not built or has pending updates.
        """
    @overload
    def accum_dfr(
        self,
        initial_states: DfrStatesAD,
        recursive_states: DfrStatesAD,
        grid: DfrGrid,
        material: DfrMaterialAD,
        options: DfrOptions = ...,
        active: drjit.cuda.ad.Bool | bool = True,
    ) -> DfrAccumAD: ...
    @overload
    def trace_dfr_paths(
        self,
        tx_positions: drjit.cuda.Array3f,
        rx_positions: drjit.cuda.Array3f,
        states: DfrStates,
        material: DfrMaterial,
        options: DfrPathOptions = ...,
        active: drjit.cuda.Bool = True,
    ) -> DfrPaths:
        """
        Export first-order diffraction paths for every (transmitter, receiver, state) triple.

        Args:
            tx_positions: Transmitter positions.
            rx_positions: Receiver positions.
            states: DfrStates, or DfrStatesAD for the AD overload.
            material: DfrMaterial, or DfrMaterialAD for the AD overload.
            options: DfrPathOptions; max_paths None or 0 exports every state.
            active: Per-state mask, one or one per state.

        Returns:
            DfrPaths with capacity n_tx * n_rx * min(n_states, max_paths), or n_tx * n_rx * n_states when max_paths is None; DfrPathsAD for the AD overload.

        Differentiable with AD inputs: VJP and JVP with respect to the continuous inputs. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """

    @overload
    def trace_dfr_paths(
        self,
        tx_positions: drjit.cuda.ad.Array3f,
        rx_positions: drjit.cuda.ad.Array3f,
        states: DfrStatesAD,
        material: DfrMaterialAD,
        options: DfrPathOptions = ...,
        active: drjit.cuda.ad.Bool = True,
    ) -> DfrPathsAD: ...
    @overload
    def shadow_test(self, ray: Ray, active: drjit.cuda.Bool = True) -> drjit.cuda.Bool:
        """
        Return a mask that is True where the ray hits any triangle within tmax.

        The result is discrete and has no derivative. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """

    @overload
    def shadow_test(self, ray: RayAD, active: drjit.cuda.ad.Bool = True) -> drjit.cuda.ad.Bool: ...
    @overload
    def visible(
        self,
        start: drjit.cuda.Array3f,
        end: drjit.cuda.Array3f,
        active: drjit.cuda.Bool | bool = True,
        *,
        ignore_prim_ids: drjit.cuda.Int = ...,
    ) -> SegmentVisibility:
        """
        Test whether the segment from start to end is unoccluded.

        Args:
            start: Segment start points.
            end: Segment end points, one per start point.
            active: Per-segment mask.
            ignore_prim_ids: Optional scene-global primitive ids to treat as transparent, K per segment laid out segment-major.

        Returns:
            SegmentVisibility, or SegmentVisibilityAD for AD points.

        The result is discrete and has no derivative. Segments with a non-finite endpoint are not visible. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def visible(
        self,
        start: drjit.cuda.ad.Array3f,
        end: drjit.cuda.ad.Array3f,
        active: drjit.cuda.ad.Bool | bool = True,
        *,
        ignore_prim_ids: drjit.cuda.Int = ...,
    ) -> SegmentVisibilityAD: ...
    @overload
    def visible_pair(
        self,
        start: drjit.cuda.Array3f,
        end_a: drjit.cuda.Array3f,
        end_b: drjit.cuda.Array3f,
        ignore_prim_ids: drjit.cuda.Int = ...,
        active: drjit.cuda.Bool | bool = True,
    ) -> SegmentPairVisibility:
        """
        Test the visibility of two endpoints from each start point in one pass.

        Returns SegmentPairVisibility with visible_a for end_a and visible_b for end_b. ignore_prim_ids works as in visible(). The result is discrete and has no derivative. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def visible_pair(
        self,
        start: drjit.cuda.ad.Array3f,
        end_a: drjit.cuda.ad.Array3f,
        end_b: drjit.cuda.ad.Array3f,
        ignore_prim_ids: drjit.cuda.Int = ...,
        active: drjit.cuda.ad.Bool | bool = True,
    ) -> SegmentPairVisibilityAD: ...
    @overload
    def visible_edge(
        self,
        source: drjit.cuda.Array3f,
        edge_position: drjit.cuda.Array3f,
        edge_direction: drjit.cuda.Array3f,
        edge_t_min: drjit.cuda.Float,
        edge_t_max: drjit.cuda.Float,
        sample_fractions: Sequence[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        active: drjit.cuda.Bool | bool = True,
    ) -> AxialEdgeVisibility:
        """
        Test whether each source sees any sampled point of an edge segment.

        Args:
            source: Source points.
            edge_position: Edge origins.
            edge_direction: Edge directions.
            edge_t_min, edge_t_max: Edge extent along edge_direction from edge_position.
            sample_fractions: 1 to 16 fractions of [edge_t_min, edge_t_max] to sample.
            active: Per-edge mask.

        Returns:
            AxialEdgeVisibility, or its AD variant for AD points.

        The result is discrete and has no derivative. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def visible_edge(
        self,
        source: drjit.cuda.ad.Array3f,
        edge_position: drjit.cuda.ad.Array3f,
        edge_direction: drjit.cuda.ad.Array3f,
        edge_t_min: drjit.cuda.ad.Float,
        edge_t_max: drjit.cuda.ad.Float,
        sample_fractions: Sequence[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        active: drjit.cuda.ad.Bool | bool = True,
    ) -> AxialEdgeVisibilityAD: ...
    @overload
    def visible_chain(
        self,
        points: drjit.cuda.Array3f,
        chain_length: drjit.cuda.Int,
        ignore_prim_per_segment: drjit.cuda.Int = ...,
        active: drjit.cuda.Bool | bool = True,
    ) -> SegmentChainVisibility:
        """
        Test every segment of each polyline chain for occlusion.

        Args:
            points: Chain vertices, chain-major, with the same number of points (at least 2) per chain.
            chain_length: Number of used points in each chain; its width is the chain count.
            ignore_prim_per_segment: Optional scene-global primitive ids to ignore, K per segment.
            active: Per-chain mask.

        Returns:
            SegmentChainVisibility with the first blocked segment and primitive of each chain.

        The result is discrete and has no derivative. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def visible_chain(
        self,
        points: drjit.cuda.ad.Array3f,
        chain_length: drjit.cuda.Int,
        ignore_prim_per_segment: drjit.cuda.Int = ...,
        active: drjit.cuda.ad.Bool | bool = True,
    ) -> SegmentChainVisibilityAD: ...
    @overload
    def nearest_edge(self, query: drjit.cuda.Array3f, active: drjit.cuda.Bool | bool = True) -> NearestPointEdge:
        """
        Find the nearest scene edge to each query point or ray.

        query is an Array3f point batch, which returns NearestPointEdge, or a Ray, which returns NearestRayEdge. A ray searches the segment [0, tmax] when tmax is finite and the half-line otherwise; rows with a non-finite origin, a zero or non-finite direction, or a non-positive tmax return the miss values. Edges disabled by set_edge_mask() are skipped.

        Differentiable: with AD inputs, distance and closest points carry derivatives (VJP and JVP) with respect to the query and the scene geometry, with the winning edge fixed by the forward pass. Raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def nearest_edge(
        self, query: drjit.cuda.ad.Array3f, active: drjit.cuda.ad.Bool | bool = True
    ) -> NearestPointEdgeAD: ...
    @overload
    def nearest_edge(self, query: Ray, active: drjit.cuda.Bool | bool = True) -> NearestRayEdge: ...
    @overload
    def nearest_edge(self, query: RayAD, active: drjit.cuda.ad.Bool | bool = True) -> NearestRayEdgeAD: ...
    @overload
    def nearest_edges(
        self, point: drjit.cuda.Array3f, k: int, active: drjit.cuda.Bool | bool = True
    ) -> NearestEdgesTopK:
        """
        Find the k nearest scene edges to each query point, nearest first, for 1 <= k <= 16.

        Returns NearestEdgesTopK. Ties are broken toward the lower global edge id.

        Differentiable: with an AD point batch, distances and closest points carry derivatives (VJP and JVP) with respect to the points and the scene geometry, with the selected edges fixed by the forward pass. A k outside [1, 16] raises RuntimeError; raises RuntimeError if the scene is not built or has pending updates (call sync() first).
        """
    @overload
    def nearest_edges(
        self, point: drjit.cuda.ad.Array3f, k: int, active: drjit.cuda.ad.Bool | bool = True
    ) -> NearestEdgesTopKAD: ...
    @property
    def num_meshes(self) -> int:
        """Number of meshes and instances."""

    @property
    def version(self) -> int:
        """Counter that increases whenever the scene geometry changes."""

    @property
    def edge_version(self) -> int:
        """Counter that increases whenever the edge set changes."""

    def __repr__(self) -> str: ...

class MixedScene:
    """
    Scene that combines triangle meshes, SDF grids, and surfel clouds.

    Supports intersect(), visible(), trace_reflections(), and transmittance(); diffraction stays on Scene. Ids number meshes first, then SDF grids, then surfel clouds.
    """

    def __init__(self, *, trace_backend: str = "auto", edge_bvh_backend: str = "auto") -> None:
        """
        Create an empty mixed scene. trace_backend and edge_bvh_backend configure its triangle Scene and take the same values as in Scene(); an unknown value raises ValueError.
        """

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        """Add a copy of mesh and return its mesh id."""

    def add_sdf(self, grid: SdfGrid, options: SdfTraceOptions = ...) -> int:
        """
        Add an SdfGrid traced with options and return its index among the SDF grids.
        """

    def add_surfel(self, surfel: SurfelCloud, options: SurfelTraceOptions = ...) -> int:
        """
        Add a surfel cloud traced with options and return its index among the surfel clouds.
        """

    def build(self) -> None:
        """
        Build every mesh, SDF, and surfel acceleration structure. Call it before any query.
        """

    def is_ready(self) -> bool:
        """Return True once build() has run and nothing was added since."""

    @property
    def num_meshes(self) -> int:
        """Number of meshes."""

    @property
    def num_sdfs(self) -> int:
        """Number of SDF grids."""

    @property
    def num_surfel_scenes(self) -> int:
        """Number of surfel clouds."""

    @overload
    def intersect(self, ray: Ray, active: drjit.cuda.Bool = True, flags: RayFlags = RayFlags.All) -> Intersection:
        """
        Find the closest mesh, SDF, or surfel hit of each ray.

        Returns an Intersection (IntersectionAD for a RayAD). shape_id counts meshes, then SDF grids, then surfel clouds; global_prim_id counts mesh faces, then one id per SDF grid, then surfels.

        Differentiable: with a RayAD, the selected hit's continuous fields carry derivatives (VJP and JVP) with respect to the ray and the selected geometry, with the winner fixed by the forward pass. Raises RuntimeError if the scene is not built.
        """

    @overload
    def intersect(
        self, ray: RayAD, active: drjit.cuda.ad.Bool = True, flags: RayFlags = RayFlags.All
    ) -> IntersectionAD: ...
    @overload
    def visible(
        self, start: drjit.cuda.Array3f, end: drjit.cuda.Array3f, active: drjit.cuda.Bool = True
    ) -> SegmentVisibility:
        """
        Test whether the segment from start to end is unoccluded by any mesh, SDF, or surfel.

        Returns SegmentVisibility. The result is discrete and has no derivative. Raises RuntimeError if the scene is not built.
        """

    @overload
    def visible(
        self, start: drjit.cuda.ad.Array3f, end: drjit.cuda.ad.Array3f, active: drjit.cuda.ad.Bool = True
    ) -> SegmentVisibilityAD: ...
    @overload
    def trace_reflections(self, ray: Ray, max_bounces: int, active: drjit.cuda.Bool = True) -> ReflectionChain:
        """
        Trace specular reflections off meshes, SDFs, and surfels for up to max_bounces bounces.

        Returns a ReflectionChain. max_bounces must be non-negative. Raises RuntimeError if the scene is not built.
        """

    @overload
    def trace_reflections(
        self, ray: RayAD, max_bounces: int, active: drjit.cuda.ad.Bool = True
    ) -> ReflectionChainAD: ...
    @overload
    def transmittance(self, ray: Ray, active: drjit.cuda.Bool = True) -> drjit.cuda.Float:
        """
        Return the transmittance along each ray.

        A mesh hit makes the ray opaque (0), surfel alpha attenuates it multiplicatively, and SDF grids are ignored. Inactive rays return 1.

        Differentiable: with a RayAD, the result carries derivatives (VJP and JVP) with respect to the ray and the surfel fields. Raises RuntimeError if the scene is not built.
        """

    @overload
    def transmittance(self, ray: RayAD, active: drjit.cuda.ad.Bool = True) -> drjit.cuda.ad.Float: ...
