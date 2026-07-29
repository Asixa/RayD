// Copyright Xingyu Chen.
// Declares the Dr.Jit reflection tracing, accumulation, and correction API.

#pragma once

#include <string>
#include <type_traits>
#include <vector>
#include <drjit/complex.h>
#include <rayd/jit/core.h>

namespace rayd {

/// Axis-aligned 2D accumulation grid: a plane at `position` along `axis`, spanning
/// [coord0_min, coord0_max] x [coord1_min, coord1_max] at resolution0 x resolution1 cells.
struct AccumGrid {
    int axis = 2;         ///< Plane normal axis (0 = x, 1 = y, 2 = z).
    float position = 0.f; ///< Plane offset along `axis`.
    float coord0_min = 0.f;
    float coord0_max = 0.f;
    float coord1_min = 0.f;
    float coord1_max = 0.f;
    int resolution0 = 0; ///< Cell count along the first in-plane axis.
    int resolution1 = 0; ///< Cell count along the second in-plane axis.
};

/// Parameters for native reflection accumulation (field synthesis and Russian-roulette termination).
struct AccumOptions {
    float wavelength = 1.f;              ///< Wavelength in world units.
    float k = 0.f;                       ///< Wavenumber (2*pi / wavelength).
    float solid_angle_per_ray = 1.f;     ///< Solid angle each ray represents.
    float cell_area = 1.f;               ///< Area of one grid cell.
    int seed = 0;                        ///< RNG seed.
    int rr_depth = 0;                    ///< Bounce depth at which Russian roulette starts.
    float rr_prob = 1.f;                 ///< Survival probability per roulette test.
    float stop_threshold = 0.f;          ///< Stop a path once its contribution falls below this.
    bool collect_wedges = false;         ///< Record diffraction-wedge events.
    bool collect_wedge_prefixes = false; ///< Also record the reflection prefix of each wedge event.
    int wedge_capacity = 0;              ///< Maximum wedge events to store.
    int wedge_sample_stride = 1;         ///< Prefix wedge event sampling stride.
};

/// Per-primitive material used by reflection accumulation (electromagnetic surface parameters).
template <typename Float_> struct MaterialData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    Float_ eta_r = full<Float_>(1.f, 1); ///< Relative permittivity.
    Float_ sigma = full<Float_>(0.f, 1); ///< Conductivity.
    Float_ gain = full<Float_>(1.f, 1);  ///< Extra gain factor.
    Float_ mu_r = full<Float_>(1.f, 1);  ///< Relative permeability.
    Mask_ valid = full<Mask_>(false, 1); ///< Whether this entry holds a real material.

    DRJIT_STRUCT(MaterialData, eta_r, sigma, gain, mu_r, valid)
};

/// Compacted list of diffraction-wedge events recorded during accumulation; `count`
/// entries (up to `capacity`) of the per-event arrays are valid.
template <typename Float_> struct WedgeEventsData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int capacity = 0;                           ///< Allocated event slots.
    Int_ count = full<Int_>(0, 1);              ///< Number of recorded events.
    Int_ ray_index = full<Int_>(-1, 1);         ///< Source ray of each event.
    Vec3f hit_points = zeros<Vec3f>(1);         ///< Wedge location.
    Vec3f normals = zeros<Vec3f>(1);            ///< Surface normal at the wedge.
    Int_ prim_id = full<Int_>(-1, 1);           ///< Primitive forming the wedge.
    Vec3f directions = zeros<Vec3f>(1);         ///< Incident direction at the wedge.
    Vec3f source_points = zeros<Vec3f>(1);      ///< Virtual source used before the wedge event.
    Float_ src_power = zeros<Float_>(1);        ///< Source field power carried into the wedge.
    Vec3f initial_directions = zeros<Vec3f>(1); ///< Primary ray direction that produced the event.
    Int_ bounce_depth = full<Int_>(-1, 1);      ///< Reflection depth at which the event occurred.

    DRJIT_STRUCT(WedgeEventsData, count, ray_index, hit_points, normals, prim_id, directions, source_points, src_power,
                 initial_directions, bounce_depth)
};

/// Result of reflection accumulation: power and complex field accumulated per grid cell,
/// plus optional diffraction-wedge events. Grid arrays have grid_cell_count entries.
template <typename Float_> struct AccumResultData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using FloatArray = Float_;
    using ComplexArray = drjit::Complex<Float_>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using WedgeBuffer = WedgeEventsData<Float_>;

    int ray_count = 0;
    int max_bounces = 0;
    int grid_cell_count = 0;
    FloatArray reflection_power = zeros<FloatArray>(1); ///< Accumulated power per grid cell.
    ComplexArray reflection_field_x =                   ///< Accumulated complex x field per cell.
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    ComplexArray reflection_field_y = ///< Accumulated complex y field per cell.
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    ComplexArray reflection_field_z = ///< Accumulated complex z field per cell.
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    Int_ reflection_count = full<Int_>(0, 1); ///< Total reflection contributions accumulated.
    WedgeBuffer wedge_events;                 ///< Recorded wedge events (if collection was enabled).

    DRJIT_STRUCT(AccumResultData, reflection_power, reflection_field_x, reflection_field_y, reflection_field_z,
                 reflection_count, wedge_events)
};

} // namespace rayd

namespace rayd {

/// Options for equivalent-path-correction (EPC) reflection traces. The expected
/// reflector sequence and optional surface-group tables steer which primitives a
/// path is allowed to use and which to ignore during visibility checks.
struct ReflEpcOptions {
    Int expected_prim_ids;          ///< Expected reflector per slot (n_rays * max_bounces).
    Int surface_group_id;           ///< Surface-group id per triangle (optional).
    Int surface_group_size;         ///< Member count of each surface group.
    Int surface_group_members;      ///< Flattened group membership table (surface_group_count * max_group_size).
    int surface_max_group_size = 0; ///< Width of the group-members table.
    std::string visibility_ignore_mode = "primitive"; ///< "primitive" or "surface_group" ignore semantics.
    float plane_tolerance = 1e-5f;                    ///< Relative out-of-plane containment tolerance.
    Int final_ignore_group_ids;                       ///< Groups ignored on the final receiver segment.
};

/// EPC options extended with per-slot material/geometry and the field-evaluation
/// parameters (frequency, polarization) plus output selectors.
template <bool Detached> struct ReflEpcFieldOptionsT : ReflEpcOptions {
    using Float_ = FloatT<Detached>;
    using Vec3f = Vector3fT<Detached>;

    Vec3f slot_plane_point = zeros<Vec3f>(1);  ///< Override reflecting-plane point per slot.
    Vec3f slot_plane_normal = zeros<Vec3f>(1); ///< Override reflecting-plane normal per slot.
    Float_ slot_eta_r = full<Float_>(1.f, 1);  ///< Relative permittivity per slot.
    Float_ slot_mu_r = full<Float_>(1.f, 1);   ///< Relative permeability per slot.
    Float_ slot_sigma = full<Float_>(0.f, 1);  ///< Conductivity per slot.
    Float_ slot_gain = full<Float_>(1.f, 1);   ///< Extra gain factor per slot.
    Vec3f tx_polarization =                    ///< Transmitter polarization vector.
        Vec3f(full<Float_>(1.f, 1), full<Float_>(0.f, 1), full<Float_>(0.f, 1));
    float omega = 2.f * 3.14159265358979323846f * 299792458.f; ///< Angular frequency (rad/s).
    float wavelength = 1.f;                                    ///< Wavelength in world units.
    bool return_geom = false;             ///< Master switch for the per-slot geometry outputs below.
    bool return_endpoints = false;        ///< Emit tx/first-hit/last-hit positions.
    bool return_hit_points = true;        ///< Emit per-slot hit points (requires return_geom).
    bool return_normals = true;           ///< Emit per-slot normals (requires return_geom).
    bool return_resolved_prim_ids = true; ///< Emit resolved primitive ids (requires return_geom).
    bool return_surface_group_ids = true; ///< Emit surface-group ids (requires return_geom).
};

/// Result of an EPC reflection trace. Per-slot arrays are ray_count * max_bounces;
/// per-ray arrays carry the path validity, length, and first occlusion encountered.
template <typename Float_> struct ReflEpcData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;

    int ray_count = 0;
    int max_bounces = 0;

    Mask_ valid = full<Mask_>(false, 1);            ///< Per-ray: a corrected path to the receiver exists.
    Int_ bounce_count = full<Int_>(0, 1);           ///< Per-ray reflection count.
    Float_ path_length = full<Float_>(Infinity, 1); ///< Per-ray total corrected path length.
    Vec3f reflection_points = zeros<Vec3f>(1);      ///< Per-slot corrected reflection point.
    Int_ prim_ids = full<Int_>(-1, 1);              ///< Per-slot reflector primitive id.
    Int_ trace_prim_ids = full<Int_>(-1, 1);        ///< Per-slot primitive actually hit during tracing.
    Int_ resolved_prim_ids = full<Int_>(-1, 1);     ///< Per-slot primitive after surface-group resolution.
    Int_ surface_group_ids = full<Int_>(-1, 1);     ///< Per-slot surface-group id.
    Vec3f plane_normals = zeros<Vec3f>(1);          ///< Per-slot reflecting-plane normal.
    Int_ first_blocked_segment = full<Int_>(-1, 1); ///< Per-ray index of the first occluded segment; -1 if none.
    Int_ first_blocked_prim = full<Int_>(-1, 1);    ///< Per-ray primitive that blocked it.
    Int_ first_blocked_group = full<Int_>(-1, 1);   ///< Per-ray surface group that blocked it.

    DRJIT_STRUCT(ReflEpcData, valid, bounce_count, path_length, reflection_points, prim_ids, trace_prim_ids,
                 resolved_prim_ids, surface_group_ids, plane_normals, first_blocked_segment, first_blocked_prim,
                 first_blocked_group)
};

template <bool Detached> using ReflEpcT = ReflEpcData<FloatT<Detached>>;

using ReflEpcAD = ReflEpcT<false>;
using ReflEpc = ReflEpcT<true>;

/// Result of an EPC reflection trace that also evaluates the complex field per ray.
/// The field is the per-component (x, y, z) complex phasor; geometry/endpoint arrays
/// are present only when the corresponding ReflEpcFieldOptions flags are set.
template <typename Float_> struct ReflEpcFieldData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;

    int ray_count = 0;
    int max_bounces = 0;

    Mask_ valid = full<Mask_>(false, 1);            ///< Per-ray: a valid path contributed field.
    Int_ bounce_count = full<Int_>(0, 1);           ///< Per-ray reflection count.
    Float_ path_length = full<Float_>(Infinity, 1); ///< Per-ray total path length.

    Float_ field_x_re = zeros<Float_>(1); ///< Real part of the x field component.
    Float_ field_x_im = zeros<Float_>(1); ///< Imaginary part of the x field component.
    Float_ field_y_re = zeros<Float_>(1); ///< Real part of the y field component.
    Float_ field_y_im = zeros<Float_>(1); ///< Imaginary part of the y field component.
    Float_ field_z_re = zeros<Float_>(1); ///< Real part of the z field component.
    Float_ field_z_im = zeros<Float_>(1); ///< Imaginary part of the z field component.

    Vec3f tx_pos = zeros<Vec3f>(1);    ///< Transmitter position (if return_endpoints).
    Vec3f first_hit = zeros<Vec3f>(1); ///< First reflection point (if return_endpoints).
    Vec3f last_hit = zeros<Vec3f>(1);  ///< Last reflection point (if return_endpoints).

    Vec3f hit_points = zeros<Vec3f>(1);         ///< Per-slot hit points (if return_hit_points).
    Vec3f normals = zeros<Vec3f>(1);            ///< Per-slot normals (if return_normals).
    Int_ resolved_prim_ids = full<Int_>(-1, 1); ///< Per-slot resolved primitive ids (if requested).
    Int_ surface_group_ids = full<Int_>(-1, 1); ///< Per-slot surface-group ids (if requested).

    DRJIT_STRUCT(ReflEpcFieldData, valid, bounce_count, path_length, field_x_re, field_x_im, field_y_re, field_y_im,
                 field_z_re, field_z_im, tx_pos, first_hit, last_hit, hit_points, normals, resolved_prim_ids,
                 surface_group_ids)
};

template <bool Detached> using ReflEpcFieldT = ReflEpcFieldData<FloatT<Detached>>;

using ReflEpcFieldAD = ReflEpcFieldT<false>;
using ReflEpcField = ReflEpcFieldT<true>;

} // namespace rayd

namespace rayd {

enum ReflectionExportMode {
    RAYD_REFLECTION_EXPORT_FULL = 0,
    RAYD_REFLECTION_EXPORT_MINIMAL = 1,
    RAYD_REFLECTION_EXPORT_COUNT_ONLY = 2,
};

/// Options controlling specular reflection traces.
struct ReflectionTraceOptions {
    bool deduplicate = false;                      ///< Merge paths that share the same sequence of reflectors.
    Int canonical_prim_table;                      ///< Optional map collapsing primitives to a canonical id for dedup.
    float image_source_tolerance = 1e-5f;          ///< Distance tolerance when comparing image sources for dedup.
    int export_mode = RAYD_REFLECTION_EXPORT_FULL; ///< Output payload size; defaults to full.
    bool return_trailing = true;                   ///< Trace and export the segment after the last reflection.
};

/// One reflection bounce for a batch of rays (the per-bounce slice of a chain).
template <typename Float_> struct ReflectionBounceData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    /// Per-lane mask of bounces that hit a reflector (prim_ids >= 0).
    Mask_ is_valid() const { return prim_ids >= 0; }

    Float_ t = full<Float_>(Infinity, 1);     ///< Distance from the previous point to this reflection.
    Vec3f hit_points = zeros<Vec3f>(1);       ///< Reflection point in world space.
    Vec3f geo_normals = zeros<Vec3f>(1);      ///< Geometric normal at the reflection point.
    Vec3f image_sources = zeros<Vec3f>(1);    ///< Mirror image of the source across this reflector.
    Vec3f plane_points = zeros<Vec3f>(1);     ///< A point on the reflecting plane.
    Vec3f plane_normals = zeros<Vec3f>(1);    ///< Unit normal of the reflecting plane.
    Int_ shape_ids = full<Int_>(-1, 1);       ///< Owning mesh id; -1 when no reflection.
    Int_ prim_ids = full<Int_>(-1, 1);        ///< Per-mesh face id of the reflector.
    Int_ local_prim_ids = full<Int_>(-1, 1);  ///< Same as prim_ids (per-mesh face id).
    Int_ global_prim_ids = full<Int_>(-1, 1); ///< Scene-global face id of the reflector.

    DRJIT_STRUCT(ReflectionBounceData, t, hit_points, geo_normals, image_sources, plane_points, plane_normals,
                 shape_ids, prim_ids, local_prim_ids, global_prim_ids)
};

/// Full reflection path per ray. The per-bounce arrays are laid out ray-major as
/// ray_count * max_bounces; bounce_count gives how many slots are valid per ray.
template <typename Float_> struct ReflectionChainData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    /// Per-slot mask of valid reflections (prim_ids >= 0).
    Mask_ is_valid() const { return prim_ids >= 0; }

    int max_bounces = 0; ///< Per-ray slot capacity.
    int ray_count = 0;   ///< Number of input rays.

    Int_ bounce_count = full<Int_>(0, 1);              ///< Valid reflections per ray.
    Int_ discovery_count = full<Int_>(0, 1);           ///< Paths collapsed into this one by dedup.
    Int_ representative_ray_index = full<Int_>(-1, 1); ///< RayAD chosen to represent a deduplicated group.
    Float_ t = full<Float_>(Infinity, 1);              ///< Per-slot reflection distance.
    Vec3f hit_points = zeros<Vec3f>(1);                ///< Per-slot reflection point.
    Vec3f geo_normals = zeros<Vec3f>(1);               ///< Per-slot geometric normal.
    Vec3f image_sources = zeros<Vec3f>(1);             ///< Per-slot image source.
    Vec3f plane_points = zeros<Vec3f>(1);              ///< Per-slot point on the reflecting plane.
    Vec3f plane_normals = zeros<Vec3f>(1);             ///< Per-slot reflecting-plane normal.
    Int_ shape_ids = full<Int_>(-1, 1);                ///< Per-slot owning mesh id.
    Int_ prim_ids = full<Int_>(-1, 1);                 ///< Per-slot per-mesh reflector face id.
    Int_ local_prim_ids = full<Int_>(-1, 1);           ///< Per-slot per-mesh face id (alias of prim_ids).
    Int_ global_prim_ids = full<Int_>(-1, 1);          ///< Per-slot scene-global reflector face id.
    Float_ trailing_t = full<Float_>(Infinity, 1);     ///< Distance of the final segment past the last reflection.
    Int_ trailing_prim = full<Int_>(-1, 1);            ///< Primitive hit by the trailing segment, if any.
    Vec3f trailing_dir = zeros<Vec3f>(1);              ///< Direction of the trailing segment.
    Vec3f trailing_origin = zeros<Vec3f>(1);           ///< Origin of the trailing segment.

    DRJIT_STRUCT(ReflectionChainData, bounce_count, discovery_count, representative_ray_index, t, hit_points,
                 geo_normals, image_sources, plane_points, plane_normals, shape_ids, prim_ids, local_prim_ids,
                 global_prim_ids, trailing_t, trailing_prim, trailing_dir, trailing_origin)
};

/// Reflection trace organized as a list of per-bounce slices (one ReflectionBounceData
/// per bounce level) rather than flattened arrays. Returned by Scene::trace_bounces().
template <typename Float_> struct ReflectionTraceData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using Bounce = ReflectionBounceData<Float_>;

    /// Per-ray mask of rays with at least one reflection.
    Mask_ is_valid() const { return bounce_count > 0; }

    int max_bounces = 0;
    int ray_count = 0;
    bool deduplicate_requested = false; ///< Whether deduplication was requested.
    bool deduplicate_applied = false;   ///< Whether deduplication actually ran.

    Int_ bounce_count = full<Int_>(0, 1);              ///< Valid reflections per ray.
    Int_ discovery_count = full<Int_>(0, 1);           ///< Paths collapsed into this one by dedup.
    Int_ representative_ray_index = full<Int_>(-1, 1); ///< Representative ray for a deduplicated group.
    Mask_ dedup_keep_mask = full<Mask_>(false, 1);     ///< Which rays survived deduplication.
    std::vector<Bounce> bounces;                       ///< One entry per bounce level (size up to max_bounces).
};

} // namespace rayd
