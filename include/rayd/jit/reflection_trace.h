// Copyright Xingyu Chen.
// Declares the Dr.Jit reflection trace API.

#pragma once

#include <vector>

#include <rayd/jit/core.h>

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
