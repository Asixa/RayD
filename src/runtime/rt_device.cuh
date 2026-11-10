// Copyright Xingyu Chen.
// Defines internal ray-traversal contracts and OptiX device helpers.

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include <rayd/math.h>

namespace rayd::shared::rt {

// Backend-neutral, decoded closest-hit record produced by a Traverser
// (docs/RAY_TRACING_BACKEND_ARCHITECTURE.md §7). It is the host-side mirror of the
// OptiX six-register TriangleHitPayload (shared/reflection/optix_hit.h) after the
// float-as-uint payload codec has been undone: `t`, `bary_u`, `bary_v` are plain
// floats, `prim` is the mesh-local primitive id, `instance` is the shape id, and
// `hit` is 0 on a miss. Keeping it a plain POD lets both the OptiX shim and the
// pure-CUDA BVH traverser hand identical hits to the shared algorithm bodies.
struct TriangleHit {
    float t;
    float bary_u;
    float bary_v;
    std::int32_t prim;
    std::int32_t instance;
    std::uint32_t hit;
};

static_assert(std::is_standard_layout_v<TriangleHit>);
static_assert(std::is_trivially_copyable_v<TriangleHit>);

// Traverser concept (C++17 traits + static_assert). A Traverser is a
// backend-neutral, per-lane closest-hit/occlusion oracle over one acceleration
// structure. Every migrated pipeline algorithm is templated over a Traverser and
// calls only these four const methods; the concrete traversers
// (shared/reflection/optix_traverser.h, shared/bvh/cuda_bvh_traverser.h) each assert
// themselves against is_traverser_v so a signature drift is a compile error, not
// a silent divergence.
//
//   TriangleHit trace_closest(origin, direction, tmin, tmax) const;
//   bool        trace_occluded(origin, direction, tmin, tmax) const;
//   bool        trace_occluded_ignore(origin, direction, tmin, tmax,
//                                     const int32_t* ignore, int ignore_count) const;
//   TriangleHit trace_first_blocker(origin, direction, tmin, tmax,
//                                   const int32_t* ignore, int ignore_count) const;
//
// origin/direction are math::Vec3f; the ignore list names mesh-global primitive
// ids to treat as non-occluding (parity with the OptiX anyhit ignore filter).
template <typename T, typename = void> struct is_traverser : std::false_type {};

template <typename T>
struct is_traverser<
    T, std::void_t<
           decltype(std::declval<const T&>().trace_closest(std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
                                                           std::declval<float>(), std::declval<float>())),
           decltype(std::declval<const T&>().trace_occluded(std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
                                                            std::declval<float>(), std::declval<float>())),
           decltype(std::declval<const T&>().trace_occluded_ignore(
               std::declval<math::Vec3f>(), std::declval<math::Vec3f>(), std::declval<float>(), std::declval<float>(),
               std::declval<const std::int32_t*>(), std::declval<int>())),
           decltype(std::declval<const T&>().trace_first_blocker(
               std::declval<math::Vec3f>(), std::declval<math::Vec3f>(), std::declval<float>(), std::declval<float>(),
               std::declval<const std::int32_t*>(), std::declval<int>()))>>
    : std::bool_constant<
          std::is_same_v<
              decltype(std::declval<const T&>().trace_closest(std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
                                                              std::declval<float>(), std::declval<float>())),
              TriangleHit> &&
          std::is_same_v<
              decltype(std::declval<const T&>().trace_occluded(std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
                                                               std::declval<float>(), std::declval<float>())),
              bool> &&
          std::is_same_v<decltype(std::declval<const T&>().trace_occluded_ignore(
                             std::declval<math::Vec3f>(), std::declval<math::Vec3f>(), std::declval<float>(),
                             std::declval<float>(), std::declval<const std::int32_t*>(), std::declval<int>())),
                         bool> &&
          std::is_same_v<decltype(std::declval<const T&>().trace_first_blocker(
                             std::declval<math::Vec3f>(), std::declval<math::Vec3f>(), std::declval<float>(),
                             std::declval<float>(), std::declval<const std::int32_t*>(), std::declval<int>())),
                         TriangleHit>> {};

template <typename T> inline constexpr bool is_traverser_v = is_traverser<T>::value;

// TraceConfig merges the two independent axes of a migrated pipeline (audit A3):
//   * Layout    - the backend storage/layout policy (AoS vs SoA inputs, packed
//                 triangles, output-layout honoring, empty-slot clearing, ...).
//                 For reflection trace this is the existing ReflectionTracePolicy.
//   * Traverser - the acceleration-structure oracle (OptiX shim or CUDA BVH).
//
// An algorithm body is templated over a single TraceConfig and reads
// Config::Layout::<flag> for compile-time layout branches and constructs /
// consumes Config::Traverser instances for all ray casts.
//
// Instantiation matrix (the only combinations any backend builds):
//
//     Layout \\ Traverser | OptixTraverser | CudaBvhTraverser
//     --------------------+----------------+-----------------
//     DrJit  (drjit)      |       X        |        X
//     Torch  (torch)      |       X        |       (none)
//
// Torch has no committed CUDA-BVH traverser path; it is OptiX-only. The CUDA BVH
// traverser is a Dr.Jit-backend eager-native path.
template <typename LayoutPolicy, typename TraverserType> struct TraceConfig {
    using Layout = LayoutPolicy;
    using Traverser = TraverserType;
};

// Rays that leave a surface. A ray-triangle test computes plane distances from float32 coordinates, so its rounding
// grows with the magnitude of the numbers involved, including the triangle's own vertices: a ray starting on a
// triangle can re-hit it at any angle, and no fixed minimum distance covers every scene size. The tolerance below
// bounds that rounding. A ray leaving a known surface starts on the outgoing side of its plane by
// surface_spawn_offset(), and a segment whose blocker's plane passes through one of its ends is not blocked by it.
// Both are scale-relative only, so they vanish with the coordinates and small scenes keep their close geometry.
inline constexpr float kSurfaceToleranceRelative = 1.0f / 8388608.0f; // 2^-23 of the magnitude.
inline constexpr int kSegmentSurfaceRetries = 4;

RAYD_HOST_DEVICE float coordinate_magnitude(math::Vec3f value) {
    return fmaxf(fabsf(value.x), fmaxf(fabsf(value.y), fabsf(value.z)));
}

RAYD_HOST_DEVICE float triangle_coordinate_magnitude(math::Vec3f p0, math::Vec3f e1, math::Vec3f e2) {
    return fmaxf(coordinate_magnitude(p0),
                 fmaxf(coordinate_magnitude(math::add(p0, e1)), coordinate_magnitude(math::add(p0, e2))));
}

RAYD_HOST_DEVICE float surface_tolerance(float coordinate_scale) {
    return kSurfaceToleranceRelative * coordinate_scale;
}

RAYD_HOST_DEVICE float surface_spawn_offset(float coordinate_scale) {
    return 2.0f * surface_tolerance(coordinate_scale);
}

/// Origin for a ray leaving `point` on the side of `outgoing_normal` (unit, on the ray's side of the surface).
RAYD_HOST_DEVICE math::Vec3f spawn_from_surface(math::Vec3f point, math::Vec3f outgoing_normal,
                                                float coordinate_scale) {
    return math::add(point, math::scale(outgoing_normal, surface_spawn_offset(coordinate_scale)));
}

/// Where a visibility segment should end for a point on a wedge's edge: just off both faces, along the exterior
/// bisector. A segment ending exactly on the edge grazes the wedge's own faces, and one running into the solid crosses
/// no face at all before it stops. That end takes no along-segment bias, which could step across a face it has to
/// cross. Opposite normals (a thin screen) have no bisector, so the point stays and keeps the bias instead.
struct WedgeVisibilityEnd {
    math::Vec3f point;
    float bias;
};

RAYD_HOST_DEVICE WedgeVisibilityEnd wedge_visibility_end(math::Vec3f edge_point, math::Vec3f n0, math::Vec3f n1,
                                                         float bias) {
    const math::Vec3f bisector = math::add(n0, n1);
    const float squared = math::dot(bisector, bisector);
    if (!(squared > 1.0e-12f)) {
        return {edge_point, bias};
    }
    const float offset = fmaxf(bias, surface_spawn_offset(coordinate_magnitude(edge_point)));
    return {math::add(edge_point, math::scale(bisector, offset / sqrtf(squared))), 0.0f};
}

/// The plane of a candidate blocker, as stored for the scene's triangles.
struct SurfacePlane {
    math::Vec3f point;
    math::Vec3f unit_normal;
    float coordinate_scale;
};

/// Closest-or-first true blocker of the segment start-end. `trace(origin, direction, tmax)` casts one occlusion ray
/// and `load_plane(hit, plane)` returns false when the blocker has no known plane. A blocker whose plane passes
/// through an end of the segment (within rounding) cannot lie between the ends, because a line crosses a plane once:
/// that end is moved off the plane, away from the other end, and the segment is cast again. Corners take one retry
/// per plane. A blocker still reported after the last retry stands, so the answer errs towards blocked.
template <typename Trace, typename LoadPlane>
RAYD_DEVICE TriangleHit trace_segment_off_surfaces(math::Vec3f start, math::Vec3f end, float min_length, float bias,
                                                   const Trace& trace, const LoadPlane& load_plane) {
    TriangleHit hit;
    hit.t = 0.0f;
    hit.bary_u = 0.0f;
    hit.bary_v = 0.0f;
    hit.prim = -1;
    hit.instance = -1;
    hit.hit = 0u;
    math::Vec3f near_end = start;
    math::Vec3f far_end = end;
    for (int attempt = 0; attempt <= kSegmentSurfaceRetries; ++attempt) {
        math::Vec3f direction = math::subtract(far_end, near_end);
        const float length = sqrtf(math::dot(direction, direction));
        if (length <= min_length) {
            hit.hit = 0u;
            return hit;
        }
        direction = math::scale(direction, 1.0f / length);
        hit = trace(math::add(near_end, math::scale(direction, bias)), direction, fmaxf(length - 2.0f * bias, 0.0f));
        SurfacePlane plane;
        if (hit.hit == 0u || attempt == kSegmentSurfaceRetries || !load_plane(hit, plane)) {
            return hit;
        }
        const float scale =
            fmaxf(plane.coordinate_scale, fmaxf(coordinate_magnitude(start), coordinate_magnitude(end)));
        const float tolerance = surface_tolerance(scale);
        const bool start_on_plane =
            fabsf(math::dot(math::subtract(start, plane.point), plane.unit_normal)) <= tolerance;
        const bool end_on_plane = fabsf(math::dot(math::subtract(end, plane.point), plane.unit_normal)) <= tolerance;
        if (!start_on_plane && !end_on_plane) {
            return hit;
        }
        const float side = math::dot(direction, plane.unit_normal) >= 0.0f ? 1.0f : -1.0f;
        const math::Vec3f shift = math::scale(plane.unit_normal, side * surface_spawn_offset(scale));
        // An end moves away from the other end, except that a segment lying in the plane keeps both ends on one side.
        if (start_on_plane) {
            near_end = math::add(near_end, shift);
        }
        if (end_on_plane) {
            far_end = start_on_plane ? math::add(far_end, shift) : math::subtract(far_end, shift);
        }
    }
    return hit;
}

} // namespace rayd::shared::rt

#ifdef __CUDACC__

namespace rayd::shared::optix {

static __forceinline__ __device__ int global_primitive_id(int shape_id, int local_primitive, const int* face_offsets,
                                                          int mesh_count) {
    const int face_offset = (shape_id >= 0 && shape_id < mesh_count) ? face_offsets[shape_id] : 0;
    return face_offset + local_primitive;
}

} // namespace rayd::shared::optix

#endif
