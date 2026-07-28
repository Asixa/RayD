#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include <rayd/shared/math/vec3.h>

namespace rayd::shared::rt {

// Backend-neutral, decoded closest-hit record produced by a Traverser
// (RAY_TRACING_BACKEND_ARCHITECTURE.md §7). It is the host-side mirror of the
// OptiX six-register TriangleHitPayload (shared/optix/device_hit.h) after the
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
// (shared/optix/optix_traverser.h, shared/bvh/cuda_bvh_traverser.h) each assert
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
template <typename T, typename = void>
struct is_traverser : std::false_type {};

template <typename T>
struct is_traverser<
    T,
    std::void_t<
        decltype(std::declval<const T &>().trace_closest(
            std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
            std::declval<float>(), std::declval<float>())),
        decltype(std::declval<const T &>().trace_occluded(
            std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
            std::declval<float>(), std::declval<float>())),
        decltype(std::declval<const T &>().trace_occluded_ignore(
            std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
            std::declval<float>(), std::declval<float>(),
            std::declval<const std::int32_t *>(), std::declval<int>())),
        decltype(std::declval<const T &>().trace_first_blocker(
            std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
            std::declval<float>(), std::declval<float>(),
            std::declval<const std::int32_t *>(), std::declval<int>()))>>
    : std::bool_constant<
          std::is_same_v<
              decltype(std::declval<const T &>().trace_closest(
                  std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
                  std::declval<float>(), std::declval<float>())),
              TriangleHit> &&
          std::is_same_v<
              decltype(std::declval<const T &>().trace_occluded(
                  std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
                  std::declval<float>(), std::declval<float>())),
              bool> &&
          std::is_same_v<
              decltype(std::declval<const T &>().trace_occluded_ignore(
                  std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
                  std::declval<float>(), std::declval<float>(),
                  std::declval<const std::int32_t *>(), std::declval<int>())),
              bool> &&
          std::is_same_v<
              decltype(std::declval<const T &>().trace_first_blocker(
                  std::declval<math::Vec3f>(), std::declval<math::Vec3f>(),
                  std::declval<float>(), std::declval<float>(),
                  std::declval<const std::int32_t *>(), std::declval<int>())),
              TriangleHit>> {};

template <typename T>
inline constexpr bool is_traverser_v = is_traverser<T>::value;

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
template <typename LayoutPolicy, typename TraverserType>
struct TraceConfig {
    using Layout = LayoutPolicy;
    using Traverser = TraverserType;
};

} // namespace rayd::shared::rt
