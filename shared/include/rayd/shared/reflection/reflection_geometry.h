#pragma once

#include <rayd/shared/math/vec3.h>

#if defined(__CUDACC__)
#  define RAYD_SHARED_REFLECTION_INLINE __host__ __device__ __forceinline__
#else
#  define RAYD_SHARED_REFLECTION_INLINE inline
#endif

namespace rayd::shared::reflection {

RAYD_SHARED_REFLECTION_INLINE math::Vec3f orient_normal_against(
    math::Vec3f incident_direction,
    math::Vec3f unit_normal) {
    return math::dot(incident_direction, unit_normal) > 0.0f
        ? math::scale(unit_normal, -1.0f)
        : unit_normal;
}

RAYD_SHARED_REFLECTION_INLINE math::Vec3f reflect_direction(
    math::Vec3f incident_direction,
    math::Vec3f unit_normal) {
    return math::subtract(
        incident_direction,
        math::scale(unit_normal, 2.0f * math::dot(incident_direction, unit_normal)));
}

RAYD_SHARED_REFLECTION_INLINE math::Vec3f reflect_point_across_plane(
    math::Vec3f point,
    math::Vec3f plane_point,
    math::Vec3f unit_normal) {
    return math::subtract(
        point,
        math::scale(
            unit_normal,
            2.0f * math::dot(math::subtract(point, plane_point), unit_normal)));
}

RAYD_SHARED_REFLECTION_INLINE bool intersect_segment_plane(
    math::Vec3f segment_start,
    math::Vec3f segment_end,
    math::Vec3f plane_point,
    math::Vec3f plane_normal,
    float parallel_tolerance,
    float segment_tolerance,
    math::Vec3f &intersection) {
    const math::Vec3f direction = math::subtract(segment_end, segment_start);
    const float denominator = math::dot(direction, plane_normal);
    if (fabsf(denominator) <= parallel_tolerance) {
        return false;
    }
    const float t = math::dot(
        math::subtract(plane_point, segment_start), plane_normal) / denominator;
    if (t < -segment_tolerance || t > 1.0f + segment_tolerance) {
        return false;
    }
    intersection = math::add(segment_start, math::scale(direction, t));
    return true;
}

} // namespace rayd::shared::reflection

#undef RAYD_SHARED_REFLECTION_INLINE
