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

} // namespace rayd::shared::reflection

#undef RAYD_SHARED_REFLECTION_INLINE
