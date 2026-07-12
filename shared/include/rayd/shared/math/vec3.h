#pragma once

#include <cmath>

#if defined(__CUDACC__)
#  define RAYD_SHARED_MATH_INLINE __host__ __device__ __forceinline__
#else
#  define RAYD_SHARED_MATH_INLINE inline
#endif

namespace rayd::shared::math {

struct Vec3f {
    float x;
    float y;
    float z;
};

RAYD_SHARED_MATH_INLINE Vec3f make_vec3(float x, float y, float z) {
    return {x, y, z};
}

RAYD_SHARED_MATH_INLINE Vec3f add(Vec3f a, Vec3f b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

RAYD_SHARED_MATH_INLINE Vec3f subtract(Vec3f a, Vec3f b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

RAYD_SHARED_MATH_INLINE Vec3f scale(Vec3f value, float factor) {
    return {value.x * factor, value.y * factor, value.z * factor};
}

RAYD_SHARED_MATH_INLINE float dot(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

RAYD_SHARED_MATH_INLINE float squared_norm(Vec3f value) {
    return dot(value, value);
}

RAYD_SHARED_MATH_INLINE Vec3f cross(Vec3f a, Vec3f b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

} // namespace rayd::shared::math

#undef RAYD_SHARED_MATH_INLINE
