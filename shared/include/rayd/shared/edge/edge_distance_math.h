#pragma once

#include <cmath>

#include <rayd/shared/math/vec3.h>

#if defined(__CUDACC__)
#  define RAYD_SHARED_EDGE_MATH_INLINE __host__ __device__ __forceinline__
#else
#  define RAYD_SHARED_EDGE_MATH_INLINE inline
#endif

namespace rayd::shared::edge {

inline constexpr float EdgeDistanceDeviceEpsilon = 1.0e-7f;
inline constexpr float EdgeDistanceFloatMax = 3.4028234663852886e38f;

struct PointSegmentDistance {
    float edge_parameter;
    math::Vec3f edge_point;
    float squared_distance;
};

struct SegmentSegmentDistance {
    float query_parameter;
    math::Vec3f query_point;
    float edge_parameter;
    math::Vec3f edge_point;
    float squared_distance;
};

struct RaySegmentDistance {
    float ray_parameter;
    math::Vec3f ray_point;
    float edge_parameter;
    math::Vec3f edge_point;
    float squared_distance;
};

struct PointSegmentJvp {
    float distance;
    float edge_parameter;
    math::Vec3f edge_point;
    math::Vec3f separation;
};

struct PointSegmentVjp {
    math::Vec3f point;
    math::Vec3f edge_start;
    math::Vec3f edge_end;
};

RAYD_SHARED_EDGE_MATH_INLINE float clamp_unit(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

RAYD_SHARED_EDGE_MATH_INLINE PointSegmentDistance point_segment_distance(
    math::Vec3f point,
    math::Vec3f edge_origin,
    math::Vec3f edge_vector,
    float epsilon = EdgeDistanceDeviceEpsilon) {
    const float edge_length_squared = math::squared_norm(edge_vector);
    const float edge_parameter = edge_length_squared > epsilon
        ? clamp_unit(math::dot(math::subtract(point, edge_origin), edge_vector) /
                     edge_length_squared)
        : 0.0f;
    const math::Vec3f edge_point =
        math::add(edge_origin, math::scale(edge_vector, edge_parameter));
    return {
        edge_parameter,
        edge_point,
        math::squared_norm(math::subtract(point, edge_point)),
    };
}

RAYD_SHARED_EDGE_MATH_INLINE void update_segment_best(
    math::Vec3f query_origin,
    math::Vec3f query_vector,
    math::Vec3f edge_origin,
    math::Vec3f edge_vector,
    float query_parameter,
    float edge_parameter,
    bool enabled,
    SegmentSegmentDistance &best) {
    if (!enabled) {
        return;
    }
    const math::Vec3f query_point =
        math::add(query_origin, math::scale(query_vector, query_parameter));
    const math::Vec3f edge_point =
        math::add(edge_origin, math::scale(edge_vector, edge_parameter));
    const float squared_distance =
        math::squared_norm(math::subtract(query_point, edge_point));
    if (squared_distance < best.squared_distance) {
        best = {
            query_parameter,
            query_point,
            edge_parameter,
            edge_point,
            squared_distance,
        };
    }
}

RAYD_SHARED_EDGE_MATH_INLINE SegmentSegmentDistance segment_segment_distance(
    math::Vec3f query_origin,
    math::Vec3f query_vector,
    math::Vec3f edge_origin,
    math::Vec3f edge_vector,
    float epsilon = EdgeDistanceDeviceEpsilon) {
    const math::Vec3f w0 = math::subtract(query_origin, edge_origin);
    const math::Vec3f query_end = math::add(query_origin, query_vector);
    const math::Vec3f edge_end = math::add(edge_origin, edge_vector);
    const float a = math::squared_norm(query_vector);
    const float b = math::dot(query_vector, edge_vector);
    const float c = math::squared_norm(edge_vector);
    const float d = math::dot(query_vector, w0);
    const float e = math::dot(edge_vector, w0);
    const float determinant = a * c - b * b;

    SegmentSegmentDistance best = {
        0.0f,
        query_origin,
        0.0f,
        edge_origin,
        EdgeDistanceFloatMax,
    };

    const PointSegmentDistance query_start =
        point_segment_distance(query_origin, edge_origin, edge_vector, epsilon);
    update_segment_best(query_origin, query_vector, edge_origin, edge_vector,
                        0.0f, query_start.edge_parameter, true, best);

    const PointSegmentDistance query_finish =
        point_segment_distance(query_end, edge_origin, edge_vector, epsilon);
    update_segment_best(query_origin, query_vector, edge_origin, edge_vector,
                        1.0f, query_finish.edge_parameter, true, best);

    const PointSegmentDistance edge_start =
        point_segment_distance(edge_origin, query_origin, query_vector, epsilon);
    update_segment_best(query_origin, query_vector, edge_origin, edge_vector,
                        edge_start.edge_parameter, 0.0f, true, best);

    const PointSegmentDistance edge_finish =
        point_segment_distance(edge_end, query_origin, query_vector, epsilon);
    update_segment_best(query_origin, query_vector, edge_origin, edge_vector,
                        edge_finish.edge_parameter, 1.0f, true, best);

    const bool interior =
        a > epsilon && c > epsilon && fabsf(determinant) > epsilon;
    if (interior) {
        const float query_parameter = (b * e - c * d) / determinant;
        const float edge_parameter = (a * e - b * d) / determinant;
        update_segment_best(
            query_origin,
            query_vector,
            edge_origin,
            edge_vector,
            query_parameter,
            edge_parameter,
            query_parameter >= 0.0f && query_parameter <= 1.0f &&
                edge_parameter >= 0.0f && edge_parameter <= 1.0f,
            best);
    }
    return best;
}

RAYD_SHARED_EDGE_MATH_INLINE RaySegmentDistance ray_segment_distance(
    math::Vec3f ray_origin,
    math::Vec3f ray_direction,
    math::Vec3f edge_origin,
    math::Vec3f edge_vector,
    float epsilon = EdgeDistanceDeviceEpsilon) {
    const math::Vec3f w0 = math::subtract(ray_origin, edge_origin);
    const math::Vec3f edge_end = math::add(edge_origin, edge_vector);
    const float a = math::squared_norm(ray_direction);
    const float b = math::dot(ray_direction, edge_vector);
    const float c = math::squared_norm(edge_vector);
    const float d = math::dot(ray_direction, w0);
    const float e = math::dot(edge_vector, w0);
    const float determinant = a * c - b * b;

    SegmentSegmentDistance best = {
        0.0f,
        ray_origin,
        0.0f,
        edge_origin,
        EdgeDistanceFloatMax,
    };
    const PointSegmentDistance ray_start =
        point_segment_distance(ray_origin, edge_origin, edge_vector, epsilon);
    update_segment_best(ray_origin, ray_direction, edge_origin, edge_vector,
                        0.0f, ray_start.edge_parameter, true, best);

    const float safe_a = a > epsilon ? a : 1.0f;
    update_segment_best(ray_origin, ray_direction, edge_origin, edge_vector,
                        a > epsilon ? fmaxf(-d / safe_a, 0.0f) : 0.0f,
                        0.0f, true, best);
    update_segment_best(ray_origin, ray_direction, edge_origin, edge_vector,
                        a > epsilon ? fmaxf((b - d) / safe_a, 0.0f) : 0.0f,
                        1.0f, true, best);

    const bool interior =
        a > epsilon && c > epsilon && fabsf(determinant) > epsilon;
    if (interior) {
        const float ray_parameter = (b * e - c * d) / determinant;
        const float edge_parameter = (a * e - b * d) / determinant;
        update_segment_best(
            ray_origin,
            ray_direction,
            edge_origin,
            edge_vector,
            ray_parameter,
            edge_parameter,
            ray_parameter >= 0.0f && edge_parameter >= 0.0f && edge_parameter <= 1.0f,
            best);
    }

    return {
        best.query_parameter,
        best.query_point,
        best.edge_parameter,
        best.edge_point,
        best.squared_distance,
    };
}

RAYD_SHARED_EDGE_MATH_INLINE PointSegmentJvp point_segment_jvp_fixed_winner(
    math::Vec3f point,
    math::Vec3f edge_start,
    math::Vec3f edge_end,
    float edge_parameter,
    math::Vec3f separation,
    math::Vec3f tangent_point,
    math::Vec3f tangent_edge_start,
    math::Vec3f tangent_edge_end) {
    const math::Vec3f edge_vector = math::subtract(edge_end, edge_start);
    const math::Vec3f tangent_edge_vector =
        math::subtract(tangent_edge_end, tangent_edge_start);
    const math::Vec3f point_from_start = math::subtract(point, edge_start);
    const math::Vec3f tangent_point_from_start =
        math::subtract(tangent_point, tangent_edge_start);
    const float denominator = fmaxf(math::squared_norm(edge_vector), 1.0e-20f);
    const float numerator = math::dot(point_from_start, edge_vector);
    float tangent_parameter = 0.0f;
    if (edge_parameter > 0.0f && edge_parameter < 1.0f) {
        tangent_parameter =
            (math::dot(tangent_point_from_start, edge_vector) +
             math::dot(point_from_start, tangent_edge_vector)) /
                denominator -
            numerator * (2.0f * math::dot(edge_vector, tangent_edge_vector)) /
                (denominator * denominator);
    }

    const math::Vec3f tangent_edge_point = math::add(
        math::add(tangent_edge_start,
                  math::scale(tangent_edge_vector, edge_parameter)),
        math::scale(edge_vector, tangent_parameter));
    const math::Vec3f tangent_separation =
        math::subtract(tangent_point, tangent_edge_point);
    const float distance = sqrtf(fmaxf(math::squared_norm(separation), 1.0e-20f));
    const float tangent_distance =
        math::dot(math::scale(separation, 1.0f / distance), tangent_separation);
    return {
        tangent_distance,
        tangent_parameter,
        tangent_edge_point,
        tangent_separation,
    };
}

RAYD_SHARED_EDGE_MATH_INLINE PointSegmentVjp point_segment_vjp_fixed_winner(
    math::Vec3f point,
    math::Vec3f edge_start,
    math::Vec3f edge_end,
    float edge_parameter,
    math::Vec3f separation,
    float grad_distance,
    math::Vec3f grad_edge_point,
    float grad_edge_parameter) {
    const math::Vec3f edge_vector = math::subtract(edge_end, edge_start);
    const float distance = sqrtf(fmaxf(math::squared_norm(separation), 1.0e-20f));
    const math::Vec3f grad_separation =
        math::scale(separation, grad_distance / distance);
    const math::Vec3f edge_point_bar =
        math::subtract(grad_edge_point, grad_separation);
    PointSegmentVjp result = {
        grad_separation,
        math::scale(edge_point_bar, 1.0f - edge_parameter),
        math::scale(edge_point_bar, edge_parameter),
    };

    float parameter_bar =
        math::dot(edge_point_bar, edge_vector) + grad_edge_parameter;
    if (parameter_bar != 0.0f &&
        edge_parameter > 0.0f && edge_parameter < 1.0f) {
        const float denominator =
            fmaxf(math::squared_norm(edge_vector), 1.0e-20f);
        const math::Vec3f point_from_start = math::subtract(point, edge_start);
        const float numerator = math::dot(point_from_start, edge_vector);
        const math::Vec3f point_term =
            math::scale(edge_vector, parameter_bar / denominator);
        const math::Vec3f edge_term = math::scale(
            math::subtract(
                math::scale(point_from_start, 1.0f / denominator),
                math::scale(edge_vector,
                            (2.0f * numerator) / (denominator * denominator))),
            parameter_bar);
        result.point = math::add(result.point, point_term);
        result.edge_start = math::subtract(
            result.edge_start, math::add(point_term, edge_term));
        result.edge_end = math::add(result.edge_end, edge_term);
    }
    return result;
}

} // namespace rayd::shared::edge

#undef RAYD_SHARED_EDGE_MATH_INLINE
