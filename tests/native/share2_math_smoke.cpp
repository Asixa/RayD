// Copyright Xingyu Chen.
// Exercises share2 math smoke in a native smoke test.

#include <cmath>

#include <src/edge/edge_distance.h>
#include <src/reflection/reflection_algorithms.cuh>

namespace {

bool close(float actual, float expected) {
    return std::fabs(actual - expected) <= 1.0e-6f;
}

bool close(rayd::shared::math::Vec3f actual, rayd::shared::math::Vec3f expected) {
    return close(actual.x, expected.x) && close(actual.y, expected.y) && close(actual.z, expected.z);
}

} // namespace

int main() {
    using rayd::shared::math::make_vec3;

    const auto point =
        rayd::shared::edge::point_segment_distance(make_vec3(0.25f, 0.5f, 0.0f), make_vec3(0.0f, 0.0f, 0.0f),
                                                   make_vec3(1.0f, 0.0f, 0.0f));
    if (!close(point.edge_parameter, 0.25f) || !close(point.edge_point, make_vec3(0.25f, 0.0f, 0.0f)) ||
        !close(point.squared_distance, 0.25f)) {
        return 1;
    }

    const auto segments =
        rayd::shared::edge::segment_segment_distance(make_vec3(0.5f, -1.0f, 0.0f), make_vec3(0.0f, 2.0f, 0.0f),
                                                     make_vec3(0.0f, 0.0f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f));
    if (!close(segments.query_parameter, 0.5f) || !close(segments.edge_parameter, 0.5f) ||
        !close(segments.squared_distance, 0.0f)) {
        return 2;
    }

    const auto ray = rayd::shared::edge::ray_segment_distance(make_vec3(-1.0f, 1.0f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f),
                                                              make_vec3(0.0f, 0.0f, 0.0f), make_vec3(0.0f, 2.0f, 0.0f));
    if (!close(ray.ray_parameter, 1.0f) || !close(ray.edge_parameter, 0.5f) || !close(ray.squared_distance, 0.0f)) {
        return 3;
    }

    const auto jvp =
        rayd::shared::edge::point_segment_jvp_fixed_winner(make_vec3(0.25f, 0.5f, 0.0f), make_vec3(0.0f, 0.0f, 0.0f),
                                                           make_vec3(1.0f, 0.0f, 0.0f), 0.25f,
                                                           make_vec3(0.0f, 0.5f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f),
                                                           make_vec3(0.0f, 0.0f, 0.0f), make_vec3(0.0f, 0.0f, 0.0f));
    if (!close(jvp.edge_parameter, 1.0f) || !close(jvp.edge_point, make_vec3(1.0f, 0.0f, 0.0f)) ||
        !close(jvp.distance, 0.0f)) {
        return 4;
    }

    const auto vjp =
        rayd::shared::edge::point_segment_vjp_fixed_winner(make_vec3(0.25f, 0.5f, 0.0f), make_vec3(0.0f, 0.0f, 0.0f),
                                                           make_vec3(1.0f, 0.0f, 0.0f), 0.25f,
                                                           make_vec3(0.0f, 0.5f, 0.0f), 0.0f,
                                                           make_vec3(0.0f, 0.0f, 0.0f), 1.0f);
    if (!close(vjp.point, make_vec3(1.0f, 0.0f, 0.0f)) || !close(vjp.edge_start, make_vec3(-0.75f, -0.5f, 0.0f)) ||
        !close(vjp.edge_end, make_vec3(-0.25f, 0.5f, 0.0f))) {
        return 5;
    }

    const auto ray_jvp =
        rayd::shared::edge::ray_segment_jvp_fixed_winner(make_vec3(0.5f, -0.25f, 1.0f), make_vec3(0.0f, 0.0f, -1.0f),
                                                         make_vec3(0.0f, 0.0f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f), 1.0f,
                                                         0.5f, false, rayd::shared::edge::EdgeDistanceFloatMax,
                                                         make_vec3(0.0f, 0.0f, 0.2f), make_vec3(0.0f, 0.0f, 0.0f),
                                                         make_vec3(0.0f, 0.0f, 0.3f), make_vec3(0.0f, 0.0f, 0.3f));
    if (!close(ray_jvp.distance, 0.0f) || !close(ray_jvp.ray_parameter, -0.1f) ||
        !close(ray_jvp.ray_point, make_vec3(0.0f, 0.0f, 0.3f)) || !close(ray_jvp.edge_parameter, 0.0f) ||
        !close(ray_jvp.edge_point, make_vec3(0.0f, 0.0f, 0.3f))) {
        return 6;
    }

    const auto ray_vjp =
        rayd::shared::edge::ray_segment_vjp_fixed_winner(make_vec3(0.5f, -0.25f, 1.0f), make_vec3(0.0f, 0.0f, -1.0f),
                                                         make_vec3(0.0f, 0.0f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f), 1.0f,
                                                         0.5f, true, 2.0f, 1.0f, 0.0f, make_vec3(0.0f, 0.0f, 0.0f),
                                                         0.0f, make_vec3(0.0f, 0.0f, 0.0f));
    if (!close(ray_vjp.ray_origin, make_vec3(0.0f, -1.0f, 0.0f)) ||
        !close(ray_vjp.ray_direction, make_vec3(0.0f, -1.0f, 0.0f)) ||
        !close(ray_vjp.edge_start, make_vec3(0.0f, 0.5f, 0.0f)) ||
        !close(ray_vjp.edge_end, make_vec3(0.0f, 0.5f, 0.0f))) {
        return 7;
    }

    const auto normal =
        rayd::shared::reflection::orient_normal_against(make_vec3(0.0f, 0.0f, 1.0f), make_vec3(0.0f, 0.0f, 1.0f));
    const auto reflected = rayd::shared::reflection::reflect_direction(make_vec3(0.0f, 0.0f, 1.0f), normal);
    const auto image =
        rayd::shared::reflection::reflect_point_across_plane(make_vec3(0.0f, 0.0f, 2.0f), make_vec3(0.0f, 0.0f, 0.0f),
                                                             make_vec3(0.0f, 0.0f, 1.0f));
    rayd::shared::math::Vec3f plane_hit{};
    const bool plane_hit_valid =
        rayd::shared::reflection::intersect_segment_plane(make_vec3(0.0f, 0.0f, -1.0f), make_vec3(0.0f, 0.0f, 1.0f),
                                                          make_vec3(0.0f, 0.0f, 0.0f), make_vec3(0.0f, 0.0f, 1.0f),
                                                          1.0e-7f, 1.0e-4f, plane_hit);
    if (!close(normal, make_vec3(0.0f, 0.0f, -1.0f)) || !close(reflected, make_vec3(0.0f, 0.0f, -1.0f)) ||
        !close(image, make_vec3(0.0f, 0.0f, -2.0f)) || !plane_hit_valid ||
        !close(plane_hit, make_vec3(0.0f, 0.0f, 0.0f))) {
        return 8;
    }
    return 0;
}
