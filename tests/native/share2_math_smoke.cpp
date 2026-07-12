#include <cmath>

#include <rayd/shared/edge/edge_distance_math.h>
#include <rayd/shared/reflection/reflection_geometry.h>

namespace {

bool close(float actual, float expected) {
    return std::fabs(actual - expected) <= 1.0e-6f;
}

bool close(rayd::shared::math::Vec3f actual,
           rayd::shared::math::Vec3f expected) {
    return close(actual.x, expected.x) &&
           close(actual.y, expected.y) &&
           close(actual.z, expected.z);
}

} // namespace

int main() {
    using rayd::shared::math::make_vec3;

    const auto point = rayd::shared::edge::point_segment_distance(
        make_vec3(0.25f, 0.5f, 0.0f),
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(1.0f, 0.0f, 0.0f));
    if (!close(point.edge_parameter, 0.25f) ||
        !close(point.edge_point, make_vec3(0.25f, 0.0f, 0.0f)) ||
        !close(point.squared_distance, 0.25f)) {
        return 1;
    }

    const auto segments = rayd::shared::edge::segment_segment_distance(
        make_vec3(0.5f, -1.0f, 0.0f),
        make_vec3(0.0f, 2.0f, 0.0f),
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(1.0f, 0.0f, 0.0f));
    if (!close(segments.query_parameter, 0.5f) ||
        !close(segments.edge_parameter, 0.5f) ||
        !close(segments.squared_distance, 0.0f)) {
        return 2;
    }

    const auto ray = rayd::shared::edge::ray_segment_distance(
        make_vec3(-1.0f, 1.0f, 0.0f),
        make_vec3(1.0f, 0.0f, 0.0f),
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(0.0f, 2.0f, 0.0f));
    if (!close(ray.ray_parameter, 1.0f) ||
        !close(ray.edge_parameter, 0.5f) ||
        !close(ray.squared_distance, 0.0f)) {
        return 3;
    }

    const auto jvp = rayd::shared::edge::point_segment_jvp_fixed_winner(
        make_vec3(0.25f, 0.5f, 0.0f),
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(1.0f, 0.0f, 0.0f),
        0.25f,
        make_vec3(0.0f, 0.5f, 0.0f),
        make_vec3(1.0f, 0.0f, 0.0f),
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(0.0f, 0.0f, 0.0f));
    if (!close(jvp.edge_parameter, 1.0f) ||
        !close(jvp.edge_point, make_vec3(1.0f, 0.0f, 0.0f)) ||
        !close(jvp.distance, 0.0f)) {
        return 4;
    }

    const auto vjp = rayd::shared::edge::point_segment_vjp_fixed_winner(
        make_vec3(0.25f, 0.5f, 0.0f),
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(1.0f, 0.0f, 0.0f),
        0.25f,
        make_vec3(0.0f, 0.5f, 0.0f),
        0.0f,
        make_vec3(0.0f, 0.0f, 0.0f),
        1.0f);
    if (!close(vjp.point, make_vec3(1.0f, 0.0f, 0.0f)) ||
        !close(vjp.edge_start, make_vec3(-0.75f, -0.5f, 0.0f)) ||
        !close(vjp.edge_end, make_vec3(-0.25f, 0.5f, 0.0f))) {
        return 5;
    }

    const auto normal = rayd::shared::reflection::orient_normal_against(
        make_vec3(0.0f, 0.0f, 1.0f),
        make_vec3(0.0f, 0.0f, 1.0f));
    const auto reflected = rayd::shared::reflection::reflect_direction(
        make_vec3(0.0f, 0.0f, 1.0f), normal);
    const auto image = rayd::shared::reflection::reflect_point_across_plane(
        make_vec3(0.0f, 0.0f, 2.0f),
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(0.0f, 0.0f, 1.0f));
    if (!close(normal, make_vec3(0.0f, 0.0f, -1.0f)) ||
        !close(reflected, make_vec3(0.0f, 0.0f, -1.0f)) ||
        !close(image, make_vec3(0.0f, 0.0f, -2.0f))) {
        return 6;
    }
    return 0;
}
