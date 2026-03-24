#pragma once

#include <array>
#include <vector>

#include <rayd/rayd.h>

namespace rayd {

template <typename ArrayD, typename Mask_>
DRJIT_INLINE ArrayD compressD(const ArrayD &array, const Mask_ &active) {
    auto idx = compress(active);
    return gather<ArrayD>(array, idx);
}

template <typename ArrayD>
DRJIT_INLINE size_t slices(const ArrayD &cuda_array) {
    if constexpr (depth_v<ArrayD> == 1) {
        return cuda_array.size();
    } else {
        return cuda_array[0].size();
    }
}

template <typename T, size_t n, bool async = false>
DRJIT_INLINE void copy_cuda_array(const Array<CUDAArray<T>, n> &cuda_array,
                                  std::array<std::vector<T>, n> &cpu_array) {
    size_t m = slices<Array<CUDAArray<T>, n>>(cuda_array);
    for (size_t i = 0; i < n; ++i) {
        cpu_array[i].resize(m);
        drjit::store(cpu_array[i].data(), cuda_array[i]);
    }
}

template <bool Detached>
DRJIT_INLINE Vector3fT<Detached> bilinear(const Vector3fT<Detached> &p0,
                                          const Vector3fT<Detached> &e1,
                                          const Vector3fT<Detached> &e2,
                                          const Vector2fT<Detached> &st) {
    return fmadd(e1, st.x(), fmadd(e2, st.y(), p0));
}

template <bool Detached>
DRJIT_INLINE Vector2fT<Detached> bilinear2(const Vector2fT<Detached> &p0,
                                           const Vector2fT<Detached> &e1,
                                           const Vector2fT<Detached> &e2,
                                           const Vector2fT<Detached> &st) {
    return fmadd(e1, st.x(), fmadd(e2, st.y(), p0));
}

template <bool Detached>
DRJIT_INLINE auto ray_intersect_triangle(const Vector3fT<Detached> &p0,
                                         const Vector3fT<Detached> &e1,
                                         const Vector3fT<Detached> &e2,
                                         const RayT<Detached> &ray) {
    Vector3fT<Detached> h = cross(ray.d, e2);
    FloatT<Detached> a = dot(e1, h);
    MaskT<Detached> valid = neq(a, 0.f);
    FloatT<Detached> safe_a = select(valid, a, 1.f);
    FloatT<Detached> f = rcp(safe_a);
    Vector3fT<Detached> s = ray.o - p0;
    FloatT<Detached> u = f * dot(s, h);
    Vector3fT<Detached> q = cross(s, e1);
    FloatT<Detached> v = f * dot(ray.d, q);
    FloatT<Detached> t = f * dot(e2, q);
    u = select(valid, u, 0.f);
    v = select(valid, v, 0.f);
    t = select(valid, t, full<FloatT<Detached>>(Infinity, slices(ray.o)));
    return std::make_pair(Vector2fT<Detached>(u, v), t);
}

template <typename Float_>
DRJIT_INLINE Float_ clamp01(const Float_ &value) {
    return maximum(minimum(value, Float_(1.f)), Float_(0.f));
}

template <bool Detached>
DRJIT_INLINE auto closest_point_on_segment(const Vector3fT<Detached> &point,
                                           const Vector3fT<Detached> &p0,
                                           const Vector3fT<Detached> &e1) {
    const FloatT<Detached> edge_length_sq = squared_norm(e1);
    const MaskT<Detached> valid_edge = edge_length_sq > Epsilon;
    const FloatT<Detached> safe_edge_length_sq = select(valid_edge, edge_length_sq, FloatT<Detached>(1.f));
    const FloatT<Detached> edge_t =
        select(valid_edge, clamp01< FloatT<Detached> >(dot(point - p0, e1) / safe_edge_length_sq), FloatT<Detached>(0.f));
    const Vector3fT<Detached> edge_point = fmadd(e1, edge_t, p0);
    const FloatT<Detached> distance_sq = squared_norm(point - edge_point);
    return std::make_tuple(edge_t, edge_point, distance_sq);
}

template <bool Detached>
DRJIT_INLINE auto closest_segment_segment(const Vector3fT<Detached> &query_origin,
                                          const Vector3fT<Detached> &query_edge,
                                          const Vector3fT<Detached> &edge_origin,
                                          const Vector3fT<Detached> &edge_vector) {
    const Vector3fT<Detached> w0 = query_origin - edge_origin;
    const Vector3fT<Detached> query_end = query_origin + query_edge;
    const Vector3fT<Detached> edge_end = edge_origin + edge_vector;

    const FloatT<Detached> a = squared_norm(query_edge);
    const FloatT<Detached> b = dot(query_edge, edge_vector);
    const FloatT<Detached> c = squared_norm(edge_vector);
    const FloatT<Detached> d = dot(query_edge, w0);
    const FloatT<Detached> e = dot(edge_vector, w0);
    const FloatT<Detached> det = fmsub(a, c, b * b);

    FloatT<Detached> best_distance_sq = full<FloatT<Detached>>(Infinity, slices(query_origin));
    FloatT<Detached> best_query_t = zeros<FloatT<Detached>>(slices(query_origin));
    FloatT<Detached> best_edge_t = zeros<FloatT<Detached>>(slices(query_origin));
    Vector3fT<Detached> best_query_point = query_origin;
    Vector3fT<Detached> best_edge_point = edge_origin;

    auto update = [&](const MaskT<Detached> &mask,
                      const FloatT<Detached> &query_t,
                      const FloatT<Detached> &edge_t) {
        const Vector3fT<Detached> query_point = fmadd(query_edge, query_t, query_origin);
        const Vector3fT<Detached> edge_point = fmadd(edge_vector, edge_t, edge_origin);
        const FloatT<Detached> distance_sq = squared_norm(query_point - edge_point);
        const MaskT<Detached> better = mask && (distance_sq < best_distance_sq);
        best_distance_sq = select(better, distance_sq, best_distance_sq);
        best_query_t = select(better, query_t, best_query_t);
        best_edge_t = select(better, edge_t, best_edge_t);
        best_query_point = select(better, query_point, best_query_point);
        best_edge_point = select(better, edge_point, best_edge_point);
    };

    {
        FloatT<Detached> edge_t;
        Vector3fT<Detached> edge_point;
        FloatT<Detached> distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<Detached>(query_origin, edge_origin, edge_vector);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(query_origin)), FloatT<Detached>(0.f), edge_t);
    }

    {
        FloatT<Detached> edge_t;
        Vector3fT<Detached> edge_point;
        FloatT<Detached> distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<Detached>(query_end, edge_origin, edge_vector);
        DRJIT_MARK_USED(edge_point);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(query_origin)), FloatT<Detached>(1.f), edge_t);
    }

    {
        FloatT<Detached> query_t;
        Vector3fT<Detached> query_point;
        FloatT<Detached> distance_sq;
        std::tie(query_t, query_point, distance_sq) = closest_point_on_segment<Detached>(edge_origin, query_origin, query_edge);
        DRJIT_MARK_USED(query_point);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(query_origin)), query_t, FloatT<Detached>(0.f));
    }

    {
        FloatT<Detached> query_t;
        Vector3fT<Detached> query_point;
        FloatT<Detached> distance_sq;
        std::tie(query_t, query_point, distance_sq) = closest_point_on_segment<Detached>(edge_end, query_origin, query_edge);
        DRJIT_MARK_USED(query_point);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(query_origin)), query_t, FloatT<Detached>(1.f));
    }

    const MaskT<Detached> interior =
        (a > Epsilon) && (c > Epsilon) && (abs(det) > Epsilon);
    const FloatT<Detached> safe_det = select(interior, det, FloatT<Detached>(1.f));
    const FloatT<Detached> query_t_line = (b * e - c * d) / safe_det;
    const FloatT<Detached> edge_t_line = (a * e - b * d) / safe_det;
    update(interior && query_t_line >= 0.f && query_t_line <= 1.f && edge_t_line >= 0.f && edge_t_line <= 1.f,
           query_t_line,
           edge_t_line);

    return std::make_tuple(best_query_t, best_query_point, best_edge_t, best_edge_point, best_distance_sq);
}

template <bool Detached>
DRJIT_INLINE auto closest_ray_segment(const Vector3fT<Detached> &ray_origin,
                                      const Vector3fT<Detached> &ray_direction,
                                      const Vector3fT<Detached> &edge_origin,
                                      const Vector3fT<Detached> &edge_vector) {
    const Vector3fT<Detached> w0 = ray_origin - edge_origin;
    const Vector3fT<Detached> edge_end = edge_origin + edge_vector;

    const FloatT<Detached> a = squared_norm(ray_direction);
    const FloatT<Detached> b = dot(ray_direction, edge_vector);
    const FloatT<Detached> c = squared_norm(edge_vector);
    const FloatT<Detached> d = dot(ray_direction, w0);
    const FloatT<Detached> e = dot(edge_vector, w0);
    const FloatT<Detached> det = fmsub(a, c, b * b);

    FloatT<Detached> best_distance_sq = full<FloatT<Detached>>(Infinity, slices(ray_origin));
    FloatT<Detached> best_query_t = zeros<FloatT<Detached>>(slices(ray_origin));
    FloatT<Detached> best_edge_t = zeros<FloatT<Detached>>(slices(ray_origin));
    Vector3fT<Detached> best_query_point = ray_origin;
    Vector3fT<Detached> best_edge_point = edge_origin;

    auto update = [&](const MaskT<Detached> &mask,
                      const FloatT<Detached> &query_t,
                      const FloatT<Detached> &edge_t) {
        const Vector3fT<Detached> query_point = fmadd(ray_direction, query_t, ray_origin);
        const Vector3fT<Detached> edge_point = fmadd(edge_vector, edge_t, edge_origin);
        const FloatT<Detached> distance_sq = squared_norm(query_point - edge_point);
        const MaskT<Detached> better = mask && (distance_sq < best_distance_sq);
        best_distance_sq = select(better, distance_sq, best_distance_sq);
        best_query_t = select(better, query_t, best_query_t);
        best_edge_t = select(better, edge_t, best_edge_t);
        best_query_point = select(better, query_point, best_query_point);
        best_edge_point = select(better, edge_point, best_edge_point);
    };

    {
        FloatT<Detached> edge_t;
        Vector3fT<Detached> edge_point;
        FloatT<Detached> distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<Detached>(ray_origin, edge_origin, edge_vector);
        DRJIT_MARK_USED(edge_point);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(ray_origin)), FloatT<Detached>(0.f), edge_t);
    }

    const MaskT<Detached> valid_ray = a > Epsilon;
    const FloatT<Detached> safe_a = select(valid_ray, a, FloatT<Detached>(1.f));
    update(full<MaskT<Detached>>(true, slices(ray_origin)),
           select(valid_ray, maximum(-d / safe_a, FloatT<Detached>(0.f)), FloatT<Detached>(0.f)),
           FloatT<Detached>(0.f));
    update(full<MaskT<Detached>>(true, slices(ray_origin)),
           select(valid_ray, maximum((b - d) / safe_a, FloatT<Detached>(0.f)), FloatT<Detached>(0.f)),
           FloatT<Detached>(1.f));

    const MaskT<Detached> interior =
        valid_ray && (c > Epsilon) && (abs(det) > Epsilon);
    const FloatT<Detached> safe_det = select(interior, det, FloatT<Detached>(1.f));
    const FloatT<Detached> query_t_line = (b * e - c * d) / safe_det;
    const FloatT<Detached> edge_t_line = (a * e - b * d) / safe_det;
    update(interior && query_t_line >= 0.f && edge_t_line >= 0.f && edge_t_line <= 1.f,
           query_t_line,
           edge_t_line);

    return std::make_tuple(best_query_t, best_query_point, best_edge_t, best_edge_point, best_distance_sq);
}

template <typename Float_>
DRJIT_INLINE auto point_aabb_distance_sq(const Array<Float_, 3> &point,
                                         const Array<Float_, 3> &bbox_min,
                                         const Array<Float_, 3> &bbox_max) {
    const Array<Float_, 3> clamped = maximum(minimum(point, bbox_max), bbox_min);
    return squared_norm(point - clamped);
}

template <typename Float_>
DRJIT_INLINE auto segment_aabb_lower_bound_sq(const Array<Float_, 3> &origin,
                                              const Array<Float_, 3> &segment,
                                              const Array<Float_, 3> &bbox_min,
                                              const Array<Float_, 3> &bbox_max) {
    const Array<Float_, 3> segment_end = origin + segment;
    const Array<Float_, 3> path_min = minimum(origin, segment_end);
    const Array<Float_, 3> path_max = maximum(origin, segment_end);
    const Array<Float_, 3> below = maximum(bbox_min - path_max, Array<Float_, 3>(0.f));
    const Array<Float_, 3> above = maximum(path_min - bbox_max, Array<Float_, 3>(0.f));
    return squared_norm(below + above);
}

template <typename Float_>
DRJIT_INLINE auto ray_aabb_lower_bound_sq(const Array<Float_, 3> &origin,
                                          const Array<Float_, 3> &direction,
                                          const Array<Float_, 3> &bbox_min,
                                          const Array<Float_, 3> &bbox_max) {
    using Mask_ = mask_t<Float_>;

    auto axis_distance = [](const Float_ &o,
                            const Float_ &d,
                            const Float_ &axis_min,
                            const Float_ &axis_max) {
        const Mask_ positive = d > Epsilon;
        const Mask_ negative = d < -Epsilon;
        const Mask_ stationary = !(positive || negative);

        Float_ delta = zeros<Float_>(slices(o));
        delta = select(positive, maximum(o - axis_max, Float_(0.f)), delta);
        delta = select(negative, maximum(axis_min - o, Float_(0.f)), delta);
        delta = select(stationary,
                       maximum(axis_min - o, Float_(0.f)) + maximum(o - axis_max, Float_(0.f)),
                       delta);
        return delta;
    };

    const Float_ dx = axis_distance(origin.x(), direction.x(), bbox_min.x(), bbox_max.x());
    const Float_ dy = axis_distance(origin.y(), direction.y(), bbox_min.y(), bbox_max.y());
    const Float_ dz = axis_distance(origin.z(), direction.z(), bbox_min.z(), bbox_max.z());
    return fmadd(dx, dx, fmadd(dy, dy, dz * dz));
}

} // namespace rayd
