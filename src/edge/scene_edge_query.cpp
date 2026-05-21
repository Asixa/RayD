#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include <rayd/scene/scene.h>
#include <rayd/edge/scene_edge.h>

#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>

#include "../scene/scene_internal.h"

namespace rayd {

namespace {

bool edge_backend_uses_optix_point(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

bool edge_backend_uses_optix_ray(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_uses_optix_topk(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

template <bool Detached>
NearestPointEdgeT<Detached> initialize_nearest_point_edge_result(int query_count) {
    NearestPointEdgeT<Detached> result;
    result.distance = full<FloatT<Detached>>(Infinity, query_count);
    result.point = zeros<Vector3fT<Detached>>(query_count);
    result.edge_t = zeros<FloatT<Detached>>(query_count);
    result.edge_point = zeros<Vector3fT<Detached>>(query_count);
    result.shape_id = full<IntT<Detached>>(-1, query_count);
    result.edge_id = full<IntT<Detached>>(-1, query_count);
    result.global_edge_id = full<IntT<Detached>>(-1, query_count);
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> initialize_nearest_ray_edge_result(int query_count) {
    NearestRayEdgeT<Detached> result;
    result.distance = full<FloatT<Detached>>(Infinity, query_count);
    result.ray_t = zeros<FloatT<Detached>>(query_count);
    result.point = zeros<Vector3fT<Detached>>(query_count);
    result.edge_t = zeros<FloatT<Detached>>(query_count);
    result.edge_point = zeros<Vector3fT<Detached>>(query_count);
    result.shape_id = full<IntT<Detached>>(-1, query_count);
    result.edge_id = full<IntT<Detached>>(-1, query_count);
    result.global_edge_id = full<IntT<Detached>>(-1, query_count);
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
}

} // namespace

template <bool Detached>
NearestPointEdgeT<Detached> Scene::nearest_edge(const Vector3fT<Detached> &point, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(point): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(point): scene has pending updates. Call Scene::sync() first.");

    const int query_count = static_cast<int>(slices(point));
    NearestPointEdgeT<Detached> result = initialize_nearest_point_edge_result<Detached>(query_count);
    if (edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    MaskDetached active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x()));
        active_detached &= drjit::isfinite(detach<false>(point.y()));
        active_detached &= drjit::isfinite(detach<false>(point.z()));
        active &= Mask(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) && drjit::isfinite(point.y()) && drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_point(edge_bvh_backend_);
    ClosestEdgeCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edge<Detached>(point, query_mask)
            : edge_bvh_->template nearest_edge<Detached>(point, query_mask);
    const MaskDetached valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const IntDetached global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_id
            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(global_edge_id_detached);
        const Vector3f p0 = gather<Vector3f>(edge_info_.start, global_edge_id, valid);
        const Vector3f e1 = gather<Vector3f>(edge_info_.edge, global_edge_id, valid);
        const Mask is_boundary = gather<Mask>(edge_info_.is_boundary, global_edge_id, valid);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<false>(point, p0, e1);

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.point = select(valid, point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, Int(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, Int(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3fDetached p0 =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3fDetached e1 =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const MaskDetached is_boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        FloatDetached edge_t;
        Vector3fDetached edge_point;
        FloatDetached distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<true>(point, p0, e1);

        result.distance = select(valid_detached, sqrt(distance_sq), result.distance);
        result.point = select(valid_detached, point, result.point);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_point = select(valid_detached, edge_point, result.edge_point);
        result.shape_id = select(valid_detached, shape_id_detached, result.shape_id);
        result.edge_id = select(valid_detached, edge_id_detached, result.edge_id);
        result.global_edge_id = select(valid_detached, global_edge_id_detached, result.global_edge_id);
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> Scene::nearest_edge(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(ray): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(ray): scene has pending updates. Call Scene::sync() first.");

    const int query_count = static_cast<int>(slices(ray.o));
    NearestRayEdgeT<Detached> result = initialize_nearest_ray_edge_result<Detached>(query_count);
    if (edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    FloatDetached t_max_input;
    MaskDetached active_detached;
    if constexpr (!Detached) {
        t_max_input = detach<false>(ray.tmax);
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) &&
                           drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) &&
                           drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &= squared_norm(Vector3fDetached(detach<false>(ray.d.x()),
                                                        detach<false>(ray.d.y()),
                                                        detach<false>(ray.d.z()))) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active &= Mask(active_detached);
    } else {
        t_max_input = ray.tmax;
        active_detached = active;
        active_detached &= drjit::isfinite(ray.o.x()) && drjit::isfinite(ray.o.y()) && drjit::isfinite(ray.o.z());
        active_detached &= drjit::isfinite(ray.d.x()) && drjit::isfinite(ray.d.y()) && drjit::isfinite(ray.d.z());
        active_detached &= squared_norm(ray.d) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_ray(edge_bvh_backend_);
    ClosestEdgeCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edge<Detached>(ray, query_mask)
            : edge_bvh_->template nearest_edge<Detached>(ray, query_mask);
    const MaskDetached valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const MaskDetached finite_tmax = drjit::isfinite(t_max_input);
    const IntDetached global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_id
            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(global_edge_id_detached);
        const Vector3f p0 = gather<Vector3f>(edge_info_.start, global_edge_id, valid);
        const Vector3f e1 = gather<Vector3f>(edge_info_.edge, global_edge_id, valid);
        const Mask is_boundary = gather<Mask>(edge_info_.is_boundary, global_edge_id, valid);

        const Mask finite_mask = valid && Mask(finite_tmax);
        const Mask infinite_mask = valid && !Mask(finite_tmax);
        const Float safe_tmax = select(finite_mask, Float(t_max_input), zeros<Float>(query_count));

        Float query_t = zeros<Float>(query_count);
        Vector3f query_point = zeros<Vector3f>(query_count);
        Float edge_t = zeros<Float>(query_count);
        Vector3f edge_point = zeros<Vector3f>(query_count);
        Float distance_sq = full<Float>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            Float segment_query_t;
            Vector3f segment_query_point;
            Float segment_edge_t;
            Vector3f segment_edge_point;
            Float segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<false>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            Float ray_query_t;
            Vector3f ray_query_point;
            Float ray_edge_t;
            Vector3f ray_edge_point;
            Float ray_distance_sq;
            std::tie(ray_query_t, ray_query_point, ray_edge_t, ray_edge_point, ray_distance_sq) =
                closest_ray_segment<false>(ray.o, ray.d, p0, e1);

            query_t = select(infinite_mask, ray_query_t, query_t);
            query_point = select(infinite_mask, ray_query_point, query_point);
            edge_t = select(infinite_mask, ray_edge_t, edge_t);
            edge_point = select(infinite_mask, ray_edge_point, edge_point);
            distance_sq = select(infinite_mask, ray_distance_sq, distance_sq);
        }

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.ray_t = select(valid, query_t, result.ray_t);
        result.point = select(valid, query_point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, Int(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, Int(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3fDetached p0 =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3fDetached e1 =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const MaskDetached is_boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        const MaskDetached finite_mask = valid_detached && finite_tmax;
        const MaskDetached infinite_mask = valid_detached && !finite_tmax;
        const FloatDetached safe_tmax = select(finite_mask, t_max_input, zeros<FloatDetached>(query_count));

        FloatDetached query_t = zeros<FloatDetached>(query_count);
        Vector3fDetached query_point = zeros<Vector3fDetached>(query_count);
        FloatDetached edge_t = zeros<FloatDetached>(query_count);
        Vector3fDetached edge_point = zeros<Vector3fDetached>(query_count);
        FloatDetached distance_sq = full<FloatDetached>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            FloatDetached segment_query_t;
            Vector3fDetached segment_query_point;
            FloatDetached segment_edge_t;
            Vector3fDetached segment_edge_point;
            FloatDetached segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<true>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            FloatDetached ray_query_t;
            Vector3fDetached ray_query_point;
            FloatDetached ray_edge_t;
            Vector3fDetached ray_edge_point;
            FloatDetached ray_distance_sq;
            std::tie(ray_query_t, ray_query_point, ray_edge_t, ray_edge_point, ray_distance_sq) =
                closest_ray_segment<true>(ray.o, ray.d, p0, e1);

            query_t = select(infinite_mask, ray_query_t, query_t);
            query_point = select(infinite_mask, ray_query_point, query_point);
            edge_t = select(infinite_mask, ray_edge_t, edge_t);
            edge_point = select(infinite_mask, ray_edge_point, edge_point);
            distance_sq = select(infinite_mask, ray_distance_sq, distance_sq);
        }

        result.distance = select(valid_detached, sqrt(distance_sq), result.distance);
        result.ray_t = select(valid_detached, query_t, result.ray_t);
        result.point = select(valid_detached, query_point, result.point);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_point = select(valid_detached, edge_point, result.edge_point);
        result.shape_id = select(valid_detached, shape_id_detached, result.shape_id);
        result.edge_id = select(valid_detached, edge_id_detached, result.edge_id);
        result.global_edge_id = select(valid_detached, global_edge_id_detached, result.global_edge_id);
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template <bool Detached>
NearestEdgesTopKT<Detached> Scene::nearest_edges_topk(const Vector3fT<Detached> &point,
                                                       int k,
                                                       MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edges_topk(point): scene is not built.");
    require(!pending_updates_,
            "Scene::nearest_edges_topk(point): scene has pending updates. Call Scene::sync() first.");
    require(k > 0, "Scene::nearest_edges_topk(point): k must be positive.");
    require(k <= 16, "Scene::nearest_edges_topk(point): k must be <= 16.");

    const int query_count = static_cast<int>(slices(point));
    const int output_count = query_count * k;
    NearestEdgesTopKT<Detached> result;
    result.query_count = query_count;
    result.k = k;
    result.is_valid = full<MaskT<Detached>>(false, output_count);
    result.distances = full<FloatT<Detached>>(Infinity, output_count);
    result.points = zeros<Vector3fT<Detached>>(output_count);
    result.edge_t = zeros<FloatT<Detached>>(output_count);
    result.edge_points = zeros<Vector3fT<Detached>>(output_count);
    result.shape_ids = full<IntT<Detached>>(-1, output_count);
    result.edge_ids = full<IntT<Detached>>(-1, output_count);
    result.global_edge_ids = full<IntT<Detached>>(-1, output_count);
    result.is_boundary = full<MaskT<Detached>>(false, output_count);
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    MaskDetached active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x())) &&
                           drjit::isfinite(detach<false>(point.y())) &&
                           drjit::isfinite(detach<false>(point.z()));
        active &= Mask(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) &&
                           drjit::isfinite(point.y()) &&
                           drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_topk(edge_bvh_backend_);
    const ClosestEdgeTopKCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edges_topk<Detached>(point, k, query_mask)
            : edge_bvh_->template nearest_edges_topk<Detached>(point, k, query_mask);
    const MaskDetached valid_detached = candidate.is_valid;
    if (drjit::none(valid_detached)) {
        return result;
    }

    const IntDetached output_index = arange<IntDetached>(output_count);
    const IntDetached output_query = output_index / k;
    const IntDetached global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_ids
            : edge_bvh_->map_to_global(candidate.global_edge_ids, valid_detached);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(global_edge_id_detached);
        const Int query_id = Int(output_query);
        const Vector3f output_point = gather<Vector3f>(point, query_id, valid);
        const Vector3f edge_start =
            gather<Vector3f>(edge_info_.start, global_edge_id, valid);
        const Vector3f edge_vector =
            gather<Vector3f>(edge_info_.edge, global_edge_id, valid);
        const Mask boundary =
            gather<Mask>(edge_info_.is_boundary, global_edge_id, valid);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<false>(output_point, edge_start, edge_vector);

        result.is_valid = valid;
        result.distances = select(valid, sqrt(distance_sq), result.distances);
        result.points = select(valid, output_point, result.points);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_points = select(valid, edge_point, result.edge_points);
        result.shape_ids = select(valid, Int(shape_id_detached), result.shape_ids);
        result.edge_ids = select(valid, Int(edge_id_detached), result.edge_ids);
        result.global_edge_ids = select(valid, global_edge_id, result.global_edge_ids);
        result.is_boundary = select(valid, boundary, result.is_boundary);
    } else {
        const Vector3fDetached output_point =
            gather<Vector3fDetached>(point, output_query, valid_detached);
        const Vector3fDetached edge_start =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3fDetached edge_vector =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const MaskDetached boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        FloatDetached edge_t;
        Vector3fDetached edge_point;
        FloatDetached distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<true>(output_point, edge_start, edge_vector);

        result.is_valid = valid_detached;
        result.distances = select(valid_detached, sqrt(distance_sq), result.distances);
        result.points = select(valid_detached, output_point, result.points);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_points = select(valid_detached, edge_point, result.edge_points);
        result.shape_ids = select(valid_detached, shape_id_detached, result.shape_ids);
        result.edge_ids = select(valid_detached, edge_id_detached, result.edge_ids);
        result.global_edge_ids = select(valid_detached, global_edge_id_detached, result.global_edge_ids);
        result.is_boundary = select(valid_detached, boundary, result.is_boundary);
    }
    return result;
}

template NearestPointEdgeDetached Scene::nearest_edge<true>(const Vector3fDetached &point, MaskDetached active) const;

template NearestPointEdge Scene::nearest_edge<false>(const Vector3f &point, Mask active) const;

template NearestRayEdgeDetached Scene::nearest_edge<true>(const RayDetached &ray, MaskDetached active) const;

template NearestRayEdge Scene::nearest_edge<false>(const Ray &ray, Mask active) const;

template NearestEdgesTopKDetached Scene::nearest_edges_topk<true>(
    const Vector3fDetached &point,
    int k,
    MaskDetached active) const;

template NearestEdgesTopK Scene::nearest_edges_topk<false>(
    const Vector3f &point,
    int k,
    Mask active) const;

} // namespace rayd
