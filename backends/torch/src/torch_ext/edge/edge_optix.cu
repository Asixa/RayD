#include <optix.h>
#include <optix_device.h>

#include <rayd/torch/common/math.cuh>
#include <rayd/torch/edge/optix_params.h>
#include <rayd/shared/contracts.h>
#include <rayd/shared/edge/edge_distance_math.h>
#include <rayd/shared/optix/scene_edge_device.cuh>

namespace rayd::torch_backend {

extern "C" {
__constant__ EdgeOptixQueryParams params;
}

namespace {

constexpr float kInfiniteRayTMax = 1.0e8f;
constexpr float kPointProbeTMax = shared::EdgeEpsilon;
constexpr uint32_t kInvalidEdgeId = shared::InvalidUnsignedId;

static __forceinline__ __device__ shared::optix::EdgeGeometrySoAView edge_geometry_view() {
    return { params.edge_p0_x, params.edge_p0_y, params.edge_p0_z,
             params.edge_e1_x, params.edge_e1_y, params.edge_e1_z,
             params.edge_mask, params.edge_count, params.search_radius };
}

static __forceinline__ __device__ shared::optix::EdgeQuerySoAView edge_query_view() {
    return { params.query_x, params.query_y, params.query_z,
             params.ray_dx, params.ray_dy, params.ray_dz, params.ray_tmax,
             params.active_mask, params.query_count, params.k };
}

static __forceinline__ __device__ shared::optix::EdgeQueryOutputView edge_output_view() {
    return { params.out_edge_ids, params.out_distance_sq, params.out_ray_t,
             params.out_edge_t, params.out_valid };
}

static __forceinline__ __device__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

static __forceinline__ __device__ bool is_active(unsigned int query) {
    return shared::optix::edge_query_active(edge_query_view(), query);
}

static __forceinline__ __device__ bool edge_visible(unsigned int edge) {
    return shared::optix::edge_geometry_active(edge_geometry_view(), edge);
}

static __forceinline__ __device__ float3 load_query_point(unsigned int query) {
    return shared::optix::load_edge_query_origin(edge_query_view(), query);
}

static __forceinline__ __device__ float3 load_ray_direction(unsigned int query) {
    return shared::optix::load_edge_query_direction(edge_query_view(), query);
}

static __forceinline__ __device__ float3 load_edge_start(unsigned int edge) {
    return shared::optix::load_edge_start(edge_geometry_view(), edge);
}

static __forceinline__ __device__ float3 load_edge_vector(unsigned int edge) {
    return shared::optix::load_edge_vector(edge_geometry_view(), edge);
}

static __forceinline__ __device__ float safe_search_radius() {
    return shared::optix::safe_edge_search_radius(edge_geometry_view());
}

static __forceinline__ __device__ int active_tier_count() {
    if (params.tier_count <= 0) {
        return 1;
    }
    return params.tier_count < EdgeOptixMaxTiers ? params.tier_count : EdgeOptixMaxTiers;
}

static __forceinline__ __device__ uint64_t tier_handle(int tier) {
    return params.tier_count > 0 ? params.tier_handles[tier] : params.handle;
}

static __forceinline__ __device__ float tier_search_radius(int tier) {
    return params.tier_count > 0 ? fmaxf(params.tier_search_radii[tier], 0.0f)
                                 : safe_search_radius();
}

static __forceinline__ __device__ shared::math::Vec3f to_shared_vec3(float3 value) {
    return shared::math::make_vec3(value.x, value.y, value.z);
}

static __forceinline__ __device__ void point_segment_distance(float3 point,
                                                              float3 p0,
                                                              float3 e1,
                                                              float &edge_t,
                                                              float &distance_sq) {
    const shared::edge::PointSegmentDistance result =
        shared::edge::point_segment_distance(
            to_shared_vec3(point), to_shared_vec3(p0), to_shared_vec3(e1));
    edge_t = result.edge_parameter;
    distance_sq = result.squared_distance;
}

static __forceinline__ __device__ void segment_segment_distance(float3 query_origin,
                                                                float3 query_edge,
                                                                float3 edge_origin,
                                                                float3 edge_vector,
                                                                float &query_t,
                                                                float &edge_t,
                                                                float &distance_sq) {
    const shared::edge::SegmentSegmentDistance result =
        shared::edge::segment_segment_distance(
            to_shared_vec3(query_origin),
            to_shared_vec3(query_edge),
            to_shared_vec3(edge_origin),
            to_shared_vec3(edge_vector));
    query_t = result.query_parameter;
    edge_t = result.edge_parameter;
    distance_sq = result.squared_distance;
}

static __forceinline__ __device__ void write_invalid(unsigned int query) {
    shared::optix::write_invalid_edge_result(edge_output_view(), query);
}

static __forceinline__ __device__ void write_final_point_output(unsigned int query,
                                                                unsigned int edge,
                                                                float distance_sq,
                                                                float edge_t) {
    if (params.write_point_outputs == 0) {
        return;
    }

    const float s = clamp01(edge_t);
    const float3 p = load_query_point(query);
    const float3 a = load_edge_start(edge);
    const float3 e = load_edge_vector(edge);
    const float3 q = a + e * s;
    const float3 d = p - q;

    params.final_distance[query] = sqrtf(fmaxf(distance_sq, 0.0f));
    params.final_edge_t[query] = s;
    params.final_shape_id[query] = params.edge_shape_id[edge];
    params.final_edge_id[query] = params.edge_local_id[edge];
    params.final_global_edge_id[query] = static_cast<int>(edge);
    params.final_edge_point[query * 3 + 0] = q.x;
    params.final_edge_point[query * 3 + 1] = q.y;
    params.final_edge_point[query * 3 + 2] = q.z;
    if (params.final_tape_edge_id != nullptr) {
        params.final_tape_edge_id[query] = static_cast<int>(edge);
    }
    if (params.final_tape_s != nullptr) {
        params.final_tape_s[query] = s;
    }
    if (params.final_tape_d != nullptr) {
        params.final_tape_d[query * 3 + 0] = d.x;
        params.final_tape_d[query * 3 + 1] = d.y;
        params.final_tape_d[query * 3 + 2] = d.z;
    }
    if (params.final_unresolved != nullptr) {
        params.final_unresolved[query] = 0u;
    }
}

#ifndef RAYD_TORCH_EDGE_POINT_RAY_ONLY
static __forceinline__ __device__ void insert_topk_candidate(unsigned int query,
                                                             int edge_id,
                                                             float distance_sq,
                                                             float edge_t) {
    const int k = params.k;
    if (k <= 0 || k > EdgeOptixTopKMax) {
        return;
    }

    const int base = static_cast<int>(query) * k;
    if (distance_sq >= params.out_distance_sq[base + k - 1]) {
        return;
    }

    int insert = k - 1;
    while (insert > 0 && distance_sq < params.out_distance_sq[base + insert - 1]) {
        params.out_distance_sq[base + insert] = params.out_distance_sq[base + insert - 1];
        params.out_edge_ids[base + insert] = params.out_edge_ids[base + insert - 1];
        params.out_edge_t[base + insert] = params.out_edge_t[base + insert - 1];
        params.out_valid[base + insert] = params.out_valid[base + insert - 1];
        --insert;
    }

    params.out_distance_sq[base + insert] = distance_sq;
    params.out_edge_ids[base + insert] = edge_id;
    params.out_edge_t[base + insert] = edge_t;
    params.out_valid[base + insert] = 1u;
}

#endif

} // namespace

// OptiX programs for the custom-AABB edge backend. Each raygen launch handles one
// query (launch index x); intersection programs report the point/segment-to-edge
// distance, and the anyhit/closesthit programs keep the running nearest edge.

/// IntersectionAD for point queries: report the point-to-edge distance if within the search radius.
#ifndef RAYD_TORCH_EDGE_TOPK_ONLY
extern "C" __global__ void __intersection__edge_point() {
    const unsigned int edge = optixGetPrimitiveIndex();
    if (edge >= static_cast<unsigned int>(params.edge_count) || !edge_visible(edge)) {
        return;
    }

    float edge_t = 0.0f;
    float distance_sq = 0.0f;
    point_segment_distance(optixGetWorldRayOrigin(),
                           load_edge_start(edge),
                           load_edge_vector(edge),
                           edge_t,
                           distance_sq);
    const float distance = sqrtf(fmaxf(distance_sq, 0.0f));
    if (distance <= optixGetRayTmax() && distance <= safe_search_radius()) {
        optixReportIntersection(distance, 0u, __float_as_uint(edge_t));
    }
}

/// IntersectionAD for ray queries: report the ray-to-edge closest approach within the search radius.
extern "C" __global__ void __intersection__edge_ray() {
    const unsigned int edge = optixGetPrimitiveIndex();
    if (edge >= static_cast<unsigned int>(params.edge_count) || !edge_visible(edge)) {
        return;
    }

    const float trace_tmax = optixGetRayTmax();
    const float payload_radius = __uint_as_float(optixGetPayload_4());
    const float search_radius = payload_radius > 0.0f ? payload_radius : safe_search_radius();
    float query_t = 0.0f;
    float edge_t = 0.0f;
    float distance_sq = 0.0f;
    segment_segment_distance(optixGetWorldRayOrigin(),
                             optixGetWorldRayDirection() * trace_tmax,
                             load_edge_start(edge),
                             load_edge_vector(edge),
                             query_t,
                             edge_t,
                             distance_sq);
    const float ray_t = query_t * trace_tmax;
    if (ray_t >= optixGetRayTmin() && ray_t <= trace_tmax &&
        sqrtf(fmaxf(distance_sq, 0.0f)) <= search_radius) {
        optixReportIntersection(ray_t,
                                0u,
                                __float_as_uint(distance_sq),
                                __float_as_uint(ray_t),
                                __float_as_uint(edge_t));
    }
}
#endif

/// IntersectionAD for top-k point queries: report every edge within the search radius for ranking.
#ifndef RAYD_TORCH_EDGE_POINT_RAY_ONLY
extern "C" __global__ void __intersection__edge_topk_point() {
    const unsigned int edge = optixGetPrimitiveIndex();
    if (edge >= static_cast<unsigned int>(params.edge_count) || !edge_visible(edge)) {
        return;
    }

    float edge_t = 0.0f;
    float distance_sq = 0.0f;
    point_segment_distance(optixGetWorldRayOrigin(),
                           load_edge_start(edge),
                           load_edge_vector(edge),
                           edge_t,
                           distance_sq);
    const float distance = sqrtf(fmaxf(distance_sq, 0.0f));
    if (distance <= safe_search_radius()) {
        optixReportIntersection(kPointProbeTMax * 0.5f,
                                0u,
                                __float_as_uint(distance_sq),
                                __float_as_uint(edge_t));
    }
}
#endif

/// Closest-hit for point queries: publish the winning edge id, distance, and edge parameter to payload.
#ifndef RAYD_TORCH_EDGE_TOPK_ONLY
extern "C" __global__ void __closesthit__edge_point() {
    const float distance = optixGetRayTmax();
    shared::optix::set_edge_point_payload(
        distance, optixGetPrimitiveIndex(), optixGetAttribute_0());
}

/// Anyhit for ray queries: keep the nearest edge so far in payload, then ignore the hit to continue.
extern "C" __global__ void __anyhit__edge_ray() {
    const float candidate_distance_sq = __uint_as_float(optixGetAttribute_0());
    const float best_distance_sq = __uint_as_float(optixGetPayload_1());
    if (candidate_distance_sq < best_distance_sq) {
        optixSetPayload_0(optixGetPrimitiveIndex());
        optixSetPayload_1(__float_as_uint(candidate_distance_sq));
        optixSetPayload_2(optixGetAttribute_1());
        optixSetPayload_3(optixGetAttribute_2());
    }
    optixIgnoreIntersection();
}
#endif

/// Anyhit for top-k point queries: insert the candidate into the per-query top-k (payload for k<=8,
/// global buffer otherwise), then ignore the hit to keep traversing.
#ifndef RAYD_TORCH_EDGE_POINT_RAY_ONLY
extern "C" __global__ void __anyhit__edge_topk_point() {
    if (params.k <= 8) {
        shared::optix::insert_edge_topk_payload_candidate(
            params.k,
            static_cast<int>(optixGetPrimitiveIndex()),
            __uint_as_float(optixGetAttribute_0()));
    } else {
        insert_topk_candidate(optixGetLaunchIndex().x,
                              static_cast<int>(optixGetPrimitiveIndex()),
                              __uint_as_float(optixGetAttribute_0()),
                              __uint_as_float(optixGetAttribute_1()));
    }
    optixIgnoreIntersection();
}
#endif

/// Miss program: no edge within range; outputs are left at their invalid defaults.
extern "C" __global__ void __miss__edge_query() {
}

/// Raygen for point queries: trace a degenerate ray from each query point and write the nearest edge.
#ifndef RAYD_TORCH_EDGE_TOPK_ONLY
extern "C" __global__ void __raygen__edge_point() {
    const unsigned int query = optixGetLaunchIndex().x;
    if (query >= static_cast<unsigned int>(params.query_count) || !is_active(query) ||
        params.edge_count <= 0) {
        if (params.write_point_outputs == 0) {
            write_invalid(query);
        }
        return;
    }

    const int tiers = active_tier_count();
    for (int tier = 0; tier < tiers; ++tier) {
        const uint64_t handle = tier_handle(tier);
        const float radius = tier_search_radius(tier);
        if (handle == 0ull || !(radius > 0.0f)) {
            continue;
        }

        uint32_t edge_id = kInvalidEdgeId;
        uint32_t distance_sq = __float_as_uint(3.4028234663852886e38f);
        uint32_t edge_t = __float_as_uint(0.0f);
        uint32_t valid = 0u;
        optixTrace(static_cast<OptixTraversableHandle>(handle),
                   load_query_point(query),
                   make_float3(0.0f, 0.0f, -1.0f),
                   0.0f,
                   radius,
                   0.0f,
                   255u,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   0,
                   1,
                   0,
                   edge_id,
                   distance_sq,
                   edge_t,
                   valid);

        if (valid == 0u || edge_id == kInvalidEdgeId) {
            continue;
        }

        if (params.write_point_outputs != 0) {
            write_final_point_output(query,
                                     edge_id,
                                     __uint_as_float(distance_sq),
                                     __uint_as_float(edge_t));
        } else {
            params.out_edge_ids[query] = static_cast<int>(edge_id);
            params.out_distance_sq[query] = __uint_as_float(distance_sq);
            params.out_edge_t[query] = __uint_as_float(edge_t);
            if (params.out_ray_t != nullptr) {
                params.out_ray_t[query] = 0.0f;
            }
            if (params.out_valid != nullptr) {
                params.out_valid[query] = 1u;
            }
        }
        return;
    }

    if (params.write_point_outputs == 0) {
        write_invalid(query);
    }
}

/// Raygen for ray queries: trace each query ray (anyhit-enforced) and write the nearest edge.
extern "C" __global__ void __raygen__edge_ray() {
    const unsigned int query = optixGetLaunchIndex().x;
    if (query >= static_cast<unsigned int>(params.query_count) || !is_active(query) ||
        params.edge_count <= 0) {
        write_invalid(query);
        return;
    }

    float trace_tmax = params.ray_tmax != nullptr ? params.ray_tmax[query] : kInfiniteRayTMax;
    if (!(trace_tmax > 0.0f) || isinf(trace_tmax)) {
        trace_tmax = kInfiniteRayTMax;
    }

    const int tiers = active_tier_count();
    for (int tier = 0; tier < tiers; ++tier) {
        const uint64_t handle = tier_handle(tier);
        const float radius = tier_search_radius(tier);
        if (handle == 0ull || !(radius > 0.0f)) {
            continue;
        }

        uint32_t edge_id = kInvalidEdgeId;
        uint32_t distance_sq = __float_as_uint(3.4028234663852886e38f);
        uint32_t ray_t = __float_as_uint(0.0f);
        uint32_t edge_t = __float_as_uint(0.0f);
        uint32_t radius_bits = __float_as_uint(radius);
        optixTrace(static_cast<OptixTraversableHandle>(handle),
                   load_query_point(query),
                   load_ray_direction(query),
                   0.0f,
                   trace_tmax,
                   0.0f,
                   255u,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
                   1,
                   1,
                   0,
                   edge_id,
                   distance_sq,
                   ray_t,
                   edge_t,
                   radius_bits);

        if (edge_id == kInvalidEdgeId) {
            continue;
        }

        params.out_edge_ids[query] = static_cast<int>(edge_id);
        params.out_distance_sq[query] = __uint_as_float(distance_sq);
        params.out_ray_t[query] = __uint_as_float(ray_t);
        params.out_edge_t[query] = __uint_as_float(edge_t);
        if (params.out_valid != nullptr) {
            params.out_valid[query] = 1u;
        }
        return;
    }

    write_invalid(query);
}
#endif

/// Raygen for top-k point queries: initialize the per-query top-k slots, trace, and emit the sorted neighbors.
#ifndef RAYD_TORCH_EDGE_POINT_RAY_ONLY
extern "C" __global__ void __raygen__edge_topk_point() {
    const unsigned int query = optixGetLaunchIndex().x;
    const int k = params.k;
    if (query >= static_cast<unsigned int>(params.query_count) || k <= 0 || k > EdgeOptixTopKMax) {
        return;
    }

    const int base = static_cast<int>(query) * k;
    for (int slot = 0; slot < EdgeOptixTopKMax; ++slot) {
        if (slot < k) {
            params.out_edge_ids[base + slot] = -1;
            params.out_distance_sq[base + slot] = 3.4028234663852886e38f;
            params.out_edge_t[base + slot] = 0.0f;
            params.out_valid[base + slot] = 0u;
        }
    }

    if (!is_active(query) || params.handle == 0ull || params.edge_count <= 0) {
        return;
    }

    if (k <= 8) {
        uint32_t edge0 = kInvalidEdgeId;
        uint32_t edge1 = kInvalidEdgeId;
        uint32_t edge2 = kInvalidEdgeId;
        uint32_t edge3 = kInvalidEdgeId;
        uint32_t edge4 = kInvalidEdgeId;
        uint32_t edge5 = kInvalidEdgeId;
        uint32_t edge6 = kInvalidEdgeId;
        uint32_t edge7 = kInvalidEdgeId;
        uint32_t dist0 = __float_as_uint(3.4028234663852886e38f);
        uint32_t dist1 = dist0;
        uint32_t dist2 = dist0;
        uint32_t dist3 = dist0;
        uint32_t dist4 = dist0;
        uint32_t dist5 = dist0;
        uint32_t dist6 = dist0;
        uint32_t dist7 = dist0;

        optixTrace(static_cast<OptixTraversableHandle>(params.handle),
                   load_query_point(query),
                   make_float3(0.0f, 0.0f, -1.0f),
                   0.0f,
                   kPointProbeTMax,
                   0.0f,
                   255u,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
                   0,
                   1,
                   0,
                   edge0,
                   edge1,
                   edge2,
                   edge3,
                   edge4,
                   edge5,
                   edge6,
                   edge7,
                   dist0,
                   dist1,
                   dist2,
                   dist3,
                   dist4,
                   dist5,
                   dist6,
                   dist7);

        uint32_t edges[8] = { edge0, edge1, edge2, edge3, edge4, edge5, edge6, edge7 };
        uint32_t distances[8] = { dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7 };
        for (int slot = 0; slot < k; ++slot) {
            const int out_index = base + slot;
            const bool valid = edges[slot] != kInvalidEdgeId;
            params.out_edge_ids[out_index] = valid ? static_cast<int>(edges[slot]) : -1;
            params.out_distance_sq[out_index] = valid
                ? __uint_as_float(distances[slot])
                : 3.4028234663852886e38f;
            params.out_edge_t[out_index] = 0.0f;
            params.out_valid[out_index] = valid ? 1u : 0u;
        }
    } else {
        uint32_t dummy = 0u;
        optixTrace(static_cast<OptixTraversableHandle>(params.handle),
                   load_query_point(query),
                   make_float3(0.0f, 0.0f, -1.0f),
                   0.0f,
                   kPointProbeTMax,
                   0.0f,
                   255u,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
                   0,
                   1,
                   0,
                   dummy);
    }
}
#endif

} // namespace rayd::torch_backend
