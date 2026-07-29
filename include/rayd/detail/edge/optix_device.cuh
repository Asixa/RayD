#pragma once

#include <optix_device.h>

#include <rayd/detail/edge/edge_distance_math.h>
#include <rayd/detail/edge/optix_contracts.h>

namespace rayd::shared::optix {

static __forceinline__ __device__ float3 edge_load_vec3(const float *x,
                                                        const float *y,
                                                        const float *z,
                                                        unsigned int index) {
    return make_float3(x[index], y[index], z[index]);
}

static __forceinline__ __device__ bool edge_query_active(const EdgeQuerySoAView &view,
                                                         unsigned int query) {
    return view.active == nullptr || view.active[query] != 0u;
}

static __forceinline__ __device__ bool edge_geometry_active(const EdgeGeometrySoAView &view,
                                                            unsigned int edge) {
    return view.active == nullptr || view.active[edge] != 0u;
}

static __forceinline__ __device__ float3 load_edge_query_origin(const EdgeQuerySoAView &view,
                                                                unsigned int query) {
    return edge_load_vec3(view.origin_x, view.origin_y, view.origin_z, query);
}

static __forceinline__ __device__ float3 load_edge_query_direction(const EdgeQuerySoAView &view,
                                                                   unsigned int query) {
    return edge_load_vec3(view.direction_x, view.direction_y, view.direction_z, query);
}

static __forceinline__ __device__ float3 load_edge_start(const EdgeGeometrySoAView &view,
                                                         unsigned int edge) {
    return edge_load_vec3(view.p0_x, view.p0_y, view.p0_z, edge);
}

static __forceinline__ __device__ float3 load_edge_vector(const EdgeGeometrySoAView &view,
                                                          unsigned int edge) {
    return edge_load_vec3(view.e1_x, view.e1_y, view.e1_z, edge);
}

static __forceinline__ __device__ float safe_edge_search_radius(
    const EdgeGeometrySoAView &view) {
    return fmaxf(view.search_radius, 0.0f);
}

static __forceinline__ __device__ void write_invalid_edge_result(
    const EdgeQueryOutputView &output,
    unsigned int query) {
    output.edge_ids[query] = -1;
    output.squared_distance[query] = edge::EdgeDistanceFloatMax;
    if (output.ray_parameter != nullptr) {
        output.ray_parameter[query] = 0.0f;
    }
    output.edge_parameter[query] = 0.0f;
    if (output.valid != nullptr) {
        output.valid[query] = 0u;
    }
}

static __forceinline__ __device__ void set_edge_point_payload(float distance,
                                                              unsigned int edge_id,
                                                              unsigned int edge_parameter) {
    optixSetPayload_0(edge_id);
    optixSetPayload_1(__float_as_uint(distance * distance));
    optixSetPayload_2(edge_parameter);
    optixSetPayload_3(1u);
}

static __forceinline__ __device__ std::uint32_t get_edge_topk_payload_id(int slot) {
    switch (slot) {
    case 0: return optixGetPayload_0();
    case 1: return optixGetPayload_1();
    case 2: return optixGetPayload_2();
    case 3: return optixGetPayload_3();
    case 4: return optixGetPayload_4();
    case 5: return optixGetPayload_5();
    case 6: return optixGetPayload_6();
    default: return optixGetPayload_7();
    }
}

static __forceinline__ __device__ std::uint32_t get_edge_topk_payload_distance(int slot) {
    switch (slot) {
    case 0: return optixGetPayload_8();
    case 1: return optixGetPayload_9();
    case 2: return optixGetPayload_10();
    case 3: return optixGetPayload_11();
    case 4: return optixGetPayload_12();
    case 5: return optixGetPayload_13();
    case 6: return optixGetPayload_14();
    default: return optixGetPayload_15();
    }
}

static __forceinline__ __device__ void set_edge_topk_payload_slot(
    int slot,
    std::uint32_t edge_id,
    std::uint32_t squared_distance) {
    switch (slot) {
    case 0: optixSetPayload_0(edge_id); optixSetPayload_8(squared_distance); break;
    case 1: optixSetPayload_1(edge_id); optixSetPayload_9(squared_distance); break;
    case 2: optixSetPayload_2(edge_id); optixSetPayload_10(squared_distance); break;
    case 3: optixSetPayload_3(edge_id); optixSetPayload_11(squared_distance); break;
    case 4: optixSetPayload_4(edge_id); optixSetPayload_12(squared_distance); break;
    case 5: optixSetPayload_5(edge_id); optixSetPayload_13(squared_distance); break;
    case 6: optixSetPayload_6(edge_id); optixSetPayload_14(squared_distance); break;
    default: optixSetPayload_7(edge_id); optixSetPayload_15(squared_distance); break;
    }
}

static __forceinline__ __device__ void insert_edge_topk_payload_candidate(
    int k,
    int edge_id,
    float squared_distance) {
    if (k <= 0 || k > EdgePayloadTopKMax) {
        return;
    }

    const std::uint32_t candidate_distance = __float_as_uint(squared_distance);
    if (squared_distance >= __uint_as_float(get_edge_topk_payload_distance(k - 1))) {
        return;
    }

    int insert = k - 1;
    while (insert > 0 &&
           squared_distance < __uint_as_float(get_edge_topk_payload_distance(insert - 1))) {
        set_edge_topk_payload_slot(insert,
                                   get_edge_topk_payload_id(insert - 1),
                                   get_edge_topk_payload_distance(insert - 1));
        --insert;
    }
    set_edge_topk_payload_slot(insert,
                               static_cast<std::uint32_t>(edge_id),
                               candidate_distance);
}

} // namespace rayd::shared::optix
