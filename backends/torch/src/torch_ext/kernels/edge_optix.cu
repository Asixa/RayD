#include <raydtorch/edge_optix_params.h>

#include <optix_device.h>

extern "C" {
__constant__ raydtorch::EdgeOptixQueryParams params;
}

namespace {

__forceinline__ __device__ float3 make_f3(const float *ptr) {
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__forceinline__ __device__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 mul3(float s, float3 a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__forceinline__ __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float point_edge_distance(unsigned int edge_id) {
    const int i0 = params.edge_v0[edge_id];
    const int i1 = params.edge_v1[edge_id];
    const unsigned int query = optixGetLaunchIndex().x;
    const float3 p = make_f3(params.point + query * 3);
    const float3 a = make_f3(params.vertices + i0 * 3);
    const float3 b = make_f3(params.vertices + i1 * 3);
    const float3 ab = sub3(b, a);
    const float denom = fmaxf(dot3(ab, ab), 1.0e-20f);
    float s = dot3(sub3(p, a), ab) / denom;
    s = fminf(1.f, fmaxf(0.f, s));
    const float3 q = add3(a, mul3(s, ab));
    const float3 d = sub3(p, q);
    return sqrtf(fmaxf(dot3(d, d), 0.f));
}

} // namespace

extern "C" __global__ void __intersection__edge_point() {
    const unsigned int edge_id = optixGetPrimitiveIndex();
    if (edge_id >= static_cast<unsigned int>(params.edge_count))
        return;
    const float distance = point_edge_distance(edge_id);
    if (distance <= optixGetRayTmax())
        optixReportIntersection(distance, 0u);
}

extern "C" __global__ void __closesthit__edge_point() {
    optixSetPayload_0(optixGetPrimitiveIndex());
}

extern "C" __global__ void __miss__edge_point() {
}

extern "C" __global__ void __raygen__edge_point() {
    const unsigned int query = optixGetLaunchIndex().x;
    if (query >= static_cast<unsigned int>(params.point_count) || params.traversable == 0 || params.edge_count <= 0) {
        params.out_edge_id[query] = -1;
        return;
    }

    unsigned int edge_id = 0xffffffffu;
    const float3 origin = make_f3(params.point + query * 3);
    optixTrace(
        params.traversable,
        origin,
        make_float3(0.f, 0.f, 1.f),
        0.f,
        params.search_radius,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,
        1,
        0,
        edge_id);
    params.out_edge_id[query] = edge_id == 0xffffffffu ? -1 : static_cast<int>(edge_id);
}
