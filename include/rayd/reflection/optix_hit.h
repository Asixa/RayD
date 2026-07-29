// Copyright Xingyu Chen.
// Defines shared reflection support for optix hit.

#pragma once

#ifdef __CUDACC__

#include <optix.h>
#include <optix_device.h>

#include <rayd/rt/optix_primitive_id.h>

namespace rayd::shared::optix {

struct TriangleHitPayload {
    unsigned int hit = 0u;
    unsigned int t = 0u;
    unsigned int bary_u = 0u;
    unsigned int bary_v = 0u;
    unsigned int prim = 0u;
    unsigned int instance = 0u;
};

struct VisibilityPayload {
    unsigned int visible = 1u;
    unsigned int blocker = 0xFFFFFFFFu;
};

static __forceinline__ __device__ void clear_triangle_hit(TriangleHitPayload& payload, float miss_t) {
    payload.hit = 0u;
    payload.t = __float_as_uint(miss_t);
    payload.bary_u = 0u;
    payload.bary_v = 0u;
    payload.prim = 0u;
    payload.instance = 0u;
}

static __forceinline__ __device__ void set_triangle_hit_payload(const TriangleHitPayload& payload) {
    optixSetPayload_0(payload.hit);
    optixSetPayload_1(payload.t);
    optixSetPayload_2(payload.bary_u);
    optixSetPayload_3(payload.bary_v);
    optixSetPayload_4(payload.prim);
    optixSetPayload_5(payload.instance);
}

static __forceinline__ __device__ TriangleHitPayload choose_nearest_hit(const TriangleHitPayload& a,
                                                                        const TriangleHitPayload& b) {
    if (a.hit == 0u)
        return b;
    if (b.hit == 0u)
        return a;
    return __uint_as_float(b.t) < __uint_as_float(a.t) ? b : a;
}

} // namespace rayd::shared::optix

#endif
