#include <optix.h>
#include <optix_device.h>

#include <src/reflection/accumulation_params_jit.h>
#include <rayd/detail/reflection/accumulation_optix_device.cuh>

namespace rayd {

namespace shared_accum = shared::multipath::reflection_accumulation;

struct ReflectionAccumulationPolicy {
    static __forceinline__ __device__ bool include_depth(
        const AccumParams &,
        int depth) {
        return depth > 0;
    }

    static __forceinline__ __device__ void commit(
        const AccumParams &params,
        unsigned int,
        int,
        int cell,
        shared::field::Complex3 field,
        float power) {
        atomicAdd(params.out_field_x_re + cell, field.x.r);
        atomicAdd(params.out_field_x_im + cell, field.x.i);
        atomicAdd(params.out_field_y_re + cell, field.y.r);
        atomicAdd(params.out_field_y_im + cell, field.y.i);
        atomicAdd(params.out_field_z_re + cell, field.z.r);
        atomicAdd(params.out_field_z_im + cell, field.z.i);
        atomicAdd(params.out_reflection_power + cell, power);
        atomicAdd(params.out_reflection_count, 1);
    }
};

extern "C" {
__constant__ AccumParams params;
}

extern "C" __global__ void __closesthit__reflection_accumulation() {
    shared_accum::closest_hit();
}

extern "C" __global__ void __miss__reflection_accumulation() {
    shared_accum::miss();
}

extern "C" __global__ void __raygen__reflection_accumulation() {
    shared_accum::raygen<AccumParams, ReflectionAccumulationPolicy>(params);
}

} // namespace rayd
