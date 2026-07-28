#include <optix.h>
#include <optix_device.h>

#include <src/runtime/math.cuh>
#include <src/reflection/accum_params.h>
#include <rayd/shared/reflection/accumulation_optix_device.cuh>

namespace rayd::torch_backend {

namespace shared_accum = shared::multipath::reflection_accumulation;

struct ReflectionAccumulationPolicy {
    static __forceinline__ __device__ bool include_depth(
        const AccumParams &params,
        int depth) {
        return depth > 0 || params.include_los != 0;
    }

    static __forceinline__ __device__ void commit(
        const AccumParams &params,
        unsigned int ray_index,
        int depth,
        int cell,
        shared::field::Complex3 field,
        float power) {
        if (params.stage_cell != nullptr && params.stage_value != nullptr) {
            const long long stride = static_cast<long long>(params.max_bounces) + 1ll;
            const long long slot =
                static_cast<long long>(ray_index) * stride + static_cast<long long>(depth);
            ReflAccumStagedValue value;
            value.a = make_float4(power, field.x.r, field.x.i, field.y.r);
            value.b = make_float4(field.y.i, field.z.r, field.z.i, 1.0f);
            params.stage_cell[slot] = cell;
            params.stage_value[slot] = value;
            return;
        }

        const WarpCellGroup group = warp_cell_group(cell);
        atomic_add_same_cell(params.out_field_x_re, cell, field.x.r, group);
        atomic_add_same_cell(params.out_field_x_im, cell, field.x.i, group);
        atomic_add_same_cell(params.out_field_y_re, cell, field.y.r, group);
        atomic_add_same_cell(params.out_field_y_im, cell, field.y.i, group);
        atomic_add_same_cell(params.out_field_z_re, cell, field.z.r, group);
        atomic_add_same_cell(params.out_field_z_im, cell, field.z.i, group);
        atomic_add_same_cell(params.out_reflection_power, cell, power, group);
        atomic_add_warp(params.out_reflection_count, 1);
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

} // namespace rayd::torch_backend
