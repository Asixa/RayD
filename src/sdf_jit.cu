// Copyright Xingyu Chen.
// Implements the Dr.Jit SDF forward CUDA launcher.

#include <src/sdf_jit.h>
#include <src/sdf_device.cuh>

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace rayd {

namespace {

namespace sm = shared::sdf_math;

constexpr float kMissDistance = std::numeric_limits<float>::infinity();

__global__ void sdf_intersect_jit_kernel(SdfJitLaunchParams params) {
    const int ray = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= params.ray_count)
        return;

    params.out_t[ray] = kMissDistance;
    params.out_hit[ray] = 0u;
    params.out_position_x[ray] = 0.0f;
    params.out_position_y[ray] = 0.0f;
    params.out_position_z[ray] = 0.0f;
    params.out_normal_x[ray] = 0.0f;
    params.out_normal_y[ray] = 0.0f;
    params.out_normal_z[ray] = 0.0f;
    params.out_steps[ray] = 0;
    params.out_base_x[ray] = 0;
    params.out_base_y[ray] = 0;
    params.out_base_z[ray] = 0;
    params.out_denominator[ray] = 1.0f;

    if (params.active[ray] == 0u)
        return;

    const float position[3] = {params.position_x[0], params.position_y[0], params.position_z[0]};
    const float rotation[4] = {params.rotation[0], params.rotation[1], params.rotation[2], params.rotation[3]};
    const float scale[3] = {params.scale_x[0], params.scale_y[0], params.scale_z[0]};
    const sm::Vec3f origin = sm::vmath::make_vec3(params.origin_x[ray], params.origin_y[ray], params.origin_z[ray]);
    const sm::Vec3f direction =
        sm::vmath::make_vec3(params.direction_x[ray], params.direction_y[ray], params.direction_z[ray]);
    const sm::Lane lane = sm::make_lane(position, rotation, scale, origin, direction,
                                        sm::core::GridExtent{params.nx, params.ny, params.nz});
    const sm::ForwardResult result = sm::intersect_forward(params.values, lane, params.ray_tmax[ray], params.max_steps,
                                                           params.relaxation, params.eps_hit);
    params.out_steps[ray] = result.steps;
    if (!result.hit)
        return;

    params.out_t[ray] = result.t;
    params.out_hit[ray] = 1u;
    params.out_position_x[ray] = result.position.x;
    params.out_position_y[ray] = result.position.y;
    params.out_position_z[ray] = result.position.z;
    params.out_normal_x[ray] = result.normal.x;
    params.out_normal_y[ray] = result.normal.y;
    params.out_normal_z[ray] = result.normal.z;
    params.out_base_x[ray] = result.base.i;
    params.out_base_y[ray] = result.base.j;
    params.out_base_z[ray] = result.base.k;
    params.out_denominator[ray] = result.denominator;
}

void check_cuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(result));
}

} // namespace

void launch_sdf_intersect_jit(const SdfJitLaunchParams& params, void* stream) {
    if (params.ray_count == 0)
        return;
    constexpr int threads = 128;
    const int blocks = (params.ray_count + threads - 1) / threads;
    sdf_intersect_jit_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(params);
    check_cuda(cudaGetLastError(), "launch_sdf_intersect_jit(): kernel launch failed");
}

} // namespace rayd
