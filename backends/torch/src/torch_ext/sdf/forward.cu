#include <rayd/torch/sdf/kernels.h>
#include <rayd/torch/sdf/device_math.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <limits>

// ADR-0037 primal: one detached relaxed sphere trace per ray, producing the
// public result and the frozen-winner tape the derivative passes consume. No
// atomics, no device-to-host read, no stream synchronization.

namespace rayd::torch_backend {

namespace {

namespace sm = sdf_math;

// ADR-0037 section 5: the `operations.json` `distance` miss sentinel, and the
// only non-finite value any output of this operation may carry. It is spelled
// locally rather than taken from `rt/numeric_policy.h`, which is inside a
// committed-PTX include closure that this device code must stay out of
// (ADR-0037 section 9).
constexpr float kMissDistance = std::numeric_limits<float>::infinity();

__global__ void sdf_intersect_forward_kernel(
    const float *__restrict__ values,
    int nx,
    int ny,
    int nz,
    const float *__restrict__ box_position,
    const float *__restrict__ box_rotation,
    const float *__restrict__ box_scale,
    const float *__restrict__ origins,
    const float *__restrict__ directions,
    float tmax,
    int max_steps,
    float relaxation,
    float eps_hit_request,
    int64_t ray_count,
    float *__restrict__ out_t,
    bool *__restrict__ out_hit,
    float *__restrict__ out_position,
    float *__restrict__ out_normal,
    int *__restrict__ out_steps,
    float *__restrict__ out_tape_t,
    int *__restrict__ out_tape_base) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= ray_count)
        return;

    // Every lane writes its whole row exactly once, so a missed lane is bitwise
    // inert by construction rather than by a later fixup (ADR-0037 section 5).
    out_t[ray] = kMissDistance;
    out_hit[ray] = false;
    out_steps[ray] = 0;
    out_tape_t[ray] = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        out_position[ray * 3 + axis] = 0.0f;
        out_normal[ray * 3 + axis] = 0.0f;
        out_tape_base[ray * 3 + axis] = 0;
    }

    const sm::Lane lane = sm::make_lane(
        box_position,
        box_rotation,
        box_scale,
        sm::vmath::make_vec3(origins[ray * 3 + 0], origins[ray * 3 + 1], origins[ray * 3 + 2]),
        sm::vmath::make_vec3(
            directions[ray * 3 + 0], directions[ray * 3 + 1], directions[ray * 3 + 2]),
        sm::core::GridExtent{nx, ny, nz});
    if (!lane.usable)
        return;

    const sm::core::Interval interval =
        sm::core::clip_ray_to_box(lane.local_origin, lane.local_direction, lane.scale, tmax);
    if (!interval.valid)
        return;

    sm::GridSampler sampler = sm::make_sampler(values, lane);
    sm::core::MarchConfig config{
        interval.t_lo,
        interval.t_hi,
        sm::core::resolve_eps_hit(eps_hit_request, lane.scale, lane.cells),
        relaxation,
        max_steps,
    };
    const sm::core::MarchResult march = sm::core::sphere_trace(sampler, config);
    out_steps[ray] = march.steps;
    if (!march.hit)
        return;

    const sm::FrozenHit hit = sm::evaluate_frozen(values, lane, sampler.base, march.t);
    out_t[ray] = march.t;
    out_hit[ray] = true;
    out_tape_t[ray] = march.t;
    out_tape_base[ray * 3 + 0] = sampler.base.i;
    out_tape_base[ray * 3 + 1] = sampler.base.j;
    out_tape_base[ray * 3 + 2] = sampler.base.k;
    out_position[ray * 3 + 0] = hit.world_point.x;
    out_position[ray * 3 + 1] = hit.world_point.y;
    out_position[ray * 3 + 2] = hit.world_point.z;
    out_normal[ray * 3 + 0] = hit.normal.x;
    out_normal[ray * 3 + 1] = hit.normal.y;
    out_normal[ray * 3 + 2] = hit.normal.z;
}

} // namespace

SdfIntersectForwardOutputs sdf_intersect_forward_cuda(
    const SdfGridTensors &grid,
    const at::Tensor &origins,
    const at::Tensor &directions,
    const SdfTraceParams &params) {
    const int64_t ray_count = origins.size(0);
    const auto float_options = origins.options();
    SdfIntersectForwardOutputs out;
    out.t = at::empty({ray_count}, float_options);
    out.hit_mask = at::empty({ray_count}, float_options.dtype(at::kBool));
    out.hit_position = at::empty({ray_count, 3}, float_options);
    out.normal = at::empty({ray_count, 3}, float_options);
    out.steps = at::empty({ray_count}, float_options.dtype(at::kInt));
    out.tape_t = at::empty({ray_count}, float_options);
    out.tape_base = at::empty({ray_count, 3}, float_options.dtype(at::kInt));
    if (ray_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(origins.get_device()).stream();
    sdf_intersect_forward_kernel<<<blocks, threads, 0, stream>>>(
        grid.values.data_ptr<float>(),
        static_cast<int>(grid.values.size(0)),
        static_cast<int>(grid.values.size(1)),
        static_cast<int>(grid.values.size(2)),
        grid.position.data_ptr<float>(),
        grid.rotation.data_ptr<float>(),
        grid.scale.data_ptr<float>(),
        origins.data_ptr<float>(),
        directions.data_ptr<float>(),
        static_cast<float>(params.tmax),
        static_cast<int>(params.max_steps),
        static_cast<float>(params.relaxation),
        static_cast<float>(params.eps_hit),
        ray_count,
        out.t.data_ptr<float>(),
        out.hit_mask.data_ptr<bool>(),
        out.hit_position.data_ptr<float>(),
        out.normal.data_ptr<float>(),
        out.steps.data_ptr<int>(),
        out.tape_t.data_ptr<float>(),
        out.tape_base.data_ptr<int>());
    return out;
}

} // namespace rayd::torch_backend
