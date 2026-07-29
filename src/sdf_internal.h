// Copyright Xingyu Chen.
// Declares the private Torch SDF launcher boundary.

#pragma once

#include <ATen/ATen.h>

// Declares the standalone Torch launchers for SDF primal and derivative passes.

namespace rayd::torch_backend {

// Caller-owned field and its oriented box. `values` is contiguous CUDA float32
// `[Nx, Ny, Nz]` with `N_i >= 2`; `position` is `[3]`, `rotation` is a
// scalar-first `[4]` quaternion, `scale` is `[3]` full side lengths.
struct SdfGridTensors {
    at::Tensor values;
    at::Tensor position;
    at::Tensor rotation;
    at::Tensor scale;
};

// Host march parameters. None of them is differentiable. A non-positive
// `eps_hit` is the ADR-0037 section 7 sentinel meaning "derive
// `kSdfEpsHitVoxelFraction * h_min` on the device", which is why the operation
// never reads `scale` back to the host.
struct SdfTraceParams {
    double tmax;
    int64_t max_steps;
    double relaxation;
    double eps_hit;
};

// The frozen discrete decisions of one march, and the whole of the tape:
// `t` is the frozen hit distance, `hit` the frozen mask, `base` the frozen base
// voxel index `[N, 3]` int32.
struct SdfTapeTensors {
    at::Tensor t;
    at::Tensor hit;
    at::Tensor base;
};

struct SdfIntersectForwardOutputs {
    at::Tensor t;            // [N] float32, +inf on miss
    at::Tensor hit_mask;     // [N] bool
    at::Tensor hit_position; // [N, 3] float32, +0.0 on miss
    at::Tensor normal;       // [N, 3] float32, +0.0 on miss
    at::Tensor steps;        // [N] int32, field evaluations performed
    at::Tensor tape_t;       // [N] float32
    at::Tensor tape_base;    // [N, 3] int32
};

// Upstream gradients of the three differentiable outputs, plus which inputs the
// caller actually needs a gradient for. A null gradient pointer is an absent
// gradient, not a zero one; an unneeded input gets an undefined tensor back.
struct SdfIntersectGradRequest {
    const at::Tensor* grad_t;
    const at::Tensor* grad_hit_position;
    const at::Tensor* grad_normal;
    bool need_grad_values;
    bool need_grad_position;
    bool need_grad_rotation;
    bool need_grad_scale;
    bool need_grad_origins;
    bool need_grad_directions;
};

struct SdfIntersectBackwardOutputs {
    at::Tensor grad_values;
    at::Tensor grad_position;
    at::Tensor grad_rotation;
    at::Tensor grad_scale;
    at::Tensor grad_origins;
    at::Tensor grad_directions;
};

// Tangents of the six supported inputs. A null pointer is a zero tangent.
struct SdfIntersectTangentInputs {
    const at::Tensor* values;
    const at::Tensor* position;
    const at::Tensor* rotation;
    const at::Tensor* scale;
    const at::Tensor* origins;
    const at::Tensor* directions;
};

struct SdfIntersectJvpOutputs {
    at::Tensor tangent_t;
    at::Tensor tangent_hit_position;
    at::Tensor tangent_normal;
};

SdfIntersectForwardOutputs sdf_intersect_forward_cuda(const SdfGridTensors& grid, const at::Tensor& origins,
                                                      const at::Tensor& directions, const SdfTraceParams& params);

SdfIntersectBackwardOutputs sdf_intersect_backward_cuda(const SdfGridTensors& grid, const at::Tensor& origins,
                                                        const at::Tensor& directions, const SdfTapeTensors& tape,
                                                        const SdfIntersectGradRequest& request);

SdfIntersectJvpOutputs sdf_intersect_jvp_cuda(const SdfGridTensors& grid, const at::Tensor& origins,
                                              const at::Tensor& directions, const SdfTapeTensors& tape,
                                              const SdfIntersectTangentInputs& tangents);

} // namespace rayd::torch_backend
