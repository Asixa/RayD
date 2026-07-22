#pragma once

#include <rayd/shared/optix/reflection_trace_params.h>
#include <rayd/shared/optix/reflection_epc_params.h>
#include <rayd/shared/optix/segment_visibility_params.h>
#include <rayd/torch/reflection/accum_params.h>
#include <rayd/torch/diffraction/paths_params.h>
#include <rayd/torch/diffraction/accum_params.h>

namespace rayd::torch_backend {

struct SceneCache;

enum class CudaVisibilityVariant : int {
    Single = 0,
    Pair = 1,
    AxialEdge = 2,
    Chain = 3,
};

void launch_reflection_trace_cuda(
    const SceneCache &scene,
    const shared::optix::ReflectionTraceParams &params,
    int lane_count);

void launch_segment_visibility_cuda(
    const SceneCache &scene,
    const shared::optix::SegmentVisibilityParams &params,
    CudaVisibilityVariant variant,
    int lane_count);

void launch_reflection_accumulation_cuda(
    const SceneCache &scene,
    const AccumParams &params,
    int lane_count);

void launch_reflection_epc_cuda(
    const SceneCache &scene,
    const shared::optix::ReflEpcParams &params,
    bool direct_only,
    bool primary_visibility_only,
    int lane_count);

void launch_diffraction_paths_cuda(
    const SceneCache &scene,
    const DfrPathParams &params,
    int lane_count);

void launch_diffraction_accumulation_cuda(
    const SceneCache &scene,
    const DfrAccumParams &params,
    int pipeline_variant,
    int lane_count);

} // namespace rayd::torch_backend
