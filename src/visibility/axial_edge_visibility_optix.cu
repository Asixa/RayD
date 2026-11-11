// Copyright Xingyu Chen.
// Implements visibility support for axial edge visibility optix.

#define RAYD_OPTIX_DEVICE_PROGRAM 1

#include <src/visibility/axial_edge_visibility.cuh>

namespace rayd::torch_backend {

extern "C" {
__constant__ AxialEdgeVisibilityParams params;
}

namespace {

using ExactPolicy = shared::optix::SegmentVisibilityDevicePolicy<true, false>;
using ExactTraverser = shared::optix::segment_visibility::SegmentVisibilityOptixTraverser<true>;
using ExactConfig = shared::rt::TraceConfig<ExactPolicy, ExactTraverser>;

} // namespace

extern "C" __global__ void __closesthit__axial_edge_visibility_exact() {
    optixSetPayload_0(0u);
    optixSetPayload_1(0xFFFFFFFFu);
}

extern "C" __global__ void __miss__axial_edge_visibility_exact() {}

extern "C" __global__ void __raygen__axial_edge_visibility_exact() {
    const ExactTraverser traverser{static_cast<::OptixTraversableHandle>(params.trace.handle), nullptr};
    axial_edge_visibility::exact_state<ExactConfig>(params, optixGetLaunchIndex().x, traverser);
}

} // namespace rayd::torch_backend
