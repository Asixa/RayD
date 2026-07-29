// Copyright Xingyu Chen.
// Implements visibility support for visibility optix Dr.Jit.

#define RAYD_OPTIX_DEVICE_PROGRAM 1

#include <src/visibility/segment_params_jit.h>
#include <src/visibility/segment_visibility.cuh>

namespace rayd {

extern "C" {
__constant__ SegmentVisibilityParams params;
}

using SegmentVisibilityPolicy = shared::optix::SegmentVisibilityDevicePolicy<false, false>;

extern "C" __global__ void __anyhit__segment_visibility() {
    shared::optix::segment_visibility::anyhit(params);
}

extern "C" __global__ void __closesthit__segment_visibility() {
    shared::optix::segment_visibility::closesthit(params);
}

extern "C" __global__ void __miss__segment_visibility() {
    shared::optix::segment_visibility::miss();
}

extern "C" __global__ void __raygen__segment_visibility() {
    shared::optix::segment_visibility::raygen_segment<SegmentVisibilityPolicy>(params);
}

extern "C" __global__ void __raygen__segment_pair_visibility() {
    shared::optix::segment_visibility::raygen_segment_pair<SegmentVisibilityPolicy>(params);
}

extern "C" __global__ void __raygen__axial_edge_visibility() {
    shared::optix::segment_visibility::raygen_axial_edge<SegmentVisibilityPolicy>(params);
}

extern "C" __global__ void __raygen__segment_chain_visibility() {
    shared::optix::segment_visibility::raygen_segment_chain<SegmentVisibilityPolicy>(params);
}

} // namespace rayd
