#include <optix.h>
#include <optix_device.h>

#include <rayd/multipath/reflection_trace_params.h>
#include <rayd/shared/optix/device_hit.h>
#include <rayd/shared/optix/reflection_trace_device.cuh>

namespace rayd {

extern "C" {
__constant__ ReflectionTraceParams params;
}

extern "C" __global__ void __closesthit__reflection() {
    shared::optix::reflection_trace_closest_hit();
}

extern "C" __global__ void __miss__reflection() {
    shared::optix::reflection_trace_miss();
}

extern "C" __global__ void __raygen__reflection_trace() {
    shared::optix::reflection_trace_raygen<shared::optix::DrJitReflectionTracePolicy>(params);
}

} // namespace rayd
