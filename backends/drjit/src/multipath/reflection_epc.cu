#include <rayd/multipath/reflection_epc_params.h>
#include <rayd/shared/optix/reflection_epc_device.cuh>

namespace rayd::shared::optix {

extern "C" {
__constant__ ReflEpcParams params;
}

namespace {

struct DrJitEpcPolicy {
    static constexpr bool DisableAnyHitWithoutIgnore = false;
};

} // namespace

extern "C" __global__ void __anyhit__reflection_epc() {
    reflection_epc_device::anyhit();
}

extern "C" __global__ void __closesthit__reflection_epc() {
    reflection_epc_device::closesthit();
}

extern "C" __global__ void __miss__reflection_epc() {
    reflection_epc_device::miss();
}

extern "C" __global__ void __raygen__reflection_epc() {
    reflection_epc_device::run_reflection_epc_raygen<DrJitEpcPolicy, false, false>();
}

extern "C" __global__ void __raygen__reflection_epc_direct() {
    reflection_epc_device::run_reflection_epc_raygen<DrJitEpcPolicy, true, false>();
}

extern "C" __global__ void __raygen__reflection_epc_direct_primary() {
    reflection_epc_device::run_reflection_epc_raygen<DrJitEpcPolicy, true, true>();
}

} // namespace rayd::shared::optix
