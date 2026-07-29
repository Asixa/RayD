// Copyright Xingyu Chen.
// Implements reflection support for epc optix.

#include <src/reflection/epc_params.h>
#include <rayd/reflection/epc_optix_device.cuh>

namespace rayd::shared::optix {

extern "C" {
__constant__ ReflEpcParams params;
}

namespace {

struct TorchEpcPolicy {
    static constexpr bool DisableAnyHitWithoutIgnore = true;
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
    reflection_epc_device::run_reflection_epc_raygen<TorchEpcPolicy, false, false>();
}

extern "C" __global__ void __raygen__reflection_epc_direct() {
    reflection_epc_device::run_reflection_epc_raygen<TorchEpcPolicy, true, false>();
}

extern "C" __global__ void __raygen__reflection_epc_direct_primary() {
    reflection_epc_device::run_reflection_epc_raygen<TorchEpcPolicy, true, true>();
}

} // namespace rayd::shared::optix
