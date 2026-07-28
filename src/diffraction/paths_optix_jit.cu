#include <optix.h>
#include <optix_device.h>

#include <rayd/diffraction/drjit/paths.h>
#include <src/diffraction/paths_params_jit.h>
#include <rayd/shared/diffraction/paths_optix_device.cuh>

// Dr.Jit OptiX adapter for first-order diffraction path export. The algorithm
// body now lives in the host-compilable, traverser-templated
// rayd/shared/diffraction/paths_algo.h behind the OptiX shim
// rayd/shared/diffraction/paths_optix_device.cuh; this file only wires the
// backend's constant params block and the extern "C" program entry points.

namespace rayd {

namespace shim = ::rayd::shared::optix::diffraction_paths;

extern "C" {
__constant__ DfrPathParams params;
}

extern "C" __global__ void __closesthit__diffraction_paths() {
    shim::closesthit();
}

extern "C" __global__ void __miss__diffraction_paths() {
    shim::miss();
}

extern "C" __global__ void __raygen__diffraction_paths_order1_primary() {
    shim::raygen_order1<DfrPathParams, false>(params);
}

extern "C" __global__ void __raygen__diffraction_paths_order1() {
    shim::raygen_order1<DfrPathParams, true>(params);
}

extern "C" __global__ void __raygen__diffraction_paths_order1_source_visibility_primary() {
    shim::raygen_source_visibility<DfrPathParams>(params);
}

extern "C" __global__ void __raygen__diffraction_paths_order1_target_export_primary() {
    shim::raygen_target_export<DfrPathParams>(params);
}

} // namespace rayd
