#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

#include <ATen/ATen.h>

#include <cstdint>

namespace raydtorch {

struct TorchCudaContext {
    int device_index = 0;
    cudaStream_t stream = nullptr;
};

struct OptixDeviceContextEntry {
    int device_index = 0;
    CUcontext cuda_context = nullptr;
    OptixDeviceContext optix_context = nullptr;
    OptixModule intersect_module = nullptr;
    OptixPipeline intersect_pipeline = nullptr;
    OptixProgramGroup intersect_raygen_group = nullptr;
    OptixProgramGroup intersect_miss_group = nullptr;
    OptixProgramGroup intersect_hitgroup = nullptr;
    OptixShaderBindingTable intersect_sbt = {};
    at::Tensor intersect_raygen_record;
    at::Tensor intersect_miss_record;
    at::Tensor intersect_hitgroup_record;
};

TorchCudaContext current_torch_cuda_context();
OptixDeviceContextEntry &get_optix_context(int device_index);
void ensure_intersect_pipeline(OptixDeviceContextEntry &entry);
void optix_check(OptixResult result, const char *expr, const char *file, int line);

} // namespace raydtorch

#define raydtorch_OPTIX_CHECK(expr) ::raydtorch::optix_check((expr), #expr, __FILE__, __LINE__)
