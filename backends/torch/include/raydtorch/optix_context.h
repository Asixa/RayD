#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

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
};

TorchCudaContext current_torch_cuda_context();
OptixDeviceContextEntry &get_optix_context(int device_index);
void optix_check(OptixResult result, const char *expr, const char *file, int line);

} // namespace raydtorch

#define raydtorch_OPTIX_CHECK(expr) ::raydtorch::optix_check((expr), #expr, __FILE__, __LINE__)
