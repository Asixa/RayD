#include <raydtorch/optix_context.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace raydtorch {

namespace {
std::mutex context_mutex;
std::unordered_map<int, OptixDeviceContextEntry> contexts;
} // namespace

TorchCudaContext current_torch_cuda_context() {
    TorchCudaContext out;
    out.device_index = c10::cuda::current_device();
    out.stream = at::cuda::getCurrentCUDAStream(out.device_index).stream();
    return out;
}

OptixDeviceContextEntry &get_optix_context(int device_index) {
    std::lock_guard<std::mutex> lock(context_mutex);
    auto it = contexts.find(device_index);
    if (it != contexts.end())
        return it->second;

    c10::cuda::CUDAGuard guard(device_index);
    CUcontext cu_ctx = nullptr;
    CUresult cu_result = cuCtxGetCurrent(&cu_ctx);
    if (cu_result != CUDA_SUCCESS || cu_ctx == nullptr)
        throw std::runtime_error("Could not get current CUDA context for OptiX.");

    OptixDeviceContext optix_ctx = nullptr;
    raydtorch_OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    raydtorch_OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &optix_ctx));

    OptixDeviceContextEntry entry;
    entry.device_index = device_index;
    entry.cuda_context = cu_ctx;
    entry.optix_context = optix_ctx;
    auto [inserted, _] = contexts.emplace(device_index, entry);
    return inserted->second;
}

void optix_check(OptixResult result, const char *expr, const char *file, int line) {
    if (result == OPTIX_SUCCESS)
        return;
    throw std::runtime_error(
        std::string("OptiX error in ") + expr + " at " + file + ":" + std::to_string(line) +
        " code=" + std::to_string(static_cast<int>(result)));
}

} // namespace raydtorch
