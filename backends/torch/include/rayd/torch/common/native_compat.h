#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

inline void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

// Device-explicit stream accessor. Callers pass the device that owns the
// tensors the launch reads and writes (the scene device, or the device of the
// launch's own tensors); this is the form new call sites must use.
inline void *jit_cuda_stream(int device_index) {
    return at::cuda::getCurrentCUDAStream(device_index).stream();
}

// Ambient-device stream accessor, retained for the pointer/params-based launch
// helpers that carry no device index. It is correct only while the calling op
// entry holds a device guard on the owning device.
inline void *jit_cuda_stream() {
    return jit_cuda_stream(at::cuda::current_device());
}

inline void audit_cuda_kernel_launch(
    const char *,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint64_t) {}

inline void audit_cuda_memset_async() {}
inline void audit_cuda_memcpy_async() {}
inline void audit_cuda_stream_synchronize() {}
inline void audit_cub_sort() {}
inline void audit_cub_scan() {}

} // namespace rayd::torch_backend
