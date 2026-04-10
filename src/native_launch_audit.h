#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rayd {

enum class NativeLaunchStage {
    Unknown = 0,
    Build,
    Sync,
    TraceReflections
};

struct NativeKernelLaunchStat {
    std::string label;
    uint64_t launches = 0;
    uint64_t total_threads = 0;
    uint64_t max_threads = 0;
    uint64_t total_items = 0;
    uint64_t max_items = 0;
};

struct NativeLaunchStageStats {
    uint64_t cuda_kernel_launches = 0;
    uint64_t cuda_kernel_total_threads = 0;
    uint64_t cuda_memcpy = 0;
    uint64_t cuda_memcpy_async = 0;
    uint64_t cuda_memset_async = 0;
    uint64_t cuda_stream_synchronize = 0;
    uint64_t cuda_event_record = 0;
    uint64_t cuda_stream_wait_event = 0;
    uint64_t cub_reduce = 0;
    uint64_t cub_sort = 0;
    uint64_t cub_scan = 0;
    uint64_t jit_memcpy = 0;
    uint64_t jit_memcpy_async = 0;
    uint64_t optix_accel_build = 0;
    uint64_t optix_accel_compact = 0;
    uint64_t optix_launch = 0;
    std::vector<NativeKernelLaunchStat> kernels;
};

struct NativeLaunchAuditSnapshot {
    NativeLaunchStageStats unknown;
    NativeLaunchStageStats build;
    NativeLaunchStageStats sync;
    NativeLaunchStageStats trace_reflections;
};

class ScopedNativeLaunchStage {
public:
    explicit ScopedNativeLaunchStage(NativeLaunchStage stage);
    ~ScopedNativeLaunchStage();

    ScopedNativeLaunchStage(const ScopedNativeLaunchStage &) = delete;
    ScopedNativeLaunchStage &operator=(const ScopedNativeLaunchStage &) = delete;

private:
    NativeLaunchStage previous_;
};

void native_launch_audit_clear();
NativeLaunchAuditSnapshot native_launch_audit_snapshot();

void audit_cuda_kernel_launch(const char *label,
                              uint32_t grid_x,
                              uint32_t grid_y,
                              uint32_t grid_z,
                              uint32_t block_x,
                              uint32_t block_y,
                              uint32_t block_z,
                              uint64_t items = 0);
void audit_cuda_memcpy();
void audit_cuda_memcpy_async();
void audit_cuda_memset_async();
void audit_cuda_stream_synchronize();
void audit_cuda_event_record();
void audit_cuda_stream_wait_event();
void audit_cub_reduce();
void audit_cub_sort();
void audit_cub_scan();
void audit_jit_memcpy();
void audit_jit_memcpy_async();
void audit_optix_accel_build();
void audit_optix_accel_compact();
void audit_optix_launch();

} // namespace rayd
