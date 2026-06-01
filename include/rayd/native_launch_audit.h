#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rayd {

// Lightweight instrumentation that tallies CUDA/CUB/OptiX launches and memory
// traffic, attributed to the high-level operation in flight. Counting is keyed
// by the current thread's NativeLaunchStage (set via ScopedNativeLaunchStage)
// and is only populated when the audit hooks are compiled in.

/// High-level operation an audited launch is attributed to.
enum class NativeLaunchStage {
    Unknown = 0,
    Build,
    Sync,
    TraceReflections,
    AccumulateReflections,
    AccumDfr,
    SurfelTrace
};

/// Aggregated launch counts for one named kernel within a stage.
struct NativeKernelLaunchStat {
    std::string label;             ///< Kernel name, or "unnamed".
    uint64_t launches = 0;          ///< Number of launches.
    uint64_t total_threads = 0;     ///< Sum of thread counts across launches.
    uint64_t max_threads = 0;       ///< Largest single-launch thread count.
    uint64_t total_items = 0;       ///< Sum of caller-reported item counts.
    uint64_t max_items = 0;         ///< Largest single-launch item count.
};

/// Per-stage tallies of device launches and host/device memory operations.
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
    double optix_launch_time_ms = 0.0;
    double optix_launch_time_min_ms = 0.0;
    double optix_launch_time_max_ms = 0.0;
    std::vector<NativeKernelLaunchStat> kernels;
};

/// Full audit state: one NativeLaunchStageStats per NativeLaunchStage.
struct NativeLaunchAuditSnapshot {
    NativeLaunchStageStats unknown;
    NativeLaunchStageStats build;
    NativeLaunchStageStats sync;
    NativeLaunchStageStats trace_reflections;
    NativeLaunchStageStats accumulate_reflections;
    NativeLaunchStageStats accum_dfr;
    NativeLaunchStageStats surfel_trace;
};

/// RAII guard that sets the current thread's audit stage and restores the previous one on scope exit.
class ScopedNativeLaunchStage {
public:
    explicit ScopedNativeLaunchStage(NativeLaunchStage stage);
    ~ScopedNativeLaunchStage();

    ScopedNativeLaunchStage(const ScopedNativeLaunchStage &) = delete;
    ScopedNativeLaunchStage &operator=(const ScopedNativeLaunchStage &) = delete;

private:
    NativeLaunchStage previous_;
};

/// Reset all audit counters to zero.
void native_launch_audit_clear();
/// Return a copy of the current audit counters across all stages.
NativeLaunchAuditSnapshot native_launch_audit_snapshot();

// Record-one-event hooks called from the CUDA/CUB/OptiX wrappers; each adds to
// the counters of the current thread's stage. \p label names the kernel and
// \p items is an optional caller-defined work count (e.g. rays processed).
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
void audit_optix_launch_duration_ms(double elapsed_ms);

/// Return true when native OptiX launch timing should record CUDA event durations.
bool native_launch_audit_timing_enabled();

} // namespace rayd
