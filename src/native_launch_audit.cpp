#include "native_launch_audit.h"

#include <algorithm>
#include <mutex>

namespace rayd {

namespace {

std::mutex &audit_mutex() {
    static std::mutex mutex;
    return mutex;
}

NativeLaunchAuditSnapshot &audit_snapshot_storage() {
    static NativeLaunchAuditSnapshot snapshot;
    return snapshot;
}

thread_local NativeLaunchStage current_stage = NativeLaunchStage::Unknown;

NativeLaunchStageStats &stage_stats(NativeLaunchAuditSnapshot &snapshot,
                                    NativeLaunchStage stage) {
    switch (stage) {
    case NativeLaunchStage::Build:
        return snapshot.build;
    case NativeLaunchStage::Sync:
        return snapshot.sync;
    case NativeLaunchStage::TraceReflections:
        return snapshot.trace_reflections;
    case NativeLaunchStage::Unknown:
    default:
        return snapshot.unknown;
    }
}

void clear_stage_stats(NativeLaunchStageStats &stats) {
    stats = NativeLaunchStageStats();
}

void update_kernel_stat(NativeLaunchStageStats &stats,
                        const char *label,
                        uint64_t threads,
                        uint64_t items) {
    const std::string key = label != nullptr ? std::string(label) : std::string("unnamed");
    auto it = std::find_if(stats.kernels.begin(),
                           stats.kernels.end(),
                           [&key](const NativeKernelLaunchStat &entry) {
                               return entry.label == key;
                           });
    if (it == stats.kernels.end()) {
        stats.kernels.push_back(NativeKernelLaunchStat { key, 1, threads, threads, items, items });
        return;
    }

    it->launches += 1;
    it->total_threads += threads;
    it->max_threads = std::max(it->max_threads, threads);
    it->total_items += items;
    it->max_items = std::max(it->max_items, items);
}

template <typename Member>
void increment_counter(Member NativeLaunchStageStats::*member) {
    std::lock_guard<std::mutex> guard(audit_mutex());
    NativeLaunchStageStats &stats = stage_stats(audit_snapshot_storage(), current_stage);
    stats.*member += 1;
}

} // namespace

ScopedNativeLaunchStage::ScopedNativeLaunchStage(NativeLaunchStage stage)
    : previous_(current_stage) {
    current_stage = stage;
}

ScopedNativeLaunchStage::~ScopedNativeLaunchStage() {
    current_stage = previous_;
}

void native_launch_audit_clear() {
    std::lock_guard<std::mutex> guard(audit_mutex());
    NativeLaunchAuditSnapshot &snapshot = audit_snapshot_storage();
    clear_stage_stats(snapshot.unknown);
    clear_stage_stats(snapshot.build);
    clear_stage_stats(snapshot.sync);
    clear_stage_stats(snapshot.trace_reflections);
}

NativeLaunchAuditSnapshot native_launch_audit_snapshot() {
    std::lock_guard<std::mutex> guard(audit_mutex());
    return audit_snapshot_storage();
}

void audit_cuda_kernel_launch(const char *label,
                              uint32_t grid_x,
                              uint32_t grid_y,
                              uint32_t grid_z,
                              uint32_t block_x,
                              uint32_t block_y,
                              uint32_t block_z,
                              uint64_t items) {
    const uint64_t threads =
        static_cast<uint64_t>(grid_x) * static_cast<uint64_t>(grid_y) * static_cast<uint64_t>(grid_z) *
        static_cast<uint64_t>(block_x) * static_cast<uint64_t>(block_y) * static_cast<uint64_t>(block_z);

    std::lock_guard<std::mutex> guard(audit_mutex());
    NativeLaunchStageStats &stats = stage_stats(audit_snapshot_storage(), current_stage);
    stats.cuda_kernel_launches += 1;
    stats.cuda_kernel_total_threads += threads;
    update_kernel_stat(stats, label, threads, items);
}

void audit_cuda_memcpy() {
    increment_counter(&NativeLaunchStageStats::cuda_memcpy);
}

void audit_cuda_memcpy_async() {
    increment_counter(&NativeLaunchStageStats::cuda_memcpy_async);
}

void audit_cuda_memset_async() {
    increment_counter(&NativeLaunchStageStats::cuda_memset_async);
}

void audit_cuda_stream_synchronize() {
    increment_counter(&NativeLaunchStageStats::cuda_stream_synchronize);
}

void audit_cuda_event_record() {
    increment_counter(&NativeLaunchStageStats::cuda_event_record);
}

void audit_cuda_stream_wait_event() {
    increment_counter(&NativeLaunchStageStats::cuda_stream_wait_event);
}

void audit_cub_reduce() {
    increment_counter(&NativeLaunchStageStats::cub_reduce);
}

void audit_cub_sort() {
    increment_counter(&NativeLaunchStageStats::cub_sort);
}

void audit_cub_scan() {
    increment_counter(&NativeLaunchStageStats::cub_scan);
}

void audit_jit_memcpy() {
    increment_counter(&NativeLaunchStageStats::jit_memcpy);
}

void audit_jit_memcpy_async() {
    increment_counter(&NativeLaunchStageStats::jit_memcpy_async);
}

void audit_optix_accel_build() {
    increment_counter(&NativeLaunchStageStats::optix_accel_build);
}

void audit_optix_accel_compact() {
    increment_counter(&NativeLaunchStageStats::optix_accel_compact);
}

void audit_optix_launch() {
    increment_counter(&NativeLaunchStageStats::optix_launch);
}

} // namespace rayd
