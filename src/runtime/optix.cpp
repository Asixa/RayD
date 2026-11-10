// Copyright Xingyu Chen.
// Implements runtime support for optix.

#include <src/runtime/optix_pipeline.h>
#include <src/runtime/rt_internal.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <src/runtime/optix_context.h>
#include <rayd/contracts.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace rayd::torch_backend {

namespace {

static_assert(shared::optix::SbtRecordAlignment == OPTIX_SBT_RECORD_ALIGNMENT);
static_assert(shared::optix::SbtRecordHeaderSize == OPTIX_SBT_RECORD_HEADER_SIZE);
using EmptySbtRecord = shared::optix::EmptySbtRecord;

// Multipath launch pipelines keep their own exception flags, separate from the scene and edge pipelines. With no
// exception flags, cold optixPipelineCreate() rejects the larger multipath modules (for example the six-raygen
// diffraction-path module) on some driver and OptiX-header combinations; see docs/optix_pipeline_create_failures.md.
// Only the flags that every supported OptiX ABI defines are used, so the value is valid with 8.x and 9.x headers.
constexpr unsigned int kMultipathExceptionFlags =
    OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;

using PipelineCacheKey = std::tuple<OptixDeviceContext, const char*, size_t, std::string, std::string, std::string,
                                    std::string, int, int, size_t>;

std::mutex& pipeline_cache_mutex() {
    static std::mutex* mutex = new std::mutex();
    return *mutex;
}

std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>>& pipeline_cache() {
    static std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>>* cache =
        new std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>>();
    return *cache;
}

// One build lock per OptiX context, guarded by pipeline_cache_mutex(); entries live as long as the process.
std::map<OptixDeviceContext, std::unique_ptr<std::mutex>>& pipeline_build_mutexes() {
    static auto* mutexes = new std::map<OptixDeviceContext, std::unique_ptr<std::mutex>>();
    return *mutexes;
}

// The retry paths below keep results unchanged but cost a rebuild or a full compile, and a failure that recurs in every
// process points at the driver or an unhealthy disk cache. Say so once per process instead of staying silent.
void warn_pipeline_fallback_once(const char* entry, const char* how, const std::string& first_failure) {
    static std::atomic<bool> warned{false};
    if (warned.exchange(true))
        return;
    TORCH_WARN("RayD: creating the OptiX multipath pipeline for '", entry, "' failed and succeeded on retry ", how,
               ". First failure: ", first_failure);
}

std::string entry_key(const char* entry) {
    return entry != nullptr ? std::string(entry) : std::string();
}

std::string raygen_key(const std::vector<const char*>& entries) {
    std::string key;
    for (const char* entry : entries) {
        if (!key.empty())
            key.push_back('\n');
        key += entry_key(entry);
    }
    return key;
}

void cuda_check(cudaError_t result, const char* expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

// A module, program-group or pipeline creation that OptiX rejected while compiling or linking, or while reading its
// disk cache. Only these failures are retried by shared_optix_launch_pipeline(); invalid arguments, out-of-memory and
// CUDA errors are reported as they are.
class OptixCompileError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

bool is_retryable_compile_result(OptixResult result) {
    switch (result) {
    case OPTIX_ERROR_FILE_IO_ERROR:
    case OPTIX_ERROR_INVALID_FILE_FORMAT:
    case OPTIX_ERROR_DISK_CACHE_INVALID_PATH:
    case OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR:
    case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR:
    case OPTIX_ERROR_DISK_CACHE_INVALID_DATA:
    case OPTIX_ERROR_PIPELINE_LINK_ERROR:
    case OPTIX_ERROR_INTERNAL_COMPILER_ERROR:
    case OPTIX_ERROR_INTERNAL_ERROR:
    case OPTIX_ERROR_UNKNOWN:
        return true;
    default:
        return false;
    }
}

// Reports a failed OptiX compile/link call together with the log OptiX wrote for it.
void optix_check_with_log(OptixResult result, const char* expr, const char* log, size_t log_size) {
    if (result == OPTIX_SUCCESS)
        return;
    std::string message = std::string("OptiX error in ") + expr + ": " + optixGetErrorName(result) +
                          " (code=" + std::to_string(static_cast<int>(result)) + ")";
    const size_t length = log == nullptr ? 0 : strnlen(log, log_size);
    if (length != 0)
        message += " log=" + std::string(log, length);
    if (is_retryable_compile_result(result))
        throw OptixCompileError(message);
    throw std::runtime_error(message);
}

void create_program_group(OptixDeviceContext context, const OptixProgramGroupDesc& desc, OptixProgramGroup* out_group) {
    OptixProgramGroupOptions options = {};
    char log[4096] = {};
    size_t log_size = sizeof(log);
    optix_check_with_log(optixProgramGroupCreate(context, &desc, 1, &options, log, &log_size, out_group),
                         "optixProgramGroupCreate", log, log_size);
}

OptixProgramGroup create_program_group(OptixDeviceContext context, const OptixProgramGroupDesc& desc) {
    OptixProgramGroup group = nullptr;
    create_program_group(context, desc, &group);
    return group;
}

// Owns the OptiX objects a lazily built pipeline has created so far. A failure part-way through destroys them, so a
// later call rebuilds from scratch instead of finding a half-built pipeline; commit() hands them to their owner.
class PendingOptixObjects {
  public:
    PendingOptixObjects() = default;
    PendingOptixObjects(const PendingOptixObjects&) = delete;
    PendingOptixObjects& operator=(const PendingOptixObjects&) = delete;

    ~PendingOptixObjects() {
        for (auto it = pipelines_.rbegin(); it != pipelines_.rend(); ++it)
            optixPipelineDestroy(*it);
        for (auto it = groups_.rbegin(); it != groups_.rend(); ++it)
            optixProgramGroupDestroy(*it);
        for (auto it = modules_.rbegin(); it != modules_.rend(); ++it)
            optixModuleDestroy(*it);
    }

    OptixModule add(OptixModule module) {
        if (module != nullptr)
            modules_.push_back(module);
        return module;
    }

    OptixProgramGroup add(OptixProgramGroup group) {
        if (group != nullptr)
            groups_.push_back(group);
        return group;
    }

    OptixPipeline add(OptixPipeline pipeline) {
        if (pipeline != nullptr)
            pipelines_.push_back(pipeline);
        return pipeline;
    }

    void commit() noexcept {
        modules_.clear();
        groups_.clear();
        pipelines_.clear();
    }

  private:
    std::vector<OptixModule> modules_;
    std::vector<OptixProgramGroup> groups_;
    std::vector<OptixPipeline> pipelines_;
};

at::Tensor make_sbt_record(int device_index, OptixProgramGroup program_group, cudaStream_t stream) {
    EmptySbtRecord host_record = {};
    rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &host_record));
    at::Tensor record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))},
                                  at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
    cuda_check(cudaMemcpyAsync(record.data_ptr<uint8_t>(), &host_record, sizeof(host_record), cudaMemcpyHostToDevice,
                               stream),
               "cudaMemcpyAsync(multipath SBT record)");
    return record;
}

int hitgroup_record_capacity(int hitgroup_record_count) {
    // Round up to a power of two so pipelines are shared across scenes with
    // similar mesh counts, but do not pad a single-record SBT to 64 entries.
    int capacity = 1;
    while (capacity < hitgroup_record_count)
        capacity *= 2;
    return capacity;
}

} // namespace

OptixLaunchPipeline::~OptixLaunchPipeline() {
    // Each event is recorded after this pipeline's optixLaunch, so waiting on all of them guarantees no launch
    // still uses the pipeline when it is destroyed below (the cache evicts pipelines while the process runs).
    for (cudaEvent_t& event : params_staging_events_) {
        if (event != nullptr) {
            cudaEventSynchronize(event);
            cudaEventDestroy(event);
            event = nullptr;
        }
    }
    if (pipeline_ != nullptr && optixPipelineDestroy != nullptr)
        optixPipelineDestroy(pipeline_);
    if (hitgroup_ != nullptr && optixProgramGroupDestroy != nullptr)
        optixProgramGroupDestroy(hitgroup_);
    if (miss_group_ != nullptr && optixProgramGroupDestroy != nullptr)
        optixProgramGroupDestroy(miss_group_);
    for (OptixProgramGroup group : raygen_groups_) {
        if (group != nullptr && optixProgramGroupDestroy != nullptr)
            optixProgramGroupDestroy(group);
    }
    if (module_ != nullptr && optixModuleDestroy != nullptr)
        optixModuleDestroy(module_);
}

void OptixLaunchPipeline::build(OptixDeviceContext context, int device_index, int hitgroup_record_count,
                                const OptixPipelineConfig& config) {
    if (context == nullptr)
        throw std::runtime_error("OptixLaunchPipeline::build(): invalid OptiX context.");
    if (hitgroup_record_count <= 0)
        throw std::runtime_error("OptixLaunchPipeline::build(): invalid hitgroup count.");
    if (config.raygen_entries.empty())
        throw std::runtime_error("OptixLaunchPipeline::build(): missing raygen entry.");

    c10::cuda::CUDAGuard guard(device_index);
    device_index_ = device_index;

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = config.num_payload_values;
    pipeline_options.numAttributeValues = shared::optix::TriangleAttributeCount;
    pipeline_options.exceptionFlags = kMultipathExceptionFlags;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipeline_options.allowOpacityMicromaps = false;

    char log[8192] = {};
    size_t log_size = sizeof(log);
    optix_check_with_log(optixModuleCreate(context, &module_options, &pipeline_options, config.ptx, config.ptx_size,
                                           log, &log_size, &module_),
                         "optixModuleCreate(multipath)", log, log_size);

    for (const char* entry : config.raygen_entries) {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = entry;
        OptixProgramGroup group = nullptr;
        create_program_group(context, desc, &group);
        raygen_groups_.push_back(group);
    }

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = config.miss_entry;
    create_program_group(context, miss_desc, &miss_group_);

    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameCH = config.closesthit_entry;
    hitgroup_desc.hitgroup.moduleAH = config.anyhit_entry != nullptr ? module_ : nullptr;
    hitgroup_desc.hitgroup.entryFunctionNameAH = config.anyhit_entry;
    create_program_group(context, hitgroup_desc, &hitgroup_);

    std::vector<OptixProgramGroup> program_groups = raygen_groups_;
    program_groups.push_back(miss_group_);
    program_groups.push_back(hitgroup_);

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;

    log_size = sizeof(log);
    optix_check_with_log(optixPipelineCreate(context, &pipeline_options, &link_options, program_groups.data(),
                                             static_cast<unsigned int>(program_groups.size()), log, &log_size,
                                             &pipeline_),
                         "optixPipelineCreate(multipath)", log, log_size);
    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup group : program_groups)
        rayd_torch_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack_sizes, pipeline_));
    uint32_t direct_callable_stack_from_traversal = 0;
    uint32_t direct_callable_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    rayd_torch_OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, link_options.maxTraceDepth, 0, 0,
                                                      &direct_callable_stack_from_traversal,
                                                      &direct_callable_stack_from_state, &continuation_stack));
    rayd_torch_OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, direct_callable_stack_from_traversal,
                                                     direct_callable_stack_from_state, continuation_stack, 2));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    for (OptixProgramGroup group : raygen_groups_)
        raygen_records_.push_back(make_sbt_record(device_index, group, stream));
    miss_record_ = make_sbt_record(device_index, miss_group_, stream);

    std::vector<EmptySbtRecord> hitgroup_host(static_cast<size_t>(hitgroup_record_count));
    for (EmptySbtRecord& record : hitgroup_host)
        rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_, &record));
    hitgroup_records_ = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord) * hitgroup_host.size())},
                                  at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
    cuda_check(cudaMemcpyAsync(hitgroup_records_.data_ptr<uint8_t>(), hitgroup_host.data(),
                               sizeof(EmptySbtRecord) * hitgroup_host.size(), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(multipath hitgroup records)");

    params_size_ = config.params_size;
    const int64_t params_capacity = static_cast<int64_t>(std::max<size_t>(params_size_, 1024));
    for (int slot = 0; slot < kParamsStagingSlots; ++slot) {
        params_buffers_[slot] =
            at::empty({params_capacity},
                      at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
        params_staging_[slot] =
            at::empty({params_capacity}, at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));
        cuda_check(cudaEventCreateWithFlags(&params_staging_events_[slot], cudaEventDisableTiming),
                   "cudaEventCreateWithFlags(params staging)");
    }
    // SBT records are immutable after construction and may immediately be
    // consumed on another CUDA stream after this shared pipeline is returned.
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SBT initialization)");
    hitgroup_record_count_ = hitgroup_record_count;
    ready_ = true;
}

std::shared_ptr<OptixLaunchPipeline> shared_optix_launch_pipeline(OptixDeviceContext context, int device_index,
                                                                  int hitgroup_record_count,
                                                                  const OptixPipelineConfig& config) {
    const int hitgroup_capacity = hitgroup_record_capacity(hitgroup_record_count);
    PipelineCacheKey key{
        context,
        config.ptx,
        config.ptx_size,
        raygen_key(config.raygen_entries),
        entry_key(config.miss_entry),
        entry_key(config.closesthit_entry),
        entry_key(config.anyhit_entry),
        hitgroup_capacity,
        config.num_payload_values,
        config.params_size,
    };

    std::mutex* build_mutex = nullptr;
    {
        std::lock_guard<std::mutex> guard(pipeline_cache_mutex());
        auto& cache = pipeline_cache();
        auto it = cache.find(key);
        if (it != cache.end())
            return it->second;
        auto& context_mutex = pipeline_build_mutexes()[context];
        if (!context_mutex)
            context_mutex = std::make_unique<std::mutex>();
        build_mutex = context_mutex.get();
    }

    // A cold build can compile for many seconds. Builds on one OptiX context are serialized (they also share the
    // context's disk-cache switch below), but the cache lock is not held across them, so lookups of pipelines that
    // are already built never wait for a compile.
    std::lock_guard<std::mutex> build_guard(*build_mutex);
    {
        std::lock_guard<std::mutex> guard(pipeline_cache_mutex());
        auto& cache = pipeline_cache();
        auto it = cache.find(key);
        if (it != cache.end())
            return it->second;
    }

    const char* entry = config.raygen_entries.empty() ? "" : config.raygen_entries.front();
    auto pipeline = std::make_shared<OptixLaunchPipeline>();
    try {
        pipeline->build(context, device_index, hitgroup_capacity, config);
    } catch (const OptixCompileError& first_failure) {
        // optixPipelineCreate() can fail with a bare COMPILE ERROR for a pipeline that is valid: on some drivers one
        // context cannot hold several large multipath pipelines at once, and the shared OptiX disk cache can get into a
        // state where large modules fail. Release the cached pipelines of this context that no caller still holds and
        // build again, which stays a quick cache hit when pipelines take turns; if that fails too, build once more
        // without the disk cache, a full compile. Each is the same module and configuration, so results are unchanged,
        // and a last failure is reported as is. Other failures (invalid input, out of memory, CUDA errors) are not
        // retried.
        const std::string first_message = first_failure.what();
        {
            std::lock_guard<std::mutex> guard(pipeline_cache_mutex());
            auto& cache = pipeline_cache();
            for (auto cached = cache.begin(); cached != cache.end();) {
                if (std::get<0>(cached->first) == context && cached->second.use_count() == 1) {
                    cached = cache.erase(cached);
                } else {
                    ++cached;
                }
            }
        }
        pipeline = std::make_shared<OptixLaunchPipeline>();
        try {
            pipeline->build(context, device_index, hitgroup_capacity, config);
            warn_pipeline_fallback_once(entry, "after releasing this context's idle cached pipelines", first_message);
        } catch (const OptixCompileError&) {
            int cache_enabled = 0;
            if (optixDeviceContextGetCacheEnabled(context, &cache_enabled) != OPTIX_SUCCESS || cache_enabled == 0)
                throw;
            optixDeviceContextSetCacheEnabled(context, 0);
            pipeline = std::make_shared<OptixLaunchPipeline>();
            try {
                pipeline->build(context, device_index, hitgroup_capacity, config);
            } catch (...) {
                optixDeviceContextSetCacheEnabled(context, 1);
                throw;
            }
            optixDeviceContextSetCacheEnabled(context, 1);
            warn_pipeline_fallback_once(entry,
                                        "with the OptiX disk cache disabled (a full compile; if this repeats in every "
                                        "process, the disk cache is unhealthy: try a fresh OPTIX_CACHE_PATH)",
                                        first_message);
        }
    }
    std::lock_guard<std::mutex> guard(pipeline_cache_mutex());
    pipeline_cache()[key] = pipeline;
    return pipeline;
}

void OptixLaunchPipeline::launch_impl(int raygen_index, const void* params, size_t actual_params_size,
                                      unsigned int n_rays, cudaStream_t stream) {
    std::lock_guard<std::mutex> guard(launch_mutex_);
    if (!ready_)
        throw std::runtime_error("OptixLaunchPipeline::launch(): pipeline is not ready.");
    if (raygen_index < 0 || raygen_index >= static_cast<int>(raygen_records_.size()))
        throw std::runtime_error("OptixLaunchPipeline::launch(): raygen index out of range.");
    const size_t launch_params_size = (std::max)(params_size_, actual_params_size);
    if (params_buffers_[0].numel() < static_cast<int64_t>(launch_params_size)) {
        // This only occurs when a caller supplies a source-compatible extended
        // launch structure. Retire every slot before replacing its storage.
        const int64_t capacity = static_cast<int64_t>(launch_params_size);
        for (int slot = 0; slot < kParamsStagingSlots; ++slot) {
            cuda_check(cudaEventSynchronize(params_staging_events_[slot]),
                       "cudaEventSynchronize(resize params buffer)");
            params_buffers_[slot] =
                at::empty({capacity},
                          at::TensorOptions().device(at::Device(at::kCUDA, device_index_)).dtype(at::kByte));
            params_staging_[slot] =
                at::empty({capacity}, at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));
        }
    }

    const int slot = params_staging_cursor_;
    params_staging_cursor_ = (params_staging_cursor_ + 1) % kParamsStagingSlots;
    // Wait until the DMA that last read this pinned slot has finished before
    // overwriting it; with the ring depth this is almost always a no-op.
    cuda_check(cudaEventSynchronize(params_staging_events_[slot]), "cudaEventSynchronize(multipath params staging)");
    std::memcpy(params_staging_[slot].data_ptr<uint8_t>(), params, launch_params_size);
    cuda_check(cudaMemcpyAsync(params_buffers_[slot].data_ptr<uint8_t>(), params_staging_[slot].data_ptr<uint8_t>(),
                               launch_params_size, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(multipath params)");
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_records_[raygen_index].data_ptr<uint8_t>());
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record_.data_ptr<uint8_t>());
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_records_.data_ptr<uint8_t>());
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_record_count_);

    const OptixResult launch_result =
        optixLaunch(pipeline_, stream, reinterpret_cast<CUdeviceptr>(params_buffers_[slot].data_ptr<uint8_t>()),
                    launch_params_size, &sbt, n_rays, 1, 1);
    cuda_check(cudaEventRecord(params_staging_events_[slot], stream), "cudaEventRecord(multipath launch params)");
    rayd_torch_OPTIX_CHECK(launch_result);
}

} // namespace rayd::torch_backend

// Shared OptiX context and pipeline ownership.

#include <src/runtime/optix_context.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <src/edge/optix_params.h>
#include <rayd/edge/point_ray_torch_ptx.h>
#include <rayd/edge/topk_torch_ptx.h>
#include <rayd/scene/intersection_torch_ptx.h>
#include <src/edge/optix_contracts.h>
#include <src/scene/scene_internal.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace rayd::torch_backend {

namespace {
std::mutex context_mutex;

struct ContextKey {
    int device_index;
    CUcontext cuda_context;

    bool operator==(const ContextKey& other) const noexcept {
        return device_index == other.device_index && cuda_context == other.cuda_context;
    }
};

struct ContextKeyHash {
    size_t operator()(const ContextKey& key) const noexcept {
        const size_t device_hash = std::hash<int>{}(key.device_index);
        const size_t context_hash = std::hash<std::uintptr_t>{}(reinterpret_cast<std::uintptr_t>(key.cuda_context));
        return device_hash ^ (context_hash + 0x9e3779b9u + (device_hash << 6u) + (device_hash >> 2u));
    }
};

std::unordered_map<ContextKey, std::unique_ptr<OptixDeviceContextEntry>, ContextKeyHash> contexts;

class OptixCapabilityUnavailable final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

static_assert(shared::optix::SbtRecordAlignment == OPTIX_SBT_RECORD_ALIGNMENT);
static_assert(shared::optix::SbtRecordHeaderSize == OPTIX_SBT_RECORD_HEADER_SIZE);
using EmptySbtRecord = shared::optix::EmptySbtRecord;

void copy_sbt_record(OptixProgramGroup program_group, at::Tensor& device_record, cudaStream_t stream) {
    EmptySbtRecord host_record = {};
    rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &host_record));
    cuda_check(cudaMemcpyAsync(device_record.data_ptr<uint8_t>(), &host_record, sizeof(host_record),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(SBT record)");
}

void copy_edge_hitgroup_records(OptixProgramGroup point_group, OptixProgramGroup ray_group, at::Tensor& device_records,
                                cudaStream_t stream) {
    EmptySbtRecord host_records[2] = {};
    rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(point_group, &host_records[0]));
    rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(ray_group, &host_records[1]));
    cuda_check(cudaMemcpyAsync(device_records.data_ptr<uint8_t>(), host_records, sizeof(host_records),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(edge hitgroup SBT records)");
}

// OptiX log levels: 1 fatal, 2 error, 3 warning, 4 print. Fatal and error messages explain a failing call that the
// thrown exception only names by code; warnings and prints are informational and stay silent.
constexpr int kOptixLogCallbackLevel = 2;

void optix_log_callback(unsigned int level, const char* tag, const char* message, void*) {
    std::fprintf(stderr, "[RayD OptiX][%u][%s]: %s\n", level, tag != nullptr ? tag : "",
                 message != nullptr ? message : "");
}

bool env_value_is_true(const char* raw) {
    if (raw == nullptr)
        return false;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value == "1" || value == "true";
}

bool optix_disabled_by_environment() {
    const char* disabled = std::getenv("RAYD_DISABLE_OPTIX");
    if (disabled == nullptr)
        disabled = std::getenv("RAYD_TORCH_DISABLE_OPTIX");
    return env_value_is_true(disabled);
}

#ifdef _WIN32
constexpr const char* kOptixLibraryName = "nvoptix.dll";
#else
constexpr const char* kOptixLibraryName = "libnvoptix.so.1";
#endif

// The two optixInit() failures that mean "this machine has no usable OptiX", explained with what fixes them.
std::string optix_runtime_unavailable_reason(OptixResult result) {
    if (result == OPTIX_ERROR_LIBRARY_NOT_FOUND)
        return std::string("the driver's OptiX library (") + kOptixLibraryName +
               ") was not found; in a container, start it with NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility "
               "(or all), because the default compute,utility does not mount it";
    int driver_version = 0;
    const std::string driver = cudaDriverGetVersion(&driver_version) == cudaSuccess
                                   ? "this driver supports CUDA " + std::to_string(driver_version / 1000) + "." +
                                         std::to_string((driver_version % 1000) / 10)
                                   : "the driver version is unknown";
    // OptiX 8.1 (ABI 93), the release-wheel SDK, first shipped in driver R555.
    const std::string minimum =
        OPTIX_ABI_VERSION == 93 ? "R555 or newer" : "one that ships OptiX ABI " + std::to_string(OPTIX_ABI_VERSION);
    return "the NVIDIA driver is older than the OptiX ABI " + std::to_string(OPTIX_ABI_VERSION) +
           " this build requests (" + driver + "); update it to " + minimum;
}

// trace_backend='auto' still works without OptiX, but on the much slower CUDA traversal. Say why once per process.
void warn_optix_fallback_once(int device_index, const char* reason) {
    static std::atomic<bool> warned{false};
    if (warned.exchange(true))
        return;
    TORCH_WARN("RayD: OptiX is unavailable on CUDA device ", device_index,
               ", so trace_backend='auto' uses the slower CUDA trace backend: ", reason,
               ". Pass trace_backend='cuda' to choose it explicitly, set RAYD_DISABLE_OPTIX=1 to silence this, or run "
               "`python -m rayd.torch.doctor` for a full report.");
}
} // namespace

TorchCudaContext current_torch_cuda_context() {
    TorchCudaContext out;
    out.device_index = c10::cuda::current_device();
    out.stream = at::cuda::getCurrentCUDAStream(out.device_index).stream();
    return out;
}

OptixDeviceContextEntry& get_optix_context(int device_index) {
    std::lock_guard<std::mutex> lock(context_mutex);
    c10::cuda::CUDAGuard guard(device_index);
    void* raw_get_current = nullptr;
    const cudaError_t lookup_result = cudaGetDriverEntryPoint("cuCtxGetCurrent", &raw_get_current, cudaEnableDefault);
    if (lookup_result != cudaSuccess || raw_get_current == nullptr) {
        throw std::runtime_error(std::string("Could not resolve cuCtxGetCurrent through the CUDA runtime: ") +
                                 cudaGetErrorString(lookup_result) + ".");
    }
    const auto get_current = reinterpret_cast<PFN_cuCtxGetCurrent>(raw_get_current);
    CUcontext cu_ctx = nullptr;
    CUresult cu_result = get_current(&cu_ctx);
    if (cu_result == CUDA_SUCCESS && cu_ctx == nullptr) {
        // A thread that has not made a CUDA runtime call yet has no current context: the device guard skips
        // cudaSetDevice when the requested device is already the thread's default. Any runtime call binds the
        // device's primary context (the one Torch uses) to this thread.
        cuda_check(cudaFree(nullptr), "cudaFree(nullptr) to bind the CUDA primary context for OptiX");
        cu_result = get_current(&cu_ctx);
    }
    if (cu_result != CUDA_SUCCESS || cu_ctx == nullptr)
        throw std::runtime_error("Could not get current CUDA context for OptiX on device " +
                                 std::to_string(device_index) + " (CUresult " + std::to_string(cu_result) + ").");

    if (optix_disabled_by_environment()) {
        throw OptixCapabilityUnavailable("OptiX is disabled by RAYD_DISABLE_OPTIX.");
    }

    const ContextKey key{device_index, cu_ctx};
    auto it = contexts.find(key);
    if (it != contexts.end())
        return *it->second;

    OptixDeviceContext optix_ctx = nullptr;
    const OptixResult init_result = optixInit();
    if (init_result == OPTIX_ERROR_LIBRARY_NOT_FOUND || init_result == OPTIX_ERROR_UNSUPPORTED_ABI_VERSION) {
        throw OptixCapabilityUnavailable(std::string("OptiX runtime is unavailable (") +
                                         optixGetErrorName(init_result) +
                                         "): " + optix_runtime_unavailable_reason(init_result));
    }
    rayd_torch_OPTIX_CHECK(init_result);
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optix_log_callback;
    options.logCallbackLevel = kOptixLogCallbackLevel;
    const OptixResult context_result = optixDeviceContextCreate(cu_ctx, &options, &optix_ctx);
    if (context_result == OPTIX_ERROR_NOT_SUPPORTED || context_result == OPTIX_ERROR_NOT_COMPATIBLE) {
        throw OptixCapabilityUnavailable(std::string("OptiX is unavailable on CUDA device ") +
                                         std::to_string(device_index) + ": " + optixGetErrorName(context_result));
    }
    // CUDA/context failures and resource exhaustion are operational errors,
    // not capability discovery. Preserve them instead of silently falling
    // back after a real OptiX failure.
    rayd_torch_OPTIX_CHECK(context_result);

    auto entry = std::make_unique<OptixDeviceContextEntry>();
    entry->device_index = device_index;
    entry->cuda_context = cu_ctx;
    entry->optix_context = optix_ctx;
    auto [inserted, _] = contexts.emplace(key, std::move(entry));
    return *inserted->second;
}

bool optix_context_available(int device_index) {
    try {
        (void)get_optix_context(device_index);
        return true;
    } catch (const OptixCapabilityUnavailable& error) {
        if (!optix_disabled_by_environment())
            warn_optix_fallback_once(device_index, error.what());
        return false;
    }
}

void ensure_intersect_pipeline(OptixDeviceContextEntry& entry) {
    std::lock_guard<std::mutex> lock(entry.pipeline_mutex);
    if (entry.intersect_pipeline != nullptr)
        return;

    c10::cuda::CUDAGuard guard(entry.device_index);
    PendingOptixObjects pending;

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = shared::optix::SceneIntersectionPayloadCount;
    pipeline_options.numAttributeValues = shared::optix::TriangleAttributeCount;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";

    char log[8192] = {};
    size_t log_size = sizeof(log);
    OptixModule module = nullptr;
    optix_check_with_log(optixModuleCreate(entry.optix_context, &module_options, &pipeline_options,
                                           rayd_torch_optix_intersect_ptx, sizeof(rayd_torch_optix_intersect_ptx), log,
                                           &log_size, &module),
                         "optixModuleCreate(intersect)", log, log_size);
    pending.add(module);

    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module;
    raygen_desc.raygen.entryFunctionName = "__raygen__intersect";
    OptixProgramGroup raygen_group = pending.add(create_program_group(entry.optix_context, raygen_desc));

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module;
    miss_desc.miss.entryFunctionName = "__miss__intersect";
    OptixProgramGroup miss_group = pending.add(create_program_group(entry.optix_context, miss_desc));

    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = module;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__intersect";
    OptixProgramGroup hitgroup = pending.add(create_program_group(entry.optix_context, hitgroup_desc));

    OptixProgramGroup program_groups[] = {raygen_group, miss_group, hitgroup};

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    OptixPipeline pipeline = nullptr;
    optix_check_with_log(optixPipelineCreate(entry.optix_context, &pipeline_options, &link_options, program_groups, 3,
                                             log, &log_size, &pipeline),
                         "optixPipelineCreate(intersect)", log, log_size);
    pending.add(pipeline);

    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup group : program_groups)
        rayd_torch_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack_sizes, pipeline));
    uint32_t direct_callable_stack_from_traversal = 0;
    uint32_t direct_callable_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    rayd_torch_OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 1, &direct_callable_stack_from_traversal,
                                                      &direct_callable_stack_from_state, &continuation_stack));
    rayd_torch_OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_from_traversal,
                                                     direct_callable_stack_from_state, continuation_stack, 2));

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, entry.device_index)).dtype(at::kByte);
    at::Tensor raygen_record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    at::Tensor miss_record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    at::Tensor hitgroup_record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(entry.device_index).stream();
    copy_sbt_record(raygen_group, raygen_record, stream);
    copy_sbt_record(miss_group, miss_record, stream);
    copy_sbt_record(hitgroup, hitgroup_record, stream);

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record.data_ptr<uint8_t>());
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record.data_ptr<uint8_t>());
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_record.data_ptr<uint8_t>());
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.hitgroupRecordCount = 1;
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(intersect SBT initialization)");

    // Publish only a complete pipeline; `intersect_pipeline` is the readiness flag and is written last.
    entry.intersect_module = module;
    entry.intersect_raygen_group = raygen_group;
    entry.intersect_miss_group = miss_group;
    entry.intersect_hitgroup = hitgroup;
    entry.intersect_raygen_record = std::move(raygen_record);
    entry.intersect_miss_record = std::move(miss_record);
    entry.intersect_hitgroup_record = std::move(hitgroup_record);
    entry.intersect_sbt = sbt;
    entry.intersect_pipeline = pipeline;
    pending.commit();
}

void ensure_edge_pipeline(OptixDeviceContextEntry& entry) {
    std::lock_guard<std::mutex> lock(entry.pipeline_mutex);
    if (entry.edge_pipeline != nullptr && entry.edge_topk_pipeline != nullptr)
        return;

    c10::cuda::CUDAGuard guard(entry.device_index);
    PendingOptixObjects pending;

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    auto make_pipeline_options = [](unsigned int payload_count) {
        OptixPipelineCompileOptions options = {};
        options.usesMotionBlur = false;
        options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        options.numPayloadValues = payload_count;
        options.numAttributeValues = shared::optix::EdgeAttributeCount;
        options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        options.pipelineLaunchParamsVariableName = "params";
        options.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);
        return options;
    };
    OptixPipelineCompileOptions point_ray_options = make_pipeline_options(shared::optix::EdgePointRayPayloadCount);
    OptixPipelineCompileOptions topk_options = make_pipeline_options(shared::optix::EdgeTopKPayloadCount);

    char log[8192] = {};
    size_t log_size = sizeof(log);
    OptixModule edge_module = nullptr;
    optix_check_with_log(optixModuleCreate(entry.optix_context, &module_options, &point_ray_options,
                                           rayd_torch_edge_optix_point_ray_ptx,
                                           sizeof(rayd_torch_edge_optix_point_ray_ptx), log, &log_size, &edge_module),
                         "optixModuleCreate(edge point/ray)", log, log_size);
    pending.add(edge_module);
    log_size = sizeof(log);
    OptixModule topk_module = nullptr;
    optix_check_with_log(optixModuleCreate(entry.optix_context, &module_options, &topk_options,
                                           rayd_torch_edge_optix_topk_ptx, sizeof(rayd_torch_edge_optix_topk_ptx), log,
                                           &log_size, &topk_module),
                         "optixModuleCreate(edge top-k)", log, log_size);
    pending.add(topk_module);

    auto make_raygen_group = [&](OptixModule module, const char* entry_name) {
        OptixProgramGroupDesc raygen_desc = {};
        raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_desc.raygen.module = module;
        raygen_desc.raygen.entryFunctionName = entry_name;
        return pending.add(create_program_group(entry.optix_context, raygen_desc));
    };
    OptixProgramGroup raygen_point_group = make_raygen_group(edge_module, "__raygen__edge_point");
    OptixProgramGroup raygen_ray_group = make_raygen_group(edge_module, "__raygen__edge_ray");
    OptixProgramGroup raygen_topk_group = make_raygen_group(topk_module, "__raygen__edge_topk_point");

    auto make_miss_group = [&](OptixModule module) {
        OptixProgramGroupDesc miss_desc = {};
        miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_desc.miss.module = module;
        miss_desc.miss.entryFunctionName = "__miss__edge_query";
        return pending.add(create_program_group(entry.optix_context, miss_desc));
    };
    OptixProgramGroup miss_group = make_miss_group(edge_module);
    OptixProgramGroup topk_miss_group = make_miss_group(topk_module);

    auto make_hitgroup = [&](OptixModule module, const char* closesthit, const char* anyhit, const char* intersection) {
        OptixProgramGroupDesc hitgroup_desc = {};
        hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_desc.hitgroup.moduleCH = closesthit != nullptr ? module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameCH = closesthit;
        hitgroup_desc.hitgroup.moduleAH = anyhit != nullptr ? module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameAH = anyhit;
        hitgroup_desc.hitgroup.moduleIS = intersection != nullptr ? module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameIS = intersection;
        return pending.add(create_program_group(entry.optix_context, hitgroup_desc));
    };
    make_hitgroup(entry.edge_module, "__closesthit__edge_point", nullptr, "__intersection__edge_point",
                  &entry.edge_hit_point_group);
    make_hitgroup(entry.edge_module, nullptr, "__anyhit__edge_ray", "__intersection__edge_ray",
                  &entry.edge_hit_ray_group);
    make_hitgroup(entry.edge_topk_module, nullptr, "__anyhit__edge_topk_point", "__intersection__edge_topk_point",
                  &entry.edge_hit_topk_group);

    OptixProgramGroup point_ray_groups[] = {
        raygen_point_group, raygen_ray_group, miss_group, hit_point_group, hit_ray_group,
    };
    OptixProgramGroup topk_groups[] = {
        raygen_topk_group,
        topk_miss_group,
        hit_topk_group,
    };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    OptixPipeline point_ray_pipeline = nullptr;
    optix_check_with_log(optixPipelineCreate(entry.optix_context, &point_ray_options, &link_options, point_ray_groups,
                                             5, log, &log_size, &point_ray_pipeline),
                         "optixPipelineCreate(edge point/ray)", log, log_size);
    pending.add(point_ray_pipeline);
    log_size = sizeof(log);
    OptixPipeline topk_pipeline = nullptr;
    optix_check_with_log(optixPipelineCreate(entry.optix_context, &topk_options, &link_options, topk_groups, 3, log,
                                             &log_size, &topk_pipeline),
                         "optixPipelineCreate(edge top-k)", log, log_size);
    pending.add(topk_pipeline);

    auto set_stack_size = [&](OptixPipeline pipeline, const OptixProgramGroup* groups, int group_count) {
        OptixStackSizes stack_sizes = {};
        for (int i = 0; i < group_count; ++i)
            rayd_torch_OPTIX_CHECK(optixUtilAccumulateStackSizes(groups[i], &stack_sizes, pipeline));
        uint32_t direct_callable_stack_from_traversal = 0;
        uint32_t direct_callable_stack_from_state = 0;
        uint32_t continuation_stack = 0;
        rayd_torch_OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 1, &direct_callable_stack_from_traversal,
                                                          &direct_callable_stack_from_state, &continuation_stack));
        rayd_torch_OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_from_traversal,
                                                         direct_callable_stack_from_state, continuation_stack, 1));
    };
    set_stack_size(point_ray_pipeline, point_ray_groups, 5);
    set_stack_size(topk_pipeline, topk_groups, 3);

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, entry.device_index)).dtype(at::kByte);
    auto make_record = [&](int64_t count) {
        return at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord)) * count}, byte_options);
    };
    at::Tensor raygen_point_record = make_record(1);
    at::Tensor raygen_ray_record = make_record(1);
    at::Tensor raygen_topk_record = make_record(1);
    at::Tensor miss_record = make_record(1);
    at::Tensor topk_miss_record = make_record(1);
    at::Tensor hitgroup_records = make_record(2);
    at::Tensor topk_hitgroup_record = make_record(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(entry.device_index).stream();
    copy_sbt_record(raygen_point_group, raygen_point_record, stream);
    copy_sbt_record(raygen_ray_group, raygen_ray_record, stream);
    copy_sbt_record(raygen_topk_group, raygen_topk_record, stream);
    copy_sbt_record(miss_group, miss_record, stream);
    copy_sbt_record(topk_miss_group, topk_miss_record, stream);
    copy_edge_hitgroup_records(hit_point_group, hit_ray_group, hitgroup_records, stream);
    copy_sbt_record(hit_topk_group, topk_hitgroup_record, stream);

    auto point_ray_sbt = [&](const at::Tensor& raygen_record) {
        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(const_cast<uint8_t*>(raygen_record.data_ptr<uint8_t>()));
        sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record.data_ptr<uint8_t>());
        sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_records.data_ptr<uint8_t>());
        sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt.hitgroupRecordCount = 2;
        return sbt;
    };
    OptixShaderBindingTable topk_sbt = {};
    topk_sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_topk_record.data_ptr<uint8_t>());
    topk_sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(topk_miss_record.data_ptr<uint8_t>());
    topk_sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    topk_sbt.missRecordCount = 1;
    topk_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(topk_hitgroup_record.data_ptr<uint8_t>());
    topk_sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    topk_sbt.hitgroupRecordCount = 1;
    const OptixShaderBindingTable point_sbt = point_ray_sbt(raygen_point_record);
    const OptixShaderBindingTable ray_sbt = point_ray_sbt(raygen_ray_record);
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(edge SBT initialization)");

    // Publish only complete pipelines; the two pipeline handles are the readiness flags and are written last.
    entry.edge_module = edge_module;
    entry.edge_topk_module = topk_module;
    entry.edge_raygen_point_group = raygen_point_group;
    entry.edge_raygen_ray_group = raygen_ray_group;
    entry.edge_raygen_topk_group = raygen_topk_group;
    entry.edge_miss_group = miss_group;
    entry.edge_topk_miss_group = topk_miss_group;
    entry.edge_hit_point_group = hit_point_group;
    entry.edge_hit_ray_group = hit_ray_group;
    entry.edge_hit_topk_group = hit_topk_group;
    entry.edge_raygen_point_record = std::move(raygen_point_record);
    entry.edge_raygen_ray_record = std::move(raygen_ray_record);
    entry.edge_raygen_topk_record = std::move(raygen_topk_record);
    entry.edge_miss_record = std::move(miss_record);
    entry.edge_topk_miss_record = std::move(topk_miss_record);
    entry.edge_hitgroup_records = std::move(hitgroup_records);
    entry.edge_topk_hitgroup_record = std::move(topk_hitgroup_record);
    entry.edge_point_sbt = point_sbt;
    entry.edge_ray_sbt = ray_sbt;
    entry.edge_topk_sbt = topk_sbt;
    entry.edge_pipeline = point_ray_pipeline;
    entry.edge_topk_pipeline = topk_pipeline;
    pending.commit();
}

OptixPipeline edge_pipeline(const OptixDeviceContextEntry& entry, EdgeOptixLaunchKind kind) {
    return kind == EdgeOptixLaunchKind::PointTopK ? entry.edge_topk_pipeline : entry.edge_pipeline;
}

const OptixShaderBindingTable& edge_sbt(const OptixDeviceContextEntry& entry, EdgeOptixLaunchKind kind) {
    switch (kind) {
    case EdgeOptixLaunchKind::Point:
        return entry.edge_point_sbt;
    case EdgeOptixLaunchKind::Ray:
        return entry.edge_ray_sbt;
    case EdgeOptixLaunchKind::PointTopK:
        return entry.edge_topk_sbt;
    }
    return entry.edge_point_sbt;
}

void optix_check(OptixResult result, const char* expr, const char* file, int line) {
    if (result == OPTIX_SUCCESS)
        return;
    throw std::runtime_error(std::string("OptiX error in ") + expr + " at " + file + ":" + std::to_string(line) + ": " +
                             optixGetErrorName(result) + " (code=" + std::to_string(static_cast<int>(result)) + ")");
}

} // namespace rayd::torch_backend
