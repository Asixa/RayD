#include <rayd/torch/runtime/optix_pipeline.h>
#include <rayd/shared/optix/pipeline_contracts.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <rayd/torch/runtime/optix_context.h>
#include <rayd/shared/optix/scene_edge_contracts.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace rayd::torch_backend {

namespace {

static_assert(shared::optix::SbtRecordAlignment == OPTIX_SBT_RECORD_ALIGNMENT);
static_assert(shared::optix::SbtRecordHeaderSize == OPTIX_SBT_RECORD_HEADER_SIZE);
using EmptySbtRecord = shared::optix::EmptySbtRecord;

using PipelineCacheKey = std::tuple<
    OptixDeviceContext,
    const char *,
    size_t,
    std::string,
    std::string,
    std::string,
    std::string,
    int,
    int,
    size_t>;

std::mutex &pipeline_cache_mutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>> &pipeline_cache() {
    static std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>> *cache =
        new std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>>();
    return *cache;
}

std::string entry_key(const char *entry) {
    return entry != nullptr ? std::string(entry) : std::string();
}

std::string raygen_key(const std::vector<const char *> &entries) {
    std::string key;
    for (const char *entry : entries) {
        if (!key.empty())
            key.push_back('\n');
        key += entry_key(entry);
    }
    return key;
}

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void create_program_group(
    OptixDeviceContext context,
    const OptixProgramGroupDesc &desc,
    OptixProgramGroup *out_group) {
    OptixProgramGroupOptions options = {};
    char log[4096] = {};
    size_t log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(
        optixProgramGroupCreate(context, &desc, 1, &options, log, &log_size, out_group));
}

at::Tensor make_sbt_record(
    int device_index,
    OptixProgramGroup program_group,
    cudaStream_t stream) {
    EmptySbtRecord host_record = {};
    rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &host_record));
    at::Tensor record = at::empty(
        {static_cast<int64_t>(sizeof(EmptySbtRecord))},
        at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
    cuda_check(
        cudaMemcpyAsync(
            record.data_ptr<uint8_t>(),
            &host_record,
            sizeof(host_record),
            cudaMemcpyHostToDevice,
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
    for (cudaEvent_t &event : params_staging_events_) {
        if (event != nullptr) {
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

void OptixLaunchPipeline::build(
    OptixDeviceContext context,
    int device_index,
    int hitgroup_record_count,
    const OptixPipelineConfig &config) {
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
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = config.num_payload_values;
    pipeline_options.numAttributeValues = shared::optix::TriangleAttributeCount;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipeline_options.allowOpacityMicromaps = false;

    char log[8192] = {};
    size_t log_size = sizeof(log);
    OptixResult result = optixModuleCreate(
        context,
        &module_options,
        &pipeline_options,
        config.ptx,
        config.ptx_size,
        log,
        &log_size,
        &module_);
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(
            std::string("OptiX error in optixModuleCreate(multipath): code=") +
            std::to_string(static_cast<int>(result)) + " log=" + std::string(log, log_size));
    }

    for (const char *entry : config.raygen_entries) {
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
    result = optixPipelineCreate(
        context,
        &pipeline_options,
        &link_options,
        program_groups.data(),
        static_cast<unsigned int>(program_groups.size()),
        log,
        &log_size,
        &pipeline_);
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(
            std::string("OptiX error in optixPipelineCreate(multipath): code=") +
            std::to_string(static_cast<int>(result)) + " log=" + std::string(log, log_size));
    }
    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup group : program_groups)
        rayd_torch_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack_sizes, pipeline_));
    uint32_t direct_callable_stack_from_traversal = 0;
    uint32_t direct_callable_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    rayd_torch_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        link_options.maxTraceDepth,
        0,
        0,
        &direct_callable_stack_from_traversal,
        &direct_callable_stack_from_state,
        &continuation_stack));
    rayd_torch_OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        direct_callable_stack_from_traversal,
        direct_callable_stack_from_state,
        continuation_stack,
        2));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    for (OptixProgramGroup group : raygen_groups_)
        raygen_records_.push_back(make_sbt_record(device_index, group, stream));
    miss_record_ = make_sbt_record(device_index, miss_group_, stream);

    std::vector<EmptySbtRecord> hitgroup_host(static_cast<size_t>(hitgroup_record_count));
    for (EmptySbtRecord &record : hitgroup_host)
        rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_, &record));
    hitgroup_records_ = at::empty(
        {static_cast<int64_t>(sizeof(EmptySbtRecord) * hitgroup_host.size())},
        at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
    cuda_check(
        cudaMemcpyAsync(
            hitgroup_records_.data_ptr<uint8_t>(),
            hitgroup_host.data(),
            sizeof(EmptySbtRecord) * hitgroup_host.size(),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(multipath hitgroup records)");

    params_size_ = config.params_size;
    const int64_t params_capacity = static_cast<int64_t>(std::max<size_t>(params_size_, 1024));
    for (int slot = 0; slot < kParamsStagingSlots; ++slot) {
        params_buffers_[slot] = at::empty(
            {params_capacity},
            at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
        params_staging_[slot] = at::empty(
            {params_capacity},
            at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));
        cuda_check(
            cudaEventCreateWithFlags(&params_staging_events_[slot], cudaEventDisableTiming),
            "cudaEventCreateWithFlags(params staging)");
    }
    // SBT records are immutable after construction and may immediately be
    // consumed on another CUDA stream after this shared pipeline is returned.
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SBT initialization)");
    hitgroup_record_count_ = hitgroup_record_count;
    ready_ = true;
}

std::shared_ptr<OptixLaunchPipeline> shared_optix_launch_pipeline(
    OptixDeviceContext context,
    int device_index,
    int hitgroup_record_count,
    const OptixPipelineConfig &config) {
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

    std::lock_guard<std::mutex> guard(pipeline_cache_mutex());
    auto &cache = pipeline_cache();
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    auto pipeline = std::make_shared<OptixLaunchPipeline>();
    pipeline->build(context, device_index, hitgroup_capacity, config);
    cache[key] = pipeline;
    return pipeline;
}

void OptixLaunchPipeline::launch_impl(
    int raygen_index,
    const void *params,
    size_t actual_params_size,
    unsigned int n_rays,
    cudaStream_t stream) {
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
            cuda_check(
                cudaEventSynchronize(params_staging_events_[slot]),
                "cudaEventSynchronize(resize params buffer)");
            params_buffers_[slot] = at::empty(
                {capacity},
                at::TensorOptions()
                    .device(at::Device(at::kCUDA, device_index_))
                    .dtype(at::kByte));
            params_staging_[slot] = at::empty(
                {capacity},
                at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));
        }
    }

    const int slot = params_staging_cursor_;
    params_staging_cursor_ = (params_staging_cursor_ + 1) % kParamsStagingSlots;
    // Wait until the DMA that last read this pinned slot has finished before
    // overwriting it; with the ring depth this is almost always a no-op.
    cuda_check(
        cudaEventSynchronize(params_staging_events_[slot]),
        "cudaEventSynchronize(multipath params staging)");
    std::memcpy(params_staging_[slot].data_ptr<uint8_t>(), params, launch_params_size);
    cuda_check(
        cudaMemcpyAsync(
            params_buffers_[slot].data_ptr<uint8_t>(),
            params_staging_[slot].data_ptr<uint8_t>(),
            launch_params_size,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(multipath params)");
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord =
        reinterpret_cast<CUdeviceptr>(raygen_records_[raygen_index].data_ptr<uint8_t>());
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record_.data_ptr<uint8_t>());
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_records_.data_ptr<uint8_t>());
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_record_count_);

    const OptixResult launch_result = optixLaunch(
        pipeline_,
        stream,
        reinterpret_cast<CUdeviceptr>(params_buffers_[slot].data_ptr<uint8_t>()),
        launch_params_size,
        &sbt,
        n_rays,
        1,
        1);
    cuda_check(
        cudaEventRecord(params_staging_events_[slot], stream),
        "cudaEventRecord(multipath launch params)");
    rayd_torch_OPTIX_CHECK(launch_result);
}

} // namespace rayd::torch_backend


// ---- merged from src/runtime/optix_context_part.cpp ----

#include <rayd/torch/runtime/optix_context.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <rayd/torch/edge/optix_params.h>
#include <rayd/torch/edge_optix_point_ray_ptx.h>
#include <rayd/torch/edge_optix_topk_ptx.h>
#include <rayd/torch/optix_intersect_ptx.h>
#include <rayd/torch/reflection_trace_optix_ptx.h>
#include <rayd/shared/optix/scene_edge_contracts.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
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

    bool operator==(const ContextKey &other) const noexcept {
        return device_index == other.device_index && cuda_context == other.cuda_context;
    }
};

struct ContextKeyHash {
    size_t operator()(const ContextKey &key) const noexcept {
        const size_t device_hash = std::hash<int>{}(key.device_index);
        const size_t context_hash = std::hash<std::uintptr_t>{}(
            reinterpret_cast<std::uintptr_t>(key.cuda_context));
        return device_hash ^ (context_hash + 0x9e3779b9u + (device_hash << 6u) +
                              (device_hash >> 2u));
    }
};

std::unordered_map<ContextKey, std::unique_ptr<OptixDeviceContextEntry>, ContextKeyHash>
    contexts;

class OptixCapabilityUnavailable final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

static_assert(shared::optix::SbtRecordAlignment == OPTIX_SBT_RECORD_ALIGNMENT);
static_assert(shared::optix::SbtRecordHeaderSize == OPTIX_SBT_RECORD_HEADER_SIZE);
using EmptySbtRecord = shared::optix::EmptySbtRecord;


void copy_sbt_record(
    OptixProgramGroup program_group,
    at::Tensor &device_record,
    cudaStream_t stream) {
    EmptySbtRecord host_record = {};
    rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &host_record));
    cuda_check(
        cudaMemcpyAsync(
            device_record.data_ptr<uint8_t>(),
            &host_record,
            sizeof(host_record),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(SBT record)");
}

void copy_edge_hitgroup_records(
    OptixProgramGroup point_group,
    OptixProgramGroup ray_group,
    at::Tensor &device_records,
    cudaStream_t stream) {
    EmptySbtRecord host_records[2] = {};
    rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(point_group, &host_records[0]));
    rayd_torch_OPTIX_CHECK(optixSbtRecordPackHeader(ray_group, &host_records[1]));
    cuda_check(
        cudaMemcpyAsync(
            device_records.data_ptr<uint8_t>(),
            host_records,
            sizeof(host_records),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(edge hitgroup SBT records)");
}


bool env_value_is_true(const char *raw) {
    if (raw == nullptr)
        return false;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value == "1" || value == "true";
}
} // namespace

TorchCudaContext current_torch_cuda_context() {
    TorchCudaContext out;
    out.device_index = c10::cuda::current_device();
    out.stream = at::cuda::getCurrentCUDAStream(out.device_index).stream();
    return out;
}

OptixDeviceContextEntry &get_optix_context(int device_index) {
    std::lock_guard<std::mutex> lock(context_mutex);
    c10::cuda::CUDAGuard guard(device_index);
    CUcontext cu_ctx = nullptr;
    CUresult cu_result = cuCtxGetCurrent(&cu_ctx);
    if (cu_result != CUDA_SUCCESS || cu_ctx == nullptr)
        throw std::runtime_error("Could not get current CUDA context for OptiX.");

    const char *disabled = std::getenv("RAYD_DISABLE_OPTIX");
    if (disabled == nullptr)
        disabled = std::getenv("RAYD_TORCH_DISABLE_OPTIX");
    if (env_value_is_true(disabled)) {
        throw OptixCapabilityUnavailable(
            "OptiX is disabled by RAYD_DISABLE_OPTIX.");
    }

    const ContextKey key{device_index, cu_ctx};
    auto it = contexts.find(key);
    if (it != contexts.end())
        return *it->second;

    OptixDeviceContext optix_ctx = nullptr;
    const OptixResult init_result = optixInit();
    if (init_result == OPTIX_ERROR_LIBRARY_NOT_FOUND ||
        init_result == OPTIX_ERROR_UNSUPPORTED_ABI_VERSION) {
        throw OptixCapabilityUnavailable(
            std::string("OptiX runtime is unavailable: ") + optixGetErrorName(init_result));
    }
    rayd_torch_OPTIX_CHECK(init_result);
    OptixDeviceContextOptions options = {};
    const OptixResult context_result =
        optixDeviceContextCreate(cu_ctx, &options, &optix_ctx);
    if (context_result == OPTIX_ERROR_NOT_SUPPORTED ||
        context_result == OPTIX_ERROR_NOT_COMPATIBLE) {
        throw OptixCapabilityUnavailable(
            std::string("OptiX is unavailable on CUDA device ") +
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
    } catch (const OptixCapabilityUnavailable &) {
        return false;
    }
}

void ensure_intersect_pipeline(OptixDeviceContextEntry &entry) {
    std::lock_guard<std::mutex> lock(entry.pipeline_mutex);
    if (entry.intersect_pipeline != nullptr)
        return;

    c10::cuda::CUDAGuard guard(entry.device_index);

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = shared::optix::SceneIntersectionPayloadCount;
    pipeline_options.numAttributeValues = shared::optix::TriangleAttributeCount;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";

    char log[8192] = {};
    size_t log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(optixModuleCreate(
        entry.optix_context,
        &module_options,
        &pipeline_options,
        rayd_torch_optix_intersect_ptx,
        sizeof(rayd_torch_optix_intersect_ptx),
        log,
        &log_size,
        &entry.intersect_module));

    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = entry.intersect_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__intersect";
    create_program_group(entry.optix_context, raygen_desc, &entry.intersect_raygen_group);

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = entry.intersect_module;
    miss_desc.miss.entryFunctionName = "__miss__intersect";
    create_program_group(entry.optix_context, miss_desc, &entry.intersect_miss_group);

    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = entry.intersect_module;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__intersect";
    create_program_group(entry.optix_context, hitgroup_desc, &entry.intersect_hitgroup);

    OptixProgramGroup program_groups[] = {
        entry.intersect_raygen_group,
        entry.intersect_miss_group,
        entry.intersect_hitgroup,
    };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(optixPipelineCreate(
        entry.optix_context,
        &pipeline_options,
        &link_options,
        program_groups,
        3,
        log,
        &log_size,
        &entry.intersect_pipeline));

    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup group : program_groups)
        rayd_torch_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack_sizes, entry.intersect_pipeline));
    uint32_t direct_callable_stack_from_traversal = 0;
    uint32_t direct_callable_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    rayd_torch_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        1,
        0,
        1,
        &direct_callable_stack_from_traversal,
        &direct_callable_stack_from_state,
        &continuation_stack));
    rayd_torch_OPTIX_CHECK(optixPipelineSetStackSize(
        entry.intersect_pipeline,
        direct_callable_stack_from_traversal,
        direct_callable_stack_from_state,
        continuation_stack,
        2));

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, entry.device_index)).dtype(at::kByte);
    entry.intersect_raygen_record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.intersect_miss_record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.intersect_hitgroup_record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(entry.device_index).stream();
    copy_sbt_record(entry.intersect_raygen_group, entry.intersect_raygen_record, stream);
    copy_sbt_record(entry.intersect_miss_group, entry.intersect_miss_record, stream);
    copy_sbt_record(entry.intersect_hitgroup, entry.intersect_hitgroup_record, stream);

    entry.intersect_sbt = {};
    entry.intersect_sbt.raygenRecord =
        reinterpret_cast<CUdeviceptr>(entry.intersect_raygen_record.data_ptr<uint8_t>());
    entry.intersect_sbt.missRecordBase =
        reinterpret_cast<CUdeviceptr>(entry.intersect_miss_record.data_ptr<uint8_t>());
    entry.intersect_sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    entry.intersect_sbt.missRecordCount = 1;
    entry.intersect_sbt.hitgroupRecordBase =
        reinterpret_cast<CUdeviceptr>(entry.intersect_hitgroup_record.data_ptr<uint8_t>());
    entry.intersect_sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    entry.intersect_sbt.hitgroupRecordCount = 1;
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(intersect SBT initialization)");
}

void ensure_edge_pipeline(OptixDeviceContextEntry &entry) {
    std::lock_guard<std::mutex> lock(entry.pipeline_mutex);
    if (entry.edge_pipeline != nullptr && entry.edge_topk_pipeline != nullptr)
        return;

    c10::cuda::CUDAGuard guard(entry.device_index);

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
        options.usesPrimitiveTypeFlags =
            static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);
        return options;
    };
    OptixPipelineCompileOptions point_ray_options =
        make_pipeline_options(shared::optix::EdgePointRayPayloadCount);
    OptixPipelineCompileOptions topk_options =
        make_pipeline_options(shared::optix::EdgeTopKPayloadCount);

    char log[8192] = {};
    size_t log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(optixModuleCreate(
        entry.optix_context,
        &module_options,
        &point_ray_options,
        rayd_torch_edge_optix_point_ray_ptx,
        sizeof(rayd_torch_edge_optix_point_ray_ptx),
        log,
        &log_size,
        &entry.edge_module));
    log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(optixModuleCreate(
        entry.optix_context,
        &module_options,
        &topk_options,
        rayd_torch_edge_optix_topk_ptx,
        sizeof(rayd_torch_edge_optix_topk_ptx),
        log,
        &log_size,
        &entry.edge_topk_module));

    auto make_raygen_group = [&](OptixModule module,
                                 const char *entry_name,
                                 OptixProgramGroup *out_group) {
        OptixProgramGroupDesc raygen_desc = {};
        raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_desc.raygen.module = module;
        raygen_desc.raygen.entryFunctionName = entry_name;
        create_program_group(entry.optix_context, raygen_desc, out_group);
    };
    make_raygen_group(entry.edge_module, "__raygen__edge_point", &entry.edge_raygen_point_group);
    make_raygen_group(entry.edge_module, "__raygen__edge_ray", &entry.edge_raygen_ray_group);
    make_raygen_group(
        entry.edge_topk_module,
        "__raygen__edge_topk_point",
        &entry.edge_raygen_topk_group);

    auto make_miss_group = [&](OptixModule module, OptixProgramGroup *out_group) {
        OptixProgramGroupDesc miss_desc = {};
        miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_desc.miss.module = module;
        miss_desc.miss.entryFunctionName = "__miss__edge_query";
        create_program_group(entry.optix_context, miss_desc, out_group);
    };
    make_miss_group(entry.edge_module, &entry.edge_miss_group);
    make_miss_group(entry.edge_topk_module, &entry.edge_topk_miss_group);

    auto make_hitgroup = [&](
                             OptixModule module,
                             const char *closesthit,
                             const char *anyhit,
                             const char *intersection,
                             OptixProgramGroup *out_group) {
        OptixProgramGroupDesc hitgroup_desc = {};
        hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_desc.hitgroup.moduleCH = closesthit != nullptr ? module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameCH = closesthit;
        hitgroup_desc.hitgroup.moduleAH = anyhit != nullptr ? module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameAH = anyhit;
        hitgroup_desc.hitgroup.moduleIS = intersection != nullptr ? module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameIS = intersection;
        create_program_group(entry.optix_context, hitgroup_desc, out_group);
    };
    make_hitgroup(
        entry.edge_module,
        "__closesthit__edge_point",
        nullptr,
        "__intersection__edge_point",
        &entry.edge_hit_point_group);
    make_hitgroup(
        entry.edge_module,
        nullptr,
        "__anyhit__edge_ray",
        "__intersection__edge_ray",
        &entry.edge_hit_ray_group);
    make_hitgroup(
        entry.edge_topk_module,
        nullptr,
        "__anyhit__edge_topk_point",
        "__intersection__edge_topk_point",
        &entry.edge_hit_topk_group);

    OptixProgramGroup point_ray_groups[] = {
        entry.edge_raygen_point_group,
        entry.edge_raygen_ray_group,
        entry.edge_miss_group,
        entry.edge_hit_point_group,
        entry.edge_hit_ray_group,
    };
    OptixProgramGroup topk_groups[] = {
        entry.edge_raygen_topk_group,
        entry.edge_topk_miss_group,
        entry.edge_hit_topk_group,
    };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(optixPipelineCreate(
        entry.optix_context,
        &point_ray_options,
        &link_options,
        point_ray_groups,
        5,
        log,
        &log_size,
        &entry.edge_pipeline));
    log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(optixPipelineCreate(
        entry.optix_context,
        &topk_options,
        &link_options,
        topk_groups,
        3,
        log,
        &log_size,
        &entry.edge_topk_pipeline));

    auto set_stack_size = [&](OptixPipeline pipeline,
                              const OptixProgramGroup *groups,
                              int group_count) {
        OptixStackSizes stack_sizes = {};
        for (int i = 0; i < group_count; ++i)
            rayd_torch_OPTIX_CHECK(optixUtilAccumulateStackSizes(groups[i], &stack_sizes, pipeline));
        uint32_t direct_callable_stack_from_traversal = 0;
        uint32_t direct_callable_stack_from_state = 0;
        uint32_t continuation_stack = 0;
        rayd_torch_OPTIX_CHECK(optixUtilComputeStackSizes(
            &stack_sizes,
            1,
            0,
            1,
            &direct_callable_stack_from_traversal,
            &direct_callable_stack_from_state,
            &continuation_stack));
        rayd_torch_OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline,
            direct_callable_stack_from_traversal,
            direct_callable_stack_from_state,
            continuation_stack,
            1));
    };
    set_stack_size(entry.edge_pipeline, point_ray_groups, 5);
    set_stack_size(entry.edge_topk_pipeline, topk_groups, 3);

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, entry.device_index)).dtype(at::kByte);
    entry.edge_raygen_point_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_raygen_ray_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_raygen_topk_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_miss_record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_topk_miss_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_hitgroup_records =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord) * 2)}, byte_options);
    entry.edge_topk_hitgroup_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(entry.device_index).stream();
    copy_sbt_record(entry.edge_raygen_point_group, entry.edge_raygen_point_record, stream);
    copy_sbt_record(entry.edge_raygen_ray_group, entry.edge_raygen_ray_record, stream);
    copy_sbt_record(entry.edge_raygen_topk_group, entry.edge_raygen_topk_record, stream);
    copy_sbt_record(entry.edge_miss_group, entry.edge_miss_record, stream);
    copy_sbt_record(entry.edge_topk_miss_group, entry.edge_topk_miss_record, stream);
    copy_edge_hitgroup_records(
        entry.edge_hit_point_group,
        entry.edge_hit_ray_group,
        entry.edge_hitgroup_records,
        stream);
    copy_sbt_record(entry.edge_hit_topk_group, entry.edge_topk_hitgroup_record, stream);

    auto init_point_ray_sbt = [&](OptixShaderBindingTable &sbt, const at::Tensor &raygen_record) {
        sbt = {};
        sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(
            const_cast<uint8_t *>(raygen_record.data_ptr<uint8_t>()));
        sbt.missRecordBase =
            reinterpret_cast<CUdeviceptr>(entry.edge_miss_record.data_ptr<uint8_t>());
        sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase =
            reinterpret_cast<CUdeviceptr>(entry.edge_hitgroup_records.data_ptr<uint8_t>());
        sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt.hitgroupRecordCount = 2;
    };
    auto init_topk_sbt = [&]() {
        entry.edge_topk_sbt = {};
        entry.edge_topk_sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(
            entry.edge_raygen_topk_record.data_ptr<uint8_t>());
        entry.edge_topk_sbt.missRecordBase =
            reinterpret_cast<CUdeviceptr>(entry.edge_topk_miss_record.data_ptr<uint8_t>());
        entry.edge_topk_sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
        entry.edge_topk_sbt.missRecordCount = 1;
        entry.edge_topk_sbt.hitgroupRecordBase =
            reinterpret_cast<CUdeviceptr>(entry.edge_topk_hitgroup_record.data_ptr<uint8_t>());
        entry.edge_topk_sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        entry.edge_topk_sbt.hitgroupRecordCount = 1;
    };
    init_point_ray_sbt(entry.edge_point_sbt, entry.edge_raygen_point_record);
    init_point_ray_sbt(entry.edge_ray_sbt, entry.edge_raygen_ray_record);
    init_topk_sbt();
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(edge SBT initialization)");
}

OptixPipeline edge_pipeline(const OptixDeviceContextEntry &entry, EdgeOptixLaunchKind kind) {
    return kind == EdgeOptixLaunchKind::PointTopK ? entry.edge_topk_pipeline : entry.edge_pipeline;
}

const OptixShaderBindingTable &edge_sbt(const OptixDeviceContextEntry &entry, EdgeOptixLaunchKind kind) {
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

void ensure_reflection_trace_pipeline(OptixDeviceContextEntry &entry) {
    std::lock_guard<std::mutex> lock(entry.pipeline_mutex);
    if (entry.reflection_trace_pipeline != nullptr)
        return;

    c10::cuda::CUDAGuard guard(entry.device_index);

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = shared::optix::TriangleHitPayloadCount;
    pipeline_options.numAttributeValues = shared::optix::TriangleAttributeCount;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    char log[8192] = {};
    size_t log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(optixModuleCreate(
        entry.optix_context,
        &module_options,
        &pipeline_options,
        rayd_torch_reflection_trace_optix_ptx,
        sizeof(rayd_torch_reflection_trace_optix_ptx),
        log,
        &log_size,
        &entry.reflection_trace_module));

    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = entry.reflection_trace_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__reflection_trace";
    create_program_group(entry.optix_context, raygen_desc, &entry.reflection_trace_raygen_group);

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = entry.reflection_trace_module;
    miss_desc.miss.entryFunctionName = "__miss__reflection";
    create_program_group(entry.optix_context, miss_desc, &entry.reflection_trace_miss_group);

    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = entry.reflection_trace_module;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__reflection";
    create_program_group(entry.optix_context, hitgroup_desc, &entry.reflection_trace_hitgroup);

    OptixProgramGroup program_groups[] = {
        entry.reflection_trace_raygen_group,
        entry.reflection_trace_miss_group,
        entry.reflection_trace_hitgroup,
    };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    rayd_torch_OPTIX_CHECK(optixPipelineCreate(
        entry.optix_context,
        &pipeline_options,
        &link_options,
        program_groups,
        3,
        log,
        &log_size,
        &entry.reflection_trace_pipeline));

    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup group : program_groups)
        rayd_torch_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack_sizes, entry.reflection_trace_pipeline));
    uint32_t direct_callable_stack_from_traversal = 0;
    uint32_t direct_callable_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    rayd_torch_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        1,
        0,
        1,
        &direct_callable_stack_from_traversal,
        &direct_callable_stack_from_state,
        &continuation_stack));
    rayd_torch_OPTIX_CHECK(optixPipelineSetStackSize(
        entry.reflection_trace_pipeline,
        direct_callable_stack_from_traversal,
        direct_callable_stack_from_state,
        continuation_stack,
        1));

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, entry.device_index)).dtype(at::kByte);
    entry.reflection_trace_raygen_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.reflection_trace_miss_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.reflection_trace_hitgroup_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(entry.device_index).stream();
    copy_sbt_record(entry.reflection_trace_raygen_group, entry.reflection_trace_raygen_record, stream);
    copy_sbt_record(entry.reflection_trace_miss_group, entry.reflection_trace_miss_record, stream);
    copy_sbt_record(entry.reflection_trace_hitgroup, entry.reflection_trace_hitgroup_record, stream);

    entry.reflection_trace_sbt = {};
    entry.reflection_trace_sbt.raygenRecord =
        reinterpret_cast<CUdeviceptr>(entry.reflection_trace_raygen_record.data_ptr<uint8_t>());
    entry.reflection_trace_sbt.missRecordBase =
        reinterpret_cast<CUdeviceptr>(entry.reflection_trace_miss_record.data_ptr<uint8_t>());
    entry.reflection_trace_sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    entry.reflection_trace_sbt.missRecordCount = 1;
    entry.reflection_trace_sbt.hitgroupRecordBase =
        reinterpret_cast<CUdeviceptr>(entry.reflection_trace_hitgroup_record.data_ptr<uint8_t>());
    entry.reflection_trace_sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    entry.reflection_trace_sbt.hitgroupRecordCount = 1;
    cuda_check(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize(reflection trace SBT initialization)");
}

void optix_check(OptixResult result, const char *expr, const char *file, int line) {
    if (result == OPTIX_SUCCESS)
        return;
    throw std::runtime_error(
        std::string("OptiX error in ") + expr + " at " + file + ":" + std::to_string(line) +
        " code=" + std::to_string(static_cast<int>(result)));
}

} // namespace rayd::torch_backend
