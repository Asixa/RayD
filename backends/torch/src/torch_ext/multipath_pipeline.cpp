#include <raydtorch/multipath_pipeline.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <raydtorch/optix_context.h>
#include <raydtorch/diffraction_accumulation_optix_ptx.h>
#include <raydtorch/diffraction_paths_optix_ptx.h>
#include <raydtorch/reflection_accumulation_optix_ptx.h>
#include <raydtorch/reflection_epc_optix_ptx.h>
#include <raydtorch/reflection_trace_optix_ptx.h>
#include <raydtorch/segment_visibility_optix_ptx.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace raydtorch {

namespace {

struct EmptySbtData {
};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using EmptySbtRecord = SbtRecord<EmptySbtData>;

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
    raydtorch_OPTIX_CHECK(
        optixProgramGroupCreate(context, &desc, 1, &options, log, &log_size, out_group));
}

at::Tensor make_sbt_record(
    int device_index,
    OptixProgramGroup program_group,
    cudaStream_t stream) {
    EmptySbtRecord host_record = {};
    raydtorch_OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &host_record));
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
    constexpr int kMinHitgroupRecordCapacity = 64;
    int capacity = kMinHitgroupRecordCapacity;
    while (capacity < hitgroup_record_count)
        capacity *= 2;
    return capacity;
}

} // namespace

OptixLaunchPipeline::~OptixLaunchPipeline() {
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
    pipeline_options.numAttributeValues = 2;
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
    link_options.maxContinuationCallableDepth = 0;
    link_options.maxDirectCallableDepthFromState = 0;
    link_options.maxDirectCallableDepthFromTraversal = 0;
    link_options.maxTraversableGraphDepth = 2;

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
    raydtorch_OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, 0, 0, 4096, 2));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    for (OptixProgramGroup group : raygen_groups_)
        raygen_records_.push_back(make_sbt_record(device_index, group, stream));
    miss_record_ = make_sbt_record(device_index, miss_group_, stream);

    std::vector<EmptySbtRecord> hitgroup_host(static_cast<size_t>(hitgroup_record_count));
    for (EmptySbtRecord &record : hitgroup_host)
        raydtorch_OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_, &record));
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
    params_buffer_ = at::empty(
        {static_cast<int64_t>(std::max<size_t>(params_size_, 1024))},
        at::TensorOptions().device(at::Device(at::kCUDA, device_index)).dtype(at::kByte));
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
    if (!ready_)
        throw std::runtime_error("OptixLaunchPipeline::launch(): pipeline is not ready.");
    if (raygen_index < 0 || raygen_index >= static_cast<int>(raygen_records_.size()))
        throw std::runtime_error("OptixLaunchPipeline::launch(): raygen index out of range.");
    const size_t launch_params_size = (std::max)(params_size_, actual_params_size);
    if (params_buffer_.numel() < static_cast<int64_t>(launch_params_size))
        throw std::runtime_error("OptixLaunchPipeline::launch(): params buffer is too small.");

    cuda_check(
        cudaMemcpyAsync(
            params_buffer_.data_ptr<uint8_t>(),
            params,
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

    raydtorch_OPTIX_CHECK(optixLaunch(
        pipeline_,
        stream,
        reinterpret_cast<CUdeviceptr>(params_buffer_.data_ptr<uint8_t>()),
        launch_params_size,
        &sbt,
        n_rays,
        1,
        1));
}

OptixPipelineConfig reflection_trace_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_reflection_trace_optix_ptx;
    config.ptx_size = sizeof(raydtorch_reflection_trace_optix_ptx);
    config.raygen_entries = {"__raygen__reflection_trace"};
    config.miss_entry = "__miss__reflection";
    config.closesthit_entry = "__closesthit__reflection";
    config.num_payload_values = 6;
    config.params_size = sizeof(ReflectionTraceParams);
    return config;
}

OptixPipelineConfig segment_visibility_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_segment_visibility_optix_ptx;
    config.ptx_size = sizeof(raydtorch_segment_visibility_optix_ptx);
    config.raygen_entries = {
        "__raygen__segment_visibility",
        "__raygen__segment_pair_visibility",
        "__raygen__axial_edge_visibility",
        "__raygen__segment_chain_visibility",
    };
    config.miss_entry = "__miss__segment_visibility";
    config.closesthit_entry = "__closesthit__segment_visibility";
    config.anyhit_entry = "__anyhit__segment_visibility";
    config.num_payload_values = 3;
    config.params_size = sizeof(SegmentVisibilityParams);
    return config;
}

OptixPipelineConfig reflection_epc_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_reflection_epc_optix_ptx;
    config.ptx_size = sizeof(raydtorch_reflection_epc_optix_ptx);
    config.raygen_entries = {
        "__raygen__reflection_epc",
        "__raygen__reflection_epc_direct",
        "__raygen__reflection_epc_direct_primary",
    };
    config.miss_entry = "__miss__reflection_epc";
    config.closesthit_entry = "__closesthit__reflection_epc";
    config.anyhit_entry = "__anyhit__reflection_epc";
    config.num_payload_values = 6;
    config.params_size = sizeof(ReflEpcParams);
    return config;
}

OptixPipelineConfig reflection_accumulation_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_reflection_accumulation_optix_ptx;
    config.ptx_size = sizeof(raydtorch_reflection_accumulation_optix_ptx);
    config.raygen_entries = {"__raygen__reflection_accumulation"};
    config.miss_entry = "__miss__reflection_accumulation";
    config.closesthit_entry = "__closesthit__reflection_accumulation";
    config.num_payload_values = 6;
    config.params_size = sizeof(AccumParams);
    return config;
}

OptixPipelineConfig diffraction_paths_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_diffraction_paths_optix_ptx;
    config.ptx_size = sizeof(raydtorch_diffraction_paths_optix_ptx);
    config.raygen_entries = {
        "__raygen__diffraction_paths_order1_primary",
        "__raygen__diffraction_paths_order1",
        "__raygen__diffraction_paths_order1_source_visibility_primary",
        "__raygen__diffraction_paths_order1_target_export_primary",
    };
    config.miss_entry = "__miss__diffraction_paths";
    config.closesthit_entry = "__closesthit__diffraction_paths";
    config.num_payload_values = 4;
    config.params_size = sizeof(DfrPathParams);
    return config;
}

OptixPipelineConfig diffraction_accumulation_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = raydtorch_diffraction_accumulation_optix_ptx;
    config.ptx_size = sizeof(raydtorch_diffraction_accumulation_optix_ptx);
    config.raygen_entries = {
        "__raygen__diffraction_order1_accumulation",
        "__raygen__diffraction_order1_accumulation_primary",
        "__raygen__diffraction_order1_accumulation_no_suffix",
        "__raygen__diffraction_order1_accumulation_no_suffix_primary",
        "__raygen__diffraction_order1_accumulation_suffix",
        "__raygen__diffraction_order1_accumulation_suffix_primary",
        "__raygen__diffraction_order1_source_visibility_primary",
        "__raygen__diffraction_order1_no_suffix_target_accumulation_primary",
        "__raygen__diffraction_order1_suffix_first_visibility_primary",
        "__raygen__diffraction_order1_suffix_target_accumulation_primary",
        "__raygen__diffraction_order1_coherent_accumulation",
        "__raygen__diffraction_order1_coherent_accumulation_primary",
        "__raygen__diffraction_chain_accumulation",
        "__raygen__diffraction_chain_accumulation_primary",
    };
    config.miss_entry = "__miss__diffraction_accumulation";
    config.closesthit_entry = "__closesthit__diffraction_accumulation";
    config.num_payload_values = 4;
    config.params_size = sizeof(DfrAccumParams);
    return config;
}

} // namespace raydtorch
