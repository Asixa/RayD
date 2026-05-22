#include <rayd/multipath/pipelines.h>

#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include <rayd/native_launch_audit.h>

#include <rayd/multipath/reflection_trace_ptx.h>
#include <rayd/multipath/reflection_epc_ptx.h>
#include <rayd/multipath/reflection_accumulation_ptx.h>
#include <rayd/multipath/diffraction_accumulation_ptx.h>
#include <rayd/multipath/diffraction_paths_ptx.h>
#include <rayd/multipath/segment_visibility_ptx.h>
#include <rayd/multipath/reflection_trace_params.h>
#include <rayd/multipath/reflection_epc_params.h>
#include <rayd/multipath/reflection_accumulation_params.h>
#include <rayd/multipath/diffraction_accumulation_params.h>
#include <rayd/multipath/diffraction_paths_params.h>
#include <rayd/multipath/segment_visibility_params.h>

namespace rayd {

namespace {

using PipelineCacheKey = std::tuple<
    OptixDeviceContext,
    const char *,
    const char *,
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

int hitgroup_record_capacity(int hitgroup_record_count) {
    constexpr int kMinHitgroupRecordCapacity = 64;
    int capacity = kMinHitgroupRecordCapacity;
    while (capacity < hitgroup_record_count) {
        capacity *= 2;
    }
    return capacity;
}

} // namespace

OptixLaunchPipeline::~OptixLaunchPipeline() {
    if (pipeline_ != nullptr && optixPipelineDestroy != nullptr) {
        optixPipelineDestroy(pipeline_);
    }
    if (pg_hitgroup_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_hitgroup_);
    }
    if (pg_miss_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_miss_);
    }
    for (OptixProgramGroup pg : pg_raygens_) {
        if (pg != nullptr && optixProgramGroupDestroy != nullptr) {
            optixProgramGroupDestroy(pg);
        }
    }
    if (module_ != nullptr && optixModuleDestroy != nullptr) {
        optixModuleDestroy(module_);
    }
    if (params_buffer_ != nullptr) {
        jit_free(params_buffer_);
    }
    if (sbt_hitgroup_records_ != nullptr) {
        jit_free(sbt_hitgroup_records_);
    }
    if (sbt_miss_record_ != nullptr) {
        jit_free(sbt_miss_record_);
    }
    for (void *record : sbt_raygen_records_) {
        if (record != nullptr) {
            jit_free(record);
        }
    }
}

/// Compile the module, create program groups, link the pipeline, and build the SBT and
/// params buffer from \p config. The shared build sequence for all four multipath pipelines.
void OptixLaunchPipeline::build(OptixDeviceContext context,
                                int hitgroup_record_count,
                                const OptixPipelineConfig &config) {
    require(context != nullptr, "OptixLaunchPipeline::build(): invalid OptiX context.");
    require(hitgroup_record_count > 0,
            "OptixLaunchPipeline::build(): hitgroup_record_count must be positive.");
    require(!config.raygen_entries.empty(),
            "OptixLaunchPipeline::build(): config requires at least one raygen entry.");
    init_optix_api();

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = config.num_payload_values;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipeline_options.allowOpacityMicromaps = 0;

    char log[2048];
    size_t log_size = sizeof(log);
    check_optix(optixModuleCreate(context,
                                  &module_options,
                                  &pipeline_options,
                                  config.ptx,
                                  config.ptx_size,
                                  log,
                                  &log_size,
                                  &module_),
                "optixModuleCreate(multipath)");

    for (const char *entry : config.raygen_entries) {
        pg_raygens_.push_back(make_raygen_group(context, module_, entry));
    }
    pg_miss_ = make_miss_group(context, module_, config.miss_entry);
    pg_hitgroup_ = make_hitgroup(context, module_, config.closesthit_entry,
                                 config.anyhit_entry, nullptr);

    std::vector<OptixProgramGroup> groups = pg_raygens_;
    groups.push_back(pg_miss_);
    groups.push_back(pg_hitgroup_);

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    link_options.maxContinuationCallableDepth = 0;
    link_options.maxDirectCallableDepthFromState = 0;
    link_options.maxDirectCallableDepthFromTraversal = 0;
    link_options.maxTraversableGraphDepth = 2;

    log_size = sizeof(log);
    check_optix(optixPipelineCreate(context,
                                    &pipeline_options,
                                    &link_options,
                                    groups.data(),
                                    static_cast<unsigned int>(groups.size()),
                                    log,
                                    &log_size,
                                    &pipeline_),
                "optixPipelineCreate(multipath)");

    check_optix(optixPipelineSetStackSize(pipeline_, 0, 0, 4096, 2),
                "optixPipelineSetStackSize(multipath)");

    for (OptixProgramGroup pg : pg_raygens_) {
        sbt_raygen_records_.push_back(make_sbt_record(pg));
    }
    sbt_miss_record_ = make_sbt_record(pg_miss_);

    std::vector<EmptySbtRecord> hitgroup_records(static_cast<size_t>(hitgroup_record_count));
    for (EmptySbtRecord &record : hitgroup_records) {
        check_optix(optixSbtRecordPackHeader(pg_hitgroup_, &record),
                    "optixSbtRecordPackHeader(hitgroup)");
    }
    sbt_hitgroup_records_ = jit_malloc(AllocType::Device,
                                       sizeof(EmptySbtRecord) * hitgroup_records.size());
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA,
               sbt_hitgroup_records_,
               hitgroup_records.data(),
               sizeof(EmptySbtRecord) * hitgroup_records.size());

    for (OptixProgramGroup &pg : pg_raygens_) {
        if (pg != nullptr) {
            check_optix(optixProgramGroupDestroy(pg),
                        "optixProgramGroupDestroy(raygen)");
            pg = nullptr;
        }
    }
    if (pg_hitgroup_ != nullptr) {
        check_optix(optixProgramGroupDestroy(pg_hitgroup_),
                    "optixProgramGroupDestroy(hitgroup)");
        pg_hitgroup_ = nullptr;
    }
    if (pg_miss_ != nullptr) {
        check_optix(optixProgramGroupDestroy(pg_miss_),
                    "optixProgramGroupDestroy(miss)");
        pg_miss_ = nullptr;
    }
    if (module_ != nullptr) {
        check_optix(optixModuleDestroy(module_), "optixModuleDestroy(multipath)");
        module_ = nullptr;
    }

    params_size_ = config.params_size;
    params_buffer_ = jit_malloc(AllocType::Device, params_size_);
    hitgroup_record_count_ = hitgroup_record_count;
    ready_ = true;
}

std::shared_ptr<OptixLaunchPipeline> shared_optix_launch_pipeline(
    OptixDeviceContext context,
    int hitgroup_record_count,
    const OptixPipelineConfig &config) {
    int hitgroup_capacity = hitgroup_record_capacity(hitgroup_record_count);
    PipelineCacheKey key{
        context,
        config.ptx,
        config.raygen_entries.empty() ? nullptr : config.raygen_entries.front(),
        hitgroup_capacity,
        config.num_payload_values,
        config.params_size,
    };

    std::lock_guard<std::mutex> guard(pipeline_cache_mutex());
    auto &cache = pipeline_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    auto pipeline = std::make_shared<OptixLaunchPipeline>();
    pipeline->build(context, hitgroup_capacity, config);
    cache[key] = pipeline;
    return pipeline;
}

/// Upload \p params and launch the pipeline with the \p raygen_index'th raygen entry over n_rays threads.
void OptixLaunchPipeline::launch_impl(int raygen_index,
                                      const void *params,
                                      unsigned int n_rays) const {
    require(ready_, "OptixLaunchPipeline::launch(): pipeline is not ready.");
    require(raygen_index >= 0 &&
                raygen_index < static_cast<int>(sbt_raygen_records_.size()),
            "OptixLaunchPipeline::launch(): raygen index out of range.");

    audit_jit_memcpy_async();
    jit_memcpy_async(JitBackend::CUDA, params_buffer_, params, params_size_);

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(sbt_raygen_records_[raygen_index]);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(sbt_miss_record_);
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(sbt_hitgroup_records_);
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_record_count_);

    audit_optix_launch();
    check_optix(optixLaunch(pipeline_,
                            jit_cuda_stream(),
                            reinterpret_cast<CUdeviceptr>(params_buffer_),
                            params_size_,
                            &sbt,
                            n_rays,
                            1,
                            1),
                "optixLaunch(multipath)");
}

OptixPipelineConfig reflection_trace_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = reflection_trace_ptx;
    config.ptx_size = reflection_trace_ptx_size;
    config.raygen_entries = {"__raygen__reflection_trace"};
    config.miss_entry = "__miss__reflection";
    config.closesthit_entry = "__closesthit__reflection";
    config.num_payload_values = 6;
    config.params_size = sizeof(ReflectionTraceParams);
    return config;
}

OptixPipelineConfig reflection_epc_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = reflection_epc_ptx;
    config.ptx_size = reflection_epc_ptx_size;
    config.raygen_entries = {"__raygen__reflection_epc"};
    config.miss_entry = "__miss__reflection_epc";
    config.closesthit_entry = "__closesthit__reflection_epc";
    config.anyhit_entry = "__anyhit__reflection_epc";
    config.num_payload_values = 6;
    config.params_size = sizeof(ReflectionEpcParams);
    return config;
}

OptixPipelineConfig reflection_accumulation_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = reflection_accumulation_ptx;
    config.ptx_size = reflection_accumulation_ptx_size;
    config.raygen_entries = {"__raygen__reflection_accumulation"};
    config.miss_entry = "__miss__reflection_accumulation";
    config.closesthit_entry = "__closesthit__reflection_accumulation";
    config.num_payload_values = 6;
    config.params_size = sizeof(AccumParams);
    return config;
}

OptixPipelineConfig diffraction_accumulation_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = diffraction_accumulation_ptx;
    config.ptx_size = diffraction_accumulation_ptx_size;
    config.raygen_entries = {
        "__raygen__diffraction_order1_accumulation",
        "__raygen__diffraction_chain_accumulation",
    };
    config.miss_entry = "__miss__diffraction_accumulation";
    config.closesthit_entry = "__closesthit__diffraction_accumulation";
    config.num_payload_values = 4;
    config.params_size = sizeof(DiffractionAccumParams);
    return config;
}

OptixPipelineConfig diffraction_paths_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = diffraction_paths_ptx;
    config.ptx_size = diffraction_paths_ptx_size;
    config.raygen_entries = {"__raygen__diffraction_paths_order1"};
    config.miss_entry = "__miss__diffraction_paths";
    config.closesthit_entry = "__closesthit__diffraction_paths";
    config.num_payload_values = 4;
    config.params_size = sizeof(DiffractionPathParams);
    return config;
}

OptixPipelineConfig segment_visibility_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = segment_visibility_ptx;
    config.ptx_size = segment_visibility_ptx_size;
    config.raygen_entries = {"__raygen__segment_visibility"};
    config.miss_entry = "__miss__segment_visibility";
    config.closesthit_entry = "__closesthit__segment_visibility";
    config.anyhit_entry = "__anyhit__segment_visibility";
    config.num_payload_values = 3;
    config.params_size = sizeof(SegmentVisibilityParams);
    return config;
}

OptixPipelineConfig segment_pair_visibility_pipeline_config() {
    OptixPipelineConfig config = segment_visibility_pipeline_config();
    config.raygen_entries = {"__raygen__segment_pair_visibility"};
    return config;
}

OptixPipelineConfig axial_edge_visibility_pipeline_config() {
    OptixPipelineConfig config = segment_visibility_pipeline_config();
    config.raygen_entries = {"__raygen__axial_edge_visibility"};
    return config;
}

OptixPipelineConfig segment_chain_visibility_pipeline_config() {
    OptixPipelineConfig config = segment_visibility_pipeline_config();
    config.raygen_entries = {"__raygen__segment_chain_visibility"};
    return config;
}

} // namespace rayd
