#include "reflection_trace_host.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "reflection_trace_ptx.h"

namespace rayd {

namespace {

void check_optix(OptixResult result, const char *message) {
    if (result != 0) {
        throw std::runtime_error(std::string("OptiX error in ") + message);
    }
}

} // namespace

ReflectionTracePipeline::~ReflectionTracePipeline() {
    if (pipeline_ != nullptr && optixPipelineDestroy != nullptr) {
        optixPipelineDestroy(pipeline_);
    }
    if (pg_hitgroup_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_hitgroup_);
    }
    if (pg_miss_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_miss_);
    }
    if (pg_raygen_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_raygen_);
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
    if (sbt_raygen_record_ != nullptr) {
        jit_free(sbt_raygen_record_);
    }
}

void ReflectionTracePipeline::build(OptixDeviceContext context,
                                    int hitgroup_record_count) {
    require(context != nullptr, "ReflectionTracePipeline::build(): invalid OptiX context.");
    require(hitgroup_record_count > 0,
            "ReflectionTracePipeline::build(): hitgroup_record_count must be positive.");
    init_optix_api();

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = 6;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipeline_options.allowOpacityMicromaps = 0;

    char log[2048];
    size_t log_size = sizeof(log);
    check_optix(
        optixModuleCreate(context,
                          &module_options,
                          &pipeline_options,
                          reflection_trace_ptx,
                          reflection_trace_ptx_size,
                          log,
                          &log_size,
                          &module_),
        "optixModuleCreate(reflection_trace)");

    OptixProgramGroupOptions pg_options = {};

    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module_;
    raygen_desc.raygen.entryFunctionName = "__raygen__reflection_trace";
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &raygen_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &pg_raygen_),
                "optixProgramGroupCreate(raygen)");

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = "__miss__reflection";
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &miss_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &pg_miss_),
                "optixProgramGroupCreate(miss)");

    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__reflection";
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &hitgroup_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &pg_hitgroup_),
                "optixProgramGroupCreate(hitgroup)");

    OptixProgramGroup groups[] = { pg_raygen_, pg_miss_, pg_hitgroup_ };
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
                                    groups,
                                    3,
                                    log,
                                    &log_size,
                                    &pipeline_),
                "optixPipelineCreate(reflection_trace)");

    check_optix(optixPipelineSetStackSize(pipeline_,
                                          0,
                                          0,
                                          4096,
                                          2),
                "optixPipelineSetStackSize(reflection_trace)");

    EmptySbtRecord raygen_record = {};
    check_optix(optixSbtRecordPackHeader(pg_raygen_, &raygen_record),
                "optixSbtRecordPackHeader(raygen)");
    sbt_raygen_record_ = jit_malloc(AllocType::Device, sizeof(EmptySbtRecord));
    jit_memcpy(JitBackend::CUDA,
               sbt_raygen_record_,
               &raygen_record,
               sizeof(EmptySbtRecord));

    EmptySbtRecord miss_record = {};
    check_optix(optixSbtRecordPackHeader(pg_miss_, &miss_record),
                "optixSbtRecordPackHeader(miss)");
    sbt_miss_record_ = jit_malloc(AllocType::Device, sizeof(EmptySbtRecord));
    jit_memcpy(JitBackend::CUDA,
               sbt_miss_record_,
               &miss_record,
               sizeof(EmptySbtRecord));

    std::vector<EmptySbtRecord> hitgroup_records(static_cast<size_t>(hitgroup_record_count));
    for (EmptySbtRecord &record : hitgroup_records) {
        check_optix(optixSbtRecordPackHeader(pg_hitgroup_, &record),
                    "optixSbtRecordPackHeader(hitgroup)");
    }
    sbt_hitgroup_records_ = jit_malloc(AllocType::Device,
                                       sizeof(EmptySbtRecord) * hitgroup_records.size());
    jit_memcpy(JitBackend::CUDA,
               sbt_hitgroup_records_,
               hitgroup_records.data(),
               sizeof(EmptySbtRecord) * hitgroup_records.size());

    params_buffer_ = jit_malloc(AllocType::Device, sizeof(ReflectionTraceParams));
    hitgroup_record_count_ = hitgroup_record_count;
    ready_ = true;
}

void ReflectionTracePipeline::launch(const ReflectionTraceParams &params) const {
    require(ready_, "ReflectionTracePipeline::launch(): pipeline is not ready.");

    jit_memcpy_async(JitBackend::CUDA,
                     params_buffer_,
                     &params,
                     sizeof(ReflectionTraceParams));

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(sbt_raygen_record_);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(sbt_miss_record_);
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(sbt_hitgroup_records_);
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_record_count_);

    check_optix(optixLaunch(pipeline_,
                            jit_cuda_stream(),
                            reinterpret_cast<CUdeviceptr>(params_buffer_),
                            sizeof(ReflectionTraceParams),
                            &sbt,
                            static_cast<unsigned int>(params.n_rays),
                            1,
                            1),
                "optixLaunch(reflection_trace)");
}

} // namespace rayd
