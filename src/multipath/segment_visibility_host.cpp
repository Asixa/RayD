#include "segment_visibility_host.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "segment_visibility_ptx.h"
#include "../native_launch_audit.h"

namespace rayd {

namespace {

void check_optix(OptixResult result, const char *message) {
    if (result != 0) {
        throw std::runtime_error(std::string("OptiX error in ") + message);
    }
}

OptixProgramGroup make_raygen_group(OptixDeviceContext context,
                                    OptixModule module,
                                    const char *entry_name) {
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = entry_name;

    char log[2048];
    size_t log_size = sizeof(log);
    OptixProgramGroup group = nullptr;
    check_optix(optixProgramGroupCreate(context,
                                        &desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &group),
                "optixProgramGroupCreate(segment_visibility raygen)");
    return group;
}

void *make_sbt_record(OptixProgramGroup group) {
    EmptySbtRecord record = {};
    check_optix(optixSbtRecordPackHeader(group, &record),
                "optixSbtRecordPackHeader(segment_visibility)");

    void *device_record = jit_malloc(AllocType::Device, sizeof(EmptySbtRecord));
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA,
               device_record,
               &record,
               sizeof(EmptySbtRecord));
    return device_record;
}

} // namespace

SegmentVisibilityPipeline::~SegmentVisibilityPipeline() {
    if (pipeline_ != nullptr && optixPipelineDestroy != nullptr) {
        optixPipelineDestroy(pipeline_);
    }
    if (pg_hitgroup_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_hitgroup_);
    }
    if (pg_miss_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_miss_);
    }
    if (pg_raygen_axial_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_raygen_axial_);
    }
    if (pg_raygen_chain_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_raygen_chain_);
    }
    if (pg_raygen_pair_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_raygen_pair_);
    }
    if (pg_raygen_segment_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_raygen_segment_);
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
    if (sbt_raygen_axial_ != nullptr) {
        jit_free(sbt_raygen_axial_);
    }
    if (sbt_raygen_chain_ != nullptr) {
        jit_free(sbt_raygen_chain_);
    }
    if (sbt_raygen_pair_ != nullptr) {
        jit_free(sbt_raygen_pair_);
    }
    if (sbt_raygen_segment_ != nullptr) {
        jit_free(sbt_raygen_segment_);
    }
}

void SegmentVisibilityPipeline::build(OptixDeviceContext context,
                                      int hitgroup_record_count) {
    require(context != nullptr, "SegmentVisibilityPipeline::build(): invalid OptiX context.");
    require(hitgroup_record_count > 0,
            "SegmentVisibilityPipeline::build(): hitgroup_record_count must be positive.");
    init_optix_api();

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = 3;
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
                                  segment_visibility_ptx,
                                  segment_visibility_ptx_size,
                                  log,
                                  &log_size,
                                  &module_),
                "optixModuleCreate(segment_visibility)");

    pg_raygen_segment_ = make_raygen_group(context,
                                           module_,
                                           "__raygen__segment_visibility");
    pg_raygen_pair_ = make_raygen_group(context,
                                        module_,
                                        "__raygen__segment_pair_visibility");
    pg_raygen_axial_ = make_raygen_group(context,
                                         module_,
                                         "__raygen__axial_edge_visibility");
    pg_raygen_chain_ = make_raygen_group(context,
                                         module_,
                                         "__raygen__segment_chain_visibility");

    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = "__miss__segment_visibility";
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &miss_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &pg_miss_),
                "optixProgramGroupCreate(segment_visibility miss)");

    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__segment_visibility";
    hitgroup_desc.hitgroup.moduleAH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__segment_visibility";
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &hitgroup_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &pg_hitgroup_),
                "optixProgramGroupCreate(segment_visibility hitgroup)");

    OptixProgramGroup groups[] = {
        pg_raygen_segment_,
        pg_raygen_pair_,
        pg_raygen_axial_,
        pg_raygen_chain_,
        pg_miss_,
        pg_hitgroup_
    };
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
                                    6,
                                    log,
                                    &log_size,
                                    &pipeline_),
                "optixPipelineCreate(segment_visibility)");

    check_optix(optixPipelineSetStackSize(pipeline_,
                                          0,
                                          0,
                                          4096,
                                          2),
                "optixPipelineSetStackSize(segment_visibility)");

    sbt_raygen_segment_ = make_sbt_record(pg_raygen_segment_);
    sbt_raygen_pair_ = make_sbt_record(pg_raygen_pair_);
    sbt_raygen_axial_ = make_sbt_record(pg_raygen_axial_);
    sbt_raygen_chain_ = make_sbt_record(pg_raygen_chain_);
    sbt_miss_record_ = make_sbt_record(pg_miss_);

    std::vector<EmptySbtRecord> hitgroup_records(static_cast<size_t>(hitgroup_record_count));
    for (EmptySbtRecord &record : hitgroup_records) {
        check_optix(optixSbtRecordPackHeader(pg_hitgroup_, &record),
                    "optixSbtRecordPackHeader(segment_visibility hitgroup)");
    }
    sbt_hitgroup_records_ = jit_malloc(
        AllocType::Device,
        sizeof(EmptySbtRecord) * hitgroup_records.size());
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA,
               sbt_hitgroup_records_,
               hitgroup_records.data(),
               sizeof(EmptySbtRecord) * hitgroup_records.size());

    params_buffer_ = jit_malloc(AllocType::Device, sizeof(SegmentVisibilityParams));
    hitgroup_record_count_ = hitgroup_record_count;
    ready_ = true;
}

void *SegmentVisibilityPipeline::raygen_record(SegmentVisibilityLaunchKind kind) const {
    switch (kind) {
    case SegmentVisibilityLaunchKind::Segment:
        return sbt_raygen_segment_;
    case SegmentVisibilityLaunchKind::SegmentPair:
        return sbt_raygen_pair_;
    case SegmentVisibilityLaunchKind::AxialEdge:
        return sbt_raygen_axial_;
    case SegmentVisibilityLaunchKind::SegmentChain:
        return sbt_raygen_chain_;
    }
    return sbt_raygen_segment_;
}

void SegmentVisibilityPipeline::launch(SegmentVisibilityLaunchKind kind,
                                       const SegmentVisibilityParams &params) const {
    require(ready_, "SegmentVisibilityPipeline::launch(): pipeline is not ready.");

    audit_jit_memcpy_async();
    jit_memcpy_async(JitBackend::CUDA,
                     params_buffer_,
                     &params,
                     sizeof(SegmentVisibilityParams));

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record(kind));
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
                            sizeof(SegmentVisibilityParams),
                            &sbt,
                            static_cast<unsigned int>(params.n_rays),
                            1,
                            1),
                "optixLaunch(segment_visibility)");
}

} // namespace rayd
