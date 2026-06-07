#include <raydtorch/common/optix_context.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <raydtorch/edge_optix_ptx.h>
#include <raydtorch/optix_intersect_ptx.h>
#include <raydtorch/reflection_trace_optix_ptx.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace raydtorch {

namespace {
std::mutex context_mutex;
std::unordered_map<int, OptixDeviceContextEntry> contexts;

struct EmptySbtData {
};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using EmptySbtRecord = SbtRecord<EmptySbtData>;

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void copy_sbt_record(
    OptixProgramGroup program_group,
    at::Tensor &device_record,
    cudaStream_t stream) {
    EmptySbtRecord host_record = {};
    raydtorch_OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &host_record));
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
    OptixProgramGroup topk_group,
    at::Tensor &device_records,
    cudaStream_t stream) {
    EmptySbtRecord host_records[3] = {};
    raydtorch_OPTIX_CHECK(optixSbtRecordPackHeader(point_group, &host_records[0]));
    raydtorch_OPTIX_CHECK(optixSbtRecordPackHeader(ray_group, &host_records[1]));
    raydtorch_OPTIX_CHECK(optixSbtRecordPackHeader(topk_group, &host_records[2]));
    cuda_check(
        cudaMemcpyAsync(
            device_records.data_ptr<uint8_t>(),
            host_records,
            sizeof(host_records),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(edge hitgroup SBT records)");
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
} // namespace

TorchCudaContext current_torch_cuda_context() {
    TorchCudaContext out;
    out.device_index = c10::cuda::current_device();
    out.stream = at::cuda::getCurrentCUDAStream(out.device_index).stream();
    return out;
}

OptixDeviceContextEntry &get_optix_context(int device_index) {
    std::lock_guard<std::mutex> lock(context_mutex);
    auto it = contexts.find(device_index);
    if (it != contexts.end())
        return it->second;

    c10::cuda::CUDAGuard guard(device_index);
    CUcontext cu_ctx = nullptr;
    CUresult cu_result = cuCtxGetCurrent(&cu_ctx);
    if (cu_result != CUDA_SUCCESS || cu_ctx == nullptr)
        throw std::runtime_error("Could not get current CUDA context for OptiX.");

    OptixDeviceContext optix_ctx = nullptr;
    raydtorch_OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    raydtorch_OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &optix_ctx));

    OptixDeviceContextEntry entry;
    entry.device_index = device_index;
    entry.cuda_context = cu_ctx;
    entry.optix_context = optix_ctx;
    auto [inserted, _] = contexts.emplace(device_index, entry);
    return inserted->second;
}

void ensure_intersect_pipeline(OptixDeviceContextEntry &entry) {
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
    pipeline_options.numPayloadValues = 5;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";

    char log[8192] = {};
    size_t log_size = sizeof(log);
    raydtorch_OPTIX_CHECK(optixModuleCreate(
        entry.optix_context,
        &module_options,
        &pipeline_options,
        raydtorch_optix_intersect_ptx,
        sizeof(raydtorch_optix_intersect_ptx),
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
    raydtorch_OPTIX_CHECK(optixPipelineCreate(
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
        raydtorch_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack_sizes, entry.intersect_pipeline));
    uint32_t direct_callable_stack_from_traversal = 0;
    uint32_t direct_callable_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    raydtorch_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        1,
        0,
        1,
        &direct_callable_stack_from_traversal,
        &direct_callable_stack_from_state,
        &continuation_stack));
    raydtorch_OPTIX_CHECK(optixPipelineSetStackSize(
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
}

void ensure_edge_pipeline(OptixDeviceContextEntry &entry) {
    if (entry.edge_pipeline != nullptr)
        return;

    c10::cuda::CUDAGuard guard(entry.device_index);

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 16;
    pipeline_options.numAttributeValues = 3;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

    char log[8192] = {};
    size_t log_size = sizeof(log);
    raydtorch_OPTIX_CHECK(optixModuleCreate(
        entry.optix_context,
        &module_options,
        &pipeline_options,
        raydtorch_edge_optix_ptx,
        sizeof(raydtorch_edge_optix_ptx),
        log,
        &log_size,
        &entry.edge_module));

    auto make_raygen_group = [&](const char *entry_name, OptixProgramGroup *out_group) {
        OptixProgramGroupDesc raygen_desc = {};
        raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_desc.raygen.module = entry.edge_module;
        raygen_desc.raygen.entryFunctionName = entry_name;
        create_program_group(entry.optix_context, raygen_desc, out_group);
    };
    make_raygen_group("__raygen__edge_point", &entry.edge_raygen_point_group);
    make_raygen_group("__raygen__edge_ray", &entry.edge_raygen_ray_group);
    make_raygen_group("__raygen__edge_topk_point", &entry.edge_raygen_topk_group);

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = entry.edge_module;
    miss_desc.miss.entryFunctionName = "__miss__edge_query";
    create_program_group(entry.optix_context, miss_desc, &entry.edge_miss_group);

    auto make_hitgroup = [&](
                             const char *closesthit,
                             const char *anyhit,
                             const char *intersection,
                             OptixProgramGroup *out_group) {
        OptixProgramGroupDesc hitgroup_desc = {};
        hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_desc.hitgroup.moduleCH = closesthit != nullptr ? entry.edge_module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameCH = closesthit;
        hitgroup_desc.hitgroup.moduleAH = anyhit != nullptr ? entry.edge_module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameAH = anyhit;
        hitgroup_desc.hitgroup.moduleIS = intersection != nullptr ? entry.edge_module : nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameIS = intersection;
        create_program_group(entry.optix_context, hitgroup_desc, out_group);
    };
    make_hitgroup(
        "__closesthit__edge_point",
        nullptr,
        "__intersection__edge_point",
        &entry.edge_hit_point_group);
    make_hitgroup(
        nullptr,
        "__anyhit__edge_ray",
        "__intersection__edge_ray",
        &entry.edge_hit_ray_group);
    make_hitgroup(
        nullptr,
        "__anyhit__edge_topk_point",
        "__intersection__edge_topk_point",
        &entry.edge_hit_topk_group);

    OptixProgramGroup program_groups[] = {
        entry.edge_raygen_point_group,
        entry.edge_raygen_ray_group,
        entry.edge_raygen_topk_group,
        entry.edge_miss_group,
        entry.edge_hit_point_group,
        entry.edge_hit_ray_group,
        entry.edge_hit_topk_group,
    };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    raydtorch_OPTIX_CHECK(optixPipelineCreate(
        entry.optix_context,
        &pipeline_options,
        &link_options,
        program_groups,
        7,
        log,
        &log_size,
        &entry.edge_pipeline));

    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup group : program_groups)
        raydtorch_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack_sizes, entry.edge_pipeline));
    uint32_t direct_callable_stack_from_traversal = 0;
    uint32_t direct_callable_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    raydtorch_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        1,
        0,
        1,
        &direct_callable_stack_from_traversal,
        &direct_callable_stack_from_state,
        &continuation_stack));
    raydtorch_OPTIX_CHECK(optixPipelineSetStackSize(
        entry.edge_pipeline,
        direct_callable_stack_from_traversal,
        direct_callable_stack_from_state,
        continuation_stack,
        1));

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, entry.device_index)).dtype(at::kByte);
    entry.edge_raygen_point_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_raygen_ray_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_raygen_topk_record =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_miss_record = at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord))}, byte_options);
    entry.edge_hitgroup_records =
        at::empty({static_cast<int64_t>(sizeof(EmptySbtRecord) * 3)}, byte_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(entry.device_index).stream();
    copy_sbt_record(entry.edge_raygen_point_group, entry.edge_raygen_point_record, stream);
    copy_sbt_record(entry.edge_raygen_ray_group, entry.edge_raygen_ray_record, stream);
    copy_sbt_record(entry.edge_raygen_topk_group, entry.edge_raygen_topk_record, stream);
    copy_sbt_record(entry.edge_miss_group, entry.edge_miss_record, stream);
    copy_edge_hitgroup_records(
        entry.edge_hit_point_group,
        entry.edge_hit_ray_group,
        entry.edge_hit_topk_group,
        entry.edge_hitgroup_records,
        stream);

    auto init_edge_sbt = [&](OptixShaderBindingTable &sbt, const at::Tensor &raygen_record) {
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
        sbt.hitgroupRecordCount = 3;
    };
    init_edge_sbt(entry.edge_point_sbt, entry.edge_raygen_point_record);
    init_edge_sbt(entry.edge_ray_sbt, entry.edge_raygen_ray_record);
    init_edge_sbt(entry.edge_topk_sbt, entry.edge_raygen_topk_record);
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
    if (entry.reflection_trace_pipeline != nullptr)
        return;

    c10::cuda::CUDAGuard guard(entry.device_index);

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 6;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    char log[8192] = {};
    size_t log_size = sizeof(log);
    raydtorch_OPTIX_CHECK(optixModuleCreate(
        entry.optix_context,
        &module_options,
        &pipeline_options,
        raydtorch_reflection_trace_optix_ptx,
        sizeof(raydtorch_reflection_trace_optix_ptx),
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
    raydtorch_OPTIX_CHECK(optixPipelineCreate(
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
        raydtorch_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stack_sizes, entry.reflection_trace_pipeline));
    uint32_t direct_callable_stack_from_traversal = 0;
    uint32_t direct_callable_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    raydtorch_OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        1,
        0,
        1,
        &direct_callable_stack_from_traversal,
        &direct_callable_stack_from_state,
        &continuation_stack));
    raydtorch_OPTIX_CHECK(optixPipelineSetStackSize(
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
}

void optix_check(OptixResult result, const char *expr, const char *file, int line) {
    if (result == OPTIX_SUCCESS)
        return;
    throw std::runtime_error(
        std::string("OptiX error in ") + expr + " at " + file + ":" + std::to_string(line) +
        " code=" + std::to_string(static_cast<int>(result)));
}

} // namespace raydtorch
