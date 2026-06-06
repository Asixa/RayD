#include <raydtorch/optix_context.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <raydtorch/optix_intersect_ptx.h>

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
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 4;
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
        1));

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

void optix_check(OptixResult result, const char *expr, const char *file, int line) {
    if (result == OPTIX_SUCCESS)
        return;
    throw std::runtime_error(
        std::string("OptiX error in ") + expr + " at " + file + ":" + std::to_string(line) +
        " code=" + std::to_string(static_cast<int>(result)));
}

} // namespace raydtorch
