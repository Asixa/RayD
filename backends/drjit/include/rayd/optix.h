#pragma once

#define RAYD_OPTIX_TARGET_VERSION 80100
#define RAYD_OPTIX_TARGET_ABI 93

#include <drjit-core/optix.h>

#include <rayd/shared/optix/scene_edge_contracts.h>

#include <string>

// Minimal host-side OptiX declarations used by RayD.
// Keep this aligned with the OptiX 8.1.0 host API subset that the project targets.
#ifndef OPTIX_VERSION
#  define OPTIX_VERSION RAYD_OPTIX_TARGET_VERSION
#endif

// =====================================================
//       Various opaque handles and enumerations
// =====================================================

using CUdeviceptr = void *;
using CUstream = void *;
using OptixPipeline = void *;
using OptixModule = void *;
using OptixProgramGroup = void *;
using OptixResult = int;
using OptixTraversableHandle = unsigned long long;
using OptixBuildOperation = int;
using OptixBuildInputType = int;
using OptixVertexFormat = int;
using OptixIndicesFormat = int;
using OptixTransformFormat = int;
using OptixAccelPropertyType = int;
using OptixProgramGroupKind = int;
using OptixDeviceProperty = int;
using OptixQueryFunctionTableOptions = int;
using OptixInstanceFlags = unsigned int;
using OptixVisibilityMask = unsigned int;

// =====================================================
//            Commonly used OptiX constants
// =====================================================

#define OPTIX_BUILD_INPUT_TYPE_TRIANGLES 0x2141
#define OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES 0x2142
#define OPTIX_BUILD_INPUT_TYPE_INSTANCES 0x2143
#define OPTIX_BUILD_OPERATION_BUILD 0x2161
#define OPTIX_BUILD_OPERATION_UPDATE 0x2162
#define OPTIX_GEOMETRY_FLAG_NONE 0
#define OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT 1
#define OPTIX_VERTEX_FORMAT_FLOAT3 0x2121
#define OPTIX_SBT_RECORD_HEADER_SIZE 32
#define OPTIX_INDICES_FORMAT_UNSIGNED_INT3 0x2103
#define OPTIX_TRANSFORM_FORMAT_NONE 0
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 0x2340
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 0x2343
#define OPTIX_COMPILE_OPTIMIZATION_DEFAULT 0
#define OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL 0x2351
#define OPTIX_COMPILE_DEBUG_LEVEL_NONE 0x2350
#define OPTIX_BUILD_FLAG_ALLOW_UPDATE 1
#define OPTIX_BUILD_FLAG_ALLOW_COMPACTION 2
#define OPTIX_BUILD_FLAG_PREFER_FAST_TRACE 4
#define OPTIX_PROPERTY_TYPE_COMPACTED_SIZE 0x2181
#define OPTIX_DEVICE_PROPERTY_RTCORE_VERSION 0x2005
#define OPTIX_QUERY_FUNCTION_TABLE_OPTION_DUMMY 0
#define OPTIX_INSTANCE_FLAG_NONE 0
#define OPTIX_EXCEPTION_FLAG_NONE 0
#define OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW 1
#define OPTIX_EXCEPTION_FLAG_TRACE_DEPTH 2
#define OPTIX_EXCEPTION_FLAG_DEBUG 8
#define OPTIX_PROGRAM_GROUP_KIND_RAYGEN 0x2421
#define OPTIX_PROGRAM_GROUP_KIND_MISS 0x2422
#define OPTIX_PROGRAM_GROUP_KIND_EXCEPTION 0x2423
#define OPTIX_PROGRAM_GROUP_KIND_HITGROUP 0x2424
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 1
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING 2
#define OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM (1u << 0)
#define OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE (1u << 31)
#define OPTIX_RAY_FLAG_DISABLE_ANYHIT 1u
#define OPTIX_RAY_FLAG_ENFORCE_ANYHIT (1u << 1)
#define OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT (1u << 2)
#define OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT (1u << 3)
#define OPTIX_SBT_RECORD_ALIGNMENT 16ull
#define OPTIX_SBT_RECORD_HEADER_SIZE 32

// =====================================================
//          Commonly used OptiX data structures
// =====================================================

struct OptixMotionOptions {
    unsigned short numKeys;
    unsigned short flags;
    float timeBegin;
    float timeEnd;
};

struct OptixAccelBuildOptions {
    unsigned int buildFlags;
    OptixBuildOperation operation;
    OptixMotionOptions motionOptions;
};

struct OptixAccelBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
    size_t tempUpdateSizeInBytes;
};

struct OptixBuildInputTriangleArray {
    const CUdeviceptr *vertexBuffers;
    unsigned int numVertices;
    OptixVertexFormat vertexFormat;
    unsigned int vertexStrideInBytes;
    CUdeviceptr indexBuffer;
    unsigned int numIndexTriplets;
    OptixIndicesFormat indexFormat;
    unsigned int indexStrideInBytes;
    CUdeviceptr preTransform;
    const unsigned int *flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
    OptixTransformFormat transformFormat;
};

struct OptixBuildInputCustomPrimitiveArray {
    const CUdeviceptr *aabbBuffers;
    unsigned int numPrimitives;
    unsigned int strideInBytes;
    const unsigned int *flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
};

struct OptixInstance {
    float transform[12];
    unsigned int instanceId;
    unsigned int sbtOffset;
    OptixVisibilityMask visibilityMask;
    OptixInstanceFlags flags;
    OptixTraversableHandle traversableHandle;
    unsigned int pad[2];
};

struct OptixBuildInputInstanceArray {
    CUdeviceptr instances;
    unsigned int numInstances;
    unsigned int instanceStride;
};

struct OptixBuildInput {
    OptixBuildInputType type;
    union {
        OptixBuildInputTriangleArray triangleArray;
        OptixBuildInputCustomPrimitiveArray customPrimitiveArray;
        OptixBuildInputInstanceArray instanceArray;
        char pad[1024];
    };
};

struct OptixPayloadType {
    unsigned int numPayloadValues;
    const unsigned int *payloadSemantics;
};

struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
    const void *boundValues;
    unsigned int numBoundValues;
    unsigned int numPayloadTypes;
    const OptixPayloadType *payloadTypes;
};

struct OptixPipelineCompileOptions {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char *pipelineLaunchParamsVariableName;
    unsigned int usesPrimitiveTypeFlags;
    int allowOpacityMicromaps;
};

// OptiX 9.1 host calls read a larger compile-options layout than Dr.Jit's
// jit_optix_configure_pipeline wrapper accepts. Keep the legacy struct above for
// Dr.Jit calls and use this only with direct optixModuleCreate/optixPipelineCreate.
struct OptixPipelineCompileOptionsDirect {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char *pipelineLaunchParamsVariableName;
    size_t pipelineLaunchParamsSizeInBytes;
    unsigned int usesPrimitiveTypeFlags;
    int allowOpacityMicromaps;
    int allowClusteredGeometry;
};

inline const OptixPipelineCompileOptions *direct_optix_pipeline_compile_options(
    const OptixPipelineCompileOptionsDirect &options) {
    return reinterpret_cast<const OptixPipelineCompileOptions *>(&options);
}

struct OptixAccelEmitDesc {
    CUdeviceptr result;
    OptixAccelPropertyType type;
};

struct OptixProgramGroupSingleModule {
    OptixModule module;
    const char *entryFunctionName;
};

struct OptixProgramGroupHitgroup {
    OptixModule moduleCH;
    const char *entryFunctionNameCH;
    OptixModule moduleAH;
    const char *entryFunctionNameAH;
    OptixModule moduleIS;
    const char *entryFunctionNameIS;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixProgramGroupOptions {
    const OptixPayloadType *payloadType;
};

struct OptixPipelineLinkOptions {
    unsigned int maxTraceDepth;
    unsigned int maxContinuationCallableDepth;
    unsigned int maxDirectCallableDepthFromState;
    unsigned int maxDirectCallableDepthFromTraversal;
    unsigned int maxTraversableGraphDepth;
};

struct OptixShaderBindingTable {
    CUdeviceptr raygenRecord;
    CUdeviceptr exceptionRecord;
    CUdeviceptr missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    CUdeviceptr hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    CUdeviceptr callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

/// Per-mesh hit-group payload stored in the SBT and read back on a hit.
struct OptixHitGroupData {
    int shape_offset;  ///< Face-offset added to the local primitive index to globalize it.
    int shape_id;      ///< Owning mesh id.
};

static_assert(rayd::shared::optix::SbtRecordAlignment == OPTIX_SBT_RECORD_ALIGNMENT);
static_assert(rayd::shared::optix::SbtRecordHeaderSize == OPTIX_SBT_RECORD_HEADER_SIZE);

template <typename T>
using SbtRecord = rayd::shared::optix::SbtRecord<T>;

using EmptySbtRecord = rayd::shared::optix::EmptySbtRecord;

using MissSbtRecord = EmptySbtRecord;
using HitGroupSbtRecord = SbtRecord<OptixHitGroupData>;

// =====================================================
//             Commonly used OptiX functions
// =====================================================

#if defined(OPTIX_STUBS_IMPL)
#  define D(name, ...) OptixResult (*name)(__VA_ARGS__) = nullptr;
#else
#  define D(name, ...) extern OptixResult (*name)(__VA_ARGS__)
#endif

D(optixAccelComputeMemoryUsage, OptixDeviceContext,
  const OptixAccelBuildOptions *, const OptixBuildInput *, unsigned int,
  OptixAccelBufferSizes *);
D(optixAccelBuild, OptixDeviceContext, CUstream, const OptixAccelBuildOptions *,
  const OptixBuildInput *, unsigned int, CUdeviceptr, size_t, CUdeviceptr,
  size_t, OptixTraversableHandle *, const OptixAccelEmitDesc *, unsigned int);
D(optixModuleCreate, OptixDeviceContext,
  const OptixModuleCompileOptions *, const OptixPipelineCompileOptions *,
  const char *, size_t, char *, size_t *, OptixModule *);
D(optixDeviceContextGetProperty, OptixDeviceContext, OptixDeviceProperty, void *, size_t);
D(optixModuleDestroy, OptixModule);
D(optixProgramGroupCreate, OptixDeviceContext, const OptixProgramGroupDesc *,
  unsigned int, const OptixProgramGroupOptions *, char *, size_t *,
  OptixProgramGroup *);
D(optixProgramGroupDestroy, OptixProgramGroup);
D(optixPipelineCreate, OptixDeviceContext,
  const OptixPipelineCompileOptions *, const OptixPipelineLinkOptions *,
  const OptixProgramGroup *, unsigned int, char *, size_t *, OptixPipeline *);
D(optixPipelineDestroy, OptixPipeline);
D(optixPipelineSetStackSize, OptixPipeline, unsigned int, unsigned int,
  unsigned int, unsigned int);
D(optixSbtRecordPackHeader, OptixProgramGroup, void *);
D(optixLaunch, OptixPipeline, CUstream, CUdeviceptr, size_t,
  const OptixShaderBindingTable *, unsigned int, unsigned int, unsigned int);
D(optixAccelCompact, OptixDeviceContext, CUstream, OptixTraversableHandle,
  CUdeviceptr, size_t, OptixTraversableHandle *);

#undef D

/// Resolve the OptiX host entry points (the D(...) function pointers above) from the driver.
extern void init_optix_api();

namespace rayd {

/// Snapshot of the loaded OptiX runtime: which entry points resolved, the ABI/RTcore
/// versions probed, and the on-disk driver module that backs them.
struct OptixRuntimeInfo {
    int target_version = RAYD_OPTIX_TARGET_VERSION; ///< OptiX version RayD was built against.
    int target_abi = RAYD_OPTIX_TARGET_ABI;         ///< OptiX ABI RayD requests from the driver.
    bool module_create_available = false;            ///< optixModuleCreate resolved.
    bool device_context_get_property_available = false; ///< optixDeviceContextGetProperty resolved.
    bool query_function_table_available = false;     ///< Driver exposes optixQueryFunctionTable.
    bool target_abi_supported = false;               ///< Driver accepts the target ABI.
    int abi_probe_result = 0;                         ///< Raw result code from the ABI probe.
    int rtcore_version = -1;                           ///< RT core version, or -1 if unavailable.
    std::string module_path;                          ///< Path to the resolved OptiX driver module.
    std::string module_version;                        ///< Version string of that module.
};

/// Probe the active OptiX driver and report what resolved; initializes the OptiX API as a side effect.
OptixRuntimeInfo query_optix_runtime_info();

// Shared OptiX host helpers used by the multipath and edge pipelines.

/// Throw std::runtime_error tagged with \p message when \p result is not OPTIX_SUCCESS.
void check_optix(OptixResult result, const char *message);
/// Create a ray-generation program group for \p entry_name in \p module.
OptixProgramGroup make_raygen_group(OptixDeviceContext context,
                                    OptixModule module,
                                    const char *entry_name);
/// Create a miss program group for \p entry_name in \p module.
OptixProgramGroup make_miss_group(OptixDeviceContext context,
                                  OptixModule module,
                                  const char *entry_name);
/// Create a hit-group program group; any of the entry-point names may be null to omit that stage.
OptixProgramGroup make_hitgroup(OptixDeviceContext context,
                                OptixModule module,
                                const char *closesthit,
                                const char *anyhit,
                                const char *intersection);
/// Allocate and upload a header-only SBT record for \p group; returns the owning device pointer.
void *make_sbt_record(OptixProgramGroup group);

} // namespace rayd
