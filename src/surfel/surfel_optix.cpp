#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include <rayd/native_launch_audit.h>
#include <rayd/surfel/surfel_optix.h>
#include <rayd/surfel/surfel_trace_params.h>
// Embedded native surfel OptiX programs: analytic first-hit tracing and k-buffer compositing.
#include <rayd/surfel/surfel_trace_ptx.h>

namespace rayd {

namespace dr = drjit;

#ifndef RAYD_OPTIX_EXCEPTION_FLAGS
#  define RAYD_OPTIX_EXCEPTION_FLAGS OPTIX_EXCEPTION_FLAG_NONE
#endif

#ifndef RAYD_OPTIX_MODULE_OPT_LEVEL
#  define RAYD_OPTIX_MODULE_OPT_LEVEL OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
#endif

namespace {

enum class SurfelOptixLaunchKind {
    Intersect,
    Composite
};

struct RetiredSurfelOptixJitResources {
    UInt pipeline_handle;
    UInt sbt_handle;
};

std::mutex &retired_surfel_optix_resources_mutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

std::vector<RetiredSurfelOptixJitResources> &retired_surfel_optix_resources() {
    static std::vector<RetiredSurfelOptixJitResources> *resources =
        new std::vector<RetiredSurfelOptixJitResources>();
    return *resources;
}

} // namespace

struct SurfelOptixState {
    OptixDeviceContext context = 0;

    UInt64 handle;
    UInt pipeline_handle;
    UInt sbt_handle;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc pgd[2] = {};
    OptixProgramGroup pg[2] = {};
    OptixShaderBindingTable sbt = {};

    void *vertex_buffer = nullptr;
    void *vertex_buffer_ptr = nullptr;
    void *gas_temp_buffer = nullptr;
    size_t gas_temp_buffer_size = 0;
    void *gas_buffer = nullptr;
    size_t gas_buffer_size = 0;
    OptixTraversableHandle gas_handle = 0;
    OptixAccelBuildOptions accel_options = {};
    OptixAccelBufferSizes gas_buffer_sizes = {};

    OptixModule trace_module = nullptr;
    OptixPipeline trace_pipeline = nullptr;
    OptixProgramGroup trace_pg_raygen = nullptr;
    OptixProgramGroup trace_pg_raygen_composite = nullptr;
    OptixProgramGroup trace_pg_miss = nullptr;
    OptixProgramGroup trace_pg_hit_composite = nullptr;
    OptixProgramGroup trace_pg_hit_intersect = nullptr;
    void *trace_sbt_raygen = nullptr;
    void *trace_sbt_raygen_composite = nullptr;
    void *trace_sbt_miss = nullptr;
    void *trace_sbt_hitgroup = nullptr;
    void *trace_params_buffer = nullptr;

    int vertex_count = 0;
    int triangle_count = 0;

    void *trace_raygen_record(SurfelOptixLaunchKind kind) const {
        return kind == SurfelOptixLaunchKind::Composite
            ? trace_sbt_raygen_composite
            : trace_sbt_raygen;
    }

    void launch_trace(SurfelOptixLaunchKind kind, const SurfelTraceParams &params) const {
        audit_jit_memcpy_async();
        jit_memcpy_async(JitBackend::CUDA,
                         trace_params_buffer,
                         &params,
                         sizeof(SurfelTraceParams));

        OptixShaderBindingTable trace_sbt = {};
        trace_sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(trace_raygen_record(kind));
        trace_sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(trace_sbt_miss);
        trace_sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
        trace_sbt.missRecordCount = 1;
        trace_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(trace_sbt_hitgroup);
        trace_sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        trace_sbt.hitgroupRecordCount = 2;

        audit_optix_launch();
        check_optix(optixLaunch(trace_pipeline,
                                jit_cuda_stream(),
                                reinterpret_cast<CUdeviceptr>(trace_params_buffer),
                                sizeof(SurfelTraceParams),
                                &trace_sbt,
                                static_cast<unsigned int>(params.ray_count),
                                1,
                                1),
                    "optixLaunch(surfel trace)");
    }
};

static void retire_surfel_optix_jit_resources(SurfelOptixState *state) {
    if (state == nullptr || state->pipeline_handle.index() == 0 || state->sbt_handle.index() == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(retired_surfel_optix_resources_mutex());
    auto &resources = retired_surfel_optix_resources();
    resources.push_back({ state->pipeline_handle, state->sbt_handle });

    state->pipeline_handle = UInt();
    state->sbt_handle = UInt();

    constexpr size_t MaxRetiredOptixResourceSets = 32;
    if (resources.size() >= MaxRetiredOptixResourceSets) {
        jit_flush_kernel_cache();
        resources.clear();
    }
}

static void destroy_surfel_optix_state(SurfelOptixState *state) {
    if (state == nullptr) {
        return;
    }

    jit_sync_thread();

    if (state->trace_pipeline != nullptr && optixPipelineDestroy != nullptr) {
        optixPipelineDestroy(state->trace_pipeline);
    }
    if (state->trace_pg_hit_intersect != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_hit_intersect);
    }
    if (state->trace_pg_hit_composite != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_hit_composite);
    }
    if (state->trace_pg_miss != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_miss);
    }
    if (state->trace_pg_raygen != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_raygen);
    }
    if (state->trace_pg_raygen_composite != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_raygen_composite);
    }
    if (state->trace_module != nullptr && optixModuleDestroy != nullptr) {
        optixModuleDestroy(state->trace_module);
    }
    if (state->trace_params_buffer != nullptr) {
        jit_free(state->trace_params_buffer);
        state->trace_params_buffer = nullptr;
    }
    if (state->trace_sbt_hitgroup != nullptr) {
        jit_free(state->trace_sbt_hitgroup);
        state->trace_sbt_hitgroup = nullptr;
    }
    if (state->trace_sbt_miss != nullptr) {
        jit_free(state->trace_sbt_miss);
        state->trace_sbt_miss = nullptr;
    }
    if (state->trace_sbt_raygen_composite != nullptr) {
        jit_free(state->trace_sbt_raygen_composite);
        state->trace_sbt_raygen_composite = nullptr;
    }
    if (state->trace_sbt_raygen != nullptr) {
        jit_free(state->trace_sbt_raygen);
        state->trace_sbt_raygen = nullptr;
    }

    retire_surfel_optix_jit_resources(state);

    if (state->vertex_buffer != nullptr) {
        jit_free(state->vertex_buffer);
        state->vertex_buffer = nullptr;
        state->vertex_buffer_ptr = nullptr;
    }
    if (state->gas_temp_buffer != nullptr) {
        jit_free(state->gas_temp_buffer);
        state->gas_temp_buffer = nullptr;
    }
    if (state->gas_buffer != nullptr) {
        jit_free(state->gas_buffer);
        state->gas_buffer = nullptr;
    }
    delete state;
}

static void ensure_surfel_trace_pipeline(SurfelOptixState *state) {
    require(state != nullptr, "ensure_surfel_trace_pipeline(): state is null.");
    if (state->trace_pipeline != nullptr) {
        return;
    }

    init_optix_api();
    state->context = jit_optix_context();

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = RAYD_OPTIX_MODULE_OPT_LEVEL;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 1;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipeline_options.allowOpacityMicromaps = 0;

    char log[2048];
    size_t log_size = sizeof(log);
    check_optix(optixModuleCreate(state->context,
                                  &module_options,
                                  &pipeline_options,
                                  surfel_trace_ptx,
                                  surfel_trace_ptx_size,
                                  log,
                                  &log_size,
                                  &state->trace_module),
                "optixModuleCreate(surfel trace)");

    state->trace_pg_raygen =
        make_raygen_group(state->context, state->trace_module, "__raygen__surfel_trace");
    state->trace_pg_raygen_composite =
        make_raygen_group(state->context, state->trace_module, "__raygen__surfel_composite");
    state->trace_pg_miss =
        make_miss_group(state->context, state->trace_module, "__miss__surfel_trace");
    state->trace_pg_hit_composite =
        make_hitgroup(state->context,
                      state->trace_module,
                      nullptr,
                      "__anyhit__surfel_composite",
                      nullptr);
    state->trace_pg_hit_intersect =
        make_hitgroup(state->context,
                      state->trace_module,
                      nullptr,
                      "__anyhit__surfel_intersect",
                      nullptr);

    OptixProgramGroup groups[] = {
        state->trace_pg_raygen,
        state->trace_pg_raygen_composite,
        state->trace_pg_miss,
        state->trace_pg_hit_composite,
        state->trace_pg_hit_intersect,
    };
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    link_options.maxContinuationCallableDepth = 0;
    link_options.maxDirectCallableDepthFromState = 0;
    link_options.maxDirectCallableDepthFromTraversal = 0;
    link_options.maxTraversableGraphDepth = 1;

    log_size = sizeof(log);
    check_optix(optixPipelineCreate(state->context,
                                    &pipeline_options,
                                    &link_options,
                                    groups,
                                    5,
                                    log,
                                    &log_size,
                                    &state->trace_pipeline),
                "optixPipelineCreate(surfel trace)");
    check_optix(optixPipelineSetStackSize(state->trace_pipeline,
                                          0,
                                          0,
                                          4096,
                                          1),
                "optixPipelineSetStackSize(surfel trace)");

    state->trace_sbt_raygen = make_sbt_record(state->trace_pg_raygen);
    state->trace_sbt_raygen_composite = make_sbt_record(state->trace_pg_raygen_composite);
    state->trace_sbt_miss = make_sbt_record(state->trace_pg_miss);
    std::vector<EmptySbtRecord> hitgroups(2);
    check_optix(optixSbtRecordPackHeader(state->trace_pg_hit_composite, &hitgroups[0]),
                "optixSbtRecordPackHeader(surfel composite hitgroup)");
    check_optix(optixSbtRecordPackHeader(state->trace_pg_hit_intersect, &hitgroups[1]),
                "optixSbtRecordPackHeader(surfel intersect hitgroup)");
    state->trace_sbt_hitgroup = jit_malloc(AllocType::Device,
                                           sizeof(EmptySbtRecord) * hitgroups.size());
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA,
               state->trace_sbt_hitgroup,
               hitgroups.data(),
               sizeof(EmptySbtRecord) * hitgroups.size());
    state->trace_params_buffer = jit_malloc(AllocType::Device, sizeof(SurfelTraceParams));
}

void SurfelOptixIntersection::reserve(int64_t size) {
    require(size >= 0, "SurfelOptixIntersection::reserve(): size must be non-negative.");
    if (size != m_size) {
        m_size = size;
        triangle_id = empty<Int>(size);
        barycentric = empty<Vector2f>(size);
        t = empty<Float>(size);
    }
}

void SurfelOptixComposite::reserve(int64_t size) {
    require(size >= 0, "SurfelOptixComposite::reserve(): size must be non-negative.");
    if (size != m_size) {
        m_size = size;
        hit_capacity = 0;
        intensity = zeros<Float>(size);
        alpha = zeros<Float>(size);
        transmittance = full<Float>(1.f, size);
        depth = full<Float>(Infinity, size);
        surfel_id = Int();
        hit_t = Float();
        hit_alpha = Float();
        hit_value = Float();
    }
}

SurfelOptixScene::SurfelOptixScene() = default;

SurfelOptixScene::~SurfelOptixScene() {
    destroy_surfel_optix_state(m_accel);
}

void SurfelOptixScene::build(const Float &vertex_buffer,
                             const Int &face_buffer,
                             int vertex_count,
                             int triangle_count,
                             bool build_hitobject_pipeline) {
    require(vertex_count > 0, "SurfelOptixScene::build(): vertex_count must be positive.");
    require(triangle_count > 0, "SurfelOptixScene::build(): triangle_count must be positive.");

    destroy_surfel_optix_state(m_accel);

    init_optix_api();
    m_accel = new SurfelOptixState();
    m_accel->context = jit_optix_context();
    m_accel->vertex_count = vertex_count;
    m_accel->triangle_count = triangle_count;

    if (build_hitobject_pipeline) {
        m_accel->pipeline_compile_options.usesMotionBlur = false;
        m_accel->pipeline_compile_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        m_accel->pipeline_compile_options.numPayloadValues = 0;
        m_accel->pipeline_compile_options.numAttributeValues = 2;
        m_accel->pipeline_compile_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
        m_accel->pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        m_accel->pipeline_compile_options.usesPrimitiveTypeFlags =
            static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
        m_accel->pipeline_compile_options.allowOpacityMicromaps = 0;

        std::memset(m_accel->pgd, 0, sizeof(m_accel->pgd));
        m_accel->pgd[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        m_accel->pgd[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        char log[1024];
        size_t log_size = sizeof(log);
        jit_optix_check(optixProgramGroupCreate(m_accel->context,
                                                m_accel->pgd,
                                                2,
                                                &m_accel->pgo,
                                                log,
                                                &log_size,
                                                m_accel->pg));

        m_accel->sbt.missRecordBase =
            reinterpret_cast<CUdeviceptr>(jit_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE));
        m_accel->sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        m_accel->sbt.missRecordCount = 1;
        jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[0],
                                                 reinterpret_cast<void *>(m_accel->sbt.missRecordBase)));

        EmptySbtRecord *hit_record =
            reinterpret_cast<EmptySbtRecord *>(jit_malloc(AllocType::HostPinned, sizeof(EmptySbtRecord)));
        jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[1], hit_record));
        m_accel->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hit_record);
        m_accel->sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        m_accel->sbt.hitgroupRecordCount = 1;

        m_accel->sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(
            jit_malloc_migrate(reinterpret_cast<void *>(m_accel->sbt.missRecordBase), AllocType::Device, 1));
        m_accel->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(
            jit_malloc_migrate(reinterpret_cast<void *>(m_accel->sbt.hitgroupRecordBase), AllocType::Device, 1));

        m_accel->pipeline_handle = UInt::steal(jit_optix_configure_pipeline(
            &m_accel->pipeline_compile_options,
            nullptr,
            m_accel->pg,
            2));
        m_accel->sbt_handle = UInt::steal(
            jit_optix_configure_sbt(&m_accel->sbt, m_accel->pipeline_handle.index()));
    }

    m_accel->vertex_buffer = jit_malloc(AllocType::Device, sizeof(float) * vertex_count * 3);
    m_accel->vertex_buffer_ptr = m_accel->vertex_buffer;
    jit_memcpy(JitBackend::CUDA,
               m_accel->vertex_buffer,
               vertex_buffer.data(),
               sizeof(float) * vertex_count * 3);

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers =
        reinterpret_cast<const CUdeviceptr *>(&m_accel->vertex_buffer_ptr);
    build_input.triangleArray.numVertices = static_cast<unsigned int>(vertex_count);
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer =
        reinterpret_cast<CUdeviceptr>(const_cast<int *>(face_buffer.data()));
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(triangle_count);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.preTransform = nullptr;
    build_input.triangleArray.numSbtRecords = 1;
    build_input.triangleArray.sbtIndexOffsetBuffer = nullptr;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    build_input.triangleArray.primitiveIndexOffset = 0;
    build_input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

    unsigned int triangle_input_flags[] = { 0u };
    build_input.triangleArray.flags = triangle_input_flags;

    m_accel->accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    m_accel->accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    jit_optix_check(optixAccelComputeMemoryUsage(m_accel->context,
                                                 &m_accel->accel_options,
                                                 &build_input,
                                                 1,
                                                 &m_accel->gas_buffer_sizes));
    m_accel->gas_temp_buffer =
        jit_malloc(AllocType::Device, m_accel->gas_buffer_sizes.tempSizeInBytes);
    m_accel->gas_temp_buffer_size = m_accel->gas_buffer_sizes.tempSizeInBytes;
    m_accel->gas_buffer =
        jit_malloc(AllocType::Device, m_accel->gas_buffer_sizes.outputSizeInBytes);
    m_accel->gas_buffer_size = m_accel->gas_buffer_sizes.outputSizeInBytes;

    jit_optix_check(optixAccelBuild(m_accel->context,
                                    jit_cuda_stream(),
                                    &m_accel->accel_options,
                                    &build_input,
                                    1,
                                    m_accel->gas_temp_buffer,
                                    m_accel->gas_buffer_sizes.tempSizeInBytes,
                                    m_accel->gas_buffer,
                                    m_accel->gas_buffer_sizes.outputSizeInBytes,
                                    &m_accel->gas_handle,
                                    nullptr,
                                    0));
}

bool SurfelOptixScene::is_ready() const {
    return m_accel != nullptr;
}

template <bool Detached>
SurfelOptixIntersection SurfelOptixScene::intersect(const RayT<Detached> &ray,
                                                    MaskT<Detached> &active) const {
    return intersect<Detached>(ray, FloatT<Detached>(RayEpsilon), active);
}

template <bool Detached>
SurfelOptixIntersection SurfelOptixScene::intersect(const RayT<Detached> &ray,
                                                    const FloatT<Detached> &t_min_input,
                                                    MaskT<Detached> &active) const {
    require(m_accel != nullptr, "SurfelOptixScene::intersect(): scene is not built.");
    require(m_accel->pipeline_handle.index() != 0 && m_accel->sbt_handle.index() != 0,
            "SurfelOptixScene::intersect(): legacy HitObject pipeline is not built.");
    const int ray_count = static_cast<int>(slices(ray.o));

    SurfelOptixIntersection intersection;
    intersection.reserve(ray_count);

    Float ox;
    Float oy;
    Float oz;
    Float dx;
    Float dy;
    Float dz;
    Float t_min;
    Float t_max_input;
    if constexpr (!Detached) {
        ox = detach<false>(ray.o.x());
        oy = detach<false>(ray.o.y());
        oz = detach<false>(ray.o.z());
        dx = detach<false>(ray.d.x());
        dy = detach<false>(ray.d.y());
        dz = detach<false>(ray.d.z());
        t_min = maximum(detach<false>(t_min_input), Float(RayEpsilon));
        t_max_input = detach<false>(ray.tmax);
    } else {
        ox = ray.o.x();
        oy = ray.o.y();
        oz = ray.o.z();
        dx = ray.d.x();
        dy = ray.d.y();
        dz = ray.d.z();
        t_min = maximum(t_min_input, Float(RayEpsilon));
        t_max_input = ray.tmax;
    }

    Mask active_detached = detach<false>(active);

    Float t_max = select(drjit::isfinite(t_max_input), t_max_input, full<Float>(1e8f, ray_count));
    Float time = 0.f;
    UInt ray_mask(255);
    UInt ray_flags(OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
    UInt sbt_offset(0);
    UInt sbt_stride(1);
    UInt miss_sbt_index(0);

    m_accel->handle = dr::opaque<UInt64>(m_accel->gas_handle);
    uint32_t trace_args[] {
        m_accel->handle.index(),
        ox.index(), oy.index(), oz.index(),
        dx.index(), dy.index(), dz.index(),
        t_min.index(), t_max.index(), time.index(),
        ray_mask.index(), ray_flags.index(),
        sbt_offset.index(), sbt_stride.index(),
        miss_sbt_index.index(),
    };

    OptixHitObjectField fields[] {
        OptixHitObjectField::IsHit,
        OptixHitObjectField::RayTMax,
        OptixHitObjectField::Attribute0,
        OptixHitObjectField::Attribute1,
        OptixHitObjectField::PrimitiveIndex,
    };
    uint32_t hitobject_out[5];

    jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t),
                        trace_args,
                        5,
                        fields,
                        hitobject_out,
                        0, 0, 0,
                        0,
                        active_detached.index(),
                        m_accel->pipeline_handle.index(),
                        m_accel->sbt_handle.index());

    Mask is_hit = UInt::steal(hitobject_out[0]) != 0u;
    active_detached &= is_hit;

    using Single = drjit::float32_array_t<Float>;
    intersection.t =
        drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[1]));
    intersection.barycentric[0] =
        drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[2]));
    intersection.barycentric[1] =
        drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[3]));
    intersection.triangle_id = Int(UInt::steal(hitobject_out[4]));

    intersection.t[!active_detached] = Infinity;
    intersection.triangle_id[!active_detached] = -1;

    if constexpr (!Detached) {
        active &= MaskAD(active_detached);
    } else {
        active = active_detached;
    }
    return intersection;
}

template <bool Detached>
SurfelOptixIntersection SurfelOptixScene::trace_analytic_candidates(
    const RayT<Detached> &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    MaskT<Detached> &active) const {
    require(m_accel != nullptr, "SurfelOptixScene::trace_analytic_candidates(): scene is not built.");
    require(max_candidate_hits > 0,
            "SurfelOptixScene::trace_analytic_candidates(): max_candidate_hits must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    SurfelOptixIntersection intersection;
    intersection.reserve(ray_count);
    intersection.triangle_id = full<Int>(-1, ray_count);
    intersection.barycentric = zeros<Vector2f>(ray_count);
    intersection.t = full<Float>(Infinity, ray_count);
    if (ray_count == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return intersection;
    }

    Float ox;
    Float oy;
    Float oz;
    Float dx;
    Float dy;
    Float dz;
    Float t_max_input;
    if constexpr (!Detached) {
        ox = detach<false>(ray.o.x());
        oy = detach<false>(ray.o.y());
        oz = detach<false>(ray.o.z());
        dx = detach<false>(ray.d.x());
        dy = detach<false>(ray.d.y());
        dz = detach<false>(ray.d.z());
        t_max_input = detach<false>(ray.tmax);
    } else {
        ox = ray.o.x();
        oy = ray.o.y();
        oz = ray.o.z();
        dx = ray.d.x();
        dy = ray.d.y();
        dz = ray.d.z();
        t_max_input = ray.tmax;
    }

    const Float zero = zeros<Float>(ray_count);
    ox += zero;
    oy += zero;
    oz += zero;
    dx += zero;
    dy += zero;
    dz += zero;
    t_max_input += zero;

    Mask active_detached = detach<false>(active);
    active_detached = active_detached && full<Mask>(true, ray_count);
    Mask valid = empty<Mask>(ray_count);

    drjit::eval(ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                t_max_input,
                active_detached,
                triangle_to_surfel_id,
                center,
                tangent_u,
                tangent_v,
                opacity);

    ensure_surfel_trace_pipeline(m_accel);

    SurfelTraceParams params = {};
    params.handle = m_accel->gas_handle;
    params.ray_ox = ox.data();
    params.ray_oy = oy.data();
    params.ray_oz = oz.data();
    params.ray_dx = dx.data();
    params.ray_dy = dy.data();
    params.ray_dz = dz.data();
    params.ray_tmax = t_max_input.data();
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.ray_count = ray_count;
    params.triangle_to_surfel_id = triangle_to_surfel_id.data();
    params.triangle_count = m_accel->triangle_count;
    params.surfel_count = static_cast<int>(slices(center));
    params.center_x = center.x().data();
    params.center_y = center.y().data();
    params.center_z = center.z().data();
    params.tangent_u_x = tangent_u.x().data();
    params.tangent_u_y = tangent_u.y().data();
    params.tangent_u_z = tangent_u.z().data();
    params.tangent_v_x = tangent_v.x().data();
    params.tangent_v_y = tangent_v.y().data();
    params.tangent_v_z = tangent_v.z().data();
    params.opacity = opacity.data();
    params.alpha_min = alpha_min;
    params.alpha_cap = alpha_cap;
    params.ray_epsilon = RayEpsilon;
    params.tmax_fallback = 1.0e8f;
    params.max_candidate_hits = max_candidate_hits;
    params.face_forward = face_forward ? 1 : 0;
    params.out_triangle_id = intersection.triangle_id.data();
    params.out_proxy_t = intersection.t.data();
    params.out_valid = reinterpret_cast<uint8_t *>(valid.data());

    {
        ScopedNativeLaunchStage stage(NativeLaunchStage::SurfelTrace);
        m_accel->launch_trace(SurfelOptixLaunchKind::Intersect, params);
    }

    const Mask hit = valid && (intersection.triangle_id >= 0);
    intersection.t[!hit] = Infinity;
    intersection.triangle_id[!hit] = -1;

    active_detached &= hit;
    if constexpr (!Detached) {
        active &= MaskAD(active_detached);
    } else {
        active = active_detached;
    }
    return intersection;
}

template <bool Detached>
SurfelOptixComposite SurfelOptixScene::composite_alpha(
    const RayT<Detached> &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &value,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    MaskT<Detached> active) const {
    require(m_accel != nullptr, "SurfelOptixScene::composite_alpha(): scene is not built.");
    require(max_candidate_hits > 0,
            "SurfelOptixScene::composite_alpha(): max_candidate_hits must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    SurfelOptixComposite composite;
    composite.reserve(ray_count);
    if (ray_count == 0) {
        return composite;
    }

    Float ox;
    Float oy;
    Float oz;
    Float dx;
    Float dy;
    Float dz;
    Float t_max_input;
    if constexpr (!Detached) {
        ox = detach<false>(ray.o.x());
        oy = detach<false>(ray.o.y());
        oz = detach<false>(ray.o.z());
        dx = detach<false>(ray.d.x());
        dy = detach<false>(ray.d.y());
        dz = detach<false>(ray.d.z());
        t_max_input = detach<false>(ray.tmax);
    } else {
        ox = ray.o.x();
        oy = ray.o.y();
        oz = ray.o.z();
        dx = ray.d.x();
        dy = ray.d.y();
        dz = ray.d.z();
        t_max_input = ray.tmax;
    }

    const Float zero = zeros<Float>(ray_count);
    ox += zero;
    oy += zero;
    oz += zero;
    dx += zero;
    dy += zero;
    dz += zero;
    t_max_input += zero;

    Mask active_detached = detach<false>(active);
    active_detached = active_detached && full<Mask>(true, ray_count);
    Mask valid = empty<Mask>(ray_count);

    const int k = max_candidate_hits;
    const int scratch_count = ray_count * k;
    Int scratch_surfel_id = empty<Int>(scratch_count);
    Float scratch_t = empty<Float>(scratch_count);
    Float scratch_alpha = empty<Float>(scratch_count);
    Float scratch_value = empty<Float>(scratch_count);

    drjit::eval(ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                t_max_input,
                active_detached,
                triangle_to_surfel_id,
                center,
                tangent_u,
                tangent_v,
                opacity,
                value);

    ensure_surfel_trace_pipeline(m_accel);

    SurfelTraceParams params = {};
    params.handle = m_accel->gas_handle;
    params.ray_ox = ox.data();
    params.ray_oy = oy.data();
    params.ray_oz = oz.data();
    params.ray_dx = dx.data();
    params.ray_dy = dy.data();
    params.ray_dz = dz.data();
    params.ray_tmax = t_max_input.data();
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.ray_count = ray_count;
    params.triangle_to_surfel_id = triangle_to_surfel_id.data();
    params.triangle_count = m_accel->triangle_count;
    params.surfel_count = static_cast<int>(slices(center));
    params.center_x = center.x().data();
    params.center_y = center.y().data();
    params.center_z = center.z().data();
    params.tangent_u_x = tangent_u.x().data();
    params.tangent_u_y = tangent_u.y().data();
    params.tangent_u_z = tangent_u.z().data();
    params.tangent_v_x = tangent_v.x().data();
    params.tangent_v_y = tangent_v.y().data();
    params.tangent_v_z = tangent_v.z().data();
    params.opacity = opacity.data();
    params.value = value.data();
    params.alpha_min = alpha_min;
    params.alpha_cap = alpha_cap;
    params.ray_epsilon = RayEpsilon;
    params.tmax_fallback = 1.0e8f;
    params.max_candidate_hits = max_candidate_hits;
    params.face_forward = face_forward ? 1 : 0;
    params.out_valid = reinterpret_cast<uint8_t *>(valid.data());
    params.composite_hit_capacity = k;
    params.scratch_surfel_id = scratch_surfel_id.data();
    params.scratch_t = scratch_t.data();
    params.scratch_alpha = scratch_alpha.data();
    params.scratch_value = scratch_value.data();
    params.out_intensity = composite.intensity.data();
    params.out_alpha = composite.alpha.data();
    params.out_transmittance = composite.transmittance.data();
    params.out_depth = composite.depth.data();

    {
        ScopedNativeLaunchStage stage(NativeLaunchStage::SurfelTrace);
        m_accel->launch_trace(SurfelOptixLaunchKind::Composite, params);
    }

    composite.hit_capacity = k;
    composite.surfel_id = scratch_surfel_id;
    composite.hit_t = scratch_t;
    composite.hit_alpha = scratch_alpha;
    composite.hit_value = scratch_value;
    return composite;
}

template SurfelOptixIntersection SurfelOptixScene::intersect<true>(const Ray &ray, Mask &active) const;
template SurfelOptixIntersection SurfelOptixScene::intersect<false>(const RayAD &ray, MaskAD &active) const;
template SurfelOptixIntersection SurfelOptixScene::intersect<true>(const Ray &ray, const Float &t_min, Mask &active) const;
template SurfelOptixIntersection SurfelOptixScene::intersect<false>(const RayAD &ray, const FloatAD &t_min, MaskAD &active) const;
template SurfelOptixIntersection SurfelOptixScene::trace_analytic_candidates<true>(
    const Ray &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    Mask &active) const;
template SurfelOptixIntersection SurfelOptixScene::trace_analytic_candidates<false>(
    const RayAD &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    MaskAD &active) const;
template SurfelOptixComposite SurfelOptixScene::composite_alpha<true>(
    const Ray &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &value,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    Mask active) const;
template SurfelOptixComposite SurfelOptixScene::composite_alpha<false>(
    const RayAD &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &value,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    MaskAD active) const;

} // namespace rayd
