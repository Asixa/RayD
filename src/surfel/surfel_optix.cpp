#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include <rayd/surfel/surfel_optix.h>

namespace rayd {

namespace dr = drjit;

#ifndef RAYD_OPTIX_EXCEPTION_FLAGS
#  define RAYD_OPTIX_EXCEPTION_FLAGS OPTIX_EXCEPTION_FLAG_NONE
#endif

namespace {

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

    int vertex_count = 0;
    int triangle_count = 0;
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

void SurfelOptixIntersection::reserve(int64_t size) {
    require(size >= 0, "SurfelOptixIntersection::reserve(): size must be non-negative.");
    if (size != m_size) {
        m_size = size;
        triangle_id = empty<Int>(size);
        barycentric = empty<Vector2f>(size);
        t = empty<Float>(size);
    }
}

SurfelOptixScene::SurfelOptixScene() = default;

SurfelOptixScene::~SurfelOptixScene() {
    destroy_surfel_optix_state(m_accel);
}

void SurfelOptixScene::build(const Float &vertex_buffer,
                             const Int &face_buffer,
                             int vertex_count,
                             int triangle_count) {
    require(vertex_count > 0, "SurfelOptixScene::build(): vertex_count must be positive.");
    require(triangle_count > 0, "SurfelOptixScene::build(): triangle_count must be positive.");

    destroy_surfel_optix_state(m_accel);

    init_optix_api();
    m_accel = new SurfelOptixState();
    m_accel->context = jit_optix_context();
    m_accel->vertex_count = vertex_count;
    m_accel->triangle_count = triangle_count;

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
    require(m_accel != nullptr, "SurfelOptixScene::intersect(): scene is not built.");
    const int ray_count = static_cast<int>(slices(ray.o));

    SurfelOptixIntersection intersection;
    intersection.reserve(ray_count);

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

    Mask active_detached = detach<false>(active);

    Float t_min = RayEpsilon;
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

template SurfelOptixIntersection SurfelOptixScene::intersect<true>(const Ray &ray, Mask &active) const;
template SurfelOptixIntersection SurfelOptixScene::intersect<false>(const RayAD &ray, MaskAD &active) const;

} // namespace rayd
