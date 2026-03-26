#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <vector>

#include <rayd/ray.h>
#include <rayd/mesh.h>
#include <rayd/scene/scene_optix.h>

namespace rayd {

// No PTX module — HitObject API with invoke=0 never executes programs.

namespace dr = drjit;

#ifndef RAYD_OPTIX_MODULE_OPT_LEVEL
#  define RAYD_OPTIX_MODULE_OPT_LEVEL OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
#endif

#ifndef RAYD_OPTIX_EXCEPTION_FLAGS
#  define RAYD_OPTIX_EXCEPTION_FLAGS OPTIX_EXCEPTION_FLAG_NONE
#endif

namespace {

void fill_optix_transform(float out[12], const Matrix4f &matrix) {
    Matrix4fDetached detached = detach<false>(matrix);
    drjit::eval(detached);
    drjit::sync_thread();

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            drjit::store(&out[row * 4 + col], detached(row, col));
        }
    }
}

struct OptixMeshState {
    bool dynamic = false;
    int face_offset = 0;
    int mesh_id = -1;
    void *vertex_buffer = nullptr;
    void *vertex_buffer_ptr = nullptr;
    void *gas_temp_buffer = nullptr;
    size_t gas_temp_buffer_size = 0;
    void *gas_buffer = nullptr;
    size_t gas_buffer_size = 0;
    OptixTraversableHandle gas_handle = 0;
    OptixAccelBuildOptions accel_options = {};
    OptixAccelBufferSizes gas_buffer_sizes = {};
};

struct RetiredOptixJitResources {
    UIntDetached pipeline_handle;
    UIntDetached sbt_handle;
};

std::mutex &retired_optix_resources_mutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

std::vector<RetiredOptixJitResources> &retired_optix_resources() {
    static std::vector<RetiredOptixJitResources> *resources =
        new std::vector<RetiredOptixJitResources>();
    return *resources;
}

} // namespace

struct OptixState {
    OptixDeviceContext context = 0;

    UInt64Detached handle;
    UIntDetached pipeline_handle;
    UIntDetached sbt_handle;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc pgd[2] = {};
    OptixProgramGroup pg[2] = {};
    OptixShaderBindingTable sbt = {};

    std::vector<HitGroupSbtRecord> hg_sbts;
    std::vector<OptixMeshState> mesh_states;
    std::vector<OptixInstance> instances;

    void *instance_buffer = nullptr;
    void *ias_temp_buffer = nullptr;
    size_t ias_temp_buffer_size = 0;
    void *ias_buffer = nullptr;
    size_t ias_buffer_size = 0;
    OptixTraversableHandle ias_handle = 0;
    OptixAccelBuildOptions ias_options = {};
    OptixAccelBufferSizes ias_buffer_sizes = {};
};

static void retire_optix_jit_resources(OptixState *state) {
    if (state == nullptr) {
        return;
    }

    if (state->pipeline_handle.index() == 0 && state->sbt_handle.index() == 0) {
        return;
    }

    // Dr.Jit 1.3.x can recycle OptiX pipeline/SBT JIT handles after a scene is
    // destroyed while cached kernels referring to those handles are still
    // around. Keep a bounded retirement list so handle IDs aren't immediately
    // reused, and only flush the kernel cache when the list grows too large.
    std::lock_guard<std::mutex> guard(retired_optix_resources_mutex());
    std::vector<RetiredOptixJitResources> &resources = retired_optix_resources();
    resources.push_back({ state->pipeline_handle, state->sbt_handle });

    state->pipeline_handle = UIntDetached();
    state->sbt_handle = UIntDetached();

    constexpr size_t MaxRetiredOptixResourceSets = 32;
    if (resources.size() >= MaxRetiredOptixResourceSets) {
        jit_flush_kernel_cache();
        resources.clear();
    }
}

void OptixIntersection::reserve(int64_t size) {
    require(size >= 0, "OptixIntersection::reserve(): size must be non-negative.");
    if (size != m_size) {
        m_size = size;
        shape_id = empty<IntDetached>(size);
        global_prim_id = empty<IntDetached>(size);
        barycentric = empty<Vector2fDetached>(size);
        t = empty<FloatDetached>(size);
    }
}

static void destroy_mesh_state(OptixMeshState &state) {
    if (state.vertex_buffer != nullptr) {
        jit_free(state.vertex_buffer);
        state.vertex_buffer = nullptr;
        state.vertex_buffer_ptr = nullptr;
    }
    if (state.gas_temp_buffer != nullptr) {
        jit_free(state.gas_temp_buffer);
        state.gas_temp_buffer = nullptr;
        state.gas_temp_buffer_size = 0;
    }
    if (state.gas_buffer != nullptr) {
        jit_free(state.gas_buffer);
        state.gas_buffer = nullptr;
        state.gas_buffer_size = 0;
        state.gas_handle = 0;
    }
}

static void destroy_optix_state(OptixState *state) {
    if (state == nullptr) {
        return;
    }

    jit_sync_thread();
    retire_optix_jit_resources(state);

    for (OptixMeshState &mesh_state : state->mesh_states) {
        destroy_mesh_state(mesh_state);
    }

    if (state->instance_buffer != nullptr) {
        jit_free(state->instance_buffer);
    }
    if (state->ias_temp_buffer != nullptr) {
        jit_free(state->ias_temp_buffer);
    }
    if (state->ias_buffer != nullptr) {
        jit_free(state->ias_buffer);
    }
    delete state;
}

static void ensure_device_buffer(void *&buffer, size_t &buffer_size, size_t required_size) {
    if (required_size == 0) {
        return;
    }

    if (buffer != nullptr && buffer_size >= required_size) {
        return;
    }

    if (buffer != nullptr) {
        jit_free(buffer);
    }
    buffer = jit_malloc(AllocType::Device, required_size);
    buffer_size = required_size;
}

static void build_gas(OptixState *state, OptixMeshState &mesh_state, const OptixSceneMeshDesc &mesh_desc) {
    const Mesh &mesh = *mesh_desc.mesh;

    if (mesh_state.vertex_buffer == nullptr) {
        mesh_state.vertex_buffer = jit_malloc(AllocType::Device, sizeof(float) * mesh.vertex_count() * 3);
        mesh_state.vertex_buffer_ptr = mesh_state.vertex_buffer;
    }

    jit_memcpy(JitBackend::CUDA,
               mesh_state.vertex_buffer,
               mesh.vertex_buffer().data(),
               sizeof(float) * mesh.vertex_count() * 3);

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers =
        reinterpret_cast<const CUdeviceptr *>(&mesh_state.vertex_buffer_ptr);
    build_input.triangleArray.numVertices = static_cast<unsigned int>(mesh.vertex_count());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer =
        reinterpret_cast<CUdeviceptr>(const_cast<int *>(mesh.face_buffer().data()));
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.face_count());
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.preTransform = nullptr;
    build_input.triangleArray.numSbtRecords = 1;
    build_input.triangleArray.sbtIndexOffsetBuffer = nullptr;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    build_input.triangleArray.primitiveIndexOffset = 0;
    build_input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

    unsigned int triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    build_input.triangleArray.flags = triangle_input_flags;

    mesh_state.accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (mesh_state.dynamic) {
        mesh_state.accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    } else {
        mesh_state.accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    }
    mesh_state.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    jit_optix_check(optixAccelComputeMemoryUsage(state->context,
                                                 &mesh_state.accel_options,
                                                 &build_input,
                                                 1,
                                                 &mesh_state.gas_buffer_sizes));

    ensure_device_buffer(mesh_state.gas_temp_buffer,
                         mesh_state.gas_temp_buffer_size,
                         std::max(mesh_state.gas_buffer_sizes.tempSizeInBytes,
                                  mesh_state.gas_buffer_sizes.tempUpdateSizeInBytes));
    void *gas_output = jit_malloc(AllocType::Device, mesh_state.gas_buffer_sizes.outputSizeInBytes);

    OptixAccelEmitDesc emit_desc = {};
    size_t *d_compacted_size = nullptr;
    unsigned int emit_count = 0;
    if (!mesh_state.dynamic) {
        d_compacted_size = static_cast<size_t *>(jit_malloc(AllocType::Device, sizeof(size_t)));
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = d_compacted_size;
        emit_count = 1;
    }

    jit_optix_check(optixAccelBuild(state->context,
                                    jit_cuda_stream(),
                                    &mesh_state.accel_options,
                                    &build_input,
                                    1,
                                    mesh_state.gas_temp_buffer,
                                    mesh_state.gas_buffer_sizes.tempSizeInBytes,
                                    gas_output,
                                    mesh_state.gas_buffer_sizes.outputSizeInBytes,
                                    &mesh_state.gas_handle,
                                    emit_count != 0 ? &emit_desc : nullptr,
                                    emit_count));

    if (mesh_state.dynamic) {
        mesh_state.gas_buffer = gas_output;
        mesh_state.gas_buffer_size = mesh_state.gas_buffer_sizes.outputSizeInBytes;
        return;
    }

    size_t compacted_size = mesh_state.gas_buffer_sizes.outputSizeInBytes;
    jit_memcpy(JitBackend::CUDA, &compacted_size, d_compacted_size, sizeof(size_t));
    jit_free(d_compacted_size);

    if (compacted_size < mesh_state.gas_buffer_sizes.outputSizeInBytes) {
        void *gas_compact = jit_malloc(AllocType::Device, compacted_size);
        jit_optix_check(optixAccelCompact(state->context,
                                          jit_cuda_stream(),
                                          mesh_state.gas_handle,
                                          gas_compact,
                                          compacted_size,
                                          &mesh_state.gas_handle));
        jit_free(gas_output);
        gas_output = gas_compact;
        mesh_state.gas_buffer_size = compacted_size;
    } else {
        mesh_state.gas_buffer_size = mesh_state.gas_buffer_sizes.outputSizeInBytes;
    }

    mesh_state.gas_buffer = gas_output;
}

static void update_gas(OptixState *state, OptixMeshState &mesh_state, const Mesh &mesh) {
    jit_memcpy(JitBackend::CUDA,
               mesh_state.vertex_buffer,
               mesh.vertex_buffer().data(),
               sizeof(float) * mesh.vertex_count() * 3);

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers =
        reinterpret_cast<const CUdeviceptr *>(&mesh_state.vertex_buffer_ptr);
    build_input.triangleArray.numVertices = static_cast<unsigned int>(mesh.vertex_count());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer =
        reinterpret_cast<CUdeviceptr>(const_cast<int *>(mesh.face_buffer().data()));
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.face_count());
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.preTransform = nullptr;
    build_input.triangleArray.numSbtRecords = 1;
    build_input.triangleArray.sbtIndexOffsetBuffer = nullptr;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    build_input.triangleArray.primitiveIndexOffset = 0;
    build_input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

    unsigned int triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    build_input.triangleArray.flags = triangle_input_flags;

    mesh_state.accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    const size_t update_temp_size = mesh_state.gas_buffer_sizes.tempUpdateSizeInBytes;
    ensure_device_buffer(mesh_state.gas_temp_buffer,
                         mesh_state.gas_temp_buffer_size,
                         update_temp_size);

    jit_optix_check(optixAccelBuild(state->context,
                                    jit_cuda_stream(),
                                    &mesh_state.accel_options,
                                    &build_input,
                                    1,
                                    mesh_state.gas_temp_buffer,
                                    update_temp_size,
                                    mesh_state.gas_buffer,
                                    mesh_state.gas_buffer_size,
                                    &mesh_state.gas_handle,
                                    nullptr,
                                    0));

    mesh_state.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
}

static void update_instances(OptixState *state, const std::vector<OptixSceneMeshDesc> &meshes) {
    state->instances.resize(meshes.size());
    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
        OptixInstance &instance = state->instances[mesh_index];
        std::memset(&instance, 0, sizeof(instance));
        fill_optix_transform(instance.transform, meshes[mesh_index].mesh->full_transform());
        instance.instanceId = static_cast<unsigned int>(meshes[mesh_index].mesh_id);
        instance.sbtOffset = static_cast<unsigned int>(mesh_index);
        instance.visibilityMask = 255u;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = state->mesh_states[mesh_index].gas_handle;
    }

    const size_t instance_bytes = sizeof(OptixInstance) * state->instances.size();
    if (state->instance_buffer == nullptr) {
        state->instance_buffer = jit_malloc(AllocType::Device, instance_bytes);
    }

    jit_memcpy(JitBackend::CUDA, state->instance_buffer, state->instances.data(), instance_bytes);
}

static void build_ias(OptixState *state, const std::vector<OptixSceneMeshDesc> &meshes, bool update) {
    update_instances(state, meshes);

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(state->instance_buffer);
    build_input.instanceArray.numInstances = static_cast<unsigned int>(meshes.size());
    build_input.instanceArray.instanceStride = sizeof(OptixInstance);

    state->ias_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    state->ias_options.operation = update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;

    if (!update) {
        jit_optix_check(optixAccelComputeMemoryUsage(state->context,
                                                     &state->ias_options,
                                                     &build_input,
                                                     1,
                                                     &state->ias_buffer_sizes));
        if (state->ias_buffer != nullptr) {
            jit_free(state->ias_buffer);
        }
        state->ias_buffer = jit_malloc(AllocType::Device, state->ias_buffer_sizes.outputSizeInBytes);
        state->ias_buffer_size = state->ias_buffer_sizes.outputSizeInBytes;
        ensure_device_buffer(state->ias_temp_buffer,
                             state->ias_temp_buffer_size,
                             std::max(state->ias_buffer_sizes.tempSizeInBytes,
                                      state->ias_buffer_sizes.tempUpdateSizeInBytes));
    }

    const size_t temp_size = update ? state->ias_buffer_sizes.tempUpdateSizeInBytes
                                    : state->ias_buffer_sizes.tempSizeInBytes;

    jit_optix_check(optixAccelBuild(state->context,
                                    jit_cuda_stream(),
                                    &state->ias_options,
                                    &build_input,
                                    1,
                                    state->ias_temp_buffer,
                                    temp_size,
                                    state->ias_buffer,
                                    state->ias_buffer_size,
                                    &state->ias_handle,
                                    nullptr,
                                    0));

    state->ias_options.operation = OPTIX_BUILD_OPERATION_BUILD;
}

OptixScene::OptixScene() = default;

OptixScene::~OptixScene() {
    destroy_optix_state(m_accel);
}

void OptixScene::configure(const std::vector<OptixSceneMeshDesc> &meshes) {
    require(!meshes.empty(), "OptixScene::configure(): missing meshes.");

    destroy_optix_state(m_accel);

    init_optix_api();
    m_accel = new OptixState();
    m_accel->context = jit_optix_context();

    m_accel->pipeline_compile_options.usesMotionBlur = false;
    m_accel->pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_accel->pipeline_compile_options.numPayloadValues = 0;
    m_accel->pipeline_compile_options.numAttributeValues = 2;
    m_accel->pipeline_compile_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
    m_accel->pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    m_accel->pipeline_compile_options.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    m_accel->pipeline_compile_options.allowOpacityMicromaps = 0;

    // No PTX module needed — built-in triangle intersection with HitObject API
    // (invoke=0) never executes CH/miss programs.  Program groups with nullptr
    // modules are valid for this configuration, matching Mitsuba's approach.
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

    m_accel->sbt.missRecordBase = jit_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE);
    m_accel->sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
    m_accel->sbt.missRecordCount = 1;
    jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[0], reinterpret_cast<void *>(m_accel->sbt.missRecordBase)));

    m_accel->hg_sbts = std::vector<HitGroupSbtRecord>(meshes.size());
    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
        m_accel->hg_sbts[mesh_index].data.shape_offset = meshes[mesh_index].face_offset;
        m_accel->hg_sbts[mesh_index].data.shape_id = meshes[mesh_index].mesh_id;
        jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[1], &m_accel->hg_sbts[mesh_index]));
    }

    m_accel->sbt.hitgroupRecordBase = jit_malloc(AllocType::HostPinned, meshes.size() * sizeof(HitGroupSbtRecord));
    m_accel->sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_accel->sbt.hitgroupRecordCount = static_cast<unsigned int>(meshes.size());
    jit_memcpy_async(JitBackend::CUDA,
                     reinterpret_cast<void *>(m_accel->sbt.hitgroupRecordBase),
                     m_accel->hg_sbts.data(),
                     meshes.size() * sizeof(HitGroupSbtRecord));

    m_accel->sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(
        jit_malloc_migrate(reinterpret_cast<void *>(m_accel->sbt.missRecordBase), AllocType::Device, 1));
    m_accel->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(
        jit_malloc_migrate(reinterpret_cast<void *>(m_accel->sbt.hitgroupRecordBase), AllocType::Device, 1));

    m_accel->pipeline_handle = UIntDetached::steal(jit_optix_configure_pipeline(&m_accel->pipeline_compile_options,
                                                                                nullptr,
                                                                                m_accel->pg,
                                                                                2));
    m_accel->sbt_handle = UIntDetached::steal(
        jit_optix_configure_sbt(&m_accel->sbt, m_accel->pipeline_handle.index()));

    m_accel->mesh_states.resize(meshes.size());
    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
        OptixMeshState &mesh_state = m_accel->mesh_states[mesh_index];
        mesh_state.dynamic = meshes[mesh_index].dynamic;
        mesh_state.face_offset = meshes[mesh_index].face_offset;
        mesh_state.mesh_id = meshes[mesh_index].mesh_id;
        build_gas(m_accel, mesh_state, meshes[mesh_index]);
    }

    build_ias(m_accel, meshes, false);
}

void OptixScene::commit_updates(const std::vector<OptixSceneMeshDesc> &meshes,
                                 const std::vector<OptixSceneMeshUpdate> &updates) {
    require(m_accel != nullptr, "OptixScene::commit_updates(): scene is not configured.");
    last_commit_profile_ = OptixCommitProfile();

    if (updates.empty()) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();

    for (const OptixSceneMeshUpdate &update : updates) {
        require(update.mesh_id >= 0 && update.mesh_id < static_cast<int>(m_accel->mesh_states.size()),
                "OptixScene::commit_updates(): mesh_id is out of range.");
        if (update.vertices_dirty) {
            ++last_commit_profile_.updated_vertex_meshes;
        }
        if (update.transform_dirty) {
            ++last_commit_profile_.updated_transform_meshes;
        }
        if (!update.vertices_dirty) {
            continue;
        }

        OptixMeshState &mesh_state = m_accel->mesh_states[static_cast<size_t>(update.mesh_id)];
        require(mesh_state.dynamic,
                "OptixScene::commit_updates(): attempted to update a non-dynamic mesh.");
        const auto gas_start = Clock::now();
        update_gas(m_accel, mesh_state, *meshes[static_cast<size_t>(update.mesh_id)].mesh);
        last_commit_profile_.gas_update_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - gas_start).count();
    }

    const auto ias_start = Clock::now();
    build_ias(m_accel, meshes, true);
    last_commit_profile_.ias_update_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - ias_start).count();
    last_commit_profile_.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
}

bool OptixScene::is_ready() const {
    return m_accel != nullptr;
}

template <bool Detached>
OptixIntersection OptixScene::intersect(const RayT<Detached> &ray, MaskT<Detached> &active) const {
    const int ray_count = static_cast<int>(slices(ray.o));

    OptixIntersection intersection;
    intersection.reserve(ray_count);

    FloatDetached ox;
    FloatDetached oy;
    FloatDetached oz;
    FloatDetached dx;
    FloatDetached dy;
    FloatDetached dz;
    FloatDetached t_max_input;
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

    MaskDetached active_detached = detach<false>(active);

    if (drjit::none(active_detached)) {
        return intersection;
    }

    FloatDetached t_min = RayEpsilon;
    FloatDetached t_max = select(drjit::isfinite(t_max_input),
                                 t_max_input,
                                 full<FloatDetached>(1e8f, ray_count));
    FloatDetached time = 0.f;
    UIntDetached ray_mask(255);
    UIntDetached ray_flags(OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                           OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
    UIntDetached sbt_offset(0);
    UIntDetached sbt_stride(1);
    UIntDetached miss_sbt_index(0);

    m_accel->handle = dr::opaque<UInt64Detached>(m_accel->ias_handle);
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
        OptixHitObjectField::SBTDataPointer,
    };
    uint32_t hitobject_out[6];

    jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t),
                        trace_args,
                        6,
                        fields,
                        hitobject_out,
                        0, 0, 0,
                        0,
                        active_detached.index(),
                        m_accel->pipeline_handle.index(),
                        m_accel->sbt_handle.index());

    MaskDetached is_hit = UIntDetached::steal(hitobject_out[0]) != 0u;
    active_detached &= is_hit;

    using Single = drjit::float32_array_t<FloatDetached>;
    intersection.t =
        drjit::reinterpret_array<Single, UIntDetached>(UIntDetached::steal(hitobject_out[1]));
    intersection.barycentric[0] =
        drjit::reinterpret_array<Single, UIntDetached>(UIntDetached::steal(hitobject_out[2]));
    intersection.barycentric[1] =
        drjit::reinterpret_array<Single, UIntDetached>(UIntDetached::steal(hitobject_out[3]));
    UIntDetached raw_prim_index = UIntDetached::steal(hitobject_out[4]);

    // Extract shape_offset and shape_id from the hit-group SBT data.
    UInt64Detached sbt_data_ptr = UInt64Detached::steal(hitobject_out[5]);
    IntDetached shape_offset = IntDetached(UIntDetached::steal(
        jit_optix_sbt_data_load(sbt_data_ptr.index(), VarType::UInt32, 0, active_detached.index())));
    intersection.shape_id = IntDetached(UIntDetached::steal(
        jit_optix_sbt_data_load(sbt_data_ptr.index(), VarType::UInt32, 4, active_detached.index())));
    intersection.global_prim_id = IntDetached(raw_prim_index) + shape_offset;

    // Clear invalid lanes.
    intersection.t[!active_detached] = Infinity;
    intersection.shape_id[!active_detached] = -1;
    intersection.global_prim_id[!active_detached] = -1;

    if constexpr (!Detached) {
        active &= Mask(active_detached);
    } else {
        active = active_detached;
    }
    return intersection;
}

template <bool Detached>
MaskT<Detached> OptixScene::shadow_test(const RayT<Detached> &ray, MaskT<Detached> active) const {
    const int ray_count = static_cast<int>(slices(ray.o));
    MaskT<Detached> hit = full<MaskT<Detached>>(false, ray_count);

    FloatDetached ox;
    FloatDetached oy;
    FloatDetached oz;
    FloatDetached dx;
    FloatDetached dy;
    FloatDetached dz;
    FloatDetached t_max_input;
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

    MaskDetached active_detached = detach<false>(active);

    if (drjit::none(active_detached)) {
        return hit;
    }

    FloatDetached t_min = RayEpsilon;
    FloatDetached t_max = select(drjit::isfinite(t_max_input),
                                 t_max_input,
                                 full<FloatDetached>(1e8f, ray_count));
    FloatDetached time = 0.f;
    UIntDetached ray_mask(255);
    UIntDetached ray_flags(OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                           OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                           OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    UIntDetached sbt_offset(0);
    UIntDetached sbt_stride(1);
    UIntDetached miss_sbt_index(0);

    m_accel->handle = dr::opaque<UInt64Detached>(m_accel->ias_handle);
    uint32_t trace_args[] {
        m_accel->handle.index(),
        ox.index(), oy.index(), oz.index(),
        dx.index(), dy.index(), dz.index(),
        t_min.index(), t_max.index(), time.index(),
        ray_mask.index(), ray_flags.index(),
        sbt_offset.index(), sbt_stride.index(),
        miss_sbt_index.index(),
    };

    OptixHitObjectField fields[] { OptixHitObjectField::IsHit };
    uint32_t hitobject_out[1];

    jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t),
                        trace_args,
                        1,
                        fields,
                        hitobject_out,
                        0, 0, 0,
                        0,
                        active_detached.index(),
                        m_accel->pipeline_handle.index(),
                        m_accel->sbt_handle.index());

    const MaskDetached hit_detached = active_detached && (UIntDetached::steal(hitobject_out[0]) != 0u);
    if constexpr (!Detached) {
        hit = Mask(hit_detached);
    } else {
        hit = hit_detached;
    }
    return hit;
}

template OptixIntersection OptixScene::intersect<true>(const RayDetached &ray, MaskDetached &active) const;
template OptixIntersection OptixScene::intersect<false>(const Ray &ray, Mask &active) const;
template MaskDetached OptixScene::shadow_test<true>(const RayDetached &ray, MaskDetached active) const;
template Mask OptixScene::shadow_test<false>(const Ray &ray, Mask active) const;

} // namespace rayd
