#include <rayd/edge/scene_edge_optix.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <rayd/optix.h>

#include <rayd/edge/edge_bvh.h>
#include <rayd/edge/edge_optix_params.h>
#include <rayd/edge/edge_optix_ptx.h>
#include <rayd/native_launch_audit.h>

namespace rayd {

namespace {

/// Capture the calling thread's Dr.Jit CUDA binding for the edge BVH entry
/// points, which take their device and stream explicitly rather than reading
/// whatever happens to be current. `jit_cuda_device_raw()` (not
/// `jit_cuda_device()`) is the raw ordinal `cudaSetDevice` expects.
EdgeBvhCudaContext current_edge_bvh_context() {
    return { jit_cuda_device_raw(), reinterpret_cast<cudaStream_t>(jit_cuda_stream()) };
}

/// Which edge query to launch; selects the matching raygen program and SBT record.
enum class EdgeOptixLaunchKind {
    Point,
    RayAD,
    PointTopK
};

/// Grow \p buffer to at least \p required_size, reallocating only when too small.
void ensure_device_buffer(void *&buffer, size_t &buffer_size, size_t required_size) {
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

bool any_active_lane(const Mask &mask) {
    drjit::eval(mask);
    return !drjit::none(mask);
}

bool prepare_stage_mask(const Mask &mask, bool early_exit, bool first_stage) {
    if (first_stage || !early_exit) {
        drjit::eval(mask);
        return true;
    }
    return any_active_lane(mask);
}

bool edge_optix_stage_early_exit_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("RAYD_EDGE_OPTIX_STAGE_EARLY_EXIT");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

} // namespace

/// All OptiX device state for the edge backend: pipeline, program groups, SBT, params buffer,
/// and one custom-AABB GAS per search radius (edges may be bucketed by inflation radius).
struct EdgeOptixState {
    struct Gas {
        void *aabb_buffer = nullptr;
        size_t aabb_buffer_size = 0;
        void *gas_temp_buffer = nullptr;
        size_t gas_temp_buffer_size = 0;
        void *gas_buffer = nullptr;
        size_t gas_buffer_size = 0;
        OptixTraversableHandle gas_handle = 0;
        OptixAccelBufferSizes gas_buffer_sizes = {};
        float radius = 0.0f;
    };

    OptixDeviceContext context = nullptr;

    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup pg_raygen_point = nullptr;
    OptixProgramGroup pg_raygen_ray = nullptr;
    OptixProgramGroup pg_raygen_topk = nullptr;
    OptixProgramGroup pg_miss = nullptr;
    OptixProgramGroup pg_hit_point = nullptr;
    OptixProgramGroup pg_hit_ray = nullptr;
    OptixProgramGroup pg_hit_topk = nullptr;

    void *sbt_raygen_point = nullptr;
    void *sbt_raygen_ray = nullptr;
    void *sbt_raygen_topk = nullptr;
    void *sbt_miss = nullptr;
    void *sbt_hitgroups = nullptr;
    void *params_buffer = nullptr;

    std::vector<Gas> gases;

    void *raygen_record(EdgeOptixLaunchKind kind) const {
        switch (kind) {
        case EdgeOptixLaunchKind::Point:
            return sbt_raygen_point;
        case EdgeOptixLaunchKind::RayAD:
            return sbt_raygen_ray;
        case EdgeOptixLaunchKind::PointTopK:
            return sbt_raygen_topk;
        }
        return sbt_raygen_point;
    }

    void launch(EdgeOptixLaunchKind kind, const EdgeOptixQueryParams &params) const {
        audit_jit_memcpy_async();
        jit_memcpy_async(JitBackend::CUDA,
                         params_buffer,
                         &params,
                         sizeof(EdgeOptixQueryParams));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record(kind));
        sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(sbt_miss);
        sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(sbt_hitgroups);
        sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        sbt.hitgroupRecordCount = 3;

        audit_optix_launch();
        check_optix(optixLaunch(pipeline,
                                     jit_cuda_stream(),
                                     reinterpret_cast<CUdeviceptr>(params_buffer),
                                     sizeof(EdgeOptixQueryParams),
                                     &sbt,
                                     static_cast<unsigned int>(params.query_count),
                                     1,
                                     1),
                         "optixLaunch(edge query)");
    }
};

SceneEdgeOptix::SceneEdgeOptix()
    : state_(new EdgeOptixState()) {}

SceneEdgeOptix::~SceneEdgeOptix() {
    if (state_ == nullptr) {
        return;
    }

    jit_sync_thread();
    if (state_->pipeline != nullptr && optixPipelineDestroy != nullptr) {
        optixPipelineDestroy(state_->pipeline);
    }
    if (state_->pg_hit_topk != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state_->pg_hit_topk);
    }
    if (state_->pg_hit_ray != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state_->pg_hit_ray);
    }
    if (state_->pg_hit_point != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state_->pg_hit_point);
    }
    if (state_->pg_miss != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state_->pg_miss);
    }
    if (state_->pg_raygen_topk != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state_->pg_raygen_topk);
    }
    if (state_->pg_raygen_ray != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state_->pg_raygen_ray);
    }
    if (state_->pg_raygen_point != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state_->pg_raygen_point);
    }
    if (state_->module != nullptr && optixModuleDestroy != nullptr) {
        optixModuleDestroy(state_->module);
    }

    for (EdgeOptixState::Gas &gas : state_->gases) {
        if (gas.gas_buffer != nullptr) {
            jit_free(gas.gas_buffer);
        }
        if (gas.gas_temp_buffer != nullptr) {
            jit_free(gas.gas_temp_buffer);
        }
        if (gas.aabb_buffer != nullptr) {
            jit_free(gas.aabb_buffer);
        }
    }
    if (state_->params_buffer != nullptr) {
        jit_free(state_->params_buffer);
    }
    if (state_->sbt_hitgroups != nullptr) {
        jit_free(state_->sbt_hitgroups);
    }
    if (state_->sbt_miss != nullptr) {
        jit_free(state_->sbt_miss);
    }
    if (state_->sbt_raygen_topk != nullptr) {
        jit_free(state_->sbt_raygen_topk);
    }
    if (state_->sbt_raygen_ray != nullptr) {
        jit_free(state_->sbt_raygen_ray);
    }
    if (state_->sbt_raygen_point != nullptr) {
        jit_free(state_->sbt_raygen_point);
    }
    delete state_;
}

/// Lazily create the OptiX module, program groups, pipeline, and SBT for the edge programs.
void SceneEdgeOptix::ensure_pipeline() {
    if (state_->pipeline != nullptr) {
        return;
    }

    init_optix_api();
    state_->context = jit_optix_context();

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = RAYD_OPTIX_MODULE_OPT_LEVEL;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = shared::optix::EdgeTopKPayloadCount;
    pipeline_options.numAttributeValues = shared::optix::EdgeAttributeCount;
    pipeline_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);
    pipeline_options.allowOpacityMicromaps = 0;

    char log[2048];
    size_t log_size = sizeof(log);
    check_optix(optixModuleCreate(state_->context,
                                       &module_options,
                                       &pipeline_options,
                                       edge_optix_ptx,
                                       edge_optix_ptx_size,
                                       log,
                                       &log_size,
                                       &state_->module),
                     "optixModuleCreate(edge)");

    state_->pg_raygen_point = make_raygen_group(state_->context, state_->module, "__raygen__edge_point");
    state_->pg_raygen_ray = make_raygen_group(state_->context, state_->module, "__raygen__edge_ray");
    state_->pg_raygen_topk = make_raygen_group(state_->context, state_->module, "__raygen__edge_topk_point");

    state_->pg_miss = make_miss_group(state_->context, state_->module, "__miss__edge_query");

    state_->pg_hit_point = make_hitgroup(state_->context,
                                         state_->module,
                                         "__closesthit__edge_point",
                                         nullptr,
                                         "__intersection__edge_point");
    state_->pg_hit_ray = make_hitgroup(state_->context,
                                       state_->module,
                                       nullptr,
                                       "__anyhit__edge_ray",
                                       "__intersection__edge_ray");
    state_->pg_hit_topk = make_hitgroup(state_->context,
                                        state_->module,
                                        nullptr,
                                        "__anyhit__edge_topk_point",
                                        "__intersection__edge_topk_point");

    OptixProgramGroup groups[] = {
        state_->pg_raygen_point,
        state_->pg_raygen_ray,
        state_->pg_raygen_topk,
        state_->pg_miss,
        state_->pg_hit_point,
        state_->pg_hit_ray,
        state_->pg_hit_topk
    };
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    link_options.maxContinuationCallableDepth = 0;
    link_options.maxDirectCallableDepthFromState = 0;
    link_options.maxDirectCallableDepthFromTraversal = 0;
    link_options.maxTraversableGraphDepth = 1;

    log_size = sizeof(log);
    check_optix(optixPipelineCreate(state_->context,
                                         &pipeline_options,
                                         &link_options,
                                         groups,
                                         7,
                                         log,
                                         &log_size,
                                         &state_->pipeline),
                     "optixPipelineCreate(edge)");
    check_optix(optixPipelineSetStackSize(state_->pipeline,
                                               0,
                                               0,
                                               4096,
                                               1),
                     "optixPipelineSetStackSize(edge)");

    state_->sbt_raygen_point = make_sbt_record(state_->pg_raygen_point);
    state_->sbt_raygen_ray = make_sbt_record(state_->pg_raygen_ray);
    state_->sbt_raygen_topk = make_sbt_record(state_->pg_raygen_topk);
    state_->sbt_miss = make_sbt_record(state_->pg_miss);

    std::vector<EmptySbtRecord> hitgroups(3);
    check_optix(optixSbtRecordPackHeader(state_->pg_hit_point, &hitgroups[0]),
                     "optixSbtRecordPackHeader(edge point hitgroup)");
    check_optix(optixSbtRecordPackHeader(state_->pg_hit_ray, &hitgroups[1]),
                     "optixSbtRecordPackHeader(edge ray hitgroup)");
    check_optix(optixSbtRecordPackHeader(state_->pg_hit_topk, &hitgroups[2]),
                     "optixSbtRecordPackHeader(edge topk hitgroup)");

    state_->sbt_hitgroups = jit_malloc(AllocType::Device, sizeof(EmptySbtRecord) * hitgroups.size());
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA,
               state_->sbt_hitgroups,
               hitgroups.data(),
               sizeof(EmptySbtRecord) * hitgroups.size());

    state_->params_buffer = jit_malloc(AllocType::Device, sizeof(EdgeOptixQueryParams));
}

/// Upload the current edge endpoints and recompute per-edge search radii from \p edge_info.
void SceneEdgeOptix::refresh_geometry(const SecondaryEdgeInfoAD &edge_info) {
    primitive_count_ = edge_info.size();
    edge_p0_ = detach<false>(edge_info.start);
    edge_e1_ = detach<false>(edge_info.edge);
}

std::vector<float> SceneEdgeOptix::compute_search_radii(const SecondaryEdgeInfoAD &edge_info) const {
    const int edge_count = edge_info.size();
    if (edge_count <= 0) {
        return {};
    }

    const Vector3f p0 = detach<false>(edge_info.start);
    const Vector3f e1 = detach<false>(edge_info.edge);
    drjit::eval(p0, e1);

    std::vector<float> p0_x(static_cast<size_t>(edge_count));
    std::vector<float> p0_y(static_cast<size_t>(edge_count));
    std::vector<float> p0_z(static_cast<size_t>(edge_count));
    std::vector<float> e1_x(static_cast<size_t>(edge_count));
    std::vector<float> e1_y(static_cast<size_t>(edge_count));
    std::vector<float> e1_z(static_cast<size_t>(edge_count));
    drjit::store(p0_x.data(), p0.x());
    drjit::store(p0_y.data(), p0.y());
    drjit::store(p0_z.data(), p0.z());
    drjit::store(e1_x.data(), e1.x());
    drjit::store(e1_y.data(), e1.y());
    drjit::store(e1_z.data(), e1.z());

    float min_x = std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float min_z = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();
    float max_edge_length = 0.0f;
    for (int index = 0; index < edge_count; ++index) {
        const float x0 = p0_x[static_cast<size_t>(index)];
        const float y0 = p0_y[static_cast<size_t>(index)];
        const float z0 = p0_z[static_cast<size_t>(index)];
        const float ex = e1_x[static_cast<size_t>(index)];
        const float ey = e1_y[static_cast<size_t>(index)];
        const float ez = e1_z[static_cast<size_t>(index)];
        const float x1 = x0 + ex;
        const float y1 = y0 + ey;
        const float z1 = z0 + ez;
        min_x = std::min(min_x, std::min(x0, x1));
        min_y = std::min(min_y, std::min(y0, y1));
        min_z = std::min(min_z, std::min(z0, z1));
        max_x = std::max(max_x, std::max(x0, x1));
        max_y = std::max(max_y, std::max(y0, y1));
        max_z = std::max(max_z, std::max(z0, z1));
        max_edge_length = std::max(max_edge_length, std::sqrt(ex * ex + ey * ey + ez * ez));
    }

    const float dx = std::max(max_x - min_x, 0.0f);
    const float dy = std::max(max_y - min_y, 0.0f);
    const float dz = std::max(max_z - min_z, 0.0f);
    const float full_radius = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0e-3f);
    const float edge_scale = std::max(max_edge_length, full_radius * 1.0e-4f);

    std::vector<float> radii;
    radii.reserve(3);
    auto add_radius = [&](float radius) {
        if (!std::isfinite(radius) || radius <= 0.0f) {
            return;
        }
        radii.push_back(std::min(std::max(radius, 1.0e-5f), full_radius));
    };

    add_radius(edge_scale * 4.0f);
    add_radius(edge_scale * 34.0f);
    add_radius(full_radius);

    std::sort(radii.begin(), radii.end());
    std::vector<float> unique_radii;
    unique_radii.reserve(radii.size());
    for (float radius : radii) {
        if (unique_radii.empty() ||
            radius > unique_radii.back() * 1.01f + 1.0e-6f) {
            unique_radii.push_back(radius);
        }
    }
    if (unique_radii.empty() ||
        unique_radii.back() < full_radius * 0.999f) {
        unique_radii.push_back(full_radius);
    } else {
        unique_radii.back() = full_radius;
    }
    return unique_radii;
}

/// Build (or refit when \p update) the custom-AABB GAS over the edge primitives.
void SceneEdgeOptix::build_gases(bool update) {
    auto release_gas = [](EdgeOptixState::Gas &gas) {
        if (gas.gas_buffer != nullptr) {
            jit_free(gas.gas_buffer);
            gas.gas_buffer = nullptr;
        }
        if (gas.gas_temp_buffer != nullptr) {
            jit_free(gas.gas_temp_buffer);
            gas.gas_temp_buffer = nullptr;
        }
        if (gas.aabb_buffer != nullptr) {
            jit_free(gas.aabb_buffer);
            gas.aabb_buffer = nullptr;
        }
        gas.aabb_buffer_size = 0;
        gas.gas_temp_buffer_size = 0;
        gas.gas_buffer_size = 0;
        gas.gas_handle = 0;
        gas.gas_buffer_sizes = {};
        gas.radius = 0.0f;
    };

    if (primitive_count_ <= 0) {
        for (EdgeOptixState::Gas &gas : state_->gases) {
            release_gas(gas);
        }
        state_->gases.clear();
        return;
    }

    drjit::eval(edge_p0_, edge_e1_);
    if (state_->gases.size() > search_radii_.size()) {
        for (size_t gas_index = search_radii_.size(); gas_index < state_->gases.size(); ++gas_index) {
            release_gas(state_->gases[gas_index]);
        }
    }
    state_->gases.resize(search_radii_.size());

    for (size_t gas_index = 0; gas_index < search_radii_.size(); ++gas_index) {
        EdgeOptixState::Gas &gas = state_->gases[gas_index];
        gas.radius = search_radii_[gas_index];

        ensure_device_buffer(gas.aabb_buffer,
                             gas.aabb_buffer_size,
                             sizeof(float) * 6u * static_cast<size_t>(primitive_count_));
        compute_edge_optix_aabbs_gpu(current_edge_bvh_context(),
                                     primitive_count_,
                                     edge_p0_.x().data(),
                                     edge_p0_.y().data(),
                                     edge_p0_.z().data(),
                                     edge_e1_.x().data(),
                                     edge_e1_.y().data(),
                                     edge_e1_.z().data(),
                                     gas.radius,
                                     static_cast<float *>(gas.aabb_buffer));

        CUdeviceptr aabb_buffer = reinterpret_cast<CUdeviceptr>(gas.aabb_buffer);
        unsigned int input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };

        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(primitive_count_);
        build_input.customPrimitiveArray.strideInBytes = sizeof(float) * 6u;
        build_input.customPrimitiveArray.flags = input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;
        build_input.customPrimitiveArray.sbtIndexOffsetBuffer = nullptr;
        build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
        build_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;
        build_input.customPrimitiveArray.primitiveIndexOffset = 0;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags =
            OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation =
            update && gas.gas_buffer != nullptr ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;

        if (accel_options.operation == OPTIX_BUILD_OPERATION_BUILD) {
            jit_optix_check(optixAccelComputeMemoryUsage(state_->context,
                                                         &accel_options,
                                                         &build_input,
                                                         1,
                                                         &gas.gas_buffer_sizes));
            ensure_device_buffer(gas.gas_temp_buffer,
                                 gas.gas_temp_buffer_size,
                                 std::max(gas.gas_buffer_sizes.tempSizeInBytes,
                                          gas.gas_buffer_sizes.tempUpdateSizeInBytes));
            if (gas.gas_buffer != nullptr) {
                jit_free(gas.gas_buffer);
            }
            gas.gas_buffer = jit_malloc(AllocType::Device, gas.gas_buffer_sizes.outputSizeInBytes);
            gas.gas_buffer_size = gas.gas_buffer_sizes.outputSizeInBytes;
        } else {
            ensure_device_buffer(gas.gas_temp_buffer,
                                 gas.gas_temp_buffer_size,
                                 gas.gas_buffer_sizes.tempUpdateSizeInBytes);
        }

        const size_t temp_size =
            accel_options.operation == OPTIX_BUILD_OPERATION_UPDATE
                ? gas.gas_buffer_sizes.tempUpdateSizeInBytes
                : gas.gas_buffer_sizes.tempSizeInBytes;

        audit_optix_accel_build();
        jit_optix_check(optixAccelBuild(state_->context,
                                        jit_cuda_stream(),
                                        &accel_options,
                                        &build_input,
                                        1,
                                        gas.gas_temp_buffer,
                                        temp_size,
                                        gas.gas_buffer,
                                        gas.gas_buffer_size,
                                        &gas.gas_handle,
                                        nullptr,
                                        0));
    }
}

void SceneEdgeOptix::build(const SecondaryEdgeInfoAD &edge_info,
                           const Mask &mask) {
    require(static_cast<int>(mask.size()) == edge_info.size(),
            "SceneEdgeOptix::build(): mask size must match the edge count.");
    ensure_pipeline();
    refresh_geometry(edge_info);
    edge_mask_ = mask;
    search_radii_ = compute_search_radii(edge_info);
    build_gases(false);
    ready_ = true;
}

void SceneEdgeOptix::set_mask(const Mask &mask) {
    require(ready_, "SceneEdgeOptix::set_mask(): GAS is not built.");
    require(static_cast<int>(mask.size()) == primitive_count_,
            "SceneEdgeOptix::set_mask(): mask size must match the edge count.");
    edge_mask_ = mask;
}

void SceneEdgeOptix::refit(const SecondaryEdgeInfoAD &edge_info,
                           const std::vector<EdgeDirtyRange> &dirty_ranges) {
    require(ready_, "SceneEdgeOptix::refit(): GAS is not built.");
    if (primitive_count_ == 0 || dirty_ranges.empty()) {
        return;
    }

    std::vector<float> new_radii = compute_search_radii(edge_info);
    const bool rebuild = new_radii.size() != search_radii_.size() ||
                         edge_info.size() != primitive_count_ ||
                         !std::equal(new_radii.begin(),
                                     new_radii.end(),
                                     search_radii_.begin(),
                                     [](float lhs, float rhs) {
                                         return lhs <= rhs * 1.01f && lhs >= rhs * 0.99f;
                                     });
    search_radii_ = std::move(new_radii);
    refresh_geometry(edge_info);
    build_gases(!rebuild);
}

SceneEdgeBVHStats SceneEdgeOptix::stats() const {
    require(ready_, "SceneEdgeOptix::stats(): GAS is not built.");
    SceneEdgeBVHStats result;
    result.primitive_count = primitive_count_;
    result.node_count = primitive_count_ > 0 ? 1 : 0;
    result.leaf_node_count = primitive_count_ > 0 ? primitive_count_ : 0;
    result.min_leaf_size = primitive_count_ > 0 ? 1 : 0;
    result.max_leaf_size = primitive_count_ > 0 ? 1 : 0;
    result.avg_leaf_size = primitive_count_ > 0 ? 1.0 : 0.0;
    result.leaf_size_histogram.assign(2, 0);
    if (primitive_count_ > 0) {
        result.leaf_size_histogram[1] = primitive_count_;
    }
    return result;
}

template <bool Detached>
ClosestEdgeCandidate SceneEdgeOptix::nearest_edge(const Vector3fT<Detached> &point,
                                                  MaskT<Detached> &active) const {
    require(ready_, "SceneEdgeOptix::nearest_edge(point): GAS is not built.");

    const int query_count = static_cast<int>(slices(point));
    ClosestEdgeCandidate result;
    result.global_edge_id = full<Int>(-1, query_count);
    result.distance_sq = full<Float>(Infinity, query_count);
    if (primitive_count_ == 0 || query_count == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return result;
    }

    const Vector3f point_detached = detach<false>(point);
    const Mask active_detached = detach<false>(active);
    if (drjit::none(active_detached)) {
        return result;
    }

    drjit::eval(point_detached, active_detached, edge_mask_);

    Mask unresolved = active_detached;
    const bool early_exit = edge_optix_stage_early_exit_enabled();
    bool first_stage = true;
    for (const EdgeOptixState::Gas &gas : state_->gases) {
        if (!prepare_stage_mask(unresolved, early_exit, first_stage)) {
            break;
        }
        first_stage = false;
        ClosestEdgeCandidate stage;
        stage.global_edge_id = full<Int>(-1, query_count);
        stage.distance_sq = full<Float>(Infinity, query_count);
        Float edge_t = empty<Float>(query_count);
        Mask valid = empty<Mask>(query_count);

        EdgeOptixQueryParams params = {};
        params.handle = gas.gas_handle;
        params.edge_p0_x = edge_p0_.x().data();
        params.edge_p0_y = edge_p0_.y().data();
        params.edge_p0_z = edge_p0_.z().data();
        params.edge_e1_x = edge_e1_.x().data();
        params.edge_e1_y = edge_e1_.y().data();
        params.edge_e1_z = edge_e1_.z().data();
        params.edge_mask = reinterpret_cast<const uint8_t *>(edge_mask_.data());
        params.edge_count = primitive_count_;
        params.search_radius = gas.radius;
        params.query_x = point_detached.x().data();
        params.query_y = point_detached.y().data();
        params.query_z = point_detached.z().data();
        params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data());
        params.query_count = query_count;
        params.out_edge_ids = stage.global_edge_id.data();
        params.out_distance_sq = stage.distance_sq.data();
        params.out_edge_t = edge_t.data();
        params.out_valid = reinterpret_cast<uint8_t *>(valid.data());

        state_->launch(EdgeOptixLaunchKind::Point, params);

        const Mask hit = stage.global_edge_id >= 0;
        result.global_edge_id = select(hit, stage.global_edge_id, result.global_edge_id);
        result.distance_sq = select(hit, stage.distance_sq, result.distance_sq);
        unresolved &= !hit;
    }

    const Mask hit = result.global_edge_id >= 0;
    if constexpr (!Detached) {
        active &= MaskAD(hit);
    } else {
        active &= hit;
    }
    return result;
}

template <bool Detached>
ClosestEdgeCandidate SceneEdgeOptix::nearest_edge(const RayT<Detached> &ray,
                                                  MaskT<Detached> &active) const {
    require(ready_, "SceneEdgeOptix::nearest_edge(ray): GAS is not built.");

    const int query_count = static_cast<int>(slices(ray.o));
    ClosestEdgeCandidate result;
    result.global_edge_id = full<Int>(-1, query_count);
    result.distance_sq = full<Float>(Infinity, query_count);
    if (primitive_count_ == 0 || query_count == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return result;
    }

    Ray ray_detached(detach<false>(ray.o), detach<false>(ray.d));
    ray_detached.tmax = detach<false>(ray.tmax);
    const Mask active_detached = detach<false>(active);
    if (drjit::none(active_detached)) {
        return result;
    }

    drjit::eval(ray_detached.o, ray_detached.d, ray_detached.tmax, active_detached, edge_mask_);

    Mask unresolved = active_detached;
    for (const EdgeOptixState::Gas &gas : state_->gases) {
        ClosestEdgeCandidate stage;
        stage.global_edge_id = full<Int>(-1, query_count);
        stage.distance_sq = full<Float>(Infinity, query_count);
        Float ray_t = empty<Float>(query_count);
        Float edge_t = empty<Float>(query_count);
        Mask valid = empty<Mask>(query_count);
        drjit::eval(unresolved);

        EdgeOptixQueryParams params = {};
        params.handle = gas.gas_handle;
        params.edge_p0_x = edge_p0_.x().data();
        params.edge_p0_y = edge_p0_.y().data();
        params.edge_p0_z = edge_p0_.z().data();
        params.edge_e1_x = edge_e1_.x().data();
        params.edge_e1_y = edge_e1_.y().data();
        params.edge_e1_z = edge_e1_.z().data();
        params.edge_mask = reinterpret_cast<const uint8_t *>(edge_mask_.data());
        params.edge_count = primitive_count_;
        params.search_radius = gas.radius;
        params.query_x = ray_detached.o.x().data();
        params.query_y = ray_detached.o.y().data();
        params.query_z = ray_detached.o.z().data();
        params.ray_dx = ray_detached.d.x().data();
        params.ray_dy = ray_detached.d.y().data();
        params.ray_dz = ray_detached.d.z().data();
        params.ray_tmax = ray_detached.tmax.data();
        params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data());
        params.query_count = query_count;
        params.out_edge_ids = stage.global_edge_id.data();
        params.out_distance_sq = stage.distance_sq.data();
        params.out_ray_t = ray_t.data();
        params.out_edge_t = edge_t.data();
        params.out_valid = reinterpret_cast<uint8_t *>(valid.data());

        state_->launch(EdgeOptixLaunchKind::RayAD, params);

        const Mask hit = stage.global_edge_id >= 0;
        result.global_edge_id = select(hit, stage.global_edge_id, result.global_edge_id);
        result.distance_sq = select(hit, stage.distance_sq, result.distance_sq);
        unresolved &= !hit;
    }

    const Mask hit = result.global_edge_id >= 0;
    if constexpr (!Detached) {
        active &= MaskAD(hit);
    } else {
        active &= hit;
    }
    return result;
}

template <bool Detached>
ClosestEdgeTopKCandidate SceneEdgeOptix::nearest_edges(const Vector3fT<Detached> &point,
                                                            int k,
                                                            MaskT<Detached> &active) const {
    require(ready_, "SceneEdgeOptix::nearest_edges(point): GAS is not built.");
    require(k > 0, "SceneEdgeOptix::nearest_edges(point): k must be positive.");
    require(k <= EdgeOptixTopKMax, "SceneEdgeOptix::nearest_edges(point): k must be <= 16.");

    const int query_count = static_cast<int>(slices(point));
    const int output_count = query_count * k;
    ClosestEdgeTopKCandidate result;
    result.query_count = query_count;
    result.k = k;
    result.is_valid = full<Mask>(false, output_count);
    result.global_edge_ids = full<Int>(-1, output_count);
    result.distance_sq = full<Float>(Infinity, output_count);
    if (primitive_count_ == 0 || query_count == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return result;
    }

    const Vector3f point_detached = detach<false>(point);
    const Mask active_detached = detach<false>(active);
    if (drjit::none(active_detached)) {
        return result;
    }

    drjit::eval(point_detached, active_detached, edge_mask_);

    Mask unresolved = active_detached;
    const Int output_indices = arange<Int>(output_count);
    const Int output_query_indices = output_indices / k;
    const Mask output_active = full<Mask>(true, output_count);
    const Int kth_slot = arange<Int>(query_count) * k + (k - 1);
    const bool early_exit = edge_optix_stage_early_exit_enabled();
    bool first_stage = true;
    for (const EdgeOptixState::Gas &gas : state_->gases) {
        if (!prepare_stage_mask(unresolved, early_exit, first_stage)) {
            break;
        }
        first_stage = false;
        ClosestEdgeTopKCandidate stage;
        stage.query_count = query_count;
        stage.k = k;
        stage.is_valid = full<Mask>(false, output_count);
        stage.global_edge_ids = full<Int>(-1, output_count);
        stage.distance_sq = full<Float>(Infinity, output_count);
        Float edge_t = empty<Float>(output_count);

        EdgeOptixQueryParams params = {};
        params.handle = gas.gas_handle;
        params.edge_p0_x = edge_p0_.x().data();
        params.edge_p0_y = edge_p0_.y().data();
        params.edge_p0_z = edge_p0_.z().data();
        params.edge_e1_x = edge_e1_.x().data();
        params.edge_e1_y = edge_e1_.y().data();
        params.edge_e1_z = edge_e1_.z().data();
        params.edge_mask = reinterpret_cast<const uint8_t *>(edge_mask_.data());
        params.edge_count = primitive_count_;
        params.search_radius = gas.radius;
        params.query_x = point_detached.x().data();
        params.query_y = point_detached.y().data();
        params.query_z = point_detached.z().data();
        params.active_mask = reinterpret_cast<const uint8_t *>(unresolved.data());
        params.query_count = query_count;
        params.k = k;
        params.out_edge_ids = stage.global_edge_ids.data();
        params.out_distance_sq = stage.distance_sq.data();
        params.out_edge_t = edge_t.data();
        params.out_valid = reinterpret_cast<uint8_t *>(stage.is_valid.data());

        state_->launch(EdgeOptixLaunchKind::PointTopK, params);

        const Mask take_slot =
            gather<Mask>(unresolved, output_query_indices, output_active);
        result.is_valid = select(take_slot, stage.is_valid, result.is_valid);
        result.global_edge_ids = select(take_slot, stage.global_edge_ids, result.global_edge_ids);
        result.distance_sq = select(take_slot, stage.distance_sq, result.distance_sq);

        const Mask has_k = gather<Mask>(stage.is_valid, kth_slot, unresolved);
        unresolved &= !has_k;
    }

    const Int first_slot = arange<Int>(query_count) * k;
    const Mask has_any = gather<Mask>(result.is_valid, first_slot, active_detached);
    if constexpr (!Detached) {
        active &= MaskAD(has_any);
    } else {
        active &= has_any;
    }
    return result;
}

template ClosestEdgeCandidate SceneEdgeOptix::nearest_edge<true>(const Vector3f &point,
                                                                 Mask &active) const;
template ClosestEdgeCandidate SceneEdgeOptix::nearest_edge<false>(const Vector3fAD &point,
                                                                  MaskAD &active) const;
template ClosestEdgeCandidate SceneEdgeOptix::nearest_edge<true>(const Ray &ray,
                                                                 Mask &active) const;
template ClosestEdgeCandidate SceneEdgeOptix::nearest_edge<false>(const RayAD &ray,
                                                                  MaskAD &active) const;
template ClosestEdgeTopKCandidate SceneEdgeOptix::nearest_edges<true>(
    const Vector3f &point,
    int k,
    Mask &active) const;
template ClosestEdgeTopKCandidate SceneEdgeOptix::nearest_edges<false>(
    const Vector3fAD &point,
    int k,
    MaskAD &active) const;

} // namespace rayd
