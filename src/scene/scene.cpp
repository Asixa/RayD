#include <algorithm>
#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <rayd/intersection.h>
#include <rayd/camera.h>
#include <rayd/ray.h>
#include <rayd/scene/scene.h>
#include <rayd/scene/scene_edge.h>

#include "../multipath/reflection_dedup.h"
#include "../multipath/reflection_accumulation_host.h"
#include "../multipath/reflection_trace_host.h"
#include "../multipath/segment_visibility_host.h"
#include "../native_launch_audit.h"

namespace rayd {

namespace {

enum class OptixSplitMode {
    Auto,
    Off,
    On
};

enum class TraceVisibilityBackend {
    Auto,
    Jit,
    Native
};

std::string normalize_optix_split_mode_value(const char *value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

OptixSplitMode active_optix_split_mode() {
    static const OptixSplitMode value = []() {
        const char *raw = std::getenv("RAYD_OPTIX_SPLIT_MODE");
        const std::string normalized = normalize_optix_split_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return normalized.empty() ? OptixSplitMode::Off : OptixSplitMode::Auto;
        }
        if (normalized == "off" || normalized == "false" || normalized == "0") {
            return OptixSplitMode::Off;
        }
        if (normalized == "on" || normalized == "true" || normalized == "1") {
            return OptixSplitMode::On;
        }
        throw std::runtime_error(
            "Invalid RAYD_OPTIX_SPLIT_MODE. Expected one of: auto, off, on.");
    }();
    return value;
}

TraceVisibilityBackend active_trace_visibility_backend() {
    static const TraceVisibilityBackend value = []() {
        const char *raw = std::getenv("RAYD_TRACE_VISIBILITY_BACKEND");
        const std::string normalized = normalize_optix_split_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return TraceVisibilityBackend::Auto;
        }
        if (normalized == "jit" || normalized == "drjit" ||
            normalized == "hitobject" || normalized == "hit_object") {
            return TraceVisibilityBackend::Jit;
        }
        if (normalized == "native" || normalized == "optixlaunch" ||
            normalized == "optix_launch") {
            return TraceVisibilityBackend::Native;
        }
        throw std::runtime_error(
            "Invalid RAYD_TRACE_VISIBILITY_BACKEND. Expected one of: auto, jit, native.");
    }();
    return value;
}

bool use_jit_trace_visibility_path(int ignore_k) {
    const TraceVisibilityBackend backend = active_trace_visibility_backend();
    if (backend == TraceVisibilityBackend::Native) {
        return false;
    }
    if (backend == TraceVisibilityBackend::Jit) {
        require(ignore_k == 0,
                "RAYD_TRACE_VISIBILITY_BACKEND=jit does not support ignore lists yet.");
        return true;
    }
    return ignore_k == 0;
}

std::string normalize_edge_backend_value(const std::string &value) {
    std::string normalized = value;
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

EdgeBVHBackend parse_edge_backend(const std::string &value) {
    const std::string normalized = normalize_edge_backend_value(value);
    if (normalized.empty() || normalized == "drjit" ||
        normalized == "dr_jit" || normalized == "software") {
        return EdgeBVHBackend::DrJit;
    }
    if (normalized == "optix" || normalized == "custom_aabb") {
        return EdgeBVHBackend::Optix;
    }
    if (normalized == "hybrid" || normalized == "mixed" ||
        normalized == "optix_ray" || normalized == "ray_optix") {
        return EdgeBVHBackend::Hybrid;
    }
    throw std::runtime_error(
        "Invalid edge_bvh_backend. Expected one of: 'drjit', 'optix', 'hybrid'.");
}

const char *edge_backend_name(EdgeBVHBackend backend) {
    switch (backend) {
    case EdgeBVHBackend::DrJit:
        return "drjit";
    case EdgeBVHBackend::Optix:
        return "optix";
    case EdgeBVHBackend::Hybrid:
        return "hybrid";
    }
    return "drjit";
}

bool edge_backend_builds_drjit(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::DrJit ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_builds_optix(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_uses_optix_point(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

bool edge_backend_uses_optix_ray(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_uses_optix_topk(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

bool should_split_optix_scene(OptixSplitMode mode,
                              int static_mesh_count,
                              int dynamic_mesh_count) {
    if (static_mesh_count == 0 || dynamic_mesh_count == 0) {
        return false;
    }
    if (mode == OptixSplitMode::On) {
        return true;
    }
    if (mode == OptixSplitMode::Off) {
        return false;
    }

    // The measured mixed-scene query tax is still too large to justify enabling
    // split mode automatically. Keep "on" available for calibration, but bias
    // "auto" to the stable single-scene path until a better heuristic exists.
    return false;
}

bool uses_symbolic_optix_query_path() {
    // Dr.Jit symbolic recording cannot mix multiple OptiX pipelines/SBTs
    // within a single captured kernel. Fall back to the unified scene path.
    return jit_flag(JitFlag::Recording);
}

bool recording_reflections() {
    return jit_flag(JitFlag::Recording);
}

template <bool Detached>
NearestPointEdgeT<Detached> initialize_nearest_point_edge_result(int query_count) {
    NearestPointEdgeT<Detached> result;
    result.distance = full<FloatT<Detached>>(Infinity, query_count);
    result.point = zeros<Vector3fT<Detached>>(query_count);
    result.edge_t = zeros<FloatT<Detached>>(query_count);
    result.edge_point = zeros<Vector3fT<Detached>>(query_count);
    result.shape_id = full<IntT<Detached>>(-1, query_count);
    result.edge_id = full<IntT<Detached>>(-1, query_count);
    result.global_edge_id = full<IntT<Detached>>(-1, query_count);
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> initialize_nearest_ray_edge_result(int query_count) {
    NearestRayEdgeT<Detached> result;
    result.distance = full<FloatT<Detached>>(Infinity, query_count);
    result.ray_t = zeros<FloatT<Detached>>(query_count);
    result.point = zeros<Vector3fT<Detached>>(query_count);
    result.edge_t = zeros<FloatT<Detached>>(query_count);
    result.edge_point = zeros<Vector3fT<Detached>>(query_count);
    result.shape_id = full<IntT<Detached>>(-1, query_count);
    result.edge_id = full<IntT<Detached>>(-1, query_count);
    result.global_edge_id = full<IntT<Detached>>(-1, query_count);
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
}

struct ReflectionTraceRaw {
    int max_bounces = 0;
    int ray_count = 0;
    IntDetached bounce_count;
    IntDetached discovery_count;
    IntDetached representative_ray_index;
    IntDetached shape_ids;
    IntDetached prim_ids;
    FloatDetached t;
    FloatDetached bary_u;
    FloatDetached bary_v;
    FloatDetached hit_x;
    FloatDetached hit_y;
    FloatDetached hit_z;
    FloatDetached norm_x;
    FloatDetached norm_y;
    FloatDetached norm_z;
    FloatDetached img_x;
    FloatDetached img_y;
    FloatDetached img_z;
    FloatDetached trailing_t;
    IntDetached trailing_prim;
    FloatDetached trailing_dir_x;
    FloatDetached trailing_dir_y;
    FloatDetached trailing_dir_z;
    FloatDetached trailing_origin_x;
    FloatDetached trailing_origin_y;
    FloatDetached trailing_origin_z;
};

struct ReflectionAccumulationRaw {
    int ray_count = 0;
    int max_bounces = 0;
    int grid_cell_count = 0;
    int wedge_capacity = 0;
    FloatDetached reflection_power;
    IntDetached reflection_count;
    IntDetached wedge_count;
    IntDetached wedge_ray_index;
    FloatDetached wedge_hit_x;
    FloatDetached wedge_hit_y;
    FloatDetached wedge_hit_z;
    FloatDetached wedge_normal_x;
    FloatDetached wedge_normal_y;
    FloatDetached wedge_normal_z;
    IntDetached wedge_prim_id;
    FloatDetached wedge_dir_x;
    FloatDetached wedge_dir_y;
    FloatDetached wedge_dir_z;
    IntDetached wedge_bounce_depth;
};

IntDetached globalize_primitive_ids(const IntDetached &local_prim_ids,
                                    const IntDetached &shape_ids,
                                    const IntDetached &face_offsets) {
    const int ray_count = static_cast<int>(slices(local_prim_ids));
    if (ray_count == 0) {
        return IntDetached();
    }

    const int mesh_count = std::max(0, static_cast<int>(slices(face_offsets)) - 1);
    const MaskDetached valid =
        (local_prim_ids >= 0) && (shape_ids >= 0) && (shape_ids < mesh_count);
    const IntDetached safe_shape_ids = select(valid, shape_ids, zeros<IntDetached>(ray_count));
    const IntDetached mesh_face_offsets =
        gather<IntDetached>(face_offsets, safe_shape_ids, valid);
    return select(valid,
                  local_prim_ids + mesh_face_offsets,
                  full<IntDetached>(-1, ray_count));
}

template <bool Detached>
ReflectionChainT<Detached> initialize_reflection_chain_result(int ray_count,
                                                              int max_bounces) {
    ReflectionChainT<Detached> result;
    result.max_bounces = max_bounces;
    result.ray_count = ray_count;

    const int slot_count = ray_count * max_bounces;
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.discovery_count = full<IntT<Detached>>(0, ray_count);
    result.representative_ray_index = full<IntT<Detached>>(-1, ray_count);
    result.t = full<FloatT<Detached>>(Infinity, slot_count);
    result.hit_points = zeros<Vector3fT<Detached>>(slot_count);
    result.geo_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.image_sources = zeros<Vector3fT<Detached>>(slot_count);
    result.plane_points = zeros<Vector3fT<Detached>>(slot_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.shape_ids = full<IntT<Detached>>(-1, slot_count);
    result.prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.local_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.global_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.trailing_t = full<FloatT<Detached>>(Infinity, ray_count);
    result.trailing_prim = full<IntT<Detached>>(-1, ray_count);
    result.trailing_dir = zeros<Vector3fT<Detached>>(ray_count);
    result.trailing_origin = zeros<Vector3fT<Detached>>(ray_count);
    return result;
}

template <bool Detached>
ReflectionBounceT<Detached> initialize_reflection_bounce_result(int ray_count) {
    ReflectionBounceT<Detached> result;
    result.t = full<FloatT<Detached>>(Infinity, ray_count);
    result.hit_points = zeros<Vector3fT<Detached>>(ray_count);
    result.geo_normals = zeros<Vector3fT<Detached>>(ray_count);
    result.image_sources = zeros<Vector3fT<Detached>>(ray_count);
    result.plane_points = zeros<Vector3fT<Detached>>(ray_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(ray_count);
    result.shape_ids = full<IntT<Detached>>(-1, ray_count);
    result.prim_ids = full<IntT<Detached>>(-1, ray_count);
    result.local_prim_ids = full<IntT<Detached>>(-1, ray_count);
    result.global_prim_ids = full<IntT<Detached>>(-1, ray_count);
    return result;
}

template <bool Detached>
ReflectionTraceT<Detached> initialize_reflection_trace_result(
    int ray_count,
    int max_bounces) {
    ReflectionTraceT<Detached> result;
    result.max_bounces = max_bounces;
    result.ray_count = ray_count;
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.discovery_count = full<IntT<Detached>>(0, ray_count);
    result.representative_ray_index = full<IntT<Detached>>(-1, ray_count);
    result.dedup_keep_mask = full<MaskT<Detached>>(false, ray_count);
    result.bounces.reserve(static_cast<size_t>(max_bounces));
    return result;
}

ReflectionTraceRaw allocate_reflection_trace_raw(int ray_count, int max_bounces) {
    const int slot_count = ray_count * max_bounces;

    ReflectionTraceRaw raw;
    raw.max_bounces = max_bounces;
    raw.ray_count = ray_count;
    raw.bounce_count = empty<IntDetached>(ray_count);
    raw.discovery_count = empty<IntDetached>(ray_count);
    raw.representative_ray_index = empty<IntDetached>(ray_count);
    raw.shape_ids = empty<IntDetached>(slot_count);
    raw.prim_ids = empty<IntDetached>(slot_count);
    raw.t = empty<FloatDetached>(slot_count);
    raw.bary_u = empty<FloatDetached>(slot_count);
    raw.bary_v = empty<FloatDetached>(slot_count);
    raw.hit_x = empty<FloatDetached>(slot_count);
    raw.hit_y = empty<FloatDetached>(slot_count);
    raw.hit_z = empty<FloatDetached>(slot_count);
    raw.norm_x = empty<FloatDetached>(slot_count);
    raw.norm_y = empty<FloatDetached>(slot_count);
    raw.norm_z = empty<FloatDetached>(slot_count);
    raw.img_x = empty<FloatDetached>(slot_count);
    raw.img_y = empty<FloatDetached>(slot_count);
    raw.img_z = empty<FloatDetached>(slot_count);
    raw.trailing_t = empty<FloatDetached>(ray_count);
    raw.trailing_prim = empty<IntDetached>(ray_count);
    raw.trailing_dir_x = empty<FloatDetached>(ray_count);
    raw.trailing_dir_y = empty<FloatDetached>(ray_count);
    raw.trailing_dir_z = empty<FloatDetached>(ray_count);
    raw.trailing_origin_x = empty<FloatDetached>(ray_count);
    raw.trailing_origin_y = empty<FloatDetached>(ray_count);
    raw.trailing_origin_z = empty<FloatDetached>(ray_count);
    return raw;
}

void initialize_reflection_trace_raw(ReflectionTraceRaw &raw) {
    const int ray_count = raw.ray_count;
    const int slot_count = raw.ray_count * raw.max_bounces;
    const int zero_i = 0;
    const int minus_one_i = -1;
    const float zero_f = 0.f;
    const float inf_f = Infinity;

    jit_memset_async(JitBackend::CUDA, raw.bounce_count.data(), ray_count, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.discovery_count.data(), ray_count, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.representative_ray_index.data(),
                     ray_count,
                     sizeof(int),
                     &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.shape_ids.data(), slot_count, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.prim_ids.data(), slot_count, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.t.data(), slot_count, sizeof(float), &inf_f);
    jit_memset_async(JitBackend::CUDA, raw.bary_u.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.bary_v.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.hit_x.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.hit_y.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.hit_z.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.norm_x.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.norm_y.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.norm_z.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.img_x.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.img_y.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.img_z.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_t.data(), ray_count, sizeof(float), &inf_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_prim.data(), ray_count, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.trailing_dir_x.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_dir_y.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_dir_z.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_origin_x.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_origin_y.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_origin_z.data(), ray_count, sizeof(float), &zero_f);
}

ReflectionAccumulationRaw allocate_reflection_accumulation_raw(int ray_count,
                                                               int max_bounces,
                                                               int grid_cell_count,
                                                               int wedge_capacity) {
    ReflectionAccumulationRaw raw;
    raw.ray_count = ray_count;
    raw.max_bounces = max_bounces;
    raw.grid_cell_count = grid_cell_count;
    raw.wedge_capacity = wedge_capacity;
    raw.reflection_power = empty<FloatDetached>(grid_cell_count);
    raw.reflection_count = empty<IntDetached>(1);
    raw.wedge_count = empty<IntDetached>(1);
    const int event_count = std::max(1, wedge_capacity);
    raw.wedge_ray_index = empty<IntDetached>(event_count);
    raw.wedge_hit_x = empty<FloatDetached>(event_count);
    raw.wedge_hit_y = empty<FloatDetached>(event_count);
    raw.wedge_hit_z = empty<FloatDetached>(event_count);
    raw.wedge_normal_x = empty<FloatDetached>(event_count);
    raw.wedge_normal_y = empty<FloatDetached>(event_count);
    raw.wedge_normal_z = empty<FloatDetached>(event_count);
    raw.wedge_prim_id = empty<IntDetached>(event_count);
    raw.wedge_dir_x = empty<FloatDetached>(event_count);
    raw.wedge_dir_y = empty<FloatDetached>(event_count);
    raw.wedge_dir_z = empty<FloatDetached>(event_count);
    raw.wedge_bounce_depth = empty<IntDetached>(event_count);
    return raw;
}

void initialize_reflection_accumulation_raw(ReflectionAccumulationRaw &raw) {
    const int zero_i = 0;
    const int minus_one_i = -1;
    const float zero_f = 0.f;
    const int event_count = std::max(1, raw.wedge_capacity);

    jit_memset_async(JitBackend::CUDA,
                     raw.reflection_power.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.reflection_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.wedge_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_ray_index.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_prim_id.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_bounce_depth.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
}

template <typename ArrayD>
ArrayD prefix_array(const ArrayD &value, int count) {
    return gather<ArrayD>(value, arange<IntDetached>(count));
}

template <typename ArrayD>
ArrayD concat_array_sequence(const std::vector<ArrayD> &parts) {
    require(!parts.empty(),
            "concat_array_sequence(): at least one array is required.");
    ArrayD result = parts.front();
    for (size_t i = 1; i < parts.size(); ++i) {
        result = concat(result, parts[i]);
    }
    return result;
}

IntDetached reflection_trace_ray_major_indices(int ray_count, int max_bounces) {
    const IntDetached slot = arange<IntDetached>(ray_count * max_bounces);
    const IntDetached ray_index = slot / IntDetached(max_bounces);
    const IntDetached bounce_index = slot - ray_index * IntDetached(max_bounces);
    return bounce_index * IntDetached(ray_count) + ray_index;
}

template <bool Detached>
MaskDetached sanitize_reflection_active(const RayT<Detached> &ray,
                                        MaskT<Detached> active) {
    MaskDetached active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) &&
                           drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) &&
                           drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &= squared_norm(Vector3fDetached(detach<false>(ray.d.x()),
                                                        detach<false>(ray.d.y()),
                                                        detach<false>(ray.d.z()))) > 0.f;
        active_detached &= ~drjit::isfinite(detach<false>(ray.tmax)) ||
                           (detach<false>(ray.tmax) > 0.f);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(ray.o.x()) &&
                           drjit::isfinite(ray.o.y()) &&
                           drjit::isfinite(ray.o.z());
        active_detached &= drjit::isfinite(ray.d.x()) &&
                           drjit::isfinite(ray.d.y()) &&
                           drjit::isfinite(ray.d.z());
        active_detached &= squared_norm(ray.d) > 0.f;
        active_detached &= ~drjit::isfinite(ray.tmax) || (ray.tmax > 0.f);
    }
    return active_detached;
}

template <bool Detached>
MaskDetached sanitize_segment_active(const Vector3fT<Detached> &start,
                                     const Vector3fT<Detached> &end,
                                     MaskT<Detached> active) {
    MaskDetached active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(start.x())) &&
                           drjit::isfinite(detach<false>(start.y())) &&
                           drjit::isfinite(detach<false>(start.z()));
        active_detached &= drjit::isfinite(detach<false>(end.x())) &&
                           drjit::isfinite(detach<false>(end.y())) &&
                           drjit::isfinite(detach<false>(end.z()));
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(start.x()) &&
                           drjit::isfinite(start.y()) &&
                           drjit::isfinite(start.z());
        active_detached &= drjit::isfinite(end.x()) &&
                           drjit::isfinite(end.y()) &&
                           drjit::isfinite(end.z());
    }
    return active_detached;
}

void ensure_segment_visibility_pipeline(
    std::unique_ptr<SegmentVisibilityPipeline> &pipeline,
    const OptixScene &optix_scene,
    int mesh_count) {
    if (!pipeline) {
        pipeline = std::make_unique<SegmentVisibilityPipeline>();
        pipeline->build(optix_scene.context(), mesh_count);
    }
}

void eval_segment_visibility_common(const Vector3fDetached &start,
                                    const IntDetached &face_offsets,
                                    const IntDetached &ignore_prim_ids,
                                    int ignore_k,
                                    const MaskDetached &active_detached) {
    if (ignore_k > 0) {
        drjit::eval(start, face_offsets, ignore_prim_ids, active_detached);
    } else {
        drjit::eval(start, face_offsets, active_detached);
    }
}

SegmentVisibilityParams make_segment_visibility_params(
    const OptixScene &optix_scene,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &start,
    const IntDetached &ignore_prim_ids,
    int ignore_k,
    const MaskDetached &active_detached,
    int ray_count) {
    SegmentVisibilityParams params = {};
    params.handle = optix_scene.ias_handle();
    params.face_offsets = face_offsets.data();
    params.n_meshes = mesh_count;
    params.start_x = start.x().data();
    params.start_y = start.y().data();
    params.start_z = start.z().data();
    params.ignore_prim_ids = ignore_k > 0 ? ignore_prim_ids.data() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = ray_count;
    return params;
}

MaskDetached launch_segment_visibility_detached(
    const OptixScene &optix_scene,
    const SegmentVisibilityPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &start,
    const Vector3fDetached &end,
    const IntDetached &ignore_prim_ids,
    int ignore_k,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    if (ray_count == 0) {
        return MaskDetached();
    }

    MaskDetached visible = empty<MaskDetached>(ray_count);
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       start,
                                       ignore_prim_ids,
                                       ignore_k,
                                       active_detached,
                                       ray_count);
    params.end_x = end.x().data();
    params.end_y = end.y().data();
    params.end_z = end.z().data();
    params.out_visible = reinterpret_cast<uint8_t *>(visible.data());
    pipeline.launch(SegmentVisibilityLaunchKind::Segment, params);
    return visible;
}

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3fDetached &start,
    const Vector3fDetached &end,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const OptixSegmentHit hit =
        optix_scene.segment_hit<true>(start, end, active_detached);
    if constexpr (!Detached) {
        result.visible = Mask(hit.visible);
    } else {
        result.visible = hit.visible;
    }
    return result;
}

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_native(
    const OptixScene &optix_scene,
    const SegmentVisibilityPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &start,
    const Vector3fDetached &end,
    const IntDetached &ignore_prim_ids,
    int ignore_k,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const MaskDetached visible_detached =
        launch_segment_visibility_detached(optix_scene,
                                           pipeline,
                                           face_offsets,
                                           mesh_count,
                                           start,
                                           end,
                                           ignore_prim_ids,
                                           ignore_k,
                                           active_detached);
    if constexpr (!Detached) {
        result.visible = Mask(visible_detached);
    } else {
        result.visible = visible_detached;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3fDetached &start,
    const Vector3fDetached &end_a,
    const Vector3fDetached &end_b,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const OptixSegmentHit hit_a =
        optix_scene.segment_hit<true>(start, end_a, active_detached);
    const OptixSegmentHit hit_b =
        optix_scene.segment_hit<true>(start, end_b, active_detached);
    if constexpr (!Detached) {
        result.visible_a = Mask(hit_a.visible);
        result.visible_b = Mask(hit_b.visible);
    } else {
        result.visible_a = hit_a.visible;
        result.visible_b = hit_b.visible;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_native(
    const OptixScene &optix_scene,
    const SegmentVisibilityPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &start,
    const Vector3fDetached &end_a,
    const Vector3fDetached &end_b,
    const IntDetached &ignore_prim_ids,
    int ignore_k,
    const MaskDetached &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    MaskDetached visible_a = empty<MaskDetached>(ray_count);
    MaskDetached visible_b = empty<MaskDetached>(ray_count);
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end_a, end_b);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       start,
                                       ignore_prim_ids,
                                       ignore_k,
                                       active_detached,
                                       ray_count);
    params.end_x = end_a.x().data();
    params.end_y = end_a.y().data();
    params.end_z = end_a.z().data();
    params.end_b_x = end_b.x().data();
    params.end_b_y = end_b.y().data();
    params.end_b_z = end_b.z().data();
    params.out_visible = reinterpret_cast<uint8_t *>(visible_a.data());
    params.out_visible_b = reinterpret_cast<uint8_t *>(visible_b.data());
    pipeline.launch(SegmentVisibilityLaunchKind::SegmentPair, params);

    if constexpr (!Detached) {
        result.visible_a = Mask(visible_a);
        result.visible_b = Mask(visible_b);
    } else {
        result.visible_a = visible_a;
        result.visible_b = visible_b;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_jit(
    const OptixScene &optix_scene,
    const Vector3fDetached &source_pos,
    const Vector3fDetached &edge_pos,
    const Vector3fDetached &edge_dir,
    const FloatDetached &edge_line_min,
    const FloatDetached &edge_line_max,
    const std::vector<float> &sample_fractions,
    const MaskDetached &active_detached) {
    const int state_count = static_cast<int>(slices(source_pos));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    MaskDetached any_visible = full<MaskDetached>(false, state_count);
    const FloatDetached span =
        maximum(edge_line_max - edge_line_min, FloatDetached(0.f));
    for (float fraction : sample_fractions) {
        const FloatDetached sample_t = edge_line_min + fraction * span;
        const Vector3fDetached sample_pos = edge_pos + sample_t * edge_dir;
        const OptixSegmentHit hit =
            optix_scene.segment_hit<true>(source_pos, sample_pos, active_detached);
        any_visible = any_visible || hit.visible;
    }

    if constexpr (!Detached) {
        result.any_visible = Mask(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_native(
    const OptixScene &optix_scene,
    const SegmentVisibilityPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &source_pos,
    const Vector3fDetached &edge_pos,
    const Vector3fDetached &edge_dir,
    const FloatDetached &edge_line_min,
    const FloatDetached &edge_line_max,
    const std::vector<float> &sample_fractions,
    const MaskDetached &active_detached) {
    const int state_count = static_cast<int>(slices(source_pos));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    MaskDetached any_visible = empty<MaskDetached>(state_count);
    drjit::eval(source_pos,
                edge_pos,
                edge_dir,
                edge_line_min,
                edge_line_max,
                face_offsets,
                active_detached);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       source_pos,
                                       IntDetached(),
                                       0,
                                       active_detached,
                                       state_count);
    params.end_x = edge_pos.x().data();
    params.end_y = edge_pos.y().data();
    params.end_z = edge_pos.z().data();
    params.edge_dir_x = edge_dir.x().data();
    params.edge_dir_y = edge_dir.y().data();
    params.edge_dir_z = edge_dir.z().data();
    params.edge_line_min = edge_line_min.data();
    params.edge_line_max = edge_line_max.data();
    params.sample_count = static_cast<int>(sample_fractions.size());
    for (size_t i = 0; i < sample_fractions.size(); ++i) {
        params.sample_fractions[i] = sample_fractions[i];
    }
    params.out_visible = reinterpret_cast<uint8_t *>(any_visible.data());
    pipeline.launch(SegmentVisibilityLaunchKind::AxialEdge, params);

    if constexpr (!Detached) {
        result.any_visible = Mask(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3fDetached &points,
    const IntDetached &chain_length,
    int chain_count,
    int max_points,
    int max_segments,
    const MaskDetached &active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    const IntDetached chain_index = arange<IntDetached>(chain_count);
    const IntDetached chain_base = chain_index * max_points;
    MaskDetached all_visible = active_detached;
    IntDetached first_blocked_segment = full<IntDetached>(-1, chain_count);
    IntDetached first_blocked_prim = full<IntDetached>(-1, chain_count);

    for (int segment = 0; segment < max_segments; ++segment) {
        const MaskDetached segment_active =
            active_detached && all_visible && (chain_length > segment);
        const IntDetached start_index = chain_base + segment;
        const Vector3fDetached start_point =
            gather<Vector3fDetached>(points, start_index, segment_active);
        const Vector3fDetached end_point =
            gather<Vector3fDetached>(points, start_index + 1, segment_active);
        const OptixSegmentHit hit =
            optix_scene.segment_hit<true>(start_point, end_point, segment_active);
        const MaskDetached blocked = segment_active && !hit.visible;
        all_visible &= !blocked;
        first_blocked_segment =
            select(blocked, IntDetached(segment), first_blocked_segment);
        first_blocked_prim =
            select(blocked, hit.global_prim_id, first_blocked_prim);
    }

    if constexpr (!Detached) {
        result.all_visible = Mask(all_visible);
        result.first_blocked_segment = Int(first_blocked_segment);
        result.first_blocked_prim = Int(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_native(
    const OptixScene &optix_scene,
    const SegmentVisibilityPipeline &pipeline,
    const IntDetached &face_offsets,
    int mesh_count,
    const Vector3fDetached &points,
    const IntDetached &chain_length,
    const IntDetached &ignore_prim_per_segment,
    int ignore_k,
    int chain_count,
    int max_points,
    int max_segments,
    const MaskDetached &active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    MaskDetached all_visible = empty<MaskDetached>(chain_count);
    IntDetached first_blocked_segment = empty<IntDetached>(chain_count);
    IntDetached first_blocked_prim = empty<IntDetached>(chain_count);
    if (ignore_k > 0) {
        drjit::eval(points,
                    chain_length,
                    ignore_prim_per_segment,
                    face_offsets,
                    active_detached);
    } else {
        drjit::eval(points, chain_length, face_offsets, active_detached);
    }

    SegmentVisibilityParams params = {};
    params.handle = optix_scene.ias_handle();
    params.face_offsets = face_offsets.data();
    params.n_meshes = mesh_count;
    params.chain_point_x = points.x().data();
    params.chain_point_y = points.y().data();
    params.chain_point_z = points.z().data();
    params.chain_length = chain_length.data();
    params.max_points = max_points;
    params.max_segments = max_segments;
    params.ignore_prim_ids = ignore_k > 0 ? ignore_prim_per_segment.data() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = chain_count;
    params.out_visible = reinterpret_cast<uint8_t *>(all_visible.data());
    params.out_first_blocked_segment = first_blocked_segment.data();
    params.out_first_blocked_prim = first_blocked_prim.data();
    pipeline.launch(SegmentVisibilityLaunchKind::SegmentChain, params);

    if constexpr (!Detached) {
        result.all_visible = Mask(all_visible);
        result.first_blocked_segment = Int(first_blocked_segment);
        result.first_blocked_prim = Int(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

template <bool Detached>
ReflectionTraceT<Detached> trace_bounces_impl(
    const Scene &scene,
    const RayT<Detached> &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskT<Detached> active) {
    require(!options.deduplicate,
            "Scene::trace_reflections(): deduplicate=true is not implemented with symbolic=true yet.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionTraceT<Detached> result =
        initialize_reflection_trace_result<Detached>(ray_count, max_bounces);
    result.deduplicate_requested = options.deduplicate;
    if (ray_count == 0) {
        return result;
    }

    const MaskDetached sanitized_active_detached =
        sanitize_reflection_active<Detached>(ray, active);

    RayT<Detached> current_ray = ray;
    MaskT<Detached> current_active;
    if constexpr (Detached) {
        current_active = sanitized_active_detached;
        result.representative_ray_index = arange<IntDetached>(ray_count);
    } else {
        current_active = Mask(sanitized_active_detached);
        result.representative_ray_index = Int(arange<IntDetached>(ray_count));
    }
    Vector3fT<Detached> current_image_source = ray.o;

    const FloatT<Detached> miss_t = full<FloatT<Detached>>(Infinity, ray_count);
    const IntT<Detached> miss_id = full<IntT<Detached>>(-1, ray_count);
    const Vector3fT<Detached> zero_v = zeros<Vector3fT<Detached>>(ray_count);
    const IntT<Detached> one_i = full<IntT<Detached>>(1, ray_count);
    const IntT<Detached> zero_i = full<IntT<Detached>>(0, ray_count);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        const IntersectionT<Detached> its =
            scene.template intersect<Detached>(current_ray, current_active, RayFlags::Geometric);
        const MaskT<Detached> bounce_hit = current_active && its.is_valid();

        Vector3fT<Detached> geo_normal = its.geo_n;
        geo_normal = select(dot(current_ray.d, geo_normal) > 0.f, -geo_normal, geo_normal);
        const FloatT<Detached> plane_distance =
            dot(current_image_source - its.p, geo_normal);
        const Vector3fT<Detached> reflected_image_source =
            current_image_source - 2.f * plane_distance * geo_normal;

        ReflectionBounceT<Detached> bounce_result =
            initialize_reflection_bounce_result<Detached>(ray_count);
        bounce_result.t = select(bounce_hit, its.t, miss_t);
        bounce_result.hit_points = select(bounce_hit, its.p, zero_v);
        bounce_result.geo_normals = select(bounce_hit, geo_normal, zero_v);
        bounce_result.image_sources = select(bounce_hit, reflected_image_source, zero_v);
        bounce_result.plane_points = bounce_result.hit_points;
        bounce_result.plane_normals = bounce_result.geo_normals;
        bounce_result.shape_ids = select(bounce_hit, its.shape_id, miss_id);
        bounce_result.prim_ids = select(bounce_hit, its.prim_id, miss_id);
        bounce_result.local_prim_ids = select(bounce_hit, its.local_prim_id, miss_id);
        bounce_result.global_prim_ids = select(bounce_hit, its.global_prim_id, miss_id);
        result.bounces.push_back(std::move(bounce_result));

        result.bounce_count += select(bounce_hit, one_i, zero_i);

        const FloatT<Detached> ray_dot_normal = dot(current_ray.d, geo_normal);
        const Vector3fT<Detached> reflected_direction =
            current_ray.d - 2.f * ray_dot_normal * geo_normal;
        current_ray.o = select(bounce_hit,
                               its.p + Epsilon * reflected_direction,
                               current_ray.o);
        current_ray.d = select(bounce_hit, reflected_direction, current_ray.d);
        current_ray.tmax = select(bounce_hit,
                                  full<FloatT<Detached>>(Infinity, ray_count),
                                  current_ray.tmax);
        current_image_source =
            select(bounce_hit, reflected_image_source, current_image_source);
        current_active = bounce_hit;
    }

    result.dedup_keep_mask = result.bounce_count > 0;
    result.discovery_count = select(result.dedup_keep_mask, one_i, zero_i);
    result.representative_ray_index =
        select(result.dedup_keep_mask,
               result.representative_ray_index,
               full<IntT<Detached>>(-1, ray_count));
    return result;
}

int face_edge_slot(const std::array<int, 3> &face_vertices, int v0, int v1) {
    auto matches = [v0, v1](int a, int b) {
        return (a == v0 && b == v1) || (a == v1 && b == v0);
    };

    if (matches(face_vertices[0], face_vertices[1])) {
        return 0;
    }
    if (matches(face_vertices[1], face_vertices[2])) {
        return 1;
    }
    if (matches(face_vertices[2], face_vertices[0])) {
        return 2;
    }
    return -1;
}

int face_opposite_vertex(const std::array<int, 3> &face_vertices, int v0, int v1) {
    for (int vertex : face_vertices) {
        if (vertex != v0 && vertex != v1) {
            return vertex;
        }
    }
    return -1;
}

} // namespace

Scene::Scene(const std::string &edge_bvh_backend)
    : optix_scene_(std::make_unique<OptixScene>()),
      optix_static_scene_(std::make_unique<OptixScene>()),
      optix_dynamic_scene_(std::make_unique<OptixScene>()),
      edge_bvh_(std::make_unique<SceneEdge>()),
      edge_optix_(std::make_unique<SceneEdgeOptix>()),
      edge_bvh_backend_(parse_edge_backend(edge_bvh_backend)) {}

Scene::~Scene() {
    for (Camera *camera : primary_edge_observers_) {
        if (camera != nullptr) {
            camera->clear_primary_edge_scene_binding(this);
        }
    }
}

std::string Scene::to_string() const {
    std::stringstream stream;
    stream << "Scene[num_meshes=" << mesh_count_
           << ", ready=" << is_ready()
           << ", pending_updates=" << pending_updates_
           << "]";
    return stream.str();
}

std::vector<const Mesh *> Scene::meshes() const {
    std::vector<const Mesh *> result;
    result.reserve(mesh_records_.size());
    for (const SceneMeshRecord &record : mesh_records_) {
        result.push_back(record.mesh.get());
    }
    return result;
}

int Scene::add_mesh(const Mesh &mesh, bool dynamic) {
    SceneMeshRecord record;
    record.mesh = std::make_unique<Mesh>(mesh);
    record.mesh->set_mesh_id(static_cast<int>(mesh_records_.size()));
    record.dynamic = dynamic;
    mesh_records_.push_back(std::move(record));

    mesh_count_ = static_cast<int>(mesh_records_.size());
    is_ready_ = false;
    pending_updates_ = false;
    vertex_offsets_ = IntDetached();
    global_geometry_ = SceneGlobalGeometry();
    edge_mask_ = MaskDetached();
    pending_edge_bvh_dirty_ranges_.clear();
    edge_bvh_dirty_ = false;
    mask_dirty_ = false;
    optix_split_active_ = false;
    optix_static_mesh_indices_.clear();
    optix_dynamic_mesh_indices_.clear();
    optix_dynamic_mesh_local_index_.clear();
    reflection_pipeline_.reset();
    segment_visibility_pipeline_.reset();
    invalidate_primary_edge_observers();
    return mesh_count_ - 1;
}

Scene::SceneMeshRecord &Scene::mesh_record(int mesh_id) {
    require(mesh_id >= 0 && mesh_id < static_cast<int>(mesh_records_.size()),
            "Scene: mesh_id is out of range.");
    return mesh_records_[static_cast<size_t>(mesh_id)];
}

const Scene::SceneMeshRecord &Scene::mesh_record(int mesh_id) const {
    require(mesh_id >= 0 && mesh_id < static_cast<int>(mesh_records_.size()),
            "Scene: mesh_id is out of range.");
    return mesh_records_[static_cast<size_t>(mesh_id)];
}

void Scene::scatter_mesh_data(const SceneMeshRecord &record, bool include_static) {
    const Mesh &mesh = *record.mesh;
    const int mesh_face_count = mesh.face_count();
    if (mesh_face_count == 0) {
        return;
    }

    const TriangleInfo *mesh_triangle_info = mesh.triangle_info();
    const Int scatter_indices = arange<Int>(mesh_face_count) + record.face_offset;
    const IntDetached scatter_indices_detached =
        arange<IntDetached>(mesh_face_count) + record.face_offset;

    scatter(triangle_info_.p0, mesh_triangle_info->p0, scatter_indices);
    scatter(triangle_info_.e1, mesh_triangle_info->e1, scatter_indices);
    scatter(triangle_info_.e2, mesh_triangle_info->e2, scatter_indices);
    scatter(triangle_info_.n0, mesh_triangle_info->n0, scatter_indices);
    scatter(triangle_info_.n1, mesh_triangle_info->n1, scatter_indices);
    scatter(triangle_info_.n2, mesh_triangle_info->n2, scatter_indices);
    scatter(triangle_info_.face_normal, mesh_triangle_info->face_normal, scatter_indices);
    scatter(triangle_info_.face_area, mesh_triangle_info->face_area, scatter_indices);

    scatter(triangle_info_detached_.p0, detach<false>(mesh_triangle_info->p0), scatter_indices_detached);
    scatter(triangle_info_detached_.e1, detach<false>(mesh_triangle_info->e1), scatter_indices_detached);
    scatter(triangle_info_detached_.e2, detach<false>(mesh_triangle_info->e2), scatter_indices_detached);
    scatter(triangle_info_detached_.n0, detach<false>(mesh_triangle_info->n0), scatter_indices_detached);
    scatter(triangle_info_detached_.n1, detach<false>(mesh_triangle_info->n1), scatter_indices_detached);
    scatter(triangle_info_detached_.n2, detach<false>(mesh_triangle_info->n2), scatter_indices_detached);
    scatter(triangle_info_detached_.face_normal,
            detach<false>(mesh_triangle_info->face_normal),
            scatter_indices_detached);
    scatter(triangle_info_detached_.face_area,
            detach<false>(mesh_triangle_info->face_area),
            scatter_indices_detached);

    if (!include_static) {
        return;
    }

    scatter(triangle_info_.face_indices, mesh_triangle_info->face_indices, scatter_indices);
    scatter(triangle_info_detached_.face_indices,
            detach<false>(mesh_triangle_info->face_indices),
            scatter_indices_detached);
    scatter(triangle_face_normal_mask_,
            full<Mask>(mesh.use_face_normals(), mesh_face_count),
            scatter_indices);
    scatter(triangle_face_normal_mask_detached_,
            full<MaskDetached>(mesh.use_face_normals(), mesh_face_count),
            scatter_indices_detached);

    if (mesh.has_uv() && mesh.triangle_uv() != nullptr) {
        scatter(triangle_uv_[0], (*mesh.triangle_uv())[0], scatter_indices);
        scatter(triangle_uv_[1], (*mesh.triangle_uv())[1], scatter_indices);
        scatter(triangle_uv_[2], (*mesh.triangle_uv())[2], scatter_indices);

        scatter(triangle_uv_detached_[0], detach<false>((*mesh.triangle_uv())[0]), scatter_indices_detached);
        scatter(triangle_uv_detached_[1], detach<false>((*mesh.triangle_uv())[1]), scatter_indices_detached);
        scatter(triangle_uv_detached_[2], detach<false>((*mesh.triangle_uv())[2]), scatter_indices_detached);
    }
}

void Scene::scatter_mesh_edge_data(const SceneMeshRecord &record, bool include_static_ids) {
    const Mesh &mesh = *record.mesh;
    const SecondaryEdgeInfo *mesh_edge_info = mesh.secondary_edge_info();
    const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
    if (mesh_edge_count == 0) {
        return;
    }

    const Int scatter_indices = arange<Int>(mesh_edge_count) + record.edge_offset;
    scatter(edge_info_.start, mesh_edge_info->start, scatter_indices);
    scatter(edge_info_.edge, mesh_edge_info->edge, scatter_indices);
    scatter(edge_info_.normal0, mesh_edge_info->normal0, scatter_indices);
    scatter(edge_info_.normal1, mesh_edge_info->normal1, scatter_indices);
    scatter(edge_info_.opposite, mesh_edge_info->opposite, scatter_indices);
    scatter(edge_info_.is_boundary, mesh_edge_info->is_boundary, scatter_indices);

    if (!include_static_ids) {
        return;
    }

    const IntDetached scatter_indices_detached = arange<IntDetached>(mesh_edge_count) + record.edge_offset;
    scatter(edge_shape_ids_,
            full<IntDetached>(mesh.mesh_id(), mesh_edge_count),
            scatter_indices_detached);
    scatter(edge_local_ids_,
            arange<IntDetached>(mesh_edge_count),
            scatter_indices_detached);
}

void Scene::ensure_scene_edge_data_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        if (!record.edge_dirty) {
            continue;
        }

        const_cast<Scene *>(this)->scatter_mesh_edge_data(record, false);
        record.edge_dirty = false;
    }

    ensure_edge_bvh_ready();
}

void Scene::ensure_edge_bvh_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    Scene *scene = const_cast<Scene *>(this);
    if (mask_dirty_) {
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            scene->edge_bvh_->set_mask(scene->edge_mask_);
            if (edge_backend_builds_optix(edge_bvh_backend_)) {
                scene->edge_bvh_->materialize();
            }
        }
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            scene->edge_optix_->set_mask(scene->edge_mask_);
            if (edge_backend_builds_drjit(edge_bvh_backend_)) {
                scene->edge_bvh_->materialize();
            }
        }
        scene->mask_dirty_ = false;
    }

    if (pending_edge_bvh_dirty_ranges_.empty()) {
        scene->edge_bvh_dirty_ = false;
        return;
    }

    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        scene->edge_bvh_->refit(scene->edge_info_, scene->pending_edge_bvh_dirty_ranges_);
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            scene->edge_bvh_->materialize();
        }
    }
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        scene->edge_optix_->refit(scene->edge_info_, scene->pending_edge_bvh_dirty_ranges_);
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            scene->edge_bvh_->materialize();
        }
    }
    scene->pending_edge_bvh_dirty_ranges_.clear();
    scene->edge_bvh_dirty_ = false;
}

void Scene::register_primary_edge_observer(Camera *camera) {
    auto it = std::find(primary_edge_observers_.begin(), primary_edge_observers_.end(), camera);
    if (it == primary_edge_observers_.end()) {
        primary_edge_observers_.push_back(camera);
    }
}

void Scene::unregister_primary_edge_observer(Camera *camera) {
    auto it = std::remove(primary_edge_observers_.begin(), primary_edge_observers_.end(), camera);
    primary_edge_observers_.erase(it, primary_edge_observers_.end());
}

void Scene::invalidate_primary_edge_observers() {
    for (Camera *camera : primary_edge_observers_) {
        if (camera != nullptr) {
            camera->invalidate_primary_edges_from_scene(this);
        }
    }
}

void Scene::build() {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::Build);
    require(!mesh_records_.empty(), "Scene::build(): missing meshes.");

    std::vector<int> face_offsets;
    face_offsets.reserve(mesh_records_.size() + 1);
    face_offsets.push_back(0);

    std::vector<int> vertex_offsets;
    vertex_offsets.reserve(mesh_records_.size() + 1);
    vertex_offsets.push_back(0);

    std::vector<int> edge_offsets;
    edge_offsets.reserve(mesh_records_.size() + 1);
    edge_offsets.push_back(0);

    std::vector<OptixSceneMeshDesc> mesh_descs;
    mesh_descs.reserve(mesh_records_.size());

    std::vector<int> topology_v0;
    std::vector<int> topology_v1;
    std::vector<int> topology_v0_global;
    std::vector<int> topology_v1_global;
    std::vector<int> topology_face0_local;
    std::vector<int> topology_face1_local;
    std::vector<int> topology_face0_global;
    std::vector<int> topology_face1_global;
    std::vector<int> topology_opposite0;
    std::vector<int> topology_opposite1;
    std::vector<int> topology_opposite0_global;
    std::vector<int> topology_opposite1_global;

    for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
        SceneMeshRecord &record = mesh_records_[mesh_index];
        Mesh &mesh = *record.mesh;
        mesh.set_mesh_id(static_cast<int>(mesh_index));
        mesh.build();
        record.vertex_offset = vertex_offsets.back();
        record.face_offset = face_offsets.back();
        const SecondaryEdgeInfo *mesh_edge_info = mesh.secondary_edge_info();
        const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
        record.edge_offset = edge_offsets.back();
        record.vertices_dirty = false;
        record.transform_dirty = false;
        record.edge_dirty = false;

        vertex_offsets.push_back(vertex_offsets.back() + mesh.vertex_count());
        face_offsets.push_back(face_offsets.back() + mesh.face_count());
        edge_offsets.push_back(edge_offsets.back() + mesh_edge_count);
        mesh_descs.push_back({ &mesh, record.dynamic, record.face_offset, static_cast<int>(mesh_index) });
    }

    mesh_count_ = static_cast<int>(mesh_records_.size());
    const int total_vertex_count = vertex_offsets.back();
    const int total_face_count = face_offsets.back();
    require(total_face_count > 0, "Scene::build(): scene has no triangles.");

    edge_count_ = edge_offsets.back();
    topology_v0.reserve(edge_count_);
    topology_v1.reserve(edge_count_);
    topology_v0_global.reserve(edge_count_);
    topology_v1_global.reserve(edge_count_);
    topology_face0_local.reserve(edge_count_);
    topology_face1_local.reserve(edge_count_);
    topology_face0_global.reserve(edge_count_);
    topology_face1_global.reserve(edge_count_);
    topology_opposite0.reserve(edge_count_);
    topology_opposite1.reserve(edge_count_);
    topology_opposite0_global.reserve(edge_count_);
    topology_opposite1_global.reserve(edge_count_);

    std::array<std::vector<int>, 3> global_face_indices_cpu;
    for (auto &global_face_indices : global_face_indices_cpu) {
        global_face_indices.reserve(total_face_count);
    }
    std::vector<int> global_shape_ids_cpu;
    std::vector<int> global_local_prim_ids_cpu;
    std::vector<int> global_prim_ids_cpu;
    global_shape_ids_cpu.reserve(total_face_count);
    global_local_prim_ids_cpu.reserve(total_face_count);
    global_prim_ids_cpu.reserve(total_face_count);

    std::array<std::vector<int>, 3> triangle_edge_ids_cpu;
    for (auto &triangle_edge_ids : triangle_edge_ids_cpu) {
        triangle_edge_ids.assign(total_face_count, -1);
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        const Mesh &mesh = *record.mesh;
        const auto &mesh_edge_indices = mesh.edge_indices();
        const int mesh_edge_count = mesh.edges_enabled() ? static_cast<int>(slices(mesh_edge_indices)) : 0;
        const Vector3iDetached mesh_face_indices(detach<false>(mesh.face_indices()[0]),
                                                 detach<false>(mesh.face_indices()[1]),
                                                 detach<false>(mesh.face_indices()[2]));
        std::array<std::vector<int>, 3> mesh_face_cpu;
        copy_cuda_array(mesh_face_indices, mesh_face_cpu);
        for (int local_face_id = 0; local_face_id < mesh.face_count(); ++local_face_id) {
            global_face_indices_cpu[0].push_back(record.vertex_offset + mesh_face_cpu[0][local_face_id]);
            global_face_indices_cpu[1].push_back(record.vertex_offset + mesh_face_cpu[1][local_face_id]);
            global_face_indices_cpu[2].push_back(record.vertex_offset + mesh_face_cpu[2][local_face_id]);
            global_shape_ids_cpu.push_back(mesh.mesh_id());
            global_local_prim_ids_cpu.push_back(local_face_id);
            global_prim_ids_cpu.push_back(record.face_offset + local_face_id);
        }

        if (mesh_edge_count == 0) {
            continue;
        }

        std::array<std::vector<int>, 5> mesh_edge_cpu;
        copy_cuda_array(mesh_edge_indices, mesh_edge_cpu);

        for (int local_edge_id = 0; local_edge_id < mesh_edge_count; ++local_edge_id) {
            const int v0 = mesh_edge_cpu[0][local_edge_id];
            const int v1 = mesh_edge_cpu[1][local_edge_id];
            const int v0_global = record.vertex_offset + v0;
            const int v1_global = record.vertex_offset + v1;
            const int face0_local = mesh_edge_cpu[2][local_edge_id];
            const int face1_local = mesh_edge_cpu[3][local_edge_id];
            const int face0_global = record.face_offset + face0_local;
            const int face1_global = face1_local >= 0 ? record.face_offset + face1_local : -1;
            const int opposite0 = mesh_edge_cpu[4][local_edge_id];
            const int opposite0_global = opposite0 >= 0 ? record.vertex_offset + opposite0 : -1;
            const int global_edge_id = record.edge_offset + local_edge_id;

            const std::array<int, 3> face0_vertices {
                mesh_face_cpu[0][face0_local],
                mesh_face_cpu[1][face0_local],
                mesh_face_cpu[2][face0_local]
            };

            int opposite1 = -1;
            if (face1_local >= 0) {
                const std::array<int, 3> face1_vertices {
                    mesh_face_cpu[0][face1_local],
                    mesh_face_cpu[1][face1_local],
                    mesh_face_cpu[2][face1_local]
                };
                opposite1 = face_opposite_vertex(face1_vertices, v0, v1);
                const int opposite1_global = opposite1 >= 0 ? record.vertex_offset + opposite1 : -1;
                const int face1_slot = face_edge_slot(face1_vertices, v0, v1);
                if (face1_slot >= 0) {
                    triangle_edge_ids_cpu[face1_slot][face1_global] = global_edge_id;
                }
                topology_opposite1_global.push_back(opposite1_global);
            } else {
                topology_opposite1_global.push_back(-1);
            }

            const int face0_slot = face_edge_slot(face0_vertices, v0, v1);
            if (face0_slot >= 0) {
                triangle_edge_ids_cpu[face0_slot][face0_global] = global_edge_id;
            }

            topology_v0.push_back(v0);
            topology_v1.push_back(v1);
            topology_v0_global.push_back(v0_global);
            topology_v1_global.push_back(v1_global);
            topology_face0_local.push_back(face0_local);
            topology_face1_local.push_back(face1_local);
            topology_face0_global.push_back(face0_global);
            topology_face1_global.push_back(face1_global);
            topology_opposite0.push_back(opposite0);
            topology_opposite1.push_back(opposite1);
            topology_opposite0_global.push_back(opposite0_global);
        }
    }

    auto load_or_empty = [](const std::vector<int> &values) {
        return values.empty() ? IntDetached() : load<IntDetached>(values.data(), values.size());
    };

    face_offsets_ = load<IntDetached>(face_offsets.data(), face_offsets.size());
    edge_offsets_ = load<IntDetached>(edge_offsets.data(), edge_offsets.size());
    vertex_offsets_ = load<IntDetached>(vertex_offsets.data(), vertex_offsets.size());
    triangle_info_ = empty<TriangleInfo>(total_face_count);
    triangle_info_detached_ = empty<TriangleInfoDetached>(total_face_count);
    triangle_uv_ = zeros<TriangleUV>(total_face_count);
    triangle_uv_detached_ = zeros<TriangleUVDetached>(total_face_count);
    triangle_face_normal_mask_ = empty<Mask>(total_face_count);
    triangle_face_normal_mask_detached_ = empty<MaskDetached>(total_face_count);
    global_geometry_.vertices = total_vertex_count > 0 ? empty<Vector3f>(total_vertex_count) : Vector3f();
    global_geometry_.faces = Vector3iDetached(
        load<IntDetached>(global_face_indices_cpu[0].data(), total_face_count),
        load<IntDetached>(global_face_indices_cpu[1].data(), total_face_count),
        load<IntDetached>(global_face_indices_cpu[2].data(), total_face_count));
    global_geometry_.shape_id = load<IntDetached>(global_shape_ids_cpu.data(), total_face_count);
    global_geometry_.local_prim_id =
        load<IntDetached>(global_local_prim_ids_cpu.data(), total_face_count);
    global_geometry_.global_prim_id = load<IntDetached>(global_prim_ids_cpu.data(), total_face_count);
    triangle_edge_ids_ = VectoriT<3, true>(load<IntDetached>(triangle_edge_ids_cpu[0].data(), total_face_count),
                                           load<IntDetached>(triangle_edge_ids_cpu[1].data(), total_face_count),
                                           load<IntDetached>(triangle_edge_ids_cpu[2].data(), total_face_count));
    if (edge_count_ > 0) {
        edge_info_ = empty<SecondaryEdgeInfo>(edge_count_);
        edge_topology_ = SceneEdgeTopology {
            load_or_empty(topology_v0),
            load_or_empty(topology_v1),
            load_or_empty(topology_v0_global),
            load_or_empty(topology_v1_global),
            load_or_empty(topology_face0_local),
            load_or_empty(topology_face1_local),
            load_or_empty(topology_face0_global),
            load_or_empty(topology_face1_global),
            load_or_empty(topology_opposite0),
            load_or_empty(topology_opposite1),
            load_or_empty(topology_opposite0_global),
            load_or_empty(topology_opposite1_global)
        };
        edge_shape_ids_ = empty<IntDetached>(edge_count_);
        edge_local_ids_ = empty<IntDetached>(edge_count_);
        edge_mask_ = full<MaskDetached>(true, edge_count_);
    } else {
        edge_info_ = SecondaryEdgeInfo();
        edge_topology_ = SceneEdgeTopology();
        edge_shape_ids_ = IntDetached();
        edge_local_ids_ = IntDetached();
        edge_mask_ = MaskDetached();
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        scatter_mesh_data(record, true);
        scatter_mesh_edge_data(record, true);
        const Mesh &mesh = *record.mesh;
        const int mesh_vertex_count = mesh.vertex_count();
        if (mesh_vertex_count > 0) {
            const Int vertex_scatter_indices = arange<Int>(mesh_vertex_count) + record.vertex_offset;
            scatter(global_geometry_.vertices, mesh.vertex_positions_world(), vertex_scatter_indices);
        }
    }
    global_geometry_.face_normal = triangle_info_.face_normal;

    int static_mesh_count = 0;
    int dynamic_mesh_count = 0;
    for (const SceneMeshRecord &record : mesh_records_) {
        if (record.dynamic) {
            ++dynamic_mesh_count;
        } else {
            ++static_mesh_count;
        }
    }

    optix_split_active_ =
        should_split_optix_scene(active_optix_split_mode(), static_mesh_count, dynamic_mesh_count);
    optix_static_mesh_indices_.clear();
    optix_dynamic_mesh_indices_.clear();
    optix_dynamic_mesh_local_index_.assign(mesh_records_.size(), -1);

    if (optix_split_active_) {
        std::vector<OptixSceneMeshDesc> static_mesh_descs;
        std::vector<OptixSceneMeshDesc> dynamic_mesh_descs;
        static_mesh_descs.reserve(static_mesh_count);
        dynamic_mesh_descs.reserve(dynamic_mesh_count);

        for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
            if (mesh_records_[mesh_index].dynamic) {
                optix_dynamic_mesh_local_index_[mesh_index] =
                    static_cast<int>(dynamic_mesh_descs.size());
                optix_dynamic_mesh_indices_.push_back(static_cast<int>(mesh_index));
                dynamic_mesh_descs.push_back(mesh_descs[mesh_index]);
            } else {
                optix_static_mesh_indices_.push_back(static_cast<int>(mesh_index));
                static_mesh_descs.push_back(mesh_descs[mesh_index]);
            }
        }

        optix_scene_ = std::make_unique<OptixScene>();
        optix_static_scene_ = std::make_unique<OptixScene>();
        optix_dynamic_scene_ = std::make_unique<OptixScene>();
        optix_scene_->build(mesh_descs);
        optix_static_scene_->build(static_mesh_descs, optix_scene_.get());
        optix_dynamic_scene_->build(dynamic_mesh_descs, optix_scene_.get());
    } else {
        optix_scene_ = std::make_unique<OptixScene>();
        optix_static_scene_ = std::make_unique<OptixScene>();
        optix_dynamic_scene_ = std::make_unique<OptixScene>();
        optix_scene_->build(mesh_descs);
    }
    reflection_pipeline_.reset();
    segment_visibility_pipeline_.reset();
    mask_dirty_ = false;
    edge_bvh_ = std::make_unique<SceneEdge>();
    edge_optix_ = std::make_unique<SceneEdgeOptix>();
    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        edge_bvh_->build(edge_info_, edge_mask_);
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            edge_bvh_->materialize();
        }
    }
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        edge_optix_->build(edge_info_, edge_mask_);
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            edge_bvh_->materialize();
        }
    }
    is_ready_ = true;
    pending_updates_ = false;
    ++scene_version_;
    ++edge_version_;
    invalidate_primary_edge_observers();
}

void Scene::update_mesh_vertices(int mesh_id, const Vector3f &positions) {
    require(is_ready(), "Scene::update_mesh_vertices(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::update_mesh_vertices(): target mesh is not dynamic.");
    require(static_cast<int>(slices(positions)) == record.mesh->vertex_count(),
            "Scene::update_mesh_vertices(): vertex count must remain unchanged.");

    record.mesh->set_vertex_positions(positions);
    record.vertices_dirty = true;
    pending_updates_ = true;
}

void Scene::set_mesh_transform(int mesh_id, const Matrix4f &matrix, bool set_left) {
    require(is_ready(), "Scene::set_mesh_transform(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::set_mesh_transform(): target mesh is not dynamic.");

    record.mesh->set_transform(matrix, set_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::append_mesh_transform(int mesh_id, const Matrix4f &matrix, bool append_left) {
    require(is_ready(), "Scene::append_mesh_transform(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::append_mesh_transform(): target mesh is not dynamic.");

    record.mesh->append_transform(matrix, append_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::set_edge_mask(const MaskDetached &mask) {
    require(is_ready(), "Scene::set_edge_mask(): scene is not built.");
    require(static_cast<int>(mask.size()) == edge_count_,
            "Scene::set_edge_mask(): mask size must match the scene edge count.");

    if (mask.size() == edge_mask_.size() && drjit::all(mask == edge_mask_)) {
        return;
    }

    edge_mask_ = mask;
    mask_dirty_ = true;
    edge_bvh_dirty_ = true;
    pending_updates_ = true;
}

void Scene::sync() {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::Sync);
    require(is_ready(), "Scene::sync(): scene is not built.");
    last_sync_profile_ = SceneSyncProfile();

    if (!pending_updates_) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    const bool mask_dirty_before = mask_dirty_;

    std::vector<OptixSceneMeshDesc> mesh_descs;
    mesh_descs.reserve(mesh_records_.size());

    std::vector<OptixSceneMeshUpdate> updates;
    updates.reserve(mesh_records_.size());

    for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
        SceneMeshRecord &record = mesh_records_[mesh_index];
        mesh_descs.push_back({ record.mesh.get(), record.dynamic, record.face_offset, static_cast<int>(mesh_index) });

        if (!record.vertices_dirty && !record.transform_dirty) {
            continue;
        }

        const auto mesh_update_start = Clock::now();
        record.mesh->update_runtime_data(record.vertices_dirty, record.transform_dirty);
        last_sync_profile_.mesh_update_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - mesh_update_start).count();

        const auto scatter_start = Clock::now();
        scatter_mesh_data(record, false);
        const int mesh_vertex_count = record.mesh->vertex_count();
        if (mesh_vertex_count > 0) {
            const Int vertex_scatter_indices = arange<Int>(mesh_vertex_count) + record.vertex_offset;
            scatter(global_geometry_.vertices,
                    record.mesh->vertex_positions_world(),
                    vertex_scatter_indices);
        }
        last_sync_profile_.triangle_scatter_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - scatter_start).count();

        const int mesh_edge_count =
            record.mesh->edges_enabled() ? static_cast<int>(slices(record.mesh->edge_indices())) : 0;
        if (mesh_edge_count > 0 && !record.edge_dirty) {
            pending_edge_bvh_dirty_ranges_.push_back({ record.edge_offset, mesh_edge_count });
            record.edge_dirty = true;
            edge_bvh_dirty_ = true;
            ++last_sync_profile_.updated_edge_meshes;
            last_sync_profile_.updated_edges += mesh_edge_count;
        }

        updates.push_back({ static_cast<int>(mesh_index), record.vertices_dirty, record.transform_dirty });
        ++last_sync_profile_.updated_meshes;
        if (record.vertices_dirty) {
            ++last_sync_profile_.updated_vertex_meshes;
        }
        if (record.transform_dirty) {
            ++last_sync_profile_.updated_transform_meshes;
        }
        record.vertices_dirty = false;
        record.transform_dirty = false;
    }
    if (!updates.empty()) {
        global_geometry_.face_normal = triangle_info_.face_normal;
    }

    if (edge_bvh_dirty_) {
        const auto edge_scatter_start = Clock::now();
        for (SceneMeshRecord &record : mesh_records_) {
            if (!record.edge_dirty) {
                continue;
            }

            scatter_mesh_edge_data(record, false);
            record.edge_dirty = false;
        }
        last_sync_profile_.edge_scatter_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - edge_scatter_start).count();

        const auto edge_refit_start = Clock::now();
        ensure_edge_bvh_ready();
        last_sync_profile_.edge_refit_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - edge_refit_start).count();
    }

    const auto optix_start = Clock::now();
    if (optix_split_active_) {
        if (!updates.empty()) {
            optix_scene_->sync(mesh_descs, updates);
        }

        std::vector<OptixSceneMeshDesc> dynamic_mesh_descs;
        dynamic_mesh_descs.reserve(optix_dynamic_mesh_indices_.size());
        for (int mesh_index : optix_dynamic_mesh_indices_) {
            dynamic_mesh_descs.push_back(mesh_descs[static_cast<size_t>(mesh_index)]);
        }

        std::vector<OptixSceneMeshUpdate> dynamic_updates;
        dynamic_updates.reserve(updates.size());
        for (const OptixSceneMeshUpdate &update : updates) {
            const int dynamic_local_index =
                optix_dynamic_mesh_local_index_[static_cast<size_t>(update.mesh_id)];
            if (dynamic_local_index < 0) {
                continue;
            }
            dynamic_updates.push_back(
                { dynamic_local_index, update.vertices_dirty, update.transform_dirty });
        }

        if (!dynamic_updates.empty()) {
            optix_dynamic_scene_->sync(dynamic_mesh_descs, dynamic_updates);
        }
        last_sync_profile_.optix_sync_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - optix_start).count();
        if (!updates.empty()) {
            const OptixSyncProfile &optix_profile = optix_scene_->last_sync_profile();
            last_sync_profile_.optix_gas_update_ms += optix_profile.gas_update_ms;
            last_sync_profile_.optix_ias_update_ms += optix_profile.ias_update_ms;
        }
        if (!dynamic_updates.empty()) {
            const OptixSyncProfile &optix_profile = optix_dynamic_scene_->last_sync_profile();
            last_sync_profile_.optix_gas_update_ms += optix_profile.gas_update_ms;
            last_sync_profile_.optix_ias_update_ms += optix_profile.ias_update_ms;
        }
    } else {
        optix_scene_->sync(mesh_descs, updates);
        last_sync_profile_.optix_sync_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - optix_start).count();
        const OptixSyncProfile &optix_profile = optix_scene_->last_sync_profile();
        last_sync_profile_.optix_gas_update_ms = optix_profile.gas_update_ms;
        last_sync_profile_.optix_ias_update_ms = optix_profile.ias_update_ms;
    }
    pending_updates_ = false;
    if (!updates.empty()) {
        ++scene_version_;
    }
    if (mask_dirty_before || last_sync_profile_.updated_edge_meshes > 0) {
        ++edge_version_;
    }
    if (!updates.empty()) {
        invalidate_primary_edge_observers();
    }
    last_sync_profile_.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
}

SceneEdgeInfo Scene::edge_info() const {
    require(is_ready(), "Scene::edge_info(): scene is not built.");
    require(!pending_updates_, "Scene::edge_info(): scene has pending updates. Call Scene::sync() first.");

    ensure_scene_edge_data_ready();

    SceneEdgeInfo info;
    info.start = edge_info_.start;
    info.edge = edge_info_.edge;
    info.end = edge_info_.start + edge_info_.edge;
    info.length = norm(edge_info_.edge);
    info.normal0 = edge_info_.normal0;
    info.normal1 = edge_info_.normal1;
    info.is_boundary = edge_info_.is_boundary;
    info.shape_id = edge_shape_ids_;
    info.local_edge_id = edge_local_ids_;
    info.global_edge_id = arange<IntDetached>(edge_count_);
    return info;
}

std::string Scene::edge_bvh_backend() const {
    return edge_backend_name(edge_bvh_backend_);
}

SceneEdgeBVHStats Scene::edge_bvh_stats() const {
    require(is_ready(), "Scene::edge_bvh_stats(): scene is not built.");
    require(!pending_updates_,
            "Scene::edge_bvh_stats(): scene has pending updates. Call Scene::sync() first.");
    ensure_edge_bvh_ready();
    return edge_bvh_backend_ == EdgeBVHBackend::Optix ? edge_optix_->stats() : edge_bvh_->stats();
}

const SceneEdgeTopology &Scene::edge_topology() const {
    require(is_ready(), "Scene::edge_topology(): scene is not built.");
    return edge_topology_;
}

const MaskDetached &Scene::edge_mask() const {
    require(is_ready(), "Scene::edge_mask(): scene is not built.");
    return edge_mask_;
}

const SceneGlobalGeometry &Scene::global_geometry() const {
    require(is_ready(), "Scene::global_geometry(): scene is not built.");
    require(!pending_updates_,
            "Scene::global_geometry(): scene has pending updates. Call Scene::sync() first.");
    return global_geometry_;
}

VectoriT<3, true> Scene::triangle_edge_indices(const IntDetached &prim_id, bool global) const {
    require(is_ready(), "Scene::triangle_edge_indices(): scene is not built.");

    const int query_count = static_cast<int>(slices(prim_id));
    VectoriT<3, true> result(full<IntDetached>(-1, query_count),
                             full<IntDetached>(-1, query_count),
                             full<IntDetached>(-1, query_count));
    if (query_count == 0) {
        return result;
    }

    const int face_count = static_cast<int>(slices(triangle_edge_ids_[0]));
    const MaskDetached valid = prim_id >= 0 && prim_id < face_count;
    const IntDetached edge0 = gather<IntDetached>(triangle_edge_ids_[0], prim_id, valid);
    const IntDetached edge1 = gather<IntDetached>(triangle_edge_ids_[1], prim_id, valid);
    const IntDetached edge2 = gather<IntDetached>(triangle_edge_ids_[2], prim_id, valid);

    if (global) {
        result[0] = select(valid, edge0, result[0]);
        result[1] = select(valid, edge1, result[1]);
        result[2] = select(valid, edge2, result[2]);
        return result;
    }

    const MaskDetached valid0 = valid && edge0 >= 0;
    const MaskDetached valid1 = valid && edge1 >= 0;
    const MaskDetached valid2 = valid && edge2 >= 0;
    result[0] = select(valid0, gather<IntDetached>(edge_local_ids_, edge0, valid0), result[0]);
    result[1] = select(valid1, gather<IntDetached>(edge_local_ids_, edge1, valid1), result[1]);
    result[2] = select(valid2, gather<IntDetached>(edge_local_ids_, edge2, valid2), result[2]);
    return result;
}

VectoriT<2, true> Scene::edge_adjacent_faces(const IntDetached &edge_id, bool global) const {
    require(is_ready(), "Scene::edge_adjacent_faces(): scene is not built.");

    const int query_count = static_cast<int>(slices(edge_id));
    VectoriT<2, true> result(full<IntDetached>(-1, query_count),
                             full<IntDetached>(-1, query_count));
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    const MaskDetached valid = edge_id >= 0 && edge_id < edge_count_;
    const IntDetached face0 = global
        ? gather<IntDetached>(edge_topology_.face0_global, edge_id, valid)
        : gather<IntDetached>(edge_topology_.face0_local, edge_id, valid);
    const IntDetached face1 = global
        ? gather<IntDetached>(edge_topology_.face1_global, edge_id, valid)
        : gather<IntDetached>(edge_topology_.face1_local, edge_id, valid);
    result[0] = select(valid, face0, result[0]);
    result[1] = select(valid, face1, result[1]);
    return result;
}

bool Scene::is_ready() const {
    const bool optix_ready =
        optix_split_active_
            ? (optix_scene_ != nullptr && optix_static_scene_ != nullptr &&
               optix_dynamic_scene_ != nullptr && optix_scene_->is_ready() &&
               optix_static_scene_->is_ready() && optix_dynamic_scene_->is_ready())
            : (optix_scene_ != nullptr && optix_scene_->is_ready());
    bool edge_ready = true;
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        edge_ready &= edge_optix_ != nullptr && edge_optix_->is_ready();
    }
    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        edge_ready &= edge_bvh_ != nullptr && edge_bvh_->is_ready();
    }
    return is_ready_ && edge_ready && optix_ready;
}

template <bool Detached>
IntersectionT<Detached> Scene::intersect(const RayT<Detached> &ray, MaskT<Detached> active, RayFlags flags) const {
    require(is_ready(), "Scene::intersect(): scene is not built.");
    require(!pending_updates_, "Scene::intersect(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(ray.o));
    const bool want_geo_n   = has_flag(flags, RayFlags::Geometric);
    const bool want_shading = has_flag(flags, RayFlags::ShadingN);
    const bool want_uv      = has_flag(flags, RayFlags::UV);
    const bool symbolic_optix_query = optix_split_active_ && uses_symbolic_optix_query_path();

    IntersectionT<Detached> intersection;
    intersection.t = full<FloatT<Detached>>(Infinity, ray_count);
    intersection.p = zeros<Vector3fT<Detached>>(ray_count);
    intersection.n = zeros<Vector3fT<Detached>>(ray_count);
    intersection.geo_n = zeros<Vector3fT<Detached>>(ray_count);
    intersection.uv = zeros<Vector2fT<Detached>>(ray_count);
    intersection.barycentric = zeros<Vector3fT<Detached>>(ray_count);
    intersection.shape_id = full<IntT<Detached>>(-1, ray_count);
    intersection.prim_id = full<IntT<Detached>>(-1, ray_count);

    MaskT<Detached> hit_mask = active;
    OptixIntersection optix_hit;
    if (optix_split_active_ && !symbolic_optix_query) {
        MaskT<Detached> static_hit_mask = active;
        MaskT<Detached> dynamic_hit_mask = active;
        const OptixIntersection static_hit =
            optix_static_scene_->template intersect<Detached>(ray, static_hit_mask);
        const OptixIntersection dynamic_hit =
            optix_dynamic_scene_->template intersect<Detached>(ray, dynamic_hit_mask);

        const MaskDetached static_hit_mask_detached = detach<false>(static_hit_mask);
        const MaskDetached dynamic_hit_mask_detached = detach<false>(dynamic_hit_mask);
        const MaskDetached choose_dynamic =
            dynamic_hit_mask_detached &&
            (!static_hit_mask_detached || (dynamic_hit.t < static_hit.t));
        const MaskDetached any_hit = static_hit_mask_detached || dynamic_hit_mask_detached;

        optix_hit.reserve(ray_count);
        optix_hit.t = select(choose_dynamic, dynamic_hit.t, static_hit.t);
        optix_hit.barycentric[0] =
            select(choose_dynamic, dynamic_hit.barycentric[0], static_hit.barycentric[0]);
        optix_hit.barycentric[1] =
            select(choose_dynamic, dynamic_hit.barycentric[1], static_hit.barycentric[1]);
        optix_hit.shape_id = select(choose_dynamic, dynamic_hit.shape_id, static_hit.shape_id);
        optix_hit.global_prim_id =
            select(choose_dynamic, dynamic_hit.global_prim_id, static_hit.global_prim_id);

        if constexpr (!Detached) {
            hit_mask = Mask(any_hit);
        } else {
            hit_mask = any_hit;
        }
    } else {
        optix_hit = optix_scene_->template intersect<Detached>(ray, hit_mask);
    }

    const IntDetached shape_id = optix_hit.shape_id;
    const IntDetached global_primitive_id = optix_hit.global_prim_id;
    const MaskDetached hit_mask_detached = detach<false>(hit_mask);
    const IntDetached mesh_face_offset = gather<IntDetached>(face_offsets_, shape_id, hit_mask_detached);
    const IntDetached local_primitive_id = global_primitive_id - mesh_face_offset;

    Vector2fT<Detached> triangle_uv_coords;
    FloatT<Detached> hit_distance;

    if constexpr (!Detached) {
        // AD path: re-gather vertex data and recompute intersection for gradients.
        const Int global_primitive_id_ad = Int(global_primitive_id);
        const Vector3f triangle_p0 = gather<Vector3f>(triangle_info_.p0, global_primitive_id_ad, hit_mask);
        const Vector3f triangle_e1 = gather<Vector3f>(triangle_info_.e1, global_primitive_id_ad, hit_mask);
        const Vector3f triangle_e2 = gather<Vector3f>(triangle_info_.e2, global_primitive_id_ad, hit_mask);
        std::tie(triangle_uv_coords, hit_distance) = ray_intersect_triangle<Detached>(triangle_p0, triangle_e1, triangle_e2, ray);

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal = gather<Vector3f>(triangle_info_.face_normal, global_primitive_id_ad, hit_mask);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 = gather<Vector3f>(triangle_info_.n0, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n1 = gather<Vector3f>(triangle_info_.n1, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n2 = gather<Vector3f>(triangle_info_.n2, global_primitive_id_ad, hit_mask);
                MaskT<Detached> use_face_normal_mask = gather<Mask>(triangle_face_normal_mask_, global_primitive_id_ad, hit_mask);
                const Vector2fT<Detached> safe_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal =
                    normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data = gather<TriangleUV>(triangle_uv_, global_primitive_id_ad, hit_mask);
            const Vector2fT<Detached> safe_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0], triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask, uv, intersection.uv);
        }
    } else {
        // Detached path: use OptiX results directly, gather only what is needed.
        triangle_uv_coords = optix_hit.barycentric;
        hit_distance = optix_hit.t;

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal = gather<Vector3fDetached>(triangle_info_detached_.face_normal, global_primitive_id, hit_mask_detached);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 = gather<Vector3fDetached>(triangle_info_detached_.n0, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n1 = gather<Vector3fDetached>(triangle_info_detached_.n1, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n2 = gather<Vector3fDetached>(triangle_info_detached_.n2, global_primitive_id, hit_mask_detached);
                MaskT<Detached> use_face_normal_mask = gather<MaskDetached>(triangle_face_normal_mask_detached_, global_primitive_id, hit_mask_detached);
                const Vector2fT<Detached> safe_uv = select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal =
                    normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask_detached, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask_detached, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data = gather<TriangleUVDetached>(triangle_uv_detached_, global_primitive_id, hit_mask_detached);
            const Vector2fT<Detached> safe_uv = select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0], triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask_detached, uv, intersection.uv);
        }
    }

    hit_mask &= drjit::isfinite(hit_distance) && (hit_distance < ray.tmax);

    const FloatT<Detached> safe_hit_distance = select(hit_mask, hit_distance, zeros<FloatT<Detached>>(ray_count));
    const Vector2fT<Detached> safe_triangle_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));

    const Vector3fT<Detached> barycentric_coordinates(1.f - safe_triangle_uv.x() - safe_triangle_uv.y(),
                                                      safe_triangle_uv.x(),
                                                      safe_triangle_uv.y());
    const Vector3fT<Detached> hit_position = ray(safe_hit_distance);

    intersection.t = select(hit_mask, safe_hit_distance, intersection.t);
    intersection.p = select(hit_mask, hit_position, intersection.p);
    intersection.barycentric = select(hit_mask, barycentric_coordinates, intersection.barycentric);
    intersection.shape_id = select(hit_mask, IntT<Detached>(shape_id), intersection.shape_id);
    const IntT<Detached> local_primitive_id_t = IntT<Detached>(local_primitive_id);
    const IntT<Detached> global_primitive_id_t = IntT<Detached>(global_primitive_id);
    intersection.prim_id = select(hit_mask, local_primitive_id_t, intersection.prim_id);
    intersection.local_prim_id =
        select(hit_mask, local_primitive_id_t, intersection.local_prim_id);
    intersection.global_prim_id =
        select(hit_mask, global_primitive_id_t, intersection.global_prim_id);
    return intersection;
}

template <bool Detached>
ReflectionChainT<Detached> Scene::trace_reflections(const RayT<Detached> &ray,
                                                    int max_bounces,
                                                    MaskT<Detached> active) const {
    return this->template trace_reflections<Detached>(
        ray, max_bounces, ReflectionTraceOptions(), active);
}

template <bool Detached>
ReflectionTraceT<Detached> Scene::trace_bounces(
    const RayT<Detached> &ray,
    int max_bounces,
    MaskT<Detached> active) const {
    return this->template trace_bounces<Detached>(
        ray, max_bounces, ReflectionTraceOptions(), active);
}

template <bool Detached>
ReflectionTraceT<Detached> Scene::trace_bounces(
    const RayT<Detached> &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflections(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflections(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_reflections(): max_bounces must be positive.");
    return trace_bounces_impl<Detached>(
        *this, ray, max_bounces, options, active);
}

template <bool Detached>
ReflectionChainT<Detached> Scene::trace_reflections(const RayT<Detached> &ray,
                                                    int max_bounces,
                                                    const ReflectionTraceOptions &options,
                                                    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflections(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflections(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0, "Scene::trace_reflections(): max_bounces must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionChainT<Detached> result =
        initialize_reflection_chain_result<Detached>(ray_count, max_bounces);
    if (ray_count == 0) {
        return result;
    }

    const bool symbolic_reflection_trace = recording_reflections();
    require(!symbolic_reflection_trace || !options.deduplicate,
            "Scene::trace_reflections(): symbolic recording does not support deduplicate=true yet.");
    if (symbolic_reflection_trace) {
        require(max_bounces == 1,
                "Scene::trace_reflections(): symbolic recording currently supports max_bounces=1 only.");
        const ReflectionTraceT<Detached> trace =
            this->template trace_bounces<Detached>(ray, 1, options, active);
        result.bounce_count = trace.bounce_count;
        result.discovery_count = trace.discovery_count;
        result.representative_ray_index = trace.representative_ray_index;
        if (!trace.bounces.empty()) {
            const ReflectionBounceT<Detached> &bounce = trace.bounces.front();
            result.t = bounce.t;
            result.hit_points = bounce.hit_points;
            result.geo_normals = bounce.geo_normals;
            result.image_sources = bounce.image_sources;
            result.plane_points = bounce.plane_points;
            result.plane_normals = bounce.plane_normals;
            result.shape_ids = bounce.shape_ids;
            result.prim_ids = bounce.prim_ids;
            result.local_prim_ids = bounce.local_prim_ids;
            result.global_prim_ids = bounce.global_prim_ids;

            const MaskT<Detached> trailing_active = trace.bounce_count > 0;
            const Vector3fT<Detached> reflected_direction =
                ray.d - 2.f * dot(ray.d, bounce.geo_normals) * bounce.geo_normals;
            const Vector3fT<Detached> trailing_origin =
                bounce.hit_points + Epsilon * reflected_direction;
            RayT<Detached> trailing_ray(
                trailing_origin,
                reflected_direction,
                full<FloatT<Detached>>(Infinity, ray_count));
            const IntersectionT<Detached> trailing =
                this->template intersect<Detached>(
                    trailing_ray, trailing_active, RayFlags::Geometric);
            const MaskT<Detached> trailing_hit =
                trailing_active && trailing.is_valid();
            result.trailing_t =
                select(trailing_hit,
                       trailing.t,
                       full<FloatT<Detached>>(Infinity, ray_count));
            result.trailing_prim =
                select(trailing_hit,
                       trailing.global_prim_id,
                       full<IntT<Detached>>(-1, ray_count));
            result.trailing_dir =
                select(trailing_active,
                       reflected_direction,
                       zeros<Vector3fT<Detached>>(ray_count));
            result.trailing_origin =
                select(trailing_active,
                       trailing_origin,
                       zeros<Vector3fT<Detached>>(ray_count));
        }
        return result;
    }

    const MaskDetached active_detached = sanitize_reflection_active<Detached>(ray, active);
    if (drjit::none(active_detached)) {
        return result;
    }

    const OptixScene *primary_scene = nullptr;
    const OptixScene *secondary_scene = nullptr;
    int split_mode = 0;
    int hitgroup_record_count = mesh_count_;
    if (optix_split_active_) {
        primary_scene = optix_static_scene_.get();
        secondary_scene = optix_dynamic_scene_.get();
        split_mode = 1;
        hitgroup_record_count = static_cast<int>(
            std::max(optix_static_mesh_indices_.size(), optix_dynamic_mesh_indices_.size()));
    } else {
        primary_scene = optix_scene_.get();
    }

    require(primary_scene != nullptr && primary_scene->is_ready(),
            "Scene::trace_reflections(): OptiX scene is not ready.");
    require(hitgroup_record_count > 0,
            "Scene::trace_reflections(): invalid hitgroup record count.");

    if (!reflection_pipeline_) {
        reflection_pipeline_ = std::make_unique<ReflectionTracePipeline>();
        reflection_pipeline_->build(primary_scene->context(), hitgroup_record_count);
    }

    RayDetached broadphase_ray;
    if constexpr (!Detached) {
        broadphase_ray = RayDetached(detach<false>(ray.o),
                                     detach<false>(ray.d),
                                     detach<false>(ray.tmax));
    } else {
        broadphase_ray = ray;
    }

    drjit::eval(broadphase_ray.o,
                broadphase_ray.d,
                broadphase_ray.tmax,
                active_detached,
                triangle_info_detached_.p0,
                triangle_info_detached_.e1,
                triangle_info_detached_.e2,
                triangle_info_detached_.face_normal,
                face_offsets_);
    if (options.deduplicate && slices(options.canonical_prim_table) > 0) {
        drjit::eval(options.canonical_prim_table);
    }

    ReflectionTraceRaw raw = allocate_reflection_trace_raw(ray_count, max_bounces);
    initialize_reflection_trace_raw(raw);

    ReflectionTraceParams params = {};
    params.primary_handle = primary_scene->ias_handle();
    params.secondary_handle =
        secondary_scene != nullptr && secondary_scene->is_ready() ? secondary_scene->ias_handle() : 0ull;
    params.split_mode = split_mode;
    params.tri_p0_x = triangle_info_detached_.p0.x().data();
    params.tri_p0_y = triangle_info_detached_.p0.y().data();
    params.tri_p0_z = triangle_info_detached_.p0.z().data();
    params.tri_e1_x = triangle_info_detached_.e1.x().data();
    params.tri_e1_y = triangle_info_detached_.e1.y().data();
    params.tri_e1_z = triangle_info_detached_.e1.z().data();
    params.tri_e2_x = triangle_info_detached_.e2.x().data();
    params.tri_e2_y = triangle_info_detached_.e2.y().data();
    params.tri_e2_z = triangle_info_detached_.e2.z().data();
    params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
    params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
    params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
    params.face_offsets = face_offsets_.data();
    params.n_meshes = mesh_count_;
    params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
    params.ray_ox = broadphase_ray.o.x().data();
    params.ray_oy = broadphase_ray.o.y().data();
    params.ray_oz = broadphase_ray.o.z().data();
    params.ray_dx = broadphase_ray.d.x().data();
    params.ray_dy = broadphase_ray.d.y().data();
    params.ray_dz = broadphase_ray.d.z().data();
    params.ray_tmax = broadphase_ray.tmax.data();
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = ray_count;
    params.max_bounces = max_bounces;
    params.out_bounce_count = raw.bounce_count.data();
    params.out_shape_ids = raw.shape_ids.data();
    params.out_prim_ids = raw.prim_ids.data();
    params.out_t = raw.t.data();
    params.out_bary_u = raw.bary_u.data();
    params.out_bary_v = raw.bary_v.data();
    params.out_hit_x = raw.hit_x.data();
    params.out_hit_y = raw.hit_y.data();
    params.out_hit_z = raw.hit_z.data();
    params.out_norm_x = raw.norm_x.data();
    params.out_norm_y = raw.norm_y.data();
    params.out_norm_z = raw.norm_z.data();
    params.out_img_x = raw.img_x.data();
    params.out_img_y = raw.img_y.data();
    params.out_img_z = raw.img_z.data();
    params.out_trailing_t = raw.trailing_t.data();
    params.out_trailing_prim = raw.trailing_prim.data();
    params.out_trailing_dir_x = raw.trailing_dir_x.data();
    params.out_trailing_dir_y = raw.trailing_dir_y.data();
    params.out_trailing_dir_z = raw.trailing_dir_z.data();
    params.out_trailing_origin_x = raw.trailing_origin_x.data();
    params.out_trailing_origin_y = raw.trailing_origin_y.data();
    params.out_trailing_origin_z = raw.trailing_origin_z.data();

    reflection_pipeline_->launch(params);

    int trace_ray_count = ray_count;
    IntDetached trace_bounce_count = raw.bounce_count;
    IntDetached trace_discovery_count =
        select(raw.bounce_count > 0,
               full<IntDetached>(1, ray_count),
               full<IntDetached>(0, ray_count));
    IntDetached trace_representative_ray_index = arange<IntDetached>(ray_count);
    IntDetached trace_shape_ids = raw.shape_ids;
    IntDetached trace_prim_ids = raw.prim_ids;
    FloatDetached trace_t = raw.t;
    FloatDetached trace_hit_x = raw.hit_x;
    FloatDetached trace_hit_y = raw.hit_y;
    FloatDetached trace_hit_z = raw.hit_z;
    FloatDetached trace_norm_x = raw.norm_x;
    FloatDetached trace_norm_y = raw.norm_y;
    FloatDetached trace_norm_z = raw.norm_z;
    FloatDetached trace_img_x = raw.img_x;
    FloatDetached trace_img_y = raw.img_y;
    FloatDetached trace_img_z = raw.img_z;
    FloatDetached trace_trailing_t = raw.trailing_t;
    IntDetached trace_trailing_prim = raw.trailing_prim;
    FloatDetached trace_trailing_dir_x = raw.trailing_dir_x;
    FloatDetached trace_trailing_dir_y = raw.trailing_dir_y;
    FloatDetached trace_trailing_dir_z = raw.trailing_dir_z;
    FloatDetached trace_trailing_origin_x = raw.trailing_origin_x;
    FloatDetached trace_trailing_origin_y = raw.trailing_origin_y;
    FloatDetached trace_trailing_origin_z = raw.trailing_origin_z;

    if (options.deduplicate) {
        ReflectionTraceRaw compacted = allocate_reflection_trace_raw(ray_count, max_bounces);
        initialize_reflection_trace_raw(compacted);

        const IntDetached canonical_table = options.canonical_prim_table;
        const int canonical_table_size = static_cast<int>(slices(canonical_table));
        const int n_unique = reflection_dedup_gpu(
            ray_count,
            max_bounces,
            raw.bounce_count.data(),
            raw.shape_ids.data(),
            raw.prim_ids.data(),
            raw.t.data(),
            raw.bary_u.data(),
            raw.bary_v.data(),
            raw.hit_x.data(),
            raw.hit_y.data(),
            raw.hit_z.data(),
            raw.norm_x.data(),
            raw.norm_y.data(),
            raw.norm_z.data(),
            raw.img_x.data(),
            raw.img_y.data(),
            raw.img_z.data(),
            face_offsets_.data(),
            mesh_count_,
            canonical_table_size > 0 ? canonical_table.data() : nullptr,
            canonical_table_size,
            options.image_source_tolerance,
            compacted.bounce_count.data(),
            compacted.shape_ids.data(),
            compacted.prim_ids.data(),
            compacted.t.data(),
            compacted.bary_u.data(),
            compacted.bary_v.data(),
            compacted.hit_x.data(),
            compacted.hit_y.data(),
            compacted.hit_z.data(),
            compacted.norm_x.data(),
            compacted.norm_y.data(),
            compacted.norm_z.data(),
            compacted.img_x.data(),
            compacted.img_y.data(),
            compacted.img_z.data(),
            compacted.discovery_count.data(),
            compacted.representative_ray_index.data());

        trace_ray_count = n_unique;
        const int unique_slot_count = trace_ray_count * max_bounces;
        trace_bounce_count = prefix_array(compacted.bounce_count, trace_ray_count);
        trace_discovery_count = prefix_array(compacted.discovery_count, trace_ray_count);
        trace_representative_ray_index =
            prefix_array(compacted.representative_ray_index, trace_ray_count);
        trace_shape_ids = prefix_array(compacted.shape_ids, unique_slot_count);
        trace_prim_ids = prefix_array(compacted.prim_ids, unique_slot_count);
        trace_t = prefix_array(compacted.t, unique_slot_count);
        trace_hit_x = prefix_array(compacted.hit_x, unique_slot_count);
        trace_hit_y = prefix_array(compacted.hit_y, unique_slot_count);
        trace_hit_z = prefix_array(compacted.hit_z, unique_slot_count);
        trace_norm_x = prefix_array(compacted.norm_x, unique_slot_count);
        trace_norm_y = prefix_array(compacted.norm_y, unique_slot_count);
        trace_norm_z = prefix_array(compacted.norm_z, unique_slot_count);
        trace_img_x = prefix_array(compacted.img_x, unique_slot_count);
        trace_img_y = prefix_array(compacted.img_y, unique_slot_count);
        trace_img_z = prefix_array(compacted.img_z, unique_slot_count);
        const MaskDetached unique_mask = full<MaskDetached>(true, trace_ray_count);
        trace_trailing_t =
            gather<FloatDetached>(raw.trailing_t, trace_representative_ray_index, unique_mask);
        trace_trailing_prim =
            gather<IntDetached>(raw.trailing_prim, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_x =
            gather<FloatDetached>(raw.trailing_dir_x, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_y =
            gather<FloatDetached>(raw.trailing_dir_y, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_z =
            gather<FloatDetached>(raw.trailing_dir_z, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_x =
            gather<FloatDetached>(raw.trailing_origin_x, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_y =
            gather<FloatDetached>(raw.trailing_origin_y, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_z =
            gather<FloatDetached>(raw.trailing_origin_z, trace_representative_ray_index, unique_mask);
        result.ray_count = trace_ray_count;
    }

    const IntDetached trace_global_prim_ids =
        globalize_primitive_ids(trace_prim_ids, trace_shape_ids, face_offsets_);

    if constexpr (Detached) {
        const Vector3fDetached hit_points(trace_hit_x, trace_hit_y, trace_hit_z);
        const Vector3fDetached plane_normals(trace_norm_x, trace_norm_y, trace_norm_z);
        result.bounce_count = trace_bounce_count;
        result.discovery_count = trace_discovery_count;
        result.representative_ray_index = trace_representative_ray_index;
        result.t = trace_t;
        result.hit_points = hit_points;
        result.geo_normals = plane_normals;
        result.image_sources = Vector3fDetached(trace_img_x, trace_img_y, trace_img_z);
        result.plane_points = hit_points;
        result.plane_normals = plane_normals;
        result.shape_ids = trace_shape_ids;
        result.prim_ids = trace_prim_ids;
        result.local_prim_ids = trace_prim_ids;
        result.global_prim_ids = trace_global_prim_ids;
        result.trailing_t = trace_trailing_t;
        result.trailing_prim = trace_trailing_prim;
        result.trailing_dir = Vector3fDetached(trace_trailing_dir_x,
                                               trace_trailing_dir_y,
                                               trace_trailing_dir_z);
        result.trailing_origin = Vector3fDetached(trace_trailing_origin_x,
                                                  trace_trailing_origin_y,
                                                  trace_trailing_origin_z);
        return result;
    } else {
        result = initialize_reflection_chain_result<false>(trace_ray_count, max_bounces);
        result.bounce_count = Int(trace_bounce_count);
        result.discovery_count = Int(trace_discovery_count);
        result.representative_ray_index = Int(trace_representative_ray_index);
        result.shape_ids = Int(trace_shape_ids);
        result.prim_ids = Int(trace_prim_ids);
        result.local_prim_ids = Int(trace_prim_ids);
        result.global_prim_ids = Int(trace_global_prim_ids);
        result.trailing_t = Float(trace_trailing_t);
        result.trailing_prim = Int(trace_trailing_prim);
        result.trailing_dir = Vector3f(Float(trace_trailing_dir_x),
                                       Float(trace_trailing_dir_y),
                                       Float(trace_trailing_dir_z));
        result.trailing_origin = Vector3f(Float(trace_trailing_origin_x),
                                          Float(trace_trailing_origin_y),
                                          Float(trace_trailing_origin_z));

        if (trace_ray_count == 0) {
            return result;
        }

        const Mask representative_mask = full<Mask>(true, trace_ray_count);
        const MaskDetached representative_mask_detached =
            full<MaskDetached>(true, trace_ray_count);
        const Int representative_ray_index = Int(trace_representative_ray_index);
        Ray current_ray(
            gather<Vector3f>(ray.o, representative_ray_index, representative_mask),
            gather<Vector3f>(ray.d, representative_ray_index, representative_mask),
            gather<Float>(ray.tmax, representative_ray_index, representative_mask));
        MaskDetached current_active_detached =
            gather<MaskDetached>(active_detached,
                                 trace_representative_ray_index,
                                 representative_mask_detached);
        Vector3f current_image_source = current_ray.o;
        const IntDetached bounce_slots =
            arange<IntDetached>(trace_ray_count) * IntDetached(max_bounces);

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const IntDetached slot_detached = bounce_slots + bounce;
            const Int slot = Int(slot_detached);
            const IntDetached shape_id_detached =
                gather<IntDetached>(trace_shape_ids, slot_detached, current_active_detached);
            const IntDetached prim_id_detached =
                gather<IntDetached>(trace_prim_ids, slot_detached, current_active_detached);
            const MaskDetached broadphase_hit =
                current_active_detached && (shape_id_detached >= 0) && (prim_id_detached >= 0);
            if (drjit::none(broadphase_hit)) {
                break;
            }

            const IntDetached mesh_face_offset =
                gather<IntDetached>(face_offsets_, shape_id_detached, broadphase_hit);
            const IntDetached global_prim_detached = mesh_face_offset + prim_id_detached;
            const Int global_prim = Int(global_prim_detached);
            const Mask hit_mask = Mask(broadphase_hit);

            const Vector3f triangle_p0 = gather<Vector3f>(triangle_info_.p0, global_prim, hit_mask);
            const Vector3f triangle_e1 = gather<Vector3f>(triangle_info_.e1, global_prim, hit_mask);
            const Vector3f triangle_e2 = gather<Vector3f>(triangle_info_.e2, global_prim, hit_mask);

            Vector2f triangle_barycentric;
            Float hit_distance;
            std::tie(triangle_barycentric, hit_distance) =
                ray_intersect_triangle<false>(triangle_p0, triangle_e1, triangle_e2, current_ray);

            Mask bounce_hit =
                hit_mask && drjit::isfinite(hit_distance) && (hit_distance < current_ray.tmax);
            const Float safe_t =
                select(bounce_hit, hit_distance, full<Float>(Infinity, trace_ray_count));
            Vector3f geo_normal = gather<Vector3f>(triangle_info_.face_normal, global_prim, hit_mask);
            geo_normal = normalize(select(hit_mask, geo_normal, Vector3f(0.f, 0.f, 1.f)));
            geo_normal = select(dot(current_ray.d, geo_normal) > 0.f, -geo_normal, geo_normal);
            const Vector3f hit_point =
                current_ray(select(bounce_hit, safe_t, zeros<Float>(trace_ray_count)));
            const Float plane_distance = dot(current_image_source - hit_point, geo_normal);
            const Vector3f reflected_image_source =
                current_image_source - 2.f * plane_distance * geo_normal;

            scatter(result.t, safe_t, slot, bounce_hit);
            scatter(result.hit_points, hit_point, slot, bounce_hit);
            scatter(result.geo_normals, geo_normal, slot, bounce_hit);
            scatter(result.image_sources, reflected_image_source, slot, bounce_hit);
            scatter(result.plane_points, hit_point, slot, bounce_hit);
            scatter(result.plane_normals, geo_normal, slot, bounce_hit);

            const Float ray_dot_normal = dot(current_ray.d, geo_normal);
            const Vector3f reflected_direction =
                current_ray.d - 2.f * ray_dot_normal * geo_normal;
            current_ray.o = select(bounce_hit,
                                   hit_point + Epsilon * reflected_direction,
                                   current_ray.o);
            current_ray.d = select(bounce_hit, reflected_direction, current_ray.d);
            current_ray.tmax = select(bounce_hit,
                                      full<Float>(Infinity, trace_ray_count),
                                      current_ray.tmax);
            current_image_source =
                select(bounce_hit, reflected_image_source, current_image_source);
            current_active_detached = detach<false>(bounce_hit);
        }

        const Mask trailing_active = result.bounce_count > 0;
        const Intersection trailing =
            this->template intersect<false>(
                current_ray, trailing_active, RayFlags::Geometric);
        const Mask trailing_hit = trailing_active && trailing.is_valid();
        result.trailing_t =
            select(trailing_hit,
                   trailing.t,
                   full<Float>(Infinity, trace_ray_count));
        result.trailing_prim =
            select(trailing_hit,
                   trailing.global_prim_id,
                   full<Int>(-1, trace_ray_count));
        result.trailing_dir =
            select(trailing_active,
                   current_ray.d,
                   zeros<Vector3f>(trace_ray_count));
        result.trailing_origin =
            select(trailing_active,
                   current_ray.o,
                   zeros<Vector3f>(trace_ray_count));

        return result;
    }
}

template <bool Detached>
ReflectionAccumulationResultT<Detached> Scene::trace_reflections_accumulating(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &tx_position,
    const ReflectionAccumulationGrid &grid,
    const PrimitiveMaterialPayloadT<Detached> &material,
    int max_bounces,
    const ReflectionAccumulationOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::TraceReflectionsAccumulating);
    require(is_ready(), "Scene::trace_reflections_accumulating(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflections_accumulating(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_reflections_accumulating(): max_bounces must be positive.");
    if constexpr (!Detached) {
        throw std::runtime_error(
            "Scene::trace_reflections_accumulating(): native accumulation is a non-AD native fast path. "
            "Use detached inputs, or use the existing AD tape path explicitly.");
    }
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::trace_reflections_accumulating(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::trace_reflections_accumulating(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::trace_reflections_accumulating(): grid bounds must be ordered.");
    require(options.wavelength > 0.f,
            "Scene::trace_reflections_accumulating(): wavelength must be positive.");
    require(options.cell_area > 0.f,
            "Scene::trace_reflections_accumulating(): cell_area must be positive.");
    require(options.solid_angle_per_ray >= 0.f,
            "Scene::trace_reflections_accumulating(): solid_angle_per_ray must be non-negative.");
    require(options.wedge_capacity >= 0,
            "Scene::trace_reflections_accumulating(): wedge_capacity must be non-negative.");

    ReflectionAccumulationResultT<Detached> result;
    const int ray_count = static_cast<int>(slices(ray.o));
    const int grid_cell_count = grid.resolution0 * grid.resolution1;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    result.grid_cell_count = grid_cell_count;

    if constexpr (!Detached) {
        throw std::runtime_error(
            "Scene::trace_reflections_accumulating(): native accumulation is a non-AD native fast path. "
            "Use detached inputs, or use the existing AD tape path explicitly.");
    } else {
        result.reflection_power = zeros<FloatDetached>(grid_cell_count);
        result.reflection_count = full<IntDetached>(0, 1);
        result.wedge_events.capacity = options.wedge_capacity;
        result.wedge_events.count = full<IntDetached>(0, 1);
        const int event_count = std::max(1, options.wedge_capacity);
        result.wedge_events.ray_index = full<IntDetached>(-1, event_count);
        result.wedge_events.hit_points = zeros<Vector3fDetached>(event_count);
        result.wedge_events.normals = zeros<Vector3fDetached>(event_count);
        result.wedge_events.prim_id = full<IntDetached>(-1, event_count);
        result.wedge_events.directions = zeros<Vector3fDetached>(event_count);
        result.wedge_events.bounce_depth = full<IntDetached>(-1, event_count);
        if (ray_count == 0) {
            return result;
        }

        require(static_cast<int>(slices(ray.d)) == ray_count &&
                    static_cast<int>(slices(ray.tmax)) == ray_count,
                "Scene::trace_reflections_accumulating(): ray fields must have matching widths.");
        const int tx_count = static_cast<int>(slices(tx_position));
        require(tx_count == 1 || tx_count == ray_count,
                "Scene::trace_reflections_accumulating(): tx_position width must be 1 or match ray count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::trace_reflections_accumulating(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::trace_reflections_accumulating(): material payload fields must have matching widths.");

        const int triangle_count = static_cast<int>(slices(triangle_info_detached_.p0));
        require(material_count >= triangle_count,
                "Scene::trace_reflections_accumulating(): material payload must provide one entry per global primitive.");

        Vector3fDetached tx_detached = tx_position;
        if (tx_count == 1 && ray_count > 1) {
            const IntDetached zero_index = full<IntDetached>(0, ray_count);
            tx_detached = Vector3fDetached(
                gather<FloatDetached>(tx_position.x(), zero_index),
                gather<FloatDetached>(tx_position.y(), zero_index),
                gather<FloatDetached>(tx_position.z(), zero_index));
        }

        MaskDetached active_detached = sanitize_reflection_active<true>(ray, active);
        active_detached &= drjit::isfinite(tx_detached.x()) &&
                           drjit::isfinite(tx_detached.y()) &&
                           drjit::isfinite(tx_detached.z());
        if (drjit::none(active_detached)) {
            return result;
        }

        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        int hitgroup_record_count = mesh_count_;
        if (optix_split_active_) {
            primary_scene = optix_static_scene_.get();
            secondary_scene = optix_dynamic_scene_.get();
            split_mode = 1;
            hitgroup_record_count = static_cast<int>(
                std::max(optix_static_mesh_indices_.size(), optix_dynamic_mesh_indices_.size()));
        } else {
            primary_scene = optix_scene_.get();
        }

        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::trace_reflections_accumulating(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::trace_reflections_accumulating(): invalid hitgroup record count.");

        if (!reflection_accumulation_pipeline_) {
            reflection_accumulation_pipeline_ =
                std::make_unique<ReflectionAccumulationPipeline>();
            reflection_accumulation_pipeline_->build(primary_scene->context(),
                                                     hitgroup_record_count);
        }

        drjit::eval(ray.o,
                    ray.d,
                    ray.tmax,
                    tx_detached,
                    active_detached,
                    triangle_info_detached_.p0,
                    triangle_info_detached_.e1,
                    triangle_info_detached_.e2,
                    triangle_info_detached_.face_normal,
                    face_offsets_,
                    material.eta_r,
                    material.sigma,
                    material.gain,
                    material.mu_r,
                    material.valid);

        ReflectionAccumulationRaw raw = allocate_reflection_accumulation_raw(
            ray_count, max_bounces, grid_cell_count, options.wedge_capacity);
        initialize_reflection_accumulation_raw(raw);

        ReflectionAccumulationParams params = {};
        params.primary_handle = primary_scene->ias_handle();
        params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready() ? secondary_scene->ias_handle() : 0ull;
        params.split_mode = split_mode;
        params.tri_p0_x = triangle_info_detached_.p0.x().data();
        params.tri_p0_y = triangle_info_detached_.p0.y().data();
        params.tri_p0_z = triangle_info_detached_.p0.z().data();
        params.tri_e1_x = triangle_info_detached_.e1.x().data();
        params.tri_e1_y = triangle_info_detached_.e1.y().data();
        params.tri_e1_z = triangle_info_detached_.e1.z().data();
        params.tri_e2_x = triangle_info_detached_.e2.x().data();
        params.tri_e2_y = triangle_info_detached_.e2.y().data();
        params.tri_e2_z = triangle_info_detached_.e2.z().data();
        params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        params.face_offsets = face_offsets_.data();
        params.n_meshes = mesh_count_;
        params.n_triangles = triangle_count;
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.ray_dx = ray.d.x().data();
        params.ray_dy = ray.d.y().data();
        params.ray_dz = ray.d.z().data();
        params.ray_tmax = ray.tmax.data();
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.n_rays = ray_count;
        params.tx_x = tx_detached.x().data();
        params.tx_y = tx_detached.y().data();
        params.tx_z = tx_detached.z().data();
        params.max_bounces = max_bounces;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.solid_angle_per_ray = options.solid_angle_per_ray;
        params.cell_area = options.cell_area;
        params.seed = options.seed;
        params.rr_depth = options.rr_depth;
        params.rr_prob = options.rr_prob;
        params.stop_threshold = options.stop_threshold;
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_gain = material.gain.data();
        params.material_mu_r = material.mu_r.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.collect_wedges = options.collect_wedges ? 1 : 0;
        params.collect_wedge_prefixes = options.collect_wedge_prefixes ? 1 : 0;
        params.wedge_capacity = options.wedge_capacity;
        params.out_reflection_power = raw.reflection_power.data();
        params.out_reflection_count = raw.reflection_count.data();
        params.out_wedge_count = raw.wedge_count.data();
        params.out_wedge_ray_index = raw.wedge_ray_index.data();
        params.out_wedge_hit_x = raw.wedge_hit_x.data();
        params.out_wedge_hit_y = raw.wedge_hit_y.data();
        params.out_wedge_hit_z = raw.wedge_hit_z.data();
        params.out_wedge_normal_x = raw.wedge_normal_x.data();
        params.out_wedge_normal_y = raw.wedge_normal_y.data();
        params.out_wedge_normal_z = raw.wedge_normal_z.data();
        params.out_wedge_prim_id = raw.wedge_prim_id.data();
        params.out_wedge_dir_x = raw.wedge_dir_x.data();
        params.out_wedge_dir_y = raw.wedge_dir_y.data();
        params.out_wedge_dir_z = raw.wedge_dir_z.data();
        params.out_wedge_bounce_depth = raw.wedge_bounce_depth.data();

        reflection_accumulation_pipeline_->launch(params);

        result.reflection_power = raw.reflection_power;
        result.reflection_count = raw.reflection_count;
        result.wedge_events.capacity = options.wedge_capacity;
        result.wedge_events.count = raw.wedge_count;
        result.wedge_events.ray_index = raw.wedge_ray_index;
        result.wedge_events.hit_points =
            Vector3fDetached(raw.wedge_hit_x, raw.wedge_hit_y, raw.wedge_hit_z);
        result.wedge_events.normals =
            Vector3fDetached(raw.wedge_normal_x, raw.wedge_normal_y, raw.wedge_normal_z);
        result.wedge_events.prim_id = raw.wedge_prim_id;
        result.wedge_events.directions =
            Vector3fDetached(raw.wedge_dir_x, raw.wedge_dir_y, raw.wedge_dir_z);
        result.wedge_events.bounce_depth = raw.wedge_bounce_depth;
        return result;
    }
}

template <bool Detached>
MaskT<Detached> Scene::shadow_test(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::shadow_test(): scene is not built.");
    require(!pending_updates_, "Scene::shadow_test(): scene has pending updates. Call Scene::sync() first.");

    const bool symbolic_optix_query = optix_split_active_ && uses_symbolic_optix_query_path();
    if (!optix_split_active_ || symbolic_optix_query) {
        return optix_scene_->template shadow_test<Detached>(ray, active);
    }

    const MaskT<Detached> static_hit =
        optix_static_scene_->template shadow_test<Detached>(ray, active);
    const MaskT<Detached> dynamic_hit =
        optix_dynamic_scene_->template shadow_test<Detached>(ray, active);
    return static_hit || dynamic_hit;
}

template <bool Detached>
SegmentVisibilityT<Detached> Scene::trace_segment_visibility(
    const Vector3fT<Detached> &start,
    const Vector3fT<Detached> &end,
    const IntDetached &ignore_prim_ids,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::trace_segment_visibility(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_segment_visibility(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end)) == ray_count,
            "Scene::trace_segment_visibility(): start and end must have the same width.");

    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;
    result.visible = full<MaskT<Detached>>(false, ray_count);
    if (ray_count == 0) {
        return result;
    }

    const int ignore_count = static_cast<int>(slices(ignore_prim_ids));
    int ignore_k = 0;
    if (ignore_count > 0) {
        require(ignore_count % ray_count == 0,
                "Scene::trace_segment_visibility(): ignore_prim_ids width must be a multiple of ray count.");
        ignore_k = ignore_count / ray_count;
    }

    const MaskDetached active_detached = sanitize_segment_active<Detached>(start, end, active);
    Vector3fDetached start_detached;
    Vector3fDetached end_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_detached = detach<false>(end);
    } else {
        start_detached = start;
        end_detached = end;
    }

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_visibility_jit_no_ignore<Detached>(
            *optix_scene_, start_detached, end_detached, active_detached);
    }

    ensure_segment_visibility_pipeline(segment_visibility_pipeline_, *optix_scene_, mesh_count_);
    return trace_segment_visibility_native<Detached>(
        *optix_scene_,
        *segment_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        start_detached,
        end_detached,
        ignore_prim_ids,
        ignore_k,
        active_detached);
}

template <bool Detached>
SegmentPairVisibilityT<Detached> Scene::trace_segment_pair_visibility(
    const Vector3fT<Detached> &start,
    const Vector3fT<Detached> &end_a,
    const Vector3fT<Detached> &end_b,
    const IntDetached &ignore_prim_ids,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::trace_segment_pair_visibility(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_segment_pair_visibility(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end_a)) == ray_count &&
                static_cast<int>(slices(end_b)) == ray_count,
            "Scene::trace_segment_pair_visibility(): start, end_a, and end_b must have the same width.");

    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;
    result.visible_a = full<MaskT<Detached>>(false, ray_count);
    result.visible_b = full<MaskT<Detached>>(false, ray_count);
    if (ray_count == 0) {
        return result;
    }

    const int ignore_count = static_cast<int>(slices(ignore_prim_ids));
    int ignore_k = 0;
    if (ignore_count > 0) {
        require(ignore_count % ray_count == 0,
                "Scene::trace_segment_pair_visibility(): ignore_prim_ids width must be a multiple of ray count.");
        ignore_k = ignore_count / ray_count;
    }

    const MaskDetached active_detached =
        sanitize_segment_active<Detached>(start, end_a, active) &&
        sanitize_segment_active<Detached>(start, end_b, active);
    Vector3fDetached start_detached;
    Vector3fDetached end_a_detached;
    Vector3fDetached end_b_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_a_detached = detach<false>(end_a);
        end_b_detached = detach<false>(end_b);
    } else {
        start_detached = start;
        end_a_detached = end_a;
        end_b_detached = end_b;
    }

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_pair_visibility_jit_no_ignore<Detached>(
            *optix_scene_,
            start_detached,
            end_a_detached,
            end_b_detached,
            active_detached);
    }

    ensure_segment_visibility_pipeline(segment_visibility_pipeline_, *optix_scene_, mesh_count_);
    return trace_segment_pair_visibility_native<Detached>(
        *optix_scene_,
        *segment_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        start_detached,
        end_a_detached,
        end_b_detached,
        ignore_prim_ids,
        ignore_k,
        active_detached);
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> Scene::trace_axial_edge_visibility(
    const Vector3fT<Detached> &source_pos,
    const Vector3fT<Detached> &edge_pos,
    const Vector3fT<Detached> &edge_dir,
    const FloatT<Detached> &edge_line_min,
    const FloatT<Detached> &edge_line_max,
    const std::vector<float> &sample_fractions,
    MaskT<Detached> active) const {
    require(!sample_fractions.empty(),
            "Scene::trace_axial_edge_visibility(): sample_fractions must not be empty.");
    require(sample_fractions.size() <= SegmentVisibilityMaxSamples,
            "Scene::trace_axial_edge_visibility(): at most 16 sample fractions are supported.");
    require(is_ready(), "Scene::trace_axial_edge_visibility(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_axial_edge_visibility(): scene has pending updates. Call Scene::sync() first.");

    const int state_count = static_cast<int>(slices(source_pos));
    require(static_cast<int>(slices(edge_pos)) == state_count &&
                static_cast<int>(slices(edge_dir)) == state_count &&
                static_cast<int>(slices(edge_line_min)) == state_count &&
                static_cast<int>(slices(edge_line_max)) == state_count,
            "Scene::trace_axial_edge_visibility(): all inputs must have the same width.");

    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;
    result.any_visible = full<MaskT<Detached>>(false, state_count);
    if (state_count == 0) {
        return result;
    }

    MaskDetached active_detached;
    Vector3fDetached source_detached;
    Vector3fDetached edge_pos_detached;
    Vector3fDetached edge_dir_detached;
    FloatDetached edge_line_min_detached;
    FloatDetached edge_line_max_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        source_detached = detach<false>(source_pos);
        edge_pos_detached = detach<false>(edge_pos);
        edge_dir_detached = detach<false>(edge_dir);
        edge_line_min_detached = detach<false>(edge_line_min);
        edge_line_max_detached = detach<false>(edge_line_max);
    } else {
        active_detached = active;
        source_detached = source_pos;
        edge_pos_detached = edge_pos;
        edge_dir_detached = edge_dir;
        edge_line_min_detached = edge_line_min;
        edge_line_max_detached = edge_line_max;
    }

    active_detached &= drjit::isfinite(source_detached.x()) &&
                       drjit::isfinite(source_detached.y()) &&
                       drjit::isfinite(source_detached.z()) &&
                       drjit::isfinite(edge_pos_detached.x()) &&
                       drjit::isfinite(edge_pos_detached.y()) &&
                       drjit::isfinite(edge_pos_detached.z()) &&
                       drjit::isfinite(edge_dir_detached.x()) &&
                       drjit::isfinite(edge_dir_detached.y()) &&
                       drjit::isfinite(edge_dir_detached.z()) &&
                       drjit::isfinite(edge_line_min_detached) &&
                       drjit::isfinite(edge_line_max_detached);

    if (active_trace_visibility_backend() != TraceVisibilityBackend::Native) {
        return trace_axial_edge_visibility_jit<Detached>(
            *optix_scene_,
            source_detached,
            edge_pos_detached,
            edge_dir_detached,
            edge_line_min_detached,
            edge_line_max_detached,
            sample_fractions,
            active_detached);
    }

    ensure_segment_visibility_pipeline(segment_visibility_pipeline_, *optix_scene_, mesh_count_);
    return trace_axial_edge_visibility_native<Detached>(
        *optix_scene_,
        *segment_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        source_detached,
        edge_pos_detached,
        edge_dir_detached,
        edge_line_min_detached,
        edge_line_max_detached,
        sample_fractions,
        active_detached);
}

template <bool Detached>
SegmentChainVisibilityT<Detached> Scene::trace_segment_chain_visibility(
    const Vector3fT<Detached> &points,
    const IntDetached &chain_length,
    const IntDetached &ignore_prim_per_segment,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::trace_segment_chain_visibility(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_segment_chain_visibility(): scene has pending updates. Call Scene::sync() first.");

    const int chain_count = static_cast<int>(slices(chain_length));
    const int point_count = static_cast<int>(slices(points));

    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = 0;
    result.all_visible = full<MaskT<Detached>>(false, chain_count);
    result.first_blocked_segment = full<IntT<Detached>>(-1, chain_count);
    result.first_blocked_prim = full<IntT<Detached>>(-1, chain_count);
    if (chain_count == 0) {
        return result;
    }

    require(point_count % chain_count == 0,
            "Scene::trace_segment_chain_visibility(): points width must be a multiple of chain count.");
    const int max_points = point_count / chain_count;
    require(max_points >= 2,
            "Scene::trace_segment_chain_visibility(): each chain must contain at least two points.");
    const int max_segments = max_points - 1;
    result.max_segments = max_segments;

    const int ignore_count = static_cast<int>(slices(ignore_prim_per_segment));
    int ignore_k = 0;
    if (ignore_count > 0) {
        const int ignore_slots = chain_count * max_segments;
        require(ignore_count % ignore_slots == 0,
                "Scene::trace_segment_chain_visibility(): ignore_prim_per_segment width must be a multiple of chain_count * max_segments.");
        ignore_k = ignore_count / ignore_slots;
    }

    MaskDetached active_detached;
    Vector3fDetached points_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        points_detached = detach<false>(points);
    } else {
        active_detached = active;
        points_detached = points;
    }
    active_detached &= chain_length >= 0;

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_chain_visibility_jit_no_ignore<Detached>(
            *optix_scene_,
            points_detached,
            chain_length,
            chain_count,
            max_points,
            max_segments,
            active_detached);
    }

    ensure_segment_visibility_pipeline(segment_visibility_pipeline_, *optix_scene_, mesh_count_);
    return trace_segment_chain_visibility_native<Detached>(
        *optix_scene_,
        *segment_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        points_detached,
        chain_length,
        ignore_prim_per_segment,
        ignore_k,
        chain_count,
        max_points,
        max_segments,
        active_detached);
}

template <bool Detached>
NearestPointEdgeT<Detached> Scene::nearest_edge(const Vector3fT<Detached> &point, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(point): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(point): scene has pending updates. Call Scene::sync() first.");

    const int query_count = static_cast<int>(slices(point));
    NearestPointEdgeT<Detached> result = initialize_nearest_point_edge_result<Detached>(query_count);
    if (edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    MaskDetached active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x()));
        active_detached &= drjit::isfinite(detach<false>(point.y()));
        active_detached &= drjit::isfinite(detach<false>(point.z()));
        active &= Mask(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) && drjit::isfinite(point.y()) && drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_point(edge_bvh_backend_);
    ClosestEdgeCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edge<Detached>(point, query_mask)
            : edge_bvh_->template nearest_edge<Detached>(point, query_mask);
    const MaskDetached valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const IntDetached global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_id
            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(global_edge_id_detached);
        const Vector3f p0 = gather<Vector3f>(edge_info_.start, global_edge_id, valid);
        const Vector3f e1 = gather<Vector3f>(edge_info_.edge, global_edge_id, valid);
        const Mask is_boundary = gather<Mask>(edge_info_.is_boundary, global_edge_id, valid);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<false>(point, p0, e1);

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.point = select(valid, point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, Int(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, Int(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3fDetached p0 =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3fDetached e1 =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const MaskDetached is_boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        FloatDetached edge_t;
        Vector3fDetached edge_point;
        FloatDetached distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<true>(point, p0, e1);

        result.distance = select(valid_detached, sqrt(distance_sq), result.distance);
        result.point = select(valid_detached, point, result.point);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_point = select(valid_detached, edge_point, result.edge_point);
        result.shape_id = select(valid_detached, shape_id_detached, result.shape_id);
        result.edge_id = select(valid_detached, edge_id_detached, result.edge_id);
        result.global_edge_id = select(valid_detached, global_edge_id_detached, result.global_edge_id);
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> Scene::nearest_edge(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(ray): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(ray): scene has pending updates. Call Scene::sync() first.");

    const int query_count = static_cast<int>(slices(ray.o));
    NearestRayEdgeT<Detached> result = initialize_nearest_ray_edge_result<Detached>(query_count);
    if (edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    FloatDetached t_max_input;
    MaskDetached active_detached;
    if constexpr (!Detached) {
        t_max_input = detach<false>(ray.tmax);
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) &&
                           drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) &&
                           drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &= squared_norm(Vector3fDetached(detach<false>(ray.d.x()),
                                                        detach<false>(ray.d.y()),
                                                        detach<false>(ray.d.z()))) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active &= Mask(active_detached);
    } else {
        t_max_input = ray.tmax;
        active_detached = active;
        active_detached &= drjit::isfinite(ray.o.x()) && drjit::isfinite(ray.o.y()) && drjit::isfinite(ray.o.z());
        active_detached &= drjit::isfinite(ray.d.x()) && drjit::isfinite(ray.d.y()) && drjit::isfinite(ray.d.z());
        active_detached &= squared_norm(ray.d) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_ray(edge_bvh_backend_);
    ClosestEdgeCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edge<Detached>(ray, query_mask)
            : edge_bvh_->template nearest_edge<Detached>(ray, query_mask);
    const MaskDetached valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const MaskDetached finite_tmax = drjit::isfinite(t_max_input);
    const IntDetached global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_id
            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(global_edge_id_detached);
        const Vector3f p0 = gather<Vector3f>(edge_info_.start, global_edge_id, valid);
        const Vector3f e1 = gather<Vector3f>(edge_info_.edge, global_edge_id, valid);
        const Mask is_boundary = gather<Mask>(edge_info_.is_boundary, global_edge_id, valid);

        const Mask finite_mask = valid && Mask(finite_tmax);
        const Mask infinite_mask = valid && !Mask(finite_tmax);
        const Float safe_tmax = select(finite_mask, Float(t_max_input), zeros<Float>(query_count));

        Float query_t = zeros<Float>(query_count);
        Vector3f query_point = zeros<Vector3f>(query_count);
        Float edge_t = zeros<Float>(query_count);
        Vector3f edge_point = zeros<Vector3f>(query_count);
        Float distance_sq = full<Float>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            Float segment_query_t;
            Vector3f segment_query_point;
            Float segment_edge_t;
            Vector3f segment_edge_point;
            Float segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<false>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            Float ray_query_t;
            Vector3f ray_query_point;
            Float ray_edge_t;
            Vector3f ray_edge_point;
            Float ray_distance_sq;
            std::tie(ray_query_t, ray_query_point, ray_edge_t, ray_edge_point, ray_distance_sq) =
                closest_ray_segment<false>(ray.o, ray.d, p0, e1);

            query_t = select(infinite_mask, ray_query_t, query_t);
            query_point = select(infinite_mask, ray_query_point, query_point);
            edge_t = select(infinite_mask, ray_edge_t, edge_t);
            edge_point = select(infinite_mask, ray_edge_point, edge_point);
            distance_sq = select(infinite_mask, ray_distance_sq, distance_sq);
        }

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.ray_t = select(valid, query_t, result.ray_t);
        result.point = select(valid, query_point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, Int(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, Int(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3fDetached p0 =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3fDetached e1 =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const MaskDetached is_boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        const MaskDetached finite_mask = valid_detached && finite_tmax;
        const MaskDetached infinite_mask = valid_detached && !finite_tmax;
        const FloatDetached safe_tmax = select(finite_mask, t_max_input, zeros<FloatDetached>(query_count));

        FloatDetached query_t = zeros<FloatDetached>(query_count);
        Vector3fDetached query_point = zeros<Vector3fDetached>(query_count);
        FloatDetached edge_t = zeros<FloatDetached>(query_count);
        Vector3fDetached edge_point = zeros<Vector3fDetached>(query_count);
        FloatDetached distance_sq = full<FloatDetached>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            FloatDetached segment_query_t;
            Vector3fDetached segment_query_point;
            FloatDetached segment_edge_t;
            Vector3fDetached segment_edge_point;
            FloatDetached segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<true>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            FloatDetached ray_query_t;
            Vector3fDetached ray_query_point;
            FloatDetached ray_edge_t;
            Vector3fDetached ray_edge_point;
            FloatDetached ray_distance_sq;
            std::tie(ray_query_t, ray_query_point, ray_edge_t, ray_edge_point, ray_distance_sq) =
                closest_ray_segment<true>(ray.o, ray.d, p0, e1);

            query_t = select(infinite_mask, ray_query_t, query_t);
            query_point = select(infinite_mask, ray_query_point, query_point);
            edge_t = select(infinite_mask, ray_edge_t, edge_t);
            edge_point = select(infinite_mask, ray_edge_point, edge_point);
            distance_sq = select(infinite_mask, ray_distance_sq, distance_sq);
        }

        result.distance = select(valid_detached, sqrt(distance_sq), result.distance);
        result.ray_t = select(valid_detached, query_t, result.ray_t);
        result.point = select(valid_detached, query_point, result.point);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_point = select(valid_detached, edge_point, result.edge_point);
        result.shape_id = select(valid_detached, shape_id_detached, result.shape_id);
        result.edge_id = select(valid_detached, edge_id_detached, result.edge_id);
        result.global_edge_id = select(valid_detached, global_edge_id_detached, result.global_edge_id);
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template <bool Detached>
NearestEdgesTopKT<Detached> Scene::nearest_edges_topk(const Vector3fT<Detached> &point,
                                                       int k,
                                                       MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edges_topk(point): scene is not built.");
    require(!pending_updates_,
            "Scene::nearest_edges_topk(point): scene has pending updates. Call Scene::sync() first.");
    require(k > 0, "Scene::nearest_edges_topk(point): k must be positive.");
    require(k <= 16, "Scene::nearest_edges_topk(point): k must be <= 16.");

    const int query_count = static_cast<int>(slices(point));
    const int output_count = query_count * k;
    NearestEdgesTopKT<Detached> result;
    result.query_count = query_count;
    result.k = k;
    result.is_valid = full<MaskT<Detached>>(false, output_count);
    result.distances = full<FloatT<Detached>>(Infinity, output_count);
    result.points = zeros<Vector3fT<Detached>>(output_count);
    result.edge_t = zeros<FloatT<Detached>>(output_count);
    result.edge_points = zeros<Vector3fT<Detached>>(output_count);
    result.shape_ids = full<IntT<Detached>>(-1, output_count);
    result.edge_ids = full<IntT<Detached>>(-1, output_count);
    result.global_edge_ids = full<IntT<Detached>>(-1, output_count);
    result.is_boundary = full<MaskT<Detached>>(false, output_count);
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    MaskDetached active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x())) &&
                           drjit::isfinite(detach<false>(point.y())) &&
                           drjit::isfinite(detach<false>(point.z()));
        active &= Mask(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) &&
                           drjit::isfinite(point.y()) &&
                           drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_topk(edge_bvh_backend_);
    const ClosestEdgeTopKCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edges_topk<Detached>(point, k, query_mask)
            : edge_bvh_->template nearest_edges_topk<Detached>(point, k, query_mask);
    const MaskDetached valid_detached = candidate.is_valid;
    if (drjit::none(valid_detached)) {
        return result;
    }

    const IntDetached output_index = arange<IntDetached>(output_count);
    const IntDetached output_query = output_index / k;
    const IntDetached global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_ids
            : edge_bvh_->map_to_global(candidate.global_edge_ids, valid_detached);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(global_edge_id_detached);
        const Int query_id = Int(output_query);
        const Vector3f output_point = gather<Vector3f>(point, query_id, valid);
        const Vector3f edge_start =
            gather<Vector3f>(edge_info_.start, global_edge_id, valid);
        const Vector3f edge_vector =
            gather<Vector3f>(edge_info_.edge, global_edge_id, valid);
        const Mask boundary =
            gather<Mask>(edge_info_.is_boundary, global_edge_id, valid);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<false>(output_point, edge_start, edge_vector);

        result.is_valid = valid;
        result.distances = select(valid, sqrt(distance_sq), result.distances);
        result.points = select(valid, output_point, result.points);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_points = select(valid, edge_point, result.edge_points);
        result.shape_ids = select(valid, Int(shape_id_detached), result.shape_ids);
        result.edge_ids = select(valid, Int(edge_id_detached), result.edge_ids);
        result.global_edge_ids = select(valid, global_edge_id, result.global_edge_ids);
        result.is_boundary = select(valid, boundary, result.is_boundary);
    } else {
        const Vector3fDetached output_point =
            gather<Vector3fDetached>(point, output_query, valid_detached);
        const Vector3fDetached edge_start =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3fDetached edge_vector =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const MaskDetached boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        FloatDetached edge_t;
        Vector3fDetached edge_point;
        FloatDetached distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<true>(output_point, edge_start, edge_vector);

        result.is_valid = valid_detached;
        result.distances = select(valid_detached, sqrt(distance_sq), result.distances);
        result.points = select(valid_detached, output_point, result.points);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_points = select(valid_detached, edge_point, result.edge_points);
        result.shape_ids = select(valid_detached, shape_id_detached, result.shape_ids);
        result.edge_ids = select(valid_detached, edge_id_detached, result.edge_ids);
        result.global_edge_ids = select(valid_detached, global_edge_id_detached, result.global_edge_ids);
        result.is_boundary = select(valid_detached, boundary, result.is_boundary);
    }
    return result;
}

template IntersectionDetached Scene::intersect<true>(const RayDetached &ray, MaskDetached active, RayFlags flags) const;
template Intersection Scene::intersect<false>(const Ray &ray, Mask active, RayFlags flags) const;
template ReflectionChainDetached Scene::trace_reflections<true>(const RayDetached &ray,
                                                                int max_bounces,
                                                                const ReflectionTraceOptions &options,
                                                                MaskDetached active) const;
template ReflectionChain Scene::trace_reflections<false>(const Ray &ray,
                                                         int max_bounces,
                                                         const ReflectionTraceOptions &options,
                                                         Mask active) const;
template ReflectionChainDetached Scene::trace_reflections<true>(const RayDetached &ray,
                                                                int max_bounces,
                                                                MaskDetached active) const;
template ReflectionChain Scene::trace_reflections<false>(const Ray &ray,
                                                         int max_bounces,
                                                         Mask active) const;
template ReflectionAccumulationResultDetached Scene::trace_reflections_accumulating<true>(
    const RayDetached &ray,
    const Vector3fDetached &tx_position,
    const ReflectionAccumulationGrid &grid,
    const PrimitiveMaterialPayloadDetached &material,
    int max_bounces,
    const ReflectionAccumulationOptions &options,
    MaskDetached active) const;
template ReflectionAccumulationResult Scene::trace_reflections_accumulating<false>(
    const Ray &ray,
    const Vector3f &tx_position,
    const ReflectionAccumulationGrid &grid,
    const PrimitiveMaterialPayload &material,
    int max_bounces,
    const ReflectionAccumulationOptions &options,
    Mask active) const;
template ReflectionTraceDetached Scene::trace_bounces<true>(
    const RayDetached &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskDetached active) const;
template ReflectionTrace Scene::trace_bounces<false>(
    const Ray &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    Mask active) const;
template ReflectionTraceDetached Scene::trace_bounces<true>(
    const RayDetached &ray,
    int max_bounces,
    MaskDetached active) const;
template ReflectionTrace Scene::trace_bounces<false>(
    const Ray &ray,
    int max_bounces,
    Mask active) const;
template MaskDetached Scene::shadow_test<true>(const RayDetached &ray, MaskDetached active) const;
template Mask Scene::shadow_test<false>(const Ray &ray, Mask active) const;
template SegmentVisibilityDetached Scene::trace_segment_visibility<true>(
    const Vector3fDetached &start,
    const Vector3fDetached &end,
    const IntDetached &ignore_prim_ids,
    MaskDetached active) const;
template SegmentVisibility Scene::trace_segment_visibility<false>(
    const Vector3f &start,
    const Vector3f &end,
    const IntDetached &ignore_prim_ids,
    Mask active) const;
template SegmentPairVisibilityDetached Scene::trace_segment_pair_visibility<true>(
    const Vector3fDetached &start,
    const Vector3fDetached &end_a,
    const Vector3fDetached &end_b,
    const IntDetached &ignore_prim_ids,
    MaskDetached active) const;
template SegmentPairVisibility Scene::trace_segment_pair_visibility<false>(
    const Vector3f &start,
    const Vector3f &end_a,
    const Vector3f &end_b,
    const IntDetached &ignore_prim_ids,
    Mask active) const;
template AxialEdgeVisibilityDetached Scene::trace_axial_edge_visibility<true>(
    const Vector3fDetached &source_pos,
    const Vector3fDetached &edge_pos,
    const Vector3fDetached &edge_dir,
    const FloatDetached &edge_line_min,
    const FloatDetached &edge_line_max,
    const std::vector<float> &sample_fractions,
    MaskDetached active) const;
template AxialEdgeVisibility Scene::trace_axial_edge_visibility<false>(
    const Vector3f &source_pos,
    const Vector3f &edge_pos,
    const Vector3f &edge_dir,
    const Float &edge_line_min,
    const Float &edge_line_max,
    const std::vector<float> &sample_fractions,
    Mask active) const;
template SegmentChainVisibilityDetached Scene::trace_segment_chain_visibility<true>(
    const Vector3fDetached &points,
    const IntDetached &chain_length,
    const IntDetached &ignore_prim_per_segment,
    MaskDetached active) const;
template SegmentChainVisibility Scene::trace_segment_chain_visibility<false>(
    const Vector3f &points,
    const IntDetached &chain_length,
    const IntDetached &ignore_prim_per_segment,
    Mask active) const;
template NearestPointEdgeDetached Scene::nearest_edge<true>(const Vector3fDetached &point, MaskDetached active) const;
template NearestPointEdge Scene::nearest_edge<false>(const Vector3f &point, Mask active) const;
template NearestRayEdgeDetached Scene::nearest_edge<true>(const RayDetached &ray, MaskDetached active) const;
template NearestRayEdge Scene::nearest_edge<false>(const Ray &ray, Mask active) const;
template NearestEdgesTopKDetached Scene::nearest_edges_topk<true>(
    const Vector3fDetached &point,
    int k,
    MaskDetached active) const;
template NearestEdgesTopK Scene::nearest_edges_topk<false>(
    const Vector3f &point,
    int k,
    Mask active) const;

} // namespace rayd

