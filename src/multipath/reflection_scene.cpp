#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include <rayd/scene/scene.h>
#include <rayd/edge/scene_edge.h>

#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>

#include "../scene/scene_internal.h"

namespace rayd {

namespace {

enum class ReflectionEpcVisibilityIgnoreMode {
    Primitive,
    SurfaceGroup
};

ReflectionEpcVisibilityIgnoreMode parse_reflection_epc_visibility_ignore_mode(
    const std::string &value) {
    const std::string normalized = normalize_optix_split_mode_value(value.c_str());
    if (normalized.empty() || normalized == "primitive" ||
        normalized == "prim" || normalized == "exact") {
        return ReflectionEpcVisibilityIgnoreMode::Primitive;
    }
    if (normalized == "surface_group" || normalized == "surface-group" ||
        normalized == "group") {
        return ReflectionEpcVisibilityIgnoreMode::SurfaceGroup;
    }
    throw std::runtime_error(
        "Invalid ReflectionEpcOptions.visibility_ignore_mode. "
        "Expected one of: 'primitive', 'surface_group'.");
}

bool recording_reflections() {
    return jit_flag(JitFlag::Recording);
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

struct ReflectionEpcRaw {
    int ray_count = 0;
    int max_bounces = 0;
    MaskDetached valid;
    IntDetached bounce_count;
    FloatDetached path_length;
    FloatDetached point_x;
    FloatDetached point_y;
    FloatDetached point_z;
    IntDetached trace_prim_ids;
    IntDetached resolved_prim_ids;
    IntDetached surface_group_ids;
    FloatDetached plane_normal_x;
    FloatDetached plane_normal_y;
    FloatDetached plane_normal_z;
    IntDetached first_blocked_segment;
    IntDetached first_blocked_prim;
    IntDetached first_blocked_group;
};

struct ReflectionAccumulationRaw {
    int ray_count = 0;
    int max_bounces = 0;
    int grid_cell_count = 0;
    int wedge_capacity = 0;
    FloatDetached reflection_power;
    FloatDetached field_x_re;
    FloatDetached field_x_im;
    FloatDetached field_y_re;
    FloatDetached field_y_im;
    FloatDetached field_z_re;
    FloatDetached field_z_im;
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
ReflectionEpcResultT<Detached> initialize_reflection_epc_result(int ray_count,
                                                                int max_bounces) {
    ReflectionEpcResultT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    const int slot_count = ray_count * max_bounces;
    result.valid = full<MaskT<Detached>>(false, ray_count);
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.path_length = full<FloatT<Detached>>(Infinity, ray_count);
    result.reflection_points = zeros<Vector3fT<Detached>>(slot_count);
    result.prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.trace_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.resolved_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.surface_group_ids = full<IntT<Detached>>(-1, slot_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.first_blocked_segment = full<IntT<Detached>>(-1, ray_count);
    result.first_blocked_prim = full<IntT<Detached>>(-1, ray_count);
    result.first_blocked_group = full<IntT<Detached>>(-1, ray_count);
    return result;
}

ReflectionEpcRaw allocate_reflection_epc_raw(int ray_count, int max_bounces) {
    const int slot_count = ray_count * max_bounces;
    ReflectionEpcRaw raw;
    raw.ray_count = ray_count;
    raw.max_bounces = max_bounces;
    raw.valid = empty<MaskDetached>(ray_count);
    raw.bounce_count = empty<IntDetached>(ray_count);
    raw.path_length = empty<FloatDetached>(ray_count);
    raw.point_x = empty<FloatDetached>(slot_count);
    raw.point_y = empty<FloatDetached>(slot_count);
    raw.point_z = empty<FloatDetached>(slot_count);
    raw.trace_prim_ids = empty<IntDetached>(slot_count);
    raw.resolved_prim_ids = empty<IntDetached>(slot_count);
    raw.surface_group_ids = empty<IntDetached>(slot_count);
    raw.plane_normal_x = empty<FloatDetached>(slot_count);
    raw.plane_normal_y = empty<FloatDetached>(slot_count);
    raw.plane_normal_z = empty<FloatDetached>(slot_count);
    raw.first_blocked_segment = empty<IntDetached>(ray_count);
    raw.first_blocked_prim = empty<IntDetached>(ray_count);
    raw.first_blocked_group = empty<IntDetached>(ray_count);
    return raw;
}

template <bool Detached>
ReflectionEpcFieldResultT<Detached> initialize_reflection_epc_field_result(
    int ray_count,
    int max_bounces,
    const ReflectionEpcFieldOptions &options) {
    ReflectionEpcFieldResultT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    const int slot_count = ray_count * max_bounces;
    const bool return_geometry = options.return_geometry;
    const bool return_endpoints = options.return_endpoints;
    const bool return_hit_points =
        return_geometry && options.return_hit_points;
    const bool return_normals = return_geometry && options.return_normals;
    const bool return_resolved_prim_ids =
        return_geometry && options.return_resolved_prim_ids;
    const bool return_surface_group_ids =
        return_geometry && options.return_surface_group_ids;

    result.valid = empty<MaskT<Detached>>(ray_count);
    result.bounce_count = empty<IntT<Detached>>(ray_count);
    result.path_length = empty<FloatT<Detached>>(ray_count);
    result.field_x_re = empty<FloatT<Detached>>(ray_count);
    result.field_x_im = empty<FloatT<Detached>>(ray_count);
    result.field_y_re = empty<FloatT<Detached>>(ray_count);
    result.field_y_im = empty<FloatT<Detached>>(ray_count);
    result.field_z_re = empty<FloatT<Detached>>(ray_count);
    result.field_z_im = empty<FloatT<Detached>>(ray_count);

    if (return_endpoints) {
        result.tx_pos =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
        result.first_hit =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
        result.last_hit =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
    } else {
        result.tx_pos = zeros<Vector3fT<Detached>>(0);
        result.first_hit = zeros<Vector3fT<Detached>>(0);
        result.last_hit = zeros<Vector3fT<Detached>>(0);
    }

    if (return_hit_points) {
        result.hit_points =
            Vector3fT<Detached>(empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count));
    } else {
        result.hit_points = zeros<Vector3fT<Detached>>(0);
    }
    if (return_normals) {
        result.normals =
            Vector3fT<Detached>(empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count));
    } else {
        result.normals = zeros<Vector3fT<Detached>>(0);
    }
    if (return_resolved_prim_ids) {
        result.resolved_prim_ids = empty<IntT<Detached>>(slot_count);
    } else {
        result.resolved_prim_ids = full<IntT<Detached>>(-1, 0);
    }
    if (return_surface_group_ids) {
        result.surface_group_ids = empty<IntT<Detached>>(slot_count);
    } else {
        result.surface_group_ids = full<IntT<Detached>>(-1, 0);
    }

    return result;
}

ReflectionEpcOptions epc_options_from_field_options(
    const ReflectionEpcFieldOptions &options) {
    ReflectionEpcOptions epc_options;
    epc_options.expected_prim_ids = options.expected_prim_ids;
    epc_options.surface_group_id = options.surface_group_id;
    epc_options.surface_group_size = options.surface_group_size;
    epc_options.surface_group_members = options.surface_group_members;
    epc_options.surface_max_group_size = options.surface_max_group_size;
    epc_options.visibility_ignore_mode = options.visibility_ignore_mode;
    epc_options.final_ignore_group_ids = options.final_ignore_group_ids;
    return epc_options;
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
    raw.field_x_re = empty<FloatDetached>(grid_cell_count);
    raw.field_x_im = empty<FloatDetached>(grid_cell_count);
    raw.field_y_re = empty<FloatDetached>(grid_cell_count);
    raw.field_y_im = empty<FloatDetached>(grid_cell_count);
    raw.field_z_re = empty<FloatDetached>(grid_cell_count);
    raw.field_z_im = empty<FloatDetached>(grid_cell_count);
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
    jit_memset_async(JitBackend::CUDA,
                     raw.field_x_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_x_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_y_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_y_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_z_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_z_im.data(),
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

} // namespace

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

    const OptixSceneSelection scenes = select_optix_scenes();
    const OptixScene *primary_scene = scenes.primary;
    const OptixScene *secondary_scene = scenes.secondary;
    int split_mode = scenes.split_mode;
    int hitgroup_record_count = scenes.hitgroup_record_count;

    require(primary_scene != nullptr && primary_scene->is_ready(),
            "Scene::trace_reflections(): OptiX scene is not ready.");
    require(hitgroup_record_count > 0,
            "Scene::trace_reflections(): invalid hitgroup record count.");

    ensure_pipeline(reflection_pipeline_, primary_scene->context(),
                    hitgroup_record_count, reflection_trace_pipeline_config());

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

    reflection_pipeline_->launch(0, params);

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
ReflectionEpcResultT<Detached> Scene::trace_reflection_epc(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflectionEpcOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflection_epc(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflection_epc(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_reflection_epc(): max_bounces must be positive.");
    require(max_bounces <= ReflectionEpcMaxBounces,
            "Scene::trace_reflection_epc(): max_bounces exceeds the native EPC limit.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionEpcResultT<Detached> result =
        initialize_reflection_epc_result<Detached>(ray_count, max_bounces);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_reflection_epc(): native EPC is a non-AD native fast path. "
                "Pass RayDetached and detached receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_reflection_epc(): receiver width must be 1 or match ray count.");
        const ReflectionEpcVisibilityIgnoreMode visibility_ignore_mode =
            parse_reflection_epc_visibility_ignore_mode(options.visibility_ignore_mode);
        const bool surface_group_ignore =
            visibility_ignore_mode == ReflectionEpcVisibilityIgnoreMode::SurfaceGroup;
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        const int surface_group_id_count = static_cast<int>(slices(options.surface_group_id));
        const int surface_group_count = static_cast<int>(slices(options.surface_group_size));
        const int surface_group_member_count =
            static_cast<int>(slices(options.surface_group_members));
        const int final_ignore_group_count =
            static_cast<int>(slices(options.final_ignore_group_ids));
        const bool has_surface_groups =
            surface_group_id_count > 0 ||
            surface_group_count > 0 ||
            surface_group_member_count > 0 ||
            options.surface_max_group_size > 0;
        require(expected_prim_count == 0 || expected_prim_count == slot_count,
                "Scene::trace_reflection_epc(): expected_prim_ids width must be n_rays * max_bounces.");
        require(final_ignore_group_count == 0 ||
                    final_ignore_group_count == 1 ||
                    final_ignore_group_count == ray_count,
                "Scene::trace_reflection_epc(): final_ignore_group_ids width must be 1 or match ray count.");
        if (has_surface_groups) {
            const int triangle_count =
                static_cast<int>(slices(triangle_info_detached_.p0));
            require(surface_group_id_count == triangle_count,
                    "Scene::trace_reflection_epc(): surface_group_id width must match triangle count.");
            require(surface_group_count > 0,
                    "Scene::trace_reflection_epc(): surface_group_size must be non-empty when surface groups are provided.");
            require(options.surface_max_group_size > 0,
                    "Scene::trace_reflection_epc(): surface_max_group_size must be positive when surface groups are provided.");
            require(surface_group_member_count >=
                        surface_group_count * options.surface_max_group_size,
                    "Scene::trace_reflection_epc(): surface_group_members must contain surface_group_size * surface_max_group_size entries.");
        }
        require(!surface_group_ignore || has_surface_groups,
                "Scene::trace_reflection_epc(): visibility_ignore_mode='surface_group' requires surface group tables.");
        require(final_ignore_group_count == 0 || surface_group_ignore,
                "Scene::trace_reflection_epc(): final_ignore_group_ids require visibility_ignore_mode='surface_group'.");

        const MaskDetached active_detached =
            sanitize_reflection_active<Detached>(ray, active);
        if (drjit::none(active_detached)) {
            return result;
        }

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        int split_mode = scenes.split_mode;
        int hitgroup_record_count = scenes.hitgroup_record_count;

        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::trace_reflection_epc(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::trace_reflection_epc(): invalid hitgroup record count.");

        ensure_pipeline(reflection_epc_pipeline_, primary_scene->context(),
                        hitgroup_record_count, reflection_epc_pipeline_config());

        ensure_reflection_epc_geometry_ready();
        drjit::eval(ray.o,
                    ray.d,
                    ray.tmax,
                    receiver,
                    active_detached);
        if (expected_prim_count > 0) {
            drjit::eval(options.expected_prim_ids);
        }
        if (has_surface_groups) {
            drjit::eval(options.surface_group_id,
                        options.surface_group_size,
                        options.surface_group_members);
        }
        if (final_ignore_group_count > 0) {
            drjit::eval(options.final_ignore_group_ids);
        }

        ReflectionEpcRaw raw = allocate_reflection_epc_raw(ray_count, max_bounces);

        ReflectionEpcParams params = {};
        params.primary_handle = primary_scene->ias_handle();
        params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready()
                ? secondary_scene->ias_handle()
                : 0ull;
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
        params.expected_prim_ids =
            expected_prim_count > 0 ? options.expected_prim_ids.data() : nullptr;
        params.expected_prim_count = expected_prim_count;
        params.surface_group_id =
            has_surface_groups ? options.surface_group_id.data() : nullptr;
        params.surface_group_id_count = surface_group_id_count;
        params.surface_group_size =
            has_surface_groups ? options.surface_group_size.data() : nullptr;
        params.surface_group_count = surface_group_count;
        params.surface_group_members =
            has_surface_groups ? options.surface_group_members.data() : nullptr;
        params.surface_max_group_size =
            has_surface_groups ? options.surface_max_group_size : 0;
        params.visibility_ignore_mode =
            surface_group_ignore ? ReflectionEpcVisibilityIgnoreSurfaceGroup
                                 : ReflectionEpcVisibilityIgnorePrimitive;
        params.final_ignore_group_ids =
            final_ignore_group_count > 0 ? options.final_ignore_group_ids.data() : nullptr;
        params.final_ignore_group_count = final_ignore_group_count;
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.ray_dx = ray.d.x().data();
        params.ray_dy = ray.d.y().data();
        params.ray_dz = ray.d.z().data();
        params.ray_tmax = ray.tmax.data();
        params.rx_x = receiver.x().data();
        params.rx_y = receiver.y().data();
        params.rx_z = receiver.z().data();
        params.rx_count = receiver_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.n_rays = ray_count;
        params.max_bounces = max_bounces;
        params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        params.out_bounce_count = raw.bounce_count.data();
        params.out_path_length = raw.path_length.data();
        params.out_point_x = raw.point_x.data();
        params.out_point_y = raw.point_y.data();
        params.out_point_z = raw.point_z.data();
        params.out_trace_prim_ids = raw.trace_prim_ids.data();
        params.out_resolved_prim_ids = raw.resolved_prim_ids.data();
        params.out_surface_group_ids = raw.surface_group_ids.data();
        params.out_plane_normal_x = raw.plane_normal_x.data();
        params.out_plane_normal_y = raw.plane_normal_y.data();
        params.out_plane_normal_z = raw.plane_normal_z.data();
        params.out_first_blocked_segment = raw.first_blocked_segment.data();
        params.out_first_blocked_prim = raw.first_blocked_prim.data();
        params.out_first_blocked_group = raw.first_blocked_group.data();

        reflection_epc_pipeline_->launch(0, params);

        result.valid = raw.valid;
        result.bounce_count = raw.bounce_count;
        result.path_length = raw.path_length;
        result.reflection_points =
            Vector3fDetached(raw.point_x, raw.point_y, raw.point_z);
        result.prim_ids = raw.trace_prim_ids;
        result.trace_prim_ids = raw.trace_prim_ids;
        result.resolved_prim_ids = raw.resolved_prim_ids;
        result.surface_group_ids = raw.surface_group_ids;
        result.plane_normals =
            Vector3fDetached(raw.plane_normal_x,
                             raw.plane_normal_y,
                             raw.plane_normal_z);
        result.first_blocked_segment = raw.first_blocked_segment;
        result.first_blocked_prim = raw.first_blocked_prim;
        result.first_blocked_group = raw.first_blocked_group;
        return result;
    }
}

template <bool Detached>
ReflectionEpcFieldResultT<Detached> Scene::trace_reflection_epc_field(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflectionEpcFieldOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflection_epc_field(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflection_epc_field(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_reflection_epc_field(): max_bounces must be positive.");
    require(max_bounces <= ReflectionEpcMaxBounces,
            "Scene::trace_reflection_epc_field(): max_bounces exceeds the native EPC limit.");
    require(options.omega > 0.f,
            "Scene::trace_reflection_epc_field(): omega must be positive.");
    require(options.wavelength > 0.f,
            "Scene::trace_reflection_epc_field(): wavelength must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionEpcFieldResultT<Detached> result =
        initialize_reflection_epc_field_result<Detached>(
            ray_count,
            max_bounces,
            options);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_reflection_epc_field(): native EPC field is a non-AD native fast path. "
                "Pass RayDetached and detached receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_reflection_epc_field(): receiver width must be 1 or match ray count.");
        const int slot_count = ray_count * max_bounces;
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_reflection_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_reflection_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_reflection_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_reflection_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_reflection_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_reflection_epc_field(): tx_polarization width must be 1 or match ray count.");

        drjit::eval(options.slot_plane_normal,
                    options.slot_eta_r,
                    options.slot_mu_r,
                    options.slot_sigma,
                    options.slot_gain,
                    options.tx_polarization);

        const ReflectionEpcOptions epc_options =
            epc_options_from_field_options(options);
        const ReflectionEpcResultDetached epc =
            trace_reflection_epc<true>(
                ray,
                receiver,
                max_bounces,
                epc_options,
                active);

        ReflectionEpcFieldParams params = {};
        params.n_rays = ray_count;
        params.max_bounces = max_bounces;
        params.epc_valid = reinterpret_cast<const uint8_t *>(epc.valid.data());
        params.epc_bounce_count = epc.bounce_count.data();
        params.epc_path_length = epc.path_length.data();
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.rx_x = receiver.x().data();
        params.rx_y = receiver.y().data();
        params.rx_z = receiver.z().data();
        params.rx_count = receiver_count;
        params.hit_x = epc.reflection_points.x().data();
        params.hit_y = epc.reflection_points.y().data();
        params.hit_z = epc.reflection_points.z().data();
        params.epc_normal_x = epc.plane_normals.x().data();
        params.epc_normal_y = epc.plane_normals.y().data();
        params.epc_normal_z = epc.plane_normals.z().data();
        const bool return_resolved_prim_ids =
            options.return_geometry && options.return_resolved_prim_ids;
        const bool return_surface_group_ids =
            options.return_geometry && options.return_surface_group_ids;
        params.resolved_prim_ids =
            return_resolved_prim_ids ? epc.resolved_prim_ids.data() : nullptr;
        params.surface_group_ids =
            return_surface_group_ids ? epc.surface_group_ids.data() : nullptr;
        params.slot_normal_x = options.slot_plane_normal.x().data();
        params.slot_normal_y = options.slot_plane_normal.y().data();
        params.slot_normal_z = options.slot_plane_normal.z().data();
        params.slot_eta_r = options.slot_eta_r.data();
        params.slot_mu_r = options.slot_mu_r.data();
        params.slot_sigma = options.slot_sigma.data();
        params.slot_gain = options.slot_gain.data();
        params.tx_pol_x = options.tx_polarization.x().data();
        params.tx_pol_y = options.tx_polarization.y().data();
        params.tx_pol_z = options.tx_polarization.z().data();
        params.tx_pol_count = tx_pol_count;
        params.omega = options.omega;
        params.wavelength = options.wavelength;
        params.out_valid = reinterpret_cast<uint8_t *>(result.valid.data());
        params.out_bounce_count = result.bounce_count.data();
        params.out_path_length = result.path_length.data();
        params.out_field_x_re = result.field_x_re.data();
        params.out_field_x_im = result.field_x_im.data();
        params.out_field_y_re = result.field_y_re.data();
        params.out_field_y_im = result.field_y_im.data();
        params.out_field_z_re = result.field_z_re.data();
        params.out_field_z_im = result.field_z_im.data();

        if (options.return_endpoints) {
            params.out_tx_x = result.tx_pos.x().data();
            params.out_tx_y = result.tx_pos.y().data();
            params.out_tx_z = result.tx_pos.z().data();
            params.out_first_hit_x = result.first_hit.x().data();
            params.out_first_hit_y = result.first_hit.y().data();
            params.out_first_hit_z = result.first_hit.z().data();
            params.out_last_hit_x = result.last_hit.x().data();
            params.out_last_hit_y = result.last_hit.y().data();
            params.out_last_hit_z = result.last_hit.z().data();
        }
        if (options.return_geometry && options.return_hit_points) {
            params.out_hit_x = result.hit_points.x().data();
            params.out_hit_y = result.hit_points.y().data();
            params.out_hit_z = result.hit_points.z().data();
        }
        if (options.return_geometry && options.return_normals) {
            params.out_normal_x = result.normals.x().data();
            params.out_normal_y = result.normals.y().data();
            params.out_normal_z = result.normals.z().data();
        }
        if (return_resolved_prim_ids) {
            params.out_resolved_prim_ids = result.resolved_prim_ids.data();
        }
        if (return_surface_group_ids) {
            params.out_surface_group_ids = result.surface_group_ids.data();
        }

        reflection_epc_field_gpu(params);
        return result;
    }
}

template <bool Detached>
ReflectionEpcFieldResultT<Detached> Scene::trace_reflection_epc_field_direct(
    const Vector3fT<Detached> &tx_position,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflectionEpcFieldOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflection_epc_field_direct(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflection_epc_field_direct(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_reflection_epc_field_direct(): max_bounces must be positive.");
    require(max_bounces <= ReflectionEpcMaxBounces,
            "Scene::trace_reflection_epc_field_direct(): max_bounces exceeds the native EPC limit.");
    require(options.omega > 0.f,
            "Scene::trace_reflection_epc_field_direct(): omega must be positive.");
    require(options.wavelength > 0.f,
            "Scene::trace_reflection_epc_field_direct(): wavelength must be positive.");

    const int ray_count = static_cast<int>(slices(tx_position));
    ReflectionEpcFieldResultT<Detached> result =
        initialize_reflection_epc_field_result<Detached>(
            ray_count,
            max_bounces,
            options);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_reflection_epc_field_direct(): native EPC field is a non-AD native fast path. "
                "Pass detached transmitter and receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_reflection_epc_field_direct(): receiver width must be 1 or match transmitter count.");
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        require(expected_prim_count == slot_count,
                "Scene::trace_reflection_epc_field_direct(): expected_prim_ids width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_point)) == slot_count,
                "Scene::trace_reflection_epc_field_direct(): slot_plane_point width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_reflection_epc_field_direct(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_reflection_epc_field_direct(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_reflection_epc_field_direct(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_reflection_epc_field_direct(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_reflection_epc_field_direct(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_reflection_epc_field_direct(): tx_polarization width must be 1 or match transmitter count.");

        const ReflectionEpcVisibilityIgnoreMode visibility_ignore_mode =
            parse_reflection_epc_visibility_ignore_mode(options.visibility_ignore_mode);
        const bool surface_group_ignore =
            visibility_ignore_mode == ReflectionEpcVisibilityIgnoreMode::SurfaceGroup;
        const int surface_group_id_count = static_cast<int>(slices(options.surface_group_id));
        const int surface_group_count = static_cast<int>(slices(options.surface_group_size));
        const int surface_group_member_count =
            static_cast<int>(slices(options.surface_group_members));
        const int final_ignore_group_count =
            static_cast<int>(slices(options.final_ignore_group_ids));
        const bool has_surface_groups =
            surface_group_id_count > 0 ||
            surface_group_count > 0 ||
            surface_group_member_count > 0 ||
            options.surface_max_group_size > 0;
        require(final_ignore_group_count == 0 ||
                    final_ignore_group_count == 1 ||
                    final_ignore_group_count == ray_count,
                "Scene::trace_reflection_epc_field_direct(): final_ignore_group_ids width must be 1 or match transmitter count.");
        if (has_surface_groups) {
            const int triangle_count =
                static_cast<int>(slices(triangle_info_detached_.p0));
            require(surface_group_id_count == triangle_count,
                    "Scene::trace_reflection_epc_field_direct(): surface_group_id width must match triangle count.");
            require(surface_group_count > 0,
                    "Scene::trace_reflection_epc_field_direct(): surface_group_size must be non-empty when surface groups are provided.");
            require(options.surface_max_group_size > 0,
                    "Scene::trace_reflection_epc_field_direct(): surface_max_group_size must be positive when surface groups are provided.");
            require(surface_group_member_count >=
                        surface_group_count * options.surface_max_group_size,
                    "Scene::trace_reflection_epc_field_direct(): surface_group_members must contain surface_group_size * surface_max_group_size entries.");
        }
        require(!surface_group_ignore || has_surface_groups,
                "Scene::trace_reflection_epc_field_direct(): visibility_ignore_mode='surface_group' requires surface group tables.");
        require(final_ignore_group_count == 0 || surface_group_ignore,
                "Scene::trace_reflection_epc_field_direct(): final_ignore_group_ids require visibility_ignore_mode='surface_group'.");

        MaskDetached active_detached = sanitize_segment_active<Detached>(
            tx_position,
            receiver,
            active);
        if (drjit::none(active_detached)) {
            return result;
        }

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        int split_mode = scenes.split_mode;
        int hitgroup_record_count = scenes.hitgroup_record_count;
        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::trace_reflection_epc_field_direct(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::trace_reflection_epc_field_direct(): invalid hitgroup record count.");

        ensure_pipeline(reflection_epc_pipeline_, primary_scene->context(),
                        hitgroup_record_count, reflection_epc_pipeline_config());

        ensure_reflection_epc_geometry_ready();
        drjit::eval(tx_position,
                    receiver,
                    active_detached,
                    options.expected_prim_ids,
                    options.slot_plane_point,
                    options.slot_plane_normal,
                    options.slot_eta_r,
                    options.slot_mu_r,
                    options.slot_sigma,
                    options.slot_gain,
                    options.tx_polarization);
        if (has_surface_groups) {
            drjit::eval(options.surface_group_id,
                        options.surface_group_size,
                        options.surface_group_members);
        }
        if (final_ignore_group_count > 0) {
            drjit::eval(options.final_ignore_group_ids);
        }

        ReflectionEpcRaw raw = allocate_reflection_epc_raw(ray_count, max_bounces);
        ReflectionEpcParams epc_params = {};
        epc_params.primary_handle = primary_scene->ias_handle();
        epc_params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready()
                ? secondary_scene->ias_handle()
                : 0ull;
        epc_params.split_mode = split_mode;
        epc_params.tri_p0_x = triangle_info_detached_.p0.x().data();
        epc_params.tri_p0_y = triangle_info_detached_.p0.y().data();
        epc_params.tri_p0_z = triangle_info_detached_.p0.z().data();
        epc_params.tri_e1_x = triangle_info_detached_.e1.x().data();
        epc_params.tri_e1_y = triangle_info_detached_.e1.y().data();
        epc_params.tri_e1_z = triangle_info_detached_.e1.z().data();
        epc_params.tri_e2_x = triangle_info_detached_.e2.x().data();
        epc_params.tri_e2_y = triangle_info_detached_.e2.y().data();
        epc_params.tri_e2_z = triangle_info_detached_.e2.z().data();
        epc_params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        epc_params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        epc_params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        epc_params.face_offsets = face_offsets_.data();
        epc_params.n_meshes = mesh_count_;
        epc_params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        epc_params.expected_prim_ids = options.expected_prim_ids.data();
        epc_params.expected_prim_count = expected_prim_count;
        epc_params.surface_group_id =
            has_surface_groups ? options.surface_group_id.data() : nullptr;
        epc_params.surface_group_id_count = surface_group_id_count;
        epc_params.surface_group_size =
            has_surface_groups ? options.surface_group_size.data() : nullptr;
        epc_params.surface_group_count = surface_group_count;
        epc_params.surface_group_members =
            has_surface_groups ? options.surface_group_members.data() : nullptr;
        epc_params.surface_max_group_size =
            has_surface_groups ? options.surface_max_group_size : 0;
        epc_params.visibility_ignore_mode =
            surface_group_ignore ? ReflectionEpcVisibilityIgnoreSurfaceGroup
                                 : ReflectionEpcVisibilityIgnorePrimitive;
        epc_params.final_ignore_group_ids =
            final_ignore_group_count > 0 ? options.final_ignore_group_ids.data() : nullptr;
        epc_params.final_ignore_group_count = final_ignore_group_count;
        epc_params.ray_ox = tx_position.x().data();
        epc_params.ray_oy = tx_position.y().data();
        epc_params.ray_oz = tx_position.z().data();
        epc_params.ray_dx = nullptr;
        epc_params.ray_dy = nullptr;
        epc_params.ray_dz = nullptr;
        epc_params.ray_tmax = nullptr;
        epc_params.direct_plane_point_x = options.slot_plane_point.x().data();
        epc_params.direct_plane_point_y = options.slot_plane_point.y().data();
        epc_params.direct_plane_point_z = options.slot_plane_point.z().data();
        epc_params.direct_plane_normal_x = options.slot_plane_normal.x().data();
        epc_params.direct_plane_normal_y = options.slot_plane_normal.y().data();
        epc_params.direct_plane_normal_z = options.slot_plane_normal.z().data();
        epc_params.rx_x = receiver.x().data();
        epc_params.rx_y = receiver.y().data();
        epc_params.rx_z = receiver.z().data();
        epc_params.rx_count = receiver_count;
        epc_params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        epc_params.n_rays = ray_count;
        epc_params.max_bounces = max_bounces;
        epc_params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        epc_params.out_bounce_count = raw.bounce_count.data();
        epc_params.out_path_length = raw.path_length.data();
        epc_params.out_point_x = raw.point_x.data();
        epc_params.out_point_y = raw.point_y.data();
        epc_params.out_point_z = raw.point_z.data();
        epc_params.out_trace_prim_ids = raw.trace_prim_ids.data();
        epc_params.out_resolved_prim_ids = raw.resolved_prim_ids.data();
        epc_params.out_surface_group_ids = raw.surface_group_ids.data();
        epc_params.out_plane_normal_x = raw.plane_normal_x.data();
        epc_params.out_plane_normal_y = raw.plane_normal_y.data();
        epc_params.out_plane_normal_z = raw.plane_normal_z.data();
        epc_params.out_first_blocked_segment = raw.first_blocked_segment.data();
        epc_params.out_first_blocked_prim = raw.first_blocked_prim.data();
        epc_params.out_first_blocked_group = raw.first_blocked_group.data();
        reflection_epc_pipeline_->launch(0, epc_params);

        ReflectionEpcFieldParams field_params = {};
        field_params.n_rays = ray_count;
        field_params.max_bounces = max_bounces;
        field_params.epc_valid = reinterpret_cast<const uint8_t *>(raw.valid.data());
        field_params.epc_bounce_count = raw.bounce_count.data();
        field_params.epc_path_length = raw.path_length.data();
        field_params.ray_ox = tx_position.x().data();
        field_params.ray_oy = tx_position.y().data();
        field_params.ray_oz = tx_position.z().data();
        field_params.rx_x = receiver.x().data();
        field_params.rx_y = receiver.y().data();
        field_params.rx_z = receiver.z().data();
        field_params.rx_count = receiver_count;
        field_params.hit_x = raw.point_x.data();
        field_params.hit_y = raw.point_y.data();
        field_params.hit_z = raw.point_z.data();
        field_params.epc_normal_x = raw.plane_normal_x.data();
        field_params.epc_normal_y = raw.plane_normal_y.data();
        field_params.epc_normal_z = raw.plane_normal_z.data();
        const bool return_resolved_prim_ids =
            options.return_geometry && options.return_resolved_prim_ids;
        const bool return_surface_group_ids =
            options.return_geometry && options.return_surface_group_ids;
        field_params.resolved_prim_ids =
            return_resolved_prim_ids ? raw.resolved_prim_ids.data() : nullptr;
        field_params.surface_group_ids =
            return_surface_group_ids ? raw.surface_group_ids.data() : nullptr;
        field_params.slot_normal_x = options.slot_plane_normal.x().data();
        field_params.slot_normal_y = options.slot_plane_normal.y().data();
        field_params.slot_normal_z = options.slot_plane_normal.z().data();
        field_params.slot_eta_r = options.slot_eta_r.data();
        field_params.slot_mu_r = options.slot_mu_r.data();
        field_params.slot_sigma = options.slot_sigma.data();
        field_params.slot_gain = options.slot_gain.data();
        field_params.tx_pol_x = options.tx_polarization.x().data();
        field_params.tx_pol_y = options.tx_polarization.y().data();
        field_params.tx_pol_z = options.tx_polarization.z().data();
        field_params.tx_pol_count = tx_pol_count;
        field_params.omega = options.omega;
        field_params.wavelength = options.wavelength;
        field_params.out_valid = reinterpret_cast<uint8_t *>(result.valid.data());
        field_params.out_bounce_count = result.bounce_count.data();
        field_params.out_path_length = result.path_length.data();
        field_params.out_field_x_re = result.field_x_re.data();
        field_params.out_field_x_im = result.field_x_im.data();
        field_params.out_field_y_re = result.field_y_re.data();
        field_params.out_field_y_im = result.field_y_im.data();
        field_params.out_field_z_re = result.field_z_re.data();
        field_params.out_field_z_im = result.field_z_im.data();

        if (options.return_endpoints) {
            field_params.out_tx_x = result.tx_pos.x().data();
            field_params.out_tx_y = result.tx_pos.y().data();
            field_params.out_tx_z = result.tx_pos.z().data();
            field_params.out_first_hit_x = result.first_hit.x().data();
            field_params.out_first_hit_y = result.first_hit.y().data();
            field_params.out_first_hit_z = result.first_hit.z().data();
            field_params.out_last_hit_x = result.last_hit.x().data();
            field_params.out_last_hit_y = result.last_hit.y().data();
            field_params.out_last_hit_z = result.last_hit.z().data();
        }
        if (options.return_geometry && options.return_hit_points) {
            field_params.out_hit_x = result.hit_points.x().data();
            field_params.out_hit_y = result.hit_points.y().data();
            field_params.out_hit_z = result.hit_points.z().data();
        }
        if (options.return_geometry && options.return_normals) {
            field_params.out_normal_x = result.normals.x().data();
            field_params.out_normal_y = result.normals.y().data();
            field_params.out_normal_z = result.normals.z().data();
        }
        if (return_resolved_prim_ids) {
            field_params.out_resolved_prim_ids = result.resolved_prim_ids.data();
        }
        if (return_surface_group_ids) {
            field_params.out_surface_group_ids = result.surface_group_ids.data();
        }

        reflection_epc_field_gpu(field_params);
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
    MaskT<Detached> active,
    const Vector3fT<Detached> &tx_polarization) const {
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
        result.reflection_field_x =
            drjit::Complex<FloatDetached>(zeros<FloatDetached>(grid_cell_count),
                                          zeros<FloatDetached>(grid_cell_count));
        result.reflection_field_y =
            drjit::Complex<FloatDetached>(zeros<FloatDetached>(grid_cell_count),
                                          zeros<FloatDetached>(grid_cell_count));
        result.reflection_field_z =
            drjit::Complex<FloatDetached>(zeros<FloatDetached>(grid_cell_count),
                                          zeros<FloatDetached>(grid_cell_count));
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
        const int tx_pol_count = static_cast<int>(slices(tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_reflections_accumulating(): tx_polarization width must be 1 or match ray count.");

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
        Vector3fDetached tx_pol_detached = tx_polarization;
        if (tx_pol_count == 1 && ray_count > 1) {
            const IntDetached zero_index = full<IntDetached>(0, ray_count);
            tx_pol_detached = Vector3fDetached(
                gather<FloatDetached>(tx_polarization.x(), zero_index),
                gather<FloatDetached>(tx_polarization.y(), zero_index),
                gather<FloatDetached>(tx_polarization.z(), zero_index));
        }

        MaskDetached active_detached = sanitize_reflection_active<true>(ray, active);
        active_detached &= drjit::isfinite(tx_detached.x()) &&
                           drjit::isfinite(tx_detached.y()) &&
                           drjit::isfinite(tx_detached.z()) &&
                           drjit::isfinite(tx_pol_detached.x()) &&
                           drjit::isfinite(tx_pol_detached.y()) &&
                           drjit::isfinite(tx_pol_detached.z());
        if (drjit::none(active_detached)) {
            return result;
        }

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        int split_mode = scenes.split_mode;
        int hitgroup_record_count = scenes.hitgroup_record_count;

        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::trace_reflections_accumulating(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::trace_reflections_accumulating(): invalid hitgroup record count.");

        ensure_pipeline(reflection_accumulation_pipeline_, primary_scene->context(),
                        hitgroup_record_count, reflection_accumulation_pipeline_config());

        drjit::eval(ray.o,
                    ray.d,
                    ray.tmax,
                    tx_detached,
                    tx_pol_detached,
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
        params.tx_pol_x = tx_pol_detached.x().data();
        params.tx_pol_y = tx_pol_detached.y().data();
        params.tx_pol_z = tx_pol_detached.z().data();
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
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
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

        reflection_accumulation_pipeline_->launch(0, params);

        result.reflection_power = raw.reflection_power;
        result.reflection_field_x =
            drjit::Complex<FloatDetached>(raw.field_x_re, raw.field_x_im);
        result.reflection_field_y =
            drjit::Complex<FloatDetached>(raw.field_y_re, raw.field_y_im);
        result.reflection_field_z =
            drjit::Complex<FloatDetached>(raw.field_z_re, raw.field_z_im);
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
    MaskDetached active,
    const Vector3fDetached &tx_polarization) const;

template ReflectionAccumulationResult Scene::trace_reflections_accumulating<false>(
    const Ray &ray,
    const Vector3f &tx_position,
    const ReflectionAccumulationGrid &grid,
    const PrimitiveMaterialPayload &material,
    int max_bounces,
    const ReflectionAccumulationOptions &options,
    Mask active,
    const Vector3f &tx_polarization) const;

template ReflectionEpcResultDetached Scene::trace_reflection_epc<true>(
    const RayDetached &ray,
    const Vector3fDetached &receiver,
    int max_bounces,
    const ReflectionEpcOptions &options,
    MaskDetached active) const;

template ReflectionEpcResult Scene::trace_reflection_epc<false>(
    const Ray &ray,
    const Vector3f &receiver,
    int max_bounces,
    const ReflectionEpcOptions &options,
    Mask active) const;

template ReflectionEpcFieldResultDetached Scene::trace_reflection_epc_field<true>(
    const RayDetached &ray,
    const Vector3fDetached &receiver,
    int max_bounces,
    const ReflectionEpcFieldOptions &options,
    MaskDetached active) const;

template ReflectionEpcFieldResult Scene::trace_reflection_epc_field<false>(
    const Ray &ray,
    const Vector3f &receiver,
    int max_bounces,
    const ReflectionEpcFieldOptions &options,
    Mask active) const;

template ReflectionEpcFieldResultDetached Scene::trace_reflection_epc_field_direct<true>(
    const Vector3fDetached &tx_position,
    const Vector3fDetached &receiver,
    int max_bounces,
    const ReflectionEpcFieldOptions &options,
    MaskDetached active) const;

template ReflectionEpcFieldResult Scene::trace_reflection_epc_field_direct<false>(
    const Vector3f &tx_position,
    const Vector3f &receiver,
    int max_bounces,
    const ReflectionEpcFieldOptions &options,
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

} // namespace rayd
