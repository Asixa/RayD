#pragma once

#include <rayd/scene/torch.h>

#include <cstdint>
#include <optional>

namespace rayd::torch {

struct ReflectionTraceRequest {
    RayBatch rays;
    std::int64_t max_bounces = 1;
};

struct ReflectionTraceResult {
    at::Tensor valid;
    at::Tensor t;
    at::Tensor prim_ids;
};

struct ReflectionTraceTapeResult {
    at::Tensor valid;
    at::Tensor t;
    at::Tensor image_sources;
    at::Tensor prim_ids;
    at::Tensor tape_prim_id;
    at::Tensor tape_barycentric;
    at::Tensor tape_hit_points;
    at::Tensor tape_normals;
    at::Tensor active_ctx;
};

ReflectionTraceResult trace_reflections_forward(
    const SceneResource &scene,
    const ReflectionTraceRequest &request);
ReflectionTraceTapeResult trace_reflections_forward_tape(
    const SceneResource &scene,
    const ReflectionTraceRequest &request);

struct ReflectionTraceBackwardRequest {
    RayBatch rays;
    at::Tensor tape_prim_id;
    at::Tensor tape_barycentric;
    at::Tensor tape_hit_points;
    at::Tensor tape_normals;
    at::Tensor image_sources;
    std::optional<at::Tensor> grad_t;
    std::optional<at::Tensor> grad_image_sources;
};

struct ReflectionTraceBackwardResult {
    at::Tensor grad_vertices;
    at::Tensor grad_ray_o;
    at::Tensor grad_ray_d;
    at::Tensor grad_ray_tmax;
};

ReflectionTraceBackwardResult trace_reflections_backward(
    const SceneResource &scene,
    const ReflectionTraceBackwardRequest &request);

struct ReflectionTraceJvpRequest {
    at::Tensor ray_o;
    at::Tensor ray_d;
    std::optional<at::Tensor> active;
    at::Tensor tape_prim_id;
    at::Tensor tape_barycentric;
    at::Tensor tape_hit_points;
    at::Tensor tape_normals;
    std::optional<at::Tensor> tangent_vertices;
    std::optional<at::Tensor> tangent_ray_o;
    std::optional<at::Tensor> tangent_ray_d;
    at::Tensor image_sources;
};

struct ReflectionTraceJvpResult {
    at::Tensor tangent_t;
    at::Tensor tangent_image_sources;
};

ReflectionTraceJvpResult trace_reflections_jvp(
    const SceneResource &scene,
    const ReflectionTraceJvpRequest &request);


struct ReflectionAccumulationConfig {
    RayBatch rays;
    at::Tensor tx;
    at::Tensor tx_pol;
    MaterialPayload material;
    std::int64_t max_bounces = 1;
    Grid2D grid;
    double wavelength = 0.0;
    double solid_angle_per_ray = 0.0;
    bool collect_wedges = false;
    bool collect_wedge_prefixes = false;
    std::int64_t wedge_capacity = 0;
    std::int64_t wedge_sample_stride = 1;
    std::int64_t accumulation_strategy = 0;
    std::int64_t compact_min_samples = 0;
    std::int64_t staged_min_samples_per_cell = 0;
    std::int64_t procedural_sample_count = 0;
    bool include_los = false;
};

struct ReflectionAccumulationResult {
    at::Tensor power;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
    at::Tensor field_y_re;
    at::Tensor field_y_im;
    at::Tensor field_z_re;
    at::Tensor field_z_im;
    at::Tensor reflection_count;
    at::Tensor wedge_count;
    at::Tensor wedge_ray_index;
    at::Tensor wedge_hit;
    at::Tensor wedge_normal;
    at::Tensor wedge_prim_id;
    at::Tensor wedge_direction;
    at::Tensor wedge_source;
    at::Tensor wedge_source_power;
    at::Tensor wedge_initial_direction;
    at::Tensor wedge_bounce_depth;
};

ReflectionAccumulationResult reflection_accumulation_forward(
    const SceneResource &scene,
    const ReflectionAccumulationConfig &config);

struct ReflectionEpcRequest {
    at::Tensor source;
    at::Tensor receiver;
    std::optional<at::Tensor> active;
    at::Tensor expected_prim_ids;
    at::Tensor direct_plane_points;
    at::Tensor direct_plane_normals;
    at::Tensor surface_group_id;
    at::Tensor surface_group_size;
    at::Tensor surface_group_members;
    std::int64_t max_bounces = 1;
    std::int64_t visibility_ignore_mode = 0;
    double plane_tolerance = 0.0;
};

struct ReflectionEpcResult {
    at::Tensor valid;
    at::Tensor path_length;
    at::Tensor resolved_prim_ids;
    at::Tensor surface_group_ids;
    at::Tensor hit_positions;
    at::Tensor normals;
};

ReflectionEpcResult reflection_epc_paths_forward(
    const SceneResource &scene,
    const ReflectionEpcRequest &request);

struct ReflectionEpcBackwardRequest {
    at::Tensor source;
    at::Tensor receiver;
    at::Tensor sequence;
    at::Tensor plane_points;
    at::Tensor plane_normals;
    at::Tensor valid;
    at::Tensor bounce_count;
    std::optional<at::Tensor> grad_points;
    std::optional<at::Tensor> grad_normals;
    std::optional<at::Tensor> grad_path_length;
    bool need_grad_vertices = false;
    bool need_grad_source = false;
    bool need_grad_receiver = false;
};

struct ReflectionEpcBackwardResult {
    at::Tensor grad_vertices;
    at::Tensor grad_source;
    at::Tensor grad_receiver;
};

ReflectionEpcBackwardResult reflection_epc_paths_backward(
    const SceneResource &scene,
    const ReflectionEpcBackwardRequest &request);

struct ReflectionEpcJvpRequest {
    at::Tensor source;
    at::Tensor receiver;
    at::Tensor sequence;
    at::Tensor plane_points;
    at::Tensor plane_normals;
    at::Tensor valid;
    at::Tensor bounce_count;
    std::optional<at::Tensor> tangent_vertices;
    std::optional<at::Tensor> tangent_source;
    std::optional<at::Tensor> tangent_receiver;
};

struct ReflectionEpcJvpResult {
    at::Tensor tangent_points;
    at::Tensor tangent_normals;
    at::Tensor tangent_path_length;
};

ReflectionEpcJvpResult reflection_epc_paths_jvp(
    const SceneResource &scene,
    const ReflectionEpcJvpRequest &request);

at::Tensor scene_face_normals_backward(
    const SceneResource &scene,
    const at::Tensor &grad_face_normals);
at::Tensor scene_face_normals_jvp(
    const SceneResource &scene,
    const at::Tensor &tangent_vertices);


} // namespace rayd::torch
