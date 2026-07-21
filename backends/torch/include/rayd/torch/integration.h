#pragma once

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include <rayd/torch/rf/diffraction.h>
#include <rayd/torch/rf/scattering.h>
#include <rayd/torch/rf/transmission.h>

namespace rayd::torch {

inline constexpr std::uint32_t kIntegrationApiVersion = 5;
inline constexpr std::string_view kIntegrationHeaderIdentity =
    "rayd.torch.integration";

namespace detail {
struct IntegrationAccess;
} // namespace detail

struct MeshInput {
    at::Tensor vertices;
    at::Tensor faces;
    at::Tensor uv;
    at::Tensor face_uv;
    at::Tensor to_world_left;
    at::Tensor to_world_right;
    bool use_face_normals = false;
    bool edges_enabled = true;
    bool dynamic = false;
};

class SceneResource final {
public:
    SceneResource(SceneResource &&) noexcept;
    SceneResource &operator=(SceneResource &&) noexcept;
    ~SceneResource() noexcept;

    SceneResource(const SceneResource &) = delete;
    SceneResource &operator=(const SceneResource &) = delete;

    bool valid() const noexcept;
    int device_index() const;

private:
    class Impl;
    explicit SceneResource(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;

    friend struct detail::IntegrationAccess;
    friend SceneResource create_scene(std::vector<MeshInput> meshes);
};

SceneResource create_scene(std::vector<MeshInput> meshes);

struct SceneEdgeRecordsResult {
    at::Tensor global_vertices;
    at::Tensor global_faces;
    at::Tensor tri_fn_x;
    at::Tensor tri_fn_y;
    at::Tensor tri_fn_z;
    at::Tensor edge_v0;
    at::Tensor edge_v1;
    at::Tensor edge_face0_global;
    at::Tensor edge_face1_global;
    at::Tensor edge_shape_id;
    at::Tensor edge_local_id;
    at::Tensor edge_opposite;
};

SceneEdgeRecordsResult scene_edge_records(const SceneResource &scene);

struct RayBatch {
    at::Tensor ray_o;
    at::Tensor ray_d;
    // Absent means no per-ray upper bound; a present tensor follows the
    // established scalar-f32 batch contract, including a defined-empty value.
    std::optional<at::Tensor> ray_tmax;
    std::optional<at::Tensor> active;
};

struct IntersectResult {
    at::Tensor t;
    at::Tensor p;
    at::Tensor n;
    at::Tensor geo_n;
    at::Tensor uv;
    at::Tensor barycentric;
    at::Tensor shape_id;
    at::Tensor prim_id;
    at::Tensor local_prim_id;
    at::Tensor global_prim_id;
};

IntersectResult intersect_forward(
    const SceneResource &scene,
    const RayBatch &rays,
    std::int64_t flags);

struct IntersectBackwardRequest {
    RayBatch rays;
    at::Tensor tape_prim_id;
    at::Tensor tape_barycentric;
    std::optional<at::Tensor> grad_t;
    std::optional<at::Tensor> grad_p;
    std::optional<at::Tensor> grad_n;
    std::optional<at::Tensor> grad_geo_n;
    std::optional<at::Tensor> grad_uv;
    std::optional<at::Tensor> grad_barycentric;
    bool need_grad_vertices = false;
    bool need_grad_ray_o = false;
    bool need_grad_ray_d = false;
    bool need_grad_ray_tmax = false;
};

struct IntersectBackwardResult {
    at::Tensor grad_vertices;
    at::Tensor grad_ray_o;
    at::Tensor grad_ray_d;
    at::Tensor grad_ray_tmax;
};

IntersectBackwardResult intersect_backward(
    const SceneResource &scene,
    const IntersectBackwardRequest &request);

struct IntersectJvpRequest {
    at::Tensor ray_o;
    at::Tensor ray_d;
    std::optional<at::Tensor> active;
    at::Tensor tape_prim_id;
    at::Tensor tape_barycentric;
    std::optional<at::Tensor> tangent_vertices;
    std::optional<at::Tensor> tangent_ray_o;
    std::optional<at::Tensor> tangent_ray_d;
    std::int64_t flags = 0;
};

struct IntersectJvpResult {
    at::Tensor tangent_t;
    at::Tensor tangent_p;
    at::Tensor tangent_n;
    at::Tensor tangent_geo_n;
    at::Tensor tangent_uv;
    at::Tensor tangent_barycentric;
};

IntersectJvpResult intersect_jvp(
    const SceneResource &scene,
    const IntersectJvpRequest &request);

struct VisibilityRequest {
    at::Tensor start;
    at::Tensor end;
    std::optional<at::Tensor> active;
};

struct VisibilityResult {
    at::Tensor visible;
    at::Tensor blocker_prim;
    at::Tensor tape_t;
};

VisibilityResult visibility_forward(
    const SceneResource &scene,
    const VisibilityRequest &request);

inline constexpr std::array<std::uint32_t, 4>
    kDiffractionTxAxialEdgeFractionBits = {
        0x3ca3d70au,
        0x3eaaaaabu,
        0x3f2aaaabu,
        0x3f7ae148u,
    };

struct AxialEdgeVisibilityConfig {
    std::array<std::uint32_t, 4> sample_fraction_bits =
        kDiffractionTxAxialEdgeFractionBits;
};

struct AxialEdgeVisibilityRequest {
    at::Tensor tx;
    at::Tensor edge_position;
    at::Tensor edge_direction;
    at::Tensor edge_t_min;
    at::Tensor edge_t_max;
    std::optional<at::Tensor> active;
    AxialEdgeVisibilityConfig config;
};

struct AxialEdgeVisibilityResult {
    at::Tensor any_visible;
};

AxialEdgeVisibilityResult axial_edge_visibility_forward(
    const SceneResource &scene,
    const AxialEdgeVisibilityRequest &request);

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

struct MaterialPayload {
    at::Tensor eta_r;
    at::Tensor sigma;
    at::Tensor mu_r;
    at::Tensor gain;
    at::Tensor valid;
};

struct Grid2D {
    std::int64_t axis = 0;
    double position = 0.0;
    double coord0_min = 0.0;
    double coord0_max = 0.0;
    double coord1_min = 0.0;
    double coord1_max = 0.0;
    std::int64_t resolution0 = 0;
    std::int64_t resolution1 = 0;
    double cell_area = 0.0;
};

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

struct DiffractionState {
    at::Tensor edge_index;
    at::Tensor edge_pos;
    at::Tensor edge_dir;
    at::Tensor edge_t_min;
    at::Tensor edge_t_max;
    at::Tensor n0;
    at::Tensor n1;
    at::Tensor prim0;
    at::Tensor prim1;
    at::Tensor exterior_angle;
    at::Tensor src;
    at::Tensor src_power;
    std::optional<at::Tensor> wi;
    std::optional<at::Tensor> d0;
};

enum class DiffractionPathLayout : std::uint8_t {
    Compact = 0,
    SourceLane = 1,
};

struct RecursiveDiffractionState {
    std::optional<at::Tensor> active;
    at::Tensor edge_index;
    at::Tensor edge_pos;
    at::Tensor edge_dir;
    at::Tensor edge_t_min;
    at::Tensor edge_t_max;
    at::Tensor n0;
    at::Tensor n1;
    at::Tensor prim0;
    at::Tensor prim1;
    at::Tensor exterior_angle;
    std::int64_t state_limit = 0;
};

struct DiffractionPathConfig {
    at::Tensor tx_pos;
    at::Tensor tx_pol;
    at::Tensor rx_pos;
    at::Tensor active;
    DiffractionState state;
    MaterialPayload material;
    std::int64_t state_limit = 0;
    std::int64_t capacity = 0;
    double wavelength = 0.0;
    double isb_taper_width_scale = 0.0;
    DiffractionPathLayout layout = DiffractionPathLayout::Compact;
};

struct DiffractionPathResult {
    at::Tensor count;
    at::Tensor valid;
    at::Tensor tx_id;
    at::Tensor rx_id;
    at::Tensor order;
    at::Tensor edge0;
    at::Tensor edge1;
    at::Tensor edge2;
    at::Tensor delay;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
    at::Tensor field_y_re;
    at::Tensor field_y_im;
    at::Tensor field_z_re;
    at::Tensor field_z_im;
    at::Tensor p0;
    at::Tensor p1;
    at::Tensor p2;
};

DiffractionPathResult diffraction_paths_order1_forward(
    const SceneResource &scene,
    const DiffractionPathConfig &config);

struct DiffractionAccumulationConfig {
    std::optional<at::Tensor> active;
    DiffractionState state;
    MaterialPayload material;
    std::int64_t state_limit = 0;
    Grid2D grid;
    double wavelength = 0.0;
    std::int64_t direct_samples = 0;
    std::int64_t keller_samples = 0;
    std::int64_t suffix_samples = 0;
    std::int64_t seed = 0;
    std::int64_t max_order = 1;
    std::optional<RecursiveDiffractionState> recursive_state;
    bool export_tape = false;
    std::optional<at::Tensor> sample_state_index;
    std::optional<at::Tensor> sample_edge_weight;
};

struct DiffractionAccumulationResult {
    at::Tensor power;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
    at::Tensor field_y_re;
    at::Tensor field_y_im;
    at::Tensor field_z_re;
    at::Tensor field_z_im;
    at::Tensor direct_count;
    at::Tensor keller_count;
    at::Tensor suffix_count;
    at::Tensor visibility_rejects;
    at::Tensor edge_visibility_rejects;
    at::Tensor utd_rejects;
    at::Tensor edge_uses;
    at::Tensor tape_active;
    at::Tensor tape_state_idx;
    at::Tensor tape_cell;
    at::Tensor tape_material_idx;
    at::Tensor tape_edge_u;
};

DiffractionAccumulationResult diffraction_accumulation_forward(
    const SceneResource &scene,
    const DiffractionAccumulationConfig &config);

struct CoherentDiffractionConfig {
    std::optional<at::Tensor> active;
    DiffractionState state;
    MaterialPayload material;
    std::int64_t state_limit = 0;
    Grid2D grid;
    double wavelength = 0.0;
    bool select_diffraction_point = false;
    bool prefilter_visibility = false;
};

struct CoherentDiffractionResult {
    at::Tensor direct_x_re;
    at::Tensor direct_x_im;
    at::Tensor direct_y_re;
    at::Tensor direct_y_im;
    at::Tensor direct_z_re;
    at::Tensor direct_z_im;
    at::Tensor multi_x_re;
    at::Tensor multi_x_im;
    at::Tensor multi_y_re;
    at::Tensor multi_y_im;
    at::Tensor multi_z_re;
    at::Tensor multi_z_im;
    at::Tensor direct_count;
    at::Tensor multi_count;
    at::Tensor visibility_reject_count;
    at::Tensor utd_reject_count;
};

CoherentDiffractionResult diffraction_coherent_accumulation_forward(
    const SceneResource &scene,
    const CoherentDiffractionConfig &config);

} // namespace rayd::torch
