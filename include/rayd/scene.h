// Copyright Xingyu Chen.
// Declares the Torch scene API.

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rayd::torch {

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

} // namespace rayd::torch
