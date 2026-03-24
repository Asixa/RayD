#pragma once

#include <memory>

#include <raydi/raydi.h>
#include <raydi/edge.h>

namespace raydi {

/// Triangle mesh plus cached geometric data used by RayDi queries.
class Mesh final {
public:
    Mesh() = default;
    Mesh(const Vector3fDetached &vertex_positions,
         const Vector3iDetached &face_indices,
         const Vector2fDetached &vertex_uv = Vector2fDetached(),
         const Vector3iDetached &face_uv_indices = Vector3iDetached(),
         bool verbose = false);
    Mesh(const Mesh &other);
    ~Mesh();

    /// Initialize object-space vertex, index, and optional UV data.
    void init(const Vector3fDetached &vertex_positions,
              const Vector3iDetached &face_indices,
              const Vector2fDetached &vertex_uv = Vector2fDetached(),
              const Vector3iDetached &face_uv_indices = Vector3iDetached(),
              bool verbose = false);

    /// Build derived geometry caches and GPU buffers from the current mesh state.
    void configure();
    void prepare_optix_buffers();
    void update_runtime_data(bool vertices_dirty, bool transform_dirty);

    void set_transform(const Matrix4f &matrix, bool set_left = true);
    void append_transform(const Matrix4f &matrix, bool append_left = true);
    Matrix4f full_transform() const { return left_transform_ * object_to_world_ * right_transform_; }

    int mesh_id() const { return mesh_id_; }
    void set_mesh_id(int mesh_id) { mesh_id_ = mesh_id; }

    bool is_ready() const { return is_ready_; }
    bool use_face_normals() const { return use_face_normals_; }
    void set_use_face_normals(bool value) {
        use_face_normals_ = value;
        is_ready_ = false;
    }

    bool has_uv() const { return has_uv_; }
    bool edges_enabled() const { return edges_enabled_; }
    void set_edges_enabled(bool value) {
        edges_enabled_ = value;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    int vertex_count() const { return vertex_count_; }
    int face_count() const { return face_count_; }

    const Matrix4f &to_world() const { return object_to_world_; }
    void set_to_world(const Matrix4f &matrix) {
        object_to_world_ = matrix;
        world_positions_dirty_ = true;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    const Matrix4f &to_world_left() const { return left_transform_; }
    void set_to_world_left(const Matrix4f &matrix) {
        left_transform_ = matrix;
        world_positions_dirty_ = true;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    const Matrix4f &to_world_right() const { return right_transform_; }
    void set_to_world_right(const Matrix4f &matrix) {
        right_transform_ = matrix;
        world_positions_dirty_ = true;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    const Vector3f &vertex_positions() const { return vertex_positions_object_; }
    void set_vertex_positions(const Vector3f &positions) {
        vertex_positions_object_ = positions;
        vertex_count_ = static_cast<int>(slices(positions));
        world_positions_dirty_ = true;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    const Vector3f &vertex_positions_world() const;
    const Vector3f &vertex_normals() const { return vertex_normals_object_; }

    const Vector2f &vertex_uv() const { return vertex_uv_; }
    void set_vertex_uv(const Vector2f &uv) {
        vertex_uv_ = uv;
        has_uv_ = slices(uv) > 0;
        is_ready_ = false;
    }

    const Vector3i &face_indices() const { return face_vertex_indices_; }
    void set_face_indices(const Vector3i &indices) {
        face_vertex_indices_ = indices;
        face_count_ = static_cast<int>(slices(indices));
        optix_face_buffer_dirty_ = true;
        is_ready_ = false;
    }

    const Vector3i &face_uv_indices() const { return face_uv_indices_; }
    void set_face_uv_indices(const Vector3i &indices) {
        face_uv_indices_ = indices;
        is_ready_ = false;
    }

    const VectoriT<5, true> &edge_indices() const { return edge_indices_; }
    const SecondaryEdgeInfo *secondary_edge_info() const;
    const TriangleInfo *triangle_info() const { return triangle_info_.get(); }
    const TriangleUV *triangle_uv() const { return triangle_uv_.get(); }

    const FloatDetached &vertex_buffer() const { return optix_vertex_buffer_; }
    const IntDetached &face_buffer() const { return optix_face_buffer_; }

    std::string to_string() const;

private:
    void update_world_triangle_info();
    void update_secondary_edge_info();
    void update_optix_vertex_buffer();
    void ensure_optix_face_buffer_ready();
    void ensure_world_positions_ready() const;
    void ensure_secondary_edge_info_ready() const;

    int mesh_id_ = -1;
    bool is_ready_ = false;
    bool use_face_normals_ = false;
    bool has_uv_ = false;
    bool edges_enabled_ = true;

    Matrix4f object_to_world_ = drjit::identity<Matrix4f>();
    Matrix4f left_transform_ = drjit::identity<Matrix4f>();
    Matrix4f right_transform_ = drjit::identity<Matrix4f>();

    int vertex_count_ = 0;
    int face_count_ = 0;

    Vector3f vertex_positions_object_;
    Vector3f vertex_normals_object_;
    Vector2f vertex_uv_;
    mutable Vector3f vertex_positions_world_;
    Vector3i face_vertex_indices_;
    Vector3i face_uv_indices_;
    VectoriT<5, true> edge_indices_;

    std::unique_ptr<TriangleInfo> triangle_info_object_;
    std::unique_ptr<TriangleInfo> triangle_info_;
    std::unique_ptr<TriangleUV> triangle_uv_;
    mutable std::unique_ptr<SecondaryEdgeInfo> secondary_edge_info_;

    mutable bool world_positions_dirty_ = true;
    mutable bool secondary_edge_info_dirty_ = true;
    bool optix_face_buffer_dirty_ = true;

    FloatDetached optix_vertex_buffer_;
    IntDetached optix_face_buffer_;
};

} // namespace raydi
