#pragma once

#include <memory>

#include <rayd/jit/core.h>
#include <rayd/jit/edge.h>

namespace rayd {

/// Triangle mesh plus cached geometric data used by RayD queries.
class Mesh final {
public:
    Mesh() = default;
    Mesh(const Vector3f &vertex_positions,
         const Vector3i &face_indices,
         const Vector2f &vertex_uv = Vector2f(),
         const Vector3i &face_uv_indices = Vector3i(),
         bool verbose = false);
    Mesh(const Mesh &other);
    ~Mesh();

    /// Initialize object-space vertex, index, and optional UV data.
    void init(const Vector3f &vertex_positions,
              const Vector3i &face_indices,
              const Vector2f &vertex_uv = Vector2f(),
              const Vector3i &face_uv_indices = Vector3i(),
              bool verbose = false);

    /// Build derived geometry caches and GPU buffers from the current mesh state.
    void build();
    /// Allocate and fill the OptiX vertex/index buffers; requires build() first.
    void prepare_optix_buffers();
    /// Refresh cached world-space geometry after a vertex or transform edit, skipping a full build().
    void update_runtime_data(bool vertices_dirty, bool transform_dirty);

    // The world transform is the product full = left * object_to_world * right.
    // set_*/append_* mark derived caches dirty; full_transform() returns the product.

    /// Replace the left (\p set_left true) or right transform factor.
    void set_transform(const Matrix4fAD &matrix, bool set_left = true);
    /// Compose \p matrix onto the left (pre-multiply) or right (post-multiply) transform factor.
    void append_transform(const Matrix4fAD &matrix, bool append_left = true);
    /// Full object-to-world transform: left * object_to_world * right.
    Matrix4fAD full_transform() const { return left_transform_ * object_to_world_ * right_transform_; }

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

    const Matrix4fAD &to_world() const { return object_to_world_; }
    void set_to_world(const Matrix4fAD &matrix) {
        object_to_world_ = matrix;
        transform_identity_ = false;
        world_positions_dirty_ = true;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    const Matrix4fAD &to_world_left() const { return left_transform_; }
    void set_to_world_left(const Matrix4fAD &matrix) {
        left_transform_ = matrix;
        transform_identity_ = false;
        world_positions_dirty_ = true;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    const Matrix4fAD &to_world_right() const { return right_transform_; }
    void set_to_world_right(const Matrix4fAD &matrix) {
        right_transform_ = matrix;
        transform_identity_ = false;
        world_positions_dirty_ = true;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    const Vector3fAD &vertex_positions() const { return vertex_positions_object_; }
    void set_vertex_positions(const Vector3fAD &positions) {
        vertex_positions_object_ = positions;
        vertex_count_ = static_cast<int>(slices(positions));
        world_positions_dirty_ = true;
        secondary_edge_info_dirty_ = true;
        is_ready_ = false;
    }

    /// World-space vertex positions; recomputed lazily from the object positions and full transform.
    const Vector3fAD &vertex_positions_world() const;
    /// Object-space per-vertex shading normals (area-weighted), populated by build().
    const Vector3fAD &vertex_normals() const { return vertex_normals_object_; }

    const Vector2fAD &vertex_uv() const { return vertex_uv_; }
    void set_vertex_uv(const Vector2fAD &uv) {
        vertex_uv_ = uv;
        has_uv_ = slices(uv) > 0;
        is_ready_ = false;
    }

    const Vector3iAD &face_indices() const { return face_vertex_indices_; }
    void set_face_indices(const Vector3iAD &indices) {
        face_vertex_indices_ = indices;
        face_count_ = static_cast<int>(slices(indices));
        optix_face_buffer_dirty_ = true;
        is_ready_ = false;
    }

    const Vector3iAD &face_uv_indices() const { return face_uv_indices_; }
    void set_face_uv_indices(const Vector3iAD &indices) {
        face_uv_indices_ = indices;
        is_ready_ = false;
    }

    /// Per-edge connectivity, packed as (vertex0, vertex1, face0, face1, opposite_vertex);
    /// face1 is -1 for boundary edges.
    const VectoriT<5, true> &edge_indices() const { return edge_indices_; }
    /// World-space per-edge data used by edge queries; computed lazily.
    const SecondaryEdgeInfoAD *secondary_edge_info() const;
    /// World-space cached triangle geometry; null until build().
    const TriangleInfoAD *triangle_info() const { return triangle_info_.get(); }
    /// Per-triangle UV coordinates; null when the mesh has no UVs.
    const TriangleUVAD *triangle_uv() const { return triangle_uv_.get(); }

    /// Flat [x,y,z,...] vertex buffer in the layout OptiX GAS builds expect.
    const Float &vertex_buffer() const { return optix_vertex_buffer_; }
    /// Flat [i,j,k,...] triangle index buffer in the layout OptiX GAS builds expect.
    const Int &face_buffer() const { return optix_face_buffer_; }

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
    // True only while no transform setter/composition API has ever been used.
    // This structural flag, rather than a numerical identity comparison,
    // preserves gradients through caller-provided identity-valued matrices.
    bool transform_identity_ = true;

    Matrix4fAD object_to_world_ = drjit::identity<Matrix4fAD>();
    Matrix4fAD left_transform_ = drjit::identity<Matrix4fAD>();
    Matrix4fAD right_transform_ = drjit::identity<Matrix4fAD>();

    int vertex_count_ = 0;
    int face_count_ = 0;

    Vector3fAD vertex_positions_object_;
    Vector3fAD vertex_normals_object_;
    Vector2fAD vertex_uv_;
    mutable Vector3fAD vertex_positions_world_;
    Vector3iAD face_vertex_indices_;
    Vector3iAD face_uv_indices_;
    VectoriT<5, true> edge_indices_;

    std::unique_ptr<TriangleInfoAD> triangle_info_object_;
    std::unique_ptr<TriangleInfoAD> triangle_info_;
    std::unique_ptr<TriangleUVAD> triangle_uv_;
    mutable std::unique_ptr<SecondaryEdgeInfoAD> secondary_edge_info_;

    mutable bool world_positions_dirty_ = true;
    mutable bool secondary_edge_info_dirty_ = true;
    bool optix_face_buffer_dirty_ = true;

    Float optix_vertex_buffer_;
    Int optix_face_buffer_;
};

} // namespace rayd
