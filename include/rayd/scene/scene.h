#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <rayd/rayd.h>
#include <rayd/edge.h>
#include <rayd/intersection.h>
#include <rayd/mesh.h>
#include <rayd/scene/scene_edge.h>
#include <rayd/scene/scene_optix.h>

namespace rayd {

struct SceneSyncProfile {
    double mesh_update_ms = 0.0;
    double triangle_scatter_ms = 0.0;
    double triangle_eval_ms = 0.0;
    double edge_scatter_ms = 0.0;
    double edge_refit_ms = 0.0;
    double optix_sync_ms = 0.0;
    double total_ms = 0.0;
    double optix_gas_update_ms = 0.0;
    double optix_ias_update_ms = 0.0;
    int updated_meshes = 0;
    int updated_vertex_meshes = 0;
    int updated_transform_meshes = 0;
    int updated_edge_meshes = 0;
    int updated_edges = 0;
};

/// Collection of built meshes and the acceleration data required for intersection queries.
class Scene final {
public:
    Scene();
    ~Scene();

    int add_mesh(const Mesh &mesh, bool dynamic = false);
    void build();
    bool is_ready() const;
    bool has_pending_updates() const { return pending_updates_; }

    void update_mesh_vertices(int mesh_id, const Vector3f &positions);
    void set_mesh_transform(int mesh_id, const Matrix4f &matrix, bool set_left = true);
    void append_mesh_transform(int mesh_id, const Matrix4f &matrix, bool append_left = true);
    void set_edge_mask(const MaskDetached &mask);
    void set_edge_mask(const Mask &mask) { set_edge_mask(detach<false>(mask)); }
    void sync();
    const SceneSyncProfile &last_sync_profile() const { return last_sync_profile_; }
    SceneEdgeInfo edge_info() const;
    const SceneEdgeTopology &edge_topology() const;
    const MaskDetached &edge_mask() const;
    const IntDetached &mesh_face_offsets() const { return face_offsets_; }
    const IntDetached &mesh_edge_offsets() const { return edge_offsets_; }
    uint64_t version() const { return scene_version_; }
    uint64_t edge_version() const { return edge_version_; }
    VectoriT<3, true> triangle_edge_indices(const IntDetached &prim_id, bool global = true) const;
    VectoriT<2, true> edge_adjacent_faces(const IntDetached &edge_id, bool global = true) const;

    template <bool Detached>
    IntersectionT<Detached> intersect(const RayT<Detached> &ray,
                                       MaskT<Detached> active = true,
                                       RayFlags flags = RayFlags::All) const;
    template <bool Detached>
    MaskT<Detached> shadow_test(const RayT<Detached> &ray, MaskT<Detached> active = true) const;
    template <bool Detached>
    NearestPointEdgeT<Detached> nearest_edge(const Vector3fT<Detached> &point,
                                             MaskT<Detached> active = true) const;
    template <bool Detached>
    NearestRayEdgeT<Detached> nearest_edge(const RayT<Detached> &ray,
                                           MaskT<Detached> active = true) const;

    int num_meshes() const { return mesh_count_; }
    std::vector<const Mesh *> meshes() const;

    std::string to_string() const;

private:
    struct SceneMeshRecord {
        std::unique_ptr<Mesh> mesh;
        bool dynamic = false;
        bool vertices_dirty = false;
        bool transform_dirty = false;
        mutable bool edge_dirty = false;
        int face_offset = 0;
        int edge_offset = 0;
    };

    SceneMeshRecord &mesh_record(int mesh_id);
    const SceneMeshRecord &mesh_record(int mesh_id) const;
    void scatter_mesh_data(const SceneMeshRecord &record, bool include_static);
    void scatter_mesh_edge_data(const SceneMeshRecord &record, bool include_static_ids);
    void ensure_scene_edge_data_ready() const;
    void ensure_edge_bvh_ready() const;
    void register_primary_edge_observer(Camera *camera);
    void unregister_primary_edge_observer(Camera *camera);
    void invalidate_primary_edge_observers();

    int mesh_count_ = 0;
    std::vector<SceneMeshRecord> mesh_records_;

    IntDetached face_offsets_;
    IntDetached edge_offsets_;
    TriangleInfo triangle_info_;
    TriangleInfoDetached triangle_info_detached_;
    TriangleUV triangle_uv_;
    TriangleUVDetached triangle_uv_detached_;
    Mask triangle_face_normal_mask_;
    MaskDetached triangle_face_normal_mask_detached_;
    SecondaryEdgeInfo edge_info_;
    SceneEdgeTopology edge_topology_;
    IntDetached edge_shape_ids_;
    IntDetached edge_local_ids_;
    MaskDetached edge_mask_;
    VectoriT<3, true> triangle_edge_ids_;

    bool is_ready_ = false;
    bool pending_updates_ = false;
    bool mask_dirty_ = false;
    uint64_t scene_version_ = 0;
    uint64_t edge_version_ = 0;
    int edge_count_ = 0;
    mutable bool edge_bvh_dirty_ = false;
    mutable std::vector<EdgeDirtyRange> pending_edge_bvh_dirty_ranges_;
    std::vector<Camera *> primary_edge_observers_;
    std::unique_ptr<OptixScene> optix_scene_;
    std::unique_ptr<SceneEdge> edge_bvh_;
    SceneSyncProfile last_sync_profile_;

    friend class Camera;
};

} // namespace rayd

