#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rayd/rayd.h>
#include <rayd/edge/edge.h>
#include <rayd/ray.h>
#include <rayd/mesh.h>
#include <rayd/multipath/reflection_accumulation.h>
#include <rayd/multipath/reflection_epc.h>
#include <rayd/multipath/reflection.h>
#include <rayd/multipath/segment_visibility.h>
#include <rayd/edge/scene_edge.h>
#include <rayd/edge/scene_edge_optix.h>
#include <rayd/scene/scene_optix.h>

namespace rayd {

class OptixLaunchPipeline;

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

enum class EdgeBVHBackend {
    DrJit,
    Optix,
    Hybrid
};

/// Collection of built meshes and the acceleration data required for intersection queries.
class Scene final {
public:
    explicit Scene(const std::string &edge_bvh_backend = "drjit");
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
    std::string edge_bvh_backend() const;
    SceneEdgeBVHStats edge_bvh_stats() const;
    const SceneEdgeTopology &edge_topology() const;
    const MaskDetached &edge_mask() const;
    const IntDetached &mesh_face_offsets() const { return face_offsets_; }
    const IntDetached &mesh_edge_offsets() const { return edge_offsets_; }
    const IntDetached &mesh_vertex_offsets() const { return vertex_offsets_; }
    const SceneGlobalGeometry &global_geometry() const;
    uint64_t version() const { return scene_version_; }
    uint64_t edge_version() const { return edge_version_; }
    VectoriT<3, true> triangle_edge_indices(const IntDetached &prim_id, bool global = true) const;
    VectoriT<2, true> edge_adjacent_faces(const IntDetached &edge_id, bool global = true) const;

    template <bool Detached>
    IntersectionT<Detached> intersect(const RayT<Detached> &ray,
                                       MaskT<Detached> active = true,
                                       RayFlags flags = RayFlags::All) const;
    template <bool Detached>
    ReflectionChainT<Detached> trace_reflections(const RayT<Detached> &ray,
                                                 int max_bounces,
                                                 const ReflectionTraceOptions &options,
                                                 MaskT<Detached> active) const;
    template <bool Detached>
    ReflectionChainT<Detached> trace_reflections(const RayT<Detached> &ray,
                                                 int max_bounces,
                                                 MaskT<Detached> active = true) const;
    template <bool Detached>
    ReflectionAccumulationResultT<Detached> trace_reflections_accumulating(
        const RayT<Detached> &ray,
        const Vector3fT<Detached> &tx_position,
        const ReflectionAccumulationGrid &grid,
        const PrimitiveMaterialPayloadT<Detached> &material,
        int max_bounces,
        const ReflectionAccumulationOptions &options,
        MaskT<Detached> active,
        const Vector3fT<Detached> &tx_polarization) const;
    template <bool Detached>
    ReflectionEpcResultT<Detached> trace_reflection_epc(
        const RayT<Detached> &ray,
        const Vector3fT<Detached> &receiver,
        int max_bounces,
        MaskT<Detached> active = true) const {
        return trace_reflection_epc(ray,
                                    receiver,
                                    max_bounces,
                                    ReflectionEpcOptions(),
                                    active);
    }
    template <bool Detached>
    ReflectionEpcResultT<Detached> trace_reflection_epc(
        const RayT<Detached> &ray,
        const Vector3fT<Detached> &receiver,
        int max_bounces,
        const ReflectionEpcOptions &options,
        MaskT<Detached> active = true) const;
    template <bool Detached>
    ReflectionEpcFieldResultT<Detached> trace_reflection_epc_field(
        const RayT<Detached> &ray,
        const Vector3fT<Detached> &receiver,
        int max_bounces,
        const ReflectionEpcFieldOptions &options,
        MaskT<Detached> active = true) const;
    template <bool Detached>
    ReflectionEpcFieldResultT<Detached> trace_reflection_epc_field_direct(
        const Vector3fT<Detached> &tx_position,
        const Vector3fT<Detached> &receiver,
        int max_bounces,
        const ReflectionEpcFieldOptions &options,
        MaskT<Detached> active = true) const;
    template <bool Detached>
    ReflectionTraceT<Detached> trace_bounces(
        const RayT<Detached> &ray,
        int max_bounces,
        const ReflectionTraceOptions &options,
        MaskT<Detached> active) const;
    template <bool Detached>
    ReflectionTraceT<Detached> trace_bounces(
        const RayT<Detached> &ray,
        int max_bounces,
        MaskT<Detached> active = true) const;
    template <bool Detached>
    MaskT<Detached> shadow_test(const RayT<Detached> &ray, MaskT<Detached> active = true) const;
    template <bool Detached>
    SegmentVisibilityT<Detached> trace_segment_visibility(
        const Vector3fT<Detached> &start,
        const Vector3fT<Detached> &end,
        const IntDetached &ignore_prim_ids = IntDetached(),
        MaskT<Detached> active = true) const;
    template <bool Detached>
    SegmentPairVisibilityT<Detached> trace_segment_pair_visibility(
        const Vector3fT<Detached> &start,
        const Vector3fT<Detached> &end_a,
        const Vector3fT<Detached> &end_b,
        const IntDetached &ignore_prim_ids = IntDetached(),
        MaskT<Detached> active = true) const;
    template <bool Detached>
    AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility(
        const Vector3fT<Detached> &source_pos,
        const Vector3fT<Detached> &edge_pos,
        const Vector3fT<Detached> &edge_dir,
        const FloatT<Detached> &edge_line_min,
        const FloatT<Detached> &edge_line_max,
        const std::vector<float> &sample_fractions,
        MaskT<Detached> active = true) const;
    template <bool Detached>
    SegmentChainVisibilityT<Detached> trace_segment_chain_visibility(
        const Vector3fT<Detached> &points,
        const IntDetached &chain_length,
        const IntDetached &ignore_prim_per_segment = IntDetached(),
        MaskT<Detached> active = true) const;
    template <bool Detached>
    NearestPointEdgeT<Detached> nearest_edge(const Vector3fT<Detached> &point,
                                             MaskT<Detached> active = true) const;
    template <bool Detached>
    NearestRayEdgeT<Detached> nearest_edge(const RayT<Detached> &ray,
                                           MaskT<Detached> active = true) const;
    template <bool Detached>
    NearestEdgesTopKT<Detached> nearest_edges_topk(const Vector3fT<Detached> &point,
                                                   int k,
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
        int vertex_offset = 0;
        int face_offset = 0;
        int edge_offset = 0;
    };

    struct OptixSceneSelection {
        const OptixScene *primary = nullptr;
        const OptixScene *secondary = nullptr;
        int split_mode = 0;
        int hitgroup_record_count = 0;
    };

    OptixSceneSelection select_optix_scenes() const;
    SceneMeshRecord &mesh_record(int mesh_id);
    const SceneMeshRecord &mesh_record(int mesh_id) const;
    void scatter_mesh_data(const SceneMeshRecord &record, bool include_static);
    void scatter_mesh_edge_data(const SceneMeshRecord &record, bool include_static_ids);
    void ensure_scene_edge_data_ready() const;
    void ensure_edge_bvh_ready() const;
    void ensure_reflection_epc_geometry_ready() const;
    int mesh_count_ = 0;
    std::vector<SceneMeshRecord> mesh_records_;

    IntDetached face_offsets_;
    IntDetached edge_offsets_;
    IntDetached vertex_offsets_;
    TriangleInfo triangle_info_;
    TriangleInfoDetached triangle_info_detached_;
    TriangleUV triangle_uv_;
    TriangleUVDetached triangle_uv_detached_;
    Mask triangle_face_normal_mask_;
    MaskDetached triangle_face_normal_mask_detached_;
    SceneGlobalGeometry global_geometry_;
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
    bool optix_split_active_ = false;
    std::vector<int> optix_static_mesh_indices_;
    std::vector<int> optix_dynamic_mesh_indices_;
    std::vector<int> optix_dynamic_mesh_local_index_;
    std::unique_ptr<OptixScene> optix_scene_;
    std::unique_ptr<OptixScene> optix_static_scene_;
    std::unique_ptr<OptixScene> optix_dynamic_scene_;
    mutable std::unique_ptr<OptixLaunchPipeline> reflection_pipeline_;
    mutable std::unique_ptr<OptixLaunchPipeline> reflection_accumulation_pipeline_;
    mutable std::unique_ptr<OptixLaunchPipeline> reflection_epc_pipeline_;
    mutable std::unique_ptr<OptixLaunchPipeline> segment_visibility_pipeline_;
    mutable bool reflection_epc_geometry_ready_ = false;
    std::unique_ptr<SceneEdge> edge_bvh_;
    std::unique_ptr<SceneEdgeOptix> edge_optix_;
    EdgeBVHBackend edge_bvh_backend_ = EdgeBVHBackend::DrJit;
    SceneSyncProfile last_sync_profile_;
};

} // namespace rayd

