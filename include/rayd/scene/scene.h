#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rayd/rayd.h>
#include <rayd/edge/edge.h>
#include <rayd/ray.h>
#include <rayd/mesh.h>
#include <rayd/multipath/diffraction_accumulation.h>
#include <rayd/multipath/diffraction_paths.h>
#include <rayd/multipath/reflection_accumulation.h>
#include <rayd/multipath/reflection_epc.h>
#include <rayd/multipath/reflection.h>
#include <rayd/multipath/segment_visibility.h>
#include <rayd/edge/scene_edge.h>
#include <rayd/edge/scene_edge_optix.h>
#include <rayd/scene/scene_optix.h>

namespace rayd {

class OptixLaunchPipeline;

/// Timing and counts for the most recent Scene::sync(); all times are milliseconds.
struct SceneSyncProfile {
    double mesh_update_ms = 0.0;        ///< Uploading updated mesh vertices.
    double triangle_scatter_ms = 0.0;  ///< Scattering triangles into scene-global buffers.
    double triangle_eval_ms = 0.0;     ///< Evaluating the scattered triangle arrays.
    double edge_scatter_ms = 0.0;      ///< Scattering edge data into scene-global buffers.
    double edge_refit_ms = 0.0;        ///< Refitting the edge BVH.
    double optix_sync_ms = 0.0;        ///< Total OptiX GAS/IAS update time.
    double total_ms = 0.0;             ///< Wall-clock time for the whole sync().
    double optix_gas_update_ms = 0.0;  ///< OptiX per-mesh GAS refit time.
    double optix_ias_update_ms = 0.0;  ///< OptiX top-level IAS rebuild time.
    int updated_meshes = 0;            ///< Meshes touched by this sync().
    int updated_vertex_meshes = 0;     ///< Meshes whose vertices changed.
    int updated_transform_meshes = 0;  ///< Meshes whose transform changed.
    int updated_edge_meshes = 0;       ///< Meshes whose edge data was re-scattered.
    int updated_edges = 0;             ///< Total edges re-scattered.
};

/// Acceleration backend used for nearest-edge queries.
enum class EdgeBVHBackend {
    DrJit,   ///< Custom Dr.Jit/CUDA BVH.
    Optix,   ///< OptiX custom-AABB backend (default).
    Hybrid   ///< Dr.Jit point queries with OptiX ray queries.
};

/// Collection of built meshes and the acceleration data required for intersection queries.
class Scene final {
public:
    /// Construct an empty scene; \p edge_bvh_backend selects the nearest-edge backend
    /// ("drjit", "optix", or "hybrid"). See EdgeBVHBackend.
    explicit Scene(const std::string &edge_bvh_backend = "optix");
    ~Scene();

    /// \brief Add a copy of \p mesh to the scene and return its mesh id.
    ///
    /// Invalidates any prior build(); call build() again before querying. Mark a mesh
    /// \p dynamic to allow later vertex/transform edits via sync() without a full rebuild.
    int add_mesh(const Mesh &mesh, bool dynamic = false);
    /// Build all acceleration structures (OptiX GAS/IAS and the edge BVH); call before any query.
    void build();
    /// True once the scene is built and no acceleration structure needs a rebuild.
    bool is_ready() const;
    /// True when dynamic edits have been queued but sync() has not yet been called.
    bool has_pending_updates() const { return pending_updates_; }

    /// Queue new object-space \p positions for a dynamic mesh; applied on the next sync().
    void update_mesh_vertices(int mesh_id, const Vector3fAD &positions);
    /// Queue a transform for a dynamic mesh; \p set_left chooses the left vs. right factor. See Mesh::set_transform.
    void set_mesh_transform(int mesh_id, const Matrix4fAD &matrix, bool set_left = true);
    /// Queue a transform composed onto a dynamic mesh's existing transform; applied on the next sync().
    void append_mesh_transform(int mesh_id, const Matrix4fAD &matrix, bool append_left = true);
    /// Set the per-edge active mask used by edge queries; size must equal the scene edge count.
    void set_edge_mask(const Mask &mask);
    void set_edge_mask(const MaskAD &mask) { set_edge_mask(detach<false>(mask)); }
    /// Apply all queued dynamic edits, refitting acceleration structures in place.
    void sync();
    const SceneSyncProfile &last_sync_profile() const { return last_sync_profile_; }
    /// Summary of the scene-global edge set (counts and buffer handles).
    SceneEdgeInfo edge_info() const;
    /// Name of the active edge backend ("drjit", "optix", or "hybrid").
    std::string edge_bvh_backend() const;
    /// Build/traversal statistics for the edge BVH.
    SceneEdgeBVHStats edge_bvh_stats() const;
    /// Scene-global edge connectivity tables.
    const SceneEdgeTopology &edge_topology() const;
    /// Current per-edge active mask. See set_edge_mask.
    const Mask &edge_mask() const;
    /// Prefix-sum of per-mesh face counts; mesh m owns faces [offset[m], offset[m+1]).
    const Int &mesh_face_offsets() const { return face_offsets_; }
    /// Prefix-sum of per-mesh edge counts.
    const Int &mesh_edge_offsets() const { return edge_offsets_; }
    /// Prefix-sum of per-mesh vertex counts.
    const Int &mesh_vertex_offsets() const { return vertex_offsets_; }
    /// Flattened scene-global geometry (vertices, faces, normals, ids).
    const SceneGeometry &global_geometry() const;
    /// Detached scene-global triangle edge-vector data used by native kernels.
    const TriangleInfo &triangle_info_detached() const { return triangle_info_detached_; }
    /// Monotonic version counter bumped whenever geometry changes; for cache invalidation.
    uint64_t version() const { return scene_version_; }
    /// Monotonic version counter bumped whenever the edge set changes.
    uint64_t edge_version() const { return edge_version_; }
    /// The three edge ids of each queried triangle; \p global selects scene-global vs. per-mesh ids.
    VectoriT<3, true> triangle_edge_indices(const Int &prim_id, bool global = true) const;
    /// The two faces adjacent to each queried edge (second is -1 on a boundary); \p global selects the id space.
    VectoriT<2, true> edge_adjacent_faces(const Int &edge_id, bool global = true) const;

    /// \brief Closest-hit ray-triangle intersection against the built scene.
    ///
    /// In the AD path (Detached == false) the broad phase runs detached through OptiX,
    /// then vertex data is re-gathered and the intersection recomputed so gradients flow
    /// through the result.
    ///
    /// \tparam Detached  When true, operate on detached (non-AD) arrays; when false,
    ///                   gradients flow through the recomputed hit fields.
    /// \param ray     RayAD batch (origin, direction, tmax).
    /// \param active  Per-lane mask; inactive lanes are skipped and returned invalid.
    /// \param flags   Selects which intersection fields are computed (see RayFlags).
    /// \return Per-ray intersection; check is_valid() before reading other fields.
    template <bool Detached>
    IntersectionT<Detached> intersect(const RayT<Detached> &ray,
                                       MaskT<Detached> active = true,
                                       RayFlags flags = RayFlags::All) const;
    /// Trace specular reflection paths with explicit options (see multipath/reflection.h).
    template <bool Detached>
    ReflectionChainT<Detached> trace_reflections(const RayT<Detached> &ray,
                                                 int max_bounces,
                                                 const ReflectionTraceOptions &options,
                                                 MaskT<Detached> active) const;
    /// Trace specular reflection paths with default options.
    template <bool Detached>
    ReflectionChainT<Detached> trace_reflections(const RayT<Detached> &ray,
                                                 int max_bounces,
                                                 MaskT<Detached> active = true) const;
    /// Native accumulation of reflected field/power onto a grid (non-AD fast path; see multipath/reflection_accumulation.h).
    template <bool Detached>
    AccumResultT<Detached> accumulate_reflections(
        const RayT<Detached> &ray,
        const Vector3fT<Detached> &tx_position,
        const AccumGrid &grid,
        const MaterialT<Detached> &material,
        int max_bounces,
        const AccumOptions &options,
        MaskT<Detached> active,
        const Vector3fT<Detached> &tx_polarization) const;
    /// Native direct diffraction accumulation onto a grid (non-AD fast path).
    template <bool Detached>
    DfrAccumT<Detached> accum_dfr_direct(
        const DfrStatesT<Detached> &states,
        const DfrGrid &grid,
        const DfrMaterialT<Detached> &material,
        const DfrOptions &options,
        MaskT<Detached> active) const;
    /// Native exact coherent first-order diffraction accumulation onto a grid.
    template <bool Detached>
    DfrCoherentAccumT<Detached> accum_dfr_coherent_direct(
        const DfrStatesT<Detached> &states,
        const DfrGrid &grid,
        const DfrMaterialT<Detached> &material,
        const DfrCoherentOptions &options,
        MaskT<Detached> active) const;
    /// Native exact coherent first-order deterministic UTD vector accumulation onto a grid.
    template <bool Detached>
    DfrCoherentAccumT<Detached> accum_dfr_coherent_direct(
        const DfrCoherentUtdStatesT<Detached> &states,
        const DfrGrid &grid,
        const DfrCoherentOptions &options,
        MaskT<Detached> active) const;
    /// Build compact first-order direct-Tx deterministic UTD states.
    template <bool Detached>
    DfrCoherentUtdStatesT<Detached> build_dfr_coherent_tx_states(
        const DfrCoherentEdgeT<Detached> &edges,
        const Vector3fT<Detached> &tx_position,
        const DfrMaterialT<Detached> &material,
        const DfrCoherentOptions &options,
        MaskT<Detached> active) const;
    /// Native higher-order direct-chain diffraction accumulation onto a grid (non-AD fast path).
    template <bool Detached>
    DfrAccumT<Detached> accum_dfr(
        const DfrStatesT<Detached> &initial_states,
        const DfrStatesT<Detached> &recursive_states,
        const DfrGrid &grid,
        const DfrMaterialT<Detached> &material,
        const DfrOptions &options,
        MaskT<Detached> active) const;
    /// Native compact direct diffraction path export (non-AD fast path).
    template <bool Detached>
    DfrPathsT<Detached> trace_dfr_paths(
        const Vector3fT<Detached> &tx_positions,
        const Vector3fT<Detached> &rx_positions,
        const DfrStatesT<Detached> &states,
        const DfrMaterialT<Detached> &material,
        const DfrPathOptions &options,
        MaskT<Detached> active) const;
    /// Equivalent-path-correction reflection trace toward \p receiver with default options.
    template <bool Detached>
    ReflEpcT<Detached> trace_refl_epc(
        const RayT<Detached> &ray,
        const Vector3fT<Detached> &receiver,
        int max_bounces,
        MaskT<Detached> active = true) const {
        return trace_refl_epc(ray,
                                    receiver,
                                    max_bounces,
                                    ReflEpcOptions(),
                                    active);
    }
    /// Equivalent-path-correction reflection trace toward \p receiver (see multipath/reflection_epc.h).
    template <bool Detached>
    ReflEpcT<Detached> trace_refl_epc(
        const RayT<Detached> &ray,
        const Vector3fT<Detached> &receiver,
        int max_bounces,
        const ReflEpcOptions &options,
        MaskT<Detached> active = true) const;
    /// EPC reflection trace returning accumulated field, seeded from a ray (see multipath/reflection_epc_field.h).
    template <bool Detached>
    ReflEpcFieldT<Detached> trace_refl_epc_field(
        const RayT<Detached> &ray,
        const Vector3fT<Detached> &receiver,
        int max_bounces,
        const ReflEpcFieldOptionsT<Detached> &options,
        MaskT<Detached> active = true) const;
    /// EPC reflection field trace seeded from a transmitter position rather than a ray.
    template <bool Detached>
    ReflEpcFieldT<Detached> trace_refl_epc_field(
        const Vector3fT<Detached> &tx_position,
        const Vector3fT<Detached> &receiver,
        int max_bounces,
        const ReflEpcFieldOptionsT<Detached> &options,
        MaskT<Detached> active = true) const;
    /// Trace per-bounce reflection records with explicit options (see multipath/reflection.h).
    template <bool Detached>
    ReflectionTraceT<Detached> trace_bounces(
        const RayT<Detached> &ray,
        int max_bounces,
        const ReflectionTraceOptions &options,
        MaskT<Detached> active) const;
    /// Trace per-bounce reflection records with default options.
    template <bool Detached>
    ReflectionTraceT<Detached> trace_bounces(
        const RayT<Detached> &ray,
        int max_bounces,
        MaskT<Detached> active = true) const;
    /// Any-hit occlusion test: per-lane mask, true where the ray hits any surface within tmax.
    template <bool Detached>
    MaskT<Detached> shadow_test(const RayT<Detached> &ray, MaskT<Detached> active = true) const;
    /// \brief Mutual visibility of segment endpoints [start, end].
    /// \param ignore_prim_ids Optional per-ray list of primitive ids to treat as non-occluding.
    /// \param active          Per-lane mask; inactive lanes return invalid.
    template <bool Detached>
    SegmentVisibilityT<Detached> visible(
        const Vector3fT<Detached> &start,
        const Vector3fT<Detached> &end,
        const Int &ignore_prim_ids = Int(),
        MaskT<Detached> active = true) const;
    /// Visibility from \p start to two endpoints in one pass (shared origin, see visible()).
    template <bool Detached>
    SegmentPairVisibilityT<Detached> visible_pair(
        const Vector3fT<Detached> &start,
        const Vector3fT<Detached> &end_a,
        const Vector3fT<Detached> &end_b,
        const Int &ignore_prim_ids = Int(),
        MaskT<Detached> active = true) const;
    /// \brief Whether \p src sees any sample point along an edge segment.
    ///
    /// Samples the edge at \p sample_fractions of [edge_t_min, edge_t_max] along
    /// \p edge_dir from \p edge_pos and reports whether any sample is visible.
    template <bool Detached>
    AxialEdgeVisibilityT<Detached> visible_edge(
        const Vector3fT<Detached> &src,
        const Vector3fT<Detached> &edge_pos,
        const Vector3fT<Detached> &edge_dir,
        const FloatT<Detached> &edge_t_min,
        const FloatT<Detached> &edge_t_max,
        const std::vector<float> &sample_fractions,
        MaskT<Detached> active = true) const;
    /// \brief Per-segment visibility along polyline chains.
    /// \param points       Flattened chain vertices, concatenated across all chains.
    /// \param chain_length Number of points in each chain (parallel to the batch).
    /// \param ignore_prim_per_segment Optional per-segment primitive ids to treat as non-occluding.
    template <bool Detached>
    SegmentChainVisibilityT<Detached> visible_chain(
        const Vector3fT<Detached> &points,
        const Int &chain_length,
        const Int &ignore_prim_per_segment = Int(),
        MaskT<Detached> active = true) const;
    /// Nearest scene edge to each query point (see edge BVH; multipath/edge headers).
    template <bool Detached>
    NearestPointEdgeT<Detached> nearest_edge(const Vector3fT<Detached> &point,
                                             MaskT<Detached> active = true) const;
    /// Nearest scene edge to each query ray; uses segment semantics on [0, tmax] when tmax is finite.
    template <bool Detached>
    NearestRayEdgeT<Detached> nearest_edge(const RayT<Detached> &ray,
                                           MaskT<Detached> active = true) const;
    /// The \p k nearest scene edges to each query point (k <= 16).
    template <bool Detached>
    NearestEdgesTopKT<Detached> nearest_edges(const Vector3fT<Detached> &point,
                                                   int k,
                                                   MaskT<Detached> active = true) const;

    int num_meshes() const { return mesh_count_; }
    /// Non-owning pointers to the scene's meshes, indexed by mesh id.
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
    void reset_multipath_pipelines();
    SceneMeshRecord &mesh_record(int mesh_id);
    const SceneMeshRecord &mesh_record(int mesh_id) const;
    void scatter_mesh_data(const SceneMeshRecord &record, bool include_static);
    void scatter_mesh_edge_data(const SceneMeshRecord &record, bool include_static_ids);
    void ensure_scene_edge_data_ready() const;
    void ensure_edge_bvh_ready() const;
    void ensure_reflection_epc_geometry_ready() const;
    int mesh_count_ = 0;
    std::vector<SceneMeshRecord> mesh_records_;

    Int face_offsets_;
    Int edge_offsets_;
    Int vertex_offsets_;
    TriangleInfoAD triangle_info_;
    TriangleInfo triangle_info_detached_;
    TriangleUVAD triangle_uv_;
    TriangleUV triangle_uv_detached_;
    MaskAD triangle_face_normal_mask_;
    Mask triangle_face_normal_mask_detached_;
    SceneGeometry global_geometry_;
    SecondaryEdgeInfoAD edge_info_;
    SceneEdgeTopology edge_topology_;
    Int edge_shape_ids_;
    Int edge_local_ids_;
    Mask edge_mask_;
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
    mutable std::shared_ptr<OptixLaunchPipeline> reflection_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> reflection_accumulation_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_accumulation_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_paths_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> reflection_epc_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> segment_visibility_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> segment_pair_visibility_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> axial_edge_visibility_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> segment_chain_visibility_pipeline_;
    mutable bool reflection_epc_geometry_ready_ = false;
    std::unique_ptr<SceneEdge> edge_bvh_;
    std::unique_ptr<SceneEdgeOptix> edge_optix_;
    EdgeBVHBackend edge_bvh_backend_ = EdgeBVHBackend::Optix;
    SceneSyncProfile last_sync_profile_;
};

} // namespace rayd

