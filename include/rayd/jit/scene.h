// Copyright Xingyu Chen.
// Declares the Dr.Jit scene API and triangle trace backends.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <rayd/jit/core.h>
#include <rayd/jit/diffraction.h>
#include <rayd/jit/edge.h>
#include <rayd/jit/mesh.h>
#include <rayd/jit/optix.h>
#include <rayd/jit/reflection.h>
#include <rayd/jit/visibility.h>

namespace rayd::shared::rt {

// Backend-neutral trace-backend descriptors (RAY_TRACING_BACKEND_ARCHITECTURE.md
// §5, §12). These are host-safe enums/POD only: they name backends and report
// capabilities, but include no CUDA/OptiX/Embree headers and use no device
// qualifiers, so a third backend can include this header cleanly.

// Which triangle traversal backend a scene resolves to. `Auto` is a request that
// is resolved at construction to a concrete kind; `None` means no triangle trace
// backend was built (edge-only scenes, or a machine without the OptiX driver).
enum class TraceBackendKind : std::uint8_t { Auto, Optix, Cuda, Embree, None };

// The integration axis (§4.2): whether a backend folds into a Dr.Jit symbolic
// megakernel or runs as an eager native batch dispatch.
enum class IntegrationMode : std::uint8_t { JitSymbolic, EagerNative };

// Static capability report for a trace backend. All flags default to false so a
// backend only advertises what it actually supports.
struct TraceCapabilities {
    bool closest_hit = false;
    bool any_hit = false;
    bool first_blocker = false;
    bool ignore_primitives = false;
    bool instancing = false;
    bool refit = false;
    bool compaction = false;
    bool device_callable = false; ///< Traverser can inline the trace on-device.
    bool jit_symbolic = false;    ///< Can fold into a Dr.Jit megakernel (axis two).
    bool fused_multipath = false;
    bool cpu = false;
};

} // namespace rayd::shared::rt

namespace rayd {

// Bring the backend-neutral descriptors into the rayd namespace so the Dr.Jit
// frontend (Scene, bindings) can refer to them without the shared::rt qualifier.
using shared::rt::IntegrationMode;
using shared::rt::TraceBackendKind;
using shared::rt::TraceCapabilities;

/// \brief Host-side lifecycle interface for a triangle trace backend.
///
/// This interface is deliberately narrow: it covers backend lifecycle and
/// capability introspection only. Per-ray batch trace methods are an eager-axis
/// concern that arrives in a later phase (RAY_TRACING_BACKEND_ARCHITECTURE.md §5)
/// and are intentionally absent here, so no virtual call can land in a per-ray
/// hot loop. Concrete backends (today only OptixTraceBackend) expose their own
/// typed build/sync/trace entry points.
class TraceBackend {
  public:
    virtual ~TraceBackend() = default;

    /// Concrete backend kind (never Auto or None for a live backend).
    virtual TraceBackendKind kind() const = 0;
    /// Static capability report for this backend.
    virtual TraceCapabilities capabilities() const = 0;
    /// True once the backend's acceleration structures are built and refit-clean.
    virtual bool is_ready() const = 0;
};

} // namespace rayd

namespace rayd {

struct OptixState;
class Mesh;

/// Raw closest-hit results from an OptiX trace (detached; the scene re-gathers AD data on top).
struct OptixIntersection {
    /// (Re)allocate the result buffers to hold \p size lanes; a no-op if already sized.
    void reserve(int64_t size);

    int64_t m_size = 0;   ///< Allocated lane count.
    Int shape_id;         ///< Owning mesh id per ray; -1 when no hit.
    Int local_prim_id;    ///< Face index within the owning mesh; -1 when no hit.
    Vector2f barycentric; ///< Barycentric (u, v) of the hit.
    Float t;              ///< Hit distance; Infinity when no hit.
};

/// Result of a single segment occlusion test.
struct OptixSegmentHit {
    Mask visible;       ///< True when the segment endpoints are mutually unoccluded.
    Mask hit;           ///< True when an occluder was found along the segment.
    Int global_prim_id; ///< Scene-global face index of the occluder; -1 when none.
};

/// Describes one mesh as input to an OptiX GAS/IAS build.
struct OptixSceneMeshDesc {
    const Mesh* mesh = nullptr; ///< Source mesh (not owned).
    bool dynamic = false;       ///< Whether the GAS is built for in-place refit.
    int face_offset = 0;        ///< Offset added to local primitive ids to globalize them.
    int mesh_id = -1;           ///< Instance id / SBT record index.
};

/// Marks which aspects of a mesh changed, driving an incremental OptixScene::sync().
struct OptixSceneMeshUpdate {
    int mesh_id = -1;
    bool vertices_dirty = false;  ///< Vertices moved: refit the GAS.
    bool transform_dirty = false; ///< Transform changed: rewrite the IAS instance.
};

/// Timing breakdown of the most recent OptixScene::sync().
struct OptixSyncProfile {
    double gas_update_ms = 0.0; ///< Time spent refitting per-mesh GAS structures.
    double ias_update_ms = 0.0; ///< Time spent rebuilding the top-level IAS.
    double total_ms = 0.0;
    int updated_vertex_meshes = 0;
    int updated_transform_meshes = 0;
};

/// OptiX acceleration structure (per-mesh GAS under a single IAS) and the trace entry points over it.
class OptixScene {
  public:
    OptixScene();
    ~OptixScene();

    /// \brief Build the GAS/IAS for \p meshes.
    ///
    /// \param meshes        Meshes to accelerate, each carrying its id and dynamic flag.
    /// \param trace_source  When non-null, reuse that scene's pipeline and SBT handles
    ///                      instead of creating new ones (used by split static/dynamic scenes).
    void build(const std::vector<OptixSceneMeshDesc>& meshes, const OptixScene* trace_source = nullptr);
    /// Apply incremental \p updates by refitting affected GAS structures and rebuilding the IAS.
    void sync(const std::vector<OptixSceneMeshDesc>& meshes, const std::vector<OptixSceneMeshUpdate>& updates);
    bool is_ready() const;
    const OptixSyncProfile& last_sync_profile() const { return last_sync_profile_; }
    OptixDeviceContext context() const;
    /// Traversable handle of the top-level instance acceleration structure.
    OptixTraversableHandle ias_handle() const;

    /// \brief Closest-hit trace; clears \p active lanes that miss and returns their raw hit data.
    /// \tparam Detached  When true, operate on detached (non-AD) arrays.
    template <bool Detached> OptixIntersection intersect(const RayT<Detached>& ray, MaskT<Detached>& active) const;
    /// Any-hit occlusion test: per-lane mask, true where the ray hits any surface within tmax.
    template <bool Detached> MaskT<Detached> shadow_test(const RayT<Detached>& ray, MaskT<Detached> active) const;
    /// Occlusion test along the finite segment [start, end] with self-intersection epsilons applied.
    template <bool Detached>
    OptixSegmentHit segment_hit(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                                MaskT<Detached> active) const;

  private:
    OptixState* m_accel = nullptr;
    OptixSyncProfile last_sync_profile_;
};

} // namespace rayd

// Host-callable CUDA orchestration entry points for the pure-CUDA triangle BVH
// backend. These mirror the edge BVH's edge_bvh.h free functions: they receive
// evaluated Dr.Jit device pointers, create their own non-blocking CUDA streams
// and RAII scratch, drive the shared BVH build/refit/traversal kernels, record
// native launch-audit hooks, and synchronize before returning. They allocate no
// persistent state and never touch Dr.Jit.

namespace rayd {

/// Read-only SoA world-space triangle geometry pointers (edge-vector form).
struct TriBvhTrianglePtrs {
    const float* p0_x;
    const float* p0_y;
    const float* p0_z;
    const float* e1_x;
    const float* e1_y;
    const float* e1_z;
    const float* e2_x;
    const float* e2_y;
    const float* e2_z;
};

/// Mutable per-node/per-primitive AABB SoA pointers (build/refit outputs).
struct TriBvhBoundsPtrs {
    float* min_x;
    float* min_y;
    float* min_z;
    float* max_x;
    float* max_y;
    float* max_z;
};

/// Read-only per-node AABB SoA pointers (query inputs).
struct TriBvhConstBoundsPtrs {
    const float* min_x;
    const float* min_y;
    const float* min_z;
    const float* max_x;
    const float* max_y;
    const float* max_z;
};

/// Read-only ray batch pointers. `t_max` is already remapped; `active` is one
/// int per ray (null means all active).
struct TriBvhRayPtrs {
    const float* origin_x;
    const float* origin_y;
    const float* origin_z;
    const float* dir_x;
    const float* dir_y;
    const float* dir_z;
    const float* t_max;
    const int* active;
};

/// Build a scene-level triangle LBVH (Morton/radix/finalize) into the caller's
/// raw (2N-1)-node topology and bounds buffers. Pure LBVH: no treelet pass.
void build_triangle_bvh_gpu(int primitive_count, TriBvhTrianglePtrs triangles, TriBvhBoundsPtrs primitive_bounds,
                            TriBvhBoundsPtrs node_bounds, int* left_child, int* right_child, int* leaf_primitive,
                            int* is_leaf, int* primitive_leaf_node);

/// Refit the compacted BVH node bounds in place after the triangles moved
/// (topology unchanged): recompute leaf-node bounds, then refit internal nodes
/// level by level in ascending height order.
void refit_triangle_bvh_gpu(int node_count, TriBvhTrianglePtrs triangles, const int* left_child, const int* right_child,
                            const int* leaf_primitives, const int* leaf_nodes, int leaf_node_count,
                            const int* level_nodes, const int* level_offsets, int level_count,
                            TriBvhBoundsPtrs node_bounds);

/// Closest-hit query over the compacted BVH.
void query_triangle_closest_hit_gpu(int ray_count, int primitive_count, int node_count, int leaf_primitive_count,
                                    float t_min, TriBvhTrianglePtrs triangles, TriBvhConstBoundsPtrs node_bounds,
                                    const int* left_child, const int* right_child, const int* leaf_primitives,
                                    TriBvhRayPtrs rays, const int* shape_id, const int* local_prim_id, float* out_t,
                                    float* out_bary_u, float* out_bary_v, int* out_shape_id, int* out_local_prim_id,
                                    int* stack_nodes, int* overflow);

/// Any-hit occlusion query over the compacted BVH.
void query_triangle_occluded_gpu(int ray_count, int primitive_count, int node_count, int leaf_primitive_count,
                                 float t_min, TriBvhTrianglePtrs triangles, TriBvhConstBoundsPtrs node_bounds,
                                 const int* left_child, const int* right_child, const int* leaf_primitives,
                                 TriBvhRayPtrs rays, int* out_hit, int* stack_nodes, int* overflow);

/// Closest-blocker query honoring a per-ray ignore list.
void query_triangle_first_blocker_gpu(int ray_count, int primitive_count, int node_count, int leaf_primitive_count,
                                      float t_min, TriBvhTrianglePtrs triangles, TriBvhConstBoundsPtrs node_bounds,
                                      const int* left_child, const int* right_child, const int* leaf_primitives,
                                      TriBvhRayPtrs rays, const int* ignore_prim_ids, int ignore_stride,
                                      int* out_global_prim_id, int* stack_nodes, int* overflow);

} // namespace rayd

namespace rayd {

// Forward declarations keep CUDA and OptiX launch details out of this widely
// included host header and prevent device-only types from leaking into host TUs.
struct AccumParams;
struct DfrPathParams;
struct DfrAccumParams;
struct CudaMultipathBvh;
enum class CudaSegmentVisibilityVariant : int;
namespace shared::optix {
struct ReflectionTraceParams;
struct SegmentVisibilityParams;
struct ReflEpcParams;
} // namespace shared::optix

/// \brief Pure-CUDA triangle trace backend (eager native axis).
///
/// Owns a single scene-level LBVH over the scene's world-space triangles
/// (`Scene::triangle_info_detached_`; transforms already baked, indexed by
/// global primitive id) and answers closest-hit / occlusion / first-blocker
/// queries with raw CUDA kernels. There is no BLAS/TLAS: the detached triangle
/// arrays are already in world space, so a single acceleration structure over
/// them is sufficient (unlike OptiX, which stores object-space GAS and needs an
/// IAS). The backend is deliberately outside any Dr.Jit symbolic recording: its
/// queries evaluate eagerly and cannot be captured into a megakernel.
class CudaTraceBackend final : public TraceBackend {
  public:
    CudaTraceBackend();
    ~CudaTraceBackend() override;

    TraceBackendKind kind() const override { return TraceBackendKind::Cuda; }
    TraceCapabilities capabilities() const override;
    bool is_ready() const override { return ready_; }

    /// Build the scene-level triangle BVH. `triangles` is the detached
    /// world-space edge-vector geometry indexed by global primitive id;
    /// `shape_id`/`local_prim_id` are the parallel scene id maps.
    void build(const TriangleInfo& triangles, const Int& shape_id, const Int& local_prim_id);
    /// Refit node bounds in place after vertex/transform edits (topology kept).
    void sync(const TriangleInfo& triangles, const Int& shape_id, const Int& local_prim_id);

    /// Closest-hit broad phase. Mirrors OptixScene::intersect: clears the missed
    /// lanes of \p active and returns the detached winner (t, barycentric,
    /// shape_id, local_prim_id) that the caller re-gathers AD geometry on top of.
    template <bool Detached> OptixIntersection intersect(const RayT<Detached>& ray, MaskT<Detached>& active) const;

    /// Any-hit occlusion test: per-lane mask, true where the ray hits any
    /// surface within tmax. Mirrors OptixScene::shadow_test.
    template <bool Detached> MaskT<Detached> shadow_test(const RayT<Detached>& ray, MaskT<Detached> active) const;

    /// P3 test hook (no public API yet): closest blocker global primitive id per
    /// ray with an optional per-ray ignore list, returned eagerly to the host.
    std::vector<int> first_blocker_selftest(const Vector3f& origin, const Vector3f& direction, const Float& tmax,
                                            const std::vector<int>& ignore_prim_ids) const;

    // -- CUDA fused multipath execution -----------------------------------------
    // Each entry mirrors the native OptiX launch contract, applies the
    // single-scene CUDA convention, materializes the BVH buffers, drains the
    // Dr.Jit stream, and launches the pure-CUDA kernel over the scene triangle
    // BVH. The parameters follow the same pointer layout as the OptiX path.
    // `params` is taken by value so CUDA-scene overrides remain local to the
    // selected backend.

    /// scene.trace_reflections(..., symbolic=True) native path.
    void run_reflection_trace(shared::optix::ReflectionTraceParams params, int lane_count) const;

    /// scene.visible / visible_pair / visible_edge / visible_chain native arm.
    void run_segment_visibility(shared::optix::SegmentVisibilityParams params, CudaSegmentVisibilityVariant variant,
                                int lane_count) const;

    /// scene.accumulate_reflections native path.
    void run_reflection_accumulation(AccumParams params, int lane_count) const;

    /// scene.trace_refl_epc / trace_refl_epc_field discovery native path.
    /// `direct_only` / `primary_visibility_only` mirror the OptiX raygen variants.
    void run_reflection_epc(shared::optix::ReflEpcParams params, bool direct_only, bool primary_visibility_only,
                            int lane_count) const;

    /// scene.trace_dfr_paths native path (single-scene two-phase export).
    void run_dfr_paths(DfrPathParams params, int lane_count) const;

    /// scene.accum_dfr_direct native path (single-scene staged: source-visibility
    /// prepass then no-suffix target and/or suffix-first + suffix-target phases).
    void run_dfr_accum_direct(DfrAccumParams params, bool has_non_suffix_strategy, bool has_suffix_strategy,
                              int lane_count) const;

    /// scene.accum_dfr_coherent_direct native path (single primary-only launch).
    void run_dfr_accum_coherent(DfrAccumParams params, int lane_count) const;

    /// scene.accum_dfr chain native path (order 2/3, single primary-only launch).
    void run_dfr_accum_chain(DfrAccumParams params, int lane_count) const;

    /// Combined 5-bool order-1 accumulation body; defensive split-scene arm (the
    /// single-scene CUDA path always uses run_dfr_accum_direct's staged phases).
    void run_dfr_accum_combined(DfrAccumParams params, bool has_non_suffix_strategy, bool has_suffix_strategy,
                                int lane_count) const;

  private:
    /// Gather this backend's persistent BVH buffers into a raw-pointer view for
    /// the fused kernels. Caller must have materialized the buffers first.
    CudaMultipathBvh multipath_bvh() const;
    /// Materialize the persistent BVH buffers and drain the Dr.Jit stream so the
    /// fused kernel (on its own stream) sees consistent inputs (b7f7226 protocol).
    void materialize_for_fused_launch() const;

    void build_or_refit(const TriangleInfo& triangles, const Int& shape_id, const Int& local_prim_id, bool refit);
    template <bool Detached> OptixIntersection intersect_impl(const RayT<Detached>& ray, MaskT<Detached>& active) const;
    template <bool Detached> MaskT<Detached> shadow_test_impl(const RayT<Detached>& ray, MaskT<Detached> active) const;

    bool ready_ = false;
    int primitive_count_ = 0;
    int node_count_ = 0;

    // World-space triangle geometry (edge-vector form) and scene id maps.
    Vector3f tri_p0_;
    Vector3f tri_e1_;
    Vector3f tri_e2_;
    Int shape_id_;
    Int local_prim_id_;

    // Compacted preorder BVH used by the pure-CUDA trace backend.
    Vector3f node_bbox_min_;
    Vector3f node_bbox_max_;
    Int left_child_;
    Int right_child_;
    Int leaf_primitives_;

    // Refit metadata.
    Int primitive_leaf_node_;
    Int leaf_nodes_;
    Int refit_level_nodes_; ///< Internal nodes concatenated by ascending height.
    std::vector<int> refit_level_offsets_;
};

} // namespace rayd

namespace rayd {

/// Which OptiX scene(s) a query should traverse, plus the SBT hit-group count.
/// In split mode a query traces the static and dynamic scenes separately and
/// merges the winners; otherwise it traces the single unified scene.
struct OptixSceneSelection {
    const OptixScene* primary = nullptr;   ///< Unified scene, or the static scene in split mode.
    const OptixScene* secondary = nullptr; ///< Dynamic scene in split mode; null otherwise.
    int split_mode = 0;                    ///< 1 when the static/dynamic split is active.
    int hitgroup_record_count = 0;         ///< SBT hit-group record count (scene mesh count).
};

/// Aggregate OptiX GAS/IAS timing returned from a sync(), summed across scenes.
struct OptixTraceSyncResult {
    double gas_update_ms = 0.0; ///< Total per-mesh GAS refit time.
    double ias_update_ms = 0.0; ///< Total top-level IAS rebuild time.
};

/// \brief OptiX triangle trace backend owning the GAS/IAS scene(s) and
/// static/dynamic split bookkeeping.
///
/// Build and sync preserve the scene's OptiX ordering and update behavior.
/// Both operations retain the existing acceleration-structure contracts.
class OptixTraceBackend final : public TraceBackend {
  public:
    OptixTraceBackend();
    ~OptixTraceBackend() override;

    TraceBackendKind kind() const override { return TraceBackendKind::Optix; }
    TraceCapabilities capabilities() const override;
    bool is_ready() const override;

    /// \brief Build the OptiX acceleration structure(s) for \p meshes.
    /// \param meshes         Full ordered mesh descriptor list (SBT order).
    /// \param dynamic_flags  Per-mesh dynamic flag, parallel to \p meshes; drives
    ///                       the static/dynamic split decision.
    void build(const std::vector<OptixSceneMeshDesc>& meshes, const std::vector<bool>& dynamic_flags);

    /// Apply incremental \p updates; returns the aggregate GAS/IAS timing.
    OptixTraceSyncResult sync(const std::vector<OptixSceneMeshDesc>& meshes,
                              const std::vector<OptixSceneMeshUpdate>& updates);

    /// True when the static/dynamic split is active for this build.
    bool split_active() const { return split_active_; }
    /// The scene(s) a query should traverse (see OptixSceneSelection).
    OptixSceneSelection select_scenes() const;

    /// Unified scene (also the split-mode trace source that owns the pipeline/SBT).
    OptixScene& primary() const { return *scene_; }
    /// Static-only scene (split mode).
    OptixScene& static_scene() const { return *static_scene_; }
    /// Dynamic-only scene (split mode).
    OptixScene& dynamic_scene() const { return *dynamic_scene_; }

  private:
    std::unique_ptr<OptixScene> scene_;
    std::unique_ptr<OptixScene> static_scene_;
    std::unique_ptr<OptixScene> dynamic_scene_;
    bool split_active_ = false;
    int hitgroup_record_count_ = 0;
    std::vector<int> static_mesh_indices_;
    std::vector<int> dynamic_mesh_indices_;
    std::vector<int> dynamic_mesh_local_index_;
};

} // namespace rayd

namespace rayd {

class OptixLaunchPipeline;

/// Timing and counts for the most recent Scene::sync(); all times are milliseconds.
struct SceneSyncProfile {
    double mesh_update_ms = 0.0;      ///< Uploading updated mesh vertices.
    double triangle_scatter_ms = 0.0; ///< Scattering triangles into scene-global buffers.
    double triangle_eval_ms = 0.0;    ///< Evaluating the scattered triangle arrays.
    double edge_scatter_ms = 0.0;     ///< Scattering edge data into scene-global buffers.
    double edge_refit_ms = 0.0;       ///< Refitting the edge BVH.
    double optix_sync_ms = 0.0;       ///< Total OptiX GAS/IAS update time.
    double total_ms = 0.0;            ///< Wall-clock time for the whole sync().
    double optix_gas_update_ms = 0.0; ///< OptiX per-mesh GAS refit time.
    double optix_ias_update_ms = 0.0; ///< OptiX top-level IAS rebuild time.
    int updated_meshes = 0;           ///< Meshes touched by this sync().
    int updated_vertex_meshes = 0;    ///< Meshes whose vertices changed.
    int updated_transform_meshes = 0; ///< Meshes whose transform changed.
    int updated_edge_meshes = 0;      ///< Meshes whose edge data was re-scattered.
    int updated_edges = 0;            ///< Total edges re-scattered.
};

/// Acceleration backend used for nearest-edge queries.
enum class EdgeBVHBackend {
    DrJit,      ///< Custom Dr.Jit/CUDA BVH.
    Optix,      ///< OptiX custom-AABB backend (default).
    OptixDrJit, ///< Dr.Jit point/top-k queries with OptiX ray queries.
    Hybrid [[deprecated("use EdgeBVHBackend::OptixDrJit")]] = OptixDrJit
};

/// Collection of built meshes and the acceleration data required for intersection queries.
class Scene final {
  public:
    /// \brief Construct an empty scene.
    ///
    /// \param edge_bvh_backend Nearest-edge backend. "auto" resolves to
    ///        "optix" when available and "drjit" otherwise; explicit choices
    ///        are "drjit", "optix", and "optix_drjit" (see EdgeBVHBackend).
    /// \param trace_backend Triangle trace backend: "auto" resolves to OptiX when
    ///        the driver is available and to CUDA otherwise; "optix" forces
    ///        OptiX (availability is enforced at build()), "cuda" forces the
    ///        software CUDA backend, and "none" builds no triangle backend.
    explicit Scene(const std::string& edge_bvh_backend = "auto", const std::string& trace_backend = "auto");
    ~Scene();

    /// \brief Add a copy of \p mesh to the scene and return its mesh id.
    ///
    /// Invalidates any prior build(); call build() again before querying. Mark a mesh
    /// \p dynamic to allow later vertex/transform edits via sync() without a full rebuild.
    int add_mesh(const Mesh& mesh, bool dynamic = false);
    /// Build all acceleration structures (OptiX GAS/IAS and the edge BVH); call before any query.
    void build();
    /// True once the scene is built and no acceleration structure needs a rebuild.
    bool is_ready() const;
    /// True when dynamic edits have been queued but sync() has not yet been called.
    bool has_pending_updates() const { return pending_updates_; }

    /// Queue new object-space \p positions for a dynamic mesh; applied on the next sync().
    void update_mesh_vertices(int mesh_id, const Vector3fAD& positions);
    /// Queue a transform for a dynamic mesh; \p set_left chooses the left vs. right factor. See Mesh::set_transform.
    void set_mesh_transform(int mesh_id, const Matrix4fAD& matrix, bool set_left = true);
    /// Queue a transform composed onto a dynamic mesh's existing transform; applied on the next sync().
    void append_mesh_transform(int mesh_id, const Matrix4fAD& matrix, bool append_left = true);
    /// Set the per-edge active mask used by edge queries; size must equal the scene edge count.
    void set_edge_mask(const Mask& mask);
    void set_edge_mask(const MaskAD& mask) { set_edge_mask(detach<false>(mask)); }
    /// Apply all queued dynamic edits, refitting acceleration structures in place.
    void sync();
    const SceneSyncProfile& last_sync_profile() const { return last_sync_profile_; }
    /// Summary of the scene-global edge set (counts and buffer handles).
    SceneEdgeInfo edge_info() const;
    /// Canonical name of the active edge backend ("drjit", "optix", or "optix_drjit").
    std::string edge_bvh_backend() const;
    /// Resolved triangle trace backend kind.
    TraceBackendKind trace_backend_kind() const { return triangle_kind_; }
    /// The active triangle trace backend, or null when trace_backend='none'.
    const TraceBackend* trace_backend() const { return trace_backend_.get(); }
    /// Build/traversal statistics for the edge BVH.
    SceneEdgeBVHStats edge_bvh_stats() const;
    /// Scene-global edge connectivity tables.
    const SceneEdgeTopology& edge_topology() const;
    /// Current per-edge active mask. See set_edge_mask.
    const Mask& edge_mask() const;
    /// Prefix-sum of per-mesh face counts; mesh m owns faces [offset[m], offset[m+1]).
    const Int& mesh_face_offsets() const { return face_offsets_; }
    /// Prefix-sum of per-mesh edge counts.
    const Int& mesh_edge_offsets() const { return edge_offsets_; }
    /// Prefix-sum of per-mesh vertex counts.
    const Int& mesh_vertex_offsets() const { return vertex_offsets_; }
    /// Flattened scene-global geometry (vertices, faces, normals, ids).
    const SceneGeometry& global_geometry() const;
    /// Detached scene-global triangle edge-vector data used by native kernels.
    const TriangleInfo& triangle_info_detached() const { return triangle_info_detached_; }
    /// Monotonic version counter bumped whenever geometry changes; for cache invalidation.
    uint64_t version() const { return scene_version_; }
    /// Monotonic version counter bumped whenever the edge set changes.
    uint64_t edge_version() const { return edge_version_; }
    /// The three edge ids of each queried triangle; \p global selects scene-global vs. per-mesh ids.
    VectoriT<3, true> triangle_edge_indices(const Int& prim_id, bool global = true) const;
    /// The two faces adjacent to each queried edge (second is -1 on a boundary); \p global selects the id space.
    VectoriT<2, true> edge_adjacent_faces(const Int& edge_id, bool global = true) const;

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
    IntersectionT<Detached> intersect(const RayT<Detached>& ray, MaskT<Detached> active = true,
                                      RayFlags flags = RayFlags::All) const;
    /// Trace specular reflection paths with explicit options.
    template <bool Detached>
    ReflectionChainT<Detached> trace_reflections(const RayT<Detached>& ray, int max_bounces,
                                                 const ReflectionTraceOptions& options, MaskT<Detached> active) const;
    /// Trace specular reflection paths with default options.
    template <bool Detached>
    ReflectionChainT<Detached> trace_reflections(const RayT<Detached>& ray, int max_bounces,
                                                 MaskT<Detached> active = true) const;
    /// Native accumulation of reflected field or power onto a grid.
    template <bool Detached>
    AccumResultT<Detached> accumulate_reflections(const RayT<Detached>& ray, const Vector3fT<Detached>& tx_position,
                                                  const AccumGrid& grid, const MaterialT<Detached>& material,
                                                  int max_bounces, const AccumOptions& options, MaskT<Detached> active,
                                                  const Vector3fT<Detached>& tx_polarization) const;
    /// Native direct diffraction accumulation onto a grid (non-AD fast path).
    template <bool Detached>
    DfrAccumT<Detached> accum_dfr_direct(const DfrStatesT<Detached>& states, const DfrGrid& grid,
                                         const DfrMaterialT<Detached>& material, const DfrOptions& options,
                                         MaskT<Detached> active) const;
    /// Native exact coherent first-order diffraction accumulation onto a grid.
    template <bool Detached>
    DfrCoherentAccumT<Detached> accum_dfr_coherent_direct(const DfrStatesT<Detached>& states, const DfrGrid& grid,
                                                          const DfrMaterialT<Detached>& material,
                                                          const DfrCoherentOptions& options,
                                                          MaskT<Detached> active) const;
    /// Native exact coherent first-order deterministic UTD vector accumulation onto a grid.
    template <bool Detached>
    DfrCoherentAccumT<Detached> accum_dfr_coherent_direct(const DfrCoherentUtdStatesT<Detached>& states,
                                                          const DfrGrid& grid, const DfrCoherentOptions& options,
                                                          MaskT<Detached> active) const;
    /// Build compact first-order direct-Tx deterministic UTD states.
    template <bool Detached>
    DfrCoherentUtdStatesT<Detached> build_dfr_coherent_tx_states(const DfrCoherentEdgeT<Detached>& edges,
                                                                 const Vector3fT<Detached>& tx_position,
                                                                 const DfrMaterialT<Detached>& material,
                                                                 const DfrCoherentOptions& options,
                                                                 MaskT<Detached> active) const;
    /// Build higher-order coherent edge candidate pairs from outgoing state bases.
    template <bool Detached>
    DfrCoherentCandidatePairsT<Detached> build_dfr_coherent_higher_candidates(
        const DfrCoherentUtdStatesT<Detached>& prev_states, const DfrCoherentEdgeT<Detached>& edges,
        const IntT<Detached>& global_to_local_edge_index, const DfrCoherentOptions& options,
        MaskT<Detached> active) const;
    /// Native higher-order direct-chain diffraction accumulation onto a grid (non-AD fast path).
    template <bool Detached>
    DfrAccumT<Detached> accum_dfr(const DfrStatesT<Detached>& initial_states,
                                  const DfrStatesT<Detached>& recursive_states, const DfrGrid& grid,
                                  const DfrMaterialT<Detached>& material, const DfrOptions& options,
                                  MaskT<Detached> active) const;
    /// Native compact direct diffraction path export (non-AD fast path).
    template <bool Detached>
    DfrPathsT<Detached> trace_dfr_paths(const Vector3fT<Detached>& tx_positions,
                                        const Vector3fT<Detached>& rx_positions, const DfrStatesT<Detached>& states,
                                        const DfrMaterialT<Detached>& material, const DfrPathOptions& options,
                                        MaskT<Detached> active) const;
    /// Equivalent-path-correction reflection trace toward \p receiver with default options.
    template <bool Detached>
    ReflEpcT<Detached> trace_refl_epc(const RayT<Detached>& ray, const Vector3fT<Detached>& receiver, int max_bounces,
                                      MaskT<Detached> active = true) const {
        return trace_refl_epc(ray, receiver, max_bounces, ReflEpcOptions(), active);
    }
    /// Equivalent-path-correction reflection trace toward \p receiver.
    template <bool Detached>
    ReflEpcT<Detached> trace_refl_epc(const RayT<Detached>& ray, const Vector3fT<Detached>& receiver, int max_bounces,
                                      const ReflEpcOptions& options, MaskT<Detached> active = true) const;
    /// EPC reflection trace returning accumulated field, seeded from a ray.
    template <bool Detached>
    ReflEpcFieldT<Detached> trace_refl_epc_field(const RayT<Detached>& ray, const Vector3fT<Detached>& receiver,
                                                 int max_bounces, const ReflEpcFieldOptionsT<Detached>& options,
                                                 MaskT<Detached> active = true) const;
    /// EPC reflection field trace seeded from a transmitter position rather than a ray.
    template <bool Detached>
    ReflEpcFieldT<Detached> trace_refl_epc_field(const Vector3fT<Detached>& tx_position,
                                                 const Vector3fT<Detached>& receiver, int max_bounces,
                                                 const ReflEpcFieldOptionsT<Detached>& options,
                                                 MaskT<Detached> active = true) const;
    /// Trace per-bounce reflection records with explicit options.
    template <bool Detached>
    ReflectionTraceT<Detached> trace_bounces(const RayT<Detached>& ray, int max_bounces,
                                             const ReflectionTraceOptions& options, MaskT<Detached> active) const;
    /// Trace per-bounce reflection records with default options.
    template <bool Detached>
    ReflectionTraceT<Detached> trace_bounces(const RayT<Detached>& ray, int max_bounces,
                                             MaskT<Detached> active = true) const;
    /// Any-hit occlusion test: per-lane mask, true where the ray hits any surface within tmax.
    template <bool Detached>
    MaskT<Detached> shadow_test(const RayT<Detached>& ray, MaskT<Detached> active = true) const;
    /// \brief P3 CUDA-backend test hook (not part of the public query surface):
    /// closest blocker global primitive id per ray, honoring an ignore list.
    /// Requires a scene built with trace_backend='cuda'.
    std::vector<int> cuda_first_blocker_selftest(const Vector3f& origin, const Vector3f& direction, const Float& tmax,
                                                 const std::vector<int>& ignore_prim_ids) const;
    /// \brief Mutual visibility of segment endpoints [start, end].
    /// \param ignore_prim_ids Optional per-ray list of primitive ids to treat as non-occluding.
    /// \param active          Per-lane mask; inactive lanes return invalid.
    template <bool Detached>
    SegmentVisibilityT<Detached> visible(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                                         const Int& ignore_prim_ids = Int(), MaskT<Detached> active = true) const;
    /// Visibility from \p start to two endpoints in one pass (shared origin, see visible()).
    template <bool Detached>
    SegmentPairVisibilityT<Detached> visible_pair(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end_a,
                                                  const Vector3fT<Detached>& end_b, const Int& ignore_prim_ids = Int(),
                                                  MaskT<Detached> active = true) const;
    /// \brief Whether \p src sees any sample point along an edge segment.
    ///
    /// Samples the edge at \p sample_fractions of [edge_t_min, edge_t_max] along
    /// \p edge_dir from \p edge_pos and reports whether any sample is visible.
    template <bool Detached>
    AxialEdgeVisibilityT<Detached> visible_edge(const Vector3fT<Detached>& src, const Vector3fT<Detached>& edge_pos,
                                                const Vector3fT<Detached>& edge_dir, const FloatT<Detached>& edge_t_min,
                                                const FloatT<Detached>& edge_t_max,
                                                const std::vector<float>& sample_fractions,
                                                MaskT<Detached> active = true) const;
    /// \brief Per-segment visibility along polyline chains.
    /// \param points       Flattened chain vertices, concatenated across all chains.
    /// \param chain_length Number of points in each chain (parallel to the batch).
    /// \param ignore_prim_per_segment Optional per-segment primitive ids to treat as non-occluding.
    template <bool Detached>
    SegmentChainVisibilityT<Detached> visible_chain(const Vector3fT<Detached>& points, const Int& chain_length,
                                                    const Int& ignore_prim_per_segment = Int(),
                                                    MaskT<Detached> active = true) const;
    /// Nearest scene edge to each query point (see edge BVH; multipath/edge headers).
    template <bool Detached>
    NearestPointEdgeT<Detached> nearest_edge(const Vector3fT<Detached>& point, MaskT<Detached> active = true) const;
    /// Nearest scene edge to each query ray; uses segment semantics on [0, tmax] when tmax is finite.
    template <bool Detached>
    NearestRayEdgeT<Detached> nearest_edge(const RayT<Detached>& ray, MaskT<Detached> active = true) const;
    /// The \p k nearest scene edges to each query point (k <= 16).
    template <bool Detached>
    NearestEdgesTopKT<Detached> nearest_edges(const Vector3fT<Detached>& point, int k,
                                              MaskT<Detached> active = true) const;

    int num_meshes() const { return mesh_count_; }
    /// Non-owning pointers to the scene's meshes, indexed by mesh id.
    std::vector<const Mesh*> meshes() const;

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

    // Triangle-trace-backend accessors. Each requires a live trace backend (built
    // only when trace_backend != 'none' and OptiX is available); the rest of the
    // code uses these instead of touching the backend directly, so migrating to a
    // second backend later stays localized here.
    OptixTraceBackend& optix_backend() const;
    CudaTraceBackend& cuda_backend() const;
    OptixScene& optix_scene() const;
    OptixScene& optix_static_scene() const;
    OptixScene& optix_dynamic_scene() const;
    bool optix_split_active() const;
    OptixSceneSelection select_optix_scenes() const;
    void reset_multipath_pipelines();
    void ensure_dfr_order1_accumulation_pipeline() const;
    void ensure_dfr_chain_accumulation_pipeline() const;
    SceneMeshRecord& mesh_record(int mesh_id);
    const SceneMeshRecord& mesh_record(int mesh_id) const;
    void scatter_mesh_data(const SceneMeshRecord& record, bool include_static);
    void scatter_mesh_edge_data(const SceneMeshRecord& record, bool include_static_ids);
    void ensure_scene_edge_data_ready() const;
    void ensure_edge_bvh_ready() const;
    void ensure_reflection_epc_geometry_ready() const;
    // Every buffer, acceleration structure, and OptiX resource below belongs to
    // the Dr.Jit CUDA device that ran build(); \p context names the failing
    // entry point in the error message.
    void require_build_device(const char* context) const;
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
    // Sticky once differentiable geometry enters the scene. Incremental
    // in-place scatters otherwise chain every scene-global AD buffer to its
    // predecessor and retain all prior update graphs.
    bool differentiable_geometry_active_ = false;
    bool mask_dirty_ = false;
    uint64_t scene_version_ = 0;
    uint64_t edge_version_ = 0;
    int edge_count_ = 0;
    int build_device_ = -1;
    mutable bool edge_bvh_dirty_ = false;
    mutable std::vector<EdgeDirtyRange> pending_edge_bvh_dirty_ranges_;
    TraceBackendKind triangle_kind_ = TraceBackendKind::None;
    std::unique_ptr<TraceBackend> trace_backend_;
    mutable std::shared_ptr<OptixLaunchPipeline> reflection_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> reflection_accumulation_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_accumulation_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_accumulation_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_accumulation_no_suffix_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_accumulation_no_suffix_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_accumulation_suffix_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_accumulation_suffix_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_source_visibility_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_no_suffix_target_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_suffix_first_visibility_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_order1_suffix_target_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_chain_accumulation_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_chain_accumulation_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_coherent_accumulation_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_coherent_accumulation_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_paths_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_paths_source_visibility_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_paths_target_export_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> diffraction_paths_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> reflection_epc_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> reflection_epc_direct_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> reflection_epc_direct_primary_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> segment_visibility_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> segment_pair_visibility_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> axial_edge_visibility_pipeline_;
    mutable std::shared_ptr<OptixLaunchPipeline> segment_chain_visibility_pipeline_;
    mutable bool reflection_epc_geometry_ready_ = false;
    std::unique_ptr<SceneEdge> edge_bvh_;
    std::unique_ptr<SceneEdgeOptix> edge_optix_;
    EdgeBVHBackend edge_bvh_backend_ = EdgeBVHBackend::DrJit;
    SceneSyncProfile last_sync_profile_;
};

} // namespace rayd
