#pragma once

#include <rayd/rayd.h>
#include <rayd/optix.h>
#include <vector>

namespace rayd {

struct OptixState;
class Mesh;

/// Raw closest-hit results from an OptiX trace (detached; the scene re-gathers AD data on top).
struct OptixIntersection {
    /// (Re)allocate the result buffers to hold \p size lanes; a no-op if already sized.
    void reserve(int64_t size);

    int64_t m_size = 0;             ///< Allocated lane count.
    Int shape_id;          ///< Owning mesh id per ray; -1 when no hit.
    Int global_prim_id;    ///< Scene-global face index per ray; -1 when no hit.
    Vector2f barycentric;  ///< Barycentric (u, v) of the hit.
    Float t;               ///< Hit distance; Infinity when no hit.
};

/// Result of a single segment occlusion test.
struct OptixSegmentHit {
    Mask visible;          ///< True when the segment endpoints are mutually unoccluded.
    Mask hit;              ///< True when an occluder was found along the segment.
    Int global_prim_id;    ///< Scene-global face index of the occluder; -1 when none.
};

/// Describes one mesh as input to an OptiX GAS/IAS build.
struct OptixSceneMeshDesc {
    const Mesh *mesh = nullptr;    ///< Source mesh (not owned).
    bool dynamic = false;          ///< Whether the GAS is built for in-place refit.
    int face_offset = 0;           ///< Offset added to local primitive ids to globalize them.
    int mesh_id = -1;              ///< Instance id / SBT record index.
};

/// Marks which aspects of a mesh changed, driving an incremental OptixScene::sync().
struct OptixSceneMeshUpdate {
    int mesh_id = -1;
    bool vertices_dirty = false;   ///< Vertices moved: refit the GAS.
    bool transform_dirty = false;  ///< Transform changed: rewrite the IAS instance.
};

/// Timing breakdown of the most recent OptixScene::sync().
struct OptixSyncProfile {
    double gas_update_ms = 0.0;     ///< Time spent refitting per-mesh GAS structures.
    double ias_update_ms = 0.0;     ///< Time spent rebuilding the top-level IAS.
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
    void build(const std::vector<OptixSceneMeshDesc> &meshes,
               const OptixScene *trace_source = nullptr);
    /// Apply incremental \p updates by refitting affected GAS structures and rebuilding the IAS.
    void sync(const std::vector<OptixSceneMeshDesc> &meshes,
              const std::vector<OptixSceneMeshUpdate> &updates);
    bool is_ready() const;
    const OptixSyncProfile &last_sync_profile() const { return last_sync_profile_; }
    OptixDeviceContext context() const;
    /// Traversable handle of the top-level instance acceleration structure.
    OptixTraversableHandle ias_handle() const;

    /// \brief Closest-hit trace; clears \p active lanes that miss and returns their raw hit data.
    /// \tparam Detached  When true, operate on detached (non-AD) arrays.
    template <bool Detached>
    OptixIntersection intersect(const RayT<Detached> &ray,
                                MaskT<Detached> &active) const;
    /// Any-hit occlusion test: per-lane mask, true where the ray hits any surface within tmax.
    template <bool Detached>
    MaskT<Detached> shadow_test(const RayT<Detached> &ray,
                                MaskT<Detached> active) const;
    /// Occlusion test along the finite segment [start, end] with self-intersection epsilons applied.
    template <bool Detached>
    OptixSegmentHit segment_hit(const Vector3fT<Detached> &start,
                                const Vector3fT<Detached> &end,
                                MaskT<Detached> active) const;

private:
    OptixState *m_accel = nullptr;
    OptixSyncProfile last_sync_profile_;
};

} // namespace rayd
