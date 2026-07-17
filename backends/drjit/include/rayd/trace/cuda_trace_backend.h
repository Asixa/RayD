#pragma once

#include <vector>

#include <rayd/rayd.h>
#include <rayd/ray.h>
#include <rayd/types.h>
#include <rayd/scene/scene_optix.h>
#include <rayd/trace/trace_backend.h>

namespace rayd {

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
    void build(const TriangleInfo &triangles, const Int &shape_id, const Int &local_prim_id);
    /// Refit node bounds in place after vertex/transform edits (topology kept).
    void sync(const TriangleInfo &triangles, const Int &shape_id, const Int &local_prim_id);

    /// Closest-hit broad phase. Mirrors OptixScene::intersect: clears the missed
    /// lanes of \p active and returns the detached winner (t, barycentric,
    /// shape_id, local_prim_id) that the caller re-gathers AD geometry on top of.
    template <bool Detached>
    OptixIntersection intersect(const RayT<Detached> &ray, MaskT<Detached> &active) const;

    /// Any-hit occlusion test: per-lane mask, true where the ray hits any
    /// surface within tmax. Mirrors OptixScene::shadow_test.
    template <bool Detached>
    MaskT<Detached> shadow_test(const RayT<Detached> &ray, MaskT<Detached> active) const;

    /// P3 test hook (no public API yet): closest blocker global primitive id per
    /// ray with an optional per-ray ignore list, returned eagerly to the host.
    std::vector<int> first_blocker_selftest(const Vector3f &origin,
                                            const Vector3f &direction,
                                            const Float &tmax,
                                            const std::vector<int> &ignore_prim_ids) const;

private:
    void build_or_refit(const TriangleInfo &triangles,
                        const Int &shape_id,
                        const Int &local_prim_id,
                        bool refit);
    template <bool Detached>
    OptixIntersection intersect_impl(const RayT<Detached> &ray, MaskT<Detached> &active) const;
    template <bool Detached>
    MaskT<Detached> shadow_test_impl(const RayT<Detached> &ray, MaskT<Detached> active) const;

    bool ready_ = false;
    int primitive_count_ = 0;
    int node_count_ = 0;

    // World-space triangle geometry (edge-vector form) and scene id maps.
    Vector3f tri_p0_;
    Vector3f tri_e1_;
    Vector3f tri_e2_;
    Int shape_id_;
    Int local_prim_id_;

    // Compacted preorder BVH (see shared/bvh/topology.h CompactBvhTopologyView).
    Vector3f node_bbox_min_;
    Vector3f node_bbox_max_;
    Int left_child_;
    Int right_child_;
    Int leaf_primitives_;

    // Refit metadata.
    Int primitive_leaf_node_;
    Int leaf_nodes_;
    Int refit_level_nodes_;              ///< Internal nodes concatenated by ascending height.
    std::vector<int> refit_level_offsets_;
};

} // namespace rayd
