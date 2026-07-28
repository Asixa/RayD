#pragma once

#include <vector>

#include <rayd/core/drjit.h>
#include <rayd/ray/drjit.h>
#include <rayd/core/drjit/types.h>
#include <rayd/scene/drjit/scene_optix.h>
#include <rayd/trace/drjit/trace_backend.h>

namespace rayd {

// Forward declarations for the CUDA fused multipath executor surface. The full
// definitions live in <src/scene/cuda_multipath_gpu_jit.h>, which pulls the OptiX
// launch-param structs (some include <vector_types.h>); keeping it out of this
// widely-included header avoids leaking CUDA headers into pure-host TUs (e.g. the
// nanobind module). Only cuda_trace_backend.cpp / scene_multipath.cpp include it.
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

    // -- CUDA fused multipath executor (P4 Stage D) -----------------------------
    // Each entry marshals exactly like the OptiX native launch (the caller in
    // scene_multipath.cpp assembles the params with the same .data() pointers),
    // then forces the single-scene CUDA convention (split_mode = 0, null handles
    // / a non-zero visibility sentinel), materializes the BVH buffers, drains the
    // Dr.Jit stream, and launches the pure-CUDA kernel over the scene triangle
    // BVH. `params` is taken by value; the CUDA-scene overrides are applied here.

    /// scene.trace_reflections(..., symbolic=True) native path.
    void run_reflection_trace(shared::optix::ReflectionTraceParams params, int lane_count) const;

    /// scene.visible / visible_pair / visible_edge / visible_chain native arm.
    void run_segment_visibility(shared::optix::SegmentVisibilityParams params,
                                CudaSegmentVisibilityVariant variant, int lane_count) const;

    /// scene.accumulate_reflections native path.
    void run_reflection_accumulation(AccumParams params, int lane_count) const;

    /// scene.trace_refl_epc / trace_refl_epc_field discovery native path.
    /// `direct_only` / `primary_visibility_only` mirror the OptiX raygen variants.
    void run_reflection_epc(shared::optix::ReflEpcParams params, bool direct_only,
                            bool primary_visibility_only, int lane_count) const;

    /// scene.trace_dfr_paths native path (single-scene two-phase export).
    void run_dfr_paths(DfrPathParams params, int lane_count) const;

    /// scene.accum_dfr_direct native path (single-scene staged: source-visibility
    /// prepass then no-suffix target and/or suffix-first + suffix-target phases).
    void run_dfr_accum_direct(DfrAccumParams params, bool has_non_suffix_strategy,
                              bool has_suffix_strategy, int lane_count) const;

    /// scene.accum_dfr_coherent_direct native path (single primary-only launch).
    void run_dfr_accum_coherent(DfrAccumParams params, int lane_count) const;

    /// scene.accum_dfr chain native path (order 2/3, single primary-only launch).
    void run_dfr_accum_chain(DfrAccumParams params, int lane_count) const;

    /// Combined 5-bool order-1 accumulation body; defensive split-scene arm (the
    /// single-scene CUDA path always uses run_dfr_accum_direct's staged phases).
    void run_dfr_accum_combined(DfrAccumParams params, bool has_non_suffix_strategy,
                                bool has_suffix_strategy, int lane_count) const;

private:
    /// Gather this backend's persistent BVH buffers into a raw-pointer view for
    /// the fused kernels. Caller must have materialized the buffers first.
    CudaMultipathBvh multipath_bvh() const;
    /// Materialize the persistent BVH buffers and drain the Dr.Jit stream so the
    /// fused kernel (on its own stream) sees consistent inputs (b7f7226 protocol).
    void materialize_for_fused_launch() const;

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
