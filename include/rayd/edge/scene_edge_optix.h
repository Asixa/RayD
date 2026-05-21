#pragma once

#include <rayd/rayd.h>
#include <rayd/edge/edge.h>
#include <rayd/ray.h>
#include <rayd/edge/scene_edge.h>

#include <vector>

namespace rayd {

struct EdgeOptixState;

/// Experimental OptiX edge backend: edges are custom AABB primitives traversed by
/// OptiX. Mirrors the SceneEdge query surface; selected via EdgeBVHBackend::Optix.
class SceneEdgeOptix {
public:
    SceneEdgeOptix();
    ~SceneEdgeOptix();

    SceneEdgeOptix(const SceneEdgeOptix &) = delete;
    SceneEdgeOptix &operator=(const SceneEdgeOptix &) = delete;

    /// Build the custom-AABB GAS over the edges in \p edge_info, masked by \p mask.
    void build(const SecondaryEdgeInfo &edge_info,
               const MaskDetached &mask);
    /// Update the per-edge active mask without rebuilding the GAS.
    void set_mask(const MaskDetached &mask);
    /// Refit the GAS after the edges in \p dirty_ranges moved.
    void refit(const SecondaryEdgeInfo &edge_info,
               const std::vector<EdgeDirtyRange> &dirty_ranges);
    bool is_ready() const { return ready_; }
    bool has_edges() const { return primitive_count_ > 0; }
    SceneEdgeBVHStats stats() const;

    /// Nearest active edge to each query point; clears \p active lanes that find none.
    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const Vector3fT<Detached> &point,
                                      MaskT<Detached> &active) const;

    /// Nearest active edge to each query ray.
    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const RayT<Detached> &ray,
                                      MaskT<Detached> &active) const;

    /// The \p k nearest active edges to each query point.
    template <bool Detached>
    ClosestEdgeTopKCandidate nearest_edges_topk(const Vector3fT<Detached> &point,
                                                int k,
                                                MaskT<Detached> &active) const;

private:
    void build_gases(bool update);
    void ensure_pipeline();
    void refresh_geometry(const SecondaryEdgeInfo &edge_info);
    /// Per-edge OptiX AABB inflation radius, sized to bound the nearest-edge search.
    std::vector<float> compute_search_radii(const SecondaryEdgeInfo &edge_info) const;

    EdgeOptixState *state_ = nullptr;
    int primitive_count_ = 0;
    bool ready_ = false;
    std::vector<float> search_radii_;

    Vector3fDetached edge_p0_;
    Vector3fDetached edge_e1_;
    MaskDetached edge_mask_;
};

} // namespace rayd
