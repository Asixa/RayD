#pragma once

#include <rayd/rayd.h>
#include <rayd/edge/edge.h>
#include <rayd/ray.h>
#include <rayd/edge/scene_edge.h>

#include <vector>

namespace rayd {

struct EdgeOptixState;

class SceneEdgeOptix {
public:
    SceneEdgeOptix();
    ~SceneEdgeOptix();

    SceneEdgeOptix(const SceneEdgeOptix &) = delete;
    SceneEdgeOptix &operator=(const SceneEdgeOptix &) = delete;

    void build(const SecondaryEdgeInfo &edge_info,
               const MaskDetached &mask);
    void set_mask(const MaskDetached &mask);
    void refit(const SecondaryEdgeInfo &edge_info,
               const std::vector<EdgeDirtyRange> &dirty_ranges);
    bool is_ready() const { return ready_; }
    bool has_edges() const { return primitive_count_ > 0; }
    SceneEdgeBVHStats stats() const;

    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const Vector3fT<Detached> &point,
                                      MaskT<Detached> &active) const;

    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const RayT<Detached> &ray,
                                      MaskT<Detached> &active) const;

    template <bool Detached>
    ClosestEdgeTopKCandidate nearest_edges_topk(const Vector3fT<Detached> &point,
                                                int k,
                                                MaskT<Detached> &active) const;

private:
    void build_gases(bool update);
    void ensure_pipeline();
    void refresh_geometry(const SecondaryEdgeInfo &edge_info);
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
