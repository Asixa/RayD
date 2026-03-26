#pragma once

#include <rayd/rayd.h>
#include <rayd/edge.h>
#include <rayd/ray.h>
#include <vector>

namespace rayd {

struct EdgeDirtyRange {
    int offset = 0;
    int count = 0;
};

struct ClosestEdgeCandidate {
    IntDetached global_edge_id;
    FloatDetached distance_sq;
};

class SceneEdge {
public:
    SceneEdge() = default;
    ~SceneEdge() = default;

    void configure(const SecondaryEdgeInfo &edge_info);
    void refit(const SecondaryEdgeInfo &edge_info,
               const std::vector<EdgeDirtyRange> &dirty_ranges);
    bool is_ready() const { return ready_; }
    bool has_edges() const { return primitive_count_ > 0; }

    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const Vector3fT<Detached> &point,
                                      MaskT<Detached> &active) const;

    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const RayT<Detached> &ray,
                                      MaskT<Detached> &active) const;

private:
    ClosestEdgeCandidate nearest_edge_point_detached(const Vector3fDetached &point,
                                                     const MaskDetached &active) const;
    ClosestEdgeCandidate nearest_edge_finite_ray_detached(const Vector3fDetached &origin,
                                                          const Vector3fDetached &segment,
                                                          const MaskDetached &active) const;
    ClosestEdgeCandidate nearest_edge_infinite_ray_detached(const Vector3fDetached &origin,
                                                            const Vector3fDetached &direction,
                                                            const MaskDetached &active) const;

    int primitive_count_ = 0;
    int node_count_ = 0;
    bool ready_ = false;

    Vector3fDetached edge_p0_;
    Vector3fDetached edge_e1_;
    Vector3fDetached primitive_bbox_min_;
    Vector3fDetached primitive_bbox_max_;
    Vector3fDetached node_bbox_min_;
    Vector3fDetached node_bbox_max_;

    IntDetached left_child_;
    IntDetached right_child_;
    IntDetached leaf_primitives_;
    IntDetached primitive_leaf_node_;

    std::vector<IntDetached> refit_levels_;
};

} // namespace rayd

