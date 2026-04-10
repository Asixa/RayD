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

struct SceneEdgeBVHStats {
    int primitive_count = 0;
    int node_count = 0;
    int internal_node_count = 0;
    int leaf_node_count = 0;
    int max_height = 0;
    int refit_level_count = 0;
    int min_leaf_size = 0;
    int max_leaf_size = 0;
    double avg_leaf_size = 0.0;
    double root_surface_area = 0.0;
    double internal_surface_area_sum = 0.0;
    double sibling_overlap_surface_area_sum = 0.0;
    double sibling_overlap_surface_area_avg = 0.0;
    double normalized_sibling_overlap = 0.0;
    std::vector<int> leaf_size_histogram;
};

class SceneEdge {
public:
    SceneEdge() = default;
    ~SceneEdge() = default;

    void build(const SecondaryEdgeInfo &edge_info);
    void build(const SecondaryEdgeInfo &edge_info,
               const MaskDetached &mask);
    void set_mask(const MaskDetached &mask);
    void refit(const SecondaryEdgeInfo &edge_info,
               const std::vector<EdgeDirtyRange> &dirty_ranges);
    void refit(const SecondaryEdgeInfo &edge_info,
               const IntDetached &primitive_indices);
    IntDetached map_to_global(const IntDetached &bvh_ids,
                              const MaskDetached &valid) const;
    bool is_ready() const { return ready_; }
    bool has_edges() const { return primitive_count_ > 0; }
    SceneEdgeBVHStats stats() const;

    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const Vector3fT<Detached> &point,
                                      MaskT<Detached> &active) const;

    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const RayT<Detached> &ray,
                                      MaskT<Detached> &active) const;

private:
    void build_bvh(const SecondaryEdgeInfo &edge_info);
    void set_all_active_state();
    void update_active_counts_from_mask(const MaskDetached &mask);
    IntDetached refit_leaf_nodes_from_primitive_indices(const SecondaryEdgeInfo &edge_info,
                                                        const IntDetached &primitive_indices);
    void refit_internal_nodes_full();
    void refit_internal_nodes_dirty(const std::vector<IntDetached> &dirty_leaf_chunks);
    ClosestEdgeCandidate nearest_edge_point_detached(const Vector3fDetached &point,
                                                     const MaskDetached &active) const;
    ClosestEdgeCandidate nearest_edge_finite_ray_detached(const Vector3fDetached &origin,
                                                          const Vector3fDetached &segment,
                                                          const MaskDetached &active) const;
    ClosestEdgeCandidate nearest_edge_infinite_ray_detached(const Vector3fDetached &origin,
                                                            const Vector3fDetached &direction,
                                                            const MaskDetached &active) const;
    void rebuild_packed_node_layout();
    void scatter_node_bounds(const IntDetached &node_indices,
                             const Vector3fDetached &bbox_min,
                             const Vector3fDetached &bbox_max);
    IntDetached gather_node_left_child(const IntDetached &node_indices,
                                       const MaskDetached &active) const;
    IntDetached gather_node_right_child(const IntDetached &node_indices,
                                        const MaskDetached &active) const;
    IntDetached gather_node_active_count(const IntDetached &node_indices,
                                         const MaskDetached &active) const;
    Vector3fDetached gather_node_bbox_min(const IntDetached &node_indices,
                                          const MaskDetached &active) const;
    Vector3fDetached gather_node_bbox_max(const IntDetached &node_indices,
                                          const MaskDetached &active) const;

    int primitive_count_ = 0;
    int node_count_ = 0;
    bool ready_ = false;
    bool all_active_ = true;
    bool packed_node_layout_enabled_ = false;

    Vector3fDetached edge_p0_;
    Vector3fDetached edge_e1_;
    Vector3fDetached primitive_bbox_min_;
    Vector3fDetached primitive_bbox_max_;
    Vector3fDetached node_bbox_min_;
    Vector3fDetached node_bbox_max_;
    FloatDetached packed_node_bounds_;

    IntDetached left_child_;
    IntDetached right_child_;
    IntDetached packed_node_children_;
    IntDetached leaf_primitives_;
    IntDetached primitive_leaf_node_;
    IntDetached leaf_nodes_;
    IntDetached primitive_active_flags_;
    IntDetached node_active_count_;
    IntDetached node_subtree_primitive_count_;
    IntDetached node_parent_;
    IntDetached dirty_node_marks_;
    IntDetached dirty_level_nodes_;
    IntDetached dirty_level_count_;

    int active_primitive_count_ = 0;
    int full_refit_node_count_ = 0;
    std::vector<IntDetached> refit_levels_;
};

} // namespace rayd

