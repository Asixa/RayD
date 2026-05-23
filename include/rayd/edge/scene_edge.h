#pragma once

#include <rayd/rayd.h>
#include <rayd/edge/edge.h>
#include <rayd/ray.h>
#include <vector>

namespace rayd {

/// Contiguous span [offset, offset + count) of edge primitives changed since the last refit.
struct EdgeDirtyRange {
    int offset = 0;
    int count = 0;
};

/// Broad-phase winner of a nearest-edge query (detached; squared distance, scene-global id).
struct ClosestEdgeCandidate {
    Int global_edge_id;
    Float distance_sq;
};

/// Broad-phase winners of a k-nearest-edges query, laid out as query_count * k slots.
struct ClosestEdgeTopKCandidate {
    int query_count = 0;
    int k = 0;
    Mask is_valid;        ///< Whether each slot holds a valid edge.
    Int global_edge_ids;  ///< Scene-global edge id per slot.
    Float distance_sq;    ///< Squared distance per slot.
};

/// Structural and quality metrics of a built edge BVH (for diagnostics/tuning).
struct SceneEdgeBVHStats {
    int primitive_count = 0;       ///< Number of edge primitives.
    int node_count = 0;            ///< Total BVH nodes.
    int internal_node_count = 0;
    int leaf_node_count = 0;
    int max_height = 0;            ///< Maximum root-to-leaf depth.
    int refit_level_count = 0;     ///< Number of levels touched during refit.
    int min_leaf_size = 0;
    int max_leaf_size = 0;
    double avg_leaf_size = 0.0;
    double root_surface_area = 0.0;
    double internal_surface_area_sum = 0.0;
    double sibling_overlap_surface_area_sum = 0.0;
    double sibling_overlap_surface_area_avg = 0.0;
    double normalized_sibling_overlap = 0.0; ///< Sibling overlap normalized by root area (BVH quality).
    std::vector<int> leaf_size_histogram;    ///< Count of leaves by primitive count.
};

/// Custom Dr.Jit/CUDA BVH over scene-global edges; the default nearest-edge backend.
class SceneEdge {
public:
    SceneEdge() = default;
    ~SceneEdge() = default;

    /// Build the BVH over all edges in \p edge_info (all edges active).
    void build(const SecondaryEdgeInfoAD &edge_info);
    /// Build the BVH, restricting queries to edges where \p mask is true.
    void build(const SecondaryEdgeInfoAD &edge_info,
               const Mask &mask);
    /// Update the per-edge active mask without rebuilding the tree.
    void set_mask(const Mask &mask);
    /// Refit node bounds after the edges in \p dirty_ranges moved (topology unchanged).
    void refit(const SecondaryEdgeInfoAD &edge_info,
               const std::vector<EdgeDirtyRange> &dirty_ranges);
    /// Refit node bounds after the edges at \p primitive_indices moved.
    void refit(const SecondaryEdgeInfoAD &edge_info,
               const Int &primitive_indices);
    /// Force evaluation of the lazily built BVH device buffers.
    void materialize() const;
    /// Translate internal BVH primitive ids to scene-global edge ids; \p valid gates the gather.
    Int map_to_global(const Int &bvh_ids,
                              const Mask &valid) const;
    bool is_ready() const { return ready_; }
    bool has_edges() const { return primitive_count_ > 0; }
    SceneEdgeBVHStats stats() const;

    /// Nearest active edge to each query point; clears \p active lanes that find none.
    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const Vector3fT<Detached> &point,
                                      MaskT<Detached> &active) const;

    /// The \p k nearest active edges to each query point (results in query_count * k order).
    template <bool Detached>
    ClosestEdgeTopKCandidate nearest_edges(const Vector3fT<Detached> &point,
                                                int k,
                                                MaskT<Detached> &active) const;

    /// Nearest active edge to each ray; uses segment semantics on [0, tmax] when tmax is finite.
    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const RayT<Detached> &ray,
                                      MaskT<Detached> &active) const;

private:
    void build_bvh(const SecondaryEdgeInfoAD &edge_info);
    void set_all_active_state();
    void update_active_counts_from_mask(const Mask &mask);
    Int refit_leaf_nodes_from_primitive_indices(const SecondaryEdgeInfoAD &edge_info,
                                                        const Int &primitive_indices);
    void refit_internal_nodes_full();
    void refit_internal_nodes_dirty(const std::vector<Int> &dirty_leaf_chunks);
    ClosestEdgeCandidate nearest_edge_point_detached(const Vector3f &point,
                                                     const Mask &active) const;
    ClosestEdgeTopKCandidate nearest_edges_point_detached(const Vector3f &point,
                                                               int k,
                                                               const Mask &active) const;
    ClosestEdgeCandidate nearest_edge_finite_ray_detached(const Vector3f &origin,
                                                          const Vector3f &segment,
                                                          const Mask &active) const;
    ClosestEdgeCandidate nearest_edge_infinite_ray_detached(const Vector3f &origin,
                                                            const Vector3f &direction,
                                                            const Mask &active) const;
    void rebuild_packed_node_layout();
    void scatter_node_bounds(const Int &node_indices,
                             const Vector3f &bbox_min,
                             const Vector3f &bbox_max);
    Int gather_node_left_child(const Int &node_indices,
                                       const Mask &active) const;
    Int gather_node_right_child(const Int &node_indices,
                                        const Mask &active) const;
    Int gather_node_active_count(const Int &node_indices,
                                         const Mask &active) const;
    Vector3f gather_node_bbox_min(const Int &node_indices,
                                          const Mask &active) const;
    Vector3f gather_node_bbox_max(const Int &node_indices,
                                          const Mask &active) const;

    int primitive_count_ = 0;
    int node_count_ = 0;
    bool ready_ = false;
    bool all_active_ = true;
    bool packed_node_layout_enabled_ = false;

    Vector3f edge_p0_;
    Vector3f edge_e1_;
    Vector3f primitive_bbox_min_;
    Vector3f primitive_bbox_max_;
    Vector3f node_bbox_min_;
    Vector3f node_bbox_max_;
    Float packed_node_bounds_;

    Int left_child_;
    Int right_child_;
    Int packed_node_children_;
    Int leaf_primitives_;
    Int primitive_leaf_node_;
    Int leaf_nodes_;
    Int primitive_active_flags_;
    Int node_active_count_;
    Int node_subtree_primitive_count_;
    Int node_parent_;
    Int dirty_node_marks_;
    Int dirty_level_nodes_;
    Int dirty_level_count_;

    int active_primitive_count_ = 0;
    int full_refit_node_count_ = 0;
    std::vector<Int> refit_levels_;
};

} // namespace rayd

