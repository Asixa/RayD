// Copyright Xingyu Chen.
// Declares the Dr.Jit edge-query API and its native launch contracts.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <rayd/contracts.h>
#include <rayd/jit/core.h>

namespace rayd {

/// Result of a nearest-edge query from a point, one entry per query.
template <typename Float_> struct NearestPointEdgeData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    /// Per-lane mask of queries that found an edge (edge_id >= 0).
    Mask_ is_valid() const { return edge_id >= 0; }

    Float_ distance = Infinity;         ///< Distance from the query point to the nearest edge.
    Vec3f point = zeros<Vec3f>(1);      ///< The query point (echoed back).
    Float_ edge_t = zeros<Float_>(1);   ///< Parameter in [0, 1] of the closest point along the edge.
    Vec3f edge_point = zeros<Vec3f>(1); ///< Closest point on the edge.
    Int_ shape_id = full<Int_>(shared::InvalidSignedId, 1);       ///< Owning mesh id; -1 when no edge found.
    Int_ edge_id = full<Int_>(shared::InvalidSignedId, 1);        ///< Edge index within the owning mesh; -1 when none.
    Int_ global_edge_id = full<Int_>(shared::InvalidSignedId, 1); ///< Scene-global edge index.
    Mask_ is_boundary = full<Mask_>(false, 1); ///< Whether the nearest edge is a boundary (open) edge.

    DRJIT_STRUCT(NearestPointEdgeData, distance, point, edge_t, edge_point, shape_id, edge_id, global_edge_id,
                 is_boundary)
};

template <bool Detached> using NearestPointEdgeT = NearestPointEdgeData<FloatT<Detached>>;

using NearestPointEdgeAD = NearestPointEdgeT<false>;
using NearestPointEdge = NearestPointEdgeT<true>;

/// Result of a nearest-edge query from a ray or segment, one entry per query.
template <typename Float_> struct NearestRayEdgeData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    /// Per-lane mask of queries that found an edge (edge_id >= 0).
    Mask_ is_valid() const { return edge_id >= 0; }

    Float_ distance = Infinity;         ///< Closest distance between the ray and the nearest edge.
    Float_ ray_t = zeros<Float_>(1);    ///< Parameter along the ray of the closest approach.
    Vec3f point = zeros<Vec3f>(1);      ///< Closest point on the ray.
    Float_ edge_t = zeros<Float_>(1);   ///< Parameter in [0, 1] of the closest point along the edge.
    Vec3f edge_point = zeros<Vec3f>(1); ///< Closest point on the edge.
    Int_ shape_id = full<Int_>(shared::InvalidSignedId, 1);       ///< Owning mesh id; -1 when no edge found.
    Int_ edge_id = full<Int_>(shared::InvalidSignedId, 1);        ///< Edge index within the owning mesh; -1 when none.
    Int_ global_edge_id = full<Int_>(shared::InvalidSignedId, 1); ///< Scene-global edge index.
    Mask_ is_boundary = full<Mask_>(false, 1); ///< Whether the nearest edge is a boundary (open) edge.

    DRJIT_STRUCT(NearestRayEdgeData, distance, ray_t, point, edge_t, edge_point, shape_id, edge_id, global_edge_id,
                 is_boundary)
};

template <bool Detached> using NearestRayEdgeT = NearestRayEdgeData<FloatT<Detached>>;

using NearestRayEdgeAD = NearestRayEdgeT<false>;
using NearestRayEdge = NearestRayEdgeT<true>;

/// Result of a k-nearest-edges query. The per-result arrays are laid out as
/// query_count * k, with each query's k results contiguous and ordered nearest-first.
template <typename Float_> struct NearestEdgesTopKData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int query_count = 0; ///< Number of query points.
    int k = 0;           ///< Neighbors requested per query.

    Mask_ is_valid = full<Mask_>(false, 1);       ///< Whether each of the query_count * k slots holds an edge.
    Float_ distances = full<Float_>(Infinity, 1); ///< Distance to each result edge.
    Vec3f points = zeros<Vec3f>(1);               ///< Query point echoed per slot.
    Float_ edge_t = zeros<Float_>(1);             ///< Closest-point parameter in [0, 1] along each edge.
    Vec3f edge_points = zeros<Vec3f>(1);          ///< Closest point on each result edge.
    Int_ shape_ids = full<Int_>(shared::InvalidSignedId, 1);       ///< Owning mesh id per slot.
    Int_ edge_ids = full<Int_>(shared::InvalidSignedId, 1);        ///< Per-mesh edge id per slot.
    Int_ global_edge_ids = full<Int_>(shared::InvalidSignedId, 1); ///< Scene-global edge id per slot.
    Mask_ is_boundary = full<Mask_>(false, 1);                     ///< Boundary-edge flag per slot.

    DRJIT_STRUCT(NearestEdgesTopKData, is_valid, distances, points, edge_t, edge_points, shape_ids, edge_ids,
                 global_edge_ids, is_boundary)
};

template <bool Detached> using NearestEdgesTopKT = NearestEdgesTopKData<FloatT<Detached>>;

using NearestEdgesTopKAD = NearestEdgesTopKT<false>;
using NearestEdgesTopK = NearestEdgesTopKT<true>;

static_assert(static_cast<std::uint8_t>(shared::NearestPointEdgeField::IsBoundary) == 7u);
static_assert(static_cast<std::uint8_t>(shared::NearestRayEdgeField::IsBoundary) == 8u);
static_assert(static_cast<std::uint8_t>(shared::NearestEdgesTopKField::IsBoundary) == 8u);

/// World-space geometry per edge consumed by the edge BVH and edge queries.
template <typename Float_> struct SecondaryEdgeInfoData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    /// First edge endpoint in world space.
    Vec3f start;
    /// Edge vector such that `start + edge` is the second endpoint.
    Vec3f edge;
    /// Face normal on one side of the edge.
    Vec3f normal0;
    /// Face normal on the opposite side of the edge.
    Vec3f normal1;
    /// Third vertex of the face associated with `normal0`.
    Vec3f opposite;
    /// Boundary marker for edges that have no opposite face.
    Mask_ is_boundary;

    int size() const { return is_boundary.size(); }

    DRJIT_STRUCT(SecondaryEdgeInfoData, start, edge, normal0, normal1, opposite, is_boundary)
};

using SecondaryEdgeInfoAD = SecondaryEdgeInfoData<FloatAD>;
using SecondaryEdgeInfo = SecondaryEdgeInfoData<Float>;

/// Scene-global per-edge geometry and ids, as returned by Scene::edge_info().
struct SceneEdgeInfo {
    Vector3fAD start;   ///< First edge endpoint in world space.
    Vector3fAD edge;    ///< Edge vector; start + edge is the second endpoint.
    Vector3fAD end;     ///< Second edge endpoint in world space.
    FloatAD length;     ///< Edge length.
    Vector3fAD normal0; ///< Face normal on one side of the edge.
    Vector3fAD normal1; ///< Face normal on the other side (undefined for boundary edges).
    MaskAD is_boundary; ///< Whether the edge has only one adjacent face.
    Int shape_id;       ///< Owning mesh id.
    Int local_edge_id;  ///< Edge index within the owning mesh.
    Int global_edge_id; ///< Edge index within the scene-global edge set.

    int size() const { return global_edge_id.size(); }

    DRJIT_STRUCT(SceneEdgeInfo, start, edge, end, length, normal0, normal1, is_boundary, shape_id, local_edge_id,
                 global_edge_id)
};

/// Scene-global edge connectivity. Each field is one entry per edge; `*_global`
/// variants index the scene-global vertex/face buffers, the others are per-mesh.
/// face1 / opposite_vertex1 are -1 for boundary edges.
struct SceneEdgeTopology {
    Int v0;                      ///< First endpoint vertex id (per-mesh).
    Int v1;                      ///< Second endpoint vertex id (per-mesh).
    Int v0_global;               ///< First endpoint vertex id (scene-global).
    Int v1_global;               ///< Second endpoint vertex id (scene-global).
    Int face0_local;             ///< First adjacent face id (per-mesh).
    Int face1_local;             ///< Second adjacent face id (per-mesh); -1 if boundary.
    Int face0_global;            ///< First adjacent face id (scene-global).
    Int face1_global;            ///< Second adjacent face id (scene-global); -1 if boundary.
    Int opposite_vertex0;        ///< Vertex of face0 opposite the edge (per-mesh).
    Int opposite_vertex1;        ///< Vertex of face1 opposite the edge (per-mesh); -1 if boundary.
    Int opposite_vertex0_global; ///< opposite_vertex0 in scene-global indexing.
    Int opposite_vertex1_global; ///< opposite_vertex1 in scene-global indexing; -1 if boundary.

    int size() const { return v0.size(); }

    DRJIT_STRUCT(SceneEdgeTopology, v0, v1, v0_global, v1_global, face0_local, face1_local, face0_global, face1_global,
                 opposite_vertex0, opposite_vertex1, opposite_vertex0_global, opposite_vertex1_global)
};

} // namespace rayd

namespace rayd {

// Retained edge-BVH build controls after configuration convergence. GpuTreelet
// plus Overlap is the product path. None is a benchmark-only pure-LBVH baseline,
// while Serial is a deterministic debug mode without a performance commitment.

/// Optional optimization pass applied after the initial BVH build.
enum class EdgeBVHPostBuildStrategy {
    None,      ///< Benchmark/reference pure-LBVH baseline only.
    GpuTreelet ///< GPU treelet reoptimization (default).
};

/// Whether build stages run serially or overlap across CUDA streams.
enum class EdgeBVHBuildStreamMode {
    Serial, ///< Deterministic debug mode.
    Overlap ///< Product default.
};

constexpr EdgeBVHPostBuildStrategy EdgeBVHDefaultPostBuildStrategy = EdgeBVHPostBuildStrategy::GpuTreelet;
constexpr EdgeBVHBuildStreamMode EdgeBVHDefaultBuildStreamMode = EdgeBVHBuildStreamMode::Overlap;
constexpr int EdgeBVHLeafSize = shared::BvhLeafSize;

/// Lower-case an env-var value and map '-' to '_' so mode names compare uniformly.
inline std::string normalize_edge_bvh_mode_value(const char* value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) -> char {
        if (ch == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(ch));
    });
    return normalized;
}

// The active_* readers each resolve their mode once from the named environment
// variable (falling back to the default above) and cache the result for the process.

/// Post-build strategy from RAYD_EDGE_BVH_POST_BUILD_STRATEGY.
inline EdgeBVHPostBuildStrategy active_edge_bvh_post_build_strategy() {
    static const EdgeBVHPostBuildStrategy value = []() {
        const char* raw = std::getenv("RAYD_EDGE_BVH_POST_BUILD_STRATEGY");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultPostBuildStrategy;
        }
        if (normalized == "none") {
            return EdgeBVHPostBuildStrategy::None;
        }
        if (normalized == "gpu_treelet") {
            return EdgeBVHPostBuildStrategy::GpuTreelet;
        }
        throw std::runtime_error("Invalid RAYD_EDGE_BVH_POST_BUILD_STRATEGY. Expected one of: none, gpu_treelet.");
    }();
    return value;
}

/// Build stream mode from RAYD_EDGE_BVH_BUILD_STREAM_MODE.
inline EdgeBVHBuildStreamMode active_edge_bvh_build_stream_mode() {
    static const EdgeBVHBuildStreamMode value = []() {
        const char* raw = std::getenv("RAYD_EDGE_BVH_BUILD_STREAM_MODE");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultBuildStreamMode;
        }
        if (normalized == "serial") {
            return EdgeBVHBuildStreamMode::Serial;
        }
        if (normalized == "overlap") {
            return EdgeBVHBuildStreamMode::Overlap;
        }
        throw std::runtime_error("Invalid RAYD_EDGE_BVH_BUILD_STREAM_MODE. Expected one of: serial, overlap.");
    }();
    return value;
}

// Treelet reoptimization thresholds (GpuTreelet post-build strategy).
constexpr int EdgeBVHTreeletMaxLeaves = shared::BvhTreeletMaxLeaves;
constexpr int EdgeBVHTreeletMinPrimitives = shared::BvhTreeletMinPrimitives;
constexpr int EdgeBVHTreeletMaxPrimitives = shared::BvhTreeletMaxPrimitives;
constexpr int EdgeBVHTreeletMinSubtreeLeaves = shared::BvhTreeletMinSubtreeLeaves;
constexpr float EdgeBVHTreeletCostInflationRatio = shared::BvhTreeletCostInflationRatio;

} // namespace rayd

namespace rayd {

/// Maximum k supported by the OptiX top-k edge intersection program.
constexpr int EdgeOptixTopKMax = shared::EdgeOptixTopKMax;

/// Launch parameters for the OptiX edge-query programs (point / ray / top-k).
/// Inputs are flat SoA device pointers; \p k selects point vs. top-k semantics.
struct EdgeOptixQueryParams {
    uint64_t handle = 0; ///< Traversable handle of the edge GAS.

    const float* edge_p0_x = nullptr; ///< Edge start x (one per edge).
    const float* edge_p0_y = nullptr;
    const float* edge_p0_z = nullptr;
    const float* edge_e1_x = nullptr; ///< Edge vector x (start + e1 is the far endpoint).
    const float* edge_e1_y = nullptr;
    const float* edge_e1_z = nullptr;
    const uint8_t* edge_mask = nullptr; ///< Per-edge active flag, or null for all-active.
    int edge_count = 0;
    float search_radius = 0.0f; ///< Distance cutoff; hits beyond this are rejected.

    const float* query_x = nullptr; ///< Query point / ray origin x (one per query).
    const float* query_y = nullptr;
    const float* query_z = nullptr;
    const float* ray_dx = nullptr; ///< RayAD direction x (ray queries only).
    const float* ray_dy = nullptr;
    const float* ray_dz = nullptr;
    const float* ray_tmax = nullptr;      ///< Per-ray max parameter (ray queries only).
    const uint8_t* active_mask = nullptr; ///< Per-query active flag.
    int query_count = 0;
    int k = 0; ///< Neighbors per query; results in query_count * k order.

    int* out_edge_ids = nullptr;      ///< Winning edge id(s).
    float* out_distance_sq = nullptr; ///< Squared distance to the winner(s).
    float* out_ray_t = nullptr;       ///< RayAD parameter at closest approach (ray queries).
    float* out_edge_t = nullptr;      ///< Closest-point parameter along the edge.
    uint8_t* out_valid = nullptr;     ///< Whether each output slot holds a hit.
};

} // namespace rayd

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
    Mask is_valid;       ///< Whether each slot holds a valid edge.
    Int global_edge_ids; ///< Scene-global edge id per slot.
    Float distance_sq;   ///< Squared distance per slot.
};

/// Structural and quality metrics of a built edge BVH (for diagnostics/tuning).
struct SceneEdgeBVHStats {
    int primitive_count = 0; ///< Number of edge primitives.
    int node_count = 0;      ///< Total BVH nodes.
    int internal_node_count = 0;
    int leaf_node_count = 0;
    int max_height = 0;        ///< Maximum root-to-leaf depth.
    int refit_level_count = 0; ///< Number of levels touched during refit.
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
    void build(const SecondaryEdgeInfoAD& edge_info);
    /// Build while retaining dynamic-refit state only when \p allow_refit is true.
    void build(const SecondaryEdgeInfoAD& edge_info, bool allow_refit);
    /// Build the BVH, restricting queries to edges where \p mask is true.
    void build(const SecondaryEdgeInfoAD& edge_info, const Mask& mask);
    /// Masked build with optional dynamic-refit state retention.
    void build(const SecondaryEdgeInfoAD& edge_info, const Mask& mask, bool allow_refit);
    /// Update the per-edge active mask without rebuilding the tree.
    void set_mask(const Mask& mask);
    /// Refit node bounds after the edges in \p dirty_ranges moved (topology unchanged).
    void refit(const SecondaryEdgeInfoAD& edge_info, const std::vector<EdgeDirtyRange>& dirty_ranges);
    /// Refit node bounds after the edges at \p primitive_indices moved.
    void refit(const SecondaryEdgeInfoAD& edge_info, const Int& primitive_indices);
    /// Force evaluation of the lazily built BVH device buffers.
    void materialize() const;
    /// Translate internal BVH primitive ids to scene-global edge ids; \p valid gates the gather.
    Int map_to_global(const Int& bvh_ids, const Mask& valid) const;
    bool is_ready() const { return ready_; }
    bool has_edges() const { return primitive_count_ > 0; }
    SceneEdgeBVHStats stats() const;

    /// Nearest active edge to each query point; clears \p active lanes that find none.
    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const Vector3fT<Detached>& point, MaskT<Detached>& active) const;

    /// The \p k nearest active edges to each query point (results in query_count * k order).
    template <bool Detached>
    ClosestEdgeTopKCandidate nearest_edges(const Vector3fT<Detached>& point, int k, MaskT<Detached>& active) const;

    /// Nearest active edge to each ray; uses segment semantics on [0, tmax] when tmax is finite.
    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const RayT<Detached>& ray, MaskT<Detached>& active) const;

  private:
    void build_bvh(const SecondaryEdgeInfoAD& edge_info, bool allow_refit);
    void set_all_active_state();
    void update_active_counts_from_mask(const Mask& mask);
    Int refit_leaf_nodes_from_primitive_indices(const SecondaryEdgeInfoAD& edge_info, const Int& primitive_indices);
    void refit_internal_nodes_full();
    void refit_internal_nodes_dirty(const std::vector<Int>& dirty_leaf_chunks);
    ClosestEdgeCandidate nearest_edge_point_detached(const Vector3f& point, const Mask& active) const;
    ClosestEdgeTopKCandidate nearest_edges_point_detached(const Vector3f& point, int k, const Mask& active) const;
    ClosestEdgeCandidate nearest_edge_finite_ray_detached(const Vector3f& origin, const Vector3f& segment,
                                                          const Mask& active) const;
    ClosestEdgeCandidate nearest_edge_infinite_ray_detached(const Vector3f& origin, const Vector3f& direction,
                                                            const Mask& active) const;
    void scatter_node_bounds(const Int& node_indices, const Vector3f& bbox_min, const Vector3f& bbox_max);
    Int gather_node_left_child(const Int& node_indices, const Mask& active) const;
    Int gather_node_right_child(const Int& node_indices, const Mask& active) const;
    Int gather_node_active_count(const Int& node_indices, const Mask& active) const;
    Vector3f gather_node_bbox_min(const Int& node_indices, const Mask& active) const;
    Vector3f gather_node_bbox_max(const Int& node_indices, const Mask& active) const;

    int primitive_count_ = 0;
    int node_count_ = 0;
    bool ready_ = false;
    bool all_active_ = true;
    bool refit_enabled_ = true;

    Vector3f edge_p0_;
    Vector3f edge_e1_;
    Vector3f primitive_bbox_min_;
    Vector3f primitive_bbox_max_;
    Vector3f node_bbox_min_;
    Vector3f node_bbox_max_;
    Int left_child_;
    Int right_child_;
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

namespace rayd {

struct EdgeOptixState;

/// Experimental OptiX edge backend: edges are custom AABB primitives traversed by
/// OptiX. Mirrors the SceneEdge query surface; selected via EdgeBVHBackend::Optix.
class SceneEdgeOptix {
  public:
    SceneEdgeOptix();
    ~SceneEdgeOptix();

    SceneEdgeOptix(const SceneEdgeOptix&) = delete;
    SceneEdgeOptix& operator=(const SceneEdgeOptix&) = delete;

    /// Build the custom-AABB GAS over the edges in \p edge_info, masked by \p mask.
    void build(const SecondaryEdgeInfoAD& edge_info, const Mask& mask);
    /// Update the per-edge active mask without rebuilding the GAS.
    void set_mask(const Mask& mask);
    /// Refit the GAS after the edges in \p dirty_ranges moved.
    void refit(const SecondaryEdgeInfoAD& edge_info, const std::vector<EdgeDirtyRange>& dirty_ranges);
    bool is_ready() const { return ready_; }
    bool has_edges() const { return primitive_count_ > 0; }
    SceneEdgeBVHStats stats() const;

    /// Nearest active edge to each query point; clears \p active lanes that find none.
    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const Vector3fT<Detached>& point, MaskT<Detached>& active) const;

    /// Nearest active edge to each query ray.
    template <bool Detached>
    ClosestEdgeCandidate nearest_edge(const RayT<Detached>& ray, MaskT<Detached>& active) const;

    /// The \p k nearest active edges to each query point.
    template <bool Detached>
    ClosestEdgeTopKCandidate nearest_edges(const Vector3fT<Detached>& point, int k, MaskT<Detached>& active) const;

  private:
    void build_gases(bool update);
    void ensure_pipeline();
    void refresh_geometry(const SecondaryEdgeInfoAD& edge_info);
    /// Per-edge OptiX AABB inflation radius, sized to bound the nearest-edge search.
    std::vector<float> compute_search_radii(const SecondaryEdgeInfoAD& edge_info) const;

    EdgeOptixState* state_ = nullptr;
    int primitive_count_ = 0;
    bool ready_ = false;
    std::vector<float> search_radii_;

    Vector3f edge_p0_;
    Vector3f edge_e1_;
    Mask edge_mask_;
};

} // namespace rayd
