// Copyright Xingyu Chen.
// Declares the Dr.Jit edge API.

#pragma once

#include <rayd/jit/core.h>
#include <rayd/jit/ray.h>

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
