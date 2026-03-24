#pragma once

#include <raydi/raydi.h>
#include <raydi/ray.h>

namespace raydi {

template <typename Float_>
struct NearestPointEdgeData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;
    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;

    Mask_ is_valid() const {
        return edge_id >= 0;
    }

    Float_ distance = Infinity;
    Vec3f point = zeros<Vec3f>(1);
    Float_ edge_t = zeros<Float_>(1);
    Vec3f edge_point = zeros<Vec3f>(1);
    Int_ shape_id = full<Int_>(-1, 1);
    Int_ edge_id = full<Int_>(-1, 1);
    Mask_ is_boundary = full<Mask_>(false, 1);

    DRJIT_STRUCT(NearestPointEdgeData,
                 distance,
                 point,
                 edge_t,
                 edge_point,
                 shape_id,
                 edge_id,
                 is_boundary)
};

template <bool Detached>
using NearestPointEdgeT = NearestPointEdgeData<FloatT<Detached>>;

using NearestPointEdge = NearestPointEdgeT<false>;
using NearestPointEdgeDetached = NearestPointEdgeT<true>;

template <typename Float_>
struct NearestRayEdgeData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;
    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;

    Mask_ is_valid() const {
        return edge_id >= 0;
    }

    Float_ distance = Infinity;
    Float_ ray_t = zeros<Float_>(1);
    Vec3f point = zeros<Vec3f>(1);
    Float_ edge_t = zeros<Float_>(1);
    Vec3f edge_point = zeros<Vec3f>(1);
    Int_ shape_id = full<Int_>(-1, 1);
    Int_ edge_id = full<Int_>(-1, 1);
    Mask_ is_boundary = full<Mask_>(false, 1);

    DRJIT_STRUCT(NearestRayEdgeData,
                 distance,
                 ray_t,
                 point,
                 edge_t,
                 edge_point,
                 shape_id,
                 edge_id,
                 is_boundary)
};

template <bool Detached>
using NearestRayEdgeT = NearestRayEdgeData<FloatT<Detached>>;

using NearestRayEdge = NearestRayEdgeT<false>;
using NearestRayEdgeDetached = NearestRayEdgeT<true>;

/// Primary-edge sample returned by image-space edge sampling.
struct PrimaryEdgeSample {
    Float x_dot_n;
    IntDetached idx;
    RayDetached ray_n;
    RayDetached ray_p;
    FloatDetached pdf;
};

template <typename Float_>
struct PrimaryEdgeInfoData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Vec2f = std::conditional_t<IsDetached, Vector2fDetached, Vector2f>;
    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;

#ifdef RAYDI_PRIMARY_EDGE_VIS_CHECK
    Vec3f p0;
    Vec3f p1;
#else
    Vec2f p0;
    Vec2f p1;
#endif
    Vec2f edge_normal;
    Float_ edge_length;

    DRJIT_STRUCT(PrimaryEdgeInfoData, p0, p1, edge_normal, edge_length)
};

using PrimaryEdgeInfo = PrimaryEdgeInfoData<Float>;
using PrimaryEdgeInfoDetached = PrimaryEdgeInfoData<FloatDetached>;

template <typename Float_>
struct SecondaryEdgeInfoData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;
    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;

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

using SecondaryEdgeInfo = SecondaryEdgeInfoData<Float>;
using SecondaryEdgeInfoDetached = SecondaryEdgeInfoData<FloatDetached>;

} // namespace raydi
