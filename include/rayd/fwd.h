#pragma once

#include <rayd/types.h>

namespace rayd {

template <typename> struct RayData;
template <bool Detached>
using RayT = RayData<FloatT<Detached>>;
using Ray = RayT<false>;
using RayDetached = RayT<true>;

template <typename> struct IntersectionData;
template <bool Detached>
using IntersectionT = IntersectionData<FloatT<Detached>>;
using Intersection = IntersectionT<false>;
using IntersectionDetached = IntersectionT<true>;

template <typename> struct ReflectionChainData;
template <bool Detached>
using ReflectionChainT = ReflectionChainData<FloatT<Detached>>;
using ReflectionChain = ReflectionChainT<false>;
using ReflectionChainDetached = ReflectionChainT<true>;

template <typename> struct ReflectionBounceData;
template <bool Detached>
using ReflectionBounceT = ReflectionBounceData<FloatT<Detached>>;
using ReflectionBounce = ReflectionBounceT<false>;
using ReflectionBounceDetached = ReflectionBounceT<true>;

template <typename> struct ReflectionTraceData;
template <bool Detached>
using ReflectionTraceT = ReflectionTraceData<FloatT<Detached>>;
using ReflectionTrace = ReflectionTraceT<false>;
using ReflectionTraceDetached = ReflectionTraceT<true>;

template <typename> struct NearestPointEdgeData;
template <bool Detached>
using NearestPointEdgeT = NearestPointEdgeData<FloatT<Detached>>;
using NearestPointEdge = NearestPointEdgeT<false>;
using NearestPointEdgeDetached = NearestPointEdgeT<true>;

template <typename> struct NearestRayEdgeData;
template <bool Detached>
using NearestRayEdgeT = NearestRayEdgeData<FloatT<Detached>>;
using NearestRayEdge = NearestRayEdgeT<false>;
using NearestRayEdgeDetached = NearestRayEdgeT<true>;

template <typename> struct NearestEdgesTopKData;
template <bool Detached>
using NearestEdgesTopKT = NearestEdgesTopKData<FloatT<Detached>>;
using NearestEdgesTopK = NearestEdgesTopKT<false>;
using NearestEdgesTopKDetached = NearestEdgesTopKT<true>;

template <typename> struct SegmentVisibilityData;
template <bool Detached>
using SegmentVisibilityT = SegmentVisibilityData<FloatT<Detached>>;
using SegmentVisibility = SegmentVisibilityT<false>;
using SegmentVisibilityDetached = SegmentVisibilityT<true>;

template <typename> struct SegmentPairVisibilityData;
template <bool Detached>
using SegmentPairVisibilityT = SegmentPairVisibilityData<FloatT<Detached>>;
using SegmentPairVisibility = SegmentPairVisibilityT<false>;
using SegmentPairVisibilityDetached = SegmentPairVisibilityT<true>;

template <typename> struct AxialEdgeVisibilityData;
template <bool Detached>
using AxialEdgeVisibilityT = AxialEdgeVisibilityData<FloatT<Detached>>;
using AxialEdgeVisibility = AxialEdgeVisibilityT<false>;
using AxialEdgeVisibilityDetached = AxialEdgeVisibilityT<true>;

struct PrimaryEdgeSample;
struct OptixIntersection;

class Mesh;
class Camera;
class OptixScene;
class SceneEdge;
class SegmentVisibilityPipeline;
class Scene;

} // namespace rayd

