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

struct PrimaryEdgeSample;
struct OptixIntersection;

class Mesh;
class Camera;
class OptixScene;
class SceneEdge;
class Scene;

} // namespace rayd

