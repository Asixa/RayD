#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

constexpr int RAYD_NUM_CHANNELS = 3;

#include <rayd/types.h>

namespace rayd {

constexpr float Epsilon = 1e-5f;
constexpr float RayEpsilon = 1e-3f;
constexpr float ShadowEpsilon = 1e-3f;
constexpr float Pi = 3.14159265358979323846f;
constexpr float Infinity = std::numeric_limits<float>::infinity();

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

template <typename> struct ReflectionEpcResultData;
template <bool Detached>
using ReflectionEpcResultT = ReflectionEpcResultData<FloatT<Detached>>;
using ReflectionEpcResult = ReflectionEpcResultT<false>;
using ReflectionEpcResultDetached = ReflectionEpcResultT<true>;
struct ReflectionEpcOptions;
struct ReflectionEpcFieldOptions;
template <typename> struct ReflectionEpcFieldResultData;
template <bool Detached>
using ReflectionEpcFieldResultT = ReflectionEpcFieldResultData<FloatT<Detached>>;
using ReflectionEpcFieldResult = ReflectionEpcFieldResultT<false>;
using ReflectionEpcFieldResultDetached = ReflectionEpcFieldResultT<true>;

template <typename> struct MaterialData;
template <bool Detached>
using MaterialT = MaterialData<FloatT<Detached>>;
using Material = MaterialT<false>;
using MaterialDetached = MaterialT<true>;

template <typename> struct WedgeEventsData;
template <bool Detached>
using WedgeEventsT = WedgeEventsData<FloatT<Detached>>;
using WedgeEvents = WedgeEventsT<false>;
using WedgeEventsDetached = WedgeEventsT<true>;

template <typename> struct AccumResultData;
template <bool Detached>
using AccumResultT = AccumResultData<FloatT<Detached>>;
using AccumResult = AccumResultT<false>;
using AccumResultDetached = AccumResultT<true>;

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

template <typename> struct SegmentChainVisibilityData;
template <bool Detached>
using SegmentChainVisibilityT = SegmentChainVisibilityData<FloatT<Detached>>;
using SegmentChainVisibility = SegmentChainVisibilityT<false>;
using SegmentChainVisibilityDetached = SegmentChainVisibilityT<true>;

struct OptixIntersection;

class Mesh;
class OptixScene;
class OptixLaunchPipeline;
class SceneEdge;
class Scene;

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

} // namespace rayd

#include <rayd/utils.h>
