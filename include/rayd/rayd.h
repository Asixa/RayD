#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

/// Number of field channels carried per ray (e.g. RGB or three polarization components).
constexpr int RAYD_NUM_CHANNELS = 3;

#include <rayd/types.h>

namespace rayd {

constexpr float Epsilon = 1e-5f;       // General-purpose geometric tolerance.
constexpr float RayEpsilon = 1e-3f;    // RayAD t_min offset to avoid self-intersection.
constexpr float ShadowEpsilon = 1e-3f; // Shadow/visibility ray offset.
constexpr float Pi = 3.14159265358979323846f;
constexpr float Infinity = std::numeric_limits<float>::infinity();

// Forward declarations and the canonical alias pattern for RayD's value types.
// Each type X is defined as a template XData<Float_>; XT<Detached> selects the
// non-AD or AD Float, and X (non-AD) / XAD (AD) are the two concrete instantiations.
template <typename> struct RayData;
template <bool Detached>
using RayT = RayData<FloatT<Detached>>;
using RayAD = RayT<false>;
using Ray = RayT<true>;

template <typename> struct IntersectionData;
template <bool Detached>
using IntersectionT = IntersectionData<FloatT<Detached>>;
using IntersectionAD = IntersectionT<false>;
using Intersection = IntersectionT<true>;

template <typename> struct ReflectionChainData;
template <bool Detached>
using ReflectionChainT = ReflectionChainData<FloatT<Detached>>;
using ReflectionChainAD = ReflectionChainT<false>;
using ReflectionChain = ReflectionChainT<true>;

template <typename> struct ReflectionBounceData;
template <bool Detached>
using ReflectionBounceT = ReflectionBounceData<FloatT<Detached>>;
using ReflectionBounceAD = ReflectionBounceT<false>;
using ReflectionBounce = ReflectionBounceT<true>;

template <typename> struct ReflectionTraceData;
template <bool Detached>
using ReflectionTraceT = ReflectionTraceData<FloatT<Detached>>;
using ReflectionTraceAD = ReflectionTraceT<false>;
using ReflectionTrace = ReflectionTraceT<true>;

template <typename> struct ReflectionEpcResultData;
template <bool Detached>
using ReflectionEpcResultT = ReflectionEpcResultData<FloatT<Detached>>;
using ReflectionEpcResultAD = ReflectionEpcResultT<false>;
using ReflectionEpcResult = ReflectionEpcResultT<true>;
struct ReflectionEpcOptions;
struct ReflectionEpcFieldOptions;
template <typename> struct ReflectionEpcFieldResultData;
template <bool Detached>
using ReflectionEpcFieldResultT = ReflectionEpcFieldResultData<FloatT<Detached>>;
using ReflectionEpcFieldResultAD = ReflectionEpcFieldResultT<false>;
using ReflectionEpcFieldResult = ReflectionEpcFieldResultT<true>;

template <typename> struct MaterialData;
template <bool Detached>
using MaterialT = MaterialData<FloatT<Detached>>;
using MaterialAD = MaterialT<false>;
using Material = MaterialT<true>;

template <typename> struct WedgeEventsData;
template <bool Detached>
using WedgeEventsT = WedgeEventsData<FloatT<Detached>>;
using WedgeEventsAD = WedgeEventsT<false>;
using WedgeEvents = WedgeEventsT<true>;

template <typename> struct AccumResultData;
template <bool Detached>
using AccumResultT = AccumResultData<FloatT<Detached>>;
using AccumResultAD = AccumResultT<false>;
using AccumResult = AccumResultT<true>;

template <typename> struct NearestPointEdgeData;
template <bool Detached>
using NearestPointEdgeT = NearestPointEdgeData<FloatT<Detached>>;
using NearestPointEdgeAD = NearestPointEdgeT<false>;
using NearestPointEdge = NearestPointEdgeT<true>;

template <typename> struct NearestRayEdgeData;
template <bool Detached>
using NearestRayEdgeT = NearestRayEdgeData<FloatT<Detached>>;
using NearestRayEdgeAD = NearestRayEdgeT<false>;
using NearestRayEdge = NearestRayEdgeT<true>;

template <typename> struct NearestEdgesTopKData;
template <bool Detached>
using NearestEdgesTopKT = NearestEdgesTopKData<FloatT<Detached>>;
using NearestEdgesTopKAD = NearestEdgesTopKT<false>;
using NearestEdgesTopK = NearestEdgesTopKT<true>;

template <typename> struct SegmentVisibilityData;
template <bool Detached>
using SegmentVisibilityT = SegmentVisibilityData<FloatT<Detached>>;
using SegmentVisibilityAD = SegmentVisibilityT<false>;
using SegmentVisibility = SegmentVisibilityT<true>;

template <typename> struct SegmentPairVisibilityData;
template <bool Detached>
using SegmentPairVisibilityT = SegmentPairVisibilityData<FloatT<Detached>>;
using SegmentPairVisibilityAD = SegmentPairVisibilityT<false>;
using SegmentPairVisibility = SegmentPairVisibilityT<true>;

template <typename> struct AxialEdgeVisibilityData;
template <bool Detached>
using AxialEdgeVisibilityT = AxialEdgeVisibilityData<FloatT<Detached>>;
using AxialEdgeVisibilityAD = AxialEdgeVisibilityT<false>;
using AxialEdgeVisibility = AxialEdgeVisibilityT<true>;

template <typename> struct SegmentChainVisibilityData;
template <bool Detached>
using SegmentChainVisibilityT = SegmentChainVisibilityData<FloatT<Detached>>;
using SegmentChainVisibilityAD = SegmentChainVisibilityT<false>;
using SegmentChainVisibility = SegmentChainVisibilityT<true>;

struct OptixIntersection;

class Mesh;
class OptixScene;
class OptixLaunchPipeline;
class SceneEdge;
class Scene;

/// Throw std::runtime_error with \p message when \p condition is false (host-side precondition check).
inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

} // namespace rayd

#include <rayd/utils.h>
