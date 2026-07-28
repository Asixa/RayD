#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <rayd/shared/contracts.h>

/// Number of field channels carried per ray (e.g. RGB or three polarization components).
constexpr int RAYD_NUM_CHANNELS = 3;

#include <rayd/core/drjit/types.h>

namespace rayd {

constexpr float Epsilon = shared::GeneralEpsilon; // General-purpose geometric tolerance.
constexpr float RayEpsilon = shared::RayEpsilon;  // RayAD t_min offset to avoid self-intersection.
constexpr float ShadowEpsilon = shared::ShadowEpsilon; // Shadow/visibility ray offset.
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

template <typename> struct SurfelIntersectionData;
template <bool Detached>
using SurfelIntersectionT = SurfelIntersectionData<FloatT<Detached>>;
using SurfelIntersectionAD = SurfelIntersectionT<false>;
using SurfelIntersection = SurfelIntersectionT<true>;

template <typename> struct SurfelCompositeData;
template <bool Detached>
using SurfelCompositeT = SurfelCompositeData<FloatT<Detached>>;
using SurfelCompositeAD = SurfelCompositeT<false>;
using SurfelComposite = SurfelCompositeT<true>;

template <typename> struct SurfelRenderData;
template <bool Detached>
using SurfelRenderT = SurfelRenderData<FloatT<Detached>>;
using SurfelRenderAD = SurfelRenderT<false>;
using SurfelRender = SurfelRenderT<true>;

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

template <typename> struct ReflEpcData;
template <bool Detached>
using ReflEpcT = ReflEpcData<FloatT<Detached>>;
using ReflEpcAD = ReflEpcT<false>;
using ReflEpc = ReflEpcT<true>;
struct ReflEpcOptions;
template <bool Detached> struct ReflEpcFieldOptionsT;
using ReflEpcFieldOptionsAD = ReflEpcFieldOptionsT<false>;
using ReflEpcFieldOptions = ReflEpcFieldOptionsT<true>;
template <typename> struct ReflEpcFieldData;
template <bool Detached>
using ReflEpcFieldT = ReflEpcFieldData<FloatT<Detached>>;
using ReflEpcFieldAD = ReflEpcFieldT<false>;
using ReflEpcField = ReflEpcFieldT<true>;

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

struct DfrGrid;
struct DfrOptions;
struct DfrCoherentOptions;
struct DfrPathOptions;

template <typename> struct DfrMaterialData;
template <bool Detached>
using DfrMaterialT = DfrMaterialData<FloatT<Detached>>;
using DfrMaterialAD = DfrMaterialT<false>;
using DfrMaterial = DfrMaterialT<true>;

template <typename> struct DfrStatesData;
template <bool Detached>
using DfrStatesT = DfrStatesData<FloatT<Detached>>;
using DfrStatesAD = DfrStatesT<false>;
using DfrStates = DfrStatesT<true>;

template <typename> struct DfrCoherentUtdStatesData;
template <bool Detached>
using DfrCoherentUtdStatesT = DfrCoherentUtdStatesData<FloatT<Detached>>;
using DfrCoherentUtdStatesAD = DfrCoherentUtdStatesT<false>;
using DfrCoherentUtdStates = DfrCoherentUtdStatesT<true>;

template <typename> struct DfrCoherentEdgeData;
template <bool Detached>
using DfrCoherentEdgeT = DfrCoherentEdgeData<FloatT<Detached>>;
using DfrCoherentEdgeAD = DfrCoherentEdgeT<false>;
using DfrCoherentEdge = DfrCoherentEdgeT<true>;

template <typename> struct DfrCoherentCandidatePairsData;
template <bool Detached>
using DfrCoherentCandidatePairsT = DfrCoherentCandidatePairsData<FloatT<Detached>>;
using DfrCoherentCandidatePairsAD = DfrCoherentCandidatePairsT<false>;
using DfrCoherentCandidatePairs = DfrCoherentCandidatePairsT<true>;

template <typename> struct DfrAccumData;
template <bool Detached>
using DfrAccumT = DfrAccumData<FloatT<Detached>>;
using DfrAccumAD = DfrAccumT<false>;
using DfrAccum = DfrAccumT<true>;

template <typename> struct DfrCoherentAccumData;
template <bool Detached>
using DfrCoherentAccumT = DfrCoherentAccumData<FloatT<Detached>>;
using DfrCoherentAccumAD = DfrCoherentAccumT<false>;
using DfrCoherentAccum = DfrCoherentAccumT<true>;

template <typename> struct DfrPathsData;
template <bool Detached>
using DfrPathsT = DfrPathsData<FloatT<Detached>>;
using DfrPathsAD = DfrPathsT<false>;
using DfrPaths = DfrPathsT<true>;

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
struct SurfelOptixState;

class Mesh;
class OptixScene;
class SurfelCloud;
class SurfelScene;
class SurfelOptixScene;
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

#include <rayd/core/drjit/utils.h>
