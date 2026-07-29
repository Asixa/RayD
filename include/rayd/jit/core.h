// Copyright Xingyu Chen.
// Declares the Dr.Jit core types, rays, transforms, and utilities.

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>
#include <drjit/array.h>
#include <drjit/autodiff.h>
#include <drjit/jit.h>
#include <drjit/matrix.h>
#include <drjit/quaternion.h>
#include <drjit/sphere.h>
#include <drjit/transform.h>
#include <rayd/contracts.h>

namespace rayd {

using namespace drjit;

// Core Dr.Jit type aliases. Every array type comes in two flavors: the bare name
// is the non-AD (detached) type and the "AD" suffix is the autodiff-enabled
// CUDADiffArray (e.g. Float / FloatAD). Width-batched templates take a
// `bool Detached_` parameter (true = non-AD) and resolve to the matching flavor
// via the FloatT / IntT / MaskT selectors.

/// Detached (non-AD) counterpart of a Dr.Jit type.
template <typename T> using Detached = drjit::detached_t<T>;

using FloatAD = drjit::CUDADiffArray<float>;
using Float = Detached<FloatAD>;

using IntAD = drjit::CUDADiffArray<int32_t>;
using Int = Detached<IntAD>;

using UIntAD = drjit::CUDADiffArray<uint32_t>;
using UInt = Detached<UIntAD>;

using UInt64AD = drjit::CUDADiffArray<uint64_t>;
using UInt64 = Detached<UInt64AD>;

template <bool Detached_> using FloatT = std::conditional_t<Detached_, Float, FloatAD>;

template <bool Detached_> using IntT = std::conditional_t<Detached_, Int, IntAD>;

using ScalarFloat = drjit::scalar_t<float>;

template <int n, bool Detached_> using VectorfT = drjit::Array<FloatT<Detached_>, n>;

template <int n, bool Detached_> using VectoriT = drjit::Array<IntT<Detached_>, n>;

template <int n, bool Detached_> using MatrixfT = drjit::Matrix<FloatT<Detached_>, n>;

template <bool Detached_> using Vector1fT = VectorfT<1, Detached_>;

template <bool Detached_> using Vector2fT = VectorfT<2, Detached_>;

template <bool Detached_> using Vector2iT = VectoriT<2, Detached_>;

template <bool Detached_> using Vector3fT = VectorfT<3, Detached_>;

template <bool Detached_> using Vector3iT = VectoriT<3, Detached_>;

using Vector1fAD = Vector1fT<false>;
using Vector1f = Vector1fT<true>;

using Vector2fAD = Vector2fT<false>;
using Vector2f = Vector2fT<true>;

using Vector2iAD = Vector2iT<false>;
using Vector2i = Vector2iT<true>;

using Vector3fAD = Vector3fT<false>;
using Vector3f = Vector3fT<true>;

using Vector3iAD = Vector3iT<false>;
using Vector3i = Vector3iT<true>;

template <bool Detached_> using Matrix4fT = MatrixfT<4, Detached_>;

using Matrix4fAD = Matrix4fT<false>;
using Matrix4f = Matrix4fT<true>;

using MaskAD = drjit::mask_t<FloatAD>;
using Mask = drjit::mask_t<Float>;

template <bool Detached_> using MaskT = std::conditional_t<Detached_, Mask, MaskAD>;

using ScalarVector3f = drjit::Array<float, 3>;
using ScalarVector4f = drjit::Array<float, 4>;
using ScalarMatrix4f = drjit::Matrix<float, 4>;

/// Per-triangle cached geometry: edge-vector form of each face plus its normals.
template <typename Float_> struct TriangleInfoData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = drjit::mask_t<Float_>;
    using Vec3f = drjit::Array<Float_, 3>;
    using Vec3i = drjit::Array<std::conditional_t<IsDetached, Int, IntAD>, 3>;

    Vec3f p0;           ///< First vertex; the triangle is parameterized as p0 + s*e1 + t*e2.
    Vec3f e1;           ///< Edge vector from vertex 0 to vertex 1.
    Vec3f e2;           ///< Edge vector from vertex 0 to vertex 2.
    Vec3f n0;           ///< Shading normal at vertex 0.
    Vec3f n1;           ///< Shading normal at vertex 1.
    Vec3f n2;           ///< Shading normal at vertex 2.
    Vec3f face_normal;  ///< Unit geometric face normal (cross(e1, e2) normalized).
    Vec3i face_indices; ///< Vertex indices of the three corners.
    Float_ face_area;   ///< Triangle area in world units.

    DRJIT_STRUCT(TriangleInfoData, p0, e1, e2, n0, n1, n2, face_normal, face_indices, face_area)
};

using TriangleInfoAD = TriangleInfoData<FloatAD>;
using TriangleInfo = TriangleInfoData<Float>;

template <bool Detached_> using TriangleInfoT = std::conditional_t<Detached_, TriangleInfo, TriangleInfoAD>;

/// Per-triangle UV coordinates: one 2D coordinate at each of the three corners.
template <typename Float_> using TriangleUvData = drjit::Array<drjit::Array<Float_, 2>, 3>;

using TriangleUVAD = TriangleUvData<FloatAD>;
using TriangleUV = TriangleUvData<Float>;

template <bool Detached_> using TriangleUVT = std::conditional_t<Detached_, TriangleUV, TriangleUVAD>;

/// Scene-global geometry: all meshes' vertices and faces flattened into one indexing space.
struct SceneGeometry {
    Vector3fAD vertices;    ///< World-space vertex positions, concatenated across meshes.
    Vector3i faces;         ///< Face vertex indices into the scene-global vertex buffer.
    Vector3fAD face_normal; ///< Unit geometric normal per face.
    Int shape_id;           ///< Owning mesh id per face.
    Int local_prim_id;      ///< Face index within its owning mesh.
    Int global_prim_id;     ///< Face index within the scene-global face buffer.

    int vertex_count() const { return vertices.x().size(); }
    int face_count() const { return global_prim_id.size(); }

    DRJIT_STRUCT(SceneGeometry, vertices, faces, face_normal, shape_id, local_prim_id, global_prim_id)
};

} // namespace rayd

/// Number of field channels carried per ray (e.g. RGB or three polarization components).
constexpr int RAYD_NUM_CHANNELS = 3;

namespace rayd {

constexpr float Epsilon = shared::GeneralEpsilon;      // General-purpose geometric tolerance.
constexpr float RayEpsilon = shared::RayEpsilon;       // RayAD t_min offset to avoid self-intersection.
constexpr float ShadowEpsilon = shared::ShadowEpsilon; // Shadow/visibility ray offset.
constexpr float Pi = 3.14159265358979323846f;
constexpr float Infinity = std::numeric_limits<float>::infinity();

// Forward declarations and the canonical alias pattern for RayD's value types.
// Each type X is defined as a template XData<Float_>; XT<Detached> selects the
// non-AD or AD Float, and X (non-AD) / XAD (AD) are the two concrete instantiations.
template <typename> struct RayData;
template <bool Detached> using RayT = RayData<FloatT<Detached>>;
using RayAD = RayT<false>;
using Ray = RayT<true>;

template <typename> struct IntersectionData;
template <bool Detached> using IntersectionT = IntersectionData<FloatT<Detached>>;
using IntersectionAD = IntersectionT<false>;
using Intersection = IntersectionT<true>;

template <typename> struct SurfelIntersectionData;
template <bool Detached> using SurfelIntersectionT = SurfelIntersectionData<FloatT<Detached>>;
using SurfelIntersectionAD = SurfelIntersectionT<false>;
using SurfelIntersection = SurfelIntersectionT<true>;

template <typename> struct SurfelCompositeData;
template <bool Detached> using SurfelCompositeT = SurfelCompositeData<FloatT<Detached>>;
using SurfelCompositeAD = SurfelCompositeT<false>;
using SurfelComposite = SurfelCompositeT<true>;

template <typename> struct SurfelRenderData;
template <bool Detached> using SurfelRenderT = SurfelRenderData<FloatT<Detached>>;
using SurfelRenderAD = SurfelRenderT<false>;
using SurfelRender = SurfelRenderT<true>;

template <typename> struct ReflectionChainData;
template <bool Detached> using ReflectionChainT = ReflectionChainData<FloatT<Detached>>;
using ReflectionChainAD = ReflectionChainT<false>;
using ReflectionChain = ReflectionChainT<true>;

template <typename> struct ReflectionBounceData;
template <bool Detached> using ReflectionBounceT = ReflectionBounceData<FloatT<Detached>>;
using ReflectionBounceAD = ReflectionBounceT<false>;
using ReflectionBounce = ReflectionBounceT<true>;

template <typename> struct ReflectionTraceData;
template <bool Detached> using ReflectionTraceT = ReflectionTraceData<FloatT<Detached>>;
using ReflectionTraceAD = ReflectionTraceT<false>;
using ReflectionTrace = ReflectionTraceT<true>;

template <typename> struct ReflEpcData;
template <bool Detached> using ReflEpcT = ReflEpcData<FloatT<Detached>>;
using ReflEpcAD = ReflEpcT<false>;
using ReflEpc = ReflEpcT<true>;
struct ReflEpcOptions;
template <bool Detached> struct ReflEpcFieldOptionsT;
using ReflEpcFieldOptionsAD = ReflEpcFieldOptionsT<false>;
using ReflEpcFieldOptions = ReflEpcFieldOptionsT<true>;
template <typename> struct ReflEpcFieldData;
template <bool Detached> using ReflEpcFieldT = ReflEpcFieldData<FloatT<Detached>>;
using ReflEpcFieldAD = ReflEpcFieldT<false>;
using ReflEpcField = ReflEpcFieldT<true>;

template <typename> struct MaterialData;
template <bool Detached> using MaterialT = MaterialData<FloatT<Detached>>;
using MaterialAD = MaterialT<false>;
using Material = MaterialT<true>;

template <typename> struct WedgeEventsData;
template <bool Detached> using WedgeEventsT = WedgeEventsData<FloatT<Detached>>;
using WedgeEventsAD = WedgeEventsT<false>;
using WedgeEvents = WedgeEventsT<true>;

template <typename> struct AccumResultData;
template <bool Detached> using AccumResultT = AccumResultData<FloatT<Detached>>;
using AccumResultAD = AccumResultT<false>;
using AccumResult = AccumResultT<true>;

struct DfrGrid;
struct DfrOptions;
struct DfrCoherentOptions;
struct DfrPathOptions;

template <typename> struct DfrMaterialData;
template <bool Detached> using DfrMaterialT = DfrMaterialData<FloatT<Detached>>;
using DfrMaterialAD = DfrMaterialT<false>;
using DfrMaterial = DfrMaterialT<true>;

template <typename> struct DfrStatesData;
template <bool Detached> using DfrStatesT = DfrStatesData<FloatT<Detached>>;
using DfrStatesAD = DfrStatesT<false>;
using DfrStates = DfrStatesT<true>;

template <typename> struct DfrCoherentUtdStatesData;
template <bool Detached> using DfrCoherentUtdStatesT = DfrCoherentUtdStatesData<FloatT<Detached>>;
using DfrCoherentUtdStatesAD = DfrCoherentUtdStatesT<false>;
using DfrCoherentUtdStates = DfrCoherentUtdStatesT<true>;

template <typename> struct DfrCoherentEdgeData;
template <bool Detached> using DfrCoherentEdgeT = DfrCoherentEdgeData<FloatT<Detached>>;
using DfrCoherentEdgeAD = DfrCoherentEdgeT<false>;
using DfrCoherentEdge = DfrCoherentEdgeT<true>;

template <typename> struct DfrCoherentCandidatePairsData;
template <bool Detached> using DfrCoherentCandidatePairsT = DfrCoherentCandidatePairsData<FloatT<Detached>>;
using DfrCoherentCandidatePairsAD = DfrCoherentCandidatePairsT<false>;
using DfrCoherentCandidatePairs = DfrCoherentCandidatePairsT<true>;

template <typename> struct DfrAccumData;
template <bool Detached> using DfrAccumT = DfrAccumData<FloatT<Detached>>;
using DfrAccumAD = DfrAccumT<false>;
using DfrAccum = DfrAccumT<true>;

template <typename> struct DfrCoherentAccumData;
template <bool Detached> using DfrCoherentAccumT = DfrCoherentAccumData<FloatT<Detached>>;
using DfrCoherentAccumAD = DfrCoherentAccumT<false>;
using DfrCoherentAccum = DfrCoherentAccumT<true>;

template <typename> struct DfrPathsData;
template <bool Detached> using DfrPathsT = DfrPathsData<FloatT<Detached>>;
using DfrPathsAD = DfrPathsT<false>;
using DfrPaths = DfrPathsT<true>;

template <typename> struct NearestPointEdgeData;
template <bool Detached> using NearestPointEdgeT = NearestPointEdgeData<FloatT<Detached>>;
using NearestPointEdgeAD = NearestPointEdgeT<false>;
using NearestPointEdge = NearestPointEdgeT<true>;

template <typename> struct NearestRayEdgeData;
template <bool Detached> using NearestRayEdgeT = NearestRayEdgeData<FloatT<Detached>>;
using NearestRayEdgeAD = NearestRayEdgeT<false>;
using NearestRayEdge = NearestRayEdgeT<true>;

template <typename> struct NearestEdgesTopKData;
template <bool Detached> using NearestEdgesTopKT = NearestEdgesTopKData<FloatT<Detached>>;
using NearestEdgesTopKAD = NearestEdgesTopKT<false>;
using NearestEdgesTopK = NearestEdgesTopKT<true>;

template <typename> struct SegmentVisibilityData;
template <bool Detached> using SegmentVisibilityT = SegmentVisibilityData<FloatT<Detached>>;
using SegmentVisibilityAD = SegmentVisibilityT<false>;
using SegmentVisibility = SegmentVisibilityT<true>;

template <typename> struct SegmentPairVisibilityData;
template <bool Detached> using SegmentPairVisibilityT = SegmentPairVisibilityData<FloatT<Detached>>;
using SegmentPairVisibilityAD = SegmentPairVisibilityT<false>;
using SegmentPairVisibility = SegmentPairVisibilityT<true>;

template <typename> struct AxialEdgeVisibilityData;
template <bool Detached> using AxialEdgeVisibilityT = AxialEdgeVisibilityData<FloatT<Detached>>;
using AxialEdgeVisibilityAD = AxialEdgeVisibilityT<false>;
using AxialEdgeVisibility = AxialEdgeVisibilityT<true>;

template <typename> struct SegmentChainVisibilityData;
template <bool Detached> using SegmentChainVisibilityT = SegmentChainVisibilityData<FloatT<Detached>>;
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

namespace rayd {

/// Keep only the lanes where \p active is true, packed contiguously.
template <typename ArrayD, typename Mask_> DRJIT_INLINE ArrayD compressD(const ArrayD& array, const Mask_& active) {
    auto idx = compress(active);
    return gather<ArrayD>(array, idx);
}

/// Number of lanes (the batch width) of a scalar or vector Dr.Jit array.
template <typename ArrayD> DRJIT_INLINE size_t slices(const ArrayD& cuda_array) {
    if constexpr (depth_v<ArrayD> == 1) {
        return cuda_array.size();
    } else {
        return cuda_array[0].size();
    }
}

/// Copy each component of an n-wide device array into the matching host vector.
template <typename T, size_t n, bool async = false>
DRJIT_INLINE void copy_cuda_array(const Array<CUDAArray<T>, n>& cuda_array, std::array<std::vector<T>, n>& cpu_array) {
    size_t m = slices<Array<CUDAArray<T>, n>>(cuda_array);
    for (size_t i = 0; i < n; ++i) {
        cpu_array[i].resize(m);
        drjit::store(cpu_array[i].data(), cuda_array[i]);
    }
}

/// Interpolate a 3D attribute over a triangle in edge-vector form: p0 + s*e1 + t*e2.
template <bool Detached>
DRJIT_INLINE Vector3fT<Detached> bilinear(const Vector3fT<Detached>& p0, const Vector3fT<Detached>& e1,
                                          const Vector3fT<Detached>& e2, const Vector2fT<Detached>& st) {
    return fmadd(e1, st.x(), fmadd(e2, st.y(), p0));
}

/// Interpolate a 2D attribute over a triangle in edge-vector form: p0 + s*e1 + t*e2.
template <bool Detached>
DRJIT_INLINE Vector2fT<Detached> bilinear2(const Vector2fT<Detached>& p0, const Vector2fT<Detached>& e1,
                                           const Vector2fT<Detached>& e2, const Vector2fT<Detached>& st) {
    return fmadd(e1, st.x(), fmadd(e2, st.y(), p0));
}

/// \brief Möller-Trumbore ray-triangle intersection for a triangle in edge-vector form.
///
/// \tparam Detached  When true, operate on detached (non-AD) arrays.
/// \param p0   First triangle vertex.
/// \param e1   Edge vector to the second vertex.
/// \param e2   Edge vector to the third vertex.
/// \param ray  RayAD batch to test.
/// \return Pair of (barycentric (u, v), hit distance t); t is Infinity on degenerate triangles.
template <bool Detached>
DRJIT_INLINE auto ray_intersect_triangle(const Vector3fT<Detached>& p0, const Vector3fT<Detached>& e1,
                                         const Vector3fT<Detached>& e2, const RayT<Detached>& ray) {
    Vector3fT<Detached> h = cross(ray.d, e2);
    FloatT<Detached> a = dot(e1, h);
    MaskT<Detached> valid = neq(a, 0.f);
    FloatT<Detached> safe_a = select(valid, a, 1.f);
    FloatT<Detached> f = rcp(safe_a);
    Vector3fT<Detached> s = ray.o - p0;
    FloatT<Detached> u = f * dot(s, h);
    Vector3fT<Detached> q = cross(s, e1);
    FloatT<Detached> v = f * dot(ray.d, q);
    FloatT<Detached> t = f * dot(e2, q);
    u = select(valid, u, 0.f);
    v = select(valid, v, 0.f);
    t = select(valid, t, full<FloatT<Detached>>(Infinity, slices(ray.o)));
    return std::make_pair(Vector2fT<Detached>(u, v), t);
}

/// Clamp a value to the unit interval [0, 1].
template <typename Float_> DRJIT_INLINE Float_ clamp01(const Float_& value) {
    return maximum(minimum(value, Float_(1.f)), Float_(0.f));
}

/// \brief Closest point on the segment [p0, p0 + e1] to \p point.
///
/// \return Tuple of (segment parameter in [0, 1], closest point, squared distance).
template <bool Detached>
DRJIT_INLINE auto closest_point_on_segment(const Vector3fT<Detached>& point, const Vector3fT<Detached>& p0,
                                           const Vector3fT<Detached>& e1) {
    const FloatT<Detached> edge_length_sq = squared_norm(e1);
    const MaskT<Detached> valid_edge = edge_length_sq > Epsilon;
    const FloatT<Detached> safe_edge_length_sq = select(valid_edge, edge_length_sq, FloatT<Detached>(1.f));
    const FloatT<Detached> edge_t =
        select(valid_edge, clamp01<FloatT<Detached>>(dot(point - p0, e1) / safe_edge_length_sq), FloatT<Detached>(0.f));
    const Vector3fT<Detached> edge_point = fmadd(e1, edge_t, p0);
    const FloatT<Detached> distance_sq = squared_norm(point - edge_point);
    return std::make_tuple(edge_t, edge_point, distance_sq);
}

/// \brief Closest pair of points between two finite segments.
///
/// Both segments are given in origin + vector form: the query segment spans
/// [query_origin, query_origin + query_edge] and the target spans
/// [edge_origin, edge_origin + edge_vector]. Endpoint and interior cases are all
/// evaluated and the nearest is returned, so the result is robust to parallel
/// and degenerate inputs.
///
/// \return Tuple of (query_t, query_point, edge_t, edge_point, squared distance),
///         where the parameters lie in [0, 1] along their respective segments.
template <bool Detached>
DRJIT_INLINE auto closest_segment_segment(const Vector3fT<Detached>& query_origin,
                                          const Vector3fT<Detached>& query_edge, const Vector3fT<Detached>& edge_origin,
                                          const Vector3fT<Detached>& edge_vector) {
    const Vector3fT<Detached> w0 = query_origin - edge_origin;
    const Vector3fT<Detached> query_end = query_origin + query_edge;
    const Vector3fT<Detached> edge_end = edge_origin + edge_vector;

    const FloatT<Detached> a = squared_norm(query_edge);
    const FloatT<Detached> b = dot(query_edge, edge_vector);
    const FloatT<Detached> c = squared_norm(edge_vector);
    const FloatT<Detached> d = dot(query_edge, w0);
    const FloatT<Detached> e = dot(edge_vector, w0);
    const FloatT<Detached> det = fmsub(a, c, b * b);

    FloatT<Detached> best_distance_sq = full<FloatT<Detached>>(Infinity, slices(query_origin));
    FloatT<Detached> best_query_t = zeros<FloatT<Detached>>(slices(query_origin));
    FloatT<Detached> best_edge_t = zeros<FloatT<Detached>>(slices(query_origin));
    Vector3fT<Detached> best_query_point = query_origin;
    Vector3fT<Detached> best_edge_point = edge_origin;

    auto update = [&](const MaskT<Detached>& mask, const FloatT<Detached>& query_t, const FloatT<Detached>& edge_t) {
        const Vector3fT<Detached> query_point = fmadd(query_edge, query_t, query_origin);
        const Vector3fT<Detached> edge_point = fmadd(edge_vector, edge_t, edge_origin);
        const FloatT<Detached> distance_sq = squared_norm(query_point - edge_point);
        const MaskT<Detached> better = mask && (distance_sq < best_distance_sq);
        best_distance_sq = select(better, distance_sq, best_distance_sq);
        best_query_t = select(better, query_t, best_query_t);
        best_edge_t = select(better, edge_t, best_edge_t);
        best_query_point = select(better, query_point, best_query_point);
        best_edge_point = select(better, edge_point, best_edge_point);
    };

    {
        FloatT<Detached> edge_t;
        Vector3fT<Detached> edge_point;
        FloatT<Detached> distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<Detached>(query_origin, edge_origin, edge_vector);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(query_origin)), FloatT<Detached>(0.f), edge_t);
    }

    {
        FloatT<Detached> edge_t;
        Vector3fT<Detached> edge_point;
        FloatT<Detached> distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<Detached>(query_end, edge_origin, edge_vector);
        DRJIT_MARK_USED(edge_point);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(query_origin)), FloatT<Detached>(1.f), edge_t);
    }

    {
        FloatT<Detached> query_t;
        Vector3fT<Detached> query_point;
        FloatT<Detached> distance_sq;
        std::tie(query_t, query_point, distance_sq) =
            closest_point_on_segment<Detached>(edge_origin, query_origin, query_edge);
        DRJIT_MARK_USED(query_point);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(query_origin)), query_t, FloatT<Detached>(0.f));
    }

    {
        FloatT<Detached> query_t;
        Vector3fT<Detached> query_point;
        FloatT<Detached> distance_sq;
        std::tie(query_t, query_point, distance_sq) =
            closest_point_on_segment<Detached>(edge_end, query_origin, query_edge);
        DRJIT_MARK_USED(query_point);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(query_origin)), query_t, FloatT<Detached>(1.f));
    }

    const MaskT<Detached> interior = (a > Epsilon) && (c > Epsilon) && (abs(det) > Epsilon);
    const FloatT<Detached> safe_det = select(interior, det, FloatT<Detached>(1.f));
    const FloatT<Detached> query_t_line = (b * e - c * d) / safe_det;
    const FloatT<Detached> edge_t_line = (a * e - b * d) / safe_det;
    update(interior && query_t_line >= 0.f && query_t_line <= 1.f && edge_t_line >= 0.f && edge_t_line <= 1.f,
           query_t_line, edge_t_line);

    return std::make_tuple(best_query_t, best_query_point, best_edge_t, best_edge_point, best_distance_sq);
}

/// \brief Closest pair of points between a ray and a finite segment.
///
/// The ray spans [ray_origin, ray_origin + t * ray_direction] for t >= 0; the
/// segment spans [edge_origin, edge_origin + edge_vector].
///
/// \return Tuple of (ray_t >= 0, ray_point, edge_t in [0, 1], edge_point, squared distance).
template <bool Detached>
DRJIT_INLINE auto closest_ray_segment(const Vector3fT<Detached>& ray_origin, const Vector3fT<Detached>& ray_direction,
                                      const Vector3fT<Detached>& edge_origin, const Vector3fT<Detached>& edge_vector) {
    const Vector3fT<Detached> w0 = ray_origin - edge_origin;
    const Vector3fT<Detached> edge_end = edge_origin + edge_vector;

    const FloatT<Detached> a = squared_norm(ray_direction);
    const FloatT<Detached> b = dot(ray_direction, edge_vector);
    const FloatT<Detached> c = squared_norm(edge_vector);
    const FloatT<Detached> d = dot(ray_direction, w0);
    const FloatT<Detached> e = dot(edge_vector, w0);
    const FloatT<Detached> det = fmsub(a, c, b * b);

    FloatT<Detached> best_distance_sq = full<FloatT<Detached>>(Infinity, slices(ray_origin));
    FloatT<Detached> best_query_t = zeros<FloatT<Detached>>(slices(ray_origin));
    FloatT<Detached> best_edge_t = zeros<FloatT<Detached>>(slices(ray_origin));
    Vector3fT<Detached> best_query_point = ray_origin;
    Vector3fT<Detached> best_edge_point = edge_origin;

    auto update = [&](const MaskT<Detached>& mask, const FloatT<Detached>& query_t, const FloatT<Detached>& edge_t) {
        const Vector3fT<Detached> query_point = fmadd(ray_direction, query_t, ray_origin);
        const Vector3fT<Detached> edge_point = fmadd(edge_vector, edge_t, edge_origin);
        const FloatT<Detached> distance_sq = squared_norm(query_point - edge_point);
        const MaskT<Detached> better = mask && (distance_sq < best_distance_sq);
        best_distance_sq = select(better, distance_sq, best_distance_sq);
        best_query_t = select(better, query_t, best_query_t);
        best_edge_t = select(better, edge_t, best_edge_t);
        best_query_point = select(better, query_point, best_query_point);
        best_edge_point = select(better, edge_point, best_edge_point);
    };

    {
        FloatT<Detached> edge_t;
        Vector3fT<Detached> edge_point;
        FloatT<Detached> distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<Detached>(ray_origin, edge_origin, edge_vector);
        DRJIT_MARK_USED(edge_point);
        DRJIT_MARK_USED(distance_sq);
        update(full<MaskT<Detached>>(true, slices(ray_origin)), FloatT<Detached>(0.f), edge_t);
    }

    const MaskT<Detached> valid_ray = a > Epsilon;
    const FloatT<Detached> safe_a = select(valid_ray, a, FloatT<Detached>(1.f));
    update(full<MaskT<Detached>>(true, slices(ray_origin)),
           select(valid_ray, maximum(-d / safe_a, FloatT<Detached>(0.f)), FloatT<Detached>(0.f)),
           FloatT<Detached>(0.f));
    update(full<MaskT<Detached>>(true, slices(ray_origin)),
           select(valid_ray, maximum((b - d) / safe_a, FloatT<Detached>(0.f)), FloatT<Detached>(0.f)),
           FloatT<Detached>(1.f));

    const MaskT<Detached> interior = valid_ray && (c > Epsilon) && (abs(det) > Epsilon);
    const FloatT<Detached> safe_det = select(interior, det, FloatT<Detached>(1.f));
    const FloatT<Detached> query_t_line = (b * e - c * d) / safe_det;
    const FloatT<Detached> edge_t_line = (a * e - b * d) / safe_det;
    update(interior && query_t_line >= 0.f && edge_t_line >= 0.f && edge_t_line <= 1.f, query_t_line, edge_t_line);

    return std::make_tuple(best_query_t, best_query_point, best_edge_t, best_edge_point, best_distance_sq);
}

/// Squared distance from \p point to an axis-aligned box; zero when inside.
template <typename Float_>
DRJIT_INLINE auto point_aabb_distance_sq(const Array<Float_, 3>& point, const Array<Float_, 3>& bbox_min,
                                         const Array<Float_, 3>& bbox_max) {
    const Array<Float_, 3> clamped = maximum(minimum(point, bbox_max), bbox_min);
    return squared_norm(point - clamped);
}

/// Conservative lower bound on the squared distance from a ray to an AABB (BVH pruning).
template <typename Float_>
DRJIT_INLINE auto ray_aabb_lower_bound_sq(const Array<Float_, 3>& origin, const Array<Float_, 3>& direction,
                                          const Array<Float_, 3>& bbox_min, const Array<Float_, 3>& bbox_max);

/// Lower bound on the squared distance from an infinite line to an AABB, via the
/// box's bounding sphere; conservative and cheap, used to prune BVH branches.
template <typename Float_>
DRJIT_INLINE auto line_aabb_sphere_lower_bound_sq(const Array<Float_, 3>& origin, const Array<Float_, 3>& direction,
                                                  const Array<Float_, 3>& bbox_min, const Array<Float_, 3>& bbox_max) {
    const Float_ direction_length_sq = squared_norm(direction);
    const mask_t<Float_> valid_direction = direction_length_sq > Epsilon;
    const Float_ safe_direction_length_sq = select(valid_direction, direction_length_sq, Float_(1.f));

    const Array<Float_, 3> bbox_center = (bbox_min + bbox_max) * Float_(0.5f);
    const Array<Float_, 3> half_extent = (bbox_max - bbox_min) * Float_(0.5f);
    const Float_ line_t = dot(bbox_center - origin, direction) / safe_direction_length_sq;
    const Array<Float_, 3> closest_point = fmadd(direction, line_t, origin);
    const Float_ center_distance = sqrt(maximum(squared_norm(bbox_center - closest_point), Float_(0.f)));
    const Float_ sphere_radius = sqrt(maximum(squared_norm(half_extent), Float_(0.f)));
    const Float_ separation = maximum(center_distance - sphere_radius, Float_(0.f));
    return select(valid_direction, separation * separation, Float_(0.f));
}

/// Conservative lower bound on the squared distance from a finite segment to an AABB.
template <typename Float_>
DRJIT_INLINE auto segment_aabb_lower_bound_sq(const Array<Float_, 3>& origin, const Array<Float_, 3>& segment,
                                              const Array<Float_, 3>& bbox_min, const Array<Float_, 3>& bbox_max) {
    const Array<Float_, 3> segment_end = origin + segment;
    const Array<Float_, 3> path_min = minimum(origin, segment_end);
    const Array<Float_, 3> path_max = maximum(origin, segment_end);
    const Array<Float_, 3> below = maximum(bbox_min - path_max, Array<Float_, 3>(0.f));
    const Array<Float_, 3> above = maximum(path_min - bbox_max, Array<Float_, 3>(0.f));
    const Float_ path_bbox_bound = squared_norm(below + above);
    const Float_ direction_bound = line_aabb_sphere_lower_bound_sq(origin, segment, bbox_min, bbox_max);
    return maximum(path_bbox_bound, direction_bound);
}

template <typename Float_>
DRJIT_INLINE auto ray_aabb_lower_bound_sq(const Array<Float_, 3>& origin, const Array<Float_, 3>& direction,
                                          const Array<Float_, 3>& bbox_min, const Array<Float_, 3>& bbox_max) {
    using Mask_ = mask_t<Float_>;

    auto axis_distance = [](const Float_& o, const Float_& d, const Float_& axis_min, const Float_& axis_max) {
        const Mask_ positive = d > Epsilon;
        const Mask_ negative = d < -Epsilon;
        const Mask_ stationary = !(positive || negative);

        Float_ delta = zeros<Float_>(slices(o));
        delta = select(positive, maximum(o - axis_max, Float_(0.f)), delta);
        delta = select(negative, maximum(axis_min - o, Float_(0.f)), delta);
        delta = select(stationary, maximum(axis_min - o, Float_(0.f)) + maximum(o - axis_max, Float_(0.f)), delta);
        return delta;
    };

    const Float_ dx = axis_distance(origin.x(), direction.x(), bbox_min.x(), bbox_max.x());
    const Float_ dy = axis_distance(origin.y(), direction.y(), bbox_min.y(), bbox_max.y());
    const Float_ dz = axis_distance(origin.z(), direction.z(), bbox_min.z(), bbox_max.z());
    const Float_ axis_bound = fmadd(dx, dx, fmadd(dy, dy, dz * dz));
    const Float_ direction_bound = line_aabb_sphere_lower_bound_sq(origin, direction, bbox_min, bbox_max);
    return maximum(axis_bound, direction_bound);
}

} // namespace rayd

namespace rayd {

/// Batch of rays, each with an origin, direction, and maximum parametric extent.
template <typename Float_> struct RayData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;

    RayData(const Vec3f& origin, const Vec3f& direction, const Float_& t_max) : o(origin), d(direction), tmax(t_max) {}

    /// Construct rays with unbounded extent (tmax = Infinity), one entry per direction lane.
    RayData(const Vec3f& origin, const Vec3f& direction) : o(origin), d(direction) {
        tmax = drjit::full<Float_>(Infinity, slices<Vec3f>(direction));
    }

    /// Same ray with the direction flipped; origin and tmax are unchanged.
    RayData reversed() const { return RayData<Float_>(o, -d, tmax); }

    int size() const { return tmax.size(); }

    /// Point at parametric distance \p t along the ray: o + t * d.
    Vec3f operator()(const Float_& t) const { return drjit::fmadd(d, t, o); }

    Vec3f o = drjit::zeros<Vec3f>(1);               ///< RayAD origins.
    Vec3f d = Vec3f(0.f, 0.f, 1.f);                 ///< RayAD directions (not required to be normalized).
    Float_ tmax = drjit::full<Float_>(Infinity, 1); ///< Maximum t; hits beyond this are ignored.

    DRJIT_STRUCT(RayData, o, d, tmax)
};

/// Bit flags selecting which intersection fields Scene::intersect() computes.
enum class RayFlags : uint32_t {
    None = static_cast<uint32_t>(shared::RayFlagBits::None),
    Geometric = static_cast<uint32_t>(shared::RayFlagBits::Geometric), // t, p, barycentric, ids, geo_n
    ShadingN = static_cast<uint32_t>(shared::RayFlagBits::ShadingN),   // interpolated shading normal (n)
    UV = static_cast<uint32_t>(shared::RayFlagBits::UV),               // interpolated texture UV (uv)
    All = Geometric | ShadingN | UV,
};

static_assert(static_cast<uint32_t>(RayFlags::All) == static_cast<uint32_t>(shared::RayFlagBits::All));
static_assert(static_cast<std::uint8_t>(shared::IntersectionField::GlobalPrimId) == 9u);

inline constexpr RayFlags operator|(RayFlags a, RayFlags b) {
    return static_cast<RayFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr RayFlags operator&(RayFlags a, RayFlags b) {
    return static_cast<RayFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr bool has_flag(RayFlags set, RayFlags flag) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(flag)) != 0;
}

/// Result of a ray-triangle intersection query, one entry per input ray.
template <typename Float_> struct IntersectionData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Vec2f = std::conditional_t<IsDetached, Vector2f, Vector2fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    /// Per-lane mask of lanes that hit a surface (prim_id >= 0).
    Mask_ is_valid() const { return prim_id >= 0; }

    Float_ t = Infinity;                 ///< Hit distance along the ray; Infinity when no hit.
    Vec3f p = zeros<Vec3f>(1);           ///< World-space hit position.
    Vec3f n = zeros<Vec3f>(1);           ///< Interpolated shading normal (valid only if RayFlags::ShadingN requested).
    Vec3f geo_n = zeros<Vec3f>(1);       ///< Geometric face normal (valid only if RayFlags::Geometric requested).
    Vec2f uv = zeros<Vec2f>(1);          ///< Interpolated texture UV (valid only if RayFlags::UV requested).
    Vec3f barycentric = zeros<Vec3f>(1); ///< Barycentric coordinates of the hit within the triangle.
    Int_ shape_id = full<Int_>(shared::InvalidSignedId, 1); ///< Owning mesh id; -1 when no hit.
    Int_ prim_id = full<Int_>(shared::InvalidSignedId, 1);  ///< Face index within the owning mesh; -1 when no hit.
    Int_ local_prim_id =
        full<Int_>(shared::InvalidSignedId, 1); ///< Same as prim_id (face index within the owning mesh).
    Int_ global_prim_id = full<Int_>(shared::InvalidSignedId, 1); ///< Scene-global face index.

    DRJIT_STRUCT(IntersectionData, t, p, n, geo_n, uv, barycentric, shape_id, prim_id, local_prim_id, global_prim_id)
};

} // namespace rayd

namespace rayd {

/// Builders for 4x4 homogeneous transforms (column-vector convention, applied on the left).
namespace transform {

/// Translation matrix by \p vector.
template <typename Float_> drjit::Matrix<Float_, 4> translate(const drjit::Array<Float_, 3>& vector) {
    return drjit::translate<drjit::Matrix<Float_, 4>>(vector);
}

/// Non-uniform scale matrix with per-axis factors \p vector.
template <typename Float_> drjit::Matrix<Float_, 4> scale(const drjit::Array<Float_, 3>& vector) {
    return drjit::scale<drjit::Matrix<Float_, 4>>(vector);
}

/// Rotation matrix about \p axis by \p angle_degrees (degrees).
template <typename Float_> drjit::Matrix<Float_, 4> rotate(const drjit::Array<Float_, 3>& axis, float angle_degrees) {
    return drjit::rotate<drjit::Matrix<Float_, 4>>(axis, drjit::deg_to_rad(angle_degrees));
}

/// Perspective projection from a vertical field of view and near/far clip planes.
inline ScalarMatrix4f perspective(float fov_degrees, float near_clip, float far_clip) {
    const float reciprocal_depth = 1.f / (far_clip - near_clip);
    const float tangent = drjit::tan(drjit::deg_to_rad(fov_degrees * 0.5f));
    const float cotangent = 1.f / tangent;

    ScalarMatrix4f transform = drjit::diag(ScalarVector4f(cotangent, cotangent, far_clip * reciprocal_depth, 0.f));
    transform(2, 3) = -near_clip * far_clip * reciprocal_depth;
    transform(3, 2) = 1.f;
    return transform;
}

/// Perspective projection from pinhole-camera intrinsics (focal lengths and principal point).
inline ScalarMatrix4f perspective_intrinsic(float fx, float fy, float cx, float cy, float near_clip, float far_clip) {
    const float reciprocal_depth = 1.f / (far_clip - near_clip);

    ScalarMatrix4f transform = drjit::diag(ScalarVector4f(1.f, 1.f, far_clip * reciprocal_depth, 0.f));
    transform(2, 3) = -near_clip * far_clip * reciprocal_depth;
    transform(3, 2) = 1.f;

    return translate(ScalarVector3f(1.f - 2.f * cx, 1.f - 2.f * cy, 0.f)) *
           scale(ScalarVector3f(2.f * fx, 2.f * fy, 1.f)) * transform;
}

/// Orthographic projection mapping z from [near_clip, far_clip] into [0, 1].
inline ScalarMatrix4f orthographic(float near_clip, float far_clip) {
    return scale(drjit::Array<float, 3>(1.f, 1.f, 1.f / (far_clip - near_clip))) *
           translate(drjit::Array<float, 3>(0.f, 0.f, -near_clip));
}

/// Camera-to-world matrix placing the camera at \p origin looking toward \p target,
/// with \p up defining the camera roll.
template <typename Float_>
drjit::Matrix<Float_, 4> look_at(const drjit::Array<Float_, 3>& origin, const drjit::Array<Float_, 3>& target,
                                 const drjit::Array<Float_, 3>& up) {
    const drjit::Array<Float_, 3> direction = drjit::normalize(target - origin);
    const drjit::Array<Float_, 3> left = drjit::normalize(drjit::cross(up, direction));
    const drjit::Array<Float_, 3> new_up = drjit::cross(direction, left);
    const drjit::Array<Float_, 1> z(0);

    return drjit::transpose(drjit::Matrix<Float_, 4>(drjit::concat(left, z), drjit::concat(new_up, z),
                                                     drjit::concat(direction, z),
                                                     drjit::Array<Float_, 4>(origin[0], origin[1], origin[2], 1.f)));
}

} // namespace transform

/// Transform a position by a 4x4 matrix (w = 1) and perspective-divide back to 3D.
template <typename Float_>
drjit::Array<Float_, 3> transform_pos(const drjit::Matrix<Float_, 4>& matrix, const drjit::Array<Float_, 3>& vector) {
    const drjit::Array<Float_, 4> transformed = matrix * drjit::concat(vector, 1.f);
    return drjit::head<3>(transformed) / transformed.w();
}

/// Transform a direction by a 4x4 matrix (w = 0); ignores translation, no normalization.
template <typename Float_>
drjit::Array<Float_, 3> transform_dir(const drjit::Matrix<Float_, 4>& matrix, const drjit::Array<Float_, 3>& vector) {
    return drjit::head<3>(matrix * drjit::concat(vector, 0.f));
}

/// Transform a 2D position by a 3x3 matrix (w = 1) and perspective-divide back to 2D.
template <typename Float_>
drjit::Array<Float_, 2> transform2d_pos(const drjit::Matrix<Float_, 3>& matrix, const drjit::Array<Float_, 2>& vector) {
    const drjit::Array<Float_, 3> transformed = matrix * drjit::Array<Float_, 3>(vector[0], vector[1], 1.f);
    return drjit::head<2>(transformed) / transformed.z();
}

} // namespace rayd
