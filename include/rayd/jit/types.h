// Copyright Xingyu Chen.
// Declares the Dr.Jit types API.

#pragma once

#include <cstdint>
#include <type_traits>

#include <drjit/array.h>
#include <drjit/autodiff.h>
#include <drjit/jit.h>
#include <drjit/matrix.h>

namespace rayd {

using namespace drjit;

// Core Dr.Jit type aliases. Every array type comes in two flavors: the bare name
// is the non-AD (detached) type and the "AD" suffix is the autodiff-enabled
// CUDADiffArray (e.g. Float / FloatAD). Width-batched templates take a
// `bool Detached_` parameter (true = non-AD) and resolve to the matching flavor
// via the FloatT / IntT / MaskT selectors.

/// Detached (non-AD) counterpart of a Dr.Jit type.
template <typename T>
using Detached = drjit::detached_t<T>;

using FloatAD = drjit::CUDADiffArray<float>;
using Float = Detached<FloatAD>;

using IntAD = drjit::CUDADiffArray<int32_t>;
using Int = Detached<IntAD>;

using UIntAD = drjit::CUDADiffArray<uint32_t>;
using UInt = Detached<UIntAD>;

using UInt64AD = drjit::CUDADiffArray<uint64_t>;
using UInt64 = Detached<UInt64AD>;

template <bool Detached_>
using FloatT = std::conditional_t<Detached_, Float, FloatAD>;

template <bool Detached_>
using IntT = std::conditional_t<Detached_, Int, IntAD>;

using ScalarFloat = drjit::scalar_t<float>;

template <int n, bool Detached_>
using VectorfT = drjit::Array<FloatT<Detached_>, n>;

template <int n, bool Detached_>
using VectoriT = drjit::Array<IntT<Detached_>, n>;

template <int n, bool Detached_>
using MatrixfT = drjit::Matrix<FloatT<Detached_>, n>;

template <bool Detached_>
using Vector1fT = VectorfT<1, Detached_>;

template <bool Detached_>
using Vector2fT = VectorfT<2, Detached_>;

template <bool Detached_>
using Vector2iT = VectoriT<2, Detached_>;

template <bool Detached_>
using Vector3fT = VectorfT<3, Detached_>;

template <bool Detached_>
using Vector3iT = VectoriT<3, Detached_>;

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

template <bool Detached_>
using Matrix4fT = MatrixfT<4, Detached_>;

using Matrix4fAD = Matrix4fT<false>;
using Matrix4f = Matrix4fT<true>;

using MaskAD = drjit::mask_t<FloatAD>;
using Mask = drjit::mask_t<Float>;

template <bool Detached_>
using MaskT = std::conditional_t<Detached_, Mask, MaskAD>;

using ScalarVector3f = drjit::Array<float, 3>;
using ScalarVector4f = drjit::Array<float, 4>;
using ScalarMatrix4f = drjit::Matrix<float, 4>;

/// Per-triangle cached geometry: edge-vector form of each face plus its normals.
template <typename Float_>
struct TriangleInfoData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = drjit::mask_t<Float_>;
    using Vec3f = drjit::Array<Float_, 3>;
    using Vec3i = drjit::Array<std::conditional_t<IsDetached, Int, IntAD>, 3>;

    Vec3f p0;            ///< First vertex; the triangle is parameterized as p0 + s*e1 + t*e2.
    Vec3f e1;            ///< Edge vector from vertex 0 to vertex 1.
    Vec3f e2;            ///< Edge vector from vertex 0 to vertex 2.
    Vec3f n0;            ///< Shading normal at vertex 0.
    Vec3f n1;            ///< Shading normal at vertex 1.
    Vec3f n2;            ///< Shading normal at vertex 2.
    Vec3f face_normal;   ///< Unit geometric face normal (cross(e1, e2) normalized).
    Vec3i face_indices;  ///< Vertex indices of the three corners.
    Float_ face_area;    ///< Triangle area in world units.

    DRJIT_STRUCT(TriangleInfoData, p0, e1, e2,
                                n0, n1, n2,
                                face_normal,
                                face_indices,
                                face_area)
};

using TriangleInfoAD = TriangleInfoData<FloatAD>;
using TriangleInfo = TriangleInfoData<Float>;

template <bool Detached_>
using TriangleInfoT = std::conditional_t<Detached_, TriangleInfo, TriangleInfoAD>;

/// Per-triangle UV coordinates: one 2D coordinate at each of the three corners.
template <typename Float_>
using TriangleUvData = drjit::Array<drjit::Array<Float_, 2>, 3>;

using TriangleUVAD = TriangleUvData<FloatAD>;
using TriangleUV = TriangleUvData<Float>;

template <bool Detached_>
using TriangleUVT = std::conditional_t<Detached_, TriangleUV, TriangleUVAD>;

/// Scene-global geometry: all meshes' vertices and faces flattened into one indexing space.
struct SceneGeometry {
    Vector3fAD vertices;          ///< World-space vertex positions, concatenated across meshes.
    Vector3i faces;     ///< Face vertex indices into the scene-global vertex buffer.
    Vector3fAD face_normal;       ///< Unit geometric normal per face.
    Int shape_id;       ///< Owning mesh id per face.
    Int local_prim_id;  ///< Face index within its owning mesh.
    Int global_prim_id; ///< Face index within the scene-global face buffer.

    int vertex_count() const { return vertices.x().size(); }
    int face_count() const { return global_prim_id.size(); }

    DRJIT_STRUCT(SceneGeometry,
                 vertices,
                 faces,
                 face_normal,
                 shape_id,
                 local_prim_id,
                 global_prim_id)
};

} // namespace rayd
