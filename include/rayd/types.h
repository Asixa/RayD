#pragma once

#include <cstdint>
#include <type_traits>

#include <drjit/array.h>
#include <drjit/autodiff.h>
#include <drjit/jit.h>
#include <drjit/matrix.h>

namespace rayd {

using namespace drjit;

template <typename T>
using Detached = drjit::detached_t<T>;

using Float = drjit::CUDADiffArray<float>;
using FloatDetached = Detached<Float>;

using Int = drjit::CUDADiffArray<int32_t>;
using IntDetached = Detached<Int>;

using UInt = drjit::CUDADiffArray<uint32_t>;
using UIntDetached = Detached<UInt>;

using UInt64 = drjit::CUDADiffArray<uint64_t>;
using UInt64Detached = Detached<UInt64>;

template <bool Detached_>
using FloatT = std::conditional_t<Detached_, FloatDetached, Float>;

template <bool Detached_>
using IntT = std::conditional_t<Detached_, IntDetached, Int>;

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

using Vector1f = Vector1fT<false>;
using Vector1fDetached = Vector1fT<true>;

using Vector2f = Vector2fT<false>;
using Vector2fDetached = Vector2fT<true>;

using Vector2i = Vector2iT<false>;
using Vector2iDetached = Vector2iT<true>;

using Vector3f = Vector3fT<false>;
using Vector3fDetached = Vector3fT<true>;

using Vector3i = Vector3iT<false>;
using Vector3iDetached = Vector3iT<true>;

template <bool Detached_>
using Matrix4fT = MatrixfT<4, Detached_>;

using Matrix4f = Matrix4fT<false>;
using Matrix4fDetached = Matrix4fT<true>;

using Mask = drjit::mask_t<Float>;
using MaskDetached = drjit::mask_t<FloatDetached>;

template <bool Detached_>
using MaskT = std::conditional_t<Detached_, MaskDetached, Mask>;

using ScalarVector3f = drjit::Array<float, 3>;
using ScalarVector4f = drjit::Array<float, 4>;
using ScalarMatrix4f = drjit::Matrix<float, 4>;

template <typename Float_>
struct TriangleInfoData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = drjit::mask_t<Float_>;
    using Vec3f = drjit::Array<Float_, 3>;
    using Vec3i = drjit::Array<std::conditional_t<IsDetached, IntDetached, Int>, 3>;

    Vec3f p0;
    Vec3f e1;
    Vec3f e2;
    Vec3f n0;
    Vec3f n1;
    Vec3f n2;
    Vec3f face_normal;
    Vec3i face_indices;
    Float_ face_area;

    DRJIT_STRUCT(TriangleInfoData, p0, e1, e2,
                                n0, n1, n2,
                                face_normal,
                                face_indices,
                                face_area)
};

using TriangleInfo = TriangleInfoData<Float>;
using TriangleInfoDetached = TriangleInfoData<FloatDetached>;

template <bool Detached_>
using TriangleInfoT = std::conditional_t<Detached_, TriangleInfoDetached, TriangleInfo>;

template <typename Float_>
using TriangleUvData = drjit::Array<drjit::Array<Float_, 2>, 3>;

using TriangleUV = TriangleUvData<Float>;
using TriangleUVDetached = TriangleUvData<FloatDetached>;

template <bool Detached_>
using TriangleUVT = std::conditional_t<Detached_, TriangleUVDetached, TriangleUV>;

} // namespace rayd
