#pragma once

#include <raydi/raydi.h>

namespace raydi {

template <typename Float_>
struct IntersectionData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;
    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;
    using Vec2f = std::conditional_t<IsDetached, Vector2fDetached, Vector2f>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;

    Mask_ is_valid() const {
        return prim_id >= 0;
    }

    Float_ t = Infinity;
    Vec3f p = zeros<Vec3f>(1);
    Vec3f n = zeros<Vec3f>(1);
    Vec3f geo_n = zeros<Vec3f>(1);
    Vec2f uv = zeros<Vec2f>(1);
    Vec3f barycentric = zeros<Vec3f>(1);
    Int_ shape_id = full<Int_>(-1, 1);
    Int_ prim_id = full<Int_>(-1, 1);

    DRJIT_STRUCT(IntersectionData, t, p, n, geo_n, uv, barycentric, shape_id, prim_id)
};

} // namespace raydi
