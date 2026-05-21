#pragma once

#include <rayd/rayd.h>

namespace rayd {

template <typename Float_>
struct RayData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;

    RayData(const Vec3f &origin, const Vec3f &direction, const Float_ &t_max)
        : o(origin), d(direction), tmax(t_max) {}

    RayData(const Vec3f &origin, const Vec3f &direction)
        : o(origin), d(direction) {
        tmax = drjit::full<Float_>(Infinity, slices<Vec3f>(direction));
    }

    RayData reversed() const { return RayData<Float_>(o, -d, tmax); }

    int size() const { return tmax.size(); }

    Vec3f operator()(const Float_ &t) const { return drjit::fmadd(d, t, o); }

    Vec3f o = drjit::zeros<Vec3f>(1);
    Vec3f d = Vec3f(0.f, 0.f, 1.f);
    Float_ tmax = drjit::full<Float_>(Infinity, 1);

    DRJIT_STRUCT(RayData, o, d, tmax)
};

enum class RayFlags : uint32_t {
    None      = 0x00,
    Geometric = 0x01,   // t, p, barycentric, shape_id, prim_id, geo_n
    ShadingN  = 0x02,   // interpolated shading normal (n)
    UV        = 0x04,   // interpolated texture UV (uv)
    All       = Geometric | ShadingN | UV,
};

inline constexpr RayFlags operator|(RayFlags a, RayFlags b) {
    return static_cast<RayFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr RayFlags operator&(RayFlags a, RayFlags b) {
    return static_cast<RayFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr bool has_flag(RayFlags set, RayFlags flag) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(flag)) != 0;
}

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
    Int_ local_prim_id = full<Int_>(-1, 1);
    Int_ global_prim_id = full<Int_>(-1, 1);

    DRJIT_STRUCT(IntersectionData,
                 t,
                 p,
                 n,
                 geo_n,
                 uv,
                 barycentric,
                 shape_id,
                 prim_id,
                 local_prim_id,
                 global_prim_id)
};

} // namespace rayd
