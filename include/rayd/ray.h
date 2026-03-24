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

} // namespace rayd
