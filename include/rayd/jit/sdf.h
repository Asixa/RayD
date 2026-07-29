// Copyright Xingyu Chen.
// Declares the Dr.Jit signed-distance-field query API.

#pragma once

#include <cstdint>

#include <rayd/jit/core.h>
#include <rayd/jit/reflection.h>

namespace rayd {

class MixedScene;

struct SdfTraceOptions {
    int max_steps = 64;
    float relaxation = 0.9f;
    float eps_hit = -1.0f;
};

template <typename Float_> struct SdfIntersectionData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    Mask_ is_valid() const { return hit_mask; }

    Float_ t = full<Float_>(Infinity, 1);
    Mask_ hit_mask = full<Mask_>(false, 1);
    Vec3f position = zeros<Vec3f>(1);
    Vec3f normal = zeros<Vec3f>(1);
    Int_ steps = zeros<Int_>(1);

    DRJIT_STRUCT(SdfIntersectionData, t, hit_mask, position, normal, steps)
};

template <bool Detached> using SdfIntersectionT = SdfIntersectionData<FloatT<Detached>>;

using SdfIntersection = SdfIntersectionT<true>;
using SdfIntersectionAD = SdfIntersectionT<false>;

class SdfGrid {
  public:
    SdfGrid() = default;
    SdfGrid(const Float& values, int nx, int ny, int nz, const Vector3f& position, const Float& rotation,
            const Vector3f& scale);
    SdfGrid(const FloatAD& values, int nx, int ny, int nz, const Vector3fAD& position, const FloatAD& rotation,
            const Vector3fAD& scale);

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int value_count() const { return nx_ * ny_ * nz_; }

    const FloatAD& values() const { return values_; }
    const Vector3fAD& position() const { return position_; }
    const FloatAD& rotation() const { return rotation_; }
    const Vector3fAD& scale() const { return scale_; }

    template <bool Detached>
    SdfIntersectionT<Detached> intersect(const RayT<Detached>& ray, const SdfTraceOptions& options = {},
                                         MaskT<Detached> active = true) const;

    template <bool Detached>
    MaskT<Detached> visible(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                            const SdfTraceOptions& options = {}, MaskT<Detached> active = true) const;

    template <bool Detached>
    ReflectionChainT<Detached> trace_reflections(const RayT<Detached>& ray, int max_bounces,
                                                 const SdfTraceOptions& options = {},
                                                 MaskT<Detached> active = true) const;

  private:
    friend class MixedScene;
    template <bool Detached> FloatT<Detached> query_bias(const SdfTraceOptions& options, int ray_count) const;

    void initialize(const FloatAD& values, int nx, int ny, int nz, const Vector3fAD& position, const FloatAD& rotation,
                    const Vector3fAD& scale);

    FloatAD values_;
    Vector3fAD position_;
    FloatAD rotation_;
    Vector3fAD scale_;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
};

template <bool Detached>
SdfIntersectionT<Detached> sdf_intersect(const SdfGrid& grid, const RayT<Detached>& ray,
                                         const SdfTraceOptions& options = {}, MaskT<Detached> active = true) {
    return grid.intersect<Detached>(ray, options, active);
}

} // namespace rayd
