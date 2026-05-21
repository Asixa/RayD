#pragma once

#include <rayd/rayd.h>

namespace rayd {

template <typename Float_>
struct ReflectionEpcResultData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;
    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;

    int ray_count = 0;
    int max_bounces = 0;

    Mask_ valid = full<Mask_>(false, 1);
    Int_ bounce_count = full<Int_>(0, 1);
    Float_ path_length = full<Float_>(Infinity, 1);
    Vec3f reflection_points = zeros<Vec3f>(1);
    Int_ prim_ids = full<Int_>(-1, 1);
    Int_ first_blocked_segment = full<Int_>(-1, 1);
    Int_ first_blocked_prim = full<Int_>(-1, 1);

    DRJIT_STRUCT(ReflectionEpcResultData,
                 valid,
                 bounce_count,
                 path_length,
                 reflection_points,
                 prim_ids,
                 first_blocked_segment,
                 first_blocked_prim)
};

template <bool Detached>
using ReflectionEpcResultT = ReflectionEpcResultData<FloatT<Detached>>;

using ReflectionEpcResult = ReflectionEpcResultT<false>;
using ReflectionEpcResultDetached = ReflectionEpcResultT<true>;

} // namespace rayd
