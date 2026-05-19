#pragma once

#include <rayd/rayd.h>

namespace rayd {

template <typename Float_>
struct SegmentVisibilityData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;

    int ray_count = 0;
    Mask_ visible = full<Mask_>(false, 1);

    DRJIT_STRUCT(SegmentVisibilityData, visible)
};

template <bool Detached>
using SegmentVisibilityT = SegmentVisibilityData<FloatT<Detached>>;

using SegmentVisibility = SegmentVisibilityT<false>;
using SegmentVisibilityDetached = SegmentVisibilityT<true>;

template <typename Float_>
struct SegmentPairVisibilityData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;

    int ray_count = 0;
    Mask_ visible_a = full<Mask_>(false, 1);
    Mask_ visible_b = full<Mask_>(false, 1);

    DRJIT_STRUCT(SegmentPairVisibilityData, visible_a, visible_b)
};

template <bool Detached>
using SegmentPairVisibilityT = SegmentPairVisibilityData<FloatT<Detached>>;

using SegmentPairVisibility = SegmentPairVisibilityT<false>;
using SegmentPairVisibilityDetached = SegmentPairVisibilityT<true>;

template <typename Float_>
struct AxialEdgeVisibilityData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;

    int state_count = 0;
    Mask_ any_visible = full<Mask_>(false, 1);

    DRJIT_STRUCT(AxialEdgeVisibilityData, any_visible)
};

template <bool Detached>
using AxialEdgeVisibilityT = AxialEdgeVisibilityData<FloatT<Detached>>;

using AxialEdgeVisibility = AxialEdgeVisibilityT<false>;
using AxialEdgeVisibilityDetached = AxialEdgeVisibilityT<true>;

} // namespace rayd
