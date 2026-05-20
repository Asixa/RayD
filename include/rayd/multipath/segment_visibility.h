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

template <typename Float_>
struct SegmentChainVisibilityData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;

    int chain_count = 0;
    int max_segments = 0;
    Mask_ all_visible = full<Mask_>(false, 1);
    Int_ first_blocked_segment = full<Int_>(-1, 1);
    Int_ first_blocked_prim = full<Int_>(-1, 1);

    DRJIT_STRUCT(SegmentChainVisibilityData,
                 all_visible,
                 first_blocked_segment,
                 first_blocked_prim)
};

template <bool Detached>
using SegmentChainVisibilityT = SegmentChainVisibilityData<FloatT<Detached>>;

using SegmentChainVisibility = SegmentChainVisibilityT<false>;
using SegmentChainVisibilityDetached = SegmentChainVisibilityT<true>;

} // namespace rayd
