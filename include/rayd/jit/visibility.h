#pragma once

#include <rayd/jit/core.h>

namespace rayd {

/// Result of a batched segment-visibility query: one mask per segment.
template <typename Float_>
struct SegmentVisibilityData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    int ray_count = 0;
    Mask_ visible = full<Mask_>(false, 1);  ///< True where the segment endpoints are mutually unoccluded.

    DRJIT_STRUCT(SegmentVisibilityData, visible)
};

template <bool Detached>
using SegmentVisibilityT = SegmentVisibilityData<FloatT<Detached>>;

using SegmentVisibilityAD = SegmentVisibilityT<false>;
using SegmentVisibility = SegmentVisibilityT<true>;

/// Result of a segment-pair query: visibility of each of two endpoints from a shared origin.
template <typename Float_>
struct SegmentPairVisibilityData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    int ray_count = 0;
    Mask_ visible_a = full<Mask_>(false, 1);  ///< Visibility of the first endpoint.
    Mask_ visible_b = full<Mask_>(false, 1);  ///< Visibility of the second endpoint.

    DRJIT_STRUCT(SegmentPairVisibilityData, visible_a, visible_b)
};

template <bool Detached>
using SegmentPairVisibilityT = SegmentPairVisibilityData<FloatT<Detached>>;

using SegmentPairVisibilityAD = SegmentPairVisibilityT<false>;
using SegmentPairVisibility = SegmentPairVisibilityT<true>;

/// Result of an axial-edge query: whether the source sees any sampled point along each edge.
template <typename Float_>
struct AxialEdgeVisibilityData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    int state_count = 0;
    Mask_ any_visible = full<Mask_>(false, 1);  ///< True where at least one edge sample is visible.

    DRJIT_STRUCT(AxialEdgeVisibilityData, any_visible)
};

template <bool Detached>
using AxialEdgeVisibilityT = AxialEdgeVisibilityData<FloatT<Detached>>;

using AxialEdgeVisibilityAD = AxialEdgeVisibilityT<false>;
using AxialEdgeVisibility = AxialEdgeVisibilityT<true>;

/// Result of a chain-visibility query: whether every segment of each polyline is
/// unoccluded, plus the first occluder encountered if not.
template <typename Float_>
struct SegmentChainVisibilityData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int chain_count = 0;
    int max_segments = 0;
    Mask_ all_visible = full<Mask_>(false, 1);       ///< True where every segment in the chain is unoccluded.
    Int_ first_blocked_segment = full<Int_>(-1, 1);  ///< Index of the first occluded segment; -1 if none.
    Int_ first_blocked_prim = full<Int_>(-1, 1);     ///< Primitive that blocked it; -1 if none.

    DRJIT_STRUCT(SegmentChainVisibilityData,
                 all_visible,
                 first_blocked_segment,
                 first_blocked_prim)
};

template <bool Detached>
using SegmentChainVisibilityT = SegmentChainVisibilityData<FloatT<Detached>>;

using SegmentChainVisibilityAD = SegmentChainVisibilityT<false>;
using SegmentChainVisibility = SegmentChainVisibilityT<true>;

} // namespace rayd
