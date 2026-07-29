// Copyright Xingyu Chen.
// Declares internal visibility support for segment params Dr.Jit.

#pragma once

#include <src/visibility/segment_visibility.cuh>

namespace rayd {

using SegmentVisibilityParams = shared::optix::SegmentVisibilityParams;
inline constexpr int SegmentVisibilityMaxSamples = shared::optix::SegmentVisibilityMaxSamples;

} // namespace rayd
