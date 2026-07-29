// Copyright Xingyu Chen.
// Declares internal visibility support for segment params Dr.Jit.

#pragma once

#include <rayd/visibility/segment_params.h>

namespace rayd {

using SegmentVisibilityParams = shared::optix::SegmentVisibilityParams;
inline constexpr int SegmentVisibilityMaxSamples = shared::optix::SegmentVisibilityMaxSamples;

} // namespace rayd
