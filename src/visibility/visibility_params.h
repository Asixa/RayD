// Copyright Xingyu Chen.
// Declares internal visibility support for visibility params.

#pragma once

#include <rayd/visibility/segment_params.h>

namespace rayd::torch_backend {

using SegmentVisibilityParams = shared::optix::SegmentVisibilityParams;
inline constexpr int SegmentVisibilityMaxSamples = shared::optix::SegmentVisibilityMaxSamples;

} // namespace rayd::torch_backend
