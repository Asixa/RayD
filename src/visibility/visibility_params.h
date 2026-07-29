// Copyright Xingyu Chen.
// Declares internal visibility support for visibility params.

#pragma once

#include <src/visibility/segment_visibility.cuh>

namespace rayd::torch_backend {

using SegmentVisibilityParams = shared::optix::SegmentVisibilityParams;
inline constexpr int SegmentVisibilityMaxSamples = shared::optix::SegmentVisibilityMaxSamples;

} // namespace rayd::torch_backend
