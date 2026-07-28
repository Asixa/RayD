#pragma once

#include <rayd/shared/optix/segment_visibility_params.h>

namespace rayd::torch_backend {

using SegmentVisibilityParams = shared::optix::SegmentVisibilityParams;
inline constexpr int SegmentVisibilityMaxSamples = shared::optix::SegmentVisibilityMaxSamples;

} // namespace rayd::torch_backend
