#pragma once

#include <cstdint>
#include <type_traits>

namespace rayd::shared::rt {

// Backend-neutral ray/segment batch views (RAY_TRACING_BACKEND_ARCHITECTURE.md
// §7). SoA pointer layout follows the repo precedent of ReflectionTraceParams
// (separate x/y/z streams), not the draft's array-of-structs vector sketch, so
// JIT paths can hand device pointers straight through without materializing
// packed buffers.
struct RayBatchView {
    const float *origin_x;
    const float *origin_y;
    const float *origin_z;
    const float *direction_x;
    const float *direction_y;
    const float *direction_z;
    const float *tmax;
    const std::uint8_t *active;
    std::uint32_t count;
};

struct SegmentBatchView {
    const float *start_x;
    const float *start_y;
    const float *start_z;
    const float *end_x;
    const float *end_y;
    const float *end_z;
    const std::uint8_t *active;
    std::uint32_t count;
};

static_assert(std::is_standard_layout_v<RayBatchView>);
static_assert(std::is_trivially_copyable_v<RayBatchView>);

static_assert(std::is_standard_layout_v<SegmentBatchView>);
static_assert(std::is_trivially_copyable_v<SegmentBatchView>);

} // namespace rayd::shared::rt
