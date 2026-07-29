// Copyright Xingyu Chen.
// Defines shared edge support for optix contracts.

#pragma once

#include <cstdint>
#include <type_traits>

#include <rayd/rt/optix_pipeline_contracts.h>

namespace rayd::shared::optix {

inline constexpr int EdgeTopKMax = 16;
inline constexpr int EdgePayloadTopKMax = 8;

static_assert(EdgeTopKMax == static_cast<int>(EdgeTopKPayloadCount));

enum class EdgePointPayloadSlot : std::uint8_t {
    EdgeId = 0,
    SquaredDistance = 1,
    EdgeParameter = 2,
    Valid = 3,
    Count = 4,
};

enum class EdgeRayPayloadSlot : std::uint8_t {
    EdgeId = 0,
    SquaredDistance = 1,
    RayParameter = 2,
    EdgeParameter = 3,
    CommonCount = 4,
};

// Slot 4 is an explicit backend adapter extension. Dr.Jit uses it as a valid
// bit, while Torch carries the current tier radius through the trace. Keeping
// the meanings separate prevents the shared prefix from promising an ABI that
// the device programs do not actually share yet.
enum class DrJitEdgeRayPayloadSlot : std::uint8_t {
    Valid = 4,
    Count = 5,
};

enum class TorchEdgeRayPayloadSlot : std::uint8_t {
    TierRadius = 4,
    Count = 5,
};

enum class EdgeIntersectionAttributeSlot : std::uint8_t {
    SquaredDistance = 0,
    RayParameter = 1,
    EdgeParameter = 2,
    Count = 3,
};

/// Backend-neutral view of the common edge-geometry portion of launch params.
struct EdgeGeometrySoAView {
    const float *p0_x;
    const float *p0_y;
    const float *p0_z;
    const float *e1_x;
    const float *e1_y;
    const float *e1_z;
    const std::uint8_t *active;
    std::int32_t count;
    float search_radius;
};

/// Backend-neutral view of the common edge-query portion of launch params.
struct EdgeQuerySoAView {
    const float *origin_x;
    const float *origin_y;
    const float *origin_z;
    const float *direction_x;
    const float *direction_y;
    const float *direction_z;
    const float *ray_tmax;
    const std::uint8_t *active;
    std::int32_t count;
    std::int32_t k;
};

/// Backend-neutral view of the common intermediate edge-query outputs.
struct EdgeQueryOutputView {
    std::int32_t *edge_ids;
    float *squared_distance;
    float *ray_parameter;
    float *edge_parameter;
    std::uint8_t *valid;
};

#define RAYD_SHARED_EDGE_OPTIX_ASSERT_POD(Type) \
    static_assert(std::is_standard_layout_v<Type>); \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_EDGE_OPTIX_ASSERT_POD(EdgeGeometrySoAView);
RAYD_SHARED_EDGE_OPTIX_ASSERT_POD(EdgeQuerySoAView);
RAYD_SHARED_EDGE_OPTIX_ASSERT_POD(EdgeQueryOutputView);

#undef RAYD_SHARED_EDGE_OPTIX_ASSERT_POD

static_assert(static_cast<std::uint8_t>(EdgePointPayloadSlot::Count) == 4u);
static_assert(static_cast<std::uint8_t>(EdgeRayPayloadSlot::CommonCount) == 4u);
static_assert(static_cast<std::uint8_t>(DrJitEdgeRayPayloadSlot::Valid) == 4u);
static_assert(static_cast<std::uint8_t>(TorchEdgeRayPayloadSlot::TierRadius) == 4u);
static_assert(static_cast<std::uint8_t>(DrJitEdgeRayPayloadSlot::Count) == 5u);
static_assert(static_cast<std::uint8_t>(TorchEdgeRayPayloadSlot::Count) == 5u);
static_assert(static_cast<std::uint8_t>(EdgeIntersectionAttributeSlot::Count) == 3u);
static_assert(static_cast<std::uint8_t>(DrJitEdgeRayPayloadSlot::Count) ==
              EdgePointRayPayloadCount);
static_assert(static_cast<std::uint8_t>(EdgeIntersectionAttributeSlot::Count) ==
              EdgeAttributeCount);

} // namespace rayd::shared::optix