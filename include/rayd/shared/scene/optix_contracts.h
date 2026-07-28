#pragma once

#include <cstdint>
#include <type_traits>

#include <rayd/shared/rt/optix_pipeline_contracts.h>

namespace rayd::shared::optix {

enum class SceneIntersectionPayloadSlot : std::uint8_t {
    RayT = 0,
    ShapeId = 1,
    BarycentricU = 2,
    BarycentricV = 3,
    LocalPrimitiveId = 4,
    Count = 5,
};

// Output order requested from Dr.Jit's hit-object API. Slots 1..5 match the
// logical result carried by SceneIntersectionPayloadSlot.
enum class SceneHitObjectFieldSlot : std::uint8_t {
    IsHit = 0,
    RayT = 1,
    BarycentricU = 2,
    BarycentricV = 3,
    LocalPrimitiveId = 4,
    ShapeId = 5,
    Count = 6,
};

struct SceneIntersectionPayload {
    std::uint32_t ray_t;
    std::uint32_t shape_id;
    std::uint32_t barycentric_u;
    std::uint32_t barycentric_v;
    std::uint32_t local_primitive_id;
};

static_assert(std::is_standard_layout_v<SceneIntersectionPayload>);
static_assert(std::is_trivially_copyable_v<SceneIntersectionPayload>);
static_assert(static_cast<std::uint8_t>(SceneIntersectionPayloadSlot::Count) == 5u);
static_assert(static_cast<std::uint8_t>(SceneHitObjectFieldSlot::Count) == 6u);
static_assert(static_cast<std::uint8_t>(SceneIntersectionPayloadSlot::Count) ==
              SceneIntersectionPayloadCount);
static_assert(sizeof(SceneIntersectionPayload) == 5u * sizeof(std::uint32_t));

} // namespace rayd::shared::optix