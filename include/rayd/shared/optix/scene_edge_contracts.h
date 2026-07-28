#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <rayd/shared/optix/pipeline_contracts.h>

namespace rayd::shared::optix {

// OptiX fixes these values as part of the shader binding table ABI. Keeping
// them here lets both backends share record layouts without including either
// backend's OptiX host declarations.
inline constexpr std::size_t SbtRecordAlignment = 16u;
inline constexpr std::size_t SbtRecordHeaderSize = 32u;

inline constexpr int EdgeTopKMax = 16;
inline constexpr int EdgePayloadTopKMax = 8;

static_assert(EdgeTopKMax == static_cast<int>(EdgeTopKPayloadCount));

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

struct SceneIntersectionPayload {
    std::uint32_t ray_t;
    std::uint32_t shape_id;
    std::uint32_t barycentric_u;
    std::uint32_t barycentric_v;
    std::uint32_t local_primitive_id;
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

template <typename T>
struct alignas(SbtRecordAlignment) SbtRecord {
    std::byte header[SbtRecordHeaderSize];
    T data;
};

struct alignas(SbtRecordAlignment) EmptySbtRecord {
    std::byte header[SbtRecordHeaderSize];
};

#define RAYD_SHARED_SCENE_EDGE_ASSERT_POD(Type) \
    static_assert(std::is_standard_layout_v<Type>); \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_SCENE_EDGE_ASSERT_POD(SceneIntersectionPayload);
RAYD_SHARED_SCENE_EDGE_ASSERT_POD(EdgeGeometrySoAView);
RAYD_SHARED_SCENE_EDGE_ASSERT_POD(EdgeQuerySoAView);
RAYD_SHARED_SCENE_EDGE_ASSERT_POD(EdgeQueryOutputView);
RAYD_SHARED_SCENE_EDGE_ASSERT_POD(EmptySbtRecord);

#undef RAYD_SHARED_SCENE_EDGE_ASSERT_POD

static_assert(static_cast<std::uint8_t>(SceneIntersectionPayloadSlot::Count) == 5u);
static_assert(static_cast<std::uint8_t>(SceneHitObjectFieldSlot::Count) == 6u);
static_assert(static_cast<std::uint8_t>(EdgePointPayloadSlot::Count) == 4u);
static_assert(static_cast<std::uint8_t>(EdgeRayPayloadSlot::CommonCount) == 4u);
static_assert(static_cast<std::uint8_t>(DrJitEdgeRayPayloadSlot::Valid) == 4u);
static_assert(static_cast<std::uint8_t>(TorchEdgeRayPayloadSlot::TierRadius) == 4u);
static_assert(static_cast<std::uint8_t>(DrJitEdgeRayPayloadSlot::Count) == 5u);
static_assert(static_cast<std::uint8_t>(TorchEdgeRayPayloadSlot::Count) == 5u);
static_assert(static_cast<std::uint8_t>(EdgeIntersectionAttributeSlot::Count) == 3u);
static_assert(static_cast<std::uint8_t>(SceneIntersectionPayloadSlot::Count) ==
              SceneIntersectionPayloadCount);
static_assert(static_cast<std::uint8_t>(DrJitEdgeRayPayloadSlot::Count) ==
              EdgePointRayPayloadCount);
static_assert(static_cast<std::uint8_t>(EdgeIntersectionAttributeSlot::Count) ==
              EdgeAttributeCount);
static_assert(sizeof(SceneIntersectionPayload) == 5u * sizeof(std::uint32_t));
static_assert(alignof(EmptySbtRecord) == SbtRecordAlignment);
static_assert(sizeof(EmptySbtRecord) == SbtRecordHeaderSize);
static_assert(offsetof(SbtRecord<std::uint32_t>, data) == SbtRecordHeaderSize);

} // namespace rayd::shared::optix
