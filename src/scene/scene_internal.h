// Copyright Xingyu Chen.
// Defines internal scene packing and OptiX payload contracts.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <src/runtime/rt_internal.h>

namespace rayd::shared::scene {

/// Stable 16-byte representation for a packed float3 with explicit zero padding.
struct alignas(16) PackedFloat4 {
    float x;
    float y;
    float z;
    float w;
};

struct GlobalGeometryPackingParams {
    const float* mesh_vertices;
    const std::int32_t* mesh_faces;
    std::int32_t vertex_count;
    std::int32_t face_count;
    std::int32_t vertex_offset;
    std::int32_t face_offset;
    std::int32_t shape_id;
    float* global_vertices;
    std::int32_t* global_faces;
    std::int32_t* face_shape_id;
    std::int32_t* face_local_id;
    cudaStream_t stream;
};

struct GlobalVertexTangentPackingParams {
    const float* mesh_tangent;
    std::int32_t vertex_count;
    std::int32_t vertex_offset;
    float* global_tangent;
    cudaStream_t stream;
};

struct GlobalVertexTangentZeroParams {
    std::int32_t vertex_count;
    std::int32_t vertex_offset;
    float* global_tangent;
    cudaStream_t stream;
};

/// Enqueue global vertex/face packing on the caller-owned stream and storage.
cudaError_t launch_pack_global_geometry_async(const GlobalGeometryPackingParams& params) noexcept;

/// Enqueue a mesh vertex-tangent copy into a scene-global tangent range.
cudaError_t launch_pack_global_vertex_tangent_async(const GlobalVertexTangentPackingParams& params) noexcept;

/// Enqueue zero initialization of a scene-global tangent range.
cudaError_t launch_zero_global_vertex_tangent_range_async(const GlobalVertexTangentZeroParams& params) noexcept;

static_assert(sizeof(PackedFloat4) == 4u * sizeof(float));
static_assert(alignof(PackedFloat4) == 4u * alignof(float));
static_assert(offsetof(PackedFloat4, w) == 3u * sizeof(float));

#define RAYD_SHARED_SCENE_ASSERT_POD(Type)                                                                             \
    static_assert(std::is_standard_layout_v<Type>);                                                                    \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_SCENE_ASSERT_POD(PackedFloat4);
RAYD_SHARED_SCENE_ASSERT_POD(GlobalGeometryPackingParams);
RAYD_SHARED_SCENE_ASSERT_POD(GlobalVertexTangentPackingParams);
RAYD_SHARED_SCENE_ASSERT_POD(GlobalVertexTangentZeroParams);

#undef RAYD_SHARED_SCENE_ASSERT_POD

} // namespace rayd::shared::scene

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
static_assert(static_cast<std::uint8_t>(SceneIntersectionPayloadSlot::Count) == SceneIntersectionPayloadCount);
static_assert(sizeof(SceneIntersectionPayload) == 5u * sizeof(std::uint32_t));

} // namespace rayd::shared::optix
