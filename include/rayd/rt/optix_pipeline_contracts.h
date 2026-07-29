// Copyright Xingyu Chen.
// Defines shared rt support for optix pipeline contracts.

#pragma once

namespace rayd::shared::optix {

// OptiX module and pipeline compilation must agree with the payload and
// attribute slots consumed by the shared contracts/device helpers.
inline constexpr unsigned int SceneIntersectionPayloadCount = 5u;
inline constexpr unsigned int TriangleHitPayloadCount = 6u;
inline constexpr unsigned int VisibilityPayloadCount = 3u;
inline constexpr unsigned int DiffractionPayloadCount = 4u;
inline constexpr unsigned int EdgePointRayPayloadCount = 5u;
inline constexpr unsigned int EdgeTopKPayloadCount = 16u;

inline constexpr unsigned int TriangleAttributeCount = 2u;
inline constexpr unsigned int EdgeAttributeCount = 3u;

} // namespace rayd::shared::optix
