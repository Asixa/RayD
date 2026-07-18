#pragma once

#include <limits>

#include <rayd/shared/contracts.h>

namespace rayd::shared::rt {

// Backend-neutral numeric policy contract (RAY_TRACING_BACKEND_ARCHITECTURE.md
// §7). Freezes the current per-backend epsilon divergences behind explicit
// legacy profiles so a future third backend cannot silently reinterpret them.
struct NumericPolicy {
    float ray_tmin;
    float shadow_tmin;
    float endpoint_offset;
    float parallel_epsilon;
    bool watertight_triangles;
};

// Field order: {ray_tmin, shadow_tmin, endpoint_offset, parallel_epsilon,
// watertight_triangles}. endpoint_offset is the shared segment-visibility
// RayBias; parallel_epsilon is the reflection/EPC plane parallel tolerance.
// Dr.Jit: scene intersect/shadow use RayEpsilon (1e-3); Torch intersect uses
// SmallEpsilon (1e-6) and routes shadow through shared segment-visibility
// TraceTMin (1e-5). Both divergences are frozen legacy values, not bugs.
inline constexpr NumericPolicy kDrJitLegacyProfile{
    RayEpsilon, ShadowEpsilon, 1.0e-5f, 1.0e-7f, false};
inline constexpr NumericPolicy kTorchLegacyProfile{
    SmallEpsilon, 1.0e-5f, 1.0e-5f, 1.0e-7f, false};

// Multipath trace family constants shared by both backends' device programs.
inline constexpr float kMultipathTraceTMin = 1.0e-5f;
inline constexpr float kTraceTMaxFinite = 1.0e8f;
inline constexpr float kMultipathRayBias = 1.0e-5f;
inline constexpr float kMinSegmentLength = 2.0e-5f;
inline constexpr float kEpcBarycentricSlack = 1.0e-4f;
inline constexpr float kNormalizeFloor = 1.0e-12f;
inline constexpr float kEdgeDistanceEpsilon = 1.0e-7f;

// Dr.Jit-only exception: surfel visibility offsets both endpoints by
// ShadowEpsilon (1e-3), unlike the 1e-5 segment-visibility RayBias.
inline constexpr float kSurfelEndpointOffset = ShadowEpsilon;

// Miss sentinels. Primary intersect/visibility/EPC report a missed hit as
// t=+inf; the reflection-trace family instead clears to kTraceTMaxFinite and
// never emits +inf, a frozen legacy exception preserved by contract.
inline constexpr float kMissDistance = std::numeric_limits<float>::infinity();
inline constexpr float kReflectionTraceMissDistance = 1.0e8f;

static_assert(kDrJitLegacyProfile.ray_tmin == RayEpsilon);
static_assert(kDrJitLegacyProfile.shadow_tmin == ShadowEpsilon);
static_assert(kTorchLegacyProfile.ray_tmin == SmallEpsilon);
static_assert(kSurfelEndpointOffset == ShadowEpsilon);
static_assert(kMultipathTraceTMin == GeneralEpsilon);
static_assert(kReflectionTraceMissDistance == kTraceTMaxFinite);

// Frozen legacy divergences: do not "fix" without a versioned contract bump.
static_assert(kDrJitLegacyProfile.ray_tmin != kTorchLegacyProfile.ray_tmin);
static_assert(kReflectionTraceMissDistance != kMissDistance);

} // namespace rayd::shared::rt
