#pragma once

#include <cstdint>
#include <type_traits>

namespace rayd::shared {

// Backend-neutral scalar contracts, usable from host and device translation
// units without pulling in either backend.
inline constexpr std::int32_t InvalidSignedId = -1;
inline constexpr std::uint32_t InvalidUnsignedId = 0xffffffffu;

inline constexpr float GeneralEpsilon = 1.0e-5f;
inline constexpr float RayEpsilon = 1.0e-3f;
inline constexpr float ShadowEpsilon = 1.0e-3f;
inline constexpr float EdgeEpsilon = 1.0e-5f;
inline constexpr float SmallEpsilon = 1.0e-6f;

inline constexpr float VacuumPermittivity = 8.854187817e-12f;
inline constexpr float SpeedOfLight = 299792458.0f;

enum class RayFlagBits : std::uint32_t {
    None = 0x00u,
    Geometric = 0x01u,
    ShadingN = 0x02u,
    UV = 0x04u,
    All = 0x07u,
};

enum class IntersectionField : std::uint8_t {
    T = 0,
    P,
    N,
    GeoN,
    UV,
    Barycentric,
    ShapeId,
    PrimId,
    LocalPrimId,
    GlobalPrimId,
    Count,
};

enum class NearestPointEdgeField : std::uint8_t {
    Distance = 0,
    Point,
    EdgeT,
    EdgePoint,
    ShapeId,
    EdgeId,
    GlobalEdgeId,
    IsBoundary,
    Count,
};

enum class NearestRayEdgeField : std::uint8_t {
    Distance = 0,
    RayT,
    Point,
    EdgeT,
    EdgePoint,
    ShapeId,
    EdgeId,
    GlobalEdgeId,
    IsBoundary,
    Count,
};

enum class NearestEdgesTopKField : std::uint8_t {
    IsValid = 0,
    Distances,
    Points,
    EdgeT,
    EdgePoints,
    ShapeIds,
    EdgeIds,
    GlobalEdgeIds,
    IsBoundary,
    Count,
};

#define RAYD_SHARED_ASSERT_ENUM_POD(Type)                                    \
    static_assert(std::is_standard_layout_v<Type>);                          \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_ASSERT_ENUM_POD(RayFlagBits);
RAYD_SHARED_ASSERT_ENUM_POD(IntersectionField);
RAYD_SHARED_ASSERT_ENUM_POD(NearestPointEdgeField);
RAYD_SHARED_ASSERT_ENUM_POD(NearestRayEdgeField);
RAYD_SHARED_ASSERT_ENUM_POD(NearestEdgesTopKField);

#undef RAYD_SHARED_ASSERT_ENUM_POD

static_assert(static_cast<std::uint32_t>(InvalidSignedId) == InvalidUnsignedId);
static_assert(static_cast<std::uint32_t>(RayFlagBits::All) ==
              (static_cast<std::uint32_t>(RayFlagBits::Geometric) |
               static_cast<std::uint32_t>(RayFlagBits::ShadingN) |
               static_cast<std::uint32_t>(RayFlagBits::UV)));
static_assert(static_cast<std::uint8_t>(IntersectionField::Count) == 10u);
static_assert(static_cast<std::uint8_t>(NearestPointEdgeField::Count) == 8u);
static_assert(static_cast<std::uint8_t>(NearestRayEdgeField::Count) == 9u);
static_assert(static_cast<std::uint8_t>(NearestEdgesTopKField::Count) == 9u);

} // namespace rayd::shared
