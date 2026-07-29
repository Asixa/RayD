#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rayd::shared::multipath {

inline constexpr std::int32_t InvalidPathId = -1;

enum class PathInteractionKind : std::int32_t {
    None = 0,
    Reflection = 1,
    Diffraction = 2,
};

enum class PathProvenance : std::int32_t {
    Unknown = 0,
    ReflectionTrace = 1,
    ReflectionEpc = 2,
    DiffractionDirect = 3,
    DiffractionChain = 4,
    Imported = 5,
};

enum class PathDerivativeMode : std::int32_t {
    None = 0,
    Tangent = 1,
    Adjoint = 2,
};

enum PathDerivativeField : std::uint32_t {
    PathDerivativeNone = 0u,
    PathDerivativeInteractionPosition = 1u << 0u,
    PathDerivativeInteractionNormal = 1u << 1u,
    PathDerivativeTotalLength = 1u << 2u,
    PathDerivativeDelay = 1u << 3u,
    PathDerivativeAod = 1u << 4u,
    PathDerivativeAoa = 1u << 5u,
    PathDerivativeComplexField = 1u << 6u,
    PathDerivativePower = 1u << 7u,
};

struct PathVec3f {
    float x;
    float y;
    float z;
};

struct PathComplex3f {
    float x_re;
    float x_im;
    float y_re;
    float y_im;
    float z_re;
    float z_im;
};

/// One interaction in the flattened interaction table. Exactly one of the
/// global primitive/edge IDs is normally valid for reflection/diffraction.
struct PathInteractionRecord {
    PathInteractionKind kind;
    std::int32_t global_primitive_id;
    std::int32_t global_edge_id;
    std::int32_t reserved;
    PathVec3f position;
    PathVec3f normal;
};

/// Scalar primal record. Interactions occupy
/// [interaction_offset, interaction_offset + interaction_count).
struct PathRecord {
    std::uint8_t valid;
    std::uint8_t fixed_winner;
    std::uint16_t reserved;
    std::int32_t order;
    std::int32_t source_index;
    std::int32_t receiver_index;
    PathProvenance provenance;
    std::uint32_t available_fields;
    std::uint32_t differentiable_fields;
    std::uint32_t interaction_offset;
    std::uint32_t interaction_count;
    float total_length;
    float delay;
    PathVec3f aod;
    PathVec3f aoa;
    PathComplex3f field;
    float power;
};

/// Optional derivative payload for continuous per-path fields. Discrete IDs,
/// order, validity, provenance, and interaction selection always come from the
/// primal fixed winner and are never differentiated.
struct PathDerivativeRecord {
    float total_length;
    float delay;
    PathVec3f aod;
    PathVec3f aoa;
    PathComplex3f field;
    float power;
};

struct PathInteractionDerivativeRecord {
    PathVec3f position;
    PathVec3f normal;
};

/// Non-owning batch view. Derivative pointers are optional and, when present,
/// must use the same path/interaction indexing as the primal tables.
struct PathRecordBatchView {
    const PathRecord *paths;
    const PathInteractionRecord *interactions;
    const PathDerivativeRecord *path_derivatives;
    const PathInteractionDerivativeRecord *interaction_derivatives;
    std::size_t path_count;
    std::size_t interaction_count;
    PathDerivativeMode derivative_mode;
    std::uint32_t reserved;
};

static_assert(sizeof(PathInteractionKind) == sizeof(std::int32_t));
static_assert(sizeof(PathProvenance) == sizeof(std::int32_t));
static_assert(sizeof(PathDerivativeMode) == sizeof(std::int32_t));
static_assert(sizeof(PathVec3f) == 12);
static_assert(sizeof(PathComplex3f) == 24);
static_assert(sizeof(PathInteractionRecord) == 40);
static_assert(sizeof(PathRecord) == 96);
static_assert(sizeof(PathDerivativeRecord) == 60);
static_assert(sizeof(PathInteractionDerivativeRecord) == 24);
static_assert(offsetof(PathRecord, order) == 4);
static_assert(offsetof(PathRecord, total_length) == 36);
static_assert(offsetof(PathRecord, field) == 68);

#define RAYD_ASSERT_PATH_RECORD_POD(Type)                                    \
    static_assert(std::is_standard_layout_v<Type>);                          \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_ASSERT_PATH_RECORD_POD(PathVec3f);
RAYD_ASSERT_PATH_RECORD_POD(PathComplex3f);
RAYD_ASSERT_PATH_RECORD_POD(PathInteractionRecord);
RAYD_ASSERT_PATH_RECORD_POD(PathRecord);
RAYD_ASSERT_PATH_RECORD_POD(PathDerivativeRecord);
RAYD_ASSERT_PATH_RECORD_POD(PathInteractionDerivativeRecord);
RAYD_ASSERT_PATH_RECORD_POD(PathRecordBatchView);

#undef RAYD_ASSERT_PATH_RECORD_POD

} // namespace rayd::shared::multipath
