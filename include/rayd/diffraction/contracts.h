// Copyright Xingyu Chen.
// Defines shared diffraction support for contracts.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rayd::shared::optix {

enum class DiffractionStrategyBit : std::int32_t {
    Direct = 1 << 0,
    Keller = 1 << 1,
    SuffixReflection = 1 << 2,
};

enum class DiffractionSampleSequence : std::int32_t {
    Hash = 0,
    Sobol = 1,
};

enum class DiffractionReceiverModel : std::int32_t {
    MatchedIsotropic = 0,
};

enum class DiffractionPathField : std::uint8_t {
    Valid = 0,
    TxId,
    RxId,
    Order,
    Edge0,
    Edge1,
    Edge2,
    Delay,
    FieldXReal,
    FieldXImag,
    FieldYReal,
    FieldYImag,
    FieldZReal,
    FieldZImag,
    P0X,
    P0Y,
    P0Z,
    P1X,
    P1Y,
    P1Z,
    P2X,
    P2Y,
    P2Z,
    Count,
};

enum class DiffractionAccumTapeField : std::uint8_t {
    Active = 0,
    StateIndex,
    Cell,
    MaterialIndex,
    EdgeU,
    Count,
};

enum class DiffractionAccumOutputField : std::uint8_t {
    Power = 0,
    FieldXReal,
    FieldXImag,
    FieldYReal,
    FieldYImag,
    FieldZReal,
    FieldZImag,
    DirectCount,
    KellerCount,
    SuffixCount,
    VisibilityRejects,
    EdgeVisibilityRejects,
    UtdRejects,
    EdgeUses,
    DirectFieldXReal,
    DirectFieldXImag,
    DirectFieldYReal,
    DirectFieldYImag,
    DirectFieldZReal,
    DirectFieldZImag,
    MultiFieldXReal,
    MultiFieldXImag,
    MultiFieldYReal,
    MultiFieldYImag,
    MultiFieldZReal,
    MultiFieldZImag,
    MultiCount,
    CoherentVisibilityRejectCount,
    CoherentUtdRejectCount,
    Count,
};

/// Scalar grid block shared by both diffraction accumulation launch layouts.
struct DiffractionGridParams {
    std::int32_t axis;
    float position;
    float coord0_min;
    float coord0_max;
    float coord1_min;
    float coord1_max;
    std::int32_t resolution0;
    std::int32_t resolution1;
    float cell_area;
};

/// Common path-result prefix. Torch-specific AoS output starts after p0_z.
struct DiffractionPathOutputPrefix {
    std::int32_t *count;
    std::uint8_t *valid;
    std::int32_t *tx_id;
    std::int32_t *rx_id;
    std::int32_t *order;
    std::int32_t *edge0;
    std::int32_t *edge1;
    std::int32_t *edge2;
    float *delay;
    float *field_x_re;
    float *field_x_im;
    float *field_y_re;
    float *field_y_im;
    float *field_z_re;
    float *field_z_im;
    float *p0_x;
    float *p0_y;
    float *p0_z;
};

/// Common path geometry tail following any backend-specific p0 staging field.
struct DiffractionPathGeometryTail {
    float *p1_x;
    float *p1_y;
    float *p1_z;
    float *p2_x;
    float *p2_y;
    float *p2_z;
};

/// Common public accumulation result pointers in their ABI order.
struct DiffractionAccumOutputPointers {
    float *power;
    float *field_x_re;
    float *field_x_im;
    float *field_y_re;
    float *field_y_im;
    float *field_z_re;
    float *field_z_im;
    std::int32_t *direct_count;
    std::int32_t *keller_count;
    std::int32_t *suffix_count;
    std::int32_t *visibility_rejects;
    std::int32_t *edge_visibility_rejects;
    std::int32_t *utd_rejects;
    std::int32_t *edge_uses;
    float *direct_field_x_re;
    float *direct_field_x_im;
    float *direct_field_y_re;
    float *direct_field_y_im;
    float *direct_field_z_re;
    float *direct_field_z_im;
    float *multi_field_x_re;
    float *multi_field_x_im;
    float *multi_field_y_re;
    float *multi_field_y_im;
    float *multi_field_z_re;
    float *multi_field_z_im;
    std::int32_t *multi_count;
    std::int32_t *visibility_reject_count;
    std::int32_t *utd_reject_count;
};

/// Fixed-winner AD tape exported by both accumulation launch layouts.
struct DiffractionAccumTapePointers {
    std::uint8_t *active;
    std::int32_t *state_index;
    std::int32_t *cell;
    std::int32_t *material_index;
    float *edge_u;
};

#define RAYD_SHARED_DIFFRACTION_ASSERT_POD(Type)                            \
    static_assert(std::is_standard_layout_v<Type>);                         \
    static_assert(std::is_trivially_copyable_v<Type>)

RAYD_SHARED_DIFFRACTION_ASSERT_POD(DiffractionGridParams);
RAYD_SHARED_DIFFRACTION_ASSERT_POD(DiffractionPathOutputPrefix);
RAYD_SHARED_DIFFRACTION_ASSERT_POD(DiffractionPathGeometryTail);
RAYD_SHARED_DIFFRACTION_ASSERT_POD(DiffractionAccumOutputPointers);
RAYD_SHARED_DIFFRACTION_ASSERT_POD(DiffractionAccumTapePointers);

#undef RAYD_SHARED_DIFFRACTION_ASSERT_POD

static_assert(static_cast<std::uint8_t>(DiffractionPathField::Count) == 23u);
static_assert(static_cast<std::uint8_t>(DiffractionAccumOutputField::Count) == 29u);
static_assert(static_cast<std::uint8_t>(DiffractionAccumTapeField::Count) == 5u);

} // namespace rayd::shared::optix
