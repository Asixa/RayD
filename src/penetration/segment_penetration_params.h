// Copyright Xingyu Chen.
// Declares internal penetration support for segment penetration params.

#pragma once

#include <optix.h>

#include <cstdint>

namespace rayd::torch_backend {

inline constexpr std::int32_t SegmentPenetrationEnumeratedFullDistance = 0;
inline constexpr std::int32_t SegmentPenetrationMonteCarloTargetInset = 1;

inline constexpr std::uint8_t SegmentPenetrationRestartConstant = 0u;
inline constexpr std::uint8_t SegmentPenetrationRestartPosition = 1u;

struct SegmentPenetrationParams {
    OptixTraversableHandle traversable = 0;
    const float *origins = nullptr;
    const float *targets = nullptr;
    const std::uint8_t *input_active = nullptr;
    const float *vertices = nullptr;
    const int *faces = nullptr;
    const int *face_offsets = nullptr;
    int *capacity_failure_state = nullptr;

    std::uint8_t *valid = nullptr;
    int *num_hits = nullptr;
    std::uint8_t *reached_target = nullptr;
    std::uint8_t *overflow = nullptr;
    float *distance = nullptr;
    float *direction = nullptr;
    float *t = nullptr;
    float *position = nullptr;
    float *normal = nullptr;
    float *geometric_normal = nullptr;
    int *global_primitive_id = nullptr;

    int *tape_primitive_id = nullptr;
    float *tape_barycentric = nullptr;
    float *tape_restart_epsilon = nullptr;
    std::uint8_t *tape_restart_branch = nullptr;
    std::uint8_t *tape_restart_tie_mask = nullptr;
    std::uint8_t *tape_direction_denominator_branch = nullptr;

    std::int32_t segment_count = 0;
    std::int32_t hit_capacity = 0;
    std::int32_t mesh_count = 0;
    std::int32_t policy = SegmentPenetrationEnumeratedFullDistance;
    std::int32_t failure_bit = 0;
    float scene_diagonal = 0.0f;
};

} // namespace rayd::torch_backend
