#pragma once

#include <cstdint>

#include <rayd/detail/edge/optix_contracts.h>

namespace rayd {

/// Maximum k supported by the OptiX top-k edge intersection program.
constexpr int EdgeOptixTopKMax = shared::optix::EdgeTopKMax;

/// Launch parameters for the OptiX edge-query programs (point / ray / top-k).
/// Inputs are flat SoA device pointers; \p k selects point vs. top-k semantics.
struct EdgeOptixQueryParams {
    uint64_t handle = 0;          ///< Traversable handle of the edge GAS.

    const float *edge_p0_x = nullptr; ///< Edge start x (one per edge).
    const float *edge_p0_y = nullptr;
    const float *edge_p0_z = nullptr;
    const float *edge_e1_x = nullptr; ///< Edge vector x (start + e1 is the far endpoint).
    const float *edge_e1_y = nullptr;
    const float *edge_e1_z = nullptr;
    const uint8_t *edge_mask = nullptr; ///< Per-edge active flag, or null for all-active.
    int edge_count = 0;
    float search_radius = 0.0f;   ///< Distance cutoff; hits beyond this are rejected.

    const float *query_x = nullptr; ///< Query point / ray origin x (one per query).
    const float *query_y = nullptr;
    const float *query_z = nullptr;
    const float *ray_dx = nullptr;  ///< RayAD direction x (ray queries only).
    const float *ray_dy = nullptr;
    const float *ray_dz = nullptr;
    const float *ray_tmax = nullptr; ///< Per-ray max parameter (ray queries only).
    const uint8_t *active_mask = nullptr; ///< Per-query active flag.
    int query_count = 0;
    int k = 0;                    ///< Neighbors per query; results in query_count * k order.

    int *out_edge_ids = nullptr;   ///< Winning edge id(s).
    float *out_distance_sq = nullptr; ///< Squared distance to the winner(s).
    float *out_ray_t = nullptr;    ///< RayAD parameter at closest approach (ray queries).
    float *out_edge_t = nullptr;   ///< Closest-point parameter along the edge.
    uint8_t *out_valid = nullptr;  ///< Whether each output slot holds a hit.
};

} // namespace rayd
