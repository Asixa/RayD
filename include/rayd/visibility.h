// Copyright Xingyu Chen.
// Declares the Torch visibility API.

#pragma once

#include <rayd/scene.h>

#include <array>
#include <cstdint>
#include <optional>

namespace rayd::torch {

struct VisibilityRequest {
    at::Tensor start;
    at::Tensor end;
    std::optional<at::Tensor> active;
};

struct VisibilityResult {
    at::Tensor visible;
    at::Tensor blocker_prim;
    at::Tensor tape_t;
};

VisibilityResult visibility_forward(const SceneResource& scene, const VisibilityRequest& request);

inline constexpr std::array<std::uint32_t, 4> kDiffractionTxAxialEdgeFractionBits = {
    0x3ca3d70au,
    0x3eaaaaabu,
    0x3f2aaaabu,
    0x3f7ae148u,
};

struct AxialEdgeVisibilityConfig {
    std::array<std::uint32_t, 4> sample_fraction_bits = kDiffractionTxAxialEdgeFractionBits;
};

struct AxialEdgeVisibilityRequest {
    at::Tensor tx;
    at::Tensor edge_position;
    at::Tensor edge_direction;
    at::Tensor edge_t_min;
    at::Tensor edge_t_max;
    std::optional<at::Tensor> active;
    AxialEdgeVisibilityConfig config;
};

struct AxialEdgeVisibilityResult {
    at::Tensor any_visible;
};

AxialEdgeVisibilityResult axial_edge_visibility_forward(const SceneResource& scene,
                                                        const AxialEdgeVisibilityRequest& request);

} // namespace rayd::torch
