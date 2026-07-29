// Copyright Xingyu Chen.
// Declares the Dr.Jit edge bvh config API.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <rayd/edge/bvh_types.h>

namespace rayd {

// Retained edge-BVH build controls after configuration convergence. GpuTreelet
// plus Overlap is the product path. None is a benchmark-only pure-LBVH baseline,
// while Serial is a deterministic debug mode without a performance commitment.

/// Optional optimization pass applied after the initial BVH build.
enum class EdgeBVHPostBuildStrategy {
    None,      ///< Benchmark/reference pure-LBVH baseline only.
    GpuTreelet ///< GPU treelet reoptimization (default).
};

/// Whether build stages run serially or overlap across CUDA streams.
enum class EdgeBVHBuildStreamMode {
    Serial, ///< Deterministic debug mode.
    Overlap ///< Product default.
};

constexpr EdgeBVHPostBuildStrategy EdgeBVHDefaultPostBuildStrategy = EdgeBVHPostBuildStrategy::GpuTreelet;
constexpr EdgeBVHBuildStreamMode EdgeBVHDefaultBuildStreamMode = EdgeBVHBuildStreamMode::Overlap;
constexpr int EdgeBVHLeafSize = shared::edge::kBvhLeafSize;

/// Lower-case an env-var value and map '-' to '_' so mode names compare uniformly.
inline std::string normalize_edge_bvh_mode_value(const char* value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) -> char {
        if (ch == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(ch));
    });
    return normalized;
}

// The active_* readers each resolve their mode once from the named environment
// variable (falling back to the default above) and cache the result for the process.

/// Post-build strategy from RAYD_EDGE_BVH_POST_BUILD_STRATEGY.
inline EdgeBVHPostBuildStrategy active_edge_bvh_post_build_strategy() {
    static const EdgeBVHPostBuildStrategy value = []() {
        const char* raw = std::getenv("RAYD_EDGE_BVH_POST_BUILD_STRATEGY");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultPostBuildStrategy;
        }
        if (normalized == "none") {
            return EdgeBVHPostBuildStrategy::None;
        }
        if (normalized == "gpu_treelet") {
            return EdgeBVHPostBuildStrategy::GpuTreelet;
        }
        throw std::runtime_error("Invalid RAYD_EDGE_BVH_POST_BUILD_STRATEGY. Expected one of: none, gpu_treelet.");
    }();
    return value;
}

/// Build stream mode from RAYD_EDGE_BVH_BUILD_STREAM_MODE.
inline EdgeBVHBuildStreamMode active_edge_bvh_build_stream_mode() {
    static const EdgeBVHBuildStreamMode value = []() {
        const char* raw = std::getenv("RAYD_EDGE_BVH_BUILD_STREAM_MODE");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultBuildStreamMode;
        }
        if (normalized == "serial") {
            return EdgeBVHBuildStreamMode::Serial;
        }
        if (normalized == "overlap") {
            return EdgeBVHBuildStreamMode::Overlap;
        }
        throw std::runtime_error("Invalid RAYD_EDGE_BVH_BUILD_STREAM_MODE. Expected one of: serial, overlap.");
    }();
    return value;
}

// Treelet reoptimization thresholds (GpuTreelet post-build strategy).
constexpr int EdgeBVHTreeletMaxLeaves = shared::edge::kBvhTreeletMaxLeaves;
constexpr int EdgeBVHTreeletMinPrimitives = shared::edge::kBvhTreeletMinPrimitives;
constexpr int EdgeBVHTreeletMaxPrimitives = shared::edge::kBvhTreeletMaxPrimitives;
constexpr int EdgeBVHTreeletMinSubtreeLeaves = shared::edge::kBvhTreeletMinSubtreeLeaves;
constexpr float EdgeBVHTreeletCostInflationRatio = shared::edge::kBvhTreeletCostInflationRatio;

} // namespace rayd
