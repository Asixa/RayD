#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace rayd {

// Compile/runtime tuning knobs for the edge BVH builder. Each mode has a default
// below and an environment-variable override (see the active_* readers); most are
// experimental calibration switches rather than supported configuration.

/// Optional optimization pass applied after the initial BVH build.
enum class EdgeBVHPostBuildStrategy {
    None,
    HybridTopLevelSAH,  ///< Rebuild the top levels with SAH (experiment).
    GpuTreelet          ///< GPU treelet reoptimization (default).
};

/// Whether build stages run serially or overlap across CUDA streams.
enum class EdgeBVHBuildStreamMode {
    Serial,
    Overlap
};

/// How node bounds are finalized during the build.
enum class EdgeBVHFinalizeMode {
    Atomic,
    LevelByLevel
};

/// Upload scheduling for treelet optimization.
enum class EdgeBVHTreeletScheduleMode {
    PerLevelUploads,
    FlatLevels
};

/// How the compacted BVH is produced and uploaded.
enum class EdgeBVHCompactionMode {
    HostUploadRaw,
    HostUploadExact,
    GpuEmit
};

/// Device memory layout for BVH nodes.
enum class EdgeBVHNodeLayoutMode {
    ScalarArrays,  ///< Separate arrays per field.
    Packed         ///< Interleaved/packed node records.
};

constexpr EdgeBVHPostBuildStrategy EdgeBVHDefaultPostBuildStrategy =
    EdgeBVHPostBuildStrategy::GpuTreelet;
constexpr EdgeBVHBuildStreamMode EdgeBVHDefaultBuildStreamMode =
    EdgeBVHBuildStreamMode::Overlap;
constexpr EdgeBVHFinalizeMode EdgeBVHDefaultFinalizeMode =
    EdgeBVHFinalizeMode::Atomic;
constexpr EdgeBVHTreeletScheduleMode EdgeBVHDefaultTreeletScheduleMode =
    EdgeBVHTreeletScheduleMode::FlatLevels;
constexpr EdgeBVHCompactionMode EdgeBVHDefaultCompactionMode =
    EdgeBVHCompactionMode::HostUploadRaw;
constexpr EdgeBVHNodeLayoutMode EdgeBVHDefaultNodeLayoutMode =
    EdgeBVHNodeLayoutMode::ScalarArrays;
constexpr int EdgeBVHLeafSize = 4; ///< Target primitives per BVH leaf.

/// Lower-case an env-var value and map '-' to '_' so mode names compare uniformly.
inline std::string normalize_edge_bvh_mode_value(const char *value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
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
        const char *raw = std::getenv("RAYD_EDGE_BVH_POST_BUILD_STRATEGY");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultPostBuildStrategy;
        }
        if (normalized == "none") {
            return EdgeBVHPostBuildStrategy::None;
        }
        if (normalized == "hybrid_top_level_sah") {
            return EdgeBVHPostBuildStrategy::HybridTopLevelSAH;
        }
        if (normalized == "gpu_treelet") {
            return EdgeBVHPostBuildStrategy::GpuTreelet;
        }
        throw std::runtime_error(
            "Invalid RAYD_EDGE_BVH_POST_BUILD_STRATEGY. Expected one of: none, "
            "hybrid_top_level_sah, gpu_treelet.");
    }();
    return value;
}

/// Build stream mode from RAYD_EDGE_BVH_BUILD_STREAM_MODE.
inline EdgeBVHBuildStreamMode active_edge_bvh_build_stream_mode() {
    static const EdgeBVHBuildStreamMode value = []() {
        const char *raw = std::getenv("RAYD_EDGE_BVH_BUILD_STREAM_MODE");
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
        throw std::runtime_error(
            "Invalid RAYD_EDGE_BVH_BUILD_STREAM_MODE. Expected one of: serial, overlap.");
    }();
    return value;
}

/// Finalize mode from RAYD_EDGE_BVH_FINALIZE_MODE.
inline EdgeBVHFinalizeMode active_edge_bvh_finalize_mode() {
    static const EdgeBVHFinalizeMode value = []() {
        const char *raw = std::getenv("RAYD_EDGE_BVH_FINALIZE_MODE");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultFinalizeMode;
        }
        if (normalized == "atomic") {
            return EdgeBVHFinalizeMode::Atomic;
        }
        if (normalized == "level_by_level") {
            return EdgeBVHFinalizeMode::LevelByLevel;
        }
        throw std::runtime_error(
            "Invalid RAYD_EDGE_BVH_FINALIZE_MODE. Expected one of: atomic, level_by_level.");
    }();
    return value;
}

/// Treelet schedule mode from RAYD_EDGE_BVH_TREELET_SCHEDULE_MODE.
inline EdgeBVHTreeletScheduleMode active_edge_bvh_treelet_schedule_mode() {
    static const EdgeBVHTreeletScheduleMode value = []() {
        const char *raw = std::getenv("RAYD_EDGE_BVH_TREELET_SCHEDULE_MODE");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultTreeletScheduleMode;
        }
        if (normalized == "per_level_uploads") {
            return EdgeBVHTreeletScheduleMode::PerLevelUploads;
        }
        if (normalized == "flat_levels") {
            return EdgeBVHTreeletScheduleMode::FlatLevels;
        }
        throw std::runtime_error(
            "Invalid RAYD_EDGE_BVH_TREELET_SCHEDULE_MODE. Expected one of: per_level_uploads, "
            "flat_levels.");
    }();
    return value;
}

/// Compaction mode from RAYD_EDGE_BVH_COMPACTION_MODE.
inline EdgeBVHCompactionMode active_edge_bvh_compaction_mode() {
    static const EdgeBVHCompactionMode value = []() {
        const char *raw = std::getenv("RAYD_EDGE_BVH_COMPACTION_MODE");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultCompactionMode;
        }
        if (normalized == "host_upload_raw") {
            return EdgeBVHCompactionMode::HostUploadRaw;
        }
        if (normalized == "host_upload_exact") {
            return EdgeBVHCompactionMode::HostUploadExact;
        }
        if (normalized == "gpu_emit") {
            return EdgeBVHCompactionMode::GpuEmit;
        }
        throw std::runtime_error(
            "Invalid RAYD_EDGE_BVH_COMPACTION_MODE. Expected one of: host_upload_raw, "
            "host_upload_exact, gpu_emit.");
    }();
    return value;
}

/// Node layout mode from RAYD_EDGE_BVH_NODE_LAYOUT_MODE.
inline EdgeBVHNodeLayoutMode active_edge_bvh_node_layout_mode() {
    static const EdgeBVHNodeLayoutMode value = []() {
        const char *raw = std::getenv("RAYD_EDGE_BVH_NODE_LAYOUT_MODE");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultNodeLayoutMode;
        }
        if (normalized == "scalar_arrays") {
            return EdgeBVHNodeLayoutMode::ScalarArrays;
        }
        if (normalized == "packed") {
            return EdgeBVHNodeLayoutMode::Packed;
        }
        throw std::runtime_error(
            "Invalid RAYD_EDGE_BVH_NODE_LAYOUT_MODE. Expected one of: scalar_arrays, packed.");
    }();
    return value;
}

// Treelet reoptimization thresholds (GpuTreelet post-build strategy).
constexpr int EdgeBVHTreeletMaxLeaves = 7;          ///< Leaves per treelet reorganized at once.
constexpr int EdgeBVHTreeletMinPrimitives = 65536;  ///< Skip treelet optimization below this primitive count.
constexpr int EdgeBVHTreeletMinSubtreeLeaves = 32;  ///< Minimum subtree size eligible for a treelet.
constexpr float EdgeBVHTreeletCostInflationRatio = 1e-4f; ///< SAH-cost improvement required to accept a reorg.

} // namespace rayd
