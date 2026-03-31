#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace rayd {

enum class EdgeBVHPostBuildStrategy {
    None,
    HybridTopLevelSAH,
    GpuTreelet
};

enum class EdgeBVHBuildStreamMode {
    Serial,
    Overlap
};

enum class EdgeBVHFinalizeMode {
    Atomic,
    LevelByLevel
};

enum class EdgeBVHTreeletScheduleMode {
    PerLevelUploads,
    FlatLevels
};

enum class EdgeBVHCompactionMode {
    HostUploadRaw,
    HostUploadExact,
    GpuEmit
};

enum class EdgeBVHBuildAlgorithm {
    LBVH,
    PLOC
};

enum class EdgeBVHNodeLayoutMode {
    ScalarArrays,
    Packed
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
// Stable default remains LBVH + GPU treelet. PLOC stays available as an experimental override.
constexpr EdgeBVHBuildAlgorithm EdgeBVHDefaultBuildAlgorithm =
    EdgeBVHBuildAlgorithm::LBVH;
constexpr EdgeBVHNodeLayoutMode EdgeBVHDefaultNodeLayoutMode =
    EdgeBVHNodeLayoutMode::ScalarArrays;
constexpr int EdgeBVHLeafSize = 4;

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

inline EdgeBVHBuildAlgorithm active_edge_bvh_build_algorithm() {
    static const EdgeBVHBuildAlgorithm value = []() {
        const char *raw = std::getenv("RAYD_EDGE_BVH_BUILD_ALGORITHM");
        const std::string normalized = normalize_edge_bvh_mode_value(raw);
        if (normalized.empty()) {
            return EdgeBVHDefaultBuildAlgorithm;
        }
        if (normalized == "lbvh") {
            return EdgeBVHBuildAlgorithm::LBVH;
        }
        if (normalized == "ploc") {
            return EdgeBVHBuildAlgorithm::PLOC;
        }
        throw std::runtime_error(
            "Invalid RAYD_EDGE_BVH_BUILD_ALGORITHM. Expected one of: lbvh, ploc.");
    }();
    return value;
}

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

constexpr int EdgeBVHTreeletMaxLeaves = 7;
constexpr int EdgeBVHTreeletMinPrimitives = 65536;
constexpr int EdgeBVHTreeletMinSubtreeLeaves = 32;
constexpr float EdgeBVHTreeletCostInflationRatio = 1e-4f;

} // namespace rayd
