#pragma once

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
    HostUpload,
    GpuEmit
};

constexpr EdgeBVHPostBuildStrategy EdgeBVHActivePostBuildStrategy =
    EdgeBVHPostBuildStrategy::GpuTreelet;
constexpr EdgeBVHBuildStreamMode EdgeBVHActiveBuildStreamMode =
    EdgeBVHBuildStreamMode::Overlap;
constexpr EdgeBVHFinalizeMode EdgeBVHActiveFinalizeMode =
    EdgeBVHFinalizeMode::Atomic;
constexpr EdgeBVHTreeletScheduleMode EdgeBVHActiveTreeletScheduleMode =
    EdgeBVHTreeletScheduleMode::FlatLevels;
constexpr EdgeBVHCompactionMode EdgeBVHActiveCompactionMode =
    EdgeBVHCompactionMode::HostUpload;

constexpr int EdgeBVHTreeletMaxLeaves = 7;
constexpr int EdgeBVHTreeletMinPrimitives = 65536;
constexpr int EdgeBVHTreeletMinSubtreeLeaves = 32;
constexpr float EdgeBVHTreeletCostInflationRatio = 1e-4f;

} // namespace rayd
