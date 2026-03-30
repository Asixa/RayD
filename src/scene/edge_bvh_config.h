#pragma once

namespace rayd {

enum class EdgeBVHPostBuildStrategy {
    None,
    HybridTopLevelSAH,
    GpuTreelet
};

constexpr EdgeBVHPostBuildStrategy EdgeBVHActivePostBuildStrategy =
    EdgeBVHPostBuildStrategy::GpuTreelet;

constexpr int EdgeBVHTreeletMaxLeaves = 7;
constexpr int EdgeBVHTreeletMinPrimitives = 65536;
constexpr int EdgeBVHTreeletMinSubtreeLeaves = 32;
constexpr float EdgeBVHTreeletCostInflationRatio = 1e-4f;

} // namespace rayd
