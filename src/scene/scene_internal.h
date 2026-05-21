#pragma once

// Internal shared helpers for the Scene implementation, split across
// scene.cpp and the per-area translation units (reflection_scene.cpp,
// visibility_scene.cpp, scene_edge_query.cpp). Not part of the public API.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <rayd/scene/scene.h>
#include <rayd/multipath/pipelines.h>

namespace rayd {

inline std::string normalize_optix_split_mode_value(const char *value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

inline bool uses_symbolic_optix_query_path() {
    // Dr.Jit symbolic recording cannot mix multiple OptiX pipelines/SBTs
    // within a single captured kernel. Fall back to the unified scene path.
    return jit_flag(JitFlag::Recording);
}

template <typename ArrayD>
ArrayD prefix_array(const ArrayD &value, int count) {
    return gather<ArrayD>(value, arange<IntDetached>(count));
}

template <typename ArrayD>
ArrayD concat_array_sequence(const std::vector<ArrayD> &parts) {
    require(!parts.empty(),
            "concat_array_sequence(): at least one array is required.");
    ArrayD result = parts.front();
    for (size_t i = 1; i < parts.size(); ++i) {
        result = concat(result, parts[i]);
    }
    return result;
}

template <bool Detached>
MaskDetached sanitize_segment_active(const Vector3fT<Detached> &start,
                                     const Vector3fT<Detached> &end,
                                     MaskT<Detached> active) {
    MaskDetached active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(start.x())) &&
                           drjit::isfinite(detach<false>(start.y())) &&
                           drjit::isfinite(detach<false>(start.z()));
        active_detached &= drjit::isfinite(detach<false>(end.x())) &&
                           drjit::isfinite(detach<false>(end.y())) &&
                           drjit::isfinite(detach<false>(end.z()));
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(start.x()) &&
                           drjit::isfinite(start.y()) &&
                           drjit::isfinite(start.z());
        active_detached &= drjit::isfinite(end.x()) &&
                           drjit::isfinite(end.y()) &&
                           drjit::isfinite(end.z());
    }
    return active_detached;
}

inline void ensure_pipeline(std::unique_ptr<OptixLaunchPipeline> &pipeline,
                     OptixDeviceContext context,
                     int hitgroup_record_count,
                     const OptixPipelineConfig &config) {
    if (!pipeline) {
        pipeline = std::make_unique<OptixLaunchPipeline>();
        pipeline->build(context, hitgroup_record_count, config);
    }
}

} // namespace rayd
