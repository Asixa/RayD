#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include "scene_internal.h"
#include <rayd/multipath/diffraction_accumulation_ad.h>
#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>
#include <rayd/trace/cuda_multipath_gpu.h>

#include "scene_multipath_internal.h"

namespace rayd {

namespace {

std::string normalize_optix_split_mode_value(const char *value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

} // namespace

namespace multipath_detail {

TraceVisibilityBackend active_trace_visibility_backend() {
    static const TraceVisibilityBackend value = []() {
        const char *raw = std::getenv("RAYD_TRACE_VISIBILITY_BACKEND");
        const std::string normalized = normalize_optix_split_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return TraceVisibilityBackend::Auto;
        }
        if (normalized == "jit" || normalized == "drjit" ||
            normalized == "hitobject" || normalized == "hit_object") {
            return TraceVisibilityBackend::Jit;
        }
        if (normalized == "native" || normalized == "optixlaunch" ||
            normalized == "optix_launch") {
            return TraceVisibilityBackend::Native;
        }
        throw std::runtime_error(
            "Invalid RAYD_TRACE_VISIBILITY_BACKEND. Expected one of: auto, jit, native.");
    }();
    return value;
}

ReflEpcVisibilityIgnoreMode parse_refl_epc_vis_ignore(
    const std::string &value) {
    const std::string normalized = normalize_optix_split_mode_value(value.c_str());
    if (normalized.empty() || normalized == "primitive" ||
        normalized == "prim" || normalized == "exact") {
        return ReflEpcVisibilityIgnoreMode::Primitive;
    }
    if (normalized == "surface_group" || normalized == "surface-group" ||
        normalized == "group") {
        return ReflEpcVisibilityIgnoreMode::SurfaceGroup;
    }
    throw std::runtime_error(
        "Invalid ReflEpcOptions.visibility_ignore_mode. "
        "Expected one of: 'primitive', 'surface_group'.");
}

bool use_jit_trace_visibility_path(int ignore_k) {
    const TraceVisibilityBackend backend = active_trace_visibility_backend();
    if (backend == TraceVisibilityBackend::Native) {
        return false;
    }
    if (backend == TraceVisibilityBackend::Jit) {
        require(ignore_k == 0,
                "RAYD_TRACE_VISIBILITY_BACKEND=jit does not support ignore lists yet.");
        return true;
    }
    return ignore_k == 0;
}

bool recording_reflections() {
    return jit_flag(JitFlag::Recording);
}

bool uses_symbolic_optix_query_path() {
    // Dr.Jit symbolic recording cannot mix multiple OptiX pipelines/SBTs
    // within a single captured kernel. Fall back to the unified scene path.
    return jit_flag(JitFlag::Recording);
}

void ensure_pipeline(std::shared_ptr<OptixLaunchPipeline> &pipeline,
                     OptixDeviceContext context,
                     int hitgroup_record_count,
                     const OptixPipelineConfig &config) {
    if (!pipeline) {
        pipeline = shared_optix_launch_pipeline(context, hitgroup_record_count, config);
    }
}

} // namespace multipath_detail

} // namespace rayd
