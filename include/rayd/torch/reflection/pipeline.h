#pragma once

#include <rayd/torch/runtime/optix_pipeline.h>

namespace rayd::torch_backend {

OptixPipelineConfig refl_trace_pipeline_config();
OptixPipelineConfig refl_visibility_pipeline_config();
OptixPipelineConfig axial_edge_visibility_pipeline_config();
OptixPipelineConfig refl_epc_pipeline_config();
OptixPipelineConfig refl_accum_pipeline_config();

inline OptixPipelineConfig reflection_trace_pipeline_config() {
    return refl_trace_pipeline_config();
}

inline OptixPipelineConfig segment_visibility_pipeline_config() {
    return refl_visibility_pipeline_config();
}

inline OptixPipelineConfig reflection_epc_pipeline_config() {
    return refl_epc_pipeline_config();
}

inline OptixPipelineConfig reflection_accumulation_pipeline_config() {
    return refl_accum_pipeline_config();
}

} // namespace rayd::torch_backend
