#include <rayd/torch/penetration/segment_penetration_kernels.h>
#include <rayd/torch/penetration/segment_penetration_params.h>

#include <rayd/shared/optix/pipeline_contracts.h>
#include <rayd/torch/segment_penetration_optix_ptx.h>

namespace rayd::torch_backend {

OptixPipelineConfig segment_penetration_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_segment_penetration_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_segment_penetration_optix_ptx);
    config.raygen_entries = {"__raygen__segment_penetration"};
    config.miss_entry = "__miss__segment_penetration";
    config.closesthit_entry = "__closesthit__segment_penetration";
    config.num_payload_values = shared::optix::SceneIntersectionPayloadCount;
    config.params_size = sizeof(SegmentPenetrationParams);
    return config;
}

} // namespace rayd::torch_backend
