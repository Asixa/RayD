#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <rayd/optix.h>
#include <rayd/rayd.h>
#include <rayd/multipath/reflection_trace_params.h>
#include <rayd/multipath/reflection_epc_params.h>
#include <rayd/multipath/reflection_accumulation_params.h>
#include <rayd/multipath/diffraction_accumulation_params.h>
#include <rayd/multipath/diffraction_paths_params.h>
#include <rayd/multipath/segment_visibility_params.h>

namespace rayd {

/// Declarative description of a single-module OptiX launch pipeline. All four
/// multipath pipelines share the exact same build sequence and differ only in
/// the PTX blob, entry-point names, payload count, and launch-params size.
struct OptixPipelineConfig {
    const char *ptx = nullptr;
    size_t ptx_size = 0;
    std::vector<const char *> raygen_entries;
    const char *miss_entry = nullptr;
    const char *closesthit_entry = nullptr;
    const char *anyhit_entry = nullptr;  // optional
    int num_payload_values = 0;
    size_t params_size = 0;
};

/// Owns the OptiX module, program groups, shader binding table, and launch
/// params buffer for one multipath pipeline. A pipeline may expose several
/// raygen entry points (e.g. segment visibility); `launch()` selects one by
/// index.
class OptixLaunchPipeline {
public:
    OptixLaunchPipeline() = default;
    ~OptixLaunchPipeline();

    OptixLaunchPipeline(const OptixLaunchPipeline &) = delete;
    OptixLaunchPipeline &operator=(const OptixLaunchPipeline &) = delete;

    void build(OptixDeviceContext context,
               int hitgroup_record_count,
               const OptixPipelineConfig &config);
    bool is_ready() const { return ready_; }
    /// Dr.Jit CUDA device the pipeline was built on, or -1 before build().
    int device() const { return device_; }

    template <typename Params>
    void launch(int raygen_index, const Params &params) const {
        launch_impl(raygen_index,
                    &params,
                    sizeof(Params),
                    static_cast<unsigned int>(params.n_rays));
    }

private:
    void launch_impl(int raygen_index,
                     const void *params,
                     size_t actual_params_size,
                     unsigned int n_rays) const;

    bool ready_ = false;
    // The module, SBT records, and params buffer below are allocated on the
    // Dr.Jit CUDA device that ran build(); launches are rejected elsewhere.
    int device_ = -1;
    int hitgroup_record_count_ = 0;
    size_t params_size_ = 0;
    size_t params_buffer_size_ = 0;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    std::vector<OptixProgramGroup> pg_raygens_;
    OptixProgramGroup pg_miss_ = nullptr;
    OptixProgramGroup pg_hitgroup_ = nullptr;
    std::vector<void *> sbt_raygen_records_;
    void *sbt_miss_record_ = nullptr;
    void *sbt_hitgroup_records_ = nullptr;
    void *params_buffer_ = nullptr;
};

std::shared_ptr<OptixLaunchPipeline> shared_optix_launch_pipeline(
    OptixDeviceContext context,
    int hitgroup_record_count,
    const OptixPipelineConfig &config);

// Pre-filled pipeline configs (PTX blob, entry points, payload/params sizes) for
// each multipath pipeline; pass to OptixLaunchPipeline::build().
OptixPipelineConfig reflection_trace_pipeline_config();
OptixPipelineConfig reflection_epc_pipeline_config();
OptixPipelineConfig reflection_epc_direct_pipeline_config();
OptixPipelineConfig reflection_epc_direct_primary_pipeline_config();
OptixPipelineConfig reflection_accumulation_pipeline_config();
OptixPipelineConfig diffraction_accumulation_pipeline_config();
OptixPipelineConfig diffraction_order1_accumulation_pipeline_config();
OptixPipelineConfig diffraction_order1_accumulation_primary_pipeline_config();
OptixPipelineConfig diffraction_order1_accumulation_no_suffix_pipeline_config();
OptixPipelineConfig diffraction_order1_accumulation_no_suffix_primary_pipeline_config();
OptixPipelineConfig diffraction_order1_accumulation_suffix_pipeline_config();
OptixPipelineConfig diffraction_order1_accumulation_suffix_primary_pipeline_config();
OptixPipelineConfig diffraction_order1_source_visibility_primary_pipeline_config();
OptixPipelineConfig diffraction_order1_no_suffix_target_primary_pipeline_config();
OptixPipelineConfig diffraction_order1_suffix_first_visibility_primary_pipeline_config();
OptixPipelineConfig diffraction_order1_suffix_target_primary_pipeline_config();
OptixPipelineConfig diffraction_chain_accumulation_pipeline_config();
OptixPipelineConfig diffraction_chain_accumulation_primary_pipeline_config();
OptixPipelineConfig diffraction_coherent_accumulation_pipeline_config();
OptixPipelineConfig diffraction_coherent_accumulation_primary_pipeline_config();
OptixPipelineConfig diffraction_paths_pipeline_config();
OptixPipelineConfig diffraction_paths_primary_pipeline_config();
OptixPipelineConfig diffraction_paths_source_visibility_primary_pipeline_config();
OptixPipelineConfig diffraction_paths_target_export_primary_pipeline_config();
OptixPipelineConfig segment_visibility_pipeline_config();
OptixPipelineConfig segment_pair_visibility_pipeline_config();
OptixPipelineConfig axial_edge_visibility_pipeline_config();
OptixPipelineConfig segment_chain_visibility_pipeline_config();

} // namespace rayd
