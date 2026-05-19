#pragma once

#include <vector>

#include <rayd/optix.h>
#include <rayd/rayd.h>

#include "segment_visibility_params.h"

namespace rayd {

enum class SegmentVisibilityLaunchKind {
    Segment,
    SegmentPair,
    AxialEdge
};

class SegmentVisibilityPipeline {
public:
    SegmentVisibilityPipeline() = default;
    ~SegmentVisibilityPipeline();

    SegmentVisibilityPipeline(const SegmentVisibilityPipeline &) = delete;
    SegmentVisibilityPipeline &operator=(const SegmentVisibilityPipeline &) = delete;

    void build(OptixDeviceContext context, int hitgroup_record_count);
    bool is_ready() const { return ready_; }

    void launch(SegmentVisibilityLaunchKind kind,
                const SegmentVisibilityParams &params) const;

private:
    void *raygen_record(SegmentVisibilityLaunchKind kind) const;

    bool ready_ = false;
    int hitgroup_record_count_ = 0;

    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixProgramGroup pg_raygen_segment_ = nullptr;
    OptixProgramGroup pg_raygen_pair_ = nullptr;
    OptixProgramGroup pg_raygen_axial_ = nullptr;
    OptixProgramGroup pg_miss_ = nullptr;
    OptixProgramGroup pg_hitgroup_ = nullptr;

    void *sbt_raygen_segment_ = nullptr;
    void *sbt_raygen_pair_ = nullptr;
    void *sbt_raygen_axial_ = nullptr;
    void *sbt_miss_record_ = nullptr;
    void *sbt_hitgroup_records_ = nullptr;
    void *params_buffer_ = nullptr;
};

} // namespace rayd
