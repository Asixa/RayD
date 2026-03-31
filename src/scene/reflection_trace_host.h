#pragma once

#include <rayd/rayd.h>
#include <rayd/optix.h>

#include "reflection_trace_params.h"

namespace rayd {

struct ReflectionTraceBuffers {
    IntDetached bounce_count;
    IntDetached shape_ids;
    IntDetached prim_ids;
    FloatDetached t;
    FloatDetached bary_u;
    FloatDetached bary_v;
    FloatDetached hit_x;
    FloatDetached hit_y;
    FloatDetached hit_z;
    FloatDetached norm_x;
    FloatDetached norm_y;
    FloatDetached norm_z;
    FloatDetached img_x;
    FloatDetached img_y;
    FloatDetached img_z;
};

class ReflectionTracePipeline {
public:
    ReflectionTracePipeline() = default;
    ~ReflectionTracePipeline();

    void build(OptixDeviceContext context, int hitgroup_record_count);
    bool is_ready() const { return ready_; }

    void launch(const ReflectionTraceParams &params) const;

private:
    bool ready_ = false;
    int hitgroup_record_count_ = 0;
    OptixModule module_ = nullptr;
    OptixProgramGroup pg_raygen_ = nullptr;
    OptixProgramGroup pg_miss_ = nullptr;
    OptixProgramGroup pg_hitgroup_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    void *sbt_raygen_record_ = nullptr;
    void *sbt_miss_record_ = nullptr;
    void *sbt_hitgroup_records_ = nullptr;
    void *params_buffer_ = nullptr;
};

} // namespace rayd
