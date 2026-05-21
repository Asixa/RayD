#pragma once

#include <rayd/rayd.h>
#include <rayd/optix.h>

#include "reflection_epc_params.h"

namespace rayd {

class ReflectionEpcPipeline {
public:
    ReflectionEpcPipeline() = default;
    ~ReflectionEpcPipeline();

    void build(OptixDeviceContext context, int hitgroup_record_count);
    bool is_ready() const { return ready_; }

    void launch(const ReflectionEpcParams &params) const;

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
