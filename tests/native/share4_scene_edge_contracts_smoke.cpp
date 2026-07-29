#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <rayd/detail/edge/optix_contracts.h>
#include <rayd/detail/rt/optix_sbt.h>
#include <rayd/detail/scene/optix_contracts.h>

using namespace rayd::shared::optix;

struct HitGroupData {
    std::int32_t shape_offset;
    std::int32_t shape_id;
};

static_assert(SbtRecordAlignment == 16u);
static_assert(SbtRecordHeaderSize == 32u);
static_assert(sizeof(EmptySbtRecord) == 32u);
static_assert(alignof(EmptySbtRecord) == 16u);
static_assert(offsetof(SbtRecord<HitGroupData>, data) == 32u);
static_assert(sizeof(SbtRecord<HitGroupData>) == 48u);
static_assert(alignof(SbtRecord<HitGroupData>) == 16u);
static_assert(sizeof(SceneIntersectionPayload) == 20u);
static_assert(std::is_standard_layout_v<EdgeGeometrySoAView>);
static_assert(std::is_standard_layout_v<EdgeQuerySoAView>);
static_assert(std::is_standard_layout_v<EdgeQueryOutputView>);
static_assert(static_cast<std::uint8_t>(SceneHitObjectFieldSlot::ShapeId) == 5u);
static_assert(static_cast<std::uint8_t>(EdgeRayPayloadSlot::CommonCount) == 4u);
static_assert(static_cast<std::uint8_t>(DrJitEdgeRayPayloadSlot::Valid) == 4u);
static_assert(static_cast<std::uint8_t>(TorchEdgeRayPayloadSlot::TierRadius) == 4u);

int main() {
    return EdgeTopKMax == 16 && EdgePayloadTopKMax == 8 ? 0 : 1;
}