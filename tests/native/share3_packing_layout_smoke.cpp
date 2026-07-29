#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <rayd/detail/scene/packing.h>

using rayd::shared::scene::GlobalGeometryPackingParams;
using rayd::shared::scene::GlobalVertexTangentPackingParams;
using rayd::shared::scene::GlobalVertexTangentZeroParams;
using rayd::shared::scene::PackedFloat4;

static_assert(sizeof(PackedFloat4) == 16);
static_assert(alignof(PackedFloat4) == 16);
static_assert(offsetof(PackedFloat4, x) == 0);
static_assert(offsetof(PackedFloat4, y) == 4);
static_assert(offsetof(PackedFloat4, z) == 8);
static_assert(offsetof(PackedFloat4, w) == 12);
static_assert(std::is_standard_layout_v<GlobalGeometryPackingParams>);
static_assert(std::is_trivially_copyable_v<GlobalGeometryPackingParams>);
static_assert(std::is_standard_layout_v<GlobalVertexTangentPackingParams>);
static_assert(std::is_trivially_copyable_v<GlobalVertexTangentPackingParams>);
static_assert(std::is_standard_layout_v<GlobalVertexTangentZeroParams>);
static_assert(std::is_trivially_copyable_v<GlobalVertexTangentZeroParams>);

int main() {
    const PackedFloat4 value = {1.0f, 2.0f, 3.0f, 0.0f};
    return value.x == 1.0f && value.y == 2.0f &&
                   value.z == 3.0f && value.w == 0.0f
               ? 0
               : 1;
}
