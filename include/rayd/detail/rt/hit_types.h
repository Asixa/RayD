#pragma once

#include <cstdint>
#include <type_traits>

namespace rayd::shared::rt {

// Backend-neutral closest-hit result (RAY_TRACING_BACKEND_ARCHITECTURE.md §7).
// shape_id is the scene mesh/shape id; local_prim_id is the mesh-local triangle
// id; global_prim_id is the scene-global triangle id. A miss sets t=+inf and
// all three ids to -1 (rayd::shared::InvalidSignedId).
struct RawHit {
    float t;
    float bary_u;
    float bary_v;
    std::int32_t global_prim_id;
    std::int32_t shape_id;
    std::int32_t local_prim_id;
};

// Backend-neutral first-blocker result: a miss sets global_prim_id to -1.
struct RawBlocker {
    std::int32_t global_prim_id;
};

static_assert(std::is_standard_layout_v<RawHit>);
static_assert(std::is_trivially_copyable_v<RawHit>);
static_assert(sizeof(RawHit) == 24);

static_assert(std::is_standard_layout_v<RawBlocker>);
static_assert(std::is_trivially_copyable_v<RawBlocker>);
static_assert(sizeof(RawBlocker) == 4);

} // namespace rayd::shared::rt
