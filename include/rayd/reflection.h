#pragma once

#include <rayd/rayd.h>

namespace rayd {

struct ReflectionTraceOptions {
    bool deduplicate = false;
    IntDetached canonical_prim_table;
    float image_source_tolerance = 1e-5f;
};

template <typename Float_>
struct ReflectionChainData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;
    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;

    Mask_ is_valid() const {
        return prim_ids >= 0;
    }

    int max_bounces = 0;
    int ray_count = 0;

    Int_ bounce_count = full<Int_>(0, 1);
    Int_ discovery_count = full<Int_>(0, 1);
    Float_ t = full<Float_>(Infinity, 1);
    Vec3f hit_points = zeros<Vec3f>(1);
    Vec3f geo_normals = zeros<Vec3f>(1);
    Vec3f image_sources = zeros<Vec3f>(1);
    Vec3f plane_points = zeros<Vec3f>(1);
    Vec3f plane_normals = zeros<Vec3f>(1);
    Int_ shape_ids = full<Int_>(-1, 1);
    Int_ prim_ids = full<Int_>(-1, 1);

    DRJIT_STRUCT(ReflectionChainData,
                 bounce_count,
                 discovery_count,
                 t,
                 hit_points,
                 geo_normals,
                 image_sources,
                 plane_points,
                 plane_normals,
                 shape_ids,
                 prim_ids)
};

} // namespace rayd
