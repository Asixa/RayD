#pragma once

#include <type_traits>

#include <drjit/complex.h>

#include <rayd/rayd.h>

namespace rayd {

struct AccumGrid {
    int axis = 2;
    float position = 0.f;
    float coord0_min = 0.f;
    float coord0_max = 0.f;
    float coord1_min = 0.f;
    float coord1_max = 0.f;
    int resolution0 = 0;
    int resolution1 = 0;
};

struct AccumOptions {
    float wavelength = 1.f;
    float k = 0.f;
    float solid_angle_per_ray = 1.f;
    float cell_area = 1.f;
    int seed = 0;
    int rr_depth = 0;
    float rr_prob = 1.f;
    float stop_threshold = 0.f;
    bool collect_wedges = false;
    bool collect_wedge_prefixes = false;
    int wedge_capacity = 0;
};

template <typename Float_>
struct MaterialData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;

    Float_ eta_r = full<Float_>(1.f, 1);
    Float_ sigma = full<Float_>(0.f, 1);
    Float_ gain = full<Float_>(1.f, 1);
    Float_ mu_r = full<Float_>(1.f, 1);
    Mask_ valid = full<Mask_>(false, 1);

    DRJIT_STRUCT(MaterialData,
                 eta_r,
                 sigma,
                 gain,
                 mu_r,
                 valid)
};

template <typename Float_>
struct WedgeEventsData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;

    int capacity = 0;
    Int_ count = full<Int_>(0, 1);
    Int_ ray_index = full<Int_>(-1, 1);
    Vec3f hit_points = zeros<Vec3f>(1);
    Vec3f normals = zeros<Vec3f>(1);
    Int_ prim_id = full<Int_>(-1, 1);
    Vec3f directions = zeros<Vec3f>(1);
    Int_ bounce_depth = full<Int_>(-1, 1);

    DRJIT_STRUCT(WedgeEventsData,
                 count,
                 ray_index,
                 hit_points,
                 normals,
                 prim_id,
                 directions,
                 bounce_depth)
};

template <typename Float_>
struct AccumResultData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using FloatArray = Float_;
    using ComplexArray = drjit::Complex<Float_>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;
    using WedgeBuffer = WedgeEventsData<Float_>;

    int ray_count = 0;
    int max_bounces = 0;
    int grid_cell_count = 0;
    FloatArray reflection_power = zeros<FloatArray>(1);
    ComplexArray reflection_field_x =
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    ComplexArray reflection_field_y =
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    ComplexArray reflection_field_z =
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    Int_ reflection_count = full<Int_>(0, 1);
    WedgeBuffer wedge_events;

    DRJIT_STRUCT(AccumResultData,
                 reflection_power,
                 reflection_field_x,
                 reflection_field_y,
                 reflection_field_z,
                 reflection_count,
                 wedge_events)
};

} // namespace rayd
