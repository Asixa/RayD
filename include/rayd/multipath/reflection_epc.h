#pragma once

#include <string>
#include <type_traits>

#include <rayd/rayd.h>

namespace rayd {

struct ReflectionEpcOptions {
    IntDetached expected_prim_ids;
    IntDetached surface_group_id;
    IntDetached surface_group_size;
    IntDetached surface_group_members;
    int surface_max_group_size = 0;
    std::string visibility_ignore_mode = "primitive";
    IntDetached final_ignore_group_ids;
};

struct ReflectionEpcFieldOptions : ReflectionEpcOptions {
    Vector3fDetached slot_plane_point = zeros<Vector3fDetached>(1);
    Vector3fDetached slot_plane_normal = zeros<Vector3fDetached>(1);
    FloatDetached slot_eta_r = full<FloatDetached>(1.f, 1);
    FloatDetached slot_mu_r = full<FloatDetached>(1.f, 1);
    FloatDetached slot_sigma = full<FloatDetached>(0.f, 1);
    FloatDetached slot_gain = full<FloatDetached>(1.f, 1);
    Vector3fDetached tx_polarization =
        Vector3fDetached(full<FloatDetached>(1.f, 1),
                         full<FloatDetached>(0.f, 1),
                         full<FloatDetached>(0.f, 1));
    float omega = 2.f * 3.14159265358979323846f * 299792458.f;
    float wavelength = 1.f;
    bool return_geometry = false;
    bool return_endpoints = false;
    bool return_hit_points = true;
    bool return_normals = true;
    bool return_resolved_prim_ids = true;
    bool return_surface_group_ids = true;
};

template <typename Float_>
struct ReflectionEpcResultData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;
    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;

    int ray_count = 0;
    int max_bounces = 0;

    Mask_ valid = full<Mask_>(false, 1);
    Int_ bounce_count = full<Int_>(0, 1);
    Float_ path_length = full<Float_>(Infinity, 1);
    Vec3f reflection_points = zeros<Vec3f>(1);
    Int_ prim_ids = full<Int_>(-1, 1);
    Int_ trace_prim_ids = full<Int_>(-1, 1);
    Int_ resolved_prim_ids = full<Int_>(-1, 1);
    Int_ surface_group_ids = full<Int_>(-1, 1);
    Vec3f plane_normals = zeros<Vec3f>(1);
    Int_ first_blocked_segment = full<Int_>(-1, 1);
    Int_ first_blocked_prim = full<Int_>(-1, 1);
    Int_ first_blocked_group = full<Int_>(-1, 1);

    DRJIT_STRUCT(ReflectionEpcResultData,
                 valid,
                 bounce_count,
                 path_length,
                 reflection_points,
                 prim_ids,
                 trace_prim_ids,
                 resolved_prim_ids,
                 surface_group_ids,
                 plane_normals,
                 first_blocked_segment,
                 first_blocked_prim,
                 first_blocked_group)
};

template <bool Detached>
using ReflectionEpcResultT = ReflectionEpcResultData<FloatT<Detached>>;

using ReflectionEpcResult = ReflectionEpcResultT<false>;
using ReflectionEpcResultDetached = ReflectionEpcResultT<true>;

template <typename Float_>
struct ReflectionEpcFieldResultData {
    static constexpr bool IsDetached = std::is_same_v<Float_, FloatDetached>;

    using Mask_ = std::conditional_t<IsDetached, MaskDetached, Mask>;
    using Int_ = std::conditional_t<IsDetached, IntDetached, Int>;
    using Vec3f = std::conditional_t<IsDetached, Vector3fDetached, Vector3f>;

    int ray_count = 0;
    int max_bounces = 0;

    Mask_ valid = full<Mask_>(false, 1);
    Int_ bounce_count = full<Int_>(0, 1);
    Float_ path_length = full<Float_>(Infinity, 1);

    Float_ field_x_re = zeros<Float_>(1);
    Float_ field_x_im = zeros<Float_>(1);
    Float_ field_y_re = zeros<Float_>(1);
    Float_ field_y_im = zeros<Float_>(1);
    Float_ field_z_re = zeros<Float_>(1);
    Float_ field_z_im = zeros<Float_>(1);

    Vec3f tx_pos = zeros<Vec3f>(1);
    Vec3f first_hit = zeros<Vec3f>(1);
    Vec3f last_hit = zeros<Vec3f>(1);

    Vec3f hit_points = zeros<Vec3f>(1);
    Vec3f normals = zeros<Vec3f>(1);
    Int_ resolved_prim_ids = full<Int_>(-1, 1);
    Int_ surface_group_ids = full<Int_>(-1, 1);

    DRJIT_STRUCT(ReflectionEpcFieldResultData,
                 valid,
                 bounce_count,
                 path_length,
                 field_x_re,
                 field_x_im,
                 field_y_re,
                 field_y_im,
                 field_z_re,
                 field_z_im,
                 tx_pos,
                 first_hit,
                 last_hit,
                 hit_points,
                 normals,
                 resolved_prim_ids,
                 surface_group_ids)
};

template <bool Detached>
using ReflectionEpcFieldResultT =
    ReflectionEpcFieldResultData<FloatT<Detached>>;

using ReflectionEpcFieldResult = ReflectionEpcFieldResultT<false>;
using ReflectionEpcFieldResultDetached = ReflectionEpcFieldResultT<true>;

} // namespace rayd
