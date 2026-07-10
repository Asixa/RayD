#pragma once

#include <string>
#include <type_traits>

#include <rayd/rayd.h>

namespace rayd {

/// Options for equivalent-path-correction (EPC) reflection traces. The expected
/// reflector sequence and optional surface-group tables steer which primitives a
/// path is allowed to use and which to ignore during visibility checks.
struct ReflEpcOptions {
    Int expected_prim_ids;       ///< Expected reflector per slot (n_rays * max_bounces).
    Int surface_group_id;        ///< Surface-group id per triangle (optional).
    Int surface_group_size;      ///< Member count of each surface group.
    Int surface_group_members;   ///< Flattened group membership table (surface_group_count * max_group_size).
    int surface_max_group_size = 0;      ///< Width of the group-members table.
    std::string visibility_ignore_mode = "primitive"; ///< "primitive" or "surface_group" ignore semantics.
    Int final_ignore_group_ids;  ///< Groups ignored on the final receiver segment.
};

/// EPC options extended with per-slot material/geometry and the field-evaluation
/// parameters (frequency, polarization) plus output selectors.
template <bool Detached>
struct ReflEpcFieldOptionsT : ReflEpcOptions {
    using Float_ = FloatT<Detached>;
    using Vec3f = Vector3fT<Detached>;

    Vec3f slot_plane_point = zeros<Vec3f>(1);  ///< Override reflecting-plane point per slot.
    Vec3f slot_plane_normal = zeros<Vec3f>(1); ///< Override reflecting-plane normal per slot.
    Float_ slot_eta_r = full<Float_>(1.f, 1);  ///< Relative permittivity per slot.
    Float_ slot_mu_r = full<Float_>(1.f, 1);   ///< Relative permeability per slot.
    Float_ slot_sigma = full<Float_>(0.f, 1);  ///< Conductivity per slot.
    Float_ slot_gain = full<Float_>(1.f, 1);   ///< Extra gain factor per slot.
    Vec3f tx_polarization =                    ///< Transmitter polarization vector.
        Vec3f(full<Float_>(1.f, 1),
              full<Float_>(0.f, 1),
              full<Float_>(0.f, 1));
    float omega = 2.f * 3.14159265358979323846f * 299792458.f; ///< Angular frequency (rad/s).
    float wavelength = 1.f;               ///< Wavelength in world units.
    bool return_geom = false;         ///< Master switch for the per-slot geometry outputs below.
    bool return_endpoints = false;        ///< Emit tx/first-hit/last-hit positions.
    bool return_hit_points = true;        ///< Emit per-slot hit points (requires return_geom).
    bool return_normals = true;           ///< Emit per-slot normals (requires return_geom).
    bool return_resolved_prim_ids = true; ///< Emit resolved primitive ids (requires return_geom).
    bool return_surface_group_ids = true; ///< Emit surface-group ids (requires return_geom).
};

/// Result of an EPC reflection trace. Per-slot arrays are ray_count * max_bounces;
/// per-ray arrays carry the path validity, length, and first occlusion encountered.
template <typename Float_>
struct ReflEpcData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;

    int ray_count = 0;
    int max_bounces = 0;

    Mask_ valid = full<Mask_>(false, 1);              ///< Per-ray: a corrected path to the receiver exists.
    Int_ bounce_count = full<Int_>(0, 1);             ///< Per-ray reflection count.
    Float_ path_length = full<Float_>(Infinity, 1);   ///< Per-ray total corrected path length.
    Vec3f reflection_points = zeros<Vec3f>(1);        ///< Per-slot corrected reflection point.
    Int_ prim_ids = full<Int_>(-1, 1);                ///< Per-slot reflector primitive id.
    Int_ trace_prim_ids = full<Int_>(-1, 1);          ///< Per-slot primitive actually hit during tracing.
    Int_ resolved_prim_ids = full<Int_>(-1, 1);       ///< Per-slot primitive after surface-group resolution.
    Int_ surface_group_ids = full<Int_>(-1, 1);       ///< Per-slot surface-group id.
    Vec3f plane_normals = zeros<Vec3f>(1);            ///< Per-slot reflecting-plane normal.
    Int_ first_blocked_segment = full<Int_>(-1, 1);   ///< Per-ray index of the first occluded segment; -1 if none.
    Int_ first_blocked_prim = full<Int_>(-1, 1);      ///< Per-ray primitive that blocked it.
    Int_ first_blocked_group = full<Int_>(-1, 1);     ///< Per-ray surface group that blocked it.

    DRJIT_STRUCT(ReflEpcData,
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
using ReflEpcT = ReflEpcData<FloatT<Detached>>;

using ReflEpcAD = ReflEpcT<false>;
using ReflEpc = ReflEpcT<true>;

/// Result of an EPC reflection trace that also evaluates the complex field per ray.
/// The field is the per-component (x, y, z) complex phasor; geometry/endpoint arrays
/// are present only when the corresponding ReflEpcFieldOptions flags are set.
template <typename Float_>
struct ReflEpcFieldData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;

    int ray_count = 0;
    int max_bounces = 0;

    Mask_ valid = full<Mask_>(false, 1);            ///< Per-ray: a valid path contributed field.
    Int_ bounce_count = full<Int_>(0, 1);           ///< Per-ray reflection count.
    Float_ path_length = full<Float_>(Infinity, 1); ///< Per-ray total path length.

    Float_ field_x_re = zeros<Float_>(1);  ///< Real part of the x field component.
    Float_ field_x_im = zeros<Float_>(1);  ///< Imaginary part of the x field component.
    Float_ field_y_re = zeros<Float_>(1);  ///< Real part of the y field component.
    Float_ field_y_im = zeros<Float_>(1);  ///< Imaginary part of the y field component.
    Float_ field_z_re = zeros<Float_>(1);  ///< Real part of the z field component.
    Float_ field_z_im = zeros<Float_>(1);  ///< Imaginary part of the z field component.

    Vec3f tx_pos = zeros<Vec3f>(1);     ///< Transmitter position (if return_endpoints).
    Vec3f first_hit = zeros<Vec3f>(1);  ///< First reflection point (if return_endpoints).
    Vec3f last_hit = zeros<Vec3f>(1);   ///< Last reflection point (if return_endpoints).

    Vec3f hit_points = zeros<Vec3f>(1); ///< Per-slot hit points (if return_hit_points).
    Vec3f normals = zeros<Vec3f>(1);    ///< Per-slot normals (if return_normals).
    Int_ resolved_prim_ids = full<Int_>(-1, 1); ///< Per-slot resolved primitive ids (if requested).
    Int_ surface_group_ids = full<Int_>(-1, 1); ///< Per-slot surface-group ids (if requested).

    DRJIT_STRUCT(ReflEpcFieldData,
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
using ReflEpcFieldT =
    ReflEpcFieldData<FloatT<Detached>>;

using ReflEpcFieldAD = ReflEpcFieldT<false>;
using ReflEpcField = ReflEpcFieldT<true>;

} // namespace rayd
