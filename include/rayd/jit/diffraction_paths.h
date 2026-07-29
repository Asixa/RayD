// Copyright Xingyu Chen.
// Declares the Dr.Jit diffraction paths API.

#pragma once

#include <type_traits>

#include <drjit/complex.h>

#include <rayd/jit/diffraction_accumulation.h>
#include <rayd/jit/core.h>

namespace rayd {

/// Options for first-order native diffraction path export.
struct DfrPathOptions {
    float wavelength = 1.f;
    float k = 0.f;
    int seed = 0;
    int max_order = 1;
    int max_paths = 0;
    int max_rx = 0;
    int strategy_mask = RAYD_DFR_DIRECT;
    int sample_count = 0;
    int return_geom = 0;
    int receiver_model = RAYD_DFR_MATCHED_ISO;
};

/// Compact native diffraction path output. Arrays are allocated to `capacity`;
/// the first `count[0]` entries are valid compact paths.
template <typename Float_> struct DfrPathsData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using ComplexArray = drjit::Complex<Float_>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;

    int capacity = 0;
    Int_ count = full<Int_>(0, 1);
    Mask_ valid = full<Mask_>(false, 1);
    Int_ tx_id = full<Int_>(-1, 1);
    Int_ rx_id = full<Int_>(-1, 1);
    Int_ order = full<Int_>(0, 1);
    Int_ edge0 = full<Int_>(-1, 1);
    Int_ edge1 = full<Int_>(-1, 1);
    Int_ edge2 = full<Int_>(-1, 1);
    Float_ delay = zeros<Float_>(1);
    ComplexArray field_x = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray field_y = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray field_z = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    Vec3f p0 = zeros<Vec3f>(1);
    Vec3f p1 = zeros<Vec3f>(1);
    Vec3f p2 = zeros<Vec3f>(1);

    DRJIT_STRUCT(DfrPathsData, count, valid, tx_id, rx_id, order, edge0, edge1, edge2, delay, field_x, field_y, field_z,
                 p0, p1, p2)
};

} // namespace rayd
