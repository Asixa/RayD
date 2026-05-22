#pragma once

#include <type_traits>

#include <drjit/complex.h>

#include <rayd/multipath/diffraction_accumulation.h>
#include <rayd/rayd.h>

namespace rayd {

/// Options for first-order native diffraction path export.
struct DiffractionPathOptions {
    float wavelength = 1.f;
    float k = 0.f;
    int seed = 0;
    int max_order = 1;
    int max_paths = 0;
    int max_receivers = 0;
    int strategy_mask = RAYD_DIFF_DIRECT;
    int sample_count = 0;
    int return_geometry = 0;
    int receiver_model = RAYD_DIFF_MATCHED_ISOTROPIC;
};

/// Compact native diffraction path output. Arrays are allocated to `capacity`;
/// the first `count[0]` entries are valid compact paths.
template <typename Float_>
struct DiffractionPathResultData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using ComplexArray = drjit::Complex<Float_>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;

    int capacity = 0;
    Int_ count = full<Int_>(0, 1);
    Mask_ valid = full<Mask_>(false, 1);
    Int_ tx_index = full<Int_>(-1, 1);
    Int_ rx_index = full<Int_>(-1, 1);
    Int_ order = full<Int_>(0, 1);
    Int_ edge_index_0 = full<Int_>(-1, 1);
    Int_ edge_index_1 = full<Int_>(-1, 1);
    Int_ edge_index_2 = full<Int_>(-1, 1);
    Float_ delay = zeros<Float_>(1);
    ComplexArray field_x = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray field_y = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray field_z = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    Vec3f point_0 = zeros<Vec3f>(1);
    Vec3f point_1 = zeros<Vec3f>(1);
    Vec3f point_2 = zeros<Vec3f>(1);

    DRJIT_STRUCT(DiffractionPathResultData,
                 count,
                 valid,
                 tx_index,
                 rx_index,
                 order,
                 edge_index_0,
                 edge_index_1,
                 edge_index_2,
                 delay,
                 field_x,
                 field_y,
                 field_z,
                 point_0,
                 point_1,
                 point_2)
};

} // namespace rayd
