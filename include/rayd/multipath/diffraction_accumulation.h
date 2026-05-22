#pragma once

#include <type_traits>

#include <drjit/complex.h>

#include <rayd/rayd.h>

namespace rayd {

/// Axis-aligned 2D accumulation grid for diffraction power/field output.
struct DiffractionGrid {
    int axis = 2;          ///< Plane normal axis (0 = x, 1 = y, 2 = z).
    float position = 0.f;  ///< Plane offset along `axis`.
    float coord0_min = 0.f;
    float coord0_max = 0.f;
    float coord1_min = 0.f;
    float coord1_max = 0.f;
    int resolution0 = 0;   ///< Cell count along the first in-plane axis.
    int resolution1 = 0;   ///< Cell count along the second in-plane axis.
    float cell_area = 1.f; ///< Area of one grid cell.
};

enum DiffractionStrategyMask {
    RAYD_DIFF_DIRECT = 1 << 0,
    RAYD_DIFF_KELLER = 1 << 1,
    RAYD_DIFF_SUFFIX_REFLECTION = 1 << 2
};

enum DiffractionSampleSequence {
    RAYD_DIFF_HASH = 0,
    RAYD_DIFF_SOBOL = 1
};

enum DiffractionReceiverModel {
    RAYD_DIFF_MATCHED_ISOTROPIC = 0
};

/// Options for native diffraction accumulation kernels.
struct DiffractionAccumOptions {
    float wavelength = 1.f;
    float k = 0.f;
    int seed = 0;
    int samples = 0;
    int max_order = 1;
    int direct_samples = 0;
    int keller_samples = 0;
    int suffix_samples = 0;
    int strategy_mask = RAYD_DIFF_DIRECT | RAYD_DIFF_KELLER;
    int sample_sequence = RAYD_DIFF_HASH;
    int receiver_model = RAYD_DIFF_MATCHED_ISOTROPIC;
    bool collect_edge_use = false;
    bool collect_debug_counts = false;
};

/// Per-primitive electromagnetic material payload used by diffraction kernels.
template <typename Float_>
struct DiffractionMaterialData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    Float_ eta_r = full<Float_>(1.f, 1);
    Float_ sigma = full<Float_>(0.f, 1);
    Float_ mu_r = full<Float_>(1.f, 1);
    Float_ gain = full<Float_>(1.f, 1);
    Mask_ valid = full<Mask_>(false, 1);

    DRJIT_STRUCT(DiffractionMaterialData,
                 eta_r,
                 sigma,
                 mu_r,
                 gain,
                 valid)
};

/// Sampled diffraction states shared by grid accumulation and path-export kernels.
template <typename Float_>
struct DiffractionStateTableData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int count = 0;
    Int_ edge_index = full<Int_>(-1, 1);
    Vec3f edge_pos = zeros<Vec3f>(1);
    Vec3f edge_dir = zeros<Vec3f>(1);
    Float_ edge_line_min = zeros<Float_>(1);
    Float_ edge_line_max = zeros<Float_>(1);
    Vec3f face0_normal = zeros<Vec3f>(1);
    Vec3f face1_normal = zeros<Vec3f>(1);
    Int_ face0_prim_id = full<Int_>(-1, 1);
    Int_ face1_prim_id = full<Int_>(-1, 1);
    Float_ exterior_angle = zeros<Float_>(1);
    Vec3f source_pos = zeros<Vec3f>(1);
    Float_ source_power = zeros<Float_>(1);
    Vec3f incident_direction = zeros<Vec3f>(1);
    Vec3f initial_direction = zeros<Vec3f>(1);
    Int_ prefix_reflection_depth = full<Int_>(0, 1);

    DRJIT_STRUCT(DiffractionStateTableData,
                 edge_index,
                 edge_pos,
                 edge_dir,
                 edge_line_min,
                 edge_line_max,
                 face0_normal,
                 face1_normal,
                 face0_prim_id,
                 face1_prim_id,
                 exterior_angle,
                 source_pos,
                 source_power,
                 incident_direction,
                 initial_direction,
                 prefix_reflection_depth)
};

/// Result of native diffraction accumulation. Grid arrays have grid_cell_count entries.
template <typename Float_>
struct DiffractionAccumResultData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using ComplexArray = drjit::Complex<Float_>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int grid_cell_count = 0;
    Float_ diffraction_power = zeros<Float_>(1);
    ComplexArray diffraction_field_x =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray diffraction_field_y =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray diffraction_field_z =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    Int_ direct_count = full<Int_>(0, 1);
    Int_ keller_count = full<Int_>(0, 1);
    Int_ suffix_count = full<Int_>(0, 1);
    Int_ visibility_reject_count = full<Int_>(0, 1);
    Int_ utd_reject_count = full<Int_>(0, 1);
    Int_ edge_use_count = full<Int_>(0, 1);

    DRJIT_STRUCT(DiffractionAccumResultData,
                 diffraction_power,
                 diffraction_field_x,
                 diffraction_field_y,
                 diffraction_field_z,
                 direct_count,
                 keller_count,
                 suffix_count,
                 visibility_reject_count,
                 utd_reject_count,
                 edge_use_count)
};

} // namespace rayd
