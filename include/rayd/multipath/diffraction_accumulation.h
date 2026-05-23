#pragma once

#include <type_traits>

#include <drjit/complex.h>

#include <rayd/rayd.h>

namespace rayd {

/// Axis-aligned 2D accumulation grid for diffraction power/field output.
struct DfrGrid {
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

enum DfrStrategyMask {
    RAYD_DFR_DIRECT = 1 << 0,
    RAYD_DFR_KELLER = 1 << 1,
    RAYD_DFR_SUFFIX_REFL = 1 << 2
};

enum DfrSampleSequence {
    RAYD_DFR_HASH = 0,
    RAYD_DFR_SOBOL = 1
};

enum DfrReceiverModel {
    RAYD_DFR_MATCHED_ISO = 0
};

/// Options for native diffraction accumulation kernels.
struct DfrOptions {
    float wavelength = 1.f;
    float k = 0.f;
    int seed = 0;
    int samples = 0;
    int max_order = 1;
    int direct_samples = 0;
    int keller_samples = 0;
    int suffix_samples = 0;
    int strategy_mask = RAYD_DFR_DIRECT | RAYD_DFR_KELLER;
    int sample_sequence = RAYD_DFR_HASH;
    int receiver_model = RAYD_DFR_MATCHED_ISO;
    bool collect_edge_use = false;
    bool collect_debug_counts = false;
};

/// Options for exact coherent deterministic diffraction accumulation.
struct DfrCoherentOptions {
    float wavelength = 1.f;
    float k = 0.f;
    int max_order = 1;
    int receiver_model = RAYD_DFR_MATCHED_ISO;
    bool select_diffraction_point = true;
    bool prefilter_visibility = true;
    bool collect_debug_counts = false;
    float omega = 0.f;
    float tx_pol_x = 1.f;
    float tx_pol_y = 0.f;
    float tx_pol_z = 0.f;
};

/// Per-primitive electromagnetic material payload used by diffraction kernels.
template <typename Float_>
struct DfrMaterialData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    Float_ eta_r = full<Float_>(1.f, 1);
    Float_ sigma = full<Float_>(0.f, 1);
    Float_ mu_r = full<Float_>(1.f, 1);
    Float_ gain = full<Float_>(1.f, 1);
    Mask_ valid = full<Mask_>(false, 1);

    DRJIT_STRUCT(DfrMaterialData,
                 eta_r,
                 sigma,
                 mu_r,
                 gain,
                 valid)
};

/// Sampled diffraction states shared by grid accumulation and path-export kernels.
template <typename Float_>
struct DfrStatesData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int count = 0;
    Int_ edge_index = full<Int_>(-1, 1);
    Vec3f edge_pos = zeros<Vec3f>(1);
    Vec3f edge_dir = zeros<Vec3f>(1);
    Float_ edge_t_min = zeros<Float_>(1);
    Float_ edge_t_max = zeros<Float_>(1);
    Vec3f n0 = zeros<Vec3f>(1);
    Vec3f n1 = zeros<Vec3f>(1);
    Int_ prim0 = full<Int_>(-1, 1);
    Int_ prim1 = full<Int_>(-1, 1);
    Float_ exterior_angle = zeros<Float_>(1);
    Vec3f src = zeros<Vec3f>(1);
    Float_ src_power = zeros<Float_>(1);
    Vec3f wi = zeros<Vec3f>(1);
    Vec3f d0 = zeros<Vec3f>(1);
    Int_ prefix_depth = full<Int_>(0, 1);

    DRJIT_STRUCT(DfrStatesData,
                 edge_index,
                 edge_pos,
                 edge_dir,
                 edge_t_min,
                 edge_t_max,
                 n0,
                 n1,
                 prim0,
                 prim1,
                 exterior_angle,
                 src,
                 src_power,
                 wi,
                 d0,
                 prefix_depth)
};


/// Full deterministic UTD state payload for exact coherent vector accumulation.
template <typename Float_>
struct DfrCoherentUtdStatesData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using ComplexArray = drjit::Complex<Float_>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int count = 0;
    Vec3f edge_pos = zeros<Vec3f>(1);
    Vec3f edge_dir = zeros<Vec3f>(1);
    Vec3f n0 = zeros<Vec3f>(1);
    Vec3f n_face_n = zeros<Vec3f>(1);
    Float_ wedge_n = zeros<Float_>(1);
    Float_ edge_line_min = zeros<Float_>(1);
    Float_ edge_line_max = zeros<Float_>(1);
    Vec3f source_pos = zeros<Vec3f>(1);
    ComplexArray incident_field = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_normal_derivative = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray r_face0 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray r_face_n = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_vector_x = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_vector_y = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_vector_z = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_normal_derivative_vector_x = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_normal_derivative_vector_y = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_normal_derivative_vector_z = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_jones_u = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_jones_v = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_derivative_jones_u = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray incident_derivative_jones_v = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    Vec3f incident_basis_u = zeros<Vec3f>(1);
    Vec3f incident_basis_v = zeros<Vec3f>(1);
    Vec3f incident_basis_k = zeros<Vec3f>(1);
    ComplexArray face0_operator_m00 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray face0_operator_m01 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray face0_operator_m10 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray face0_operator_m11 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray face1_operator_m00 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray face1_operator_m01 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray face1_operator_m10 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray face1_operator_m11 = ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    Float_ face0_eta_r = zeros<Float_>(1);
    Float_ face0_mu_r = full<Float_>(1.f, 1);
    Float_ face0_sigma = zeros<Float_>(1);
    Float_ face0_gain = full<Float_>(1.f, 1);
    Float_ face0_use_fresnel = zeros<Float_>(1);
    Float_ face1_eta_r = zeros<Float_>(1);
    Float_ face1_mu_r = full<Float_>(1.f, 1);
    Float_ face1_sigma = zeros<Float_>(1);
    Float_ face1_gain = full<Float_>(1.f, 1);
    Float_ face1_use_fresnel = zeros<Float_>(1);
    Float_ select_stationary_point = zeros<Float_>(1);
    Int_ owner_code = full<Int_>(0, 1);
    Int_ adjacent_face0 = full<Int_>(-1, 1);
    Int_ adjacent_face1 = full<Int_>(-1, 1);

    DRJIT_STRUCT(DfrCoherentUtdStatesData,
                 edge_pos,
                 edge_dir,
                 n0,
                 n_face_n,
                 wedge_n,
                 edge_line_min,
                 edge_line_max,
                 source_pos,
                 incident_field,
                 incident_normal_derivative,
                 r_face0,
                 r_face_n,
                 incident_vector_x,
                 incident_vector_y,
                 incident_vector_z,
                 incident_normal_derivative_vector_x,
                 incident_normal_derivative_vector_y,
                 incident_normal_derivative_vector_z,
                 incident_jones_u,
                 incident_jones_v,
                 incident_derivative_jones_u,
                 incident_derivative_jones_v,
                 incident_basis_u,
                 incident_basis_v,
                 incident_basis_k,
                 face0_operator_m00,
                 face0_operator_m01,
                 face0_operator_m10,
                 face0_operator_m11,
                 face1_operator_m00,
                 face1_operator_m01,
                 face1_operator_m10,
                 face1_operator_m11,
                 face0_eta_r,
                 face0_mu_r,
                 face0_sigma,
                 face0_gain,
                 face0_use_fresnel,
                 face1_eta_r,
                 face1_mu_r,
                 face1_sigma,
                 face1_gain,
                 face1_use_fresnel,
                 select_stationary_point,
                 owner_code,
                 adjacent_face0,
                 adjacent_face1)
};

/// Result of native diffraction accumulation. Grid arrays have grid_cell_count entries.
template <typename Float_>
struct DfrAccumData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using ComplexArray = drjit::Complex<Float_>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int grid_cell_count = 0;
    Float_ power = zeros<Float_>(1);
    ComplexArray field_x =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray field_y =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray field_z =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    Int_ direct_count = full<Int_>(0, 1);
    Int_ keller_count = full<Int_>(0, 1);
    Int_ suffix_count = full<Int_>(0, 1);
    Int_ vis_rejects = full<Int_>(0, 1);
    Int_ edge_vis_rejects = full<Int_>(0, 1);
    Int_ utd_rejects = full<Int_>(0, 1);
    Int_ edge_uses = full<Int_>(0, 1);

    DRJIT_STRUCT(DfrAccumData,
                 power,
                 field_x,
                 field_y,
                 field_z,
                 direct_count,
                 keller_count,
                 suffix_count,
                 vis_rejects,
                 edge_vis_rejects,
                 utd_rejects,
                 edge_uses)
};

/// Exact coherent deterministic diffraction accumulation result.
template <typename Float_>
struct DfrCoherentAccumData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using ComplexArray = drjit::Complex<Float_>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int grid_cell_count = 0;
    ComplexArray direct_field_x =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray direct_field_y =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray direct_field_z =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray multi_field_x =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray multi_field_y =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    ComplexArray multi_field_z =
        ComplexArray(zeros<Float_>(1), zeros<Float_>(1));
    Int_ direct_count = full<Int_>(0, 1);
    Int_ multi_count = full<Int_>(0, 1);
    Int_ visibility_reject_count = full<Int_>(0, 1);
    Int_ utd_reject_count = full<Int_>(0, 1);

    DRJIT_STRUCT(DfrCoherentAccumData,
                 direct_field_x,
                 direct_field_y,
                 direct_field_z,
                 multi_field_x,
                 multi_field_y,
                 multi_field_z,
                 direct_count,
                 multi_count,
                 visibility_reject_count,
                 utd_reject_count)
};

} // namespace rayd
