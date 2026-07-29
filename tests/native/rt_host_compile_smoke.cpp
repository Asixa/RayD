// Host-compile smoke for the migrated multipath algorithm headers (P4 key gate).
//
// This translation unit is compiled by tests/test_rt_host_compile.py with a pure
// host C++ compiler (cl.exe / g++), no CUDA or OptiX device compiler. It proves
// that the de-CUDA-ised algorithm bodies parse and fully type-check off-device:
// they must spell no optixTrace / payload register / launch index, and every ray
// cast must go through the rt::Traverser concept, which a plain host struct can
// satisfy. `<vector_types.h>` (a header-only POD from the CUDA toolkit, pulled in
// by reflection_trace_params.h for the optional packed-triangle inputs) is the
// only CUDA include on the host path.

#include <cstdint>

#include <rayd/detail/diffraction/paths_algo.h>
#include <rayd/detail/diffraction/accumulation_algo.h>
#include <rayd/detail/reflection/accumulation_algo.h>
#include <rayd/detail/reflection/epc_algo.h>
#include <rayd/detail/reflection/trace_algo.h>
#include <rayd/detail/visibility/segment_algo.h>
#include <rayd/detail/rt/qualifiers.h>
#include <rayd/detail/rt/traverser.h>

namespace {

using rayd::shared::math::Vec3f;
using rayd::shared::rt::TriangleHit;

// A host layout policy exercising every compile-time layout branch (the Torch
// superset). Instantiating the algorithm against it type-checks the AoS-input,
// packed-triangle, output-layout, empty-slot, nullable-tmax, and extended-output
// paths under the host compiler.
struct HostLayoutPolicy {
    static constexpr bool allow_aos_inputs = true;
    static constexpr bool allow_packed_triangles = true;
    static constexpr bool honor_output_layout = true;
    static constexpr bool clear_empty_slots = true;
    static constexpr bool nullable_ray_tmax = true;
    static constexpr bool allow_extended_outputs = true;
};

// A trivial host traverser satisfying rt::is_traverser: enough for the algorithm
// body to compile and route all four traversal calls off-device.
struct HostTraverser {
    TriangleHit trace_closest(Vec3f, Vec3f, float, float) const {
        return TriangleHit{0.0f, 0.0f, 0.0f, -1, -1, 0u};
    }
    bool trace_occluded(Vec3f, Vec3f, float, float) const { return false; }
    bool trace_occluded_ignore(Vec3f, Vec3f, float, float,
                               const std::int32_t *, int) const {
        return false;
    }
    TriangleHit trace_first_blocker(Vec3f, Vec3f, float, float,
                                    const std::int32_t *, int) const {
        return TriangleHit{0.0f, 0.0f, 0.0f, -1, -1, 0u};
    }
};

static_assert(rayd::shared::rt::is_traverser_v<HostTraverser>,
              "HostTraverser must satisfy the rt::Traverser concept off-device.");

using HostConfig = rayd::shared::rt::TraceConfig<HostLayoutPolicy, HostTraverser>;

using AlgoFn = void (*)(const rayd::shared::optix::ReflectionTraceParams &,
                        std::uint32_t, const HostTraverser &, const HostTraverser &);

// Segment-visibility layout policy (both compile-time knobs on) and its config.
struct HostSegmentLayoutPolicy {
    static constexpr bool disable_anyhit_without_ignore = true;
    static constexpr bool write_output_t = true;
};
using HostSegmentConfig =
    rayd::shared::rt::TraceConfig<HostSegmentLayoutPolicy, HostTraverser>;
using SegmentAlgoFn = void (*)(const rayd::shared::optix::SegmentVisibilityParams &,
                               std::uint32_t, const HostTraverser &);

// Reflection-EPC layout policy and its config. The EPC algorithm reads only the
// traverser axis; the layout only carries the anyhit-disable knob the OptiX shim's
// traverser template consumes.
struct HostEpcLayoutPolicy {
    static constexpr bool DisableAnyHitWithoutIgnore = true;
};
using HostEpcConfig = rayd::shared::rt::TraceConfig<HostEpcLayoutPolicy, HostTraverser>;
using EpcAlgoFn = void (*)(const rayd::shared::optix::ReflEpcParams &,
                           std::uint32_t, const HostTraverser &, const HostTraverser &);

// Reflection-accumulation host stubs. AccumParams mirrors the backend SoA layout
// minus the OptiX handles (which the algorithm never reads, since the ray casts
// go through the Traverser); the policy supplies the compile-time include-depth
// predicate and a no-op grid commit.
struct HostAccumParams {
    int split_mode;
    const float *tri_p0_x;
    const float *tri_p0_y;
    const float *tri_p0_z;
    const float *tri_e1_x;
    const float *tri_e1_y;
    const float *tri_e1_z;
    const float *tri_e2_x;
    const float *tri_e2_y;
    const float *tri_e2_z;
    const float *tri_fn_x;
    const float *tri_fn_y;
    const float *tri_fn_z;
    const int *face_offsets;
    int n_meshes;
    int n_triangles;
    const float *ray_ox;
    const float *ray_oy;
    const float *ray_oz;
    const float *ray_dx;
    const float *ray_dy;
    const float *ray_dz;
    const float *ray_tmax;
    const std::uint8_t *active_mask;
    int n_rays;
    const float *tx_x;
    const float *tx_y;
    const float *tx_z;
    const float *tx_pol_x;
    const float *tx_pol_y;
    const float *tx_pol_z;
    int max_bounces;
    float wavelength;
    float k;
    float solid_angle_per_ray;
    float cell_area;
    int seed;
    int rr_depth;
    float rr_prob;
    float stop_threshold;
    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;
    const float *material_eta_r;
    const float *material_sigma;
    const float *material_gain;
    const float *material_mu_r;
    const std::uint8_t *material_valid;
    int material_count;
    int collect_wedges;
    int collect_wedge_prefixes;
    int wedge_capacity;
    int wedge_sample_stride;
    int *out_wedge_count;
    int *out_wedge_ray_index;
    float *out_wedge_hit_x;
    float *out_wedge_hit_y;
    float *out_wedge_hit_z;
    float *out_wedge_normal_x;
    float *out_wedge_normal_y;
    float *out_wedge_normal_z;
    int *out_wedge_prim_id;
    float *out_wedge_dir_x;
    float *out_wedge_dir_y;
    float *out_wedge_dir_z;
    float *out_wedge_source_x;
    float *out_wedge_source_y;
    float *out_wedge_source_z;
    float *out_wedge_source_power;
    float *out_wedge_initial_dir_x;
    float *out_wedge_initial_dir_y;
    float *out_wedge_initial_dir_z;
    int *out_wedge_bounce_depth;
};

struct HostAccumPolicy {
    static bool include_depth(const HostAccumParams &, int depth) { return depth > 0; }
    static void commit(const HostAccumParams &, unsigned int, int, int,
                       rayd::shared::field::Complex3, float) {}
};
using ReflAccumAlgoFn = void (*)(const HostAccumParams &, std::uint32_t,
                                 const HostTraverser &, const HostTraverser &);

// Diffraction path-export host stub. DfrPathParams mirrors the backend SoA layout
// of the fields the algorithm reads (the OptiX handles, which only the shim
// touches, are omitted).
struct HostPathParams {
    int split_mode;
    int n_rays;
    int capacity;
    int tx_count;
    const float *rx_pos_x;
    const float *rx_pos_y;
    const float *rx_pos_z;
    int rx_count;
    const std::uint8_t *active_mask;
    int active_width;
    int state_count;
    int state_limit;
    const int *state_edge_index;
    const float *state_edge_pos_x;
    const float *state_edge_pos_y;
    const float *state_edge_pos_z;
    const float *state_edge_dir_x;
    const float *state_edge_dir_y;
    const float *state_edge_dir_z;
    const float *state_edge_t_min;
    const float *state_edge_t_max;
    const float *state_n0_x;
    const float *state_n0_y;
    const float *state_n0_z;
    const float *state_n1_x;
    const float *state_n1_y;
    const float *state_n1_z;
    const int *state_prim0;
    const int *state_prim1;
    const float *state_exterior_angle;
    const float *state_src_x;
    const float *state_src_y;
    const float *state_src_z;
    const float *state_src_power;
    const float *material_eta_r;
    const float *material_sigma;
    const float *material_mu_r;
    const float *material_gain;
    const std::uint8_t *material_valid;
    int material_count;
    float k;
    float omega;
    int max_order;
    int strategy_mask;
    int receiver_model;
    std::uint8_t *temp_visibility;
    int *out_count;
    std::uint8_t *out_valid;
    int *out_tx_id;
    int *out_rx_id;
    int *out_order;
    int *out_edge0;
    int *out_edge1;
    int *out_edge2;
    float *out_delay;
    float *out_field_x_re;
    float *out_field_x_im;
    float *out_field_y_re;
    float *out_field_y_im;
    float *out_field_z_re;
    float *out_field_z_im;
    float *out_p0_x;
    float *out_p0_y;
    float *out_p0_z;
    float *out_p1_x;
    float *out_p1_y;
    float *out_p1_z;
    float *out_p2_x;
    float *out_p2_y;
    float *out_p2_z;
};
using PathAlgoFn = void (*)(const HostPathParams &, std::uint32_t,
                            const HostTraverser &, const HostTraverser &);

struct HostDfrAccumParams {
    int split_mode;

    int n_rays;

    const std::uint8_t *active_mask;
    int state_count;
    const int *state_edge_index;
    const float *state_edge_pos_x;
    const float *state_edge_pos_y;
    const float *state_edge_pos_z;
    const float *state_edge_dir_x;
    const float *state_edge_dir_y;
    const float *state_edge_dir_z;
    const float *state_edge_t_min;
    const float *state_edge_t_max;
    const float *state_n0_x;
    const float *state_n0_y;
    const float *state_n0_z;
    const float *state_n1_x;
    const float *state_n1_y;
    const float *state_n1_z;
    const int *state_prim0;
    const int *state_prim1;
    const float *state_exterior_angle;
    const float *state_src_x;
    const float *state_src_y;
    const float *state_src_z;
    const float *state_src_power;
    const float *state_wi_x;
    const float *state_wi_y;
    const float *state_wi_z;
    const float *state_d0_x;
    const float *state_d0_y;
    const float *state_d0_z;
    const int *state_prefix_depth;
    const float *utd_epx;
    const float *utd_epy;
    const float *utd_epz;
    const float *utd_edx;
    const float *utd_edy;
    const float *utd_edz;
    const float *utd_n0x;
    const float *utd_n0y;
    const float *utd_n0z;
    const float *utd_nnx;
    const float *utd_nny;
    const float *utd_nnz;
    const float *utd_wn;
    const float *utd_elm;
    const float *utd_elx;
    const float *utd_spx;
    const float *utd_spy;
    const float *utd_spz;
    const float *utd_ifr;
    const float *utd_ifi;
    const float *utd_inr;
    const float *utd_ini;
    const float *utd_r0r;
    const float *utd_r0i;
    const float *utd_rnr;
    const float *utd_rni;
    const float *utd_vxr;
    const float *utd_vxi;
    const float *utd_vyr;
    const float *utd_vyi;
    const float *utd_vzr;
    const float *utd_vzi;
    const float *utd_dxr;
    const float *utd_dxi;
    const float *utd_dyr;
    const float *utd_dyi;
    const float *utd_dzr;
    const float *utd_dzi;
    const float *utd_jur;
    const float *utd_jui;
    const float *utd_jvr;
    const float *utd_jvi;
    const float *utd_djur;
    const float *utd_djui;
    const float *utd_djvr;
    const float *utd_djvi;
    const float *utd_bux;
    const float *utd_buy;
    const float *utd_buz;
    const float *utd_bvx;
    const float *utd_bvy;
    const float *utd_bvz;
    const float *utd_bkx;
    const float *utd_bky;
    const float *utd_bkz;
    const float *utd_f0m00r;
    const float *utd_f0m00i;
    const float *utd_f0m01r;
    const float *utd_f0m01i;
    const float *utd_f0m10r;
    const float *utd_f0m10i;
    const float *utd_f0m11r;
    const float *utd_f0m11i;
    const float *utd_f1m00r;
    const float *utd_f1m00i;
    const float *utd_f1m01r;
    const float *utd_f1m01i;
    const float *utd_f1m10r;
    const float *utd_f1m10i;
    const float *utd_f1m11r;
    const float *utd_f1m11i;
    const float *utd_f0er;
    const float *utd_f0mu;
    const float *utd_f0sg;
    const float *utd_f0g;
    const float *utd_f0uf;
    const float *utd_f0pr;
    const float *utd_f1er;
    const float *utd_f1mu;
    const float *utd_f1sg;
    const float *utd_f1g;
    const float *utd_f1uf;
    const float *utd_f1pr;
    const float *utd_select;
    const float *const *coherent_utd_slots;
    int coherent_utd_slot_count;
    const int *coherent_owner_code;
    const int *coherent_adjacent_face0;
    const int *coherent_adjacent_face1;

    int recursive_state_count;
    const std::uint8_t *recursive_active_mask;
    const int *recursive_state_edge_index;
    const float *recursive_state_edge_pos_x;
    const float *recursive_state_edge_pos_y;
    const float *recursive_state_edge_pos_z;
    const float *recursive_state_edge_dir_x;
    const float *recursive_state_edge_dir_y;
    const float *recursive_state_edge_dir_z;
    const float *recursive_state_edge_t_min;
    const float *recursive_state_edge_t_max;
    const float *recursive_state_n0_x;
    const float *recursive_state_n0_y;
    const float *recursive_state_n0_z;
    const float *recursive_state_n1_x;
    const float *recursive_state_n1_y;
    const float *recursive_state_n1_z;
    const int *recursive_state_prim0;
    const int *recursive_state_prim1;
    const float *recursive_state_exterior_angle;

    int grid_axis;
    float grid_position;
    float grid_coord0_min;
    float grid_coord0_max;
    float grid_coord1_min;
    float grid_coord1_max;
    int grid_resolution0;
    int grid_resolution1;
    float grid_cell_area;

    const float *tri_p0_x;
    const float *tri_p0_y;
    const float *tri_p0_z;
    const float *tri_e1_x;
    const float *tri_e1_y;
    const float *tri_e1_z;
    const float *tri_e2_x;
    const float *tri_e2_y;
    const float *tri_e2_z;
    const float *tri_fn_x;
    const float *tri_fn_y;
    const float *tri_fn_z;
    const int *face_offsets;
    int n_meshes;
    int n_triangles;
    const std::uint32_t *suffix_candidate_prim_id;
    int suffix_candidate_count;

    const float *material_eta_r;
    const float *material_sigma;
    const float *material_mu_r;
    const float *material_gain;
    const std::uint8_t *material_valid;
    int material_count;

    float wavelength;
    float k;
    int seed;
    int samples;
    int max_order;
    int direct_samples;
    int keller_samples;
    int suffix_samples;
    int strategy_mask;
    int sample_sequence;
    int receiver_model;
    int select_diffraction_point;
    int prefilter_visibility;
    int collect_edge_use;
    int collect_debug_counts;
    float omega;
    float tx_pol_x;
    float tx_pol_y;
    float tx_pol_z;

    float *out_power;
    float *out_field_x_re;
    float *out_field_x_im;
    float *out_field_y_re;
    float *out_field_y_im;
    float *out_field_z_re;
    float *out_field_z_im;
    int *out_direct_count;
    int *out_keller_count;
    int *out_suffix_count;
    int *out_vis_rejects;
    int *out_edge_vis_rejects;
    int *out_utd_rejects;
    int *out_edge_uses;

    float *out_direct_field_x_re;
    float *out_direct_field_x_im;
    float *out_direct_field_y_re;
    float *out_direct_field_y_im;
    float *out_direct_field_z_re;
    float *out_direct_field_z_im;
    float *out_multi_field_x_re;
    float *out_multi_field_x_im;
    float *out_multi_field_y_re;
    float *out_multi_field_y_im;
    float *out_multi_field_z_re;
    float *out_multi_field_z_im;
    int *out_multi_count;
    int *out_visibility_reject_count;
    int *out_utd_reject_count;

    std::uint8_t *temp_visibility;
    std::uint8_t *tape_active;
    int *tape_state_idx;
    int *tape_cell;
    int *tape_material_idx;
    float *tape_edge_u;
};

struct HostDiffractionPolicy {
    struct CellGroup {};
    static constexpr int kDirect = 1;
    static constexpr int kKeller = 2;
    static constexpr int kSuffix = 4;
    static const HostDfrAccumParams &params() {
        static HostDfrAccumParams p{};
        return p;
    }
    static int sample_state_index_for_lane(unsigned int) { return 0; }
    static int state_edge_index_at(int) { return 0; }
    static Vec3f state_edge_pos_at(int) { return Vec3f{}; }
    static Vec3f state_edge_dir_at(int) { return Vec3f{0.f, 0.f, 1.f}; }
    static float state_edge_t_min_at(int) { return 0.f; }
    static float state_edge_t_max_at(int) { return 1.f; }
    static float sample_edge_weight_for_lane(int, unsigned int, int) { return 1.f; }
    static int state_prim0_at(int) { return -1; }
    static int state_prim1_at(int) { return -1; }
    static float state_exterior_angle_at(int) { return 1.f; }
    static float state_src_power_at(int) { return 1.f; }
    static Vec3f state_src_at(int) { return Vec3f{}; }
    static int recursive_state_edge_index_at(int) { return 0; }
    static Vec3f recursive_state_edge_pos_at(int) { return Vec3f{}; }
    static Vec3f recursive_state_edge_dir_at(int) { return Vec3f{0.f, 0.f, 1.f}; }
    static float recursive_state_edge_t_min_at(int) { return 0.f; }
    static float recursive_state_edge_t_max_at(int) { return 1.f; }
    static int recursive_state_prim0_at(int) { return -1; }
    static int recursive_state_prim1_at(int) { return -1; }
    static float recursive_state_exterior_angle_at(int) { return 1.f; }
    static bool material_valid_at(int) { return false; }
    static float material_gain_at(int) { return 1.f; }
    static bool active_state(int) { return true; }
    static bool recursive_active_state(int) { return true; }
    static CellGroup cell_group(int) { return {}; }
    static void atomic_add_same_cell(float *, int, float, CellGroup) {}
    static void atomic_add_same_cell(int *, int, int, CellGroup) {}
    static void atomic_add_warp(float *, float) {}
    static void atomic_add_warp(int *, int) {}
    static bool stage_order1(unsigned int, int, float, float, bool, bool) { return false; }
    static bool stage_coherent(int, int, bool, float, float, float, float, float, float) { return false; }
};
using HostDiffractionAlgo =
    rayd::shared::multipath::DiffractionAccumulationAlgo<HostDiffractionPolicy, HostTraverser>;
using DfrAccumPmf = void (HostDiffractionAlgo::*)(std::uint32_t) const;

}  // namespace

// Returning the address forces full host instantiation of each algorithm body.
AlgoFn rt_host_compile_smoke_reflection_trace() {
    return &rayd::shared::multipath::reflection_trace_algo<HostConfig>;
}

SegmentAlgoFn rt_host_compile_smoke_segment_visibility() {
    // Instantiating one entry pulls in the shared trace_segment core; take the
    // address of every launch variant so all four fully type-check off-device.
    volatile SegmentAlgoFn pair =
        &rayd::shared::multipath::segment_pair_visibility_algo<HostSegmentConfig>;
    volatile SegmentAlgoFn axial =
        &rayd::shared::multipath::axial_edge_visibility_algo<HostSegmentConfig>;
    volatile SegmentAlgoFn chain =
        &rayd::shared::multipath::segment_chain_visibility_algo<HostSegmentConfig>;
    (void)pair;
    (void)axial;
    (void)chain;
    return &rayd::shared::multipath::segment_visibility_algo<HostSegmentConfig>;
}

EpcAlgoFn rt_host_compile_smoke_reflection_epc() {
    // Exercise the DirectOnly / PrimaryVisibilityOnly template axes off-device.
    volatile EpcAlgoFn direct =
        &rayd::shared::multipath::run_reflection_epc_algo<HostEpcConfig, true, false>;
    volatile EpcAlgoFn primary_only =
        &rayd::shared::multipath::run_reflection_epc_algo<HostEpcConfig, false, true>;
    (void)direct;
    (void)primary_only;
    return &rayd::shared::multipath::run_reflection_epc_algo<HostEpcConfig, false, false>;
}

ReflAccumAlgoFn rt_host_compile_smoke_reflection_accumulation() {
    return &rayd::shared::multipath::reflection_accumulation_algo<
        HostAccumParams, HostAccumPolicy, HostTraverser>;
}

PathAlgoFn rt_host_compile_smoke_diffraction_paths() {
    // Exercise the SplitScene axis and both two-phase lanes off-device.
    volatile PathAlgoFn split =
        &rayd::shared::multipath::trace_paths_order1_algo<HostPathParams, HostTraverser, true>;
    volatile PathAlgoFn source_vis =
        &rayd::shared::multipath::trace_paths_source_visibility_algo<HostPathParams, HostTraverser>;
    volatile PathAlgoFn target_export =
        &rayd::shared::multipath::trace_paths_target_export_algo<HostPathParams, HostTraverser>;
    (void)split;
    (void)source_vis;
    (void)target_export;
    return &rayd::shared::multipath::trace_paths_order1_algo<HostPathParams, HostTraverser, false>;
}

DfrAccumPmf rt_host_compile_smoke_diffraction_accumulation() {
    // Force host instantiation of every diffraction-accumulation lane body.
    volatile DfrAccumPmf source_vis =
        &HostDiffractionAlgo::run_diffraction_order1_source_visibility_algo<true>;
    volatile DfrAccumPmf no_suffix =
        &HostDiffractionAlgo::run_diffraction_order1_no_suffix_target_accumulation_algo<true>;
    volatile DfrAccumPmf suffix_first =
        &HostDiffractionAlgo::run_diffraction_order1_suffix_first_visibility_algo<true>;
    volatile DfrAccumPmf suffix_target =
        &HostDiffractionAlgo::run_diffraction_order1_suffix_target_accumulation_algo<true>;
    volatile DfrAccumPmf coherent =
        &HostDiffractionAlgo::run_diffraction_order1_coherent_accumulation_algo<true>;
    volatile DfrAccumPmf chain =
        &HostDiffractionAlgo::run_diffraction_chain_accumulation_algo<false>;
    (void)source_vis;
    (void)no_suffix;
    (void)suffix_first;
    (void)suffix_target;
    (void)coherent;
    (void)chain;
    return &HostDiffractionAlgo::run_diffraction_order1_accumulation_algo<false, false, true, true, true>;
}
