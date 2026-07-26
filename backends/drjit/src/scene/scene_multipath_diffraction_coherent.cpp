#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include "scene_internal.h"
#include <rayd/multipath/diffraction_accumulation_ad.h>
#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>
#include <rayd/trace/cuda_multipath_gpu.h>

#include "scene_multipath_internal.h"

namespace rayd {

using namespace multipath_detail;

namespace {

struct DfrCoherentAccumRaw {
    int grid_cell_count = 0;
    Float direct_field_x_re;
    Float direct_field_x_im;
    Float direct_field_y_re;
    Float direct_field_y_im;
    Float direct_field_z_re;
    Float direct_field_z_im;
    Float multi_field_x_re;
    Float multi_field_x_im;
    Float multi_field_y_re;
    Float multi_field_y_im;
    Float multi_field_z_re;
    Float multi_field_z_im;
    Int direct_count;
    Int multi_count;
    Int visibility_reject_count;
    Int utd_reject_count;
};

DfrCoherentAccumRaw alloc_dfr_coherent_accum_raw(int grid_cell_count) {
    DfrCoherentAccumRaw raw;
    raw.grid_cell_count = grid_cell_count;
    raw.direct_field_x_re = empty<Float>(grid_cell_count);
    raw.direct_field_x_im = empty<Float>(grid_cell_count);
    raw.direct_field_y_re = empty<Float>(grid_cell_count);
    raw.direct_field_y_im = empty<Float>(grid_cell_count);
    raw.direct_field_z_re = empty<Float>(grid_cell_count);
    raw.direct_field_z_im = empty<Float>(grid_cell_count);
    raw.multi_field_x_re = empty<Float>(grid_cell_count);
    raw.multi_field_x_im = empty<Float>(grid_cell_count);
    raw.multi_field_y_re = empty<Float>(grid_cell_count);
    raw.multi_field_y_im = empty<Float>(grid_cell_count);
    raw.multi_field_z_re = empty<Float>(grid_cell_count);
    raw.multi_field_z_im = empty<Float>(grid_cell_count);
    raw.direct_count = empty<Int>(grid_cell_count);
    raw.multi_count = empty<Int>(grid_cell_count);
    raw.visibility_reject_count = empty<Int>(grid_cell_count);
    raw.utd_reject_count = empty<Int>(grid_cell_count);
    return raw;
}

void init_dfr_coherent_accum_raw(DfrCoherentAccumRaw &raw) {
    const int zero_i = 0;
    const float zero_f = 0.f;
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_x_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_x_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_y_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_y_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_z_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_field_z_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_x_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_x_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_y_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_y_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_z_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_field_z_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.direct_count.data(),
                     raw.grid_cell_count,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.multi_count.data(),
                     raw.grid_cell_count,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.visibility_reject_count.data(),
                     raw.grid_cell_count,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.utd_reject_count.data(),
                     raw.grid_cell_count,
                     sizeof(int),
                     &zero_i);
}

Vector3f normalize_with_fallback(const Vector3f &value, const Vector3f &fallback) {
    const Float value_norm = norm(value);
    const Float fallback_norm = norm(fallback);
    return select(value_norm > Float(1.0e-8f),
                  value / (value_norm + Float(1.0e-12f)),
                  fallback / (fallback_norm + Float(1.0e-12f)));
}

Vector3f stable_perpendicular_basis_jit(const Vector3f &ray_dir, const Vector3f &preferred) {
    const Vector3f proj = preferred - dot(preferred, ray_dir) * ray_dir;
    const Mask use_z = abs(ray_dir.z()) < Float(0.9f);
    const Vector3f alt_axis = select(use_z,
                                     Vector3f(Float(0.f), Float(0.f), Float(1.f)),
                                     Vector3f(Float(0.f), Float(1.f), Float(0.f)));
    const Vector3f alt_proj = alt_axis - dot(alt_axis, ray_dir) * ray_dir;
    return normalize_with_fallback(proj, alt_proj);
}

Mask wedge_exterior_mask_jit(const Vector3f &direction_from_edge,
                             const Vector3f &edge_dir,
                             const Vector3f &n0,
                             const Vector3f &nn) {
    const Vector3f direction_proj =
        direction_from_edge - dot(direction_from_edge, edge_dir) * edge_dir;
    return (norm(direction_proj) > Float(1.0e-8f)) &&
           ((dot(direction_proj, n0) >= Float(-1.0e-8f)) ||
            (dot(direction_proj, nn) >= Float(-1.0e-8f)));
}

Int interleave_two_ignore_slots(const Int &slot0, const Int &slot1, int width) {
    if (width <= 0) {
        return zeros<Int>(0);
    }
    const Int slot_major = concat(slot0, slot1);
    const UInt dst_idx = arange<UInt>(width * 2);
    const UInt ray_idx = dst_idx / UInt(2);
    const UInt slot_idx = dst_idx - ray_idx * UInt(2);
    const UInt src_idx = slot_idx * UInt(width) + ray_idx;
    return gather<Int>(slot_major, src_idx);
}

Int interleave_four_ignore_slots(const Int &slot0,
                                 const Int &slot1,
                                 const Int &slot2,
                                 const Int &slot3,
                                 int width) {
    if (width <= 0) {
        return zeros<Int>(0);
    }
    const Int slot_major = concat(concat(slot0, slot1), concat(slot2, slot3));
    const UInt dst_idx = arange<UInt>(width * 4);
    const UInt ray_idx = dst_idx / UInt(4);
    const UInt slot_idx = dst_idx - ray_idx * UInt(4);
    const UInt src_idx = slot_idx * UInt(width) + ray_idx;
    return gather<Int>(slot_major, src_idx);
}

Float gather_material_float(const Float &values,
                            const Int &face,
                            const Mask &valid,
                            float fallback) {
    const UInt safe = UInt(select(valid, face, Int(0)));
    return select(valid, gather<Float>(values, safe), Float(fallback));
}

Mask gather_material_mask(const Mask &values, const Int &face, const Mask &valid) {
    const UInt safe = UInt(select(valid, face, Int(0)));
    return valid && gather<Mask>(values, safe);
}

} // namespace

template <bool Detached>
DfrCoherentUtdStatesT<Detached> Scene::build_dfr_coherent_tx_states(
    const DfrCoherentEdgeT<Detached> &edges,
    const Vector3fT<Detached> &tx_position,
    const DfrMaterialT<Detached> &material,
    const DfrCoherentOptions &options,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::build_dfr_coherent_tx_states(): scene is not built.");
    require(!pending_updates_,
            "Scene::build_dfr_coherent_tx_states(): scene has pending updates. Call Scene::sync() first.");
    require(options.wavelength > 0.f,
            "Scene::build_dfr_coherent_tx_states(): wavelength must be positive.");
    require(options.k > 0.f,
            "Scene::build_dfr_coherent_tx_states(): k must be positive.");
    if constexpr (!Detached) {
        (void)edges;
        (void)tx_position;
        (void)material;
        (void)active;
        throw std::runtime_error(
            "Scene::build_dfr_coherent_tx_states(): AD inputs are not supported yet.");
    } else {
        const int edge_count = edges.count;
        require(edge_count >= 0,
                "Scene::build_dfr_coherent_tx_states(): invalid edge count.");
        DfrCoherentUtdStates result;
        if (edge_count == 0) {
            result.count = 0;
            return result;
        }
        require(static_cast<int>(slices(edges.edge_index)) >= edge_count &&
                    static_cast<int>(slices(edges.edge_pos)) >= edge_count &&
                    static_cast<int>(slices(edges.edge_dir)) >= edge_count &&
                    static_cast<int>(slices(edges.n0)) >= edge_count &&
                    static_cast<int>(slices(edges.n_face_n)) >= edge_count &&
                    static_cast<int>(slices(edges.wedge_n)) >= edge_count &&
                    static_cast<int>(slices(edges.edge_line_min)) >= edge_count &&
                    static_cast<int>(slices(edges.edge_line_max)) >= edge_count &&
                    static_cast<int>(slices(edges.adjacent_face0)) >= edge_count &&
                    static_cast<int>(slices(edges.adjacent_face1)) >= edge_count,
                "Scene::build_dfr_coherent_tx_states(): edge fields must cover edge count.");
        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::build_dfr_coherent_tx_states(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::build_dfr_coherent_tx_states(): material payload fields must have matching widths.");
        const int ignore_count = static_cast<int>(slices(edges.ignore_prim_ids));
        int ignore_k = edges.ignore_k;
        if (ignore_count > 0) {
            require(ignore_k > 0,
                    "Scene::build_dfr_coherent_tx_states(): ignore_k must be positive when ignore_prim_ids is provided.");
            require(ignore_count == edge_count * ignore_k,
                    "Scene::build_dfr_coherent_tx_states(): ignore_prim_ids width must equal edge count * ignore_k.");
        } else {
            ignore_k = 0;
        }

        Vector3f source_pos = tx_position;
        if (static_cast<int>(slices(source_pos)) == 1 && edge_count > 1) {
            source_pos = gather<Vector3f>(source_pos, zeros<UInt>(edge_count));
        } else {
            require(static_cast<int>(slices(source_pos)) == edge_count,
                    "Scene::build_dfr_coherent_tx_states(): tx_position width must be 1 or match edge count.");
        }
        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && edge_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<UInt>(edge_count));
        } else {
            require(active_width == edge_count,
                    "Scene::build_dfr_coherent_tx_states(): active width must be 1 or match edge count.");
        }

        const Vector3f edge_dir = normalize_with_fallback(
            edges.edge_dir, Vector3f(Float(0.f), Float(0.f), Float(1.f)));
        require(trace_backend_ != nullptr && optix_scene().is_ready(),
                "Scene::build_dfr_coherent_tx_states(): OptiX scene is not ready.");
        ensure_pipeline(segment_visibility_pipeline_,
                        optix_scene().context(),
                        mesh_count_,
                        segment_visibility_pipeline_config());
        const SegmentVisibility visibility_result =
            trace_segment_visibility_native<true>(optix_scene(),
                                                  *segment_visibility_pipeline_,
                                                  face_offsets_,
                                                  mesh_count_,
                                                  source_pos,
                                                  edges.edge_pos,
                                                  edges.ignore_prim_ids,
                                                  ignore_k,
                                                  active_detached);
        const Mask visibility = visibility_result.visible;
        const Mask source_exterior =
            wedge_exterior_mask_jit(source_pos - edges.edge_pos, edge_dir, edges.n0, edges.n_face_n);
        const Mask finite_line =
            (edges.edge_line_max - edges.edge_line_min) > Float(1.0e-8f);
        const Mask valid = visibility && source_exterior && finite_line && active_detached;
        const UInt keep = compress(valid);
        const int state_count = static_cast<int>(slices(keep));
        result.count = state_count;
        if (state_count == 0) {
            return result;
        }

        result.edge_index = gather<Int>(edges.edge_index, keep);
        result.edge_pos = gather<Vector3f>(edges.edge_pos, keep);
        result.edge_dir = gather<Vector3f>(edge_dir, keep);
        result.n0 = gather<Vector3f>(edges.n0, keep);
        result.n_face_n = gather<Vector3f>(edges.n_face_n, keep);
        result.wedge_n = gather<Float>(edges.wedge_n, keep);
        result.edge_line_min = gather<Float>(edges.edge_line_min, keep);
        result.edge_line_max = gather<Float>(edges.edge_line_max, keep);
        result.source_pos = gather<Vector3f>(source_pos, keep);
        result.adjacent_face0 = gather<Int>(edges.adjacent_face0, keep);
        result.adjacent_face1 = gather<Int>(edges.adjacent_face1, keep);

        const Vector3f source_to_edge = result.edge_pos - result.source_pos;
        const Float distance = norm(source_to_edge) + Float(1.0e-12f);
        const Float source_gain = Float(1.f) / (Float(2.f) * Float(options.k) * distance);
        const drjit::Complex<Float> phase =
            exp(drjit::Complex<Float>(zeros<Float>(state_count), -Float(options.k) * distance));
        result.incident_field = phase * source_gain;
        result.incident_normal_derivative =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));
        const Vector3f ray_dir = source_to_edge / distance;
        const Vector3f tx_pol(Float(options.tx_pol_x), Float(options.tx_pol_y), Float(options.tx_pol_z));
        const Vector3f pol_dir = stable_perpendicular_basis_jit(ray_dir, tx_pol);
        result.incident_vector_x = result.incident_field * pol_dir.x();
        result.incident_vector_y = result.incident_field * pol_dir.y();
        result.incident_vector_z = result.incident_field * pol_dir.z();
        result.incident_normal_derivative_vector_x =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));
        result.incident_normal_derivative_vector_y =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));
        result.incident_normal_derivative_vector_z =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));

        result.incident_basis_k = ray_dir;
        result.incident_basis_u = stable_perpendicular_basis_jit(ray_dir, result.edge_dir);
        result.incident_basis_v = normalize_with_fallback(
            cross(ray_dir, result.incident_basis_u),
            stable_perpendicular_basis_jit(ray_dir, Vector3f(Float(0.f), Float(1.f), Float(0.f))));
        result.incident_jones_u = result.incident_vector_x * result.incident_basis_u.x() +
                                  result.incident_vector_y * result.incident_basis_u.y() +
                                  result.incident_vector_z * result.incident_basis_u.z();
        result.incident_jones_v = result.incident_vector_x * result.incident_basis_v.x() +
                                  result.incident_vector_y * result.incident_basis_v.y() +
                                  result.incident_vector_z * result.incident_basis_v.z();
        result.incident_derivative_jones_u =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));
        result.incident_derivative_jones_v =
            drjit::Complex<Float>(zeros<Float>(state_count), zeros<Float>(state_count));

        const Mask valid0 = result.adjacent_face0 >= Int(0) &&
                            result.adjacent_face0 < Int(material_count);
        const Mask valid1 = result.adjacent_face1 >= Int(0) &&
                            result.adjacent_face1 < Int(material_count);
        const Mask mat0 = gather_material_mask(material.valid, result.adjacent_face0, valid0);
        const Mask mat1 = gather_material_mask(material.valid, result.adjacent_face1, valid1);
        result.face0_eta_r = gather_material_float(material.eta_r, result.adjacent_face0, mat0, 1.f);
        result.face0_mu_r = gather_material_float(material.mu_r, result.adjacent_face0, mat0, 1.f);
        result.face0_sigma = gather_material_float(material.sigma, result.adjacent_face0, mat0, 0.f);
        result.face0_gain = gather_material_float(material.gain, result.adjacent_face0, mat0, 1.f);
        result.face0_use_fresnel = Float(mat0);
        result.face1_eta_r = gather_material_float(material.eta_r, result.adjacent_face1, mat1, 1.f);
        result.face1_mu_r = gather_material_float(material.mu_r, result.adjacent_face1, mat1, 1.f);
        result.face1_sigma = gather_material_float(material.sigma, result.adjacent_face1, mat1, 0.f);
        result.face1_gain = gather_material_float(material.gain, result.adjacent_face1, mat1, 1.f);
        result.face1_use_fresnel = Float(mat1);

        const drjit::Complex<Float> zero_c(zeros<Float>(state_count), zeros<Float>(state_count));
        const drjit::Complex<Float> pec_c(full<Float>(-1.f, state_count), zeros<Float>(state_count));
        result.r_face0 = pec_c;
        result.r_face_n = pec_c;
        result.face0_operator_m00 = pec_c;
        result.face0_operator_m01 = zero_c;
        result.face0_operator_m10 = zero_c;
        result.face0_operator_m11 = pec_c;
        result.face1_operator_m00 = pec_c;
        result.face1_operator_m01 = zero_c;
        result.face1_operator_m10 = zero_c;
        result.face1_operator_m11 = pec_c;

        result.select_stationary_point = full<Float>(1.f, state_count);
        result.owner_code = zeros<Int>(state_count);
        result.path_length_prefix = distance;
        result.first_interaction_pos = result.edge_pos;
        result.source_type_code = zeros<Int>(state_count);
        result.prefix_reflection_depth = zeros<Int>(state_count);
        result.intermediate_reflection_depth = zeros<Int>(state_count);
        result.suffix_reflection_depth = zeros<Int>(state_count);
        result.approximation_mode_code = zeros<Int>(state_count);
        result.order = full<Int>(1, state_count);
        return result;
    }
}

template <bool Detached>
DfrCoherentCandidatePairsT<Detached> Scene::build_dfr_coherent_higher_candidates(
    const DfrCoherentUtdStatesT<Detached> &prev_states,
    const DfrCoherentEdgeT<Detached> &edges,
    const IntT<Detached> &global_to_local_edge_index,
    const DfrCoherentOptions &options,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::build_dfr_coherent_higher_candidates(): scene is not built.");
    require(!pending_updates_,
            "Scene::build_dfr_coherent_higher_candidates(): scene has pending updates. Call Scene::sync() first.");
    require(options.higher_probe_radius_scale > 0.f,
            "Scene::build_dfr_coherent_higher_candidates(): probe radius scale must be positive.");
    require(options.higher_probe_radius_min >= 0.f &&
                options.higher_probe_radius_min <= options.higher_probe_radius_max,
            "Scene::build_dfr_coherent_higher_candidates(): probe radius bounds must be ordered.");
    if constexpr (!Detached) {
        (void)prev_states;
        (void)edges;
        (void)global_to_local_edge_index;
        (void)active;
        throw std::runtime_error(
            "Scene::build_dfr_coherent_higher_candidates(): AD inputs are not supported yet.");
    } else {
        const int prev_count = prev_states.count;
        const int edge_count = edges.count;
        require(prev_count >= 0 && edge_count >= 0,
                "Scene::build_dfr_coherent_higher_candidates(): invalid state or edge count.");
        DfrCoherentCandidatePairs result;
        if (prev_count == 0 || edge_count == 0) {
            result.count = 0;
            return result;
        }
        require(static_cast<int>(slices(prev_states.edge_index)) >= prev_count &&
                    static_cast<int>(slices(prev_states.edge_pos)) >= prev_count &&
                    static_cast<int>(slices(prev_states.source_pos)) >= prev_count &&
                    static_cast<int>(slices(prev_states.incident_basis_u)) >= prev_count &&
                    static_cast<int>(slices(prev_states.incident_basis_v)) >= prev_count &&
                    static_cast<int>(slices(prev_states.incident_basis_k)) >= prev_count,
                "Scene::build_dfr_coherent_higher_candidates(): previous state fields must cover state count.");
        require(static_cast<int>(slices(global_to_local_edge_index)) > 0,
                "Scene::build_dfr_coherent_higher_candidates(): global-to-local edge index map must not be empty.");

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && prev_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<UInt>(prev_count));
        } else {
            require(active_width == prev_count,
                    "Scene::build_dfr_coherent_higher_candidates(): active width must be 1 or match previous state count.");
        }

        constexpr int probe_count = 18;
        constexpr int probe_grid_count = probe_count / 2;
        const int probe_lane_count = prev_count * probe_count;
        const UInt probe_idx = arange<UInt>(probe_lane_count);
        const UInt prev_idx_all = probe_idx / UInt(probe_count);
        const UInt probe_slot = probe_idx - prev_idx_all * UInt(probe_count);
        const UInt probe_grid_slot = probe_slot % UInt(probe_grid_count);
        const Float probe_u = Float(probe_grid_slot / UInt(3)) - Float(1.f);
        const Float probe_v = Float(probe_grid_slot % UInt(3)) - Float(1.f);
        const Float probe_sign = select(probe_slot < UInt(probe_grid_count), Float(1.f), Float(-1.f));

        const Vector3f edge_pos = gather<Vector3f>(prev_states.edge_pos, prev_idx_all);
        const Vector3f source_pos = gather<Vector3f>(prev_states.source_pos, prev_idx_all);
        const Vector3f basis_u = gather<Vector3f>(prev_states.incident_basis_u, prev_idx_all);
        const Vector3f basis_v = gather<Vector3f>(prev_states.incident_basis_v, prev_idx_all);
        const Vector3f basis_k = gather<Vector3f>(prev_states.incident_basis_k, prev_idx_all);
        const Int prev_edge_idx = gather<Int>(prev_states.edge_index, prev_idx_all);
        const Mask probe_active = gather<Mask>(active_detached, prev_idx_all);

        const Float source_distance = norm(edge_pos - source_pos);
        const Float unclamped_radius = source_distance * Float(options.higher_probe_radius_scale);
        const Float probe_radius =
            minimum(maximum(unclamped_radius, Float(options.higher_probe_radius_min)),
                    Float(options.higher_probe_radius_max));
        const Vector3f ray_origin = edge_pos +
                                    basis_u * (probe_radius * probe_u) +
                                    basis_v * (probe_radius * probe_v);
        const Vector3f ray_dir = basis_k * probe_sign;

        const NearestRayEdge nearest =
            this->template nearest_edge<true>(Ray(ray_origin, ray_dir), probe_active);
        Mask valid = nearest.global_edge_id >= Int(0);
        const Int safe_global_edge =
            select(valid, nearest.global_edge_id, Int(0));
        valid &= safe_global_edge < Int(static_cast<int>(slices(global_to_local_edge_index)));
        const Int local_edge_idx =
            gather<Int>(global_to_local_edge_index, UInt(safe_global_edge), valid);
        valid &= local_edge_idx >= Int(0);
        valid &= prev_edge_idx != local_edge_idx;
        valid &= probe_active;

        if (options.higher_filter_visibility) {
            require(static_cast<int>(slices(edges.edge_pos)) >= edge_count &&
                        static_cast<int>(slices(edges.adjacent_face0)) >= edge_count &&
                        static_cast<int>(slices(edges.adjacent_face1)) >= edge_count &&
                        static_cast<int>(slices(prev_states.adjacent_face0)) >= prev_count &&
                        static_cast<int>(slices(prev_states.adjacent_face1)) >= prev_count,
                    "Scene::build_dfr_coherent_higher_candidates(): visibility filtering requires edge positions and adjacent faces.");
            require(trace_backend_ != nullptr && optix_scene().is_ready(),
                    "Scene::build_dfr_coherent_higher_candidates(): OptiX scene is not ready.");
            const UInt safe_local_edge = UInt(select(valid, local_edge_idx, Int(0)));
            const Vector3f next_edge_pos = gather<Vector3f>(edges.edge_pos, safe_local_edge, valid);
            const Int prev_adjacent_face0 = gather<Int>(prev_states.adjacent_face0, prev_idx_all, valid);
            const Int prev_adjacent_face1 = gather<Int>(prev_states.adjacent_face1, prev_idx_all, valid);
            const Int next_adjacent_face0 = gather<Int>(edges.adjacent_face0, safe_local_edge, valid);
            const Int next_adjacent_face1 = gather<Int>(edges.adjacent_face1, safe_local_edge, valid);
            const Int ignore_prim_ids = interleave_four_ignore_slots(
                prev_adjacent_face0,
                prev_adjacent_face1,
                next_adjacent_face0,
                next_adjacent_face1,
                probe_lane_count);
            ensure_pipeline(segment_visibility_pipeline_,
                            optix_scene().context(),
                            mesh_count_,
                            segment_visibility_pipeline_config());
            const SegmentVisibility visibility_result =
                trace_segment_visibility_native<true>(optix_scene(),
                                                      *segment_visibility_pipeline_,
                                                      face_offsets_,
                                                      mesh_count_,
                                                      edge_pos,
                                                      next_edge_pos,
                                                      ignore_prim_ids,
                                                      4,
                                                      valid);
            valid &= visibility_result.visible;
            result.visibility_filtered = 1;
        }

        const UInt keep = compress(valid);
        const int candidate_count = static_cast<int>(slices(keep));
        result.count = candidate_count;
        if (candidate_count == 0) {
            return result;
        }
        result.prev_index = Int(gather<UInt>(prev_idx_all, keep));
        result.edge_index = gather<Int>(local_edge_idx, keep);
        return result;
    }
}

template <bool Detached>
DfrCoherentAccumT<Detached> Scene::accum_dfr_coherent_direct(
    const DfrCoherentUtdStatesT<Detached> &states,
    const DfrGrid &grid,
    const DfrCoherentOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::accum_dfr_coherent_direct(): scene is not built.");
    require(!pending_updates_,
            "Scene::accum_dfr_coherent_direct(): scene has pending updates. Call Scene::sync() first.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accum_dfr_coherent_direct(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accum_dfr_coherent_direct(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accum_dfr_coherent_direct(): grid bounds must be ordered.");
    require(options.wavelength > 0.f,
            "Scene::accum_dfr_coherent_direct(): wavelength must be positive.");
    require(options.max_order == 1,
            "Scene::accum_dfr_coherent_direct(): only max_order == 1 is supported.");
    require(options.receiver_model == RAYD_DFR_MATCHED_ISO,
            "Scene::accum_dfr_coherent_direct(): only matched isotropic receivers are supported.");
    if constexpr (!Detached) {
        (void)states;
        (void)active;
        throw std::runtime_error(
            "Scene::accum_dfr_coherent_direct(): AD inputs are not supported yet.");
    } else {
        const int state_count = states.count;
        require(state_count > 0,
                "Scene::accum_dfr_coherent_direct(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n_face_n)) >= state_count &&
                    static_cast<int>(slices(states.source_pos)) >= state_count &&
                    static_cast<int>(slices(states.incident_basis_u)) >= state_count &&
                    static_cast<int>(slices(states.incident_basis_v)) >= state_count &&
                    static_cast<int>(slices(states.incident_basis_k)) >= state_count &&
                    static_cast<int>(slices(states.wedge_n)) >= state_count &&
                    static_cast<int>(slices(states.edge_line_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_line_max)) >= state_count &&
                    static_cast<int>(slices(states.face0_eta_r)) >= state_count &&
                    static_cast<int>(slices(states.face0_mu_r)) >= state_count &&
                    static_cast<int>(slices(states.face0_sigma)) >= state_count &&
                    static_cast<int>(slices(states.face0_gain)) >= state_count &&
                    static_cast<int>(slices(states.face0_use_fresnel)) >= state_count &&
                    static_cast<int>(slices(states.face1_eta_r)) >= state_count &&
                    static_cast<int>(slices(states.face1_mu_r)) >= state_count &&
                    static_cast<int>(slices(states.face1_sigma)) >= state_count &&
                    static_cast<int>(slices(states.face1_gain)) >= state_count &&
                    static_cast<int>(slices(states.face1_use_fresnel)) >= state_count &&
                    static_cast<int>(slices(states.select_stationary_point)) >= state_count &&
                    static_cast<int>(slices(states.incident_field)) >= state_count &&
                    static_cast<int>(slices(states.incident_normal_derivative)) >= state_count &&
                    static_cast<int>(slices(states.r_face0)) >= state_count &&
                    static_cast<int>(slices(states.r_face_n)) >= state_count &&
                    static_cast<int>(slices(states.incident_vector_x)) >= state_count &&
                    static_cast<int>(slices(states.incident_vector_y)) >= state_count &&
                    static_cast<int>(slices(states.incident_vector_z)) >= state_count &&
                    static_cast<int>(slices(states.incident_normal_derivative_vector_x)) >= state_count &&
                    static_cast<int>(slices(states.incident_normal_derivative_vector_y)) >= state_count &&
                    static_cast<int>(slices(states.incident_normal_derivative_vector_z)) >= state_count &&
                    static_cast<int>(slices(states.incident_jones_u)) >= state_count &&
                    static_cast<int>(slices(states.incident_jones_v)) >= state_count &&
                    static_cast<int>(slices(states.incident_derivative_jones_u)) >= state_count &&
                    static_cast<int>(slices(states.incident_derivative_jones_v)) >= state_count &&
                    static_cast<int>(slices(states.face0_operator_m00)) >= state_count &&
                    static_cast<int>(slices(states.face0_operator_m01)) >= state_count &&
                    static_cast<int>(slices(states.face0_operator_m10)) >= state_count &&
                    static_cast<int>(slices(states.face0_operator_m11)) >= state_count &&
                    static_cast<int>(slices(states.face1_operator_m00)) >= state_count &&
                    static_cast<int>(slices(states.face1_operator_m01)) >= state_count &&
                    static_cast<int>(slices(states.face1_operator_m10)) >= state_count &&
                    static_cast<int>(slices(states.face1_operator_m11)) >= state_count &&
                    static_cast<int>(slices(states.owner_code)) >= state_count &&
                    static_cast<int>(slices(states.adjacent_face0)) >= state_count &&
                    static_cast<int>(slices(states.adjacent_face1)) >= state_count,
                "Scene::accum_dfr_coherent_direct(): full UTD state fields must cover state count.");
        const int grid_cell_count = grid.resolution0 * grid.resolution1;
        const int launch_count = state_count * grid_cell_count;
        require(launch_count > 0,
                "Scene::accum_dfr_coherent_direct(): invalid launch count.");

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && state_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(state_count));
        } else {
            require(active_width == state_count,
                    "Scene::accum_dfr_coherent_direct(): active width must be 1 or match state count.");
        }
        active_detached &= drjit::isfinite(states.source_pos.x()) &&
                           drjit::isfinite(states.source_pos.y()) &&
                           drjit::isfinite(states.source_pos.z()) &&
                           drjit::isfinite(states.edge_pos.x()) &&
                           drjit::isfinite(states.edge_pos.y()) &&
                           drjit::isfinite(states.edge_pos.z());

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr_coherent_direct(): OptiX scene is not ready.");
            require(scenes.hitgroup_record_count > 0,
                    "Scene::accum_dfr_coherent_direct(): invalid hitgroup record count.");

            auto &dfr_pipeline = split_mode == 0
                ? diffraction_coherent_accumulation_primary_pipeline_
                : diffraction_coherent_accumulation_pipeline_;
            const OptixPipelineConfig dfr_pipeline_config = split_mode == 0
                ? diffraction_coherent_accumulation_primary_pipeline_config()
                : diffraction_coherent_accumulation_pipeline_config();

            ensure_pipeline(dfr_pipeline,
                            primary_scene->context(),
                            scenes.hitgroup_record_count,
                            dfr_pipeline_config);
        }

        drjit::eval(states.edge_pos,
                    states.edge_dir,
                    states.n0,
                    states.n_face_n,
                    states.source_pos,
                    states.incident_basis_u,
                    states.incident_basis_v,
                    states.incident_basis_k,
                    states.wedge_n,
                    states.edge_line_min,
                    states.edge_line_max,
                    states.face0_eta_r,
                    states.face0_mu_r,
                    states.face0_sigma,
                    states.face0_gain,
                    states.face0_use_fresnel,
                    states.face1_eta_r,
                    states.face1_mu_r,
                    states.face1_sigma,
                    states.face1_gain,
                    states.face1_use_fresnel,
                    states.select_stationary_point,
                    states.incident_field,
                    states.incident_normal_derivative,
                    states.r_face0,
                    states.r_face_n,
                    states.incident_vector_x,
                    states.incident_vector_y,
                    states.incident_vector_z,
                    states.incident_normal_derivative_vector_x,
                    states.incident_normal_derivative_vector_y,
                    states.incident_normal_derivative_vector_z,
                    states.incident_jones_u,
                    states.incident_jones_v,
                    states.incident_derivative_jones_u,
                    states.incident_derivative_jones_v,
                    states.face0_operator_m00,
                    states.face0_operator_m01,
                    states.face0_operator_m10,
                    states.face0_operator_m11,
                    states.face1_operator_m00,
                    states.face1_operator_m01,
                    states.face1_operator_m10,
                    states.face1_operator_m11,
                    states.owner_code,
                    states.adjacent_face0,
                    states.adjacent_face1,
                    active_detached,
                    triangle_info_detached_.face_normal,
                    face_offsets_);

        DfrCoherentAccum result;
        result.grid_cell_count = grid_cell_count;
        DfrCoherentAccumRaw raw = alloc_dfr_coherent_accum_raw(grid_cell_count);
        init_dfr_coherent_accum_raw(raw);

        DfrAccumParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.n_rays = launch_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.state_count = state_count;
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.grid_cell_area = grid.cell_area;
        params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        params.face_offsets = face_offsets_.data();
        params.n_meshes = mesh_count_;
        params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.max_order = options.max_order;
        params.receiver_model = options.receiver_model;
        params.select_diffraction_point = options.select_diffraction_point ? 1 : 0;
        params.prefilter_visibility = options.prefilter_visibility ? 1 : 0;
        params.collect_debug_counts = options.collect_debug_counts ? 1 : 0;
        params.omega = options.omega;
        params.tx_pol_x = options.tx_pol_x;
        params.tx_pol_y = options.tx_pol_y;
        params.tx_pol_z = options.tx_pol_z;
        params.coherent_utd_slot_count = 84;
        params.utd_epx = states.edge_pos.x().data();
        params.utd_epy = states.edge_pos.y().data();
        params.utd_epz = states.edge_pos.z().data();
        params.utd_edx = states.edge_dir.x().data();
        params.utd_edy = states.edge_dir.y().data();
        params.utd_edz = states.edge_dir.z().data();
        params.utd_n0x = states.n0.x().data();
        params.utd_n0y = states.n0.y().data();
        params.utd_n0z = states.n0.z().data();
        params.utd_nnx = states.n_face_n.x().data();
        params.utd_nny = states.n_face_n.y().data();
        params.utd_nnz = states.n_face_n.z().data();
        params.utd_wn = states.wedge_n.data();
        params.utd_elm = states.edge_line_min.data();
        params.utd_elx = states.edge_line_max.data();
        params.utd_spx = states.source_pos.x().data();
        params.utd_spy = states.source_pos.y().data();
        params.utd_spz = states.source_pos.z().data();
        params.utd_ifr = drjit::real(states.incident_field).data();
        params.utd_ifi = drjit::imag(states.incident_field).data();
        params.utd_inr = drjit::real(states.incident_normal_derivative).data();
        params.utd_ini = drjit::imag(states.incident_normal_derivative).data();
        params.utd_r0r = drjit::real(states.r_face0).data();
        params.utd_r0i = drjit::imag(states.r_face0).data();
        params.utd_rnr = drjit::real(states.r_face_n).data();
        params.utd_rni = drjit::imag(states.r_face_n).data();
        params.utd_vxr = drjit::real(states.incident_vector_x).data();
        params.utd_vxi = drjit::imag(states.incident_vector_x).data();
        params.utd_vyr = drjit::real(states.incident_vector_y).data();
        params.utd_vyi = drjit::imag(states.incident_vector_y).data();
        params.utd_vzr = drjit::real(states.incident_vector_z).data();
        params.utd_vzi = drjit::imag(states.incident_vector_z).data();
        params.utd_dxr = drjit::real(states.incident_normal_derivative_vector_x).data();
        params.utd_dxi = drjit::imag(states.incident_normal_derivative_vector_x).data();
        params.utd_dyr = drjit::real(states.incident_normal_derivative_vector_y).data();
        params.utd_dyi = drjit::imag(states.incident_normal_derivative_vector_y).data();
        params.utd_dzr = drjit::real(states.incident_normal_derivative_vector_z).data();
        params.utd_dzi = drjit::imag(states.incident_normal_derivative_vector_z).data();
        params.utd_jur = drjit::real(states.incident_jones_u).data();
        params.utd_jui = drjit::imag(states.incident_jones_u).data();
        params.utd_jvr = drjit::real(states.incident_jones_v).data();
        params.utd_jvi = drjit::imag(states.incident_jones_v).data();
        params.utd_djur = drjit::real(states.incident_derivative_jones_u).data();
        params.utd_djui = drjit::imag(states.incident_derivative_jones_u).data();
        params.utd_djvr = drjit::real(states.incident_derivative_jones_v).data();
        params.utd_djvi = drjit::imag(states.incident_derivative_jones_v).data();
        params.utd_bux = states.incident_basis_u.x().data();
        params.utd_buy = states.incident_basis_u.y().data();
        params.utd_buz = states.incident_basis_u.z().data();
        params.utd_bvx = states.incident_basis_v.x().data();
        params.utd_bvy = states.incident_basis_v.y().data();
        params.utd_bvz = states.incident_basis_v.z().data();
        params.utd_bkx = states.incident_basis_k.x().data();
        params.utd_bky = states.incident_basis_k.y().data();
        params.utd_bkz = states.incident_basis_k.z().data();
        params.utd_f0m00r = drjit::real(states.face0_operator_m00).data();
        params.utd_f0m00i = drjit::imag(states.face0_operator_m00).data();
        params.utd_f0m01r = drjit::real(states.face0_operator_m01).data();
        params.utd_f0m01i = drjit::imag(states.face0_operator_m01).data();
        params.utd_f0m10r = drjit::real(states.face0_operator_m10).data();
        params.utd_f0m10i = drjit::imag(states.face0_operator_m10).data();
        params.utd_f0m11r = drjit::real(states.face0_operator_m11).data();
        params.utd_f0m11i = drjit::imag(states.face0_operator_m11).data();
        params.utd_f1m00r = drjit::real(states.face1_operator_m00).data();
        params.utd_f1m00i = drjit::imag(states.face1_operator_m00).data();
        params.utd_f1m01r = drjit::real(states.face1_operator_m01).data();
        params.utd_f1m01i = drjit::imag(states.face1_operator_m01).data();
        params.utd_f1m10r = drjit::real(states.face1_operator_m10).data();
        params.utd_f1m10i = drjit::imag(states.face1_operator_m10).data();
        params.utd_f1m11r = drjit::real(states.face1_operator_m11).data();
        params.utd_f1m11i = drjit::imag(states.face1_operator_m11).data();
        params.utd_f0er = states.face0_eta_r.data();
        params.utd_f0mu = states.face0_mu_r.data();
        params.utd_f0sg = states.face0_sigma.data();
        params.utd_f0g = states.face0_gain.data();
        params.utd_f0uf = states.face0_use_fresnel.data();
        params.utd_f1er = states.face1_eta_r.data();
        params.utd_f1mu = states.face1_mu_r.data();
        params.utd_f1sg = states.face1_sigma.data();
        params.utd_f1g = states.face1_gain.data();
        params.utd_f1uf = states.face1_use_fresnel.data();
        params.utd_select = states.select_stationary_point.data();
        params.coherent_owner_code = states.owner_code.data();
        params.coherent_adjacent_face0 = states.adjacent_face0.data();
        params.coherent_adjacent_face1 = states.adjacent_face1.data();
        params.out_direct_count = raw.direct_count.data();
        params.out_direct_field_x_re = raw.direct_field_x_re.data();
        params.out_direct_field_x_im = raw.direct_field_x_im.data();
        params.out_direct_field_y_re = raw.direct_field_y_re.data();
        params.out_direct_field_y_im = raw.direct_field_y_im.data();
        params.out_direct_field_z_re = raw.direct_field_z_re.data();
        params.out_direct_field_z_im = raw.direct_field_z_im.data();
        params.out_multi_field_x_re = raw.multi_field_x_re.data();
        params.out_multi_field_x_im = raw.multi_field_x_im.data();
        params.out_multi_field_y_re = raw.multi_field_y_re.data();
        params.out_multi_field_y_im = raw.multi_field_y_im.data();
        params.out_multi_field_z_re = raw.multi_field_z_re.data();
        params.out_multi_field_z_im = raw.multi_field_z_im.data();
        params.out_multi_count = raw.multi_count.data();
        params.out_visibility_reject_count = raw.visibility_reject_count.data();
        params.out_utd_reject_count = raw.utd_reject_count.data();

        if (cuda_trace) {
            cuda_backend().run_dfr_accum_coherent(params, launch_count);
        } else {
            (split_mode == 0 ? diffraction_coherent_accumulation_primary_pipeline_
                             : diffraction_coherent_accumulation_pipeline_)
                ->launch(0, params);
        }

        result.direct_field_x = drjit::Complex<Float>(raw.direct_field_x_re, raw.direct_field_x_im);
        result.direct_field_y = drjit::Complex<Float>(raw.direct_field_y_re, raw.direct_field_y_im);
        result.direct_field_z = drjit::Complex<Float>(raw.direct_field_z_re, raw.direct_field_z_im);
        result.multi_field_x = drjit::Complex<Float>(raw.multi_field_x_re, raw.multi_field_x_im);
        result.multi_field_y = drjit::Complex<Float>(raw.multi_field_y_re, raw.multi_field_y_im);
        result.multi_field_z = drjit::Complex<Float>(raw.multi_field_z_re, raw.multi_field_z_im);
        result.direct_count = raw.direct_count;
        result.multi_count = raw.multi_count;
        result.visibility_reject_count = raw.visibility_reject_count;
        result.utd_reject_count = raw.utd_reject_count;
        return result;
    }
}

template <bool Detached>
DfrCoherentAccumT<Detached> Scene::accum_dfr_coherent_direct(
    const DfrStatesT<Detached> &states,
    const DfrGrid &grid,
    const DfrMaterialT<Detached> &material,
    const DfrCoherentOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::accum_dfr_coherent_direct(): scene is not built.");
    require(!pending_updates_,
            "Scene::accum_dfr_coherent_direct(): scene has pending updates. Call Scene::sync() first.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accum_dfr_coherent_direct(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accum_dfr_coherent_direct(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accum_dfr_coherent_direct(): grid bounds must be ordered.");
    require(grid.cell_area > 0.f,
            "Scene::accum_dfr_coherent_direct(): grid.cell_area must be positive.");
    require(options.wavelength > 0.f,
            "Scene::accum_dfr_coherent_direct(): wavelength must be positive.");
    require(options.max_order == 1,
            "Scene::accum_dfr_coherent_direct(): only max_order == 1 is supported.");
    require(options.receiver_model == RAYD_DFR_MATCHED_ISO,
            "Scene::accum_dfr_coherent_direct(): only matched isotropic receivers are supported.");
    if constexpr (!Detached) {
        (void)states;
        (void)material;
        (void)active;
        throw std::runtime_error(
            "Scene::accum_dfr_coherent_direct(): AD inputs are not supported yet.");
    } else {
        const int state_count = states.count;
        require(state_count > 0,
                "Scene::accum_dfr_coherent_direct(): invalid state count.");
        require(static_cast<int>(slices(states.edge_index)) >= state_count &&
                    static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count &&
                    static_cast<int>(slices(states.wi)) >= state_count &&
                    static_cast<int>(slices(states.d0)) >= state_count &&
                    static_cast<int>(slices(states.prefix_depth)) >= state_count,
                "Scene::accum_dfr_coherent_direct(): state fields must cover state count.");
        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr_coherent_direct(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr_coherent_direct(): material payload fields must have matching widths.");

        DfrCoherentAccum result;
        const int grid_cell_count = grid.resolution0 * grid.resolution1;
        const int launch_count = state_count * grid_cell_count;
        result.grid_cell_count = grid_cell_count;
        require(launch_count > 0,
                "Scene::accum_dfr_coherent_direct(): invalid launch count.");

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && state_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(state_count));
        } else {
            require(active_width == state_count,
                    "Scene::accum_dfr_coherent_direct(): active width must be 1 or match state count.");
        }
        active_detached &= drjit::isfinite(states.src.x()) &&
                           drjit::isfinite(states.src.y()) &&
                           drjit::isfinite(states.src.z()) &&
                           drjit::isfinite(states.edge_pos.x()) &&
                           drjit::isfinite(states.edge_pos.y()) &&
                           drjit::isfinite(states.edge_pos.z()) &&
                           drjit::isfinite(states.src_power);

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr_coherent_direct(): OptiX scene is not ready.");
            require(scenes.hitgroup_record_count > 0,
                    "Scene::accum_dfr_coherent_direct(): invalid hitgroup record count.");

            auto &dfr_pipeline = split_mode == 0
                ? diffraction_coherent_accumulation_primary_pipeline_
                : diffraction_coherent_accumulation_pipeline_;
            const OptixPipelineConfig dfr_pipeline_config = split_mode == 0
                ? diffraction_coherent_accumulation_primary_pipeline_config()
                : diffraction_coherent_accumulation_pipeline_config();

            ensure_pipeline(dfr_pipeline,
                            primary_scene->context(),
                            scenes.hitgroup_record_count,
                            dfr_pipeline_config);
        }

        drjit::eval(states.edge_index,
                    states.edge_pos,
                    states.edge_dir,
                    states.edge_t_min,
                    states.edge_t_max,
                    states.n0,
                    states.n1,
                    states.prim0,
                    states.prim1,
                    states.exterior_angle,
                    states.src,
                    states.src_power,
                    states.wi,
                    states.d0,
                    states.prefix_depth,
                    active_detached,
                    material.eta_r,
                    material.sigma,
                    material.mu_r,
                    material.gain,
                    material.valid);

        DfrCoherentAccumRaw raw =
            alloc_dfr_coherent_accum_raw(grid_cell_count);
        init_dfr_coherent_accum_raw(raw);

        DfrAccumParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.n_rays = launch_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.state_count = state_count;
        params.state_edge_index = states.edge_index.data();
        params.state_edge_pos_x = states.edge_pos.x().data();
        params.state_edge_pos_y = states.edge_pos.y().data();
        params.state_edge_pos_z = states.edge_pos.z().data();
        params.state_edge_dir_x = states.edge_dir.x().data();
        params.state_edge_dir_y = states.edge_dir.y().data();
        params.state_edge_dir_z = states.edge_dir.z().data();
        params.state_edge_t_min = states.edge_t_min.data();
        params.state_edge_t_max = states.edge_t_max.data();
        params.state_n0_x = states.n0.x().data();
        params.state_n0_y = states.n0.y().data();
        params.state_n0_z = states.n0.z().data();
        params.state_n1_x = states.n1.x().data();
        params.state_n1_y = states.n1.y().data();
        params.state_n1_z = states.n1.z().data();
        params.state_prim0 = states.prim0.data();
        params.state_prim1 = states.prim1.data();
        params.state_exterior_angle = states.exterior_angle.data();
        params.state_src_x = states.src.x().data();
        params.state_src_y = states.src.y().data();
        params.state_src_z = states.src.z().data();
        params.state_src_power = states.src_power.data();
        params.state_wi_x = states.wi.x().data();
        params.state_wi_y = states.wi.y().data();
        params.state_wi_z = states.wi.z().data();
        params.state_d0_x = states.d0.x().data();
        params.state_d0_y = states.d0.y().data();
        params.state_d0_z = states.d0.z().data();
        params.state_prefix_depth = states.prefix_depth.data();
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.grid_cell_area = grid.cell_area;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_mu_r = material.mu_r.data();
        params.material_gain = material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.max_order = options.max_order;
        params.receiver_model = options.receiver_model;
        params.select_diffraction_point = options.select_diffraction_point ? 1 : 0;
        params.prefilter_visibility = options.prefilter_visibility ? 1 : 0;
        params.collect_debug_counts = options.collect_debug_counts ? 1 : 0;
        params.out_direct_count = raw.direct_count.data();
        params.out_direct_field_x_re = raw.direct_field_x_re.data();
        params.out_direct_field_x_im = raw.direct_field_x_im.data();
        params.out_direct_field_y_re = raw.direct_field_y_re.data();
        params.out_direct_field_y_im = raw.direct_field_y_im.data();
        params.out_direct_field_z_re = raw.direct_field_z_re.data();
        params.out_direct_field_z_im = raw.direct_field_z_im.data();
        params.out_multi_field_x_re = raw.multi_field_x_re.data();
        params.out_multi_field_x_im = raw.multi_field_x_im.data();
        params.out_multi_field_y_re = raw.multi_field_y_re.data();
        params.out_multi_field_y_im = raw.multi_field_y_im.data();
        params.out_multi_field_z_re = raw.multi_field_z_re.data();
        params.out_multi_field_z_im = raw.multi_field_z_im.data();
        params.out_multi_count = raw.multi_count.data();
        params.out_visibility_reject_count =
            raw.visibility_reject_count.data();
        params.out_utd_reject_count = raw.utd_reject_count.data();

        if (cuda_trace) {
            cuda_backend().run_dfr_accum_coherent(params, launch_count);
        } else {
            (split_mode == 0 ? diffraction_coherent_accumulation_primary_pipeline_
                             : diffraction_coherent_accumulation_pipeline_)
                ->launch(0, params);
        }

        result.direct_field_x =
            drjit::Complex<Float>(raw.direct_field_x_re, raw.direct_field_x_im);
        result.direct_field_y =
            drjit::Complex<Float>(raw.direct_field_y_re, raw.direct_field_y_im);
        result.direct_field_z =
            drjit::Complex<Float>(raw.direct_field_z_re, raw.direct_field_z_im);
        result.multi_field_x =
            drjit::Complex<Float>(raw.multi_field_x_re, raw.multi_field_x_im);
        result.multi_field_y =
            drjit::Complex<Float>(raw.multi_field_y_re, raw.multi_field_y_im);
        result.multi_field_z =
            drjit::Complex<Float>(raw.multi_field_z_re, raw.multi_field_z_im);
        result.direct_count = raw.direct_count;
        result.multi_count = raw.multi_count;
        result.visibility_reject_count = raw.visibility_reject_count;
        result.utd_reject_count = raw.utd_reject_count;
        return result;
    }
}

template DfrCoherentAccum Scene::accum_dfr_coherent_direct<true>(
    const DfrCoherentUtdStates &states,
    const DfrGrid &grid,
    const DfrCoherentOptions &options,
    Mask active) const;
template DfrCoherentAccumAD Scene::accum_dfr_coherent_direct<false>(
    const DfrCoherentUtdStatesAD &states,
    const DfrGrid &grid,
    const DfrCoherentOptions &options,
    MaskAD active) const;
template DfrCoherentUtdStates Scene::build_dfr_coherent_tx_states<true>(
    const DfrCoherentEdge &edges,
    const Vector3f &tx_position,
    const DfrMaterial &material,
    const DfrCoherentOptions &options,
    Mask active) const;
template DfrCoherentUtdStatesAD Scene::build_dfr_coherent_tx_states<false>(
    const DfrCoherentEdgeAD &edges,
    const Vector3fAD &tx_position,
    const DfrMaterialAD &material,
    const DfrCoherentOptions &options,
    MaskAD active) const;
template DfrCoherentCandidatePairs Scene::build_dfr_coherent_higher_candidates<true>(
    const DfrCoherentUtdStates &prev_states,
    const DfrCoherentEdge &edges,
    const Int &global_to_local_edge_index,
    const DfrCoherentOptions &options,
    Mask active) const;
template DfrCoherentCandidatePairsAD Scene::build_dfr_coherent_higher_candidates<false>(
    const DfrCoherentUtdStatesAD &prev_states,
    const DfrCoherentEdgeAD &edges,
    const IntAD &global_to_local_edge_index,
    const DfrCoherentOptions &options,
    MaskAD active) const;
template DfrCoherentAccum Scene::accum_dfr_coherent_direct<true>(
    const DfrStates &states,
    const DfrGrid &grid,
    const DfrMaterial &material,
    const DfrCoherentOptions &options,
    Mask active) const;
template DfrCoherentAccumAD Scene::accum_dfr_coherent_direct<false>(
    const DfrStatesAD &states,
    const DfrGrid &grid,
    const DfrMaterialAD &material,
    const DfrCoherentOptions &options,
    MaskAD active) const;

} // namespace rayd
