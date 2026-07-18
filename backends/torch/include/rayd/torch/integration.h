#pragma once

#include <ATen/ATen.h>

#include <cstdint>

// Source-level integration contract for consumers built in the SAME CMake
// GRAPH as rayd_torch_native_core. Every caller must use the exact compiler,
// CRT, CUDA toolkit, LibTorch headers/libraries, compile definitions, and link
// configuration selected by that graph.
//
// This is NOT an independently binary-stable C ABI and must not be exposed as
// a wheel-to-wheel FFI boundary. `extern "C"` only fixes symbol name mangling;
// it does not stabilize at::Tensor layout, ownership, exceptions, the C++
// runtime, or LibTorch ABI. Independently built plugins must use dispatcher
// schemas or a separately versioned POD C ABI instead of this header.

extern "C" int64_t rayd_torch_native_scene_create(
    const at::Tensor *vertices,
    const at::Tensor *faces,
    const at::Tensor *uv,
    const at::Tensor *face_uv,
    const at::Tensor *to_world_left,
    const at::Tensor *to_world_right,
    const int64_t *mesh_flags,
    int64_t mesh_count);

extern "C" void rayd_torch_native_scene_destroy(int64_t scene_handle);

extern "C" int64_t rayd_torch_native_scene_edge_records(
    int64_t scene_handle,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_intersect_forward(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    int64_t flags,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" void rayd_torch_native_visibility_forward(
    int64_t scene_handle,
    const at::Tensor *start,
    const at::Tensor *end,
    const at::Tensor *active,
    at::Tensor *visible,
    at::Tensor *blocker_prim,
    at::Tensor *tape_t);

// Differentiable-geometry companions (fixed-winner contract). The tape
// tensors (prim ids, barycentrics, hit points, normals) are detached winner
// records produced by the matching forward; backward/jvp entry points
// recompute continuous quantities from them and never differentiate the
// discrete winner selection itself.
//
// All backward/jvp entries validate their inputs host-side before any kernel
// launch: per-tensor dtype/device/rank checks, cross-tensor ray-batch and
// bounce-count consistency, gradient/tangent shapes against their primals,
// and vertex tangents against the scene's global vertex table. Violations
// throw std::runtime_error. Empty-tape semantics differ by direction:
// rayd_torch_native_intersect_backward accepts a defined-but-empty (numel 0)
// tape_barycentric and recomputes barycentrics on the fly (width-0 path),
// while rayd_torch_native_intersect_jvp has no recompute path and rejects an
// empty barycentric tape for a nonzero ray batch. The reflection-chain and
// EPC companions require full tapes in both directions.

extern "C" int64_t rayd_torch_native_intersect_backward(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    const at::Tensor *tape_prim_id,
    const at::Tensor *tape_barycentric,
    const at::Tensor *grad_t,
    const at::Tensor *grad_p,
    const at::Tensor *grad_n,
    const at::Tensor *grad_geo_n,
    const at::Tensor *grad_uv,
    const at::Tensor *grad_barycentric,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_intersect_jvp(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *active,
    const at::Tensor *tape_prim_id,
    const at::Tensor *tape_barycentric,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d,
    int64_t flags,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_trace_reflections_forward(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    int64_t max_bounces,
    at::Tensor *outputs,
    int64_t output_capacity);

// Tape-emitting variant of the reflection chain forward. Emits the nine
// tensors of the internal AD forward: valid, t, image_sources, prim_ids,
// tape_prim_id (alias of prim_ids), tape_barycentric, tape_hit_points,
// tape_normals, active_ctx.
extern "C" int64_t rayd_torch_native_trace_reflections_forward_tape(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    int64_t max_bounces,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_trace_reflections_backward(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    const at::Tensor *tape_prim_id,
    const at::Tensor *tape_barycentric,
    const at::Tensor *tape_hit_points,
    const at::Tensor *tape_normals,
    const at::Tensor *image_sources,
    const at::Tensor *grad_t,
    const at::Tensor *grad_image_sources,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_trace_reflections_jvp(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *active,
    const at::Tensor *tape_prim_id,
    const at::Tensor *tape_barycentric,
    const at::Tensor *tape_hit_points,
    const at::Tensor *tape_normals,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d,
    const at::Tensor *image_sources,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_reflection_accumulation_forward(
    int64_t scene_handle,
    const at::Tensor *ray_o,
    const at::Tensor *ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    const at::Tensor *tx,
    const at::Tensor *tx_pol,
    const at::Tensor *material_eta_r,
    const at::Tensor *material_sigma,
    const at::Tensor *material_mu_r,
    const at::Tensor *material_gain,
    const at::Tensor *material_valid,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength,
    double solid_angle_per_ray,
    bool collect_wedges,
    bool collect_wedge_prefixes,
    int64_t wedge_capacity,
    int64_t wedge_sample_stride,
    int64_t accumulation_strategy,
    int64_t compact_min_samples,
    int64_t staged_min_samples_per_cell,
    int64_t procedural_sample_count,
    bool include_los,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_reflection_epc_paths_forward(
    int64_t scene_handle,
    const at::Tensor *source,
    const at::Tensor *receiver,
    const at::Tensor *active,
    const at::Tensor *expected_prim_ids,
    const at::Tensor *direct_plane_points,
    const at::Tensor *direct_plane_normals,
    const at::Tensor *surface_group_id,
    const at::Tensor *surface_group_size,
    const at::Tensor *surface_group_members,
    int64_t max_bounces,
    int64_t visibility_ignore_mode,
    double plane_tolerance,
    at::Tensor *outputs,
    int64_t output_capacity);

// Fixed-winner geometry companions of the reflection EPC path export
// (direct-plane mode). `sequence` is the frozen winner face sequence and
// `plane_points` / `plane_normals` are exactly the direct-plane arrays the
// matching forward consumed; `valid` / `bounce_count` are frozen discovery
// records. The entries differentiate only the continuous chain geometry (the
// mirror loop, the plane intersections and the path length) and chain each
// bounce's plane cotangents to the winner triangle's vertices, so no OptiX is
// involved. Gradients/tangents may be strided; null gradient/tangent inputs
// are treated as zero and invalid rows contribute nothing.

extern "C" int64_t rayd_torch_native_reflection_epc_paths_backward(
    int64_t scene_handle,
    const at::Tensor *source,
    const at::Tensor *receiver,
    const at::Tensor *sequence,
    const at::Tensor *plane_points,
    const at::Tensor *plane_normals,
    const at::Tensor *valid,
    const at::Tensor *bounce_count,
    const at::Tensor *grad_points,
    const at::Tensor *grad_normals,
    const at::Tensor *grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_reflection_epc_paths_jvp(
    int64_t scene_handle,
    const at::Tensor *source,
    const at::Tensor *receiver,
    const at::Tensor *sequence,
    const at::Tensor *plane_points,
    const at::Tensor *plane_normals,
    const at::Tensor *valid,
    const at::Tensor *bounce_count,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_source,
    const at::Tensor *tangent_receiver,
    at::Tensor *outputs,
    int64_t output_capacity);

// Adjoint / tangent of the scene's unit face-normal table
// cross(v1 - v0, v2 - v0) / fmaxf(|cross|, 1e-6) with respect to the global
// vertex table. The 1e-6 denominator clamp matches how the consumer builds
// the table from the raw edge-record cross products (kEpcFaceNormalMinNorm
// in shared/reflection/epc_chain.h).

extern "C" int64_t rayd_torch_native_scene_face_normals_backward(
    int64_t scene_handle,
    const at::Tensor *grad_face_normals,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_scene_face_normals_jvp(
    int64_t scene_handle,
    const at::Tensor *tangent_vertices,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_diffraction_paths_order1_forward(
    int64_t scene_handle,
    const at::Tensor *tx_pos,
    const at::Tensor *tx_pol,
    const at::Tensor *rx_pos,
    const at::Tensor *active,
    const at::Tensor *state_edge_index,
    const at::Tensor *state_edge_pos,
    const at::Tensor *state_edge_dir,
    const at::Tensor *state_edge_t_min,
    const at::Tensor *state_edge_t_max,
    const at::Tensor *state_n0,
    const at::Tensor *state_n1,
    const at::Tensor *state_prim0,
    const at::Tensor *state_prim1,
    const at::Tensor *state_exterior_angle,
    const at::Tensor *state_src,
    const at::Tensor *state_src_power,
    const at::Tensor *material_eta_r,
    const at::Tensor *material_sigma,
    const at::Tensor *material_mu_r,
    const at::Tensor *material_gain,
    const at::Tensor *material_valid,
    int64_t state_limit,
    int64_t capacity,
    double wavelength,
    double isb_taper_width_scale,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_diffraction_accumulation_forward(
    int64_t scene_handle,
    const at::Tensor *active,
    const at::Tensor *state_edge_index,
    const at::Tensor *state_edge_pos,
    const at::Tensor *state_edge_dir,
    const at::Tensor *state_edge_t_min,
    const at::Tensor *state_edge_t_max,
    const at::Tensor *state_n0,
    const at::Tensor *state_n1,
    const at::Tensor *state_prim0,
    const at::Tensor *state_prim1,
    const at::Tensor *state_exterior_angle,
    const at::Tensor *state_src,
    const at::Tensor *state_src_power,
    const at::Tensor *state_wi,
    const at::Tensor *state_d0,
    const at::Tensor *material_eta_r,
    const at::Tensor *material_sigma,
    const at::Tensor *material_mu_r,
    const at::Tensor *material_gain,
    const at::Tensor *material_valid,
    int64_t state_limit,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    int64_t max_order,
    int64_t recursive_state_limit,
    const at::Tensor *recursive_active,
    const at::Tensor *recursive_state_edge_index,
    const at::Tensor *recursive_state_edge_pos,
    const at::Tensor *recursive_state_edge_dir,
    const at::Tensor *recursive_state_edge_t_min,
    const at::Tensor *recursive_state_edge_t_max,
    const at::Tensor *recursive_state_n0,
    const at::Tensor *recursive_state_n1,
    const at::Tensor *recursive_state_prim0,
    const at::Tensor *recursive_state_prim1,
    const at::Tensor *recursive_state_exterior_angle,
    int64_t export_tape,
    const at::Tensor *sample_state_index,
    const at::Tensor *sample_edge_weight,
    at::Tensor *outputs,
    int64_t output_capacity);

extern "C" int64_t rayd_torch_native_diffraction_coherent_accumulation_forward(
    int64_t scene_handle,
    const at::Tensor *active,
    const at::Tensor *state_edge_index,
    const at::Tensor *state_edge_pos,
    const at::Tensor *state_edge_dir,
    const at::Tensor *state_edge_t_min,
    const at::Tensor *state_edge_t_max,
    const at::Tensor *state_n0,
    const at::Tensor *state_n1,
    const at::Tensor *state_prim0,
    const at::Tensor *state_prim1,
    const at::Tensor *state_exterior_angle,
    const at::Tensor *state_src,
    const at::Tensor *state_src_power,
    const at::Tensor *state_wi,
    const at::Tensor *state_d0,
    const at::Tensor *material_eta_r,
    const at::Tensor *material_sigma,
    const at::Tensor *material_mu_r,
    const at::Tensor *material_gain,
    const at::Tensor *material_valid,
    int64_t state_limit,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    bool select_diffraction_point,
    bool prefilter_visibility,
    at::Tensor *outputs,
    int64_t output_capacity);
