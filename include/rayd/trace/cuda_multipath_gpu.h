#pragma once

#include <cstdint>

#include <rayd/multipath/diffraction_accumulation_params.h>
#include <rayd/multipath/diffraction_paths_params.h>
#include <rayd/multipath/reflection_accumulation_params.h>
#include <rayd/shared/optix/reflection_epc_params.h>
#include <rayd/shared/optix/reflection_trace_params.h>
#include <rayd/shared/optix/segment_visibility_params.h>

// Host/device seam for the CUDA fused multipath executor (P4 Stage D). The
// launchers below run the migrated, traverser-templated multipath algorithm
// bodies (shared/multipath/*_algo.h) with Traverser = CudaBvhTraverser over the
// scene-level triangle BVH, one thread per lane (the lane index is the former
// optixGetLaunchIndex). They are the pure-CUDA counterpart of the OptiX
// pipeline launches in scene_multipath.cpp; each owns its own non-blocking
// stream and traversal-stack scratch and synchronizes before returning, exactly
// like the P3 triangle_bvh.cu query launchers.
//
// This header is host-safe (the params structs are POD with std::uint64_t
// handles) so both cuda_trace_backend.cpp and cuda_multipath.cu include it.

namespace rayd {

/// Read-only raw pointers for one built scene triangle BVH, gathered by
/// CudaTraceBackend from its persistent Dr.Jit buffers. Mirrors the pieces
/// shared::bvh::CudaBvhView needs (triangle SoA, node-bounds SoA, compacted
/// topology, per-primitive id map).
struct CudaMultipathBvh {
    const float *p0_x;
    const float *p0_y;
    const float *p0_z;
    const float *e1_x;
    const float *e1_y;
    const float *e1_z;
    const float *e2_x;
    const float *e2_y;
    const float *e2_z;
    const float *node_min_x;
    const float *node_min_y;
    const float *node_min_z;
    const float *node_max_x;
    const float *node_max_y;
    const float *node_max_z;
    const int *left_child;
    const int *right_child;
    const int *leaf_primitives;
    const int *shape_id;
    const int *local_prim_id;
    int primitive_count;
    int node_count;
    int leaf_primitive_count;
};

/// Reflection-path trace (scene.trace_reflections symbolic=True native path).
void launch_reflection_trace_cuda(const shared::optix::ReflectionTraceParams &params,
                                  const CudaMultipathBvh &bvh,
                                  int lane_count);

/// Segment visibility family (scene.visible / visible_pair / visible_edge /
/// visible_chain). `variant` selects the raygen body.
enum class CudaSegmentVisibilityVariant : int {
    Single = 0,
    Pair = 1,
    AxialEdge = 2,
    Chain = 3
};

void launch_segment_visibility_cuda(const shared::optix::SegmentVisibilityParams &params,
                                    const CudaMultipathBvh &bvh,
                                    CudaSegmentVisibilityVariant variant,
                                    int lane_count);

/// Reflection EPC discovery (scene.trace_refl_epc / trace_refl_epc_field).
/// `direct_only` / `primary_visibility_only` mirror the two template axes of the
/// OptiX raygen (run_reflection_epc_raygen<Policy, DirectOnly, PrimaryOnly>).
void launch_reflection_epc_cuda(const shared::optix::ReflEpcParams &params,
                                const CudaMultipathBvh &bvh,
                                bool direct_only,
                                bool primary_visibility_only,
                                int lane_count);

/// Reflection accumulation (scene.accumulate_reflections).
void launch_reflection_accumulation_cuda(const AccumParams &params,
                                         const CudaMultipathBvh &bvh,
                                         int lane_count);

/// First-order diffraction compact path export (scene.trace_dfr_paths). Runs the
/// single-scene two-phase export: source-visibility prepass then target export.
void launch_dfr_paths_cuda(const DfrPathParams &params,
                           const CudaMultipathBvh &bvh,
                           int lane_count);

/// First-order diffraction accumulation (scene.accum_dfr_direct). Runs the
/// single-scene staged path: source-visibility prepass, then the no-suffix
/// target and/or the suffix-first-visibility + suffix-target phases, all ordered
/// on one stream (mirrors the split_mode==0 OptiX dispatch). `params` is staged
/// into the file-local __constant__ before each phase.
void launch_dfr_accum_direct_cuda(const DfrAccumParams &params,
                                  const CudaMultipathBvh &bvh,
                                  bool has_non_suffix_strategy,
                                  bool has_suffix_strategy,
                                  int lane_count);

/// Coherent order-1 diffraction accumulation (scene.accum_dfr_coherent_direct,
/// both the full-UTD and the simple-state overloads). Single primary-only launch.
void launch_dfr_accum_coherent_cuda(const DfrAccumParams &params,
                                    const CudaMultipathBvh &bvh,
                                    int lane_count);

/// Chain (order 2/3) diffraction accumulation (scene.accum_dfr). Single
/// primary-only launch.
void launch_dfr_accum_chain_cuda(const DfrAccumParams &params,
                                 const CudaMultipathBvh &bvh,
                                 int lane_count);

/// Combined 5-bool order-1 accumulation body. The single-scene CUDA backend
/// always takes the staged path above; this exists so every OptiX raygen variant
/// has a CUDA kernel and is wired defensively for a future split-scene CUDA arm.
void launch_dfr_accum_combined_cuda(const DfrAccumParams &params,
                                    const CudaMultipathBvh &bvh,
                                    bool has_non_suffix_strategy,
                                    bool has_suffix_strategy,
                                    int lane_count);

} // namespace rayd
