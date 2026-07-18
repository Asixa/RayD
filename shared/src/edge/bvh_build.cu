#include <rayd/shared/edge/bvh_build.h>

#include <rayd/shared/bvh/build.h>
#include <rayd/shared/bvh/refit.h>

namespace rayd::shared::edge {
namespace {

constexpr int kBlockSize = 256;

__host__ __device__ inline BvhFloat3 min3(const BvhFloat3 &a, const BvhFloat3 &b) {
    return { fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z) };
}

__host__ __device__ inline BvhFloat3 max3(const BvhFloat3 &a, const BvhFloat3 &b) {
    return { fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z) };
}

__device__ inline void store_bounds(int index,
                                    const BvhBounds3 &value,
                                    MutableAabbSoAView bounds) {
    bounds.min_x[index] = value.min.x;
    bounds.min_y[index] = value.min.y;
    bounds.min_z[index] = value.min.z;
    bounds.max_x[index] = value.max.x;
    bounds.max_y[index] = value.max.y;
    bounds.max_z[index] = value.max.z;
}

__global__ void compute_primitive_bounds_kernel(PrimitiveBoundsParams params) {
    const int primitive = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (primitive >= static_cast<int>(params.edges.count)) return;
    const BvhFloat3 p0{ params.edges.p0_x[primitive],
                        params.edges.p0_y[primitive],
                        params.edges.p0_z[primitive] };
    const BvhFloat3 p1{ p0.x + params.edges.direction_x[primitive],
                        p0.y + params.edges.direction_y[primitive],
                        p0.z + params.edges.direction_z[primitive] };
    const BvhBounds3 bounds{ min3(p0, p1), max3(p0, p1) };
    store_bounds(primitive, bounds, params.primitive_bounds);
    params.packed_bounds[primitive] = bounds;
}

} // namespace

void launch_compute_primitive_bounds_async(const PrimitiveBoundsParams &params) {
    const int count = static_cast<int>(params.edges.count);
    if (count == 0) return;
    compute_primitive_bounds_kernel<<<(count + kBlockSize - 1) / kBlockSize,
                                      kBlockSize,
                                      0,
                                      params.stream>>>(params);
}

// Thin edge adapters that forward to the shared primitive-agnostic launchers.
// The edge parameter names alias the shared bvh structs (see bvh_build.h), so
// these calls stay byte-for-byte identical to the shared launch sequence.
void launch_init_sequence_async(const SequenceInitParams &params) {
    bvh::launch_init_sequence_async(params);
}

void launch_compute_morton_codes_async(const MortonCodeParams &params) {
    bvh::launch_compute_morton_codes_async(params);
}

void launch_build_radix_tree_async(const RadixTreeParams &params) {
    bvh::launch_build_radix_tree_async(params);
}

void launch_finalize_leaves_and_bounds_async(const LeafBoundsFinalizeParams &params) {
    bvh::launch_finalize_leaves_and_bounds_async(params);
}

void launch_initialize_leaf_costs_async(const LeafCostParams &params) {
    bvh::launch_initialize_leaf_costs_async(params);
}

void launch_initialize_internal_costs_async(const InternalCostParams &params) {
    bvh::launch_initialize_internal_costs_async(params);
}

void launch_optimize_selected_treelets_async(const TreeletOptimizeParams &params) {
    bvh::launch_optimize_selected_treelets_async(params);
}

void launch_mark_dirty_ancestors_async(const DirtyAncestorMarkParams &params) {
    bvh::launch_mark_dirty_ancestors_async(params);
}

void launch_compact_dirty_level_async(const DirtyLevelCompactParams &params) {
    bvh::launch_compact_dirty_level_async(params);
}

void launch_refit_selected_internal_nodes_async(const InternalNodeRefitParams &params) {
    bvh::launch_refit_selected_internal_nodes_async(params);
}

} // namespace rayd::shared::edge
