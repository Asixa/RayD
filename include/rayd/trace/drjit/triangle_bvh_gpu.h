#pragma once

// Host-callable CUDA orchestration entry points for the pure-CUDA triangle BVH
// backend. These mirror the edge BVH's edge_bvh.h free functions: they receive
// evaluated Dr.Jit device pointers, create their own non-blocking CUDA streams
// and RAII scratch, drive the shared BVH build/refit/traversal kernels, record
// native launch-audit hooks, and synchronize before returning. They allocate no
// persistent state and never touch Dr.Jit.

namespace rayd {

/// Read-only SoA world-space triangle geometry pointers (edge-vector form).
struct TriBvhTrianglePtrs {
    const float *p0_x;
    const float *p0_y;
    const float *p0_z;
    const float *e1_x;
    const float *e1_y;
    const float *e1_z;
    const float *e2_x;
    const float *e2_y;
    const float *e2_z;
};

/// Mutable per-node/per-primitive AABB SoA pointers (build/refit outputs).
struct TriBvhBoundsPtrs {
    float *min_x;
    float *min_y;
    float *min_z;
    float *max_x;
    float *max_y;
    float *max_z;
};

/// Read-only per-node AABB SoA pointers (query inputs).
struct TriBvhConstBoundsPtrs {
    const float *min_x;
    const float *min_y;
    const float *min_z;
    const float *max_x;
    const float *max_y;
    const float *max_z;
};

/// Read-only ray batch pointers. `t_max` is already remapped; `active` is one
/// int per ray (null means all active).
struct TriBvhRayPtrs {
    const float *origin_x;
    const float *origin_y;
    const float *origin_z;
    const float *dir_x;
    const float *dir_y;
    const float *dir_z;
    const float *t_max;
    const int *active;
};

/// Build a scene-level triangle LBVH (Morton/radix/finalize) into the caller's
/// raw (2N-1)-node topology and bounds buffers. Pure LBVH: no treelet pass.
void build_triangle_bvh_gpu(int primitive_count,
                            TriBvhTrianglePtrs triangles,
                            TriBvhBoundsPtrs primitive_bounds,
                            TriBvhBoundsPtrs node_bounds,
                            int *left_child,
                            int *right_child,
                            int *leaf_primitive,
                            int *is_leaf,
                            int *primitive_leaf_node);

/// Refit the compacted BVH node bounds in place after the triangles moved
/// (topology unchanged): recompute leaf-node bounds, then refit internal nodes
/// level by level in ascending height order.
void refit_triangle_bvh_gpu(int node_count,
                            TriBvhTrianglePtrs triangles,
                            const int *left_child,
                            const int *right_child,
                            const int *leaf_primitives,
                            const int *leaf_nodes,
                            int leaf_node_count,
                            const int *level_nodes,
                            const int *level_offsets,
                            int level_count,
                            TriBvhBoundsPtrs node_bounds);

/// Closest-hit query over the compacted BVH.
void query_triangle_closest_hit_gpu(int ray_count,
                                    int primitive_count,
                                    int node_count,
                                    int leaf_primitive_count,
                                    float t_min,
                                    TriBvhTrianglePtrs triangles,
                                    TriBvhConstBoundsPtrs node_bounds,
                                    const int *left_child,
                                    const int *right_child,
                                    const int *leaf_primitives,
                                    TriBvhRayPtrs rays,
                                    const int *shape_id,
                                    const int *local_prim_id,
                                    float *out_t,
                                    float *out_bary_u,
                                    float *out_bary_v,
                                    int *out_shape_id,
                                    int *out_local_prim_id,
                                    int *stack_nodes,
                                    int *overflow);

/// Any-hit occlusion query over the compacted BVH.
void query_triangle_occluded_gpu(int ray_count,
                                 int primitive_count,
                                 int node_count,
                                 int leaf_primitive_count,
                                 float t_min,
                                 TriBvhTrianglePtrs triangles,
                                 TriBvhConstBoundsPtrs node_bounds,
                                 const int *left_child,
                                 const int *right_child,
                                 const int *leaf_primitives,
                                 TriBvhRayPtrs rays,
                                 int *out_hit,
                                 int *stack_nodes,
                                 int *overflow);

/// Closest-blocker query honoring a per-ray ignore list.
void query_triangle_first_blocker_gpu(int ray_count,
                                      int primitive_count,
                                      int node_count,
                                      int leaf_primitive_count,
                                      float t_min,
                                      TriBvhTrianglePtrs triangles,
                                      TriBvhConstBoundsPtrs node_bounds,
                                      const int *left_child,
                                      const int *right_child,
                                      const int *leaf_primitives,
                                      TriBvhRayPtrs rays,
                                      const int *ignore_prim_ids,
                                      int ignore_stride,
                                      int *out_global_prim_id,
                                      int *stack_nodes,
                                      int *overflow);

} // namespace rayd
