#include <src/scene/cuda_multipath_gpu_jit.h>

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <rayd/jit/native_launch_audit.h>

#include <rayd/detail/bvh/cuda_bvh_traverser.h>
#include <rayd/detail/bvh/topology.h>
#include <rayd/detail/bvh/triangle_query.h>
#include <rayd/detail/field_math.h>
#include <rayd/detail/vec3.h>
#include <src/diffraction/accumulation_params_jit.h>
#include <src/diffraction/paths_params_jit.h>
#include <rayd/detail/diffraction/accumulation_algo.h>
#include <rayd/detail/diffraction/paths_algo.h>
#include <rayd/detail/reflection/accumulation_algo.h>
#include <rayd/detail/reflection/epc_algo.h>
#include <rayd/detail/reflection/trace_algo.h>
#include <rayd/detail/visibility/segment_algo.h>
#include <rayd/detail/rt/traverser.h>

// CUDA fused multipath executor (P4 Stage D). Each launcher runs the migrated,
// traverser-templated multipath algorithm body (concept-owned shared/*/*_algo.h) with
// Traverser = shared::bvh::CudaBvhTraverser over the scene-level triangle BVH,
// one thread per lane (the lane index is the former optixGetLaunchIndex). This is
// the pure-CUDA counterpart of the OptiX pipeline launches in scene_multipath.cpp.
//
// The CUDA backend has a single scene-level BVH (no static/dynamic split and no
// IAS), so every kernel forces the single-scene traversal path: split_mode is 0,
// and the "secondary" traverser is a copy of the primary that the algorithm never
// consults. The traversal-stack scratch and the non-blocking stream are owned
// here and synchronized before returning, exactly like the P3 triangle_bvh.cu
// query launchers.

namespace rayd {

namespace {

namespace bvh = ::rayd::shared::bvh;
namespace rt = ::rayd::shared::rt;
namespace math = ::rayd::shared::math;
namespace multipath = ::rayd::shared::multipath;

constexpr int kTraversalStackDepth = bvh::kBvhTraversalStackDepth;
constexpr int kBlockSize = 256;

int block_count(int count) { return (count + kBlockSize - 1) / kBlockSize; }

[[noreturn]] void throw_runtime_error_local(const std::string &message) {
    throw std::runtime_error(message);
}

void check_cuda_call(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        throw_runtime_error_local(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

void check_cuda_last_error(const char *message) { check_cuda_call(cudaGetLastError(), message); }

/// The CUDA device that is current on this thread right now.
int current_cuda_device() {
    int device = -1;
    check_cuda_call(cudaGetDevice(&device), "cuda_multipath: failed to query the current CUDA device");
    return device;
}

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;
    explicit CudaBuffer(size_t count) { allocate(count); }
    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    void allocate(size_t count) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        device_ = -1;
        if (count == 0) {
            return;
        }
        // Device memory is only valid on the device it was allocated on. Record
        // that device here and re-check it in get(), so scratch allocated under
        // one current device can never be handed to a launch issued under
        // another.
        const int device = current_cuda_device();
        check_cuda_call(cudaMalloc(reinterpret_cast<void **>(&ptr_), sizeof(T) * count),
                        "cuda_multipath: CudaBuffer allocation failed");
        device_ = device;
    }

    T *get() {
        if (ptr_ != nullptr) {
            const int device = current_cuda_device();
            if (device != device_) {
                throw_runtime_error_local(
                    "cuda_multipath: CudaBuffer was allocated on CUDA device " +
                    std::to_string(device_) + " but device " + std::to_string(device) +
                    " is current.");
            }
        }
        return ptr_;
    }

    int device() const { return device_; }

private:
    T *ptr_ = nullptr;
    int device_ = -1;
};

class CudaStreamHandle {
public:
    CudaStreamHandle() {
        check_cuda_call(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                        "cuda_multipath: failed to create CUDA stream");
    }
    ~CudaStreamHandle() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    CudaStreamHandle(const CudaStreamHandle &) = delete;
    CudaStreamHandle &operator=(const CudaStreamHandle &) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

__device__ __forceinline__ bvh::CudaBvhView make_view(const CudaMultipathBvh &b) {
    bvh::CudaBvhView view;
    view.triangles = {b.p0_x, b.p0_y, b.p0_z, b.e1_x, b.e1_y, b.e1_z,
                      b.e2_x, b.e2_y, b.e2_z, static_cast<size_t>(b.primitive_count)};
    view.node_bounds = {b.node_min_x, b.node_min_y, b.node_min_z,
                        b.node_max_x, b.node_max_y, b.node_max_z,
                        static_cast<size_t>(b.node_count)};
    view.topology.left_child = b.left_child;
    view.topology.right_child = b.right_child;
    view.topology.leaf_primitives = b.leaf_primitives;
    view.topology.node_active_count = nullptr;
    view.topology.node_count = static_cast<size_t>(b.node_count);
    view.topology.primitive_count = static_cast<size_t>(b.primitive_count);
    view.topology.leaf_primitive_count = static_cast<size_t>(b.leaf_primitive_count);
    view.prim_map = {b.shape_id, b.local_prim_id, static_cast<size_t>(b.primitive_count)};
    return view;
}

__device__ __forceinline__ bvh::TriangleTraversalScratchView make_scratch(
    int *stack_nodes, int *overflow, int lane_count) {
    bvh::TriangleTraversalScratchView scratch;
    scratch.node_indices = stack_nodes;
    scratch.overflow = overflow;
    scratch.query_stride = static_cast<size_t>(lane_count);
    scratch.stack_depth = static_cast<size_t>(kTraversalStackDepth);
    scratch.capacity = static_cast<size_t>(lane_count) * static_cast<size_t>(kTraversalStackDepth);
    scratch.overflow_capacity = static_cast<size_t>(lane_count);
    return scratch;
}

/// One CudaBvhTraverser bound to this thread's lane. `overflow` is written per
/// lane by the traversal core; the build-time guard (max BVH height + 1 <= stack
/// depth) makes overflow structurally impossible, so it is never repaired here.
__device__ __forceinline__ bvh::CudaBvhTraverser make_traverser(
    const bvh::CudaBvhView &view, const bvh::TriangleTraversalScratchView &scratch,
    unsigned int lane) {
    bvh::CudaBvhTraverser traverser;
    traverser.view = view;
    traverser.scratch = scratch;
    traverser.lane = static_cast<size_t>(lane);
    return traverser;
}

// ---------------------------------------------------------------------------
// Reflection trace
// ---------------------------------------------------------------------------

/// Layout policy for the CUDA reflection-trace path. Bit-for-bit the DrJit
/// reflection-trace layout (shared/reflection/trace_optix_device.cuh's
/// DrJitReflectionTracePolicy); duplicated here so the object-compiled CUDA unit
/// needs no OptiX-including device shim, keeping the OptiX path byte-unchanged.
struct CudaReflectionTracePolicy {
    static constexpr bool allow_aos_inputs = false;
    static constexpr bool allow_packed_triangles = false;
    static constexpr bool honor_output_layout = false;
    static constexpr bool clear_empty_slots = false;
    static constexpr bool nullable_ray_tmax = false;
    static constexpr bool allow_extended_outputs = false;
};

__global__ void reflection_trace_kernel(shared::optix::ReflectionTraceParams params,
                                        CudaMultipathBvh bvh, int *stack_nodes, int *overflow,
                                        int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count)) {
        return;
    }
    const bvh::CudaBvhView view = make_view(bvh);
    const bvh::TriangleTraversalScratchView scratch = make_scratch(stack_nodes, overflow, lane_count);
    const bvh::CudaBvhTraverser traverser = make_traverser(view, scratch, lane);
    using Config = rt::TraceConfig<CudaReflectionTracePolicy, bvh::CudaBvhTraverser>;
    multipath::reflection_trace_algo<Config>(params, lane, traverser, traverser);
}

// ---------------------------------------------------------------------------
// Segment visibility
// ---------------------------------------------------------------------------

/// Layout policy for the CUDA segment-visibility path. Bit-for-bit the DrJit
/// SegmentVisibilityDevicePolicy<false, false> (segment_visibility.cu).
struct CudaSegmentVisibilityPolicy {
    static constexpr bool disable_anyhit_without_ignore = false;
    static constexpr bool write_output_t = false;
};

__global__ void segment_visibility_kernel(shared::optix::SegmentVisibilityParams params,
                                          CudaMultipathBvh bvh, int variant, int *stack_nodes,
                                          int *overflow, int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count)) {
        return;
    }
    const bvh::CudaBvhView view = make_view(bvh);
    const bvh::TriangleTraversalScratchView scratch = make_scratch(stack_nodes, overflow, lane_count);
    const bvh::CudaBvhTraverser traverser = make_traverser(view, scratch, lane);
    using Config = rt::TraceConfig<CudaSegmentVisibilityPolicy, bvh::CudaBvhTraverser>;
    switch (static_cast<CudaSegmentVisibilityVariant>(variant)) {
    case CudaSegmentVisibilityVariant::Single:
        multipath::segment_visibility_algo<Config>(params, lane, traverser);
        break;
    case CudaSegmentVisibilityVariant::Pair:
        multipath::segment_pair_visibility_algo<Config>(params, lane, traverser);
        break;
    case CudaSegmentVisibilityVariant::AxialEdge:
        multipath::axial_edge_visibility_algo<Config>(params, lane, traverser);
        break;
    case CudaSegmentVisibilityVariant::Chain:
        multipath::segment_chain_visibility_algo<Config>(params, lane, traverser);
        break;
    }
}

// ---------------------------------------------------------------------------
// Reflection accumulation
// ---------------------------------------------------------------------------

/// Grid-commit policy for the CUDA reflection-accumulation path. Bit-for-bit the
/// DrJit ReflectionAccumulationPolicy (reflection_accumulation.cu): include depths
/// greater than zero and atomically scatter field / power into the grid cell.
struct CudaReflectionAccumulationPolicy {
    static __forceinline__ __device__ bool include_depth(const AccumParams &, int depth) {
        return depth > 0;
    }

    static __forceinline__ __device__ void commit(const AccumParams &params, unsigned int, int,
                                                  int cell, shared::field::Complex3 field,
                                                  float power) {
        atomicAdd(params.out_field_x_re + cell, field.x.r);
        atomicAdd(params.out_field_x_im + cell, field.x.i);
        atomicAdd(params.out_field_y_re + cell, field.y.r);
        atomicAdd(params.out_field_y_im + cell, field.y.i);
        atomicAdd(params.out_field_z_re + cell, field.z.r);
        atomicAdd(params.out_field_z_im + cell, field.z.i);
        atomicAdd(params.out_reflection_power + cell, power);
        atomicAdd(params.out_reflection_count, 1);
    }
};

__global__ void reflection_accumulation_kernel(AccumParams params, CudaMultipathBvh bvh,
                                               int *stack_nodes, int *overflow, int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count)) {
        return;
    }
    const bvh::CudaBvhView view = make_view(bvh);
    const bvh::TriangleTraversalScratchView scratch = make_scratch(stack_nodes, overflow, lane_count);
    const bvh::CudaBvhTraverser traverser = make_traverser(view, scratch, lane);
    multipath::reflection_accumulation_algo<AccumParams, CudaReflectionAccumulationPolicy,
                                            bvh::CudaBvhTraverser>(params, lane, traverser,
                                                                   traverser);
}

// ---------------------------------------------------------------------------
// Reflection EPC (discovery)
// ---------------------------------------------------------------------------

using shared::optix::ReflEpcParams;
using shared::optix::ReflEpcVisibilityIgnoreSurfaceGroup;
namespace epc_detail = multipath::reflection_epc_algo_detail;

/// Layout policy for the CUDA EPC path (the DisableAnyHitWithoutIgnore flag is an
/// OptiX ray-flag choice with no CUDA analogue; kept for Config parity).
struct CudaEpcPolicy {
    static constexpr bool DisableAnyHitWithoutIgnore = false;
};

/// True when a candidate blocker is on this segment's ignore set. Primitive mode
/// (default) compares global prim ids directly; surface-group mode resolves the
/// candidate prim to its group first (parity with the OptiX anyhit filter).
__device__ __forceinline__ bool epc_prim_ignored(const ReflEpcParams &params, int prim, int ig0,
                                                 int ig1, int ig2) {
    int cand = prim;
    if (params.visibility_ignore_mode == ReflEpcVisibilityIgnoreSurfaceGroup) {
        cand = epc_detail::surface_group_for_prim(params, prim);
    }
    return (ig0 >= 0 && cand == ig0) || (ig1 >= 0 && cand == ig1) || (ig2 >= 0 && cand == ig2);
}

/// Closest non-ignored blocker over the scene BVH, group-aware. Mirrors
/// bvh::traverse_first_blocker's structure (near/far ordering, (t, prim)
/// tie-break) with the EPC group-aware ignore check. Returns the winning global
/// primitive id or -1.
__device__ __forceinline__ int epc_first_blocker(const bvh::CudaBvhView &view,
                                                 const bvh::TriangleTraversalScratchView &scratch,
                                                 std::size_t lane, const ReflEpcParams &params,
                                                 math::Vec3f origin, math::Vec3f direction,
                                                 float tmin, float tmax, int ig0, int ig1,
                                                 int ig2) {
    const float ox = origin.x, oy = origin.y, oz = origin.z;
    const float dx = direction.x, dy = direction.y, dz = direction.z;
    const float inv_dx = bvh::safe_rcp(dx), inv_dy = bvh::safe_rcp(dy), inv_dz = bvh::safe_rcp(dz);
    float best_t = tmax;
    int best_prim = -1;
    int sp = 0;
    int node = 0;
    for (;;) {
        while (node >= 0 && !bvh::is_leaf_node(view.topology, node)) {
            const int left = view.topology.left_child[node];
            const int right = view.topology.right_child[node];
            float t_left = 0.0f;
            float t_right = 0.0f;
            const bool hit_left = bvh::intersect_node_bounds(view.node_bounds, left, ox, oy, oz,
                                                             inv_dx, inv_dy, inv_dz, tmin, best_t,
                                                             t_left);
            const bool hit_right = bvh::intersect_node_bounds(view.node_bounds, right, ox, oy, oz,
                                                              inv_dx, inv_dy, inv_dz, tmin, best_t,
                                                              t_right);
            if (hit_left && hit_right) {
                int near_child = left;
                int far_child = right;
                if (!bvh::near_child_is_left(t_left, t_right, left, right)) {
                    near_child = right;
                    far_child = left;
                }
                if (!bvh::stack_push(scratch, lane, static_cast<std::size_t>(sp), far_child)) {
                    break;  // build-time guard makes overflow impossible
                }
                ++sp;
                node = near_child;
            } else if (hit_left) {
                node = left;
            } else if (hit_right) {
                node = right;
            } else {
                node = -1;
            }
        }
        if (node >= 0) {
            const int leaf_begin = -view.topology.left_child[node] - 1;
            const int leaf_count = view.topology.right_child[node];
            for (int slot = 0; slot < leaf_count; ++slot) {
                const int prim = view.topology.leaf_primitives[leaf_begin + slot];
                if (epc_prim_ignored(params, prim, ig0, ig1, ig2)) {
                    continue;
                }
                const bvh::TriangleVertices v = bvh::load_triangle(view.triangles, prim);
                const bvh::WatertightTriangleHit hit = bvh::intersect_triangle_watertight(
                    ox, oy, oz, dx, dy, dz, v.ax, v.ay, v.az, v.bx, v.by, v.bz, v.cx, v.cy, v.cz,
                    tmin, best_t);
                if (hit.hit && (hit.t < best_t || (hit.t == best_t && prim < best_prim))) {
                    best_t = hit.t;
                    best_prim = prim;
                }
            }
        }
        if (sp == 0) {
            break;
        }
        --sp;
        node = bvh::stack_load(scratch, lane, static_cast<std::size_t>(sp));
    }
    return best_prim;
}

/// EPC traverser: trace_closest is the reflector scene cast (closest hit);
/// trace_first_blocker is the group-aware segment-visibility cast. Satisfies
/// rt::is_traverser.
struct CudaEpcTraverser {
    bvh::CudaBvhView view;
    bvh::TriangleTraversalScratchView scratch;
    std::size_t lane;
    const ReflEpcParams *params;

    __device__ __forceinline__ rt::TriangleHit trace_closest(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        const bvh::CudaBvhTraverser t{view, scratch, lane};
        return t.trace_closest(origin, direction, tmin, tmax);
    }

    __device__ __forceinline__ rt::TriangleHit trace_first_blocker(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t *ignore, int ignore_count) const {
        const int ig0 = ignore_count > 0 ? ignore[0] : -1;
        const int ig1 = ignore_count > 1 ? ignore[1] : -1;
        const int ig2 = ignore_count > 2 ? ignore[2] : -1;
        const int prim = epc_first_blocker(view, scratch, lane, *params, origin, direction, tmin,
                                           tmax, ig0, ig1, ig2);
        rt::TriangleHit hit;
        hit.t = tmax;
        hit.bary_u = 0.0f;
        hit.bary_v = 0.0f;
        if (prim >= 0) {
            hit.prim = view.prim_map.local_prim_id[prim];
            hit.instance = view.prim_map.shape_id[prim];
            hit.hit = 1u;
        } else {
            hit.prim = -1;
            hit.instance = -1;
            hit.hit = 0u;
        }
        return hit;
    }

    __device__ __forceinline__ bool trace_occluded_ignore(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax,
        const std::int32_t *ignore, int ignore_count) const {
        return trace_first_blocker(origin, direction, tmin, tmax, ignore, ignore_count).hit != 0u;
    }

    __device__ __forceinline__ bool trace_occluded(
        math::Vec3f origin, math::Vec3f direction, float tmin, float tmax) const {
        return trace_occluded_ignore(origin, direction, tmin, tmax, nullptr, 0);
    }
};

static_assert(rt::is_traverser_v<CudaEpcTraverser>,
              "CudaEpcTraverser must satisfy the rt::Traverser concept.");

template <bool DirectOnly, bool PrimaryOnly>
__global__ void reflection_epc_kernel(ReflEpcParams params, CudaMultipathBvh bvh, int *stack_nodes,
                                     int *overflow, int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count)) {
        return;
    }
    const bvh::CudaBvhView view = make_view(bvh);
    const bvh::TriangleTraversalScratchView scratch = make_scratch(stack_nodes, overflow, lane_count);
    const CudaEpcTraverser traverser{view, scratch, static_cast<std::size_t>(lane), &params};
    using Config = rt::TraceConfig<CudaEpcPolicy, CudaEpcTraverser>;
    multipath::run_reflection_epc_algo<Config, DirectOnly, PrimaryOnly>(params, lane, traverser,
                                                                        traverser);
}

// ---------------------------------------------------------------------------
// Diffraction paths (first-order compact export)
// ---------------------------------------------------------------------------

__global__ void dfr_paths_source_visibility_kernel(DfrPathParams params, CudaMultipathBvh bvh,
                                                  int *stack_nodes, int *overflow, int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count)) {
        return;
    }
    const bvh::CudaBvhView view = make_view(bvh);
    const bvh::TriangleTraversalScratchView scratch = make_scratch(stack_nodes, overflow, lane_count);
    const bvh::CudaBvhTraverser traverser = make_traverser(view, scratch, lane);
    multipath::trace_paths_source_visibility_algo<DfrPathParams, bvh::CudaBvhTraverser>(
        params, lane, traverser, traverser);
}

__global__ void dfr_paths_target_export_kernel(DfrPathParams params, CudaMultipathBvh bvh,
                                              int *stack_nodes, int *overflow, int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count)) {
        return;
    }
    const bvh::CudaBvhView view = make_view(bvh);
    const bvh::TriangleTraversalScratchView scratch = make_scratch(stack_nodes, overflow, lane_count);
    const bvh::CudaBvhTraverser traverser = make_traverser(view, scratch, lane);
    multipath::trace_paths_target_export_algo<DfrPathParams, bvh::CudaBvhTraverser>(
        params, lane, traverser, traverser);
}

// ---------------------------------------------------------------------------
// Diffraction accumulation (order-1 accumulation, source visibility, no-suffix
// target, suffix-first visibility, suffix target, coherent, chain)
// ---------------------------------------------------------------------------

/// Per-block relay slot holding the address of the current launch's params.
///
/// Diffraction accumulation is the one migrated pipeline whose algorithm body
/// reads its storage through a *static* Policy::params() accessor (every other
/// pipeline takes params by argument). The shared body cannot be changed without
/// disturbing the byte-identical OptiX path, so the accessor has to find the
/// params without being handed them.
///
/// The params therefore travel as an ordinary `__grid_constant__` kernel
/// parameter and each block publishes their address here once at entry. This
/// replaced a file-scope `__constant__` rewritten per launch with
/// `cudaMemcpyToSymbolAsync`: that symbol was one mutable object per device
/// module, so two threads (or two streams) launching accumulation concurrently
/// raced to overwrite each other's params, and the staging copy silently
/// reordered against any launch not on the staging stream. A kernel parameter is
/// per-launch by construction, so neither failure is expressible any more.
///
/// `__shared__` inside a `__device__` function is one allocation per block (not
/// per inline expansion), and kernel parameters live in constant bank 0, so the
/// payload loads still come from constant memory exactly as before -- only the
/// eight-byte address indirection is new.
__device__ __forceinline__ const DfrAccumParams *&dfr_accum_params_slot() {
    __shared__ const DfrAccumParams *slot;
    return slot;
}

/// Publish this launch's params to the block. Must be called by every thread of
/// the block, before any early return, so the barrier is uniform.
__device__ __forceinline__ void publish_dfr_accum_params(const DfrAccumParams &params) {
    if (threadIdx.x == 0) {
        dfr_accum_params_slot() = &params;
    }
    __syncthreads();
}

/// CUDA clone of the DrJit DiffractionAccumulationPolicy
/// (src/diffraction/accumulation_optix_jit.cu). Every method is
/// byte-for-byte the DrJit OptiX policy; only params() differs (it reads the
/// per-block relay slot above instead of the OptiX module __constant__), so the
/// backend extensions (state_count lane mapping, edge-length weighting,
/// atomicAdd commit, return-false staging hooks) behave identically to the
/// OptiX path.
struct CudaDiffractionAccumulationPolicy {
    static __forceinline__ __device__ const DfrAccumParams &params() {
        return *dfr_accum_params_slot();
    }

    static constexpr int kDirect =
        static_cast<int>(shared::optix::DiffractionStrategyBit::Direct);
    static constexpr int kKeller =
        static_cast<int>(shared::optix::DiffractionStrategyBit::Keller);
    static constexpr int kSuffix =
        static_cast<int>(shared::optix::DiffractionStrategyBit::SuffixReflection);

    struct CellGroup {};

    static __forceinline__ __device__ int sample_state_index_for_lane(unsigned int lane) {
        return static_cast<int>(lane % static_cast<unsigned int>(params().state_count));
    }

    static __forceinline__ __device__ bool active_state(int i) {
        return params().active_mask == nullptr || params().active_mask[i] != 0u;
    }

    static __forceinline__ __device__ bool recursive_active_state(int i) {
        return params().recursive_active_mask == nullptr ||
               params().recursive_active_mask[i] != 0u;
    }

    static __forceinline__ __device__ int state_edge_index_at(int i) {
        return params().state_edge_index[i];
    }

    static __forceinline__ __device__ math::Vec3f state_edge_pos_at(int i) {
        return math::make_vec3(params().state_edge_pos_x[i], params().state_edge_pos_y[i],
                               params().state_edge_pos_z[i]);
    }

    static __forceinline__ __device__ math::Vec3f state_edge_dir_at(int i) {
        return math::make_vec3(params().state_edge_dir_x[i], params().state_edge_dir_y[i],
                               params().state_edge_dir_z[i]);
    }

    static __forceinline__ __device__ float state_edge_t_min_at(int i) {
        return params().state_edge_t_min[i];
    }

    static __forceinline__ __device__ float state_edge_t_max_at(int i) {
        return params().state_edge_t_max[i];
    }

    static __forceinline__ __device__ float
    sample_edge_weight_for_lane(int state_idx, unsigned int, int sample_count) {
        const float edge_length =
            fmaxf(state_edge_t_max_at(state_idx) - state_edge_t_min_at(state_idx), 0.f);
        return edge_length / fmaxf(static_cast<float>(sample_count), 1.f);
    }

    static __forceinline__ __device__ int state_prim0_at(int i) {
        return params().state_prim0[i];
    }

    static __forceinline__ __device__ int state_prim1_at(int i) {
        return params().state_prim1[i];
    }

    static __forceinline__ __device__ float state_exterior_angle_at(int i) {
        return params().state_exterior_angle[i];
    }

    static __forceinline__ __device__ float state_src_power_at(int i) {
        return params().state_src_power[i];
    }

    static __forceinline__ __device__ math::Vec3f state_src_at(int i) {
        return math::make_vec3(params().state_src_x[i], params().state_src_y[i],
                               params().state_src_z[i]);
    }

    static __forceinline__ __device__ float3 state_wi_at(int i) {
        return make_float3(params().state_wi_x[i], params().state_wi_y[i], params().state_wi_z[i]);
    }

    static __forceinline__ __device__ int recursive_state_edge_index_at(int i) {
        return params().recursive_state_edge_index[i];
    }

    static __forceinline__ __device__ math::Vec3f recursive_state_edge_pos_at(int i) {
        return math::make_vec3(params().recursive_state_edge_pos_x[i],
                               params().recursive_state_edge_pos_y[i],
                               params().recursive_state_edge_pos_z[i]);
    }

    static __forceinline__ __device__ math::Vec3f recursive_state_edge_dir_at(int i) {
        return math::make_vec3(params().recursive_state_edge_dir_x[i],
                               params().recursive_state_edge_dir_y[i],
                               params().recursive_state_edge_dir_z[i]);
    }

    static __forceinline__ __device__ float recursive_state_edge_t_min_at(int i) {
        return params().recursive_state_edge_t_min[i];
    }

    static __forceinline__ __device__ float recursive_state_edge_t_max_at(int i) {
        return params().recursive_state_edge_t_max[i];
    }

    static __forceinline__ __device__ int recursive_state_prim0_at(int i) {
        return params().recursive_state_prim0[i];
    }

    static __forceinline__ __device__ int recursive_state_prim1_at(int i) {
        return params().recursive_state_prim1[i];
    }

    static __forceinline__ __device__ float recursive_state_exterior_angle_at(int i) {
        return params().recursive_state_exterior_angle[i];
    }

    static __forceinline__ __device__ bool material_valid_at(int prim) {
        return params().material_valid == nullptr || params().material_valid[prim] != 0u;
    }

    static __forceinline__ __device__ float material_gain_at(int prim) {
        return params().material_gain[prim];
    }

    static __forceinline__ __device__ CellGroup cell_group(int) { return {}; }

    static __forceinline__ __device__ void
    atomic_add_same_cell(float *base, int i, float value, CellGroup) {
        atomicAdd(base + i, value);
    }

    static __forceinline__ __device__ void
    atomic_add_same_cell(int *base, int i, int value, CellGroup) {
        atomicAdd(base + i, value);
    }

    static __forceinline__ __device__ void atomic_add_warp(float *base, float value) {
        atomicAdd(base, value);
    }

    static __forceinline__ __device__ void atomic_add_warp(int *base, int value) {
        atomicAdd(base, value);
    }

    static __forceinline__ __device__ bool stage_order1(unsigned int, int, float, float, bool,
                                                        bool) {
        return false;
    }

    static __forceinline__ __device__ bool
    stage_coherent(int, int, bool, float, float, float, float, float, float) {
        return false;
    }
};

using CudaDiffractionAccumulationAlgo =
    multipath::DiffractionAccumulationAlgo<CudaDiffractionAccumulationPolicy, bvh::CudaBvhTraverser>;

/// Build the per-lane algorithm object over the single scene BVH. The secondary
/// traverser is a copy of the primary; split_mode is forced to 0 on the CUDA
/// path so the algorithm never consults it (matches the OptiX single-scene run).
__device__ __forceinline__ CudaDiffractionAccumulationAlgo make_dfr_accum_algo(
    const CudaMultipathBvh &bvh, int *stack_nodes, int *overflow, int lane_count,
    unsigned int lane) {
    const bvh::CudaBvhView view = make_view(bvh);
    const bvh::TriangleTraversalScratchView scratch = make_scratch(stack_nodes, overflow, lane_count);
    const bvh::CudaBvhTraverser traverser = make_traverser(view, scratch, lane);
    return CudaDiffractionAccumulationAlgo{traverser, traverser};
}

/// Enum tag selecting the diffraction-accumulation raygen body for the single,
/// PrimaryOnly kernels (source visibility / no-suffix target / suffix-first
/// visibility / suffix target / coherent / chain). The 5-bool combined
/// accumulation body is a separate template kernel below.
enum class DfrAccumPhase : int {
    SourceVisibility = 0,
    NoSuffixTarget = 1,
    SuffixFirstVisibility = 2,
    SuffixTarget = 3,
    Coherent = 4,
    Chain = 5,
};

/// Combined order-1 accumulation raygen (five template bools preserved). This is
/// the CUDA counterpart of run_diffraction_order1_accumulation_raygen<...>; the
/// scene dispatch only reaches it in the split-scene (split_mode != 0) branch,
/// which never triggers on the single-scene CUDA backend, but it is instantiated
/// for parity so every OptiX raygen variant has a CUDA kernel.
template <bool PrimaryOnly, bool IncludeCoherent, bool IncludeDirect, bool IncludeKeller,
          bool IncludeSuffix>
__global__ void dfr_accum_order1_kernel(__grid_constant__ const DfrAccumParams params,
                                        CudaMultipathBvh bvh, int *stack_nodes, int *overflow,
                                        int lane_count) {
    publish_dfr_accum_params(params);
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count)) {
        return;
    }
    make_dfr_accum_algo(bvh, stack_nodes, overflow, lane_count, lane)
        .run_diffraction_order1_accumulation_algo<PrimaryOnly, IncludeCoherent, IncludeDirect,
                                                  IncludeKeller, IncludeSuffix>(lane);
}

/// Single-body PrimaryOnly raygen kernels (source visibility / no-suffix target /
/// suffix-first visibility / suffix target / coherent / chain), one dispatch
/// switch. PrimaryOnly is always true on the single-scene CUDA path.
__global__ void dfr_accum_phase_kernel(__grid_constant__ const DfrAccumParams params,
                                       CudaMultipathBvh bvh, int phase, int *stack_nodes,
                                       int *overflow, int lane_count) {
    publish_dfr_accum_params(params);
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count)) {
        return;
    }
    CudaDiffractionAccumulationAlgo algo =
        make_dfr_accum_algo(bvh, stack_nodes, overflow, lane_count, lane);
    switch (static_cast<DfrAccumPhase>(phase)) {
    case DfrAccumPhase::SourceVisibility:
        algo.run_diffraction_order1_source_visibility_algo<true>(lane);
        break;
    case DfrAccumPhase::NoSuffixTarget:
        algo.run_diffraction_order1_no_suffix_target_accumulation_algo<true>(lane);
        break;
    case DfrAccumPhase::SuffixFirstVisibility:
        algo.run_diffraction_order1_suffix_first_visibility_algo<true>(lane);
        break;
    case DfrAccumPhase::SuffixTarget:
        algo.run_diffraction_order1_suffix_target_accumulation_algo<true>(lane);
        break;
    case DfrAccumPhase::Coherent:
        algo.run_diffraction_order1_coherent_accumulation_algo<true>(lane);
        break;
    case DfrAccumPhase::Chain:
        algo.run_diffraction_chain_accumulation_algo<true>(lane);
        break;
    }
}

/// Allocate the per-lane traversal-stack scratch on \p stream.
struct ScratchBuffers {
    CudaBuffer<int> stack;
    CudaBuffer<int> overflow;
    ScratchBuffers(int lane_count)
        : stack(static_cast<size_t>(lane_count) * static_cast<size_t>(kTraversalStackDepth)),
          overflow(static_cast<size_t>(lane_count)) {}
};

} // namespace

void launch_reflection_trace_cuda(const shared::optix::ReflectionTraceParams &params,
                                  const CudaMultipathBvh &bvh, int lane_count) {
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        reflection_trace_kernel<<<block_count(lane_count), kBlockSize, 0, stream_handle.get()>>>(
            params, bvh, scratch.stack.get(), scratch.overflow.get(), lane_count);
        audit_cuda_kernel_launch("reflection_trace_cuda_kernel",
                                 static_cast<uint32_t>(block_count(lane_count)), 1, 1, kBlockSize, 1,
                                 1, static_cast<uint64_t>(lane_count));
        check_cuda_last_error("launch_reflection_trace_cuda(): kernel launch failed");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream_handle.get()),
                        "launch_reflection_trace_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_reflection_trace_cuda(): ") + error.what());
    }
}

void launch_segment_visibility_cuda(const shared::optix::SegmentVisibilityParams &params,
                                    const CudaMultipathBvh &bvh,
                                    CudaSegmentVisibilityVariant variant, int lane_count) {
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        segment_visibility_kernel<<<block_count(lane_count), kBlockSize, 0, stream_handle.get()>>>(
            params, bvh, static_cast<int>(variant), scratch.stack.get(), scratch.overflow.get(),
            lane_count);
        audit_cuda_kernel_launch("segment_visibility_cuda_kernel",
                                 static_cast<uint32_t>(block_count(lane_count)), 1, 1, kBlockSize, 1,
                                 1, static_cast<uint64_t>(lane_count));
        check_cuda_last_error("launch_segment_visibility_cuda(): kernel launch failed");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream_handle.get()),
                        "launch_segment_visibility_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_segment_visibility_cuda(): ") + error.what());
    }
}

void launch_reflection_accumulation_cuda(const AccumParams &params, const CudaMultipathBvh &bvh,
                                         int lane_count) {
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        reflection_accumulation_kernel<<<block_count(lane_count), kBlockSize, 0,
                                         stream_handle.get()>>>(
            params, bvh, scratch.stack.get(), scratch.overflow.get(), lane_count);
        audit_cuda_kernel_launch("reflection_accumulation_cuda_kernel",
                                 static_cast<uint32_t>(block_count(lane_count)), 1, 1, kBlockSize, 1,
                                 1, static_cast<uint64_t>(lane_count));
        check_cuda_last_error("launch_reflection_accumulation_cuda(): kernel launch failed");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(stream_handle.get()),
                        "launch_reflection_accumulation_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_reflection_accumulation_cuda(): ") +
                                  error.what());
    }
}

void launch_reflection_epc_cuda(const shared::optix::ReflEpcParams &params,
                                const CudaMultipathBvh &bvh, bool direct_only,
                                bool primary_visibility_only, int lane_count) {
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        const int blocks = block_count(lane_count);
        const cudaStream_t s = stream_handle.get();
        int *stack = scratch.stack.get();
        int *ovf = scratch.overflow.get();
        // Instantiation matrix mirrors the OptiX raygen variants:
        // full (false,false), direct (true,false), direct_primary (true,true).
        if (direct_only && primary_visibility_only) {
            reflection_epc_kernel<true, true><<<blocks, kBlockSize, 0, s>>>(params, bvh, stack, ovf,
                                                                            lane_count);
        } else if (direct_only) {
            reflection_epc_kernel<true, false><<<blocks, kBlockSize, 0, s>>>(params, bvh, stack, ovf,
                                                                             lane_count);
        } else {
            reflection_epc_kernel<false, false><<<blocks, kBlockSize, 0, s>>>(params, bvh, stack,
                                                                              ovf, lane_count);
        }
        audit_cuda_kernel_launch("reflection_epc_cuda_kernel", static_cast<uint32_t>(blocks), 1, 1,
                                 kBlockSize, 1, 1, static_cast<uint64_t>(lane_count));
        check_cuda_last_error("launch_reflection_epc_cuda(): kernel launch failed");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(s),
                        "launch_reflection_epc_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_reflection_epc_cuda(): ") + error.what());
    }
}

void launch_dfr_paths_cuda(const DfrPathParams &params, const CudaMultipathBvh &bvh,
                           int lane_count) {
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        const int blocks = block_count(lane_count);
        const cudaStream_t s = stream_handle.get();
        int *stack = scratch.stack.get();
        int *ovf = scratch.overflow.get();
        // Single-scene two-phase export (mirrors the split_mode==0 OptiX path):
        // source-visibility prepass then target export, ordered on one stream.
        dfr_paths_source_visibility_kernel<<<blocks, kBlockSize, 0, s>>>(params, bvh, stack, ovf,
                                                                         lane_count);
        audit_cuda_kernel_launch("dfr_paths_source_visibility_cuda_kernel",
                                 static_cast<uint32_t>(blocks), 1, 1, kBlockSize, 1, 1,
                                 static_cast<uint64_t>(lane_count));
        check_cuda_last_error("launch_dfr_paths_cuda(): source-visibility launch failed");
        dfr_paths_target_export_kernel<<<blocks, kBlockSize, 0, s>>>(params, bvh, stack, ovf,
                                                                     lane_count);
        audit_cuda_kernel_launch("dfr_paths_target_export_cuda_kernel",
                                 static_cast<uint32_t>(blocks), 1, 1, kBlockSize, 1, 1,
                                 static_cast<uint64_t>(lane_count));
        check_cuda_last_error("launch_dfr_paths_cuda(): target-export launch failed");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(s), "launch_dfr_paths_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_dfr_paths_cuda(): ") + error.what());
    }
}

namespace {

void launch_dfr_accum_phase(const DfrAccumParams &params, DfrAccumPhase phase,
                            const char *audit_name, const CudaMultipathBvh &bvh, int *stack,
                            int *ovf, int lane_count, cudaStream_t stream) {
    const int blocks = block_count(lane_count);
    dfr_accum_phase_kernel<<<blocks, kBlockSize, 0, stream>>>(params, bvh, static_cast<int>(phase),
                                                              stack, ovf, lane_count);
    audit_cuda_kernel_launch(audit_name, static_cast<uint32_t>(blocks), 1, 1, kBlockSize, 1, 1,
                             static_cast<uint64_t>(lane_count));
    check_cuda_last_error(audit_name);
}

} // namespace

void launch_dfr_accum_direct_cuda(const DfrAccumParams &params, const CudaMultipathBvh &bvh,
                                  bool has_non_suffix_strategy, bool has_suffix_strategy,
                                  int lane_count) {
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        const cudaStream_t s = stream_handle.get();
        int *stack = scratch.stack.get();
        int *ovf = scratch.overflow.get();
        // Single-scene staged order (mirrors the split_mode==0 OptiX path): the
        // source-visibility prepass writes temp_visibility, then the target
        // phases read (and the suffix-first phase overwrites) it, all serialized
        // on one stream.
        launch_dfr_accum_phase(params, DfrAccumPhase::SourceVisibility,
                               "dfr_accum_source_visibility_cuda_kernel", bvh, stack, ovf,
                               lane_count, s);
        if (has_non_suffix_strategy) {
            launch_dfr_accum_phase(params, DfrAccumPhase::NoSuffixTarget,
                                   "dfr_accum_no_suffix_target_cuda_kernel", bvh, stack, ovf,
                                   lane_count, s);
        }
        if (has_suffix_strategy) {
            launch_dfr_accum_phase(params, DfrAccumPhase::SuffixFirstVisibility,
                                   "dfr_accum_suffix_first_visibility_cuda_kernel", bvh, stack, ovf,
                                   lane_count, s);
            launch_dfr_accum_phase(params, DfrAccumPhase::SuffixTarget,
                                   "dfr_accum_suffix_target_cuda_kernel", bvh, stack, ovf,
                                   lane_count, s);
        }
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(s),
                        "launch_dfr_accum_direct_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_dfr_accum_direct_cuda(): ") + error.what());
    }
}

void launch_dfr_accum_coherent_cuda(const DfrAccumParams &params, const CudaMultipathBvh &bvh,
                                    int lane_count) {
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        const cudaStream_t s = stream_handle.get();
        launch_dfr_accum_phase(params, DfrAccumPhase::Coherent, "dfr_accum_coherent_cuda_kernel",
                               bvh, scratch.stack.get(), scratch.overflow.get(), lane_count, s);
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(s),
                        "launch_dfr_accum_coherent_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_dfr_accum_coherent_cuda(): ") + error.what());
    }
}

void launch_dfr_accum_chain_cuda(const DfrAccumParams &params, const CudaMultipathBvh &bvh,
                                 int lane_count) {
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        const cudaStream_t s = stream_handle.get();
        launch_dfr_accum_phase(params, DfrAccumPhase::Chain, "dfr_accum_chain_cuda_kernel", bvh,
                               scratch.stack.get(), scratch.overflow.get(), lane_count, s);
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(s),
                        "launch_dfr_accum_chain_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_dfr_accum_chain_cuda(): ") + error.what());
    }
}

void launch_dfr_accum_combined_cuda(const DfrAccumParams &params, const CudaMultipathBvh &bvh,
                                    bool has_non_suffix_strategy, bool has_suffix_strategy,
                                    int lane_count) {
    // Combined 5-bool accumulation body (run_diffraction_order1_accumulation_algo).
    // The single-scene CUDA backend always takes the staged path above, so this
    // is only reachable if a split-scene CUDA arm is ever added; it is
    // instantiated (PrimaryOnly = true) so every OptiX raygen variant has a CUDA
    // kernel. IncludeCoherent is always false here (the coherent body is its own
    // kernel), matching the OptiX combined raygen entries.
    if (lane_count == 0) {
        return;
    }
    try {
        CudaStreamHandle stream_handle;
        ScratchBuffers scratch(lane_count);
        const cudaStream_t s = stream_handle.get();
        const int blocks = block_count(lane_count);
        int *stack = scratch.stack.get();
        int *ovf = scratch.overflow.get();
        if (has_non_suffix_strategy && has_suffix_strategy) {
            dfr_accum_order1_kernel<true, false, true, true, true>
                <<<blocks, kBlockSize, 0, s>>>(params, bvh, stack, ovf, lane_count);
        } else if (has_suffix_strategy) {
            dfr_accum_order1_kernel<true, false, false, false, true>
                <<<blocks, kBlockSize, 0, s>>>(params, bvh, stack, ovf, lane_count);
        } else {
            dfr_accum_order1_kernel<true, false, true, true, false>
                <<<blocks, kBlockSize, 0, s>>>(params, bvh, stack, ovf, lane_count);
        }
        audit_cuda_kernel_launch("dfr_accum_order1_cuda_kernel", static_cast<uint32_t>(blocks), 1, 1,
                                 kBlockSize, 1, 1, static_cast<uint64_t>(lane_count));
        check_cuda_last_error("launch_dfr_accum_combined_cuda(): kernel launch failed");
        audit_cuda_stream_synchronize();
        check_cuda_call(cudaStreamSynchronize(s),
                        "launch_dfr_accum_combined_cuda(): stream sync failed");
    } catch (const std::exception &error) {
        throw_runtime_error_local(std::string("launch_dfr_accum_combined_cuda(): ") + error.what());
    }
}

} // namespace rayd
