#include <src/scene/multipath_cuda.h>

#include <src/scene/cache.h>
#include <src/runtime/math.cuh>
#include <rayd/shared/bvh/cuda_bvh_traverser.h>
#include <rayd/shared/reflection/accumulation_algo.h>
#include <rayd/shared/reflection/epc_algo.h>
#include <rayd/shared/diffraction/paths_algo.h>
#include <rayd/shared/diffraction/accumulation_algo.h>
#include <rayd/shared/reflection/trace_algo.h>
#include <rayd/shared/visibility/segment_algo.h>
#include <rayd/shared/rt/traverser.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace rayd::torch_backend {
namespace {

constexpr int kBlockSize = 128;

// Keep both the by-value CUDA launch ABI and per-block dynamic shared copy
// inside the conservative limits available on sm_87 and newer devices.
static_assert(sizeof(DfrAccumParams) <= 4096);
static_assert(sizeof(DfrAccumParams) <= 48 * 1024);

void cuda_check(cudaError_t result, const char *where) {
    if (result != cudaSuccess)
        throw std::runtime_error(
            std::string("CUDA error in ") + where + ": " + cudaGetErrorString(result));
}

__device__ shared::bvh::CudaBvhView make_view(
    shared::bvh::TriangleSoAView triangles,
    shared::bvh::AabbSoAView bounds,
    shared::bvh::CompactBvhTopologyView topology,
    shared::bvh::TrianglePrimIdMapView prim_map) {
    return {triangles, bounds, topology, prim_map};
}

__device__ shared::bvh::CudaBvhTraverser make_traverser(
    shared::bvh::CudaBvhView view,
    int *stack_nodes,
    int *overflow,
    int lane_count,
    unsigned int lane) {
    shared::bvh::TriangleTraversalScratchView scratch = {
        stack_nodes,
        overflow,
        static_cast<size_t>(lane_count),
        static_cast<size_t>(shared::bvh::kBvhTraversalStackDepth),
        static_cast<size_t>(lane_count) * shared::bvh::kBvhTraversalStackDepth,
        static_cast<size_t>(lane_count),
    };
    return {view, scratch, static_cast<size_t>(lane)};
}

struct TorchCudaReflectionPolicy {
    static constexpr bool allow_aos_inputs = true;
    static constexpr bool allow_packed_triangles = true;
    static constexpr bool honor_output_layout = true;
    static constexpr bool clear_empty_slots = true;
    static constexpr bool nullable_ray_tmax = true;
    static constexpr bool allow_extended_outputs = true;
};

struct TorchCudaVisibilityPolicy {
    static constexpr bool disable_anyhit_without_ignore = true;
    static constexpr bool write_output_t = true;
};

struct TorchCudaReflectionAccumulationPolicy {
    static __forceinline__ __device__ bool include_depth(
        const AccumParams &params, int depth) {
        return depth > 0 || params.include_los != 0;
    }

    static __forceinline__ __device__ void commit(
        const AccumParams &params,
        unsigned int ray_index,
        int depth,
        int cell,
        shared::field::Complex3 field,
        float power) {
        if (params.stage_cell != nullptr && params.stage_value != nullptr) {
            const long long stride = static_cast<long long>(params.max_bounces) + 1ll;
            const long long slot =
                static_cast<long long>(ray_index) * stride + static_cast<long long>(depth);
            ReflAccumStagedValue value;
            value.a = make_float4(power, field.x.r, field.x.i, field.y.r);
            value.b = make_float4(field.y.i, field.z.r, field.z.i, 1.0f);
            params.stage_cell[slot] = cell;
            params.stage_value[slot] = value;
            return;
        }
        const WarpCellGroup group = warp_cell_group(cell);
        atomic_add_same_cell(params.out_field_x_re, cell, field.x.r, group);
        atomic_add_same_cell(params.out_field_x_im, cell, field.x.i, group);
        atomic_add_same_cell(params.out_field_y_re, cell, field.y.r, group);
        atomic_add_same_cell(params.out_field_y_im, cell, field.y.i, group);
        atomic_add_same_cell(params.out_field_z_re, cell, field.z.r, group);
        atomic_add_same_cell(params.out_field_z_im, cell, field.z.i, group);
        atomic_add_same_cell(params.out_reflection_power, cell, power, group);
        atomic_add_warp(params.out_reflection_count, 1);
    }
};

using shared::optix::ReflEpcParams;
using shared::optix::ReflEpcVisibilityIgnoreSurfaceGroup;
namespace epc_detail = shared::multipath::reflection_epc_algo_detail;

struct TorchCudaEpcPolicy {
    static constexpr bool DisableAnyHitWithoutIgnore = false;
};

__device__ __forceinline__ DfrAccumParams &cuda_dfr_params() {
    extern __shared__ __align__(16) unsigned char rayd_dfr_shared_params[];
    return *reinterpret_cast<DfrAccumParams *>(rayd_dfr_shared_params);
}

struct TorchCudaDiffractionAccumulationPolicy {
    static __forceinline__ __device__ const DfrAccumParams &params() { return cuda_dfr_params(); }
    static constexpr int kDirect = RAYD_TORCH_DFR_DIRECT;
    static constexpr int kKeller = RAYD_TORCH_DFR_KELLER;
    static constexpr int kSuffix = RAYD_TORCH_DFR_SUFFIX_REFL;
    using CellGroup = WarpCellGroup;
    static __forceinline__ __device__ bool active_for_state(
        const uint8_t *mask, int width, int stride, int idx) {
        if (mask == nullptr) return true;
        return mask[(width == 1 ? 0 : idx) * stride] != 0u;
    }
    static __forceinline__ __device__ bool active_state(int i) {
        return active_for_state(params().active_mask, params().active_width, params().active_stride, i);
    }
    static __forceinline__ __device__ bool recursive_active_state(int i) {
        return active_for_state(params().recursive_active_mask, params().recursive_active_width,
                                params().recursive_active_stride, i);
    }
    static __forceinline__ __device__ CellGroup cell_group(int cell) { return warp_cell_group(cell); }
    static __forceinline__ __device__ void atomic_add_same_cell(float *p,int i,float v,CellGroup g){
        ::rayd::torch_backend::atomic_add_same_cell(p,i,v,g); }
    static __forceinline__ __device__ void atomic_add_same_cell(int *p,int i,int v,CellGroup g){
        ::rayd::torch_backend::atomic_add_same_cell(p,i,v,g); }
    static __forceinline__ __device__ void atomic_add_warp(float *p,float v){
        ::rayd::torch_backend::atomic_add_warp(p,v); }
    static __forceinline__ __device__ void atomic_add_warp(int *p,int v){
        ::rayd::torch_backend::atomic_add_warp(p,v); }
    static __forceinline__ __device__ bool stage_order1(
        unsigned int lane,int cell,float power,float field_x_re,bool direct,bool keller){
        if(params().stage_cell==nullptr||params().stage_value==nullptr) return false;
        params().stage_cell[lane]=cell;
        params().stage_value[lane]=make_float4(power,field_x_re,direct?1.f:0.f,keller?1.f:0.f);
        return true;
    }
    static __forceinline__ __device__ bool stage_coherent(
        int cell,int state,bool multi,float xr,float xi,float yr,float yi,float zr,float zi){
        if(params().coherent_stage_key==nullptr||params().coherent_stage_value==nullptr) return false;
        const int cells=params().grid_resolution0*params().grid_resolution1;
        const int slot=cell*params().state_count+state;
        DfrCoherentStagedValue value;
        value.a=make_float4(xr,xi,yr,yi); value.b=make_float4(zr,zi,1.f,0.f);
        params().coherent_stage_key[slot]=multi?cells+cell:cell;
        params().coherent_stage_value[slot]=value; return true;
    }
    static __forceinline__ __device__ float read_f32(const float*p,int s,int i){return p[i*s];}
    static __forceinline__ __device__ int read_i32(const int*p,int s,int i){return p[i*s];}
    static __forceinline__ __device__ uint8_t read_u8(const uint8_t*p,int s,int i){return p[i*s];}
    static __forceinline__ __device__ shared::math::Vec3f vec(
        const float*x,const float*y,const float*z,int s,int i){
        const int o=i*s; return shared::math::make_vec3(x[o],y[o],z[o]); }
    static __forceinline__ __device__ float3 opt_vec(
        const float*x,const float*y,const float*z,int s,int i){
        const int o=i*s; return make_float3(x?x[o]:0.f,y?y[o]:0.f,z?z[o]:0.f); }
    static __forceinline__ __device__ int sample_state_index_for_lane(unsigned int lane){
        if(!params().sample_state_index) return int(lane%unsigned(params().state_count));
        const int i=read_i32(params().sample_state_index,params().sample_state_index_stride,int(lane));
        return i>=0&&i<params().state_count?i:-1;
    }
    static __forceinline__ __device__ int state_edge_index_at(int i){return read_i32(params().state_edge_index,params().state_edge_index_stride,i);}
    static __forceinline__ __device__ shared::math::Vec3f state_edge_pos_at(int i){return vec(params().state_edge_pos_x,params().state_edge_pos_y,params().state_edge_pos_z,params().state_edge_pos_stride,i);}
    static __forceinline__ __device__ shared::math::Vec3f state_edge_dir_at(int i){return vec(params().state_edge_dir_x,params().state_edge_dir_y,params().state_edge_dir_z,params().state_edge_dir_stride,i);}
    static __forceinline__ __device__ float state_edge_t_min_at(int i){return read_f32(params().state_edge_t_min,params().state_edge_t_min_stride,i);}
    static __forceinline__ __device__ float state_edge_t_max_at(int i){return read_f32(params().state_edge_t_max,params().state_edge_t_max_stride,i);}
    static __forceinline__ __device__ float sample_edge_weight_for_lane(int i,unsigned int lane,int n){
        if(params().sample_edge_weight) return fmaxf(read_f32(params().sample_edge_weight,params().sample_edge_weight_stride,int(lane)),0.f);
        return fmaxf(state_edge_t_max_at(i)-state_edge_t_min_at(i),0.f)*float(params().state_count)/fmaxf(float(n),1.f);
    }
    static __forceinline__ __device__ int state_prim0_at(int i){return read_i32(params().state_prim0,params().state_prim0_stride,i);}
    static __forceinline__ __device__ int state_prim1_at(int i){return read_i32(params().state_prim1,params().state_prim1_stride,i);}
    static __forceinline__ __device__ float state_exterior_angle_at(int i){return read_f32(params().state_exterior_angle,params().state_exterior_angle_stride,i);}
    static __forceinline__ __device__ float state_src_power_at(int i){return read_f32(params().state_src_power,params().state_src_power_stride,i);}
    static __forceinline__ __device__ shared::math::Vec3f state_src_at(int i){return vec(params().state_src_x,params().state_src_y,params().state_src_z,params().state_src_stride,i);}
    static __forceinline__ __device__ float3 state_wi_at(int i){return opt_vec(params().state_wi_x,params().state_wi_y,params().state_wi_z,params().state_wi_stride,i);}
    static __forceinline__ __device__ int recursive_state_edge_index_at(int i){return read_i32(params().recursive_state_edge_index,params().recursive_state_edge_index_stride,i);}
    static __forceinline__ __device__ shared::math::Vec3f recursive_state_edge_pos_at(int i){return vec(params().recursive_state_edge_pos_x,params().recursive_state_edge_pos_y,params().recursive_state_edge_pos_z,params().recursive_state_edge_pos_stride,i);}
    static __forceinline__ __device__ shared::math::Vec3f recursive_state_edge_dir_at(int i){return vec(params().recursive_state_edge_dir_x,params().recursive_state_edge_dir_y,params().recursive_state_edge_dir_z,params().recursive_state_edge_dir_stride,i);}
    static __forceinline__ __device__ float recursive_state_edge_t_min_at(int i){return read_f32(params().recursive_state_edge_t_min,params().recursive_state_edge_t_min_stride,i);}
    static __forceinline__ __device__ float recursive_state_edge_t_max_at(int i){return read_f32(params().recursive_state_edge_t_max,params().recursive_state_edge_t_max_stride,i);}
    static __forceinline__ __device__ int recursive_state_prim0_at(int i){return read_i32(params().recursive_state_prim0,params().recursive_state_prim0_stride,i);}
    static __forceinline__ __device__ int recursive_state_prim1_at(int i){return read_i32(params().recursive_state_prim1,params().recursive_state_prim1_stride,i);}
    static __forceinline__ __device__ float recursive_state_exterior_angle_at(int i){return read_f32(params().recursive_state_exterior_angle,params().recursive_state_exterior_angle_stride,i);}
    static __forceinline__ __device__ bool material_valid_at(int i){return !params().material_valid||read_u8(params().material_valid,params().material_valid_stride,i)!=0u;}
    static __forceinline__ __device__ float material_gain_at(int i){return read_f32(params().material_gain,params().material_gain_stride,i);}
};

__device__ __forceinline__ bool epc_prim_ignored(
    const ReflEpcParams &params, int prim, int ig0, int ig1, int ig2) {
    int candidate = prim;
    if (params.visibility_ignore_mode == ReflEpcVisibilityIgnoreSurfaceGroup)
        candidate = epc_detail::surface_group_for_prim(params, prim);
    return (ig0 >= 0 && candidate == ig0) ||
           (ig1 >= 0 && candidate == ig1) ||
           (ig2 >= 0 && candidate == ig2);
}

__device__ __forceinline__ int epc_first_blocker(
    const shared::bvh::CudaBvhView &view,
    const shared::bvh::TriangleTraversalScratchView &scratch,
    size_t lane,
    const ReflEpcParams &params,
    shared::math::Vec3f origin,
    shared::math::Vec3f direction,
    float tmin,
    float tmax,
    int ig0,
    int ig1,
    int ig2) {
    const float inv_dx = shared::bvh::safe_rcp(direction.x);
    const float inv_dy = shared::bvh::safe_rcp(direction.y);
    const float inv_dz = shared::bvh::safe_rcp(direction.z);
    float best_t = tmax;
    int best_prim = -1;
    int sp = 0;
    int node = 0;
    for (;;) {
        while (node >= 0 && !shared::bvh::is_leaf_node(view.topology, node)) {
            const int left = view.topology.left_child[node];
            const int right = view.topology.right_child[node];
            float t_left = 0.0f;
            float t_right = 0.0f;
            const bool hit_left = shared::bvh::intersect_node_bounds(
                view.node_bounds, left, origin.x, origin.y, origin.z,
                inv_dx, inv_dy, inv_dz, tmin, best_t, t_left);
            const bool hit_right = shared::bvh::intersect_node_bounds(
                view.node_bounds, right, origin.x, origin.y, origin.z,
                inv_dx, inv_dy, inv_dz, tmin, best_t, t_right);
            if (hit_left && hit_right) {
                int near_child = left;
                int far_child = right;
                if (!shared::bvh::near_child_is_left(t_left, t_right, left, right)) {
                    near_child = right;
                    far_child = left;
                }
                if (!shared::bvh::stack_push(scratch, lane, static_cast<size_t>(sp), far_child))
                    return -1;
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
                if (epc_prim_ignored(params, prim, ig0, ig1, ig2))
                    continue;
                const auto v = shared::bvh::load_triangle(view.triangles, prim);
                const auto hit = shared::bvh::intersect_triangle_watertight(
                    origin.x, origin.y, origin.z,
                    direction.x, direction.y, direction.z,
                    v.ax, v.ay, v.az, v.bx, v.by, v.bz, v.cx, v.cy, v.cz,
                    tmin, best_t);
                if (hit.hit && (hit.t < best_t || (hit.t == best_t && prim < best_prim))) {
                    best_t = hit.t;
                    best_prim = prim;
                }
            }
        }
        if (sp == 0)
            break;
        --sp;
        node = shared::bvh::stack_load(scratch, lane, static_cast<size_t>(sp));
    }
    return best_prim;
}

struct TorchCudaEpcTraverser {
    shared::bvh::CudaBvhView view;
    shared::bvh::TriangleTraversalScratchView scratch;
    size_t lane;
    const ReflEpcParams *params;

    __device__ __forceinline__ shared::rt::TriangleHit trace_closest(
        shared::math::Vec3f origin, shared::math::Vec3f direction,
        float tmin, float tmax) const {
        return shared::bvh::CudaBvhTraverser{view, scratch, lane}.trace_closest(
            origin, direction, tmin, tmax);
    }
    __device__ __forceinline__ shared::rt::TriangleHit trace_first_blocker(
        shared::math::Vec3f origin, shared::math::Vec3f direction,
        float tmin, float tmax, const int32_t *ignore, int ignore_count) const {
        const int ig0 = ignore_count > 0 ? ignore[0] : -1;
        const int ig1 = ignore_count > 1 ? ignore[1] : -1;
        const int ig2 = ignore_count > 2 ? ignore[2] : -1;
        const int prim = epc_first_blocker(
            view, scratch, lane, *params, origin, direction, tmin, tmax, ig0, ig1, ig2);
        shared::rt::TriangleHit hit{};
        hit.t = tmax;
        if (prim >= 0) {
            hit.prim = view.prim_map.local_prim_id[prim];
            hit.instance = view.prim_map.shape_id[prim];
            hit.hit = 1u;
        } else {
            hit.prim = -1;
            hit.instance = -1;
        }
        return hit;
    }
    __device__ __forceinline__ bool trace_occluded_ignore(
        shared::math::Vec3f origin, shared::math::Vec3f direction,
        float tmin, float tmax, const int32_t *ignore, int ignore_count) const {
        return trace_first_blocker(origin, direction, tmin, tmax, ignore, ignore_count).hit != 0u;
    }
    __device__ __forceinline__ bool trace_occluded(
        shared::math::Vec3f origin, shared::math::Vec3f direction,
        float tmin, float tmax) const {
        return trace_occluded_ignore(origin, direction, tmin, tmax, nullptr, 0);
    }
};

static_assert(shared::rt::is_traverser_v<TorchCudaEpcTraverser>);

__global__ void reflection_kernel(
    shared::optix::ReflectionTraceParams params,
    shared::bvh::TriangleSoAView triangles,
    shared::bvh::AabbSoAView bounds,
    shared::bvh::CompactBvhTopologyView topology,
    shared::bvh::TrianglePrimIdMapView prim_map,
    int *stack_nodes,
    int *overflow,
    int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count))
        return;
    const auto traverser = make_traverser(
        make_view(triangles, bounds, topology, prim_map),
        stack_nodes, overflow, lane_count, lane);
    using Config = shared::rt::TraceConfig<
        TorchCudaReflectionPolicy, shared::bvh::CudaBvhTraverser>;
    shared::multipath::reflection_trace_algo<Config>(
        params, lane, traverser, traverser);
}

__global__ void visibility_kernel(
    shared::optix::SegmentVisibilityParams params,
    int variant,
    shared::bvh::TriangleSoAView triangles,
    shared::bvh::AabbSoAView bounds,
    shared::bvh::CompactBvhTopologyView topology,
    shared::bvh::TrianglePrimIdMapView prim_map,
    int *stack_nodes,
    int *overflow,
    int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count))
        return;
    const auto traverser = make_traverser(
        make_view(triangles, bounds, topology, prim_map),
        stack_nodes, overflow, lane_count, lane);
    using Config = shared::rt::TraceConfig<
        TorchCudaVisibilityPolicy, shared::bvh::CudaBvhTraverser>;
    switch (static_cast<CudaVisibilityVariant>(variant)) {
    case CudaVisibilityVariant::Single:
        shared::multipath::segment_visibility_algo<Config>(params, lane, traverser);
        break;
    case CudaVisibilityVariant::Pair:
        shared::multipath::segment_pair_visibility_algo<Config>(params, lane, traverser);
        break;
    case CudaVisibilityVariant::AxialEdge:
        shared::multipath::axial_edge_visibility_algo<Config>(params, lane, traverser);
        break;
    case CudaVisibilityVariant::Chain:
        shared::multipath::segment_chain_visibility_algo<Config>(params, lane, traverser);
        break;
    }
}

__global__ void reflection_accumulation_kernel(
    AccumParams params,
    shared::bvh::TriangleSoAView triangles,
    shared::bvh::AabbSoAView bounds,
    shared::bvh::CompactBvhTopologyView topology,
    shared::bvh::TrianglePrimIdMapView prim_map,
    int *stack_nodes,
    int *overflow,
    int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count))
        return;
    const auto traverser = make_traverser(
        make_view(triangles, bounds, topology, prim_map),
        stack_nodes, overflow, lane_count, lane);
    shared::multipath::reflection_accumulation_algo<
        AccumParams, TorchCudaReflectionAccumulationPolicy,
        shared::bvh::CudaBvhTraverser>(params, lane, traverser, traverser);
}

template <bool SourcePass>
__global__ void diffraction_paths_kernel(
    DfrPathParams params,
    shared::bvh::TriangleSoAView triangles,
    shared::bvh::AabbSoAView bounds,
    shared::bvh::CompactBvhTopologyView topology,
    shared::bvh::TrianglePrimIdMapView prim_map,
    int *stack_nodes,
    int *overflow,
    int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count))
        return;
    const auto traverser = make_traverser(
        make_view(triangles, bounds, topology, prim_map),
        stack_nodes, overflow, lane_count, lane);
    if constexpr (SourcePass)
        shared::multipath::trace_paths_source_visibility_algo<DfrPathParams>(
            params, lane, traverser, traverser);
    else
        shared::multipath::trace_paths_target_export_algo<DfrPathParams>(
            params, lane, traverser, traverser);
}

__global__ void diffraction_accumulation_kernel(
    DfrAccumParams kernel_params,
    int variant,
    shared::bvh::TriangleSoAView triangles,
    shared::bvh::AabbSoAView bounds,
    shared::bvh::CompactBvhTopologyView topology,
    shared::bvh::TrianglePrimIdMapView prim_map,
    int *stack_nodes,
    int *overflow,
    int lane_count) {
    auto *dst = reinterpret_cast<unsigned char *>(&cuda_dfr_params());
    const auto *src = reinterpret_cast<const unsigned char *>(&kernel_params);
    for (int i = threadIdx.x; i < static_cast<int>(sizeof(DfrAccumParams)); i += blockDim.x)
        dst[i] = src[i];
    __syncthreads();
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count))
        return;
    const auto traverser = make_traverser(
        make_view(triangles, bounds, topology, prim_map),
        stack_nodes, overflow, lane_count, lane);
    using Algo = shared::multipath::DiffractionAccumulationAlgo<
        TorchCudaDiffractionAccumulationPolicy, shared::bvh::CudaBvhTraverser>;
    Algo algo{traverser, traverser};
    switch (variant) {
    case 6:
        algo.run_diffraction_order1_source_visibility_algo<true>(lane);
        break;
    case 7:
        algo.run_diffraction_order1_no_suffix_target_accumulation_algo<true>(lane);
        break;
    case 8:
        algo.run_diffraction_order1_suffix_first_visibility_algo<true>(lane);
        break;
    case 9:
        algo.run_diffraction_order1_suffix_target_accumulation_algo<true>(lane);
        break;
    case 11:
        algo.run_diffraction_order1_coherent_accumulation_algo<true>(lane);
        break;
    case 13:
        algo.run_diffraction_chain_accumulation_algo<true>(lane);
        break;
    }
}

template <bool DirectOnly, bool PrimaryOnly>
__global__ void reflection_epc_kernel(
    ReflEpcParams params,
    shared::bvh::TriangleSoAView triangles,
    shared::bvh::AabbSoAView bounds,
    shared::bvh::CompactBvhTopologyView topology,
    shared::bvh::TrianglePrimIdMapView prim_map,
    int *stack_nodes,
    int *overflow,
    int lane_count) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= static_cast<unsigned int>(lane_count))
        return;
    const auto view = make_view(triangles, bounds, topology, prim_map);
    shared::bvh::TriangleTraversalScratchView scratch = {
        stack_nodes, overflow, static_cast<size_t>(lane_count),
        static_cast<size_t>(shared::bvh::kBvhTraversalStackDepth),
        static_cast<size_t>(lane_count) * shared::bvh::kBvhTraversalStackDepth,
        static_cast<size_t>(lane_count)};
    const TorchCudaEpcTraverser traverser{view, scratch, static_cast<size_t>(lane), &params};
    using Config = shared::rt::TraceConfig<TorchCudaEpcPolicy, TorchCudaEpcTraverser>;
    shared::multipath::run_reflection_epc_algo<Config, DirectOnly, PrimaryOnly>(
        params, lane, traverser, traverser);
}

shared::bvh::TrianglePrimIdMapView scene_prim_map(const SceneCache &scene) {
    return {scene.face_shape_id.data_ptr<int>(), scene.face_local_id.data_ptr<int>(),
            static_cast<size_t>(scene.global_faces.size(0))};
}

} // namespace

void launch_reflection_trace_cuda(
    const SceneCache &scene,
    const shared::optix::ReflectionTraceParams &params,
    int lane_count) {
    if (lane_count == 0)
        return;
    const auto iopts = scene.global_faces.options();
    at::Tensor stack = at::empty(
        {shared::bvh::kBvhTraversalStackDepth, lane_count}, iopts);
    at::Tensor overflow = at::empty({lane_count}, iopts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene.device_index).stream();
    const int blocks = (lane_count + kBlockSize - 1) / kBlockSize;
    reflection_kernel<<<blocks, kBlockSize, 0, stream>>>(
        params, scene_triangle_view(scene), scene_triangle_bvh_bounds_view(scene),
        scene_triangle_bvh_topology_view(scene), scene_prim_map(scene),
        stack.data_ptr<int>(), overflow.data_ptr<int>(), lane_count);
    cuda_check(cudaGetLastError(), "launch_reflection_trace_cuda");
}

void launch_segment_visibility_cuda(
    const SceneCache &scene,
    const shared::optix::SegmentVisibilityParams &params,
    CudaVisibilityVariant variant,
    int lane_count) {
    if (lane_count == 0)
        return;
    const auto iopts = scene.global_faces.options();
    at::Tensor stack = at::empty(
        {shared::bvh::kBvhTraversalStackDepth, lane_count}, iopts);
    at::Tensor overflow = at::empty({lane_count}, iopts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene.device_index).stream();
    const int blocks = (lane_count + kBlockSize - 1) / kBlockSize;
    // The shared algorithm keeps the historical null-traversable fast guard.
    // A CUDA BVH is represented by explicit views instead of an OptiX handle,
    // so use a non-zero backend sentinel for the duration of this launch.
    shared::optix::SegmentVisibilityParams cuda_params = params;
    cuda_params.handle = 1ull;
    visibility_kernel<<<blocks, kBlockSize, 0, stream>>>(
        cuda_params, static_cast<int>(variant), scene_triangle_view(scene),
        scene_triangle_bvh_bounds_view(scene), scene_triangle_bvh_topology_view(scene),
        scene_prim_map(scene), stack.data_ptr<int>(), overflow.data_ptr<int>(), lane_count);
    cuda_check(cudaGetLastError(), "launch_segment_visibility_cuda");
}

void launch_reflection_accumulation_cuda(
    const SceneCache &scene,
    const AccumParams &params,
    int lane_count) {
    if (lane_count == 0)
        return;
    const auto iopts = scene.global_faces.options();
    at::Tensor stack = at::empty(
        {shared::bvh::kBvhTraversalStackDepth, lane_count}, iopts);
    at::Tensor overflow = at::empty({lane_count}, iopts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene.device_index).stream();
    const int blocks = (lane_count + kBlockSize - 1) / kBlockSize;
    reflection_accumulation_kernel<<<blocks, kBlockSize, 0, stream>>>(
        params, scene_triangle_view(scene), scene_triangle_bvh_bounds_view(scene),
        scene_triangle_bvh_topology_view(scene), scene_prim_map(scene),
        stack.data_ptr<int>(), overflow.data_ptr<int>(), lane_count);
    cuda_check(cudaGetLastError(), "launch_reflection_accumulation_cuda");
}

void launch_reflection_epc_cuda(
    const SceneCache &scene,
    const ReflEpcParams &params,
    bool direct_only,
    bool primary_visibility_only,
    int lane_count) {
    if (lane_count == 0)
        return;
    const auto iopts = scene.global_faces.options();
    at::Tensor stack = at::empty(
        {shared::bvh::kBvhTraversalStackDepth, lane_count}, iopts);
    at::Tensor overflow = at::empty({lane_count}, iopts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene.device_index).stream();
    const int blocks = (lane_count + kBlockSize - 1) / kBlockSize;
#define RAYD_LAUNCH_EPC(DIRECT, PRIMARY) \
    reflection_epc_kernel<DIRECT, PRIMARY><<<blocks, kBlockSize, 0, stream>>>( \
        params, scene_triangle_view(scene), scene_triangle_bvh_bounds_view(scene), \
        scene_triangle_bvh_topology_view(scene), scene_prim_map(scene), \
        stack.data_ptr<int>(), overflow.data_ptr<int>(), lane_count)
    if (direct_only && primary_visibility_only)
        RAYD_LAUNCH_EPC(true, true);
    else if (direct_only)
        RAYD_LAUNCH_EPC(true, false);
    else
        RAYD_LAUNCH_EPC(false, false);
#undef RAYD_LAUNCH_EPC
    cuda_check(cudaGetLastError(), "launch_reflection_epc_cuda");
}

void launch_diffraction_paths_cuda(
    const SceneCache &scene,
    const DfrPathParams &params,
    int lane_count) {
    if (lane_count == 0)
        return;
    const auto iopts = scene.global_faces.options();
    at::Tensor stack = at::empty(
        {shared::bvh::kBvhTraversalStackDepth, lane_count}, iopts);
    at::Tensor overflow = at::empty({lane_count}, iopts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene.device_index).stream();
    const int blocks = (lane_count + kBlockSize - 1) / kBlockSize;
#define RAYD_LAUNCH_DFR_PATH(SOURCE) \
    diffraction_paths_kernel<SOURCE><<<blocks, kBlockSize, 0, stream>>>( \
        params, scene_triangle_view(scene), scene_triangle_bvh_bounds_view(scene), \
        scene_triangle_bvh_topology_view(scene), scene_prim_map(scene), \
        stack.data_ptr<int>(), overflow.data_ptr<int>(), lane_count)
    RAYD_LAUNCH_DFR_PATH(true);
    RAYD_LAUNCH_DFR_PATH(false);
#undef RAYD_LAUNCH_DFR_PATH
    cuda_check(cudaGetLastError(), "launch_diffraction_paths_cuda");
}

void launch_diffraction_accumulation_cuda(
    const SceneCache &scene,
    const DfrAccumParams &params,
    int pipeline_variant,
    int lane_count) {
    if (lane_count == 0)
        return;
    if (pipeline_variant != 6 && pipeline_variant != 7 &&
        pipeline_variant != 8 && pipeline_variant != 9 &&
        pipeline_variant != 11 && pipeline_variant != 13)
        throw std::runtime_error("unsupported CUDA diffraction accumulation variant");
    const auto iopts = scene.global_faces.options();
    at::Tensor stack = at::empty(
        {shared::bvh::kBvhTraversalStackDepth, lane_count}, iopts);
    at::Tensor overflow = at::empty({lane_count}, iopts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(scene.device_index).stream();
    const int blocks = (lane_count + kBlockSize - 1) / kBlockSize;
    diffraction_accumulation_kernel<<<
        blocks, kBlockSize, sizeof(DfrAccumParams), stream>>>(
        params, pipeline_variant, scene_triangle_view(scene),
        scene_triangle_bvh_bounds_view(scene), scene_triangle_bvh_topology_view(scene),
        scene_prim_map(scene), stack.data_ptr<int>(), overflow.data_ptr<int>(), lane_count);
    cuda_check(cudaGetLastError(), "launch_diffraction_accumulation_cuda");
}

} // namespace rayd::torch_backend
