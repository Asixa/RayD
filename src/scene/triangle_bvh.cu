#include <src/scene/triangle_bvh.h>

#include <src/scene/cache.h>
#include <rayd/detail/bvh/triangle_query.h>
#include <rayd/detail/contracts.h>

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {
namespace {

constexpr int kBlockSize = 128;

void cuda_check(cudaError_t result, const char *where) {
    if (result != cudaSuccess)
        throw std::runtime_error(
            std::string("CUDA error in ") + where + ": " + cudaGetErrorString(result));
}

__global__ void triangle_bounds_kernel(
    const float *p0_x, const float *p0_y, const float *p0_z,
    const float *e1_x, const float *e1_y, const float *e1_z,
    const float *e2_x, const float *e2_y, const float *e2_z,
    int count,
    float *min_x, float *min_y, float *min_z,
    float *max_x, float *max_y, float *max_z,
    shared::bvh::BvhBounds3 *packed) {
    const int primitive = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (primitive >= count)
        return;
    const float ax = p0_x[primitive];
    const float ay = p0_y[primitive];
    const float az = p0_z[primitive];
    const float bx = ax + e1_x[primitive];
    const float by = ay + e1_y[primitive];
    const float bz = az + e1_z[primitive];
    const float cx = ax + e2_x[primitive];
    const float cy = ay + e2_y[primitive];
    const float cz = az + e2_z[primitive];
    shared::bvh::BvhBounds3 bounds = {
        {fminf(ax, fminf(bx, cx)), fminf(ay, fminf(by, cy)), fminf(az, fminf(bz, cz))},
        {fmaxf(ax, fmaxf(bx, cx)), fmaxf(ay, fmaxf(by, cy)), fmaxf(az, fmaxf(bz, cz))}};
    min_x[primitive] = bounds.min.x;
    min_y[primitive] = bounds.min.y;
    min_z[primitive] = bounds.min.z;
    max_x[primitive] = bounds.max.x;
    max_y[primitive] = bounds.max.y;
    max_z[primitive] = bounds.max.z;
    packed[primitive] = bounds;
}

__global__ void prepare_triangle_rays_kernel(
    const float *ray_o,
    const float *ray_d,
    const float *ray_tmax,
    const bool *active,
    int64_t ray_count,
    int64_t o_stride0,
    int64_t o_stride1,
    int64_t d_stride0,
    int64_t d_stride1,
    float *ox, float *oy, float *oz,
    float *dx, float *dy, float *dz,
    float *tmax,
    int *active_i32) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= ray_count)
        return;
    const float *o = ray_o + ray * o_stride0;
    const float *d = ray_d + ray * d_stride0;
    ox[ray] = o[0 * o_stride1];
    oy[ray] = o[1 * o_stride1];
    oz[ray] = o[2 * o_stride1];
    dx[ray] = d[0 * d_stride1];
    dy[ray] = d[1 * d_stride1];
    dz[ray] = d[2 * d_stride1];
    tmax[ray] = ray_tmax == nullptr ? FLT_MAX : ray_tmax[ray];
    active_i32[ray] = active == nullptr || active[ray] ? 1 : 0;
}

__global__ void finalize_triangle_hits_kernel(
    const int *face_local_id,
    int64_t ray_count,
    const int *winner_global,
    const float *bary_u,
    const float *bary_v,
    int *out_local,
    int *out_global,
    float *out_bary_uv) {
    const int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= ray_count)
        return;
    const int global = winner_global[ray];
    if (out_local != nullptr)
        out_local[ray] = global >= 0 ? face_local_id[global] : -1;
    if (out_global != nullptr)
        out_global[ray] = global;
    if (out_bary_uv != nullptr) {
        out_bary_uv[ray * 2 + 0] = bary_u[ray];
        out_bary_uv[ray * 2 + 1] = bary_v[ray];
    }
}

} // namespace

void compute_triangle_bvh_bounds_cuda(
    const SceneCache &scene,
    at::Tensor &primitive_min_x,
    at::Tensor &primitive_min_y,
    at::Tensor &primitive_min_z,
    at::Tensor &primitive_max_x,
    at::Tensor &primitive_max_y,
    at::Tensor &primitive_max_z,
    at::Tensor &packed_bounds,
    cudaStream_t stream) {
    const int64_t count = scene.global_faces.numel() / 3;
    if (count == 0)
        return;
    const int blocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
    triangle_bounds_kernel<<<blocks, kBlockSize, 0, stream>>>(
        scene.tri_p0_x.data_ptr<float>(), scene.tri_p0_y.data_ptr<float>(),
        scene.tri_p0_z.data_ptr<float>(), scene.tri_e1_x.data_ptr<float>(),
        scene.tri_e1_y.data_ptr<float>(), scene.tri_e1_z.data_ptr<float>(),
        scene.tri_e2_x.data_ptr<float>(), scene.tri_e2_y.data_ptr<float>(),
        scene.tri_e2_z.data_ptr<float>(), static_cast<int>(count),
        primitive_min_x.data_ptr<float>(), primitive_min_y.data_ptr<float>(),
        primitive_min_z.data_ptr<float>(), primitive_max_x.data_ptr<float>(),
        primitive_max_y.data_ptr<float>(), primitive_max_z.data_ptr<float>(),
        reinterpret_cast<shared::bvh::BvhBounds3 *>(packed_bounds.data_ptr<uint8_t>()));
    cuda_check(cudaGetLastError(), "compute_triangle_bvh_bounds_cuda");
}

void launch_intersect_cuda_bvh(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    at::Tensor &out_t,
    int *out_shape_id,
    int *out_local_prim_id,
    int *out_global_prim_id,
    float *out_bary_uv,
    cudaStream_t stream) {
    const int64_t ray_count = ray_o.size(0);
    if (ray_count == 0)
        return;
    if (!scene.custom_triangle_bvh.valid)
        throw std::runtime_error("CUDA triangle BVH is not built for this scene.");

    const auto fopts = ray_o.options();
    const auto iopts = scene.global_faces.options();
    at::Tensor ox = at::empty({ray_count}, fopts);
    at::Tensor oy = at::empty({ray_count}, fopts);
    at::Tensor oz = at::empty({ray_count}, fopts);
    at::Tensor dx = at::empty({ray_count}, fopts);
    at::Tensor dy = at::empty({ray_count}, fopts);
    at::Tensor dz = at::empty({ray_count}, fopts);
    at::Tensor tmax = at::empty({ray_count}, fopts);
    at::Tensor active_i32 = at::empty({ray_count}, iopts);
    at::Tensor bary_u = at::empty({ray_count}, fopts);
    at::Tensor bary_v = at::empty({ray_count}, fopts);
    at::Tensor shape = out_shape_id == nullptr ? at::empty({ray_count}, iopts) : at::Tensor();
    at::Tensor winner_global = at::empty({ray_count}, iopts);
    at::Tensor stack = at::empty(
        {shared::bvh::kBvhTraversalStackDepth, ray_count}, iopts);
    at::Tensor overflow = at::empty({ray_count}, iopts);

    const int blocks = static_cast<int>((ray_count + kBlockSize - 1) / kBlockSize);
    prepare_triangle_rays_kernel<<<blocks, kBlockSize, 0, stream>>>(
        ray_o.data_ptr<float>(), ray_d.data_ptr<float>(),
        !ray_tmax.defined() || ray_tmax.numel() == 0 ? nullptr : ray_tmax.data_ptr<float>(),
        !active.defined() || active.numel() == 0 ? nullptr : active.data_ptr<bool>(),
        ray_count, ray_o.stride(0), ray_o.stride(1), ray_d.stride(0), ray_d.stride(1),
        ox.data_ptr<float>(), oy.data_ptr<float>(), oz.data_ptr<float>(),
        dx.data_ptr<float>(), dy.data_ptr<float>(), dz.data_ptr<float>(),
        tmax.data_ptr<float>(), active_i32.data_ptr<int>());

    shared::bvh::TriangleClosestHitParams params = {};
    params.triangles = scene_triangle_view(scene);
    params.node_bounds = scene_triangle_bvh_bounds_view(scene);
    params.topology = scene_triangle_bvh_topology_view(scene);
    params.rays = {ox.data_ptr<float>(), oy.data_ptr<float>(), oz.data_ptr<float>(),
                   dx.data_ptr<float>(), dy.data_ptr<float>(), dz.data_ptr<float>(),
                   tmax.data_ptr<float>(), active_i32.data_ptr<int>(),
                   static_cast<size_t>(ray_count)};
    // Emit the global primitive id in the query's local-id slot. The finalizer
    // maps it back to the public mesh-local id without a host lookup.
    params.prim_map = {scene.face_shape_id.data_ptr<int>(),
                       scene.primitive_identity.data_ptr<int>(),
                       static_cast<size_t>(scene.global_faces.size(0))};
    params.output = {out_t.data_ptr<float>(), bary_u.data_ptr<float>(), bary_v.data_ptr<float>(),
                     out_shape_id == nullptr ? shape.data_ptr<int>() : out_shape_id,
                     winner_global.data_ptr<int>(), static_cast<size_t>(ray_count)};
    params.scratch = {stack.data_ptr<int>(), overflow.data_ptr<int>(),
                      static_cast<size_t>(ray_count),
                      static_cast<size_t>(shared::bvh::kBvhTraversalStackDepth),
                      static_cast<size_t>(stack.numel()), static_cast<size_t>(overflow.numel())};
    params.t_min = shared::SmallEpsilon;
    params.stream = stream;
    shared::bvh::launch_triangle_closest_hit_async(params);
    shared::bvh::launch_triangle_closest_hit_repair_async(params);
    finalize_triangle_hits_kernel<<<blocks, kBlockSize, 0, stream>>>(
        scene.face_local_id.data_ptr<int>(), ray_count, winner_global.data_ptr<int>(),
        bary_u.data_ptr<float>(), bary_v.data_ptr<float>(), out_local_prim_id,
        out_global_prim_id, out_bary_uv);
    cuda_check(cudaGetLastError(), "launch_intersect_cuda_bvh");
}

} // namespace rayd::torch_backend
