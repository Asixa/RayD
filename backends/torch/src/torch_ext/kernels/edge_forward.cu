#include <raydtorch/edge_kernels.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace raydtorch {

namespace {

__device__ float3 make_f3(const float *ptr) {
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__device__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 mul3(float s, float3 a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void edge_forward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const int *__restrict__ edge_shape_id,
    const int *__restrict__ edge_local_id,
    int64_t edge_count,
    const float *__restrict__ point,
    int64_t point_count,
    float *__restrict__ distance,
    float *__restrict__ edge_point,
    float *__restrict__ edge_t,
    int *__restrict__ shape_id,
    int *__restrict__ edge_id,
    int *__restrict__ global_edge_id,
    int *__restrict__ tape_edge_id,
    float *__restrict__ tape_s,
    float *__restrict__ tape_d) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;

    const float3 p = make_f3(point + point_idx * 3);
    float best_dist2 = CUDART_INF_F;
    float best_s = 0.f;
    int best_edge = -1;
    float3 best_edge_point = make_float3(0.f, 0.f, 0.f);
    float3 best_d = make_float3(0.f, 0.f, 0.f);

    for (int edge_idx = 0; edge_idx < edge_count; ++edge_idx) {
        const int i0 = edge_v0[edge_idx];
        const int i1 = edge_v1[edge_idx];
        const float3 a = make_f3(vertices + i0 * 3);
        const float3 b = make_f3(vertices + i1 * 3);
        const float3 ab = sub3(b, a);
        const float denom = fmaxf(dot3(ab, ab), 1e-20f);
        float s = dot3(sub3(p, a), ab) / denom;
        s = fminf(1.f, fmaxf(0.f, s));
        const float3 q = add3(a, mul3(s, ab));
        const float3 d = sub3(p, q);
        const float dist2 = dot3(d, d);
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_s = s;
            best_edge = edge_idx;
            best_edge_point = q;
            best_d = d;
        }
    }

    distance[point_idx] = sqrtf(best_dist2);
    edge_t[point_idx] = best_s;
    tape_s[point_idx] = best_s;
    tape_edge_id[point_idx] = best_edge;
    shape_id[point_idx] = best_edge >= 0 ? edge_shape_id[best_edge] : -1;
    edge_id[point_idx] = best_edge >= 0 ? edge_local_id[best_edge] : -1;
    global_edge_id[point_idx] = best_edge;
    edge_point[point_idx * 3 + 0] = best_edge_point.x;
    edge_point[point_idx * 3 + 1] = best_edge_point.y;
    edge_point[point_idx * 3 + 2] = best_edge_point.z;
    tape_d[point_idx * 3 + 0] = best_d.x;
    tape_d[point_idx * 3 + 1] = best_d.y;
    tape_d[point_idx * 3 + 2] = best_d.z;
}

} // namespace

EdgeForwardOutputs edge_forward_cuda(const SceneCache &scene, const at::Tensor &point) {
    const MeshRecord &mesh = scene.meshes[0];
    const int64_t point_count = point.size(0);
    const int64_t edge_count = scene.edge_v0.size(0);
    auto fopts = point.options();
    auto iopts = scene.edge_v0.options();

    EdgeForwardOutputs out;
    out.distance = at::empty({point_count}, fopts);
    out.edge_point = at::empty({point_count, 3}, fopts);
    out.edge_t = at::empty({point_count}, fopts);
    out.shape_id = at::empty({point_count}, iopts);
    out.edge_id = at::empty({point_count}, iopts);
    out.global_edge_id = at::empty({point_count}, iopts);
    out.tape_edge_id = out.global_edge_id;
    out.tape_s = out.edge_t;
    out.tape_d = at::empty({point_count, 3}, fopts);

    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.get_device()).stream();
    edge_forward_kernel<<<blocks, threads, 0, stream>>>(
        mesh.vertices.data_ptr<float>(),
        scene.edge_v0.data_ptr<int>(),
        scene.edge_v1.data_ptr<int>(),
        scene.edge_shape_id.data_ptr<int>(),
        scene.edge_local_id.data_ptr<int>(),
        edge_count,
        point.data_ptr<float>(),
        point_count,
        out.distance.data_ptr<float>(),
        out.edge_point.data_ptr<float>(),
        out.edge_t.data_ptr<float>(),
        out.shape_id.data_ptr<int>(),
        out.edge_id.data_ptr<int>(),
        out.global_edge_id.data_ptr<int>(),
        out.tape_edge_id.data_ptr<int>(),
        out.tape_s.data_ptr<float>(),
        out.tape_d.data_ptr<float>());
    return out;
}

} // namespace raydtorch
