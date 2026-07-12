#include <rayd/torch/edge/kernels.h>
#include <rayd/torch/common/math.cuh>
#include <rayd/shared/edge/edge_distance_math.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace rayd::torch_backend {

namespace {

__device__ shared::math::Vec3f to_shared_vec3(float3 value) {
    return shared::math::make_vec3(value.x, value.y, value.z);
}

__device__ float3 from_shared_vec3(shared::math::Vec3f value) {
    return make_float3(value.x, value.y, value.z);
}

int64_t optional_stride(const at::Tensor *tensor, int64_t dim) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0 || tensor->dim() <= dim)
        return 0;
    return tensor->stride(dim);
}

__device__ float read_scalar_or_zero(const float *base, int64_t index, int64_t stride0) {
    return base == nullptr ? 0.f : base[index * stride0];
}

__device__ float3 read_vec3_or_zero(const float *base, int64_t index, int64_t stride0, int64_t stride1) {
    return base == nullptr ? make_float3(0.f, 0.f, 0.f)
                           : make_float3(base[index * stride0 + 0 * stride1],
                                         base[index * stride0 + 1 * stride1],
                                         base[index * stride0 + 2 * stride1]);
}

__global__ void edge_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const float *__restrict__ point,
    const int *__restrict__ tape_edge_id,
    const float *__restrict__ tape_s,
    const float *__restrict__ tape_d,
    const float *__restrict__ grad_distance,
    const float *__restrict__ grad_edge_point,
    const float *__restrict__ grad_edge_t,
    const float *__restrict__ grad_edge_t_alias,
    int64_t grad_distance_stride0,
    int64_t grad_edge_point_stride0,
    int64_t grad_edge_point_stride1,
    int64_t grad_edge_t_stride0,
    int64_t grad_edge_t_alias_stride0,
    int64_t point_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_point) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;
    for (int axis = 0; axis < 3; ++axis)
        grad_point[point_idx * 3 + axis] = 0.f;
    const int edge_id = tape_edge_id[point_idx];
    if (edge_id < 0)
        return;

    const int i0 = edge_v0[edge_id];
    const int i1 = edge_v1[edge_id];
    const float3 p = make_f3(point + point_idx * 3);
    const float3 a = make_f3(vertices + i0 * 3);
    const float3 b = make_f3(vertices + i1 * 3);
    const float s = tape_s[point_idx];
    const float3 d = make_f3(tape_d + point_idx * 3);
    const float distance_bar = read_scalar_or_zero(grad_distance, point_idx, grad_distance_stride0);
    const float3 gep = read_vec3_or_zero(grad_edge_point, point_idx, grad_edge_point_stride0, grad_edge_point_stride1);
    const float edge_parameter_bar =
        read_scalar_or_zero(grad_edge_t, point_idx, grad_edge_t_stride0) +
        read_scalar_or_zero(grad_edge_t_alias, point_idx, grad_edge_t_alias_stride0);
    const shared::edge::PointSegmentVjp vjp =
        shared::edge::point_segment_vjp_fixed_winner(
            to_shared_vec3(p),
            to_shared_vec3(a),
            to_shared_vec3(b),
            s,
            to_shared_vec3(d),
            distance_bar,
            to_shared_vec3(gep),
            edge_parameter_bar);
    const float3 point_bar = from_shared_vec3(vjp.point);
    grad_point[point_idx * 3 + 0] = point_bar.x;
    grad_point[point_idx * 3 + 1] = point_bar.y;
    grad_point[point_idx * 3 + 2] = point_bar.z;
    atomic_add3(grad_vertices, i0, from_shared_vec3(vjp.edge_start));
    atomic_add3(grad_vertices, i1, from_shared_vec3(vjp.edge_end));
}

__global__ void edge_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const float *__restrict__ point,
    const int *__restrict__ tape_edge_id,
    const float *__restrict__ tape_s,
    const float *__restrict__ tape_d,
    const float *__restrict__ tangent_vertices,
    const float *__restrict__ tangent_point,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    int64_t tangent_point_stride0,
    int64_t tangent_point_stride1,
    int64_t point_count,
    float *__restrict__ tangent_distance,
    float *__restrict__ tangent_edge_point,
    float *__restrict__ tangent_edge_t,
    float *__restrict__ tangent_tape_s,
    float *__restrict__ tangent_tape_d) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= point_count)
        return;
    tangent_distance[point_idx] = 0.f;
    tangent_edge_t[point_idx] = 0.f;
    tangent_tape_s[point_idx] = 0.f;
    for (int axis = 0; axis < 3; ++axis) {
        tangent_edge_point[point_idx * 3 + axis] = 0.f;
        tangent_tape_d[point_idx * 3 + axis] = 0.f;
    }
    const int edge_id = tape_edge_id[point_idx];
    if (edge_id < 0)
        return;

    const int i0 = edge_v0[edge_id];
    const int i1 = edge_v1[edge_id];
    const float3 a = make_f3(vertices + i0 * 3);
    const float3 b = make_f3(vertices + i1 * 3);
    const float3 p = make_f3(point + point_idx * 3);
    const float3 da = read_vec3_or_zero(tangent_vertices, i0, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 db = read_vec3_or_zero(tangent_vertices, i1, tangent_vertices_stride0, tangent_vertices_stride1);
    const float3 dp = read_vec3_or_zero(tangent_point, point_idx, tangent_point_stride0, tangent_point_stride1);
    const float s = tape_s[point_idx];
    const float3 d = make_f3(tape_d + point_idx * 3);
    const shared::edge::PointSegmentJvp jvp =
        shared::edge::point_segment_jvp_fixed_winner(
            to_shared_vec3(p),
            to_shared_vec3(a),
            to_shared_vec3(b),
            s,
            to_shared_vec3(d),
            to_shared_vec3(dp),
            to_shared_vec3(da),
            to_shared_vec3(db));
    const float3 dep = from_shared_vec3(jvp.edge_point);

    tangent_distance[point_idx] = jvp.distance;
    tangent_edge_t[point_idx] = jvp.edge_parameter;
    tangent_edge_point[point_idx * 3 + 0] = dep.x;
    tangent_edge_point[point_idx * 3 + 1] = dep.y;
    tangent_edge_point[point_idx * 3 + 2] = dep.z;
}

} // namespace

EdgeBackwardOutputs edge_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor &grad_distance,
    const at::Tensor &grad_edge_point,
    const at::Tensor &grad_edge_t) {
    return edge_backward_optional_cuda(
        vertices,
        edge_v0,
        edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        &grad_distance,
        &grad_edge_point,
        &grad_edge_t,
        nullptr);
}

EdgeBackwardOutputs edge_backward_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor *grad_distance,
    const at::Tensor *grad_edge_point,
    const at::Tensor *grad_edge_t,
    const at::Tensor *grad_edge_t_alias) {
    const int64_t point_count = point.size(0);
    EdgeBackwardOutputs out;
    out.grad_vertices = at::empty_like(vertices);
    out.grad_point = at::empty_like(point);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.get_device()).stream();
    cudaMemsetAsync(out.grad_vertices.data_ptr<float>(), 0, static_cast<size_t>(out.grad_vertices.nbytes()), stream);
    if (point_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    edge_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        edge_v0.data_ptr<int>(),
        edge_v1.data_ptr<int>(),
        point.data_ptr<float>(),
        tape_edge_id.data_ptr<int>(),
        tape_s.data_ptr<float>(),
        tape_d.data_ptr<float>(),
        grad_distance == nullptr ? nullptr : grad_distance->data_ptr<float>(),
        grad_edge_point == nullptr ? nullptr : grad_edge_point->data_ptr<float>(),
        grad_edge_t == nullptr ? nullptr : grad_edge_t->data_ptr<float>(),
        grad_edge_t_alias == nullptr ? nullptr : grad_edge_t_alias->data_ptr<float>(),
        optional_stride(grad_distance, 0),
        optional_stride(grad_edge_point, 0),
        optional_stride(grad_edge_point, 1),
        optional_stride(grad_edge_t, 0),
        optional_stride(grad_edge_t_alias, 0),
        point_count,
        out.grad_vertices.data_ptr<float>(),
        out.grad_point.data_ptr<float>());
    return out;
}

EdgeJvpOutputs edge_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_point) {
    return edge_jvp_optional_cuda(
        vertices,
        edge_v0,
        edge_v1,
        point,
        tape_edge_id,
        tape_s,
        tape_d,
        &tangent_vertices,
        &tangent_point);
}

EdgeJvpOutputs edge_jvp_optional_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_point) {
    const int64_t point_count = point.size(0);
    EdgeJvpOutputs out;
    out.tangent_distance = at::empty({point_count}, point.options());
    out.tangent_edge_point = at::empty_like(point);
    out.tangent_edge_t = at::empty({point_count}, point.options());
    out.tangent_tape_s = at::empty_like(tape_s);
    out.tangent_tape_d = at::empty_like(tape_d);
    if (point_count == 0)
        return out;

    const int threads = 128;
    const int blocks = static_cast<int>((point_count + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(point.get_device()).stream();
    edge_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        edge_v0.data_ptr<int>(),
        edge_v1.data_ptr<int>(),
        point.data_ptr<float>(),
        tape_edge_id.data_ptr<int>(),
        tape_s.data_ptr<float>(),
        tape_d.data_ptr<float>(),
        tangent_vertices == nullptr ? nullptr : tangent_vertices->data_ptr<float>(),
        tangent_point == nullptr ? nullptr : tangent_point->data_ptr<float>(),
        optional_stride(tangent_vertices, 0),
        optional_stride(tangent_vertices, 1),
        optional_stride(tangent_point, 0),
        optional_stride(tangent_point, 1),
        point_count,
        out.tangent_distance.data_ptr<float>(),
        out.tangent_edge_point.data_ptr<float>(),
        out.tangent_edge_t.data_ptr<float>(),
        out.tangent_tape_s.data_ptr<float>(),
        out.tangent_tape_d.data_ptr<float>());
    return out;
}

} // namespace rayd::torch_backend
