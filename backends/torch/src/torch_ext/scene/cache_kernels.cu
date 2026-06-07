#include <raydtorch/scene/cache_kernels.h>
#include <raydtorch/common/math.cuh>
#include <raydtorch/common/optix_context.h>

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace raydtorch {

namespace {

__global__ void compute_triangle_soa_kernel(
    int triangle_count,
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    float *__restrict__ p0_x,
    float *__restrict__ p0_y,
    float *__restrict__ p0_z,
    float *__restrict__ e1_x,
    float *__restrict__ e1_y,
    float *__restrict__ e1_z,
    float *__restrict__ e2_x,
    float *__restrict__ e2_y,
    float *__restrict__ e2_z,
    float *__restrict__ fn_x,
    float *__restrict__ fn_y,
    float *__restrict__ fn_z) {
    const int tri = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri >= triangle_count) {
        return;
    }

    const int i0 = faces[tri * 3 + 0];
    const int i1 = faces[tri * 3 + 1];
    const int i2 = faces[tri * 3 + 2];
    const float3 p0 = make_f3(vertices + i0 * 3);
    const float3 p1 = make_f3(vertices + i1 * 3);
    const float3 p2 = make_f3(vertices + i2 * 3);
    const float3 edge1 = sub3(p1, p0);
    const float3 edge2 = sub3(p2, p0);
    const float3 normal = cross3(edge1, edge2);

    p0_x[tri] = p0.x;
    p0_y[tri] = p0.y;
    p0_z[tri] = p0.z;
    e1_x[tri] = edge1.x;
    e1_y[tri] = edge1.y;
    e1_z[tri] = edge1.z;
    e2_x[tri] = edge2.x;
    e2_y[tri] = edge2.y;
    e2_z[tri] = edge2.z;
    fn_x[tri] = normal.x;
    fn_y[tri] = normal.y;
    fn_z[tri] = normal.z;
}

__global__ void compute_edge_soa_kernel(
    int edge_count,
    const float *__restrict__ vertices,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    float *__restrict__ p0_x,
    float *__restrict__ p0_y,
    float *__restrict__ p0_z,
    float *__restrict__ e1_x,
    float *__restrict__ e1_y,
    float *__restrict__ e1_z) {
    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= edge_count) {
        return;
    }

    const int i0 = edge_v0[edge];
    const int i1 = edge_v1[edge];
    const float3 p0 = make_f3(vertices + i0 * 3);
    const float3 p1 = make_f3(vertices + i1 * 3);
    const float3 edge1 = sub3(p1, p0);

    p0_x[edge] = p0.x;
    p0_y[edge] = p0.y;
    p0_z[edge] = p0.z;
    e1_x[edge] = edge1.x;
    e1_y[edge] = edge1.y;
    e1_z[edge] = edge1.z;
}

void launch_require_count(int64_t count, const char *name) {
    if (count < 0 || count > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + ": count is outside int32 launch range.");
    }
}

} // namespace

void compute_triangle_soa_cuda(
    int64_t triangle_count,
    const at::Tensor &vertices,
    const at::Tensor &faces,
    at::Tensor &tri_p0_x,
    at::Tensor &tri_p0_y,
    at::Tensor &tri_p0_z,
    at::Tensor &tri_e1_x,
    at::Tensor &tri_e1_y,
    at::Tensor &tri_e1_z,
    at::Tensor &tri_e2_x,
    at::Tensor &tri_e2_y,
    at::Tensor &tri_e2_z,
    at::Tensor &tri_fn_x,
    at::Tensor &tri_fn_y,
    at::Tensor &tri_fn_z) {
    launch_require_count(triangle_count, "compute_triangle_soa_cuda()");
    if (triangle_count == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((triangle_count + block_size - 1) / block_size);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    compute_triangle_soa_kernel<<<block_count, block_size, 0, torch_ctx.stream>>>(
        static_cast<int>(triangle_count),
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        tri_p0_x.data_ptr<float>(),
        tri_p0_y.data_ptr<float>(),
        tri_p0_z.data_ptr<float>(),
        tri_e1_x.data_ptr<float>(),
        tri_e1_y.data_ptr<float>(),
        tri_e1_z.data_ptr<float>(),
        tri_e2_x.data_ptr<float>(),
        tri_e2_y.data_ptr<float>(),
        tri_e2_z.data_ptr<float>(),
        tri_fn_x.data_ptr<float>(),
        tri_fn_y.data_ptr<float>(),
        tri_fn_z.data_ptr<float>());
}

void compute_edge_soa_cuda(
    int64_t edge_count,
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    at::Tensor &edge_p0_x,
    at::Tensor &edge_p0_y,
    at::Tensor &edge_p0_z,
    at::Tensor &edge_e1_x,
    at::Tensor &edge_e1_y,
    at::Tensor &edge_e1_z) {
    launch_require_count(edge_count, "compute_edge_soa_cuda()");
    if (edge_count == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((edge_count + block_size - 1) / block_size);
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    compute_edge_soa_kernel<<<block_count, block_size, 0, torch_ctx.stream>>>(
        static_cast<int>(edge_count),
        vertices.data_ptr<float>(),
        edge_v0.data_ptr<int>(),
        edge_v1.data_ptr<int>(),
        edge_p0_x.data_ptr<float>(),
        edge_p0_y.data_ptr<float>(),
        edge_p0_z.data_ptr<float>(),
        edge_e1_x.data_ptr<float>(),
        edge_e1_y.data_ptr<float>(),
        edge_e1_z.data_ptr<float>());
}

} // namespace raydtorch
