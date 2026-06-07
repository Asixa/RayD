#include <raydtorch/scene/cache_kernels.h>
#include <raydtorch/common/math.cuh>
#include <raydtorch/common/optix_context.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
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

__global__ void compute_edge_search_stats_kernel(
    int edge_count,
    const float *__restrict__ p0_x,
    const float *__restrict__ p0_y,
    const float *__restrict__ p0_z,
    const float *__restrict__ e1_x,
    const float *__restrict__ e1_y,
    const float *__restrict__ e1_z,
    float *__restrict__ partials) {
    extern __shared__ float shared[];
    float *min_x = shared;
    float *min_y = min_x + blockDim.x;
    float *min_z = min_y + blockDim.x;
    float *max_x = min_z + blockDim.x;
    float *max_y = max_x + blockDim.x;
    float *max_z = max_y + blockDim.x;
    float *max_len = max_z + blockDim.x;

    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    float local_min_x = FLT_MAX;
    float local_min_y = FLT_MAX;
    float local_min_z = FLT_MAX;
    float local_max_x = -FLT_MAX;
    float local_max_y = -FLT_MAX;
    float local_max_z = -FLT_MAX;
    float local_max_len = 0.0f;
    if (edge < edge_count) {
        const float x0 = p0_x[edge];
        const float y0 = p0_y[edge];
        const float z0 = p0_z[edge];
        const float ex = e1_x[edge];
        const float ey = e1_y[edge];
        const float ez = e1_z[edge];
        const float x1 = x0 + ex;
        const float y1 = y0 + ey;
        const float z1 = z0 + ez;
        local_min_x = fminf(x0, x1);
        local_min_y = fminf(y0, y1);
        local_min_z = fminf(z0, z1);
        local_max_x = fmaxf(x0, x1);
        local_max_y = fmaxf(y0, y1);
        local_max_z = fmaxf(z0, z1);
        local_max_len = sqrtf(ex * ex + ey * ey + ez * ez);
    }

    const int lane = threadIdx.x;
    min_x[lane] = local_min_x;
    min_y[lane] = local_min_y;
    min_z[lane] = local_min_z;
    max_x[lane] = local_max_x;
    max_y[lane] = local_max_y;
    max_z[lane] = local_max_z;
    max_len[lane] = local_max_len;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            min_x[lane] = fminf(min_x[lane], min_x[lane + stride]);
            min_y[lane] = fminf(min_y[lane], min_y[lane + stride]);
            min_z[lane] = fminf(min_z[lane], min_z[lane + stride]);
            max_x[lane] = fmaxf(max_x[lane], max_x[lane + stride]);
            max_y[lane] = fmaxf(max_y[lane], max_y[lane + stride]);
            max_z[lane] = fmaxf(max_z[lane], max_z[lane + stride]);
            max_len[lane] = fmaxf(max_len[lane], max_len[lane + stride]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        float *out = partials + static_cast<int64_t>(blockIdx.x) * 7;
        out[0] = min_x[0];
        out[1] = min_y[0];
        out[2] = min_z[0];
        out[3] = max_x[0];
        out[4] = max_y[0];
        out[5] = max_z[0];
        out[6] = max_len[0];
    }
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

EdgeSearchStats compute_edge_search_stats_cuda(
    int64_t edge_count,
    const at::Tensor &edge_p0_x,
    const at::Tensor &edge_p0_y,
    const at::Tensor &edge_p0_z,
    const at::Tensor &edge_e1_x,
    const at::Tensor &edge_e1_y,
    const at::Tensor &edge_e1_z) {
    launch_require_count(edge_count, "compute_edge_search_stats_cuda()");
    EdgeSearchStats stats;
    if (edge_count == 0) {
        return stats;
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((edge_count + block_size - 1) / block_size);
    at::Tensor partials = at::empty({block_count, 7}, edge_p0_x.options());
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    compute_edge_search_stats_kernel<<<
        block_count,
        block_size,
        sizeof(float) * block_size * 7,
        torch_ctx.stream>>>(
        static_cast<int>(edge_count),
        edge_p0_x.data_ptr<float>(),
        edge_p0_y.data_ptr<float>(),
        edge_p0_z.data_ptr<float>(),
        edge_e1_x.data_ptr<float>(),
        edge_e1_y.data_ptr<float>(),
        edge_e1_z.data_ptr<float>(),
        partials.data_ptr<float>());

    at::Tensor partials_cpu = partials.cpu();
    const float *values = partials_cpu.data_ptr<float>();
    stats.has_edges = true;
    stats.min_x = std::numeric_limits<float>::infinity();
    stats.min_y = std::numeric_limits<float>::infinity();
    stats.min_z = std::numeric_limits<float>::infinity();
    stats.max_x = -std::numeric_limits<float>::infinity();
    stats.max_y = -std::numeric_limits<float>::infinity();
    stats.max_z = -std::numeric_limits<float>::infinity();
    stats.max_edge_length = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const float *row = values + block * 7;
        stats.min_x = std::min(stats.min_x, row[0]);
        stats.min_y = std::min(stats.min_y, row[1]);
        stats.min_z = std::min(stats.min_z, row[2]);
        stats.max_x = std::max(stats.max_x, row[3]);
        stats.max_y = std::max(stats.max_y, row[4]);
        stats.max_z = std::max(stats.max_z, row[5]);
        stats.max_edge_length = std::max(stats.max_edge_length, row[6]);
    }
    return stats;
}

} // namespace raydtorch
