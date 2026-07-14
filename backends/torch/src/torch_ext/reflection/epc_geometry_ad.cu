// Fixed-winner geometry adjoint / tangent of the reflection EPC path export
// (direct-plane mode) and of the scene's unit face-normal table.
//
// The forward being differentiated is the specular chain the EPC discovery
// raygen solves for an already-selected plane sequence: mirror the source
// through each plane, walk back from the receiver intersecting each plane,
// sum the segment lengths. That chain lives in
// shared/include/rayd/shared/reflection/epc_chain.h together with its
// reverse-mode companion, so the math here has exactly one implementation.
// Which primitive each bounce hits, the containment test and the visibility
// casts are frozen discovery decisions: invalid rows contribute nothing and
// no ray is traced, so no OptiX is involved.

#include <rayd/torch/reflection/kernels.h>
#include <rayd/torch/common/math.cuh>
#include <rayd/shared/optix/reflection_epc_params.h>
#include <rayd/shared/reflection/epc_chain.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace rayd::torch_backend {

namespace {

namespace shared_math = rayd::shared::math;
namespace shared_reflection = rayd::shared::reflection;

using shared_math::Vec3f;
using shared::optix::ReflEpcMaxBounces;

__device__ Vec3f load_shared_vec3(const float *base, int64_t index) {
    return shared_math::make_vec3(
        base[index * 3 + 0], base[index * 3 + 1], base[index * 3 + 2]);
}

__device__ Vec3f load_strided_vec3_or_zero(
    const float *base,
    int64_t index0,
    int64_t index1,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2) {
    if (base == nullptr) {
        return shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    }
    const int64_t offset = index0 * stride0 + index1 * stride1;
    return shared_math::make_vec3(
        base[offset], base[offset + stride2], base[offset + 2 * stride2]);
}

__device__ void store_vec3(float *base, int64_t index, Vec3f value) {
    base[index * 3 + 0] = value.x;
    base[index * 3 + 1] = value.y;
    base[index * 3 + 2] = value.z;
}

__device__ void atomic_add_shared_vec3(float *base, int index, Vec3f value) {
    atomicAdd(&base[index * 3 + 0], value.x);
    atomicAdd(&base[index * 3 + 1], value.y);
    atomicAdd(&base[index * 3 + 2], value.z);
}

int64_t optional_stride_or_zero(const at::Tensor *tensor, int64_t dim) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0 ||
        tensor->dim() <= dim) {
        return 0;
    }
    return tensor->stride(dim);
}

const float *optional_data_ptr(const at::Tensor *tensor) {
    if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0) {
        return nullptr;
    }
    return tensor->data_ptr<float>();
}

void zero_float_tensor_async(const at::Tensor &tensor, cudaStream_t stream) {
    if (tensor.defined() && tensor.numel() > 0) {
        cudaMemsetAsync(
            tensor.data_ptr<float>(),
            0,
            static_cast<size_t>(tensor.numel()) * sizeof(float),
            stream);
    }
}

// Re-solve the frozen-winner chain for one ray and load its plane inputs.
// Returns false when the row is invalid or the chain guard rejects it (the
// row then contributes exactly zero, matching the frozen discovery record).
__device__ bool load_row_chain(
    const float *source,
    const float *receiver,
    const float *plane_points,
    const float *plane_normals,
    const bool *valid,
    const int *bounce_count,
    int64_t ray_index,
    int max_bounces,
    Vec3f *row_plane_points,
    Vec3f *row_plane_normals,
    shared_reflection::EpcChain<ReflEpcMaxBounces> &chain) {
    if (!valid[ray_index]) {
        return false;
    }
    const int bounces = bounce_count[ray_index];
    if (bounces < 1 || bounces > max_bounces || bounces > ReflEpcMaxBounces) {
        return false;
    }
    const int64_t base = ray_index * max_bounces;
    for (int bounce = 0; bounce < bounces; ++bounce) {
        row_plane_points[bounce] = load_shared_vec3(plane_points, base + bounce);
        row_plane_normals[bounce] = load_shared_vec3(plane_normals, base + bounce);
    }
    return shared_reflection::solve_epc_chain<ReflEpcMaxBounces>(
        row_plane_points,
        row_plane_normals,
        bounces,
        load_shared_vec3(source, ray_index),
        load_shared_vec3(receiver, ray_index),
        chain);
}

__global__ void reflection_epc_paths_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const int *__restrict__ sequence,
    const float *__restrict__ plane_points,
    const float *__restrict__ plane_normals,
    const bool *__restrict__ valid,
    const int *__restrict__ bounce_count,
    const float *__restrict__ grad_points,
    const float *__restrict__ grad_normals,
    const float *__restrict__ grad_path_length,
    int64_t grad_points_stride0,
    int64_t grad_points_stride1,
    int64_t grad_points_stride2,
    int64_t grad_normals_stride0,
    int64_t grad_normals_stride1,
    int64_t grad_normals_stride2,
    int64_t grad_path_length_stride0,
    int64_t ray_count,
    int max_bounces,
    int64_t triangle_count,
    float *__restrict__ grad_vertices,
    float *__restrict__ grad_source,
    float *__restrict__ grad_receiver) {
    const int64_t ray_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray_index >= ray_count) {
        return;
    }
    const Vec3f zero = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    if (grad_source != nullptr) {
        store_vec3(grad_source, ray_index, zero);
    }
    if (grad_receiver != nullptr) {
        store_vec3(grad_receiver, ray_index, zero);
    }

    Vec3f row_plane_points[ReflEpcMaxBounces];
    Vec3f row_plane_normals[ReflEpcMaxBounces];
    shared_reflection::EpcChain<ReflEpcMaxBounces> chain;
    if (!load_row_chain(
            source,
            receiver,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
            ray_index,
            max_bounces,
            row_plane_points,
            row_plane_normals,
            chain)) {
        return;
    }
    const int bounces = chain.bounces;

    Vec3f grad_hits[ReflEpcMaxBounces];
    Vec3f grad_unit_normals[ReflEpcMaxBounces];
    for (int bounce = 0; bounce < bounces; ++bounce) {
        grad_hits[bounce] = load_strided_vec3_or_zero(
            grad_points,
            ray_index,
            bounce,
            grad_points_stride0,
            grad_points_stride1,
            grad_points_stride2);
        grad_unit_normals[bounce] = load_strided_vec3_or_zero(
            grad_normals,
            ray_index,
            bounce,
            grad_normals_stride0,
            grad_normals_stride1,
            grad_normals_stride2);
    }
    const float grad_length =
        grad_path_length == nullptr
            ? 0.0f
            : grad_path_length[ray_index * grad_path_length_stride0];

    Vec3f grad_source_row;
    Vec3f grad_receiver_row;
    Vec3f grad_plane_points[ReflEpcMaxBounces];
    Vec3f grad_plane_normals[ReflEpcMaxBounces];
    shared_reflection::adj_solve_epc_chain<ReflEpcMaxBounces>(
        chain,
        row_plane_points,
        row_plane_normals,
        load_shared_vec3(source, ray_index),
        load_shared_vec3(receiver, ray_index),
        grad_hits,
        grad_unit_normals,
        grad_length,
        grad_source_row,
        grad_receiver_row,
        grad_plane_points,
        grad_plane_normals);

    if (grad_source != nullptr) {
        store_vec3(grad_source, ray_index, grad_source_row);
    }
    if (grad_receiver != nullptr) {
        store_vec3(grad_receiver, ray_index, grad_receiver_row);
    }
    if (grad_vertices == nullptr) {
        return;
    }

    // Chain each bounce's plane cotangents to the winner triangle: the anchor
    // is v0(prim) and the plane normal is the unit face normal, exactly how
    // the consumer builds the direct-plane arrays from the scene export.
    const int64_t base = ray_index * max_bounces;
    for (int bounce = 0; bounce < bounces; ++bounce) {
        const int prim = sequence[base + bounce];
        if (prim < 0 || prim >= triangle_count) {
            continue;
        }
        const int i0 = faces[prim * 3 + 0];
        const int i1 = faces[prim * 3 + 1];
        const int i2 = faces[prim * 3 + 2];
        const Vec3f v0 = load_shared_vec3(vertices, i0);
        const Vec3f v1 = load_shared_vec3(vertices, i1);
        const Vec3f v2 = load_shared_vec3(vertices, i2);
        Vec3f grad_v0 = grad_plane_points[bounce];
        Vec3f grad_v1 = zero;
        Vec3f grad_v2 = zero;
        shared_reflection::adj_face_normal(
            v0,
            v1,
            v2,
            shared_reflection::face_unit_normal(v0, v1, v2),
            grad_plane_normals[bounce],
            grad_v0,
            grad_v1,
            grad_v2);
        atomic_add_shared_vec3(grad_vertices, i0, grad_v0);
        atomic_add_shared_vec3(grad_vertices, i1, grad_v1);
        atomic_add_shared_vec3(grad_vertices, i2, grad_v2);
    }
}

__global__ void reflection_epc_paths_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ source,
    const float *__restrict__ receiver,
    const int *__restrict__ sequence,
    const float *__restrict__ plane_points,
    const float *__restrict__ plane_normals,
    const bool *__restrict__ valid,
    const int *__restrict__ bounce_count,
    const float *__restrict__ tangent_vertices,
    const float *__restrict__ tangent_source,
    const float *__restrict__ tangent_receiver,
    int64_t tangent_vertices_stride0,
    int64_t tangent_vertices_stride1,
    int64_t tangent_source_stride0,
    int64_t tangent_source_stride1,
    int64_t tangent_receiver_stride0,
    int64_t tangent_receiver_stride1,
    int64_t ray_count,
    int max_bounces,
    int64_t triangle_count,
    float *__restrict__ tangent_points,
    float *__restrict__ tangent_normals,
    float *__restrict__ tangent_path_length) {
    const int64_t ray_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray_index >= ray_count) {
        return;
    }
    const Vec3f zero = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    const int64_t base = ray_index * max_bounces;
    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        store_vec3(tangent_points, base + bounce, zero);
        store_vec3(tangent_normals, base + bounce, zero);
    }
    tangent_path_length[ray_index] = 0.0f;

    Vec3f row_plane_points[ReflEpcMaxBounces];
    Vec3f row_plane_normals[ReflEpcMaxBounces];
    shared_reflection::EpcChain<ReflEpcMaxBounces> chain;
    if (!load_row_chain(
            source,
            receiver,
            plane_points,
            plane_normals,
            valid,
            bounce_count,
            ray_index,
            max_bounces,
            row_plane_points,
            row_plane_normals,
            chain)) {
        return;
    }
    const int bounces = chain.bounces;

    // Tangent of each plane under vertex tangents: anchor tangent is the
    // winner triangle's v0 tangent, normal tangent is the unit face-normal
    // tangent (the transpose of the vertex chaining in the backward kernel).
    Vec3f tangent_plane_points[ReflEpcMaxBounces];
    Vec3f tangent_plane_normals[ReflEpcMaxBounces];
    for (int bounce = 0; bounce < bounces; ++bounce) {
        tangent_plane_points[bounce] = zero;
        tangent_plane_normals[bounce] = zero;
        if (tangent_vertices == nullptr) {
            continue;
        }
        const int prim = sequence[base + bounce];
        if (prim < 0 || prim >= triangle_count) {
            continue;
        }
        const int i0 = faces[prim * 3 + 0];
        const int i1 = faces[prim * 3 + 1];
        const int i2 = faces[prim * 3 + 2];
        const Vec3f tangent_v0 = load_strided_vec3_or_zero(
            tangent_vertices, i0, 0, tangent_vertices_stride0, 0,
            tangent_vertices_stride1);
        const Vec3f tangent_v1 = load_strided_vec3_or_zero(
            tangent_vertices, i1, 0, tangent_vertices_stride0, 0,
            tangent_vertices_stride1);
        const Vec3f tangent_v2 = load_strided_vec3_or_zero(
            tangent_vertices, i2, 0, tangent_vertices_stride0, 0,
            tangent_vertices_stride1);
        tangent_plane_points[bounce] = tangent_v0;
        tangent_plane_normals[bounce] = shared_reflection::jvp_face_normal(
            load_shared_vec3(vertices, i0),
            load_shared_vec3(vertices, i1),
            load_shared_vec3(vertices, i2),
            tangent_v0,
            tangent_v1,
            tangent_v2);
    }

    Vec3f tangent_hits[ReflEpcMaxBounces];
    Vec3f tangent_unit_normals[ReflEpcMaxBounces];
    float tangent_length = 0.0f;
    shared_reflection::jvp_solve_epc_chain<ReflEpcMaxBounces>(
        chain,
        row_plane_points,
        row_plane_normals,
        load_shared_vec3(source, ray_index),
        load_shared_vec3(receiver, ray_index),
        load_strided_vec3_or_zero(
            tangent_source, ray_index, 0, tangent_source_stride0, 0,
            tangent_source_stride1),
        load_strided_vec3_or_zero(
            tangent_receiver, ray_index, 0, tangent_receiver_stride0, 0,
            tangent_receiver_stride1),
        tangent_plane_points,
        tangent_plane_normals,
        tangent_hits,
        tangent_unit_normals,
        tangent_length);

    for (int bounce = 0; bounce < bounces; ++bounce) {
        store_vec3(tangent_points, base + bounce, tangent_hits[bounce]);
        store_vec3(tangent_normals, base + bounce, tangent_unit_normals[bounce]);
    }
    tangent_path_length[ray_index] = tangent_length;
}

__global__ void scene_face_normals_backward_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ grad_face_normals,
    int64_t grad_stride0,
    int64_t grad_stride1,
    int64_t triangle_count,
    float *__restrict__ grad_vertices) {
    const int64_t face_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (face_index >= triangle_count) {
        return;
    }
    const Vec3f grad_normal = load_strided_vec3_or_zero(
        grad_face_normals, face_index, 0, grad_stride0, 0, grad_stride1);
    if (grad_normal.x == 0.0f && grad_normal.y == 0.0f && grad_normal.z == 0.0f) {
        return;
    }
    const int i0 = faces[face_index * 3 + 0];
    const int i1 = faces[face_index * 3 + 1];
    const int i2 = faces[face_index * 3 + 2];
    const Vec3f v0 = load_shared_vec3(vertices, i0);
    const Vec3f v1 = load_shared_vec3(vertices, i1);
    const Vec3f v2 = load_shared_vec3(vertices, i2);
    Vec3f grad_v0 = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    Vec3f grad_v1 = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    Vec3f grad_v2 = shared_math::make_vec3(0.0f, 0.0f, 0.0f);
    shared_reflection::adj_face_normal(
        v0,
        v1,
        v2,
        shared_reflection::face_unit_normal(v0, v1, v2),
        grad_normal,
        grad_v0,
        grad_v1,
        grad_v2);
    atomic_add_shared_vec3(grad_vertices, i0, grad_v0);
    atomic_add_shared_vec3(grad_vertices, i1, grad_v1);
    atomic_add_shared_vec3(grad_vertices, i2, grad_v2);
}

__global__ void scene_face_normals_jvp_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ tangent_vertices,
    int64_t tangent_stride0,
    int64_t tangent_stride1,
    int64_t triangle_count,
    float *__restrict__ tangent_face_normals) {
    const int64_t face_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (face_index >= triangle_count) {
        return;
    }
    const int i0 = faces[face_index * 3 + 0];
    const int i1 = faces[face_index * 3 + 1];
    const int i2 = faces[face_index * 3 + 2];
    const Vec3f tangent = shared_reflection::jvp_face_normal(
        load_shared_vec3(vertices, i0),
        load_shared_vec3(vertices, i1),
        load_shared_vec3(vertices, i2),
        load_strided_vec3_or_zero(
            tangent_vertices, i0, 0, tangent_stride0, 0, tangent_stride1),
        load_strided_vec3_or_zero(
            tangent_vertices, i1, 0, tangent_stride0, 0, tangent_stride1),
        load_strided_vec3_or_zero(
            tangent_vertices, i2, 0, tangent_stride0, 0, tangent_stride1));
    store_vec3(tangent_face_normals, face_index, tangent);
}

} // namespace

ReflEpcPathsBackwardOutputs reflection_epc_paths_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &sequence,
    const at::Tensor &plane_points,
    const at::Tensor &plane_normals,
    const at::Tensor &valid,
    const at::Tensor &bounce_count,
    const at::Tensor *grad_points,
    const at::Tensor *grad_normals,
    const at::Tensor *grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver) {
    const int64_t ray_count = source.size(0);
    const int max_bounces = static_cast<int>(sequence.size(1));
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    ReflEpcPathsBackwardOutputs out;
    out.grad_vertices = need_grad_vertices
        ? at::empty(vertices.sizes(), vertices.options())
        : at::Tensor();
    out.grad_source =
        need_grad_source ? at::empty(source.sizes(), source.options()) : at::Tensor();
    out.grad_receiver = need_grad_receiver
        ? at::empty(receiver.sizes(), receiver.options())
        : at::Tensor();
    zero_float_tensor_async(out.grad_vertices, stream);
    if (ray_count == 0 ||
        (!need_grad_vertices && !need_grad_source && !need_grad_receiver)) {
        return out;
    }

    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_epc_paths_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        source.data_ptr<float>(),
        receiver.data_ptr<float>(),
        sequence.data_ptr<int>(),
        plane_points.data_ptr<float>(),
        plane_normals.data_ptr<float>(),
        valid.data_ptr<bool>(),
        bounce_count.data_ptr<int>(),
        optional_data_ptr(grad_points),
        optional_data_ptr(grad_normals),
        optional_data_ptr(grad_path_length),
        optional_stride_or_zero(grad_points, 0),
        optional_stride_or_zero(grad_points, 1),
        optional_stride_or_zero(grad_points, 2),
        optional_stride_or_zero(grad_normals, 0),
        optional_stride_or_zero(grad_normals, 1),
        optional_stride_or_zero(grad_normals, 2),
        optional_stride_or_zero(grad_path_length, 0),
        ray_count,
        max_bounces,
        faces.size(0),
        need_grad_vertices ? out.grad_vertices.data_ptr<float>() : nullptr,
        need_grad_source ? out.grad_source.data_ptr<float>() : nullptr,
        need_grad_receiver ? out.grad_receiver.data_ptr<float>() : nullptr);
    return out;
}

ReflEpcPathsJvpOutputs reflection_epc_paths_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &sequence,
    const at::Tensor &plane_points,
    const at::Tensor &plane_normals,
    const at::Tensor &valid,
    const at::Tensor &bounce_count,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_source,
    const at::Tensor *tangent_receiver) {
    const int64_t ray_count = source.size(0);
    const int64_t max_bounces = sequence.size(1);
    ReflEpcPathsJvpOutputs out;
    out.tangent_points =
        at::empty({ray_count, max_bounces, 3}, source.options());
    out.tangent_normals =
        at::empty({ray_count, max_bounces, 3}, source.options());
    out.tangent_path_length = at::empty({ray_count}, source.options());
    if (ray_count == 0) {
        return out;
    }

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((ray_count + threads - 1) / threads);
    reflection_epc_paths_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        source.data_ptr<float>(),
        receiver.data_ptr<float>(),
        sequence.data_ptr<int>(),
        plane_points.data_ptr<float>(),
        plane_normals.data_ptr<float>(),
        valid.data_ptr<bool>(),
        bounce_count.data_ptr<int>(),
        optional_data_ptr(tangent_vertices),
        optional_data_ptr(tangent_source),
        optional_data_ptr(tangent_receiver),
        optional_stride_or_zero(tangent_vertices, 0),
        optional_stride_or_zero(tangent_vertices, 1),
        optional_stride_or_zero(tangent_source, 0),
        optional_stride_or_zero(tangent_source, 1),
        optional_stride_or_zero(tangent_receiver, 0),
        optional_stride_or_zero(tangent_receiver, 1),
        ray_count,
        static_cast<int>(max_bounces),
        faces.size(0),
        out.tangent_points.data_ptr<float>(),
        out.tangent_normals.data_ptr<float>(),
        out.tangent_path_length.data_ptr<float>());
    return out;
}

at::Tensor scene_face_normals_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &grad_face_normals) {
    const int64_t triangle_count = faces.size(0);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    at::Tensor grad_vertices = at::empty(vertices.sizes(), vertices.options());
    zero_float_tensor_async(grad_vertices, stream);
    if (triangle_count == 0) {
        return grad_vertices;
    }

    const int threads = 128;
    const int blocks = static_cast<int>((triangle_count + threads - 1) / threads);
    scene_face_normals_backward_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        grad_face_normals.data_ptr<float>(),
        grad_face_normals.stride(0),
        grad_face_normals.stride(1),
        triangle_count,
        grad_vertices.data_ptr<float>());
    return grad_vertices;
}

at::Tensor scene_face_normals_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &tangent_vertices) {
    const int64_t triangle_count = faces.size(0);
    at::Tensor tangent_face_normals =
        at::empty({triangle_count, 3}, vertices.options());
    if (triangle_count == 0) {
        return tangent_face_normals;
    }

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int threads = 128;
    const int blocks = static_cast<int>((triangle_count + threads - 1) / threads);
    scene_face_normals_jvp_kernel<<<blocks, threads, 0, stream>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        tangent_vertices.data_ptr<float>(),
        tangent_vertices.stride(0),
        tangent_vertices.stride(1),
        triangle_count,
        tangent_face_normals.data_ptr<float>());
    return tangent_face_normals;
}

} // namespace rayd::torch_backend
