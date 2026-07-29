#include <rayd/detail/scene/packing.h>

#include <cuda_runtime.h>

namespace rayd::shared::scene {
namespace {

constexpr int PackingBlockSize = 256;

static_assert(sizeof(PackedFloat4) == sizeof(float4));
static_assert(alignof(PackedFloat4) == alignof(float4));

__global__ void pack_global_geometry_kernel(GlobalGeometryPackingParams params) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < params.vertex_count) {
        const int source = index * 3;
        const int destination = (params.vertex_offset + index) * 3;
        params.global_vertices[destination + 0] = params.mesh_vertices[source + 0];
        params.global_vertices[destination + 1] = params.mesh_vertices[source + 1];
        params.global_vertices[destination + 2] = params.mesh_vertices[source + 2];
    }
    if (index < params.face_count) {
        const int source = index * 3;
        const int destination_face = params.face_offset + index;
        const int destination = destination_face * 3;
        params.global_faces[destination + 0] =
            params.mesh_faces[source + 0] + params.vertex_offset;
        params.global_faces[destination + 1] =
            params.mesh_faces[source + 1] + params.vertex_offset;
        params.global_faces[destination + 2] =
            params.mesh_faces[source + 2] + params.vertex_offset;
        params.face_shape_id[destination_face] = params.shape_id;
        params.face_local_id[destination_face] = index;
    }
}

__global__ void pack_global_vertex_tangent_kernel(
    GlobalVertexTangentPackingParams params) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.vertex_count) {
        return;
    }
    const int source = index * 3;
    const int destination = (params.vertex_offset + index) * 3;
    params.global_tangent[destination + 0] = params.mesh_tangent[source + 0];
    params.global_tangent[destination + 1] = params.mesh_tangent[source + 1];
    params.global_tangent[destination + 2] = params.mesh_tangent[source + 2];
}

__global__ void zero_global_vertex_tangent_range_kernel(
    GlobalVertexTangentZeroParams params) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.vertex_count) {
        return;
    }
    const int destination = (params.vertex_offset + index) * 3;
    params.global_tangent[destination + 0] = 0.0f;
    params.global_tangent[destination + 1] = 0.0f;
    params.global_tangent[destination + 2] = 0.0f;
}

int packing_block_count(int count) {
    return (count + PackingBlockSize - 1) / PackingBlockSize;
}

} // namespace

cudaError_t launch_pack_global_geometry_async(
    const GlobalGeometryPackingParams &params) noexcept {
    const int launch_count =
        params.vertex_count > params.face_count ? params.vertex_count : params.face_count;
    if (launch_count <= 0) {
        return cudaSuccess;
    }
    pack_global_geometry_kernel<<<packing_block_count(launch_count),
                                  PackingBlockSize,
                                  0,
                                  params.stream>>>(params);
    return cudaGetLastError();
}

cudaError_t launch_pack_global_vertex_tangent_async(
    const GlobalVertexTangentPackingParams &params) noexcept {
    if (params.vertex_count <= 0) {
        return cudaSuccess;
    }
    pack_global_vertex_tangent_kernel<<<packing_block_count(params.vertex_count),
                                        PackingBlockSize,
                                        0,
                                        params.stream>>>(params);
    return cudaGetLastError();
}

cudaError_t launch_zero_global_vertex_tangent_range_async(
    const GlobalVertexTangentZeroParams &params) noexcept {
    if (params.vertex_count <= 0) {
        return cudaSuccess;
    }
    zero_global_vertex_tangent_range_kernel<<<packing_block_count(params.vertex_count),
                                              PackingBlockSize,
                                              0,
                                              params.stream>>>(params);
    return cudaGetLastError();
}

} // namespace rayd::shared::scene
