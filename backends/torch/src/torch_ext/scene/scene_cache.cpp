#include <rayd/torch/scene/cache.h>
#include <rayd/torch/common/optix_context.h>
#include <rayd/torch/common/tensor_check.h>
#include <rayd/torch/edge/bvh.h>
#include <rayd/torch/scene/cache_kernels.h>
#include <rayd/torch/scene/triangle_bvh.h>
#include <rayd/shared/bvh/build.h>
#include <rayd/shared/bvh/host_topology.h>
#include <rayd/shared/edge/bvh_build.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rayd::torch_backend {

namespace {
std::atomic<int64_t> next_handle{1};
std::mutex scenes_mutex;
std::unordered_map<int64_t, std::unique_ptr<SceneCache>> scenes;

void cuda_check(cudaError_t result, const char *expr) {
    if (result == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(result));
}

void require_optional_matrix4(const at::Tensor &tensor, std::string_view name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 2, name);
    require_last_dim(tensor, 4, name);
    if (tensor.size(0) != 0 && tensor.size(0) != 4)
        throw std::runtime_error(std::string(name) + " must be empty or have shape (4, 4).");
}

void compact_accel_if_smaller(
    OptixDeviceContext optix_context,
    cudaStream_t stream,
    at::TensorOptions byte_options,
    at::Tensor &gas_buffer,
    OptixTraversableHandle &traversable,
    const at::Tensor &compacted_size_buffer,
    const char *name) {
    uint64_t compacted_size = 0;
    cuda_check(
        cudaMemcpyAsync(
            &compacted_size,
            compacted_size_buffer.data_ptr<uint8_t>(),
            sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync(compacted GAS size)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(compacted GAS size)");
    if (compacted_size == 0 || compacted_size >= static_cast<uint64_t>(gas_buffer.numel()))
        return;
    if (compacted_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(std::string(name) + ": compacted GAS size exceeds int64 range.");
    }

    at::Tensor source_buffer = gas_buffer;
    at::Tensor compacted_buffer =
        at::empty({static_cast<int64_t>(compacted_size)}, byte_options);
    OptixTraversableHandle compacted_traversable = 0;
    rayd_torch_OPTIX_CHECK(optixAccelCompact(
        optix_context,
        stream,
        traversable,
        reinterpret_cast<CUdeviceptr>(compacted_buffer.data_ptr<uint8_t>()),
        static_cast<size_t>(compacted_size),
        &compacted_traversable));
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(GAS compaction)");
    gas_buffer = compacted_buffer;
    traversable = compacted_traversable;
}

OptixTriangleAccel build_triangle_accel(
    const MeshRecord &mesh,
    OptixDeviceContext optix_context,
    cudaStream_t stream) {
    OptixTriangleAccel accel;
    accel.vertex_buffer = mesh.vertices.contiguous();
    accel.index_buffer = mesh.faces.contiguous();

    CUdeviceptr vertex_buffer =
        reinterpret_cast<CUdeviceptr>(accel.vertex_buffer.data_ptr<float>());
    CUdeviceptr index_buffer =
        reinterpret_cast<CUdeviceptr>(accel.index_buffer.data_ptr<int>());
    uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &vertex_buffer;
    build_input.triangleArray.numVertices =
        static_cast<unsigned int>(accel.vertex_buffer.size(0));
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer = index_buffer;
    build_input.triangleArray.numIndexTriplets =
        static_cast<unsigned int>(accel.index_buffer.size(0));
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.flags = &triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (mesh.dynamic)
        accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    else
        accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes = {};
    rayd_torch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    at::TensorOptions byte_options =
        at::TensorOptions().device(mesh.vertices.device()).dtype(at::kByte);
    accel.gas_temp_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
    accel.gas_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);
    at::Tensor compacted_size_buffer;
    OptixAccelEmitDesc compacted_size_emit = {};
    OptixAccelEmitDesc *emit_descs = nullptr;
    unsigned int emit_desc_count = 0;
    if (!mesh.dynamic) {
        compacted_size_buffer = at::empty({static_cast<int64_t>(sizeof(uint64_t))}, byte_options);
        compacted_size_emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        compacted_size_emit.result =
            reinterpret_cast<CUdeviceptr>(compacted_size_buffer.data_ptr<uint8_t>());
        emit_descs = &compacted_size_emit;
        emit_desc_count = 1;
    }

    rayd_torch_OPTIX_CHECK(optixAccelBuild(
        optix_context,
        stream,
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(accel.gas_temp_buffer.data_ptr<uint8_t>()),
        buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(accel.gas_buffer.data_ptr<uint8_t>()),
        buffer_sizes.outputSizeInBytes,
        &accel.traversable,
        emit_descs,
        emit_desc_count));
    if (!mesh.dynamic) {
        compact_accel_if_smaller(
            optix_context,
            stream,
            byte_options,
            accel.gas_buffer,
            accel.traversable,
            compacted_size_buffer,
            "build_triangle_accel()");
    }

    return accel;
}

void write_identity_instance(OptixInstance &instance, unsigned int instance_id, OptixTraversableHandle traversable) {
    std::memset(&instance, 0, sizeof(instance));
    instance.transform[0] = 1.0f;
    instance.transform[5] = 1.0f;
    instance.transform[10] = 1.0f;
    instance.instanceId = instance_id;
    instance.sbtOffset = 0;
    instance.visibilityMask = 255u;
    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = traversable;
}

void build_triangle_ias(SceneCache &scene, OptixDeviceContext optix_context, cudaStream_t stream) {
    if (scene.triangle_accels.empty())
        throw std::runtime_error("build_triangle_ias(): missing triangle acceleration structures.");

    std::vector<OptixInstance> instances(scene.triangle_accels.size());
    for (size_t mesh_index = 0; mesh_index < scene.triangle_accels.size(); ++mesh_index) {
        write_identity_instance(
            instances[mesh_index],
            static_cast<unsigned int>(mesh_index),
            scene.triangle_accels[mesh_index].traversable);
    }

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, scene.device_index)).dtype(at::kByte);
    scene.triangle_ias.instance_buffer =
        at::empty({static_cast<int64_t>(sizeof(OptixInstance) * instances.size())}, byte_options);
    cuda_check(
        cudaMemcpyAsync(
            scene.triangle_ias.instance_buffer.data_ptr<uint8_t>(),
            instances.data(),
            sizeof(OptixInstance) * instances.size(),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(triangle IAS instances)");

    CUdeviceptr instance_buffer =
        reinterpret_cast<CUdeviceptr>(scene.triangle_ias.instance_buffer.data_ptr<uint8_t>());
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = instance_buffer;
    build_input.instanceArray.numInstances = static_cast<unsigned int>(instances.size());
    build_input.instanceArray.instanceStride = sizeof(OptixInstance);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes = {};
    rayd_torch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    scene.triangle_ias.ias_temp_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
    scene.triangle_ias.ias_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);

    rayd_torch_OPTIX_CHECK(optixAccelBuild(
        optix_context,
        stream,
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(scene.triangle_ias.ias_temp_buffer.data_ptr<uint8_t>()),
        buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(scene.triangle_ias.ias_buffer.data_ptr<uint8_t>()),
        buffer_sizes.outputSizeInBytes,
        &scene.triangle_ias.traversable,
        nullptr,
        0));
}

void refresh_global_geometry(SceneCache &scene) {
    int64_t vertex_offset = 0;
    int64_t face_offset = 0;
    std::vector<int32_t> face_offsets;
    face_offsets.reserve(scene.meshes.size());
    for (size_t mesh_id = 0; mesh_id < scene.meshes.size(); ++mesh_id) {
        const MeshRecord &mesh = scene.meshes[mesh_id];
        if (vertex_offset > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
            face_offset > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("Scene.build(): geometry exceeds int32 indexing limits.");
        }
        face_offsets.push_back(static_cast<int32_t>(face_offset));
        vertex_offset += mesh.vertices.size(0);
        face_offset += mesh.faces.size(0);
    }
    if (vertex_offset > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
        face_offset > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Scene.build(): geometry exceeds int32 indexing limits.");
    }

    at::TensorOptions fopts = scene.meshes[0].vertices.options();
    at::TensorOptions iopts = scene.meshes[0].faces.options();
    scene.global_vertices = at::empty({vertex_offset, 3}, fopts);
    scene.global_faces = at::empty({face_offset, 3}, iopts);
    scene.face_shape_id = at::empty({face_offset}, iopts);
    scene.face_local_id = at::empty({face_offset}, iopts);
    scene.primitive_identity = at::arange(face_offset, iopts);
    scene.face_offsets = at::empty({static_cast<int64_t>(face_offsets.size())}, iopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    cuda_check(
        cudaMemcpyAsync(
            scene.face_offsets.data_ptr<int>(),
            face_offsets.data(),
            sizeof(int32_t) * face_offsets.size(),
            cudaMemcpyHostToDevice,
            torch_ctx.stream),
        "cudaMemcpyAsync(face offsets)");

    vertex_offset = 0;
    face_offset = 0;
    for (int32_t mesh_id = 0; mesh_id < static_cast<int32_t>(scene.meshes.size()); ++mesh_id) {
        const MeshRecord &mesh = scene.meshes[mesh_id];
        pack_global_geometry_cuda(
            mesh.vertices,
            mesh.faces,
            static_cast<int32_t>(vertex_offset),
            static_cast<int32_t>(face_offset),
            mesh_id,
            scene.global_vertices,
            scene.global_faces,
            scene.face_shape_id,
            scene.face_local_id);
        vertex_offset += mesh.vertices.size(0);
        face_offset += mesh.faces.size(0);
    }

    const int64_t triangle_count = scene.global_faces.size(0);
    scene.tri_p0_x = at::empty({triangle_count}, fopts);
    scene.tri_p0_y = at::empty({triangle_count}, fopts);
    scene.tri_p0_z = at::empty({triangle_count}, fopts);
    scene.tri_e1_x = at::empty({triangle_count}, fopts);
    scene.tri_e1_y = at::empty({triangle_count}, fopts);
    scene.tri_e1_z = at::empty({triangle_count}, fopts);
    scene.tri_e2_x = at::empty({triangle_count}, fopts);
    scene.tri_e2_y = at::empty({triangle_count}, fopts);
    scene.tri_e2_z = at::empty({triangle_count}, fopts);
    scene.tri_fn_x = at::empty({triangle_count}, fopts);
    scene.tri_fn_y = at::empty({triangle_count}, fopts);
    scene.tri_fn_z = at::empty({triangle_count}, fopts);
    scene.tri_p0_packed = at::empty({triangle_count, 4}, fopts);
    scene.tri_e1_packed = at::empty({triangle_count, 4}, fopts);
    scene.tri_e2_packed = at::empty({triangle_count, 4}, fopts);
    scene.tri_fn_packed = at::empty({triangle_count, 4}, fopts);
    compute_triangle_soa_cuda(
        triangle_count,
        scene.global_vertices,
        scene.global_faces,
        scene.tri_p0_x,
        scene.tri_p0_y,
        scene.tri_p0_z,
        scene.tri_e1_x,
        scene.tri_e1_y,
        scene.tri_e1_z,
        scene.tri_e2_x,
        scene.tri_e2_y,
        scene.tri_e2_z,
        scene.tri_fn_x,
        scene.tri_fn_y,
        scene.tri_fn_z,
        scene.tri_p0_packed,
        scene.tri_e1_packed,
        scene.tri_e2_packed,
        scene.tri_fn_packed);
}

void update_triangle_accel(
    const MeshRecord &mesh,
    OptixTriangleAccel &accel,
    OptixDeviceContext optix_context,
    cudaStream_t stream) {
    accel.vertex_buffer = mesh.vertices.contiguous();
    CUdeviceptr vertex_buffer =
        reinterpret_cast<CUdeviceptr>(accel.vertex_buffer.data_ptr<float>());
    CUdeviceptr index_buffer =
        reinterpret_cast<CUdeviceptr>(accel.index_buffer.data_ptr<int>());
    uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &vertex_buffer;
    build_input.triangleArray.numVertices =
        static_cast<unsigned int>(accel.vertex_buffer.size(0));
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer = index_buffer;
    build_input.triangleArray.numIndexTriplets =
        static_cast<unsigned int>(accel.index_buffer.size(0));
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.flags = &triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes buffer_sizes = {};
    rayd_torch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    at::TensorOptions byte_options =
        at::TensorOptions().device(mesh.vertices.device()).dtype(at::kByte);
    size_t temp_bytes = buffer_sizes.tempUpdateSizeInBytes;
    if (accel.gas_buffer.numel() < static_cast<int64_t>(buffer_sizes.outputSizeInBytes)) {
        accel.gas_buffer =
            at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        temp_bytes = buffer_sizes.tempSizeInBytes;
    }
    if (accel.gas_temp_buffer.numel() < static_cast<int64_t>(temp_bytes))
        accel.gas_temp_buffer = at::empty({static_cast<int64_t>(temp_bytes)}, byte_options);

    rayd_torch_OPTIX_CHECK(optixAccelBuild(
        optix_context,
        stream,
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(accel.gas_temp_buffer.data_ptr<uint8_t>()),
        temp_bytes,
        reinterpret_cast<CUdeviceptr>(accel.gas_buffer.data_ptr<uint8_t>()),
        static_cast<size_t>(accel.gas_buffer.numel()),
        &accel.traversable,
        nullptr,
        0));
}

bool scene_has_dynamic_edges(const SceneCache &scene) {
    for (const MeshRecord &mesh : scene.meshes) {
        if (mesh.dynamic && mesh.edges_enabled)
            return true;
    }
    return false;
}

void build_edge_topology(SceneCache &scene) {
    std::vector<at::Tensor> edge_v0_parts;
    std::vector<at::Tensor> edge_v1_parts;
    std::vector<at::Tensor> edge_face0_parts;
    std::vector<at::Tensor> edge_face1_parts;
    std::vector<at::Tensor> edge_opposite_parts;
    std::vector<at::Tensor> edge_shape_id_parts;
    std::vector<at::Tensor> edge_local_id_parts;
    edge_v0_parts.reserve(scene.meshes.size());
    edge_v1_parts.reserve(scene.meshes.size());
    edge_face0_parts.reserve(scene.meshes.size());
    edge_face1_parts.reserve(scene.meshes.size());
    edge_opposite_parts.reserve(scene.meshes.size());
    edge_shape_id_parts.reserve(scene.meshes.size());
    edge_local_id_parts.reserve(scene.meshes.size());

    int32_t vertex_offset = 0;
    for (int32_t shape_id = 0; shape_id < static_cast<int32_t>(scene.meshes.size()); ++shape_id) {
        const MeshRecord &mesh = scene.meshes[shape_id];
        if (mesh.edges_enabled) {
            EdgeTopology topology = build_edge_topology_cuda(mesh.faces, vertex_offset, shape_id);
            if (topology.edge_v0.numel() > 0) {
                edge_v0_parts.push_back(topology.edge_v0);
                edge_v1_parts.push_back(topology.edge_v1);
                edge_face0_parts.push_back(topology.edge_face0);
                edge_face1_parts.push_back(topology.edge_face1);
                edge_opposite_parts.push_back(topology.edge_opposite);
                edge_shape_id_parts.push_back(topology.edge_shape_id);
                edge_local_id_parts.push_back(topology.edge_local_id);
            }
        }
        vertex_offset += static_cast<int32_t>(mesh.vertices.size(0));
    }

    at::Device device(at::kCUDA, scene.device_index);
    at::TensorOptions iopts = at::TensorOptions().device(device).dtype(at::kInt);
    auto cat_or_empty = [&](std::vector<at::Tensor> &parts) {
        if (parts.empty())
            return at::empty({0}, iopts);
        return at::cat(parts, 0).contiguous();
    };
    scene.edge_v0 = cat_or_empty(edge_v0_parts);
    scene.edge_v1 = cat_or_empty(edge_v1_parts);
    scene.edge_face0 = cat_or_empty(edge_face0_parts);
    scene.edge_face1 = cat_or_empty(edge_face1_parts);
    scene.edge_opposite = cat_or_empty(edge_opposite_parts);
    scene.edge_shape_id = cat_or_empty(edge_shape_id_parts);
    scene.edge_local_id = cat_or_empty(edge_local_id_parts);
}

std::vector<float> compute_edge_search_radii(
    const EdgeSearchStats &stats) {
    if (!stats.has_edges)
        return {};

    const float dx = std::max(stats.max_x - stats.min_x, 0.0f);
    const float dy = std::max(stats.max_y - stats.min_y, 0.0f);
    const float dz = std::max(stats.max_z - stats.min_z, 0.0f);
    const float full_radius = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0e-3f);
    const float edge_scale = std::max(stats.max_edge_length, full_radius * 1.0e-4f);

    std::vector<float> radii;
    radii.reserve(3);
    auto add_radius = [&](float radius) {
        if (std::isfinite(radius) && radius > 0.0f)
            radii.push_back(std::min(std::max(radius, 1.0e-5f), full_radius));
    };
    add_radius(edge_scale * 4.0f);
    add_radius(edge_scale * 34.0f);
    add_radius(full_radius);

    std::sort(radii.begin(), radii.end());
    std::vector<float> unique_radii;
    unique_radii.reserve(radii.size());
    for (float radius : radii) {
        if (unique_radii.empty() || radius > unique_radii.back() * 1.01f + 1.0e-6f)
            unique_radii.push_back(radius);
    }
    if (unique_radii.empty() || unique_radii.back() < full_radius * 0.999f)
        unique_radii.push_back(full_radius);
    else
        unique_radii.back() = full_radius;
    return unique_radii;
}

void refresh_edge_soa(SceneCache &scene) {
    const int64_t edge_count = scene.edge_v0.size(0);
    at::Device device(at::kCUDA, scene.device_index);
    at::TensorOptions fopts = at::TensorOptions().device(device).dtype(at::kFloat);
    scene.edge_p0_x = at::empty({edge_count}, fopts);
    scene.edge_p0_y = at::empty({edge_count}, fopts);
    scene.edge_p0_z = at::empty({edge_count}, fopts);
    scene.edge_e1_x = at::empty({edge_count}, fopts);
    scene.edge_e1_y = at::empty({edge_count}, fopts);
    scene.edge_e1_z = at::empty({edge_count}, fopts);
    if (!scene.edge_mask.defined() || scene.edge_mask.numel() != edge_count) {
        scene.edge_mask = at::ones(
            {edge_count}, at::TensorOptions().device(device).dtype(at::kByte));
        scene.edge_mask_version += 1;
    }
    compute_edge_soa_cuda(
        edge_count,
        scene.global_vertices,
        scene.edge_v0,
        scene.edge_v1,
        scene.edge_p0_x,
        scene.edge_p0_y,
        scene.edge_p0_z,
        scene.edge_e1_x,
        scene.edge_e1_y,
        scene.edge_e1_z);
}

void build_edge_accel(SceneCache &scene, OptixDeviceContext optix_context, cudaStream_t stream) {
    const int64_t edge_count = scene.edge_v0.size(0);
    refresh_edge_soa(scene);
    scene.edge_accels.clear();
    if (edge_count == 0) {
        scene.edge_accel = {};
        return;
    }

    const EdgeSearchStats stats = compute_edge_search_stats_cuda(
        edge_count,
        scene.edge_p0_x,
        scene.edge_p0_y,
        scene.edge_p0_z,
        scene.edge_e1_x,
        scene.edge_e1_y,
        scene.edge_e1_z);
    std::vector<float> radii = compute_edge_search_radii(stats);
    scene.edge_accels.resize(radii.size());
    at::Device device(at::kCUDA, scene.device_index);
    at::TensorOptions byte_options = at::TensorOptions().device(device).dtype(at::kByte);
    at::TensorOptions float_options = at::TensorOptions().device(device).dtype(at::kFloat);
    const bool compact_static_edges = !scene_has_dynamic_edges(scene);

    for (size_t gas_index = 0; gas_index < radii.size(); ++gas_index) {
        OptixEdgeAccel &accel = scene.edge_accels[gas_index];
        const float radius = radii[gas_index];
        accel.aabb_buffer = at::empty({edge_count, 6}, float_options);
        compute_edge_optix_aabbs_cuda(
            edge_count,
            scene.edge_p0_x,
            scene.edge_p0_y,
            scene.edge_p0_z,
            scene.edge_e1_x,
            scene.edge_e1_y,
            scene.edge_e1_z,
            radius,
            accel.aabb_buffer);

        CUdeviceptr aabb_buffer =
            reinterpret_cast<CUdeviceptr>(accel.aabb_buffer.data_ptr<float>());
        uint32_t input_flags = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(edge_count);
        build_input.customPrimitiveArray.strideInBytes = sizeof(float) * 6;
        build_input.customPrimitiveArray.flags = &input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        if (compact_static_edges)
            accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes buffer_sizes = {};
        rayd_torch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context, &accel_options, &build_input, 1, &buffer_sizes));

        accel.gas_temp_buffer =
            at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
        accel.gas_buffer =
            at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);
        at::Tensor compacted_size_buffer;
        OptixAccelEmitDesc compacted_size_emit = {};
        OptixAccelEmitDesc *emit_descs = nullptr;
        unsigned int emit_desc_count = 0;
        if (compact_static_edges) {
            compacted_size_buffer = at::empty({static_cast<int64_t>(sizeof(uint64_t))}, byte_options);
            compacted_size_emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            compacted_size_emit.result =
                reinterpret_cast<CUdeviceptr>(compacted_size_buffer.data_ptr<uint8_t>());
            emit_descs = &compacted_size_emit;
            emit_desc_count = 1;
        }
        rayd_torch_OPTIX_CHECK(optixAccelBuild(
            optix_context,
            stream,
            &accel_options,
            &build_input,
            1,
            reinterpret_cast<CUdeviceptr>(accel.gas_temp_buffer.data_ptr<uint8_t>()),
            buffer_sizes.tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(accel.gas_buffer.data_ptr<uint8_t>()),
            buffer_sizes.outputSizeInBytes,
            &accel.traversable,
            emit_descs,
            emit_desc_count));
        if (compact_static_edges) {
            compact_accel_if_smaller(
                optix_context,
                stream,
                byte_options,
                accel.gas_buffer,
                accel.traversable,
                compacted_size_buffer,
                "build_edge_accel()");
        }
        accel.search_radius = radius;
    }
    scene.edge_accel = scene.edge_accels.back();
}

bool update_edge_accel(SceneCache &scene, OptixDeviceContext optix_context, cudaStream_t stream) {
    const int64_t edge_count = scene.edge_v0.size(0);
    if (edge_count == 0) {
        scene.edge_accels.clear();
        scene.edge_accel = {};
        return true;
    }
    if (!scene_has_dynamic_edges(scene) || scene.edge_accels.empty())
        return false;

    refresh_edge_soa(scene);
    const EdgeSearchStats stats = compute_edge_search_stats_cuda(
        edge_count,
        scene.edge_p0_x,
        scene.edge_p0_y,
        scene.edge_p0_z,
        scene.edge_e1_x,
        scene.edge_e1_y,
        scene.edge_e1_z);
    std::vector<float> radii = compute_edge_search_radii(stats);
    if (radii.size() != scene.edge_accels.size())
        return false;

    at::Device device(at::kCUDA, scene.device_index);
    at::TensorOptions byte_options = at::TensorOptions().device(device).dtype(at::kByte);
    for (size_t gas_index = 0; gas_index < radii.size(); ++gas_index) {
        OptixEdgeAccel &accel = scene.edge_accels[gas_index];
        if (!accel.aabb_buffer.defined() || accel.aabb_buffer.size(0) != edge_count ||
            !accel.gas_buffer.defined()) {
            return false;
        }

        const float radius = radii[gas_index];
        compute_edge_optix_aabbs_cuda(
            edge_count,
            scene.edge_p0_x,
            scene.edge_p0_y,
            scene.edge_p0_z,
            scene.edge_e1_x,
            scene.edge_e1_y,
            scene.edge_e1_z,
            radius,
            accel.aabb_buffer);

        CUdeviceptr aabb_buffer =
            reinterpret_cast<CUdeviceptr>(accel.aabb_buffer.data_ptr<float>());
        uint32_t input_flags = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(edge_count);
        build_input.customPrimitiveArray.strideInBytes = sizeof(float) * 6;
        build_input.customPrimitiveArray.flags = &input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes buffer_sizes = {};
        rayd_torch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context, &accel_options, &build_input, 1, &buffer_sizes));
        const size_t temp_bytes = buffer_sizes.tempSizeInBytes;
        if (accel.gas_buffer.numel() < static_cast<int64_t>(buffer_sizes.outputSizeInBytes))
            return false;
        if (accel.gas_temp_buffer.numel() < static_cast<int64_t>(temp_bytes))
            accel.gas_temp_buffer = at::empty({static_cast<int64_t>(temp_bytes)}, byte_options);

        rayd_torch_OPTIX_CHECK(optixAccelBuild(
            optix_context,
            stream,
            &accel_options,
            &build_input,
            1,
            reinterpret_cast<CUdeviceptr>(accel.gas_temp_buffer.data_ptr<uint8_t>()),
            temp_bytes,
            reinterpret_cast<CUdeviceptr>(accel.gas_buffer.data_ptr<uint8_t>()),
            static_cast<size_t>(accel.gas_buffer.numel()),
            &accel.traversable,
            nullptr,
            0));
        accel.search_radius = radius;
    }
    scene.edge_accel = scene.edge_accels.back();
    return true;
}
} // namespace

SceneHandle::~SceneHandle() {
    if (owns_handle && handle != 0)
        destroy_scene(handle);
}

std::unique_ptr<SceneCache> create_scene_cache(std::vector<MeshRecord> meshes) {
    return create_scene_cache(
        std::move(meshes), TraceBackend::Auto, EdgeBackend::Auto);
}

std::unique_ptr<SceneCache> create_scene_cache(
    std::vector<MeshRecord> meshes,
    TraceBackend requested_trace_backend,
    EdgeBackend requested_edge_backend) {
    if (meshes.empty())
        throw std::runtime_error("Scene.build(): at least one mesh is required.");

    const int64_t device_index = meshes[0].vertices.get_device();
    c10::cuda::CUDAGuard guard(static_cast<int>(device_index));
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (torch_ctx.device_index != device_index)
        throw std::runtime_error("Scene.build(): current CUDA device does not match mesh tensors.");
    for (const MeshRecord &mesh : meshes) {
        require_vec3f(mesh.vertices, "mesh.vertices");
        require_vec3i(mesh.faces, "mesh.faces");
        require_optional_matrix4(mesh.to_world_left, "mesh.to_world_left");
        require_optional_matrix4(mesh.to_world_right, "mesh.to_world_right");
        if (mesh.vertices.get_device() != device_index || mesh.faces.get_device() != device_index)
            throw std::runtime_error("Scene.build(): all tensors must be on the same CUDA device.");
        if (mesh.to_world_left.get_device() != device_index || mesh.to_world_right.get_device() != device_index)
            throw std::runtime_error("Scene.build(): transform tensors must be on the scene device.");
    }

    auto scene_unique = std::make_unique<SceneCache>();
    SceneCache *scene = scene_unique.get();
    scene->handle = next_handle.fetch_add(1);
    scene->device_index = device_index;
    if (requested_trace_backend == TraceBackend::Auto) {
        scene->trace_backend = optix_context_available(static_cast<int>(device_index))
            ? TraceBackend::Optix
            : TraceBackend::Cuda;
    } else {
        scene->trace_backend = requested_trace_backend;
    }
    scene->edge_backend = requested_edge_backend == EdgeBackend::Auto
        ? (scene->trace_backend == TraceBackend::Optix ? EdgeBackend::Optix : EdgeBackend::Cuda)
        : requested_edge_backend;
    OptixDeviceContextEntry *optix_entry = nullptr;
    if (scene->trace_backend == TraceBackend::Optix || scene->edge_backend == EdgeBackend::Optix)
        optix_entry = &get_optix_context(static_cast<int>(device_index));
    scene->meshes = std::move(meshes);
    refresh_global_geometry(*scene);
    if (scene->trace_backend == TraceBackend::Optix) {
        scene->triangle_accels.reserve(scene->meshes.size());
        for (const MeshRecord &mesh : scene->meshes)
            scene->triangle_accels.push_back(
                build_triangle_accel(mesh, optix_entry->optix_context, torch_ctx.stream));
        build_triangle_ias(*scene, optix_entry->optix_context, torch_ctx.stream);
    } else {
        ensure_custom_triangle_bvh(*scene);
    }
    build_edge_topology(*scene);
    if (scene->edge_backend == EdgeBackend::Optix)
        build_edge_accel(*scene, optix_entry->optix_context, torch_ctx.stream);
    else
        refresh_edge_soa(*scene);
    return scene_unique;
}

int64_t create_scene(std::vector<MeshRecord> meshes) {
    auto scene = create_scene_cache(std::move(meshes));
    const int64_t handle = scene->handle;
    register_scene_cache(std::move(scene));
    return handle;
}

void register_scene_cache(std::unique_ptr<SceneCache> scene) {
    const int64_t handle = scene->handle;
    std::lock_guard<std::mutex> lock(scenes_mutex);
    scenes.emplace(handle, std::move(scene));
}

void destroy_scene(int64_t handle) {
    if (handle == 0)
        return;
    // The GIL is the outermost lock in this process: every TORCH_LIBRARY op
    // wrapper holds it and then takes `scenes_mutex` through get_scene(). A
    // SceneCache owns the caller's mesh tensors, and releasing a tensor that
    // carries a Python object re-enters Python (THPVariable_clear drops the
    // GIL around the TensorImpl release and takes it back afterwards). Running
    // that destructor under `scenes_mutex` therefore waits for the GIL while
    // holding a RayD lock, which is the reverse order and deadlocks against
    // any thread that is inside an op. Detach the entry under the lock and
    // destroy it after the lock is gone; the map never exposes a dying scene.
    std::unique_ptr<SceneCache> scene;
    {
        std::lock_guard<std::mutex> lock(scenes_mutex);
        auto node = scenes.extract(handle);
        if (node.empty())
            return;
        scene = std::move(node.mapped());
    }
}

SceneCache &get_scene(int64_t handle) {
    std::lock_guard<std::mutex> lock(scenes_mutex);
    auto it = scenes.find(handle);
    if (it == scenes.end())
        throw std::runtime_error("Invalid RayD Torch scene handle.");
    return *it->second;
}

int64_t scene_version(int64_t handle) {
    return get_scene(handle).version;
}

int64_t scene_num_meshes(int64_t handle) {
    return static_cast<int64_t>(get_scene(handle).meshes.size());
}

int64_t scene_edge_count(int64_t handle) {
    return get_scene(handle).edge_v0.size(0);
}

std::string scene_trace_backend(int64_t handle) {
    return get_scene(handle).trace_backend == TraceBackend::Cuda ? "cuda" : "optix";
}

std::string scene_edge_backend(int64_t handle) {
    return get_scene(handle).edge_backend == EdgeBackend::Cuda ? "cuda" : "optix";
}

void update_mesh_vertices(int64_t handle, int64_t mesh_id, at::Tensor vertices) {
    SceneCache &scene = get_scene(handle);
    if (mesh_id < 0 || mesh_id >= static_cast<int64_t>(scene.meshes.size()))
        throw std::runtime_error("update_mesh_vertices(): invalid mesh id.");
    MeshRecord &mesh = scene.meshes[mesh_id];
    if (!mesh.dynamic)
        throw std::runtime_error("update_mesh_vertices(): target mesh is not dynamic.");
    require_vec3f(vertices, "vertices");
    if (vertices.get_device() != scene.device_index)
        throw std::runtime_error("update_mesh_vertices(): vertices must stay on the scene device.");
    if (vertices.size(0) != mesh.vertices.size(0))
        throw std::runtime_error("update_mesh_vertices(): vertex count must stay unchanged.");
    mesh.vertices = vertices.contiguous();
    mesh.pending_update = true;
}

void sync_scene(int64_t handle) {
    SceneCache &scene = get_scene(handle);
    c10::cuda::CUDAGuard guard(static_cast<int>(scene.device_index));
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (torch_ctx.device_index != scene.device_index)
        throw std::runtime_error("Scene.sync(): current CUDA device does not match scene tensors.");
    OptixDeviceContextEntry *optix_entry = nullptr;
    if (scene.trace_backend == TraceBackend::Optix || scene.edge_backend == EdgeBackend::Optix)
        optix_entry = &get_optix_context(static_cast<int>(scene.device_index));

    bool changed = false;
    for (int64_t mesh_id = 0; mesh_id < static_cast<int64_t>(scene.meshes.size()); ++mesh_id) {
        MeshRecord &mesh = scene.meshes[mesh_id];
        if (!mesh.pending_update)
            continue;
        if (scene.trace_backend == TraceBackend::Optix) {
            update_triangle_accel(
                mesh,
                scene.triangle_accels[mesh_id],
                optix_entry->optix_context,
                torch_ctx.stream);
        }
        mesh.pending_update = false;
        changed = true;
    }
    if (changed) {
        refresh_global_geometry(scene);
        scene.version += 1;
        scene.edge_version += 1;
        if (scene.trace_backend == TraceBackend::Optix)
            build_triangle_ias(scene, optix_entry->optix_context, torch_ctx.stream);
        else
            ensure_custom_triangle_bvh(scene);
        if (scene.edge_backend == EdgeBackend::Optix) {
            if (!update_edge_accel(scene, optix_entry->optix_context, torch_ctx.stream))
                build_edge_accel(scene, optix_entry->optix_context, torch_ctx.stream);
        } else {
            refresh_edge_soa(scene);
        }
    }
}

int64_t scene_version(c10::intrusive_ptr<SceneHandle> scene) {
    return scene_version(scene->handle);
}

int64_t scene_num_meshes(c10::intrusive_ptr<SceneHandle> scene) {
    return scene_num_meshes(scene->handle);
}

int64_t scene_edge_count(c10::intrusive_ptr<SceneHandle> scene) {
    return scene_edge_count(scene->handle);
}

namespace {

struct HostTreeletSchedule {
    std::vector<int> nodes;
    std::vector<int> level_offsets;
};

HostTreeletSchedule build_treelet_schedule(
    int primitive_count,
    const std::vector<int> &left_child,
    const std::vector<int> &right_child) {
    const int node_count = primitive_count * 2 - 1;
    const int leaf_base = primitive_count - 1;
    std::vector<int> height(static_cast<size_t>(node_count), 0);
    std::vector<int> leaf_count(static_cast<size_t>(node_count), 1);
    std::vector<std::pair<int, bool>> stack;
    stack.reserve(static_cast<size_t>(node_count) * 2);
    stack.emplace_back(0, false);
    int max_height = 0;
    while (!stack.empty()) {
        const auto [node, visited] = stack.back();
        stack.pop_back();
        if (node < 0 || node >= node_count)
            throw std::runtime_error("custom edge BVH contains an invalid child index.");
        if (node >= leaf_base)
            continue;
        if (!visited) {
            const int left = left_child[static_cast<size_t>(node)];
            const int right = right_child[static_cast<size_t>(node)];
            if (left < 0 || left >= node_count || right < 0 || right >= node_count)
                throw std::runtime_error("custom edge BVH contains an invalid internal node.");
            stack.emplace_back(node, true);
            stack.emplace_back(right, false);
            stack.emplace_back(left, false);
            continue;
        }
        const int left = left_child[static_cast<size_t>(node)];
        const int right = right_child[static_cast<size_t>(node)];
        height[static_cast<size_t>(node)] =
            std::max(height[static_cast<size_t>(left)], height[static_cast<size_t>(right)]) + 1;
        leaf_count[static_cast<size_t>(node)] =
            leaf_count[static_cast<size_t>(left)] + leaf_count[static_cast<size_t>(right)];
        max_height = std::max(max_height, height[static_cast<size_t>(node)]);
    }

    std::vector<std::vector<int>> levels(static_cast<size_t>(max_height + 1));
    for (int node = 0; node < leaf_base; ++node) {
        if (leaf_count[static_cast<size_t>(node)] >=
            rayd::shared::edge::kBvhTreeletMinSubtreeLeaves) {
            levels[static_cast<size_t>(height[static_cast<size_t>(node)])].push_back(node);
        }
    }
    HostTreeletSchedule schedule;
    schedule.level_offsets.resize(static_cast<size_t>(max_height + 2), 0);
    for (int level = 0; level <= max_height; ++level) {
        schedule.level_offsets[static_cast<size_t>(level)] =
            static_cast<int>(schedule.nodes.size());
        schedule.nodes.insert(schedule.nodes.end(), levels[static_cast<size_t>(level)].begin(),
                              levels[static_cast<size_t>(level)].end());
    }
    schedule.level_offsets[static_cast<size_t>(max_height + 1)] =
        static_cast<int>(schedule.nodes.size());
    return schedule;
}

} // namespace

rayd::shared::edge::EdgeSoAView scene_edge_view(const SceneCache &scene) {
    return {scene.edge_p0_x.data_ptr<float>(), scene.edge_p0_y.data_ptr<float>(),
            scene.edge_p0_z.data_ptr<float>(), scene.edge_e1_x.data_ptr<float>(),
            scene.edge_e1_y.data_ptr<float>(), scene.edge_e1_z.data_ptr<float>(),
            static_cast<size_t>(scene.edge_v0.numel())};
}

rayd::shared::edge::AabbSoAView scene_edge_bvh_bounds_view(const SceneCache &scene) {
    const CompactEdgeBvh &bvh = scene.custom_edge_bvh;
    return {bvh.node_min_x.defined() ? bvh.node_min_x.data_ptr<float>() : nullptr,
            bvh.node_min_y.defined() ? bvh.node_min_y.data_ptr<float>() : nullptr,
            bvh.node_min_z.defined() ? bvh.node_min_z.data_ptr<float>() : nullptr,
            bvh.node_max_x.defined() ? bvh.node_max_x.data_ptr<float>() : nullptr,
            bvh.node_max_y.defined() ? bvh.node_max_y.data_ptr<float>() : nullptr,
            bvh.node_max_z.defined() ? bvh.node_max_z.data_ptr<float>() : nullptr,
            static_cast<size_t>(bvh.node_count)};
}

rayd::shared::edge::CompactBvhTopologyView scene_edge_bvh_topology_view(
    const SceneCache &scene) {
    const CompactEdgeBvh &bvh = scene.custom_edge_bvh;
    const size_t primitive_count = static_cast<size_t>(scene.edge_v0.numel());
    return {bvh.left_child.defined() ? bvh.left_child.data_ptr<int>() : nullptr,
            bvh.right_child.defined() ? bvh.right_child.data_ptr<int>() : nullptr,
            bvh.leaf_primitives.defined() ? bvh.leaf_primitives.data_ptr<int>() : nullptr, nullptr,
            static_cast<size_t>(bvh.node_count), primitive_count, primitive_count};
}

void ensure_custom_edge_bvh(SceneCache &scene) {
    CompactEdgeBvh &bvh = scene.custom_edge_bvh;
    if (bvh.valid && bvh.geometry_version == scene.edge_version)
        return;

    c10::cuda::CUDAGuard guard(static_cast<int>(scene.device_index));
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (torch_ctx.device_index != scene.device_index)
        throw std::runtime_error("ensure_custom_edge_bvh(): current CUDA device does not match scene.");
    const int64_t primitive_count = scene.edge_v0.numel();
    const int64_t max_primitive_count =
        (static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) / 2;
    if (primitive_count > max_primitive_count)
        throw std::runtime_error(
            "ensure_custom_edge_bvh(): node topology exceeds int32 indexing range.");

    bvh = {};
    bvh.geometry_version = scene.edge_version;
    if (primitive_count == 0) {
        bvh.valid = true;
        return;
    }

    const int64_t node_count = primitive_count * 2 - 1;
    bvh.node_count = node_count;
    const auto fopts = scene.edge_p0_x.options();
    const auto iopts = scene.edge_v0.options();
    const auto bopts = at::TensorOptions().device(scene.edge_v0.device()).dtype(at::kByte);
    auto make_f = [&](int64_t count) { return at::empty({count}, fopts); };
    auto make_i = [&](int64_t count) { return at::empty({count}, iopts); };
    bvh.primitive_min_x = make_f(primitive_count); bvh.primitive_min_y = make_f(primitive_count);
    bvh.primitive_min_z = make_f(primitive_count); bvh.primitive_max_x = make_f(primitive_count);
    bvh.primitive_max_y = make_f(primitive_count); bvh.primitive_max_z = make_f(primitive_count);
    bvh.node_min_x = make_f(node_count); bvh.node_min_y = make_f(node_count);
    bvh.node_min_z = make_f(node_count); bvh.node_max_x = make_f(node_count);
    bvh.node_max_y = make_f(node_count); bvh.node_max_z = make_f(node_count);
    bvh.left_child = make_i(node_count).fill_(-1); bvh.right_child = make_i(node_count).fill_(-1);
    bvh.parent = make_i(node_count).fill_(-1); bvh.leaf_primitive = make_i(node_count).fill_(-1);
    bvh.is_leaf = make_i(node_count).zero_(); bvh.primitive_leaf_node = make_i(primitive_count).fill_(-1);
    bvh.leaf_primitives = make_i(primitive_count).fill_(-1);
    bvh.morton_in = make_i(primitive_count); bvh.morton_out = make_i(primitive_count);
    bvh.primitive_ids_in = make_i(primitive_count); bvh.primitive_ids_out = make_i(primitive_count);
    bvh.merge_counters = make_i(std::max<int64_t>(primitive_count - 1, 1)).zero_();
    const size_t bounds_bytes = sizeof(rayd::shared::edge::BvhBounds3);
    bvh.packed_bounds = at::empty({primitive_count * static_cast<int64_t>(bounds_bytes)}, bopts);
    bvh.reduced_bound = at::empty({static_cast<int64_t>(bounds_bytes)}, bopts);
    const size_t scratch_bytes = std::max(
        edge_bvh_bounds_reduce_scratch_bytes(primitive_count, torch_ctx.stream),
        edge_bvh_sort_scratch_bytes(primitive_count, torch_ctx.stream));
    bvh.scratch = at::empty({static_cast<int64_t>(scratch_bytes)}, bopts);

    rayd::shared::edge::launch_compute_primitive_bounds_async({
        scene_edge_view(scene),
        {bvh.primitive_min_x.data_ptr<float>(), bvh.primitive_min_y.data_ptr<float>(),
         bvh.primitive_min_z.data_ptr<float>(), bvh.primitive_max_x.data_ptr<float>(),
         bvh.primitive_max_y.data_ptr<float>(), bvh.primitive_max_z.data_ptr<float>(),
         static_cast<size_t>(primitive_count)},
        reinterpret_cast<rayd::shared::edge::BvhBounds3 *>(bvh.packed_bounds.data_ptr<uint8_t>()),
        torch_ctx.stream});
    reduce_edge_bvh_bounds_cuda(
        primitive_count, bvh.packed_bounds, bvh.reduced_bound, bvh.scratch, torch_ctx.stream);
    rayd::shared::edge::BvhBounds3 scene_bound = {};
    cuda_check(cudaMemcpyAsync(&scene_bound, bvh.reduced_bound.data_ptr<uint8_t>(), bounds_bytes,
                               cudaMemcpyDeviceToHost, torch_ctx.stream),
               "cudaMemcpyAsync(custom BVH scene bound)");
    cuda_check(cudaStreamSynchronize(torch_ctx.stream),
               "cudaStreamSynchronize(custom BVH scene bound)");

    rayd::shared::edge::launch_compute_morton_codes_async({
        {bvh.primitive_min_x.data_ptr<float>(), bvh.primitive_min_y.data_ptr<float>(),
         bvh.primitive_min_z.data_ptr<float>(), bvh.primitive_max_x.data_ptr<float>(),
         bvh.primitive_max_y.data_ptr<float>(), bvh.primitive_max_z.data_ptr<float>(),
         static_cast<size_t>(primitive_count)},
        scene_bound, reinterpret_cast<uint32_t *>(bvh.morton_in.data_ptr<int>()), torch_ctx.stream});
    rayd::shared::edge::launch_init_sequence_async({
        bvh.primitive_ids_in.data_ptr<int>(), static_cast<int>(primitive_count), torch_ctx.stream});
    sort_edge_bvh_morton_cuda(primitive_count, bvh.morton_in, bvh.morton_out,
                              bvh.primitive_ids_in, bvh.primitive_ids_out,
                              bvh.scratch, torch_ctx.stream);
    HostTreeletSchedule treelet_schedule;
    const bool optimize_treelets =
        primitive_count >= rayd::shared::edge::kBvhTreeletMinPrimitives &&
        primitive_count <= rayd::shared::edge::kBvhTreeletMaxPrimitives;
    if (primitive_count > 1) {
        rayd::shared::edge::launch_build_radix_tree_async({
            reinterpret_cast<const uint32_t *>(bvh.morton_out.data_ptr<int>()),
            bvh.primitive_ids_out.data_ptr<int>(), bvh.left_child.data_ptr<int>(),
            bvh.right_child.data_ptr<int>(), bvh.parent.data_ptr<int>(),
            static_cast<int>(primitive_count), torch_ctx.stream});
    }
    if (optimize_treelets) {
        std::vector<int> host_left(static_cast<size_t>(node_count), -1);
        std::vector<int> host_right(static_cast<size_t>(node_count), -1);
        cuda_check(cudaMemcpyAsync(host_left.data(), bvh.left_child.data_ptr<int>(),
                                   static_cast<size_t>(node_count) * sizeof(int),
                                   cudaMemcpyDeviceToHost, torch_ctx.stream),
                   "cudaMemcpyAsync(custom BVH left topology)");
        cuda_check(cudaMemcpyAsync(host_right.data(), bvh.right_child.data_ptr<int>(),
                                   static_cast<size_t>(node_count) * sizeof(int),
                                   cudaMemcpyDeviceToHost, torch_ctx.stream),
                   "cudaMemcpyAsync(custom BVH right topology)");
        cuda_check(cudaStreamSynchronize(torch_ctx.stream),
                   "cudaStreamSynchronize(custom BVH topology)");
        treelet_schedule = build_treelet_schedule(
            static_cast<int>(primitive_count), host_left, host_right);
        bvh.node_cost = make_f(node_count);
        bvh.internal_cost_arrivals = make_i(node_count).zero_();
        bvh.treelet_nodes = make_i(static_cast<int64_t>(treelet_schedule.nodes.size()));
        if (!treelet_schedule.nodes.empty()) {
            cuda_check(cudaMemcpyAsync(
                           bvh.treelet_nodes.data_ptr<int>(), treelet_schedule.nodes.data(),
                           treelet_schedule.nodes.size() * sizeof(int),
                           cudaMemcpyHostToDevice, torch_ctx.stream),
                       "cudaMemcpyAsync(custom BVH treelet schedule)");
        }
    }
    rayd::shared::edge::launch_finalize_leaves_and_bounds_async({
        bvh.primitive_ids_out.data_ptr<int>(), bvh.parent.data_ptr<int>(),
        {bvh.primitive_min_x.data_ptr<float>(), bvh.primitive_min_y.data_ptr<float>(),
         bvh.primitive_min_z.data_ptr<float>(), bvh.primitive_max_x.data_ptr<float>(),
         bvh.primitive_max_y.data_ptr<float>(), bvh.primitive_max_z.data_ptr<float>(),
         static_cast<size_t>(primitive_count)},
        bvh.left_child.data_ptr<int>(), bvh.right_child.data_ptr<int>(),
        {bvh.node_min_x.data_ptr<float>(), bvh.node_min_y.data_ptr<float>(),
         bvh.node_min_z.data_ptr<float>(), bvh.node_max_x.data_ptr<float>(),
         bvh.node_max_y.data_ptr<float>(), bvh.node_max_z.data_ptr<float>(),
         static_cast<size_t>(node_count)},
        bvh.leaf_primitive.data_ptr<int>(), bvh.is_leaf.data_ptr<int>(),
        bvh.primitive_leaf_node.data_ptr<int>(), bvh.merge_counters.data_ptr<int>(),
        static_cast<int>(primitive_count), torch_ctx.stream});
    if (optimize_treelets) {
        const float scene_scale = std::max(
            scene_bound.max.x - scene_bound.min.x,
            std::max(scene_bound.max.y - scene_bound.min.y,
                     scene_bound.max.z - scene_bound.min.z));
        const float inflation = std::max(
            scene_scale * rayd::shared::edge::kBvhTreeletCostInflationRatio, 1.0e-6f);
        rayd::shared::edge::launch_initialize_leaf_costs_async({
            {bvh.node_min_x.data_ptr<float>(), bvh.node_min_y.data_ptr<float>(),
             bvh.node_min_z.data_ptr<float>(), bvh.node_max_x.data_ptr<float>(),
             bvh.node_max_y.data_ptr<float>(), bvh.node_max_z.data_ptr<float>(),
             static_cast<size_t>(node_count)},
            bvh.node_cost.data_ptr<float>(), inflation, static_cast<int>(primitive_count),
            torch_ctx.stream});
        rayd::shared::edge::launch_initialize_internal_costs_async({
            bvh.left_child.data_ptr<int>(), bvh.right_child.data_ptr<int>(),
            bvh.parent.data_ptr<int>(),
            {bvh.node_min_x.data_ptr<float>(), bvh.node_min_y.data_ptr<float>(),
             bvh.node_min_z.data_ptr<float>(), bvh.node_max_x.data_ptr<float>(),
             bvh.node_max_y.data_ptr<float>(), bvh.node_max_z.data_ptr<float>(),
             static_cast<size_t>(node_count)},
            bvh.node_cost.data_ptr<float>(), bvh.internal_cost_arrivals.data_ptr<int>(),
            inflation, static_cast<int>(primitive_count), torch_ctx.stream});
        const int max_height = static_cast<int>(treelet_schedule.level_offsets.size()) - 2;
        for (int height = 1; height <= max_height; ++height) {
            const int begin = treelet_schedule.level_offsets[static_cast<size_t>(height)];
            const int end = treelet_schedule.level_offsets[static_cast<size_t>(height + 1)];
            if (end == begin)
                continue;
            rayd::shared::edge::launch_optimize_selected_treelets_async({
                bvh.treelet_nodes.data_ptr<int>() + begin, bvh.is_leaf.data_ptr<int>(),
                bvh.left_child.data_ptr<int>(), bvh.right_child.data_ptr<int>(),
                bvh.parent.data_ptr<int>(),
                {bvh.node_min_x.data_ptr<float>(), bvh.node_min_y.data_ptr<float>(),
                 bvh.node_min_z.data_ptr<float>(), bvh.node_max_x.data_ptr<float>(),
                 bvh.node_max_y.data_ptr<float>(), bvh.node_max_z.data_ptr<float>(),
                 static_cast<size_t>(node_count)},
                bvh.leaf_primitive.data_ptr<int>(), bvh.node_cost.data_ptr<float>(),
                inflation, end - begin, torch_ctx.stream});
        }
    }
    encode_raw_edge_bvh_cuda(primitive_count, bvh.left_child, bvh.right_child,
                             bvh.leaf_primitive, bvh.leaf_primitives, torch_ctx.stream);
    cuda_check(cudaGetLastError(), "ensure_custom_edge_bvh() kernel launch");
    // The cache may be consumed by a later dispatcher call on another Torch
    // stream. Publish it only after its persistent topology is fully built.
    cuda_check(cudaStreamSynchronize(torch_ctx.stream),
               "cudaStreamSynchronize(custom BVH build)");
    bvh.valid = true;
}

std::vector<at::Tensor> scene_edge_records(SceneCache &scene) {
    at::Tensor edge_shape_index = scene.edge_shape_id.to(at::kLong);
    at::Tensor face_offsets = scene.face_offsets.index_select(0, edge_shape_index);
    at::Tensor edge_face0_global = scene.edge_face0 + face_offsets;
    at::Tensor edge_face1_global = at::where(
        scene.edge_face1.ge(0),
        scene.edge_face1 + face_offsets,
        scene.edge_face1);

    return {
        scene.global_vertices,
        scene.global_faces,
        scene.tri_fn_x,
        scene.tri_fn_y,
        scene.tri_fn_z,
        scene.edge_v0,
        scene.edge_v1,
        edge_face0_global,
        edge_face1_global,
        scene.edge_shape_id,
        scene.edge_local_id,
        scene.edge_opposite,
    };
}

rayd::shared::bvh::TriangleSoAView scene_triangle_view(const SceneCache &scene) {
    return {scene.tri_p0_x.data_ptr<float>(), scene.tri_p0_y.data_ptr<float>(),
            scene.tri_p0_z.data_ptr<float>(), scene.tri_e1_x.data_ptr<float>(),
            scene.tri_e1_y.data_ptr<float>(), scene.tri_e1_z.data_ptr<float>(),
            scene.tri_e2_x.data_ptr<float>(), scene.tri_e2_y.data_ptr<float>(),
            scene.tri_e2_z.data_ptr<float>(),
            static_cast<size_t>(scene.global_faces.size(0))};
}

rayd::shared::bvh::AabbSoAView scene_triangle_bvh_bounds_view(const SceneCache &scene) {
    const CompactTriangleBvh &bvh = scene.custom_triangle_bvh;
    return {bvh.node_min_x.defined() ? bvh.node_min_x.data_ptr<float>() : nullptr,
            bvh.node_min_y.defined() ? bvh.node_min_y.data_ptr<float>() : nullptr,
            bvh.node_min_z.defined() ? bvh.node_min_z.data_ptr<float>() : nullptr,
            bvh.node_max_x.defined() ? bvh.node_max_x.data_ptr<float>() : nullptr,
            bvh.node_max_y.defined() ? bvh.node_max_y.data_ptr<float>() : nullptr,
            bvh.node_max_z.defined() ? bvh.node_max_z.data_ptr<float>() : nullptr,
            static_cast<size_t>(bvh.node_count)};
}

rayd::shared::bvh::CompactBvhTopologyView scene_triangle_bvh_topology_view(
    const SceneCache &scene) {
    const CompactTriangleBvh &bvh = scene.custom_triangle_bvh;
    const size_t primitive_count = static_cast<size_t>(scene.global_faces.size(0));
    return {bvh.left_child.defined() ? bvh.left_child.data_ptr<int>() : nullptr,
            bvh.right_child.defined() ? bvh.right_child.data_ptr<int>() : nullptr,
            bvh.leaf_primitives.defined() ? bvh.leaf_primitives.data_ptr<int>() : nullptr,
            nullptr, static_cast<size_t>(bvh.node_count), primitive_count, primitive_count};
}

void ensure_custom_triangle_bvh(SceneCache &scene) {
    CompactTriangleBvh &bvh = scene.custom_triangle_bvh;
    if (bvh.valid && bvh.geometry_version == scene.version)
        return;

    c10::cuda::CUDAGuard guard(static_cast<int>(scene.device_index));
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (torch_ctx.device_index != scene.device_index)
        throw std::runtime_error(
            "ensure_custom_triangle_bvh(): current CUDA device does not match scene.");
    const int64_t primitive_count = scene.global_faces.size(0);
    const int64_t max_primitive_count =
        (static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) / 2;
    if (primitive_count > max_primitive_count)
        throw std::runtime_error(
            "ensure_custom_triangle_bvh(): topology exceeds int32 indexing range.");

    bvh = {};
    bvh.geometry_version = scene.version;
    if (primitive_count == 0) {
        bvh.valid = true;
        return;
    }

    const int64_t node_count = primitive_count * 2 - 1;
    bvh.node_count = node_count;
    const auto fopts = scene.global_vertices.options();
    const auto iopts = scene.global_faces.options();
    const auto bopts = at::TensorOptions().device(scene.global_faces.device()).dtype(at::kByte);
    auto make_f = [&](int64_t count) { return at::empty({count}, fopts); };
    auto make_i = [&](int64_t count) { return at::empty({count}, iopts); };
    bvh.primitive_min_x = make_f(primitive_count); bvh.primitive_min_y = make_f(primitive_count);
    bvh.primitive_min_z = make_f(primitive_count); bvh.primitive_max_x = make_f(primitive_count);
    bvh.primitive_max_y = make_f(primitive_count); bvh.primitive_max_z = make_f(primitive_count);
    bvh.node_min_x = make_f(node_count); bvh.node_min_y = make_f(node_count);
    bvh.node_min_z = make_f(node_count); bvh.node_max_x = make_f(node_count);
    bvh.node_max_y = make_f(node_count); bvh.node_max_z = make_f(node_count);
    bvh.left_child = make_i(node_count).fill_(-1); bvh.right_child = make_i(node_count).fill_(-1);
    bvh.parent = make_i(node_count).fill_(-1); bvh.leaf_primitive = make_i(node_count).fill_(-1);
    bvh.is_leaf = make_i(node_count).zero_();
    bvh.primitive_leaf_node = make_i(primitive_count).fill_(-1);
    bvh.leaf_primitives = make_i(primitive_count).fill_(-1);
    bvh.morton_in = make_i(primitive_count); bvh.morton_out = make_i(primitive_count);
    bvh.primitive_ids_in = make_i(primitive_count); bvh.primitive_ids_out = make_i(primitive_count);
    bvh.merge_counters = make_i(std::max<int64_t>(primitive_count - 1, 1)).zero_();
    const size_t bounds_bytes = sizeof(rayd::shared::bvh::BvhBounds3);
    bvh.packed_bounds = at::empty(
        {primitive_count * static_cast<int64_t>(bounds_bytes)}, bopts);
    bvh.reduced_bound = at::empty({static_cast<int64_t>(bounds_bytes)}, bopts);
    const size_t scratch_bytes = std::max(
        edge_bvh_bounds_reduce_scratch_bytes(primitive_count, torch_ctx.stream),
        edge_bvh_sort_scratch_bytes(primitive_count, torch_ctx.stream));
    bvh.scratch = at::empty({static_cast<int64_t>(scratch_bytes)}, bopts);

    compute_triangle_bvh_bounds_cuda(
        scene, bvh.primitive_min_x, bvh.primitive_min_y, bvh.primitive_min_z,
        bvh.primitive_max_x, bvh.primitive_max_y, bvh.primitive_max_z,
        bvh.packed_bounds, torch_ctx.stream);
    reduce_edge_bvh_bounds_cuda(
        primitive_count, bvh.packed_bounds, bvh.reduced_bound, bvh.scratch, torch_ctx.stream);
    rayd::shared::bvh::BvhBounds3 scene_bound = {};
    cuda_check(cudaMemcpyAsync(
                   &scene_bound, bvh.reduced_bound.data_ptr<uint8_t>(), bounds_bytes,
                   cudaMemcpyDeviceToHost, torch_ctx.stream),
               "cudaMemcpyAsync(triangle BVH scene bound)");
    cuda_check(cudaStreamSynchronize(torch_ctx.stream),
               "cudaStreamSynchronize(triangle BVH scene bound)");

    rayd::shared::bvh::launch_compute_morton_codes_async({
        {bvh.primitive_min_x.data_ptr<float>(), bvh.primitive_min_y.data_ptr<float>(),
         bvh.primitive_min_z.data_ptr<float>(), bvh.primitive_max_x.data_ptr<float>(),
         bvh.primitive_max_y.data_ptr<float>(), bvh.primitive_max_z.data_ptr<float>(),
         static_cast<size_t>(primitive_count)},
        scene_bound, reinterpret_cast<uint32_t *>(bvh.morton_in.data_ptr<int>()),
        torch_ctx.stream});
    rayd::shared::bvh::launch_init_sequence_async({
        bvh.primitive_ids_in.data_ptr<int>(), static_cast<int>(primitive_count), torch_ctx.stream});
    sort_edge_bvh_morton_cuda(
        primitive_count, bvh.morton_in, bvh.morton_out, bvh.primitive_ids_in,
        bvh.primitive_ids_out, bvh.scratch, torch_ctx.stream);
    if (primitive_count > 1) {
        rayd::shared::bvh::launch_build_radix_tree_async({
            reinterpret_cast<const uint32_t *>(bvh.morton_out.data_ptr<int>()),
            bvh.primitive_ids_out.data_ptr<int>(), bvh.left_child.data_ptr<int>(),
            bvh.right_child.data_ptr<int>(), bvh.parent.data_ptr<int>(),
            static_cast<int>(primitive_count), torch_ctx.stream});
    }
    rayd::shared::bvh::launch_finalize_leaves_and_bounds_async({
        bvh.primitive_ids_out.data_ptr<int>(), bvh.parent.data_ptr<int>(),
        {bvh.primitive_min_x.data_ptr<float>(), bvh.primitive_min_y.data_ptr<float>(),
         bvh.primitive_min_z.data_ptr<float>(), bvh.primitive_max_x.data_ptr<float>(),
         bvh.primitive_max_y.data_ptr<float>(), bvh.primitive_max_z.data_ptr<float>(),
         static_cast<size_t>(primitive_count)},
        bvh.left_child.data_ptr<int>(), bvh.right_child.data_ptr<int>(),
        {bvh.node_min_x.data_ptr<float>(), bvh.node_min_y.data_ptr<float>(),
         bvh.node_min_z.data_ptr<float>(), bvh.node_max_x.data_ptr<float>(),
         bvh.node_max_y.data_ptr<float>(), bvh.node_max_z.data_ptr<float>(),
         static_cast<size_t>(node_count)},
        bvh.leaf_primitive.data_ptr<int>(), bvh.is_leaf.data_ptr<int>(),
        bvh.primitive_leaf_node.data_ptr<int>(), bvh.merge_counters.data_ptr<int>(),
        static_cast<int>(primitive_count), torch_ctx.stream});

    // The fused CUDA traverser intentionally uses a fixed, caller-owned stack.
    // Reject a topology that cannot fit before making it visible to queries;
    // unlike the standalone intersection path, multipath launches cannot repair
    // an overflowed lane without changing their tracing decisions.
    std::vector<int> host_left(static_cast<size_t>(node_count));
    std::vector<int> host_right(static_cast<size_t>(node_count));
    std::vector<int> host_is_leaf(static_cast<size_t>(node_count));
    const size_t topology_bytes = static_cast<size_t>(node_count) * sizeof(int);
    cuda_check(cudaMemcpyAsync(host_left.data(), bvh.left_child.data_ptr<int>(),
                               topology_bytes, cudaMemcpyDeviceToHost, torch_ctx.stream),
               "cudaMemcpyAsync(triangle BVH left child)");
    cuda_check(cudaMemcpyAsync(host_right.data(), bvh.right_child.data_ptr<int>(),
                               topology_bytes, cudaMemcpyDeviceToHost, torch_ctx.stream),
               "cudaMemcpyAsync(triangle BVH right child)");
    cuda_check(cudaMemcpyAsync(host_is_leaf.data(), bvh.is_leaf.data_ptr<int>(),
                               topology_bytes, cudaMemcpyDeviceToHost, torch_ctx.stream),
               "cudaMemcpyAsync(triangle BVH leaf flags)");
    cuda_check(cudaStreamSynchronize(torch_ctx.stream),
               "cudaStreamSynchronize(triangle BVH topology guard)");
    std::vector<int> heights(static_cast<size_t>(node_count), -1);
    const int tree_height = rayd::shared::bvh::compute_node_height(
        0, host_left, host_right, host_is_leaf, heights);
    if (tree_height > rayd::shared::bvh::kBvhTraversalStackDepth)
        throw std::runtime_error(
            "CUDA triangle BVH height " + std::to_string(tree_height) +
            " exceeds traversal stack capacity " +
            std::to_string(rayd::shared::bvh::kBvhTraversalStackDepth) + ".");

    encode_raw_edge_bvh_cuda(
        primitive_count, bvh.left_child, bvh.right_child, bvh.leaf_primitive,
        bvh.leaf_primitives, torch_ctx.stream);
    cuda_check(cudaGetLastError(), "ensure_custom_triangle_bvh() kernel launch");
    cuda_check(cudaStreamSynchronize(torch_ctx.stream),
               "cudaStreamSynchronize(triangle BVH build)");
    bvh.valid = true;
}

std::vector<at::Tensor> scene_edge_records(c10::intrusive_ptr<SceneHandle> scene_handle) {
    return scene_edge_records(get_scene(scene_handle->handle));
}

std::vector<at::Tensor> scene_global_geometry(
    c10::intrusive_ptr<SceneHandle> scene_handle) {
    SceneCache &scene = get_scene(scene_handle->handle);
    at::Tensor face_normal = at::stack(
        {scene.tri_fn_x, scene.tri_fn_y, scene.tri_fn_z}, 1);
    at::Tensor squared_normal = face_normal.square().sum(1, true);
    at::Tensor inverse_normal = squared_normal
        .clamp_min(std::numeric_limits<float>::min())
        .rsqrt();
    face_normal = at::where(
        squared_normal.gt(0.0f), face_normal * inverse_normal,
        at::zeros_like(face_normal));
    at::Tensor global_prim_id = at::arange(
        scene.global_faces.size(0), scene.face_local_id.options());
    return {
        scene.global_vertices,
        scene.global_faces,
        face_normal,
        scene.face_shape_id,
        scene.face_local_id,
        global_prim_id,
    };
}

at::Tensor get_scene_edge_mask(c10::intrusive_ptr<SceneHandle> scene_handle) {
    SceneCache &scene = get_scene(scene_handle->handle);
    return scene.edge_mask.to(at::kBool);
}

void set_scene_edge_mask(
    c10::intrusive_ptr<SceneHandle> scene_handle,
    at::Tensor mask) {
    SceneCache &scene = get_scene(scene_handle->handle);
    require_cuda(mask, "edge_mask");
    require_contiguous(mask, "edge_mask");
    require_rank(mask, 1, "edge_mask");
    if (mask.get_device() != scene.device_index)
        throw std::runtime_error("set_scene_edge_mask(): mask must be on the scene device.");
    if (mask.numel() != scene.edge_v0.numel())
        throw std::runtime_error("set_scene_edge_mask(): mask length must equal edge count.");
    if (mask.scalar_type() != at::kBool && mask.scalar_type() != at::kByte)
        throw std::runtime_error("set_scene_edge_mask(): mask must have bool or uint8 dtype.");
    scene.edge_mask = mask.to(at::kByte).contiguous().clone();
    scene.edge_mask_version += 1;
    scene.version += 1;
}

void update_mesh_vertices(c10::intrusive_ptr<SceneHandle> scene, int64_t mesh_id, at::Tensor vertices) {
    update_mesh_vertices(scene->handle, mesh_id, std::move(vertices));
}

void sync_scene(c10::intrusive_ptr<SceneHandle> scene) {
    sync_scene(scene->handle);
}

} // namespace rayd::torch_backend
