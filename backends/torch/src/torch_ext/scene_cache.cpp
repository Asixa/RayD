#include <raydtorch/scene_cache.h>
#include <raydtorch/optix_context.h>
#include <raydtorch/tensor_check.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <map>
#include <stdexcept>
#include <string>

namespace raydtorch {

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
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes = {};
    raydtorch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    at::TensorOptions byte_options =
        at::TensorOptions().device(mesh.vertices.device()).dtype(at::kByte);
    accel.gas_temp_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
    accel.gas_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);

    raydtorch_OPTIX_CHECK(optixAccelBuild(
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
        nullptr,
        0));

    return accel;
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
    raydtorch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
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

    raydtorch_OPTIX_CHECK(optixAccelBuild(
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

void build_edge_topology(SceneCache &scene) {
    std::vector<int32_t> edge_v0;
    std::vector<int32_t> edge_v1;
    std::vector<int32_t> edge_face0;
    std::vector<int32_t> edge_face1;
    std::vector<int32_t> edge_shape_id;
    std::vector<int32_t> edge_local_id;
    std::map<std::pair<int32_t, int32_t>, int32_t> edge_lookup;

    for (int32_t shape_id = 0; shape_id < static_cast<int32_t>(scene.meshes.size()); ++shape_id) {
        at::Tensor faces_cpu = scene.meshes[shape_id].faces.cpu();
        const int *faces = faces_cpu.data_ptr<int>();
        for (int32_t face_id = 0; face_id < static_cast<int32_t>(faces_cpu.size(0)); ++face_id) {
            const int32_t tri[3] = {
                static_cast<int32_t>(faces[face_id * 3 + 0]),
                static_cast<int32_t>(faces[face_id * 3 + 1]),
                static_cast<int32_t>(faces[face_id * 3 + 2]),
            };
            const int32_t pairs[3][2] = {{tri[0], tri[1]}, {tri[1], tri[2]}, {tri[2], tri[0]}};
            for (int local_edge = 0; local_edge < 3; ++local_edge) {
                int32_t a = pairs[local_edge][0];
                int32_t b = pairs[local_edge][1];
                auto key = std::minmax(a, b);
                auto it = edge_lookup.find(key);
                if (it == edge_lookup.end()) {
                    const int32_t edge_id = static_cast<int32_t>(edge_v0.size());
                    edge_lookup.emplace(key, edge_id);
                    edge_v0.push_back(a);
                    edge_v1.push_back(b);
                    edge_face0.push_back(face_id);
                    edge_face1.push_back(-1);
                    edge_shape_id.push_back(shape_id);
                    edge_local_id.push_back(edge_id);
                } else if (edge_face1[it->second] < 0) {
                    edge_face1[it->second] = face_id;
                }
            }
        }
    }

    at::TensorOptions cpu_iopts = at::TensorOptions().device(at::kCPU).dtype(at::kInt);
    at::Device device(at::kCUDA, scene.device_index);
    scene.edge_v0 = at::tensor(edge_v0, cpu_iopts).to(device);
    scene.edge_v1 = at::tensor(edge_v1, cpu_iopts).to(device);
    scene.edge_face0 = at::tensor(edge_face0, cpu_iopts).to(device);
    scene.edge_face1 = at::tensor(edge_face1, cpu_iopts).to(device);
    scene.edge_shape_id = at::tensor(edge_shape_id, cpu_iopts).to(device);
    scene.edge_local_id = at::tensor(edge_local_id, cpu_iopts).to(device);
}

void build_edge_accel(SceneCache &scene, OptixDeviceContext optix_context, cudaStream_t stream) {
    const int64_t edge_count = scene.edge_v0.size(0);
    if (edge_count == 0)
        return;

    constexpr float search_radius = 1.0e10f;
    at::Tensor vertices_cpu = scene.meshes[0].vertices.cpu();
    at::Tensor edge_v0_cpu = scene.edge_v0.cpu();
    at::Tensor edge_v1_cpu = scene.edge_v1.cpu();
    const float *vertices = vertices_cpu.data_ptr<float>();
    const int *edge_v0 = edge_v0_cpu.data_ptr<int>();
    const int *edge_v1 = edge_v1_cpu.data_ptr<int>();

    std::vector<float> aabb_host(static_cast<size_t>(edge_count) * 6);
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        const int i0 = edge_v0[edge];
        const int i1 = edge_v1[edge];
        for (int axis = 0; axis < 3; ++axis) {
            const float a = vertices[i0 * 3 + axis];
            const float b = vertices[i1 * 3 + axis];
            aabb_host[edge * 6 + axis] = std::min(a, b) - search_radius;
            aabb_host[edge * 6 + 3 + axis] = std::max(a, b) + search_radius;
        }
    }

    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, scene.device_index)).dtype(at::kByte);
    scene.edge_accel.aabb_buffer =
        at::empty({static_cast<int64_t>(aabb_host.size() * sizeof(float))}, byte_options);
    cuda_check(
        cudaMemcpyAsync(
            scene.edge_accel.aabb_buffer.data_ptr<uint8_t>(),
            aabb_host.data(),
            aabb_host.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(edge AABB)");

    CUdeviceptr aabb_buffer =
        reinterpret_cast<CUdeviceptr>(scene.edge_accel.aabb_buffer.data_ptr<uint8_t>());
    uint32_t input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
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
    raydtorch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    scene.edge_accel.gas_temp_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
    scene.edge_accel.gas_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);
    raydtorch_OPTIX_CHECK(optixAccelBuild(
        optix_context,
        stream,
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(scene.edge_accel.gas_temp_buffer.data_ptr<uint8_t>()),
        buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(scene.edge_accel.gas_buffer.data_ptr<uint8_t>()),
        buffer_sizes.outputSizeInBytes,
        &scene.edge_accel.traversable,
        nullptr,
        0));
    scene.edge_accel.search_radius = search_radius;
}
} // namespace

int64_t create_scene(std::vector<MeshRecord> meshes) {
    if (meshes.empty())
        throw std::runtime_error("Scene.build(): at least one mesh is required.");

    const int64_t device_index = meshes[0].vertices.get_device();
    c10::cuda::CUDAGuard guard(static_cast<int>(device_index));
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (torch_ctx.device_index != device_index)
        throw std::runtime_error("Scene.build(): current CUDA device does not match mesh tensors.");
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(device_index));

    for (const MeshRecord &mesh : meshes) {
        require_vec3f(mesh.vertices, "mesh.vertices");
        require_vec3i(mesh.faces, "mesh.faces");
        if (mesh.vertices.get_device() != device_index || mesh.faces.get_device() != device_index)
            throw std::runtime_error("Scene.build(): all tensors must be on the same CUDA device.");
    }

    auto scene = std::make_unique<SceneCache>();
    scene->handle = next_handle.fetch_add(1);
    scene->device_index = device_index;
    scene->meshes = std::move(meshes);
    scene->triangle_accels.reserve(scene->meshes.size());
    for (const MeshRecord &mesh : scene->meshes)
        scene->triangle_accels.push_back(
            build_triangle_accel(mesh, optix_entry.optix_context, torch_ctx.stream));
    build_edge_topology(*scene);
    build_edge_accel(*scene, optix_entry.optix_context, torch_ctx.stream);
    const int64_t handle = scene->handle;

    std::lock_guard<std::mutex> lock(scenes_mutex);
    scenes.emplace(handle, std::move(scene));
    return handle;
}

void destroy_scene(int64_t handle) {
    if (handle == 0)
        return;
    std::lock_guard<std::mutex> lock(scenes_mutex);
    scenes.erase(handle);
}

SceneCache &get_scene(int64_t handle) {
    std::lock_guard<std::mutex> lock(scenes_mutex);
    auto it = scenes.find(handle);
    if (it == scenes.end())
        throw std::runtime_error("Invalid RayDTorch scene handle.");
    return *it->second;
}

int64_t scene_version(int64_t handle) {
    return get_scene(handle).version;
}

int64_t scene_num_meshes(int64_t handle) {
    return static_cast<int64_t>(get_scene(handle).meshes.size());
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
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));

    bool changed = false;
    for (int64_t mesh_id = 0; mesh_id < static_cast<int64_t>(scene.meshes.size()); ++mesh_id) {
        MeshRecord &mesh = scene.meshes[mesh_id];
        if (!mesh.pending_update)
            continue;
        update_triangle_accel(
            mesh,
            scene.triangle_accels[mesh_id],
            optix_entry.optix_context,
            torch_ctx.stream);
        mesh.pending_update = false;
        changed = true;
    }
    if (changed) {
        scene.version += 1;
        scene.edge_version += 1;
    }
}

} // namespace raydtorch
