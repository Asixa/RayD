#include <raydtorch/scene_cache.h>
#include <raydtorch/optix_context.h>
#include <raydtorch/tensor_check.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

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
    raydtorch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context, &accel_options, &build_input, 1, &buffer_sizes));

    scene.triangle_ias.ias_temp_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.tempSizeInBytes)}, byte_options);
    scene.triangle_ias.ias_buffer =
        at::empty({static_cast<int64_t>(buffer_sizes.outputSizeInBytes)}, byte_options);

    raydtorch_OPTIX_CHECK(optixAccelBuild(
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
    std::vector<int32_t> local_edge_counts(scene.meshes.size(), 0);
    std::map<std::tuple<int32_t, int32_t, int32_t>, int32_t> edge_lookup;

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
                auto sorted = std::minmax(a, b);
                auto key = std::make_tuple(shape_id, sorted.first, sorted.second);
                auto it = edge_lookup.find(key);
                if (it == edge_lookup.end()) {
                    const int32_t edge_id = static_cast<int32_t>(edge_v0.size());
                    edge_lookup.emplace(key, edge_id);
                    edge_v0.push_back(a);
                    edge_v1.push_back(b);
                    edge_face0.push_back(face_id);
                    edge_face1.push_back(-1);
                    edge_shape_id.push_back(shape_id);
                    edge_local_id.push_back(local_edge_counts[shape_id]++);
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

std::vector<float> compute_edge_search_radii(
    const std::vector<float> &p0_x,
    const std::vector<float> &p0_y,
    const std::vector<float> &p0_z,
    const std::vector<float> &e1_x,
    const std::vector<float> &e1_y,
    const std::vector<float> &e1_z) {
    const int64_t edge_count = static_cast<int64_t>(p0_x.size());
    if (edge_count <= 0)
        return {};

    float min_x = std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float min_z = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();
    float max_edge_length = 0.0f;

    for (int64_t edge = 0; edge < edge_count; ++edge) {
        const float x0 = p0_x[edge];
        const float y0 = p0_y[edge];
        const float z0 = p0_z[edge];
        const float ex = e1_x[edge];
        const float ey = e1_y[edge];
        const float ez = e1_z[edge];
        const float x1 = x0 + ex;
        const float y1 = y0 + ey;
        const float z1 = z0 + ez;
        min_x = std::min(min_x, std::min(x0, x1));
        min_y = std::min(min_y, std::min(y0, y1));
        min_z = std::min(min_z, std::min(z0, z1));
        max_x = std::max(max_x, std::max(x0, x1));
        max_y = std::max(max_y, std::max(y0, y1));
        max_z = std::max(max_z, std::max(z0, z1));
        max_edge_length = std::max(max_edge_length, std::sqrt(ex * ex + ey * ey + ez * ez));
    }

    const float dx = std::max(max_x - min_x, 0.0f);
    const float dy = std::max(max_y - min_y, 0.0f);
    const float dz = std::max(max_z - min_z, 0.0f);
    const float full_radius = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0e-3f);
    const float edge_scale = std::max(max_edge_length, full_radius * 1.0e-4f);

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
    at::Tensor edge_v0_cpu = scene.edge_v0.cpu();
    at::Tensor edge_v1_cpu = scene.edge_v1.cpu();
    at::Tensor edge_shape_cpu = scene.edge_shape_id.cpu();
    const int *edge_v0 = edge_v0_cpu.data_ptr<int>();
    const int *edge_v1 = edge_v1_cpu.data_ptr<int>();
    const int *edge_shape = edge_shape_cpu.data_ptr<int>();

    std::vector<at::Tensor> vertices_cpu;
    vertices_cpu.reserve(scene.meshes.size());
    for (const MeshRecord &mesh : scene.meshes)
        vertices_cpu.push_back(mesh.vertices.cpu());

    std::vector<float> p0_x(edge_count);
    std::vector<float> p0_y(edge_count);
    std::vector<float> p0_z(edge_count);
    std::vector<float> e1_x(edge_count);
    std::vector<float> e1_y(edge_count);
    std::vector<float> e1_z(edge_count);
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        const int shape_id = edge_shape[edge];
        const float *vertices = vertices_cpu[shape_id].data_ptr<float>();
        const int i0 = edge_v0[edge];
        const int i1 = edge_v1[edge];
        const float x0 = vertices[i0 * 3 + 0];
        const float y0 = vertices[i0 * 3 + 1];
        const float z0 = vertices[i0 * 3 + 2];
        const float x1 = vertices[i1 * 3 + 0];
        const float y1 = vertices[i1 * 3 + 1];
        const float z1 = vertices[i1 * 3 + 2];
        p0_x[edge] = x0;
        p0_y[edge] = y0;
        p0_z[edge] = z0;
        e1_x[edge] = x1 - x0;
        e1_y[edge] = y1 - y0;
        e1_z[edge] = z1 - z0;
    }

    at::TensorOptions cpu_fopts = at::TensorOptions().device(at::kCPU).dtype(at::kFloat);
    at::Device device(at::kCUDA, scene.device_index);
    scene.edge_p0_x = at::tensor(p0_x, cpu_fopts).to(device);
    scene.edge_p0_y = at::tensor(p0_y, cpu_fopts).to(device);
    scene.edge_p0_z = at::tensor(p0_z, cpu_fopts).to(device);
    scene.edge_e1_x = at::tensor(e1_x, cpu_fopts).to(device);
    scene.edge_e1_y = at::tensor(e1_y, cpu_fopts).to(device);
    scene.edge_e1_z = at::tensor(e1_z, cpu_fopts).to(device);
    scene.edge_mask = at::ones({edge_count}, at::TensorOptions().device(device).dtype(at::kByte));
}

void build_edge_accel(SceneCache &scene, OptixDeviceContext optix_context, cudaStream_t stream) {
    const int64_t edge_count = scene.edge_v0.size(0);
    refresh_edge_soa(scene);
    scene.edge_accels.clear();
    if (edge_count == 0) {
        scene.edge_accel = {};
        return;
    }

    std::vector<float> p0_x(edge_count);
    std::vector<float> p0_y(edge_count);
    std::vector<float> p0_z(edge_count);
    std::vector<float> e1_x(edge_count);
    std::vector<float> e1_y(edge_count);
    std::vector<float> e1_z(edge_count);
    std::memcpy(p0_x.data(), scene.edge_p0_x.cpu().data_ptr<float>(), p0_x.size() * sizeof(float));
    std::memcpy(p0_y.data(), scene.edge_p0_y.cpu().data_ptr<float>(), p0_y.size() * sizeof(float));
    std::memcpy(p0_z.data(), scene.edge_p0_z.cpu().data_ptr<float>(), p0_z.size() * sizeof(float));
    std::memcpy(e1_x.data(), scene.edge_e1_x.cpu().data_ptr<float>(), e1_x.size() * sizeof(float));
    std::memcpy(e1_y.data(), scene.edge_e1_y.cpu().data_ptr<float>(), e1_y.size() * sizeof(float));
    std::memcpy(e1_z.data(), scene.edge_e1_z.cpu().data_ptr<float>(), e1_z.size() * sizeof(float));

    std::vector<float> radii = compute_edge_search_radii(p0_x, p0_y, p0_z, e1_x, e1_y, e1_z);
    scene.edge_accels.resize(radii.size());
    at::TensorOptions byte_options =
        at::TensorOptions().device(at::Device(at::kCUDA, scene.device_index)).dtype(at::kByte);

    for (size_t gas_index = 0; gas_index < radii.size(); ++gas_index) {
        OptixEdgeAccel &accel = scene.edge_accels[gas_index];
        const float radius = radii[gas_index];
        std::vector<float> aabb_host(static_cast<size_t>(edge_count) * 6);
        for (int64_t edge = 0; edge < edge_count; ++edge) {
            const float x0 = p0_x[edge];
            const float y0 = p0_y[edge];
            const float z0 = p0_z[edge];
            const float x1 = x0 + e1_x[edge];
            const float y1 = y0 + e1_y[edge];
            const float z1 = z0 + e1_z[edge];
            aabb_host[edge * 6 + 0] = std::min(x0, x1) - radius;
            aabb_host[edge * 6 + 1] = std::min(y0, y1) - radius;
            aabb_host[edge * 6 + 2] = std::min(z0, z1) - radius;
            aabb_host[edge * 6 + 3] = std::max(x0, x1) + radius;
            aabb_host[edge * 6 + 4] = std::max(y0, y1) + radius;
            aabb_host[edge * 6 + 5] = std::max(z0, z1) + radius;
        }

        accel.aabb_buffer =
            at::empty({static_cast<int64_t>(aabb_host.size() * sizeof(float))}, byte_options);
        cuda_check(
            cudaMemcpyAsync(
                accel.aabb_buffer.data_ptr<uint8_t>(),
                aabb_host.data(),
                aabb_host.size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync(edge AABB)");

        CUdeviceptr aabb_buffer =
            reinterpret_cast<CUdeviceptr>(accel.aabb_buffer.data_ptr<uint8_t>());
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
        raydtorch_OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context, &accel_options, &build_input, 1, &buffer_sizes));

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
        accel.search_radius = radius;
    }
    scene.edge_accel = scene.edge_accels.back();
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
    build_triangle_ias(*scene, optix_entry.optix_context, torch_ctx.stream);
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
        build_triangle_ias(scene, optix_entry.optix_context, torch_ctx.stream);
        build_edge_accel(scene, optix_entry.optix_context, torch_ctx.stream);
        scene.version += 1;
        scene.edge_version += 1;
    }
}

} // namespace raydtorch
