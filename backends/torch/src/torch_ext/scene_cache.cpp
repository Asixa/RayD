#include <raydtorch/scene_cache.h>
#include <raydtorch/optix_context.h>
#include <raydtorch/tensor_check.h>

#include <atomic>
#include <stdexcept>

namespace raydtorch {

namespace {
std::atomic<int64_t> next_handle{1};
std::mutex scenes_mutex;
std::unordered_map<int64_t, std::unique_ptr<SceneCache>> scenes;
} // namespace

int64_t create_scene(std::vector<MeshRecord> meshes) {
    if (meshes.empty())
        throw std::runtime_error("Scene.build(): at least one mesh is required.");

    const int64_t device_index = meshes[0].vertices.get_device();
    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (torch_ctx.device_index != device_index)
        throw std::runtime_error("Scene.build(): current CUDA device does not match mesh tensors.");
    get_optix_context(static_cast<int>(device_index));

    for (const MeshRecord &mesh : meshes) {
        require_vec3f(mesh.vertices, "mesh.vertices");
        require_vec3i(mesh.faces, "mesh.faces");
        if (mesh.vertices.get_device() != device_index)
            throw std::runtime_error("Scene.build(): all tensors must be on the same CUDA device.");
    }

    auto scene = std::make_unique<SceneCache>();
    scene->handle = next_handle.fetch_add(1);
    scene->device_index = device_index;
    scene->meshes = std::move(meshes);
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

} // namespace raydtorch
