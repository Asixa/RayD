#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace raydtorch {

struct MeshRecord {
    at::Tensor vertices;
    at::Tensor faces;
    at::Tensor uv;
    at::Tensor face_uv;
    at::Tensor to_world_left;
    at::Tensor to_world_right;
    bool use_face_normals = false;
    bool edges_enabled = true;
    bool dynamic = false;
};

struct SceneCache {
    int64_t handle = 0;
    int64_t version = 1;
    int64_t edge_version = 1;
    int64_t device_index = 0;
    std::vector<MeshRecord> meshes;
};

int64_t create_scene(std::vector<MeshRecord> meshes);
void destroy_scene(int64_t handle);
SceneCache &get_scene(int64_t handle);
int64_t scene_version(int64_t handle);
int64_t scene_num_meshes(int64_t handle);

} // namespace raydtorch
