#include <rayd/torch/scene/cache.h>
#include <rayd/torch/scene/cache_kernels.h>
#include <rayd/torch/common/tensor_check.h>
#include <rayd/torch/integration.h>

#include "../integration_internal.h"

#include <torch/extension.h>

#include <string>

namespace rayd::torch_backend {

namespace {

constexpr int64_t kMeshUseFaceNormals = 1;
constexpr int64_t kMeshEdgesEnabled = 2;
constexpr int64_t kMeshDynamic = 4;

void require_mesh_vertex_tangent(const at::Tensor &tensor, const MeshRecord &mesh, const char *name) {
    require_vec3f(tensor, name);
    if (tensor.size(0) != mesh.vertices.size(0)) {
        throw std::runtime_error(std::string(name) + " must match its mesh vertex count.");
    }
}

} // namespace

MeshRecord integration_mesh_record(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor uv,
    at::Tensor face_uv,
    at::Tensor to_world_left,
    at::Tensor to_world_right,
    int64_t flags) {
    MeshRecord record;
    record.vertices = std::move(vertices);
    record.faces = std::move(faces);
    record.uv = std::move(uv);
    record.face_uv = std::move(face_uv);
    record.to_world_left = std::move(to_world_left);
    record.to_world_right = std::move(to_world_right);
    record.use_face_normals = (flags & kMeshUseFaceNormals) != 0;
    record.edges_enabled = (flags & kMeshEdgesEnabled) != 0;
    record.dynamic = (flags & kMeshDynamic) != 0;
    return record;
}

c10::intrusive_ptr<SceneHandle> create_scene_cache_from_flat(
    std::vector<at::Tensor> vertices,
    std::vector<at::Tensor> faces,
    std::vector<at::Tensor> uv,
    std::vector<at::Tensor> face_uv,
    std::vector<at::Tensor> to_world_left,
    std::vector<at::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags) {
    const size_t mesh_count = vertices.size();
    if (faces.size() != mesh_count ||
        uv.size() != mesh_count ||
        face_uv.size() != mesh_count ||
        to_world_left.size() != mesh_count ||
        to_world_right.size() != mesh_count ||
        mesh_flags.size() != mesh_count) {
        throw std::runtime_error("Scene init lists must have the same length.");
    }
    std::vector<MeshRecord> meshes;
    meshes.reserve(mesh_count);
    for (size_t i = 0; i < mesh_count; ++i) {
        meshes.push_back(integration_mesh_record(
            std::move(vertices[i]),
            std::move(faces[i]),
            std::move(uv[i]),
            std::move(face_uv[i]),
            std::move(to_world_left[i]),
            std::move(to_world_right[i]),
            mesh_flags[i]));
    }
    const int64_t handle = create_scene(std::move(meshes));
    return c10::make_intrusive<SceneHandle>(handle);
}

int64_t create_scene_op(py::list mesh_specs) {
    std::vector<MeshRecord> meshes;
    meshes.reserve(py::len(mesh_specs));
    for (py::handle item : mesh_specs) {
        py::dict spec = py::reinterpret_borrow<py::dict>(item);
        MeshRecord record;
        record.vertices = spec["vertices"].cast<at::Tensor>();
        record.faces = spec["faces"].cast<at::Tensor>();
        record.uv = spec["uv"].cast<at::Tensor>();
        record.face_uv = spec["face_uv"].cast<at::Tensor>();
        record.to_world_left = spec["to_world_left"].cast<at::Tensor>();
        record.to_world_right = spec["to_world_right"].cast<at::Tensor>();
        record.use_face_normals = spec["use_face_normals"].cast<bool>();
        record.edges_enabled = spec["edges_enabled"].cast<bool>();
        record.dynamic = spec["dynamic"].cast<bool>();
        meshes.push_back(record);
    }
    return create_scene(std::move(meshes));
}

py::tuple split_scene_vertex_grad_op(int64_t handle, at::Tensor grad_vertices) {
    std::vector<at::Tensor> parts =
        split_scene_vertex_grad(c10::make_intrusive<SceneHandle>(handle, false), grad_vertices);
    py::tuple result(parts.size());
    for (size_t i = 0; i < parts.size(); ++i)
        result[i] = parts[i];
    return result;
}

std::vector<at::Tensor> split_scene_vertex_grad(
    c10::intrusive_ptr<SceneHandle> scene_handle,
    at::Tensor grad_vertices) {
    SceneCache &scene = get_scene(scene_handle->handle);
    require_vec3f(grad_vertices, "grad_vertices");
    if (grad_vertices.size(0) != scene.global_vertices.size(0)) {
        throw std::runtime_error("grad_vertices must match scene global vertex count.");
    }

    std::vector<at::Tensor> result;
    result.reserve(scene.meshes.size());
    int64_t vertex_offset = 0;
    for (size_t mesh_index = 0; mesh_index < scene.meshes.size(); ++mesh_index) {
        const int64_t vertex_count = scene.meshes[mesh_index].vertices.size(0);
        result.push_back(grad_vertices.narrow(0, vertex_offset, vertex_count));
        vertex_offset += vertex_count;
    }
    return result;
}

py::object pack_scene_vertex_tangents_op(int64_t handle, py::args tangent_args) {
    c10::intrusive_ptr<SceneHandle> scene = c10::make_intrusive<SceneHandle>(handle, false);
    SceneCache &cache = get_scene(handle);
    if (static_cast<size_t>(py::len(tangent_args)) != cache.meshes.size()) {
        throw std::runtime_error("pack_scene_vertex_tangents() expects one tangent per mesh.");
    }
    std::vector<c10::optional<at::Tensor>> tangents;
    tangents.reserve(py::len(tangent_args));
    for (size_t mesh_index = 0; mesh_index < cache.meshes.size(); ++mesh_index) {
        if (tangent_args[mesh_index].is_none())
            tangents.emplace_back(c10::nullopt);
        else
            tangents.emplace_back(tangent_args[mesh_index].cast<at::Tensor>());
    }
    at::Tensor packed = pack_scene_vertex_tangents(scene, std::move(tangents));
    if (!packed.defined())
        return py::none();
    return py::cast(packed);
}

at::Tensor pack_scene_vertex_tangents(
    c10::intrusive_ptr<SceneHandle> scene_handle,
    std::vector<c10::optional<at::Tensor>> tangents) {
    SceneCache &scene = get_scene(scene_handle->handle);
    if (tangents.size() != scene.meshes.size()) {
        throw std::runtime_error("pack_scene_vertex_tangents() expects one tangent per mesh.");
    }
    bool any_tangent = false;
    for (const c10::optional<at::Tensor> &tangent : tangents) {
        if (tangent.has_value() && tangent->defined() && tangent->numel() != 0) {
            any_tangent = true;
            break;
        }
    }
    if (!any_tangent)
        return at::Tensor();

    at::Tensor global_tangent = at::empty_like(scene.global_vertices);
    int64_t vertex_offset = 0;
    for (size_t mesh_index = 0; mesh_index < scene.meshes.size(); ++mesh_index) {
        const MeshRecord &mesh = scene.meshes[mesh_index];
        const int64_t vertex_count = mesh.vertices.size(0);
        const c10::optional<at::Tensor> &tangent_obj = tangents[mesh_index];
        if (!tangent_obj.has_value() || !tangent_obj->defined() || tangent_obj->numel() == 0) {
            zero_global_vertex_tangent_range_cuda(vertex_offset, vertex_count, global_tangent);
        } else {
            at::Tensor tangent = *tangent_obj;
            require_mesh_vertex_tangent(tangent, mesh, "mesh tangent");
            pack_global_vertex_tangent_cuda(tangent, vertex_offset, vertex_count, global_tangent);
        }
        vertex_offset += vertex_count;
    }
    return global_tangent;
}

} // namespace rayd::torch_backend

namespace rayd::torch {

class SceneResource::Impl final {
public:
    explicit Impl(std::unique_ptr<torch_backend::SceneCache> scene)
        : scene(std::move(scene)) {}

    std::unique_ptr<torch_backend::SceneCache> scene;
};

SceneResource::SceneResource(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
SceneResource::SceneResource(SceneResource &&) noexcept = default;
SceneResource &SceneResource::operator=(SceneResource &&) noexcept = default;
SceneResource::~SceneResource() noexcept = default;

bool SceneResource::valid() const noexcept {
    return impl_ != nullptr && impl_->scene != nullptr;
}

int SceneResource::device_index() const {
    return static_cast<int>(detail::IntegrationAccess::scene_cache(*this).device_index);
}

torch_backend::SceneCache &detail::IntegrationAccess::scene_cache(const SceneResource &scene) {
    if (!scene.impl_ || !scene.impl_->scene)
        throw std::runtime_error("rayd::torch operation received an invalid SceneResource");
    return *scene.impl_->scene;
}

SceneResource create_scene(std::vector<MeshInput> meshes) {
    if (meshes.empty())
        throw std::runtime_error("rayd::torch::create_scene requires at least one mesh");
    std::vector<torch_backend::MeshRecord> records;
    records.reserve(meshes.size());
    for (MeshInput &mesh : meshes) {
        int64_t flags = 0;
        if (mesh.use_face_normals)
            flags |= 1;
        if (mesh.edges_enabled)
            flags |= 2;
        if (mesh.dynamic)
            flags |= 4;
        records.push_back(torch_backend::integration_mesh_record(
            std::move(mesh.vertices),
            std::move(mesh.faces),
            std::move(mesh.uv),
            std::move(mesh.face_uv),
            std::move(mesh.to_world_left),
            std::move(mesh.to_world_right),
            flags));
    }
    auto owner = torch_backend::create_scene_cache(std::move(records));
    return SceneResource(std::make_unique<SceneResource::Impl>(std::move(owner)));
}

SceneEdgeRecordsResult scene_edge_records(const SceneResource &scene) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    std::vector<at::Tensor> values = torch_backend::scene_edge_records(cache);
    if (values.size() != 12)
        throw std::runtime_error("rayd::torch::scene_edge_records returned an unexpected output count");
    return {
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
    };
}

} // namespace rayd::torch
