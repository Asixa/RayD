#include <raydtorch/scene/cache.h>
#include <raydtorch/scene/cache_kernels.h>
#include <raydtorch/common/tensor_check.h>

#include <torch/extension.h>

#include <string>

namespace raydtorch {

namespace {

void require_mesh_vertex_tangent(const at::Tensor &tensor, const MeshRecord &mesh, const char *name) {
    require_vec3f(tensor, name);
    if (tensor.size(0) != mesh.vertices.size(0)) {
        throw std::runtime_error(std::string(name) + " must match its mesh vertex count.");
    }
}

} // namespace

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
    require_vec3f(grad_vertices, "grad_vertices");
    SceneCache &scene = get_scene(handle);
    if (grad_vertices.size(0) != scene.global_vertices.size(0)) {
        throw std::runtime_error("grad_vertices must match scene global vertex count.");
    }

    py::tuple result(scene.meshes.size());
    int64_t vertex_offset = 0;
    for (size_t mesh_index = 0; mesh_index < scene.meshes.size(); ++mesh_index) {
        const int64_t vertex_count = scene.meshes[mesh_index].vertices.size(0);
        result[mesh_index] = grad_vertices.narrow(0, vertex_offset, vertex_count);
        vertex_offset += vertex_count;
    }
    return result;
}

py::object pack_scene_vertex_tangents_op(int64_t handle, py::args tangent_args) {
    SceneCache &scene = get_scene(handle);
    if (static_cast<size_t>(py::len(tangent_args)) != scene.meshes.size()) {
        throw std::runtime_error("pack_scene_vertex_tangents() expects one tangent per mesh.");
    }
    bool any_tangent = false;
    for (size_t mesh_index = 0; mesh_index < scene.meshes.size(); ++mesh_index) {
        if (!tangent_args[mesh_index].is_none()) {
            any_tangent = true;
            break;
        }
    }
    if (!any_tangent) {
        return py::none();
    }

    at::Tensor global_tangent = at::empty_like(scene.global_vertices);
    int64_t vertex_offset = 0;
    for (size_t mesh_index = 0; mesh_index < scene.meshes.size(); ++mesh_index) {
        const MeshRecord &mesh = scene.meshes[mesh_index];
        const int64_t vertex_count = mesh.vertices.size(0);
        py::handle tangent_obj = tangent_args[mesh_index];
        if (tangent_obj.is_none()) {
            zero_global_vertex_tangent_range_cuda(vertex_offset, vertex_count, global_tangent);
        } else {
            at::Tensor tangent = tangent_obj.cast<at::Tensor>();
            require_mesh_vertex_tangent(tangent, mesh, "mesh tangent");
            pack_global_vertex_tangent_cuda(tangent, vertex_offset, vertex_count, global_tangent);
        }
        vertex_offset += vertex_count;
    }
    return py::cast(global_tangent);
}

void bind_scene_ops(py::module_ &m) {
    m.def("create_scene", &create_scene_op);
    m.def("destroy_scene", &destroy_scene);
    m.def("scene_version", &scene_version);
    m.def("scene_num_meshes", &scene_num_meshes);
    m.def("scene_edge_count", &scene_edge_count);
    m.def("split_scene_vertex_grad", &split_scene_vertex_grad_op);
    m.def("pack_scene_vertex_tangents", &pack_scene_vertex_tangents_op);
    m.def("update_mesh_vertices", &update_mesh_vertices);
    m.def("sync_scene", &sync_scene);
}

} // namespace raydtorch
