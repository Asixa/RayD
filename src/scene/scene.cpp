#include <algorithm>
#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include <rayd/scene/scene.h>
#include <rayd/edge/scene_edge.h>

#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>

#include "scene_internal.h"

namespace rayd {

namespace {

enum class OptixSplitMode {
    Auto,
    Off,
    On
};

OptixSplitMode active_optix_split_mode() {
    static const OptixSplitMode value = []() {
        const char *raw = std::getenv("RAYD_OPTIX_SPLIT_MODE");
        const std::string normalized = normalize_optix_split_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return normalized.empty() ? OptixSplitMode::Off : OptixSplitMode::Auto;
        }
        if (normalized == "off" || normalized == "false" || normalized == "0") {
            return OptixSplitMode::Off;
        }
        if (normalized == "on" || normalized == "true" || normalized == "1") {
            return OptixSplitMode::On;
        }
        throw std::runtime_error(
            "Invalid RAYD_OPTIX_SPLIT_MODE. Expected one of: auto, off, on.");
    }();
    return value;
}

std::string normalize_edge_backend_value(const std::string &value) {
    std::string normalized = value;
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

EdgeBVHBackend parse_edge_backend(const std::string &value) {
    const std::string normalized = normalize_edge_backend_value(value);
    if (normalized.empty() || normalized == "drjit" ||
        normalized == "dr_jit" || normalized == "software") {
        return EdgeBVHBackend::DrJit;
    }
    if (normalized == "optix" || normalized == "custom_aabb") {
        return EdgeBVHBackend::Optix;
    }
    if (normalized == "hybrid" || normalized == "mixed" ||
        normalized == "optix_ray" || normalized == "ray_optix") {
        return EdgeBVHBackend::Hybrid;
    }
    throw std::runtime_error(
        "Invalid edge_bvh_backend. Expected one of: 'drjit', 'optix', 'hybrid'.");
}

const char *edge_backend_name(EdgeBVHBackend backend) {
    switch (backend) {
    case EdgeBVHBackend::DrJit:
        return "drjit";
    case EdgeBVHBackend::Optix:
        return "optix";
    case EdgeBVHBackend::Hybrid:
        return "hybrid";
    }
    return "drjit";
}

bool edge_backend_builds_drjit(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::DrJit ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_builds_optix(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix ||
           backend == EdgeBVHBackend::Hybrid;
}

bool should_split_optix_scene(OptixSplitMode mode,
                              int static_mesh_count,
                              int dynamic_mesh_count) {
    if (static_mesh_count == 0 || dynamic_mesh_count == 0) {
        return false;
    }
    if (mode == OptixSplitMode::On) {
        return true;
    }
    if (mode == OptixSplitMode::Off) {
        return false;
    }

    // The measured mixed-scene query tax is still too large to justify enabling
    // split mode automatically. Keep "on" available for calibration, but bias
    // "auto" to the stable single-scene path until a better heuristic exists.
    return false;
}

int face_edge_slot(const std::array<int, 3> &face_vertices, int v0, int v1) {
    auto matches = [v0, v1](int a, int b) {
        return (a == v0 && b == v1) || (a == v1 && b == v0);
    };

    if (matches(face_vertices[0], face_vertices[1])) {
        return 0;
    }
    if (matches(face_vertices[1], face_vertices[2])) {
        return 1;
    }
    if (matches(face_vertices[2], face_vertices[0])) {
        return 2;
    }
    return -1;
}

int face_opposite_vertex(const std::array<int, 3> &face_vertices, int v0, int v1) {
    for (int vertex : face_vertices) {
        if (vertex != v0 && vertex != v1) {
            return vertex;
        }
    }
    return -1;
}

} // namespace

Scene::Scene(const std::string &edge_bvh_backend)
    : optix_scene_(std::make_unique<OptixScene>()),
      optix_static_scene_(std::make_unique<OptixScene>()),
      optix_dynamic_scene_(std::make_unique<OptixScene>()),
      edge_bvh_(std::make_unique<SceneEdge>()),
      edge_optix_(std::make_unique<SceneEdgeOptix>()),
      edge_bvh_backend_(parse_edge_backend(edge_bvh_backend)) {}

Scene::~Scene() = default;

std::string Scene::to_string() const {
    std::stringstream stream;
    stream << "Scene[num_meshes=" << mesh_count_
           << ", ready=" << is_ready()
           << ", pending_updates=" << pending_updates_
           << "]";
    return stream.str();
}

std::vector<const Mesh *> Scene::meshes() const {
    std::vector<const Mesh *> result;
    result.reserve(mesh_records_.size());
    for (const SceneMeshRecord &record : mesh_records_) {
        result.push_back(record.mesh.get());
    }
    return result;
}

int Scene::add_mesh(const Mesh &mesh, bool dynamic) {
    SceneMeshRecord record;
    record.mesh = std::make_unique<Mesh>(mesh);
    record.mesh->set_mesh_id(static_cast<int>(mesh_records_.size()));
    record.dynamic = dynamic;
    mesh_records_.push_back(std::move(record));

    mesh_count_ = static_cast<int>(mesh_records_.size());
    is_ready_ = false;
    pending_updates_ = false;
    vertex_offsets_ = IntDetached();
    global_geometry_ = SceneGlobalGeometry();
    edge_mask_ = MaskDetached();
    pending_edge_bvh_dirty_ranges_.clear();
    edge_bvh_dirty_ = false;
    mask_dirty_ = false;
    optix_split_active_ = false;
    optix_static_mesh_indices_.clear();
    optix_dynamic_mesh_indices_.clear();
    optix_dynamic_mesh_local_index_.clear();
    reflection_pipeline_.reset();
    reflection_epc_pipeline_.reset();
    reflection_epc_geometry_ready_ = false;
    segment_visibility_pipeline_.reset();
    return mesh_count_ - 1;
}

Scene::SceneMeshRecord &Scene::mesh_record(int mesh_id) {
    require(mesh_id >= 0 && mesh_id < static_cast<int>(mesh_records_.size()),
            "Scene: mesh_id is out of range.");
    return mesh_records_[static_cast<size_t>(mesh_id)];
}

const Scene::SceneMeshRecord &Scene::mesh_record(int mesh_id) const {
    require(mesh_id >= 0 && mesh_id < static_cast<int>(mesh_records_.size()),
            "Scene: mesh_id is out of range.");
    return mesh_records_[static_cast<size_t>(mesh_id)];
}

void Scene::scatter_mesh_data(const SceneMeshRecord &record, bool include_static) {
    const Mesh &mesh = *record.mesh;
    const int mesh_face_count = mesh.face_count();
    if (mesh_face_count == 0) {
        return;
    }

    const TriangleInfo *mesh_triangle_info = mesh.triangle_info();
    const Int scatter_indices = arange<Int>(mesh_face_count) + record.face_offset;
    const IntDetached scatter_indices_detached =
        arange<IntDetached>(mesh_face_count) + record.face_offset;

    scatter(triangle_info_.p0, mesh_triangle_info->p0, scatter_indices);
    scatter(triangle_info_.e1, mesh_triangle_info->e1, scatter_indices);
    scatter(triangle_info_.e2, mesh_triangle_info->e2, scatter_indices);
    scatter(triangle_info_.n0, mesh_triangle_info->n0, scatter_indices);
    scatter(triangle_info_.n1, mesh_triangle_info->n1, scatter_indices);
    scatter(triangle_info_.n2, mesh_triangle_info->n2, scatter_indices);
    scatter(triangle_info_.face_normal, mesh_triangle_info->face_normal, scatter_indices);
    scatter(triangle_info_.face_area, mesh_triangle_info->face_area, scatter_indices);

    scatter(triangle_info_detached_.p0, detach<false>(mesh_triangle_info->p0), scatter_indices_detached);
    scatter(triangle_info_detached_.e1, detach<false>(mesh_triangle_info->e1), scatter_indices_detached);
    scatter(triangle_info_detached_.e2, detach<false>(mesh_triangle_info->e2), scatter_indices_detached);
    scatter(triangle_info_detached_.n0, detach<false>(mesh_triangle_info->n0), scatter_indices_detached);
    scatter(triangle_info_detached_.n1, detach<false>(mesh_triangle_info->n1), scatter_indices_detached);
    scatter(triangle_info_detached_.n2, detach<false>(mesh_triangle_info->n2), scatter_indices_detached);
    scatter(triangle_info_detached_.face_normal,
            detach<false>(mesh_triangle_info->face_normal),
            scatter_indices_detached);
    scatter(triangle_info_detached_.face_area,
            detach<false>(mesh_triangle_info->face_area),
            scatter_indices_detached);

    if (!include_static) {
        return;
    }

    scatter(triangle_info_.face_indices, mesh_triangle_info->face_indices, scatter_indices);
    scatter(triangle_info_detached_.face_indices,
            detach<false>(mesh_triangle_info->face_indices),
            scatter_indices_detached);
    scatter(triangle_face_normal_mask_,
            full<Mask>(mesh.use_face_normals(), mesh_face_count),
            scatter_indices);
    scatter(triangle_face_normal_mask_detached_,
            full<MaskDetached>(mesh.use_face_normals(), mesh_face_count),
            scatter_indices_detached);

    if (mesh.has_uv() && mesh.triangle_uv() != nullptr) {
        scatter(triangle_uv_[0], (*mesh.triangle_uv())[0], scatter_indices);
        scatter(triangle_uv_[1], (*mesh.triangle_uv())[1], scatter_indices);
        scatter(triangle_uv_[2], (*mesh.triangle_uv())[2], scatter_indices);

        scatter(triangle_uv_detached_[0], detach<false>((*mesh.triangle_uv())[0]), scatter_indices_detached);
        scatter(triangle_uv_detached_[1], detach<false>((*mesh.triangle_uv())[1]), scatter_indices_detached);
        scatter(triangle_uv_detached_[2], detach<false>((*mesh.triangle_uv())[2]), scatter_indices_detached);
    }
}

void Scene::scatter_mesh_edge_data(const SceneMeshRecord &record, bool include_static_ids) {
    const Mesh &mesh = *record.mesh;
    const SecondaryEdgeInfo *mesh_edge_info = mesh.secondary_edge_info();
    const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
    if (mesh_edge_count == 0) {
        return;
    }

    const Int scatter_indices = arange<Int>(mesh_edge_count) + record.edge_offset;
    scatter(edge_info_.start, mesh_edge_info->start, scatter_indices);
    scatter(edge_info_.edge, mesh_edge_info->edge, scatter_indices);
    scatter(edge_info_.normal0, mesh_edge_info->normal0, scatter_indices);
    scatter(edge_info_.normal1, mesh_edge_info->normal1, scatter_indices);
    scatter(edge_info_.opposite, mesh_edge_info->opposite, scatter_indices);
    scatter(edge_info_.is_boundary, mesh_edge_info->is_boundary, scatter_indices);

    if (!include_static_ids) {
        return;
    }

    const IntDetached scatter_indices_detached = arange<IntDetached>(mesh_edge_count) + record.edge_offset;
    scatter(edge_shape_ids_,
            full<IntDetached>(mesh.mesh_id(), mesh_edge_count),
            scatter_indices_detached);
    scatter(edge_local_ids_,
            arange<IntDetached>(mesh_edge_count),
            scatter_indices_detached);
}

void Scene::ensure_scene_edge_data_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        if (!record.edge_dirty) {
            continue;
        }

        const_cast<Scene *>(this)->scatter_mesh_edge_data(record, false);
        record.edge_dirty = false;
    }

    ensure_edge_bvh_ready();
}

void Scene::ensure_edge_bvh_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    Scene *scene = const_cast<Scene *>(this);
    if (mask_dirty_) {
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            scene->edge_bvh_->set_mask(scene->edge_mask_);
            if (edge_backend_builds_optix(edge_bvh_backend_)) {
                scene->edge_bvh_->materialize();
            }
        }
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            scene->edge_optix_->set_mask(scene->edge_mask_);
            if (edge_backend_builds_drjit(edge_bvh_backend_)) {
                scene->edge_bvh_->materialize();
            }
        }
        scene->mask_dirty_ = false;
    }

    if (pending_edge_bvh_dirty_ranges_.empty()) {
        scene->edge_bvh_dirty_ = false;
        return;
    }

    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        scene->edge_bvh_->refit(scene->edge_info_, scene->pending_edge_bvh_dirty_ranges_);
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            scene->edge_bvh_->materialize();
        }
    }
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        scene->edge_optix_->refit(scene->edge_info_, scene->pending_edge_bvh_dirty_ranges_);
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            scene->edge_bvh_->materialize();
        }
    }
    scene->pending_edge_bvh_dirty_ranges_.clear();
    scene->edge_bvh_dirty_ = false;
}

void Scene::ensure_reflection_epc_geometry_ready() const {
    if (reflection_epc_geometry_ready_) {
        return;
    }

    drjit::eval(triangle_info_detached_.p0,
                triangle_info_detached_.e1,
                triangle_info_detached_.e2,
                triangle_info_detached_.face_normal,
                face_offsets_);
    reflection_epc_geometry_ready_ = true;
}

void Scene::build() {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::Build);
    require(!mesh_records_.empty(), "Scene::build(): missing meshes.");

    std::vector<int> face_offsets;
    face_offsets.reserve(mesh_records_.size() + 1);
    face_offsets.push_back(0);

    std::vector<int> vertex_offsets;
    vertex_offsets.reserve(mesh_records_.size() + 1);
    vertex_offsets.push_back(0);

    std::vector<int> edge_offsets;
    edge_offsets.reserve(mesh_records_.size() + 1);
    edge_offsets.push_back(0);

    std::vector<OptixSceneMeshDesc> mesh_descs;
    mesh_descs.reserve(mesh_records_.size());

    std::vector<int> topology_v0;
    std::vector<int> topology_v1;
    std::vector<int> topology_v0_global;
    std::vector<int> topology_v1_global;
    std::vector<int> topology_face0_local;
    std::vector<int> topology_face1_local;
    std::vector<int> topology_face0_global;
    std::vector<int> topology_face1_global;
    std::vector<int> topology_opposite0;
    std::vector<int> topology_opposite1;
    std::vector<int> topology_opposite0_global;
    std::vector<int> topology_opposite1_global;

    for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
        SceneMeshRecord &record = mesh_records_[mesh_index];
        Mesh &mesh = *record.mesh;
        mesh.set_mesh_id(static_cast<int>(mesh_index));
        mesh.build();
        record.vertex_offset = vertex_offsets.back();
        record.face_offset = face_offsets.back();
        const SecondaryEdgeInfo *mesh_edge_info = mesh.secondary_edge_info();
        const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
        record.edge_offset = edge_offsets.back();
        record.vertices_dirty = false;
        record.transform_dirty = false;
        record.edge_dirty = false;

        vertex_offsets.push_back(vertex_offsets.back() + mesh.vertex_count());
        face_offsets.push_back(face_offsets.back() + mesh.face_count());
        edge_offsets.push_back(edge_offsets.back() + mesh_edge_count);
        mesh_descs.push_back({ &mesh, record.dynamic, record.face_offset, static_cast<int>(mesh_index) });
    }

    mesh_count_ = static_cast<int>(mesh_records_.size());
    const int total_vertex_count = vertex_offsets.back();
    const int total_face_count = face_offsets.back();
    require(total_face_count > 0, "Scene::build(): scene has no triangles.");

    edge_count_ = edge_offsets.back();
    topology_v0.reserve(edge_count_);
    topology_v1.reserve(edge_count_);
    topology_v0_global.reserve(edge_count_);
    topology_v1_global.reserve(edge_count_);
    topology_face0_local.reserve(edge_count_);
    topology_face1_local.reserve(edge_count_);
    topology_face0_global.reserve(edge_count_);
    topology_face1_global.reserve(edge_count_);
    topology_opposite0.reserve(edge_count_);
    topology_opposite1.reserve(edge_count_);
    topology_opposite0_global.reserve(edge_count_);
    topology_opposite1_global.reserve(edge_count_);

    std::array<std::vector<int>, 3> global_face_indices_cpu;
    for (auto &global_face_indices : global_face_indices_cpu) {
        global_face_indices.reserve(total_face_count);
    }
    std::vector<int> global_shape_ids_cpu;
    std::vector<int> global_local_prim_ids_cpu;
    std::vector<int> global_prim_ids_cpu;
    global_shape_ids_cpu.reserve(total_face_count);
    global_local_prim_ids_cpu.reserve(total_face_count);
    global_prim_ids_cpu.reserve(total_face_count);

    std::array<std::vector<int>, 3> triangle_edge_ids_cpu;
    for (auto &triangle_edge_ids : triangle_edge_ids_cpu) {
        triangle_edge_ids.assign(total_face_count, -1);
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        const Mesh &mesh = *record.mesh;
        const auto &mesh_edge_indices = mesh.edge_indices();
        const int mesh_edge_count = mesh.edges_enabled() ? static_cast<int>(slices(mesh_edge_indices)) : 0;
        const Vector3iDetached mesh_face_indices(detach<false>(mesh.face_indices()[0]),
                                                 detach<false>(mesh.face_indices()[1]),
                                                 detach<false>(mesh.face_indices()[2]));
        std::array<std::vector<int>, 3> mesh_face_cpu;
        copy_cuda_array(mesh_face_indices, mesh_face_cpu);
        for (int local_face_id = 0; local_face_id < mesh.face_count(); ++local_face_id) {
            global_face_indices_cpu[0].push_back(record.vertex_offset + mesh_face_cpu[0][local_face_id]);
            global_face_indices_cpu[1].push_back(record.vertex_offset + mesh_face_cpu[1][local_face_id]);
            global_face_indices_cpu[2].push_back(record.vertex_offset + mesh_face_cpu[2][local_face_id]);
            global_shape_ids_cpu.push_back(mesh.mesh_id());
            global_local_prim_ids_cpu.push_back(local_face_id);
            global_prim_ids_cpu.push_back(record.face_offset + local_face_id);
        }

        if (mesh_edge_count == 0) {
            continue;
        }

        std::array<std::vector<int>, 5> mesh_edge_cpu;
        copy_cuda_array(mesh_edge_indices, mesh_edge_cpu);

        for (int local_edge_id = 0; local_edge_id < mesh_edge_count; ++local_edge_id) {
            const int v0 = mesh_edge_cpu[0][local_edge_id];
            const int v1 = mesh_edge_cpu[1][local_edge_id];
            const int v0_global = record.vertex_offset + v0;
            const int v1_global = record.vertex_offset + v1;
            const int face0_local = mesh_edge_cpu[2][local_edge_id];
            const int face1_local = mesh_edge_cpu[3][local_edge_id];
            const int face0_global = record.face_offset + face0_local;
            const int face1_global = face1_local >= 0 ? record.face_offset + face1_local : -1;
            const int opposite0 = mesh_edge_cpu[4][local_edge_id];
            const int opposite0_global = opposite0 >= 0 ? record.vertex_offset + opposite0 : -1;
            const int global_edge_id = record.edge_offset + local_edge_id;

            const std::array<int, 3> face0_vertices {
                mesh_face_cpu[0][face0_local],
                mesh_face_cpu[1][face0_local],
                mesh_face_cpu[2][face0_local]
            };

            int opposite1 = -1;
            if (face1_local >= 0) {
                const std::array<int, 3> face1_vertices {
                    mesh_face_cpu[0][face1_local],
                    mesh_face_cpu[1][face1_local],
                    mesh_face_cpu[2][face1_local]
                };
                opposite1 = face_opposite_vertex(face1_vertices, v0, v1);
                const int opposite1_global = opposite1 >= 0 ? record.vertex_offset + opposite1 : -1;
                const int face1_slot = face_edge_slot(face1_vertices, v0, v1);
                if (face1_slot >= 0) {
                    triangle_edge_ids_cpu[face1_slot][face1_global] = global_edge_id;
                }
                topology_opposite1_global.push_back(opposite1_global);
            } else {
                topology_opposite1_global.push_back(-1);
            }

            const int face0_slot = face_edge_slot(face0_vertices, v0, v1);
            if (face0_slot >= 0) {
                triangle_edge_ids_cpu[face0_slot][face0_global] = global_edge_id;
            }

            topology_v0.push_back(v0);
            topology_v1.push_back(v1);
            topology_v0_global.push_back(v0_global);
            topology_v1_global.push_back(v1_global);
            topology_face0_local.push_back(face0_local);
            topology_face1_local.push_back(face1_local);
            topology_face0_global.push_back(face0_global);
            topology_face1_global.push_back(face1_global);
            topology_opposite0.push_back(opposite0);
            topology_opposite1.push_back(opposite1);
            topology_opposite0_global.push_back(opposite0_global);
        }
    }

    auto load_or_empty = [](const std::vector<int> &values) {
        return values.empty() ? IntDetached() : load<IntDetached>(values.data(), values.size());
    };

    face_offsets_ = load<IntDetached>(face_offsets.data(), face_offsets.size());
    edge_offsets_ = load<IntDetached>(edge_offsets.data(), edge_offsets.size());
    vertex_offsets_ = load<IntDetached>(vertex_offsets.data(), vertex_offsets.size());
    triangle_info_ = empty<TriangleInfo>(total_face_count);
    triangle_info_detached_ = empty<TriangleInfoDetached>(total_face_count);
    triangle_uv_ = zeros<TriangleUV>(total_face_count);
    triangle_uv_detached_ = zeros<TriangleUVDetached>(total_face_count);
    triangle_face_normal_mask_ = empty<Mask>(total_face_count);
    triangle_face_normal_mask_detached_ = empty<MaskDetached>(total_face_count);
    global_geometry_.vertices = total_vertex_count > 0 ? empty<Vector3f>(total_vertex_count) : Vector3f();
    global_geometry_.faces = Vector3iDetached(
        load<IntDetached>(global_face_indices_cpu[0].data(), total_face_count),
        load<IntDetached>(global_face_indices_cpu[1].data(), total_face_count),
        load<IntDetached>(global_face_indices_cpu[2].data(), total_face_count));
    global_geometry_.shape_id = load<IntDetached>(global_shape_ids_cpu.data(), total_face_count);
    global_geometry_.local_prim_id =
        load<IntDetached>(global_local_prim_ids_cpu.data(), total_face_count);
    global_geometry_.global_prim_id = load<IntDetached>(global_prim_ids_cpu.data(), total_face_count);
    triangle_edge_ids_ = VectoriT<3, true>(load<IntDetached>(triangle_edge_ids_cpu[0].data(), total_face_count),
                                           load<IntDetached>(triangle_edge_ids_cpu[1].data(), total_face_count),
                                           load<IntDetached>(triangle_edge_ids_cpu[2].data(), total_face_count));
    if (edge_count_ > 0) {
        edge_info_ = empty<SecondaryEdgeInfo>(edge_count_);
        edge_topology_ = SceneEdgeTopology {
            load_or_empty(topology_v0),
            load_or_empty(topology_v1),
            load_or_empty(topology_v0_global),
            load_or_empty(topology_v1_global),
            load_or_empty(topology_face0_local),
            load_or_empty(topology_face1_local),
            load_or_empty(topology_face0_global),
            load_or_empty(topology_face1_global),
            load_or_empty(topology_opposite0),
            load_or_empty(topology_opposite1),
            load_or_empty(topology_opposite0_global),
            load_or_empty(topology_opposite1_global)
        };
        edge_shape_ids_ = empty<IntDetached>(edge_count_);
        edge_local_ids_ = empty<IntDetached>(edge_count_);
        edge_mask_ = full<MaskDetached>(true, edge_count_);
    } else {
        edge_info_ = SecondaryEdgeInfo();
        edge_topology_ = SceneEdgeTopology();
        edge_shape_ids_ = IntDetached();
        edge_local_ids_ = IntDetached();
        edge_mask_ = MaskDetached();
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        scatter_mesh_data(record, true);
        scatter_mesh_edge_data(record, true);
        const Mesh &mesh = *record.mesh;
        const int mesh_vertex_count = mesh.vertex_count();
        if (mesh_vertex_count > 0) {
            const Int vertex_scatter_indices = arange<Int>(mesh_vertex_count) + record.vertex_offset;
            scatter(global_geometry_.vertices, mesh.vertex_positions_world(), vertex_scatter_indices);
        }
    }
    global_geometry_.face_normal = triangle_info_.face_normal;

    int static_mesh_count = 0;
    int dynamic_mesh_count = 0;
    for (const SceneMeshRecord &record : mesh_records_) {
        if (record.dynamic) {
            ++dynamic_mesh_count;
        } else {
            ++static_mesh_count;
        }
    }

    optix_split_active_ =
        should_split_optix_scene(active_optix_split_mode(), static_mesh_count, dynamic_mesh_count);
    optix_static_mesh_indices_.clear();
    optix_dynamic_mesh_indices_.clear();
    optix_dynamic_mesh_local_index_.assign(mesh_records_.size(), -1);

    if (optix_split_active_) {
        std::vector<OptixSceneMeshDesc> static_mesh_descs;
        std::vector<OptixSceneMeshDesc> dynamic_mesh_descs;
        static_mesh_descs.reserve(static_mesh_count);
        dynamic_mesh_descs.reserve(dynamic_mesh_count);

        for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
            if (mesh_records_[mesh_index].dynamic) {
                optix_dynamic_mesh_local_index_[mesh_index] =
                    static_cast<int>(dynamic_mesh_descs.size());
                optix_dynamic_mesh_indices_.push_back(static_cast<int>(mesh_index));
                dynamic_mesh_descs.push_back(mesh_descs[mesh_index]);
            } else {
                optix_static_mesh_indices_.push_back(static_cast<int>(mesh_index));
                static_mesh_descs.push_back(mesh_descs[mesh_index]);
            }
        }

        optix_scene_ = std::make_unique<OptixScene>();
        optix_static_scene_ = std::make_unique<OptixScene>();
        optix_dynamic_scene_ = std::make_unique<OptixScene>();
        optix_scene_->build(mesh_descs);
        optix_static_scene_->build(static_mesh_descs, optix_scene_.get());
        optix_dynamic_scene_->build(dynamic_mesh_descs, optix_scene_.get());
    } else {
        optix_scene_ = std::make_unique<OptixScene>();
        optix_static_scene_ = std::make_unique<OptixScene>();
        optix_dynamic_scene_ = std::make_unique<OptixScene>();
        optix_scene_->build(mesh_descs);
    }
    reflection_pipeline_.reset();
    reflection_epc_pipeline_.reset();
    reflection_epc_geometry_ready_ = false;
    segment_visibility_pipeline_.reset();
    mask_dirty_ = false;
    edge_bvh_ = std::make_unique<SceneEdge>();
    edge_optix_ = std::make_unique<SceneEdgeOptix>();
    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        edge_bvh_->build(edge_info_, edge_mask_);
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            edge_bvh_->materialize();
        }
    }
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        edge_optix_->build(edge_info_, edge_mask_);
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            edge_bvh_->materialize();
        }
    }
    is_ready_ = true;
    pending_updates_ = false;
    ++scene_version_;
    ++edge_version_;
}

void Scene::update_mesh_vertices(int mesh_id, const Vector3f &positions) {
    require(is_ready(), "Scene::update_mesh_vertices(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::update_mesh_vertices(): target mesh is not dynamic.");
    require(static_cast<int>(slices(positions)) == record.mesh->vertex_count(),
            "Scene::update_mesh_vertices(): vertex count must remain unchanged.");

    record.mesh->set_vertex_positions(positions);
    record.vertices_dirty = true;
    pending_updates_ = true;
}

void Scene::set_mesh_transform(int mesh_id, const Matrix4f &matrix, bool set_left) {
    require(is_ready(), "Scene::set_mesh_transform(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::set_mesh_transform(): target mesh is not dynamic.");

    record.mesh->set_transform(matrix, set_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::append_mesh_transform(int mesh_id, const Matrix4f &matrix, bool append_left) {
    require(is_ready(), "Scene::append_mesh_transform(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::append_mesh_transform(): target mesh is not dynamic.");

    record.mesh->append_transform(matrix, append_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::set_edge_mask(const MaskDetached &mask) {
    require(is_ready(), "Scene::set_edge_mask(): scene is not built.");
    require(static_cast<int>(mask.size()) == edge_count_,
            "Scene::set_edge_mask(): mask size must match the scene edge count.");

    if (mask.size() == edge_mask_.size() && drjit::all(mask == edge_mask_)) {
        return;
    }

    edge_mask_ = mask;
    mask_dirty_ = true;
    edge_bvh_dirty_ = true;
    pending_updates_ = true;
}

void Scene::sync() {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::Sync);
    require(is_ready(), "Scene::sync(): scene is not built.");
    last_sync_profile_ = SceneSyncProfile();

    if (!pending_updates_) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    const bool mask_dirty_before = mask_dirty_;

    std::vector<OptixSceneMeshDesc> mesh_descs;
    mesh_descs.reserve(mesh_records_.size());

    std::vector<OptixSceneMeshUpdate> updates;
    updates.reserve(mesh_records_.size());

    for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
        SceneMeshRecord &record = mesh_records_[mesh_index];
        mesh_descs.push_back({ record.mesh.get(), record.dynamic, record.face_offset, static_cast<int>(mesh_index) });

        if (!record.vertices_dirty && !record.transform_dirty) {
            continue;
        }

        const auto mesh_update_start = Clock::now();
        record.mesh->update_runtime_data(record.vertices_dirty, record.transform_dirty);
        last_sync_profile_.mesh_update_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - mesh_update_start).count();

        const auto scatter_start = Clock::now();
        scatter_mesh_data(record, false);
        const int mesh_vertex_count = record.mesh->vertex_count();
        if (mesh_vertex_count > 0) {
            const Int vertex_scatter_indices = arange<Int>(mesh_vertex_count) + record.vertex_offset;
            scatter(global_geometry_.vertices,
                    record.mesh->vertex_positions_world(),
                    vertex_scatter_indices);
        }
        last_sync_profile_.triangle_scatter_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - scatter_start).count();

        const int mesh_edge_count =
            record.mesh->edges_enabled() ? static_cast<int>(slices(record.mesh->edge_indices())) : 0;
        if (mesh_edge_count > 0 && !record.edge_dirty) {
            pending_edge_bvh_dirty_ranges_.push_back({ record.edge_offset, mesh_edge_count });
            record.edge_dirty = true;
            edge_bvh_dirty_ = true;
            ++last_sync_profile_.updated_edge_meshes;
            last_sync_profile_.updated_edges += mesh_edge_count;
        }

        updates.push_back({ static_cast<int>(mesh_index), record.vertices_dirty, record.transform_dirty });
        ++last_sync_profile_.updated_meshes;
        if (record.vertices_dirty) {
            ++last_sync_profile_.updated_vertex_meshes;
        }
        if (record.transform_dirty) {
            ++last_sync_profile_.updated_transform_meshes;
        }
        record.vertices_dirty = false;
        record.transform_dirty = false;
    }
    if (!updates.empty()) {
        global_geometry_.face_normal = triangle_info_.face_normal;
    }

    if (edge_bvh_dirty_) {
        const auto edge_scatter_start = Clock::now();
        for (SceneMeshRecord &record : mesh_records_) {
            if (!record.edge_dirty) {
                continue;
            }

            scatter_mesh_edge_data(record, false);
            record.edge_dirty = false;
        }
        last_sync_profile_.edge_scatter_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - edge_scatter_start).count();

        const auto edge_refit_start = Clock::now();
        ensure_edge_bvh_ready();
        last_sync_profile_.edge_refit_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - edge_refit_start).count();
    }

    const auto optix_start = Clock::now();
    if (optix_split_active_) {
        if (!updates.empty()) {
            optix_scene_->sync(mesh_descs, updates);
        }

        std::vector<OptixSceneMeshDesc> dynamic_mesh_descs;
        dynamic_mesh_descs.reserve(optix_dynamic_mesh_indices_.size());
        for (int mesh_index : optix_dynamic_mesh_indices_) {
            dynamic_mesh_descs.push_back(mesh_descs[static_cast<size_t>(mesh_index)]);
        }

        std::vector<OptixSceneMeshUpdate> dynamic_updates;
        dynamic_updates.reserve(updates.size());
        for (const OptixSceneMeshUpdate &update : updates) {
            const int dynamic_local_index =
                optix_dynamic_mesh_local_index_[static_cast<size_t>(update.mesh_id)];
            if (dynamic_local_index < 0) {
                continue;
            }
            dynamic_updates.push_back(
                { dynamic_local_index, update.vertices_dirty, update.transform_dirty });
        }

        if (!dynamic_updates.empty()) {
            optix_dynamic_scene_->sync(dynamic_mesh_descs, dynamic_updates);
        }
        last_sync_profile_.optix_sync_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - optix_start).count();
        if (!updates.empty()) {
            const OptixSyncProfile &optix_profile = optix_scene_->last_sync_profile();
            last_sync_profile_.optix_gas_update_ms += optix_profile.gas_update_ms;
            last_sync_profile_.optix_ias_update_ms += optix_profile.ias_update_ms;
        }
        if (!dynamic_updates.empty()) {
            const OptixSyncProfile &optix_profile = optix_dynamic_scene_->last_sync_profile();
            last_sync_profile_.optix_gas_update_ms += optix_profile.gas_update_ms;
            last_sync_profile_.optix_ias_update_ms += optix_profile.ias_update_ms;
        }
    } else {
        optix_scene_->sync(mesh_descs, updates);
        last_sync_profile_.optix_sync_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - optix_start).count();
        const OptixSyncProfile &optix_profile = optix_scene_->last_sync_profile();
        last_sync_profile_.optix_gas_update_ms = optix_profile.gas_update_ms;
        last_sync_profile_.optix_ias_update_ms = optix_profile.ias_update_ms;
    }
    pending_updates_ = false;
    if (!updates.empty()) {
        reflection_epc_geometry_ready_ = false;
    }
    if (!updates.empty()) {
        ++scene_version_;
    }
    if (mask_dirty_before || last_sync_profile_.updated_edge_meshes > 0) {
        ++edge_version_;
    }
    last_sync_profile_.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
}

SceneEdgeInfo Scene::edge_info() const {
    require(is_ready(), "Scene::edge_info(): scene is not built.");
    require(!pending_updates_, "Scene::edge_info(): scene has pending updates. Call Scene::sync() first.");

    ensure_scene_edge_data_ready();

    SceneEdgeInfo info;
    info.start = edge_info_.start;
    info.edge = edge_info_.edge;
    info.end = edge_info_.start + edge_info_.edge;
    info.length = norm(edge_info_.edge);
    info.normal0 = edge_info_.normal0;
    info.normal1 = edge_info_.normal1;
    info.is_boundary = edge_info_.is_boundary;
    info.shape_id = edge_shape_ids_;
    info.local_edge_id = edge_local_ids_;
    info.global_edge_id = arange<IntDetached>(edge_count_);
    return info;
}

std::string Scene::edge_bvh_backend() const {
    return edge_backend_name(edge_bvh_backend_);
}

SceneEdgeBVHStats Scene::edge_bvh_stats() const {
    require(is_ready(), "Scene::edge_bvh_stats(): scene is not built.");
    require(!pending_updates_,
            "Scene::edge_bvh_stats(): scene has pending updates. Call Scene::sync() first.");
    ensure_edge_bvh_ready();
    return edge_bvh_backend_ == EdgeBVHBackend::Optix ? edge_optix_->stats() : edge_bvh_->stats();
}

const SceneEdgeTopology &Scene::edge_topology() const {
    require(is_ready(), "Scene::edge_topology(): scene is not built.");
    return edge_topology_;
}

const MaskDetached &Scene::edge_mask() const {
    require(is_ready(), "Scene::edge_mask(): scene is not built.");
    return edge_mask_;
}

const SceneGlobalGeometry &Scene::global_geometry() const {
    require(is_ready(), "Scene::global_geometry(): scene is not built.");
    require(!pending_updates_,
            "Scene::global_geometry(): scene has pending updates. Call Scene::sync() first.");
    return global_geometry_;
}

VectoriT<3, true> Scene::triangle_edge_indices(const IntDetached &prim_id, bool global) const {
    require(is_ready(), "Scene::triangle_edge_indices(): scene is not built.");

    const int query_count = static_cast<int>(slices(prim_id));
    VectoriT<3, true> result(full<IntDetached>(-1, query_count),
                             full<IntDetached>(-1, query_count),
                             full<IntDetached>(-1, query_count));
    if (query_count == 0) {
        return result;
    }

    const int face_count = static_cast<int>(slices(triangle_edge_ids_[0]));
    const MaskDetached valid = prim_id >= 0 && prim_id < face_count;
    const IntDetached edge0 = gather<IntDetached>(triangle_edge_ids_[0], prim_id, valid);
    const IntDetached edge1 = gather<IntDetached>(triangle_edge_ids_[1], prim_id, valid);
    const IntDetached edge2 = gather<IntDetached>(triangle_edge_ids_[2], prim_id, valid);

    if (global) {
        result[0] = select(valid, edge0, result[0]);
        result[1] = select(valid, edge1, result[1]);
        result[2] = select(valid, edge2, result[2]);
        return result;
    }

    const MaskDetached valid0 = valid && edge0 >= 0;
    const MaskDetached valid1 = valid && edge1 >= 0;
    const MaskDetached valid2 = valid && edge2 >= 0;
    result[0] = select(valid0, gather<IntDetached>(edge_local_ids_, edge0, valid0), result[0]);
    result[1] = select(valid1, gather<IntDetached>(edge_local_ids_, edge1, valid1), result[1]);
    result[2] = select(valid2, gather<IntDetached>(edge_local_ids_, edge2, valid2), result[2]);
    return result;
}

VectoriT<2, true> Scene::edge_adjacent_faces(const IntDetached &edge_id, bool global) const {
    require(is_ready(), "Scene::edge_adjacent_faces(): scene is not built.");

    const int query_count = static_cast<int>(slices(edge_id));
    VectoriT<2, true> result(full<IntDetached>(-1, query_count),
                             full<IntDetached>(-1, query_count));
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    const MaskDetached valid = edge_id >= 0 && edge_id < edge_count_;
    const IntDetached face0 = global
        ? gather<IntDetached>(edge_topology_.face0_global, edge_id, valid)
        : gather<IntDetached>(edge_topology_.face0_local, edge_id, valid);
    const IntDetached face1 = global
        ? gather<IntDetached>(edge_topology_.face1_global, edge_id, valid)
        : gather<IntDetached>(edge_topology_.face1_local, edge_id, valid);
    result[0] = select(valid, face0, result[0]);
    result[1] = select(valid, face1, result[1]);
    return result;
}

bool Scene::is_ready() const {
    const bool optix_ready =
        optix_split_active_
            ? (optix_scene_ != nullptr && optix_static_scene_ != nullptr &&
               optix_dynamic_scene_ != nullptr && optix_scene_->is_ready() &&
               optix_static_scene_->is_ready() && optix_dynamic_scene_->is_ready())
            : (optix_scene_ != nullptr && optix_scene_->is_ready());
    bool edge_ready = true;
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        edge_ready &= edge_optix_ != nullptr && edge_optix_->is_ready();
    }
    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        edge_ready &= edge_bvh_ != nullptr && edge_bvh_->is_ready();
    }
    return is_ready_ && edge_ready && optix_ready;
}

template <bool Detached>
IntersectionT<Detached> Scene::intersect(const RayT<Detached> &ray, MaskT<Detached> active, RayFlags flags) const {
    require(is_ready(), "Scene::intersect(): scene is not built.");
    require(!pending_updates_, "Scene::intersect(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(ray.o));
    const bool want_geo_n   = has_flag(flags, RayFlags::Geometric);
    const bool want_shading = has_flag(flags, RayFlags::ShadingN);
    const bool want_uv      = has_flag(flags, RayFlags::UV);
    const bool symbolic_optix_query = optix_split_active_ && uses_symbolic_optix_query_path();

    IntersectionT<Detached> intersection;
    intersection.t = full<FloatT<Detached>>(Infinity, ray_count);
    intersection.p = zeros<Vector3fT<Detached>>(ray_count);
    intersection.n = zeros<Vector3fT<Detached>>(ray_count);
    intersection.geo_n = zeros<Vector3fT<Detached>>(ray_count);
    intersection.uv = zeros<Vector2fT<Detached>>(ray_count);
    intersection.barycentric = zeros<Vector3fT<Detached>>(ray_count);
    intersection.shape_id = full<IntT<Detached>>(-1, ray_count);
    intersection.prim_id = full<IntT<Detached>>(-1, ray_count);

    MaskT<Detached> hit_mask = active;
    OptixIntersection optix_hit;
    if (optix_split_active_ && !symbolic_optix_query) {
        MaskT<Detached> static_hit_mask = active;
        MaskT<Detached> dynamic_hit_mask = active;
        const OptixIntersection static_hit =
            optix_static_scene_->template intersect<Detached>(ray, static_hit_mask);
        const OptixIntersection dynamic_hit =
            optix_dynamic_scene_->template intersect<Detached>(ray, dynamic_hit_mask);

        const MaskDetached static_hit_mask_detached = detach<false>(static_hit_mask);
        const MaskDetached dynamic_hit_mask_detached = detach<false>(dynamic_hit_mask);
        const MaskDetached choose_dynamic =
            dynamic_hit_mask_detached &&
            (!static_hit_mask_detached || (dynamic_hit.t < static_hit.t));
        const MaskDetached any_hit = static_hit_mask_detached || dynamic_hit_mask_detached;

        optix_hit.reserve(ray_count);
        optix_hit.t = select(choose_dynamic, dynamic_hit.t, static_hit.t);
        optix_hit.barycentric[0] =
            select(choose_dynamic, dynamic_hit.barycentric[0], static_hit.barycentric[0]);
        optix_hit.barycentric[1] =
            select(choose_dynamic, dynamic_hit.barycentric[1], static_hit.barycentric[1]);
        optix_hit.shape_id = select(choose_dynamic, dynamic_hit.shape_id, static_hit.shape_id);
        optix_hit.global_prim_id =
            select(choose_dynamic, dynamic_hit.global_prim_id, static_hit.global_prim_id);

        if constexpr (!Detached) {
            hit_mask = Mask(any_hit);
        } else {
            hit_mask = any_hit;
        }
    } else {
        optix_hit = optix_scene_->template intersect<Detached>(ray, hit_mask);
    }

    const IntDetached shape_id = optix_hit.shape_id;
    const IntDetached global_primitive_id = optix_hit.global_prim_id;
    const MaskDetached hit_mask_detached = detach<false>(hit_mask);
    const IntDetached mesh_face_offset = gather<IntDetached>(face_offsets_, shape_id, hit_mask_detached);
    const IntDetached local_primitive_id = global_primitive_id - mesh_face_offset;

    Vector2fT<Detached> triangle_uv_coords;
    FloatT<Detached> hit_distance;

    if constexpr (!Detached) {
        // AD path: re-gather vertex data and recompute intersection for gradients.
        const Int global_primitive_id_ad = Int(global_primitive_id);
        const Vector3f triangle_p0 = gather<Vector3f>(triangle_info_.p0, global_primitive_id_ad, hit_mask);
        const Vector3f triangle_e1 = gather<Vector3f>(triangle_info_.e1, global_primitive_id_ad, hit_mask);
        const Vector3f triangle_e2 = gather<Vector3f>(triangle_info_.e2, global_primitive_id_ad, hit_mask);
        std::tie(triangle_uv_coords, hit_distance) = ray_intersect_triangle<Detached>(triangle_p0, triangle_e1, triangle_e2, ray);

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal = gather<Vector3f>(triangle_info_.face_normal, global_primitive_id_ad, hit_mask);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 = gather<Vector3f>(triangle_info_.n0, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n1 = gather<Vector3f>(triangle_info_.n1, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n2 = gather<Vector3f>(triangle_info_.n2, global_primitive_id_ad, hit_mask);
                MaskT<Detached> use_face_normal_mask = gather<Mask>(triangle_face_normal_mask_, global_primitive_id_ad, hit_mask);
                const Vector2fT<Detached> safe_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal =
                    normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data = gather<TriangleUV>(triangle_uv_, global_primitive_id_ad, hit_mask);
            const Vector2fT<Detached> safe_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0], triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask, uv, intersection.uv);
        }
    } else {
        // Detached path: use OptiX results directly, gather only what is needed.
        triangle_uv_coords = optix_hit.barycentric;
        hit_distance = optix_hit.t;

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal = gather<Vector3fDetached>(triangle_info_detached_.face_normal, global_primitive_id, hit_mask_detached);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 = gather<Vector3fDetached>(triangle_info_detached_.n0, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n1 = gather<Vector3fDetached>(triangle_info_detached_.n1, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n2 = gather<Vector3fDetached>(triangle_info_detached_.n2, global_primitive_id, hit_mask_detached);
                MaskT<Detached> use_face_normal_mask = gather<MaskDetached>(triangle_face_normal_mask_detached_, global_primitive_id, hit_mask_detached);
                const Vector2fT<Detached> safe_uv = select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal =
                    normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask_detached, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask_detached, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data = gather<TriangleUVDetached>(triangle_uv_detached_, global_primitive_id, hit_mask_detached);
            const Vector2fT<Detached> safe_uv = select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0], triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask_detached, uv, intersection.uv);
        }
    }

    hit_mask &= drjit::isfinite(hit_distance) && (hit_distance < ray.tmax);

    const FloatT<Detached> safe_hit_distance = select(hit_mask, hit_distance, zeros<FloatT<Detached>>(ray_count));
    const Vector2fT<Detached> safe_triangle_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));

    const Vector3fT<Detached> barycentric_coordinates(1.f - safe_triangle_uv.x() - safe_triangle_uv.y(),
                                                      safe_triangle_uv.x(),
                                                      safe_triangle_uv.y());
    const Vector3fT<Detached> hit_position = ray(safe_hit_distance);

    intersection.t = select(hit_mask, safe_hit_distance, intersection.t);
    intersection.p = select(hit_mask, hit_position, intersection.p);
    intersection.barycentric = select(hit_mask, barycentric_coordinates, intersection.barycentric);
    intersection.shape_id = select(hit_mask, IntT<Detached>(shape_id), intersection.shape_id);
    const IntT<Detached> local_primitive_id_t = IntT<Detached>(local_primitive_id);
    const IntT<Detached> global_primitive_id_t = IntT<Detached>(global_primitive_id);
    intersection.prim_id = select(hit_mask, local_primitive_id_t, intersection.prim_id);
    intersection.local_prim_id =
        select(hit_mask, local_primitive_id_t, intersection.local_prim_id);
    intersection.global_prim_id =
        select(hit_mask, global_primitive_id_t, intersection.global_prim_id);
    return intersection;
}

Scene::OptixSceneSelection Scene::select_optix_scenes() const {
    OptixSceneSelection selection;
    selection.hitgroup_record_count = mesh_count_;
    if (optix_split_active_) {
        selection.primary = optix_static_scene_.get();
        selection.secondary = optix_dynamic_scene_.get();
        selection.split_mode = 1;
        selection.hitgroup_record_count = static_cast<int>(
            std::max(optix_static_mesh_indices_.size(), optix_dynamic_mesh_indices_.size()));
    } else {
        selection.primary = optix_scene_.get();
    }
    return selection;
}

template IntersectionDetached Scene::intersect<true>(const RayDetached &ray, MaskDetached active, RayFlags flags) const;

template Intersection Scene::intersect<false>(const Ray &ray, Mask active, RayFlags flags) const;

} // namespace rayd
