#include <algorithm>
#include <array>
#include <chrono>
#include <sstream>
#include <vector>

#include <rayd/intersection.h>
#include <rayd/camera.h>
#include <rayd/ray.h>
#include <rayd/scene/scene.h>
#include <rayd/scene/scene_edge.h>

namespace rayd {

namespace {

template <bool Detached>
NearestPointEdgeT<Detached> initialize_nearest_point_edge_result(int query_count) {
    NearestPointEdgeT<Detached> result;
    result.distance = full<FloatT<Detached>>(Infinity, query_count);
    result.point = zeros<Vector3fT<Detached>>(query_count);
    result.edge_t = zeros<FloatT<Detached>>(query_count);
    result.edge_point = zeros<Vector3fT<Detached>>(query_count);
    result.shape_id = full<IntT<Detached>>(-1, query_count);
    result.edge_id = full<IntT<Detached>>(-1, query_count);
    result.global_edge_id = full<IntT<Detached>>(-1, query_count);
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> initialize_nearest_ray_edge_result(int query_count) {
    NearestRayEdgeT<Detached> result;
    result.distance = full<FloatT<Detached>>(Infinity, query_count);
    result.ray_t = zeros<FloatT<Detached>>(query_count);
    result.point = zeros<Vector3fT<Detached>>(query_count);
    result.edge_t = zeros<FloatT<Detached>>(query_count);
    result.edge_point = zeros<Vector3fT<Detached>>(query_count);
    result.shape_id = full<IntT<Detached>>(-1, query_count);
    result.edge_id = full<IntT<Detached>>(-1, query_count);
    result.global_edge_id = full<IntT<Detached>>(-1, query_count);
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
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

Scene::Scene()
    : optix_scene_(std::make_unique<OptixScene>()),
      edge_bvh_(std::make_unique<SceneEdge>()) {}

Scene::~Scene() {
    for (Camera *camera : primary_edge_observers_) {
        if (camera != nullptr) {
            camera->clear_primary_edge_scene_binding(this);
        }
    }
}

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
    edge_mask_ = MaskDetached();
    active_edge_info_ = SecondaryEdgeInfo();
    active_edge_global_ids_ = IntDetached();
    global_to_active_edge_id_ = IntDetached();
    pending_edge_bvh_dirty_ranges_.clear();
    edge_bvh_dirty_ = false;
    mask_dirty_ = false;
    all_edges_active_ = true;
    invalidate_primary_edge_observers();
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

void Scene::scatter_active_edge_data(const IntDetached &active_indices,
                                     const IntDetached &global_indices) {
    const int count = static_cast<int>(global_indices.size());
    if (count == 0) {
        return;
    }

    const Int active_indices_ad = Int(active_indices);
    const Int global_indices_ad = Int(global_indices);
    scatter(active_edge_info_.start,
            gather<Vector3f>(edge_info_.start, global_indices_ad),
            active_indices_ad);
    scatter(active_edge_info_.edge,
            gather<Vector3f>(edge_info_.edge, global_indices_ad),
            active_indices_ad);
    scatter(active_edge_info_.normal0,
            gather<Vector3f>(edge_info_.normal0, global_indices_ad),
            active_indices_ad);
    scatter(active_edge_info_.normal1,
            gather<Vector3f>(edge_info_.normal1, global_indices_ad),
            active_indices_ad);
    scatter(active_edge_info_.opposite,
            gather<Vector3f>(edge_info_.opposite, global_indices_ad),
            active_indices_ad);
    scatter(active_edge_info_.is_boundary,
            gather<Mask>(edge_info_.is_boundary, global_indices_ad),
            active_indices_ad);
}

void Scene::rebuild_active_edge_bvh() {
    pending_edge_bvh_dirty_ranges_.clear();
    active_edge_info_ = SecondaryEdgeInfo();
    active_edge_global_ids_ = IntDetached();
    global_to_active_edge_id_ = IntDetached();

    if (edge_count_ == 0) {
        all_edges_active_ = true;
        edge_bvh_->build(active_edge_info_);
        edge_bvh_dirty_ = false;
        mask_dirty_ = false;
        return;
    }

    all_edges_active_ = drjit::all(edge_mask_);
    if (all_edges_active_) {
        edge_bvh_->build(edge_info_);
        edge_bvh_dirty_ = false;
        mask_dirty_ = false;
        return;
    }

    active_edge_global_ids_ = compressD(arange<IntDetached>(edge_count_), edge_mask_);
    const int active_edge_count = static_cast<int>(active_edge_global_ids_.size());
    if (active_edge_count > 0) {
        const Int active_edge_global_ids_ad = Int(active_edge_global_ids_);
        active_edge_info_ = SecondaryEdgeInfo {
            gather<Vector3f>(edge_info_.start, active_edge_global_ids_ad),
            gather<Vector3f>(edge_info_.edge, active_edge_global_ids_ad),
            gather<Vector3f>(edge_info_.normal0, active_edge_global_ids_ad),
            gather<Vector3f>(edge_info_.normal1, active_edge_global_ids_ad),
            gather<Vector3f>(edge_info_.opposite, active_edge_global_ids_ad),
            gather<Mask>(edge_info_.is_boundary, active_edge_global_ids_ad)
        };
        global_to_active_edge_id_ = full<IntDetached>(-1, edge_count_);
        scatter(global_to_active_edge_id_,
                arange<IntDetached>(active_edge_count),
                active_edge_global_ids_);
        drjit::eval(active_edge_info_.start,
                    active_edge_info_.edge,
                    active_edge_info_.normal0,
                    active_edge_info_.normal1,
                    active_edge_info_.opposite,
                    active_edge_info_.is_boundary,
                    active_edge_global_ids_,
                    global_to_active_edge_id_);
        drjit::sync_thread();
    } else {
        global_to_active_edge_id_ = full<IntDetached>(-1, edge_count_);
        drjit::eval(global_to_active_edge_id_);
        drjit::sync_thread();
    }

    edge_bvh_->build(active_edge_info_);
    edge_bvh_dirty_ = false;
    mask_dirty_ = false;
}

IntDetached Scene::dirty_global_edge_ids_from_ranges() const {
    size_t total_count = 0;
    for (const EdgeDirtyRange &range : pending_edge_bvh_dirty_ranges_) {
        if (range.count > 0) {
            total_count += static_cast<size_t>(range.count);
        }
    }

    if (total_count == 0) {
        return IntDetached();
    }

    std::vector<int> dirty_global_ids;
    dirty_global_ids.reserve(total_count);
    for (const EdgeDirtyRange &range : pending_edge_bvh_dirty_ranges_) {
        for (int index = 0; index < range.count; ++index) {
            dirty_global_ids.push_back(range.offset + index);
        }
    }

    return load<IntDetached>(dirty_global_ids.data(), dirty_global_ids.size());
}

void Scene::ensure_scene_edge_data_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    bool updated_edge_data = false;
    for (const SceneMeshRecord &record : mesh_records_) {
        if (!record.edge_dirty) {
            continue;
        }

        const_cast<Scene *>(this)->scatter_mesh_edge_data(record, false);
        record.edge_dirty = false;
        updated_edge_data = true;
    }

    if (updated_edge_data) {
        drjit::eval(edge_info_);
        drjit::sync_thread();
    }

    ensure_edge_bvh_ready();
}

void Scene::ensure_edge_bvh_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    if (mask_dirty_) {
        const_cast<Scene *>(this)->rebuild_active_edge_bvh();
        return;
    }

    if (pending_edge_bvh_dirty_ranges_.empty()) {
        edge_bvh_dirty_ = false;
        return;
    }

    if (all_edges_active_) {
        edge_bvh_->refit(edge_info_, pending_edge_bvh_dirty_ranges_);
        pending_edge_bvh_dirty_ranges_.clear();
        edge_bvh_dirty_ = false;
        return;
    }

    const IntDetached dirty_global_ids = dirty_global_edge_ids_from_ranges();
    if (dirty_global_ids.size() == 0) {
        pending_edge_bvh_dirty_ranges_.clear();
        edge_bvh_dirty_ = false;
        return;
    }

    const IntDetached mapped_active_ids =
        gather<IntDetached>(global_to_active_edge_id_, dirty_global_ids);
    const MaskDetached active_dirty = mapped_active_ids >= 0;
    const IntDetached active_indices = compressD(mapped_active_ids, active_dirty);
    if (active_indices.size() > 0) {
        const IntDetached global_indices = compressD(dirty_global_ids, active_dirty);
        Scene *scene = const_cast<Scene *>(this);
        scene->scatter_active_edge_data(active_indices, global_indices);
        drjit::eval(scene->active_edge_info_.start,
                    scene->active_edge_info_.edge,
                    scene->active_edge_info_.normal0,
                    scene->active_edge_info_.normal1,
                    scene->active_edge_info_.opposite,
                    scene->active_edge_info_.is_boundary);
        drjit::sync_thread();
        edge_bvh_->refit(active_edge_info_, active_indices);
    }

    pending_edge_bvh_dirty_ranges_.clear();
    edge_bvh_dirty_ = false;
}

void Scene::register_primary_edge_observer(Camera *camera) {
    auto it = std::find(primary_edge_observers_.begin(), primary_edge_observers_.end(), camera);
    if (it == primary_edge_observers_.end()) {
        primary_edge_observers_.push_back(camera);
    }
}

void Scene::unregister_primary_edge_observer(Camera *camera) {
    auto it = std::remove(primary_edge_observers_.begin(), primary_edge_observers_.end(), camera);
    primary_edge_observers_.erase(it, primary_edge_observers_.end());
}

void Scene::invalidate_primary_edge_observers() {
    for (Camera *camera : primary_edge_observers_) {
        if (camera != nullptr) {
            camera->invalidate_primary_edges_from_scene(this);
        }
    }
}

void Scene::build() {
    require(!mesh_records_.empty(), "Scene::build(): missing meshes.");

    std::vector<int> face_offsets;
    face_offsets.reserve(mesh_records_.size() + 1);
    face_offsets.push_back(0);

    std::vector<int> edge_offsets;
    edge_offsets.reserve(mesh_records_.size() + 1);
    edge_offsets.push_back(0);

    std::vector<OptixSceneMeshDesc> mesh_descs;
    mesh_descs.reserve(mesh_records_.size());

    std::vector<int> topology_v0;
    std::vector<int> topology_v1;
    std::vector<int> topology_face0_local;
    std::vector<int> topology_face1_local;
    std::vector<int> topology_face0_global;
    std::vector<int> topology_face1_global;
    std::vector<int> topology_opposite0;
    std::vector<int> topology_opposite1;

    for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
        SceneMeshRecord &record = mesh_records_[mesh_index];
        Mesh &mesh = *record.mesh;
        mesh.set_mesh_id(static_cast<int>(mesh_index));
        mesh.build();
        record.face_offset = face_offsets.back();
        const SecondaryEdgeInfo *mesh_edge_info = mesh.secondary_edge_info();
        const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
        record.edge_offset = edge_offsets.back();
        record.vertices_dirty = false;
        record.transform_dirty = false;
        record.edge_dirty = false;

        face_offsets.push_back(face_offsets.back() + mesh.face_count());
        edge_offsets.push_back(edge_offsets.back() + mesh_edge_count);
        mesh_descs.push_back({ &mesh, record.dynamic, record.face_offset, static_cast<int>(mesh_index) });
    }

    mesh_count_ = static_cast<int>(mesh_records_.size());
    const int total_face_count = face_offsets.back();
    require(total_face_count > 0, "Scene::build(): scene has no triangles.");

    edge_count_ = edge_offsets.back();
    topology_v0.reserve(edge_count_);
    topology_v1.reserve(edge_count_);
    topology_face0_local.reserve(edge_count_);
    topology_face1_local.reserve(edge_count_);
    topology_face0_global.reserve(edge_count_);
    topology_face1_global.reserve(edge_count_);
    topology_opposite0.reserve(edge_count_);
    topology_opposite1.reserve(edge_count_);

    std::array<std::vector<int>, 3> triangle_edge_ids_cpu;
    for (auto &triangle_edge_ids : triangle_edge_ids_cpu) {
        triangle_edge_ids.assign(total_face_count, -1);
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        const Mesh &mesh = *record.mesh;
        const auto &mesh_edge_indices = mesh.edge_indices();
        const int mesh_edge_count = mesh.edges_enabled() ? static_cast<int>(slices(mesh_edge_indices)) : 0;
        if (mesh_edge_count == 0) {
            continue;
        }

        std::array<std::vector<int>, 5> mesh_edge_cpu;
        copy_cuda_array(mesh_edge_indices, mesh_edge_cpu);

        const Vector3iDetached mesh_face_indices(detach<false>(mesh.face_indices()[0]),
                                                 detach<false>(mesh.face_indices()[1]),
                                                 detach<false>(mesh.face_indices()[2]));
        std::array<std::vector<int>, 3> mesh_face_cpu;
        copy_cuda_array(mesh_face_indices, mesh_face_cpu);

        for (int local_edge_id = 0; local_edge_id < mesh_edge_count; ++local_edge_id) {
            const int v0 = mesh_edge_cpu[0][local_edge_id];
            const int v1 = mesh_edge_cpu[1][local_edge_id];
            const int face0_local = mesh_edge_cpu[2][local_edge_id];
            const int face1_local = mesh_edge_cpu[3][local_edge_id];
            const int face0_global = record.face_offset + face0_local;
            const int face1_global = face1_local >= 0 ? record.face_offset + face1_local : -1;
            const int opposite0 = mesh_edge_cpu[4][local_edge_id];
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
                const int face1_slot = face_edge_slot(face1_vertices, v0, v1);
                if (face1_slot >= 0) {
                    triangle_edge_ids_cpu[face1_slot][face1_global] = global_edge_id;
                }
            }

            const int face0_slot = face_edge_slot(face0_vertices, v0, v1);
            if (face0_slot >= 0) {
                triangle_edge_ids_cpu[face0_slot][face0_global] = global_edge_id;
            }

            topology_v0.push_back(v0);
            topology_v1.push_back(v1);
            topology_face0_local.push_back(face0_local);
            topology_face1_local.push_back(face1_local);
            topology_face0_global.push_back(face0_global);
            topology_face1_global.push_back(face1_global);
            topology_opposite0.push_back(opposite0);
            topology_opposite1.push_back(opposite1);
        }
    }

    auto load_or_empty = [](const std::vector<int> &values) {
        return values.empty() ? IntDetached() : load<IntDetached>(values.data(), values.size());
    };

    face_offsets_ = load<IntDetached>(face_offsets.data(), face_offsets.size());
    edge_offsets_ = load<IntDetached>(edge_offsets.data(), edge_offsets.size());
    triangle_info_ = empty<TriangleInfo>(total_face_count);
    triangle_info_detached_ = empty<TriangleInfoDetached>(total_face_count);
    triangle_uv_ = zeros<TriangleUV>(total_face_count);
    triangle_uv_detached_ = zeros<TriangleUVDetached>(total_face_count);
    triangle_face_normal_mask_ = empty<Mask>(total_face_count);
    triangle_face_normal_mask_detached_ = empty<MaskDetached>(total_face_count);
    triangle_edge_ids_ = VectoriT<3, true>(load<IntDetached>(triangle_edge_ids_cpu[0].data(), total_face_count),
                                           load<IntDetached>(triangle_edge_ids_cpu[1].data(), total_face_count),
                                           load<IntDetached>(triangle_edge_ids_cpu[2].data(), total_face_count));
    if (edge_count_ > 0) {
        edge_info_ = empty<SecondaryEdgeInfo>(edge_count_);
        active_edge_info_ = SecondaryEdgeInfo();
        edge_topology_ = SceneEdgeTopology {
            load_or_empty(topology_v0),
            load_or_empty(topology_v1),
            load_or_empty(topology_face0_local),
            load_or_empty(topology_face1_local),
            load_or_empty(topology_face0_global),
            load_or_empty(topology_face1_global),
            load_or_empty(topology_opposite0),
            load_or_empty(topology_opposite1)
        };
        edge_shape_ids_ = empty<IntDetached>(edge_count_);
        edge_local_ids_ = empty<IntDetached>(edge_count_);
        edge_mask_ = full<MaskDetached>(true, edge_count_);
        active_edge_global_ids_ = IntDetached();
        global_to_active_edge_id_ = IntDetached();
    } else {
        edge_info_ = SecondaryEdgeInfo();
        active_edge_info_ = SecondaryEdgeInfo();
        edge_topology_ = SceneEdgeTopology();
        edge_shape_ids_ = IntDetached();
        edge_local_ids_ = IntDetached();
        edge_mask_ = MaskDetached();
        active_edge_global_ids_ = IntDetached();
        global_to_active_edge_id_ = IntDetached();
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        scatter_mesh_data(record, true);
        scatter_mesh_edge_data(record, true);
    }

    drjit::eval(face_offsets_,
                triangle_info_,
                triangle_info_detached_,
                triangle_uv_,
                triangle_uv_detached_,
                triangle_face_normal_mask_,
                triangle_face_normal_mask_detached_,
                edge_offsets_,
                triangle_edge_ids_,
                edge_info_,
                edge_mask_,
                edge_topology_,
                edge_shape_ids_,
                edge_local_ids_);
    drjit::sync_thread();

    optix_scene_->build(mesh_descs);
    all_edges_active_ = true;
    mask_dirty_ = false;
    rebuild_active_edge_bvh();
    is_ready_ = true;
    pending_updates_ = false;
    ++scene_version_;
    ++edge_version_;
    invalidate_primary_edge_observers();
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
    all_edges_active_ = edge_count_ == 0 || drjit::all(edge_mask_);
    mask_dirty_ = true;
    edge_bvh_dirty_ = true;
    pending_updates_ = true;
}

void Scene::sync() {
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
        const auto eval_start = Clock::now();
        drjit::eval(triangle_info_.p0,
                    triangle_info_.e1,
                    triangle_info_.e2,
                    triangle_info_.n0,
                    triangle_info_.n1,
                    triangle_info_.n2,
                    triangle_info_.face_normal,
                    triangle_info_.face_area,
                    triangle_info_detached_.p0,
                    triangle_info_detached_.e1,
                    triangle_info_detached_.e2,
                    triangle_info_detached_.n0,
                    triangle_info_detached_.n1,
                    triangle_info_detached_.n2,
                    triangle_info_detached_.face_normal,
                    triangle_info_detached_.face_area);
        drjit::sync_thread();
        last_sync_profile_.triangle_eval_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - eval_start).count();
    }

    if (edge_bvh_dirty_) {
        const auto edge_scatter_start = Clock::now();
        bool updated_edge_data = false;
        for (SceneMeshRecord &record : mesh_records_) {
            if (!record.edge_dirty) {
                continue;
            }

            scatter_mesh_edge_data(record, false);
            record.edge_dirty = false;
            updated_edge_data = true;
        }

        if (updated_edge_data) {
            drjit::eval(edge_info_);
            drjit::sync_thread();
        }
        last_sync_profile_.edge_scatter_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - edge_scatter_start).count();

        const auto edge_refit_start = Clock::now();
        ensure_edge_bvh_ready();
        last_sync_profile_.edge_refit_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - edge_refit_start).count();
    }

    const auto optix_start = Clock::now();
    optix_scene_->sync(mesh_descs, updates);
    last_sync_profile_.optix_sync_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - optix_start).count();
    const OptixSyncProfile &optix_profile = optix_scene_->last_sync_profile();
    last_sync_profile_.optix_gas_update_ms = optix_profile.gas_update_ms;
    last_sync_profile_.optix_ias_update_ms = optix_profile.ias_update_ms;
    pending_updates_ = false;
    if (!updates.empty()) {
        ++scene_version_;
    }
    if (mask_dirty_before || last_sync_profile_.updated_edge_meshes > 0) {
        ++edge_version_;
    }
    if (!updates.empty()) {
        invalidate_primary_edge_observers();
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

const SceneEdgeTopology &Scene::edge_topology() const {
    require(is_ready(), "Scene::edge_topology(): scene is not built.");
    return edge_topology_;
}

const MaskDetached &Scene::edge_mask() const {
    require(is_ready(), "Scene::edge_mask(): scene is not built.");
    return edge_mask_;
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
    return is_ready_ && optix_scene_ != nullptr && edge_bvh_ != nullptr &&
           optix_scene_->is_ready() && edge_bvh_->is_ready();
}

template <bool Detached>
IntersectionT<Detached> Scene::intersect(const RayT<Detached> &ray, MaskT<Detached> active, RayFlags flags) const {
    require(is_ready(), "Scene::intersect(): scene is not built.");
    require(!pending_updates_, "Scene::intersect(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(ray.o));
    const bool want_geo_n   = has_flag(flags, RayFlags::Geometric);
    const bool want_shading = has_flag(flags, RayFlags::ShadingN);
    const bool want_uv      = has_flag(flags, RayFlags::UV);

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
    OptixIntersection optix_hit = optix_scene_->template intersect<Detached>(ray, hit_mask);

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
    intersection.prim_id = select(hit_mask, IntT<Detached>(local_primitive_id), intersection.prim_id);
    return intersection;
}

template <bool Detached>
MaskT<Detached> Scene::shadow_test(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::shadow_test(): scene is not built.");
    require(!pending_updates_, "Scene::shadow_test(): scene has pending updates. Call Scene::sync() first.");

    return optix_scene_->template shadow_test<Detached>(ray, active);
}

template <bool Detached>
NearestPointEdgeT<Detached> Scene::nearest_edge(const Vector3fT<Detached> &point, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(point): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(point): scene has pending updates. Call Scene::sync() first.");

    const int query_count = static_cast<int>(slices(point));
    NearestPointEdgeT<Detached> result = initialize_nearest_point_edge_result<Detached>(query_count);
    if (edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    MaskDetached active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x()));
        active_detached &= drjit::isfinite(detach<false>(point.y()));
        active_detached &= drjit::isfinite(detach<false>(point.z()));
        active &= Mask(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) && drjit::isfinite(point.y()) && drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    ClosestEdgeCandidate candidate = edge_bvh_->template nearest_edge<Detached>(point, query_mask);
    const MaskDetached valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const IntDetached global_edge_id_detached = all_edges_active_
        ? candidate.global_edge_id
        : gather<IntDetached>(active_edge_global_ids_, candidate.global_edge_id, valid_detached);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(global_edge_id_detached);
        const Vector3f p0 = gather<Vector3f>(edge_info_.start, global_edge_id, valid);
        const Vector3f e1 = gather<Vector3f>(edge_info_.edge, global_edge_id, valid);
        const Mask is_boundary = gather<Mask>(edge_info_.is_boundary, global_edge_id, valid);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<false>(point, p0, e1);

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.point = select(valid, point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, Int(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, Int(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3fDetached p0 =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3fDetached e1 =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const MaskDetached is_boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        FloatDetached edge_t;
        Vector3fDetached edge_point;
        FloatDetached distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<true>(point, p0, e1);

        result.distance = select(valid_detached, sqrt(distance_sq), result.distance);
        result.point = select(valid_detached, point, result.point);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_point = select(valid_detached, edge_point, result.edge_point);
        result.shape_id = select(valid_detached, shape_id_detached, result.shape_id);
        result.edge_id = select(valid_detached, edge_id_detached, result.edge_id);
        result.global_edge_id = select(valid_detached, global_edge_id_detached, result.global_edge_id);
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> Scene::nearest_edge(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(ray): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(ray): scene has pending updates. Call Scene::sync() first.");

    const int query_count = static_cast<int>(slices(ray.o));
    NearestRayEdgeT<Detached> result = initialize_nearest_ray_edge_result<Detached>(query_count);
    if (edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    FloatDetached t_max_input;
    MaskDetached active_detached;
    if constexpr (!Detached) {
        t_max_input = detach<false>(ray.tmax);
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) &&
                           drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) &&
                           drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &= squared_norm(Vector3fDetached(detach<false>(ray.d.x()),
                                                        detach<false>(ray.d.y()),
                                                        detach<false>(ray.d.z()))) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active &= Mask(active_detached);
    } else {
        t_max_input = ray.tmax;
        active_detached = active;
        active_detached &= drjit::isfinite(ray.o.x()) && drjit::isfinite(ray.o.y()) && drjit::isfinite(ray.o.z());
        active_detached &= drjit::isfinite(ray.d.x()) && drjit::isfinite(ray.d.y()) && drjit::isfinite(ray.d.z());
        active_detached &= squared_norm(ray.d) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    ClosestEdgeCandidate candidate = edge_bvh_->template nearest_edge<Detached>(ray, query_mask);
    const MaskDetached valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const MaskDetached finite_tmax = drjit::isfinite(t_max_input);
    const IntDetached global_edge_id_detached = all_edges_active_
        ? candidate.global_edge_id
        : gather<IntDetached>(active_edge_global_ids_, candidate.global_edge_id, valid_detached);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(global_edge_id_detached);
        const Vector3f p0 = gather<Vector3f>(edge_info_.start, global_edge_id, valid);
        const Vector3f e1 = gather<Vector3f>(edge_info_.edge, global_edge_id, valid);
        const Mask is_boundary = gather<Mask>(edge_info_.is_boundary, global_edge_id, valid);

        const Mask finite_mask = valid && Mask(finite_tmax);
        const Mask infinite_mask = valid && !Mask(finite_tmax);
        const Float safe_tmax = select(finite_mask, Float(t_max_input), zeros<Float>(query_count));

        Float query_t = zeros<Float>(query_count);
        Vector3f query_point = zeros<Vector3f>(query_count);
        Float edge_t = zeros<Float>(query_count);
        Vector3f edge_point = zeros<Vector3f>(query_count);
        Float distance_sq = full<Float>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            Float segment_query_t;
            Vector3f segment_query_point;
            Float segment_edge_t;
            Vector3f segment_edge_point;
            Float segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<false>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            Float ray_query_t;
            Vector3f ray_query_point;
            Float ray_edge_t;
            Vector3f ray_edge_point;
            Float ray_distance_sq;
            std::tie(ray_query_t, ray_query_point, ray_edge_t, ray_edge_point, ray_distance_sq) =
                closest_ray_segment<false>(ray.o, ray.d, p0, e1);

            query_t = select(infinite_mask, ray_query_t, query_t);
            query_point = select(infinite_mask, ray_query_point, query_point);
            edge_t = select(infinite_mask, ray_edge_t, edge_t);
            edge_point = select(infinite_mask, ray_edge_point, edge_point);
            distance_sq = select(infinite_mask, ray_distance_sq, distance_sq);
        }

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.ray_t = select(valid, query_t, result.ray_t);
        result.point = select(valid, query_point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, Int(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, Int(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3fDetached p0 =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3fDetached e1 =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const MaskDetached is_boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        const MaskDetached finite_mask = valid_detached && finite_tmax;
        const MaskDetached infinite_mask = valid_detached && !finite_tmax;
        const FloatDetached safe_tmax = select(finite_mask, t_max_input, zeros<FloatDetached>(query_count));

        FloatDetached query_t = zeros<FloatDetached>(query_count);
        Vector3fDetached query_point = zeros<Vector3fDetached>(query_count);
        FloatDetached edge_t = zeros<FloatDetached>(query_count);
        Vector3fDetached edge_point = zeros<Vector3fDetached>(query_count);
        FloatDetached distance_sq = full<FloatDetached>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            FloatDetached segment_query_t;
            Vector3fDetached segment_query_point;
            FloatDetached segment_edge_t;
            Vector3fDetached segment_edge_point;
            FloatDetached segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<true>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            FloatDetached ray_query_t;
            Vector3fDetached ray_query_point;
            FloatDetached ray_edge_t;
            Vector3fDetached ray_edge_point;
            FloatDetached ray_distance_sq;
            std::tie(ray_query_t, ray_query_point, ray_edge_t, ray_edge_point, ray_distance_sq) =
                closest_ray_segment<true>(ray.o, ray.d, p0, e1);

            query_t = select(infinite_mask, ray_query_t, query_t);
            query_point = select(infinite_mask, ray_query_point, query_point);
            edge_t = select(infinite_mask, ray_edge_t, edge_t);
            edge_point = select(infinite_mask, ray_edge_point, edge_point);
            distance_sq = select(infinite_mask, ray_distance_sq, distance_sq);
        }

        result.distance = select(valid_detached, sqrt(distance_sq), result.distance);
        result.ray_t = select(valid_detached, query_t, result.ray_t);
        result.point = select(valid_detached, query_point, result.point);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_point = select(valid_detached, edge_point, result.edge_point);
        result.shape_id = select(valid_detached, shape_id_detached, result.shape_id);
        result.edge_id = select(valid_detached, edge_id_detached, result.edge_id);
        result.global_edge_id = select(valid_detached, global_edge_id_detached, result.global_edge_id);
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template IntersectionDetached Scene::intersect<true>(const RayDetached &ray, MaskDetached active, RayFlags flags) const;
template Intersection Scene::intersect<false>(const Ray &ray, Mask active, RayFlags flags) const;
template MaskDetached Scene::shadow_test<true>(const RayDetached &ray, MaskDetached active) const;
template Mask Scene::shadow_test<false>(const Ray &ray, Mask active) const;
template NearestPointEdgeDetached Scene::nearest_edge<true>(const Vector3fDetached &point, MaskDetached active) const;
template NearestPointEdge Scene::nearest_edge<false>(const Vector3f &point, Mask active) const;
template NearestRayEdgeDetached Scene::nearest_edge<true>(const RayDetached &ray, MaskDetached active) const;
template NearestRayEdge Scene::nearest_edge<false>(const Ray &ray, Mask active) const;

} // namespace rayd

