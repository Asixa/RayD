#include <algorithm>
#include <chrono>
#include <sstream>
#include <vector>

#include <raydi/intersection.h>
#include <raydi/camera.h>
#include <raydi/ray.h>
#include <raydi/scene/scene.h>
#include <raydi/scene/scene_edge.h>

namespace raydi {

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
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
}

} // namespace

Scene::Scene()
    : optix_scene_(std::make_unique<OptixScene>()),
      edge_bvh_(std::make_unique<SceneEdge>()) {}

Scene::~Scene() {
    for (PerspectiveCamera *camera : primary_edge_observers_) {
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

void Scene::scatter_mesh_edge_data(const SceneMeshRecord &record) {
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

    bool updated_edge_data = false;
    for (const SceneMeshRecord &record : mesh_records_) {
        if (!record.edge_dirty) {
            continue;
        }

        const_cast<Scene *>(this)->scatter_mesh_edge_data(record);
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
    if (!edge_bvh_dirty_ || pending_edge_bvh_dirty_ranges_.empty()) {
        edge_bvh_dirty_ = false;
        return;
    }

    edge_bvh_->refit(edge_info_, pending_edge_bvh_dirty_ranges_);
    pending_edge_bvh_dirty_ranges_.clear();
    edge_bvh_dirty_ = false;
}

void Scene::register_primary_edge_observer(PerspectiveCamera *camera) {
    auto it = std::find(primary_edge_observers_.begin(), primary_edge_observers_.end(), camera);
    if (it == primary_edge_observers_.end()) {
        primary_edge_observers_.push_back(camera);
    }
}

void Scene::unregister_primary_edge_observer(PerspectiveCamera *camera) {
    auto it = std::remove(primary_edge_observers_.begin(), primary_edge_observers_.end(), camera);
    primary_edge_observers_.erase(it, primary_edge_observers_.end());
}

void Scene::invalidate_primary_edge_observers() {
    for (PerspectiveCamera *camera : primary_edge_observers_) {
        if (camera != nullptr) {
            camera->invalidate_primary_edges_from_scene(this);
        }
    }
}

void Scene::configure() {
    require(!mesh_records_.empty(), "Scene::configure(): missing meshes.");

    std::vector<int> face_offsets;
    face_offsets.reserve(mesh_records_.size() + 1);
    face_offsets.push_back(0);

    std::vector<int> edge_offsets;
    edge_offsets.reserve(mesh_records_.size() + 1);
    edge_offsets.push_back(0);

    std::vector<OptixSceneMeshDesc> mesh_descs;
    mesh_descs.reserve(mesh_records_.size());

    for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
        SceneMeshRecord &record = mesh_records_[mesh_index];
        Mesh &mesh = *record.mesh;
        mesh.set_mesh_id(static_cast<int>(mesh_index));
        mesh.configure();
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
    require(total_face_count > 0, "Scene::configure(): scene has no triangles.");

    face_offsets_ = load<IntDetached>(face_offsets.data(), face_offsets.size());
    edge_offsets_ = load<IntDetached>(edge_offsets.data(), edge_offsets.size());
    triangle_info_ = empty<TriangleInfo>(total_face_count);
    triangle_info_detached_ = empty<TriangleInfoDetached>(total_face_count);
    triangle_uv_ = zeros<TriangleUV>(total_face_count);
    triangle_uv_detached_ = zeros<TriangleUVDetached>(total_face_count);
    triangle_face_normal_mask_ = empty<Mask>(total_face_count);
    triangle_face_normal_mask_detached_ = empty<MaskDetached>(total_face_count);
    edge_count_ = edge_offsets.back();
    if (edge_count_ > 0) {
        edge_info_ = empty<SecondaryEdgeInfo>(edge_count_);
        edge_shape_ids_ = empty<IntDetached>(edge_count_);
        edge_local_ids_ = empty<IntDetached>(edge_count_);
    } else {
        edge_info_ = SecondaryEdgeInfo();
        edge_shape_ids_ = IntDetached();
        edge_local_ids_ = IntDetached();
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        scatter_mesh_data(record, true);
        scatter_mesh_edge_data(record);
    }

    drjit::eval(face_offsets_,
                triangle_info_,
                triangle_info_detached_,
                triangle_uv_,
                triangle_uv_detached_,
                triangle_face_normal_mask_,
                triangle_face_normal_mask_detached_,
                edge_offsets_,
                edge_info_,
                edge_shape_ids_,
                edge_local_ids_);
    drjit::sync_thread();

    optix_scene_->configure(mesh_descs);
    edge_bvh_->configure(edge_info_);
    pending_edge_bvh_dirty_ranges_.clear();
    edge_bvh_dirty_ = false;
    is_ready_ = true;
    pending_updates_ = false;
    ++scene_version_;
    invalidate_primary_edge_observers();
}

void Scene::update_mesh_vertices(int mesh_id, const Vector3f &positions) {
    require(is_ready(), "Scene::update_mesh_vertices(): scene is not configured.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::update_mesh_vertices(): target mesh is not dynamic.");
    require(static_cast<int>(slices(positions)) == record.mesh->vertex_count(),
            "Scene::update_mesh_vertices(): vertex count must remain unchanged.");

    record.mesh->set_vertex_positions(positions);
    record.vertices_dirty = true;
    pending_updates_ = true;
}

void Scene::set_mesh_transform(int mesh_id, const Matrix4f &matrix, bool set_left) {
    require(is_ready(), "Scene::set_mesh_transform(): scene is not configured.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::set_mesh_transform(): target mesh is not dynamic.");

    record.mesh->set_transform(matrix, set_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::append_mesh_transform(int mesh_id, const Matrix4f &matrix, bool append_left) {
    require(is_ready(), "Scene::append_mesh_transform(): scene is not configured.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::append_mesh_transform(): target mesh is not dynamic.");

    record.mesh->append_transform(matrix, append_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::commit_updates() {
    require(is_ready(), "Scene::commit_updates(): scene is not configured.");
    last_commit_profile_ = SceneCommitProfile();

    if (!pending_updates_) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();

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
        last_commit_profile_.mesh_update_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - mesh_update_start).count();

        const auto scatter_start = Clock::now();
        scatter_mesh_data(record, false);
        last_commit_profile_.triangle_scatter_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - scatter_start).count();
        const int mesh_edge_count =
            record.mesh->edges_enabled() ? static_cast<int>(slices(record.mesh->edge_indices())) : 0;
        if (mesh_edge_count > 0 && !record.edge_dirty) {
            pending_edge_bvh_dirty_ranges_.push_back({ record.edge_offset, mesh_edge_count });
            record.edge_dirty = true;
            edge_bvh_dirty_ = true;
        }

        updates.push_back({ static_cast<int>(mesh_index), record.vertices_dirty, record.transform_dirty });
        ++last_commit_profile_.updated_meshes;
        if (record.vertices_dirty) {
            ++last_commit_profile_.updated_vertex_meshes;
        }
        if (record.transform_dirty) {
            ++last_commit_profile_.updated_transform_meshes;
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
        last_commit_profile_.triangle_eval_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - eval_start).count();
    }

    const auto optix_start = Clock::now();
    optix_scene_->commit_updates(mesh_descs, updates);
    last_commit_profile_.optix_commit_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - optix_start).count();
    const OptixCommitProfile &optix_profile = optix_scene_->last_commit_profile();
    last_commit_profile_.optix_gas_update_ms = optix_profile.gas_update_ms;
    last_commit_profile_.optix_ias_update_ms = optix_profile.ias_update_ms;
    pending_updates_ = false;
    ++scene_version_;
    invalidate_primary_edge_observers();
    last_commit_profile_.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
}

bool Scene::is_ready() const {
    return is_ready_ && optix_scene_ != nullptr && edge_bvh_ != nullptr &&
           optix_scene_->is_ready() && edge_bvh_->is_ready();
}

template <bool Detached>
IntersectionT<Detached> Scene::ray_intersect(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::ray_intersect(): scene is not configured.");
    require(!pending_updates_, "Scene::ray_intersect(): scene has pending updates. Call Scene::commit_updates() first.");

    const int ray_count = static_cast<int>(slices(ray.o));

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
    OptixIntersection optix_hit = optix_scene_->template ray_intersect<Detached>(ray, hit_mask);
    if (drjit::none(hit_mask)) {
        return intersection;
    }

    const IntDetached shape_id = optix_hit.shape_id;
    const IntDetached global_primitive_id = optix_hit.global_prim_id;
    const MaskDetached hit_mask_detached = detach<false>(hit_mask);
    const IntDetached mesh_face_offset = gather<IntDetached>(face_offsets_, shape_id, hit_mask_detached);
    const IntDetached local_primitive_id = global_primitive_id - mesh_face_offset;

    Vector3fT<Detached> shading_n0;
    Vector3fT<Detached> shading_n1;
    Vector3fT<Detached> shading_n2;
    Vector3fT<Detached> geometric_normal;
    TriangleUVT<Detached> triangle_uv;
    MaskT<Detached> use_face_normal_mask;
    Vector2fT<Detached> triangle_uv_coords;
    FloatT<Detached> hit_distance;

    if constexpr (!Detached) {
        const Int global_primitive_id_ad = Int(global_primitive_id);
        shading_n0 = gather<Vector3f>(triangle_info_.n0, global_primitive_id_ad, hit_mask);
        shading_n1 = gather<Vector3f>(triangle_info_.n1, global_primitive_id_ad, hit_mask);
        shading_n2 = gather<Vector3f>(triangle_info_.n2, global_primitive_id_ad, hit_mask);
        geometric_normal = gather<Vector3f>(triangle_info_.face_normal, global_primitive_id_ad, hit_mask);
        triangle_uv = gather<TriangleUV>(triangle_uv_, global_primitive_id_ad, hit_mask);
        use_face_normal_mask = gather<Mask>(triangle_face_normal_mask_, global_primitive_id_ad, hit_mask);
        const Vector3f triangle_p0 = gather<Vector3f>(triangle_info_.p0, global_primitive_id_ad, hit_mask);
        const Vector3f triangle_e1 = gather<Vector3f>(triangle_info_.e1, global_primitive_id_ad, hit_mask);
        const Vector3f triangle_e2 = gather<Vector3f>(triangle_info_.e2, global_primitive_id_ad, hit_mask);
        std::tie(triangle_uv_coords, hit_distance) = ray_intersect_triangle<Detached>(triangle_p0, triangle_e1, triangle_e2, ray);
    } else {
        shading_n0 = gather<Vector3fDetached>(triangle_info_detached_.n0, global_primitive_id, hit_mask_detached);
        shading_n1 = gather<Vector3fDetached>(triangle_info_detached_.n1, global_primitive_id, hit_mask_detached);
        shading_n2 = gather<Vector3fDetached>(triangle_info_detached_.n2, global_primitive_id, hit_mask_detached);
        geometric_normal = gather<Vector3fDetached>(triangle_info_detached_.face_normal, global_primitive_id, hit_mask_detached);
        triangle_uv = gather<TriangleUVDetached>(triangle_uv_detached_, global_primitive_id, hit_mask_detached);
        use_face_normal_mask = gather<MaskDetached>(triangle_face_normal_mask_detached_, global_primitive_id, hit_mask_detached);
        triangle_uv_coords = optix_hit.barycentric;
        hit_distance = optix_hit.t;
    }

    hit_mask &= drjit::isfinite(hit_distance) && (hit_distance < ray.tmax);
    if (drjit::none(hit_mask)) {
        return intersection;
    }

    const FloatT<Detached> safe_hit_distance = select(hit_mask, hit_distance, zeros<FloatT<Detached>>(ray_count));
    const Vector2fT<Detached> safe_triangle_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));

    const Vector3fT<Detached> barycentric_coordinates(1.f - safe_triangle_uv.x() - safe_triangle_uv.y(),
                                                      safe_triangle_uv.x(),
                                                      safe_triangle_uv.y());
    Vector3fT<Detached> shading_normal =
        normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_triangle_uv));
    shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);

    const Vector2fT<Detached> uv =
        bilinear2<Detached>(triangle_uv[0], triangle_uv[1] - triangle_uv[0], triangle_uv[2] - triangle_uv[0], safe_triangle_uv);
    const Vector3fT<Detached> hit_position = ray(safe_hit_distance);

    intersection.t = select(hit_mask, safe_hit_distance, intersection.t);
    intersection.p = select(hit_mask, hit_position, intersection.p);
    intersection.n = select(hit_mask, shading_normal, intersection.n);
    intersection.geo_n = select(hit_mask, geometric_normal, intersection.geo_n);
    intersection.uv = select(hit_mask, uv, intersection.uv);
    intersection.barycentric = select(hit_mask, barycentric_coordinates, intersection.barycentric);
    intersection.shape_id = select(hit_mask, IntT<Detached>(shape_id), intersection.shape_id);
    intersection.prim_id = select(hit_mask, IntT<Detached>(local_primitive_id), intersection.prim_id);
    return intersection;
}

template <bool Detached>
MaskT<Detached> Scene::shadow_test(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::shadow_test(): scene is not configured.");
    require(!pending_updates_, "Scene::shadow_test(): scene has pending updates. Call Scene::commit_updates() first.");

    return optix_scene_->template shadow_test<Detached>(ray, active);
}

template <bool Detached>
NearestPointEdgeT<Detached> Scene::closest_edge(const Vector3fT<Detached> &point, MaskT<Detached> active) const {
    require(is_ready(), "Scene::closest_edge(point): scene is not configured.");
    require(!pending_updates_, "Scene::closest_edge(point): scene has pending updates. Call Scene::commit_updates() first.");

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
    ClosestEdgeCandidate candidate = edge_bvh_->template closest_edge<Detached>(point, query_mask);
    const MaskDetached valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, candidate.global_edge_id, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, candidate.global_edge_id, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(candidate.global_edge_id);
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
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3fDetached p0 =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), candidate.global_edge_id, valid_detached);
        const Vector3fDetached e1 =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), candidate.global_edge_id, valid_detached);
        const MaskDetached is_boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), candidate.global_edge_id, valid_detached);

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
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> Scene::closest_edge(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::closest_edge(ray): scene is not configured.");
    require(!pending_updates_, "Scene::closest_edge(ray): scene has pending updates. Call Scene::commit_updates() first.");

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
    ClosestEdgeCandidate candidate = edge_bvh_->template closest_edge<Detached>(ray, query_mask);
    const MaskDetached valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const MaskDetached finite_tmax = drjit::isfinite(t_max_input);
    const IntDetached shape_id_detached =
        gather<IntDetached>(edge_shape_ids_, candidate.global_edge_id, valid_detached);
    const IntDetached edge_id_detached =
        gather<IntDetached>(edge_local_ids_, candidate.global_edge_id, valid_detached);

    if constexpr (!Detached) {
        const Mask valid = Mask(valid_detached);
        const Int global_edge_id = Int(candidate.global_edge_id);
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
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3fDetached p0 =
            gather<Vector3fDetached>(detach<false>(edge_info_.start), candidate.global_edge_id, valid_detached);
        const Vector3fDetached e1 =
            gather<Vector3fDetached>(detach<false>(edge_info_.edge), candidate.global_edge_id, valid_detached);
        const MaskDetached is_boundary =
            gather<MaskDetached>(detach<false>(edge_info_.is_boundary), candidate.global_edge_id, valid_detached);

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
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template IntersectionDetached Scene::ray_intersect<true>(const RayDetached &ray, MaskDetached active) const;
template Intersection Scene::ray_intersect<false>(const Ray &ray, Mask active) const;
template MaskDetached Scene::shadow_test<true>(const RayDetached &ray, MaskDetached active) const;
template Mask Scene::shadow_test<false>(const Ray &ray, Mask active) const;
template NearestPointEdgeDetached Scene::closest_edge<true>(const Vector3fDetached &point, MaskDetached active) const;
template NearestPointEdge Scene::closest_edge<false>(const Vector3f &point, Mask active) const;
template NearestRayEdgeDetached Scene::closest_edge<true>(const RayDetached &ray, MaskDetached active) const;
template NearestRayEdge Scene::closest_edge<false>(const Ray &ray, Mask active) const;

} // namespace raydi

