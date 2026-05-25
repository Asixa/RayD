#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include <rayd/scene/scene.h>
#include <rayd/edge/scene_edge.h>
#include <rayd/native_launch_audit.h>

namespace rayd {

namespace {
/// Whether to split static and dynamic meshes into separate OptiX scenes (env RAYD_OPTIX_SPLIT_MODE).
enum class OptixSplitMode {
    Auto,
    Off,
    On
};

std::string normalize_optix_split_mode_value(const char *value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

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

/// Map a backend name ("drjit"/"optix"/"hybrid" and aliases) to EdgeBVHBackend; throws on unknown.
EdgeBVHBackend parse_edge_backend(const std::string &value) {
    const std::string normalized = normalize_edge_backend_value(value);
    if (normalized.empty() || normalized == "optix" ||
        normalized == "custom_aabb") {
        return EdgeBVHBackend::Optix;
    }
    if (normalized == "drjit" || normalized == "dr_jit" ||
        normalized == "software") {
        return EdgeBVHBackend::DrJit;
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

bool edge_backend_uses_optix_point(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

bool edge_backend_uses_optix_ray(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_uses_optix_topk(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
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

void Scene::reset_multipath_pipelines() {
    reflection_pipeline_.reset();
    reflection_accumulation_pipeline_.reset();
    diffraction_order1_accumulation_pipeline_.reset();
    diffraction_chain_accumulation_pipeline_.reset();
    diffraction_coherent_accumulation_pipeline_.reset();
    diffraction_paths_primary_pipeline_.reset();
    diffraction_paths_pipeline_.reset();
    reflection_epc_pipeline_.reset();
    reflection_epc_geometry_ready_ = false;
    segment_visibility_pipeline_.reset();
    segment_pair_visibility_pipeline_.reset();
    axial_edge_visibility_pipeline_.reset();
    segment_chain_visibility_pipeline_.reset();
}

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
    vertex_offsets_ = Int();
    global_geometry_ = SceneGeometry();
    edge_mask_ = Mask();
    pending_edge_bvh_dirty_ranges_.clear();
    edge_bvh_dirty_ = false;
    mask_dirty_ = false;
    optix_split_active_ = false;
    optix_static_mesh_indices_.clear();
    optix_dynamic_mesh_indices_.clear();
    optix_dynamic_mesh_local_index_.clear();
    reset_multipath_pipelines();
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

    const TriangleInfoAD *mesh_triangle_info = mesh.triangle_info();
    const IntAD scatter_indices = arange<IntAD>(mesh_face_count) + record.face_offset;
    const Int scatter_indices_detached =
        arange<Int>(mesh_face_count) + record.face_offset;

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
            full<MaskAD>(mesh.use_face_normals(), mesh_face_count),
            scatter_indices);
    scatter(triangle_face_normal_mask_detached_,
            full<Mask>(mesh.use_face_normals(), mesh_face_count),
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
    const SecondaryEdgeInfoAD *mesh_edge_info = mesh.secondary_edge_info();
    const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
    if (mesh_edge_count == 0) {
        return;
    }

    const IntAD scatter_indices = arange<IntAD>(mesh_edge_count) + record.edge_offset;
    scatter(edge_info_.start, mesh_edge_info->start, scatter_indices);
    scatter(edge_info_.edge, mesh_edge_info->edge, scatter_indices);
    scatter(edge_info_.normal0, mesh_edge_info->normal0, scatter_indices);
    scatter(edge_info_.normal1, mesh_edge_info->normal1, scatter_indices);
    scatter(edge_info_.opposite, mesh_edge_info->opposite, scatter_indices);
    scatter(edge_info_.is_boundary, mesh_edge_info->is_boundary, scatter_indices);

    if (!include_static_ids) {
        return;
    }

    const Int scatter_indices_detached = arange<Int>(mesh_edge_count) + record.edge_offset;
    scatter(edge_shape_ids_,
            full<Int>(mesh.mesh_id(), mesh_edge_count),
            scatter_indices_detached);
    scatter(edge_local_ids_,
            arange<Int>(mesh_edge_count),
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
        const SecondaryEdgeInfoAD *mesh_edge_info = mesh.secondary_edge_info();
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
        const Vector3i mesh_face_indices(detach<false>(mesh.face_indices()[0]),
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
        return values.empty() ? Int() : load<Int>(values.data(), values.size());
    };

    face_offsets_ = load<Int>(face_offsets.data(), face_offsets.size());
    edge_offsets_ = load<Int>(edge_offsets.data(), edge_offsets.size());
    vertex_offsets_ = load<Int>(vertex_offsets.data(), vertex_offsets.size());
    triangle_info_ = empty<TriangleInfoAD>(total_face_count);
    triangle_info_detached_ = empty<TriangleInfo>(total_face_count);
    triangle_uv_ = zeros<TriangleUVAD>(total_face_count);
    triangle_uv_detached_ = zeros<TriangleUV>(total_face_count);
    triangle_face_normal_mask_ = empty<MaskAD>(total_face_count);
    triangle_face_normal_mask_detached_ = empty<Mask>(total_face_count);
    global_geometry_.vertices = total_vertex_count > 0 ? empty<Vector3fAD>(total_vertex_count) : Vector3fAD();
    global_geometry_.faces = Vector3i(
        load<Int>(global_face_indices_cpu[0].data(), total_face_count),
        load<Int>(global_face_indices_cpu[1].data(), total_face_count),
        load<Int>(global_face_indices_cpu[2].data(), total_face_count));
    global_geometry_.shape_id = load<Int>(global_shape_ids_cpu.data(), total_face_count);
    global_geometry_.local_prim_id =
        load<Int>(global_local_prim_ids_cpu.data(), total_face_count);
    global_geometry_.global_prim_id = load<Int>(global_prim_ids_cpu.data(), total_face_count);
    triangle_edge_ids_ = VectoriT<3, true>(load<Int>(triangle_edge_ids_cpu[0].data(), total_face_count),
                                           load<Int>(triangle_edge_ids_cpu[1].data(), total_face_count),
                                           load<Int>(triangle_edge_ids_cpu[2].data(), total_face_count));
    if (edge_count_ > 0) {
        edge_info_ = empty<SecondaryEdgeInfoAD>(edge_count_);
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
        edge_shape_ids_ = empty<Int>(edge_count_);
        edge_local_ids_ = empty<Int>(edge_count_);
        edge_mask_ = full<Mask>(true, edge_count_);
    } else {
        edge_info_ = SecondaryEdgeInfoAD();
        edge_topology_ = SceneEdgeTopology();
        edge_shape_ids_ = Int();
        edge_local_ids_ = Int();
        edge_mask_ = Mask();
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        scatter_mesh_data(record, true);
        scatter_mesh_edge_data(record, true);
        const Mesh &mesh = *record.mesh;
        const int mesh_vertex_count = mesh.vertex_count();
        if (mesh_vertex_count > 0) {
            const IntAD vertex_scatter_indices = arange<IntAD>(mesh_vertex_count) + record.vertex_offset;
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
    reset_multipath_pipelines();

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

void Scene::update_mesh_vertices(int mesh_id, const Vector3fAD &positions) {
    require(is_ready(), "Scene::update_mesh_vertices(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::update_mesh_vertices(): target mesh is not dynamic.");
    require(static_cast<int>(slices(positions)) == record.mesh->vertex_count(),
            "Scene::update_mesh_vertices(): vertex count must remain unchanged.");

    record.mesh->set_vertex_positions(positions);
    record.vertices_dirty = true;
    pending_updates_ = true;
}

void Scene::set_mesh_transform(int mesh_id, const Matrix4fAD &matrix, bool set_left) {
    require(is_ready(), "Scene::set_mesh_transform(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::set_mesh_transform(): target mesh is not dynamic.");

    record.mesh->set_transform(matrix, set_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::append_mesh_transform(int mesh_id, const Matrix4fAD &matrix, bool append_left) {
    require(is_ready(), "Scene::append_mesh_transform(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::append_mesh_transform(): target mesh is not dynamic.");

    record.mesh->append_transform(matrix, append_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::set_edge_mask(const Mask &mask) {
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
            const IntAD vertex_scatter_indices = arange<IntAD>(mesh_vertex_count) + record.vertex_offset;
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
    info.global_edge_id = arange<Int>(edge_count_);
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

const Mask &Scene::edge_mask() const {
    require(is_ready(), "Scene::edge_mask(): scene is not built.");
    return edge_mask_;
}

const SceneGeometry &Scene::global_geometry() const {
    require(is_ready(), "Scene::global_geometry(): scene is not built.");
    require(!pending_updates_,
            "Scene::global_geometry(): scene has pending updates. Call Scene::sync() first.");
    return global_geometry_;
}

VectoriT<3, true> Scene::triangle_edge_indices(const Int &prim_id, bool global) const {
    require(is_ready(), "Scene::triangle_edge_indices(): scene is not built.");

    const int query_count = static_cast<int>(slices(prim_id));
    VectoriT<3, true> result(full<Int>(-1, query_count),
                             full<Int>(-1, query_count),
                             full<Int>(-1, query_count));
    if (query_count == 0) {
        return result;
    }

    const int face_count = static_cast<int>(slices(triangle_edge_ids_[0]));
    const Mask valid = prim_id >= 0 && prim_id < face_count;
    const Int edge0 = gather<Int>(triangle_edge_ids_[0], prim_id, valid);
    const Int edge1 = gather<Int>(triangle_edge_ids_[1], prim_id, valid);
    const Int edge2 = gather<Int>(triangle_edge_ids_[2], prim_id, valid);

    if (global) {
        result[0] = select(valid, edge0, result[0]);
        result[1] = select(valid, edge1, result[1]);
        result[2] = select(valid, edge2, result[2]);
        return result;
    }

    const Mask valid0 = valid && edge0 >= 0;
    const Mask valid1 = valid && edge1 >= 0;
    const Mask valid2 = valid && edge2 >= 0;
    result[0] = select(valid0, gather<Int>(edge_local_ids_, edge0, valid0), result[0]);
    result[1] = select(valid1, gather<Int>(edge_local_ids_, edge1, valid1), result[1]);
    result[2] = select(valid2, gather<Int>(edge_local_ids_, edge2, valid2), result[2]);
    return result;
}

VectoriT<2, true> Scene::edge_adjacent_faces(const Int &edge_id, bool global) const {
    require(is_ready(), "Scene::edge_adjacent_faces(): scene is not built.");

    const int query_count = static_cast<int>(slices(edge_id));
    VectoriT<2, true> result(full<Int>(-1, query_count),
                             full<Int>(-1, query_count));
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    const Mask valid = edge_id >= 0 && edge_id < edge_count_;
    const Int face0 = global
        ? gather<Int>(edge_topology_.face0_global, edge_id, valid)
        : gather<Int>(edge_topology_.face0_local, edge_id, valid);
    const Int face1 = global
        ? gather<Int>(edge_topology_.face1_global, edge_id, valid)
        : gather<Int>(edge_topology_.face1_local, edge_id, valid);
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

    Mask active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x()));
        active_detached &= drjit::isfinite(detach<false>(point.y()));
        active_detached &= drjit::isfinite(detach<false>(point.z()));
        active &= MaskAD(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) && drjit::isfinite(point.y()) && drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_point(edge_bvh_backend_);
    ClosestEdgeCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edge<Detached>(point, query_mask)
            : edge_bvh_->template nearest_edge<Detached>(point, query_mask);
    const Mask valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Int global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_id
            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const Int shape_id_detached =
        gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached =
        gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const MaskAD valid = MaskAD(valid_detached);
        const IntAD global_edge_id = IntAD(global_edge_id_detached);
        const Vector3fAD p0 = gather<Vector3fAD>(edge_info_.start, global_edge_id, valid);
        const Vector3fAD e1 = gather<Vector3fAD>(edge_info_.edge, global_edge_id, valid);
        const MaskAD is_boundary = gather<MaskAD>(edge_info_.is_boundary, global_edge_id, valid);

        FloatAD edge_t;
        Vector3fAD edge_point;
        FloatAD distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<false>(point, p0, e1);

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.point = select(valid, point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, IntAD(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, IntAD(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3f p0 =
            gather<Vector3f>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3f e1 =
            gather<Vector3f>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const Mask is_boundary =
            gather<Mask>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
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

    Float t_max_input;
    Mask active_detached;
    if constexpr (!Detached) {
        t_max_input = detach<false>(ray.tmax);
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) &&
                           drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) &&
                           drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &= squared_norm(Vector3f(detach<false>(ray.d.x()),
                                                        detach<false>(ray.d.y()),
                                                        detach<false>(ray.d.z()))) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active &= MaskAD(active_detached);
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
    const bool use_optix_candidate = edge_backend_uses_optix_ray(edge_bvh_backend_);
    ClosestEdgeCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edge<Detached>(ray, query_mask)
            : edge_bvh_->template nearest_edge<Detached>(ray, query_mask);
    const Mask valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Mask finite_tmax = drjit::isfinite(t_max_input);
    const Int global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_id
            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const Int shape_id_detached =
        gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached =
        gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const MaskAD valid = MaskAD(valid_detached);
        const IntAD global_edge_id = IntAD(global_edge_id_detached);
        const Vector3fAD p0 = gather<Vector3fAD>(edge_info_.start, global_edge_id, valid);
        const Vector3fAD e1 = gather<Vector3fAD>(edge_info_.edge, global_edge_id, valid);
        const MaskAD is_boundary = gather<MaskAD>(edge_info_.is_boundary, global_edge_id, valid);

        const MaskAD finite_mask = valid && MaskAD(finite_tmax);
        const MaskAD infinite_mask = valid && !MaskAD(finite_tmax);
        const FloatAD safe_tmax = select(finite_mask, FloatAD(t_max_input), zeros<FloatAD>(query_count));

        FloatAD query_t = zeros<FloatAD>(query_count);
        Vector3fAD query_point = zeros<Vector3fAD>(query_count);
        FloatAD edge_t = zeros<FloatAD>(query_count);
        Vector3fAD edge_point = zeros<Vector3fAD>(query_count);
        FloatAD distance_sq = full<FloatAD>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            FloatAD segment_query_t;
            Vector3fAD segment_query_point;
            FloatAD segment_edge_t;
            Vector3fAD segment_edge_point;
            FloatAD segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<false>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            FloatAD ray_query_t;
            Vector3fAD ray_query_point;
            FloatAD ray_edge_t;
            Vector3fAD ray_edge_point;
            FloatAD ray_distance_sq;
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
        result.shape_id = select(valid, IntAD(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, IntAD(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3f p0 =
            gather<Vector3f>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3f e1 =
            gather<Vector3f>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const Mask is_boundary =
            gather<Mask>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        const Mask finite_mask = valid_detached && finite_tmax;
        const Mask infinite_mask = valid_detached && !finite_tmax;
        const Float safe_tmax = select(finite_mask, t_max_input, zeros<Float>(query_count));

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
                closest_segment_segment<true>(ray.o, ray.d * safe_tmax, p0, e1);

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

template <bool Detached>
NearestEdgesTopKT<Detached> Scene::nearest_edges(const Vector3fT<Detached> &point,
                                                       int k,
                                                       MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edges(point): scene is not built.");
    require(!pending_updates_,
            "Scene::nearest_edges(point): scene has pending updates. Call Scene::sync() first.");
    require(k > 0, "Scene::nearest_edges(point): k must be positive.");
    require(k <= 16, "Scene::nearest_edges(point): k must be <= 16.");

    const int query_count = static_cast<int>(slices(point));
    const int output_count = query_count * k;
    NearestEdgesTopKT<Detached> result;
    result.query_count = query_count;
    result.k = k;
    result.is_valid = full<MaskT<Detached>>(false, output_count);
    result.distances = full<FloatT<Detached>>(Infinity, output_count);
    result.points = zeros<Vector3fT<Detached>>(output_count);
    result.edge_t = zeros<FloatT<Detached>>(output_count);
    result.edge_points = zeros<Vector3fT<Detached>>(output_count);
    result.shape_ids = full<IntT<Detached>>(-1, output_count);
    result.edge_ids = full<IntT<Detached>>(-1, output_count);
    result.global_edge_ids = full<IntT<Detached>>(-1, output_count);
    result.is_boundary = full<MaskT<Detached>>(false, output_count);
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    Mask active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x())) &&
                           drjit::isfinite(detach<false>(point.y())) &&
                           drjit::isfinite(detach<false>(point.z()));
        active &= MaskAD(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) &&
                           drjit::isfinite(point.y()) &&
                           drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_topk(edge_bvh_backend_);
    const ClosestEdgeTopKCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edges<Detached>(point, k, query_mask)
            : edge_bvh_->template nearest_edges<Detached>(point, k, query_mask);
    const Mask valid_detached = candidate.is_valid;
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Int output_index = arange<Int>(output_count);
    const Int output_query = output_index / k;
    const Int global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_ids
            : edge_bvh_->map_to_global(candidate.global_edge_ids, valid_detached);
    const Int shape_id_detached =
        gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached =
        gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const MaskAD valid = MaskAD(valid_detached);
        const IntAD global_edge_id = IntAD(global_edge_id_detached);
        const IntAD query_id = IntAD(output_query);
        const Vector3fAD output_point = gather<Vector3fAD>(point, query_id, valid);
        const Vector3fAD edge_start =
            gather<Vector3fAD>(edge_info_.start, global_edge_id, valid);
        const Vector3fAD edge_vector =
            gather<Vector3fAD>(edge_info_.edge, global_edge_id, valid);
        const MaskAD boundary =
            gather<MaskAD>(edge_info_.is_boundary, global_edge_id, valid);

        FloatAD edge_t;
        Vector3fAD edge_point;
        FloatAD distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<false>(output_point, edge_start, edge_vector);

        result.is_valid = valid;
        result.distances = select(valid, sqrt(distance_sq), result.distances);
        result.points = select(valid, output_point, result.points);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_points = select(valid, edge_point, result.edge_points);
        result.shape_ids = select(valid, IntAD(shape_id_detached), result.shape_ids);
        result.edge_ids = select(valid, IntAD(edge_id_detached), result.edge_ids);
        result.global_edge_ids = select(valid, global_edge_id, result.global_edge_ids);
        result.is_boundary = select(valid, boundary, result.is_boundary);
    } else {
        const Vector3f output_point =
            gather<Vector3f>(point, output_query, valid_detached);
        const Vector3f edge_start =
            gather<Vector3f>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3f edge_vector =
            gather<Vector3f>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const Mask boundary =
            gather<Mask>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<true>(output_point, edge_start, edge_vector);

        result.is_valid = valid_detached;
        result.distances = select(valid_detached, sqrt(distance_sq), result.distances);
        result.points = select(valid_detached, output_point, result.points);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_points = select(valid_detached, edge_point, result.edge_points);
        result.shape_ids = select(valid_detached, shape_id_detached, result.shape_ids);
        result.edge_ids = select(valid_detached, edge_id_detached, result.edge_ids);
        result.global_edge_ids = select(valid_detached, global_edge_id_detached, result.global_edge_ids);
        result.is_boundary = select(valid_detached, boundary, result.is_boundary);
    }
    return result;
}

template NearestPointEdge Scene::nearest_edge<true>(const Vector3f &point, Mask active) const;
template NearestPointEdgeAD Scene::nearest_edge<false>(const Vector3fAD &point, MaskAD active) const;
template NearestRayEdge Scene::nearest_edge<true>(const Ray &ray, Mask active) const;
template NearestRayEdgeAD Scene::nearest_edge<false>(const RayAD &ray, MaskAD active) const;
template NearestEdgesTopK Scene::nearest_edges<true>(
    const Vector3f &point,
    int k,
    Mask active) const;
template NearestEdgesTopKAD Scene::nearest_edges<false>(
    const Vector3fAD &point,
    int k,
    MaskAD active) const;

} // namespace rayd
