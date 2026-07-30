// Copyright Xingyu Chen.
// Implements scene support for scene Dr.Jit.

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <rayd/jit/core.h>
#include <rayd/jit/mesh.h>

namespace rayd {

/// \brief Compute cached triangle geometry and area-weighted vertex normals for a mesh.
///
/// \return Pair of (per-triangle info in edge-vector form, per-vertex shading normals).
///         Degenerate (zero-area) triangles fall back to a +Z normal.
template <bool Detached>
static std::pair<TriangleInfoT<Detached>, Vector3fT<Detached>> process_mesh(const Vector3fT<Detached>& vertex_positions,
                                                                            const Vector3iT<Detached>& face_indices) {
    const int vertex_count = static_cast<int>(slices<Vector3fT<Detached>>(vertex_positions));

    TriangleInfoT<Detached> triangles;
    triangles.face_indices = face_indices;
    triangles.p0 = gather<Vector3fT<Detached>>(vertex_positions, face_indices[0]);
    triangles.e1 = gather<Vector3fT<Detached>>(vertex_positions, face_indices[1]) - triangles.p0;
    triangles.e2 = gather<Vector3fT<Detached>>(vertex_positions, face_indices[2]) - triangles.p0;

    Vector3fT<Detached> raw_face_normals = cross(triangles.e1, triangles.e2);
    FloatT<Detached> raw_face_areas = norm(raw_face_normals);
    MaskT<Detached> valid_faces = raw_face_areas > Epsilon;

    Vector3fT<Detached>& face_normals = triangles.face_normal;
    FloatT<Detached>& face_areas = triangles.face_area;
    face_normals = select(valid_faces, raw_face_normals / raw_face_areas, Vector3fT<Detached>(0.f, 0.f, 1.f));
    face_areas = raw_face_areas;

    Vector3fT<Detached> vertex_normals = zeros<Vector3fT<Detached>>(vertex_count);
    FloatT<Detached> vertex_weights = zeros<FloatT<Detached>>(vertex_count);
    for (int corner = 0; corner < 3; ++corner) {
        for (int axis = 0; axis < 3; ++axis) {
            scatter_reduce(ReduceOp::Add, vertex_normals[axis], raw_face_normals[axis], face_indices[corner]);
        }
        scatter_reduce(ReduceOp::Add, vertex_weights, face_areas, face_indices[corner]);
    }

    const FloatT<Detached> safe_vertex_weights = select(vertex_weights > Epsilon, vertex_weights, 1.f);
    vertex_normals /= safe_vertex_weights;
    vertex_normals = select(vertex_weights > Epsilon, normalize(vertex_normals), Vector3fT<Detached>(0.f, 0.f, 1.f));
    triangles.n0 = gather<Vector3fT<Detached>>(vertex_normals, face_indices[0]);
    triangles.n1 = gather<Vector3fT<Detached>>(vertex_normals, face_indices[1]);
    triangles.n2 = gather<Vector3fT<Detached>>(vertex_normals, face_indices[2]);

    face_areas *= 0.5f;

    drjit::eval(triangles, vertex_normals);
    return {triangles, vertex_normals};
}

/// Normalize \p value, returning +Z for near-zero-length inputs instead of NaN.
template <bool Detached> static Vector3fT<Detached> safe_normalize(const Vector3fT<Detached>& value) {
    const FloatT<Detached> length_sq = squared_norm(value);
    const MaskT<Detached> valid = length_sq > Epsilon * Epsilon;
    const FloatT<Detached> safe_length = sqrt(select(valid, length_sq, FloatT<Detached>(1.f)));
    return select(valid, value / safe_length, Vector3fT<Detached>(0.f, 0.f, 1.f));
}

/// Transform object-space triangle info to world space; normals use the inverse-transpose
/// of \p to_world_matrix, and face normals/areas are recomputed from the transformed edges.
static TriangleInfoAD transform_triangle_info(const TriangleInfoAD& triangle_info_object,
                                              const Matrix4fAD& to_world_matrix) {
    TriangleInfoAD triangles_world;
    triangles_world.face_indices = triangle_info_object.face_indices;
    triangles_world.p0 = transform_pos(to_world_matrix, triangle_info_object.p0);
    triangles_world.e1 = transform_dir(to_world_matrix, triangle_info_object.e1);
    triangles_world.e2 = transform_dir(to_world_matrix, triangle_info_object.e2);

    const Matrix4fAD normal_to_world = transpose(inverse(to_world_matrix));
    triangles_world.n0 = safe_normalize<false>(transform_dir(normal_to_world, triangle_info_object.n0));
    triangles_world.n1 = safe_normalize<false>(transform_dir(normal_to_world, triangle_info_object.n1));
    triangles_world.n2 = safe_normalize<false>(transform_dir(normal_to_world, triangle_info_object.n2));

    const Vector3fAD raw_face_normals = cross(triangles_world.e1, triangles_world.e2);
    const FloatAD raw_face_areas = norm(raw_face_normals);
    const MaskAD valid_faces = raw_face_areas > Epsilon;
    triangles_world.face_normal = select(valid_faces, raw_face_normals / raw_face_areas, Vector3fAD(0.f, 0.f, 1.f));
    triangles_world.face_area = raw_face_areas * 0.5f;
    return triangles_world;
}

Mesh::Mesh(const Mesh& other)
    : mesh_id_(other.mesh_id_), is_ready_(false), use_face_normals_(other.use_face_normals_), has_uv_(other.has_uv_),
      edges_enabled_(other.edges_enabled_), transform_identity_(other.transform_identity_),
      object_to_world_(other.object_to_world_), left_transform_(other.left_transform_),
      right_transform_(other.right_transform_), vertex_count_(other.vertex_count_), face_count_(other.face_count_),
      vertex_positions_object_(other.vertex_positions_object_), vertex_normals_object_(other.vertex_normals_object_),
      vertex_uv_(other.vertex_uv_), vertex_positions_world_(other.vertex_positions_world_),
      face_vertex_indices_(other.face_vertex_indices_), face_uv_indices_(other.face_uv_indices_),
      edge_indices_(other.edge_indices_), world_positions_dirty_(other.world_positions_dirty_),
      secondary_edge_info_dirty_(other.secondary_edge_info_dirty_),
      optix_face_buffer_dirty_(other.optix_face_buffer_dirty_), optix_vertex_buffer_(other.optix_vertex_buffer_),
      optix_face_buffer_(other.optix_face_buffer_) {
    if (other.triangle_info_object_) {
        triangle_info_object_ = std::make_unique<TriangleInfoAD>(*other.triangle_info_object_);
    }
    if (other.triangle_info_) {
        triangle_info_ = std::make_unique<TriangleInfoAD>(*other.triangle_info_);
    }
    if (other.triangle_uv_) {
        triangle_uv_ = std::make_unique<TriangleUVAD>(*other.triangle_uv_);
    }
    if (other.secondary_edge_info_) {
        secondary_edge_info_ = std::make_unique<SecondaryEdgeInfoAD>(*other.secondary_edge_info_);
    }
}

Mesh::Mesh(const Vector3f& vertex_positions, const Vector3i& face_indices, const Vector2f& vertex_uv,
           const Vector3i& face_uv_indices, bool verbose) {
    init(vertex_positions, face_indices, vertex_uv, face_uv_indices, verbose);
}

Mesh::~Mesh() = default;

const Vector3fAD& Mesh::vertex_positions_world() const {
    ensure_world_positions_ready();
    return vertex_positions_world_;
}

const SecondaryEdgeInfoAD* Mesh::secondary_edge_info() const {
    ensure_secondary_edge_info_ready();
    return secondary_edge_info_.get();
}

void Mesh::set_transform(const Matrix4fAD& matrix, bool set_left) {
    if (set_left) {
        left_transform_ = matrix;
    } else {
        right_transform_ = matrix;
    }
    transform_identity_ = false;
    world_positions_dirty_ = true;
    secondary_edge_info_dirty_ = true;
    is_ready_ = false;
}

void Mesh::append_transform(const Matrix4fAD& matrix, bool append_left) {
    if (append_left) {
        left_transform_ = matrix * left_transform_;
    } else {
        right_transform_ *= matrix;
    }
    transform_identity_ = false;
    world_positions_dirty_ = true;
    secondary_edge_info_dirty_ = true;
    is_ready_ = false;
}

void Mesh::init(const Vector3f& vertex_positions, const Vector3i& face_indices, const Vector2f& vertex_uv,
                const Vector3i& face_uv_indices, bool verbose) {
    using namespace std::chrono;

    vertex_count_ = static_cast<int>(slices(vertex_positions));
    face_count_ = static_cast<int>(slices(face_indices));
    vertex_positions_object_ = Vector3fAD(vertex_positions);
    face_vertex_indices_ = Vector3iAD(face_indices);
    has_uv_ = slices(vertex_uv) > 0;
    if (has_uv_) {
        vertex_uv_ = Vector2fAD(vertex_uv);
        const size_t face_uv_count = slices(face_uv_indices);
        if (face_uv_count == 0) {
            require(slices(vertex_uv) == static_cast<size_t>(vertex_count_),
                    "Mesh::init(): UV count must match vertex count when face_uv_indices are omitted.");
            face_uv_indices_ = Vector3iAD(face_indices);
        } else {
            require(face_uv_count == static_cast<size_t>(face_count_),
                    "Mesh::init(): face_uv_indices must match the number of faces.");
            face_uv_indices_ = Vector3iAD(face_uv_indices);
        }
    } else {
        vertex_uv_ = Vector2fAD();
        face_uv_indices_ = Vector3iAD();
    }

    std::array<std::vector<int>, 3> face_indices_cpu;
    copy_cuda_array(face_indices, face_indices_cpu);
    drjit::eval();
    drjit::sync_thread();

    const auto start_time = high_resolution_clock::now();
    int edge_count = 0;
    if (edges_enabled_) {
        std::array<std::vector<int>, 5> edge_records;
        for (auto& record : edge_records) {
            record.reserve(3 * face_count_);
        }

        // Per edge, record every incident face together with its own opposite
        // vertex (the corner of that face not on the edge). Storing the opposite
        // vertex per face (rather than once per edge) is what lets non-manifold
        // edges emit correct wedges below.
        std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> edge_map;
        for (int face_index = 0; face_index < face_count_; ++face_index) {
            for (int local_edge = 0; local_edge < 3; ++local_edge) {
                const int start_corner = local_edge;
                const int end_corner = (local_edge + 1) % 3;
                const int opposite_corner = (local_edge + 2) % 3;

                const int start_vertex = face_indices_cpu[start_corner][face_index];
                const int end_vertex = face_indices_cpu[end_corner][face_index];
                const int opposite_vertex = face_indices_cpu[opposite_corner][face_index];
                const auto edge_key = start_vertex < end_vertex ? std::make_pair(start_vertex, end_vertex)
                                                                : std::make_pair(end_vertex, start_vertex);

                edge_map[edge_key].emplace_back(face_index, opposite_vertex);
            }
        }

        for (const auto& [edge_vertices, faces] : edge_map) {
            if (faces.size() == 1) {
                // Boundary edge: a single incident face, no second wedge face.
                edge_records[0].push_back(edge_vertices.first);
                edge_records[1].push_back(edge_vertices.second);
                edge_records[2].push_back(faces[0].first);
                edge_records[3].push_back(-1);
                edge_records[4].push_back(faces[0].second);
                ++edge_count;
                continue;
            }
            // Emit one wedge per unordered pair of incident faces. A manifold
            // edge (2 faces) yields exactly the single (face0, face1) wedge,
            // identical to before. Non-manifold edges (3+ faces) emit every
            // face pair so no wedge or diffraction path is silently dropped;
            // each wedge carries the opposite vertex of its own face0.
            for (size_t i = 0; i < faces.size(); ++i) {
                for (size_t j = i + 1; j < faces.size(); ++j) {
                    edge_records[0].push_back(edge_vertices.first);
                    edge_records[1].push_back(edge_vertices.second);
                    edge_records[2].push_back(faces[i].first);
                    edge_records[3].push_back(faces[j].first);
                    edge_records[4].push_back(faces[i].second);
                    ++edge_count;
                }
            }
        }

        edge_indices_ = VectoriT<5, true>(drjit::load<Int>(edge_records[0].data(), edge_count),
                                          drjit::load<Int>(edge_records[1].data(), edge_count),
                                          drjit::load<Int>(edge_records[2].data(), edge_count),
                                          drjit::load<Int>(edge_records[3].data(), edge_count),
                                          drjit::load<Int>(edge_records[4].data(), edge_count));
    } else {
        edge_indices_ = VectoriT<5, true>(Int(), Int(), Int(), Int(), Int());
    }

    const auto end_time = high_resolution_clock::now();
    const double seconds = duration_cast<duration<double>>(end_time - start_time).count();
    if (verbose) {
        std::cout << "Loaded " << vertex_count_ << " vertices, " << face_count_ << " faces, " << edge_count
                  << " edges in " << seconds << " seconds." << std::endl;
    }

    optix_face_buffer_dirty_ = true;
    is_ready_ = false;
}

void Mesh::build() {
    require(vertex_count_ > 0, "Mesh::build(): mesh has no vertices.");
    require(face_count_ > 0, "Mesh::build(): mesh has no faces.");

    if (!triangle_info_object_) {
        triangle_info_object_ = std::make_unique<TriangleInfoAD>();
    }
    std::tie(*triangle_info_object_, vertex_normals_object_) =
        process_mesh<false>(vertex_positions_object_, face_vertex_indices_);

    if (!triangle_info_) {
        triangle_info_ = std::make_unique<TriangleInfoAD>();
    }
    update_world_triangle_info();

    triangle_uv_.reset();
    if (has_uv_) {
        triangle_uv_ = std::make_unique<TriangleUVAD>();
        for (int corner = 0; corner < 3; ++corner) {
            (*triangle_uv_)[corner] = gather<Vector2fAD>(vertex_uv_, face_uv_indices_[corner]);
        }
    }

    world_positions_dirty_ = true;
    secondary_edge_info_dirty_ = true;
    ensure_secondary_edge_info_ready();

    is_ready_ = true;
    prepare_optix_buffers();
    drjit::eval();
    drjit::sync_thread();
}

void Mesh::update_runtime_data(bool vertices_dirty, bool transform_dirty) {
    require(vertices_dirty || transform_dirty,
            "Mesh::update_runtime_data(): expected either vertices or transform to be dirty.");
    require(vertex_count_ > 0, "Mesh::update_runtime_data(): mesh has no vertices.");
    require(face_count_ > 0, "Mesh::update_runtime_data(): mesh has no faces.");
    require(triangle_info_ != nullptr,
            "Mesh::update_runtime_data(): mesh must be built before applying incremental updates.");

    if (vertices_dirty) {
        require(triangle_info_object_ != nullptr,
                "Mesh::update_runtime_data(): object-space geometry must be built first.");
        std::tie(*triangle_info_object_, vertex_normals_object_) =
            process_mesh<false>(vertex_positions_object_, face_vertex_indices_);
    }

    update_world_triangle_info();
    world_positions_dirty_ = true;
    secondary_edge_info_dirty_ = true;

    is_ready_ = true;

    if (vertices_dirty) {
        update_optix_vertex_buffer();
        // The packed vertex buffer and the following OptiX update are issued
        // on the same Dr.Jit CUDA stream. Evaluation establishes that
        // stream-local dependency; a host-side sync here serialized geometry
        // derivation before every dynamic GAS refit.
        drjit::eval(optix_vertex_buffer_);
    }
}

void Mesh::update_world_triangle_info() {
    require(triangle_info_object_ != nullptr,
            "Mesh::update_world_triangle_info(): object-space geometry must be built first.");
    require(triangle_info_ != nullptr,
            "Mesh::update_world_triangle_info(): world-space triangle cache must be allocated first.");

    if (transform_identity_) {
        *triangle_info_ = *triangle_info_object_;
    } else {
        const Matrix4fAD to_world_matrix = left_transform_ * object_to_world_ * right_transform_;
        *triangle_info_ = transform_triangle_info(*triangle_info_object_, to_world_matrix);
    }
}

void Mesh::update_secondary_edge_info() {
    ensure_world_positions_ready();

    if (!edges_enabled_) {
        secondary_edge_info_.reset();
        secondary_edge_info_dirty_ = false;
        return;
    }

    if (!secondary_edge_info_) {
        secondary_edge_info_ = std::make_unique<SecondaryEdgeInfoAD>();
    }

    SecondaryEdgeInfoAD secondary_edges;
    const IntAD edge_vertex_0 = IntAD(edge_indices_[0]);
    const IntAD edge_vertex_1 = IntAD(edge_indices_[1]);
    const IntAD face_index_0 = IntAD(edge_indices_[2]);
    const IntAD face_index_1 = IntAD(edge_indices_[3]);
    const IntAD opposite_vertex = IntAD(edge_indices_[4]);
    const MaskAD is_boundary = face_index_1 < 0;

    secondary_edges.is_boundary = is_boundary;
    secondary_edges.start = gather<Vector3fAD>(vertex_positions_world_, edge_vertex_0);
    secondary_edges.edge = gather<Vector3fAD>(vertex_positions_world_, edge_vertex_1) - secondary_edges.start;
    secondary_edges.normal0 = gather<Vector3fAD>(triangle_info_->face_normal, face_index_0);
    secondary_edges.normal1 = gather<Vector3fAD>(triangle_info_->face_normal, face_index_1, ~is_boundary);
    secondary_edges.opposite = gather<Vector3fAD>(vertex_positions_world_, opposite_vertex);
    *secondary_edge_info_ = secondary_edges;
    secondary_edge_info_dirty_ = false;
}

void Mesh::ensure_world_positions_ready() const {
    if (!world_positions_dirty_) {
        return;
    }

    if (transform_identity_) {
        vertex_positions_world_ = vertex_positions_object_;
    } else {
        const Matrix4fAD to_world_matrix = left_transform_ * object_to_world_ * right_transform_;
        vertex_positions_world_ = transform_pos(to_world_matrix, vertex_positions_object_);
    }
    world_positions_dirty_ = false;
}

void Mesh::ensure_secondary_edge_info_ready() const {
    if (!secondary_edge_info_dirty_) {
        return;
    }

    const_cast<Mesh*>(this)->update_secondary_edge_info();
}

void Mesh::prepare_optix_buffers() {
    require(is_ready_, "Mesh::prepare_optix_buffers(): mesh must be built first.");

    update_optix_vertex_buffer();
    ensure_optix_face_buffer_ready();
}

void Mesh::update_optix_vertex_buffer() {
    const int scalar_count = vertex_count_ * 3;
    if (slices(optix_vertex_buffer_) != static_cast<size_t>(scalar_count)) {
        optix_vertex_buffer_ = empty<Float>(scalar_count);
    }

    const Int indices = arange<Int>(vertex_count_) * 3;
    for (int axis = 0; axis < 3; ++axis) {
        scatter(optix_vertex_buffer_, detach<false>(vertex_positions_object_[axis]), indices + axis);
    }
}

void Mesh::ensure_optix_face_buffer_ready() {
    const int scalar_count = face_count_ * 3;
    if (!optix_face_buffer_dirty_ && slices(optix_face_buffer_) == static_cast<size_t>(scalar_count)) {
        return;
    }

    if (slices(optix_face_buffer_) != static_cast<size_t>(scalar_count)) {
        optix_face_buffer_ = empty<Int>(scalar_count);
    }

    const Int indices = arange<Int>(face_count_) * 3;
    for (int axis = 0; axis < 3; ++axis) {
        scatter(optix_face_buffer_, detach<false>(face_vertex_indices_[axis]), indices + axis);
    }
    optix_face_buffer_dirty_ = false;
}

std::string Mesh::to_string() const {
    std::stringstream stream;
    stream << "Mesh[nv=" << vertex_count_ << ", nf=" << face_count_ << ", has_uv=" << has_uv_
           << ", edges=" << edges_enabled_ << "]";
    return stream.str();
}

} // namespace rayd

// Consolidated scene container implementation.
#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include <rayd/jit/core.h>
#include <rayd/jit/scene.h>
#include <rayd/jit/edge.h>
#include <rayd/jit/native_launch_audit.h>

namespace rayd {

namespace {
std::string normalize_edge_backend_value(const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) -> char { return static_cast<char>(std::tolower(ch)); });
    return normalized;
}

/// Resolve an edge backend name to a concrete EdgeBVHBackend.
EdgeBVHBackend parse_edge_backend(const std::string& value) {
    const std::string normalized = normalize_edge_backend_value(value);
    if (normalized.empty() || normalized == "auto") {
        return optix_available() ? EdgeBVHBackend::Optix : EdgeBVHBackend::DrJit;
    }
    if (normalized == "optix" || normalized == "custom_aabb") {
        return EdgeBVHBackend::Optix;
    }
    if (normalized == "drjit" || normalized == "dr_jit" || normalized == "software") {
        return EdgeBVHBackend::DrJit;
    }
    if (normalized == "optix_drjit" || normalized == "hybrid" || normalized == "mixed" || normalized == "optix_ray" ||
        normalized == "ray_optix") {
        return EdgeBVHBackend::OptixDrJit;
    }
    throw std::runtime_error("Invalid edge_bvh_backend. Expected one of: 'auto', 'drjit', 'optix', "
                             "'optix_drjit'.");
}

const char* edge_backend_name(EdgeBVHBackend backend) {
    switch (backend) {
    case EdgeBVHBackend::DrJit:
        return "drjit";
    case EdgeBVHBackend::Optix:
        return "optix";
    case EdgeBVHBackend::OptixDrJit:
        return "optix_drjit";
    }
    return "drjit";
}

bool edge_backend_builds_drjit(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::DrJit || backend == EdgeBVHBackend::OptixDrJit;
}

bool edge_backend_builds_optix(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix || backend == EdgeBVHBackend::OptixDrJit;
}

bool edge_backend_uses_optix_point(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

bool edge_backend_uses_optix_ray(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix || backend == EdgeBVHBackend::OptixDrJit;
}

bool edge_backend_uses_optix_topk(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

/// Parse and resolve the trace_backend selector to a concrete kind. "auto"
/// resolves to OptiX when the current CUDA device/context accepts an OptiX
/// context and to CUDA otherwise;
/// explicit OptiX availability is enforced later, at build().
TraceBackendKind resolve_trace_backend_kind(const std::string& value) {
    const std::string normalized = normalize_edge_backend_value(value);
    if (normalized.empty() || normalized == "auto") {
        return optix_available() ? TraceBackendKind::Optix : TraceBackendKind::Cuda;
    }
    if (normalized == "optix") {
        return TraceBackendKind::Optix;
    }
    if (normalized == "none") {
        return TraceBackendKind::None;
    }
    if (normalized == "cuda") {
        return TraceBackendKind::Cuda;
    }
    if (normalized == "embree") {
        throw std::runtime_error("trace_backend 'embree' is not implemented yet "
                                 "(planned: Embree in P5)");
    }
    throw std::runtime_error("Invalid trace_backend. Expected one of: 'auto', 'optix', 'cuda', 'none'.");
}

template <bool Detached> NearestPointEdgeT<Detached> initialize_nearest_point_edge_result(int query_count) {
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

template <bool Detached> NearestRayEdgeT<Detached> initialize_nearest_ray_edge_result(int query_count) {
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

int face_edge_slot(const std::array<int, 3>& face_vertices, int v0, int v1) {
    auto matches = [v0, v1](int a, int b) { return (a == v0 && b == v1) || (a == v1 && b == v0); };

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

int face_opposite_vertex(const std::array<int, 3>& face_vertices, int v0, int v1) {
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
    diffraction_order1_accumulation_primary_pipeline_.reset();
    diffraction_order1_accumulation_no_suffix_pipeline_.reset();
    diffraction_order1_accumulation_no_suffix_primary_pipeline_.reset();
    diffraction_order1_accumulation_suffix_pipeline_.reset();
    diffraction_order1_accumulation_suffix_primary_pipeline_.reset();
    diffraction_order1_source_visibility_primary_pipeline_.reset();
    diffraction_order1_no_suffix_target_primary_pipeline_.reset();
    diffraction_order1_suffix_first_visibility_primary_pipeline_.reset();
    diffraction_order1_suffix_target_primary_pipeline_.reset();
    diffraction_chain_accumulation_pipeline_.reset();
    diffraction_chain_accumulation_primary_pipeline_.reset();
    diffraction_coherent_accumulation_pipeline_.reset();
    diffraction_coherent_accumulation_primary_pipeline_.reset();
    diffraction_paths_primary_pipeline_.reset();
    diffraction_paths_source_visibility_primary_pipeline_.reset();
    diffraction_paths_target_export_primary_pipeline_.reset();
    diffraction_paths_pipeline_.reset();
    reflection_epc_pipeline_.reset();
    reflection_epc_direct_pipeline_.reset();
    reflection_epc_direct_primary_pipeline_.reset();
    reflection_epc_geometry_ready_ = false;
    segment_visibility_pipeline_.reset();
    segment_pair_visibility_pipeline_.reset();
    axial_edge_visibility_pipeline_.reset();
    segment_chain_visibility_pipeline_.reset();
}

Scene::Scene(const std::string& edge_bvh_backend, const std::string& trace_backend)
    : triangle_kind_(resolve_trace_backend_kind(trace_backend)), edge_bvh_(std::make_unique<SceneEdge>()),
      edge_optix_(std::make_unique<SceneEdgeOptix>()), edge_bvh_backend_(parse_edge_backend(edge_bvh_backend)) {}

Scene::~Scene() = default;

void Scene::require_build_device(const char* context) const {
    if (build_device_ < 0) {
        return;
    }

    const int current_device = jit_cuda_device();
    if (current_device == build_device_) {
        return;
    }

    throw std::runtime_error(
        std::string(context) + " requires the Dr.Jit CUDA device the scene was built on (device " +
        std::to_string(build_device_) + "), but the current Dr.Jit CUDA device is " + std::to_string(current_device) +
        ". Scene buffers, acceleration structures, and OptiX resources are bound "
        "to their build device; call rayd.drjit.set_device(" +
        std::to_string(build_device_) + ") before querying, or rebuild the scene on the current device.");
}

std::string Scene::to_string() const {
    std::stringstream stream;
    stream << "Scene[num_meshes=" << mesh_count_ << ", ready=" << is_ready() << ", pending_updates=" << pending_updates_
           << "]";
    return stream.str();
}

std::vector<const Mesh*> Scene::meshes() const {
    std::vector<const Mesh*> result;
    result.reserve(mesh_records_.size());
    for (const SceneMeshRecord& record : mesh_records_) {
        result.push_back(record.mesh.get());
    }
    return result;
}

int Scene::add_mesh(const Mesh& mesh, bool dynamic) {
    SceneMeshRecord record;
    record.mesh = std::make_unique<Mesh>(mesh);
    record.mesh->set_mesh_id(static_cast<int>(mesh_records_.size()));
    record.dynamic = dynamic;
    record.geometry_owner_id = static_cast<int>(mesh_records_.size());
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
    trace_backend_.reset();
    reset_multipath_pipelines();
    return mesh_count_ - 1;
}

int Scene::add_instance(int geometry_id, const Matrix4fAD& transform, bool dynamic) {
    const SceneMeshRecord& source = mesh_record(geometry_id);
    const SceneMeshRecord& owner = mesh_record(source.geometry_owner_id);
    require(!owner.dynamic, "Scene::add_instance(): dynamic source geometry cannot be instanced.");

    SceneMeshRecord record;
    record.mesh = std::make_unique<Mesh>(*source.mesh);
    record.mesh->append_transform(transform, true);
    record.mesh->set_mesh_id(static_cast<int>(mesh_records_.size()));
    record.dynamic = dynamic;
    record.geometry_owner_id = source.geometry_owner_id;
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
    trace_backend_.reset();
    reset_multipath_pipelines();
    return mesh_count_ - 1;
}

Scene::SceneMeshRecord& Scene::mesh_record(int mesh_id) {
    require(mesh_id >= 0 && mesh_id < static_cast<int>(mesh_records_.size()), "Scene: mesh_id is out of range.");
    return mesh_records_[static_cast<size_t>(mesh_id)];
}

const Scene::SceneMeshRecord& Scene::mesh_record(int mesh_id) const {
    require(mesh_id >= 0 && mesh_id < static_cast<int>(mesh_records_.size()), "Scene: mesh_id is out of range.");
    return mesh_records_[static_cast<size_t>(mesh_id)];
}

void Scene::scatter_mesh_data(const SceneMeshRecord& record, bool include_static) {
    const Mesh& mesh = *record.mesh;
    const int mesh_face_count = mesh.face_count();
    if (mesh_face_count == 0) {
        return;
    }

    const TriangleInfoAD* mesh_triangle_info = mesh.triangle_info();
    const IntAD scatter_indices = arange<IntAD>(mesh_face_count) + record.face_offset;
    const Int scatter_indices_detached = arange<Int>(mesh_face_count) + record.face_offset;

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
    scatter(triangle_info_detached_.face_normal, detach<false>(mesh_triangle_info->face_normal),
            scatter_indices_detached);
    scatter(triangle_info_detached_.face_area, detach<false>(mesh_triangle_info->face_area), scatter_indices_detached);

    if (!include_static) {
        return;
    }

    scatter(triangle_info_.face_indices, mesh_triangle_info->face_indices, scatter_indices);
    scatter(triangle_info_detached_.face_indices, detach<false>(mesh_triangle_info->face_indices),
            scatter_indices_detached);
    scatter(triangle_face_normal_mask_, full<MaskAD>(mesh.use_face_normals(), mesh_face_count), scatter_indices);
    scatter(triangle_face_normal_mask_detached_, full<Mask>(mesh.use_face_normals(), mesh_face_count),
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

void Scene::scatter_mesh_edge_data(const SceneMeshRecord& record, bool include_static_ids) {
    const Mesh& mesh = *record.mesh;
    const SecondaryEdgeInfoAD* mesh_edge_info = mesh.secondary_edge_info();
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
    scatter(edge_shape_ids_, full<Int>(mesh.mesh_id(), mesh_edge_count), scatter_indices_detached);
    scatter(edge_local_ids_, arange<Int>(mesh_edge_count), scatter_indices_detached);
}

void Scene::ensure_scene_edge_data_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    for (const SceneMeshRecord& record : mesh_records_) {
        if (!record.edge_dirty) {
            continue;
        }

        const_cast<Scene*>(this)->scatter_mesh_edge_data(record, false);
        record.edge_dirty = false;
    }

    ensure_edge_bvh_ready();
}

void Scene::ensure_edge_bvh_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    Scene* scene = const_cast<Scene*>(this);
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

    drjit::eval(triangle_info_detached_.p0, triangle_info_detached_.e1, triangle_info_detached_.e2,
                triangle_info_detached_.face_normal, face_offsets_);
    reflection_epc_geometry_ready_ = true;
}

void Scene::build() {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::Build);
    require(!mesh_records_.empty(), "Scene::build(): missing meshes.");

    // Validate the resolved backend plan before building any acceleration
    // structure so an OptiX-less machine fails fast with a clear message rather
    // than deep inside an OptiX call.
    if (triangle_kind_ == TraceBackendKind::Optix && !optix_available()) {
        throw std::runtime_error("trace_backend 'optix' requested but OptiX is unavailable for the "
                                 "current CUDA device/context");
    }
    if (edge_backend_builds_optix(edge_bvh_backend_) && !optix_available()) {
        throw std::runtime_error("edge_bvh_backend='" + std::string(edge_backend_name(edge_bvh_backend_)) +
                                 "' requires OptiX, which is unavailable for the current CUDA "
                                 "device/context; use edge_bvh_backend=\"drjit\" for a software edge "
                                 "backend");
    }

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
        SceneMeshRecord& record = mesh_records_[mesh_index];
        Mesh& mesh = *record.mesh;
        mesh.set_mesh_id(static_cast<int>(mesh_index));
        mesh.build();
        record.vertex_offset = vertex_offsets.back();
        record.face_offset = face_offsets.back();
        const SecondaryEdgeInfoAD* mesh_edge_info = mesh.secondary_edge_info();
        const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
        record.edge_offset = edge_offsets.back();
        record.vertices_dirty = false;
        record.transform_dirty = false;
        record.edge_dirty = false;

        vertex_offsets.push_back(vertex_offsets.back() + mesh.vertex_count());
        face_offsets.push_back(face_offsets.back() + mesh.face_count());
        edge_offsets.push_back(edge_offsets.back() + mesh_edge_count);
        const SceneMeshRecord& owner = mesh_records_[static_cast<size_t>(record.geometry_owner_id)];
        mesh_descs.push_back({&mesh, record.dynamic, owner.dynamic, record.face_offset, static_cast<int>(mesh_index),
                              record.geometry_owner_id});
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
    for (auto& global_face_indices : global_face_indices_cpu) {
        global_face_indices.reserve(total_face_count);
    }
    std::vector<int> global_shape_ids_cpu;
    std::vector<int> global_local_prim_ids_cpu;
    std::vector<int> global_prim_ids_cpu;
    global_shape_ids_cpu.reserve(total_face_count);
    global_local_prim_ids_cpu.reserve(total_face_count);
    global_prim_ids_cpu.reserve(total_face_count);

    std::array<std::vector<int>, 3> triangle_edge_ids_cpu;
    for (auto& triangle_edge_ids : triangle_edge_ids_cpu) {
        triangle_edge_ids.assign(total_face_count, -1);
    }

    for (const SceneMeshRecord& record : mesh_records_) {
        const Mesh& mesh = *record.mesh;
        const auto& mesh_edge_indices = mesh.edge_indices();
        const int mesh_edge_count = mesh.edges_enabled() ? static_cast<int>(slices(mesh_edge_indices)) : 0;
        const Vector3i mesh_face_indices(detach<false>(mesh.face_indices()[0]), detach<false>(mesh.face_indices()[1]),
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

            const std::array<int, 3> face0_vertices{mesh_face_cpu[0][face0_local], mesh_face_cpu[1][face0_local],
                                                    mesh_face_cpu[2][face0_local]};

            int opposite1 = -1;
            if (face1_local >= 0) {
                const std::array<int, 3> face1_vertices{mesh_face_cpu[0][face1_local], mesh_face_cpu[1][face1_local],
                                                        mesh_face_cpu[2][face1_local]};
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

    auto load_or_empty = [](const std::vector<int>& values) {
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
    global_geometry_.faces = Vector3i(load<Int>(global_face_indices_cpu[0].data(), total_face_count),
                                      load<Int>(global_face_indices_cpu[1].data(), total_face_count),
                                      load<Int>(global_face_indices_cpu[2].data(), total_face_count));
    global_geometry_.shape_id = load<Int>(global_shape_ids_cpu.data(), total_face_count);
    global_geometry_.local_prim_id = load<Int>(global_local_prim_ids_cpu.data(), total_face_count);
    global_geometry_.global_prim_id = load<Int>(global_prim_ids_cpu.data(), total_face_count);
    triangle_edge_ids_ = VectoriT<3, true>(load<Int>(triangle_edge_ids_cpu[0].data(), total_face_count),
                                           load<Int>(triangle_edge_ids_cpu[1].data(), total_face_count),
                                           load<Int>(triangle_edge_ids_cpu[2].data(), total_face_count));
    if (edge_count_ > 0) {
        edge_info_ = empty<SecondaryEdgeInfoAD>(edge_count_);
        edge_topology_ = SceneEdgeTopology{load_or_empty(topology_v0),
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
                                           load_or_empty(topology_opposite1_global)};
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

    for (const SceneMeshRecord& record : mesh_records_) {
        scatter_mesh_data(record, true);
        scatter_mesh_edge_data(record, true);
        const Mesh& mesh = *record.mesh;
        const int mesh_vertex_count = mesh.vertex_count();
        if (mesh_vertex_count > 0) {
            const IntAD vertex_scatter_indices = arange<IntAD>(mesh_vertex_count) + record.vertex_offset;
            scatter(global_geometry_.vertices, mesh.vertex_positions_world(), vertex_scatter_indices);
        }
    }
    differentiable_geometry_active_ = grad_enabled(triangle_info_, global_geometry_.vertices, edge_info_);
    global_geometry_.face_normal = triangle_info_.face_normal;

    int dynamic_mesh_count = 0;
    for (const SceneMeshRecord& record : mesh_records_) {
        if (record.dynamic) {
            ++dynamic_mesh_count;
        }
    }

    reset_multipath_pipelines();

    // The unconditional OptiX GAS build is gone: a triangle trace backend is
    // constructed only when the resolved plan asks for one. Edge-only scenes
    // (trace_backend='none') leave trace_backend_ null.
    if (triangle_kind_ == TraceBackendKind::Optix) {
        std::vector<bool> dynamic_flags;
        dynamic_flags.reserve(mesh_records_.size());
        for (const SceneMeshRecord& record : mesh_records_) {
            dynamic_flags.push_back(record.dynamic);
        }
        auto optix_backend = std::make_unique<OptixTraceBackend>();
        optix_backend->build(mesh_descs, dynamic_flags);
        trace_backend_ = std::move(optix_backend);
    } else if (triangle_kind_ == TraceBackendKind::Cuda) {
        // Pure-CUDA scene-level triangle BVH over the world-space detached
        // triangle arrays scattered above (no OptiX driver required).
        auto cuda_backend = std::make_unique<CudaTraceBackend>();
        cuda_backend->build(triangle_info_detached_, global_geometry_.shape_id, global_geometry_.local_prim_id);
        trace_backend_ = std::move(cuda_backend);
    } else {
        trace_backend_.reset();
    }
    mask_dirty_ = false;
    edge_bvh_ = std::make_unique<SceneEdge>();
    edge_optix_ = std::make_unique<SceneEdgeOptix>();
    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        edge_bvh_->build(edge_info_, edge_mask_, dynamic_mesh_count > 0);
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
    build_device_ = jit_cuda_device();
    ++scene_version_;
    ++edge_version_;
}

void Scene::update_mesh_vertices(int mesh_id, const Vector3fAD& positions) {
    require(is_ready(), "Scene::update_mesh_vertices(): scene is not built.");
    require_build_device("Scene::update_mesh_vertices()");

    SceneMeshRecord& record = mesh_record(mesh_id);
    require(record.geometry_owner_id == mesh_id,
            "Scene::update_mesh_vertices(): instance geometry is shared; update its source mesh instead.");
    require(record.dynamic, "Scene::update_mesh_vertices(): target mesh is not dynamic.");
    require(static_cast<int>(slices(positions)) == record.mesh->vertex_count(),
            "Scene::update_mesh_vertices(): vertex count must remain unchanged.");

    record.mesh->set_vertex_positions(positions);
    differentiable_geometry_active_ = differentiable_geometry_active_ || grad_enabled(positions);
    record.vertices_dirty = true;
    pending_updates_ = true;
}

void Scene::set_mesh_transform(int mesh_id, const Matrix4fAD& matrix, bool set_left) {
    require(is_ready(), "Scene::set_mesh_transform(): scene is not built.");
    require_build_device("Scene::set_mesh_transform()");

    SceneMeshRecord& record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::set_mesh_transform(): target mesh is not dynamic.");

    record.mesh->set_transform(matrix, set_left);
    differentiable_geometry_active_ = differentiable_geometry_active_ || grad_enabled(matrix);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::append_mesh_transform(int mesh_id, const Matrix4fAD& matrix, bool append_left) {
    require(is_ready(), "Scene::append_mesh_transform(): scene is not built.");
    require_build_device("Scene::append_mesh_transform()");

    SceneMeshRecord& record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::append_mesh_transform(): target mesh is not dynamic.");

    record.mesh->append_transform(matrix, append_left);
    differentiable_geometry_active_ = differentiable_geometry_active_ || grad_enabled(matrix);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::set_edge_mask(const Mask& mask) {
    require(is_ready(), "Scene::set_edge_mask(): scene is not built.");
    require(static_cast<int>(mask.size()) == edge_count_,
            "Scene::set_edge_mask(): mask size must match the scene edge count.");
    require_build_device("Scene::set_edge_mask()");

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
    require_build_device("Scene::sync()");
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
        SceneMeshRecord& record = mesh_records_[mesh_index];
        const SceneMeshRecord& owner = mesh_records_[static_cast<size_t>(record.geometry_owner_id)];
        mesh_descs.push_back({record.mesh.get(), record.dynamic, owner.dynamic, record.face_offset,
                              static_cast<int>(mesh_index), record.geometry_owner_id});

        if (!record.vertices_dirty && !record.transform_dirty) {
            continue;
        }

        const auto mesh_update_start = Clock::now();
        record.mesh->update_runtime_data(record.vertices_dirty, record.transform_dirty);
        last_sync_profile_.mesh_update_ms +=
            std::chrono::duration<double, std::milli>(Clock::now() - mesh_update_start).count();

        const int mesh_edge_count =
            record.mesh->edges_enabled() ? static_cast<int>(slices(record.mesh->edge_indices())) : 0;
        if (mesh_edge_count > 0 && !record.edge_dirty) {
            pending_edge_bvh_dirty_ranges_.push_back({record.edge_offset, mesh_edge_count});
            record.edge_dirty = true;
            edge_bvh_dirty_ = true;
            ++last_sync_profile_.updated_edge_meshes;
            last_sync_profile_.updated_edges += mesh_edge_count;
        }

        updates.push_back({static_cast<int>(mesh_index), record.vertices_dirty, record.transform_dirty});
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
        const auto scatter_start = Clock::now();

        if (differentiable_geometry_active_) {
            // A Dr.Jit scatter output depends on its prior target. Reusing the
            // scene-global AD targets here would retain the complete graph of
            // every preceding dynamic update. Rebuild the aggregate targets
            // from the current per-mesh caches instead. Static topology and UV
            // fields remain unchanged.
            const int face_count = static_cast<int>(slices(triangle_info_.face_area));
            const int vertex_count = global_geometry_.vertex_count();
            const Vector3iAD face_indices = triangle_info_.face_indices;
            const Vector3i face_indices_detached = triangle_info_detached_.face_indices;

            triangle_info_ = empty<TriangleInfoAD>(face_count);
            triangle_info_.face_indices = face_indices;
            triangle_info_detached_ = empty<TriangleInfo>(face_count);
            triangle_info_detached_.face_indices = face_indices_detached;
            global_geometry_.vertices = vertex_count > 0 ? empty<Vector3fAD>(vertex_count) : Vector3fAD();

            for (const SceneMeshRecord& record : mesh_records_) {
                scatter_mesh_data(record, false);
                const int mesh_vertex_count = record.mesh->vertex_count();
                if (mesh_vertex_count > 0) {
                    const IntAD vertex_scatter_indices = arange<IntAD>(mesh_vertex_count) + record.vertex_offset;
                    scatter(global_geometry_.vertices, record.mesh->vertex_positions_world(), vertex_scatter_indices);
                }
            }
        } else {
            for (const OptixSceneMeshUpdate& update : updates) {
                const SceneMeshRecord& record = mesh_records_[static_cast<size_t>(update.mesh_id)];
                scatter_mesh_data(record, false);
                const int mesh_vertex_count = record.mesh->vertex_count();
                if (mesh_vertex_count > 0) {
                    const IntAD vertex_scatter_indices = arange<IntAD>(mesh_vertex_count) + record.vertex_offset;
                    scatter(global_geometry_.vertices, record.mesh->vertex_positions_world(), vertex_scatter_indices);
                }
            }
        }

        last_sync_profile_.triangle_scatter_ms +=
            std::chrono::duration<double, std::milli>(Clock::now() - scatter_start).count();
        global_geometry_.face_normal = triangle_info_.face_normal;
    }

    if (edge_bvh_dirty_) {
        const auto edge_scatter_start = Clock::now();
        if (differentiable_geometry_active_ && !updates.empty()) {
            edge_info_ = edge_count_ > 0 ? empty<SecondaryEdgeInfoAD>(edge_count_) : SecondaryEdgeInfoAD();
            for (SceneMeshRecord& record : mesh_records_) {
                scatter_mesh_edge_data(record, false);
                record.edge_dirty = false;
            }
        } else {
            for (SceneMeshRecord& record : mesh_records_) {
                if (!record.edge_dirty) {
                    continue;
                }

                scatter_mesh_edge_data(record, false);
                record.edge_dirty = false;
            }
        }
        last_sync_profile_.edge_scatter_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - edge_scatter_start).count();

        const auto edge_refit_start = Clock::now();
        ensure_edge_bvh_ready();
        last_sync_profile_.edge_refit_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - edge_refit_start).count();
    }

    // The OptiX GAS/IAS refit runs only when a triangle trace backend exists;
    // an edge-only scene (trace_backend='none') syncs its edge BVH above and
    // skips this block entirely.
    if (trace_backend_ != nullptr) {
        const auto trace_sync_start = Clock::now();
        if (triangle_kind_ == TraceBackendKind::Cuda) {
            // A pure-CUDA backend refits its scene-level BVH from the triangle
            // arrays re-scattered above; there is no OptiX GAS/IAS timing.
            if (!updates.empty()) {
                cuda_backend().sync(triangle_info_detached_, global_geometry_.shape_id, global_geometry_.local_prim_id);
            }
        } else {
            const OptixTraceSyncResult optix_result = optix_backend().sync(mesh_descs, updates);
            last_sync_profile_.optix_gas_update_ms = optix_result.gas_update_ms;
            last_sync_profile_.optix_ias_update_ms = optix_result.ias_update_ms;
        }
        last_sync_profile_.optix_sync_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - trace_sync_start).count();
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
    last_sync_profile_.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
}

SceneEdgeInfo Scene::edge_info() const {
    require(is_ready(), "Scene::edge_info(): scene is not built.");
    require(!pending_updates_, "Scene::edge_info(): scene has pending updates. Call Scene::sync() first.");
    require_build_device("Scene::edge_info()");

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
    require(!pending_updates_, "Scene::edge_bvh_stats(): scene has pending updates. Call Scene::sync() first.");
    require_build_device("Scene::edge_bvh_stats()");
    ensure_edge_bvh_ready();
    return edge_bvh_backend_ == EdgeBVHBackend::Optix ? edge_optix_->stats() : edge_bvh_->stats();
}

const SceneEdgeTopology& Scene::edge_topology() const {
    require(is_ready(), "Scene::edge_topology(): scene is not built.");
    require_build_device("Scene::edge_topology()");
    return edge_topology_;
}

const Mask& Scene::edge_mask() const {
    require(is_ready(), "Scene::edge_mask(): scene is not built.");
    require_build_device("Scene::edge_mask()");
    return edge_mask_;
}

const SceneGeometry& Scene::global_geometry() const {
    require(is_ready(), "Scene::global_geometry(): scene is not built.");
    require(!pending_updates_, "Scene::global_geometry(): scene has pending updates. Call Scene::sync() first.");
    require_build_device("Scene::global_geometry()");
    return global_geometry_;
}

VectoriT<3, true> Scene::triangle_edge_indices(const Int& prim_id, bool global) const {
    require(is_ready(), "Scene::triangle_edge_indices(): scene is not built.");
    require_build_device("Scene::triangle_edge_indices()");

    const int query_count = static_cast<int>(slices(prim_id));
    VectoriT<3, true> result(full<Int>(-1, query_count), full<Int>(-1, query_count), full<Int>(-1, query_count));
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

VectoriT<2, true> Scene::edge_adjacent_faces(const Int& edge_id, bool global) const {
    require(is_ready(), "Scene::edge_adjacent_faces(): scene is not built.");
    require_build_device("Scene::edge_adjacent_faces()");

    const int query_count = static_cast<int>(slices(edge_id));
    VectoriT<2, true> result(full<Int>(-1, query_count), full<Int>(-1, query_count));
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    const Mask valid = edge_id >= 0 && edge_id < edge_count_;
    const Int face0 = global ? gather<Int>(edge_topology_.face0_global, edge_id, valid)
                             : gather<Int>(edge_topology_.face0_local, edge_id, valid);
    const Int face1 = global ? gather<Int>(edge_topology_.face1_global, edge_id, valid)
                             : gather<Int>(edge_topology_.face1_local, edge_id, valid);
    result[0] = select(valid, face0, result[0]);
    result[1] = select(valid, face1, result[1]);
    return result;
}

bool Scene::is_ready() const {
    // With a triangle trace backend, defer to it; a built edge-only scene
    // (no trace backend) reports ready so its edge queries can run.
    const bool trace_ready = trace_backend_ != nullptr ? trace_backend_->is_ready() : true;
    bool edge_ready = true;
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        edge_ready &= edge_optix_ != nullptr && edge_optix_->is_ready();
    }
    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        edge_ready &= edge_bvh_ != nullptr && edge_bvh_->is_ready();
    }
    return is_ready_ && edge_ready && trace_ready;
}

OptixTraceBackend& Scene::optix_backend() const {
    require(trace_backend_ != nullptr, "Scene: this operation requires a triangle trace backend, but the "
                                       "scene was built with trace_backend='none' (or OptiX is unavailable).");
    require(triangle_kind_ == TraceBackendKind::Optix,
            "Scene: this operation requires the OptiX trace backend; the scene was "
            "built with trace_backend='cuda'. CUDA multipath arrives with the "
            "CudaFusedExecutor (P4).");
    // Every triangle-backend query funnels through this accessor, so the
    // build-device check here also covers the query entry points that live in
    // the other Scene translation units.
    require_build_device("Scene: this operation");
    return *static_cast<OptixTraceBackend*>(trace_backend_.get());
}

CudaTraceBackend& Scene::cuda_backend() const {
    require(trace_backend_ != nullptr && triangle_kind_ == TraceBackendKind::Cuda,
            "Scene: this operation requires the CUDA trace backend.");
    require_build_device("Scene: this operation");
    return *static_cast<CudaTraceBackend*>(trace_backend_.get());
}

std::vector<int> Scene::cuda_first_blocker_selftest(const Vector3f& origin, const Vector3f& direction,
                                                    const Float& tmax, const std::vector<int>& ignore_prim_ids) const {
    require(is_ready(), "Scene::cuda_first_blocker_selftest(): scene is not built.");
    require(!pending_updates_,
            "Scene::cuda_first_blocker_selftest(): scene has pending updates. Call Scene::sync() first.");
    require_build_device("Scene::cuda_first_blocker_selftest()");
    return cuda_backend().first_blocker_selftest(origin, direction, tmax, ignore_prim_ids);
}

OptixScene& Scene::optix_scene() const {
    return optix_backend().primary();
}

OptixScene& Scene::optix_static_scene() const {
    return optix_backend().static_scene();
}

OptixScene& Scene::optix_dynamic_scene() const {
    return optix_backend().dynamic_scene();
}

bool Scene::optix_split_active() const {
    return triangle_kind_ == TraceBackendKind::Optix && optix_backend().split_active();
}

OptixSceneSelection Scene::select_optix_scenes() const {
    return optix_backend().select_scenes();
}

template <bool Detached>
NearestPointEdgeT<Detached> Scene::nearest_edge(const Vector3fT<Detached>& point, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(point): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(point): scene has pending updates. Call Scene::sync() first.");
    require_build_device("Scene::nearest_edge(point)");

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
    ClosestEdgeCandidate candidate = use_optix_candidate
                                         ? edge_optix_->template nearest_edge<Detached>(point, query_mask)
                                         : edge_bvh_->template nearest_edge<Detached>(point, query_mask);
    const Mask valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Int global_edge_id_detached = use_optix_candidate
                                            ? candidate.global_edge_id
                                            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const Int shape_id_detached = gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached = gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

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
        const Vector3f p0 = gather<Vector3f>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3f e1 = gather<Vector3f>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
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
NearestRayEdgeT<Detached> Scene::nearest_edge(const RayT<Detached>& ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(ray): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(ray): scene has pending updates. Call Scene::sync() first.");
    require_build_device("Scene::nearest_edge(ray)");

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
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) && drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) && drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &=
            squared_norm(Vector3f(detach<false>(ray.d.x()), detach<false>(ray.d.y()), detach<false>(ray.d.z()))) > 0.f;
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
    ClosestEdgeCandidate candidate = use_optix_candidate ? edge_optix_->template nearest_edge<Detached>(ray, query_mask)
                                                         : edge_bvh_->template nearest_edge<Detached>(ray, query_mask);
    const Mask valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Mask finite_tmax = drjit::isfinite(t_max_input);
    const Int global_edge_id_detached = use_optix_candidate
                                            ? candidate.global_edge_id
                                            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const Int shape_id_detached = gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached = gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

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
        const Vector3f p0 = gather<Vector3f>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3f e1 = gather<Vector3f>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
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
NearestEdgesTopKT<Detached> Scene::nearest_edges(const Vector3fT<Detached>& point, int k,
                                                 MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edges(point): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edges(point): scene has pending updates. Call Scene::sync() first.");
    require(k > 0, "Scene::nearest_edges(point): k must be positive.");
    require(k <= 16, "Scene::nearest_edges(point): k must be <= 16.");
    require_build_device("Scene::nearest_edges(point)");

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
        active_detached &= drjit::isfinite(detach<false>(point.x())) && drjit::isfinite(detach<false>(point.y())) &&
                           drjit::isfinite(detach<false>(point.z()));
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
    const bool use_optix_candidate = edge_backend_uses_optix_topk(edge_bvh_backend_);
    const ClosestEdgeTopKCandidate candidate = use_optix_candidate
                                                   ? edge_optix_->template nearest_edges<Detached>(point, k, query_mask)
                                                   : edge_bvh_->template nearest_edges<Detached>(point, k, query_mask);
    const Mask valid_detached = candidate.is_valid;
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Int output_index = arange<Int>(output_count);
    const Int output_query = output_index / k;
    const Int global_edge_id_detached = use_optix_candidate
                                            ? candidate.global_edge_ids
                                            : edge_bvh_->map_to_global(candidate.global_edge_ids, valid_detached);
    const Int shape_id_detached = gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached = gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const MaskAD valid = MaskAD(valid_detached);
        const IntAD global_edge_id = IntAD(global_edge_id_detached);
        const IntAD query_id = IntAD(output_query);
        const Vector3fAD output_point = gather<Vector3fAD>(point, query_id, valid);
        const Vector3fAD edge_start = gather<Vector3fAD>(edge_info_.start, global_edge_id, valid);
        const Vector3fAD edge_vector = gather<Vector3fAD>(edge_info_.edge, global_edge_id, valid);
        const MaskAD boundary = gather<MaskAD>(edge_info_.is_boundary, global_edge_id, valid);

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
        const Vector3f output_point = gather<Vector3f>(point, output_query, valid_detached);
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

template NearestPointEdge Scene::nearest_edge<true>(const Vector3f& point, Mask active) const;
template NearestPointEdgeAD Scene::nearest_edge<false>(const Vector3fAD& point, MaskAD active) const;
template NearestRayEdge Scene::nearest_edge<true>(const Ray& ray, Mask active) const;
template NearestRayEdgeAD Scene::nearest_edge<false>(const RayAD& ray, MaskAD active) const;
template NearestEdgesTopK Scene::nearest_edges<true>(const Vector3f& point, int k, Mask active) const;
template NearestEdgesTopKAD Scene::nearest_edges<false>(const Vector3fAD& point, int k, MaskAD active) const;

} // namespace rayd

// Consolidated scene intersection implementation.
#include <rayd/jit/core.h>
#include <rayd/jit/scene.h>

namespace rayd {

namespace {
bool uses_symbolic_optix_query_path() {
    // Dr.Jit symbolic recording cannot mix multiple OptiX pipelines/SBTs
    // within a single captured kernel. Fall back to the unified scene path.
    return jit_flag(JitFlag::Recording);
}

} // namespace

template <bool Detached>
IntersectionT<Detached> Scene::intersect(const RayT<Detached>& ray, MaskT<Detached> active, RayFlags flags) const {
    require(is_ready(), "Scene::intersect(): scene is not built.");
    require(!pending_updates_, "Scene::intersect(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(ray.o));
    const bool want_geo_n = has_flag(flags, RayFlags::Geometric);
    const bool want_shading = has_flag(flags, RayFlags::ShadingN);
    const bool want_uv = has_flag(flags, RayFlags::UV);
    const bool symbolic_optix_query = optix_split_active() && uses_symbolic_optix_query_path();

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
    if (triangle_kind_ == TraceBackendKind::Cuda) {
        require(!jit_flag(JitFlag::Recording), "trace_backend='cuda' cannot serve intersect() inside a Dr.Jit symbolic "
                                               "recording region; use trace_backend='optix' or evaluate outside the "
                                               "recorded loop.");
        optix_hit = cuda_backend().template intersect<Detached>(ray, hit_mask);
    } else if (optix_split_active() && !symbolic_optix_query) {
        MaskT<Detached> static_hit_mask = active;
        MaskT<Detached> dynamic_hit_mask = active;
        const OptixIntersection static_hit = optix_static_scene().template intersect<Detached>(ray, static_hit_mask);
        const OptixIntersection dynamic_hit = optix_dynamic_scene().template intersect<Detached>(ray, dynamic_hit_mask);

        const Mask static_hit_mask_detached = detach<false>(static_hit_mask);
        const Mask dynamic_hit_mask_detached = detach<false>(dynamic_hit_mask);
        const Mask choose_dynamic =
            dynamic_hit_mask_detached && (!static_hit_mask_detached || (dynamic_hit.t < static_hit.t));
        const Mask any_hit = static_hit_mask_detached || dynamic_hit_mask_detached;

        optix_hit.reserve(ray_count);
        optix_hit.t = select(choose_dynamic, dynamic_hit.t, static_hit.t);
        optix_hit.barycentric[0] = select(choose_dynamic, dynamic_hit.barycentric[0], static_hit.barycentric[0]);
        optix_hit.barycentric[1] = select(choose_dynamic, dynamic_hit.barycentric[1], static_hit.barycentric[1]);
        optix_hit.shape_id = select(choose_dynamic, dynamic_hit.shape_id, static_hit.shape_id);
        optix_hit.local_prim_id = select(choose_dynamic, dynamic_hit.local_prim_id, static_hit.local_prim_id);

        if constexpr (!Detached) {
            hit_mask = MaskAD(any_hit);
        } else {
            hit_mask = any_hit;
        }
    } else {
        optix_hit = optix_scene().template intersect<Detached>(ray, hit_mask);
    }

    const Int shape_id = optix_hit.shape_id;
    const Int local_primitive_id = optix_hit.local_prim_id;
    const Mask hit_mask_detached = detach<false>(hit_mask);
    const Int mesh_face_offset = gather<Int>(face_offsets_, shape_id, hit_mask_detached);
    const Int global_primitive_id = local_primitive_id + mesh_face_offset;

    Vector2fT<Detached> triangle_uv_coords;
    FloatT<Detached> hit_distance;

    if constexpr (!Detached) {
        // AD path: re-gather vertex data and recompute intersection for gradients.
        const IntAD global_primitive_id_ad = IntAD(global_primitive_id);
        const Vector3fAD triangle_p0 = gather<Vector3fAD>(triangle_info_.p0, global_primitive_id_ad, hit_mask);
        const Vector3fAD triangle_e1 = gather<Vector3fAD>(triangle_info_.e1, global_primitive_id_ad, hit_mask);
        const Vector3fAD triangle_e2 = gather<Vector3fAD>(triangle_info_.e2, global_primitive_id_ad, hit_mask);
        std::tie(triangle_uv_coords, hit_distance) =
            ray_intersect_triangle<Detached>(triangle_p0, triangle_e1, triangle_e2, ray);

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal =
                gather<Vector3fAD>(triangle_info_.face_normal, global_primitive_id_ad, hit_mask);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 =
                    gather<Vector3fAD>(triangle_info_.n0, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n1 =
                    gather<Vector3fAD>(triangle_info_.n1, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n2 =
                    gather<Vector3fAD>(triangle_info_.n2, global_primitive_id_ad, hit_mask);
                MaskT<Detached> use_face_normal_mask =
                    gather<MaskAD>(triangle_face_normal_mask_, global_primitive_id_ad, hit_mask);
                const Vector2fT<Detached> safe_uv =
                    select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal = normalize(
                    bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data =
                gather<TriangleUVAD>(triangle_uv_, global_primitive_id_ad, hit_mask);
            const Vector2fT<Detached> safe_uv =
                select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0],
                                    triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask, uv, intersection.uv);
        }
    } else {
        // Detached path: use OptiX results directly, gather only what is needed.
        triangle_uv_coords = optix_hit.barycentric;
        hit_distance = optix_hit.t;

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal =
                gather<Vector3f>(triangle_info_detached_.face_normal, global_primitive_id, hit_mask_detached);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 =
                    gather<Vector3f>(triangle_info_detached_.n0, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n1 =
                    gather<Vector3f>(triangle_info_detached_.n1, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n2 =
                    gather<Vector3f>(triangle_info_detached_.n2, global_primitive_id, hit_mask_detached);
                MaskT<Detached> use_face_normal_mask =
                    gather<Mask>(triangle_face_normal_mask_detached_, global_primitive_id, hit_mask_detached);
                const Vector2fT<Detached> safe_uv =
                    select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal = normalize(
                    bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask_detached, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask_detached, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data =
                gather<TriangleUV>(triangle_uv_detached_, global_primitive_id, hit_mask_detached);
            const Vector2fT<Detached> safe_uv =
                select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0],
                                    triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask_detached, uv, intersection.uv);
        }
    }

    hit_mask &= drjit::isfinite(hit_distance) && (hit_distance < ray.tmax);

    const FloatT<Detached> safe_hit_distance = select(hit_mask, hit_distance, zeros<FloatT<Detached>>(ray_count));
    const Vector2fT<Detached> safe_triangle_uv =
        select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));

    const Vector3fT<Detached> barycentric_coordinates(1.f - safe_triangle_uv.x() - safe_triangle_uv.y(),
                                                      safe_triangle_uv.x(), safe_triangle_uv.y());
    const Vector3fT<Detached> hit_position = ray(safe_hit_distance);

    intersection.t = select(hit_mask, safe_hit_distance, intersection.t);
    intersection.p = select(hit_mask, hit_position, intersection.p);
    intersection.barycentric = select(hit_mask, barycentric_coordinates, intersection.barycentric);
    intersection.shape_id = select(hit_mask, IntT<Detached>(shape_id), intersection.shape_id);
    const IntT<Detached> local_primitive_id_t = IntT<Detached>(local_primitive_id);
    const IntT<Detached> global_primitive_id_t = IntT<Detached>(global_primitive_id);
    intersection.prim_id = select(hit_mask, local_primitive_id_t, intersection.prim_id);
    intersection.local_prim_id = select(hit_mask, local_primitive_id_t, intersection.local_prim_id);
    intersection.global_prim_id = select(hit_mask, global_primitive_id_t, intersection.global_prim_id);
    return intersection;
}

template Intersection Scene::intersect<true>(const Ray& ray, Mask active, RayFlags flags) const;
template IntersectionAD Scene::intersect<false>(const RayAD& ray, MaskAD active, RayFlags flags) const;

} // namespace rayd

// Consolidated scene OptiX host facade.
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <rayd/jit/core.h>
#include <rayd/jit/mesh.h>
#include <rayd/jit/scene.h>

#include <rayd/jit/native_launch_audit.h>
#include <src/scene/scene_internal.h>

namespace rayd {

// No PTX module — HitObject API with invoke=0 never executes programs.

namespace dr = drjit;

#ifndef RAYD_OPTIX_MODULE_OPT_LEVEL
#define RAYD_OPTIX_MODULE_OPT_LEVEL OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
#endif

#ifndef RAYD_OPTIX_EXCEPTION_FLAGS
#define RAYD_OPTIX_EXCEPTION_FLAGS OPTIX_EXCEPTION_FLAG_NONE
#endif

namespace {

/// Build-quality bias for an OptiX acceleration structure: optimize traversal vs. build time.
enum class OptixAccelPreference { FastTrace, FastBuild };

std::string normalize_optix_mode_value(const char* value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) -> char { return static_cast<char>(std::tolower(ch)); });
    return normalized;
}

/// Read an accel-preference override from environment variable \p env_name; "auto"
/// (or unset) falls back to \p auto_value. Accepts fast_trace/trace, fast_build/build/relaxed/none.
OptixAccelPreference parse_optix_preference(const char* env_name, OptixAccelPreference auto_value) {
    const char* raw = std::getenv(env_name);
    const std::string normalized = normalize_optix_mode_value(raw);
    if (normalized.empty() || normalized == "auto") {
        return auto_value;
    }
    if (normalized == "fast_trace" || normalized == "trace") {
        return OptixAccelPreference::FastTrace;
    }
    if (normalized == "fast_build" || normalized == "build" || normalized == "relaxed" || normalized == "none") {
        return OptixAccelPreference::FastBuild;
    }
    throw std::runtime_error(std::string("Invalid ") + env_name +
                             ". Expected one of: auto, fast_trace, fast_build, relaxed.");
}

OptixAccelPreference active_dynamic_gas_preference() {
    static const OptixAccelPreference value =
        parse_optix_preference("RAYD_OPTIX_DYNAMIC_GAS_PREFERENCE", OptixAccelPreference::FastTrace);
    return value;
}

OptixAccelPreference active_dynamic_ias_preference() {
    static const OptixAccelPreference value =
        parse_optix_preference("RAYD_OPTIX_DYNAMIC_IAS_PREFERENCE", OptixAccelPreference::FastBuild);
    return value;
}

unsigned optix_preference_build_flag(OptixAccelPreference preference) {
    if (preference == OptixAccelPreference::FastBuild) {
#ifdef OPTIX_BUILD_FLAG_PREFER_FAST_BUILD
        return OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
#else
        return 0u;
#endif
    }
    return OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
}

/// Flatten the upper 3x4 of \p matrix into the row-major float[12] an OptixInstance expects.
void fill_optix_transform(float out[12], const Matrix4fAD& matrix) {
    Matrix4f detached = detach<false>(matrix);
    drjit::eval(detached);
    drjit::sync_thread();

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            drjit::store(&out[row * 4 + col], detached(row, col));
        }
    }
}

/// Per-mesh device buffers and bottom-level acceleration structure (GAS) handle.
struct OptixMeshState {
    bool dynamic = false;
    int face_offset = 0;
    int mesh_id = -1;
    void* vertex_buffer = nullptr;
    void* vertex_buffer_ptr = nullptr;
    void* gas_temp_buffer = nullptr;
    size_t gas_temp_buffer_size = 0;
    void* gas_buffer = nullptr;
    size_t gas_buffer_size = 0;
    OptixTraversableHandle gas_handle = 0;
    OptixAccelBuildOptions accel_options = {};
    OptixAccelBufferSizes gas_buffer_sizes = {};
};

void write_optix_instance(OptixInstance& instance, const OptixSceneMeshDesc& mesh_desc,
                          const OptixMeshState& mesh_state) {
    std::memset(&instance, 0, sizeof(instance));
    fill_optix_transform(instance.transform, mesh_desc.mesh->full_transform());
    instance.instanceId = static_cast<unsigned int>(mesh_desc.mesh_id);
    instance.sbtOffset = static_cast<unsigned int>(mesh_desc.mesh_id);
    instance.visibilityMask = 255u;
    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = mesh_state.gas_handle;
}

struct RetiredOptixJitResources {
    UInt pipeline_handle;
    UInt sbt_handle;
};

std::mutex& retired_optix_resources_mutex() {
    static std::mutex* mutex = new std::mutex();
    return *mutex;
}

std::vector<RetiredOptixJitResources>& retired_optix_resources() {
    static std::vector<RetiredOptixJitResources>* resources = new std::vector<RetiredOptixJitResources>();
    return *resources;
}

} // namespace

/// All OptiX device state backing one OptixScene: context, pipeline/SBT, per-mesh GAS, and the IAS.
struct OptixState {
    OptixDeviceContext context = 0;
    bool has_dynamic_meshes = false;
    bool has_static_meshes = false;
    bool owns_trace_handles = true;

    UInt64 handle;
    UInt pipeline_handle;
    UInt sbt_handle;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc pgd[2] = {};
    OptixProgramGroup pg[2] = {};
    OptixShaderBindingTable sbt = {};

    std::vector<HitGroupSbtRecord> hg_sbts;
    std::vector<OptixMeshState> mesh_states;
    std::vector<size_t> gas_owner_indices;
    std::vector<OptixInstance> instances;

    void* instance_buffer = nullptr;
    void* ias_temp_buffer = nullptr;
    size_t ias_temp_buffer_size = 0;
    void* ias_buffer = nullptr;
    size_t ias_buffer_size = 0;
    OptixTraversableHandle ias_handle = 0;
    OptixAccelBuildOptions ias_options = {};
    OptixAccelBufferSizes ias_buffer_sizes = {};
};

static void retire_optix_jit_resources(OptixState* state) {
    if (state == nullptr) {
        return;
    }

    if (state->pipeline_handle.index() == 0 && state->sbt_handle.index() == 0) {
        return;
    }

    // Dr.Jit 1.3.x can recycle OptiX pipeline/SBT JIT handles after a scene is
    // destroyed while cached kernels referring to those handles are still
    // around. Keep a bounded retirement list so handle IDs aren't immediately
    // reused, and only flush the kernel cache when the list grows too large.
    std::lock_guard<std::mutex> guard(retired_optix_resources_mutex());
    std::vector<RetiredOptixJitResources>& resources = retired_optix_resources();
    resources.push_back({state->pipeline_handle, state->sbt_handle});

    state->pipeline_handle = UInt();
    state->sbt_handle = UInt();

    constexpr size_t MaxRetiredOptixResourceSets = 32;
    if (resources.size() >= MaxRetiredOptixResourceSets) {
        jit_flush_kernel_cache();
        resources.clear();
    }
}

void OptixIntersection::reserve(int64_t size) {
    require(size >= 0, "OptixIntersection::reserve(): size must be non-negative.");
    if (size != m_size) {
        m_size = size;
        shape_id = empty<Int>(size);
        local_prim_id = empty<Int>(size);
        barycentric = empty<Vector2f>(size);
        t = empty<Float>(size);
    }
}

static void destroy_mesh_state(OptixMeshState& state) {
    if (state.vertex_buffer != nullptr) {
        jit_free(state.vertex_buffer);
        state.vertex_buffer = nullptr;
        state.vertex_buffer_ptr = nullptr;
    }
    if (state.gas_temp_buffer != nullptr) {
        jit_free(state.gas_temp_buffer);
        state.gas_temp_buffer = nullptr;
        state.gas_temp_buffer_size = 0;
    }
    if (state.gas_buffer != nullptr) {
        jit_free(state.gas_buffer);
        state.gas_buffer = nullptr;
        state.gas_buffer_size = 0;
        state.gas_handle = 0;
    }
}

static void destroy_optix_state(OptixState* state) {
    if (state == nullptr) {
        return;
    }

    jit_sync_thread();
    if (state->owns_trace_handles) {
        retire_optix_jit_resources(state);
    } else {
        state->pipeline_handle = UInt();
        state->sbt_handle = UInt();
    }

    for (OptixMeshState& mesh_state : state->mesh_states) {
        destroy_mesh_state(mesh_state);
    }

    if (state->instance_buffer != nullptr) {
        jit_free(state->instance_buffer);
    }
    if (state->ias_temp_buffer != nullptr) {
        jit_free(state->ias_temp_buffer);
    }
    if (state->ias_buffer != nullptr) {
        jit_free(state->ias_buffer);
    }
    delete state;
}

static void ensure_device_buffer(void*& buffer, size_t& buffer_size, size_t required_size) {
    if (required_size == 0) {
        return;
    }

    if (buffer != nullptr && buffer_size >= required_size) {
        return;
    }

    if (buffer != nullptr) {
        jit_free(buffer);
    }
    buffer = jit_malloc(AllocType::Device, required_size);
    buffer_size = required_size;
}

static void ensure_instance_buffer(OptixState* state) {
    const size_t instance_bytes = sizeof(OptixInstance) * state->instances.size();
    if (instance_bytes == 0) {
        return;
    }

    if (state->instance_buffer != nullptr) {
        return;
    }

    state->instance_buffer = jit_malloc(AllocType::Device, instance_bytes);
}

static void upload_instance_span(OptixState* state, size_t begin, size_t count) {
    if (count == 0) {
        return;
    }

    ensure_instance_buffer(state);
    const size_t byte_offset = begin * sizeof(OptixInstance);
    const size_t byte_count = count * sizeof(OptixInstance);
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA, reinterpret_cast<uint8_t*>(state->instance_buffer) + byte_offset,
               state->instances.data() + begin, byte_count);
}

/// Build a mesh's GAS from scratch; dynamic meshes allow update, static meshes are compacted.
static void build_gas(OptixState* state, OptixMeshState& mesh_state, const OptixSceneMeshDesc& mesh_desc) {
    const Mesh& mesh = *mesh_desc.mesh;

    if (mesh_state.vertex_buffer == nullptr) {
        mesh_state.vertex_buffer = jit_malloc(AllocType::Device, sizeof(float) * mesh.vertex_count() * 3);
        mesh_state.vertex_buffer_ptr = mesh_state.vertex_buffer;
    }

    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA, mesh_state.vertex_buffer, mesh.vertex_buffer().data(),
               sizeof(float) * mesh.vertex_count() * 3);

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = reinterpret_cast<const CUdeviceptr*>(&mesh_state.vertex_buffer_ptr);
    build_input.triangleArray.numVertices = static_cast<unsigned int>(mesh.vertex_count());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(const_cast<int*>(mesh.face_buffer().data()));
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.face_count());
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.preTransform = nullptr;
    build_input.triangleArray.numSbtRecords = 1;
    build_input.triangleArray.sbtIndexOffsetBuffer = nullptr;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    build_input.triangleArray.primitiveIndexOffset = 0;
    build_input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

    unsigned int triangle_input_flags[] = {0u};
    build_input.triangleArray.flags = triangle_input_flags;

    mesh_state.accel_options.buildFlags = mesh_state.dynamic
                                              ? optix_preference_build_flag(active_dynamic_gas_preference())
                                              : OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (mesh_state.dynamic) {
        mesh_state.accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    } else {
        mesh_state.accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    }
    mesh_state.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    jit_optix_check(optixAccelComputeMemoryUsage(state->context, &mesh_state.accel_options, &build_input, 1,
                                                 &mesh_state.gas_buffer_sizes));

    ensure_device_buffer(mesh_state.gas_temp_buffer, mesh_state.gas_temp_buffer_size,
                         std::max(mesh_state.gas_buffer_sizes.tempSizeInBytes,
                                  mesh_state.gas_buffer_sizes.tempUpdateSizeInBytes));
    void* gas_output = jit_malloc(AllocType::Device, mesh_state.gas_buffer_sizes.outputSizeInBytes);

    OptixAccelEmitDesc emit_desc = {};
    size_t* d_compacted_size = nullptr;
    unsigned int emit_count = 0;
    if (!mesh_state.dynamic) {
        d_compacted_size = static_cast<size_t*>(jit_malloc(AllocType::Device, sizeof(size_t)));
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = d_compacted_size;
        emit_count = 1;
    }

    audit_optix_accel_build();
    jit_optix_check(optixAccelBuild(state->context, jit_cuda_stream(), &mesh_state.accel_options, &build_input, 1,
                                    mesh_state.gas_temp_buffer, mesh_state.gas_buffer_sizes.tempSizeInBytes, gas_output,
                                    mesh_state.gas_buffer_sizes.outputSizeInBytes, &mesh_state.gas_handle,
                                    emit_count != 0 ? &emit_desc : nullptr, emit_count));

    if (mesh_state.dynamic) {
        mesh_state.gas_buffer = gas_output;
        mesh_state.gas_buffer_size = mesh_state.gas_buffer_sizes.outputSizeInBytes;
        return;
    }

    size_t compacted_size = mesh_state.gas_buffer_sizes.outputSizeInBytes;
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA, &compacted_size, d_compacted_size, sizeof(size_t));
    jit_free(d_compacted_size);

    if (compacted_size < mesh_state.gas_buffer_sizes.outputSizeInBytes) {
        void* gas_compact = jit_malloc(AllocType::Device, compacted_size);
        audit_optix_accel_compact();
        jit_optix_check(optixAccelCompact(state->context, jit_cuda_stream(), mesh_state.gas_handle, gas_compact,
                                          compacted_size, &mesh_state.gas_handle));
        jit_free(gas_output);
        gas_output = gas_compact;
        mesh_state.gas_buffer_size = compacted_size;
    } else {
        mesh_state.gas_buffer_size = mesh_state.gas_buffer_sizes.outputSizeInBytes;
    }

    mesh_state.gas_buffer = gas_output;
}

/// Refit an existing dynamic-mesh GAS in place after its vertices moved (no realloc).
static void update_gas(OptixState* state, OptixMeshState& mesh_state, const Mesh& mesh) {
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA, mesh_state.vertex_buffer, mesh.vertex_buffer().data(),
               sizeof(float) * mesh.vertex_count() * 3);

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = reinterpret_cast<const CUdeviceptr*>(&mesh_state.vertex_buffer_ptr);
    build_input.triangleArray.numVertices = static_cast<unsigned int>(mesh.vertex_count());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(const_cast<int*>(mesh.face_buffer().data()));
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.face_count());
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.preTransform = nullptr;
    build_input.triangleArray.numSbtRecords = 1;
    build_input.triangleArray.sbtIndexOffsetBuffer = nullptr;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    build_input.triangleArray.primitiveIndexOffset = 0;
    build_input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

    unsigned int triangle_input_flags[] = {0u};
    build_input.triangleArray.flags = triangle_input_flags;

    mesh_state.accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    const size_t update_temp_size = mesh_state.gas_buffer_sizes.tempUpdateSizeInBytes;
    ensure_device_buffer(mesh_state.gas_temp_buffer, mesh_state.gas_temp_buffer_size, update_temp_size);

    audit_optix_accel_build();
    jit_optix_check(optixAccelBuild(state->context, jit_cuda_stream(), &mesh_state.accel_options, &build_input, 1,
                                    mesh_state.gas_temp_buffer, update_temp_size, mesh_state.gas_buffer,
                                    mesh_state.gas_buffer_size, &mesh_state.gas_handle, nullptr, 0));

    mesh_state.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
}

static void initialize_instances(OptixState* state, const std::vector<OptixSceneMeshDesc>& meshes) {
    state->instances.resize(meshes.size());
    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
        write_optix_instance(state->instances[mesh_index], meshes[mesh_index],
                             state->mesh_states[state->gas_owner_indices[mesh_index]]);
    }

    upload_instance_span(state, 0, state->instances.size());
}

/// Rewrite the IAS instance records for the dirty meshes and upload them in contiguous spans.
static void update_dirty_instances(OptixState* state, const std::vector<OptixSceneMeshDesc>& meshes,
                                   const std::vector<int>& dirty_instance_indices) {
    if (dirty_instance_indices.empty()) {
        return;
    }

    require(state->instances.size() == meshes.size(),
            "OptixScene::sync(): instance cache size does not match the scene mesh count.");

    for (int mesh_index : dirty_instance_indices) {
        require(mesh_index >= 0 && mesh_index < static_cast<int>(meshes.size()),
                "OptixScene::sync(): dirty instance index is out of range.");
        write_optix_instance(state->instances[static_cast<size_t>(mesh_index)], meshes[static_cast<size_t>(mesh_index)],
                             state->mesh_states[state->gas_owner_indices[static_cast<size_t>(mesh_index)]]);
    }

    if (dirty_instance_indices.size() == state->instances.size()) {
        upload_instance_span(state, 0, state->instances.size());
        return;
    }

    size_t span_begin = static_cast<size_t>(dirty_instance_indices.front());
    size_t span_end = span_begin + 1;
    for (size_t offset = 1; offset < dirty_instance_indices.size(); ++offset) {
        const size_t instance_index = static_cast<size_t>(dirty_instance_indices[offset]);
        if (instance_index == span_end) {
            span_end = instance_index + 1;
            continue;
        }

        upload_instance_span(state, span_begin, span_end - span_begin);
        span_begin = instance_index;
        span_end = instance_index + 1;
    }

    upload_instance_span(state, span_begin, span_end - span_begin);
}

/// Build (or, when \p update, refit) the top-level instance acceleration structure over all meshes.
static void build_ias(OptixState* state, bool update) {
    require(!state->instances.empty(), "OptixScene: missing instances for IAS build.");

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(state->instance_buffer);
    build_input.instanceArray.numInstances = static_cast<unsigned int>(state->instances.size());
    build_input.instanceArray.instanceStride = sizeof(OptixInstance);

    if (state->has_dynamic_meshes) {
        state->ias_options.buildFlags =
            OPTIX_BUILD_FLAG_ALLOW_UPDATE | optix_preference_build_flag(active_dynamic_ias_preference());
    } else {
        state->ias_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    state->ias_options.operation = update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;

    if (!update) {
        jit_optix_check(optixAccelComputeMemoryUsage(state->context, &state->ias_options, &build_input, 1,
                                                     &state->ias_buffer_sizes));
        if (state->ias_buffer != nullptr) {
            jit_free(state->ias_buffer);
        }
        state->ias_buffer = jit_malloc(AllocType::Device, state->ias_buffer_sizes.outputSizeInBytes);
        state->ias_buffer_size = state->ias_buffer_sizes.outputSizeInBytes;
        ensure_device_buffer(state->ias_temp_buffer, state->ias_temp_buffer_size,
                             std::max(state->ias_buffer_sizes.tempSizeInBytes,
                                      state->ias_buffer_sizes.tempUpdateSizeInBytes));
    }

    const size_t temp_size =
        update ? state->ias_buffer_sizes.tempUpdateSizeInBytes : state->ias_buffer_sizes.tempSizeInBytes;

    audit_optix_accel_build();
    jit_optix_check(optixAccelBuild(state->context, jit_cuda_stream(), &state->ias_options, &build_input, 1,
                                    state->ias_temp_buffer, temp_size, state->ias_buffer, state->ias_buffer_size,
                                    &state->ias_handle, nullptr, 0));

    state->ias_options.operation = OPTIX_BUILD_OPERATION_BUILD;
}

OptixScene::OptixScene() = default;

OptixScene::~OptixScene() {
    destroy_optix_state(m_accel);
}

void OptixScene::build(const std::vector<OptixSceneMeshDesc>& meshes, const OptixScene* trace_source) {
    require(!meshes.empty(), "OptixScene::build(): missing meshes.");
    require(trace_source == nullptr || trace_source->m_accel != nullptr,
            "OptixScene::build(): trace_source must be built first.");

    destroy_optix_state(m_accel);

    init_optix_api();
    m_accel = new OptixState();
    m_accel->context = jit_optix_context();
    m_accel->owns_trace_handles = trace_source == nullptr;

    if (trace_source == nullptr) {
        m_accel->pipeline_compile_options.usesMotionBlur = false;
        m_accel->pipeline_compile_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        m_accel->pipeline_compile_options.numPayloadValues = 0;
        m_accel->pipeline_compile_options.numAttributeValues = 2;
        m_accel->pipeline_compile_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
        m_accel->pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        m_accel->pipeline_compile_options.usesPrimitiveTypeFlags =
            static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
        m_accel->pipeline_compile_options.allowOpacityMicromaps = 0;

        // No PTX module needed — built-in triangle intersection with HitObject API
        // (invoke=0) never executes CH/miss programs.  Program groups with nullptr
        // modules are valid for this configuration, matching Mitsuba's approach.
        std::memset(m_accel->pgd, 0, sizeof(m_accel->pgd));
        m_accel->pgd[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        m_accel->pgd[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        char log[1024];
        size_t log_size = sizeof(log);
        jit_optix_check(
            optixProgramGroupCreate(m_accel->context, m_accel->pgd, 2, &m_accel->pgo, log, &log_size, m_accel->pg));

        m_accel->sbt.missRecordBase = jit_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE);
        m_accel->sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        m_accel->sbt.missRecordCount = 1;
        jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[0], reinterpret_cast<void*>(m_accel->sbt.missRecordBase)));

        m_accel->hg_sbts = std::vector<HitGroupSbtRecord>(meshes.size());
        for (const OptixSceneMeshDesc& mesh_desc : meshes) {
            const size_t sbt_index = static_cast<size_t>(mesh_desc.mesh_id);
            require(sbt_index < m_accel->hg_sbts.size(),
                    "OptixScene::build(): mesh_id is out of range for SBT records.");
            m_accel->hg_sbts[sbt_index].data.shape_offset = mesh_desc.face_offset;
            m_accel->hg_sbts[sbt_index].data.shape_id = mesh_desc.mesh_id;
            jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[1], &m_accel->hg_sbts[sbt_index]));
        }

        m_accel->sbt.hitgroupRecordBase = jit_malloc(AllocType::HostPinned, meshes.size() * sizeof(HitGroupSbtRecord));
        m_accel->sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        m_accel->sbt.hitgroupRecordCount = static_cast<unsigned int>(meshes.size());
        audit_jit_memcpy_async();
        jit_memcpy_async(JitBackend::CUDA, reinterpret_cast<void*>(m_accel->sbt.hitgroupRecordBase),
                         m_accel->hg_sbts.data(), meshes.size() * sizeof(HitGroupSbtRecord));

        m_accel->sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(
            jit_malloc_migrate(reinterpret_cast<void*>(m_accel->sbt.missRecordBase), AllocType::Device, 1));
        m_accel->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(
            jit_malloc_migrate(reinterpret_cast<void*>(m_accel->sbt.hitgroupRecordBase), AllocType::Device, 1));

        m_accel->pipeline_handle =
            UInt::steal(jit_optix_configure_pipeline(&m_accel->pipeline_compile_options, nullptr, m_accel->pg, 2));
        m_accel->sbt_handle = UInt::steal(jit_optix_configure_sbt(&m_accel->sbt, m_accel->pipeline_handle.index()));
    } else {
        m_accel->pipeline_handle = trace_source->m_accel->pipeline_handle;
        m_accel->sbt_handle = trace_source->m_accel->sbt_handle;
    }

    m_accel->mesh_states.resize(meshes.size());
    m_accel->gas_owner_indices.resize(meshes.size());
    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
        const int owner_id = meshes[mesh_index].geometry_owner_id;
        require(owner_id >= 0 && owner_id < static_cast<int>(meshes.size()),
                "OptixScene::build(): shared GAS owner is absent from the scene.");
        const size_t owner_index = static_cast<size_t>(owner_id);
        require(meshes[owner_index].mesh_id == owner_id && meshes[owner_index].geometry_owner_id == owner_id,
                "OptixScene::build(): shared GAS owner id does not name an owning mesh.");
        m_accel->gas_owner_indices[mesh_index] = owner_index;

        OptixMeshState& mesh_state = m_accel->mesh_states[mesh_index];
        mesh_state.dynamic = meshes[mesh_index].geometry_dynamic;
        mesh_state.face_offset = meshes[mesh_index].face_offset;
        mesh_state.mesh_id = meshes[mesh_index].mesh_id;
        m_accel->has_dynamic_meshes |= meshes[mesh_index].dynamic;
        m_accel->has_static_meshes |= !meshes[mesh_index].dynamic;
        if (owner_index == mesh_index) {
            build_gas(m_accel, mesh_state, meshes[mesh_index]);
        }
    }

    initialize_instances(m_accel, meshes);
    build_ias(m_accel, false);
}

void OptixScene::sync(const std::vector<OptixSceneMeshDesc>& meshes, const std::vector<OptixSceneMeshUpdate>& updates) {
    require(m_accel != nullptr, "OptixScene::sync(): scene is not built.");
    last_sync_profile_ = OptixSyncProfile();

    if (updates.empty()) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    bool has_vertex_updates = false;
    std::vector<uint8_t> dirty_instance_mask(meshes.size(), 0);

    for (const OptixSceneMeshUpdate& update : updates) {
        require(update.mesh_id >= 0 && update.mesh_id < static_cast<int>(m_accel->mesh_states.size()),
                "OptixScene::sync(): mesh_id is out of range.");
        if (update.vertices_dirty) {
            ++last_sync_profile_.updated_vertex_meshes;
            has_vertex_updates = true;
        }
        if (update.transform_dirty) {
            ++last_sync_profile_.updated_transform_meshes;
            dirty_instance_mask[static_cast<size_t>(update.mesh_id)] = 1;
        }
        if (!update.vertices_dirty) {
            continue;
        }

        const size_t instance_index = static_cast<size_t>(update.mesh_id);
        const size_t owner_index = m_accel->gas_owner_indices[instance_index];
        require(owner_index == instance_index,
                "OptixScene::sync(): instance vertices cannot update shared source geometry.");
        OptixMeshState& mesh_state = m_accel->mesh_states[owner_index];
        require(mesh_state.dynamic, "OptixScene::sync(): attempted to update a non-dynamic mesh.");
        const auto gas_start = Clock::now();
        const OptixTraversableHandle previous_handle = mesh_state.gas_handle;
        update_gas(m_accel, mesh_state, *meshes[instance_index].mesh);
        last_sync_profile_.gas_update_ms += std::chrono::duration<double, std::milli>(Clock::now() - gas_start).count();
        if (mesh_state.gas_handle != previous_handle) {
            for (size_t mesh_index = 0; mesh_index < m_accel->gas_owner_indices.size(); ++mesh_index) {
                if (m_accel->gas_owner_indices[mesh_index] == owner_index) {
                    dirty_instance_mask[mesh_index] = 1;
                }
            }
        }
    }

    std::vector<int> dirty_instance_indices;
    dirty_instance_indices.reserve(updates.size());
    for (size_t mesh_index = 0; mesh_index < dirty_instance_mask.size(); ++mesh_index) {
        if (dirty_instance_mask[mesh_index] != 0) {
            dirty_instance_indices.push_back(static_cast<int>(mesh_index));
        }
    }

    const bool needs_instance_upload = !dirty_instance_indices.empty();
    if (needs_instance_upload) {
        update_dirty_instances(m_accel, meshes, dirty_instance_indices);
    }

    const bool needs_ias_update = has_vertex_updates || needs_instance_upload;
    if (needs_ias_update) {
        const auto ias_start = Clock::now();
        build_ias(m_accel, true);
        last_sync_profile_.ias_update_ms = std::chrono::duration<double, std::milli>(Clock::now() - ias_start).count();
    }
    last_sync_profile_.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
}

bool OptixScene::is_ready() const {
    return m_accel != nullptr;
}

OptixDeviceContext OptixScene::context() const {
    require(m_accel != nullptr, "OptixScene::context(): scene is not built.");
    return m_accel->context;
}

OptixTraversableHandle OptixScene::ias_handle() const {
    require(m_accel != nullptr, "OptixScene::ias_handle(): scene is not built.");
    return m_accel->ias_handle;
}

template <bool Detached>
OptixIntersection OptixScene::intersect(const RayT<Detached>& ray, MaskT<Detached>& active) const {
    const int ray_count = static_cast<int>(slices(ray.o));

    OptixIntersection intersection;
    intersection.reserve(ray_count);

    Float ox;
    Float oy;
    Float oz;
    Float dx;
    Float dy;
    Float dz;
    Float t_max_input;
    if constexpr (!Detached) {
        ox = detach<false>(ray.o.x());
        oy = detach<false>(ray.o.y());
        oz = detach<false>(ray.o.z());
        dx = detach<false>(ray.d.x());
        dy = detach<false>(ray.d.y());
        dz = detach<false>(ray.d.z());
        t_max_input = detach<false>(ray.tmax);
    } else {
        ox = ray.o.x();
        oy = ray.o.y();
        oz = ray.o.z();
        dx = ray.d.x();
        dy = ray.d.y();
        dz = ray.d.z();
        t_max_input = ray.tmax;
    }

    Mask active_detached = detach<false>(active);

    Float t_min = RayEpsilon;
    Float t_max = select(drjit::isfinite(t_max_input), t_max_input, full<Float>(1e8f, ray_count));
    Float time = 0.f;
    UInt ray_mask(255);
    UInt ray_flags(OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
    UInt sbt_offset(0);
    UInt sbt_stride(1);
    UInt miss_sbt_index(0);

    m_accel->handle = dr::opaque<UInt64>(m_accel->ias_handle);
    uint32_t trace_args[]{
        m_accel->handle.index(),
        ox.index(),
        oy.index(),
        oz.index(),
        dx.index(),
        dy.index(),
        dz.index(),
        t_min.index(),
        t_max.index(),
        time.index(),
        ray_mask.index(),
        ray_flags.index(),
        sbt_offset.index(),
        sbt_stride.index(),
        miss_sbt_index.index(),
    };

    constexpr std::size_t kSceneHitObjectFieldCount =
        static_cast<std::size_t>(shared::optix::SceneHitObjectFieldSlot::Count);
    OptixHitObjectField fields[kSceneHitObjectFieldCount]{
        OptixHitObjectField::IsHit,      OptixHitObjectField::RayTMax,        OptixHitObjectField::Attribute0,
        OptixHitObjectField::Attribute1, OptixHitObjectField::PrimitiveIndex, OptixHitObjectField::InstanceId,
    };
    uint32_t hitobject_out[kSceneHitObjectFieldCount];

    jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args,
                        static_cast<uint32_t>(kSceneHitObjectFieldCount), fields, hitobject_out, 0, 0, 0, 0,
                        active_detached.index(), m_accel->pipeline_handle.index(), m_accel->sbt_handle.index());

    Mask is_hit = UInt::steal(hitobject_out[0]) != 0u;
    active_detached &= is_hit;

    using Single = drjit::float32_array_t<Float>;
    intersection.t = drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[1]));
    intersection.barycentric[0] = drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[2]));
    intersection.barycentric[1] = drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[3]));
    UInt raw_prim_index = UInt::steal(hitobject_out[4]);
    intersection.local_prim_id = Int(raw_prim_index);
    intersection.shape_id = Int(UInt::steal(hitobject_out[5]));

    // Clear invalid lanes.
    intersection.t[!active_detached] = Infinity;
    intersection.shape_id[!active_detached] = -1;
    intersection.local_prim_id[!active_detached] = -1;

    if constexpr (!Detached) {
        active &= MaskAD(active_detached);
    } else {
        active = active_detached;
    }
    return intersection;
}

template <bool Detached>
MaskT<Detached> OptixScene::shadow_test(const RayT<Detached>& ray, MaskT<Detached> active) const {
    const int ray_count = static_cast<int>(slices(ray.o));
    MaskT<Detached> hit = full<MaskT<Detached>>(false, ray_count);

    Float ox;
    Float oy;
    Float oz;
    Float dx;
    Float dy;
    Float dz;
    Float t_max_input;
    if constexpr (!Detached) {
        ox = detach<false>(ray.o.x());
        oy = detach<false>(ray.o.y());
        oz = detach<false>(ray.o.z());
        dx = detach<false>(ray.d.x());
        dy = detach<false>(ray.d.y());
        dz = detach<false>(ray.d.z());
        t_max_input = detach<false>(ray.tmax);
    } else {
        ox = ray.o.x();
        oy = ray.o.y();
        oz = ray.o.z();
        dx = ray.d.x();
        dy = ray.d.y();
        dz = ray.d.z();
        t_max_input = ray.tmax;
    }

    Mask active_detached = detach<false>(active);

    Float t_min = RayEpsilon;
    Float t_max = select(drjit::isfinite(t_max_input), t_max_input, full<Float>(1e8f, ray_count));
    Float time = 0.f;
    UInt ray_mask(255);
    UInt ray_flags(OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    UInt sbt_offset(0);
    UInt sbt_stride(1);
    UInt miss_sbt_index(0);

    m_accel->handle = dr::opaque<UInt64>(m_accel->ias_handle);
    uint32_t trace_args[]{
        m_accel->handle.index(),
        ox.index(),
        oy.index(),
        oz.index(),
        dx.index(),
        dy.index(),
        dz.index(),
        t_min.index(),
        t_max.index(),
        time.index(),
        ray_mask.index(),
        ray_flags.index(),
        sbt_offset.index(),
        sbt_stride.index(),
        miss_sbt_index.index(),
    };

    OptixHitObjectField fields[]{OptixHitObjectField::IsHit};
    uint32_t hitobject_out[1];

    jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args, 1, fields, hitobject_out, 0, 0, 0, 0,
                        active_detached.index(), m_accel->pipeline_handle.index(), m_accel->sbt_handle.index());

    const Mask hit_detached = active_detached && (UInt::steal(hitobject_out[0]) != 0u);
    if constexpr (!Detached) {
        hit = MaskAD(hit_detached);
    } else {
        hit = hit_detached;
    }
    return hit;
}

template <bool Detached>
OptixSegmentHit OptixScene::segment_hit(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                                        MaskT<Detached> active) const {
    const int ray_count = static_cast<int>(slices(start));

    Vector3f start_detached;
    Vector3f end_detached;
    Mask active_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_detached = detach<false>(end);
        active_detached = detach<false>(active);
    } else {
        start_detached = start;
        end_detached = end;
        active_detached = active;
    }

    active_detached &= drjit::isfinite(start_detached.x()) && drjit::isfinite(start_detached.y()) &&
                       drjit::isfinite(start_detached.z()) && drjit::isfinite(end_detached.x()) &&
                       drjit::isfinite(end_detached.y()) && drjit::isfinite(end_detached.z());

    const Vector3f delta = end_detached - start_detached;
    const Float length_sq = squared_norm(delta);
    const Mask valid_segment = length_sq > (2.f * Epsilon) * (2.f * Epsilon);
    const Float safe_length = sqrt(select(valid_segment, length_sq, Float(1.f)));
    const Vector3f direction = delta / safe_length;
    const Vector3f origin = start_detached + Epsilon * direction;
    const Float t_min = Epsilon;
    const Float t_max = maximum(safe_length - 2.f * Epsilon, Float(0.f));
    const Float time = 0.f;
    const UInt ray_mask(255);
    const UInt ray_flags(OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                         OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    const UInt sbt_offset(0);
    const UInt sbt_stride(1);
    const UInt miss_sbt_index(0);
    const Mask trace_active = active_detached && valid_segment;

    m_accel->handle = dr::opaque<UInt64>(m_accel->ias_handle);
    uint32_t trace_args[]{
        m_accel->handle.index(), origin.x().index(),    origin.y().index(), origin.z().index(), direction.x().index(),
        direction.y().index(),   direction.z().index(), t_min.index(),      t_max.index(),      time.index(),
        ray_mask.index(),        ray_flags.index(),     sbt_offset.index(), sbt_stride.index(), miss_sbt_index.index(),
    };

    OptixHitObjectField fields[]{
        OptixHitObjectField::IsHit,
        OptixHitObjectField::PrimitiveIndex,
        OptixHitObjectField::SBTDataPointer,
    };
    uint32_t hitobject_out[3];

    jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t), trace_args, 3, fields, hitobject_out, 0, 0, 0, 0,
                        trace_active.index(), m_accel->pipeline_handle.index(), m_accel->sbt_handle.index());

    const Mask hit = trace_active && (UInt::steal(hitobject_out[0]) != 0u);
    const UInt raw_prim_index = UInt::steal(hitobject_out[1]);
    const UInt64 sbt_data_ptr = UInt64::steal(hitobject_out[2]);
    const Int shape_offset =
        Int(UInt::steal(jit_optix_sbt_data_load(sbt_data_ptr.index(), VarType::UInt32, 0, hit.index())));

    OptixSegmentHit result;
    result.visible = active_detached && (!valid_segment || !hit);
    result.hit = hit;
    result.global_prim_id = select(hit, Int(raw_prim_index) + shape_offset, full<Int>(-1, ray_count));
    return result;
}

template OptixIntersection OptixScene::intersect<true>(const Ray& ray, Mask& active) const;
template OptixIntersection OptixScene::intersect<false>(const RayAD& ray, MaskAD& active) const;
template Mask OptixScene::shadow_test<true>(const Ray& ray, Mask active) const;
template MaskAD OptixScene::shadow_test<false>(const RayAD& ray, MaskAD active) const;
template OptixSegmentHit OptixScene::segment_hit<true>(const Vector3f& start, const Vector3f& end, Mask active) const;
template OptixSegmentHit OptixScene::segment_hit<false>(const Vector3fAD& start, const Vector3fAD& end,
                                                        MaskAD active) const;

} // namespace rayd

// Consolidated scene custom-operation bindings.
#include <utility>

#include <drjit/custom.h>
#include <nanobind/nanobind.h>

#include "scene_internal_jit.h"
#include <src/diffraction/accumulation_ad_jit.h>

namespace drjit {

template <typename T> struct struct_support {
    using Traversable = traversable_t<T>;

    template <typename T1, typename F> static void apply_1(T1&& value, F&& f) {
        auto fields = Traversable::fields(value);
        traverse_1(fields, std::forward<F>(f));
    }

    template <typename T1, typename T2, typename F> static void apply_2(T1&& value_1, T2&& value_2, F&& f) {
        auto fields_1 = Traversable::fields(value_1);
        auto fields_2 = Traversable::fields(value_2);
        traverse_2(fields_1, fields_2, std::forward<F>(f));
    }
};

} // namespace drjit

namespace rayd {

thread_local DfrDirectTapeCapture* active_dfr_direct_tape_capture = nullptr;

namespace {

namespace nb = nanobind;

template <typename Output, typename Input> class RaydCustomOp : public drjit::detail::CustomOpBase {
  public:
    explicit RaydCustomOp(const Input& input) : registered_input(drjit::detail::ad_scan(*this, input, true)) {}

    void register_output(const Output& output) { registered_output = drjit::detail::ad_scan(*this, output, false); }

  protected:
    Input registered_input;
    Output registered_output;
};

struct DfrDirectAccumOpInput {
    DfrStatesAD states;
    DfrMaterialAD material;
    Vector3fAD suffix_tri_p0;
    Vector3fAD suffix_tri_face_normal;
    Vector3fAD suffix_vertices;
    Vector3i suffix_faces;
    MaskAD active;

    DRJIT_STRUCT(DfrDirectAccumOpInput, states, material, suffix_tri_p0, suffix_tri_face_normal, suffix_vertices,
                 active)
};

struct DfrDirectAccumOpInputDetached {
    DfrStates states;
    DfrMaterial material;
    Vector3f suffix_tri_p0;
    Vector3f suffix_tri_face_normal;
    Vector3f suffix_vertices;
    Vector3i suffix_faces;
    Mask active;
};

DfrDirectAccumOpInputDetached detach_dfr_direct_input(const DfrDirectAccumOpInput& input) {
    DfrDirectAccumOpInputDetached detached;
    detached.states.count = input.states.count;
    detached.states.edge_index = detach<false>(input.states.edge_index);
    detached.states.edge_pos = detach<false>(input.states.edge_pos);
    detached.states.edge_dir = detach<false>(input.states.edge_dir);
    detached.states.edge_t_min = detach<false>(input.states.edge_t_min);
    detached.states.edge_t_max = detach<false>(input.states.edge_t_max);
    detached.states.n0 = detach<false>(input.states.n0);
    detached.states.n1 = detach<false>(input.states.n1);
    detached.states.prim0 = detach<false>(input.states.prim0);
    detached.states.prim1 = detach<false>(input.states.prim1);
    detached.states.exterior_angle = detach<false>(input.states.exterior_angle);
    detached.states.src = detach<false>(input.states.src);
    detached.states.src_power = detach<false>(input.states.src_power);
    detached.states.wi = detach<false>(input.states.wi);
    detached.states.d0 = detach<false>(input.states.d0);
    detached.states.prefix_depth = detach<false>(input.states.prefix_depth);
    detached.material.eta_r = detach<false>(input.material.eta_r);
    detached.material.sigma = detach<false>(input.material.sigma);
    detached.material.mu_r = detach<false>(input.material.mu_r);
    detached.material.gain = detach<false>(input.material.gain);
    detached.material.valid = detach<false>(input.material.valid);
    detached.suffix_tri_p0 = detach<false>(input.suffix_tri_p0);
    detached.suffix_tri_face_normal = detach<false>(input.suffix_tri_face_normal);
    detached.suffix_vertices = detach<false>(input.suffix_vertices);
    detached.suffix_faces = input.suffix_faces;
    detached.active = detach<false>(input.active);
    detached.states.count = input.states.count;
    return detached;
}

DfrStates detach_dfr_states_input(const DfrStatesAD& input) {
    DfrStates detached;
    detached.count = input.count;
    detached.edge_index = detach<false>(input.edge_index);
    detached.edge_pos = detach<false>(input.edge_pos);
    detached.edge_dir = detach<false>(input.edge_dir);
    detached.edge_t_min = detach<false>(input.edge_t_min);
    detached.edge_t_max = detach<false>(input.edge_t_max);
    detached.n0 = detach<false>(input.n0);
    detached.n1 = detach<false>(input.n1);
    detached.prim0 = detach<false>(input.prim0);
    detached.prim1 = detach<false>(input.prim1);
    detached.exterior_angle = detach<false>(input.exterior_angle);
    detached.src = detach<false>(input.src);
    detached.src_power = detach<false>(input.src_power);
    detached.wi = detach<false>(input.wi);
    detached.d0 = detach<false>(input.d0);
    detached.prefix_depth = detach<false>(input.prefix_depth);
    return detached;
}

DfrAccumAD dfr_accum_to_ad(const DfrAccum& input) {
    DfrAccumAD output;
    output.grid_cell_count = input.grid_cell_count;
    output.power = FloatAD(input.power);
    output.field_x = drjit::Complex<FloatAD>(FloatAD(input.field_x.x()), FloatAD(input.field_x.y()));
    output.field_y = drjit::Complex<FloatAD>(FloatAD(input.field_y.x()), FloatAD(input.field_y.y()));
    output.field_z = drjit::Complex<FloatAD>(FloatAD(input.field_z.x()), FloatAD(input.field_z.y()));
    output.direct_count = IntAD(input.direct_count);
    output.keller_count = IntAD(input.keller_count);
    output.suffix_count = IntAD(input.suffix_count);
    output.vis_rejects = IntAD(input.vis_rejects);
    output.edge_vis_rejects = IntAD(input.edge_vis_rejects);
    output.utd_rejects = IntAD(input.utd_rejects);
    output.edge_uses = IntAD(input.edge_uses);
    return output;
}

DfrAccum zero_dfr_accum_grad(int grid_cell_count) {
    DfrAccum output;
    output.grid_cell_count = grid_cell_count;
    output.power = zeros<Float>(grid_cell_count);
    output.field_x = drjit::Complex<Float>(zeros<Float>(grid_cell_count), zeros<Float>(grid_cell_count));
    output.field_y = drjit::Complex<Float>(zeros<Float>(grid_cell_count), zeros<Float>(grid_cell_count));
    output.field_z = drjit::Complex<Float>(zeros<Float>(grid_cell_count), zeros<Float>(grid_cell_count));
    output.direct_count = full<Int>(0, 1);
    output.keller_count = full<Int>(0, 1);
    output.suffix_count = full<Int>(0, 1);
    output.vis_rejects = full<Int>(0, 1);
    output.edge_vis_rejects = full<Int>(0, 1);
    output.utd_rejects = full<Int>(0, 1);
    output.edge_uses = full<Int>(0, 1);
    return output;
}

void set_dfr_accum_output_grad(DfrAccumAD& registered_output, const DfrAccum& grad_output) {
    drjit::set_grad(registered_output.power, grad_output.power);
    drjit::set_grad(registered_output.field_x.x(), grad_output.field_x.x());
    drjit::set_grad(registered_output.field_x.y(), grad_output.field_x.y());
    drjit::set_grad(registered_output.field_y.x(), grad_output.field_y.x());
    drjit::set_grad(registered_output.field_y.y(), grad_output.field_y.y());
    drjit::set_grad(registered_output.field_z.x(), grad_output.field_z.x());
    drjit::set_grad(registered_output.field_z.y(), grad_output.field_z.y());
}

class ScopedDfrDirectTapeCapture {
  public:
    explicit ScopedDfrDirectTapeCapture(DfrDirectTapeCapture* capture) : previous_(active_dfr_direct_tape_capture) {
        active_dfr_direct_tape_capture = capture;
    }

    ~ScopedDfrDirectTapeCapture() { active_dfr_direct_tape_capture = previous_; }

  private:
    DfrDirectTapeCapture* previous_ = nullptr;
};

int dfr_direct_sample_count(const DfrOptions& options) {
    return (options.strategy_mask & RAYD_DFR_DIRECT) != 0
               ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
               : 0;
}

int dfr_keller_sample_count(const DfrOptions& options) {
    return (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
}

int dfr_suffix_sample_count(const DfrOptions& options) {
    return (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
}

int dfr_direct_custom_ad_sample_count(const DfrOptions& options) {
    return dfr_direct_sample_count(options) + dfr_keller_sample_count(options) + dfr_suffix_sample_count(options);
}

void require_dfr_direct_custom_ad_supported_impl(const DfrOptions& options) {
    require(options.max_order == 1, "Scene::accum_dfr_direct(): native AD currently supports max_order == 1.");
}

void require_dfr_chain_custom_ad_supported_impl(const DfrOptions& options) {
    require(options.max_order == 2 || options.max_order == 3,
            "Scene::accum_dfr(): native AD currently supports max_order 2 or 3.");
}

template <typename FloatLike> Float coerce_float_grad(const FloatLike& value, size_t width) {
    Float detached = detach<false>(value);
    return detached.size() == width ? detached : zeros<Float>(width);
}

template <typename VecLike> Vector3f coerce_vec3_grad(const VecLike& value, size_t width) {
    Vector3f detached = detach<false>(value);
    return slices(detached) == width ? detached : zeros<Vector3f>(width);
}

struct DfrSuffixTriangleJvp {
    Vector3f p0;
    Vector3f face_normal;
};

DfrSuffixTriangleJvp dfr_suffix_triangle_jvp_from_vertices(const Vector3f& vertices, const Vector3i& faces,
                                                           const Vector3f& dot_vertices, size_t triangle_width) {
    DfrSuffixTriangleJvp result;
    result.p0 = zeros<Vector3f>(triangle_width);
    result.face_normal = zeros<Vector3f>(triangle_width);
    if (triangle_width == 0 || slices(vertices) == 0 || slices(dot_vertices) == 0) {
        return result;
    }

    const Vector3f v0 = gather<Vector3f>(vertices, faces[0]);
    const Vector3f v1 = gather<Vector3f>(vertices, faces[1]);
    const Vector3f v2 = gather<Vector3f>(vertices, faces[2]);
    const Vector3f dot_v0 = gather<Vector3f>(dot_vertices, faces[0]);
    const Vector3f dot_v1 = gather<Vector3f>(dot_vertices, faces[1]);
    const Vector3f dot_v2 = gather<Vector3f>(dot_vertices, faces[2]);

    const Vector3f e1 = v1 - v0;
    const Vector3f e2 = v2 - v0;
    const Vector3f dot_e1 = dot_v1 - dot_v0;
    const Vector3f dot_e2 = dot_v2 - dot_v0;
    const Vector3f raw_normal = cross(e1, e2);
    const Vector3f dot_raw_normal = cross(dot_e1, e2) + cross(e1, dot_e2);
    const Float raw_normal_norm = norm(raw_normal);
    const Mask valid = raw_normal_norm > Epsilon;
    const Vector3f face_normal = select(valid, raw_normal / raw_normal_norm, Vector3f(0.f, 0.f, 1.f));
    result.p0 = dot_v0;
    result.face_normal =
        select(valid, (dot_raw_normal - face_normal * dot(face_normal, dot_raw_normal)) / raw_normal_norm,
               zeros<Vector3f>(triangle_width));
    return result;
}

Vector3f dfr_suffix_triangle_vertex_vjp(const Vector3f& vertices, const Vector3i& faces, const Vector3f& grad_tri_p0,
                                        const Vector3f& grad_tri_face_normal, size_t vertex_width) {
    Vector3f grad_vertices = zeros<Vector3f>(vertex_width);
    if (vertex_width == 0 || slices(faces[0]) == 0) {
        return grad_vertices;
    }

    const Vector3f v0 = gather<Vector3f>(vertices, faces[0]);
    const Vector3f v1 = gather<Vector3f>(vertices, faces[1]);
    const Vector3f v2 = gather<Vector3f>(vertices, faces[2]);
    const Vector3f e1 = v1 - v0;
    const Vector3f e2 = v2 - v0;
    const Vector3f raw_normal = cross(e1, e2);
    const Float raw_normal_norm = norm(raw_normal);
    const Mask valid = raw_normal_norm > Epsilon;
    const Vector3f face_normal = select(valid, raw_normal / raw_normal_norm, Vector3f(0.f, 0.f, 1.f));
    const Vector3f grad_raw_normal =
        select(valid, (grad_tri_face_normal - face_normal * dot(face_normal, grad_tri_face_normal)) / raw_normal_norm,
               zeros<Vector3f>(slices(faces[0])));
    const Vector3f grad_e1 = cross(e2, grad_raw_normal);
    const Vector3f grad_e2 = cross(grad_raw_normal, e1);
    const Vector3f grad_v0 = grad_tri_p0 - grad_e1 - grad_e2;
    const Vector3f grad_v1 = grad_e1;
    const Vector3f grad_v2 = grad_e2;

    for (int axis = 0; axis < 3; ++axis) {
        scatter_reduce(ReduceOp::Add, grad_vertices[axis], grad_v0[axis], faces[0]);
        scatter_reduce(ReduceOp::Add, grad_vertices[axis], grad_v1[axis], faces[1]);
        scatter_reduce(ReduceOp::Add, grad_vertices[axis], grad_v2[axis], faces[2]);
    }
    return grad_vertices;
}

template <typename States> int dfr_state_count_for(const States& states) {
    const int state_width = static_cast<int>(slices(states.edge_index));
    return states.count > 0 ? states.count : state_width;
}

class DfrDirectAccumOp : public RaydCustomOp<DfrAccumAD, DfrDirectAccumOpInput> {
  public:
    using Base = RaydCustomOp<DfrAccumAD, DfrDirectAccumOpInput>;
    using OutputType = DfrAccumAD;

    DfrDirectAccumOp(const DfrDirectAccumOpInput& input, const Scene* scene, const DfrGrid& grid,
                     const DfrOptions& options)
        : Base(input), scene_(scene), grid_(grid), options_(options) {}

    OutputType eval(DfrDirectAccumOpInputDetached input) {
        m_input_ = input;
        const int launch_count = dfr_direct_custom_ad_sample_count(options_);
        if (launch_count > 0) {
            m_tape_.launch_count = launch_count;
            m_tape_.active = full<Mask>(false, launch_count);
            m_tape_.state_idx = full<Int>(-1, launch_count);
            m_tape_.cell = full<Int>(-1, launch_count);
            m_tape_.material_idx = full<Int>(-1, launch_count);
            m_tape_.edge_u = zeros<Float>(launch_count);
            drjit::eval(m_tape_.active, m_tape_.state_idx, m_tape_.cell, m_tape_.material_idx, m_tape_.edge_u);
        }

        ScopedDfrDirectTapeCapture tape_scope(launch_count > 0 ? &m_tape_ : nullptr);
        DfrAccum primal = scene_->accum_dfr_direct<true>(input.states, grid_, input.material, options_, input.active);
        return dfr_accum_to_ad(primal);
    }

    void forward() override {
        const int grid_cell_count = grid_.resolution0 * grid_.resolution1;
        DfrAccum output = zero_dfr_accum_grad(grid_cell_count);
        if (m_tape_.launch_count <= 0) {
            set_dfr_accum_output_grad(this->registered_output, output);
            return;
        }

        const size_t state_width = slices(m_input_.states.edge_index);
        const size_t material_width = slices(m_input_.material.gain);
        const size_t triangle_width = slices(m_input_.suffix_tri_p0);
        const size_t vertex_width = slices(m_input_.suffix_vertices);

        const Vector3f dot_edge_pos =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.states.edge_pos), state_width);
        const Vector3f dot_edge_dir =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.states.edge_dir), state_width);
        const Float dot_edge_t_min =
            coerce_float_grad(drjit::grad<false>(this->registered_input.states.edge_t_min), state_width);
        const Float dot_edge_t_max =
            coerce_float_grad(drjit::grad<false>(this->registered_input.states.edge_t_max), state_width);
        const Vector3f dot_src = coerce_vec3_grad(drjit::grad<false>(this->registered_input.states.src), state_width);
        const Vector3f dot_wi = coerce_vec3_grad(drjit::grad<false>(this->registered_input.states.wi), state_width);
        const Float dot_src_power =
            coerce_float_grad(drjit::grad<false>(this->registered_input.states.src_power), state_width);
        const Float dot_exterior_angle =
            coerce_float_grad(drjit::grad<false>(this->registered_input.states.exterior_angle), state_width);
        const Float dot_material_gain =
            coerce_float_grad(drjit::grad<false>(this->registered_input.material.gain), material_width);
        const Vector3f dot_suffix_vertices =
            coerce_vec3_grad(drjit::grad<false>(scene_->global_geometry().vertices), vertex_width);
        const DfrSuffixTriangleJvp dot_suffix_triangles =
            dfr_suffix_triangle_jvp_from_vertices(m_input_.suffix_vertices, m_input_.suffix_faces, dot_suffix_vertices,
                                                  triangle_width);
        const Vector3f& dot_tri_p0 = dot_suffix_triangles.p0;
        const Vector3f& dot_tri_face_normal = dot_suffix_triangles.face_normal;

        drjit::eval(dot_edge_pos, dot_edge_dir, dot_edge_t_min, dot_edge_t_max, dot_src, dot_wi, dot_src_power,
                    dot_exterior_angle, dot_material_gain, dot_suffix_vertices, dot_tri_p0, dot_tri_face_normal,
                    output.power, output.field_x.x());

        DfrDirectAccumADParams params = base_ad_params();
        params.dot_state_edge_pos_x = dot_edge_pos.x().data();
        params.dot_state_edge_pos_y = dot_edge_pos.y().data();
        params.dot_state_edge_pos_z = dot_edge_pos.z().data();
        params.dot_state_edge_dir_x = dot_edge_dir.x().data();
        params.dot_state_edge_dir_y = dot_edge_dir.y().data();
        params.dot_state_edge_dir_z = dot_edge_dir.z().data();
        params.dot_state_edge_t_min = dot_edge_t_min.data();
        params.dot_state_edge_t_max = dot_edge_t_max.data();
        params.dot_state_src_x = dot_src.x().data();
        params.dot_state_src_y = dot_src.y().data();
        params.dot_state_src_z = dot_src.z().data();
        params.dot_state_wi_x = dot_wi.x().data();
        params.dot_state_wi_y = dot_wi.y().data();
        params.dot_state_wi_z = dot_wi.z().data();
        params.dot_state_src_power = dot_src_power.data();
        params.dot_state_exterior_angle = dot_exterior_angle.data();
        params.dot_material_gain = dot_material_gain.data();
        params.dot_tri_p0_x = dot_tri_p0.x().data();
        params.dot_tri_p0_y = dot_tri_p0.y().data();
        params.dot_tri_p0_z = dot_tri_p0.z().data();
        params.dot_tri_fn_x = dot_tri_face_normal.x().data();
        params.dot_tri_fn_y = dot_tri_face_normal.y().data();
        params.dot_tri_fn_z = dot_tri_face_normal.z().data();
        params.dot_out_power = output.power.data();
        params.dot_out_field_x_re = output.field_x.x().data();
        dfr_direct_accum_jvp_gpu(params);
        set_dfr_accum_output_grad(this->registered_output, output);
    }

    void backward() override {
        if (m_tape_.launch_count <= 0) {
            return;
        }

        const size_t state_width = slices(m_input_.states.edge_index);
        const size_t material_width = slices(m_input_.material.gain);
        const size_t triangle_width = slices(m_input_.suffix_tri_p0);
        const size_t vertex_width = slices(m_input_.suffix_vertices);
        Vector3f grad_edge_pos = zeros<Vector3f>(state_width);
        Vector3f grad_edge_dir = zeros<Vector3f>(state_width);
        Float grad_edge_t_min = zeros<Float>(state_width);
        Float grad_edge_t_max = zeros<Float>(state_width);
        Vector3f grad_src = zeros<Vector3f>(state_width);
        Vector3f grad_wi = zeros<Vector3f>(state_width);
        Float grad_src_power = zeros<Float>(state_width);
        Float grad_exterior_angle = zeros<Float>(state_width);
        Float grad_material_gain = zeros<Float>(material_width);
        Vector3f grad_tri_p0 = zeros<Vector3f>(triangle_width);
        Vector3f grad_tri_face_normal = zeros<Vector3f>(triangle_width);
        Vector3f grad_suffix_vertices = zeros<Vector3f>(vertex_width);
        Float grad_power =
            coerce_float_grad(drjit::grad<false>(this->registered_output.power), grid_.resolution0 * grid_.resolution1);
        Float grad_field_x_re = coerce_float_grad(drjit::grad<false>(this->registered_output.field_x.x()),
                                                  grid_.resolution0 * grid_.resolution1);

        drjit::eval(grad_edge_pos, grad_edge_dir, grad_edge_t_min, grad_edge_t_max, grad_src, grad_wi, grad_src_power,
                    grad_exterior_angle, grad_material_gain, grad_tri_p0, grad_tri_face_normal, grad_suffix_vertices,
                    grad_power, grad_field_x_re);

        DfrDirectAccumADParams params = base_ad_params();
        params.grad_out_power = grad_power.data();
        params.grad_out_field_x_re = grad_field_x_re.data();
        params.grad_state_edge_pos_x = grad_edge_pos.x().data();
        params.grad_state_edge_pos_y = grad_edge_pos.y().data();
        params.grad_state_edge_pos_z = grad_edge_pos.z().data();
        params.grad_state_edge_dir_x = grad_edge_dir.x().data();
        params.grad_state_edge_dir_y = grad_edge_dir.y().data();
        params.grad_state_edge_dir_z = grad_edge_dir.z().data();
        params.grad_state_edge_t_min = grad_edge_t_min.data();
        params.grad_state_edge_t_max = grad_edge_t_max.data();
        params.grad_state_src_x = grad_src.x().data();
        params.grad_state_src_y = grad_src.y().data();
        params.grad_state_src_z = grad_src.z().data();
        params.grad_state_wi_x = grad_wi.x().data();
        params.grad_state_wi_y = grad_wi.y().data();
        params.grad_state_wi_z = grad_wi.z().data();
        params.grad_state_src_power = grad_src_power.data();
        params.grad_state_exterior_angle = grad_exterior_angle.data();
        params.grad_material_gain = grad_material_gain.data();
        params.grad_tri_p0_x = grad_tri_p0.x().data();
        params.grad_tri_p0_y = grad_tri_p0.y().data();
        params.grad_tri_p0_z = grad_tri_p0.z().data();
        params.grad_tri_fn_x = grad_tri_face_normal.x().data();
        params.grad_tri_fn_y = grad_tri_face_normal.y().data();
        params.grad_tri_fn_z = grad_tri_face_normal.z().data();
        dfr_direct_accum_vjp_gpu(params);
        grad_suffix_vertices = dfr_suffix_triangle_vertex_vjp(m_input_.suffix_vertices, m_input_.suffix_faces,
                                                              grad_tri_p0, grad_tri_face_normal, vertex_width);
        drjit::eval(grad_suffix_vertices);

        drjit::accum_grad(this->registered_input.states.edge_pos, drjit::detach<false>(grad_edge_pos));
        drjit::accum_grad(this->registered_input.states.edge_dir, drjit::detach<false>(grad_edge_dir));
        drjit::accum_grad(this->registered_input.states.edge_t_min, drjit::detach<false>(grad_edge_t_min));
        drjit::accum_grad(this->registered_input.states.edge_t_max, drjit::detach<false>(grad_edge_t_max));
        drjit::accum_grad(this->registered_input.states.src, drjit::detach<false>(grad_src));
        drjit::accum_grad(this->registered_input.states.wi, drjit::detach<false>(grad_wi));
        drjit::accum_grad(this->registered_input.states.src_power, drjit::detach<false>(grad_src_power));
        drjit::accum_grad(this->registered_input.states.exterior_angle, drjit::detach<false>(grad_exterior_angle));
        drjit::accum_grad(this->registered_input.material.gain, drjit::detach<false>(grad_material_gain));
        drjit::accum_grad(this->registered_input.suffix_vertices, drjit::detach<false>(grad_suffix_vertices));
    }

    const char* name() const override { return "DfrDirectAccum"; }

  private:
    DfrDirectAccumADParams base_ad_params() const {
        DfrDirectAccumADParams params = {};
        params.n_rays = m_tape_.launch_count;
        params.state_count = dfr_state_count_for(m_input_.states);
        params.material_count = static_cast<int>(slices(m_input_.material.gain));
        params.grid_axis = grid_.axis;
        params.grid_position = grid_.position;
        params.grid_coord0_min = grid_.coord0_min;
        params.grid_coord0_max = grid_.coord0_max;
        params.grid_coord1_min = grid_.coord1_min;
        params.grid_coord1_max = grid_.coord1_max;
        params.grid_resolution0 = grid_.resolution0;
        params.grid_resolution1 = grid_.resolution1;
        params.grid_cell_area = grid_.cell_area;
        params.direct_samples = dfr_direct_sample_count(options_);
        params.keller_samples = dfr_keller_sample_count(options_);
        params.suffix_samples = dfr_suffix_sample_count(options_);
        params.wavelength = options_.wavelength;
        params.seed = options_.seed;
        const TriangleInfo& triangles = scene_->triangle_info_detached();
        const bool suffix_enabled = params.suffix_samples > 0;
        params.n_triangles = suffix_enabled ? static_cast<int>(slices(m_input_.suffix_tri_p0)) : 0;
        params.tape_active = reinterpret_cast<const uint8_t*>(m_tape_.active.data());
        params.tape_state_idx = m_tape_.state_idx.data();
        params.tape_cell = m_tape_.cell.data();
        params.tape_material_idx = m_tape_.material_idx.data();
        params.tape_edge_u = m_tape_.edge_u.data();
        params.state_edge_pos_x = m_input_.states.edge_pos.x().data();
        params.state_edge_pos_y = m_input_.states.edge_pos.y().data();
        params.state_edge_pos_z = m_input_.states.edge_pos.z().data();
        params.state_edge_dir_x = m_input_.states.edge_dir.x().data();
        params.state_edge_dir_y = m_input_.states.edge_dir.y().data();
        params.state_edge_dir_z = m_input_.states.edge_dir.z().data();
        params.state_edge_t_min = m_input_.states.edge_t_min.data();
        params.state_edge_t_max = m_input_.states.edge_t_max.data();
        params.state_src_x = m_input_.states.src.x().data();
        params.state_src_y = m_input_.states.src.y().data();
        params.state_src_z = m_input_.states.src.z().data();
        params.state_wi_x = m_input_.states.wi.x().data();
        params.state_wi_y = m_input_.states.wi.y().data();
        params.state_wi_z = m_input_.states.wi.z().data();
        params.state_src_power = m_input_.states.src_power.data();
        params.state_exterior_angle = m_input_.states.exterior_angle.data();
        params.state_prim0 = m_input_.states.prim0.data();
        params.state_prim1 = m_input_.states.prim1.data();
        params.tri_p0_x = suffix_enabled ? m_input_.suffix_tri_p0.x().data() : nullptr;
        params.tri_p0_y = suffix_enabled ? m_input_.suffix_tri_p0.y().data() : nullptr;
        params.tri_p0_z = suffix_enabled ? m_input_.suffix_tri_p0.z().data() : nullptr;
        params.tri_e1_x = suffix_enabled ? triangles.e1.x().data() : nullptr;
        params.tri_e1_y = suffix_enabled ? triangles.e1.y().data() : nullptr;
        params.tri_e1_z = suffix_enabled ? triangles.e1.z().data() : nullptr;
        params.tri_e2_x = suffix_enabled ? triangles.e2.x().data() : nullptr;
        params.tri_e2_y = suffix_enabled ? triangles.e2.y().data() : nullptr;
        params.tri_e2_z = suffix_enabled ? triangles.e2.z().data() : nullptr;
        params.tri_fn_x = suffix_enabled ? m_input_.suffix_tri_face_normal.x().data() : nullptr;
        params.tri_fn_y = suffix_enabled ? m_input_.suffix_tri_face_normal.y().data() : nullptr;
        params.tri_fn_z = suffix_enabled ? m_input_.suffix_tri_face_normal.z().data() : nullptr;
        params.material_gain = m_input_.material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t*>(m_input_.material.valid.data());
        return params;
    }

    const Scene* scene_ = nullptr;
    DfrGrid grid_;
    DfrOptions options_;
    DfrDirectAccumOpInputDetached m_input_;
    DfrDirectTapeCapture m_tape_;
};

DfrAccumAD dfr_direct_accum_custom_op_impl(const Scene* scene, const DfrStatesAD& states, const DfrGrid& grid,
                                           const DfrMaterialAD& material, const DfrOptions& options,
                                           const Vector3fAD& suffix_tri_p0, const Vector3fAD& suffix_tri_face_normal,
                                           const Vector3fAD& suffix_vertices, const Vector3i& suffix_faces,
                                           const MaskAD& active) {
    DfrDirectAccumOpInput input;
    input.states = states;
    input.material = material;
    input.suffix_tri_p0 = suffix_tri_p0;
    input.suffix_tri_face_normal = suffix_tri_face_normal;
    input.suffix_vertices = suffix_vertices;
    input.suffix_faces = suffix_faces;
    input.active = active;
    nb::ref<DfrDirectAccumOp> op = new DfrDirectAccumOp(input, scene, grid, options);
    DfrAccumAD output = op->eval(detach_dfr_direct_input(input));
    drjit::detail::new_grad(output);
    op->register_output(output);
    if (!ad_custom_op(op.get())) {
        drjit::disable_grad(output);
    }
    return output;
}

struct DfrChainAccumOpInput {
    DfrStatesAD initial_states;
    DfrStatesAD recursive_states;
    DfrMaterialAD material;
    MaskAD active;
    Vector3fAD suffix_tri_p0;
    Vector3fAD suffix_tri_face_normal;
    Vector3fAD suffix_vertices;
    Vector3i suffix_faces;

    DRJIT_STRUCT(DfrChainAccumOpInput, initial_states, recursive_states, material, active, suffix_tri_p0,
                 suffix_tri_face_normal, suffix_vertices)
};

struct DfrChainAccumOpInputDetached {
    DfrStates initial_states;
    DfrStates recursive_states;
    DfrMaterial material;
    Mask active;
    Vector3f suffix_tri_p0;
    Vector3f suffix_tri_face_normal;
    Vector3f suffix_vertices;
    Vector3i suffix_faces;
};

DfrChainAccumOpInputDetached detach_dfr_chain_input(const DfrChainAccumOpInput& input) {
    DfrChainAccumOpInputDetached detached;
    detached.initial_states = detach_dfr_states_input(input.initial_states);
    detached.recursive_states = detach_dfr_states_input(input.recursive_states);
    detached.material.eta_r = detach<false>(input.material.eta_r);
    detached.material.sigma = detach<false>(input.material.sigma);
    detached.material.mu_r = detach<false>(input.material.mu_r);
    detached.material.gain = detach<false>(input.material.gain);
    detached.material.valid = detach<false>(input.material.valid);
    detached.active = detach<false>(input.active);
    detached.suffix_tri_p0 = detach<false>(input.suffix_tri_p0);
    detached.suffix_tri_face_normal = detach<false>(input.suffix_tri_face_normal);
    detached.suffix_vertices = detach<false>(input.suffix_vertices);
    detached.suffix_faces = input.suffix_faces;
    return detached;
}

class DfrChainAccumOp : public RaydCustomOp<DfrAccumAD, DfrChainAccumOpInput> {
  public:
    using Base = RaydCustomOp<DfrAccumAD, DfrChainAccumOpInput>;
    using OutputType = DfrAccumAD;

    DfrChainAccumOp(const DfrChainAccumOpInput& input, const Scene* scene, const DfrGrid& grid,
                    const DfrOptions& options)
        : Base(input), scene_(scene), grid_(grid), options_(options) {}

    OutputType eval(DfrChainAccumOpInputDetached input) {
        m_input_ = input;
        const int launch_count = dfr_direct_custom_ad_sample_count(options_);
        if (launch_count > 0) {
            m_tape_.launch_count = launch_count;
            m_tape_.active = full<Mask>(false, launch_count);
            m_tape_.state_idx = full<Int>(-1, launch_count);
            m_tape_.cell = full<Int>(-1, launch_count);
            m_tape_.material_idx = full<Int>(-1, launch_count);
            m_tape_.edge_u = zeros<Float>(launch_count);
            drjit::eval(m_tape_.active, m_tape_.state_idx, m_tape_.cell, m_tape_.material_idx, m_tape_.edge_u);
        }

        ScopedDfrDirectTapeCapture tape_scope(launch_count > 0 ? &m_tape_ : nullptr);
        DfrAccum primal = scene_->accum_dfr<true>(input.initial_states, input.recursive_states, grid_, input.material,
                                                  options_, input.active);
        return dfr_accum_to_ad(primal);
    }

    void forward() override {
        const int grid_cell_count = grid_.resolution0 * grid_.resolution1;
        DfrAccum output = zero_dfr_accum_grad(grid_cell_count);
        if (m_tape_.launch_count <= 0) {
            set_dfr_accum_output_grad(this->registered_output, output);
            return;
        }

        const size_t initial_width = slices(m_input_.initial_states.edge_index);
        const size_t recursive_width = slices(m_input_.recursive_states.edge_index);
        const size_t material_width = slices(m_input_.material.gain);
        const size_t triangle_width = slices(m_input_.suffix_tri_p0);
        const size_t vertex_width = slices(m_input_.suffix_vertices);
        const Vector3f dot_initial_edge_pos =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.initial_states.edge_pos), initial_width);
        const Vector3f dot_initial_edge_dir =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.initial_states.edge_dir), initial_width);
        const Float dot_initial_edge_t_min =
            coerce_float_grad(drjit::grad<false>(this->registered_input.initial_states.edge_t_min), initial_width);
        const Float dot_initial_edge_t_max =
            coerce_float_grad(drjit::grad<false>(this->registered_input.initial_states.edge_t_max), initial_width);
        const Vector3f dot_initial_src =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.initial_states.src), initial_width);
        const Float dot_initial_src_power =
            coerce_float_grad(drjit::grad<false>(this->registered_input.initial_states.src_power), initial_width);
        const Float dot_initial_exterior =
            coerce_float_grad(drjit::grad<false>(this->registered_input.initial_states.exterior_angle), initial_width);
        const Vector3f dot_recursive_edge_pos =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.recursive_states.edge_pos), recursive_width);
        const Vector3f dot_recursive_edge_dir =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.recursive_states.edge_dir), recursive_width);
        const Float dot_recursive_edge_t_min =
            coerce_float_grad(drjit::grad<false>(this->registered_input.recursive_states.edge_t_min), recursive_width);
        const Float dot_recursive_edge_t_max =
            coerce_float_grad(drjit::grad<false>(this->registered_input.recursive_states.edge_t_max), recursive_width);
        const Float dot_recursive_exterior =
            coerce_float_grad(drjit::grad<false>(this->registered_input.recursive_states.exterior_angle),
                              recursive_width);
        const Float dot_material_gain =
            coerce_float_grad(drjit::grad<false>(this->registered_input.material.gain), material_width);
        const Vector3f dot_suffix_vertices =
            coerce_vec3_grad(drjit::grad<false>(scene_->global_geometry().vertices), vertex_width);
        const DfrSuffixTriangleJvp dot_suffix_triangles =
            dfr_suffix_triangle_jvp_from_vertices(m_input_.suffix_vertices, m_input_.suffix_faces, dot_suffix_vertices,
                                                  triangle_width);
        const Vector3f& dot_suffix_tri_p0 = dot_suffix_triangles.p0;
        const Vector3f& dot_suffix_tri_face_normal = dot_suffix_triangles.face_normal;

        drjit::eval(dot_initial_edge_pos, dot_initial_edge_dir, dot_initial_edge_t_min, dot_initial_edge_t_max,
                    dot_initial_src, dot_initial_src_power, dot_initial_exterior, dot_recursive_edge_pos,
                    dot_recursive_edge_dir, dot_recursive_edge_t_min, dot_recursive_edge_t_max, dot_recursive_exterior,
                    dot_material_gain, dot_suffix_vertices, dot_suffix_tri_p0, dot_suffix_tri_face_normal, output.power,
                    output.field_x.x());

        DfrChainAccumADParams params = base_ad_params();
        params.dot_state_edge_pos_x = dot_initial_edge_pos.x().data();
        params.dot_state_edge_pos_y = dot_initial_edge_pos.y().data();
        params.dot_state_edge_pos_z = dot_initial_edge_pos.z().data();
        params.dot_state_edge_dir_x = dot_initial_edge_dir.x().data();
        params.dot_state_edge_dir_y = dot_initial_edge_dir.y().data();
        params.dot_state_edge_dir_z = dot_initial_edge_dir.z().data();
        params.dot_state_edge_t_min = dot_initial_edge_t_min.data();
        params.dot_state_edge_t_max = dot_initial_edge_t_max.data();
        params.dot_state_src_x = dot_initial_src.x().data();
        params.dot_state_src_y = dot_initial_src.y().data();
        params.dot_state_src_z = dot_initial_src.z().data();
        params.dot_state_src_power = dot_initial_src_power.data();
        params.dot_state_exterior_angle = dot_initial_exterior.data();
        params.dot_recursive_state_edge_pos_x = dot_recursive_edge_pos.x().data();
        params.dot_recursive_state_edge_pos_y = dot_recursive_edge_pos.y().data();
        params.dot_recursive_state_edge_pos_z = dot_recursive_edge_pos.z().data();
        params.dot_recursive_state_edge_dir_x = dot_recursive_edge_dir.x().data();
        params.dot_recursive_state_edge_dir_y = dot_recursive_edge_dir.y().data();
        params.dot_recursive_state_edge_dir_z = dot_recursive_edge_dir.z().data();
        params.dot_recursive_state_edge_t_min = dot_recursive_edge_t_min.data();
        params.dot_recursive_state_edge_t_max = dot_recursive_edge_t_max.data();
        params.dot_recursive_state_exterior_angle = dot_recursive_exterior.data();
        params.dot_material_gain = dot_material_gain.data();
        params.dot_tri_p0_x = dot_suffix_tri_p0.x().data();
        params.dot_tri_p0_y = dot_suffix_tri_p0.y().data();
        params.dot_tri_p0_z = dot_suffix_tri_p0.z().data();
        params.dot_tri_fn_x = dot_suffix_tri_face_normal.x().data();
        params.dot_tri_fn_y = dot_suffix_tri_face_normal.y().data();
        params.dot_tri_fn_z = dot_suffix_tri_face_normal.z().data();
        params.dot_out_power = output.power.data();
        params.dot_out_field_x_re = output.field_x.x().data();
        dfr_chain_accum_jvp_gpu(params);
        set_dfr_accum_output_grad(this->registered_output, output);
    }

    void backward() override {
        if (m_tape_.launch_count <= 0) {
            return;
        }

        const size_t initial_width = slices(m_input_.initial_states.edge_index);
        const size_t recursive_width = slices(m_input_.recursive_states.edge_index);
        const size_t material_width = slices(m_input_.material.gain);
        const size_t triangle_width = slices(m_input_.suffix_tri_p0);
        const size_t vertex_width = slices(m_input_.suffix_vertices);
        Vector3f grad_initial_edge_pos = zeros<Vector3f>(initial_width);
        Vector3f grad_initial_edge_dir = zeros<Vector3f>(initial_width);
        Float grad_initial_edge_t_min = zeros<Float>(initial_width);
        Float grad_initial_edge_t_max = zeros<Float>(initial_width);
        Vector3f grad_initial_src = zeros<Vector3f>(initial_width);
        Float grad_initial_src_power = zeros<Float>(initial_width);
        Float grad_initial_exterior = zeros<Float>(initial_width);
        Vector3f grad_recursive_edge_pos = zeros<Vector3f>(recursive_width);
        Vector3f grad_recursive_edge_dir = zeros<Vector3f>(recursive_width);
        Float grad_recursive_edge_t_min = zeros<Float>(recursive_width);
        Float grad_recursive_edge_t_max = zeros<Float>(recursive_width);
        Float grad_recursive_exterior = zeros<Float>(recursive_width);
        Float grad_material_gain = zeros<Float>(material_width);
        Vector3f grad_suffix_tri_p0 = zeros<Vector3f>(triangle_width);
        Vector3f grad_suffix_tri_face_normal = zeros<Vector3f>(triangle_width);
        Vector3f grad_suffix_vertices = zeros<Vector3f>(vertex_width);
        Float grad_power =
            coerce_float_grad(drjit::grad<false>(this->registered_output.power), grid_.resolution0 * grid_.resolution1);
        Float grad_field_x_re = coerce_float_grad(drjit::grad<false>(this->registered_output.field_x.x()),
                                                  grid_.resolution0 * grid_.resolution1);

        drjit::eval(grad_initial_edge_pos, grad_initial_edge_dir, grad_initial_edge_t_min, grad_initial_edge_t_max,
                    grad_initial_src, grad_initial_src_power, grad_initial_exterior, grad_recursive_edge_pos,
                    grad_recursive_edge_dir, grad_recursive_edge_t_min, grad_recursive_edge_t_max,
                    grad_recursive_exterior, grad_material_gain, grad_suffix_tri_p0, grad_suffix_tri_face_normal,
                    grad_suffix_vertices, grad_power, grad_field_x_re);

        DfrChainAccumADParams params = base_ad_params();
        params.grad_out_power = grad_power.data();
        params.grad_out_field_x_re = grad_field_x_re.data();
        params.grad_state_edge_pos_x = grad_initial_edge_pos.x().data();
        params.grad_state_edge_pos_y = grad_initial_edge_pos.y().data();
        params.grad_state_edge_pos_z = grad_initial_edge_pos.z().data();
        params.grad_state_edge_dir_x = grad_initial_edge_dir.x().data();
        params.grad_state_edge_dir_y = grad_initial_edge_dir.y().data();
        params.grad_state_edge_dir_z = grad_initial_edge_dir.z().data();
        params.grad_state_edge_t_min = grad_initial_edge_t_min.data();
        params.grad_state_edge_t_max = grad_initial_edge_t_max.data();
        params.grad_state_src_x = grad_initial_src.x().data();
        params.grad_state_src_y = grad_initial_src.y().data();
        params.grad_state_src_z = grad_initial_src.z().data();
        params.grad_state_src_power = grad_initial_src_power.data();
        params.grad_state_exterior_angle = grad_initial_exterior.data();
        params.grad_recursive_state_edge_pos_x = grad_recursive_edge_pos.x().data();
        params.grad_recursive_state_edge_pos_y = grad_recursive_edge_pos.y().data();
        params.grad_recursive_state_edge_pos_z = grad_recursive_edge_pos.z().data();
        params.grad_recursive_state_edge_dir_x = grad_recursive_edge_dir.x().data();
        params.grad_recursive_state_edge_dir_y = grad_recursive_edge_dir.y().data();
        params.grad_recursive_state_edge_dir_z = grad_recursive_edge_dir.z().data();
        params.grad_recursive_state_edge_t_min = grad_recursive_edge_t_min.data();
        params.grad_recursive_state_edge_t_max = grad_recursive_edge_t_max.data();
        params.grad_recursive_state_exterior_angle = grad_recursive_exterior.data();
        params.grad_material_gain = grad_material_gain.data();
        params.grad_tri_p0_x = grad_suffix_tri_p0.x().data();
        params.grad_tri_p0_y = grad_suffix_tri_p0.y().data();
        params.grad_tri_p0_z = grad_suffix_tri_p0.z().data();
        params.grad_tri_fn_x = grad_suffix_tri_face_normal.x().data();
        params.grad_tri_fn_y = grad_suffix_tri_face_normal.y().data();
        params.grad_tri_fn_z = grad_suffix_tri_face_normal.z().data();
        dfr_chain_accum_vjp_gpu(params);
        grad_suffix_vertices =
            dfr_suffix_triangle_vertex_vjp(m_input_.suffix_vertices, m_input_.suffix_faces, grad_suffix_tri_p0,
                                           grad_suffix_tri_face_normal, vertex_width);
        drjit::eval(grad_suffix_vertices);

        drjit::accum_grad(this->registered_input.initial_states.edge_pos, drjit::detach<false>(grad_initial_edge_pos));
        drjit::accum_grad(this->registered_input.initial_states.edge_dir, drjit::detach<false>(grad_initial_edge_dir));
        drjit::accum_grad(this->registered_input.initial_states.edge_t_min,
                          drjit::detach<false>(grad_initial_edge_t_min));
        drjit::accum_grad(this->registered_input.initial_states.edge_t_max,
                          drjit::detach<false>(grad_initial_edge_t_max));
        drjit::accum_grad(this->registered_input.initial_states.src, drjit::detach<false>(grad_initial_src));
        drjit::accum_grad(this->registered_input.initial_states.src_power,
                          drjit::detach<false>(grad_initial_src_power));
        drjit::accum_grad(this->registered_input.initial_states.exterior_angle,
                          drjit::detach<false>(grad_initial_exterior));
        drjit::accum_grad(this->registered_input.recursive_states.edge_pos,
                          drjit::detach<false>(grad_recursive_edge_pos));
        drjit::accum_grad(this->registered_input.recursive_states.edge_dir,
                          drjit::detach<false>(grad_recursive_edge_dir));
        drjit::accum_grad(this->registered_input.recursive_states.edge_t_min,
                          drjit::detach<false>(grad_recursive_edge_t_min));
        drjit::accum_grad(this->registered_input.recursive_states.edge_t_max,
                          drjit::detach<false>(grad_recursive_edge_t_max));
        drjit::accum_grad(this->registered_input.recursive_states.exterior_angle,
                          drjit::detach<false>(grad_recursive_exterior));
        drjit::accum_grad(this->registered_input.material.gain, drjit::detach<false>(grad_material_gain));
        drjit::accum_grad(this->registered_input.suffix_vertices, drjit::detach<false>(grad_suffix_vertices));
    }

    const char* name() const override { return "DfrChainAccum"; }

  private:
    DfrChainAccumADParams base_ad_params() const {
        DfrChainAccumADParams params = {};
        params.n_rays = m_tape_.launch_count;
        params.state_count = dfr_state_count_for(m_input_.initial_states);
        params.recursive_state_count = dfr_state_count_for(m_input_.recursive_states);
        params.material_count = static_cast<int>(slices(m_input_.material.gain));
        params.grid_axis = grid_.axis;
        params.grid_position = grid_.position;
        params.grid_coord0_min = grid_.coord0_min;
        params.grid_coord0_max = grid_.coord0_max;
        params.grid_coord1_min = grid_.coord1_min;
        params.grid_coord1_max = grid_.coord1_max;
        params.grid_resolution0 = grid_.resolution0;
        params.grid_resolution1 = grid_.resolution1;
        params.grid_cell_area = grid_.cell_area;
        params.direct_samples = dfr_direct_sample_count(options_);
        params.keller_samples = dfr_keller_sample_count(options_);
        params.suffix_samples = dfr_suffix_sample_count(options_);
        params.max_order = options_.max_order;
        params.wavelength = options_.wavelength;
        params.seed = options_.seed;
        const TriangleInfo& triangles = scene_->triangle_info_detached();
        const bool suffix_enabled = params.suffix_samples > 0;
        params.n_triangles = suffix_enabled ? static_cast<int>(slices(m_input_.suffix_tri_p0)) : 0;
        params.tape_active = reinterpret_cast<const uint8_t*>(m_tape_.active.data());
        params.tape_cell = m_tape_.cell.data();
        params.state_edge_index = m_input_.initial_states.edge_index.data();
        params.state_edge_pos_x = m_input_.initial_states.edge_pos.x().data();
        params.state_edge_pos_y = m_input_.initial_states.edge_pos.y().data();
        params.state_edge_pos_z = m_input_.initial_states.edge_pos.z().data();
        params.state_edge_dir_x = m_input_.initial_states.edge_dir.x().data();
        params.state_edge_dir_y = m_input_.initial_states.edge_dir.y().data();
        params.state_edge_dir_z = m_input_.initial_states.edge_dir.z().data();
        params.state_edge_t_min = m_input_.initial_states.edge_t_min.data();
        params.state_edge_t_max = m_input_.initial_states.edge_t_max.data();
        params.state_src_x = m_input_.initial_states.src.x().data();
        params.state_src_y = m_input_.initial_states.src.y().data();
        params.state_src_z = m_input_.initial_states.src.z().data();
        params.state_src_power = m_input_.initial_states.src_power.data();
        params.state_exterior_angle = m_input_.initial_states.exterior_angle.data();
        params.state_prim0 = m_input_.initial_states.prim0.data();
        params.state_prim1 = m_input_.initial_states.prim1.data();
        params.recursive_state_edge_index = m_input_.recursive_states.edge_index.data();
        params.recursive_state_edge_pos_x = m_input_.recursive_states.edge_pos.x().data();
        params.recursive_state_edge_pos_y = m_input_.recursive_states.edge_pos.y().data();
        params.recursive_state_edge_pos_z = m_input_.recursive_states.edge_pos.z().data();
        params.recursive_state_edge_dir_x = m_input_.recursive_states.edge_dir.x().data();
        params.recursive_state_edge_dir_y = m_input_.recursive_states.edge_dir.y().data();
        params.recursive_state_edge_dir_z = m_input_.recursive_states.edge_dir.z().data();
        params.recursive_state_edge_t_min = m_input_.recursive_states.edge_t_min.data();
        params.recursive_state_edge_t_max = m_input_.recursive_states.edge_t_max.data();
        params.recursive_state_exterior_angle = m_input_.recursive_states.exterior_angle.data();
        params.recursive_state_prim0 = m_input_.recursive_states.prim0.data();
        params.recursive_state_prim1 = m_input_.recursive_states.prim1.data();
        params.tri_p0_x = suffix_enabled ? m_input_.suffix_tri_p0.x().data() : nullptr;
        params.tri_p0_y = suffix_enabled ? m_input_.suffix_tri_p0.y().data() : nullptr;
        params.tri_p0_z = suffix_enabled ? m_input_.suffix_tri_p0.z().data() : nullptr;
        params.tri_e1_x = suffix_enabled ? triangles.e1.x().data() : nullptr;
        params.tri_e1_y = suffix_enabled ? triangles.e1.y().data() : nullptr;
        params.tri_e1_z = suffix_enabled ? triangles.e1.z().data() : nullptr;
        params.tri_e2_x = suffix_enabled ? triangles.e2.x().data() : nullptr;
        params.tri_e2_y = suffix_enabled ? triangles.e2.y().data() : nullptr;
        params.tri_e2_z = suffix_enabled ? triangles.e2.z().data() : nullptr;
        params.tri_fn_x = suffix_enabled ? m_input_.suffix_tri_face_normal.x().data() : nullptr;
        params.tri_fn_y = suffix_enabled ? m_input_.suffix_tri_face_normal.y().data() : nullptr;
        params.tri_fn_z = suffix_enabled ? m_input_.suffix_tri_face_normal.z().data() : nullptr;
        params.material_gain = m_input_.material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t*>(m_input_.material.valid.data());
        return params;
    }

    const Scene* scene_ = nullptr;
    DfrGrid grid_;
    DfrOptions options_;
    DfrChainAccumOpInputDetached m_input_;
    DfrDirectTapeCapture m_tape_;
};

DfrAccumAD dfr_chain_accum_custom_op_impl(const Scene* scene, const DfrStatesAD& initial_states,
                                          const DfrStatesAD& recursive_states, const DfrGrid& grid,
                                          const DfrMaterialAD& material, const DfrOptions& options,
                                          const Vector3fAD& suffix_tri_p0, const Vector3fAD& suffix_tri_face_normal,
                                          const Vector3fAD& suffix_vertices, const Vector3i& suffix_faces,
                                          const MaskAD& active) {
    DfrChainAccumOpInput input;
    input.initial_states = initial_states;
    input.recursive_states = recursive_states;
    input.material = material;
    input.active = active;
    input.suffix_tri_p0 = suffix_tri_p0;
    input.suffix_tri_face_normal = suffix_tri_face_normal;
    input.suffix_vertices = suffix_vertices;
    input.suffix_faces = suffix_faces;
    nb::ref<DfrChainAccumOp> op = new DfrChainAccumOp(input, scene, grid, options);
    DfrAccumAD output = op->eval(detach_dfr_chain_input(input));
    drjit::detail::new_grad(output);
    op->register_output(output);
    if (!ad_custom_op(op.get())) {
        drjit::disable_grad(output);
    }
    return output;
}

} // namespace

void require_dfr_direct_custom_ad_supported(const DfrOptions& options) {
    require_dfr_direct_custom_ad_supported_impl(options);
}

void require_dfr_chain_custom_ad_supported(const DfrOptions& options) {
    require_dfr_chain_custom_ad_supported_impl(options);
}

DfrAccumAD dfr_direct_accum_custom_op(const Scene* scene, const DfrStatesAD& states, const DfrGrid& grid,
                                      const DfrMaterialAD& material, const DfrOptions& options,
                                      const Vector3fAD& suffix_tri_p0, const Vector3fAD& suffix_tri_face_normal,
                                      const Vector3fAD& suffix_vertices, const Vector3i& suffix_faces,
                                      const MaskAD& active) {
    return dfr_direct_accum_custom_op_impl(scene, states, grid, material, options, suffix_tri_p0,
                                           suffix_tri_face_normal, suffix_vertices, suffix_faces, active);
}

DfrAccumAD dfr_chain_accum_custom_op(const Scene* scene, const DfrStatesAD& initial_states,
                                     const DfrStatesAD& recursive_states, const DfrGrid& grid,
                                     const DfrMaterialAD& material, const DfrOptions& options,
                                     const Vector3fAD& suffix_tri_p0, const Vector3fAD& suffix_tri_face_normal,
                                     const Vector3fAD& suffix_vertices, const Vector3i& suffix_faces,
                                     const MaskAD& active) {
    return dfr_chain_accum_custom_op_impl(scene, initial_states, recursive_states, grid, material, options,
                                          suffix_tri_p0, suffix_tri_face_normal, suffix_vertices, suffix_faces, active);
}

} // namespace rayd
