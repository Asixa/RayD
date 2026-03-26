#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <rayd/transform.h>
#include <rayd/mesh.h>

namespace rayd {

template <bool Detached>
static std::pair<TriangleInfoT<Detached>, Vector3fT<Detached>> process_mesh(const Vector3fT<Detached> &vertex_positions,
                                                                             const Vector3iT<Detached> &face_indices) {
    const int vertex_count = static_cast<int>(slices<Vector3fT<Detached>>(vertex_positions));

    TriangleInfoT<Detached> triangles;
    triangles.face_indices = face_indices;
    triangles.p0 = gather<Vector3fT<Detached>>(vertex_positions, face_indices[0]);
    triangles.e1 = gather<Vector3fT<Detached>>(vertex_positions, face_indices[1]) - triangles.p0;
    triangles.e2 = gather<Vector3fT<Detached>>(vertex_positions, face_indices[2]) - triangles.p0;

    Vector3fT<Detached> raw_face_normals = cross(triangles.e1, triangles.e2);
    FloatT<Detached> raw_face_areas = norm(raw_face_normals);
    MaskT<Detached> valid_faces = raw_face_areas > Epsilon;

    Vector3fT<Detached> &face_normals = triangles.face_normal;
    FloatT<Detached> &face_areas = triangles.face_area;
    face_normals = select(valid_faces,
                          raw_face_normals / raw_face_areas,
                          Vector3fT<Detached>(0.f, 0.f, 1.f));
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
    vertex_normals = select(vertex_weights > Epsilon,
                            normalize(vertex_normals),
                            Vector3fT<Detached>(0.f, 0.f, 1.f));
    triangles.n0 = gather<Vector3fT<Detached>>(vertex_normals, face_indices[0]);
    triangles.n1 = gather<Vector3fT<Detached>>(vertex_normals, face_indices[1]);
    triangles.n2 = gather<Vector3fT<Detached>>(vertex_normals, face_indices[2]);

    face_areas *= 0.5f;

    drjit::eval(triangles, vertex_normals);
    return { triangles, vertex_normals };
}

template <bool Detached>
static Vector3fT<Detached> safe_normalize(const Vector3fT<Detached> &value) {
    const FloatT<Detached> length_sq = squared_norm(value);
    const MaskT<Detached> valid = length_sq > Epsilon * Epsilon;
    const FloatT<Detached> safe_length = sqrt(select(valid, length_sq, FloatT<Detached>(1.f)));
    return select(valid,
                  value / safe_length,
                  Vector3fT<Detached>(0.f, 0.f, 1.f));
}

static TriangleInfo transform_triangle_info(const TriangleInfo &triangle_info_object,
                                            const Matrix4f &to_world_matrix) {
    TriangleInfo triangles_world;
    triangles_world.face_indices = triangle_info_object.face_indices;
    triangles_world.p0 = transform_pos(to_world_matrix, triangle_info_object.p0);
    triangles_world.e1 = transform_dir(to_world_matrix, triangle_info_object.e1);
    triangles_world.e2 = transform_dir(to_world_matrix, triangle_info_object.e2);

    const Matrix4f normal_to_world = transpose(inverse(to_world_matrix));
    triangles_world.n0 = safe_normalize<false>(transform_dir(normal_to_world, triangle_info_object.n0));
    triangles_world.n1 = safe_normalize<false>(transform_dir(normal_to_world, triangle_info_object.n1));
    triangles_world.n2 = safe_normalize<false>(transform_dir(normal_to_world, triangle_info_object.n2));

    const Vector3f raw_face_normals = cross(triangles_world.e1, triangles_world.e2);
    const Float raw_face_areas = norm(raw_face_normals);
    const Mask valid_faces = raw_face_areas > Epsilon;
    triangles_world.face_normal = select(valid_faces,
                                         raw_face_normals / raw_face_areas,
                                         Vector3f(0.f, 0.f, 1.f));
    triangles_world.face_area = raw_face_areas * 0.5f;
    return triangles_world;
}

Mesh::Mesh(const Mesh &other)
    : mesh_id_(other.mesh_id_),
      is_ready_(false),
      use_face_normals_(other.use_face_normals_),
      has_uv_(other.has_uv_),
      edges_enabled_(other.edges_enabled_),
      object_to_world_(other.object_to_world_),
      left_transform_(other.left_transform_),
      right_transform_(other.right_transform_),
      vertex_count_(other.vertex_count_),
      face_count_(other.face_count_),
      vertex_positions_object_(other.vertex_positions_object_),
      vertex_normals_object_(other.vertex_normals_object_),
      vertex_uv_(other.vertex_uv_),
      vertex_positions_world_(other.vertex_positions_world_),
      face_vertex_indices_(other.face_vertex_indices_),
      face_uv_indices_(other.face_uv_indices_),
      edge_indices_(other.edge_indices_),
      world_positions_dirty_(other.world_positions_dirty_),
      secondary_edge_info_dirty_(other.secondary_edge_info_dirty_),
      optix_face_buffer_dirty_(other.optix_face_buffer_dirty_),
      optix_vertex_buffer_(other.optix_vertex_buffer_),
      optix_face_buffer_(other.optix_face_buffer_) {
    if (other.triangle_info_object_) {
        triangle_info_object_ = std::make_unique<TriangleInfo>(*other.triangle_info_object_);
    }
    if (other.triangle_info_) {
        triangle_info_ = std::make_unique<TriangleInfo>(*other.triangle_info_);
    }
    if (other.triangle_uv_) {
        triangle_uv_ = std::make_unique<TriangleUV>(*other.triangle_uv_);
    }
    if (other.secondary_edge_info_) {
        secondary_edge_info_ = std::make_unique<SecondaryEdgeInfo>(*other.secondary_edge_info_);
    }
}

Mesh::Mesh(const Vector3fDetached &vertex_positions,
           const Vector3iDetached &face_indices,
           const Vector2fDetached &vertex_uv,
           const Vector3iDetached &face_uv_indices,
           bool verbose) {
    init(vertex_positions, face_indices, vertex_uv, face_uv_indices, verbose);
}

Mesh::~Mesh() = default;

const Vector3f &Mesh::vertex_positions_world() const {
    ensure_world_positions_ready();
    return vertex_positions_world_;
}

const SecondaryEdgeInfo *Mesh::secondary_edge_info() const {
    ensure_secondary_edge_info_ready();
    return secondary_edge_info_.get();
}

void Mesh::set_transform(const Matrix4f &matrix, bool set_left) {
    if (set_left) {
        left_transform_ = matrix;
    } else {
        right_transform_ = matrix;
    }
    world_positions_dirty_ = true;
    secondary_edge_info_dirty_ = true;
    is_ready_ = false;
}

void Mesh::append_transform(const Matrix4f &matrix, bool append_left) {
    if (append_left) {
        left_transform_ = matrix * left_transform_;
    } else {
        right_transform_ *= matrix;
    }
    world_positions_dirty_ = true;
    secondary_edge_info_dirty_ = true;
    is_ready_ = false;
}

void Mesh::init(const Vector3fDetached &vertex_positions,
                const Vector3iDetached &face_indices,
                const Vector2fDetached &vertex_uv,
                const Vector3iDetached &face_uv_indices,
                bool verbose) {
    using namespace std::chrono;

    vertex_count_ = static_cast<int>(slices(vertex_positions));
    face_count_ = static_cast<int>(slices(face_indices));
    vertex_positions_object_ = Vector3f(vertex_positions);
    face_vertex_indices_ = Vector3i(face_indices);
    has_uv_ = slices(vertex_uv) > 0;
    if (has_uv_) {
        vertex_uv_ = Vector2f(vertex_uv);
        const size_t face_uv_count = slices(face_uv_indices);
        if (face_uv_count == 0) {
            require(slices(vertex_uv) == static_cast<size_t>(vertex_count_),
                    "Mesh::init(): UV count must match vertex count when face_uv_indices are omitted.");
            face_uv_indices_ = Vector3i(face_indices);
        } else {
            require(face_uv_count == static_cast<size_t>(face_count_),
                    "Mesh::init(): face_uv_indices must match the number of faces.");
            face_uv_indices_ = Vector3i(face_uv_indices);
        }
    } else {
        vertex_uv_ = Vector2f();
        face_uv_indices_ = Vector3i();
    }

    std::array<std::vector<int>, 3> face_indices_cpu;
    copy_cuda_array(face_indices, face_indices_cpu);
    drjit::eval();
    drjit::sync_thread();

    const auto start_time = high_resolution_clock::now();
    int edge_count = 0;
    if (edges_enabled_) {
        std::array<std::vector<int>, 5> edge_records;
        for (auto &record : edge_records) {
            record.reserve(3 * face_count_);
        }

        std::map<std::pair<int, int>, std::vector<int>> edge_map;
        for (int face_index = 0; face_index < face_count_; ++face_index) {
            for (int local_edge = 0; local_edge < 3; ++local_edge) {
                const int start_corner = local_edge;
                const int end_corner = (local_edge + 1) % 3;
                const int opposite_corner = (local_edge + 2) % 3;

                const int start_vertex = face_indices_cpu[start_corner][face_index];
                const int end_vertex = face_indices_cpu[end_corner][face_index];
                const int opposite_vertex = face_indices_cpu[opposite_corner][face_index];
                const auto edge_key = start_vertex < end_vertex
                    ? std::make_pair(start_vertex, end_vertex)
                    : std::make_pair(end_vertex, start_vertex);

                auto edge_entry = edge_map.find(edge_key);
                if (edge_entry == edge_map.end()) {
                    edge_entry = edge_map.emplace(edge_key, std::vector<int>{}).first;
                    edge_entry->second.push_back(opposite_vertex);
                }
                edge_entry->second.push_back(face_index);
            }
        }

        for (const auto &[edge_vertices, adjacency] : edge_map) {
            edge_records[0].push_back(edge_vertices.first);
            edge_records[1].push_back(edge_vertices.second);
            edge_records[2].push_back(adjacency[1]);
            if (adjacency.size() >= 3) {
                edge_records[3].push_back(adjacency[2]);
                edge_records[4].push_back(adjacency[0]);
            } else {
                edge_records[3].push_back(-1);
                edge_records[4].push_back(adjacency[0]);
            }
            ++edge_count;
        }

        edge_indices_ = VectoriT<5, true>(drjit::load<IntDetached>(edge_records[0].data(), edge_count),
                                          drjit::load<IntDetached>(edge_records[1].data(), edge_count),
                                          drjit::load<IntDetached>(edge_records[2].data(), edge_count),
                                          drjit::load<IntDetached>(edge_records[3].data(), edge_count),
                                          drjit::load<IntDetached>(edge_records[4].data(), edge_count));
    } else {
        edge_indices_ = VectoriT<5, true>(IntDetached(), IntDetached(), IntDetached(), IntDetached(), IntDetached());
    }

    const auto end_time = high_resolution_clock::now();
    const double seconds = duration_cast<duration<double>>(end_time - start_time).count();
    if (verbose) {
        std::cout << "Loaded " << vertex_count_ << " vertices, "
                  << face_count_ << " faces, "
                  << edge_count << " edges in " << seconds << " seconds." << std::endl;
    }

    optix_face_buffer_dirty_ = true;
    is_ready_ = false;
}

void Mesh::build() {
    require(vertex_count_ > 0, "Mesh::build(): mesh has no vertices.");
    require(face_count_ > 0, "Mesh::build(): mesh has no faces.");

    if (!triangle_info_object_) {
        triangle_info_object_ = std::make_unique<TriangleInfo>();
    }
    std::tie(*triangle_info_object_, vertex_normals_object_) =
        process_mesh<false>(vertex_positions_object_, face_vertex_indices_);

    if (!triangle_info_) {
        triangle_info_ = std::make_unique<TriangleInfo>();
    }
    update_world_triangle_info();

    triangle_uv_.reset();
    if (has_uv_) {
        triangle_uv_ = std::make_unique<TriangleUV>();
        for (int corner = 0; corner < 3; ++corner) {
            (*triangle_uv_)[corner] = gather<Vector2f>(vertex_uv_, face_uv_indices_[corner]);
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
        drjit::eval(optix_vertex_buffer_);
        drjit::sync_thread();
    }
}

void Mesh::update_world_triangle_info() {
    require(triangle_info_object_ != nullptr,
            "Mesh::update_world_triangle_info(): object-space geometry must be built first.");
    require(triangle_info_ != nullptr,
            "Mesh::update_world_triangle_info(): world-space triangle cache must be allocated first.");

    const Matrix4f to_world_matrix = left_transform_ * object_to_world_ * right_transform_;
    *triangle_info_ = transform_triangle_info(*triangle_info_object_, to_world_matrix);
}

void Mesh::update_secondary_edge_info() {
    ensure_world_positions_ready();

    if (!edges_enabled_) {
        secondary_edge_info_.reset();
        secondary_edge_info_dirty_ = false;
        return;
    }

    if (!secondary_edge_info_) {
        secondary_edge_info_ = std::make_unique<SecondaryEdgeInfo>();
    }

    SecondaryEdgeInfo secondary_edges;
    const Int edge_vertex_0 = Int(edge_indices_[0]);
    const Int edge_vertex_1 = Int(edge_indices_[1]);
    const Int face_index_0 = Int(edge_indices_[2]);
    const Int face_index_1 = Int(edge_indices_[3]);
    const Int opposite_vertex = Int(edge_indices_[4]);
    const Mask is_boundary = face_index_1 < 0;

    secondary_edges.is_boundary = is_boundary;
    secondary_edges.start = gather<Vector3f>(vertex_positions_world_, edge_vertex_0);
    secondary_edges.edge = gather<Vector3f>(vertex_positions_world_, edge_vertex_1) - secondary_edges.start;
    secondary_edges.normal0 = gather<Vector3f>(triangle_info_->face_normal, face_index_0);
    secondary_edges.normal1 = gather<Vector3f>(triangle_info_->face_normal, face_index_1, ~is_boundary);
    secondary_edges.opposite = gather<Vector3f>(vertex_positions_world_, opposite_vertex);
    *secondary_edge_info_ = secondary_edges;
    secondary_edge_info_dirty_ = false;
}

void Mesh::ensure_world_positions_ready() const {
    if (!world_positions_dirty_) {
        return;
    }

    const Matrix4f to_world_matrix = left_transform_ * object_to_world_ * right_transform_;
    vertex_positions_world_ = transform_pos(to_world_matrix, vertex_positions_object_);
    world_positions_dirty_ = false;
}

void Mesh::ensure_secondary_edge_info_ready() const {
    if (!secondary_edge_info_dirty_) {
        return;
    }

    const_cast<Mesh *>(this)->update_secondary_edge_info();
}

void Mesh::prepare_optix_buffers() {
    require(is_ready_, "Mesh::prepare_optix_buffers(): mesh must be built first.");

    update_optix_vertex_buffer();
    ensure_optix_face_buffer_ready();
}

void Mesh::update_optix_vertex_buffer() {
    const int scalar_count = vertex_count_ * 3;
    if (slices(optix_vertex_buffer_) != static_cast<size_t>(scalar_count)) {
        optix_vertex_buffer_ = empty<FloatDetached>(scalar_count);
    }

    const IntDetached indices = arange<IntDetached>(vertex_count_) * 3;
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
        optix_face_buffer_ = empty<IntDetached>(scalar_count);
    }

    const IntDetached indices = arange<IntDetached>(face_count_) * 3;
    for (int axis = 0; axis < 3; ++axis) {
        scatter(optix_face_buffer_, detach<false>(face_vertex_indices_[axis]), indices + axis);
    }
    optix_face_buffer_dirty_ = false;
}

std::string Mesh::to_string() const {
    std::stringstream stream;
    stream << "Mesh[nv=" << vertex_count_
           << ", nf=" << face_count_
           << ", has_uv=" << has_uv_
           << ", edges=" << edges_enabled_
           << "]";
    return stream.str();
}

} // namespace rayd
