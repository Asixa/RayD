#include <sstream>
#include <vector>

#include <rayd/ray.h>
#include <rayd/transform.h>
#include <rayd/intersection.h>
#include <rayd/mesh.h>
#include <rayd/camera.h>
#include <rayd/scene/scene.h>

namespace rayd {

namespace {

Vector2f make_pixel_centers(int width, int height) {
    const int pixel_count = width * height;
    const Int index = arange<Int>(pixel_count);
    const Float x = (Float(index % width) + 0.5f) / static_cast<float>(width);
    const Float y = (Float(index / width) + 0.5f) / static_cast<float>(height);
    return Vector2f(x, y);
}

FloatDetached compute_cdf(const FloatDetached &pmf) {
    const size_t size = pmf.size();
    const ScalarFloat *pmf_ptr = pmf.data();

    require(size > 0, "PerspectiveCamera::DiscreteDistribution::init(): empty distribution.");

    std::vector<ScalarFloat> cdf(size);
    double sum = 0.0;
    for (uint32_t index = 0; index < size; ++index) {
        const double value = static_cast<double>(*pmf_ptr++);
        require(value >= 0.0,
                "PerspectiveCamera::DiscreteDistribution::init(): entries must be non-negative.");

        sum += value;
        cdf[index] = static_cast<ScalarFloat>(sum);
    }

    return drjit::load<FloatDetached>(cdf.data(), size);
}

Float render_edge_grad_flat(const Scene &scene,
                            const PerspectiveCamera &camera,
                            int spp,
                            float background) {
    require(spp > 0, "PerspectiveCamera::render_grad(): spp must be positive.");
    require(camera.width() > 0 && camera.height() > 0,
            "PerspectiveCamera::render_grad(): camera resolution must be positive.");

    const int sample_count = camera.width() * camera.height() * spp;
    const FloatDetached samples =
        (arange<FloatDetached>(sample_count) + 0.5f) / static_cast<float>(sample_count);

    PerspectiveCamera prepared_camera(camera);
    prepared_camera.prepare_primary_edges(scene);

    const PrimaryEdgeSample edge_samples = prepared_camera.sample_primary_edge(samples);
    const MaskDetached valid_detached = edge_samples.idx >= 0;
    const Mask valid = Mask(valid_detached);

    const IntersectionDetached its_n =
        scene.intersect<true>(edge_samples.ray_n, valid_detached);
    const IntersectionDetached its_p =
        scene.intersect<true>(edge_samples.ray_p, valid_detached);

    const FloatDetached background_depth = full<FloatDetached>(background, sample_count);
    const FloatDetached depth_n = select(its_n.is_valid(), its_n.t, background_depth);
    const FloatDetached depth_p = select(its_p.is_valid(), its_p.t, background_depth);

    Float value = edge_samples.x_dot_n * Float((depth_n - depth_p) / edge_samples.pdf);
    value = select(valid, value, zeros<Float>(sample_count));
    value /= static_cast<float>(spp);
    value -= Float(detach<false>(value));

    Float image = zeros<Float>(camera.width() * camera.height());
    scatter_reduce(ReduceOp::Add, image, value, Int(edge_samples.idx), valid);
    return image;
}

} // namespace

PerspectiveCamera::PerspectiveCamera(float fov_x, float near_clip, float far_clip)
    : field_of_view_x_(fov_x),
      near_clip_(near_clip),
      far_clip_(far_clip),
      uses_intrinsics_(false) {}

PerspectiveCamera::PerspectiveCamera(float fx, float fy, float cx, float cy,
                                     float near_clip, float far_clip)
    : focal_length_x_(fx),
      focal_length_y_(fy),
      principal_point_x_(cx),
      principal_point_y_(cy),
      near_clip_(near_clip),
      far_clip_(far_clip),
      uses_intrinsics_(true) {}

PerspectiveCamera::~PerspectiveCamera() {
    if (primary_edge_scene_ != nullptr) {
        const_cast<Scene *>(primary_edge_scene_)->unregister_primary_edge_observer(this);
        primary_edge_scene_ = nullptr;
    }
}

void PerspectiveCamera::DiscreteDistribution::init(const FloatDetached &pmf) {
    size_ = static_cast<int>(pmf.size());
    sum_ = sum(pmf);
    pmf_ = pmf;

    const FloatDetached host_pmf = drjit::migrate(pmf_, AllocType::Host);
    drjit::sync_thread();
    cmf_ = compute_cdf(host_pmf);
    pmf_normalized_ = pmf / sum_;
    cmf_normalized_ = cmf_ / sum_;
}

std::pair<IntDetached, FloatDetached>
PerspectiveCamera::DiscreteDistribution::sample(const FloatDetached &samples) const {
    if (size_ == 1) {
        return { zeros<IntDetached>(), full<FloatDetached>(1.f) };
    }

    const FloatDetached scaled_samples = samples * sum_;
    const IntDetached indices = binary_search<IntDetached>(
        0, size_ - 1,
        [&](IntDetached index) DRJIT_INLINE_LAMBDA {
            return gather<FloatDetached>(cmf_, index) < scaled_samples;
        });
    return { indices, gather<FloatDetached>(pmf_, indices) / sum_ };
}

void PerspectiveCamera::set_transform(const Matrix4f &matrix, bool set_left) {
    if (set_left) {
        left_transform_ = matrix;
    } else {
        right_transform_ = matrix;
    }
    is_ready_ = false;
    primary_edges_ready_ = false;
}

void PerspectiveCamera::append_transform(const Matrix4f &matrix, bool append_left) {
    if (append_left) {
        left_transform_ = matrix * left_transform_;
    } else {
        right_transform_ *= matrix;
    }
    is_ready_ = false;
    primary_edges_ready_ = false;
}

void PerspectiveCamera::configure(bool cache) {
    require(image_width_ > 0 && image_height_ > 0,
            "PerspectiveCamera::configure(): width and height must be positive.");

    aspect_ratio_ = static_cast<float>(image_width_) / static_cast<float>(image_height_);

    ScalarMatrix4f sample_from_camera;
    if (uses_intrinsics_) {
        sample_from_camera =
            transform::scale(ScalarVector3f(-0.5f, -0.5f, 1.f)) *
            transform::translate(ScalarVector3f(-1.f, -1.f, 0.f)) *
            transform::perspective_intrinsic(focal_length_x_, focal_length_y_,
                                             principal_point_x_, principal_point_y_,
                                             near_clip_, far_clip_);
    } else {
        sample_from_camera =
            transform::scale(ScalarVector3f(-0.5f, -0.5f * aspect_ratio_, 1.f)) *
            transform::translate(ScalarVector3f(-1.f, -1.f / aspect_ratio_, 0.f)) *
            transform::perspective(field_of_view_x_, near_clip_, far_clip_);
    }

    camera_to_sample_ = Matrix4f(sample_from_camera);
    sample_to_camera_ = Matrix4f(inverse(sample_from_camera));

    const Matrix4f to_world_matrix = left_transform_ * object_to_world_ * right_transform_;
    world_to_sample_ = camera_to_sample_ * inverse(to_world_matrix);
    sample_to_world_ = to_world_matrix * sample_to_camera_;

    camera_position_ = transform_pos(to_world_matrix, zeros<Vector3f>(1));
    camera_direction_ = transform_dir(to_world_matrix, Vector3f(0.f, 0.f, 1.f));

    const Vector3f sample_corner_00 = transform_pos(sample_to_camera_, Vector3f(0.f, 0.f, 0.f));
    const Vector3f sample_corner_10 = transform_pos(sample_to_camera_, Vector3f(1.f, 0.f, 0.f));
    const Vector3f sample_corner_11 = transform_pos(sample_to_camera_, Vector3f(1.f, 1.f, 0.f));
    const Vector3f sample_center = transform_pos(sample_to_camera_, Vector3f(0.5f, 0.5f, 0.f));
    inverse_sensor_area_ = rcp(norm(sample_corner_00 - sample_corner_10) *
                               norm(sample_corner_11 - sample_corner_10)) *
                           squared_norm(sample_center);

    if (cache) {
        drjit::make_opaque(camera_to_sample_,
                           sample_to_camera_,
                           world_to_sample_,
                           sample_to_world_,
                           camera_position_,
                           camera_direction_,
                           inverse_sensor_area_);
    }

    is_ready_ = true;
    primary_edges_ready_ = false;
}

void PerspectiveCamera::prepare_primary_edges(const Scene &scene) {
    require(is_ready_, "PerspectiveCamera::prepare_primary_edges(): camera is not configured.");
    require(scene.is_ready(), "PerspectiveCamera::prepare_primary_edges(): scene is not configured.");
    require(!scene.has_pending_updates(),
            "PerspectiveCamera::prepare_primary_edges(): scene has pending updates. Call Scene::commit_updates() first.");

    const std::vector<const Mesh *> meshes = scene.meshes();
    std::vector<VectoriT<5, true>> candidate_edges(meshes.size());
    std::vector<int> edge_offsets(1, 0);

    primary_edge_info_ = PrimaryEdgeInfo();
    primary_edges_ready_ = false;

    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
        const Mesh *mesh = meshes[mesh_index];
        const VectoriT<5, true> &mesh_edge_indices = mesh->edge_indices();
        if (!mesh->edges_enabled() || slices(mesh_edge_indices) == 0) {
            edge_offsets.push_back(edge_offsets.back());
            continue;
        }

        auto &visible_edges = candidate_edges[mesh_index];
        const Int face_index_0 = Int(mesh_edge_indices[2]);
        const Int face_index_1 = Int(mesh_edge_indices[3]);
        const Mask has_opposite_face = face_index_1 >= 0;

        const TriangleInfo *triangle_info = mesh->triangle_info();
        const Vector3f view_dir_face_0 =
            normalize(camera_position_ - gather<Vector3f>(triangle_info->p0, face_index_0));
        const Vector3f view_dir_face_1 =
            normalize(camera_position_ - gather<Vector3f>(triangle_info->p0, face_index_1, has_opposite_face));
        const Vector3f face_normal_0 = gather<Vector3f>(triangle_info->face_normal, face_index_0);
        const Vector3f face_normal_1 = gather<Vector3f>(triangle_info->face_normal, face_index_1, has_opposite_face);

        Mask uv_discontinuity = zeros<Mask>(slices(has_opposite_face));
        if (mesh->has_uv()) {
            const Vector3i face_uv_indices_0 = gather<Vector3i>(mesh->face_uv_indices(), face_index_0);
            const Vector3i face_uv_indices_1 = gather<Vector3i>(mesh->face_uv_indices(), face_index_1, has_opposite_face);
            Int shared_uv_count = zeros<Int>(slices(has_opposite_face));

            shared_uv_count = select(eq(face_uv_indices_0[0], face_uv_indices_1[0]) ||
                                     eq(face_uv_indices_0[0], face_uv_indices_1[1]) ||
                                     eq(face_uv_indices_0[0], face_uv_indices_1[2]),
                                     Int(1),
                                     Int(0));
            shared_uv_count = select(eq(face_uv_indices_0[1], face_uv_indices_1[0]) ||
                                     eq(face_uv_indices_0[1], face_uv_indices_1[1]) ||
                                     eq(face_uv_indices_0[1], face_uv_indices_1[2]),
                                     shared_uv_count + 1,
                                     shared_uv_count);
            shared_uv_count = select(eq(face_uv_indices_0[2], face_uv_indices_1[0]) ||
                                     eq(face_uv_indices_0[2], face_uv_indices_1[1]) ||
                                     eq(face_uv_indices_0[2], face_uv_indices_1[2]),
                                     shared_uv_count + 1,
                                     shared_uv_count);
            uv_discontinuity = neq(shared_uv_count, 2);
        }

        if (mesh->use_face_normals()) {
            Mask culled_edges = has_opposite_face;
            culled_edges &= (dot(view_dir_face_0, face_normal_0) < Epsilon &&
                             dot(view_dir_face_1, face_normal_1) < Epsilon) ||
                            (dot(face_normal_0, face_normal_1) > 1.f - Epsilon);
            visible_edges = compressD<VectoriT<5, true>>(mesh_edge_indices,
                                                         detach<false>(mesh->has_uv() ? (~culled_edges || uv_discontinuity)
                                                                                      : ~culled_edges));
        } else {
            Mask silhouette_edges = ~has_opposite_face;
            silhouette_edges |= (dot(view_dir_face_0, face_normal_0) > Epsilon) ^
                                (dot(view_dir_face_1, face_normal_1) > Epsilon);
            visible_edges = compressD<VectoriT<5, true>>(mesh_edge_indices,
                                                         detach<false>(mesh->has_uv() ? (silhouette_edges || uv_discontinuity)
                                                                                      : silhouette_edges));
        }

        edge_offsets.push_back(edge_offsets.back() + static_cast<int>(slices(visible_edges)));
    }

    if (edge_offsets.back() == 0) {
        attach_primary_edge_scene(scene);
        return;
    }

    primary_edge_info_ = empty<PrimaryEdgeInfo>(edge_offsets.back());
    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
        const auto &visible_edges = candidate_edges[mesh_index];
        const int visible_edge_count = static_cast<int>(slices(visible_edges));
        if (visible_edge_count == 0) {
            continue;
        }

        const Mesh *mesh = meshes[mesh_index];
        Int scatter_indices = arange<Int>(visible_edge_count) + edge_offsets[mesh_index];
        const Int edge_vertex_0 = Int(visible_edges[0]);
        const Int edge_vertex_1 = Int(visible_edges[1]);

        const Vector3f edge_start = gather<Vector3f>(mesh->vertex_positions_world(), edge_vertex_0);
        const Vector3f edge_end = gather<Vector3f>(mesh->vertex_positions_world(), edge_vertex_1);

        const Vector3f sample_edge_start = transform_pos(world_to_sample_, edge_start);
        const Vector3f sample_edge_end = transform_pos(world_to_sample_, edge_end);

#ifdef RAYD_PRIMARY_EDGE_VIS_CHECK
        scatter(primary_edge_info_.p0, detach<false>(sample_edge_start), scatter_indices);
        scatter(primary_edge_info_.p1, detach<false>(sample_edge_end), scatter_indices);
#else
        scatter(primary_edge_info_.p0, head<2>(sample_edge_start), scatter_indices);
        scatter(primary_edge_info_.p1, head<2>(sample_edge_end), scatter_indices);
#endif

        Vector2f sample_edge_direction = detach<false>(head<2>(sample_edge_end - sample_edge_start));
        Float edge_length = maximum(norm(sample_edge_direction), Float(Epsilon));
        sample_edge_direction /= edge_length;
        scatter(primary_edge_info_.edge_normal,
                Vector2f(-sample_edge_direction.y(), sample_edge_direction.x()),
                scatter_indices);
        scatter(primary_edge_info_.edge_length, edge_length, scatter_indices);
    }

    primary_edge_distribution_.init(detach<false>(primary_edge_info_.edge_length));
    primary_edges_ready_ = true;
    attach_primary_edge_scene(scene);
}

void PerspectiveCamera::attach_primary_edge_scene(const Scene &scene) {
    if (primary_edge_scene_ == &scene) {
        const_cast<Scene &>(scene).register_primary_edge_observer(this);
        return;
    }

    if (primary_edge_scene_ != nullptr) {
        const_cast<Scene *>(primary_edge_scene_)->unregister_primary_edge_observer(this);
    }

    primary_edge_scene_ = &scene;
    const_cast<Scene &>(scene).register_primary_edge_observer(this);
}

void PerspectiveCamera::invalidate_primary_edges_from_scene(const Scene *scene) {
    if (primary_edge_scene_ == scene) {
        primary_edges_ready_ = false;
    }
}

void PerspectiveCamera::clear_primary_edge_scene_binding(const Scene *scene) {
    if (primary_edge_scene_ == scene) {
        primary_edge_scene_ = nullptr;
        primary_edges_ready_ = false;
    }
}

std::string PerspectiveCamera::to_string() const {
    std::stringstream stream;
    stream << "Camera[width=" << image_width_
           << ", height=" << image_height_;
    if (uses_intrinsics_) {
        stream << ", fx=" << focal_length_x_
               << ", fy=" << focal_length_y_
               << ", cx=" << principal_point_x_
               << ", cy=" << principal_point_y_;
    } else {
        stream << ", fov_x=" << field_of_view_x_;
    }
    stream << "]";
    return stream.str();
}

RayDetached PerspectiveCamera::sample_primary_ray(const Vector2fDetached &samples) const {
    require(is_ready_, "PerspectiveCamera::sample_primary_ray(): camera is not configured.");

    const Vector3fDetached sample_direction =
        normalize(transform_pos<FloatDetached>(detach<false>(sample_to_camera_),
                                        concat(samples, Vector1fDetached(0.f))));
    const Matrix4fDetached to_world_matrix = detach<false>(left_transform_ * object_to_world_ * right_transform_);
    return RayDetached(transform_pos<FloatDetached>(to_world_matrix, zeros<Vector3fDetached>(slices(samples))),
                transform_dir<FloatDetached>(to_world_matrix, sample_direction));
}

Ray PerspectiveCamera::sample_primary_ray(const Vector2f &samples) const {
    require(is_ready_, "PerspectiveCamera::sample_primary_ray(): camera is not configured.");

    const Vector3f sample_direction =
        detach<false>(normalize(transform_pos<Float>(sample_to_camera_,
                                                      concat(samples, Vector1f(0.f)))));
    const Matrix4f to_world_matrix = left_transform_ * object_to_world_ * right_transform_;
    return Ray(transform_pos<Float>(to_world_matrix, zeros<Vector3f>(slices(samples))),
                transform_dir<Float>(to_world_matrix, sample_direction));
}

PrimaryEdgeSample PerspectiveCamera::sample_primary_edge(const FloatDetached &sample) const {
    require(primary_edges_ready_,
            "PerspectiveCamera::sample_primary_edge(): primary edges are not prepared.");

    FloatDetached edge_sample = sample;
    PrimaryEdgeSample result;
    const int sample_count = static_cast<int>(slices(edge_sample));

    result.idx = full<IntDetached>(-1, sample_count);
    result.pdf = zeros<FloatDetached>(sample_count);
    result.x_dot_n = zeros<Float>(sample_count);

    IntDetached edge_index;
    std::tie(edge_index, result.pdf) = primary_edge_distribution_.sample_reuse<true>(edge_sample);

    const Int edge_index_ad = Int(edge_index);
    const FloatDetached edge_length = detach<false>(gather<Float>(primary_edge_info_.edge_length, edge_index_ad));
    result.pdf /= edge_length;

    const Vector2fDetached edge_normal = detach<false>(gather<Vector2f>(primary_edge_info_.edge_normal, edge_index_ad));
    const Vector2f edge_start = gather<Vector2f>(primary_edge_info_.p0, edge_index_ad);
    const Vector2f edge_end = gather<Vector2f>(primary_edge_info_.p1, edge_index_ad);
    const Vector2f sample_position_ad = fmadd(edge_start, 1.f - edge_sample, edge_end * edge_sample);
    const Vector2fDetached sample_position = detach<false>(sample_position_ad);
    result.x_dot_n = dot(sample_position_ad, edge_normal);

    const Vector2fDetached resolution(static_cast<float>(image_width_), static_cast<float>(image_height_));
    const Vector2iDetached pixel = floor2int<Vector2iDetached, Vector2fDetached>(sample_position * resolution);
    const MaskDetached valid_pixel = pixel.x() >= 0 && pixel.x() < image_width_ &&
                              pixel.y() >= 0 && pixel.y() < image_height_;

    masked(result.idx, valid_pixel) = pixel.y() * image_width_ + pixel.x();
    masked(result.pdf, ~valid_pixel) = 0.f;

    result.ray_p = sample_primary_ray(sample_position + Epsilon * edge_normal);
    result.ray_n = sample_primary_ray(sample_position - Epsilon * edge_normal);
    return result;
}

drjit::Tensor<Float> PerspectiveCamera::render(const Scene &scene, float background) const {
    require(is_ready_, "PerspectiveCamera::render(): camera is not configured.");
    require(scene.is_ready(), "PerspectiveCamera::render(): scene is not configured.");
    require(!scene.has_pending_updates(),
            "PerspectiveCamera::render(): scene has pending updates. Call Scene::commit_updates() first.");

    const Vector2f samples = make_pixel_centers(image_width_, image_height_);
    const Ray rays = sample_primary_ray(samples);
    const Intersection its = scene.intersect<false>(rays);
    const Float image = select(its.is_valid(), its.t, full<Float>(background, image_width_ * image_height_));

    const size_t shape[2] = {
        static_cast<size_t>(image_height_),
        static_cast<size_t>(image_width_)
    };
    return drjit::Tensor<Float>(image, 2, shape);
}

drjit::Tensor<Float> PerspectiveCamera::render_grad(const Scene &scene, int spp, float background) const {
    require(is_ready_, "PerspectiveCamera::render_grad(): camera is not configured.");

    const Float image = render_edge_grad_flat(scene, *this, spp, background);
    const size_t shape[2] = {
        static_cast<size_t>(image_height_),
        static_cast<size_t>(image_width_)
    };
    return drjit::Tensor<Float>(image, 2, shape);
}

} // namespace rayd
