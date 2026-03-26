#pragma once

#include <utility>

#include <rayd/rayd.h>
#include <rayd/edge.h>
#include <drjit/tensor.h>
#include <drjit/math.h>
#include <drjit/util.h>

namespace rayd {

class Scene;

/// Perspective camera with optional intrinsic calibration and primary-edge sampling.
class Camera final {
public:
    Camera(float fov_x = 45.f, float near_clip = 1e-4f, float far_clip = 1e4f);
    Camera(float fx, float fy, float cx, float cy,
           float near_clip = 1e-4f, float far_clip = 1e4f);
    ~Camera();

    /// Rebuild camera transforms and cached projection data.
    void configure(bool cache = true);
    /// Extract image-space edges that contribute to primary visibility changes.
    void prepare_primary_edges(const Scene &scene);

    RayDetached sample_primary_ray(const Vector2fDetached &samples) const;
    Ray sample_primary_ray(const Vector2f &samples) const;
    PrimaryEdgeSample sample_primary_edge(const FloatDetached &sample) const;
    drjit::Tensor<Float> render(const Scene &scene, float background = 0.f) const;
    drjit::Tensor<Float> render_grad(const Scene &scene, int spp = 4, float background = 0.f) const;

    void set_transform(const Matrix4f &matrix, bool set_left = true);
    void append_transform(const Matrix4f &matrix, bool append_left = true);

    int width() const { return image_width_; }
    void set_width(int width) {
        image_width_ = width;
        is_ready_ = false;
        primary_edges_ready_ = false;
    }

    int height() const { return image_height_; }
    void set_height(int height) {
        image_height_ = height;
        is_ready_ = false;
        primary_edges_ready_ = false;
    }

    const Matrix4f &to_world() const { return object_to_world_; }
    void set_to_world(const Matrix4f &matrix) {
        object_to_world_ = matrix;
        is_ready_ = false;
        primary_edges_ready_ = false;
    }

    const Matrix4f &to_world_left() const { return left_transform_; }
    void set_to_world_left(const Matrix4f &matrix) {
        left_transform_ = matrix;
        is_ready_ = false;
        primary_edges_ready_ = false;
    }

    const Matrix4f &to_world_right() const { return right_transform_; }
    void set_to_world_right(const Matrix4f &matrix) {
        right_transform_ = matrix;
        is_ready_ = false;
        primary_edges_ready_ = false;
    }

    const Matrix4f &camera_to_sample() const { return camera_to_sample_; }
    const Matrix4f &sample_to_camera() const { return sample_to_camera_; }
    const Matrix4f &world_to_sample() const { return world_to_sample_; }
    const Matrix4f &sample_to_world() const { return sample_to_world_; }

    bool is_ready() const { return is_ready_; }
    bool uses_intrinsics() const { return uses_intrinsics_; }
    bool primary_edges_ready() const { return primary_edges_ready_; }

    std::string to_string() const;

private:
    void attach_primary_edge_scene(const Scene &scene);
    void invalidate_primary_edges_from_scene(const Scene *scene);
    void clear_primary_edge_scene_binding(const Scene *scene);

    int image_width_ = 1;
    int image_height_ = 1;
    float aspect_ratio_ = 1.f;

    float field_of_view_x_ = 45.f;
    float focal_length_x_ = 0.f;
    float focal_length_y_ = 0.f;
    float principal_point_x_ = 0.f;
    float principal_point_y_ = 0.f;
    float near_clip_ = 1e-4f;
    float far_clip_ = 1e4f;

    Matrix4f object_to_world_ = drjit::identity<Matrix4f>();
    Matrix4f left_transform_ = drjit::identity<Matrix4f>();
    Matrix4f right_transform_ = drjit::identity<Matrix4f>();

    Matrix4f sample_to_camera_;
    Matrix4f camera_to_sample_;
    Matrix4f world_to_sample_;
    Matrix4f sample_to_world_;

    Vector3f camera_position_;
    Vector3f camera_direction_;
    Float inverse_sensor_area_;

    bool uses_intrinsics_ = false;
    bool is_ready_ = false;
    bool primary_edges_ready_ = false;
    const Scene *primary_edge_scene_ = nullptr;

    PrimaryEdgeInfo primary_edge_info_;
    class DiscreteDistribution {
    public:
        DiscreteDistribution() = default;

        void init(const FloatDetached &pmf);
        std::pair<IntDetached, FloatDetached> sample(const FloatDetached &samples) const;

        template <bool Detached>
        std::pair<IntDetached, FloatDetached> sample_reuse(FloatT<Detached> &samples) const {
            if (size_ == 1) {
                return { zeros<IntDetached>(), full<FloatDetached>(1.f) };
            }

            samples *= sum_;
            IntDetached indices;
            if constexpr (!Detached) {
                indices = binary_search<IntDetached>(
                    0, size_ - 1,
                    [&](IntDetached index) DRJIT_INLINE_LAMBDA {
                        return gather<FloatDetached>(cmf_, index) < detach<false>(samples);
                    });
            } else {
                indices = binary_search<IntDetached>(
                    0, size_ - 1,
                    [&](IntDetached index) DRJIT_INLINE_LAMBDA {
                        return gather<FloatDetached>(cmf_, index) < samples;
                    });
            }

            samples -= gather<FloatDetached>(cmf_, indices - 1, indices > 0);
            const FloatDetached pmf = gather<FloatDetached>(pmf_, indices);
            masked(samples, pmf > 0.f) /= pmf;
            samples = clamp(samples, 0.f, 1.f);
            return { indices, pmf / sum_ };
        }

        const FloatDetached &pmf() const { return pmf_normalized_; }
        const FloatDetached &cmf() const { return cmf_normalized_; }

    private:
        int size_ = 0;
        FloatDetached sum_;
        FloatDetached pmf_;
        FloatDetached pmf_normalized_;
        FloatDetached cmf_;
        FloatDetached cmf_normalized_;
    };

    DiscreteDistribution primary_edge_distribution_;

    friend class Scene;
};

} // namespace rayd
