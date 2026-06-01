#pragma once

#include <rayd/ray.h>
#include <rayd/surfel/surfel_optix.h>

namespace rayd {

enum class SurfelPrimitiveMode : uint32_t {
    Icosahedron20 = 0,
    QuadTriangles = 1,
    SingleTriangle = 2,
};

struct SurfelTraceOptions {
    /// Minimum Gaussian alpha included in the proxy and final alpha compositing.
    float alpha_min = 1.f / 255.f;
    /// Optional local-radius cap for debugging/backward compatibility; Infinity disables it.
    float cutoff = Infinity;
    float alpha_cap = 0.99f;
    /// Relative normal-axis scale for the thin Icosahedron20 proxy, multiplied by the Gaussian radius.
    float proxy_epsilon = 1e-3f;
    /// Small per-ray k-buffer capacity for detached native alpha compositing.
    /// The legacy retrace path also uses this as its candidate-iteration limit.
    int max_candidate_hits = 8;
    SurfelPrimitiveMode primitive_mode = SurfelPrimitiveMode::Icosahedron20;
    bool face_forward = true;
    /// Use the surfel native OptiX pipeline for detached intersect/composite calls.
    /// AD alpha compositing keeps the differentiable reference path.
    bool single_launch = true;
};

template <typename Float_>
struct SurfelIntersectionData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec2f = std::conditional_t<IsDetached, Vector2f, Vector2fAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    Mask_ is_valid() const { return surfel_id >= 0; }

    Float_ t = Infinity;
    Vec3f p = zeros<Vec3f>(1);
    Vec3f n = zeros<Vec3f>(1);
    Vec2f local_uv = zeros<Vec2f>(1);
    Float_ gaussian_weight = zeros<Float_>(1);
    Float_ opacity = zeros<Float_>(1);
    Float_ alpha = zeros<Float_>(1);
    Float_ value = zeros<Float_>(1);
    Int_ surfel_id = full<Int_>(-1, 1);
    Int_ triangle_id = full<Int_>(-1, 1);

    DRJIT_STRUCT(SurfelIntersectionData,
                 t,
                 p,
                 n,
                 local_uv,
                 gaussian_weight,
                 opacity,
                 alpha,
                 value,
                 surfel_id,
                 triangle_id)
};

template <typename Float_>
struct SurfelCompositeData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    Mask_ is_valid() const { return alpha > Float_(0.f); }

    Float_ intensity = zeros<Float_>(1);      ///< Alpha-composited scalar surfel value.
    Float_ alpha = zeros<Float_>(1);          ///< Accumulated alpha, 1 - final transmittance.
    Float_ transmittance = full<Float_>(1.f, 1);
    Float_ depth = full<Float_>(Infinity, 1); ///< Alpha-weighted depth, Infinity when empty.

    DRJIT_STRUCT(SurfelCompositeData,
                 intensity,
                 alpha,
                 transmittance,
                 depth)
};

class SurfelCloud {
public:
    SurfelCloud() = default;
    SurfelCloud(const Vector3f &center,
                const Vector3f &tangent_u,
                const Vector3f &tangent_v,
                const Float &opacity = Float(),
                const Float &value = Float());
    SurfelCloud(const Vector3fAD &center,
                const Vector3fAD &tangent_u,
                const Vector3fAD &tangent_v,
                const FloatAD &opacity = FloatAD(),
                const FloatAD &value = FloatAD());

    int surfel_count() const { return surfel_count_; }

    const Vector3fAD &center() const { return center_; }
    const Vector3fAD &tangent_u() const { return tangent_u_; }
    const Vector3fAD &tangent_v() const { return tangent_v_; }
    const FloatAD &opacity() const { return opacity_; }
    const FloatAD &value() const { return value_; }

private:
    void initialize(const Vector3fAD &center,
                    const Vector3fAD &tangent_u,
                    const Vector3fAD &tangent_v,
                    const FloatAD &opacity,
                    const FloatAD &value);

    Vector3fAD center_;
    Vector3fAD tangent_u_;
    Vector3fAD tangent_v_;
    FloatAD opacity_;
    FloatAD value_;
    int surfel_count_ = 0;
};

class SurfelScene {
public:
    SurfelScene() = default;
    explicit SurfelScene(const SurfelCloud &cloud,
                         const SurfelTraceOptions &options = SurfelTraceOptions());

    SurfelScene(const SurfelScene &) = delete;
    SurfelScene &operator=(const SurfelScene &) = delete;

    void build();
    bool is_ready() const { return ready_; }
    int surfel_count() const { return cloud_.surfel_count(); }
    int triangle_count() const { return triangle_count_; }

    template <bool Detached>
    SurfelIntersectionT<Detached> intersect(const RayT<Detached> &ray,
                                            MaskT<Detached> active) const;

    template <bool Detached>
    SurfelCompositeT<Detached> composite_alpha(const RayT<Detached> &ray,
                                               MaskT<Detached> active) const;

    template <bool Detached>
    SurfelCompositeT<Detached> composite_alpha_reference(const RayT<Detached> &ray,
                                                         MaskT<Detached> active) const;

    template <bool Detached>
    MaskT<Detached> shadow_test(const RayT<Detached> &ray,
                                MaskT<Detached> active) const;

    template <bool Detached>
    MaskT<Detached> visible(const Vector3fT<Detached> &start,
                            const Vector3fT<Detached> &end,
                            MaskT<Detached> active) const;

private:
    void build_triangle_buffers();

    SurfelCloud cloud_;
    SurfelTraceOptions options_;
    bool ready_ = false;
    int vertex_count_ = 0;
    int triangle_count_ = 0;

    Int triangle_to_surfel_id_;
    Vector3f detached_center_;
    Vector3f detached_tangent_u_;
    Vector3f detached_tangent_v_;
    Float detached_opacity_;
    Float detached_value_;
    Float optix_vertex_buffer_;
    Int optix_face_buffer_;
    SurfelOptixScene optix_scene_;
};

} // namespace rayd
