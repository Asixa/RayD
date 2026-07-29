// Copyright Xingyu Chen.
// Declares the Dr.Jit surfel API.

#pragma once

#include <rayd/jit/ray.h>
#include <rayd/jit/surfel_optix.h>

namespace rayd {

enum class SurfelPrimitiveMode : uint32_t {
    Icosahedron20 = 0,
    QuadTriangles = 1,
    SingleTriangle = 2,
};

enum class SurfelColorModel : uint32_t {
    ConstantRGB = 0,
    SH = 1,
    FeatureChannels = 2,
};

enum class SurfelRenderMode : uint32_t {
    Alpha = 0,
    Depth = 1,
    RGB = 2,
    RGBDepth = 3,
    Feature = 4,
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
    /// Collect per-ray candidate buffer occupancy diagnostics.
    bool collect_candidate_stats = false;
    /// Use per-surfel opacity when building proxy bounds.
    bool opacity_aware_proxy_bounds = false;
    /// Continue tracing after a full candidate buffer until transmittance is low enough.
    bool continue_after_full_buffer = false;
    float transmittance_min = 0.03f;
    int max_trace_segments = 4;
};

struct SurfelRenderOptions {
    SurfelRenderMode mode = SurfelRenderMode::RGB;
    SurfelColorModel color_model = SurfelColorModel::ConstantRGB;
    bool normal = false;
    int sh_degree = 0;
    int channel_count = 3;
    int channel_chunk = 0;
    ScalarVector3f background_rgb = ScalarVector3f(0.f, 0.f, 0.f);
};

template <typename Float_> struct SurfelIntersectionData {
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

    DRJIT_STRUCT(SurfelIntersectionData, t, p, n, local_uv, gaussian_weight, opacity, alpha, value, surfel_id,
                 triangle_id)
};

template <typename Float_> struct SurfelCompositeData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    Mask_ is_valid() const { return alpha > Float_(0.f); }

    Float_ intensity = zeros<Float_>(1); ///< Alpha-composited scalar surfel value.
    Float_ alpha = zeros<Float_>(1);     ///< Accumulated alpha, 1 - final transmittance.
    Float_ transmittance = full<Float_>(1.f, 1);
    Float_ depth = full<Float_>(Infinity, 1); ///< Alpha-weighted depth, Infinity when empty.
    Int_ candidate_count = zeros<Int_>(1);
    Mask_ candidate_buffer_full = full<Mask_>(false, 1);

    DRJIT_STRUCT(SurfelCompositeData, intensity, alpha, transmittance, depth, candidate_count, candidate_buffer_full)
};

template <typename Float_> struct SurfelRenderData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;
    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    Mask_ is_valid() const { return alpha > Float_(0.f); }

    Float_ channels = Float();      ///< Flat [ray_count, channel_count] output buffer.
    Vec3f rgb = zeros<Vec3f>(1);    ///< Convenience RGB view for RGB/RGBDepth modes.
    Vec3f normal = zeros<Vec3f>(1); ///< Alpha-weighted surfel normal; zero when empty or disabled.
    Float_ alpha = zeros<Float_>(1);
    Float_ transmittance = full<Float_>(1.f, 1);
    Float_ depth = full<Float_>(Infinity, 1);
    Int_ candidate_count = zeros<Int_>(1);
    Mask_ candidate_buffer_full = full<Mask_>(false, 1);
    int channel_count = 0;
};

class SurfelGeometry {
  public:
    SurfelGeometry() = default;
    SurfelGeometry(const Vector3f& center, const Vector3f& tangent_u, const Vector3f& tangent_v);
    SurfelGeometry(const Vector3fAD& center, const Vector3fAD& tangent_u, const Vector3fAD& tangent_v);

    int surfel_count() const { return surfel_count_; }

    const Vector3fAD& center() const { return center_; }
    const Vector3fAD& tangent_u() const { return tangent_u_; }
    const Vector3fAD& tangent_v() const { return tangent_v_; }

  private:
    void initialize(const Vector3fAD& center, const Vector3fAD& tangent_u, const Vector3fAD& tangent_v);

    Vector3fAD center_;
    Vector3fAD tangent_u_;
    Vector3fAD tangent_v_;
    int surfel_count_ = 0;
};

class SurfelAppearance {
  public:
    SurfelAppearance() = default;
    static SurfelAppearance rgb(const Float& opacity, const Vector3f& rgb);
    static SurfelAppearance rgb(const FloatAD& opacity, const Vector3fAD& rgb);
    static SurfelAppearance features(const Float& opacity, const Float& values, int channel_count);
    static SurfelAppearance features(const FloatAD& opacity, const FloatAD& values, int channel_count);
    static SurfelAppearance sh(const Float& opacity, const Float& coeffs, int sh_degree);
    static SurfelAppearance sh(const FloatAD& opacity, const FloatAD& coeffs, int sh_degree);

    SurfelAppearance(const Float& opacity, const Float& values, SurfelColorModel color_model, int channel_count,
                     int sh_degree = 0);
    SurfelAppearance(const FloatAD& opacity, const FloatAD& values, SurfelColorModel color_model, int channel_count,
                     int sh_degree = 0);

    int surfel_count() const { return surfel_count_; }
    int channel_count() const { return channel_count_; }
    int sh_degree() const { return sh_degree_; }
    int sh_basis_count() const { return (sh_degree_ + 1) * (sh_degree_ + 1); }
    SurfelColorModel color_model() const { return color_model_; }

    const FloatAD& opacity() const { return opacity_; }
    const FloatAD& values() const { return values_; }

  private:
    void initialize(const FloatAD& opacity, const FloatAD& values, SurfelColorModel color_model, int channel_count,
                    int sh_degree);

    FloatAD opacity_;
    FloatAD values_;
    SurfelColorModel color_model_ = SurfelColorModel::FeatureChannels;
    int channel_count_ = 0;
    int sh_degree_ = 0;
    int surfel_count_ = 0;
};

class SurfelCloud {
  public:
    SurfelCloud() = default;
    SurfelCloud(const Vector3f& center, const Vector3f& tangent_u, const Vector3f& tangent_v,
                const Float& opacity = Float(), const Float& value = Float());
    SurfelCloud(const Vector3fAD& center, const Vector3fAD& tangent_u, const Vector3fAD& tangent_v,
                const FloatAD& opacity = FloatAD(), const FloatAD& value = FloatAD());

    int surfel_count() const { return surfel_count_; }

    const Vector3fAD& center() const { return center_; }
    const Vector3fAD& tangent_u() const { return tangent_u_; }
    const Vector3fAD& tangent_v() const { return tangent_v_; }
    const FloatAD& opacity() const { return opacity_; }
    const FloatAD& value() const { return value_; }

  private:
    void initialize(const Vector3fAD& center, const Vector3fAD& tangent_u, const Vector3fAD& tangent_v,
                    const FloatAD& opacity, const FloatAD& value);

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
    explicit SurfelScene(const SurfelCloud& cloud, const SurfelTraceOptions& options = SurfelTraceOptions());
    explicit SurfelScene(const SurfelGeometry& geometry, const SurfelTraceOptions& options = SurfelTraceOptions());

    SurfelScene(const SurfelScene&) = delete;
    SurfelScene& operator=(const SurfelScene&) = delete;

    void build();
    bool is_ready() const { return ready_; }
    int surfel_count() const { return geometry_.surfel_count(); }
    int triangle_count() const { return triangle_count_; }
    int build_count() const { return build_count_; }

    void update_geometry(const SurfelGeometry& geometry);
    void update_appearance(const SurfelAppearance& appearance);

    template <bool Detached>
    SurfelIntersectionT<Detached> intersect(const RayT<Detached>& ray, MaskT<Detached> active) const;

    template <bool Detached>
    SurfelCompositeT<Detached> composite_alpha(const RayT<Detached>& ray, MaskT<Detached> active) const;

    template <bool Detached>
    SurfelCompositeT<Detached> composite_alpha_reference(const RayT<Detached>& ray, MaskT<Detached> active) const;

    template <bool Detached>
    SurfelRenderT<Detached> render(const RayT<Detached>& ray, const SurfelRenderOptions& render_options,
                                   MaskT<Detached> active) const;

    template <bool Detached> MaskT<Detached> shadow_test(const RayT<Detached>& ray, MaskT<Detached> active) const;

    template <bool Detached>
    MaskT<Detached> visible(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                            MaskT<Detached> active) const;

  private:
    void build_triangle_buffers();
    void validate_trace_options(const char* context) const;
    void refresh_detached_appearance();
    int render_channel_count(const SurfelRenderOptions& render_options) const;

    SurfelCloud cloud_;
    SurfelGeometry geometry_;
    SurfelAppearance appearance_;
    SurfelTraceOptions options_;
    bool ready_ = false;
    int build_count_ = 0;
    int vertex_count_ = 0;
    int triangle_count_ = 0;

    Int triangle_to_surfel_id_;
    Vector3f detached_center_;
    Vector3f detached_tangent_u_;
    Vector3f detached_tangent_v_;
    Float detached_opacity_;
    Float detached_value_;
    Float detached_appearance_values_;
    int appearance_channel_count_ = 1;
    int appearance_sh_degree_ = 0;
    SurfelColorModel appearance_color_model_ = SurfelColorModel::FeatureChannels;
    Float optix_vertex_buffer_;
    Int optix_face_buffer_;
    SurfelOptixScene optix_scene_;
};

} // namespace rayd
