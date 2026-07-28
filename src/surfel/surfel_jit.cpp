#include <algorithm>
#include <array>
#include <cmath>

#include <rayd/surfel/surfel.h>

namespace rayd {

namespace {

constexpr int IcosahedronVertexCount = 12;
constexpr int IcosahedronFaceCount = 20;

const std::array<std::array<float, 3>, IcosahedronVertexCount> &icosahedron_vertices_unit_inradius() {
    static const std::array<std::array<float, 3>, IcosahedronVertexCount> vertices = []() {
        constexpr float Phi = 1.618033988749895f;
        constexpr float EdgeLength = 2.f;
        constexpr float Inradius =
            EdgeLength * (3.f * 1.7320508075688772f + 3.872983346207417f) / 12.f;
        constexpr float Scale = 1.f / Inradius;
        return std::array<std::array<float, 3>, IcosahedronVertexCount> {{
            {{-Scale,  Phi * Scale, 0.f}},
            {{ Scale,  Phi * Scale, 0.f}},
            {{-Scale, -Phi * Scale, 0.f}},
            {{ Scale, -Phi * Scale, 0.f}},
            {{0.f, -Scale,  Phi * Scale}},
            {{0.f,  Scale,  Phi * Scale}},
            {{0.f, -Scale, -Phi * Scale}},
            {{0.f,  Scale, -Phi * Scale}},
            {{ Phi * Scale, 0.f, -Scale}},
            {{ Phi * Scale, 0.f,  Scale}},
            {{-Phi * Scale, 0.f, -Scale}},
            {{-Phi * Scale, 0.f,  Scale}},
        }};
    }();
    return vertices;
}

const std::array<std::array<int, 3>, IcosahedronFaceCount> &icosahedron_faces() {
    static const std::array<std::array<int, 3>, IcosahedronFaceCount> faces {{
        {{0, 11, 5}}, {{0, 5, 1}}, {{0, 1, 7}}, {{0, 7, 10}}, {{0, 10, 11}},
        {{1, 5, 9}}, {{5, 11, 4}}, {{11, 10, 2}}, {{10, 7, 6}}, {{7, 1, 8}},
        {{3, 9, 4}}, {{3, 4, 2}}, {{3, 2, 6}}, {{3, 6, 8}}, {{3, 8, 9}},
        {{4, 9, 5}}, {{2, 4, 11}}, {{6, 2, 10}}, {{8, 6, 7}}, {{9, 8, 1}},
    }};
    return faces;
}

template <bool Detached>
struct AnalyticSurfelHit {
    FloatT<Detached> t;
    Vector3fT<Detached> p;
    Vector3fT<Detached> n;
    Vector2fT<Detached> local_uv;
    FloatT<Detached> gaussian_weight;
    FloatT<Detached> opacity;
    FloatT<Detached> alpha;
    FloatT<Detached> value;
    MaskT<Detached> valid;
};

int sh_basis_count_for_degree(int degree) {
    require(degree >= 0 && degree <= 3,
            "SurfelAppearance::sh(): sh_degree must be in [0, 3].");
    return (degree + 1) * (degree + 1);
}

template <bool Detached>
FloatT<Detached> sh_basis_value(int basis, const Vector3fT<Detached> &view_dir) {
    const FloatT<Detached> x = view_dir.x();
    const FloatT<Detached> y = view_dir.y();
    const FloatT<Detached> z = view_dir.z();
    switch (basis) {
        case 0: return FloatT<Detached>(0.28209479177387814f);
        case 1: return FloatT<Detached>(0.4886025119029199f) * y;
        case 2: return FloatT<Detached>(0.4886025119029199f) * z;
        case 3: return FloatT<Detached>(0.4886025119029199f) * x;
        case 4: return FloatT<Detached>(1.0925484305920792f) * x * y;
        case 5: return FloatT<Detached>(1.0925484305920792f) * y * z;
        case 6: return FloatT<Detached>(0.31539156525252005f) *
                       (FloatT<Detached>(3.f) * z * z - FloatT<Detached>(1.f));
        case 7: return FloatT<Detached>(1.0925484305920792f) * x * z;
        case 8: return FloatT<Detached>(0.5462742152960396f) * (x * x - y * y);
        case 9: return FloatT<Detached>(0.5900435899266435f) * y *
                       (FloatT<Detached>(3.f) * x * x - y * y);
        case 10: return FloatT<Detached>(2.890611442640554f) * x * y * z;
        case 11: return FloatT<Detached>(0.4570457994644658f) * y *
                        (FloatT<Detached>(5.f) * z * z - FloatT<Detached>(1.f));
        case 12: return FloatT<Detached>(0.3731763325901154f) * z *
                        (FloatT<Detached>(5.f) * z * z - FloatT<Detached>(3.f));
        case 13: return FloatT<Detached>(0.4570457994644658f) * x *
                        (FloatT<Detached>(5.f) * z * z - FloatT<Detached>(1.f));
        case 14: return FloatT<Detached>(1.445305721320277f) * z * (x * x - y * y);
        case 15: return FloatT<Detached>(0.5900435899266435f) * x *
                        (x * x - FloatT<Detached>(3.f) * y * y);
        default: return zeros<FloatT<Detached>>(slices(view_dir));
    }
}

FloatAD pack_rgb_values(const Vector3fAD &rgb) {
    const size_t count = slices(rgb);
    FloatAD packed = empty<FloatAD>(count * 3);
    const IntAD base = arange<IntAD>(count) * 3;
    scatter(packed, rgb.x(), base + 0);
    scatter(packed, rgb.y(), base + 1);
    scatter(packed, rgb.z(), base + 2);
    return packed;
}

FloatAD first_appearance_channel(const SurfelAppearance &appearance) {
    const int count = appearance.surfel_count();
    if (count <= 0 || appearance.channel_count() <= 0 || slices(appearance.values()) == 0) {
        return full<FloatAD>(1.f, std::max(count, 1));
    }
    const IntAD index = arange<IntAD>(count) * appearance.channel_count();
    return gather<FloatAD>(appearance.values(), index);
}

template <bool Detached>
AnalyticSurfelHit<Detached> evaluate_analytic_surfel_hit(const RayT<Detached> &ray,
                                                         const Vector3fT<Detached> &center,
                                                         const Vector3fT<Detached> &tangent_u,
                                                         const Vector3fT<Detached> &tangent_v,
                                                         const FloatT<Detached> &opacity,
                                                         const FloatT<Detached> &value,
                                                         const SurfelTraceOptions &options,
                                                         MaskT<Detached> active) {
    const int ray_count = static_cast<int>(slices(ray.o));
    AnalyticSurfelHit<Detached> hit;

    const Vector3fT<Detached> raw_normal = cross(tangent_u, tangent_v);
    const FloatT<Detached> normal_len_sq = squared_norm(raw_normal);
    const MaskT<Detached> normal_valid = normal_len_sq > FloatT<Detached>(1e-16f);
    Vector3fT<Detached> normal =
        raw_normal / sqrt(select(normal_valid, normal_len_sq, FloatT<Detached>(1.f)));
    if (options.face_forward) {
        normal = select(dot(normal, ray.d) > FloatT<Detached>(0.f), -normal, normal);
    }

    const FloatT<Detached> denom = dot(ray.d, normal);
    const MaskT<Detached> plane_valid = abs(denom) > FloatT<Detached>(1e-8f);
    const FloatT<Detached> safe_denom = select(plane_valid, denom, FloatT<Detached>(1.f));
    const FloatT<Detached> plane_t = dot(center - ray.o, normal) / safe_denom;
    MaskT<Detached> valid = active &&
                            normal_valid &&
                            plane_valid &&
                            drjit::isfinite(plane_t) &&
                            (plane_t > FloatT<Detached>(RayEpsilon)) &&
                            (plane_t < ray.tmax);

    const FloatT<Detached> safe_t = select(valid, plane_t, zeros<FloatT<Detached>>(ray_count));
    const Vector3fT<Detached> hit_point = ray(safe_t);
    const Vector3fT<Detached> delta = hit_point - center;

    const FloatT<Detached> uu = dot(tangent_u, tangent_u);
    const FloatT<Detached> uv = dot(tangent_u, tangent_v);
    const FloatT<Detached> vv = dot(tangent_v, tangent_v);
    const FloatT<Detached> du = dot(delta, tangent_u);
    const FloatT<Detached> dv = dot(delta, tangent_v);
    const FloatT<Detached> basis_det = uu * vv - uv * uv;
    const MaskT<Detached> basis_valid = abs(basis_det) > FloatT<Detached>(1e-16f);
    const FloatT<Detached> safe_basis_det = select(basis_valid, basis_det, FloatT<Detached>(1.f));

    const FloatT<Detached> local_u = (du * vv - dv * uv) / safe_basis_det;
    const FloatT<Detached> local_v = (dv * uu - du * uv) / safe_basis_det;
    const FloatT<Detached> gaussian =
        exp(FloatT<Detached>(-0.5f) * (local_u * local_u + local_v * local_v));
    const FloatT<Detached> alpha_uncapped = opacity * gaussian;
    const FloatT<Detached> alpha_min_value(options.alpha_min);
    const FloatT<Detached> alpha_accept_slack =
        FloatT<Detached>(1e-6f) * maximum(FloatT<Detached>(1.f), alpha_min_value);
    valid &= basis_valid && (alpha_uncapped + alpha_accept_slack >= alpha_min_value);

    FloatT<Detached> alpha =
        minimum(FloatT<Detached>(options.alpha_cap),
                maximum(FloatT<Detached>(0.f), alpha_uncapped));
    alpha = select(valid, alpha, zeros<FloatT<Detached>>(ray_count));

    hit.t = plane_t;
    hit.p = hit_point;
    hit.n = normal;
    hit.local_uv = Vector2fT<Detached>(local_u, local_v);
    hit.gaussian_weight = gaussian;
    hit.opacity = opacity;
    hit.alpha = alpha;
    hit.value = value;
    hit.valid = valid;
    return hit;
}

template <bool Detached>
FloatT<Detached> gather_appearance_channel(const SurfelAppearance &appearance,
                                           const IntT<Detached> &surfel_id,
                                           int output_channel,
                                           int sh_degree,
                                           const Vector3fT<Detached> &view_dir,
                                           MaskT<Detached> active) {
    const int ray_count = static_cast<int>(slices(surfel_id));
    if (appearance.surfel_count() <= 0 || slices(appearance.values()) == 0) {
        return full<FloatT<Detached>>(output_channel == 0 ? 1.f : 0.f, ray_count);
    }

    const IntT<Detached> safe_surfel = maximum(surfel_id, IntT<Detached>(0));
    if (appearance.color_model() == SurfelColorModel::SH) {
        const int basis_count = sh_basis_count_for_degree(sh_degree);
        const int storage_basis_count = sh_basis_count_for_degree(appearance.sh_degree());
        const int channel = output_channel < 3 ? output_channel : 0;
        FloatT<Detached> value = zeros<FloatT<Detached>>(ray_count);
        for (int basis = 0; basis < basis_count; ++basis) {
            const IntT<Detached> index =
                (safe_surfel * storage_basis_count + IntT<Detached>(basis)) * IntT<Detached>(3) +
                IntT<Detached>(channel);
            const FloatT<Detached> coeff =
                gather<FloatT<Detached>>(appearance.values(), index, active);
            value += coeff * sh_basis_value<Detached>(basis, view_dir);
        }
        return value;
    }

    const int source_channel =
        appearance.channel_count() > 0 ? std::min(output_channel, appearance.channel_count() - 1) : 0;
    const IntT<Detached> index =
        safe_surfel * IntT<Detached>(appearance.channel_count()) + IntT<Detached>(source_channel);
    return gather<FloatT<Detached>>(appearance.values(), index, active);
}

template <bool Detached>
Vector3fT<Detached> normalized_view_direction(const RayT<Detached> &ray) {
    const int ray_count = static_cast<int>(slices(ray.o));
    const Vector3fT<Detached> view = -ray.d;
    const FloatT<Detached> len_sq = squared_norm(view);
    return view / sqrt(select(len_sq > FloatT<Detached>(1e-16f),
                              len_sq,
                              FloatT<Detached>(1.f))) +
           zeros<Vector3fT<Detached>>(ray_count);
}

} // namespace

SurfelGeometry::SurfelGeometry(const Vector3f &center,
                               const Vector3f &tangent_u,
                               const Vector3f &tangent_v) {
    initialize(Vector3fAD(center), Vector3fAD(tangent_u), Vector3fAD(tangent_v));
}

SurfelGeometry::SurfelGeometry(const Vector3fAD &center,
                               const Vector3fAD &tangent_u,
                               const Vector3fAD &tangent_v) {
    initialize(center, tangent_u, tangent_v);
}

void SurfelGeometry::initialize(const Vector3fAD &center,
                                const Vector3fAD &tangent_u,
                                const Vector3fAD &tangent_v) {
    const size_t count = slices(center);
    require(count > 0, "SurfelGeometry(): center must contain at least one surfel.");
    require(slices(tangent_u) == count,
            "SurfelGeometry(): tangent_u must have the same lane count as center.");
    require(slices(tangent_v) == count,
            "SurfelGeometry(): tangent_v must have the same lane count as center.");
    center_ = center;
    tangent_u_ = tangent_u;
    tangent_v_ = tangent_v;
    surfel_count_ = static_cast<int>(count);
}

SurfelAppearance SurfelAppearance::rgb(const Float &opacity, const Vector3f &rgb) {
    return SurfelAppearance(FloatAD(opacity),
                            pack_rgb_values(Vector3fAD(rgb)),
                            SurfelColorModel::ConstantRGB,
                            3,
                            0);
}

SurfelAppearance SurfelAppearance::rgb(const FloatAD &opacity, const Vector3fAD &rgb) {
    return SurfelAppearance(opacity,
                            pack_rgb_values(rgb),
                            SurfelColorModel::ConstantRGB,
                            3,
                            0);
}

SurfelAppearance SurfelAppearance::features(const Float &opacity,
                                            const Float &values,
                                            int channel_count) {
    return SurfelAppearance(opacity, values, SurfelColorModel::FeatureChannels, channel_count, 0);
}

SurfelAppearance SurfelAppearance::features(const FloatAD &opacity,
                                            const FloatAD &values,
                                            int channel_count) {
    return SurfelAppearance(opacity, values, SurfelColorModel::FeatureChannels, channel_count, 0);
}

SurfelAppearance SurfelAppearance::sh(const Float &opacity,
                                      const Float &coeffs,
                                      int sh_degree) {
    return SurfelAppearance(opacity, coeffs, SurfelColorModel::SH, 3, sh_degree);
}

SurfelAppearance SurfelAppearance::sh(const FloatAD &opacity,
                                      const FloatAD &coeffs,
                                      int sh_degree) {
    return SurfelAppearance(opacity, coeffs, SurfelColorModel::SH, 3, sh_degree);
}

SurfelAppearance::SurfelAppearance(const Float &opacity,
                                   const Float &values,
                                   SurfelColorModel color_model,
                                   int channel_count,
                                   int sh_degree) {
    initialize(FloatAD(opacity), FloatAD(values), color_model, channel_count, sh_degree);
}

SurfelAppearance::SurfelAppearance(const FloatAD &opacity,
                                   const FloatAD &values,
                                   SurfelColorModel color_model,
                                   int channel_count,
                                   int sh_degree) {
    initialize(opacity, values, color_model, channel_count, sh_degree);
}

void SurfelAppearance::initialize(const FloatAD &opacity,
                                  const FloatAD &values,
                                  SurfelColorModel color_model,
                                  int channel_count,
                                  int sh_degree) {
    require(channel_count > 0, "SurfelAppearance(): channel_count must be positive.");
    const size_t count = slices(opacity);
    require(count > 0, "SurfelAppearance(): opacity must contain at least one surfel.");
    int values_per_surfel = channel_count;
    if (color_model == SurfelColorModel::SH) {
        values_per_surfel = sh_basis_count_for_degree(sh_degree) * 3;
    }
    require(slices(values) == count * static_cast<size_t>(values_per_surfel),
            "SurfelAppearance(): values must have surfel_count * values_per_surfel lanes.");
    opacity_ = opacity;
    values_ = values;
    color_model_ = color_model;
    channel_count_ = channel_count;
    sh_degree_ = sh_degree;
    surfel_count_ = static_cast<int>(count);
}

SurfelCloud::SurfelCloud(const Vector3f &center,
                         const Vector3f &tangent_u,
                         const Vector3f &tangent_v,
                         const Float &opacity,
                         const Float &value) {
    initialize(Vector3fAD(center),
               Vector3fAD(tangent_u),
               Vector3fAD(tangent_v),
               FloatAD(opacity),
               FloatAD(value));
}

SurfelCloud::SurfelCloud(const Vector3fAD &center,
                         const Vector3fAD &tangent_u,
                         const Vector3fAD &tangent_v,
                         const FloatAD &opacity,
                         const FloatAD &value) {
    initialize(center, tangent_u, tangent_v, opacity, value);
}

void SurfelCloud::initialize(const Vector3fAD &center,
                             const Vector3fAD &tangent_u,
                             const Vector3fAD &tangent_v,
                             const FloatAD &opacity,
                             const FloatAD &value) {
    const size_t count = slices(center);
    require(count > 0, "SurfelCloud(): center must contain at least one surfel.");
    require(slices(tangent_u) == count,
            "SurfelCloud(): tangent_u must have the same lane count as center.");
    require(slices(tangent_v) == count,
            "SurfelCloud(): tangent_v must have the same lane count as center.");

    center_ = center;
    tangent_u_ = tangent_u;
    tangent_v_ = tangent_v;
    surfel_count_ = static_cast<int>(count);

    if (slices(opacity) == 0) {
        opacity_ = full<FloatAD>(1.f, count);
    } else {
        require(slices(opacity) == count,
                "SurfelCloud(): opacity must have the same lane count as center.");
        opacity_ = opacity;
    }

    if (slices(value) == 0) {
        value_ = full<FloatAD>(1.f, count);
    } else {
        require(slices(value) == count,
                "SurfelCloud(): value must have the same lane count as center.");
        value_ = value;
    }
}

SurfelScene::SurfelScene(const SurfelCloud &cloud, const SurfelTraceOptions &options)
    : cloud_(cloud),
      geometry_(cloud.center(), cloud.tangent_u(), cloud.tangent_v()),
      appearance_(SurfelAppearance::features(cloud.opacity(), cloud.value(), 1)),
      options_(options) {
    require(geometry_.surfel_count() > 0, "SurfelScene(): cloud must contain at least one surfel.");
    validate_trace_options("SurfelScene()");
}

SurfelScene::SurfelScene(const SurfelGeometry &geometry, const SurfelTraceOptions &options)
    : geometry_(geometry),
      appearance_(SurfelAppearance::features(full<FloatAD>(1.f, geometry.surfel_count()),
                                             full<FloatAD>(1.f, geometry.surfel_count()),
                                             1)),
      options_(options) {
    require(geometry_.surfel_count() > 0, "SurfelScene(): geometry must contain at least one surfel.");
    validate_trace_options("SurfelScene()");
}

void SurfelScene::validate_trace_options(const char *context) const {
    (void) context;
    require(options_.alpha_min > 0.f && options_.alpha_min < 1.f,
            "SurfelScene(): alpha_min must be in (0, 1).");
    require(options_.cutoff > 0.f,
            "SurfelScene(): cutoff must be positive.");
    require(options_.alpha_cap > 0.f && options_.alpha_cap <= 1.f,
            "SurfelScene(): alpha_cap must be in (0, 1].");
    require(options_.proxy_epsilon > 0.f,
            "SurfelScene(): proxy_epsilon must be positive.");
    require(options_.max_candidate_hits > 0,
            "SurfelScene(): max_candidate_hits must be positive.");
    require(options_.transmittance_min >= 0.f && options_.transmittance_min < 1.f,
            "SurfelScene(): transmittance_min must be in [0, 1).");
    require(options_.max_trace_segments > 0,
            "SurfelScene(): max_trace_segments must be positive.");
}

void SurfelScene::refresh_detached_appearance() {
    require(appearance_.surfel_count() == geometry_.surfel_count(),
            "SurfelScene::update_appearance(): appearance surfel count must match geometry.");
    detached_opacity_ = detach<false>(appearance_.opacity());
    detached_appearance_values_ = detach<false>(appearance_.values());
    detached_value_ = detach<false>(first_appearance_channel(appearance_));
    appearance_channel_count_ = appearance_.channel_count();
    appearance_sh_degree_ = appearance_.sh_degree();
    appearance_color_model_ = appearance_.color_model();
    eval(detached_opacity_, detached_appearance_values_, detached_value_);
}

void SurfelScene::update_appearance(const SurfelAppearance &appearance) {
    require(appearance.surfel_count() == geometry_.surfel_count(),
            "SurfelScene::update_appearance(): appearance surfel count must match geometry.");
    appearance_ = appearance;
    refresh_detached_appearance();
}

void SurfelScene::update_geometry(const SurfelGeometry &geometry) {
    require(ready_, "SurfelScene::update_geometry(): scene is not built.");
    require(geometry.surfel_count() == geometry_.surfel_count(),
            "SurfelScene::update_geometry(): geometry surfel count must match the built scene.");
    geometry_ = geometry;

    build_triangle_buffers();
    detached_center_ = detach<false>(geometry_.center());
    detached_tangent_u_ = detach<false>(geometry_.tangent_u());
    detached_tangent_v_ = detach<false>(geometry_.tangent_v());
    eval(optix_vertex_buffer_,
         optix_face_buffer_,
         triangle_to_surfel_id_,
         detached_center_,
         detached_tangent_u_,
         detached_tangent_v_);
    optix_scene_.build(optix_vertex_buffer_,
                       optix_face_buffer_,
                       vertex_count_,
                       triangle_count_,
                       !options_.single_launch);
    ++build_count_;
}

void SurfelScene::build() {
    require(geometry_.surfel_count() > 0, "SurfelScene::build(): geometry is empty.");
    validate_trace_options("SurfelScene::build()");

    build_triangle_buffers();
    detached_center_ = detach<false>(geometry_.center());
    detached_tangent_u_ = detach<false>(geometry_.tangent_u());
    detached_tangent_v_ = detach<false>(geometry_.tangent_v());
    refresh_detached_appearance();
    eval(optix_vertex_buffer_,
         optix_face_buffer_,
         triangle_to_surfel_id_,
         detached_center_,
         detached_tangent_u_,
         detached_tangent_v_,
         detached_opacity_,
         detached_value_,
         detached_appearance_values_);
    optix_scene_.build(optix_vertex_buffer_,
                       optix_face_buffer_,
                       vertex_count_,
                       triangle_count_,
                       !options_.single_launch);
    ready_ = true;
    ++build_count_;
}

void SurfelScene::build_triangle_buffers() {
    const int surfel_count = geometry_.surfel_count();
    const bool use_icosahedron = options_.primitive_mode == SurfelPrimitiveMode::Icosahedron20;
    const bool use_quad = options_.primitive_mode == SurfelPrimitiveMode::QuadTriangles;
    const int vertices_per_surfel = use_icosahedron ? IcosahedronVertexCount : (use_quad ? 4 : 3);
    const int triangles_per_surfel = use_icosahedron ? IcosahedronFaceCount : (use_quad ? 2 : 1);
    vertex_count_ = surfel_count * vertices_per_surfel;
    triangle_count_ = surfel_count * triangles_per_surfel;

    Vector3fAD vertices_ad = empty<Vector3fAD>(vertex_count_);
    const IntAD base = arange<IntAD>(surfel_count) * vertices_per_surfel;
    const Vector3fAD &center = geometry_.center();
    const Vector3fAD &u = geometry_.tangent_u();
    const Vector3fAD &v = geometry_.tangent_v();
    const FloatAD alpha_min(options_.alpha_min);
    const FloatAD safe_opacity = options_.opacity_aware_proxy_bounds
        ? maximum(alpha_min, appearance_.opacity())
        : full<FloatAD>(1.f, surfel_count);
    FloatAD influence_radius =
        sqrt(maximum(FloatAD(0.f), FloatAD(2.f) * log(safe_opacity / alpha_min)));
    if (std::isfinite(options_.cutoff)) {
        influence_radius = minimum(influence_radius, FloatAD(options_.cutoff));
    }
    const FloatAD proxy_radius =
        influence_radius * FloatAD(1.0001f) + FloatAD(1e-6f);

    if (use_icosahedron) {
        const Vector3fAD raw_normal = cross(u, v);
        const FloatAD normal_len_sq = squared_norm(raw_normal);
        const Vector3fAD normal =
            raw_normal / sqrt(select(normal_len_sq > FloatAD(1e-16f),
                                     normal_len_sq,
                                     FloatAD(1.f)));
        const auto &vertices = icosahedron_vertices_unit_inradius();
        for (int vertex = 0; vertex < IcosahedronVertexCount; ++vertex) {
            const FloatAD local_x(vertices[static_cast<size_t>(vertex)][0]);
            const FloatAD local_y(vertices[static_cast<size_t>(vertex)][1]);
            const FloatAD local_z(vertices[static_cast<size_t>(vertex)][2]);
            const Vector3fAD proxy_vertex =
                center +
                proxy_radius * (local_x * u + local_y * v) +
                proxy_radius * FloatAD(options_.proxy_epsilon) * local_z * normal;
            scatter(vertices_ad, proxy_vertex, base + vertex);
        }
    } else if (use_quad) {
        scatter(vertices_ad, center - proxy_radius * u - proxy_radius * v, base + 0);
        scatter(vertices_ad, center + proxy_radius * u - proxy_radius * v, base + 1);
        scatter(vertices_ad, center + proxy_radius * u + proxy_radius * v, base + 2);
        scatter(vertices_ad, center - proxy_radius * u + proxy_radius * v, base + 3);
    } else {
        scatter(vertices_ad, center + (-2.f * proxy_radius) * u - proxy_radius * v, base + 0);
        scatter(vertices_ad, center + ( 2.f * proxy_radius) * u - proxy_radius * v, base + 1);
        scatter(vertices_ad, center + ( 3.f * proxy_radius) * v, base + 2);
    }

    const Vector3f vertices = detach<false>(vertices_ad);
    optix_vertex_buffer_ = empty<Float>(vertex_count_ * 3);
    const Int vertex_indices = arange<Int>(vertex_count_) * 3;
    for (int axis = 0; axis < 3; ++axis) {
        scatter(optix_vertex_buffer_, vertices[axis], vertex_indices + axis);
    }

    const Int triangle_id = arange<Int>(triangle_count_);
    const Int surfel_id = triangle_id / triangles_per_surfel;
    const Int local_face = triangle_id - surfel_id * triangles_per_surfel;
    const Int vertex_base = surfel_id * vertices_per_surfel;

    Int corner0;
    Int corner1;
    Int corner2;
    if (use_icosahedron) {
        std::array<int, IcosahedronFaceCount> face0;
        std::array<int, IcosahedronFaceCount> face1;
        std::array<int, IcosahedronFaceCount> face2;
        const auto &faces = icosahedron_faces();
        for (int i = 0; i < IcosahedronFaceCount; ++i) {
            face0[static_cast<size_t>(i)] = faces[static_cast<size_t>(i)][0];
            face1[static_cast<size_t>(i)] = faces[static_cast<size_t>(i)][1];
            face2[static_cast<size_t>(i)] = faces[static_cast<size_t>(i)][2];
        }
        corner0 = gather<Int>(load<Int>(face0.data(), face0.size()), local_face);
        corner1 = gather<Int>(load<Int>(face1.data(), face1.size()), local_face);
        corner2 = gather<Int>(load<Int>(face2.data(), face2.size()), local_face);
    } else if (use_quad) {
        corner0 = Int(0);
        corner1 = select(local_face == Int(0), Int(1), Int(2));
        corner2 = select(local_face == Int(0), Int(2), Int(3));
    } else {
        corner0 = Int(0);
        corner1 = Int(1);
        corner2 = Int(2);
    }

    const Int face_index = triangle_id * 3;
    optix_face_buffer_ = empty<Int>(triangle_count_ * 3);
    scatter(optix_face_buffer_, vertex_base + corner0, face_index + 0);
    scatter(optix_face_buffer_, vertex_base + corner1, face_index + 1);
    scatter(optix_face_buffer_, vertex_base + corner2, face_index + 2);
    triangle_to_surfel_id_ = surfel_id;
}

template <bool Detached>
SurfelIntersectionT<Detached> SurfelScene::intersect(const RayT<Detached> &ray,
                                                     MaskT<Detached> active) const {
    require(ready_, "SurfelScene::intersect(): scene is not built.");

    const int ray_count = static_cast<int>(slices(ray.o));
    SurfelIntersectionT<Detached> result;
    result.t = full<FloatT<Detached>>(Infinity, ray_count);
    result.p = zeros<Vector3fT<Detached>>(ray_count);
    result.n = zeros<Vector3fT<Detached>>(ray_count);
    result.local_uv = zeros<Vector2fT<Detached>>(ray_count);
    result.gaussian_weight = zeros<FloatT<Detached>>(ray_count);
    result.opacity = zeros<FloatT<Detached>>(ray_count);
    result.alpha = zeros<FloatT<Detached>>(ray_count);
    result.value = zeros<FloatT<Detached>>(ray_count);
    result.surfel_id = full<IntT<Detached>>(-1, ray_count);
    result.triangle_id = full<IntT<Detached>>(-1, ray_count);

    if (options_.single_launch) {
        MaskT<Detached> hit_mask = active;
        const SurfelOptixIntersection optix_hit =
            optix_scene_.template trace_analytic_candidates<Detached>(ray,
                                                                       triangle_to_surfel_id_,
                                                                       detached_center_,
                                                                       detached_tangent_u_,
                                                                       detached_tangent_v_,
                                                                       detached_opacity_,
                                                                       options_.alpha_min,
                                                                       options_.alpha_cap,
                                                                       options_.max_candidate_hits,
                                                                       options_.face_forward,
                                                                       hit_mask);
        const Mask hit_mask_detached = detach<false>(hit_mask);
        const Int triangle_id = optix_hit.triangle_id;
        const Int surfel_id = gather<Int>(triangle_to_surfel_id_, triangle_id, hit_mask_detached);

        Vector3fT<Detached> center;
        Vector3fT<Detached> tangent_u;
        Vector3fT<Detached> tangent_v;
        FloatT<Detached> opacity;
        FloatT<Detached> value;
        if constexpr (!Detached) {
            const IntAD surfel_id_ad = IntAD(surfel_id);
            center = gather<Vector3fAD>(geometry_.center(), surfel_id_ad, hit_mask);
            tangent_u = gather<Vector3fAD>(geometry_.tangent_u(), surfel_id_ad, hit_mask);
            tangent_v = gather<Vector3fAD>(geometry_.tangent_v(), surfel_id_ad, hit_mask);
            opacity = gather<FloatAD>(appearance_.opacity(), surfel_id_ad, hit_mask);
            value = gather<FloatAD>(first_appearance_channel(appearance_), surfel_id_ad, hit_mask);
        } else {
            center = gather<Vector3f>(detached_center_, surfel_id, hit_mask_detached);
            tangent_u = gather<Vector3f>(detached_tangent_u_, surfel_id, hit_mask_detached);
            tangent_v = gather<Vector3f>(detached_tangent_v_, surfel_id, hit_mask_detached);
            opacity = gather<Float>(detached_opacity_, surfel_id, hit_mask_detached);
            value = gather<Float>(detached_value_, surfel_id, hit_mask_detached);
        }

        const AnalyticSurfelHit<Detached> analytic =
            evaluate_analytic_surfel_hit<Detached>(ray,
                                                   center,
                                                   tangent_u,
                                                   tangent_v,
                                                   opacity,
                                                   value,
                                                   options_,
                                                   hit_mask);
        const MaskT<Detached> take = hit_mask && analytic.valid;
        result.t = select(take, analytic.t, result.t);
        result.p = select(take, analytic.p, result.p);
        result.n = select(take, analytic.n, result.n);
        result.local_uv = select(take, analytic.local_uv, result.local_uv);
        result.gaussian_weight = select(take, analytic.gaussian_weight, result.gaussian_weight);
        result.opacity = select(take, analytic.opacity, result.opacity);
        result.alpha = select(take, analytic.alpha, result.alpha);
        result.value = select(take, analytic.value, result.value);
        result.surfel_id = select(take, IntT<Detached>(surfel_id), result.surfel_id);
        result.triangle_id = select(take, IntT<Detached>(triangle_id), result.triangle_id);
        return result;
    }

    MaskT<Detached> search_active = active;
    FloatT<Detached> t_min = full<FloatT<Detached>>(RayEpsilon, ray_count);

    for (int candidate = 0; candidate < options_.max_candidate_hits; ++candidate) {
        MaskT<Detached> hit_mask = search_active && (t_min < ray.tmax);
        const SurfelOptixIntersection optix_hit =
            optix_scene_.template intersect<Detached>(ray, t_min, hit_mask);
        const Mask hit_mask_detached = detach<false>(hit_mask);
        const Int triangle_id = optix_hit.triangle_id;
        const Int surfel_id = gather<Int>(triangle_to_surfel_id_, triangle_id, hit_mask_detached);

        Vector3fT<Detached> center;
        Vector3fT<Detached> tangent_u;
        Vector3fT<Detached> tangent_v;
        FloatT<Detached> opacity;
        FloatT<Detached> value;
        if constexpr (!Detached) {
            const IntAD surfel_id_ad = IntAD(surfel_id);
            center = gather<Vector3fAD>(geometry_.center(), surfel_id_ad, hit_mask);
            tangent_u = gather<Vector3fAD>(geometry_.tangent_u(), surfel_id_ad, hit_mask);
            tangent_v = gather<Vector3fAD>(geometry_.tangent_v(), surfel_id_ad, hit_mask);
            opacity = gather<FloatAD>(appearance_.opacity(), surfel_id_ad, hit_mask);
            value = gather<FloatAD>(first_appearance_channel(appearance_), surfel_id_ad, hit_mask);
        } else {
            center = gather<Vector3f>(detached_center_, surfel_id, hit_mask_detached);
            tangent_u = gather<Vector3f>(detached_tangent_u_, surfel_id, hit_mask_detached);
            tangent_v = gather<Vector3f>(detached_tangent_v_, surfel_id, hit_mask_detached);
            opacity = gather<Float>(detached_opacity_, surfel_id, hit_mask_detached);
            value = gather<Float>(detached_value_, surfel_id, hit_mask_detached);
        }

        const AnalyticSurfelHit<Detached> analytic =
            evaluate_analytic_surfel_hit<Detached>(ray,
                                                   center,
                                                   tangent_u,
                                                   tangent_v,
                                                   opacity,
                                                   value,
                                                   options_,
                                                   hit_mask);
        const MaskT<Detached> take = analytic.valid && (analytic.t < result.t);
        result.t = select(take, analytic.t, result.t);
        result.p = select(take, analytic.p, result.p);
        result.n = select(take, analytic.n, result.n);
        result.local_uv = select(take, analytic.local_uv, result.local_uv);
        result.gaussian_weight = select(take, analytic.gaussian_weight, result.gaussian_weight);
        result.opacity = select(take, analytic.opacity, result.opacity);
        result.alpha = select(take, analytic.alpha, result.alpha);
        result.value = select(take, analytic.value, result.value);
        result.surfel_id = select(take, IntT<Detached>(surfel_id), result.surfel_id);
        result.triangle_id = select(take, IntT<Detached>(triangle_id), result.triangle_id);

        t_min = select(hit_mask,
                       optix_hit.t + FloatT<Detached>(RayEpsilon),
                       t_min);
        search_active = hit_mask && (result.surfel_id < IntT<Detached>(0));
    }
    return result;
}

template <bool Detached>
SurfelCompositeT<Detached> SurfelScene::composite_alpha(const RayT<Detached> &ray,
                                                        MaskT<Detached> active) const {
    if constexpr (!Detached) {
        if (options_.continue_after_full_buffer) {
            return composite_alpha_reference<Detached>(ray, active);
        }
    }
    if (options_.single_launch) {
        const SurfelOptixComposite native =
            optix_scene_.template composite_alpha<Detached>(ray,
                                                            triangle_to_surfel_id_,
                                                            detached_center_,
                                                            detached_tangent_u_,
                                                            detached_tangent_v_,
                                                            detached_opacity_,
                                                            detached_value_,
                                                             options_.alpha_min,
                                                             options_.alpha_cap,
                                                             options_.max_candidate_hits,
                                                             options_.collect_candidate_stats,
                                                             options_.continue_after_full_buffer,
                                                             options_.transmittance_min,
                                                             options_.max_trace_segments,
                                                             options_.face_forward,
                                                             active);
        if constexpr (Detached) {
            SurfelCompositeT<Detached> result;
            result.intensity = native.intensity;
            result.alpha = native.alpha;
            result.transmittance = native.transmittance;
            result.depth = native.depth;
            result.candidate_count = native.candidate_count;
            result.candidate_buffer_full = native.candidate_buffer_full;
            return result;
        } else {
            const int ray_count = static_cast<int>(slices(ray.o));
            SurfelCompositeT<Detached> result;
            result.intensity = zeros<FloatT<Detached>>(ray_count);
            result.alpha = zeros<FloatT<Detached>>(ray_count);
            result.transmittance = full<FloatT<Detached>>(1.f, ray_count);
            result.depth = full<FloatT<Detached>>(Infinity, ray_count);
            result.candidate_count = IntAD(native.candidate_count);
            result.candidate_buffer_full = MaskAD(native.candidate_buffer_full);

            FloatT<Detached> depth_numerator = zeros<FloatT<Detached>>(ray_count);
            const Int slot_base = arange<Int>(ray_count) * native.hit_capacity;
            for (int slot = 0; slot < native.hit_capacity; ++slot) {
                const Int surfel_id = gather<Int>(native.surfel_id, slot_base + slot);
                const MaskT<Detached> hit_mask = active && (IntT<Detached>(surfel_id) >= IntT<Detached>(0));
                const IntAD surfel_id_ad = IntAD(surfel_id);
                const Vector3fAD center = gather<Vector3fAD>(geometry_.center(), surfel_id_ad, hit_mask);
                const Vector3fAD tangent_u = gather<Vector3fAD>(geometry_.tangent_u(), surfel_id_ad, hit_mask);
                const Vector3fAD tangent_v = gather<Vector3fAD>(geometry_.tangent_v(), surfel_id_ad, hit_mask);
                const FloatAD opacity = gather<FloatAD>(appearance_.opacity(), surfel_id_ad, hit_mask);
                const FloatAD value = gather<FloatAD>(first_appearance_channel(appearance_), surfel_id_ad, hit_mask);
                const AnalyticSurfelHit<Detached> analytic =
                    evaluate_analytic_surfel_hit<Detached>(ray,
                                                           center,
                                                           tangent_u,
                                                           tangent_v,
                                                           opacity,
                                                           value,
                                                           options_,
                                                           hit_mask);
                const FloatT<Detached> contribution =
                    select(analytic.valid,
                           result.transmittance * analytic.alpha,
                           zeros<FloatT<Detached>>(ray_count));
                result.intensity += contribution * analytic.value;
                result.alpha += contribution;
                depth_numerator += contribution * select(analytic.valid,
                                                         analytic.t,
                                                         zeros<FloatT<Detached>>(ray_count));
                result.transmittance *= select(analytic.valid,
                                               FloatT<Detached>(1.f) - analytic.alpha,
                                               FloatT<Detached>(1.f));
            }

            const MaskT<Detached> has_alpha = result.alpha > FloatT<Detached>(0.f);
            result.depth = select(has_alpha, depth_numerator / result.alpha, result.depth);
            return result;
        }
    }
    return composite_alpha_reference<Detached>(ray, active);
}

template <bool Detached>
SurfelCompositeT<Detached> SurfelScene::composite_alpha_reference(const RayT<Detached> &ray,
                                                                  MaskT<Detached> active) const {
    require(ready_, "SurfelScene::composite_alpha_reference(): scene is not built.");

    const int ray_count = static_cast<int>(slices(ray.o));
    SurfelCompositeT<Detached> result;
    result.intensity = zeros<FloatT<Detached>>(ray_count);
    result.alpha = zeros<FloatT<Detached>>(ray_count);
    result.transmittance = full<FloatT<Detached>>(1.f, ray_count);
    result.depth = full<FloatT<Detached>>(Infinity, ray_count);

    FloatT<Detached> depth_numerator = zeros<FloatT<Detached>>(ray_count);
    FloatT<Detached> previous_t = full<FloatT<Detached>>(-Infinity, ray_count);
    IntT<Detached> previous_id = full<IntT<Detached>>(-1, ray_count);

    for (int pick = 0; pick < geometry_.surfel_count(); ++pick) {
        FloatT<Detached> best_t = full<FloatT<Detached>>(Infinity, ray_count);
        FloatT<Detached> best_alpha = zeros<FloatT<Detached>>(ray_count);
        FloatT<Detached> best_value = zeros<FloatT<Detached>>(ray_count);
        IntT<Detached> best_id = full<IntT<Detached>>(-1, ray_count);

        for (int surfel = 0; surfel < geometry_.surfel_count(); ++surfel) {
            const IntT<Detached> surfel_id_current =
                full<IntT<Detached>>(surfel, ray_count);
            Vector3fT<Detached> center;
            Vector3fT<Detached> tangent_u;
            Vector3fT<Detached> tangent_v;
            FloatT<Detached> opacity;
            FloatT<Detached> value;
            if constexpr (!Detached) {
                const IntAD surfel_id = IntAD(full<Int>(surfel, ray_count));
                center = gather<Vector3fAD>(geometry_.center(), surfel_id, active);
                tangent_u = gather<Vector3fAD>(geometry_.tangent_u(), surfel_id, active);
                tangent_v = gather<Vector3fAD>(geometry_.tangent_v(), surfel_id, active);
                opacity = gather<FloatAD>(appearance_.opacity(), surfel_id, active);
                value = gather<FloatAD>(first_appearance_channel(appearance_), surfel_id, active);
            } else {
                const Int surfel_id = full<Int>(surfel, ray_count);
                const Mask active_detached = detach<false>(active);
                center = gather<Vector3f>(detached_center_, surfel_id, active_detached);
                tangent_u = gather<Vector3f>(detached_tangent_u_, surfel_id, active_detached);
                tangent_v = gather<Vector3f>(detached_tangent_v_, surfel_id, active_detached);
                opacity = gather<Float>(detached_opacity_, surfel_id, active_detached);
                value = gather<Float>(detached_value_, surfel_id, active_detached);
            }

            const AnalyticSurfelHit<Detached> analytic =
                evaluate_analytic_surfel_hit<Detached>(ray,
                                                       center,
                                                       tangent_u,
                                                       tangent_v,
                                                       opacity,
                                                       value,
                                                       options_,
                                                       active);

            constexpr float SortEpsilon = 1e-6f;
            const FloatT<Detached> order_eps(SortEpsilon);
            const MaskT<Detached> after_previous =
                (analytic.t > previous_t + order_eps) ||
                ((abs(analytic.t - previous_t) <= order_eps) &&
                 (surfel_id_current > previous_id));
            const MaskT<Detached> before_best =
                (analytic.t < best_t - order_eps) ||
                ((abs(analytic.t - best_t) <= order_eps) &&
                 ((best_id < IntT<Detached>(0)) || (surfel_id_current < best_id)));
            const MaskT<Detached> take = analytic.valid && after_previous && before_best;

            best_t = select(take, analytic.t, best_t);
            best_alpha = select(take, analytic.alpha, best_alpha);
            best_value = select(take, analytic.value, best_value);
            best_id = select(take, surfel_id_current, best_id);
        }

        const MaskT<Detached> has_best = best_id >= IntT<Detached>(0);
        best_alpha = select(has_best, best_alpha, zeros<FloatT<Detached>>(ray_count));
        const FloatT<Detached> safe_best_t =
            select(has_best, best_t, zeros<FloatT<Detached>>(ray_count));
        const FloatT<Detached> contribution = result.transmittance * best_alpha;
        result.intensity += contribution * best_value;
        result.alpha += contribution;
        depth_numerator += contribution * safe_best_t;
        result.transmittance *= FloatT<Detached>(1.f) - best_alpha;
        previous_t = select(has_best, best_t, previous_t);
        previous_id = select(has_best, best_id, previous_id);
    }

    const MaskT<Detached> has_alpha = result.alpha > FloatT<Detached>(0.f);
    result.depth = select(has_alpha, depth_numerator / result.alpha, result.depth);
    return result;
}

int SurfelScene::render_channel_count(const SurfelRenderOptions &render_options) const {
    switch (render_options.mode) {
        case SurfelRenderMode::Alpha:
        case SurfelRenderMode::Depth:
            return 0;
        case SurfelRenderMode::RGB:
        case SurfelRenderMode::RGBDepth:
            return 3;
        case SurfelRenderMode::Feature:
            require(render_options.channel_count > 0,
                    "SurfelRenderOptions.feature(): channel_count must be positive.");
            return render_options.channel_count;
        default:
            return 0;
    }
}

template <bool Detached>
SurfelRenderT<Detached> SurfelScene::render(const RayT<Detached> &ray,
                                            const SurfelRenderOptions &render_options,
                                            MaskT<Detached> active) const {
    require(ready_, "SurfelScene::render(): scene is not built.");
    const int ray_count = static_cast<int>(slices(ray.o));
    const int output_channels = render_channel_count(render_options);
    require(render_options.sh_degree >= 0 && render_options.sh_degree <= 3,
            "SurfelScene::render(): sh_degree must be in [0, 3].");
    require(render_options.color_model == appearance_color_model_ ||
                (render_options.color_model == SurfelColorModel::ConstantRGB &&
                 appearance_color_model_ == SurfelColorModel::FeatureChannels),
            "SurfelScene::render(): render color model must match the current appearance.");
    if (appearance_color_model_ == SurfelColorModel::SH) {
        require(render_options.sh_degree <= appearance_sh_degree_,
                "SurfelScene::render(): sh_degree exceeds appearance SH degree.");
    }
    if (render_options.mode == SurfelRenderMode::Feature) {
        require(output_channels <= appearance_channel_count_,
                "SurfelScene::render(): requested feature channel_count exceeds appearance.");
    }
    const int effective_sh_degree =
        appearance_color_model_ == SurfelColorModel::SH ? render_options.sh_degree : 0;

    SurfelRenderT<Detached> result;
    result.channel_count = output_channels;
    result.channels = output_channels > 0
        ? zeros<FloatT<Detached>>(ray_count * output_channels)
        : FloatT<Detached>();
    result.rgb = zeros<Vector3fT<Detached>>(ray_count);
    result.normal = zeros<Vector3fT<Detached>>(ray_count);
    result.alpha = zeros<FloatT<Detached>>(ray_count);
    result.transmittance = full<FloatT<Detached>>(1.f, ray_count);
    result.depth = full<FloatT<Detached>>(Infinity, ray_count);

    const SurfelOptixComposite native =
        optix_scene_.template render<Detached>(ray,
                                               triangle_to_surfel_id_,
                                               detached_center_,
                                               detached_tangent_u_,
                                               detached_tangent_v_,
                                               detached_opacity_,
                                               detached_appearance_values_,
                                               appearance_channel_count_,
                                               static_cast<int>(appearance_color_model_),
                                                appearance_sh_degree_,
                                                effective_sh_degree,
                                                output_channels,
                                                render_options.normal,
                                                render_options.background_rgb,
                                                options_.alpha_min,
                                                options_.alpha_cap,
                                                options_.max_candidate_hits,
                                                options_.collect_candidate_stats,
                                                Detached && options_.continue_after_full_buffer,
                                                options_.transmittance_min,
                                                options_.max_trace_segments,
                                                options_.face_forward,
                                                active);

    result.candidate_count = IntT<Detached>(native.candidate_count);
    result.candidate_buffer_full = MaskT<Detached>(native.candidate_buffer_full);

    if constexpr (Detached) {
        result.channels = native.channels;
        result.alpha = native.alpha;
        result.transmittance = native.transmittance;
        result.depth = native.depth;
        result.normal = native.normal;
        if (output_channels >= 3) {
            const Int channel_base = arange<Int>(ray_count) * output_channels;
            result.rgb = Vector3f(gather<Float>(result.channels, channel_base + 0),
                                  gather<Float>(result.channels, channel_base + 1),
                                  gather<Float>(result.channels, channel_base + 2));
        }
        return result;
    } else {
        FloatT<Detached> depth_numerator = zeros<FloatT<Detached>>(ray_count);
        Vector3fT<Detached> normal_numerator = zeros<Vector3fT<Detached>>(ray_count);
        const Vector3fT<Detached> view_dir = normalized_view_direction<Detached>(ray);
        const Int slot_base = arange<Int>(ray_count) * native.hit_capacity;
        for (int slot = 0; slot < native.hit_capacity; ++slot) {
            const Int surfel_id = gather<Int>(native.surfel_id, slot_base + slot);
            const MaskT<Detached> hit_mask = active && (IntT<Detached>(surfel_id) >= IntT<Detached>(0));
            const IntAD surfel_id_ad = IntAD(surfel_id);
            const Vector3fAD center = gather<Vector3fAD>(geometry_.center(), surfel_id_ad, hit_mask);
            const Vector3fAD tangent_u = gather<Vector3fAD>(geometry_.tangent_u(), surfel_id_ad, hit_mask);
            const Vector3fAD tangent_v = gather<Vector3fAD>(geometry_.tangent_v(), surfel_id_ad, hit_mask);
            const FloatAD opacity = gather<FloatAD>(appearance_.opacity(), surfel_id_ad, hit_mask);
            const AnalyticSurfelHit<Detached> analytic =
                evaluate_analytic_surfel_hit<Detached>(ray,
                                                       center,
                                                       tangent_u,
                                                       tangent_v,
                                                       opacity,
                                                       zeros<FloatAD>(ray_count),
                                                       options_,
                                                       hit_mask);
            const FloatT<Detached> contribution =
                select(analytic.valid,
                       result.transmittance * analytic.alpha,
                       zeros<FloatT<Detached>>(ray_count));
            result.alpha += contribution;
            depth_numerator += contribution * select(analytic.valid,
                                                      analytic.t,
                                                      zeros<FloatT<Detached>>(ray_count));
            if (render_options.normal) {
                normal_numerator += analytic.n * contribution;
            }
            if (output_channels > 0) {
                const IntAD out_base = IntAD(arange<Int>(ray_count) * output_channels);
                for (int channel = 0; channel < output_channels; ++channel) {
                    const FloatAD payload =
                        gather_appearance_channel<Detached>(appearance_,
                                                            IntT<Detached>(surfel_id),
                                                            channel,
                                                            effective_sh_degree,
                                                            view_dir,
                                                            hit_mask);
                    scatter_add(result.channels,
                                contribution * payload,
                                out_base + channel,
                                hit_mask);
                }
            }
            result.transmittance *= select(analytic.valid,
                                           FloatT<Detached>(1.f) - analytic.alpha,
                                           FloatT<Detached>(1.f));
        }

        if (output_channels >= 3) {
            const IntAD channel_base = IntAD(arange<Int>(ray_count) * output_channels);
            for (int channel = 0; channel < 3; ++channel) {
                scatter_add(result.channels,
                            result.transmittance * FloatAD(render_options.background_rgb[channel]),
                            channel_base + channel,
                            active);
            }
            result.rgb = Vector3fAD(gather<FloatAD>(result.channels, channel_base + 0),
                                    gather<FloatAD>(result.channels, channel_base + 1),
                                    gather<FloatAD>(result.channels, channel_base + 2));
        }
        const MaskT<Detached> has_alpha = result.alpha > FloatT<Detached>(0.f);
        result.depth = select(has_alpha, depth_numerator / result.alpha, result.depth);
        if (render_options.normal) {
            result.normal = select(has_alpha, normal_numerator / result.alpha, result.normal);
        }
        return result;
    }
}

template <bool Detached>
MaskT<Detached> SurfelScene::shadow_test(const RayT<Detached> &ray,
                                         MaskT<Detached> active) const {
    return intersect<Detached>(ray, active).is_valid();
}

template <bool Detached>
MaskT<Detached> SurfelScene::visible(const Vector3fT<Detached> &start,
                                     const Vector3fT<Detached> &end,
                                     MaskT<Detached> active) const {
    const int ray_count = static_cast<int>(slices(start));
    const Vector3fT<Detached> delta = end - start;
    const FloatT<Detached> length_sq = squared_norm(delta);
    const MaskT<Detached> valid_segment =
        length_sq > FloatT<Detached>((2.f * ShadowEpsilon) * (2.f * ShadowEpsilon));
    const FloatT<Detached> length =
        sqrt(select(valid_segment, length_sq, FloatT<Detached>(1.f)));
    const Vector3fT<Detached> direction = delta / length;
    const Vector3fT<Detached> origin = start + FloatT<Detached>(ShadowEpsilon) * direction;
    const FloatT<Detached> tmax =
        maximum(length - FloatT<Detached>(2.f * ShadowEpsilon), FloatT<Detached>(0.f));
    const RayT<Detached> ray(origin, direction, tmax);
    const MaskT<Detached> trace_active = active && valid_segment;
    const MaskT<Detached> hit = shadow_test<Detached>(ray, trace_active);
    return trace_active && !hit;
}

template SurfelIntersection SurfelScene::intersect<true>(const Ray &ray, Mask active) const;
template SurfelIntersectionAD SurfelScene::intersect<false>(const RayAD &ray, MaskAD active) const;
template SurfelComposite SurfelScene::composite_alpha<true>(const Ray &ray, Mask active) const;
template SurfelCompositeAD SurfelScene::composite_alpha<false>(const RayAD &ray, MaskAD active) const;
template SurfelComposite SurfelScene::composite_alpha_reference<true>(const Ray &ray, Mask active) const;
template SurfelCompositeAD SurfelScene::composite_alpha_reference<false>(const RayAD &ray, MaskAD active) const;
template SurfelRender SurfelScene::render<true>(const Ray &ray,
                                                const SurfelRenderOptions &render_options,
                                                Mask active) const;
template SurfelRenderAD SurfelScene::render<false>(const RayAD &ray,
                                                   const SurfelRenderOptions &render_options,
                                                   MaskAD active) const;
template Mask SurfelScene::shadow_test<true>(const Ray &ray, Mask active) const;
template MaskAD SurfelScene::shadow_test<false>(const RayAD &ray, MaskAD active) const;
template Mask SurfelScene::visible<true>(const Vector3f &start, const Vector3f &end, Mask active) const;
template MaskAD SurfelScene::visible<false>(const Vector3fAD &start, const Vector3fAD &end, MaskAD active) const;

} // namespace rayd

// Consolidated surfel OptiX host facade.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include <rayd/native_launch_audit.h>
#include <rayd/surfel/surfel_optix.h>
#include <rayd/surfel/surfel_trace_params.h>
// Embedded native surfel OptiX programs: analytic first-hit tracing and k-buffer compositing.
#include <surfel_trace_ptx.h>

namespace rayd {

namespace dr = drjit;

#ifndef RAYD_OPTIX_EXCEPTION_FLAGS
#  define RAYD_OPTIX_EXCEPTION_FLAGS OPTIX_EXCEPTION_FLAG_NONE
#endif

#ifndef RAYD_OPTIX_MODULE_OPT_LEVEL
#  define RAYD_OPTIX_MODULE_OPT_LEVEL OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
#endif

namespace {

enum class SurfelOptixLaunchKind {
    Intersect,
    Composite
};

struct RetiredSurfelOptixJitResources {
    UInt pipeline_handle;
    UInt sbt_handle;
};

std::mutex &retired_surfel_optix_resources_mutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

std::vector<RetiredSurfelOptixJitResources> &retired_surfel_optix_resources() {
    static std::vector<RetiredSurfelOptixJitResources> *resources =
        new std::vector<RetiredSurfelOptixJitResources>();
    return *resources;
}

} // namespace

struct SurfelOptixState {
    OptixDeviceContext context = 0;

    UInt64 handle;
    UInt pipeline_handle;
    UInt sbt_handle;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc pgd[2] = {};
    OptixProgramGroup pg[2] = {};
    OptixShaderBindingTable sbt = {};

    void *vertex_buffer = nullptr;
    void *vertex_buffer_ptr = nullptr;
    void *gas_temp_buffer = nullptr;
    size_t gas_temp_buffer_size = 0;
    void *gas_buffer = nullptr;
    size_t gas_buffer_size = 0;
    OptixTraversableHandle gas_handle = 0;
    OptixAccelBuildOptions accel_options = {};
    OptixAccelBufferSizes gas_buffer_sizes = {};

    OptixModule trace_module = nullptr;
    OptixPipeline trace_pipeline = nullptr;
    OptixProgramGroup trace_pg_raygen = nullptr;
    OptixProgramGroup trace_pg_raygen_composite = nullptr;
    OptixProgramGroup trace_pg_miss = nullptr;
    OptixProgramGroup trace_pg_hit_composite = nullptr;
    OptixProgramGroup trace_pg_hit_intersect = nullptr;
    void *trace_sbt_raygen = nullptr;
    void *trace_sbt_raygen_composite = nullptr;
    void *trace_sbt_miss = nullptr;
    void *trace_sbt_hitgroup = nullptr;
    void *trace_params_buffer = nullptr;

    int vertex_count = 0;
    int triangle_count = 0;

    void *trace_raygen_record(SurfelOptixLaunchKind kind) const {
        return kind == SurfelOptixLaunchKind::Composite
            ? trace_sbt_raygen_composite
            : trace_sbt_raygen;
    }

    void launch_trace(SurfelOptixLaunchKind kind, const SurfelTraceParams &params) const {
        audit_jit_memcpy_async();
        jit_memcpy_async(JitBackend::CUDA,
                         trace_params_buffer,
                         &params,
                         sizeof(SurfelTraceParams));

        OptixShaderBindingTable trace_sbt = {};
        trace_sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(trace_raygen_record(kind));
        trace_sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(trace_sbt_miss);
        trace_sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
        trace_sbt.missRecordCount = 1;
        trace_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(trace_sbt_hitgroup);
        trace_sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        trace_sbt.hitgroupRecordCount = 2;

        audit_optix_launch();
        check_optix(optixLaunch(trace_pipeline,
                                jit_cuda_stream(),
                                reinterpret_cast<CUdeviceptr>(trace_params_buffer),
                                sizeof(SurfelTraceParams),
                                &trace_sbt,
                                static_cast<unsigned int>(params.ray_count),
                                1,
                                1),
                    "optixLaunch(surfel trace)");
    }
};

static void retire_surfel_optix_jit_resources(SurfelOptixState *state) {
    if (state == nullptr || state->pipeline_handle.index() == 0 || state->sbt_handle.index() == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(retired_surfel_optix_resources_mutex());
    auto &resources = retired_surfel_optix_resources();
    resources.push_back({ state->pipeline_handle, state->sbt_handle });

    state->pipeline_handle = UInt();
    state->sbt_handle = UInt();

    constexpr size_t MaxRetiredOptixResourceSets = 32;
    if (resources.size() >= MaxRetiredOptixResourceSets) {
        jit_flush_kernel_cache();
        resources.clear();
    }
}

static void destroy_surfel_optix_state(SurfelOptixState *state) {
    if (state == nullptr) {
        return;
    }

    jit_sync_thread();

    if (state->trace_pipeline != nullptr && optixPipelineDestroy != nullptr) {
        optixPipelineDestroy(state->trace_pipeline);
    }
    if (state->trace_pg_hit_intersect != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_hit_intersect);
    }
    if (state->trace_pg_hit_composite != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_hit_composite);
    }
    if (state->trace_pg_miss != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_miss);
    }
    if (state->trace_pg_raygen != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_raygen);
    }
    if (state->trace_pg_raygen_composite != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(state->trace_pg_raygen_composite);
    }
    if (state->trace_module != nullptr && optixModuleDestroy != nullptr) {
        optixModuleDestroy(state->trace_module);
    }
    if (state->trace_params_buffer != nullptr) {
        jit_free(state->trace_params_buffer);
        state->trace_params_buffer = nullptr;
    }
    if (state->trace_sbt_hitgroup != nullptr) {
        jit_free(state->trace_sbt_hitgroup);
        state->trace_sbt_hitgroup = nullptr;
    }
    if (state->trace_sbt_miss != nullptr) {
        jit_free(state->trace_sbt_miss);
        state->trace_sbt_miss = nullptr;
    }
    if (state->trace_sbt_raygen_composite != nullptr) {
        jit_free(state->trace_sbt_raygen_composite);
        state->trace_sbt_raygen_composite = nullptr;
    }
    if (state->trace_sbt_raygen != nullptr) {
        jit_free(state->trace_sbt_raygen);
        state->trace_sbt_raygen = nullptr;
    }

    retire_surfel_optix_jit_resources(state);

    if (state->vertex_buffer != nullptr) {
        jit_free(state->vertex_buffer);
        state->vertex_buffer = nullptr;
        state->vertex_buffer_ptr = nullptr;
    }
    if (state->gas_temp_buffer != nullptr) {
        jit_free(state->gas_temp_buffer);
        state->gas_temp_buffer = nullptr;
    }
    if (state->gas_buffer != nullptr) {
        jit_free(state->gas_buffer);
        state->gas_buffer = nullptr;
    }
    delete state;
}

static void ensure_surfel_trace_pipeline(SurfelOptixState *state) {
    require(state != nullptr, "ensure_surfel_trace_pipeline(): state is null.");
    if (state->trace_pipeline != nullptr) {
        return;
    }

    init_optix_api();
    state->context = jit_optix_context();

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = RAYD_OPTIX_MODULE_OPT_LEVEL;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptionsDirect pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 1;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.pipelineLaunchParamsSizeInBytes = sizeof(SurfelTraceParams);
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipeline_options.allowOpacityMicromaps = 0;
    pipeline_options.allowClusteredGeometry = 0;

    char log[2048];
    size_t log_size = sizeof(log);
    check_optix(optixModuleCreate(state->context,
                                  &module_options,
                                  direct_optix_pipeline_compile_options(pipeline_options),
                                  surfel_trace_ptx,
                                  surfel_trace_ptx_size,
                                  log,
                                  &log_size,
                                  &state->trace_module),
                "optixModuleCreate(surfel trace)");

    state->trace_pg_raygen =
        make_raygen_group(state->context, state->trace_module, "__raygen__surfel_trace");
    state->trace_pg_raygen_composite =
        make_raygen_group(state->context, state->trace_module, "__raygen__surfel_composite");
    state->trace_pg_miss =
        make_miss_group(state->context, state->trace_module, "__miss__surfel_trace");
    state->trace_pg_hit_composite =
        make_hitgroup(state->context,
                      state->trace_module,
                      nullptr,
                      "__anyhit__surfel_composite",
                      nullptr);
    state->trace_pg_hit_intersect =
        make_hitgroup(state->context,
                      state->trace_module,
                      nullptr,
                      "__anyhit__surfel_intersect",
                      nullptr);

    OptixProgramGroup groups[] = {
        state->trace_pg_raygen,
        state->trace_pg_raygen_composite,
        state->trace_pg_miss,
        state->trace_pg_hit_composite,
        state->trace_pg_hit_intersect,
    };
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    link_options.maxContinuationCallableDepth = 0;
    link_options.maxDirectCallableDepthFromState = 0;
    link_options.maxDirectCallableDepthFromTraversal = 0;
    link_options.maxTraversableGraphDepth = 1;

    log_size = sizeof(log);
    check_optix(optixPipelineCreate(state->context,
                                    direct_optix_pipeline_compile_options(pipeline_options),
                                    &link_options,
                                    groups,
                                    5,
                                    log,
                                    &log_size,
                                    &state->trace_pipeline),
                "optixPipelineCreate(surfel trace)");
    check_optix(optixPipelineSetStackSize(state->trace_pipeline,
                                          0,
                                          0,
                                          4096,
                                          1),
                "optixPipelineSetStackSize(surfel trace)");

    state->trace_sbt_raygen = make_sbt_record(state->trace_pg_raygen);
    state->trace_sbt_raygen_composite = make_sbt_record(state->trace_pg_raygen_composite);
    state->trace_sbt_miss = make_sbt_record(state->trace_pg_miss);
    std::vector<EmptySbtRecord> hitgroups(2);
    check_optix(optixSbtRecordPackHeader(state->trace_pg_hit_composite, &hitgroups[0]),
                "optixSbtRecordPackHeader(surfel composite hitgroup)");
    check_optix(optixSbtRecordPackHeader(state->trace_pg_hit_intersect, &hitgroups[1]),
                "optixSbtRecordPackHeader(surfel intersect hitgroup)");
    state->trace_sbt_hitgroup = jit_malloc(AllocType::Device,
                                           sizeof(EmptySbtRecord) * hitgroups.size());
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA,
               state->trace_sbt_hitgroup,
               hitgroups.data(),
               sizeof(EmptySbtRecord) * hitgroups.size());
    state->trace_params_buffer = jit_malloc(AllocType::Device, sizeof(SurfelTraceParams));
}

void SurfelOptixIntersection::reserve(int64_t size) {
    require(size >= 0, "SurfelOptixIntersection::reserve(): size must be non-negative.");
    if (size != m_size) {
        m_size = size;
        triangle_id = empty<Int>(size);
        barycentric = empty<Vector2f>(size);
        t = empty<Float>(size);
    }
}

void SurfelOptixComposite::reserve(int64_t size) {
    require(size >= 0, "SurfelOptixComposite::reserve(): size must be non-negative.");
    if (size != m_size) {
        m_size = size;
        hit_capacity = 0;
        intensity = zeros<Float>(size);
        channels = Float();
        normal = zeros<Vector3f>(size);
        alpha = zeros<Float>(size);
        transmittance = full<Float>(1.f, size);
        depth = full<Float>(Infinity, size);
        surfel_id = Int();
        hit_t = Float();
        hit_alpha = Float();
        hit_value = Float();
        candidate_count = zeros<Int>(size);
        candidate_buffer_full = full<Mask>(false, size);
    }
}

SurfelOptixScene::SurfelOptixScene() = default;

SurfelOptixScene::~SurfelOptixScene() {
    destroy_surfel_optix_state(m_accel);
}

void SurfelOptixScene::build(const Float &vertex_buffer,
                             const Int &face_buffer,
                             int vertex_count,
                             int triangle_count,
                             bool build_hitobject_pipeline) {
    require(vertex_count > 0, "SurfelOptixScene::build(): vertex_count must be positive.");
    require(triangle_count > 0, "SurfelOptixScene::build(): triangle_count must be positive.");

    destroy_surfel_optix_state(m_accel);

    init_optix_api();
    m_accel = new SurfelOptixState();
    m_accel->context = jit_optix_context();
    m_accel->vertex_count = vertex_count;
    m_accel->triangle_count = triangle_count;

    if (build_hitobject_pipeline) {
        m_accel->pipeline_compile_options.usesMotionBlur = false;
        m_accel->pipeline_compile_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        m_accel->pipeline_compile_options.numPayloadValues = 0;
        m_accel->pipeline_compile_options.numAttributeValues = 2;
        m_accel->pipeline_compile_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;
        m_accel->pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        m_accel->pipeline_compile_options.usesPrimitiveTypeFlags =
            static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
        m_accel->pipeline_compile_options.allowOpacityMicromaps = 0;

        std::memset(m_accel->pgd, 0, sizeof(m_accel->pgd));
        m_accel->pgd[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        m_accel->pgd[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        char log[1024];
        size_t log_size = sizeof(log);
        jit_optix_check(optixProgramGroupCreate(m_accel->context,
                                                m_accel->pgd,
                                                2,
                                                &m_accel->pgo,
                                                log,
                                                &log_size,
                                                m_accel->pg));

        m_accel->sbt.missRecordBase =
            reinterpret_cast<CUdeviceptr>(jit_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE));
        m_accel->sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        m_accel->sbt.missRecordCount = 1;
        jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[0],
                                                 reinterpret_cast<void *>(m_accel->sbt.missRecordBase)));

        EmptySbtRecord *hit_record =
            reinterpret_cast<EmptySbtRecord *>(jit_malloc(AllocType::HostPinned, sizeof(EmptySbtRecord)));
        jit_optix_check(optixSbtRecordPackHeader(m_accel->pg[1], hit_record));
        m_accel->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hit_record);
        m_accel->sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
        m_accel->sbt.hitgroupRecordCount = 1;

        m_accel->sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(
            jit_malloc_migrate(reinterpret_cast<void *>(m_accel->sbt.missRecordBase), AllocType::Device, 1));
        m_accel->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(
            jit_malloc_migrate(reinterpret_cast<void *>(m_accel->sbt.hitgroupRecordBase), AllocType::Device, 1));

        m_accel->pipeline_handle = UInt::steal(jit_optix_configure_pipeline(
            &m_accel->pipeline_compile_options,
            nullptr,
            m_accel->pg,
            2));
        m_accel->sbt_handle = UInt::steal(
            jit_optix_configure_sbt(&m_accel->sbt, m_accel->pipeline_handle.index()));
    }

    m_accel->vertex_buffer = jit_malloc(AllocType::Device, sizeof(float) * vertex_count * 3);
    m_accel->vertex_buffer_ptr = m_accel->vertex_buffer;
    jit_memcpy(JitBackend::CUDA,
               m_accel->vertex_buffer,
               vertex_buffer.data(),
               sizeof(float) * vertex_count * 3);

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers =
        reinterpret_cast<const CUdeviceptr *>(&m_accel->vertex_buffer_ptr);
    build_input.triangleArray.numVertices = static_cast<unsigned int>(vertex_count);
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    build_input.triangleArray.indexBuffer =
        reinterpret_cast<CUdeviceptr>(const_cast<int *>(face_buffer.data()));
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(triangle_count);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    build_input.triangleArray.preTransform = nullptr;
    build_input.triangleArray.numSbtRecords = 1;
    build_input.triangleArray.sbtIndexOffsetBuffer = nullptr;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    build_input.triangleArray.primitiveIndexOffset = 0;
    build_input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

    unsigned int triangle_input_flags[] = { 0u };
    build_input.triangleArray.flags = triangle_input_flags;

    m_accel->accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    m_accel->accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    jit_optix_check(optixAccelComputeMemoryUsage(m_accel->context,
                                                 &m_accel->accel_options,
                                                 &build_input,
                                                 1,
                                                 &m_accel->gas_buffer_sizes));
    m_accel->gas_temp_buffer =
        jit_malloc(AllocType::Device, m_accel->gas_buffer_sizes.tempSizeInBytes);
    m_accel->gas_temp_buffer_size = m_accel->gas_buffer_sizes.tempSizeInBytes;
    m_accel->gas_buffer =
        jit_malloc(AllocType::Device, m_accel->gas_buffer_sizes.outputSizeInBytes);
    m_accel->gas_buffer_size = m_accel->gas_buffer_sizes.outputSizeInBytes;

    jit_optix_check(optixAccelBuild(m_accel->context,
                                    jit_cuda_stream(),
                                    &m_accel->accel_options,
                                    &build_input,
                                    1,
                                    m_accel->gas_temp_buffer,
                                    m_accel->gas_buffer_sizes.tempSizeInBytes,
                                    m_accel->gas_buffer,
                                    m_accel->gas_buffer_sizes.outputSizeInBytes,
                                    &m_accel->gas_handle,
                                    nullptr,
                                    0));
}

bool SurfelOptixScene::is_ready() const {
    return m_accel != nullptr;
}

template <bool Detached>
SurfelOptixIntersection SurfelOptixScene::intersect(const RayT<Detached> &ray,
                                                    MaskT<Detached> &active) const {
    return intersect<Detached>(ray, FloatT<Detached>(RayEpsilon), active);
}

template <bool Detached>
SurfelOptixIntersection SurfelOptixScene::intersect(const RayT<Detached> &ray,
                                                    const FloatT<Detached> &t_min_input,
                                                    MaskT<Detached> &active) const {
    require(m_accel != nullptr, "SurfelOptixScene::intersect(): scene is not built.");
    require(m_accel->pipeline_handle.index() != 0 && m_accel->sbt_handle.index() != 0,
            "SurfelOptixScene::intersect(): legacy HitObject pipeline is not built.");
    const int ray_count = static_cast<int>(slices(ray.o));

    SurfelOptixIntersection intersection;
    intersection.reserve(ray_count);

    Float ox;
    Float oy;
    Float oz;
    Float dx;
    Float dy;
    Float dz;
    Float t_min;
    Float t_max_input;
    if constexpr (!Detached) {
        ox = detach<false>(ray.o.x());
        oy = detach<false>(ray.o.y());
        oz = detach<false>(ray.o.z());
        dx = detach<false>(ray.d.x());
        dy = detach<false>(ray.d.y());
        dz = detach<false>(ray.d.z());
        t_min = maximum(detach<false>(t_min_input), Float(RayEpsilon));
        t_max_input = detach<false>(ray.tmax);
    } else {
        ox = ray.o.x();
        oy = ray.o.y();
        oz = ray.o.z();
        dx = ray.d.x();
        dy = ray.d.y();
        dz = ray.d.z();
        t_min = maximum(t_min_input, Float(RayEpsilon));
        t_max_input = ray.tmax;
    }

    Mask active_detached = detach<false>(active);

    Float t_max = select(drjit::isfinite(t_max_input), t_max_input, full<Float>(1e8f, ray_count));
    Float time = 0.f;
    UInt ray_mask(255);
    UInt ray_flags(OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
    UInt sbt_offset(0);
    UInt sbt_stride(1);
    UInt miss_sbt_index(0);

    m_accel->handle = dr::opaque<UInt64>(m_accel->gas_handle);
    uint32_t trace_args[] {
        m_accel->handle.index(),
        ox.index(), oy.index(), oz.index(),
        dx.index(), dy.index(), dz.index(),
        t_min.index(), t_max.index(), time.index(),
        ray_mask.index(), ray_flags.index(),
        sbt_offset.index(), sbt_stride.index(),
        miss_sbt_index.index(),
    };

    OptixHitObjectField fields[] {
        OptixHitObjectField::IsHit,
        OptixHitObjectField::RayTMax,
        OptixHitObjectField::Attribute0,
        OptixHitObjectField::Attribute1,
        OptixHitObjectField::PrimitiveIndex,
    };
    uint32_t hitobject_out[5];

    jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t),
                        trace_args,
                        5,
                        fields,
                        hitobject_out,
                        0, 0, 0,
                        0,
                        active_detached.index(),
                        m_accel->pipeline_handle.index(),
                        m_accel->sbt_handle.index());

    Mask is_hit = UInt::steal(hitobject_out[0]) != 0u;
    active_detached &= is_hit;

    using Single = drjit::float32_array_t<Float>;
    intersection.t =
        drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[1]));
    intersection.barycentric[0] =
        drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[2]));
    intersection.barycentric[1] =
        drjit::reinterpret_array<Single, UInt>(UInt::steal(hitobject_out[3]));
    intersection.triangle_id = Int(UInt::steal(hitobject_out[4]));

    intersection.t[!active_detached] = Infinity;
    intersection.triangle_id[!active_detached] = -1;

    if constexpr (!Detached) {
        active &= MaskAD(active_detached);
    } else {
        active = active_detached;
    }
    return intersection;
}

template <bool Detached>
SurfelOptixIntersection SurfelOptixScene::trace_analytic_candidates(
    const RayT<Detached> &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    MaskT<Detached> &active) const {
    require(m_accel != nullptr, "SurfelOptixScene::trace_analytic_candidates(): scene is not built.");
    require(max_candidate_hits > 0,
            "SurfelOptixScene::trace_analytic_candidates(): max_candidate_hits must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    SurfelOptixIntersection intersection;
    intersection.reserve(ray_count);
    intersection.triangle_id = full<Int>(-1, ray_count);
    intersection.barycentric = zeros<Vector2f>(ray_count);
    intersection.t = full<Float>(Infinity, ray_count);
    if (ray_count == 0) {
        if constexpr (!Detached) {
            active &= false;
        } else {
            active = false;
        }
        return intersection;
    }

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

    const Float zero = zeros<Float>(ray_count);
    ox += zero;
    oy += zero;
    oz += zero;
    dx += zero;
    dy += zero;
    dz += zero;
    t_max_input += zero;

    Mask active_detached = detach<false>(active);
    active_detached = active_detached && full<Mask>(true, ray_count);
    Mask valid = empty<Mask>(ray_count);

    drjit::eval(ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                t_max_input,
                active_detached,
                triangle_to_surfel_id,
                center,
                tangent_u,
                tangent_v,
                opacity);

    ensure_surfel_trace_pipeline(m_accel);

    SurfelTraceParams params = {};
    params.handle = m_accel->gas_handle;
    params.ray_ox = ox.data();
    params.ray_oy = oy.data();
    params.ray_oz = oz.data();
    params.ray_dx = dx.data();
    params.ray_dy = dy.data();
    params.ray_dz = dz.data();
    params.ray_tmax = t_max_input.data();
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.ray_count = ray_count;
    params.triangle_to_surfel_id = triangle_to_surfel_id.data();
    params.triangle_count = m_accel->triangle_count;
    params.surfel_count = static_cast<int>(slices(center));
    params.center_x = center.x().data();
    params.center_y = center.y().data();
    params.center_z = center.z().data();
    params.tangent_u_x = tangent_u.x().data();
    params.tangent_u_y = tangent_u.y().data();
    params.tangent_u_z = tangent_u.z().data();
    params.tangent_v_x = tangent_v.x().data();
    params.tangent_v_y = tangent_v.y().data();
    params.tangent_v_z = tangent_v.z().data();
    params.opacity = opacity.data();
    params.alpha_min = alpha_min;
    params.alpha_cap = alpha_cap;
    params.ray_epsilon = RayEpsilon;
    params.tmax_fallback = 1.0e8f;
    params.max_candidate_hits = max_candidate_hits;
    params.face_forward = face_forward ? 1 : 0;
    params.out_triangle_id = intersection.triangle_id.data();
    params.out_proxy_t = intersection.t.data();
    params.out_valid = reinterpret_cast<uint8_t *>(valid.data());

    {
        ScopedNativeLaunchStage stage(NativeLaunchStage::SurfelTrace);
        m_accel->launch_trace(SurfelOptixLaunchKind::Intersect, params);
    }

    const Mask hit = valid && (intersection.triangle_id >= 0);
    intersection.t[!hit] = Infinity;
    intersection.triangle_id[!hit] = -1;

    active_detached &= hit;
    if constexpr (!Detached) {
        active &= MaskAD(active_detached);
    } else {
        active = active_detached;
    }
    return intersection;
}

template <bool Detached>
SurfelOptixComposite SurfelOptixScene::composite_alpha(
    const RayT<Detached> &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &value,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool collect_candidate_stats,
    bool continue_after_full_buffer,
    float transmittance_min,
    int max_trace_segments,
    bool face_forward,
    MaskT<Detached> active) const {
    require(m_accel != nullptr, "SurfelOptixScene::composite_alpha(): scene is not built.");
    require(max_candidate_hits > 0,
            "SurfelOptixScene::composite_alpha(): max_candidate_hits must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    SurfelOptixComposite composite;
    composite.reserve(ray_count);
    if (ray_count == 0) {
        return composite;
    }

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

    const Float zero = zeros<Float>(ray_count);
    ox += zero;
    oy += zero;
    oz += zero;
    dx += zero;
    dy += zero;
    dz += zero;
    t_max_input += zero;

    Mask active_detached = detach<false>(active);
    active_detached = active_detached && full<Mask>(true, ray_count);
    Mask valid = empty<Mask>(ray_count);

    const int k = max_candidate_hits;
    const int scratch_count = ray_count * k;
    Int scratch_surfel_id = empty<Int>(scratch_count);
    Float scratch_t = empty<Float>(scratch_count);
    Float scratch_alpha = empty<Float>(scratch_count);
    Float scratch_value = empty<Float>(scratch_count);
    composite.channels = empty<Float>(ray_count);
    Int candidate_count = collect_candidate_stats ? empty<Int>(ray_count) : zeros<Int>(ray_count);
    Mask candidate_buffer_full = collect_candidate_stats
        ? empty<Mask>(ray_count)
        : full<Mask>(false, ray_count);

    drjit::eval(ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                t_max_input,
                active_detached,
                triangle_to_surfel_id,
                center,
                tangent_u,
                tangent_v,
                opacity,
                value);

    ensure_surfel_trace_pipeline(m_accel);

    SurfelTraceParams params = {};
    params.handle = m_accel->gas_handle;
    params.ray_ox = ox.data();
    params.ray_oy = oy.data();
    params.ray_oz = oz.data();
    params.ray_dx = dx.data();
    params.ray_dy = dy.data();
    params.ray_dz = dz.data();
    params.ray_tmax = t_max_input.data();
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.ray_count = ray_count;
    params.triangle_to_surfel_id = triangle_to_surfel_id.data();
    params.triangle_count = m_accel->triangle_count;
    params.surfel_count = static_cast<int>(slices(center));
    params.center_x = center.x().data();
    params.center_y = center.y().data();
    params.center_z = center.z().data();
    params.tangent_u_x = tangent_u.x().data();
    params.tangent_u_y = tangent_u.y().data();
    params.tangent_u_z = tangent_u.z().data();
    params.tangent_v_x = tangent_v.x().data();
    params.tangent_v_y = tangent_v.y().data();
    params.tangent_v_z = tangent_v.z().data();
    params.opacity = opacity.data();
    params.value = value.data();
    params.appearance_values = value.data();
    params.appearance_channel_count = 1;
    params.color_model = 2;
    params.sh_degree = 0;
    params.render_channel_count = 1;
    params.alpha_min = alpha_min;
    params.alpha_cap = alpha_cap;
    params.ray_epsilon = RayEpsilon;
    params.tmax_fallback = 1.0e8f;
    params.max_candidate_hits = max_candidate_hits;
    params.face_forward = face_forward ? 1 : 0;
    params.collect_candidate_stats = collect_candidate_stats ? 1 : 0;
    params.continue_after_full_buffer = continue_after_full_buffer ? 1 : 0;
    params.transmittance_min = transmittance_min;
    params.max_trace_segments = max_trace_segments;
    params.out_valid = reinterpret_cast<uint8_t *>(valid.data());
    params.composite_hit_capacity = k;
    params.scratch_surfel_id = scratch_surfel_id.data();
    params.scratch_t = scratch_t.data();
    params.scratch_alpha = scratch_alpha.data();
    params.scratch_value = scratch_value.data();
    params.out_intensity = composite.intensity.data();
    params.out_channels = composite.channels.data();
    params.out_alpha = composite.alpha.data();
    params.out_transmittance = composite.transmittance.data();
    params.out_depth = composite.depth.data();
    params.out_candidate_count = collect_candidate_stats ? candidate_count.data() : nullptr;
    params.out_candidate_buffer_full = collect_candidate_stats
        ? reinterpret_cast<uint8_t *>(candidate_buffer_full.data())
        : nullptr;

    {
        ScopedNativeLaunchStage stage(NativeLaunchStage::SurfelTrace);
        m_accel->launch_trace(SurfelOptixLaunchKind::Composite, params);
    }

    composite.hit_capacity = k;
    composite.surfel_id = scratch_surfel_id;
    composite.hit_t = scratch_t;
    composite.hit_alpha = scratch_alpha;
    composite.hit_value = scratch_value;
    composite.candidate_count = candidate_count;
    composite.candidate_buffer_full = candidate_buffer_full;
    return composite;
}

template <bool Detached>
SurfelOptixComposite SurfelOptixScene::render(
    const RayT<Detached> &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &appearance_values,
    int appearance_channel_count,
    int color_model,
    int appearance_sh_degree,
    int sh_degree,
    int render_channel_count,
    bool output_normal,
    const ScalarVector3f &background_rgb,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool collect_candidate_stats,
    bool continue_after_full_buffer,
    float transmittance_min,
    int max_trace_segments,
    bool face_forward,
    MaskT<Detached> active) const {
    require(m_accel != nullptr, "SurfelOptixScene::render(): scene is not built.");
    require(max_candidate_hits > 0,
            "SurfelOptixScene::render(): max_candidate_hits must be positive.");
    require(render_channel_count >= 0,
            "SurfelOptixScene::render(): render_channel_count must be non-negative.");

    const int ray_count = static_cast<int>(slices(ray.o));
    SurfelOptixComposite composite;
    composite.reserve(ray_count);
    if (ray_count == 0) {
        return composite;
    }

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

    const Float zero = zeros<Float>(ray_count);
    ox += zero;
    oy += zero;
    oz += zero;
    dx += zero;
    dy += zero;
    dz += zero;
    t_max_input += zero;

    Mask active_detached = detach<false>(active);
    active_detached = active_detached && full<Mask>(true, ray_count);
    Mask valid = empty<Mask>(ray_count);

    const int k = max_candidate_hits;
    const int scratch_count = ray_count * k;
    Int scratch_surfel_id = empty<Int>(scratch_count);
    Float scratch_t = empty<Float>(scratch_count);
    Float scratch_alpha = empty<Float>(scratch_count);
    Float scratch_value = empty<Float>(scratch_count);
    composite.channels = render_channel_count > 0
        ? empty<Float>(ray_count * render_channel_count)
        : Float();
    composite.normal = output_normal ? empty<Vector3f>(ray_count) : zeros<Vector3f>(ray_count);
    Int candidate_count = collect_candidate_stats ? empty<Int>(ray_count) : zeros<Int>(ray_count);
    Mask candidate_buffer_full = collect_candidate_stats
        ? empty<Mask>(ray_count)
        : full<Mask>(false, ray_count);

    drjit::eval(ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                t_max_input,
                active_detached,
                triangle_to_surfel_id,
                center,
                tangent_u,
                tangent_v,
                opacity,
                appearance_values);

    ensure_surfel_trace_pipeline(m_accel);

    SurfelTraceParams params = {};
    params.handle = m_accel->gas_handle;
    params.ray_ox = ox.data();
    params.ray_oy = oy.data();
    params.ray_oz = oz.data();
    params.ray_dx = dx.data();
    params.ray_dy = dy.data();
    params.ray_dz = dz.data();
    params.ray_tmax = t_max_input.data();
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.ray_count = ray_count;
    params.triangle_to_surfel_id = triangle_to_surfel_id.data();
    params.triangle_count = m_accel->triangle_count;
    params.surfel_count = static_cast<int>(slices(center));
    params.center_x = center.x().data();
    params.center_y = center.y().data();
    params.center_z = center.z().data();
    params.tangent_u_x = tangent_u.x().data();
    params.tangent_u_y = tangent_u.y().data();
    params.tangent_u_z = tangent_u.z().data();
    params.tangent_v_x = tangent_v.x().data();
    params.tangent_v_y = tangent_v.y().data();
    params.tangent_v_z = tangent_v.z().data();
    params.opacity = opacity.data();
    params.value = appearance_values.data();
    params.appearance_values = appearance_values.data();
    params.appearance_channel_count = appearance_channel_count;
    params.color_model = color_model;
    params.appearance_sh_degree = appearance_sh_degree;
    params.sh_degree = sh_degree;
    params.render_channel_count = render_channel_count;
    params.output_normal = output_normal ? 1 : 0;
    params.background_rgb[0] = background_rgb[0];
    params.background_rgb[1] = background_rgb[1];
    params.background_rgb[2] = background_rgb[2];
    params.alpha_min = alpha_min;
    params.alpha_cap = alpha_cap;
    params.ray_epsilon = RayEpsilon;
    params.tmax_fallback = 1.0e8f;
    params.max_candidate_hits = max_candidate_hits;
    params.face_forward = face_forward ? 1 : 0;
    params.collect_candidate_stats = collect_candidate_stats ? 1 : 0;
    params.continue_after_full_buffer = continue_after_full_buffer ? 1 : 0;
    params.transmittance_min = transmittance_min;
    params.max_trace_segments = max_trace_segments;
    params.out_valid = reinterpret_cast<uint8_t *>(valid.data());
    params.composite_hit_capacity = k;
    params.scratch_surfel_id = scratch_surfel_id.data();
    params.scratch_t = scratch_t.data();
    params.scratch_alpha = scratch_alpha.data();
    params.scratch_value = scratch_value.data();
    params.out_intensity = composite.intensity.data();
    params.out_channels = render_channel_count > 0 ? composite.channels.data() : nullptr;
    params.out_normal_x = output_normal ? composite.normal.x().data() : nullptr;
    params.out_normal_y = output_normal ? composite.normal.y().data() : nullptr;
    params.out_normal_z = output_normal ? composite.normal.z().data() : nullptr;
    params.out_alpha = composite.alpha.data();
    params.out_transmittance = composite.transmittance.data();
    params.out_depth = composite.depth.data();
    params.out_candidate_count = collect_candidate_stats ? candidate_count.data() : nullptr;
    params.out_candidate_buffer_full = collect_candidate_stats
        ? reinterpret_cast<uint8_t *>(candidate_buffer_full.data())
        : nullptr;

    {
        ScopedNativeLaunchStage stage(NativeLaunchStage::SurfelTrace);
        m_accel->launch_trace(SurfelOptixLaunchKind::Composite, params);
    }

    composite.hit_capacity = k;
    composite.surfel_id = scratch_surfel_id;
    composite.hit_t = scratch_t;
    composite.hit_alpha = scratch_alpha;
    composite.hit_value = scratch_value;
    composite.candidate_count = candidate_count;
    composite.candidate_buffer_full = candidate_buffer_full;
    return composite;
}

template SurfelOptixIntersection SurfelOptixScene::intersect<true>(const Ray &ray, Mask &active) const;
template SurfelOptixIntersection SurfelOptixScene::intersect<false>(const RayAD &ray, MaskAD &active) const;
template SurfelOptixIntersection SurfelOptixScene::intersect<true>(const Ray &ray, const Float &t_min, Mask &active) const;
template SurfelOptixIntersection SurfelOptixScene::intersect<false>(const RayAD &ray, const FloatAD &t_min, MaskAD &active) const;
template SurfelOptixIntersection SurfelOptixScene::trace_analytic_candidates<true>(
    const Ray &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    Mask &active) const;
template SurfelOptixIntersection SurfelOptixScene::trace_analytic_candidates<false>(
    const RayAD &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool face_forward,
    MaskAD &active) const;
template SurfelOptixComposite SurfelOptixScene::composite_alpha<true>(
    const Ray &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &value,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool collect_candidate_stats,
    bool continue_after_full_buffer,
    float transmittance_min,
    int max_trace_segments,
    bool face_forward,
    Mask active) const;
template SurfelOptixComposite SurfelOptixScene::composite_alpha<false>(
    const RayAD &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &value,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool collect_candidate_stats,
    bool continue_after_full_buffer,
    float transmittance_min,
    int max_trace_segments,
    bool face_forward,
    MaskAD active) const;
template SurfelOptixComposite SurfelOptixScene::render<true>(
    const Ray &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &appearance_values,
    int appearance_channel_count,
    int color_model,
    int appearance_sh_degree,
    int sh_degree,
    int render_channel_count,
    bool output_normal,
    const ScalarVector3f &background_rgb,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool collect_candidate_stats,
    bool continue_after_full_buffer,
    float transmittance_min,
    int max_trace_segments,
    bool face_forward,
    Mask active) const;
template SurfelOptixComposite SurfelOptixScene::render<false>(
    const RayAD &ray,
    const Int &triangle_to_surfel_id,
    const Vector3f &center,
    const Vector3f &tangent_u,
    const Vector3f &tangent_v,
    const Float &opacity,
    const Float &appearance_values,
    int appearance_channel_count,
    int color_model,
    int appearance_sh_degree,
    int sh_degree,
    int render_channel_count,
    bool output_normal,
    const ScalarVector3f &background_rgb,
    float alpha_min,
    float alpha_cap,
    int max_candidate_hits,
    bool collect_candidate_stats,
    bool continue_after_full_buffer,
    float transmittance_min,
    int max_trace_segments,
    bool face_forward,
    MaskAD active) const;

} // namespace rayd
