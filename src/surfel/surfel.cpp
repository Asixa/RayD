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
    valid &= basis_valid && (alpha_uncapped >= FloatT<Detached>(options.alpha_min));

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

} // namespace

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
    : cloud_(cloud), options_(options) {
    require(cloud_.surfel_count() > 0, "SurfelScene(): cloud must contain at least one surfel.");
    require(options_.alpha_min > 0.f && options_.alpha_min < 1.f,
            "SurfelScene(): alpha_min must be in (0, 1).");
    require(options_.cutoff > 0.f, "SurfelScene(): cutoff must be positive.");
    require(options_.alpha_cap > 0.f && options_.alpha_cap <= 1.f,
            "SurfelScene(): alpha_cap must be in (0, 1].");
    require(options_.proxy_epsilon > 0.f, "SurfelScene(): proxy_epsilon must be positive.");
    require(options_.max_candidate_hits > 0,
            "SurfelScene(): max_candidate_hits must be positive.");
}

void SurfelScene::build() {
    require(cloud_.surfel_count() > 0, "SurfelScene::build(): cloud is empty.");
    require(options_.alpha_min > 0.f && options_.alpha_min < 1.f,
            "SurfelScene::build(): alpha_min must be in (0, 1).");
    require(options_.cutoff > 0.f, "SurfelScene::build(): cutoff must be positive.");
    require(options_.alpha_cap > 0.f && options_.alpha_cap <= 1.f,
            "SurfelScene::build(): alpha_cap must be in (0, 1].");
    require(options_.proxy_epsilon > 0.f, "SurfelScene::build(): proxy_epsilon must be positive.");
    require(options_.max_candidate_hits > 0,
            "SurfelScene::build(): max_candidate_hits must be positive.");

    build_triangle_buffers();
    eval(optix_vertex_buffer_, optix_face_buffer_, triangle_to_surfel_id_);
    optix_scene_.build(optix_vertex_buffer_, optix_face_buffer_, vertex_count_, triangle_count_);
    ready_ = true;
}

void SurfelScene::build_triangle_buffers() {
    const int surfel_count = cloud_.surfel_count();
    const bool use_icosahedron = options_.primitive_mode == SurfelPrimitiveMode::Icosahedron20;
    const bool use_quad = options_.primitive_mode == SurfelPrimitiveMode::QuadTriangles;
    const int vertices_per_surfel = use_icosahedron ? IcosahedronVertexCount : (use_quad ? 4 : 3);
    const int triangles_per_surfel = use_icosahedron ? IcosahedronFaceCount : (use_quad ? 2 : 1);
    vertex_count_ = surfel_count * vertices_per_surfel;
    triangle_count_ = surfel_count * triangles_per_surfel;

    Vector3fAD vertices_ad = empty<Vector3fAD>(vertex_count_);
    const IntAD base = arange<IntAD>(surfel_count) * vertices_per_surfel;
    const Vector3fAD &center = cloud_.center();
    const Vector3fAD &u = cloud_.tangent_u();
    const Vector3fAD &v = cloud_.tangent_v();
    const FloatAD alpha_min(options_.alpha_min);
    const FloatAD safe_opacity = maximum(cloud_.opacity(), alpha_min);
    FloatAD influence_radius =
        sqrt(maximum(FloatAD(0.f), FloatAD(2.f) * log(safe_opacity / alpha_min)));
    if (std::isfinite(options_.cutoff)) {
        influence_radius = minimum(influence_radius, FloatAD(options_.cutoff));
    }

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
                influence_radius * (local_x * u + local_y * v) +
                influence_radius * FloatAD(options_.proxy_epsilon) * local_z * normal;
            scatter(vertices_ad, proxy_vertex, base + vertex);
        }
    } else if (use_quad) {
        scatter(vertices_ad, center - influence_radius * u - influence_radius * v, base + 0);
        scatter(vertices_ad, center + influence_radius * u - influence_radius * v, base + 1);
        scatter(vertices_ad, center + influence_radius * u + influence_radius * v, base + 2);
        scatter(vertices_ad, center - influence_radius * u + influence_radius * v, base + 3);
    } else {
        scatter(vertices_ad, center + (-2.f * influence_radius) * u - influence_radius * v, base + 0);
        scatter(vertices_ad, center + ( 2.f * influence_radius) * u - influence_radius * v, base + 1);
        scatter(vertices_ad, center + ( 3.f * influence_radius) * v, base + 2);
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
            center = gather<Vector3fAD>(cloud_.center(), surfel_id_ad, hit_mask);
            tangent_u = gather<Vector3fAD>(cloud_.tangent_u(), surfel_id_ad, hit_mask);
            tangent_v = gather<Vector3fAD>(cloud_.tangent_v(), surfel_id_ad, hit_mask);
            opacity = gather<FloatAD>(cloud_.opacity(), surfel_id_ad, hit_mask);
            value = gather<FloatAD>(cloud_.value(), surfel_id_ad, hit_mask);
        } else {
            center = gather<Vector3f>(detach<false>(cloud_.center()), surfel_id, hit_mask_detached);
            tangent_u = gather<Vector3f>(detach<false>(cloud_.tangent_u()), surfel_id, hit_mask_detached);
            tangent_v = gather<Vector3f>(detach<false>(cloud_.tangent_v()), surfel_id, hit_mask_detached);
            opacity = gather<Float>(detach<false>(cloud_.opacity()), surfel_id, hit_mask_detached);
            value = gather<Float>(detach<false>(cloud_.value()), surfel_id, hit_mask_detached);
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

    for (int pick = 0; pick < cloud_.surfel_count(); ++pick) {
        FloatT<Detached> best_t = full<FloatT<Detached>>(Infinity, ray_count);
        FloatT<Detached> best_alpha = zeros<FloatT<Detached>>(ray_count);
        FloatT<Detached> best_value = zeros<FloatT<Detached>>(ray_count);
        IntT<Detached> best_id = full<IntT<Detached>>(-1, ray_count);

        for (int surfel = 0; surfel < cloud_.surfel_count(); ++surfel) {
            const IntT<Detached> surfel_id_current =
                full<IntT<Detached>>(surfel, ray_count);
            Vector3fT<Detached> center;
            Vector3fT<Detached> tangent_u;
            Vector3fT<Detached> tangent_v;
            FloatT<Detached> opacity;
            FloatT<Detached> value;
            if constexpr (!Detached) {
                const IntAD surfel_id = IntAD(full<Int>(surfel, ray_count));
                center = gather<Vector3fAD>(cloud_.center(), surfel_id, active);
                tangent_u = gather<Vector3fAD>(cloud_.tangent_u(), surfel_id, active);
                tangent_v = gather<Vector3fAD>(cloud_.tangent_v(), surfel_id, active);
                opacity = gather<FloatAD>(cloud_.opacity(), surfel_id, active);
                value = gather<FloatAD>(cloud_.value(), surfel_id, active);
            } else {
                const Int surfel_id = full<Int>(surfel, ray_count);
                const Mask active_detached = detach<false>(active);
                center = gather<Vector3f>(detach<false>(cloud_.center()), surfel_id, active_detached);
                tangent_u = gather<Vector3f>(detach<false>(cloud_.tangent_u()), surfel_id, active_detached);
                tangent_v = gather<Vector3f>(detach<false>(cloud_.tangent_v()), surfel_id, active_detached);
                opacity = gather<Float>(detach<false>(cloud_.opacity()), surfel_id, active_detached);
                value = gather<Float>(detach<false>(cloud_.value()), surfel_id, active_detached);
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
template Mask SurfelScene::shadow_test<true>(const Ray &ray, Mask active) const;
template MaskAD SurfelScene::shadow_test<false>(const RayAD &ray, MaskAD active) const;
template Mask SurfelScene::visible<true>(const Vector3f &start, const Vector3f &end, Mask active) const;
template MaskAD SurfelScene::visible<false>(const Vector3fAD &start, const Vector3fAD &end, MaskAD active) const;

} // namespace rayd
