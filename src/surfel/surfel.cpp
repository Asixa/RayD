#include <vector>

#include <rayd/surfel/surfel.h>

namespace rayd {

SurfelCloud::SurfelCloud(const Vector3f &center,
                         const Vector3f &tangent_u,
                         const Vector3f &tangent_v,
                         const Float &opacity) {
    initialize(Vector3fAD(center), Vector3fAD(tangent_u), Vector3fAD(tangent_v), FloatAD(opacity));
}

SurfelCloud::SurfelCloud(const Vector3fAD &center,
                         const Vector3fAD &tangent_u,
                         const Vector3fAD &tangent_v,
                         const FloatAD &opacity) {
    initialize(center, tangent_u, tangent_v, opacity);
}

void SurfelCloud::initialize(const Vector3fAD &center,
                             const Vector3fAD &tangent_u,
                             const Vector3fAD &tangent_v,
                             const FloatAD &opacity) {
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
}

SurfelScene::SurfelScene(const SurfelCloud &cloud, const SurfelTraceOptions &options)
    : cloud_(cloud), options_(options) {
    require(cloud_.surfel_count() > 0, "SurfelScene(): cloud must contain at least one surfel.");
    require(options_.cutoff > 0.f, "SurfelScene(): cutoff must be positive.");
    require(options_.alpha_cap > 0.f && options_.alpha_cap <= 1.f,
            "SurfelScene(): alpha_cap must be in (0, 1].");
}

void SurfelScene::build() {
    require(cloud_.surfel_count() > 0, "SurfelScene::build(): cloud is empty.");
    require(options_.cutoff > 0.f, "SurfelScene::build(): cutoff must be positive.");
    require(options_.alpha_cap > 0.f && options_.alpha_cap <= 1.f,
            "SurfelScene::build(): alpha_cap must be in (0, 1].");

    build_triangle_buffers();
    eval(optix_vertex_buffer_, optix_face_buffer_, triangle_to_surfel_id_);
    optix_scene_.build(optix_vertex_buffer_, optix_face_buffer_, vertex_count_, triangle_count_);
    ready_ = true;
}

void SurfelScene::build_triangle_buffers() {
    const int surfel_count = cloud_.surfel_count();
    const bool use_quad = options_.primitive_mode == SurfelPrimitiveMode::QuadTriangles;
    const int vertices_per_surfel = use_quad ? 4 : 3;
    const int triangles_per_surfel = use_quad ? 2 : 1;
    vertex_count_ = surfel_count * vertices_per_surfel;
    triangle_count_ = surfel_count * triangles_per_surfel;

    Vector3fAD vertices_ad = empty<Vector3fAD>(vertex_count_);
    const IntAD base = arange<IntAD>(surfel_count) * vertices_per_surfel;
    const FloatAD cutoff(options_.cutoff);
    const Vector3fAD &center = cloud_.center();
    const Vector3fAD &u = cloud_.tangent_u();
    const Vector3fAD &v = cloud_.tangent_v();

    if (use_quad) {
        scatter(vertices_ad, center - cutoff * u - cutoff * v, base + 0);
        scatter(vertices_ad, center + cutoff * u - cutoff * v, base + 1);
        scatter(vertices_ad, center + cutoff * u + cutoff * v, base + 2);
        scatter(vertices_ad, center - cutoff * u + cutoff * v, base + 3);
    } else {
        scatter(vertices_ad, center + (-2.f * cutoff) * u - cutoff * v, base + 0);
        scatter(vertices_ad, center + ( 2.f * cutoff) * u - cutoff * v, base + 1);
        scatter(vertices_ad, center + ( 3.f * cutoff) * v, base + 2);
    }

    std::vector<int> face_flat(static_cast<size_t>(triangle_count_) * 3u);
    std::vector<int> triangle_to_surfel(static_cast<size_t>(triangle_count_));
    for (int surfel_id = 0; surfel_id < surfel_count; ++surfel_id) {
        const int vertex_base = surfel_id * vertices_per_surfel;
        const int triangle_base = surfel_id * triangles_per_surfel;
        if (use_quad) {
            const int tri0 = triangle_base;
            face_flat[static_cast<size_t>(tri0) * 3u + 0u] = vertex_base + 0;
            face_flat[static_cast<size_t>(tri0) * 3u + 1u] = vertex_base + 1;
            face_flat[static_cast<size_t>(tri0) * 3u + 2u] = vertex_base + 2;
            triangle_to_surfel[static_cast<size_t>(tri0)] = surfel_id;

            const int tri1 = triangle_base + 1;
            face_flat[static_cast<size_t>(tri1) * 3u + 0u] = vertex_base + 0;
            face_flat[static_cast<size_t>(tri1) * 3u + 1u] = vertex_base + 2;
            face_flat[static_cast<size_t>(tri1) * 3u + 2u] = vertex_base + 3;
            triangle_to_surfel[static_cast<size_t>(tri1)] = surfel_id;
        } else {
            face_flat[static_cast<size_t>(triangle_base) * 3u + 0u] = vertex_base + 0;
            face_flat[static_cast<size_t>(triangle_base) * 3u + 1u] = vertex_base + 1;
            face_flat[static_cast<size_t>(triangle_base) * 3u + 2u] = vertex_base + 2;
            triangle_to_surfel[static_cast<size_t>(triangle_base)] = surfel_id;
        }
    }

    const Vector3f vertices = detach<false>(vertices_ad);
    optix_vertex_buffer_ = empty<Float>(vertex_count_ * 3);
    const Int vertex_indices = arange<Int>(vertex_count_) * 3;
    for (int axis = 0; axis < 3; ++axis) {
        scatter(optix_vertex_buffer_, vertices[axis], vertex_indices + axis);
    }
    optix_face_buffer_ = load<Int>(face_flat.data(), face_flat.size());
    triangle_to_surfel_id_ = load<Int>(triangle_to_surfel.data(), triangle_to_surfel.size());
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
    result.surfel_id = full<IntT<Detached>>(-1, ray_count);
    result.triangle_id = full<IntT<Detached>>(-1, ray_count);

    MaskT<Detached> hit_mask = active;
    const SurfelOptixIntersection optix_hit = optix_scene_.template intersect<Detached>(ray, hit_mask);
    const Mask hit_mask_detached = detach<false>(hit_mask);
    const Int triangle_id = optix_hit.triangle_id;
    const Int surfel_id = gather<Int>(triangle_to_surfel_id_, triangle_id, hit_mask_detached);

    Vector3fT<Detached> center;
    Vector3fT<Detached> tangent_u;
    Vector3fT<Detached> tangent_v;
    FloatT<Detached> opacity;
    if constexpr (!Detached) {
        const IntAD surfel_id_ad = IntAD(surfel_id);
        center = gather<Vector3fAD>(cloud_.center(), surfel_id_ad, hit_mask);
        tangent_u = gather<Vector3fAD>(cloud_.tangent_u(), surfel_id_ad, hit_mask);
        tangent_v = gather<Vector3fAD>(cloud_.tangent_v(), surfel_id_ad, hit_mask);
        opacity = gather<FloatAD>(cloud_.opacity(), surfel_id_ad, hit_mask);
    } else {
        center = gather<Vector3f>(detach<false>(cloud_.center()), surfel_id, hit_mask_detached);
        tangent_u = gather<Vector3f>(detach<false>(cloud_.tangent_u()), surfel_id, hit_mask_detached);
        tangent_v = gather<Vector3f>(detach<false>(cloud_.tangent_v()), surfel_id, hit_mask_detached);
        opacity = gather<Float>(detach<false>(cloud_.opacity()), surfel_id, hit_mask_detached);
    }

    const Vector3fT<Detached> raw_normal = cross(tangent_u, tangent_v);
    const FloatT<Detached> normal_len_sq = squared_norm(raw_normal);
    const MaskT<Detached> normal_valid = normal_len_sq > FloatT<Detached>(1e-16f);
    Vector3fT<Detached> normal =
        raw_normal / sqrt(select(normal_valid, normal_len_sq, FloatT<Detached>(1.f)));
    if (options_.face_forward) {
        normal = select(dot(normal, ray.d) > FloatT<Detached>(0.f), -normal, normal);
    }

    const FloatT<Detached> denom = dot(ray.d, normal);
    const MaskT<Detached> plane_valid = abs(denom) > FloatT<Detached>(1e-8f);
    const FloatT<Detached> safe_denom = select(plane_valid, denom, FloatT<Detached>(1.f));
    const FloatT<Detached> plane_t = dot(center - ray.o, normal) / safe_denom;
    MaskT<Detached> valid = hit_mask &&
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
    const FloatT<Detached> safe_basis_det =
        select(basis_valid, basis_det, FloatT<Detached>(1.f));

    const FloatT<Detached> local_u = (du * vv - dv * uv) / safe_basis_det;
    const FloatT<Detached> local_v = (dv * uu - du * uv) / safe_basis_det;
    const FloatT<Detached> cutoff(options_.cutoff);
    const MaskT<Detached> support_valid =
        (abs(local_u) <= cutoff) && (abs(local_v) <= cutoff);
    valid &= basis_valid && support_valid;

    const FloatT<Detached> gaussian =
        exp(FloatT<Detached>(-0.5f) * (local_u * local_u + local_v * local_v));
    const Vector2fT<Detached> local_uv(local_u, local_v);

    result.t = select(valid, plane_t, result.t);
    result.p = select(valid, hit_point, result.p);
    result.n = select(valid, normal, result.n);
    result.local_uv = select(valid, local_uv, result.local_uv);
    result.gaussian_weight = select(valid, gaussian, result.gaussian_weight);
    result.opacity = select(valid, opacity, result.opacity);
    result.surfel_id = select(valid, IntT<Detached>(surfel_id), result.surfel_id);
    result.triangle_id = select(valid, IntT<Detached>(triangle_id), result.triangle_id);
    return result;
}

template <bool Detached>
SurfelCompositeT<Detached> SurfelScene::composite_alpha(const RayT<Detached> &ray,
                                                        MaskT<Detached> active) const {
    require(ready_, "SurfelScene::composite_alpha(): scene is not built.");

    const int ray_count = static_cast<int>(slices(ray.o));
    SurfelCompositeT<Detached> result;
    result.intensity = zeros<FloatT<Detached>>(ray_count);
    result.alpha = zeros<FloatT<Detached>>(ray_count);
    result.transmittance = full<FloatT<Detached>>(1.f, ray_count);
    result.depth = full<FloatT<Detached>>(Infinity, ray_count);

    FloatT<Detached> depth_numerator = zeros<FloatT<Detached>>(ray_count);

    for (int surfel = 0; surfel < cloud_.surfel_count(); ++surfel) {
        Vector3fT<Detached> center;
        Vector3fT<Detached> tangent_u;
        Vector3fT<Detached> tangent_v;
        FloatT<Detached> opacity;
        if constexpr (!Detached) {
            const IntAD surfel_id = IntAD(full<Int>(surfel, ray_count));
            center = gather<Vector3fAD>(cloud_.center(), surfel_id, active);
            tangent_u = gather<Vector3fAD>(cloud_.tangent_u(), surfel_id, active);
            tangent_v = gather<Vector3fAD>(cloud_.tangent_v(), surfel_id, active);
            opacity = gather<FloatAD>(cloud_.opacity(), surfel_id, active);
        } else {
            const Int surfel_id = full<Int>(surfel, ray_count);
            const Mask active_detached = detach<false>(active);
            center = gather<Vector3f>(detach<false>(cloud_.center()), surfel_id, active_detached);
            tangent_u = gather<Vector3f>(detach<false>(cloud_.tangent_u()), surfel_id, active_detached);
            tangent_v = gather<Vector3f>(detach<false>(cloud_.tangent_v()), surfel_id, active_detached);
            opacity = gather<Float>(detach<false>(cloud_.opacity()), surfel_id, active_detached);
        }

        const Vector3fT<Detached> raw_normal = cross(tangent_u, tangent_v);
        const FloatT<Detached> normal_len_sq = squared_norm(raw_normal);
        const MaskT<Detached> normal_valid = normal_len_sq > FloatT<Detached>(1e-16f);
        Vector3fT<Detached> normal =
            raw_normal / sqrt(select(normal_valid, normal_len_sq, FloatT<Detached>(1.f)));
        if (options_.face_forward) {
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

        const FloatT<Detached> safe_t =
            select(valid, plane_t, zeros<FloatT<Detached>>(ray_count));
        const Vector3fT<Detached> hit_point = ray(safe_t);
        const Vector3fT<Detached> delta = hit_point - center;

        const FloatT<Detached> uu = dot(tangent_u, tangent_u);
        const FloatT<Detached> uv = dot(tangent_u, tangent_v);
        const FloatT<Detached> vv = dot(tangent_v, tangent_v);
        const FloatT<Detached> du = dot(delta, tangent_u);
        const FloatT<Detached> dv = dot(delta, tangent_v);
        const FloatT<Detached> basis_det = uu * vv - uv * uv;
        const MaskT<Detached> basis_valid = abs(basis_det) > FloatT<Detached>(1e-16f);
        const FloatT<Detached> safe_basis_det =
            select(basis_valid, basis_det, FloatT<Detached>(1.f));

        const FloatT<Detached> local_u = (du * vv - dv * uv) / safe_basis_det;
        const FloatT<Detached> local_v = (dv * uu - du * uv) / safe_basis_det;
        const FloatT<Detached> cutoff(options_.cutoff);
        valid &= basis_valid && (abs(local_u) <= cutoff) && (abs(local_v) <= cutoff);

        const FloatT<Detached> gaussian =
            exp(FloatT<Detached>(-0.5f) * (local_u * local_u + local_v * local_v));
        FloatT<Detached> alpha =
            minimum(FloatT<Detached>(options_.alpha_cap),
                    maximum(FloatT<Detached>(0.f), opacity * gaussian));
        alpha = select(valid, alpha, zeros<FloatT<Detached>>(ray_count));

        const FloatT<Detached> contribution = result.transmittance * alpha;
        result.intensity += contribution;
        result.alpha += contribution;
        depth_numerator += contribution * plane_t;
        result.transmittance *= FloatT<Detached>(1.f) - alpha;
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
template Mask SurfelScene::shadow_test<true>(const Ray &ray, Mask active) const;
template MaskAD SurfelScene::shadow_test<false>(const RayAD &ray, MaskAD active) const;
template Mask SurfelScene::visible<true>(const Vector3f &start, const Vector3f &end, Mask active) const;
template MaskAD SurfelScene::visible<false>(const Vector3fAD &start, const Vector3fAD &end, MaskAD active) const;

} // namespace rayd
