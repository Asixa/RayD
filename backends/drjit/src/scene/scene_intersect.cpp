#include <rayd/ray.h>
#include <rayd/scene/scene.h>

namespace rayd {

namespace {
bool uses_symbolic_optix_query_path() {
    // Dr.Jit symbolic recording cannot mix multiple OptiX pipelines/SBTs
    // within a single captured kernel. Fall back to the unified scene path.
    return jit_flag(JitFlag::Recording);
}

} // namespace

template <bool Detached>
IntersectionT<Detached> Scene::intersect(const RayT<Detached> &ray, MaskT<Detached> active, RayFlags flags) const {
    require(is_ready(), "Scene::intersect(): scene is not built.");
    require(!pending_updates_, "Scene::intersect(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(ray.o));
    const bool want_geo_n   = has_flag(flags, RayFlags::Geometric);
    const bool want_shading = has_flag(flags, RayFlags::ShadingN);
    const bool want_uv      = has_flag(flags, RayFlags::UV);
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
        require(!jit_flag(JitFlag::Recording),
                "trace_backend='cuda' cannot serve intersect() inside a Dr.Jit symbolic "
                "recording region; use trace_backend='optix' or evaluate outside the "
                "recorded loop.");
        optix_hit = cuda_backend().template intersect<Detached>(ray, hit_mask);
    } else if (optix_split_active() && !symbolic_optix_query) {
        MaskT<Detached> static_hit_mask = active;
        MaskT<Detached> dynamic_hit_mask = active;
        const OptixIntersection static_hit =
            optix_static_scene().template intersect<Detached>(ray, static_hit_mask);
        const OptixIntersection dynamic_hit =
            optix_dynamic_scene().template intersect<Detached>(ray, dynamic_hit_mask);

        const Mask static_hit_mask_detached = detach<false>(static_hit_mask);
        const Mask dynamic_hit_mask_detached = detach<false>(dynamic_hit_mask);
        const Mask choose_dynamic =
            dynamic_hit_mask_detached &&
            (!static_hit_mask_detached || (dynamic_hit.t < static_hit.t));
        const Mask any_hit = static_hit_mask_detached || dynamic_hit_mask_detached;

        optix_hit.reserve(ray_count);
        optix_hit.t = select(choose_dynamic, dynamic_hit.t, static_hit.t);
        optix_hit.barycentric[0] =
            select(choose_dynamic, dynamic_hit.barycentric[0], static_hit.barycentric[0]);
        optix_hit.barycentric[1] =
            select(choose_dynamic, dynamic_hit.barycentric[1], static_hit.barycentric[1]);
        optix_hit.shape_id = select(choose_dynamic, dynamic_hit.shape_id, static_hit.shape_id);
        optix_hit.local_prim_id =
            select(choose_dynamic, dynamic_hit.local_prim_id, static_hit.local_prim_id);

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
        std::tie(triangle_uv_coords, hit_distance) = ray_intersect_triangle<Detached>(triangle_p0, triangle_e1, triangle_e2, ray);

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal = gather<Vector3fAD>(triangle_info_.face_normal, global_primitive_id_ad, hit_mask);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 = gather<Vector3fAD>(triangle_info_.n0, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n1 = gather<Vector3fAD>(triangle_info_.n1, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n2 = gather<Vector3fAD>(triangle_info_.n2, global_primitive_id_ad, hit_mask);
                MaskT<Detached> use_face_normal_mask = gather<MaskAD>(triangle_face_normal_mask_, global_primitive_id_ad, hit_mask);
                const Vector2fT<Detached> safe_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal =
                    normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data = gather<TriangleUVAD>(triangle_uv_, global_primitive_id_ad, hit_mask);
            const Vector2fT<Detached> safe_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0], triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask, uv, intersection.uv);
        }
    } else {
        // Detached path: use OptiX results directly, gather only what is needed.
        triangle_uv_coords = optix_hit.barycentric;
        hit_distance = optix_hit.t;

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal = gather<Vector3f>(triangle_info_detached_.face_normal, global_primitive_id, hit_mask_detached);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 = gather<Vector3f>(triangle_info_detached_.n0, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n1 = gather<Vector3f>(triangle_info_detached_.n1, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n2 = gather<Vector3f>(triangle_info_detached_.n2, global_primitive_id, hit_mask_detached);
                MaskT<Detached> use_face_normal_mask = gather<Mask>(triangle_face_normal_mask_detached_, global_primitive_id, hit_mask_detached);
                const Vector2fT<Detached> safe_uv = select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal =
                    normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask_detached, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask_detached, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data = gather<TriangleUV>(triangle_uv_detached_, global_primitive_id, hit_mask_detached);
            const Vector2fT<Detached> safe_uv = select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0], triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask_detached, uv, intersection.uv);
        }
    }

    hit_mask &= drjit::isfinite(hit_distance) && (hit_distance < ray.tmax);

    const FloatT<Detached> safe_hit_distance = select(hit_mask, hit_distance, zeros<FloatT<Detached>>(ray_count));
    const Vector2fT<Detached> safe_triangle_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));

    const Vector3fT<Detached> barycentric_coordinates(1.f - safe_triangle_uv.x() - safe_triangle_uv.y(),
                                                      safe_triangle_uv.x(),
                                                      safe_triangle_uv.y());
    const Vector3fT<Detached> hit_position = ray(safe_hit_distance);

    intersection.t = select(hit_mask, safe_hit_distance, intersection.t);
    intersection.p = select(hit_mask, hit_position, intersection.p);
    intersection.barycentric = select(hit_mask, barycentric_coordinates, intersection.barycentric);
    intersection.shape_id = select(hit_mask, IntT<Detached>(shape_id), intersection.shape_id);
    const IntT<Detached> local_primitive_id_t = IntT<Detached>(local_primitive_id);
    const IntT<Detached> global_primitive_id_t = IntT<Detached>(global_primitive_id);
    intersection.prim_id = select(hit_mask, local_primitive_id_t, intersection.prim_id);
    intersection.local_prim_id =
        select(hit_mask, local_primitive_id_t, intersection.local_prim_id);
    intersection.global_prim_id =
        select(hit_mask, global_primitive_id_t, intersection.global_prim_id);
    return intersection;
}

template Intersection Scene::intersect<true>(const Ray &ray, Mask active, RayFlags flags) const;
template IntersectionAD Scene::intersect<false>(const RayAD &ray, MaskAD active, RayFlags flags) const;

} // namespace rayd
