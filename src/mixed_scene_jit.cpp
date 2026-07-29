// Copyright Xingyu Chen.
// Implements Dr.Jit unified mesh, bounded SDF, and surfel ray queries.

#include <rayd/jit/mixed_scene.h>

#include <src/runtime/multipath_internal_jit.h>

namespace rayd {

namespace {

template <bool Detached> IntersectionT<Detached> invalid_intersection(int ray_count) {
    IntersectionT<Detached> result;
    result.t = full<FloatT<Detached>>(Infinity, ray_count);
    result.p = zeros<Vector3fT<Detached>>(ray_count);
    result.n = zeros<Vector3fT<Detached>>(ray_count);
    result.geo_n = zeros<Vector3fT<Detached>>(ray_count);
    result.uv = zeros<Vector2fT<Detached>>(ray_count);
    result.barycentric = zeros<Vector3fT<Detached>>(ray_count);
    result.shape_id = full<IntT<Detached>>(-1, ray_count);
    result.prim_id = full<IntT<Detached>>(-1, ray_count);
    result.local_prim_id = full<IntT<Detached>>(-1, ray_count);
    result.global_prim_id = full<IntT<Detached>>(-1, ray_count);
    return result;
}

template <bool Detached>
void select_candidate(IntersectionT<Detached>& result, const IntersectionT<Detached>& candidate, MaskT<Detached> take,
                      RayFlags flags) {
    result.t = select(take, candidate.t, result.t);
    if (has_flag(flags, RayFlags::Geometric)) {
        result.p = select(take, candidate.p, result.p);
        result.geo_n = select(take, candidate.geo_n, result.geo_n);
        result.barycentric = select(take, candidate.barycentric, result.barycentric);
    }
    if (has_flag(flags, RayFlags::ShadingN))
        result.n = select(take, candidate.n, result.n);
    if (has_flag(flags, RayFlags::UV))
        result.uv = select(take, candidate.uv, result.uv);
    result.shape_id = select(take, candidate.shape_id, result.shape_id);
    result.prim_id = select(take, candidate.prim_id, result.prim_id);
    result.local_prim_id = select(take, candidate.local_prim_id, result.local_prim_id);
    result.global_prim_id = select(take, candidate.global_prim_id, result.global_prim_id);
}

} // namespace

MixedScene::MixedScene(const std::string& edge_bvh_backend, const std::string& trace_backend)
    : mesh_scene_(edge_bvh_backend, trace_backend) {}

MixedScene::~MixedScene() = default;

int MixedScene::add_mesh(const Mesh& mesh, bool dynamic) {
    const int mesh_id = mesh_scene_.add_mesh(mesh, dynamic);
    ++mesh_count_;
    mesh_face_count_ += mesh.face_count();
    ready_ = false;
    return mesh_id;
}

int MixedScene::add_sdf(const SdfGrid& grid, const SdfTraceOptions& options) {
    sdfs_.push_back(SdfEntry{grid, options});
    ready_ = false;
    return static_cast<int>(sdfs_.size()) - 1;
}

int MixedScene::add_surfel(const SurfelCloud& cloud, const SurfelTraceOptions& options) {
    surfels_.push_back(std::make_unique<SurfelScene>(cloud, options));
    ready_ = false;
    return static_cast<int>(surfels_.size()) - 1;
}

void MixedScene::build() {
    if (mesh_count_ > 0)
        mesh_scene_.build();
    surfel_prefix_.clear();
    int prefix = 0;
    for (const auto& scene : surfels_) {
        scene->build();
        surfel_prefix_.push_back(prefix);
        prefix += scene->surfel_count();
    }
    ready_ = true;
}

bool MixedScene::is_ready() const {
    if (!ready_ || (mesh_count_ > 0 && !mesh_scene_.is_ready()))
        return false;
    for (const auto& scene : surfels_) {
        if (!scene->is_ready())
            return false;
    }
    return true;
}

template <bool Detached>
IntersectionT<Detached> MixedScene::intersect(const RayT<Detached>& ray, MaskT<Detached> active, RayFlags flags) const {
    require(is_ready(), "MixedScene::intersect(): scene is not built.");
    if (mesh_count_ > 0 && sdfs_.empty() && surfels_.empty())
        return mesh_scene_.intersect<Detached>(ray, active, flags);

    const int ray_count = static_cast<int>(slices(ray.o));
    IntersectionT<Detached> result = mesh_count_ > 0 ? mesh_scene_.intersect<Detached>(ray, active, flags)
                                                     : invalid_intersection<Detached>(ray_count);
    if (!has_flag(flags, RayFlags::Geometric)) {
        result.p = zeros<Vector3fT<Detached>>(ray_count);
        result.geo_n = zeros<Vector3fT<Detached>>(ray_count);
        result.barycentric = zeros<Vector3fT<Detached>>(ray_count);
    }
    if (!has_flag(flags, RayFlags::ShadingN))
        result.n = zeros<Vector3fT<Detached>>(ray_count);
    if (!has_flag(flags, RayFlags::UV))
        result.uv = zeros<Vector2fT<Detached>>(ray_count);
    for (int index = 0; index < static_cast<int>(sdfs_.size()); ++index) {
        const SdfEntry& entry = sdfs_[index];
        const SdfIntersectionT<Detached> hit = entry.grid.intersect<Detached>(ray, entry.options, active);
        IntersectionT<Detached> candidate = invalid_intersection<Detached>(ray_count);
        candidate.t = hit.t;
        candidate.p = hit.position;
        candidate.n = hit.normal;
        candidate.geo_n = hit.normal;
        candidate.shape_id = full<IntT<Detached>>(mesh_count_ + index, ray_count);
        candidate.prim_id = zeros<IntT<Detached>>(ray_count);
        candidate.local_prim_id = zeros<IntT<Detached>>(ray_count);
        candidate.global_prim_id = full<IntT<Detached>>(mesh_face_count_ + index, ray_count);
        const MaskT<Detached> take = active && hit.hit_mask && (hit.t < result.t);
        select_candidate<Detached>(result, candidate, take, flags);
    }

    const int surfel_global_base = mesh_face_count_ + static_cast<int>(sdfs_.size());
    for (int index = 0; index < static_cast<int>(surfels_.size()); ++index) {
        const SurfelIntersectionT<Detached> hit = surfels_[index]->intersect<Detached>(ray, active);
        IntersectionT<Detached> candidate = invalid_intersection<Detached>(ray_count);
        candidate.t = hit.t;
        candidate.p = hit.p;
        candidate.n = hit.n;
        candidate.geo_n = hit.n;
        candidate.uv = hit.local_uv;
        candidate.shape_id = full<IntT<Detached>>(mesh_count_ + static_cast<int>(sdfs_.size()) + index, ray_count);
        candidate.prim_id = hit.surfel_id;
        candidate.local_prim_id = hit.surfel_id;
        candidate.global_prim_id = hit.surfel_id + surfel_global_base + surfel_prefix_[index];
        const MaskT<Detached> take = active && hit.is_valid() && (hit.t < result.t);
        select_candidate<Detached>(result, candidate, take, flags);
    }
    return result;
}

template <bool Detached>
SegmentVisibilityT<Detached> MixedScene::visible(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                                                 MaskT<Detached> active) const {
    require(is_ready(), "MixedScene::visible(): scene is not built.");
    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end)) == ray_count,
            "MixedScene::visible(): start and end must have the same width.");
    if (mesh_count_ > 0 && sdfs_.empty() && surfels_.empty())
        return mesh_scene_.visible<Detached>(start, end, Int(), active);

    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;
    result.visible = active;
    if (mesh_count_ > 0)
        result.visible &= mesh_scene_.visible<Detached>(start, end, Int(), active).visible;
    for (const SdfEntry& entry : sdfs_)
        result.visible &= entry.grid.visible<Detached>(start, end, entry.options, active);
    for (const auto& scene : surfels_)
        result.visible &= scene->visible<Detached>(start, end, active);
    return result;
}

template <bool Detached>
FloatT<Detached> MixedScene::transmittance(const RayT<Detached>& ray, MaskT<Detached> active) const {
    require(is_ready(), "MixedScene::transmittance(): scene is not built.");
    const int ray_count = static_cast<int>(slices(ray.o));
    MaskT<Detached> opaque = full<MaskT<Detached>>(false, ray_count);
    if (mesh_count_ > 0)
        opaque |= mesh_scene_.shadow_test<Detached>(ray, active);
    FloatT<Detached> result = full<FloatT<Detached>>(1.0f, ray_count);
    for (const auto& scene : surfels_)
        result *= scene->composite_alpha<Detached>(ray, active).transmittance;
    return select(active, select(opaque, zeros<FloatT<Detached>>(ray_count), result),
                  full<FloatT<Detached>>(1.0f, ray_count));
}

template <bool Detached>
ReflectionChainT<Detached> MixedScene::trace_reflections(const RayT<Detached>& ray, int max_bounces,
                                                         MaskT<Detached> active) const {
    require(is_ready(), "MixedScene::trace_reflections(): scene is not built.");
    require(max_bounces >= 0, "MixedScene::trace_reflections(): max_bounces must be non-negative.");
    if (mesh_count_ > 0 && sdfs_.empty() && surfels_.empty() && max_bounces > 0)
        return mesh_scene_.trace_reflections<Detached>(ray, max_bounces, active);

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionChainT<Detached> result =
        multipath_detail::initialize_reflection_chain_result<Detached>(ray_count, max_bounces);
    if (ray_count == 0 || max_bounces == 0)
        return result;

    const FloatT<Detached> direction_length = maximum(sqrt(dot(ray.d, ray.d)), FloatT<Detached>(1.0e-12f));
    RayT<Detached> current_ray(ray.o, ray.d / direction_length, ray.tmax);
    MaskT<Detached> current_active = active;
    Vector3fT<Detached> current_image_source = ray.o;
    const Int slot_base = arange<Int>(ray_count) * max_bounces;
    const IntT<Detached> one = full<IntT<Detached>>(1, ray_count);
    const IntT<Detached> zero = zeros<IntT<Detached>>(ray_count);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        const IntersectionT<Detached> hit = intersect<Detached>(current_ray, current_active, RayFlags::All);
        const MaskT<Detached> bounce_hit = current_active && hit.is_valid();
        const Vector3fT<Detached> normal = select(dot(current_ray.d, hit.geo_n) > 0.0f, -hit.geo_n, hit.geo_n);
        const FloatT<Detached> plane_distance = dot(current_image_source - hit.p, normal);
        const Vector3fT<Detached> image_source = current_image_source - 2.0f * plane_distance * normal;
        const IntT<Detached> slot = IntT<Detached>(slot_base + bounce);
        scatter(result.t, hit.t, slot, bounce_hit);
        scatter(result.hit_points, hit.p, slot, bounce_hit);
        scatter(result.geo_normals, normal, slot, bounce_hit);
        scatter(result.image_sources, image_source, slot, bounce_hit);
        scatter(result.plane_points, hit.p, slot, bounce_hit);
        scatter(result.plane_normals, normal, slot, bounce_hit);
        scatter(result.shape_ids, hit.shape_id, slot, bounce_hit);
        scatter(result.prim_ids, hit.prim_id, slot, bounce_hit);
        scatter(result.local_prim_ids, hit.local_prim_id, slot, bounce_hit);
        scatter(result.global_prim_ids, hit.global_prim_id, slot, bounce_hit);
        result.bounce_count += select(bounce_hit, one, zero);

        const Vector3fT<Detached> reflected = current_ray.d - 2.0f * dot(current_ray.d, normal) * normal;
        FloatT<Detached> bias = full<FloatT<Detached>>(RayEpsilon, ray_count);
        for (int index = 0; index < static_cast<int>(sdfs_.size()); ++index) {
            bias = select(hit.shape_id == mesh_count_ + index,
                          sdfs_[index].grid.query_bias<Detached>(sdfs_[index].options, ray_count), bias);
        }
        current_ray.o = select(bounce_hit, hit.p + bias * reflected, current_ray.o);
        current_ray.d = select(bounce_hit, reflected, current_ray.d);
        current_ray.tmax = select(bounce_hit, full<FloatT<Detached>>(Infinity, ray_count), current_ray.tmax);
        current_image_source = select(bounce_hit, image_source, current_image_source);
        current_active = bounce_hit;
    }

    const MaskT<Detached> valid = result.bounce_count > 0;
    result.discovery_count = select(valid, one, zero);
    result.representative_ray_index =
        select(valid, IntT<Detached>(arange<Int>(ray_count)), full<IntT<Detached>>(-1, ray_count));
    result.trailing_dir = select(valid, current_ray.d, zeros<Vector3fT<Detached>>(ray_count));
    result.trailing_origin = select(valid, current_ray.o, zeros<Vector3fT<Detached>>(ray_count));
    return result;
}

template Intersection MixedScene::intersect<true>(const Ray&, Mask, RayFlags) const;
template IntersectionAD MixedScene::intersect<false>(const RayAD&, MaskAD, RayFlags) const;
template SegmentVisibility MixedScene::visible<true>(const Vector3f&, const Vector3f&, Mask) const;
template SegmentVisibilityAD MixedScene::visible<false>(const Vector3fAD&, const Vector3fAD&, MaskAD) const;
template ReflectionChain MixedScene::trace_reflections<true>(const Ray&, int, Mask) const;
template ReflectionChainAD MixedScene::trace_reflections<false>(const RayAD&, int, MaskAD) const;
template Float MixedScene::transmittance<true>(const Ray&, Mask) const;
template FloatAD MixedScene::transmittance<false>(const RayAD&, MaskAD) const;

} // namespace rayd
