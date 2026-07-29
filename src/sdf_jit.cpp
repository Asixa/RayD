// Copyright Xingyu Chen.
// Implements the Dr.Jit SDF query, visibility, and reflection paths.

#include <rayd/jit/sdf.h>

#include <src/sdf_jit.h>

#include <drjit-core/jit.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace rayd {

namespace {

struct SdfNativeIntersection {
    Float t;
    Mask hit;
    Vector3f position;
    Vector3f normal;
    Int steps;
    Int base_x;
    Int base_y;
    Int base_z;
    Float denominator;
};

struct AdRotation {
    FloatAD r00;
    FloatAD r01;
    FloatAD r02;
    FloatAD r10;
    FloatAD r11;
    FloatAD r12;
    FloatAD r20;
    FloatAD r21;
    FloatAD r22;
};

struct AdGridSample {
    FloatAD value;
    Vector3fAD local_gradient;
};

void validate_options(const SdfTraceOptions& options, const char* context) {
    require(options.max_steps >= 1, std::string(context) + ": max_steps must be at least 1.");
    require(options.relaxation > 0.0f && options.relaxation <= 1.0f,
            std::string(context) + ": relaxation must lie in (0, 1].");
}

void validate_ray_widths(const Ray& ray, int ray_count, const Mask& active, const char* context) {
    const auto compatible = [ray_count](int width) { return width == 1 || width == ray_count; };
    require(compatible(static_cast<int>(slices(ray.o))), std::string(context) + ": invalid ray origin width.");
    require(compatible(static_cast<int>(slices(ray.d))), std::string(context) + ": invalid ray direction width.");
    require(compatible(static_cast<int>(slices(ray.tmax))), std::string(context) + ": invalid ray tmax width.");
    require(compatible(static_cast<int>(slices(active))), std::string(context) + ": invalid active-mask width.");
}

SdfNativeIntersection launch_native_intersection(const SdfGrid& grid, const Ray& ray, const SdfTraceOptions& options,
                                                 Mask active) {
    const int ray_count = static_cast<int>(slices(ray.o));
    validate_ray_widths(ray, ray_count, active, "SdfGrid::intersect()");

    SdfNativeIntersection out;
    out.t = empty<Float>(ray_count);
    out.hit = empty<Mask>(ray_count);
    out.position = empty<Vector3f>(ray_count);
    out.normal = empty<Vector3f>(ray_count);
    out.steps = empty<Int>(ray_count);
    out.base_x = empty<Int>(ray_count);
    out.base_y = empty<Int>(ray_count);
    out.base_z = empty<Int>(ray_count);
    out.denominator = empty<Float>(ray_count);
    if (ray_count == 0)
        return out;

    const Float zero = zeros<Float>(ray_count);
    Float ox = ray.o.x() + zero;
    Float oy = ray.o.y() + zero;
    Float oz = ray.o.z() + zero;
    Float dx = ray.d.x() + zero;
    Float dy = ray.d.y() + zero;
    Float dz = ray.d.z() + zero;
    Float tmax = ray.tmax + zero;
    active = active && full<Mask>(true, ray_count);

    const Float values = detach<false>(grid.values());
    const Vector3f position = detach<false>(grid.position());
    const Float rotation = detach<false>(grid.rotation());
    const Vector3f scale = detach<false>(grid.scale());
    drjit::eval(values, position, rotation, scale, ox, oy, oz, dx, dy, dz, tmax, active);

    SdfJitLaunchParams params{};
    params.values = values.data();
    params.nx = grid.nx();
    params.ny = grid.ny();
    params.nz = grid.nz();
    params.position_x = position.x().data();
    params.position_y = position.y().data();
    params.position_z = position.z().data();
    params.rotation = rotation.data();
    params.scale_x = scale.x().data();
    params.scale_y = scale.y().data();
    params.scale_z = scale.z().data();
    params.origin_x = ox.data();
    params.origin_y = oy.data();
    params.origin_z = oz.data();
    params.direction_x = dx.data();
    params.direction_y = dy.data();
    params.direction_z = dz.data();
    params.ray_tmax = tmax.data();
    params.active = reinterpret_cast<const std::uint8_t*>(active.data());
    params.ray_count = ray_count;
    params.max_steps = options.max_steps;
    params.relaxation = options.relaxation;
    params.eps_hit = options.eps_hit;
    params.out_t = out.t.data();
    params.out_hit = reinterpret_cast<std::uint8_t*>(out.hit.data());
    params.out_position_x = out.position.x().data();
    params.out_position_y = out.position.y().data();
    params.out_position_z = out.position.z().data();
    params.out_normal_x = out.normal.x().data();
    params.out_normal_y = out.normal.y().data();
    params.out_normal_z = out.normal.z().data();
    params.out_steps = out.steps.data();
    params.out_base_x = out.base_x.data();
    params.out_base_y = out.base_y.data();
    params.out_base_z = out.base_z.data();
    params.out_denominator = out.denominator.data();
    launch_sdf_intersect_jit(params, jit_cuda_stream());
    return out;
}

FloatAD quaternion_component(const FloatAD& rotation, int axis, int ray_count, const MaskAD& active) {
    return gather<FloatAD>(rotation, full<IntAD>(axis, ray_count), active);
}

AdRotation make_rotation(const FloatAD& rotation, int ray_count, const MaskAD& active) {
    FloatAD w = quaternion_component(rotation, 0, ray_count, active);
    FloatAD x = quaternion_component(rotation, 1, ray_count, active);
    FloatAD y = quaternion_component(rotation, 2, ray_count, active);
    FloatAD z = quaternion_component(rotation, 3, ray_count, active);
    const FloatAD inv_length = rcp(maximum(sqrt(w * w + x * x + y * y + z * z), FloatAD(1.0e-12f)));
    w *= inv_length;
    x *= inv_length;
    y *= inv_length;
    z *= inv_length;

    AdRotation result;
    result.r00 = 1.0f - 2.0f * (y * y + z * z);
    result.r01 = 2.0f * (x * y - w * z);
    result.r02 = 2.0f * (x * z + w * y);
    result.r10 = 2.0f * (x * y + w * z);
    result.r11 = 1.0f - 2.0f * (x * x + z * z);
    result.r12 = 2.0f * (y * z - w * x);
    result.r20 = 2.0f * (x * z - w * y);
    result.r21 = 2.0f * (y * z + w * x);
    result.r22 = 1.0f - 2.0f * (x * x + y * y);
    return result;
}

Vector3fAD rotate(const AdRotation& rotation, const Vector3fAD& value) {
    return Vector3fAD(rotation.r00 * value.x() + rotation.r01 * value.y() + rotation.r02 * value.z(),
                      rotation.r10 * value.x() + rotation.r11 * value.y() + rotation.r12 * value.z(),
                      rotation.r20 * value.x() + rotation.r21 * value.y() + rotation.r22 * value.z());
}

Vector3fAD transpose_rotate(const AdRotation& rotation, const Vector3fAD& value) {
    return Vector3fAD(rotation.r00 * value.x() + rotation.r10 * value.y() + rotation.r20 * value.z(),
                      rotation.r01 * value.x() + rotation.r11 * value.y() + rotation.r21 * value.z(),
                      rotation.r02 * value.x() + rotation.r12 * value.y() + rotation.r22 * value.z());
}

AdGridSample sample_grid_ad(const SdfGrid& grid, const Vector3fAD& local_point, const Vector3fAD& scale,
                            const IntAD& base_x, const IntAD& base_y, const IntAD& base_z, const MaskAD& active) {
    const float cells_x = static_cast<float>(grid.nx() - 1);
    const float cells_y = static_cast<float>(grid.ny() - 1);
    const float cells_z = static_cast<float>(grid.nz() - 1);
    const FloatAD coord_x = minimum(maximum((local_point.x() / scale.x() + 0.5f) * cells_x, 0.0f), cells_x);
    const FloatAD coord_y = minimum(maximum((local_point.y() / scale.y() + 0.5f) * cells_y, 0.0f), cells_y);
    const FloatAD coord_z = minimum(maximum((local_point.z() / scale.z() + 0.5f) * cells_z, 0.0f), cells_z);
    const FloatAD fx = coord_x - FloatAD(base_x);
    const FloatAD fy = coord_y - FloatAD(base_y);
    const FloatAD fz = coord_z - FloatAD(base_z);
    const FloatAD wx[2] = {1.0f - fx, fx};
    const FloatAD wy[2] = {1.0f - fy, fy};
    const FloatAD wz[2] = {1.0f - fz, fz};

    FloatAD value = zeros<FloatAD>(slices(local_point));
    Vector3fAD index_gradient = zeros<Vector3fAD>(slices(local_point));
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                const IntAD index = ((base_x + di) * grid.ny() + (base_y + dj)) * grid.nz() + (base_z + dk);
                const FloatAD corner = gather<FloatAD>(grid.values(), index, active);
                value += wx[di] * wy[dj] * wz[dk] * corner;
                index_gradient.x() += (di == 0 ? -1.0f : 1.0f) * wy[dj] * wz[dk] * corner;
                index_gradient.y() += wx[di] * (dj == 0 ? -1.0f : 1.0f) * wz[dk] * corner;
                index_gradient.z() += wx[di] * wy[dj] * (dk == 0 ? -1.0f : 1.0f) * corner;
            }
        }
    }

    return AdGridSample{
        value,
        Vector3fAD(index_gradient.x() * cells_x / scale.x(), index_gradient.y() * cells_y / scale.y(),
                   index_gradient.z() * cells_z / scale.z()),
    };
}

SdfIntersectionAD attach_sdf_gradient(const SdfGrid& grid, const RayAD& ray, const SdfNativeIntersection& native) {
    const int ray_count = static_cast<int>(slices(ray.o));
    const MaskAD hit = MaskAD(native.hit);
    const Vector3fAD zero = zeros<Vector3fAD>(ray_count);
    const Vector3fAD position = grid.position() + zero;
    const Vector3fAD scale = grid.scale() + zero;
    const AdRotation rotation = make_rotation(grid.rotation(), ray_count, hit);
    const FloatAD direction_length = maximum(sqrt(dot(ray.d, ray.d)), FloatAD(1.0e-12f));
    const Vector3fAD unit_direction = ray.d / direction_length;
    const FloatAD frozen_t = FloatAD(native.t);
    const Vector3fAD frozen_world_point = ray.o + unit_direction * frozen_t;
    const Vector3fAD frozen_local_point = transpose_rotate(rotation, frozen_world_point - position);
    const IntAD base_x = IntAD(native.base_x);
    const IntAD base_y = IntAD(native.base_y);
    const IntAD base_z = IntAD(native.base_z);
    const AdGridSample frozen_sample = sample_grid_ad(grid, frozen_local_point, scale, base_x, base_y, base_z, hit);
    const FloatAD t_proxy = select(hit, -frozen_sample.value / FloatAD(native.denominator), zeros<FloatAD>(ray_count));
    const FloatAD t = replace_grad(frozen_t, t_proxy);

    const Vector3fAD world_point = ray.o + unit_direction * t;
    const Vector3fAD local_point = transpose_rotate(rotation, world_point - position);
    const AdGridSample normal_sample = sample_grid_ad(grid, local_point, scale, base_x, base_y, base_z, hit);
    const Vector3fAD world_gradient = rotate(rotation, normal_sample.local_gradient);
    const Vector3fAD normal_proxy =
        select(hit, world_gradient / maximum(sqrt(dot(world_gradient, world_gradient)), FloatAD(1.0e-12f)), zero);

    SdfIntersectionAD result;
    result.t = replace_grad(FloatAD(native.t), select(hit, t, zeros<FloatAD>(ray_count)));
    result.hit_mask = hit;
    result.position = replace_grad(Vector3fAD(native.position), select(hit, world_point, zero));
    result.normal = replace_grad(Vector3fAD(native.normal), normal_proxy);
    result.steps = IntAD(native.steps);
    return result;
}

template <bool Detached> ReflectionChainT<Detached> initialize_sdf_chain(int ray_count, int max_bounces) {
    ReflectionChainT<Detached> result;
    result.max_bounces = max_bounces;
    result.ray_count = ray_count;
    const int slot_count = ray_count * max_bounces;
    result.bounce_count = zeros<IntT<Detached>>(ray_count);
    result.discovery_count = zeros<IntT<Detached>>(ray_count);
    result.representative_ray_index = full<IntT<Detached>>(-1, ray_count);
    result.t = full<FloatT<Detached>>(Infinity, slot_count);
    result.hit_points = zeros<Vector3fT<Detached>>(slot_count);
    result.geo_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.image_sources = zeros<Vector3fT<Detached>>(slot_count);
    result.plane_points = zeros<Vector3fT<Detached>>(slot_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.shape_ids = full<IntT<Detached>>(-1, slot_count);
    result.prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.local_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.global_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.trailing_t = full<FloatT<Detached>>(Infinity, ray_count);
    result.trailing_prim = full<IntT<Detached>>(-1, ray_count);
    result.trailing_dir = zeros<Vector3fT<Detached>>(ray_count);
    result.trailing_origin = zeros<Vector3fT<Detached>>(ray_count);
    return result;
}

} // namespace

template <bool Detached> FloatT<Detached> SdfGrid::query_bias(const SdfTraceOptions& options, int ray_count) const {
    Float resolved;
    if (options.eps_hit > 0.0f) {
        resolved = full<Float>(options.eps_hit, ray_count);
    } else {
        const Vector3f detached_scale = detach<false>(scale());
        const Float voxel = minimum(detached_scale.x() / static_cast<float>(nx() - 1),
                                    minimum(detached_scale.y() / static_cast<float>(ny() - 1),
                                            detached_scale.z() / static_cast<float>(nz() - 1)));
        resolved = voxel * 1.0e-3f + zeros<Float>(ray_count);
    }
    resolved = maximum(2.0f * resolved, Float(RayEpsilon));
    if constexpr (Detached)
        return resolved;
    else
        return FloatAD(resolved);
}

SdfGrid::SdfGrid(const Float& values, int nx, int ny, int nz, const Vector3f& position, const Float& rotation,
                 const Vector3f& scale) {
    initialize(FloatAD(values), nx, ny, nz, Vector3fAD(position), FloatAD(rotation), Vector3fAD(scale));
}

SdfGrid::SdfGrid(const FloatAD& values, int nx, int ny, int nz, const Vector3fAD& position, const FloatAD& rotation,
                 const Vector3fAD& scale) {
    initialize(values, nx, ny, nz, position, rotation, scale);
}

void SdfGrid::initialize(const FloatAD& values, int nx, int ny, int nz, const Vector3fAD& position,
                         const FloatAD& rotation, const Vector3fAD& scale) {
    require(nx >= 2 && ny >= 2 && nz >= 2, "SdfGrid(): every grid extent must be at least 2.");
    require(static_cast<int64_t>(nx) * ny * nz == static_cast<int64_t>(slices(values)),
            "SdfGrid(): values must contain exactly nx * ny * nz entries.");
    require(slices(position) == 1, "SdfGrid(): position must contain one vector.");
    require(slices(rotation) == 4, "SdfGrid(): rotation must contain four quaternion entries in (w, x, y, z) order.");
    require(slices(scale) == 1, "SdfGrid(): scale must contain one vector.");
    values_ = values;
    position_ = position;
    rotation_ = rotation;
    scale_ = scale;
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
}

template <bool Detached>
SdfIntersectionT<Detached> SdfGrid::intersect(const RayT<Detached>& ray, const SdfTraceOptions& options,
                                              MaskT<Detached> active) const {
    validate_options(options, "SdfGrid::intersect()");
    const Ray detached_ray(detach<false>(ray.o), detach<false>(ray.d), detach<false>(ray.tmax));
    const SdfNativeIntersection native =
        launch_native_intersection(*this, detached_ray, options, detach<false>(active));
    if constexpr (Detached) {
        SdfIntersection result;
        result.t = native.t;
        result.hit_mask = native.hit;
        result.position = native.position;
        result.normal = native.normal;
        result.steps = native.steps;
        return result;
    } else {
        return attach_sdf_gradient(*this, ray, native);
    }
}

template <bool Detached>
MaskT<Detached> SdfGrid::visible(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                                 const SdfTraceOptions& options, MaskT<Detached> active) const {
    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end)) == ray_count, "SdfGrid::visible(): start and end must have the same width.");
    const Vector3fT<Detached> delta = end - start;
    const FloatT<Detached> length = sqrt(dot(delta, delta));
    const FloatT<Detached> bias = query_bias<Detached>(options, ray_count);
    const MaskT<Detached> short_segment = length <= 2.0f * bias;
    const Vector3fT<Detached> direction = delta / maximum(length, FloatT<Detached>(1.0e-12f));
    const RayT<Detached> ray(start + direction * bias, direction,
                             maximum(length - 2.0f * bias, FloatT<Detached>(0.0f)));
    const SdfIntersectionT<Detached> hit = intersect<Detached>(ray, options, active && !short_segment);
    return active && (short_segment || !hit.hit_mask);
}

template <bool Detached>
ReflectionChainT<Detached> SdfGrid::trace_reflections(const RayT<Detached>& ray, int max_bounces,
                                                      const SdfTraceOptions& options, MaskT<Detached> active) const {
    require(max_bounces >= 0, "SdfGrid::trace_reflections(): max_bounces must be non-negative.");
    validate_options(options, "SdfGrid::trace_reflections()");
    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionChainT<Detached> result = initialize_sdf_chain<Detached>(ray_count, max_bounces);
    if (ray_count == 0 || max_bounces == 0)
        return result;

    const FloatT<Detached> direction_length = maximum(sqrt(dot(ray.d, ray.d)), FloatT<Detached>(1.0e-12f));
    RayT<Detached> current_ray(ray.o, ray.d / direction_length, ray.tmax);
    MaskT<Detached> current_active = active;
    Vector3fT<Detached> current_image_source = ray.o;
    const Int slot_base = arange<Int>(ray_count) * max_bounces;
    const IntT<Detached> one = full<IntT<Detached>>(1, ray_count);
    const IntT<Detached> zero = zeros<IntT<Detached>>(ray_count);
    const FloatT<Detached> bias = query_bias<Detached>(options, ray_count);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        const SdfIntersectionT<Detached> hit = intersect<Detached>(current_ray, options, current_active);
        const MaskT<Detached> bounce_hit = current_active && hit.hit_mask;
        Vector3fT<Detached> normal = select(dot(current_ray.d, hit.normal) > 0.0f, -hit.normal, hit.normal);
        const FloatT<Detached> plane_distance = dot(current_image_source - hit.position, normal);
        const Vector3fT<Detached> image_source = current_image_source - 2.0f * plane_distance * normal;
        const IntT<Detached> slot = IntT<Detached>(slot_base + bounce);
        const IntT<Detached> sdf_id = zeros<IntT<Detached>>(ray_count);
        scatter(result.t, hit.t, slot, bounce_hit);
        scatter(result.hit_points, hit.position, slot, bounce_hit);
        scatter(result.geo_normals, normal, slot, bounce_hit);
        scatter(result.image_sources, image_source, slot, bounce_hit);
        scatter(result.plane_points, hit.position, slot, bounce_hit);
        scatter(result.plane_normals, normal, slot, bounce_hit);
        scatter(result.shape_ids, sdf_id, slot, bounce_hit);
        scatter(result.prim_ids, sdf_id, slot, bounce_hit);
        scatter(result.local_prim_ids, sdf_id, slot, bounce_hit);
        scatter(result.global_prim_ids, sdf_id, slot, bounce_hit);
        result.bounce_count += select(bounce_hit, one, zero);

        const Vector3fT<Detached> reflected_direction = current_ray.d - 2.0f * dot(current_ray.d, normal) * normal;
        current_ray.o = select(bounce_hit, hit.position + bias * reflected_direction, current_ray.o);
        current_ray.d = select(bounce_hit, reflected_direction, current_ray.d);
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

template SdfIntersection SdfGrid::intersect<true>(const Ray&, const SdfTraceOptions&, Mask) const;
template SdfIntersectionAD SdfGrid::intersect<false>(const RayAD&, const SdfTraceOptions&, MaskAD) const;
template Mask SdfGrid::visible<true>(const Vector3f&, const Vector3f&, const SdfTraceOptions&, Mask) const;
template MaskAD SdfGrid::visible<false>(const Vector3fAD&, const Vector3fAD&, const SdfTraceOptions&, MaskAD) const;
template ReflectionChain SdfGrid::trace_reflections<true>(const Ray&, int, const SdfTraceOptions&, Mask) const;
template ReflectionChainAD SdfGrid::trace_reflections<false>(const RayAD&, int, const SdfTraceOptions&, MaskAD) const;
template Float SdfGrid::query_bias<true>(const SdfTraceOptions&, int) const;
template FloatAD SdfGrid::query_bias<false>(const SdfTraceOptions&, int) const;

} // namespace rayd
