#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/ray.h>
#include "scene_internal.h"
#include <rayd/multipath/diffraction_accumulation_ad.h>
#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>
#include <rayd/trace/cuda_multipath_gpu.h>

#include "scene_multipath_internal.h"

namespace rayd {

using namespace multipath_detail;

namespace {

struct ReflEpcRaw {
    int ray_count = 0;
    int max_bounces = 0;
    Mask valid;
    Int bounce_count;
    Float path_length;
    Float point_x;
    Float point_y;
    Float point_z;
    Int trace_prim_ids;
    Int resolved_prim_ids;
    Int surface_group_ids;
    Float plane_normal_x;
    Float plane_normal_y;
    Float plane_normal_z;
    Int first_blocked_segment;
    Int first_blocked_prim;
    Int first_blocked_group;
};

template <bool Detached>
ReflEpcT<Detached> init_refl_epc(int ray_count,
                                                                int max_bounces) {
    ReflEpcT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    const int slot_count = ray_count * max_bounces;
    result.valid = full<MaskT<Detached>>(false, ray_count);
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.path_length = full<FloatT<Detached>>(Infinity, ray_count);
    result.reflection_points = zeros<Vector3fT<Detached>>(slot_count);
    result.prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.trace_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.resolved_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.surface_group_ids = full<IntT<Detached>>(-1, slot_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.first_blocked_segment = full<IntT<Detached>>(-1, ray_count);
    result.first_blocked_prim = full<IntT<Detached>>(-1, ray_count);
    result.first_blocked_group = full<IntT<Detached>>(-1, ray_count);
    return result;
}

ReflEpcRaw alloc_refl_epc_raw(int ray_count, int max_bounces) {
    const int slot_count = ray_count * max_bounces;
    ReflEpcRaw raw;
    raw.ray_count = ray_count;
    raw.max_bounces = max_bounces;
    raw.valid = empty<Mask>(ray_count);
    raw.bounce_count = empty<Int>(ray_count);
    raw.path_length = empty<Float>(ray_count);
    raw.point_x = empty<Float>(slot_count);
    raw.point_y = empty<Float>(slot_count);
    raw.point_z = empty<Float>(slot_count);
    raw.trace_prim_ids = empty<Int>(slot_count);
    raw.resolved_prim_ids = empty<Int>(slot_count);
    raw.surface_group_ids = empty<Int>(slot_count);
    raw.plane_normal_x = empty<Float>(slot_count);
    raw.plane_normal_y = empty<Float>(slot_count);
    raw.plane_normal_z = empty<Float>(slot_count);
    raw.first_blocked_segment = empty<Int>(ray_count);
    raw.first_blocked_prim = empty<Int>(ray_count);
    raw.first_blocked_group = empty<Int>(ray_count);
    return raw;
}

template <bool Detached>
ReflEpcFieldT<Detached> init_refl_epc_field(
    int ray_count,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options) {
    ReflEpcFieldT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    const int slot_count = ray_count * max_bounces;
    const bool return_geom = options.return_geom;
    const bool return_endpoints = options.return_endpoints;
    const bool return_hit_points =
        return_geom && options.return_hit_points;
    const bool return_normals = return_geom && options.return_normals;
    const bool return_resolved_prim_ids =
        return_geom && options.return_resolved_prim_ids;
    const bool return_surface_group_ids =
        return_geom && options.return_surface_group_ids;

    result.valid = empty<MaskT<Detached>>(ray_count);
    result.bounce_count = empty<IntT<Detached>>(ray_count);
    result.path_length = empty<FloatT<Detached>>(ray_count);
    result.field_x_re = empty<FloatT<Detached>>(ray_count);
    result.field_x_im = empty<FloatT<Detached>>(ray_count);
    result.field_y_re = empty<FloatT<Detached>>(ray_count);
    result.field_y_im = empty<FloatT<Detached>>(ray_count);
    result.field_z_re = empty<FloatT<Detached>>(ray_count);
    result.field_z_im = empty<FloatT<Detached>>(ray_count);

    if (return_endpoints) {
        result.tx_pos =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
        result.first_hit =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
        result.last_hit =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
    } else {
        result.tx_pos = zeros<Vector3fT<Detached>>(0);
        result.first_hit = zeros<Vector3fT<Detached>>(0);
        result.last_hit = zeros<Vector3fT<Detached>>(0);
    }

    if (return_hit_points) {
        result.hit_points =
            Vector3fT<Detached>(empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count));
    } else {
        result.hit_points = zeros<Vector3fT<Detached>>(0);
    }
    if (return_normals) {
        result.normals =
            Vector3fT<Detached>(empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count));
    } else {
        result.normals = zeros<Vector3fT<Detached>>(0);
    }
    if (return_resolved_prim_ids) {
        result.resolved_prim_ids = empty<IntT<Detached>>(slot_count);
    } else {
        result.resolved_prim_ids = full<IntT<Detached>>(-1, 0);
    }
    if (return_surface_group_ids) {
        result.surface_group_ids = empty<IntT<Detached>>(slot_count);
    } else {
        result.surface_group_ids = full<IntT<Detached>>(-1, 0);
    }

    return result;
}

ReflEpcOptions epc_options_from_field_options(
    const ReflEpcFieldOptions &options) {
    ReflEpcOptions epc_options;
    epc_options.expected_prim_ids = options.expected_prim_ids;
    epc_options.surface_group_id = options.surface_group_id;
    epc_options.surface_group_size = options.surface_group_size;
    epc_options.surface_group_members = options.surface_group_members;
    epc_options.surface_max_group_size = options.surface_max_group_size;
    epc_options.visibility_ignore_mode = options.visibility_ignore_mode;
    epc_options.final_ignore_group_ids = options.final_ignore_group_ids;
    return epc_options;
}

} // namespace

template <bool Detached>
ReflEpcT<Detached> Scene::trace_refl_epc(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc(): max_bounces exceeds the native EPC limit.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflEpcT<Detached> result =
        init_refl_epc<Detached>(ray_count, max_bounces);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_refl_epc(): native EPC is a non-AD native fast path. "
                "Pass a non-AD Ray and detached receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc(): receiver width must be 1 or match ray count.");
        const ReflEpcVisibilityIgnoreMode visibility_ignore_mode =
            parse_refl_epc_vis_ignore(options.visibility_ignore_mode);
        const bool surface_group_ignore =
            visibility_ignore_mode == ReflEpcVisibilityIgnoreMode::SurfaceGroup;
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        const int surface_group_id_count = static_cast<int>(slices(options.surface_group_id));
        const int surface_group_count = static_cast<int>(slices(options.surface_group_size));
        const int surface_group_member_count =
            static_cast<int>(slices(options.surface_group_members));
        const int final_ignore_group_count =
            static_cast<int>(slices(options.final_ignore_group_ids));
        const bool has_surface_groups =
            surface_group_id_count > 0 ||
            surface_group_count > 0 ||
            surface_group_member_count > 0 ||
            options.surface_max_group_size > 0;
        require(expected_prim_count == 0 || expected_prim_count == slot_count,
                "Scene::trace_refl_epc(): expected_prim_ids width must be n_rays * max_bounces.");
        require(final_ignore_group_count == 0 ||
                    final_ignore_group_count == 1 ||
                    final_ignore_group_count == ray_count,
                "Scene::trace_refl_epc(): final_ignore_group_ids width must be 1 or match ray count.");
        if (has_surface_groups) {
            const int triangle_count =
                static_cast<int>(slices(triangle_info_detached_.p0));
            require(surface_group_id_count == triangle_count,
                    "Scene::trace_refl_epc(): surface_group_id width must match triangle count.");
            require(surface_group_count > 0,
                    "Scene::trace_refl_epc(): surface_group_size must be non-empty when surface groups are provided.");
            require(options.surface_max_group_size > 0,
                    "Scene::trace_refl_epc(): surface_max_group_size must be positive when surface groups are provided.");
            require(surface_group_member_count >=
                        surface_group_count * options.surface_max_group_size,
                    "Scene::trace_refl_epc(): surface_group_members must contain surface_group_size * surface_max_group_size entries.");
        }
        require(!surface_group_ignore || has_surface_groups,
                "Scene::trace_refl_epc(): visibility_ignore_mode='surface_group' requires surface group tables.");
        require(final_ignore_group_count == 0 || surface_group_ignore,
                "Scene::trace_refl_epc(): final_ignore_group_ids require visibility_ignore_mode='surface_group'.");

        const Mask active_detached =
            sanitize_reflection_active<Detached>(ray, active);
        if (drjit::none(active_detached)) {
            return result;
        }

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            const int hitgroup_record_count = scenes.hitgroup_record_count;

            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::trace_refl_epc(): OptiX scene is not ready.");
            require(hitgroup_record_count > 0,
                    "Scene::trace_refl_epc(): invalid hitgroup record count.");

            ensure_pipeline(reflection_epc_pipeline_, primary_scene->context(),
                            hitgroup_record_count, reflection_epc_pipeline_config());
        }

        drjit::eval(ray.o,
                    ray.d,
                    ray.tmax,
                    receiver,
                    active_detached);
        if (expected_prim_count > 0) {
            drjit::eval(options.expected_prim_ids);
        }
        if (has_surface_groups) {
            drjit::eval(options.surface_group_id,
                        options.surface_group_size,
                        options.surface_group_members);
        }
        if (final_ignore_group_count > 0) {
            drjit::eval(options.final_ignore_group_ids);
        }

        ensure_reflection_epc_geometry_ready();

        ReflEpcRaw raw = alloc_refl_epc_raw(ray_count, max_bounces);

        ReflEpcParams params = {};
        params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.tri_p0_x = triangle_info_detached_.p0.x().data();
        params.tri_p0_y = triangle_info_detached_.p0.y().data();
        params.tri_p0_z = triangle_info_detached_.p0.z().data();
        params.tri_e1_x = triangle_info_detached_.e1.x().data();
        params.tri_e1_y = triangle_info_detached_.e1.y().data();
        params.tri_e1_z = triangle_info_detached_.e1.z().data();
        params.tri_e2_x = triangle_info_detached_.e2.x().data();
        params.tri_e2_y = triangle_info_detached_.e2.y().data();
        params.tri_e2_z = triangle_info_detached_.e2.z().data();
        params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        params.face_offsets = face_offsets_.data();
        params.n_meshes = mesh_count_;
        params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        params.expected_prim_ids =
            expected_prim_count > 0 ? options.expected_prim_ids.data() : nullptr;
        params.expected_prim_count = expected_prim_count;
        params.surface_group_id =
            has_surface_groups ? options.surface_group_id.data() : nullptr;
        params.surface_group_id_count = surface_group_id_count;
        params.surface_group_size =
            has_surface_groups ? options.surface_group_size.data() : nullptr;
        params.surface_group_count = surface_group_count;
        params.surface_group_members =
            has_surface_groups ? options.surface_group_members.data() : nullptr;
        params.surface_max_group_size =
            has_surface_groups ? options.surface_max_group_size : 0;
        params.visibility_ignore_mode =
            surface_group_ignore ? ReflEpcVisibilityIgnoreSurfaceGroup
                                 : ReflEpcVisibilityIgnorePrimitive;
        params.final_ignore_group_ids =
            final_ignore_group_count > 0 ? options.final_ignore_group_ids.data() : nullptr;
        params.final_ignore_group_count = final_ignore_group_count;
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.ray_dx = ray.d.x().data();
        params.ray_dy = ray.d.y().data();
        params.ray_dz = ray.d.z().data();
        params.ray_tmax = ray.tmax.data();
        params.rx_x = receiver.x().data();
        params.rx_y = receiver.y().data();
        params.rx_z = receiver.z().data();
        params.rx_count = receiver_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.n_rays = ray_count;
        params.max_bounces = max_bounces;
        params.plane_tolerance = options.plane_tolerance;
        params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        params.out_bounce_count = raw.bounce_count.data();
        params.out_path_length = raw.path_length.data();
        params.out_point_x = raw.point_x.data();
        params.out_point_y = raw.point_y.data();
        params.out_point_z = raw.point_z.data();
        params.out_trace_prim_ids = raw.trace_prim_ids.data();
        params.out_resolved_prim_ids = raw.resolved_prim_ids.data();
        params.out_surface_group_ids = raw.surface_group_ids.data();
        params.out_plane_normal_x = raw.plane_normal_x.data();
        params.out_plane_normal_y = raw.plane_normal_y.data();
        params.out_plane_normal_z = raw.plane_normal_z.data();
        params.out_first_blocked_segment = raw.first_blocked_segment.data();
        params.out_first_blocked_prim = raw.first_blocked_prim.data();
        params.out_first_blocked_group = raw.first_blocked_group.data();

        if (cuda_trace) {
            cuda_backend().run_reflection_epc(params, /*direct_only=*/false,
                                              /*primary_visibility_only=*/false, ray_count);
        } else {
            reflection_epc_pipeline_->launch(0, params);
        }

        result.valid = raw.valid;
        result.bounce_count = raw.bounce_count;
        result.path_length = raw.path_length;
        result.reflection_points =
            Vector3f(raw.point_x, raw.point_y, raw.point_z);
        result.prim_ids = raw.trace_prim_ids;
        result.trace_prim_ids = raw.trace_prim_ids;
        result.resolved_prim_ids = raw.resolved_prim_ids;
        result.surface_group_ids = raw.surface_group_ids;
        result.plane_normals =
            Vector3f(raw.plane_normal_x,
                             raw.plane_normal_y,
                             raw.plane_normal_z);
        result.first_blocked_segment = raw.first_blocked_segment;
        result.first_blocked_prim = raw.first_blocked_prim;
        result.first_blocked_group = raw.first_blocked_group;
        return result;
    }
}

template <bool Detached>
ReflEpcFieldT<Detached> Scene::trace_refl_epc_field(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc_field(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc_field(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc_field(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc_field(): max_bounces exceeds the native EPC limit.");
    require(options.omega > 0.f,
            "Scene::trace_refl_epc_field(): omega must be positive.");
    require(options.wavelength > 0.f,
            "Scene::trace_refl_epc_field(): wavelength must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflEpcFieldT<Detached> result =
        init_refl_epc_field<Detached>(
            ray_count,
            max_bounces,
            options);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_refl_epc_field(): native EPC field is a non-AD native fast path. "
                "Pass a non-AD Ray and detached receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match ray count.");
        const int slot_count = ray_count * max_bounces;
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match ray count.");

        drjit::eval(options.slot_plane_normal,
                    options.slot_eta_r,
                    options.slot_mu_r,
                    options.slot_sigma,
                    options.slot_gain,
                    options.tx_polarization);

        const ReflEpcOptions epc_options =
            epc_options_from_field_options(options);
        const ReflEpc epc =
            trace_refl_epc<true>(
                ray,
                receiver,
                max_bounces,
                epc_options,
                active);

        ReflEpcFieldParams params = {};
        params.n_rays = ray_count;
        params.max_bounces = max_bounces;
        params.epc_valid = reinterpret_cast<const uint8_t *>(epc.valid.data());
        params.epc_bounce_count = epc.bounce_count.data();
        params.epc_path_length = epc.path_length.data();
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.rx_x = receiver.x().data();
        params.rx_y = receiver.y().data();
        params.rx_z = receiver.z().data();
        params.rx_count = receiver_count;
        params.hit_x = epc.reflection_points.x().data();
        params.hit_y = epc.reflection_points.y().data();
        params.hit_z = epc.reflection_points.z().data();
        params.epc_normal_x = epc.plane_normals.x().data();
        params.epc_normal_y = epc.plane_normals.y().data();
        params.epc_normal_z = epc.plane_normals.z().data();
        const bool return_resolved_prim_ids =
            options.return_geom && options.return_resolved_prim_ids;
        const bool return_surface_group_ids =
            options.return_geom && options.return_surface_group_ids;
        params.resolved_prim_ids =
            return_resolved_prim_ids ? epc.resolved_prim_ids.data() : nullptr;
        params.surface_group_ids =
            return_surface_group_ids ? epc.surface_group_ids.data() : nullptr;
        params.slot_normal_x = options.slot_plane_normal.x().data();
        params.slot_normal_y = options.slot_plane_normal.y().data();
        params.slot_normal_z = options.slot_plane_normal.z().data();
        params.slot_eta_r = options.slot_eta_r.data();
        params.slot_mu_r = options.slot_mu_r.data();
        params.slot_sigma = options.slot_sigma.data();
        params.slot_gain = options.slot_gain.data();
        params.tx_pol_x = options.tx_polarization.x().data();
        params.tx_pol_y = options.tx_polarization.y().data();
        params.tx_pol_z = options.tx_polarization.z().data();
        params.tx_pol_count = tx_pol_count;
        params.omega = options.omega;
        params.wavelength = options.wavelength;
        params.out_valid = reinterpret_cast<uint8_t *>(result.valid.data());
        params.out_bounce_count = result.bounce_count.data();
        params.out_path_length = result.path_length.data();
        params.out_field_x_re = result.field_x_re.data();
        params.out_field_x_im = result.field_x_im.data();
        params.out_field_y_re = result.field_y_re.data();
        params.out_field_y_im = result.field_y_im.data();
        params.out_field_z_re = result.field_z_re.data();
        params.out_field_z_im = result.field_z_im.data();

        if (options.return_endpoints) {
            params.out_tx_x = result.tx_pos.x().data();
            params.out_tx_y = result.tx_pos.y().data();
            params.out_tx_z = result.tx_pos.z().data();
            params.out_first_hit_x = result.first_hit.x().data();
            params.out_first_hit_y = result.first_hit.y().data();
            params.out_first_hit_z = result.first_hit.z().data();
            params.out_last_hit_x = result.last_hit.x().data();
            params.out_last_hit_y = result.last_hit.y().data();
            params.out_last_hit_z = result.last_hit.z().data();
        }
        if (options.return_geom && options.return_hit_points) {
            params.out_hit_x = result.hit_points.x().data();
            params.out_hit_y = result.hit_points.y().data();
            params.out_hit_z = result.hit_points.z().data();
        }
        if (options.return_geom && options.return_normals) {
            params.out_normal_x = result.normals.x().data();
            params.out_normal_y = result.normals.y().data();
            params.out_normal_z = result.normals.z().data();
        }
        if (return_resolved_prim_ids) {
            params.out_resolved_prim_ids = result.resolved_prim_ids.data();
        }
        if (return_surface_group_ids) {
            params.out_surface_group_ids = result.surface_group_ids.data();
        }

        reflection_epc_field_gpu(params);
        return result;
    }
}

template <bool Detached>
ReflEpcFieldT<Detached> Scene::trace_refl_epc_field(
    const Vector3fT<Detached> &tx_position,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc_field(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc_field(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc_field(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc_field(): max_bounces exceeds the native EPC limit.");
    require(options.omega > 0.f,
            "Scene::trace_refl_epc_field(): omega must be positive.");
    require(options.wavelength > 0.f,
            "Scene::trace_refl_epc_field(): wavelength must be positive.");

    const int ray_count = static_cast<int>(slices(tx_position));
    ReflEpcFieldT<Detached> result =
        init_refl_epc_field<Detached>(
            ray_count,
            max_bounces,
            options);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match transmitter count.");
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        require(expected_prim_count == slot_count,
                "Scene::trace_refl_epc_field(): expected_prim_ids width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_point)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_point width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match transmitter count.");

        Mask active_detached = sanitize_segment_active<false>(
            tx_position,
            receiver,
            active);
        if (drjit::none(active_detached)) {
            result.valid = full<MaskAD>(false, ray_count);
            result.bounce_count = full<IntAD>(0, ray_count);
            result.path_length = full<FloatAD>(Infinity, ray_count);
            result.field_x_re = zeros<FloatAD>(ray_count);
            result.field_x_im = zeros<FloatAD>(ray_count);
            result.field_y_re = zeros<FloatAD>(ray_count);
            result.field_y_im = zeros<FloatAD>(ray_count);
            result.field_z_re = zeros<FloatAD>(ray_count);
            result.field_z_im = zeros<FloatAD>(ray_count);
            return result;
        }

        const Int slot_base = arange<Int>(ray_count) * Int(max_bounces);
        const MaskAD active_ad = MaskAD(active_detached);

        struct ComplexADValue {
            FloatAD re;
            FloatAD im;
        };
        struct ComplexVectorAD {
            Vector3fAD re;
            Vector3fAD im;
        };

        auto complex_add = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re + b.re, a.im + b.im};
        };
        auto complex_sub = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re - b.re, a.im - b.im};
        };
        auto complex_mul = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re * b.re - a.im * b.im,
                    a.re * b.im + a.im * b.re};
        };
        auto complex_scale = [](const ComplexADValue &a,
                                const FloatAD &scale) -> ComplexADValue {
            return {a.re * scale, a.im * scale};
        };
        auto complex_div = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            const FloatAD denom =
                maximum(b.re * b.re + b.im * b.im, FloatAD(Epsilon));
            return {(a.re * b.re + a.im * b.im) / denom,
                    (a.im * b.re - a.re * b.im) / denom};
        };
        auto complex_sqrt = [](const ComplexADValue &a) -> ComplexADValue {
            const FloatAD mag =
                sqrt(maximum(a.re * a.re + a.im * a.im, FloatAD(0.f)));
            const MaskAD positive_real_axis =
                (abs(a.im) <= FloatAD(Epsilon)) && (a.re > FloatAD(Epsilon));
            const FloatAD real_part =
                sqrt(maximum(FloatAD(0.5f) * (mag + a.re), FloatAD(0.f)));
            const FloatAD imag_abs =
                sqrt(maximum(FloatAD(0.5f) * (mag - a.re), FloatAD(1e-20f)));
            const FloatAD imag_sign =
                select(a.im < FloatAD(0.f), FloatAD(-1.f), FloatAD(1.f));
            return {
                select(positive_real_axis, sqrt(a.re), real_part),
                select(positive_real_axis, FloatAD(0.f), imag_sign * imag_abs),
            };
        };
        auto normalize_safe = [](const Vector3fAD &value,
                                 const Vector3fAD &fallback) -> Vector3fAD {
            const FloatAD value_norm = norm(value);
            return select(value_norm > FloatAD(Epsilon),
                          value / maximum(value_norm, FloatAD(Epsilon)),
                          fallback);
        };
        auto stable_perpendicular = [&](const Vector3fAD &direction,
                                        const Vector3fAD &preferred) -> Vector3fAD {
            const Vector3fAD dir =
                normalize_safe(direction, Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            const Vector3fAD projected = preferred - dot(preferred, dir) * dir;
            const Vector3fAD axis =
                select(abs(dir.x()) < FloatAD(0.9f),
                       Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)),
                       Vector3fAD(FloatAD(0.f), FloatAD(1.f), FloatAD(0.f)));
            const Vector3fAD fallback = axis - dot(axis, dir) * dir;
            return select(squared_norm(projected) > FloatAD(1e-12f),
                          normalize_safe(projected, axis),
                          normalize_safe(fallback,
                                         Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f))));
        };
        auto complex_dot_real = [](const ComplexVectorAD &field,
                                   const Vector3fAD &basis) -> ComplexADValue {
            return {dot(field.re, basis), dot(field.im, basis)};
        };
        auto slot_reflection_coefficients =
            [&](const IntAD &slot,
                const FloatAD &cos_theta,
                const MaskAD &slot_active) -> std::pair<ComplexADValue, ComplexADValue> {
            const FloatAD eta_r =
                maximum(gather<FloatAD>(options.slot_eta_r, slot, slot_active),
                        FloatAD(Epsilon));
            const FloatAD sigma =
                maximum(gather<FloatAD>(options.slot_sigma, slot, slot_active),
                        FloatAD(0.f));
            const FloatAD gain = gather<FloatAD>(options.slot_gain, slot, slot_active);
            const FloatAD mu_r =
                maximum(gather<FloatAD>(options.slot_mu_r, slot, slot_active),
                        FloatAD(Epsilon));
            const FloatAD omega = maximum(FloatAD(options.omega), FloatAD(Epsilon));
            const ComplexADValue eta = {
                eta_r,
                -sigma / (omega * FloatAD(8.854187817e-12f))
            };
            const ComplexADValue mu = {mu_r, FloatAD(0.f)};
            const FloatAD cos_clamped =
                minimum(maximum(abs(cos_theta), FloatAD(Epsilon)), FloatAD(1.f));
            const FloatAD sin2 =
                maximum(FloatAD(0.f), FloatAD(1.f) - cos_clamped * cos_clamped);
            const ComplexADValue a =
                complex_sqrt(complex_sub(complex_mul(mu, eta),
                                         ComplexADValue{sin2, FloatAD(0.f)}));
            const ComplexADValue mu_cos = {mu_r * cos_clamped, FloatAD(0.f)};
            const ComplexADValue eta_cos = {eta.re * cos_clamped,
                                            eta.im * cos_clamped};
            const ComplexADValue r_te =
                complex_scale(
                    complex_div(complex_sub(mu_cos, a),
                                complex_add(mu_cos, a)),
                    gain);
            const ComplexADValue r_tm =
                complex_scale(
                    complex_div(complex_sub(eta_cos, a),
                                complex_add(eta_cos, a)),
                    gain);
            return {r_te, r_tm};
        };
        auto reflect_field_vector =
            [&](const ComplexVectorAD &field,
                const Vector3fAD &incident_dir,
                const Vector3fAD &slot_normal,
                const IntAD &slot,
                const MaskAD &slot_active) -> ComplexVectorAD {
            const Vector3fAD incident_hat =
                normalize_safe(incident_dir,
                               Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            Vector3fAD normal_hat =
                normalize_safe(slot_normal,
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            normal_hat = select(dot(incident_hat, normal_hat) > FloatAD(0.f),
                                -normal_hat,
                                normal_hat);
            const FloatAD dot_dn = dot(incident_hat, normal_hat);
            const Vector3fAD reflected_dir =
                normalize_safe(incident_hat - FloatAD(2.f) * dot_dn * normal_hat,
                               -incident_hat);
            Vector3fAD s_hat = cross(normal_hat, incident_hat);
            s_hat = select(squared_norm(s_hat) > FloatAD(1e-12f),
                           normalize_safe(s_hat, stable_perpendicular(incident_hat, normal_hat)),
                           stable_perpendicular(incident_hat, normal_hat));
            Vector3fAD p_in_hat = cross(s_hat, incident_hat);
            p_in_hat =
                select(squared_norm(p_in_hat) > FloatAD(1e-12f),
                       normalize_safe(p_in_hat, stable_perpendicular(incident_hat, normal_hat)),
                       stable_perpendicular(incident_hat, normal_hat));
            Vector3fAD p_out_hat = cross(s_hat, reflected_dir);
            p_out_hat =
                select(squared_norm(p_out_hat) > FloatAD(1e-12f),
                       normalize_safe(p_out_hat, stable_perpendicular(reflected_dir, normal_hat)),
                       stable_perpendicular(reflected_dir, normal_hat));

            const auto [r_te, r_tm] =
                slot_reflection_coefficients(slot, abs(dot(incident_hat, normal_hat)), slot_active);
            const ComplexADValue e_s = complex_dot_real(field, s_hat);
            const ComplexADValue e_p = complex_dot_real(field, p_in_hat);
            const ComplexADValue out_s = complex_mul(r_te, e_s);
            const ComplexADValue out_p = complex_mul(r_tm, e_p);
            return {
                s_hat * out_s.re + p_out_hat * out_p.re,
                s_hat * out_s.im + p_out_hat * out_p.im,
            };
        };

        std::vector<Vector3fAD> images;
        images.reserve(static_cast<size_t>(max_bounces) + 1);
        images.push_back(tx_position);
        MaskAD valid = active_ad;

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            Vector3fAD plane_point =
                gather<Vector3fAD>(options.slot_plane_point, slot_ad, active_ad);
            Vector3fAD plane_normal =
                normalize_safe(gather<Vector3fAD>(options.slot_plane_normal, slot_ad, active_ad),
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            const Int expected_prim =
                gather<Int>(options.expected_prim_ids, slot, active_detached);
            valid = valid && MaskAD(expected_prim >= Int(0)) &&
                    (squared_norm(plane_normal) > FloatAD(0.f));
            const FloatAD plane_distance =
                dot(images.back() - plane_point, plane_normal);
            images.push_back(
                select(valid,
                       images.back() - FloatAD(2.f) * plane_distance * plane_normal,
                       images.back()));
        }

        Vector3fAD rx = receiver;
        if (receiver_count == 1 && ray_count > 1) {
            rx = gather<Vector3fAD>(receiver, zeros<IntAD>(ray_count), full<MaskAD>(true, ray_count));
        }
        Vector3fAD target = rx;
        std::vector<Vector3fAD> hits(static_cast<size_t>(max_bounces));
        std::vector<Vector3fAD> normals(static_cast<size_t>(max_bounces));
        for (int bounce = max_bounces - 1; bounce >= 0; --bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            const Vector3fAD plane_point =
                gather<Vector3fAD>(options.slot_plane_point, slot_ad, active_ad);
            const Vector3fAD plane_normal =
                normalize_safe(gather<Vector3fAD>(options.slot_plane_normal, slot_ad, active_ad),
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            const Vector3fAD line = target - images[static_cast<size_t>(bounce + 1)];
            const FloatAD denom = dot(line, plane_normal);
            const FloatAD t =
                dot(plane_point - images[static_cast<size_t>(bounce + 1)], plane_normal) /
                denom;
            const MaskAD hit_valid =
                valid &&
                drjit::isfinite(t) &&
                (abs(denom) > FloatAD(Epsilon)) &&
                (t > FloatAD(0.f)) &&
                (t < FloatAD(1.f));
            const Vector3fAD hit =
                images[static_cast<size_t>(bounce + 1)] + t * line;
            hits[static_cast<size_t>(bounce)] = select(hit_valid, hit, zeros<Vector3fAD>(ray_count));
            normals[static_cast<size_t>(bounce)] =
                select(hit_valid, plane_normal, zeros<Vector3fAD>(ray_count));
            if (options.return_geom && options.return_hit_points) {
                scatter(result.hit_points, hits[static_cast<size_t>(bounce)], slot_ad, hit_valid);
            }
            if (options.return_geom && options.return_normals) {
                scatter(result.normals, normals[static_cast<size_t>(bounce)], slot_ad, hit_valid);
            }
            target = select(hit_valid, hit, target);
            valid = hit_valid;
        }

        FloatAD path_length = zeros<FloatAD>(ray_count);
        Vector3fAD previous = tx_position;
        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Vector3fAD hit = hits[static_cast<size_t>(bounce)];
            path_length += norm(hit - previous);
            previous = hit;
        }
        path_length += norm(rx - previous);
        valid = valid && (path_length > FloatAD(Epsilon)) && drjit::isfinite(path_length);

        const Int pol_idx =
            tx_pol_count == 1 ? zeros<Int>(ray_count) : arange<Int>(ray_count);
        const Vector3fAD tx_pol =
            gather<Vector3fAD>(options.tx_polarization, IntAD(pol_idx), active_ad);
        const Vector3fAD first_dir =
            normalize_safe(hits.front() - tx_position,
                           Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
        const Vector3fAD transverse_pol =
            stable_perpendicular(first_dir, tx_pol);
        ComplexVectorAD field = {
            transverse_pol,
            zeros<Vector3fAD>(ray_count),
        };
        Vector3fAD field_previous = tx_position;
        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            const Vector3fAD hit = hits[static_cast<size_t>(bounce)];
            const Vector3fAD incident_dir =
                normalize_safe(hit - field_previous,
                               Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            field = reflect_field_vector(
                field,
                incident_dir,
                normals[static_cast<size_t>(bounce)],
                slot_ad,
                active_ad);
            field_previous = hit;
        }
        const FloatAD wave_k =
            FloatAD(2.f * Pi) / maximum(FloatAD(options.wavelength), FloatAD(Epsilon));
        const FloatAD phase = -wave_k * path_length;
        const FloatAD amplitude =
            FloatAD(options.wavelength) /
            (FloatAD(4.f * Pi) * maximum(path_length, FloatAD(Epsilon)));
        const FloatAD phase_cos = cos(phase);
        const FloatAD phase_sin = sin(phase);
        const Vector3fAD out_re =
            amplitude * (field.re * phase_cos - field.im * phase_sin);
        const Vector3fAD out_im =
            amplitude * (field.re * phase_sin + field.im * phase_cos);
        valid = valid &&
                drjit::isfinite(out_re.x()) &&
                drjit::isfinite(out_re.y()) &&
                drjit::isfinite(out_re.z()) &&
                drjit::isfinite(out_im.x()) &&
                drjit::isfinite(out_im.y()) &&
                drjit::isfinite(out_im.z());

        result.valid = valid;
        result.bounce_count =
            select(valid, full<IntAD>(max_bounces, ray_count), full<IntAD>(0, ray_count));
        result.path_length =
            select(valid, path_length, full<FloatAD>(Infinity, ray_count));
        result.field_x_re = select(valid, out_re.x(), FloatAD(0.f));
        result.field_x_im = select(valid, out_im.x(), FloatAD(0.f));
        result.field_y_re = select(valid, out_re.y(), FloatAD(0.f));
        result.field_y_im = select(valid, out_im.y(), FloatAD(0.f));
        result.field_z_re = select(valid, out_re.z(), FloatAD(0.f));
        result.field_z_im = select(valid, out_im.z(), FloatAD(0.f));

        if (options.return_endpoints) {
            result.tx_pos = tx_position;
            result.first_hit = max_bounces > 0 ? hits.front() : zeros<Vector3fAD>(ray_count);
            result.last_hit = max_bounces > 0 ? hits.back() : zeros<Vector3fAD>(ray_count);
        }
        if (options.return_geom && options.return_resolved_prim_ids) {
            result.resolved_prim_ids = IntAD(options.expected_prim_ids);
        }
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match transmitter count.");
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        require(expected_prim_count == slot_count,
                "Scene::trace_refl_epc_field(): expected_prim_ids width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_point)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_point width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match transmitter count.");

        const ReflEpcVisibilityIgnoreMode visibility_ignore_mode =
            parse_refl_epc_vis_ignore(options.visibility_ignore_mode);
        const bool surface_group_ignore =
            visibility_ignore_mode == ReflEpcVisibilityIgnoreMode::SurfaceGroup;
        const int surface_group_id_count = static_cast<int>(slices(options.surface_group_id));
        const int surface_group_count = static_cast<int>(slices(options.surface_group_size));
        const int surface_group_member_count =
            static_cast<int>(slices(options.surface_group_members));
        const int final_ignore_group_count =
            static_cast<int>(slices(options.final_ignore_group_ids));
        const bool has_surface_groups =
            surface_group_id_count > 0 ||
            surface_group_count > 0 ||
            surface_group_member_count > 0 ||
            options.surface_max_group_size > 0;
        require(final_ignore_group_count == 0 ||
                    final_ignore_group_count == 1 ||
                    final_ignore_group_count == ray_count,
                "Scene::trace_refl_epc_field(): final_ignore_group_ids width must be 1 or match transmitter count.");
        if (has_surface_groups) {
            const int triangle_count =
                static_cast<int>(slices(triangle_info_detached_.p0));
            require(surface_group_id_count == triangle_count,
                    "Scene::trace_refl_epc_field(): surface_group_id width must match triangle count.");
            require(surface_group_count > 0,
                    "Scene::trace_refl_epc_field(): surface_group_size must be non-empty when surface groups are provided.");
            require(options.surface_max_group_size > 0,
                    "Scene::trace_refl_epc_field(): surface_max_group_size must be positive when surface groups are provided.");
            require(surface_group_member_count >=
                        surface_group_count * options.surface_max_group_size,
                    "Scene::trace_refl_epc_field(): surface_group_members must contain surface_group_size * surface_max_group_size entries.");
        }
        require(!surface_group_ignore || has_surface_groups,
                "Scene::trace_refl_epc_field(): visibility_ignore_mode='surface_group' requires surface group tables.");
        require(final_ignore_group_count == 0 || surface_group_ignore,
                "Scene::trace_refl_epc_field(): final_ignore_group_ids require visibility_ignore_mode='surface_group'.");

        Mask active_detached = sanitize_segment_active<Detached>(
            tx_position,
            receiver,
            active);
        if (drjit::none(active_detached)) {
            return result;
        }

        const bool cuda_trace = triangle_kind_ == TraceBackendKind::Cuda;
        const OptixScene *primary_scene = nullptr;
        const OptixScene *secondary_scene = nullptr;
        int split_mode = 0;
        std::shared_ptr<OptixLaunchPipeline> *epc_pipeline = nullptr;
        if (!cuda_trace) {
            const OptixSceneSelection scenes = select_optix_scenes();
            primary_scene = scenes.primary;
            secondary_scene = scenes.secondary;
            split_mode = scenes.split_mode;
            const int hitgroup_record_count = scenes.hitgroup_record_count;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::trace_refl_epc_field(): OptiX scene is not ready.");
            require(hitgroup_record_count > 0,
                    "Scene::trace_refl_epc_field(): invalid hitgroup record count.");

            epc_pipeline = split_mode == 0 ? &reflection_epc_direct_primary_pipeline_
                                           : &reflection_epc_direct_pipeline_;
            const OptixPipelineConfig epc_pipeline_config =
                split_mode == 0 ? reflection_epc_direct_primary_pipeline_config()
                                : reflection_epc_direct_pipeline_config();

            ensure_pipeline(*epc_pipeline, primary_scene->context(),
                            hitgroup_record_count, epc_pipeline_config);
        }

        drjit::eval(tx_position,
                    receiver,
                    active_detached,
                    options.expected_prim_ids,
                    options.slot_plane_point,
                    options.slot_plane_normal,
                    options.slot_eta_r,
                    options.slot_mu_r,
                    options.slot_sigma,
                    options.slot_gain,
                    options.tx_polarization);
        if (has_surface_groups) {
            drjit::eval(options.surface_group_id,
                        options.surface_group_size,
                        options.surface_group_members);
        }
        if (final_ignore_group_count > 0) {
            drjit::eval(options.final_ignore_group_ids);
        }

        ensure_reflection_epc_geometry_ready();

        ReflEpcRaw raw = alloc_refl_epc_raw(ray_count, max_bounces);
        ReflEpcParams epc_params = {};
        epc_params.primary_handle = cuda_trace ? 0ull : primary_scene->ias_handle();
        epc_params.secondary_handle =
            (!cuda_trace && secondary_scene != nullptr && secondary_scene->is_ready())
                ? secondary_scene->ias_handle()
                : 0ull;
        epc_params.split_mode = split_mode;
        epc_params.tri_p0_x = triangle_info_detached_.p0.x().data();
        epc_params.tri_p0_y = triangle_info_detached_.p0.y().data();
        epc_params.tri_p0_z = triangle_info_detached_.p0.z().data();
        epc_params.tri_e1_x = triangle_info_detached_.e1.x().data();
        epc_params.tri_e1_y = triangle_info_detached_.e1.y().data();
        epc_params.tri_e1_z = triangle_info_detached_.e1.z().data();
        epc_params.tri_e2_x = triangle_info_detached_.e2.x().data();
        epc_params.tri_e2_y = triangle_info_detached_.e2.y().data();
        epc_params.tri_e2_z = triangle_info_detached_.e2.z().data();
        epc_params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        epc_params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        epc_params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        epc_params.face_offsets = face_offsets_.data();
        epc_params.n_meshes = mesh_count_;
        epc_params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        epc_params.expected_prim_ids = options.expected_prim_ids.data();
        epc_params.expected_prim_count = expected_prim_count;
        epc_params.surface_group_id =
            has_surface_groups ? options.surface_group_id.data() : nullptr;
        epc_params.surface_group_id_count = surface_group_id_count;
        epc_params.surface_group_size =
            has_surface_groups ? options.surface_group_size.data() : nullptr;
        epc_params.surface_group_count = surface_group_count;
        epc_params.surface_group_members =
            has_surface_groups ? options.surface_group_members.data() : nullptr;
        epc_params.surface_max_group_size =
            has_surface_groups ? options.surface_max_group_size : 0;
        epc_params.visibility_ignore_mode =
            surface_group_ignore ? ReflEpcVisibilityIgnoreSurfaceGroup
                                 : ReflEpcVisibilityIgnorePrimitive;
        epc_params.final_ignore_group_ids =
            final_ignore_group_count > 0 ? options.final_ignore_group_ids.data() : nullptr;
        epc_params.final_ignore_group_count = final_ignore_group_count;
        epc_params.ray_ox = tx_position.x().data();
        epc_params.ray_oy = tx_position.y().data();
        epc_params.ray_oz = tx_position.z().data();
        epc_params.ray_dx = nullptr;
        epc_params.ray_dy = nullptr;
        epc_params.ray_dz = nullptr;
        epc_params.ray_tmax = nullptr;
        epc_params.direct_plane_point_x = options.slot_plane_point.x().data();
        epc_params.direct_plane_point_y = options.slot_plane_point.y().data();
        epc_params.direct_plane_point_z = options.slot_plane_point.z().data();
        epc_params.direct_plane_normal_x = options.slot_plane_normal.x().data();
        epc_params.direct_plane_normal_y = options.slot_plane_normal.y().data();
        epc_params.direct_plane_normal_z = options.slot_plane_normal.z().data();
        epc_params.rx_x = receiver.x().data();
        epc_params.rx_y = receiver.y().data();
        epc_params.rx_z = receiver.z().data();
        epc_params.rx_count = receiver_count;
        epc_params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        epc_params.n_rays = ray_count;
        epc_params.max_bounces = max_bounces;
        epc_params.plane_tolerance = options.plane_tolerance;
        epc_params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        epc_params.out_bounce_count = raw.bounce_count.data();
        epc_params.out_path_length = raw.path_length.data();
        epc_params.out_point_x = raw.point_x.data();
        epc_params.out_point_y = raw.point_y.data();
        epc_params.out_point_z = raw.point_z.data();
        epc_params.out_trace_prim_ids = raw.trace_prim_ids.data();
        epc_params.out_resolved_prim_ids = raw.resolved_prim_ids.data();
        epc_params.out_surface_group_ids = raw.surface_group_ids.data();
        epc_params.out_plane_normal_x = raw.plane_normal_x.data();
        epc_params.out_plane_normal_y = raw.plane_normal_y.data();
        epc_params.out_plane_normal_z = raw.plane_normal_z.data();
        epc_params.out_first_blocked_segment = raw.first_blocked_segment.data();
        epc_params.out_first_blocked_prim = raw.first_blocked_prim.data();
        epc_params.out_first_blocked_group = raw.first_blocked_group.data();
        if (cuda_trace) {
            // Single-scene CUDA: split_mode is 0, so the direct-primary variant.
            cuda_backend().run_reflection_epc(epc_params, /*direct_only=*/true,
                                              /*primary_visibility_only=*/true, ray_count);
        } else {
            (*epc_pipeline)->launch(0, epc_params);
        }

        ReflEpcFieldParams field_params = {};
        field_params.n_rays = ray_count;
        field_params.max_bounces = max_bounces;
        field_params.epc_valid = reinterpret_cast<const uint8_t *>(raw.valid.data());
        field_params.epc_bounce_count = raw.bounce_count.data();
        field_params.epc_path_length = raw.path_length.data();
        field_params.ray_ox = tx_position.x().data();
        field_params.ray_oy = tx_position.y().data();
        field_params.ray_oz = tx_position.z().data();
        field_params.rx_x = receiver.x().data();
        field_params.rx_y = receiver.y().data();
        field_params.rx_z = receiver.z().data();
        field_params.rx_count = receiver_count;
        field_params.hit_x = raw.point_x.data();
        field_params.hit_y = raw.point_y.data();
        field_params.hit_z = raw.point_z.data();
        field_params.epc_normal_x = raw.plane_normal_x.data();
        field_params.epc_normal_y = raw.plane_normal_y.data();
        field_params.epc_normal_z = raw.plane_normal_z.data();
        const bool return_resolved_prim_ids =
            options.return_geom && options.return_resolved_prim_ids;
        const bool return_surface_group_ids =
            options.return_geom && options.return_surface_group_ids;
        field_params.resolved_prim_ids =
            return_resolved_prim_ids ? raw.resolved_prim_ids.data() : nullptr;
        field_params.surface_group_ids =
            return_surface_group_ids ? raw.surface_group_ids.data() : nullptr;
        field_params.slot_normal_x = options.slot_plane_normal.x().data();
        field_params.slot_normal_y = options.slot_plane_normal.y().data();
        field_params.slot_normal_z = options.slot_plane_normal.z().data();
        field_params.slot_eta_r = options.slot_eta_r.data();
        field_params.slot_mu_r = options.slot_mu_r.data();
        field_params.slot_sigma = options.slot_sigma.data();
        field_params.slot_gain = options.slot_gain.data();
        field_params.tx_pol_x = options.tx_polarization.x().data();
        field_params.tx_pol_y = options.tx_polarization.y().data();
        field_params.tx_pol_z = options.tx_polarization.z().data();
        field_params.tx_pol_count = tx_pol_count;
        field_params.omega = options.omega;
        field_params.wavelength = options.wavelength;
        field_params.out_valid = reinterpret_cast<uint8_t *>(result.valid.data());
        field_params.out_bounce_count = result.bounce_count.data();
        field_params.out_path_length = result.path_length.data();
        field_params.out_field_x_re = result.field_x_re.data();
        field_params.out_field_x_im = result.field_x_im.data();
        field_params.out_field_y_re = result.field_y_re.data();
        field_params.out_field_y_im = result.field_y_im.data();
        field_params.out_field_z_re = result.field_z_re.data();
        field_params.out_field_z_im = result.field_z_im.data();

        if (options.return_endpoints) {
            field_params.out_tx_x = result.tx_pos.x().data();
            field_params.out_tx_y = result.tx_pos.y().data();
            field_params.out_tx_z = result.tx_pos.z().data();
            field_params.out_first_hit_x = result.first_hit.x().data();
            field_params.out_first_hit_y = result.first_hit.y().data();
            field_params.out_first_hit_z = result.first_hit.z().data();
            field_params.out_last_hit_x = result.last_hit.x().data();
            field_params.out_last_hit_y = result.last_hit.y().data();
            field_params.out_last_hit_z = result.last_hit.z().data();
        }
        if (options.return_geom && options.return_hit_points) {
            field_params.out_hit_x = result.hit_points.x().data();
            field_params.out_hit_y = result.hit_points.y().data();
            field_params.out_hit_z = result.hit_points.z().data();
        }
        if (options.return_geom && options.return_normals) {
            field_params.out_normal_x = result.normals.x().data();
            field_params.out_normal_y = result.normals.y().data();
            field_params.out_normal_z = result.normals.z().data();
        }
        if (return_resolved_prim_ids) {
            field_params.out_resolved_prim_ids = result.resolved_prim_ids.data();
        }
        if (return_surface_group_ids) {
            field_params.out_surface_group_ids = result.surface_group_ids.data();
        }

        reflection_epc_field_gpu(field_params);
        return result;
    }
}

template ReflEpc Scene::trace_refl_epc<true>(
    const Ray &ray,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    Mask active) const;
template ReflEpcAD Scene::trace_refl_epc<false>(
    const RayAD &ray,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    MaskAD active) const;
template ReflEpcField Scene::trace_refl_epc_field<true>(
    const Ray &ray,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcFieldOptions &options,
    Mask active) const;
template ReflEpcFieldAD Scene::trace_refl_epc_field<false>(
    const RayAD &ray,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsAD &options,
    MaskAD active) const;
template ReflEpcField Scene::trace_refl_epc_field<true>(
    const Vector3f &tx_position,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcFieldOptions &options,
    Mask active) const;
template ReflEpcFieldAD Scene::trace_refl_epc_field<false>(
    const Vector3fAD &tx_position,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsAD &options,
    MaskAD active) const;

} // namespace rayd
