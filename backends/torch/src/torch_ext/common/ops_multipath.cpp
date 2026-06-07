#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <raydtorch/diffraction/accum_params.h>
#include <raydtorch/diffraction/accum_ad.h>
#include <raydtorch/diffraction/paths_params.h>
#include <raydtorch/diffraction/pipeline.h>
#include <raydtorch/scene/geometry_kernels.h>
#include <raydtorch/common/optix_pipeline.h>
#include <raydtorch/reflection/kernels.h>
#include <raydtorch/reflection/pipeline.h>
#include <raydtorch/common/optix_context.h>
#include <raydtorch/reflection/accum_params.h>
#include <raydtorch/reflection/dedup.h>
#include <raydtorch/reflection/epc_field.h>
#include <raydtorch/reflection/epc_params.h>
#include <raydtorch/reflection/trace_params.h>
#include <raydtorch/reflection/visibility_params.h>
#include <raydtorch/scene/cache.h>
#include <raydtorch/common/tensor_check.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace raydtorch {

namespace {

void require_same_batch(const at::Tensor &a, const at::Tensor &b, const char *name) {
    if (a.size(0) != b.size(0))
        throw std::runtime_error(std::string(name) + " tensors must have the same batch size.");
}

void require_flat_i32(const at::Tensor &tensor, const char *name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kInt, name);
    require_rank(tensor, 1, name);
}

void require_flat_f32(const at::Tensor &tensor, const char *name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 1, name);
}

void require_state_width(const at::Tensor &tensor, int64_t state_count, const char *name) {
    if (tensor.size(0) < state_count)
        throw std::runtime_error(std::string(name) + " must cover state_count.");
}

int32_t checked_i32(int64_t value, const char *name) {
    if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error(std::string(name) + " does not fit in int32.");
    return static_cast<int32_t>(value);
}

at::Tensor active_mask_for_states(const at::Tensor &active, int64_t state_count, const char *name) {
    if (active.size(0) == state_count)
        return active.contiguous();
    if (active.size(0) == 1)
        return active.expand({state_count}).contiguous();
    throw std::runtime_error(std::string(name) + " active width must be 1 or match state_count.");
}

at::Tensor first_bounce_column(const at::Tensor &value, int64_t ray_count) {
    if (value.dim() == 1)
        return value.reshape({ray_count}).contiguous();
    return value.slice(1, 0, 1).reshape({ray_count}).contiguous();
}

struct Vec3SoA {
    at::Tensor x;
    at::Tensor y;
    at::Tensor z;
};

Vec3SoA split_vec3(const at::Tensor &value) {
    return {
        value.select(1, 0).contiguous(),
        value.select(1, 1).contiguous(),
        value.select(1, 2).contiguous(),
    };
}

struct TriangleSoA {
    at::Tensor p0_x;
    at::Tensor p0_y;
    at::Tensor p0_z;
    at::Tensor e1_x;
    at::Tensor e1_y;
    at::Tensor e1_z;
    at::Tensor e2_x;
    at::Tensor e2_y;
    at::Tensor e2_z;
    at::Tensor fn_x;
    at::Tensor fn_y;
    at::Tensor fn_z;
    at::Tensor face_offsets;
    int32_t n_triangles = 0;
};

TriangleSoA make_triangle_soa(const MeshRecord &mesh) {
    at::Tensor faces_i64 = mesh.faces.to(at::kLong);
    at::Tensor v0 = mesh.vertices.index_select(0, faces_i64.select(1, 0)).contiguous();
    at::Tensor v1 = mesh.vertices.index_select(0, faces_i64.select(1, 1)).contiguous();
    at::Tensor v2 = mesh.vertices.index_select(0, faces_i64.select(1, 2)).contiguous();
    at::Tensor e1 = (v1 - v0).contiguous();
    at::Tensor e2 = (v2 - v0).contiguous();
    at::Tensor fn = at::cross(e1, e2, 1).contiguous();
    return {
        v0.select(1, 0).contiguous(),
        v0.select(1, 1).contiguous(),
        v0.select(1, 2).contiguous(),
        e1.select(1, 0).contiguous(),
        e1.select(1, 1).contiguous(),
        e1.select(1, 2).contiguous(),
        e2.select(1, 0).contiguous(),
        e2.select(1, 1).contiguous(),
        e2.select(1, 2).contiguous(),
        fn.select(1, 0).contiguous(),
        fn.select(1, 1).contiguous(),
        fn.select(1, 2).contiguous(),
        at::zeros({1}, mesh.faces.options()),
        static_cast<int32_t>(mesh.faces.size(0)),
    };
}

TriangleSoA make_scene_triangle_soa(const SceneCache &scene) {
    at::Tensor faces_i64 = scene.global_faces.to(at::kLong);
    at::Tensor v0 = scene.global_vertices.index_select(0, faces_i64.select(1, 0)).contiguous();
    at::Tensor v1 = scene.global_vertices.index_select(0, faces_i64.select(1, 1)).contiguous();
    at::Tensor v2 = scene.global_vertices.index_select(0, faces_i64.select(1, 2)).contiguous();
    at::Tensor e1 = (v1 - v0).contiguous();
    at::Tensor e2 = (v2 - v0).contiguous();
    at::Tensor fn = at::cross(e1, e2, 1).contiguous();
    return {
        v0.select(1, 0).contiguous(),
        v0.select(1, 1).contiguous(),
        v0.select(1, 2).contiguous(),
        e1.select(1, 0).contiguous(),
        e1.select(1, 1).contiguous(),
        e1.select(1, 2).contiguous(),
        e2.select(1, 0).contiguous(),
        e2.select(1, 1).contiguous(),
        e2.select(1, 2).contiguous(),
        fn.select(1, 0).contiguous(),
        fn.select(1, 1).contiguous(),
        fn.select(1, 2).contiguous(),
        scene.face_offsets.contiguous(),
        static_cast<int32_t>(scene.global_faces.size(0)),
    };
}

const uint8_t *mask_ptr(const at::Tensor &mask) {
    return reinterpret_cast<const uint8_t *>(mask.data_ptr<bool>());
}

uint8_t *mutable_mask_ptr(const at::Tensor &mask) {
    return reinterpret_cast<uint8_t *>(mask.data_ptr<bool>());
}

at::Tensor stack_vec3(const at::Tensor &x, const at::Tensor &y, const at::Tensor &z) {
    return at::stack({x, y, z}, 1).contiguous();
}

std::shared_ptr<OptixLaunchPipeline> optix_pipeline_for_scene(
    const SceneCache &scene,
    const OptixPipelineConfig &config) {
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    return shared_optix_launch_pipeline(
        optix_entry.optix_context,
        static_cast<int>(scene.device_index),
        1,
        config);
}

} // namespace

py::tuple visibility_forward_op(
    int64_t scene_handle,
    at::Tensor start,
    at::Tensor end,
    at::Tensor active) {
    require_vec3f(start, "start");
    require_vec3f(end, "end");
    require_mask(active, "active");
    require_same_batch(start, end, "visibility");
    if (active.size(0) != start.size(0))
        throw std::runtime_error("active must match the visibility batch size.");

    SceneCache &scene = get_scene(scene_handle);
    if (scene.meshes.size() != 1)
        throw std::runtime_error("visibility_forward: first milestone supports exactly one mesh.");
    const int64_t ray_count = start.size(0);
    at::Tensor visible = at::empty({ray_count}, active.options());
    at::Tensor blocker_prim = at::full({ray_count}, -1, scene.meshes[0].faces.options());
    at::Tensor tape_t = at::full(
        {ray_count},
        std::numeric_limits<float>::infinity(),
        start.options());
    if (ray_count == 0)
        return py::make_tuple(visible, blocker_prim, tape_t);

    Vec3SoA start_soa = split_vec3(start);
    Vec3SoA end_soa = split_vec3(end);
    at::Tensor active_contig = active.contiguous();
    at::Tensor face_offsets = at::zeros({1}, scene.meshes[0].faces.options());

    SegmentVisibilityParams params = {};
    params.handle = scene.triangle_ias.traversable;
    params.face_offsets = face_offsets.data_ptr<int>();
    params.n_meshes = 1;
    params.start_x = start_soa.x.data_ptr<float>();
    params.start_y = start_soa.y.data_ptr<float>();
    params.start_z = start_soa.z.data_ptr<float>();
    params.end_x = end_soa.x.data_ptr<float>();
    params.end_y = end_soa.y.data_ptr<float>();
    params.end_z = end_soa.z.data_ptr<float>();
    params.active_mask = mask_ptr(active_contig);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.out_visible = mutable_mask_ptr(visible);
    params.out_first_blocked_prim = blocker_prim.data_ptr<int>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    optix_pipeline_for_scene(scene, segment_visibility_pipeline_config())
        ->launch(0, params, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    return py::make_tuple(visible, blocker_prim, tape_t);
}

py::tuple trace_reflections_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    int64_t max_bounces) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    require_same_batch(ray_o, ray_d, "trace_reflections");
    if (ray_tmax.size(0) != ray_o.size(0) || active.size(0) != ray_o.size(0))
        throw std::runtime_error("ray_tmax and active must match the ray batch size.");
    if (max_bounces < 1)
        throw std::runtime_error("max_bounces must be at least 1.");

    SceneCache &scene = get_scene(scene_handle);
    if (scene.meshes.size() != 1)
        throw std::runtime_error("trace_reflections_forward: first milestone supports exactly one mesh.");
    const MeshRecord &mesh = scene.meshes[0];

    const int64_t ray_count = ray_o.size(0);
    auto fopts = ray_o.options();
    auto iopts = mesh.faces.options();
    at::Tensor t = at::full(
        {ray_count, max_bounces},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor prim_ids = at::full({ray_count, max_bounces}, -1, iopts);
    at::Tensor local_prim_ids = at::full({ray_count, max_bounces}, -1, iopts);
    at::Tensor shape_ids = at::zeros({ray_count, max_bounces}, iopts);
    at::Tensor bounce_count = at::zeros({ray_count}, iopts);
    at::Tensor bary_u = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor bary_v = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor hit_x = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor hit_y = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor hit_z = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor norm_x = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor norm_y = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor norm_z = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor img_x = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor img_y = at::zeros({ray_count, max_bounces}, fopts);
    at::Tensor img_z = at::zeros({ray_count, max_bounces}, fopts);
    if (ray_count == 0) {
        at::Tensor valid = at::zeros({ray_count, max_bounces}, active.options());
        at::Tensor image_sources = at::zeros({ray_count, max_bounces, 3}, fopts);
        at::Tensor tape_prim_id = at::full({ray_count}, -1, iopts);
        at::Tensor tape_barycentric = at::zeros({ray_count, 2}, fopts);
        at::Tensor tape_t = at::full(
            {ray_count},
            std::numeric_limits<float>::infinity(),
            fopts);
        return py::make_tuple(
            valid,
            t,
            image_sources,
            prim_ids,
            tape_prim_id,
            tape_barycentric,
            tape_t);
    }

    TriangleSoA tri = make_triangle_soa(mesh);
    Vec3SoA ray_o_soa = split_vec3(ray_o);
    Vec3SoA ray_d_soa = split_vec3(ray_d);
    at::Tensor ray_tmax_contig = ray_tmax.contiguous();
    at::Tensor active_contig = active.contiguous();

    TorchCudaContext torch_ctx = current_torch_cuda_context();

    ReflectionTraceParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = 1;
    params.n_triangles = tri.n_triangles;
    params.ray_ox = ray_o_soa.x.data_ptr<float>();
    params.ray_oy = ray_o_soa.y.data_ptr<float>();
    params.ray_oz = ray_o_soa.z.data_ptr<float>();
    params.ray_dx = ray_d_soa.x.data_ptr<float>();
    params.ray_dy = ray_d_soa.y.data_ptr<float>();
    params.ray_dz = ray_d_soa.z.data_ptr<float>();
    params.ray_tmax = ray_tmax_contig.data_ptr<float>();
    params.active_mask = mask_ptr(active_contig);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.max_bounces = static_cast<int32_t>(max_bounces);
    params.export_mode = 0;
    params.return_trailing = 0;
    params.out_bounce_count = bounce_count.data_ptr<int>();
    params.out_shape_ids = shape_ids.data_ptr<int>();
    params.out_prim_ids = local_prim_ids.data_ptr<int>();
    params.out_global_prim_ids = prim_ids.data_ptr<int>();
    params.out_t = t.data_ptr<float>();
    params.out_bary_u = bary_u.data_ptr<float>();
    params.out_bary_v = bary_v.data_ptr<float>();
    params.out_hit_x = hit_x.data_ptr<float>();
    params.out_hit_y = hit_y.data_ptr<float>();
    params.out_hit_z = hit_z.data_ptr<float>();
    params.out_norm_x = norm_x.data_ptr<float>();
    params.out_norm_y = norm_y.data_ptr<float>();
    params.out_norm_z = norm_z.data_ptr<float>();
    params.out_img_x = img_x.data_ptr<float>();
    params.out_img_y = img_y.data_ptr<float>();
    params.out_img_z = img_z.data_ptr<float>();

    optix_pipeline_for_scene(scene, reflection_trace_pipeline_config())
        ->launch(0, params, static_cast<unsigned int>(ray_count), torch_ctx.stream);

    at::Tensor bounce_index =
        at::arange(max_bounces, at::TensorOptions().device(ray_o.device()).dtype(at::kLong))
            .reshape({1, max_bounces});
    at::Tensor valid = bounce_index.lt(bounce_count.to(at::kLong).reshape({ray_count, 1}));
    at::Tensor image_sources = at::stack({img_x, img_y, img_z}, 2).contiguous();
    at::Tensor tape_prim_id = prim_ids.slice(1, 0, 1).reshape({ray_count}).contiguous();
    at::Tensor tape_barycentric = at::stack(
        {
            bary_u.slice(1, 0, 1).reshape({ray_count}),
            bary_v.slice(1, 0, 1).reshape({ray_count}),
        },
        1).contiguous();
    at::Tensor tape_t = t.slice(1, 0, 1).reshape({ray_count}).contiguous();

    return py::make_tuple(
        valid,
        t,
        image_sources,
        prim_ids,
        tape_prim_id,
        tape_barycentric,
        tape_t);
}

py::tuple trace_reflections_backward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor grad_t) {
    SceneCache &scene = get_scene(scene_handle);
    const MeshRecord &mesh = scene.meshes[0];
    ReflectionBackwardOutputs out = reflection_backward_cuda(
        mesh.vertices,
        mesh.faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        first_bounce_column(grad_t, ray_o.size(0)));
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple trace_reflections_jvp_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tangent_vertices,
    at::Tensor tangent_ray_o,
    at::Tensor tangent_ray_d,
    at::Tensor image_sources) {
    SceneCache &scene = get_scene(scene_handle);
    const MeshRecord &mesh = scene.meshes[0];
    ReflectionJvpOutputs out = reflection_jvp_cuda(
        mesh.vertices,
        mesh.faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices.contiguous(),
        tangent_ray_o.contiguous(),
        tangent_ray_d.contiguous(),
        image_sources);
    return py::make_tuple(out.tangent_t, out.tangent_image_sources);
}

py::tuple trace_refl_epc_field_forward_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor active,
    int64_t max_bounces) {
    require_vec3f(source, "source");
    require_vec3f(receiver, "receiver");
    require_mask(active, "active");
    require_same_batch(source, receiver, "trace_refl_epc_field");
    if (active.size(0) != source.size(0))
        throw std::runtime_error("active must match the EPC batch size.");
    if (max_bounces < 1)
        throw std::runtime_error("max_bounces must be at least 1.");

    SceneCache &scene = get_scene(scene_handle);
    if (scene.meshes.size() != 1)
        throw std::runtime_error("trace_refl_epc_field_forward: first milestone supports exactly one mesh.");
    const MeshRecord &mesh = scene.meshes[0];
    const int64_t ray_count = source.size(0);
    const int64_t slot_count = ray_count * max_bounces;
    auto fopts = source.options();
    auto iopts = mesh.faces.options();
    at::Tensor field_real = at::zeros({ray_count}, fopts);
    at::Tensor field_imag = at::zeros({ray_count}, fopts);
    at::Tensor path_length = at::full(
        {ray_count},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor valid = at::zeros({ray_count}, active.options());
    at::Tensor resolved_first = at::full({ray_count}, -1, iopts);
    at::Tensor tape_prim_id = at::full({ray_count}, -1, iopts);
    at::Tensor tape_barycentric = at::zeros({ray_count, 2}, fopts);
    at::Tensor tape_t = at::full(
        {ray_count},
        std::numeric_limits<float>::infinity(),
        fopts);
    if (ray_count == 0) {
        return py::make_tuple(
            field_real,
            field_imag,
            path_length,
            valid,
            resolved_first,
            tape_prim_id,
            tape_barycentric,
            tape_t);
    }

    at::Tensor ray_d = (receiver - source).contiguous();
    at::Tensor ray_tmax = at::sqrt(at::sum(ray_d * ray_d, {1})).contiguous();
    TriangleSoA tri = make_triangle_soa(mesh);
    Vec3SoA source_soa = split_vec3(source);
    Vec3SoA ray_d_soa = split_vec3(ray_d);
    Vec3SoA receiver_soa = split_vec3(receiver);
    at::Tensor active_contig = active.contiguous();

    at::Tensor epc_valid = at::zeros({ray_count}, active.options());
    at::Tensor epc_bounce_count = at::zeros({ray_count}, iopts);
    at::Tensor epc_path_length = at::full(
        {ray_count},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor point_x = at::zeros({slot_count}, fopts);
    at::Tensor point_y = at::zeros({slot_count}, fopts);
    at::Tensor point_z = at::zeros({slot_count}, fopts);
    at::Tensor trace_prim_ids = at::full({slot_count}, -1, iopts);
    at::Tensor resolved_prim_ids = at::full({slot_count}, -1, iopts);
    at::Tensor surface_group_ids = at::full({slot_count}, -1, iopts);
    at::Tensor plane_normal_x = at::zeros({slot_count}, fopts);
    at::Tensor plane_normal_y = at::zeros({slot_count}, fopts);
    at::Tensor plane_normal_z = at::zeros({slot_count}, fopts);
    at::Tensor first_blocked_segment = at::full({ray_count}, -1, iopts);
    at::Tensor first_blocked_prim = at::full({ray_count}, -1, iopts);
    at::Tensor first_blocked_group = at::full({ray_count}, -1, iopts);

    ReflEpcParams epc_params = {};
    epc_params.primary_handle = scene.triangle_ias.traversable;
    epc_params.secondary_handle = 0;
    epc_params.split_mode = 0;
    epc_params.tri_p0_x = tri.p0_x.data_ptr<float>();
    epc_params.tri_p0_y = tri.p0_y.data_ptr<float>();
    epc_params.tri_p0_z = tri.p0_z.data_ptr<float>();
    epc_params.tri_e1_x = tri.e1_x.data_ptr<float>();
    epc_params.tri_e1_y = tri.e1_y.data_ptr<float>();
    epc_params.tri_e1_z = tri.e1_z.data_ptr<float>();
    epc_params.tri_e2_x = tri.e2_x.data_ptr<float>();
    epc_params.tri_e2_y = tri.e2_y.data_ptr<float>();
    epc_params.tri_e2_z = tri.e2_z.data_ptr<float>();
    epc_params.tri_fn_x = tri.fn_x.data_ptr<float>();
    epc_params.tri_fn_y = tri.fn_y.data_ptr<float>();
    epc_params.tri_fn_z = tri.fn_z.data_ptr<float>();
    epc_params.face_offsets = tri.face_offsets.data_ptr<int>();
    epc_params.n_meshes = 1;
    epc_params.n_triangles = tri.n_triangles;
    epc_params.visibility_ignore_mode = ReflEpcVisibilityIgnorePrimitive;
    epc_params.ray_ox = source_soa.x.data_ptr<float>();
    epc_params.ray_oy = source_soa.y.data_ptr<float>();
    epc_params.ray_oz = source_soa.z.data_ptr<float>();
    epc_params.ray_dx = ray_d_soa.x.data_ptr<float>();
    epc_params.ray_dy = ray_d_soa.y.data_ptr<float>();
    epc_params.ray_dz = ray_d_soa.z.data_ptr<float>();
    epc_params.ray_tmax = ray_tmax.data_ptr<float>();
    epc_params.rx_x = receiver_soa.x.data_ptr<float>();
    epc_params.rx_y = receiver_soa.y.data_ptr<float>();
    epc_params.rx_z = receiver_soa.z.data_ptr<float>();
    epc_params.rx_count = static_cast<int32_t>(ray_count);
    epc_params.active_mask = mask_ptr(active_contig);
    epc_params.n_rays = static_cast<int32_t>(ray_count);
    epc_params.max_bounces = static_cast<int32_t>(max_bounces);
    epc_params.out_valid = mutable_mask_ptr(epc_valid);
    epc_params.out_bounce_count = epc_bounce_count.data_ptr<int>();
    epc_params.out_path_length = epc_path_length.data_ptr<float>();
    epc_params.out_point_x = point_x.data_ptr<float>();
    epc_params.out_point_y = point_y.data_ptr<float>();
    epc_params.out_point_z = point_z.data_ptr<float>();
    epc_params.out_trace_prim_ids = trace_prim_ids.data_ptr<int>();
    epc_params.out_resolved_prim_ids = resolved_prim_ids.data_ptr<int>();
    epc_params.out_surface_group_ids = surface_group_ids.data_ptr<int>();
    epc_params.out_plane_normal_x = plane_normal_x.data_ptr<float>();
    epc_params.out_plane_normal_y = plane_normal_y.data_ptr<float>();
    epc_params.out_plane_normal_z = plane_normal_z.data_ptr<float>();
    epc_params.out_first_blocked_segment = first_blocked_segment.data_ptr<int>();
    epc_params.out_first_blocked_prim = first_blocked_prim.data_ptr<int>();
    epc_params.out_first_blocked_group = first_blocked_group.data_ptr<int>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    optix_pipeline_for_scene(scene, reflection_epc_pipeline_config())
        ->launch(0, epc_params, static_cast<unsigned int>(ray_count), torch_ctx.stream);

    at::Tensor slot_eta_r = at::ones({slot_count}, fopts);
    at::Tensor slot_mu_r = at::ones({slot_count}, fopts);
    at::Tensor slot_sigma = at::zeros({slot_count}, fopts);
    at::Tensor slot_gain = at::ones({slot_count}, fopts);
    at::Tensor tx_pol_x = at::ones({1}, fopts);
    at::Tensor tx_pol_y = at::zeros({1}, fopts);
    at::Tensor tx_pol_z = at::zeros({1}, fopts);
    at::Tensor field_y_re = at::zeros({ray_count}, fopts);
    at::Tensor field_y_im = at::zeros({ray_count}, fopts);
    at::Tensor field_z_re = at::zeros({ray_count}, fopts);
    at::Tensor field_z_im = at::zeros({ray_count}, fopts);

    ReflEpcFieldParams field_params = {};
    field_params.n_rays = static_cast<int32_t>(ray_count);
    field_params.max_bounces = static_cast<int32_t>(max_bounces);
    field_params.epc_valid = mask_ptr(epc_valid);
    field_params.epc_bounce_count = epc_bounce_count.data_ptr<int>();
    field_params.epc_path_length = epc_path_length.data_ptr<float>();
    field_params.ray_ox = source_soa.x.data_ptr<float>();
    field_params.ray_oy = source_soa.y.data_ptr<float>();
    field_params.ray_oz = source_soa.z.data_ptr<float>();
    field_params.rx_x = receiver_soa.x.data_ptr<float>();
    field_params.rx_y = receiver_soa.y.data_ptr<float>();
    field_params.rx_z = receiver_soa.z.data_ptr<float>();
    field_params.rx_count = static_cast<int32_t>(ray_count);
    field_params.hit_x = point_x.data_ptr<float>();
    field_params.hit_y = point_y.data_ptr<float>();
    field_params.hit_z = point_z.data_ptr<float>();
    field_params.epc_normal_x = plane_normal_x.data_ptr<float>();
    field_params.epc_normal_y = plane_normal_y.data_ptr<float>();
    field_params.epc_normal_z = plane_normal_z.data_ptr<float>();
    field_params.resolved_prim_ids = resolved_prim_ids.data_ptr<int>();
    field_params.surface_group_ids = surface_group_ids.data_ptr<int>();
    field_params.slot_normal_x = plane_normal_x.data_ptr<float>();
    field_params.slot_normal_y = plane_normal_y.data_ptr<float>();
    field_params.slot_normal_z = plane_normal_z.data_ptr<float>();
    field_params.slot_eta_r = slot_eta_r.data_ptr<float>();
    field_params.slot_mu_r = slot_mu_r.data_ptr<float>();
    field_params.slot_sigma = slot_sigma.data_ptr<float>();
    field_params.slot_gain = slot_gain.data_ptr<float>();
    field_params.tx_pol_x = tx_pol_x.data_ptr<float>();
    field_params.tx_pol_y = tx_pol_y.data_ptr<float>();
    field_params.tx_pol_z = tx_pol_z.data_ptr<float>();
    field_params.tx_pol_count = 1;
    field_params.omega = 2.0f * 3.14159265358979323846f * 299792458.0f;
    field_params.wavelength = 1.0f;
    field_params.out_valid = mutable_mask_ptr(valid);
    field_params.out_bounce_count = epc_bounce_count.data_ptr<int>();
    field_params.out_path_length = path_length.data_ptr<float>();
    field_params.out_field_x_re = field_real.data_ptr<float>();
    field_params.out_field_x_im = field_imag.data_ptr<float>();
    field_params.out_field_y_re = field_y_re.data_ptr<float>();
    field_params.out_field_y_im = field_y_im.data_ptr<float>();
    field_params.out_field_z_re = field_z_re.data_ptr<float>();
    field_params.out_field_z_im = field_z_im.data_ptr<float>();
    reflection_epc_field_gpu(field_params);

    resolved_first = resolved_prim_ids.reshape({ray_count, max_bounces})
                         .slice(1, 0, 1)
                         .reshape({ray_count})
                         .contiguous();
    tape_prim_id = trace_prim_ids.reshape({ray_count, max_bounces})
                       .slice(1, 0, 1)
                       .reshape({ray_count})
                       .contiguous();
    tape_t = path_length.contiguous();

    return py::make_tuple(
        field_real,
        field_imag,
        path_length,
        valid,
        resolved_first,
        tape_prim_id,
        tape_barycentric,
        tape_t);
}

py::tuple trace_refl_epc_field_backward_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_t,
    at::Tensor grad_field_real,
    at::Tensor grad_field_imag,
    at::Tensor grad_path_length) {
    SceneCache &scene = get_scene(scene_handle);
    const MeshRecord &mesh = scene.meshes[0];
    ReflEpcBackwardOutputs out = refl_epc_backward_cuda(
        mesh.vertices,
        mesh.faces,
        source,
        receiver,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        grad_field_real.contiguous(),
        grad_field_imag.contiguous(),
        grad_path_length.contiguous());
    return py::make_tuple(out.grad_vertices, out.grad_source, out.grad_receiver);
}

py::tuple trace_refl_epc_field_jvp_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_t,
    at::Tensor tangent_vertices,
    at::Tensor tangent_source,
    at::Tensor tangent_receiver) {
    SceneCache &scene = get_scene(scene_handle);
    const MeshRecord &mesh = scene.meshes[0];
    ReflEpcJvpOutputs out = refl_epc_jvp_cuda(
        mesh.vertices,
        mesh.faces,
        source,
        receiver,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        tangent_vertices.contiguous(),
        tangent_source.contiguous(),
        tangent_receiver.contiguous());
    return py::make_tuple(out.tangent_field_real, out.tangent_field_imag, out.tangent_path_length);
}

py::tuple reflection_dedup_forward_op(
    at::Tensor bounce_count,
    at::Tensor shape_ids,
    at::Tensor prim_ids,
    at::Tensor t,
    at::Tensor bary_u,
    at::Tensor bary_v,
    at::Tensor hit_x,
    at::Tensor hit_y,
    at::Tensor hit_z,
    at::Tensor norm_x,
    at::Tensor norm_y,
    at::Tensor norm_z,
    at::Tensor img_x,
    at::Tensor img_y,
    at::Tensor img_z,
    int64_t max_bounces,
    double image_source_tolerance) {
    if (bounce_count.dim() != 1)
        throw std::runtime_error("bounce_count must be flat.");
    if (max_bounces <= 0)
        throw std::runtime_error("max_bounces must be positive.");
    const int64_t ray_count = bounce_count.size(0);
    const int64_t slot_count = ray_count * max_bounces;
    auto iopts = bounce_count.options();
    auto fopts = t.options();
    at::Tensor out_bounce_count = at::zeros({ray_count}, iopts);
    at::Tensor out_shape_ids = at::full({slot_count}, -1, iopts);
    at::Tensor out_prim_ids = at::full({slot_count}, -1, iopts);
    at::Tensor out_t = at::full(
        {slot_count},
        std::numeric_limits<float>::infinity(),
        fopts);
    at::Tensor out_bary_u = at::zeros({slot_count}, fopts);
    at::Tensor out_bary_v = at::zeros({slot_count}, fopts);
    at::Tensor out_hit_x = at::zeros({slot_count}, fopts);
    at::Tensor out_hit_y = at::zeros({slot_count}, fopts);
    at::Tensor out_hit_z = at::zeros({slot_count}, fopts);
    at::Tensor out_norm_x = at::zeros({slot_count}, fopts);
    at::Tensor out_norm_y = at::zeros({slot_count}, fopts);
    at::Tensor out_norm_z = at::zeros({slot_count}, fopts);
    at::Tensor out_img_x = at::zeros({slot_count}, fopts);
    at::Tensor out_img_y = at::zeros({slot_count}, fopts);
    at::Tensor out_img_z = at::zeros({slot_count}, fopts);
    at::Tensor out_discovery_count = at::zeros({ray_count}, iopts);
    at::Tensor out_representative = at::full({ray_count}, -1, iopts);
    at::Tensor face_offsets = at::zeros({1}, iopts);

    int unique_count = 0;
    if (ray_count > 0) {
        at::Tensor bounce_count_c = bounce_count.contiguous();
        at::Tensor shape_ids_c = shape_ids.contiguous();
        at::Tensor prim_ids_c = prim_ids.contiguous();
        at::Tensor t_c = t.contiguous();
        at::Tensor bary_u_c = bary_u.contiguous();
        at::Tensor bary_v_c = bary_v.contiguous();
        at::Tensor hit_x_c = hit_x.contiguous();
        at::Tensor hit_y_c = hit_y.contiguous();
        at::Tensor hit_z_c = hit_z.contiguous();
        at::Tensor norm_x_c = norm_x.contiguous();
        at::Tensor norm_y_c = norm_y.contiguous();
        at::Tensor norm_z_c = norm_z.contiguous();
        at::Tensor img_x_c = img_x.contiguous();
        at::Tensor img_y_c = img_y.contiguous();
        at::Tensor img_z_c = img_z.contiguous();
        unique_count = reflection_dedup_gpu(
            static_cast<int32_t>(ray_count),
            static_cast<int32_t>(max_bounces),
            bounce_count_c.data_ptr<int>(),
            shape_ids_c.data_ptr<int>(),
            prim_ids_c.data_ptr<int>(),
            t_c.data_ptr<float>(),
            bary_u_c.data_ptr<float>(),
            bary_v_c.data_ptr<float>(),
            hit_x_c.data_ptr<float>(),
            hit_y_c.data_ptr<float>(),
            hit_z_c.data_ptr<float>(),
            norm_x_c.data_ptr<float>(),
            norm_y_c.data_ptr<float>(),
            norm_z_c.data_ptr<float>(),
            img_x_c.data_ptr<float>(),
            img_y_c.data_ptr<float>(),
            img_z_c.data_ptr<float>(),
            face_offsets.data_ptr<int>(),
            1,
            nullptr,
            0,
            static_cast<float>(image_source_tolerance),
            out_bounce_count.data_ptr<int>(),
            out_shape_ids.data_ptr<int>(),
            out_prim_ids.data_ptr<int>(),
            out_t.data_ptr<float>(),
            out_bary_u.data_ptr<float>(),
            out_bary_v.data_ptr<float>(),
            out_hit_x.data_ptr<float>(),
            out_hit_y.data_ptr<float>(),
            out_hit_z.data_ptr<float>(),
            out_norm_x.data_ptr<float>(),
            out_norm_y.data_ptr<float>(),
            out_norm_z.data_ptr<float>(),
            out_img_x.data_ptr<float>(),
            out_img_y.data_ptr<float>(),
            out_img_z.data_ptr<float>(),
            out_discovery_count.data_ptr<int>(),
            out_representative.data_ptr<int>());
    }

    return py::make_tuple(
        unique_count,
        out_bounce_count,
        out_shape_ids,
        out_prim_ids,
        out_t,
        out_bary_u,
        out_bary_v,
        out_hit_x,
        out_hit_y,
        out_hit_z,
        out_norm_x,
        out_norm_y,
        out_norm_z,
        out_img_x,
        out_img_y,
        out_img_z,
        out_discovery_count,
        out_representative);
}

py::tuple reflection_accumulation_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tx,
    at::Tensor tx_pol,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_scalar_f(ray_tmax, "ray_tmax");
    require_mask(active, "active");
    require_vec3f(tx, "tx");
    require_vec3f(tx_pol, "tx_pol");
    require_same_batch(ray_o, ray_d, "reflection_accumulation");
    require_same_batch(ray_o, tx, "reflection_accumulation");
    require_same_batch(ray_o, tx_pol, "reflection_accumulation");
    if (ray_tmax.size(0) != ray_o.size(0) || active.size(0) != ray_o.size(0))
        throw std::runtime_error("ray_tmax and active must match the ray batch size.");
    if (max_bounces < 0)
        throw std::runtime_error("max_bounces must be non-negative.");
    if (grid_axis < 0 || grid_axis > 2)
        throw std::runtime_error("grid_axis must be 0, 1, or 2.");
    if (grid_resolution0 <= 0 || grid_resolution1 <= 0)
        throw std::runtime_error("grid resolutions must be positive.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");

    SceneCache &scene = get_scene(scene_handle);
    if (scene.meshes.size() != 1)
        throw std::runtime_error("reflection_accumulation_forward: first milestone supports exactly one mesh.");
    const MeshRecord &mesh = scene.meshes[0];
    const int64_t ray_count = ray_o.size(0);
    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    auto fopts = ray_o.options();
    auto iopts = mesh.faces.options();
    at::Tensor power = at::zeros({cell_count}, fopts);
    at::Tensor field_x_re = at::zeros({cell_count}, fopts);
    at::Tensor field_x_im = at::zeros({cell_count}, fopts);
    at::Tensor field_y_re = at::zeros({cell_count}, fopts);
    at::Tensor field_y_im = at::zeros({cell_count}, fopts);
    at::Tensor field_z_re = at::zeros({cell_count}, fopts);
    at::Tensor field_z_im = at::zeros({cell_count}, fopts);
    at::Tensor reflection_count = at::zeros({1}, iopts);
    if (ray_count == 0) {
        return py::make_tuple(
            power.reshape({grid_resolution1, grid_resolution0}),
            field_x_re.reshape({grid_resolution1, grid_resolution0}),
            field_x_im.reshape({grid_resolution1, grid_resolution0}),
            field_y_re.reshape({grid_resolution1, grid_resolution0}),
            field_y_im.reshape({grid_resolution1, grid_resolution0}),
            field_z_re.reshape({grid_resolution1, grid_resolution0}),
            field_z_im.reshape({grid_resolution1, grid_resolution0}),
            reflection_count);
    }

    TriangleSoA tri = make_triangle_soa(mesh);
    Vec3SoA ray_o_soa = split_vec3(ray_o);
    Vec3SoA ray_d_soa = split_vec3(ray_d);
    Vec3SoA tx_soa = split_vec3(tx);
    Vec3SoA tx_pol_soa = split_vec3(tx_pol);
    at::Tensor ray_tmax_contig = ray_tmax.contiguous();
    at::Tensor active_contig = active.contiguous();
    at::Tensor material_eta_r = at::ones({tri.n_triangles}, fopts);
    at::Tensor material_sigma = at::zeros({tri.n_triangles}, fopts);
    at::Tensor material_gain = at::ones({tri.n_triangles}, fopts);
    at::Tensor material_mu_r = at::ones({tri.n_triangles}, fopts);
    at::Tensor material_valid = at::ones({tri.n_triangles}, active.options());

    AccumParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = 1;
    params.n_triangles = tri.n_triangles;
    params.ray_ox = ray_o_soa.x.data_ptr<float>();
    params.ray_oy = ray_o_soa.y.data_ptr<float>();
    params.ray_oz = ray_o_soa.z.data_ptr<float>();
    params.ray_dx = ray_d_soa.x.data_ptr<float>();
    params.ray_dy = ray_d_soa.y.data_ptr<float>();
    params.ray_dz = ray_d_soa.z.data_ptr<float>();
    params.ray_tmax = ray_tmax_contig.data_ptr<float>();
    params.active_mask = mask_ptr(active_contig);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.tx_x = tx_soa.x.data_ptr<float>();
    params.tx_y = tx_soa.y.data_ptr<float>();
    params.tx_z = tx_soa.z.data_ptr<float>();
    params.tx_pol_x = tx_pol_soa.x.data_ptr<float>();
    params.tx_pol_y = tx_pol_soa.y.data_ptr<float>();
    params.tx_pol_z = tx_pol_soa.z.data_ptr<float>();
    params.max_bounces = static_cast<int32_t>(max_bounces);
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.solid_angle_per_ray = 1.0f;
    const double span0 = grid_coord0_max - grid_coord0_min;
    const double span1 = grid_coord1_max - grid_coord1_min;
    params.cell_area = static_cast<float>(
        std::abs(span0 * span1) /
        static_cast<double>(grid_resolution0 * grid_resolution1));
    params.seed = 0;
    params.rr_depth = 0;
    params.rr_prob = 1.0f;
    params.stop_threshold = 0.0f;
    params.grid_axis = static_cast<int32_t>(grid_axis);
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = static_cast<int32_t>(grid_resolution0);
    params.grid_resolution1 = static_cast<int32_t>(grid_resolution1);
    params.material_eta_r = material_eta_r.data_ptr<float>();
    params.material_sigma = material_sigma.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_mu_r = material_mu_r.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.material_count = tri.n_triangles;
    params.collect_wedges = 0;
    params.collect_wedge_prefixes = 0;
    params.wedge_capacity = 0;
    params.wedge_sample_stride = 1;
    params.out_reflection_power = power.data_ptr<float>();
    params.out_field_x_re = field_x_re.data_ptr<float>();
    params.out_field_x_im = field_x_im.data_ptr<float>();
    params.out_field_y_re = field_y_re.data_ptr<float>();
    params.out_field_y_im = field_y_im.data_ptr<float>();
    params.out_field_z_re = field_z_re.data_ptr<float>();
    params.out_field_z_im = field_z_im.data_ptr<float>();
    params.out_reflection_count = reflection_count.data_ptr<int>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    optix_pipeline_for_scene(scene, reflection_accumulation_pipeline_config())
        ->launch(0, params, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    return py::make_tuple(
        power.reshape({grid_resolution1, grid_resolution0}),
        field_x_re.reshape({grid_resolution1, grid_resolution0}),
        field_x_im.reshape({grid_resolution1, grid_resolution0}),
        field_y_re.reshape({grid_resolution1, grid_resolution0}),
        field_y_im.reshape({grid_resolution1, grid_resolution0}),
        field_z_re.reshape({grid_resolution1, grid_resolution0}),
        field_z_im.reshape({grid_resolution1, grid_resolution0}),
        reflection_count);
}

py::tuple diffraction_paths_order1_forward_op(
    int64_t scene_handle,
    at::Tensor tx_pos,
    at::Tensor rx_pos,
    at::Tensor active,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_n0,
    at::Tensor state_n1,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t capacity,
    double wavelength) {
    require_vec3f(tx_pos, "tx_pos");
    require_vec3f(rx_pos, "rx_pos");
    require_mask(active, "active");
    require_flat_i32(state_edge_index, "state_edge_index");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_vec3f(state_n0, "state_n0");
    require_vec3f(state_n1, "state_n1");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    if (capacity < 0)
        throw std::runtime_error("capacity must be non-negative.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");

    SceneCache &scene = get_scene(scene_handle);
    if (scene.meshes.size() != 1)
        throw std::runtime_error("diffraction_paths_order1_forward: first milestone supports exactly one mesh.");

    const int64_t tx_count = tx_pos.size(0);
    const int64_t rx_count = rx_pos.size(0);
    const int64_t state_count = state_edge_index.size(0);
    require_state_width(state_edge_pos, state_count, "state_edge_pos");
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_n0, state_count, "state_n0");
    require_state_width(state_n1, state_count, "state_n1");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    const int64_t material_count = material_gain.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_valid.size(0) != material_count)
        throw std::runtime_error("material_gain and material_valid must have matching widths.");

    const int64_t n_rays64 = tx_count * rx_count * state_count;
    if (n_rays64 > capacity)
        throw std::runtime_error("capacity must be at least tx_count * rx_count * state_count.");
    const int32_t n_rays = checked_i32(n_rays64, "n_rays");
    const int32_t capacity_i32 = checked_i32(capacity, "capacity");

    auto fopts = tx_pos.options();
    auto iopts = state_edge_index.options();
    at::Tensor out_count = at::zeros({1}, iopts);
    at::Tensor out_valid = at::zeros({capacity}, active.options());
    at::Tensor out_tx_id = at::full({capacity}, -1, iopts);
    at::Tensor out_rx_id = at::full({capacity}, -1, iopts);
    at::Tensor out_order = at::zeros({capacity}, iopts);
    at::Tensor out_edge0 = at::full({capacity}, -1, iopts);
    at::Tensor out_edge1 = at::full({capacity}, -1, iopts);
    at::Tensor out_edge2 = at::full({capacity}, -1, iopts);
    at::Tensor out_delay = at::zeros({capacity}, fopts);
    at::Tensor out_field_x_re = at::zeros({capacity}, fopts);
    at::Tensor out_field_x_im = at::zeros({capacity}, fopts);
    at::Tensor out_field_y_re = at::zeros({capacity}, fopts);
    at::Tensor out_field_y_im = at::zeros({capacity}, fopts);
    at::Tensor out_field_z_re = at::zeros({capacity}, fopts);
    at::Tensor out_field_z_im = at::zeros({capacity}, fopts);
    at::Tensor out_p0_x = at::zeros({capacity}, fopts);
    at::Tensor out_p0_y = at::zeros({capacity}, fopts);
    at::Tensor out_p0_z = at::zeros({capacity}, fopts);
    at::Tensor out_p1_x = at::zeros({capacity}, fopts);
    at::Tensor out_p1_y = at::zeros({capacity}, fopts);
    at::Tensor out_p1_z = at::zeros({capacity}, fopts);
    at::Tensor out_p2_x = at::zeros({capacity}, fopts);
    at::Tensor out_p2_y = at::zeros({capacity}, fopts);
    at::Tensor out_p2_z = at::zeros({capacity}, fopts);
    if (n_rays == 0 || capacity_i32 == 0) {
        return py::make_tuple(
            out_count,
            out_valid,
            out_tx_id,
            out_rx_id,
            out_order,
            out_edge0,
            out_edge1,
            out_edge2,
            out_delay,
            out_field_x_re,
            out_field_x_im,
            out_field_y_re,
            out_field_y_im,
            out_field_z_re,
            out_field_z_im,
            at::stack({out_p0_x, out_p0_y, out_p0_z}, 1).contiguous(),
            at::stack({out_p1_x, out_p1_y, out_p1_z}, 1).contiguous(),
            at::stack({out_p2_x, out_p2_y, out_p2_z}, 1).contiguous());
    }

    Vec3SoA tx_soa = split_vec3(tx_pos);
    Vec3SoA rx_soa = split_vec3(rx_pos);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_n0_soa = split_vec3(state_n0);
    Vec3SoA state_n1_soa = split_vec3(state_n1);
    Vec3SoA state_src_soa = split_vec3(state_src);
    at::Tensor active_contig = active_mask_for_states(active, state_count, "diffraction_paths_order1_forward");
    at::Tensor temp_visibility = at::zeros({n_rays}, active.options());

    DfrPathParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = n_rays;
    params.capacity = capacity_i32;
    params.tx_pos_x = tx_soa.x.data_ptr<float>();
    params.tx_pos_y = tx_soa.y.data_ptr<float>();
    params.tx_pos_z = tx_soa.z.data_ptr<float>();
    params.tx_count = checked_i32(tx_count, "tx_count");
    params.rx_pos_x = rx_soa.x.data_ptr<float>();
    params.rx_pos_y = rx_soa.y.data_ptr<float>();
    params.rx_pos_z = rx_soa.z.data_ptr<float>();
    params.rx_count = checked_i32(rx_count, "rx_count");
    params.active_mask = mask_ptr(active_contig);
    params.active_width = checked_i32(state_count, "active_width");
    params.state_count = checked_i32(state_count, "state_count");
    params.state_limit = checked_i32(state_count, "state_limit");
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_n0_x = state_n0_soa.x.data_ptr<float>();
    params.state_n0_y = state_n0_soa.y.data_ptr<float>();
    params.state_n0_z = state_n0_soa.z.data_ptr<float>();
    params.state_n1_x = state_n1_soa.x.data_ptr<float>();
    params.state_n1_y = state_n1_soa.y.data_ptr<float>();
    params.state_n1_z = state_n1_soa.z.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.seed = 0;
    params.max_order = 1;
    params.strategy_mask = RAYDTORCH_DFR_DIRECT;
    params.sample_count = 1;
    params.return_geom = 1;
    params.receiver_model = RAYDTORCH_DFR_MATCHED_ISO;
    params.temp_visibility = mutable_mask_ptr(temp_visibility);
    params.out_count = out_count.data_ptr<int>();
    params.out_valid = mutable_mask_ptr(out_valid);
    params.out_tx_id = out_tx_id.data_ptr<int>();
    params.out_rx_id = out_rx_id.data_ptr<int>();
    params.out_order = out_order.data_ptr<int>();
    params.out_edge0 = out_edge0.data_ptr<int>();
    params.out_edge1 = out_edge1.data_ptr<int>();
    params.out_edge2 = out_edge2.data_ptr<int>();
    params.out_delay = out_delay.data_ptr<float>();
    params.out_field_x_re = out_field_x_re.data_ptr<float>();
    params.out_field_x_im = out_field_x_im.data_ptr<float>();
    params.out_field_y_re = out_field_y_re.data_ptr<float>();
    params.out_field_y_im = out_field_y_im.data_ptr<float>();
    params.out_field_z_re = out_field_z_re.data_ptr<float>();
    params.out_field_z_im = out_field_z_im.data_ptr<float>();
    params.out_p0_x = out_p0_x.data_ptr<float>();
    params.out_p0_y = out_p0_y.data_ptr<float>();
    params.out_p0_z = out_p0_z.data_ptr<float>();
    params.out_p1_x = out_p1_x.data_ptr<float>();
    params.out_p1_y = out_p1_y.data_ptr<float>();
    params.out_p1_z = out_p1_z.data_ptr<float>();
    params.out_p2_x = out_p2_x.data_ptr<float>();
    params.out_p2_y = out_p2_y.data_ptr<float>();
    params.out_p2_z = out_p2_z.data_ptr<float>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    auto pipeline = optix_pipeline_for_scene(scene, diffraction_paths_pipeline_config());
    pipeline->launch(2, params, static_cast<unsigned int>(n_rays), torch_ctx.stream);
    pipeline->launch(3, params, static_cast<unsigned int>(n_rays), torch_ctx.stream);

    return py::make_tuple(
        out_count,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_order,
        out_edge0,
        out_edge1,
        out_edge2,
        out_delay,
        out_field_x_re,
        out_field_x_im,
        out_field_y_re,
        out_field_y_im,
        out_field_z_re,
        out_field_z_im,
        at::stack({out_p0_x, out_p0_y, out_p0_z}, 1).contiguous(),
        at::stack({out_p1_x, out_p1_y, out_p1_z}, 1).contiguous(),
        at::stack({out_p2_x, out_p2_y, out_p2_z}, 1).contiguous());
}

py::tuple diffraction_accumulation_forward_op(
    int64_t scene_handle,
    at::Tensor active,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_n0,
    at::Tensor state_n1,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor state_wi,
    at::Tensor state_d0,
    at::Tensor material_eta_r,
    at::Tensor material_sigma,
    at::Tensor material_mu_r,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    int64_t max_order,
    at::Tensor recursive_active,
    at::Tensor recursive_state_edge_index,
    at::Tensor recursive_state_edge_pos,
    at::Tensor recursive_state_edge_dir,
    at::Tensor recursive_state_edge_t_min,
    at::Tensor recursive_state_edge_t_max,
    at::Tensor recursive_state_n0,
    at::Tensor recursive_state_n1,
    at::Tensor recursive_state_prim0,
    at::Tensor recursive_state_prim1,
    at::Tensor recursive_state_exterior_angle) {
    require_mask(active, "active");
    require_flat_i32(state_edge_index, "state_edge_index");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_vec3f(state_n0, "state_n0");
    require_vec3f(state_n1, "state_n1");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_vec3f(state_wi, "state_wi");
    require_vec3f(state_d0, "state_d0");
    require_flat_f32(material_eta_r, "material_eta_r");
    require_flat_f32(material_sigma, "material_sigma");
    require_flat_f32(material_mu_r, "material_mu_r");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    if (grid_axis < 0 || grid_axis > 2)
        throw std::runtime_error("grid_axis must be 0, 1, or 2.");
    if (!(grid_coord0_min < grid_coord0_max) || !(grid_coord1_min < grid_coord1_max))
        throw std::runtime_error("grid bounds must be ordered.");
    if (grid_resolution0 <= 0 || grid_resolution1 <= 0)
        throw std::runtime_error("grid resolutions must be positive.");
    if (!(grid_cell_area > 0.0))
        throw std::runtime_error("grid_cell_area must be positive.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");
    if (direct_samples < 0)
        throw std::runtime_error("direct_samples must be non-negative.");
    if (keller_samples < 0)
        throw std::runtime_error("keller_samples must be non-negative.");
    if (suffix_samples < 0)
        throw std::runtime_error("suffix_samples must be non-negative.");
    if (max_order < 1 || max_order > 3)
        throw std::runtime_error("max_order must be 1, 2, or 3.");

    SceneCache &scene = get_scene(scene_handle);
    const int64_t state_count = state_edge_index.size(0);
    require_state_width(state_edge_pos, state_count, "state_edge_pos");
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_n0, state_count, "state_n0");
    require_state_width(state_n1, state_count, "state_n1");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    require_state_width(state_wi, state_count, "state_wi");
    require_state_width(state_d0, state_count, "state_d0");
    const bool use_recursive = max_order > 1;
    int64_t recursive_state_count = 0;
    if (use_recursive) {
        require_mask(recursive_active, "recursive_active");
        require_flat_i32(recursive_state_edge_index, "recursive_state_edge_index");
        require_vec3f(recursive_state_edge_pos, "recursive_state_edge_pos");
        require_vec3f(recursive_state_edge_dir, "recursive_state_edge_dir");
        require_scalar_f(recursive_state_edge_t_min, "recursive_state_edge_t_min");
        require_scalar_f(recursive_state_edge_t_max, "recursive_state_edge_t_max");
        require_vec3f(recursive_state_n0, "recursive_state_n0");
        require_vec3f(recursive_state_n1, "recursive_state_n1");
        require_flat_i32(recursive_state_prim0, "recursive_state_prim0");
        require_flat_i32(recursive_state_prim1, "recursive_state_prim1");
        require_scalar_f(recursive_state_exterior_angle, "recursive_state_exterior_angle");
        recursive_state_count = recursive_state_edge_index.size(0);
        require_state_width(recursive_active, recursive_state_count, "recursive_active");
        require_state_width(recursive_state_edge_pos, recursive_state_count, "recursive_state_edge_pos");
        require_state_width(recursive_state_edge_dir, recursive_state_count, "recursive_state_edge_dir");
        require_state_width(recursive_state_edge_t_min, recursive_state_count, "recursive_state_edge_t_min");
        require_state_width(recursive_state_edge_t_max, recursive_state_count, "recursive_state_edge_t_max");
        require_state_width(recursive_state_n0, recursive_state_count, "recursive_state_n0");
        require_state_width(recursive_state_n1, recursive_state_count, "recursive_state_n1");
        require_state_width(recursive_state_prim0, recursive_state_count, "recursive_state_prim0");
        require_state_width(recursive_state_prim1, recursive_state_count, "recursive_state_prim1");
        require_state_width(recursive_state_exterior_angle, recursive_state_count, "recursive_state_exterior_angle");
    }
    const int64_t material_count = material_eta_r.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_sigma.size(0) != material_count ||
        material_mu_r.size(0) != material_count ||
        material_gain.size(0) != material_count ||
        material_valid.size(0) != material_count) {
        throw std::runtime_error("material payload fields must have matching widths.");
    }

    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    const int32_t direct_launch_count = checked_i32(direct_samples, "direct_samples");
    const int32_t keller_launch_count = checked_i32(keller_samples, "keller_samples");
    const int32_t suffix_launch_count = checked_i32(suffix_samples, "suffix_samples");
    const int32_t launch_count = checked_i32(direct_samples + keller_samples + suffix_samples, "launch_count");
    auto fopts = state_src.options();
    auto iopts = state_edge_index.options();
    at::Tensor power = at::zeros({cell_count}, fopts);
    at::Tensor field_x_re = at::zeros({cell_count}, fopts);
    at::Tensor field_x_im = at::zeros({cell_count}, fopts);
    at::Tensor field_y_re = at::zeros({cell_count}, fopts);
    at::Tensor field_y_im = at::zeros({cell_count}, fopts);
    at::Tensor field_z_re = at::zeros({cell_count}, fopts);
    at::Tensor field_z_im = at::zeros({cell_count}, fopts);
    at::Tensor direct_count = at::zeros({1}, iopts);
    at::Tensor keller_count = at::zeros({1}, iopts);
    at::Tensor suffix_count = at::zeros({1}, iopts);
    at::Tensor vis_rejects = at::zeros({1}, iopts);
    at::Tensor edge_vis_rejects = at::zeros({1}, iopts);
    at::Tensor utd_rejects = at::zeros({1}, iopts);
    at::Tensor edge_uses = at::zeros({1}, iopts);
    at::Tensor tape_active = at::zeros({launch_count}, active.options());
    at::Tensor tape_state_idx = at::full({launch_count}, -1, iopts);
    at::Tensor tape_cell = at::full({launch_count}, -1, iopts);
    at::Tensor tape_material_idx = at::full({launch_count}, -1, iopts);
    at::Tensor tape_edge_u = at::zeros({launch_count}, fopts);
    if (state_count == 0 || launch_count == 0) {
        return py::make_tuple(
            power.reshape({grid_resolution1, grid_resolution0}),
            field_x_re.reshape({grid_resolution1, grid_resolution0}),
            field_x_im.reshape({grid_resolution1, grid_resolution0}),
            field_y_re.reshape({grid_resolution1, grid_resolution0}),
            field_y_im.reshape({grid_resolution1, grid_resolution0}),
            field_z_re.reshape({grid_resolution1, grid_resolution0}),
            field_z_im.reshape({grid_resolution1, grid_resolution0}),
            direct_count,
            keller_count,
            suffix_count,
            vis_rejects,
            edge_vis_rejects,
            utd_rejects,
            edge_uses,
            tape_active,
            tape_state_idx,
            tape_cell,
            tape_material_idx,
            tape_edge_u);
    }

    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_n0_soa = split_vec3(state_n0);
    Vec3SoA state_n1_soa = split_vec3(state_n1);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA state_wi_soa = split_vec3(state_wi);
    Vec3SoA state_d0_soa = split_vec3(state_d0);
    at::Tensor active_contig = active_mask_for_states(active, state_count, "diffraction_accumulation_forward");
    at::Tensor state_prefix_depth = at::zeros({state_count}, iopts);
    at::Tensor temp_visibility = at::zeros({launch_count}, active.options());
    at::Tensor recursive_active_contig;
    at::Tensor recursive_prefix_depth;
    Vec3SoA recursive_edge_pos_soa;
    Vec3SoA recursive_edge_dir_soa;
    Vec3SoA recursive_n0_soa;
    Vec3SoA recursive_n1_soa;
    if (use_recursive) {
        recursive_active_contig = active_mask_for_states(
            recursive_active,
            recursive_state_count,
            "diffraction_accumulation_forward recursive_active");
        recursive_prefix_depth = at::zeros({recursive_state_count}, iopts);
        recursive_edge_pos_soa = split_vec3(recursive_state_edge_pos);
        recursive_edge_dir_soa = split_vec3(recursive_state_edge_dir);
        recursive_n0_soa = split_vec3(recursive_state_n0);
        recursive_n1_soa = split_vec3(recursive_state_n1);
    }

    DfrAccumParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = launch_count;
    params.active_mask = mask_ptr(active_contig);
    params.state_count = checked_i32(state_count, "state_count");
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_n0_x = state_n0_soa.x.data_ptr<float>();
    params.state_n0_y = state_n0_soa.y.data_ptr<float>();
    params.state_n0_z = state_n0_soa.z.data_ptr<float>();
    params.state_n1_x = state_n1_soa.x.data_ptr<float>();
    params.state_n1_y = state_n1_soa.y.data_ptr<float>();
    params.state_n1_z = state_n1_soa.z.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_wi_x = state_wi_soa.x.data_ptr<float>();
    params.state_wi_y = state_wi_soa.y.data_ptr<float>();
    params.state_wi_z = state_wi_soa.z.data_ptr<float>();
    params.state_d0_x = state_d0_soa.x.data_ptr<float>();
    params.state_d0_y = state_d0_soa.y.data_ptr<float>();
    params.state_d0_z = state_d0_soa.z.data_ptr<float>();
    params.state_prefix_depth = state_prefix_depth.data_ptr<int>();
    params.recursive_state_count = checked_i32(recursive_state_count, "recursive_state_count");
    if (use_recursive) {
        params.recursive_active_mask = mask_ptr(recursive_active_contig);
        params.recursive_state_edge_index = recursive_state_edge_index.data_ptr<int>();
        params.recursive_state_edge_pos_x = recursive_edge_pos_soa.x.data_ptr<float>();
        params.recursive_state_edge_pos_y = recursive_edge_pos_soa.y.data_ptr<float>();
        params.recursive_state_edge_pos_z = recursive_edge_pos_soa.z.data_ptr<float>();
        params.recursive_state_edge_dir_x = recursive_edge_dir_soa.x.data_ptr<float>();
        params.recursive_state_edge_dir_y = recursive_edge_dir_soa.y.data_ptr<float>();
        params.recursive_state_edge_dir_z = recursive_edge_dir_soa.z.data_ptr<float>();
        params.recursive_state_edge_t_min = recursive_state_edge_t_min.data_ptr<float>();
        params.recursive_state_edge_t_max = recursive_state_edge_t_max.data_ptr<float>();
        params.recursive_state_n0_x = recursive_n0_soa.x.data_ptr<float>();
        params.recursive_state_n0_y = recursive_n0_soa.y.data_ptr<float>();
        params.recursive_state_n0_z = recursive_n0_soa.z.data_ptr<float>();
        params.recursive_state_n1_x = recursive_n1_soa.x.data_ptr<float>();
        params.recursive_state_n1_y = recursive_n1_soa.y.data_ptr<float>();
        params.recursive_state_n1_z = recursive_n1_soa.z.data_ptr<float>();
        params.recursive_state_prim0 = recursive_state_prim0.data_ptr<int>();
        params.recursive_state_prim1 = recursive_state_prim1.data_ptr<int>();
        params.recursive_state_exterior_angle = recursive_state_exterior_angle.data_ptr<float>();
    }
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.n_triangles = tri.n_triangles;
    params.suffix_candidate_prim_id = nullptr;
    params.suffix_candidate_count = 0;
    params.material_eta_r = material_eta_r.data_ptr<float>();
    params.material_sigma = material_sigma.data_ptr<float>();
    params.material_mu_r = material_mu_r.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.seed = checked_i32(seed, "seed");
    params.samples = launch_count;
    params.max_order = checked_i32(max_order, "max_order");
    params.direct_samples = direct_launch_count;
    params.keller_samples = keller_launch_count;
    params.suffix_samples = suffix_launch_count;
    params.strategy_mask =
        (direct_launch_count > 0 ? RAYDTORCH_DFR_DIRECT : 0) |
        (keller_launch_count > 0 ? RAYDTORCH_DFR_KELLER : 0) |
        (suffix_launch_count > 0 ? RAYDTORCH_DFR_SUFFIX_REFL : 0);
    params.sample_sequence = RAYDTORCH_DFR_HASH;
    params.receiver_model = RAYDTORCH_DFR_MATCHED_ISO;
    params.select_diffraction_point = 0;
    params.prefilter_visibility = 0;
    params.collect_edge_use = 1;
    params.collect_debug_counts = 1;
    params.omega = 2.0f * 3.14159265358979323846f * 299792458.0f;
    params.tx_pol_x = 1.0f;
    params.tx_pol_y = 0.0f;
    params.tx_pol_z = 0.0f;
    params.out_power = power.data_ptr<float>();
    params.out_field_x_re = field_x_re.data_ptr<float>();
    params.out_field_x_im = field_x_im.data_ptr<float>();
    params.out_field_y_re = field_y_re.data_ptr<float>();
    params.out_field_y_im = field_y_im.data_ptr<float>();
    params.out_field_z_re = field_z_re.data_ptr<float>();
    params.out_field_z_im = field_z_im.data_ptr<float>();
    params.out_direct_count = direct_count.data_ptr<int>();
    params.out_keller_count = keller_count.data_ptr<int>();
    params.out_suffix_count = suffix_count.data_ptr<int>();
    params.out_vis_rejects = vis_rejects.data_ptr<int>();
    params.out_edge_vis_rejects = edge_vis_rejects.data_ptr<int>();
    params.out_utd_rejects = utd_rejects.data_ptr<int>();
    params.out_edge_uses = edge_uses.data_ptr<int>();
    params.temp_visibility = mutable_mask_ptr(temp_visibility);
    params.tape_active = mutable_mask_ptr(tape_active);
    params.tape_state_idx = tape_state_idx.data_ptr<int>();
    params.tape_cell = tape_cell.data_ptr<int>();
    params.tape_material_idx = tape_material_idx.data_ptr<int>();
    params.tape_edge_u = tape_edge_u.data_ptr<float>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    auto pipeline = optix_pipeline_for_scene(scene, diffraction_accumulation_pipeline_config());
    if (use_recursive) {
        pipeline->launch(13, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
    } else {
        pipeline->launch(6, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
        if (direct_launch_count + keller_launch_count > 0)
            pipeline->launch(7, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
        if (suffix_launch_count > 0) {
            pipeline->launch(8, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
            pipeline->launch(9, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
        }
    }

    return py::make_tuple(
        power.reshape({grid_resolution1, grid_resolution0}),
        field_x_re.reshape({grid_resolution1, grid_resolution0}),
        field_x_im.reshape({grid_resolution1, grid_resolution0}),
        field_y_re.reshape({grid_resolution1, grid_resolution0}),
        field_y_im.reshape({grid_resolution1, grid_resolution0}),
        field_z_re.reshape({grid_resolution1, grid_resolution0}),
        field_z_im.reshape({grid_resolution1, grid_resolution0}),
        direct_count,
        keller_count,
        suffix_count,
        vis_rejects,
        edge_vis_rejects,
        utd_rejects,
        edge_uses,
        tape_active,
        tape_state_idx,
        tape_cell,
        tape_material_idx,
        tape_edge_u);
}

py::tuple diffraction_accumulation_direct_backward_op(
    int64_t scene_handle,
    at::Tensor tape_active,
    at::Tensor tape_state_idx,
    at::Tensor tape_cell,
    at::Tensor tape_material_idx,
    at::Tensor tape_edge_u,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor state_wi,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    at::Tensor grad_power,
    at::Tensor grad_field_x_re) {
    require_mask(tape_active, "tape_active");
    require_flat_i32(tape_state_idx, "tape_state_idx");
    require_flat_i32(tape_cell, "tape_cell");
    require_flat_i32(tape_material_idx, "tape_material_idx");
    require_flat_f32(tape_edge_u, "tape_edge_u");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_vec3f(state_wi, "state_wi");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    const int64_t launch_count = tape_active.size(0);
    require_state_width(tape_state_idx, launch_count, "tape_state_idx");
    require_state_width(tape_cell, launch_count, "tape_cell");
    require_state_width(tape_material_idx, launch_count, "tape_material_idx");
    require_state_width(tape_edge_u, launch_count, "tape_edge_u");
    const int64_t state_count = state_edge_pos.size(0);
    const int64_t material_count = material_gain.size(0);
    if (material_valid.size(0) != material_count)
        throw std::runtime_error("material_valid must match material_gain width.");

    SceneCache &scene = get_scene(scene_handle);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA state_wi_soa = split_vec3(state_wi);
    at::Tensor grad_power_flat = grad_power.reshape({-1}).contiguous();
    at::Tensor grad_field_x_re_flat = grad_field_x_re.reshape({-1}).contiguous();
    require_flat_f32(grad_power_flat, "grad_power");
    require_flat_f32(grad_field_x_re_flat, "grad_field_x_re");

    at::Tensor grad_edge_pos_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_pos_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_pos_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_t_min = at::zeros_like(state_edge_t_min);
    at::Tensor grad_edge_t_max = at::zeros_like(state_edge_t_max);
    at::Tensor grad_src_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_wi_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_wi_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_wi_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_power = at::zeros_like(state_src_power);
    at::Tensor grad_exterior_angle = at::zeros_like(state_exterior_angle);
    at::Tensor grad_material_gain = at::zeros_like(material_gain);
    at::Tensor grad_tri_p0_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_z = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_z = at::zeros({tri.n_triangles}, state_edge_pos.options());

    DfrDirectAccumADParams params = {};
    params.n_rays = checked_i32(launch_count, "n_rays");
    params.state_count = checked_i32(state_count, "state_count");
    params.material_count = checked_i32(material_count, "material_count");
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.direct_samples = checked_i32(direct_samples, "direct_samples");
    params.keller_samples = checked_i32(keller_samples, "keller_samples");
    params.suffix_samples = checked_i32(suffix_samples, "suffix_samples");
    params.wavelength = static_cast<float>(wavelength);
    params.seed = checked_i32(seed, "seed");
    params.n_triangles = tri.n_triangles;
    params.tape_active = mask_ptr(tape_active);
    params.tape_state_idx = tape_state_idx.data_ptr<int>();
    params.tape_cell = tape_cell.data_ptr<int>();
    params.tape_material_idx = tape_material_idx.data_ptr<int>();
    params.tape_edge_u = tape_edge_u.data_ptr<float>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_wi_x = state_wi_soa.x.data_ptr<float>();
    params.state_wi_y = state_wi_soa.y.data_ptr<float>();
    params.state_wi_z = state_wi_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.grad_out_power = grad_power_flat.data_ptr<float>();
    params.grad_out_field_x_re = grad_field_x_re_flat.data_ptr<float>();
    params.grad_state_edge_pos_x = grad_edge_pos_x.data_ptr<float>();
    params.grad_state_edge_pos_y = grad_edge_pos_y.data_ptr<float>();
    params.grad_state_edge_pos_z = grad_edge_pos_z.data_ptr<float>();
    params.grad_state_edge_dir_x = grad_edge_dir_x.data_ptr<float>();
    params.grad_state_edge_dir_y = grad_edge_dir_y.data_ptr<float>();
    params.grad_state_edge_dir_z = grad_edge_dir_z.data_ptr<float>();
    params.grad_state_edge_t_min = grad_edge_t_min.data_ptr<float>();
    params.grad_state_edge_t_max = grad_edge_t_max.data_ptr<float>();
    params.grad_state_src_x = grad_src_x.data_ptr<float>();
    params.grad_state_src_y = grad_src_y.data_ptr<float>();
    params.grad_state_src_z = grad_src_z.data_ptr<float>();
    params.grad_state_wi_x = grad_wi_x.data_ptr<float>();
    params.grad_state_wi_y = grad_wi_y.data_ptr<float>();
    params.grad_state_wi_z = grad_wi_z.data_ptr<float>();
    params.grad_state_src_power = grad_src_power.data_ptr<float>();
    params.grad_state_exterior_angle = grad_exterior_angle.data_ptr<float>();
    params.grad_material_gain = grad_material_gain.data_ptr<float>();
    params.grad_tri_p0_x = grad_tri_p0_x.data_ptr<float>();
    params.grad_tri_p0_y = grad_tri_p0_y.data_ptr<float>();
    params.grad_tri_p0_z = grad_tri_p0_z.data_ptr<float>();
    params.grad_tri_fn_x = grad_tri_fn_x.data_ptr<float>();
    params.grad_tri_fn_y = grad_tri_fn_y.data_ptr<float>();
    params.grad_tri_fn_z = grad_tri_fn_z.data_ptr<float>();
    dfr_direct_accum_vjp_gpu(params);
    return py::make_tuple(
        stack_vec3(grad_edge_pos_x, grad_edge_pos_y, grad_edge_pos_z),
        stack_vec3(grad_edge_dir_x, grad_edge_dir_y, grad_edge_dir_z),
        grad_edge_t_min,
        grad_edge_t_max,
        stack_vec3(grad_src_x, grad_src_y, grad_src_z),
        stack_vec3(grad_wi_x, grad_wi_y, grad_wi_z),
        grad_src_power,
        grad_exterior_angle,
        grad_material_gain);
}

py::tuple diffraction_accumulation_direct_jvp_op(
    int64_t scene_handle,
    at::Tensor tape_active,
    at::Tensor tape_state_idx,
    at::Tensor tape_cell,
    at::Tensor tape_material_idx,
    at::Tensor tape_edge_u,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor state_wi,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    at::Tensor dot_state_edge_pos,
    at::Tensor dot_state_edge_dir,
    at::Tensor dot_state_edge_t_min,
    at::Tensor dot_state_edge_t_max,
    at::Tensor dot_state_exterior_angle,
    at::Tensor dot_state_src,
    at::Tensor dot_state_src_power,
    at::Tensor dot_state_wi,
    at::Tensor dot_material_gain) {
    SceneCache &scene = get_scene(scene_handle);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA state_wi_soa = split_vec3(state_wi);
    Vec3SoA dot_edge_pos_soa = split_vec3(dot_state_edge_pos);
    Vec3SoA dot_edge_dir_soa = split_vec3(dot_state_edge_dir);
    Vec3SoA dot_src_soa = split_vec3(dot_state_src);
    Vec3SoA dot_wi_soa = split_vec3(dot_state_wi);
    const int64_t launch_count = tape_active.size(0);
    const int64_t state_count = state_edge_pos.size(0);
    const int64_t material_count = material_gain.size(0);
    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    at::Tensor dot_power = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor dot_field_x_re = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor zero_tri = at::zeros({tri.n_triangles}, state_edge_pos.options());

    DfrDirectAccumADParams params = {};
    params.n_rays = checked_i32(launch_count, "n_rays");
    params.state_count = checked_i32(state_count, "state_count");
    params.material_count = checked_i32(material_count, "material_count");
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.direct_samples = checked_i32(direct_samples, "direct_samples");
    params.keller_samples = checked_i32(keller_samples, "keller_samples");
    params.suffix_samples = checked_i32(suffix_samples, "suffix_samples");
    params.wavelength = static_cast<float>(wavelength);
    params.seed = checked_i32(seed, "seed");
    params.n_triangles = tri.n_triangles;
    params.tape_active = mask_ptr(tape_active);
    params.tape_state_idx = tape_state_idx.data_ptr<int>();
    params.tape_cell = tape_cell.data_ptr<int>();
    params.tape_material_idx = tape_material_idx.data_ptr<int>();
    params.tape_edge_u = tape_edge_u.data_ptr<float>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_wi_x = state_wi_soa.x.data_ptr<float>();
    params.state_wi_y = state_wi_soa.y.data_ptr<float>();
    params.state_wi_z = state_wi_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.dot_state_edge_pos_x = dot_edge_pos_soa.x.data_ptr<float>();
    params.dot_state_edge_pos_y = dot_edge_pos_soa.y.data_ptr<float>();
    params.dot_state_edge_pos_z = dot_edge_pos_soa.z.data_ptr<float>();
    params.dot_state_edge_dir_x = dot_edge_dir_soa.x.data_ptr<float>();
    params.dot_state_edge_dir_y = dot_edge_dir_soa.y.data_ptr<float>();
    params.dot_state_edge_dir_z = dot_edge_dir_soa.z.data_ptr<float>();
    params.dot_state_edge_t_min = dot_state_edge_t_min.data_ptr<float>();
    params.dot_state_edge_t_max = dot_state_edge_t_max.data_ptr<float>();
    params.dot_state_src_x = dot_src_soa.x.data_ptr<float>();
    params.dot_state_src_y = dot_src_soa.y.data_ptr<float>();
    params.dot_state_src_z = dot_src_soa.z.data_ptr<float>();
    params.dot_state_wi_x = dot_wi_soa.x.data_ptr<float>();
    params.dot_state_wi_y = dot_wi_soa.y.data_ptr<float>();
    params.dot_state_wi_z = dot_wi_soa.z.data_ptr<float>();
    params.dot_state_src_power = dot_state_src_power.data_ptr<float>();
    params.dot_state_exterior_angle = dot_state_exterior_angle.data_ptr<float>();
    params.dot_material_gain = dot_material_gain.data_ptr<float>();
    params.dot_tri_p0_x = zero_tri.data_ptr<float>();
    params.dot_tri_p0_y = zero_tri.data_ptr<float>();
    params.dot_tri_p0_z = zero_tri.data_ptr<float>();
    params.dot_tri_fn_x = zero_tri.data_ptr<float>();
    params.dot_tri_fn_y = zero_tri.data_ptr<float>();
    params.dot_tri_fn_z = zero_tri.data_ptr<float>();
    params.dot_out_power = dot_power.data_ptr<float>();
    params.dot_out_field_x_re = dot_field_x_re.data_ptr<float>();
    dfr_direct_accum_jvp_gpu(params);
    return py::make_tuple(
        dot_power.reshape({grid_resolution1, grid_resolution0}),
        dot_field_x_re.reshape({grid_resolution1, grid_resolution0}));
}

py::tuple diffraction_coherent_accumulation_forward_op(
    int64_t scene_handle,
    at::Tensor active,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_n0,
    at::Tensor state_n1,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor state_wi,
    at::Tensor state_d0,
    at::Tensor material_eta_r,
    at::Tensor material_sigma,
    at::Tensor material_mu_r,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    bool select_diffraction_point,
    bool prefilter_visibility) {
    require_mask(active, "active");
    require_flat_i32(state_edge_index, "state_edge_index");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_vec3f(state_n0, "state_n0");
    require_vec3f(state_n1, "state_n1");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_vec3f(state_wi, "state_wi");
    require_vec3f(state_d0, "state_d0");
    require_flat_f32(material_eta_r, "material_eta_r");
    require_flat_f32(material_sigma, "material_sigma");
    require_flat_f32(material_mu_r, "material_mu_r");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    if (grid_axis < 0 || grid_axis > 2)
        throw std::runtime_error("grid_axis must be 0, 1, or 2.");
    if (!(grid_coord0_min < grid_coord0_max) || !(grid_coord1_min < grid_coord1_max))
        throw std::runtime_error("grid bounds must be ordered.");
    if (grid_resolution0 <= 0 || grid_resolution1 <= 0)
        throw std::runtime_error("grid resolutions must be positive.");
    if (!(grid_cell_area > 0.0))
        throw std::runtime_error("grid_cell_area must be positive.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");

    SceneCache &scene = get_scene(scene_handle);
    const int64_t state_count = state_edge_index.size(0);
    require_state_width(state_edge_pos, state_count, "state_edge_pos");
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_n0, state_count, "state_n0");
    require_state_width(state_n1, state_count, "state_n1");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    require_state_width(state_wi, state_count, "state_wi");
    require_state_width(state_d0, state_count, "state_d0");
    const int64_t material_count = material_eta_r.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_sigma.size(0) != material_count ||
        material_mu_r.size(0) != material_count ||
        material_gain.size(0) != material_count ||
        material_valid.size(0) != material_count) {
        throw std::runtime_error("material payload fields must have matching widths.");
    }

    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    const int64_t launch_count64 = state_count * cell_count;
    const int32_t launch_count = checked_i32(launch_count64, "launch_count");
    auto fopts = state_src.options();
    auto iopts = state_edge_index.options();
    at::Tensor direct_x_re = at::zeros({cell_count}, fopts);
    at::Tensor direct_x_im = at::zeros({cell_count}, fopts);
    at::Tensor direct_y_re = at::zeros({cell_count}, fopts);
    at::Tensor direct_y_im = at::zeros({cell_count}, fopts);
    at::Tensor direct_z_re = at::zeros({cell_count}, fopts);
    at::Tensor direct_z_im = at::zeros({cell_count}, fopts);
    at::Tensor multi_x_re = at::zeros({cell_count}, fopts);
    at::Tensor multi_x_im = at::zeros({cell_count}, fopts);
    at::Tensor multi_y_re = at::zeros({cell_count}, fopts);
    at::Tensor multi_y_im = at::zeros({cell_count}, fopts);
    at::Tensor multi_z_re = at::zeros({cell_count}, fopts);
    at::Tensor multi_z_im = at::zeros({cell_count}, fopts);
    at::Tensor direct_count = at::zeros({cell_count}, iopts);
    at::Tensor multi_count = at::zeros({cell_count}, iopts);
    at::Tensor visibility_reject_count = at::zeros({cell_count}, iopts);
    at::Tensor utd_reject_count = at::zeros({cell_count}, iopts);
    if (state_count == 0 || launch_count == 0) {
        return py::make_tuple(
            direct_x_re, direct_x_im, direct_y_re, direct_y_im, direct_z_re, direct_z_im,
            multi_x_re, multi_x_im, multi_y_re, multi_y_im, multi_z_re, multi_z_im,
            direct_count, multi_count, visibility_reject_count, utd_reject_count);
    }

    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_n0_soa = split_vec3(state_n0);
    Vec3SoA state_n1_soa = split_vec3(state_n1);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA state_wi_soa = split_vec3(state_wi);
    Vec3SoA state_d0_soa = split_vec3(state_d0);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    at::Tensor active_contig = active_mask_for_states(active, state_count, "diffraction_coherent_accumulation_forward");
    at::Tensor state_prefix_depth = at::zeros({state_count}, iopts);

    DfrAccumParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = launch_count;
    params.active_mask = mask_ptr(active_contig);
    params.state_count = checked_i32(state_count, "state_count");
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_n0_x = state_n0_soa.x.data_ptr<float>();
    params.state_n0_y = state_n0_soa.y.data_ptr<float>();
    params.state_n0_z = state_n0_soa.z.data_ptr<float>();
    params.state_n1_x = state_n1_soa.x.data_ptr<float>();
    params.state_n1_y = state_n1_soa.y.data_ptr<float>();
    params.state_n1_z = state_n1_soa.z.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_wi_x = state_wi_soa.x.data_ptr<float>();
    params.state_wi_y = state_wi_soa.y.data_ptr<float>();
    params.state_wi_z = state_wi_soa.z.data_ptr<float>();
    params.state_d0_x = state_d0_soa.x.data_ptr<float>();
    params.state_d0_y = state_d0_soa.y.data_ptr<float>();
    params.state_d0_z = state_d0_soa.z.data_ptr<float>();
    params.state_prefix_depth = state_prefix_depth.data_ptr<int>();
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.n_triangles = tri.n_triangles;
    params.material_eta_r = material_eta_r.data_ptr<float>();
    params.material_sigma = material_sigma.data_ptr<float>();
    params.material_mu_r = material_mu_r.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.max_order = 1;
    params.receiver_model = RAYDTORCH_DFR_MATCHED_ISO;
    params.select_diffraction_point = select_diffraction_point ? 1 : 0;
    params.prefilter_visibility = prefilter_visibility ? 1 : 0;
    params.collect_debug_counts = 1;
    params.out_direct_count = direct_count.data_ptr<int>();
    params.out_direct_field_x_re = direct_x_re.data_ptr<float>();
    params.out_direct_field_x_im = direct_x_im.data_ptr<float>();
    params.out_direct_field_y_re = direct_y_re.data_ptr<float>();
    params.out_direct_field_y_im = direct_y_im.data_ptr<float>();
    params.out_direct_field_z_re = direct_z_re.data_ptr<float>();
    params.out_direct_field_z_im = direct_z_im.data_ptr<float>();
    params.out_multi_field_x_re = multi_x_re.data_ptr<float>();
    params.out_multi_field_x_im = multi_x_im.data_ptr<float>();
    params.out_multi_field_y_re = multi_y_re.data_ptr<float>();
    params.out_multi_field_y_im = multi_y_im.data_ptr<float>();
    params.out_multi_field_z_re = multi_z_re.data_ptr<float>();
    params.out_multi_field_z_im = multi_z_im.data_ptr<float>();
    params.out_multi_count = multi_count.data_ptr<int>();
    params.out_visibility_reject_count = visibility_reject_count.data_ptr<int>();
    params.out_utd_reject_count = utd_reject_count.data_ptr<int>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    auto pipeline = optix_pipeline_for_scene(scene, diffraction_accumulation_pipeline_config());
    pipeline->launch(11, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);

    return py::make_tuple(
        direct_x_re.reshape({grid_resolution1, grid_resolution0}),
        direct_x_im.reshape({grid_resolution1, grid_resolution0}),
        direct_y_re.reshape({grid_resolution1, grid_resolution0}),
        direct_y_im.reshape({grid_resolution1, grid_resolution0}),
        direct_z_re.reshape({grid_resolution1, grid_resolution0}),
        direct_z_im.reshape({grid_resolution1, grid_resolution0}),
        multi_x_re.reshape({grid_resolution1, grid_resolution0}),
        multi_x_im.reshape({grid_resolution1, grid_resolution0}),
        multi_y_re.reshape({grid_resolution1, grid_resolution0}),
        multi_y_im.reshape({grid_resolution1, grid_resolution0}),
        multi_z_re.reshape({grid_resolution1, grid_resolution0}),
        multi_z_im.reshape({grid_resolution1, grid_resolution0}),
        direct_count.reshape({grid_resolution1, grid_resolution0}),
        multi_count.reshape({grid_resolution1, grid_resolution0}),
        visibility_reject_count.reshape({grid_resolution1, grid_resolution0}),
        utd_reject_count.reshape({grid_resolution1, grid_resolution0}));
}

void bind_multipath_ops(py::module_ &m) {
    m.def("visibility_forward", &visibility_forward_op);
    m.def("trace_reflections_forward", &trace_reflections_forward_op);
    m.def("trace_reflections_backward", &trace_reflections_backward_op);
    m.def("trace_reflections_jvp", &trace_reflections_jvp_op);
    m.def("trace_refl_epc_field_forward", &trace_refl_epc_field_forward_op);
    m.def("trace_refl_epc_field_backward", &trace_refl_epc_field_backward_op);
    m.def("trace_refl_epc_field_jvp", &trace_refl_epc_field_jvp_op);
    m.def("reflection_dedup_forward", &reflection_dedup_forward_op);
    m.def("reflection_accumulation_forward", &reflection_accumulation_forward_op);
    m.def("diffraction_paths_order1_forward", &diffraction_paths_order1_forward_op);
    m.def("diffraction_accumulation_forward", &diffraction_accumulation_forward_op);
    m.def("diffraction_accumulation_direct_backward", &diffraction_accumulation_direct_backward_op);
    m.def("diffraction_accumulation_direct_jvp", &diffraction_accumulation_direct_jvp_op);
    m.def("diffraction_coherent_accumulation_forward", &diffraction_coherent_accumulation_forward_op);
}

} // namespace raydtorch
