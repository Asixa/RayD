// Copyright Xingyu Chen.
// Implements reflection support for reflection.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <src/diffraction/accum_params.h>
#include <src/diffraction/accum_ad.h>
#include <src/diffraction/paths_params.h>
#include <src/diffraction/pipeline.h>
#include <src/scene/geometry_kernels.h>
#include <src/runtime/optix_pipeline.h>
#include <src/reflection/kernels.h>
#include <src/reflection/pipeline.h>
#include <src/runtime/optix_context.h>
#include <src/scene/multipath_cuda.h>
#include <src/reflection/accum_reduce.h>
#include <src/reflection/accum_params.h>
#include <src/reflection/dedup.h>
#include <src/reflection/epc_field.h>
#include <src/reflection/epc_params.h>
#include <src/reflection/trace_params.h>
#include <src/scene/cache.h>
#include <src/bindings/tensor_contract.h>
#include <rayd/reflection.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rayd::torch_backend {

namespace {

constexpr int64_t kStagedReflAccumMinSamples = 2048;
constexpr int64_t kStagedReflAccumMinSamplesPerCell = 4;

void require_same_batch(const at::Tensor &a, const at::Tensor &b, const char *name) {
    if (a.size(0) != b.size(0))
        throw std::runtime_error(std::string(name) + " tensors must have the same batch size.");
}

void require_ray_tmax(const at::Tensor &ray_tmax, int64_t ray_count, const char *name) {
    if (!ray_tmax.defined())
        return;
    require_scalar_f(ray_tmax, "ray_tmax");
    if (ray_tmax.numel() != 0 && ray_tmax.size(0) != ray_count)
        throw std::runtime_error(std::string(name) + " ray_tmax must be empty or match the ray batch size.");
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

void require_flat_f32_strided(const at::Tensor &tensor, const char *name) {
    require_cuda(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 1, name);
}

void require_vec3f_strided(const at::Tensor &tensor, const char *name) {
    require_cuda(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 2, name);
    require_last_dim(tensor, 3, name);
}

// Host-side validation for the differentiable C ABI entry points
// (rayd::torch::trace_reflections_backward / trace_reflections_jvp and the reflection
// EPC path companions). These run off the hot forward path; the existing
// forward entries are untouched.

void require_ray_batch(const at::Tensor &tensor, int64_t ray_count, const char *name) {
    if (tensor.size(0) != ray_count)
        throw std::runtime_error(std::string(name) + " must match the ray batch size.");
}

void require_optional_active(const at::Tensor *active, int64_t ray_count) {
    if (active == nullptr || !active->defined())
        return;
    require_mask(*active, "active");
    if (active->numel() != 0 && active->size(0) != ray_count)
        throw std::runtime_error("active must be empty or match the ray batch size.");
}

void require_chain_tape_prim_id(const at::Tensor &tape_prim_id, int64_t ray_count) {
    require_cuda(tape_prim_id, "tape_prim_id");
    require_contiguous(tape_prim_id, "tape_prim_id");
    require_dtype(tape_prim_id, at::kInt, "tape_prim_id");
    require_rank(tape_prim_id, 2, "tape_prim_id");
    require_ray_batch(tape_prim_id, ray_count, "tape_prim_id");
    if (tape_prim_id.size(1) < 1)
        throw std::runtime_error("tape_prim_id must cover at least one bounce.");
}

void require_chain_tape_vec3(
    const at::Tensor &tensor,
    int64_t ray_count,
    int64_t bounce_count,
    const char *name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 3, name);
    require_ray_batch(tensor, ray_count, name);
    if (tensor.size(1) != bounce_count)
        throw std::runtime_error(std::string(name) + " must match the tape bounce count.");
    require_last_dim(tensor, 3, name);
}

void require_chain_tape_barycentric(
    const at::Tensor &tape_barycentric,
    int64_t ray_count,
    int64_t bounce_count) {
    require_cuda(tape_barycentric, "tape_barycentric");
    require_contiguous(tape_barycentric, "tape_barycentric");
    require_dtype(tape_barycentric, at::kFloat, "tape_barycentric");
    require_rank(tape_barycentric, 3, "tape_barycentric");
    require_ray_batch(tape_barycentric, ray_count, "tape_barycentric");
    if (tape_barycentric.size(1) != bounce_count)
        throw std::runtime_error("tape_barycentric must match the tape bounce count.");
    if (tape_barycentric.size(2) != 2 && tape_barycentric.size(2) != 3)
        throw std::runtime_error("tape_barycentric last dimension must be 2 or 3.");
}

// Gradients and tangents may be strided views (the kernels consume explicit
// strides), so contiguity is deliberately not required for them.
void require_optional_grad_vec(
    const at::Tensor *grad,
    int64_t ray_count,
    int64_t width,
    const char *name) {
    if (grad == nullptr)
        return;
    require_cuda(*grad, name);
    require_dtype(*grad, at::kFloat, name);
    if (width == 0) {
        require_rank(*grad, 1, name);
    } else {
        require_rank(*grad, 2, name);
        require_last_dim(*grad, width, name);
    }
    require_ray_batch(*grad, ray_count, name);
}

void require_optional_chain_grad_t(
    const at::Tensor *grad,
    int64_t ray_count,
    int64_t bounce_count) {
    if (grad == nullptr)
        return;
    require_cuda(*grad, "grad_t");
    require_dtype(*grad, at::kFloat, "grad_t");
    if (grad->dim() == 1) {
        require_ray_batch(*grad, ray_count, "grad_t");
        return;
    }
    require_rank(*grad, 2, "grad_t");
    require_ray_batch(*grad, ray_count, "grad_t");
    if (grad->size(1) != bounce_count)
        throw std::runtime_error("grad_t must match the tape bounce count.");
}

void require_optional_chain_grad_image_sources(
    const at::Tensor *grad,
    int64_t ray_count,
    int64_t bounce_count) {
    if (grad == nullptr)
        return;
    require_cuda(*grad, "grad_image_sources");
    require_dtype(*grad, at::kFloat, "grad_image_sources");
    require_rank(*grad, 3, "grad_image_sources");
    require_ray_batch(*grad, ray_count, "grad_image_sources");
    if (grad->size(1) != bounce_count)
        throw std::runtime_error("grad_image_sources must match the tape bounce count.");
    require_last_dim(*grad, 3, "grad_image_sources");
}

void require_optional_tangent_vertices(
    const at::Tensor *tangent,
    const at::Tensor &global_vertices,
    const char *name) {
    if (tangent == nullptr)
        return;
    require_cuda(*tangent, name);
    require_dtype(*tangent, at::kFloat, name);
    require_rank(*tangent, 2, name);
    require_last_dim(*tangent, 3, name);
    if (tangent->size(0) != global_vertices.size(0))
        throw std::runtime_error(
            std::string(name) + " must match the scene global vertex table.");
}

// Shape/dtype checks for the reflection EPC paths geometry companions. The
// winner sequence and the plane arrays are required to be the contiguous
// tensors the paths forward consumed; gradients/tangents may be strided views
// (the kernels consume explicit strides).

int64_t require_epc_paths_frozen_winner(
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &sequence,
    const at::Tensor &plane_points,
    const at::Tensor &plane_normals,
    const at::Tensor &valid,
    const at::Tensor &bounce_count) {
    require_vec3f(source, "source");
    require_vec3f(receiver, "receiver");
    const int64_t ray_count = source.size(0);
    require_ray_batch(receiver, ray_count, "receiver");
    require_cuda(sequence, "sequence");
    require_contiguous(sequence, "sequence");
    require_dtype(sequence, at::kInt, "sequence");
    require_rank(sequence, 2, "sequence");
    require_ray_batch(sequence, ray_count, "sequence");
    const int64_t bounce_width = sequence.size(1);
    if (bounce_width < 1)
        throw std::runtime_error("sequence must cover at least one bounce.");
    if (bounce_width > ReflEpcMaxBounces)
        throw std::runtime_error("sequence bounce count exceeds ReflEpcMaxBounces.");
    require_chain_tape_vec3(plane_points, ray_count, bounce_width, "plane_points");
    require_chain_tape_vec3(plane_normals, ray_count, bounce_width, "plane_normals");
    require_mask(valid, "valid");
    require_ray_batch(valid, ray_count, "valid");
    require_flat_i32(bounce_count, "bounce_count");
    require_ray_batch(bounce_count, ray_count, "bounce_count");
    return ray_count;
}

void require_optional_chain_grad_vec3(
    const at::Tensor *grad,
    int64_t ray_count,
    int64_t bounce_count,
    const char *name) {
    if (grad == nullptr)
        return;
    require_cuda(*grad, name);
    require_dtype(*grad, at::kFloat, name);
    require_rank(*grad, 3, name);
    require_ray_batch(*grad, ray_count, name);
    if (grad->size(1) != bounce_count)
        throw std::runtime_error(std::string(name) + " must match the sequence bounce count.");
    require_last_dim(*grad, 3, name);
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

const at::Tensor *optional_tensor(py::object obj, at::Tensor &storage) {
    if (obj.is_none())
        return nullptr;
    storage = obj.cast<at::Tensor>();
    if (!storage.defined() || storage.numel() == 0)
        return nullptr;
    return &storage;
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
    at::Tensor p0_packed;
    at::Tensor e1_packed;
    at::Tensor e2_packed;
    at::Tensor fn_packed;
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
        at::empty({0, 4}, mesh.vertices.options()),
        at::empty({0, 4}, mesh.vertices.options()),
        at::empty({0, 4}, mesh.vertices.options()),
        at::empty({0, 4}, mesh.vertices.options()),
        at::zeros({1}, mesh.faces.options()),
        static_cast<int32_t>(mesh.faces.size(0)),
    };
}

TriangleSoA make_scene_triangle_soa(const SceneCache &scene) {
    return {
        scene.tri_p0_x,
        scene.tri_p0_y,
        scene.tri_p0_z,
        scene.tri_e1_x,
        scene.tri_e1_y,
        scene.tri_e1_z,
        scene.tri_e2_x,
        scene.tri_e2_y,
        scene.tri_e2_z,
        scene.tri_fn_x,
        scene.tri_fn_y,
        scene.tri_fn_z,
        scene.tri_p0_packed,
        scene.tri_e1_packed,
        scene.tri_e2_packed,
        scene.tri_fn_packed,
        scene.face_offsets.contiguous(),
        static_cast<int32_t>(scene.global_faces.size(0)),
    };
}

const uint8_t *mask_ptr(const at::Tensor &mask) {
    return reinterpret_cast<const uint8_t *>(mask.data_ptr<bool>());
}

const uint8_t *optional_mask_ptr(const at::Tensor &mask) {
    if (!mask.defined() || mask.numel() == 0)
        return nullptr;
    return reinterpret_cast<const uint8_t *>(mask.data_ptr<bool>());
}

uint8_t *mutable_mask_ptr(const at::Tensor &mask) {
    return reinterpret_cast<uint8_t *>(mask.data_ptr<bool>());
}

at::Tensor optional_active_from_py(py::object active_obj, int64_t count, const char *name) {
    if (active_obj.is_none())
        return at::Tensor();
    at::Tensor active = active_obj.cast<at::Tensor>();
    require_mask(active, name);
    if (active.numel() == 0)
        return active.contiguous();
    if (active.size(0) != count)
        throw std::runtime_error(std::string(name) + " must match the batch size.");
    return active.contiguous();
}

at::Tensor optional_active_from_tensor(const at::Tensor *active_ptr, int64_t count, const char *name) {
    if (active_ptr == nullptr || !active_ptr->defined())
        return at::Tensor();
    const at::Tensor &active = *active_ptr;
    require_mask(active, name);
    if (active.numel() == 0)
        return active.contiguous();
    if (active.size(0) != count)
        throw std::runtime_error(std::string(name) + " must match the batch size.");
    return active.contiguous();
}

at::Tensor finite_vec3_rows(const at::Tensor &value) {
    return at::isfinite(value).all(1);
}

void require_scene_device(
    const SceneCache &scene,
    const at::Tensor &value,
    const char *name) {
    if (value.defined() && value.get_device() != scene.device_index)
        throw std::runtime_error(
            std::string(name) + " must be on the same CUDA device as the scene.");
}

void require_scene_device(
    const SceneCache &scene,
    const at::Tensor *value,
    const char *name) {
    if (value != nullptr)
        require_scene_device(scene, *value, name);
}

void require_scene_device(
    const SceneCache &scene,
    const c10::optional<at::Tensor> &value,
    const char *name) {
    if (value.has_value())
        require_scene_device(scene, *value, name);
}

at::Tensor stack_vec3(const at::Tensor &x, const at::Tensor &y, const at::Tensor &z) {
    return at::stack({x, y, z}, 1).contiguous();
}

py::tuple tensor_vector_to_tuple(const std::vector<at::Tensor> &values) {
    py::tuple result(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        result[i] = values[i];
    return result;
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

void launch_reflection_trace_backend(
    SceneCache &scene,
    const ReflectionTraceParams &params,
    unsigned int lane_count,
    cudaStream_t stream) {
    if (scene.trace_backend == TraceBackend::Cuda) {
        launch_reflection_trace_cuda(scene, params, static_cast<int>(lane_count));
        return;
    }
    optix_pipeline_for_scene(scene, reflection_trace_pipeline_config())
        ->launch(0, params, lane_count, stream);
}

} // namespace
std::vector<at::Tensor> trace_reflections_forward_native_impl(
    SceneCache &scene,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    const at::Tensor *active_ptr,
    int64_t max_bounces,
    bool export_tape,
    bool export_image_sources) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0), "trace_reflections");
    require_same_batch(ray_o, ray_d, "trace_reflections");
    at::Tensor active;
    if (active_ptr != nullptr && active_ptr->defined()) {
        require_mask(*active_ptr, "active");
        if (active_ptr->numel() != 0 && active_ptr->size(0) != ray_o.size(0))
            throw std::runtime_error("active must be empty or match the reflection ray batch size.");
        active = active_ptr->contiguous();
    }
    at::Tensor active_ctx = active.defined() ? active : at::empty({0}, ray_o.options().dtype(at::kBool));
    if (max_bounces < 1)
        throw std::runtime_error("max_bounces must be at least 1.");

    require_scene_device(scene, ray_o, "ray_o");
    require_scene_device(scene, ray_d, "ray_d");
    require_scene_device(scene, ray_tmax, "ray_tmax");
    require_scene_device(scene, active_ptr, "active");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));

    const int64_t ray_count = ray_o.size(0);
    auto fopts = ray_o.options();
    auto iopts = scene.global_faces.options();
    const bool write_image_sources = export_image_sources || export_tape;

    at::Tensor t = at::empty({ray_count, max_bounces}, fopts);
    at::Tensor prim_ids = at::empty({ray_count, max_bounces}, iopts);
    at::Tensor valid = at::empty({ray_count, max_bounces}, ray_o.options().dtype(at::kBool));
    at::Tensor image_sources;
    if (write_image_sources)
        image_sources = at::empty({ray_count, max_bounces, 3}, fopts);
    at::Tensor tape_barycentric;
    at::Tensor tape_hit_points;
    at::Tensor tape_normals;
    if (export_tape) {
        tape_barycentric = at::empty({ray_count, max_bounces, 3}, fopts);
        tape_hit_points = at::empty({ray_count, max_bounces, 3}, fopts);
        tape_normals = at::empty({ray_count, max_bounces, 3}, fopts);
    }

    if (ray_count == 0) {
        if (!export_tape && !export_image_sources)
            return {valid, t, prim_ids};
        if (!export_tape)
            return {valid, t, image_sources, prim_ids};
        return {
            valid,
            t,
            image_sources,
            prim_ids,
            prim_ids,
            tape_barycentric,
            tape_hit_points,
            tape_normals,
            active_ctx};
    }

    TriangleSoA tri = make_scene_triangle_soa(scene);
    at::Tensor ray_tmax_contig =
        !ray_tmax.defined() || ray_tmax.numel() == 0 ? ray_tmax : ray_tmax.contiguous();
    at::Tensor active_contig = active;

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
    params.tri_p0_packed = reinterpret_cast<const float4 *>(tri.p0_packed.data_ptr<float>());
    params.tri_e1_packed = reinterpret_cast<const float4 *>(tri.e1_packed.data_ptr<float>());
    params.tri_e2_packed = reinterpret_cast<const float4 *>(tri.e2_packed.data_ptr<float>());
    params.tri_fn_packed = reinterpret_cast<const float4 *>(tri.fn_packed.data_ptr<float>());
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.n_triangles = tri.n_triangles;
    params.ray_o_aos = ray_o.data_ptr<float>();
    params.ray_d_aos = ray_d.data_ptr<float>();
    params.ray_tmax = !ray_tmax_contig.defined() || ray_tmax_contig.numel() == 0
        ? nullptr
        : ray_tmax_contig.data_ptr<float>();
    params.active_mask = optional_mask_ptr(active_contig);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.max_bounces = static_cast<int32_t>(max_bounces);
    params.export_mode = 0;
    params.return_trailing = 0;
    params.output_layout = 0;
    params.out_valid = mutable_mask_ptr(valid);
    params.out_bounce_count = nullptr;
    params.out_shape_ids = nullptr;
    params.out_prim_ids = nullptr;
    params.out_global_prim_ids = prim_ids.data_ptr<int>();
    params.out_t = t.data_ptr<float>();
    params.out_bary_u = nullptr;
    params.out_bary_v = nullptr;
    params.out_bary = export_tape ? tape_barycentric.data_ptr<float>() : nullptr;
    params.out_hit_x = nullptr;
    params.out_hit_y = nullptr;
    params.out_hit_z = nullptr;
    params.out_hit = export_tape ? tape_hit_points.data_ptr<float>() : nullptr;
    params.out_norm_x = nullptr;
    params.out_norm_y = nullptr;
    params.out_norm_z = nullptr;
    params.out_norm = export_tape ? tape_normals.data_ptr<float>() : nullptr;
    params.out_img_x = nullptr;
    params.out_img_y = nullptr;
    params.out_img_z = nullptr;
    params.out_img = write_image_sources ? image_sources.data_ptr<float>() : nullptr;

    launch_reflection_trace_backend(
        scene, params, static_cast<unsigned int>(ray_count), torch_ctx.stream);

    if (!export_tape && !export_image_sources)
        return {valid, t, prim_ids};
    if (!export_tape)
        return {valid, t, image_sources, prim_ids};

    return {
        valid,
        t,
        image_sources,
        prim_ids,
        prim_ids,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        active_ctx};
}

py::tuple trace_reflections_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active,
    int64_t max_bounces) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(active, active_storage);
    return tensor_vector_to_tuple(trace_reflections_forward_native_impl(
        get_scene(scene_handle),
        ray_o,
        ray_d,
        ray_tmax,
        active_ptr,
        max_bounces,
        true,
        true));
}

py::tuple trace_reflections_forward_noad_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active,
    int64_t max_bounces) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(active, active_storage);
    return tensor_vector_to_tuple(trace_reflections_forward_native_impl(
        get_scene(scene_handle),
        ray_o,
        ray_d,
        ray_tmax,
        active_ptr,
        max_bounces,
        false,
        true));
}

py::tuple trace_reflections_forward_reduced_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active,
    int64_t max_bounces) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(active, active_storage);
    return tensor_vector_to_tuple(trace_reflections_forward_native_impl(
        get_scene(scene_handle),
        ray_o,
        ray_d,
        ray_tmax,
        active_ptr,
        max_bounces,
        false,
        false));
}



ReflectionBackwardOutputs integration_trace_reflections_backward_impl(
    SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor *ray_tmax,
    const at::Tensor *active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_hit_points,
    const at::Tensor &tape_normals,
    const at::Tensor &image_sources,
    const at::Tensor *grad_t,
    const at::Tensor *grad_image_sources) {
    auto maybe = [](const at::Tensor *tensor) -> const at::Tensor * {
        if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0)
            return nullptr;
        return tensor;
    };
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    const int64_t ray_count = ray_o.size(0);
    require_ray_batch(ray_d, ray_count, "ray_d");
    if (ray_tmax != nullptr && ray_tmax->defined())
        require_ray_tmax(*ray_tmax, ray_count, "trace_reflections_backward");
    require_optional_active(active, ray_count);
    require_chain_tape_prim_id(tape_prim_id, ray_count);
    const int64_t bounce_count = tape_prim_id.size(1);
    require_chain_tape_barycentric(tape_barycentric, ray_count, bounce_count);
    require_chain_tape_vec3(tape_hit_points, ray_count, bounce_count, "tape_hit_points");
    require_chain_tape_vec3(tape_normals, ray_count, bounce_count, "tape_normals");
    require_chain_tape_vec3(image_sources, ray_count, bounce_count, "image_sources");
    require_optional_chain_grad_t(maybe(grad_t), ray_count, bounce_count);
    require_optional_chain_grad_image_sources(maybe(grad_image_sources), ray_count, bounce_count);
    require_scene_device(scene, ray_o, "ray_o");
    require_scene_device(scene, ray_d, "ray_d");
    require_scene_device(scene, ray_tmax, "ray_tmax");
    require_scene_device(scene, active, "active");
    require_scene_device(scene, tape_prim_id, "tape_prim_id");
    require_scene_device(scene, tape_barycentric, "tape_barycentric");
    require_scene_device(scene, tape_hit_points, "tape_hit_points");
    require_scene_device(scene, tape_normals, "tape_normals");
    require_scene_device(scene, image_sources, "image_sources");
    require_scene_device(scene, grad_t, "grad_t");
    require_scene_device(scene, grad_image_sources, "grad_image_sources");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    at::Tensor ray_tmax_storage = ray_tmax == nullptr ? at::Tensor() : *ray_tmax;
    at::Tensor active_storage = active == nullptr ? at::Tensor() : *active;
    return reflection_chain_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        ray_tmax_storage,
        active_storage,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        image_sources,
        maybe(grad_t),
        maybe(grad_image_sources));
}

ReflectionJvpOutputs integration_trace_reflections_jvp_impl(
    SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor *active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_hit_points,
    const at::Tensor &tape_normals,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d,
    const at::Tensor &image_sources) {
    auto maybe = [](const at::Tensor *tensor) -> const at::Tensor * {
        if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0)
            return nullptr;
        return tensor;
    };
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    const int64_t ray_count = ray_o.size(0);
    require_ray_batch(ray_d, ray_count, "ray_d");
    require_optional_active(active, ray_count);
    require_chain_tape_prim_id(tape_prim_id, ray_count);
    const int64_t bounce_count = tape_prim_id.size(1);
    require_chain_tape_barycentric(tape_barycentric, ray_count, bounce_count);
    require_chain_tape_vec3(tape_hit_points, ray_count, bounce_count, "tape_hit_points");
    require_chain_tape_vec3(tape_normals, ray_count, bounce_count, "tape_normals");
    require_chain_tape_vec3(image_sources, ray_count, bounce_count, "image_sources");
    require_optional_tangent_vertices(maybe(tangent_vertices), scene.global_vertices, "tangent_vertices");
    require_optional_grad_vec(maybe(tangent_ray_o), ray_count, 3, "tangent_ray_o");
    require_optional_grad_vec(maybe(tangent_ray_d), ray_count, 3, "tangent_ray_d");
    require_scene_device(scene, ray_o, "ray_o");
    require_scene_device(scene, ray_d, "ray_d");
    require_scene_device(scene, active, "active");
    require_scene_device(scene, tape_prim_id, "tape_prim_id");
    require_scene_device(scene, tape_barycentric, "tape_barycentric");
    require_scene_device(scene, tape_hit_points, "tape_hit_points");
    require_scene_device(scene, tape_normals, "tape_normals");
    require_scene_device(scene, tangent_vertices, "tangent_vertices");
    require_scene_device(scene, tangent_ray_o, "tangent_ray_o");
    require_scene_device(scene, tangent_ray_d, "tangent_ray_d");
    require_scene_device(scene, image_sources, "image_sources");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    at::Tensor active_storage = active == nullptr ? at::Tensor() : *active;
    return reflection_chain_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active_storage,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        maybe(tangent_vertices),
        maybe(tangent_ray_o),
        maybe(tangent_ray_d),
        image_sources);
}



py::tuple trace_reflections_backward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_hit_points,
    at::Tensor tape_normals,
    at::Tensor image_sources,
    at::Tensor grad_t,
    at::Tensor grad_image_sources) {
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    ReflectionBackwardOutputs out = reflection_chain_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        image_sources,
        grad_t.numel() == 0 ? nullptr : &grad_t,
        grad_image_sources.numel() == 0 ? nullptr : &grad_image_sources);
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple trace_reflections_backward_optional_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_hit_points,
    at::Tensor tape_normals,
    at::Tensor image_sources,
    py::object grad_t_obj,
    py::object grad_image_sources_obj) {
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    at::Tensor grad_t_storage;
    at::Tensor grad_image_sources_storage;
    const at::Tensor *grad_t = optional_tensor(grad_t_obj, grad_t_storage);
    const at::Tensor *grad_image_sources = optional_tensor(grad_image_sources_obj, grad_image_sources_storage);
    ReflectionBackwardOutputs out = reflection_chain_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        image_sources,
        grad_t,
        grad_image_sources);
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple trace_reflections_jvp_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_hit_points,
    at::Tensor tape_normals,
    at::Tensor tangent_vertices,
    at::Tensor tangent_ray_o,
    at::Tensor tangent_ray_d,
    at::Tensor image_sources) {
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    ReflectionJvpOutputs out = reflection_chain_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        tangent_vertices.numel() == 0 ? nullptr : &tangent_vertices,
        tangent_ray_o.numel() == 0 ? nullptr : &tangent_ray_o,
        tangent_ray_d.numel() == 0 ? nullptr : &tangent_ray_d,
        image_sources);
    return py::make_tuple(out.tangent_t, out.tangent_image_sources);
}

py::tuple trace_reflections_jvp_optional_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_hit_points,
    at::Tensor tape_normals,
    py::object tangent_vertices_obj,
    py::object tangent_ray_o_obj,
    py::object tangent_ray_d_obj,
    at::Tensor image_sources) {
    SceneCache &scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_ray_o_storage;
    at::Tensor tangent_ray_d_storage;
    const at::Tensor *tangent_vertices = optional_tensor(tangent_vertices_obj, tangent_vertices_storage);
    const at::Tensor *tangent_ray_o = optional_tensor(tangent_ray_o_obj, tangent_ray_o_storage);
    const at::Tensor *tangent_ray_d = optional_tensor(tangent_ray_d_obj, tangent_ray_d_storage);
    ReflectionJvpOutputs out = reflection_chain_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_hit_points,
        tape_normals,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d,
        image_sources);
    return py::make_tuple(out.tangent_t, out.tangent_image_sources);
}

std::vector<at::Tensor> trace_refl_epc_field_forward_native_impl(
    int64_t scene_handle,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor *active_ptr,
    int64_t max_bounces) {
    require_vec3f(source, "source");
    require_vec3f(receiver, "receiver");
    require_same_batch(source, receiver, "trace_refl_epc_field");
    at::Tensor active = optional_active_from_tensor(active_ptr, source.size(0), "active");
    at::Tensor active_ctx = active.defined() ? active : at::empty({0}, source.options().dtype(at::kBool));
    if (max_bounces < 1)
        throw std::runtime_error("max_bounces must be at least 1.");

    SceneCache &scene = get_scene(scene_handle);
    require_scene_device(scene, source, "source");
    require_scene_device(scene, receiver, "receiver");
    require_scene_device(scene, active_ptr, "active");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    const int64_t ray_count = source.size(0);
    const int64_t slot_count = ray_count * max_bounces;
    auto fopts = source.options();
    auto iopts = scene.global_faces.options();
    at::Tensor field_real = at::empty({ray_count}, fopts);
    at::Tensor field_imag = at::empty({ray_count}, fopts);
    at::Tensor path_length = at::empty({ray_count}, fopts);
    at::Tensor valid = at::empty({ray_count}, source.options().dtype(at::kBool));
    at::Tensor resolved_first = at::empty({ray_count}, iopts);
    at::Tensor tape_prim_id = at::empty({ray_count}, iopts);
    at::Tensor tape_barycentric = at::empty({ray_count, 3}, fopts);
    if (ray_count == 0) {
        return {
            field_real,
            field_imag,
            path_length,
            valid,
            resolved_first,
            tape_prim_id,
            tape_barycentric,
            active_ctx};
    }

    TriangleSoA tri = make_scene_triangle_soa(scene);
    at::Tensor active_contig = active;

    at::Tensor source_x = at::empty({ray_count}, fopts);
    at::Tensor source_y = at::empty({ray_count}, fopts);
    at::Tensor source_z = at::empty({ray_count}, fopts);
    at::Tensor receiver_x = at::empty({ray_count}, fopts);
    at::Tensor receiver_y = at::empty({ray_count}, fopts);
    at::Tensor receiver_z = at::empty({ray_count}, fopts);
    at::Tensor ray_dx = at::empty({ray_count}, fopts);
    at::Tensor ray_dy = at::empty({ray_count}, fopts);
    at::Tensor ray_dz = at::empty({ray_count}, fopts);
    at::Tensor ray_tmax = at::empty({ray_count}, fopts);

    at::Tensor epc_valid = at::empty({ray_count}, source.options().dtype(at::kBool));
    at::Tensor epc_bounce_count = at::empty({ray_count}, iopts);
    at::Tensor epc_path_length = at::empty({ray_count}, fopts);
    at::Tensor point_x = at::empty({slot_count}, fopts);
    at::Tensor point_y = at::empty({slot_count}, fopts);
    at::Tensor point_z = at::empty({slot_count}, fopts);
    at::Tensor trace_prim_ids = at::empty({slot_count}, iopts);
    at::Tensor resolved_prim_ids = at::empty({slot_count}, iopts);
    at::Tensor surface_group_ids = at::empty({slot_count}, iopts);
    at::Tensor plane_normal_x = at::empty({slot_count}, fopts);
    at::Tensor plane_normal_y = at::empty({slot_count}, fopts);
    at::Tensor plane_normal_z = at::empty({slot_count}, fopts);
    at::Tensor first_blocked_segment = at::empty({ray_count}, iopts);
    at::Tensor first_blocked_prim = at::empty({ray_count}, iopts);
    at::Tensor first_blocked_group = at::empty({ray_count}, iopts);

    ReflEpcForwardSetupParams setup_params = {};
    setup_params.n_rays = static_cast<int32_t>(ray_count);
    setup_params.max_bounces = static_cast<int32_t>(max_bounces);
    setup_params.source_aos = source.data_ptr<float>();
    setup_params.receiver_aos = receiver.data_ptr<float>();
    setup_params.source_x = source_x.data_ptr<float>();
    setup_params.source_y = source_y.data_ptr<float>();
    setup_params.source_z = source_z.data_ptr<float>();
    setup_params.receiver_x = receiver_x.data_ptr<float>();
    setup_params.receiver_y = receiver_y.data_ptr<float>();
    setup_params.receiver_z = receiver_z.data_ptr<float>();
    setup_params.ray_dx = ray_dx.data_ptr<float>();
    setup_params.ray_dy = ray_dy.data_ptr<float>();
    setup_params.ray_dz = ray_dz.data_ptr<float>();
    setup_params.ray_tmax = ray_tmax.data_ptr<float>();
    setup_params.epc_valid = mutable_mask_ptr(epc_valid);
    setup_params.epc_bounce_count = epc_bounce_count.data_ptr<int>();
    setup_params.epc_path_length = epc_path_length.data_ptr<float>();
    setup_params.point_x = point_x.data_ptr<float>();
    setup_params.point_y = point_y.data_ptr<float>();
    setup_params.point_z = point_z.data_ptr<float>();
    setup_params.trace_prim_ids = trace_prim_ids.data_ptr<int>();
    setup_params.resolved_prim_ids = resolved_prim_ids.data_ptr<int>();
    setup_params.surface_group_ids = surface_group_ids.data_ptr<int>();
    setup_params.plane_normal_x = plane_normal_x.data_ptr<float>();
    setup_params.plane_normal_y = plane_normal_y.data_ptr<float>();
    setup_params.plane_normal_z = plane_normal_z.data_ptr<float>();
    setup_params.first_blocked_segment = first_blocked_segment.data_ptr<int>();
    setup_params.first_blocked_prim = first_blocked_prim.data_ptr<int>();
    setup_params.first_blocked_group = first_blocked_group.data_ptr<int>();
    setup_params.tape_barycentric = tape_barycentric.data_ptr<float>();
    reflection_epc_forward_setup_gpu(setup_params, static_cast<int>(scene.device_index));

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
    epc_params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    epc_params.n_triangles = tri.n_triangles;
    epc_params.visibility_ignore_mode = ReflEpcVisibilityIgnorePrimitive;
    epc_params.ray_ox = source_x.data_ptr<float>();
    epc_params.ray_oy = source_y.data_ptr<float>();
    epc_params.ray_oz = source_z.data_ptr<float>();
    epc_params.ray_dx = ray_dx.data_ptr<float>();
    epc_params.ray_dy = ray_dy.data_ptr<float>();
    epc_params.ray_dz = ray_dz.data_ptr<float>();
    epc_params.ray_tmax = ray_tmax.data_ptr<float>();
    epc_params.rx_x = receiver_x.data_ptr<float>();
    epc_params.rx_y = receiver_y.data_ptr<float>();
    epc_params.rx_z = receiver_z.data_ptr<float>();
    epc_params.rx_count = static_cast<int32_t>(ray_count);
    epc_params.active_mask = optional_mask_ptr(active_contig);
    epc_params.n_rays = static_cast<int32_t>(ray_count);
    epc_params.max_bounces = static_cast<int32_t>(max_bounces);
    epc_params.plane_tolerance = 1e-5f;
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
    if (scene.trace_backend == TraceBackend::Cuda) {
        launch_reflection_epc_cuda(
            scene, epc_params, false, false, static_cast<int>(ray_count));
    } else {
        optix_pipeline_for_scene(scene, reflection_epc_pipeline_config())
            ->launch(0, epc_params, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    }

    ReflEpcFieldParams field_params = {};
    field_params.n_rays = static_cast<int32_t>(ray_count);
    field_params.max_bounces = static_cast<int32_t>(max_bounces);
    field_params.epc_valid = mask_ptr(epc_valid);
    field_params.epc_bounce_count = epc_bounce_count.data_ptr<int>();
    field_params.epc_path_length = epc_path_length.data_ptr<float>();
    field_params.ray_ox = source_x.data_ptr<float>();
    field_params.ray_oy = source_y.data_ptr<float>();
    field_params.ray_oz = source_z.data_ptr<float>();
    field_params.rx_x = receiver_x.data_ptr<float>();
    field_params.rx_y = receiver_y.data_ptr<float>();
    field_params.rx_z = receiver_z.data_ptr<float>();
    field_params.rx_count = static_cast<int32_t>(ray_count);
    field_params.hit_x = point_x.data_ptr<float>();
    field_params.hit_y = point_y.data_ptr<float>();
    field_params.hit_z = point_z.data_ptr<float>();
    field_params.epc_normal_x = plane_normal_x.data_ptr<float>();
    field_params.epc_normal_y = plane_normal_y.data_ptr<float>();
    field_params.epc_normal_z = plane_normal_z.data_ptr<float>();
    field_params.trace_prim_ids = trace_prim_ids.data_ptr<int>();
    field_params.resolved_prim_ids = resolved_prim_ids.data_ptr<int>();
    field_params.surface_group_ids = surface_group_ids.data_ptr<int>();
    field_params.slot_normal_x = plane_normal_x.data_ptr<float>();
    field_params.slot_normal_y = plane_normal_y.data_ptr<float>();
    field_params.slot_normal_z = plane_normal_z.data_ptr<float>();
    field_params.tx_pol_count = 1;
    field_params.omega = 2.0f * 3.14159265358979323846f * 299792458.0f;
    field_params.wavelength = 1.0f;
    field_params.out_valid = mutable_mask_ptr(valid);
    field_params.out_path_length = path_length.data_ptr<float>();
    field_params.out_field_x_re = field_real.data_ptr<float>();
    field_params.out_field_x_im = field_imag.data_ptr<float>();
    field_params.out_first_resolved_prim_id = resolved_first.data_ptr<int>();
    field_params.out_first_trace_prim_id = tape_prim_id.data_ptr<int>();
    reflection_epc_field_gpu(field_params, static_cast<int>(scene.device_index));

    return {
        field_real,
        field_imag,
        path_length,
        valid,
        resolved_first,
        tape_prim_id,
        tape_barycentric,
        active_ctx};
}

py::tuple trace_refl_epc_field_forward_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    py::object active_obj,
    int64_t max_bounces) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(active_obj, active_storage);
    return tensor_vector_to_tuple(trace_refl_epc_field_forward_native_impl(
        scene_handle,
        source,
        receiver,
        active_ptr,
        max_bounces));
}

std::vector<at::Tensor> reflection_epc_paths_forward_native_impl(
    SceneCache &scene,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor *active_ptr,
    const at::Tensor &expected_prim_ids,
    const at::Tensor &direct_plane_points,
    const at::Tensor &direct_plane_normals,
    const at::Tensor &surface_group_id,
    const at::Tensor &surface_group_size,
    const at::Tensor &surface_group_members,
    int64_t max_bounces,
    int64_t visibility_ignore_mode,
    double plane_tolerance) {
    require_vec3f(source, "source");
    require_vec3f(receiver, "receiver");
    require_same_batch(source, receiver, "reflection_epc_paths");
    require_cuda(expected_prim_ids, "expected_prim_ids");
    require_contiguous(expected_prim_ids, "expected_prim_ids");
    require_dtype(expected_prim_ids, at::kInt, "expected_prim_ids");
    require_rank(expected_prim_ids, 2, "expected_prim_ids");
    require_cuda(direct_plane_points, "direct_plane_points");
    require_contiguous(direct_plane_points, "direct_plane_points");
    require_dtype(direct_plane_points, at::kFloat, "direct_plane_points");
    require_rank(direct_plane_points, 3, "direct_plane_points");
    require_last_dim(direct_plane_points, 3, "direct_plane_points");
    require_cuda(direct_plane_normals, "direct_plane_normals");
    require_contiguous(direct_plane_normals, "direct_plane_normals");
    require_dtype(direct_plane_normals, at::kFloat, "direct_plane_normals");
    require_rank(direct_plane_normals, 3, "direct_plane_normals");
    require_last_dim(direct_plane_normals, 3, "direct_plane_normals");
    require_flat_i32(surface_group_id, "surface_group_id");
    require_flat_i32(surface_group_size, "surface_group_size");
    require_flat_i32(surface_group_members, "surface_group_members");
    if (max_bounces < 1)
        throw std::runtime_error("max_bounces must be at least 1.");
    if (!(plane_tolerance >= 0.0) || !std::isfinite(plane_tolerance))
        throw std::runtime_error("plane_tolerance must be finite and non-negative.");
    if (expected_prim_ids.size(0) != source.size(0) || expected_prim_ids.size(1) != max_bounces)
        throw std::runtime_error("expected_prim_ids must have shape (N, max_bounces).");
    if (direct_plane_points.size(0) != source.size(0) || direct_plane_points.size(1) != max_bounces)
        throw std::runtime_error("direct_plane_points must have shape (N, max_bounces, 3).");
    if (!direct_plane_normals.sizes().equals(direct_plane_points.sizes()))
        throw std::runtime_error("direct_plane_normals must match direct_plane_points.");
    if (surface_group_size.size(0) <= 0)
        throw std::runtime_error("surface_group_size must contain at least one group.");
    if (surface_group_members.numel() % surface_group_size.size(0) != 0)
        throw std::runtime_error("surface_group_members must be padded to group_count * max_group_size.");

    at::Tensor active = optional_active_from_tensor(active_ptr, source.size(0), "active");
    require_scene_device(scene, source, "source");
    require_scene_device(scene, receiver, "receiver");
    require_scene_device(scene, active_ptr, "active");
    require_scene_device(scene, expected_prim_ids, "expected_prim_ids");
    require_scene_device(scene, direct_plane_points, "direct_plane_points");
    require_scene_device(scene, direct_plane_normals, "direct_plane_normals");
    require_scene_device(scene, surface_group_id, "surface_group_id");
    require_scene_device(scene, surface_group_size, "surface_group_size");
    require_scene_device(scene, surface_group_members, "surface_group_members");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    const int64_t ray_count = source.size(0);
    const int64_t slot_count = ray_count * max_bounces;
    auto fopts = source.options();
    auto iopts = scene.global_faces.options();
    at::Tensor valid = at::empty({ray_count}, source.options().dtype(at::kBool));
    at::Tensor path_length = at::empty({ray_count}, fopts);
    at::Tensor resolved_prim_ids = at::empty({ray_count, max_bounces}, iopts);
    at::Tensor surface_group_ids = at::empty({ray_count, max_bounces}, iopts);
    at::Tensor point_x = at::empty({slot_count}, fopts);
    at::Tensor point_y = at::empty({slot_count}, fopts);
    at::Tensor point_z = at::empty({slot_count}, fopts);
    at::Tensor plane_normal_x = at::empty({slot_count}, fopts);
    at::Tensor plane_normal_y = at::empty({slot_count}, fopts);
    at::Tensor plane_normal_z = at::empty({slot_count}, fopts);
    at::Tensor trace_prim_ids = at::empty({slot_count}, iopts);
    at::Tensor bounce_count = at::empty({ray_count}, iopts);
    at::Tensor first_blocked_segment = at::empty({ray_count}, iopts);
    at::Tensor first_blocked_prim = at::empty({ray_count}, iopts);
    at::Tensor first_blocked_group = at::empty({ray_count}, iopts);
    at::Tensor tape_barycentric = at::empty({ray_count, 3}, fopts);
    at::Tensor source_x = at::empty({ray_count}, fopts);
    at::Tensor source_y = at::empty({ray_count}, fopts);
    at::Tensor source_z = at::empty({ray_count}, fopts);
    at::Tensor receiver_x = at::empty({ray_count}, fopts);
    at::Tensor receiver_y = at::empty({ray_count}, fopts);
    at::Tensor receiver_z = at::empty({ray_count}, fopts);
    at::Tensor ray_dx = at::empty({ray_count}, fopts);
    at::Tensor ray_dy = at::empty({ray_count}, fopts);
    at::Tensor ray_dz = at::empty({ray_count}, fopts);
    at::Tensor ray_tmax = at::empty({ray_count}, fopts);

    if (ray_count == 0) {
        at::Tensor hit_positions = stack_vec3(point_x, point_y, point_z).reshape({ray_count, max_bounces, 3}).contiguous();
        at::Tensor normals = stack_vec3(plane_normal_x, plane_normal_y, plane_normal_z).reshape({ray_count, max_bounces, 3}).contiguous();
        return {valid, path_length, resolved_prim_ids, surface_group_ids, hit_positions, normals};
    }

    at::Tensor direct_plane_points_flat = direct_plane_points.reshape({slot_count, 3}).contiguous();
    at::Tensor direct_plane_normals_flat = direct_plane_normals.reshape({slot_count, 3}).contiguous();
    Vec3SoA plane_points = split_vec3(direct_plane_points_flat);
    Vec3SoA plane_normals = split_vec3(direct_plane_normals_flat);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    at::Tensor resolved_flat = resolved_prim_ids.reshape({slot_count}).contiguous();
    at::Tensor group_flat = surface_group_ids.reshape({slot_count}).contiguous();

    ReflEpcForwardSetupParams setup_params = {};
    setup_params.n_rays = static_cast<int32_t>(ray_count);
    setup_params.max_bounces = static_cast<int32_t>(max_bounces);
    setup_params.source_aos = source.data_ptr<float>();
    setup_params.receiver_aos = receiver.data_ptr<float>();
    setup_params.source_x = source_x.data_ptr<float>();
    setup_params.source_y = source_y.data_ptr<float>();
    setup_params.source_z = source_z.data_ptr<float>();
    setup_params.receiver_x = receiver_x.data_ptr<float>();
    setup_params.receiver_y = receiver_y.data_ptr<float>();
    setup_params.receiver_z = receiver_z.data_ptr<float>();
    setup_params.ray_dx = ray_dx.data_ptr<float>();
    setup_params.ray_dy = ray_dy.data_ptr<float>();
    setup_params.ray_dz = ray_dz.data_ptr<float>();
    setup_params.ray_tmax = ray_tmax.data_ptr<float>();
    setup_params.epc_valid = mutable_mask_ptr(valid);
    setup_params.epc_bounce_count = bounce_count.data_ptr<int>();
    setup_params.epc_path_length = path_length.data_ptr<float>();
    setup_params.point_x = point_x.data_ptr<float>();
    setup_params.point_y = point_y.data_ptr<float>();
    setup_params.point_z = point_z.data_ptr<float>();
    setup_params.trace_prim_ids = trace_prim_ids.data_ptr<int>();
    setup_params.resolved_prim_ids = resolved_flat.data_ptr<int>();
    setup_params.surface_group_ids = group_flat.data_ptr<int>();
    setup_params.plane_normal_x = plane_normal_x.data_ptr<float>();
    setup_params.plane_normal_y = plane_normal_y.data_ptr<float>();
    setup_params.plane_normal_z = plane_normal_z.data_ptr<float>();
    setup_params.first_blocked_segment = first_blocked_segment.data_ptr<int>();
    setup_params.first_blocked_prim = first_blocked_prim.data_ptr<int>();
    setup_params.first_blocked_group = first_blocked_group.data_ptr<int>();
    setup_params.tape_barycentric = tape_barycentric.data_ptr<float>();
    reflection_epc_forward_setup_gpu(setup_params, static_cast<int>(scene.device_index));

    ReflEpcParams epc_params = {};
    epc_params.primary_handle = scene.triangle_ias.traversable;
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
    epc_params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    epc_params.n_triangles = tri.n_triangles;
    epc_params.expected_prim_ids = expected_prim_ids.data_ptr<int>();
    epc_params.expected_prim_count = static_cast<int32_t>(slot_count);
    epc_params.surface_group_id = surface_group_id.data_ptr<int>();
    epc_params.surface_group_id_count = static_cast<int32_t>(surface_group_id.size(0));
    epc_params.surface_group_size = surface_group_size.data_ptr<int>();
    epc_params.surface_group_count = static_cast<int32_t>(surface_group_size.size(0));
    epc_params.surface_group_members = surface_group_members.data_ptr<int>();
    epc_params.surface_max_group_size = static_cast<int32_t>(surface_group_members.numel() / surface_group_size.size(0));
    epc_params.visibility_ignore_mode = static_cast<int32_t>(visibility_ignore_mode);
    epc_params.ray_ox = source_x.data_ptr<float>();
    epc_params.ray_oy = source_y.data_ptr<float>();
    epc_params.ray_oz = source_z.data_ptr<float>();
    epc_params.ray_dx = ray_dx.data_ptr<float>();
    epc_params.ray_dy = ray_dy.data_ptr<float>();
    epc_params.ray_dz = ray_dz.data_ptr<float>();
    epc_params.ray_tmax = ray_tmax.data_ptr<float>();
    epc_params.direct_plane_point_x = plane_points.x.data_ptr<float>();
    epc_params.direct_plane_point_y = plane_points.y.data_ptr<float>();
    epc_params.direct_plane_point_z = plane_points.z.data_ptr<float>();
    epc_params.direct_plane_normal_x = plane_normals.x.data_ptr<float>();
    epc_params.direct_plane_normal_y = plane_normals.y.data_ptr<float>();
    epc_params.direct_plane_normal_z = plane_normals.z.data_ptr<float>();
    epc_params.rx_x = receiver_x.data_ptr<float>();
    epc_params.rx_y = receiver_y.data_ptr<float>();
    epc_params.rx_z = receiver_z.data_ptr<float>();
    epc_params.rx_count = static_cast<int32_t>(ray_count);
    epc_params.active_mask = optional_mask_ptr(active);
    epc_params.n_rays = static_cast<int32_t>(ray_count);
    epc_params.max_bounces = static_cast<int32_t>(max_bounces);
    epc_params.plane_tolerance = static_cast<float>(plane_tolerance);
    epc_params.out_valid = mutable_mask_ptr(valid);
    epc_params.out_bounce_count = bounce_count.data_ptr<int>();
    epc_params.out_path_length = path_length.data_ptr<float>();
    epc_params.out_point_x = point_x.data_ptr<float>();
    epc_params.out_point_y = point_y.data_ptr<float>();
    epc_params.out_point_z = point_z.data_ptr<float>();
    epc_params.out_trace_prim_ids = trace_prim_ids.data_ptr<int>();
    epc_params.out_resolved_prim_ids = resolved_flat.data_ptr<int>();
    epc_params.out_surface_group_ids = group_flat.data_ptr<int>();
    epc_params.out_plane_normal_x = plane_normal_x.data_ptr<float>();
    epc_params.out_plane_normal_y = plane_normal_y.data_ptr<float>();
    epc_params.out_plane_normal_z = plane_normal_z.data_ptr<float>();
    epc_params.out_first_blocked_segment = first_blocked_segment.data_ptr<int>();
    epc_params.out_first_blocked_prim = first_blocked_prim.data_ptr<int>();
    epc_params.out_first_blocked_group = first_blocked_group.data_ptr<int>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (scene.trace_backend == TraceBackend::Cuda) {
        launch_reflection_epc_cuda(
            scene, epc_params, false, false, static_cast<int>(ray_count));
    } else {
        optix_pipeline_for_scene(scene, reflection_epc_pipeline_config())
            ->launch(0, epc_params, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    }

    at::Tensor hit_positions = stack_vec3(point_x, point_y, point_z).reshape({ray_count, max_bounces, 3}).contiguous();
    at::Tensor normals = stack_vec3(plane_normal_x, plane_normal_y, plane_normal_z).reshape({ray_count, max_bounces, 3}).contiguous();
    return {
        valid,
        path_length,
        resolved_flat.reshape({ray_count, max_bounces}).contiguous(),
        group_flat.reshape({ray_count, max_bounces}).contiguous(),
        hit_positions,
        normals,
    };
}

py::tuple reflection_epc_paths_forward_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    py::object active_obj,
    at::Tensor expected_prim_ids,
    at::Tensor direct_plane_points,
    at::Tensor direct_plane_normals,
    at::Tensor surface_group_id,
    at::Tensor surface_group_size,
    at::Tensor surface_group_members,
    int64_t max_bounces,
    int64_t visibility_ignore_mode,
    double plane_tolerance) {
    at::Tensor active_storage;
    const at::Tensor *active = optional_tensor(active_obj, active_storage);
    return tensor_vector_to_tuple(reflection_epc_paths_forward_native_impl(
        get_scene(scene_handle),
        source,
        receiver,
        active,
        expected_prim_ids,
        direct_plane_points,
        direct_plane_normals,
        surface_group_id,
        surface_group_size,
        surface_group_members,
        max_bounces,
        visibility_ignore_mode,
        plane_tolerance));
}

py::tuple trace_refl_epc_field_backward_op(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tape_t,
    py::object grad_field_real_obj,
    py::object grad_field_imag_obj,
    py::object grad_path_length_obj,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver) {
    SceneCache &scene = get_scene(scene_handle);
    at::Tensor grad_field_real_storage;
    at::Tensor grad_field_imag_storage;
    at::Tensor grad_path_length_storage;
    const at::Tensor *grad_field_real =
        optional_tensor(grad_field_real_obj, grad_field_real_storage);
    const at::Tensor *grad_field_imag =
        optional_tensor(grad_field_imag_obj, grad_field_imag_storage);
    const at::Tensor *grad_path_length =
        optional_tensor(grad_path_length_obj, grad_path_length_storage);
    const int64_t ray_count = source.size(0);
    if (grad_field_real != nullptr) {
        require_flat_f32_strided(*grad_field_real, "grad_field_real");
        if (grad_field_real->size(0) != ray_count)
            throw std::runtime_error("grad_field_real must match the EPC batch size.");
    }
    if (grad_field_imag != nullptr) {
        require_flat_f32_strided(*grad_field_imag, "grad_field_imag");
        if (grad_field_imag->size(0) != ray_count)
            throw std::runtime_error("grad_field_imag must match the EPC batch size.");
    }
    if (grad_path_length != nullptr) {
        require_flat_f32_strided(*grad_path_length, "grad_path_length");
        if (grad_path_length->size(0) != ray_count)
            throw std::runtime_error("grad_path_length must match the EPC batch size.");
    }
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    ReflEpcBackwardOutputs out = refl_epc_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        source,
        receiver,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        grad_field_real,
        grad_field_imag,
        grad_path_length,
        need_grad_vertices,
        need_grad_source,
        need_grad_receiver);
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
    py::object tangent_vertices_obj,
    py::object tangent_source_obj,
    py::object tangent_receiver_obj) {
    SceneCache &scene = get_scene(scene_handle);
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_source_storage;
    at::Tensor tangent_receiver_storage;
    const at::Tensor *tangent_vertices =
        optional_tensor(tangent_vertices_obj, tangent_vertices_storage);
    const at::Tensor *tangent_source =
        optional_tensor(tangent_source_obj, tangent_source_storage);
    const at::Tensor *tangent_receiver =
        optional_tensor(tangent_receiver_obj, tangent_receiver_storage);
    if (tangent_vertices != nullptr) {
        require_vec3f_strided(*tangent_vertices, "tangent_vertices");
        if (tangent_vertices->size(0) != scene.global_vertices.size(0))
            throw std::runtime_error("tangent_vertices must match the scene global vertex count.");
    }
    if (tangent_source != nullptr) {
        require_vec3f_strided(*tangent_source, "tangent_source");
        if (tangent_source->size(0) != source.size(0))
            throw std::runtime_error("tangent_source must match the EPC batch size.");
    }
    if (tangent_receiver != nullptr) {
        require_vec3f_strided(*tangent_receiver, "tangent_receiver");
        if (tangent_receiver->size(0) != receiver.size(0))
            throw std::runtime_error("tangent_receiver must match the EPC batch size.");
    }
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    ReflEpcJvpOutputs out = refl_epc_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        source,
        receiver,
        active,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        tangent_vertices,
        tangent_source,
        tangent_receiver);
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
            static_cast<int>(bounce_count.get_device()),
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

struct ReflectionAccumulationNativeOutputs {
    at::Tensor power;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
    at::Tensor field_y_re;
    at::Tensor field_y_im;
    at::Tensor field_z_re;
    at::Tensor field_z_im;
    at::Tensor reflection_count;
    at::Tensor wedge_count;
    at::Tensor wedge_ray_index;
    at::Tensor wedge_hit;
    at::Tensor wedge_normal;
    at::Tensor wedge_prim_id;
    at::Tensor wedge_direction;
    at::Tensor wedge_source;
    at::Tensor wedge_source_power;
    at::Tensor wedge_initial_direction;
    at::Tensor wedge_bounce_depth;
};

ReflectionAccumulationNativeOutputs reflection_accumulation_forward_native_impl(
    SceneCache &scene,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    const at::Tensor *active,
    at::Tensor tx,
    at::Tensor tx_pol,
    at::Tensor material_eta_r,
    at::Tensor material_sigma,
    at::Tensor material_mu_r,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength,
    double solid_angle_per_ray,
    bool collect_wedges,
    bool collect_wedge_prefixes,
    int64_t wedge_capacity,
    int64_t wedge_sample_stride,
    int64_t accumulation_strategy,
    int64_t compact_min_samples,
    int64_t staged_min_samples_per_cell,
    int64_t procedural_sample_count,
    bool include_los) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0), "reflection_accumulation");
    if (active != nullptr)
        require_mask(*active, "active");
    require_vec3f(tx, "tx");
    require_vec3f(tx_pol, "tx_pol");
    require_flat_f32(material_eta_r, "material_eta_r");
    require_flat_f32(material_sigma, "material_sigma");
    require_flat_f32(material_mu_r, "material_mu_r");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    require_same_batch(ray_o, ray_d, "reflection_accumulation");
    require_same_batch(ray_o, tx, "reflection_accumulation");
    require_same_batch(ray_o, tx_pol, "reflection_accumulation");
    if (active != nullptr && active->numel() != 0 && active->size(0) != ray_o.size(0))
        throw std::runtime_error("ray_tmax and active must match the ray batch size.");
    if (max_bounces < 0)
        throw std::runtime_error("max_bounces must be non-negative.");
    if (grid_axis < 0 || grid_axis > 2)
        throw std::runtime_error("grid_axis must be 0, 1, or 2.");
    if (grid_resolution0 <= 0 || grid_resolution1 <= 0)
        throw std::runtime_error("grid resolutions must be positive.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");
    if (!(solid_angle_per_ray >= 0.0))
        throw std::runtime_error("solid_angle_per_ray must be non-negative.");
    if (wedge_capacity < 0)
        throw std::runtime_error("wedge_capacity must be non-negative.");
    if (wedge_sample_stride <= 0)
        throw std::runtime_error("wedge_sample_stride must be positive.");
    if (accumulation_strategy < 0 || accumulation_strategy > 4)
        throw std::runtime_error("accumulation_strategy is not supported.");
    if (compact_min_samples < 0 || staged_min_samples_per_cell < 0 || procedural_sample_count < 0)
        throw std::runtime_error("accumulation thresholds and procedural_sample_count must be non-negative.");

    require_scene_device(scene, ray_o, "ray_o");
    require_scene_device(scene, ray_d, "ray_d");
    require_scene_device(scene, ray_tmax, "ray_tmax");
    require_scene_device(scene, active, "active");
    require_scene_device(scene, tx, "tx");
    require_scene_device(scene, tx_pol, "tx_pol");
    require_scene_device(scene, material_eta_r, "material_eta_r");
    require_scene_device(scene, material_sigma, "material_sigma");
    require_scene_device(scene, material_mu_r, "material_mu_r");
    require_scene_device(scene, material_gain, "material_gain");
    require_scene_device(scene, material_valid, "material_valid");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    const int64_t material_count = material_eta_r.size(0);
    if (material_count != scene.global_faces.size(0) ||
        material_sigma.size(0) != material_count ||
        material_mu_r.size(0) != material_count ||
        material_gain.size(0) != material_count ||
        material_valid.size(0) != material_count) {
        throw std::runtime_error("reflection material payload must match the scene triangle count.");
    }

    const int64_t ray_count = ray_o.size(0);
    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    const int32_t max_bounces_i = checked_i32(max_bounces, "max_bounces");
    const int64_t stage_depth_count = max_bounces + 1;
    const bool stage_sample_count_fits =
        ray_count <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) /
                         std::max<int64_t>(stage_depth_count, 1);
    const int64_t stage_sample_count =
        stage_sample_count_fits ? ray_count * stage_depth_count : 0;
    const int64_t staged_min_samples = compact_min_samples > 0
        ? compact_min_samples
        : kStagedReflAccumMinSamples;
    const int64_t staged_min_per_cell = staged_min_samples_per_cell > 0
        ? staged_min_samples_per_cell
        : kStagedReflAccumMinSamplesPerCell;
    const bool force_staged = accumulation_strategy == 2;
    const bool force_atomic = accumulation_strategy == 1 || accumulation_strategy == 3 || accumulation_strategy == 4;
    const bool staged_accum =
        !force_atomic && stage_sample_count_fits &&
        (force_staged ||
         (stage_sample_count >= staged_min_samples &&
          stage_sample_count >= cell_count * staged_min_per_cell));
    auto fopts = ray_o.options();
    auto iopts = scene.global_faces.options();
    at::Tensor power = at::zeros({cell_count}, fopts);
    at::Tensor field_x_re = at::zeros({cell_count}, fopts);
    at::Tensor field_x_im = at::zeros({cell_count}, fopts);
    at::Tensor field_y_re = at::zeros({cell_count}, fopts);
    at::Tensor field_y_im = at::zeros({cell_count}, fopts);
    at::Tensor field_z_re = at::zeros({cell_count}, fopts);
    at::Tensor field_z_im = at::zeros({cell_count}, fopts);
    at::Tensor reflection_count = at::zeros({1}, iopts);
    const int64_t wedge_capacity64 = collect_wedges ? wedge_capacity : 0;
    const int32_t wedge_capacity_i = checked_i32(wedge_capacity64, "wedge_capacity");
    const int32_t wedge_sample_stride_i = checked_i32(wedge_sample_stride, "wedge_sample_stride");
    at::Tensor wedge_count = at::zeros({1}, iopts);
    at::Tensor wedge_ray_index = at::full({wedge_capacity64}, -1, iopts);
    at::Tensor wedge_prim_id = at::full({wedge_capacity64}, -1, iopts);
    at::Tensor wedge_bounce_depth = at::full({wedge_capacity64}, -1, iopts);
    at::Tensor wedge_source_power = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_hit_x = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_hit_y = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_hit_z = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_normal_x = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_normal_y = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_normal_z = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_dir_x = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_dir_y = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_dir_z = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_source_x = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_source_y = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_source_z = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_initial_dir_x = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_initial_dir_y = at::zeros({wedge_capacity64}, fopts);
    at::Tensor wedge_initial_dir_z = at::zeros({wedge_capacity64}, fopts);
    if (ray_count == 0) {
        return {
            power.reshape({grid_resolution1, grid_resolution0}),
            field_x_re.reshape({grid_resolution1, grid_resolution0}),
            field_x_im.reshape({grid_resolution1, grid_resolution0}),
            field_y_re.reshape({grid_resolution1, grid_resolution0}),
            field_y_im.reshape({grid_resolution1, grid_resolution0}),
            field_z_re.reshape({grid_resolution1, grid_resolution0}),
            field_z_im.reshape({grid_resolution1, grid_resolution0}),
            reflection_count,
            wedge_count,
            wedge_ray_index,
            stack_vec3(wedge_hit_x, wedge_hit_y, wedge_hit_z),
            stack_vec3(wedge_normal_x, wedge_normal_y, wedge_normal_z),
            wedge_prim_id,
            stack_vec3(wedge_dir_x, wedge_dir_y, wedge_dir_z),
            stack_vec3(wedge_source_x, wedge_source_y, wedge_source_z),
            wedge_source_power,
            stack_vec3(wedge_initial_dir_x, wedge_initial_dir_y, wedge_initial_dir_z),
            wedge_bounce_depth};
    }

    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA ray_o_soa = split_vec3(ray_o);
    Vec3SoA ray_d_soa = split_vec3(ray_d);
    Vec3SoA tx_soa = split_vec3(tx);
    Vec3SoA tx_pol_soa = split_vec3(tx_pol);
    at::Tensor ray_tmax_contig =
        !ray_tmax.defined() || ray_tmax.numel() == 0 ? ray_tmax : ray_tmax.contiguous();
    at::Tensor active_contig = active == nullptr ? at::Tensor() : active->contiguous();
    at::Tensor stage_cell = staged_accum
        ? at::full({stage_sample_count}, -1, iopts)
        : at::Tensor();
    at::Tensor stage_value = staged_accum
        ? at::zeros({stage_sample_count, 8}, fopts)
        : at::Tensor();

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
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.n_triangles = tri.n_triangles;
    params.ray_ox = ray_o_soa.x.data_ptr<float>();
    params.ray_oy = ray_o_soa.y.data_ptr<float>();
    params.ray_oz = ray_o_soa.z.data_ptr<float>();
    params.ray_dx = ray_d_soa.x.data_ptr<float>();
    params.ray_dy = ray_d_soa.y.data_ptr<float>();
    params.ray_dz = ray_d_soa.z.data_ptr<float>();
    params.ray_tmax = !ray_tmax_contig.defined() || ray_tmax_contig.numel() == 0
        ? nullptr
        : ray_tmax_contig.data_ptr<float>();
    params.active_mask = optional_mask_ptr(active_contig);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.include_los = include_los ? 1 : 0;
    params.tx_x = tx_soa.x.data_ptr<float>();
    params.tx_y = tx_soa.y.data_ptr<float>();
    params.tx_z = tx_soa.z.data_ptr<float>();
    params.tx_pol_x = tx_pol_soa.x.data_ptr<float>();
    params.tx_pol_y = tx_pol_soa.y.data_ptr<float>();
    params.tx_pol_z = tx_pol_soa.z.data_ptr<float>();
    params.max_bounces = max_bounces_i;
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.solid_angle_per_ray = static_cast<float>(solid_angle_per_ray);
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
    params.material_count = static_cast<int32_t>(material_count);
    params.collect_wedges = collect_wedges ? 1 : 0;
    params.collect_wedge_prefixes = collect_wedge_prefixes ? 1 : 0;
    params.wedge_capacity = wedge_capacity_i;
    params.wedge_sample_stride = wedge_sample_stride_i;
    params.out_reflection_power = power.data_ptr<float>();
    params.out_field_x_re = field_x_re.data_ptr<float>();
    params.out_field_x_im = field_x_im.data_ptr<float>();
    params.out_field_y_re = field_y_re.data_ptr<float>();
    params.out_field_y_im = field_y_im.data_ptr<float>();
    params.out_field_z_re = field_z_re.data_ptr<float>();
    params.out_field_z_im = field_z_im.data_ptr<float>();
    params.out_reflection_count = reflection_count.data_ptr<int>();
    params.stage_cell = staged_accum ? stage_cell.data_ptr<int>() : nullptr;
    params.stage_value = staged_accum
        ? reinterpret_cast<ReflAccumStagedValue *>(stage_value.data_ptr<float>())
        : nullptr;
    params.out_wedge_count = collect_wedges ? wedge_count.data_ptr<int>() : nullptr;
    params.out_wedge_ray_index = collect_wedges ? wedge_ray_index.data_ptr<int>() : nullptr;
    params.out_wedge_hit_x = collect_wedges ? wedge_hit_x.data_ptr<float>() : nullptr;
    params.out_wedge_hit_y = collect_wedges ? wedge_hit_y.data_ptr<float>() : nullptr;
    params.out_wedge_hit_z = collect_wedges ? wedge_hit_z.data_ptr<float>() : nullptr;
    params.out_wedge_normal_x = collect_wedges ? wedge_normal_x.data_ptr<float>() : nullptr;
    params.out_wedge_normal_y = collect_wedges ? wedge_normal_y.data_ptr<float>() : nullptr;
    params.out_wedge_normal_z = collect_wedges ? wedge_normal_z.data_ptr<float>() : nullptr;
    params.out_wedge_prim_id = collect_wedges ? wedge_prim_id.data_ptr<int>() : nullptr;
    params.out_wedge_dir_x = collect_wedges ? wedge_dir_x.data_ptr<float>() : nullptr;
    params.out_wedge_dir_y = collect_wedges ? wedge_dir_y.data_ptr<float>() : nullptr;
    params.out_wedge_dir_z = collect_wedges ? wedge_dir_z.data_ptr<float>() : nullptr;
    params.out_wedge_source_x = collect_wedges ? wedge_source_x.data_ptr<float>() : nullptr;
    params.out_wedge_source_y = collect_wedges ? wedge_source_y.data_ptr<float>() : nullptr;
    params.out_wedge_source_z = collect_wedges ? wedge_source_z.data_ptr<float>() : nullptr;
    params.out_wedge_source_power = collect_wedges ? wedge_source_power.data_ptr<float>() : nullptr;
    params.out_wedge_initial_dir_x = collect_wedges ? wedge_initial_dir_x.data_ptr<float>() : nullptr;
    params.out_wedge_initial_dir_y = collect_wedges ? wedge_initial_dir_y.data_ptr<float>() : nullptr;
    params.out_wedge_initial_dir_z = collect_wedges ? wedge_initial_dir_z.data_ptr<float>() : nullptr;
    params.out_wedge_bounce_depth = collect_wedges ? wedge_bounce_depth.data_ptr<int>() : nullptr;

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (scene.trace_backend == TraceBackend::Cuda) {
        launch_reflection_accumulation_cuda(scene, params, static_cast<int>(ray_count));
    } else {
        optix_pipeline_for_scene(scene, reflection_accumulation_pipeline_config())
            ->launch(0, params, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    }
    if (staged_accum) {
        reduce_refl_accum_staged_cuda(
            stage_sample_count,
            stage_cell,
            stage_value,
            power,
            field_x_re,
            field_x_im,
            field_y_re,
            field_y_im,
            field_z_re,
            field_z_im,
            reflection_count);
    }
    return {
        power.reshape({grid_resolution1, grid_resolution0}),
        field_x_re.reshape({grid_resolution1, grid_resolution0}),
        field_x_im.reshape({grid_resolution1, grid_resolution0}),
        field_y_re.reshape({grid_resolution1, grid_resolution0}),
        field_y_im.reshape({grid_resolution1, grid_resolution0}),
        field_z_re.reshape({grid_resolution1, grid_resolution0}),
        field_z_im.reshape({grid_resolution1, grid_resolution0}),
        reflection_count,
        wedge_count,
        wedge_ray_index,
        stack_vec3(wedge_hit_x, wedge_hit_y, wedge_hit_z),
        stack_vec3(wedge_normal_x, wedge_normal_y, wedge_normal_z),
        wedge_prim_id,
        stack_vec3(wedge_dir_x, wedge_dir_y, wedge_dir_z),
        stack_vec3(wedge_source_x, wedge_source_y, wedge_source_z),
        wedge_source_power,
        stack_vec3(wedge_initial_dir_x, wedge_initial_dir_y, wedge_initial_dir_z),
        wedge_bounce_depth};
}

py::tuple reflection_accumulation_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tx,
    at::Tensor tx_pol,
    at::Tensor material_eta_r,
    at::Tensor material_sigma,
    at::Tensor material_mu_r,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength,
    double solid_angle_per_ray,
    bool collect_wedges,
    bool collect_wedge_prefixes,
    int64_t wedge_capacity,
    int64_t wedge_sample_stride,
    int64_t accumulation_strategy,
    int64_t compact_min_samples,
    int64_t staged_min_samples_per_cell,
    int64_t procedural_sample_count,
    bool include_los) {
    ReflectionAccumulationNativeOutputs out = reflection_accumulation_forward_native_impl(
        get_scene(scene_handle),
        std::move(ray_o),
        std::move(ray_d),
        std::move(ray_tmax),
        &active,
        std::move(tx),
        std::move(tx_pol),
        std::move(material_eta_r),
        std::move(material_sigma),
        std::move(material_mu_r),
        std::move(material_gain),
        std::move(material_valid),
        max_bounces,
        grid_axis,
        grid_position,
        grid_coord0_min,
        grid_coord0_max,
        grid_coord1_min,
        grid_coord1_max,
        grid_resolution0,
        grid_resolution1,
        wavelength,
        solid_angle_per_ray,
        collect_wedges,
        collect_wedge_prefixes,
        wedge_capacity,
        wedge_sample_stride,
        accumulation_strategy,
        compact_min_samples,
        staged_min_samples_per_cell,
        procedural_sample_count,
        include_los);
    return py::make_tuple(
        out.power,
        out.field_x_re,
        out.field_x_im,
        out.field_y_re,
        out.field_y_im,
        out.field_z_re,
        out.field_z_im,
        out.reflection_count,
        out.wedge_count,
        out.wedge_ray_index,
        out.wedge_hit,
        out.wedge_normal,
        out.wedge_prim_id,
        out.wedge_direction,
        out.wedge_source,
        out.wedge_source_power,
        out.wedge_initial_direction,
        out.wedge_bounce_depth);
}



ReflEpcPathsBackwardOutputs integration_reflection_epc_paths_backward_impl(
    SceneCache &scene,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &sequence,
    const at::Tensor &plane_points,
    const at::Tensor &plane_normals,
    const at::Tensor &valid,
    const at::Tensor &bounce_count,
    const at::Tensor *grad_points,
    const at::Tensor *grad_normals,
    const at::Tensor *grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver) {
    auto maybe = [](const at::Tensor *tensor) -> const at::Tensor * {
        if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0)
            return nullptr;
        return tensor;
    };
    const int64_t ray_count = require_epc_paths_frozen_winner(
        source, receiver, sequence, plane_points, plane_normals, valid, bounce_count);
    const int64_t bounce_width = sequence.size(1);
    require_optional_chain_grad_vec3(maybe(grad_points), ray_count, bounce_width, "grad_points");
    require_optional_chain_grad_vec3(maybe(grad_normals), ray_count, bounce_width, "grad_normals");
    require_optional_grad_vec(maybe(grad_path_length), ray_count, 0, "grad_path_length");
    require_scene_device(scene, source, "source");
    require_scene_device(scene, receiver, "receiver");
    require_scene_device(scene, sequence, "sequence");
    require_scene_device(scene, plane_points, "plane_points");
    require_scene_device(scene, plane_normals, "plane_normals");
    require_scene_device(scene, valid, "valid");
    require_scene_device(scene, bounce_count, "bounce_count");
    require_scene_device(scene, grad_points, "grad_points");
    require_scene_device(scene, grad_normals, "grad_normals");
    require_scene_device(scene, grad_path_length, "grad_path_length");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    return reflection_epc_paths_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        source,
        receiver,
        sequence,
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        maybe(grad_points),
        maybe(grad_normals),
        maybe(grad_path_length),
        need_grad_vertices,
        need_grad_source,
        need_grad_receiver);
}

ReflEpcPathsJvpOutputs integration_reflection_epc_paths_jvp_impl(
    SceneCache &scene,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &sequence,
    const at::Tensor &plane_points,
    const at::Tensor &plane_normals,
    const at::Tensor &valid,
    const at::Tensor &bounce_count,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_source,
    const at::Tensor *tangent_receiver) {
    auto maybe = [](const at::Tensor *tensor) -> const at::Tensor * {
        if (tensor == nullptr || !tensor->defined() || tensor->numel() == 0)
            return nullptr;
        return tensor;
    };
    const int64_t ray_count = require_epc_paths_frozen_winner(
        source, receiver, sequence, plane_points, plane_normals, valid, bounce_count);
    require_optional_tangent_vertices(maybe(tangent_vertices), scene.global_vertices, "tangent_vertices");
    require_optional_grad_vec(maybe(tangent_source), ray_count, 3, "tangent_source");
    require_optional_grad_vec(maybe(tangent_receiver), ray_count, 3, "tangent_receiver");
    require_scene_device(scene, source, "source");
    require_scene_device(scene, receiver, "receiver");
    require_scene_device(scene, sequence, "sequence");
    require_scene_device(scene, plane_points, "plane_points");
    require_scene_device(scene, plane_normals, "plane_normals");
    require_scene_device(scene, valid, "valid");
    require_scene_device(scene, bounce_count, "bounce_count");
    require_scene_device(scene, tangent_vertices, "tangent_vertices");
    require_scene_device(scene, tangent_source, "tangent_source");
    require_scene_device(scene, tangent_receiver, "tangent_receiver");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    return reflection_epc_paths_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        source,
        receiver,
        sequence,
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        maybe(tangent_vertices),
        maybe(tangent_source),
        maybe(tangent_receiver));
}

at::Tensor integration_scene_face_normals_backward_impl(
    SceneCache &scene,
    const at::Tensor &grad_face_normals) {
    require_cuda(grad_face_normals, "grad_face_normals");
    require_dtype(grad_face_normals, at::kFloat, "grad_face_normals");
    require_rank(grad_face_normals, 2, "grad_face_normals");
    require_last_dim(grad_face_normals, 3, "grad_face_normals");
    require_scene_device(scene, grad_face_normals, "grad_face_normals");
    if (grad_face_normals.size(0) != scene.global_faces.size(0))
        throw std::runtime_error("grad_face_normals must match the scene global face table.");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    return scene_face_normals_backward_cuda(
        scene.global_vertices, scene.global_faces, grad_face_normals);
}

at::Tensor integration_scene_face_normals_jvp_impl(
    SceneCache &scene,
    const at::Tensor &tangent_vertices) {
    require_optional_tangent_vertices(
        &tangent_vertices, scene.global_vertices, "tangent_vertices");
    require_scene_device(scene, tangent_vertices, "tangent_vertices");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    return scene_face_normals_jvp_cuda(
        scene.global_vertices, scene.global_faces, tangent_vertices);
}

} // namespace rayd::torch_backend

#include "../bindings/integration_internal.h"

namespace rayd::torch {

namespace {

const at::Tensor *present_optional(const std::optional<at::Tensor> &value) {
    if (!value.has_value() || !value->defined())
        return nullptr;
    return &*value;
}

template <std::size_t N>
void require_output_count(const std::vector<at::Tensor> &values, const char *operation) {
    if (values.size() != N)
        throw std::runtime_error(std::string(operation) + " returned an unexpected output count");
}

} // namespace


ReflectionTraceResult trace_reflections_forward(
    const SceneResource &scene,
    const ReflectionTraceRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    at::Tensor ray_tmax = present_optional(request.rays.ray_tmax) == nullptr
        ? at::Tensor()
        : *request.rays.ray_tmax;
    std::vector<at::Tensor> values = torch_backend::trace_reflections_forward_native_impl(
        cache,
        request.rays.ray_o,
        request.rays.ray_d,
        ray_tmax,
        present_optional(request.rays.active),
        request.max_bounces,
        false,
        false);
    require_output_count<3>(values, "rayd::torch::trace_reflections_forward");
    return {values[0], values[1], values[2]};
}

ReflectionTraceTapeResult trace_reflections_forward_tape(
    const SceneResource &scene,
    const ReflectionTraceRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    at::Tensor ray_tmax = present_optional(request.rays.ray_tmax) == nullptr
        ? at::Tensor()
        : *request.rays.ray_tmax;
    std::vector<at::Tensor> values = torch_backend::trace_reflections_forward_native_impl(
        cache,
        request.rays.ray_o,
        request.rays.ray_d,
        ray_tmax,
        present_optional(request.rays.active),
        request.max_bounces,
        true,
        true);
    require_output_count<9>(values, "rayd::torch::trace_reflections_forward_tape");
    return {
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
    };
}


ReflectionTraceBackwardResult trace_reflections_backward(
    const SceneResource &scene,
    const ReflectionTraceBackwardRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    auto ptr = [](const std::optional<at::Tensor> &value) -> const at::Tensor * {
        if (!value.has_value() || !value->defined())
            return nullptr;
        return &*value;
    };
    const at::Tensor *ray_tmax = present_optional(request.rays.ray_tmax);
    auto out = torch_backend::integration_trace_reflections_backward_impl(
        cache,
        request.rays.ray_o,
        request.rays.ray_d,
        ray_tmax,
        ptr(request.rays.active),
        request.tape_prim_id,
        request.tape_barycentric,
        request.tape_hit_points,
        request.tape_normals,
        request.image_sources,
        ptr(request.grad_t),
        ptr(request.grad_image_sources));
    return {out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax};
}

ReflectionTraceJvpResult trace_reflections_jvp(
    const SceneResource &scene,
    const ReflectionTraceJvpRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    auto ptr = [](const std::optional<at::Tensor> &value) -> const at::Tensor * {
        if (!value.has_value() || !value->defined())
            return nullptr;
        return &*value;
    };
    auto out = torch_backend::integration_trace_reflections_jvp_impl(
        cache,
        request.ray_o,
        request.ray_d,
        ptr(request.active),
        request.tape_prim_id,
        request.tape_barycentric,
        request.tape_hit_points,
        request.tape_normals,
        ptr(request.tangent_vertices),
        ptr(request.tangent_ray_o),
        ptr(request.tangent_ray_d),
        request.image_sources);
    return {out.tangent_t, out.tangent_image_sources};
}

ReflectionAccumulationResult reflection_accumulation_forward(
    const SceneResource &scene,
    const ReflectionAccumulationConfig &config) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    at::Tensor ray_tmax = present_optional(config.rays.ray_tmax) == nullptr
        ? at::Tensor()
        : *config.rays.ray_tmax;
    auto out = torch_backend::reflection_accumulation_forward_native_impl(
        cache,
        config.rays.ray_o,
        config.rays.ray_d,
        ray_tmax,
        present_optional(config.rays.active),
        config.tx,
        config.tx_pol,
        config.material.eta_r,
        config.material.sigma,
        config.material.mu_r,
        config.material.gain,
        config.material.valid,
        config.max_bounces,
        config.grid.axis,
        config.grid.position,
        config.grid.coord0_min,
        config.grid.coord0_max,
        config.grid.coord1_min,
        config.grid.coord1_max,
        config.grid.resolution0,
        config.grid.resolution1,
        config.wavelength,
        config.solid_angle_per_ray,
        config.collect_wedges,
        config.collect_wedge_prefixes,
        config.wedge_capacity,
        config.wedge_sample_stride,
        config.accumulation_strategy,
        config.compact_min_samples,
        config.staged_min_samples_per_cell,
        config.procedural_sample_count,
        config.include_los);
    return {
        out.power,
        out.field_x_re,
        out.field_x_im,
        out.field_y_re,
        out.field_y_im,
        out.field_z_re,
        out.field_z_im,
        out.reflection_count,
        out.wedge_count,
        out.wedge_ray_index,
        out.wedge_hit,
        out.wedge_normal,
        out.wedge_prim_id,
        out.wedge_direction,
        out.wedge_source,
        out.wedge_source_power,
        out.wedge_initial_direction,
        out.wedge_bounce_depth,
    };
}

ReflectionEpcResult reflection_epc_paths_forward(
    const SceneResource &scene,
    const ReflectionEpcRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    std::vector<at::Tensor> values = torch_backend::reflection_epc_paths_forward_native_impl(
        cache,
        request.source,
        request.receiver,
        present_optional(request.active),
        request.expected_prim_ids,
        request.direct_plane_points,
        request.direct_plane_normals,
        request.surface_group_id,
        request.surface_group_size,
        request.surface_group_members,
        request.max_bounces,
        request.visibility_ignore_mode,
        request.plane_tolerance);
    require_output_count<6>(values, "rayd::torch::reflection_epc_paths_forward");
    return {values[0], values[1], values[2], values[3], values[4], values[5]};
}

at::Tensor scene_face_normals_backward(
    const SceneResource &scene,
    const at::Tensor &grad_face_normals) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    return torch_backend::integration_scene_face_normals_backward_impl(
        cache, grad_face_normals);
}

at::Tensor scene_face_normals_jvp(
    const SceneResource &scene,
    const at::Tensor &tangent_vertices) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    return torch_backend::integration_scene_face_normals_jvp_impl(
        cache, tangent_vertices);
}

ReflectionEpcBackwardResult reflection_epc_paths_backward(
    const SceneResource &scene,
    const ReflectionEpcBackwardRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    auto ptr = [](const std::optional<at::Tensor> &value) -> const at::Tensor * {
        if (!value.has_value() || !value->defined())
            return nullptr;
        return &*value;
    };
    auto out = torch_backend::integration_reflection_epc_paths_backward_impl(
        cache,
        request.source,
        request.receiver,
        request.sequence,
        request.plane_points,
        request.plane_normals,
        request.valid,
        request.bounce_count,
        ptr(request.grad_points),
        ptr(request.grad_normals),
        ptr(request.grad_path_length),
        request.need_grad_vertices,
        request.need_grad_source,
        request.need_grad_receiver);
    return {out.grad_vertices, out.grad_source, out.grad_receiver};
}

ReflectionEpcJvpResult reflection_epc_paths_jvp(
    const SceneResource &scene,
    const ReflectionEpcJvpRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    auto ptr = [](const std::optional<at::Tensor> &value) -> const at::Tensor * {
        if (!value.has_value() || !value->defined())
            return nullptr;
        return &*value;
    };
    auto out = torch_backend::integration_reflection_epc_paths_jvp_impl(
        cache,
        request.source,
        request.receiver,
        request.sequence,
        request.plane_points,
        request.plane_normals,
        request.valid,
        request.bounce_count,
        ptr(request.tangent_vertices),
        ptr(request.tangent_source),
        ptr(request.tangent_receiver));
    return {out.tangent_points, out.tangent_normals, out.tangent_path_length};
}

} // namespace rayd::torch


// ---- merged from src/reflection/pipeline_part.cpp ----

#include <src/reflection/pipeline.h>

#include <src/reflection/accum_params.h>
#include <src/reflection/epc_params.h>
#include <src/reflection/trace_params.h>
#include <rayd/rt/optix_pipeline_contracts.h>
#include <rayd/reflection/accumulation_torch_ptx.h>
#include <rayd/reflection/epc_torch_ptx.h>
#include <rayd/reflection/trace_torch_ptx.h>

namespace rayd::torch_backend {

OptixPipelineConfig refl_trace_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_reflection_trace_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_reflection_trace_optix_ptx);
    config.raygen_entries = {"__raygen__reflection_trace"};
    config.miss_entry = "__miss__reflection";
    config.closesthit_entry = "__closesthit__reflection";
    config.num_payload_values = shared::optix::TriangleHitPayloadCount;
    config.params_size = sizeof(ReflectionTraceParams);
    return config;
}

OptixPipelineConfig refl_epc_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_reflection_epc_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_reflection_epc_optix_ptx);
    config.raygen_entries = {
        "__raygen__reflection_epc",
        "__raygen__reflection_epc_direct",
        "__raygen__reflection_epc_direct_primary",
    };
    config.miss_entry = "__miss__reflection_epc";
    config.closesthit_entry = "__closesthit__reflection_epc";
    config.anyhit_entry = "__anyhit__reflection_epc";
    config.num_payload_values = shared::optix::TriangleHitPayloadCount;
    config.params_size = sizeof(ReflEpcParams);
    return config;
}

OptixPipelineConfig refl_accum_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_reflection_accumulation_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_reflection_accumulation_optix_ptx);
    config.raygen_entries = {"__raygen__reflection_accumulation"};
    config.miss_entry = "__miss__reflection_accumulation";
    config.closesthit_entry = "__closesthit__reflection_accumulation";
    config.num_payload_values = shared::optix::TriangleHitPayloadCount;
    config.params_size = sizeof(AccumParams);
    return config;
}

} // namespace rayd::torch_backend
