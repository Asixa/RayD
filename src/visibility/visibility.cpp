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
#include <src/visibility/axial_edge_visibility_params.h>
#include <src/reflection/dedup.h>
#include <src/reflection/epc_field.h>
#include <src/reflection/epc_params.h>
#include <src/reflection/trace_params.h>
#include <src/visibility/visibility.h>
#include <src/visibility/visibility_params.h>
#include <rayd/visibility.h>
#include <rayd/detail/rt/optix_pipeline_contracts.h>
#include <rayd/visibility/segment_torch_ptx.h>
#include <rayd/visibility/axial_edge_torch_ptx.h>
#include <src/scene/cache.h>
#include <src/bindings/tensor_contract.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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

at::Tensor visibility_active_mask(
    const c10::optional<at::Tensor> &active,
    int64_t count,
    const at::TensorOptions &options,
    const char *name) {
    if (!active.has_value() || !active->defined() || active->numel() == 0)
        return at::ones({count}, options.dtype(at::kBool));
    require_mask(*active, name);
    if (active->size(0) != count)
        throw std::runtime_error(std::string(name) + " must match the batch size.");
    return active->contiguous();
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

at::Tensor prepare_visibility_ignore_ids(
    const c10::optional<at::Tensor> &ignore_ids,
    int64_t row_count,
    int max_rank,
    const char *name,
    int32_t &ignore_k) {
    ignore_k = 0;
    if (!ignore_ids.has_value() || !ignore_ids->defined())
        return at::Tensor();
    require_cuda(*ignore_ids, name);
    require_dtype(*ignore_ids, at::kInt, name);
    if (ignore_ids->dim() < 1 || ignore_ids->dim() > max_rank)
        throw std::runtime_error(std::string(name) + " has the wrong rank.");
    if (ignore_ids->numel() == 0)
        return at::Tensor();
    if (row_count <= 0 || ignore_ids->numel() % row_count != 0)
        throw std::runtime_error(
            std::string(name) + " size must be a multiple of its visibility row count.");
    ignore_k = checked_i32(ignore_ids->numel() / row_count, "ignore_k");
    return ignore_ids->contiguous().reshape({-1});
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

void launch_segment_visibility_backend(
    SceneCache &scene,
    const SegmentVisibilityParams &params,
    int variant,
    unsigned int lane_count,
    cudaStream_t stream) {
    if (scene.trace_backend == TraceBackend::Cuda) {
        launch_segment_visibility_cuda(
            scene, params, static_cast<CudaVisibilityVariant>(variant),
            static_cast<int>(lane_count));
        return;
    }
    optix_pipeline_for_scene(scene, segment_visibility_pipeline_config())
        ->launch(variant, params, lane_count, stream);
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

struct SegmentVisibilityNativeOutputs {
    at::Tensor visible;
    at::Tensor blocker_prim;
    at::Tensor tape_t;
};

SegmentVisibilityNativeOutputs visibility_forward_native_impl(
    SceneCache &scene,
    at::Tensor start,
    at::Tensor end,
    const at::Tensor *active_ptr) {
    require_vec3f(start, "start");
    require_vec3f(end, "end");
    require_same_batch(start, end, "visibility");
    at::Tensor active = optional_active_from_tensor(active_ptr, start.size(0), "active");

    require_scene_device(scene, start, "start");
    require_scene_device(scene, end, "end");
    require_scene_device(scene, active, "active");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    const int64_t ray_count = start.size(0);
    at::Tensor visible = at::empty({ray_count}, start.options().dtype(at::kBool));
    at::Tensor blocker_prim = at::empty({ray_count}, scene.global_faces.options());
    at::Tensor tape_t = at::empty({ray_count}, start.options());
    if (ray_count == 0)
        return {visible, blocker_prim, tape_t};

    SegmentVisibilityParams params = {};
    params.handle = scene.triangle_ias.traversable;
    params.face_offsets = scene.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.start_aos = start.data_ptr<float>();
    params.end_aos = end.data_ptr<float>();
    params.active_mask = optional_mask_ptr(active);
    params.n_rays = static_cast<int32_t>(ray_count);
    params.out_visible = mutable_mask_ptr(visible);
    params.out_first_blocked_prim = blocker_prim.data_ptr<int>();
    params.out_t = tape_t.data_ptr<float>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    launch_segment_visibility_backend(
        scene, params, 0, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    return {visible, blocker_prim, tape_t};
}

float float_from_bits(std::uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

py::tuple visibility_forward_op(
    int64_t scene_handle,
    at::Tensor start,
    at::Tensor end,
    py::object active_obj) {
    at::Tensor active_storage;
    const at::Tensor *active = optional_tensor(active_obj, active_storage);
    SegmentVisibilityNativeOutputs out =
        visibility_forward_native_impl(get_scene(scene_handle), start, end, active);
    return py::make_tuple(out.visible, out.blocker_prim, out.tape_t);
}


std::vector<at::Tensor> visible_pair_forward_impl(
    int64_t scene_handle,
    at::Tensor start,
    at::Tensor end_a,
    at::Tensor end_b,
    c10::optional<at::Tensor> ignore_prim_ids,
    c10::optional<at::Tensor> active) {
    require_vec3f(start, "start");
    require_vec3f(end_a, "end_a");
    require_vec3f(end_b, "end_b");
    require_same_batch(start, end_a, "visible_pair");
    require_same_batch(start, end_b, "visible_pair");

    SceneCache &scene = get_scene(scene_handle);
    require_scene_device(scene, start, "start");
    require_scene_device(scene, end_a, "end_a");
    require_scene_device(scene, end_b, "end_b");
    require_scene_device(scene, ignore_prim_ids, "ignore_prim_ids");
    require_scene_device(scene, active, "active");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    const int64_t ray_count = start.size(0);
    at::Tensor visible_a = at::empty({ray_count}, start.options().dtype(at::kBool));
    at::Tensor visible_b = at::empty({ray_count}, start.options().dtype(at::kBool));
    if (ray_count == 0)
        return {visible_a, visible_b};

    at::Tensor active_mask = visibility_active_mask(
        active, ray_count, start.options(), "active");
    active_mask = (active_mask & finite_vec3_rows(start) &
                   finite_vec3_rows(end_a) & finite_vec3_rows(end_b)).contiguous();
    int32_t ignore_k = 0;
    at::Tensor ignore_ids = prepare_visibility_ignore_ids(
        ignore_prim_ids, ray_count, 2, "ignore_prim_ids", ignore_k);

    SegmentVisibilityParams params = {};
    params.handle = scene.triangle_ias.traversable;
    params.face_offsets = scene.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.start_aos = start.data_ptr<float>();
    params.end_aos = end_a.data_ptr<float>();
    params.end_b_aos = end_b.data_ptr<float>();
    params.ignore_prim_ids = ignore_k > 0 ? ignore_ids.data_ptr<int>() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = mask_ptr(active_mask);
    params.n_rays = checked_i32(ray_count, "ray_count");
    params.out_visible = mutable_mask_ptr(visible_a);
    params.out_visible_b = mutable_mask_ptr(visible_b);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    launch_segment_visibility_backend(
        scene, params, 1, static_cast<unsigned int>(ray_count), torch_ctx.stream);
    return {visible_a, visible_b};
}

std::vector<at::Tensor> visible_edge_forward_impl(
    int64_t scene_handle,
    at::Tensor source,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    std::vector<double> sample_fractions,
    c10::optional<at::Tensor> active) {
    if (sample_fractions.empty())
        throw std::runtime_error("visible_edge sample_fractions must not be empty.");
    if (sample_fractions.size() > static_cast<size_t>(SegmentVisibilityMaxSamples))
        throw std::runtime_error("visible_edge supports at most 16 sample fractions.");
    if (!std::all_of(sample_fractions.begin(), sample_fractions.end(), [](double value) {
            return std::isfinite(value);
        }))
        throw std::runtime_error("visible_edge sample_fractions must be finite.");
    require_vec3f(source, "source");
    require_vec3f(edge_position, "edge_position");
    require_vec3f(edge_direction, "edge_direction");
    require_flat_f32(edge_t_min, "edge_t_min");
    require_flat_f32(edge_t_max, "edge_t_max");
    require_same_batch(source, edge_position, "visible_edge");
    require_same_batch(source, edge_direction, "visible_edge");
    if (edge_t_min.size(0) != source.size(0) ||
        edge_t_max.size(0) != source.size(0))
        throw std::runtime_error("visible_edge inputs must have the same batch size.");

    SceneCache &scene = get_scene(scene_handle);
    require_scene_device(scene, source, "source");
    require_scene_device(scene, edge_position, "edge_position");
    require_scene_device(scene, edge_direction, "edge_direction");
    require_scene_device(scene, edge_t_min, "edge_t_min");
    require_scene_device(scene, edge_t_max, "edge_t_max");
    require_scene_device(scene, active, "active");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    const int64_t state_count = source.size(0);
    at::Tensor any_visible =
        at::empty({state_count}, source.options().dtype(at::kBool));
    if (state_count == 0)
        return {any_visible};

    at::Tensor active_mask = visibility_active_mask(
        active, state_count, source.options(), "active");
    active_mask = (active_mask & finite_vec3_rows(source) &
                   finite_vec3_rows(edge_position) &
                   finite_vec3_rows(edge_direction) & at::isfinite(edge_t_min) &
                   at::isfinite(edge_t_max)).contiguous();
    Vec3SoA edge_direction_soa = split_vec3(edge_direction);

    SegmentVisibilityParams params = {};
    params.handle = scene.triangle_ias.traversable;
    params.face_offsets = scene.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.start_aos = source.data_ptr<float>();
    params.end_aos = edge_position.data_ptr<float>();
    params.edge_dir_x = edge_direction_soa.x.data_ptr<float>();
    params.edge_dir_y = edge_direction_soa.y.data_ptr<float>();
    params.edge_dir_z = edge_direction_soa.z.data_ptr<float>();
    params.edge_t_min = edge_t_min.data_ptr<float>();
    params.edge_t_max = edge_t_max.data_ptr<float>();
    params.active_mask = mask_ptr(active_mask);
    params.n_rays = checked_i32(state_count, "state_count");
    params.sample_count = checked_i32(sample_fractions.size(), "sample_count");
    for (size_t i = 0; i < sample_fractions.size(); ++i)
        params.sample_fractions[i] = static_cast<float>(sample_fractions[i]);
    params.out_visible = mutable_mask_ptr(any_visible);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    launch_segment_visibility_backend(
        scene, params, 2, static_cast<unsigned int>(state_count), torch_ctx.stream);
    return {any_visible};
}

std::vector<at::Tensor> visible_chain_forward_impl(
    int64_t scene_handle,
    at::Tensor points,
    at::Tensor chain_length,
    c10::optional<at::Tensor> ignore_prim_per_segment,
    c10::optional<at::Tensor> active) {
    require_cuda(points, "points");
    require_contiguous(points, "points");
    require_dtype(points, at::kFloat, "points");
    require_rank(points, 3, "points");
    require_last_dim(points, 3, "points");
    require_flat_i32(chain_length, "chain_length");
    if (points.size(0) != chain_length.size(0))
        throw std::runtime_error(
            "visible_chain points and chain_length must have the same batch size.");

    SceneCache &scene = get_scene(scene_handle);
    require_scene_device(scene, points, "points");
    require_scene_device(scene, chain_length, "chain_length");
    require_scene_device(scene, ignore_prim_per_segment, "ignore_prim_per_segment");
    require_scene_device(scene, active, "active");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    const int64_t chain_count = chain_length.size(0);
    at::Tensor all_visible =
        at::empty({chain_count}, points.options().dtype(at::kBool));
    at::Tensor first_blocked_segment =
        at::empty({chain_count}, chain_length.options());
    at::Tensor first_blocked_prim =
        at::empty({chain_count}, chain_length.options());
    if (chain_count == 0)
        return {all_visible, first_blocked_segment, first_blocked_prim};

    const int64_t max_points = points.size(1);
    if (max_points < 2)
        throw std::runtime_error("visible_chain requires at least two points per chain.");
    const int64_t max_segments = max_points - 1;
    at::Tensor active_mask = visibility_active_mask(
        active, chain_count, points.options(), "active");
    active_mask = (
        active_mask & chain_length.ge(0) & at::isfinite(points).all(2).all(1)
    ).contiguous();

    int32_t ignore_k = 0;
    at::Tensor ignore_ids = prepare_visibility_ignore_ids(
        ignore_prim_per_segment,
        chain_count * max_segments,
        3,
        "ignore_prim_per_segment",
        ignore_k);
    at::Tensor flat_points = points.reshape({chain_count * max_points, 3});
    Vec3SoA point_soa = split_vec3(flat_points);

    SegmentVisibilityParams params = {};
    params.handle = scene.triangle_ias.traversable;
    params.face_offsets = scene.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.chain_point_x = point_soa.x.data_ptr<float>();
    params.chain_point_y = point_soa.y.data_ptr<float>();
    params.chain_point_z = point_soa.z.data_ptr<float>();
    params.chain_length = chain_length.data_ptr<int>();
    params.max_points = checked_i32(max_points, "max_points");
    params.max_segments = checked_i32(max_segments, "max_segments");
    params.ignore_prim_ids = ignore_k > 0 ? ignore_ids.data_ptr<int>() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = mask_ptr(active_mask);
    params.n_rays = checked_i32(chain_count, "chain_count");
    params.out_visible = mutable_mask_ptr(all_visible);
    params.out_first_blocked_segment = first_blocked_segment.data_ptr<int>();
    params.out_first_blocked_prim = first_blocked_prim.data_ptr<int>();

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    launch_segment_visibility_backend(
        scene, params, 3, static_cast<unsigned int>(chain_count), torch_ctx.stream);
    return {all_visible, first_blocked_segment, first_blocked_prim};
}


at::Tensor axial_edge_visibility_forward_native_impl(
    SceneCache &scene,
    const at::Tensor &tx,
    const at::Tensor &edge_position,
    const at::Tensor &edge_direction,
    const at::Tensor &edge_t_min,
    const at::Tensor &edge_t_max,
    const at::Tensor *active,
    const std::array<std::uint32_t, AxialEdgeVisibilitySampleCount>
        &sample_fraction_bits) {
    if (scene.trace_backend == TraceBackend::Cuda)
        throw std::runtime_error(
            "ADR-0029 axial-edge visibility is unsupported by the CUDA ray-tracing backend; "
            "select trace_backend='optix'.");
    require_cuda(tx, "tx");
    require_contiguous(tx, "tx");
    require_dtype(tx, at::kFloat, "tx");
    require_rank(tx, 1, "tx");
    if (tx.size(0) != 3)
        throw std::runtime_error("tx must have shape (3,).");
    require_vec3f(edge_position, "edge_position");
    require_vec3f(edge_direction, "edge_direction");
    require_flat_f32(edge_t_min, "edge_t_min");
    require_flat_f32(edge_t_max, "edge_t_max");
    const int64_t state_count = edge_position.size(0);
    if (edge_direction.size(0) != state_count ||
        edge_t_min.size(0) != state_count ||
        edge_t_max.size(0) != state_count) {
        throw std::runtime_error(
            "axial-edge visibility inputs must have the same state count.");
    }
    if (active != nullptr) {
        require_mask(*active, "active");
        require_contiguous(*active, "active");
        if (active->size(0) != state_count) {
            throw std::runtime_error(
                "active must match the axial-edge state count.");
        }
    }

    require_scene_device(scene, tx, "tx");
    require_scene_device(scene, edge_position, "edge_position");
    require_scene_device(scene, edge_direction, "edge_direction");
    require_scene_device(scene, edge_t_min, "edge_t_min");
    require_scene_device(scene, edge_t_max, "edge_t_max");
    if (active != nullptr)
        require_scene_device(scene, *active, "active");

    std::array<float, AxialEdgeVisibilitySampleCount> sample_fractions{};
    for (std::size_t index = 0; index < sample_fractions.size(); ++index) {
        sample_fractions[index] = float_from_bits(sample_fraction_bits[index]);
        if (!std::isfinite(sample_fractions[index])) {
            throw std::runtime_error(
                "axial-edge visibility fraction bits must encode finite float32 values.");
        }
    }

    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    at::Tensor any_visible =
        at::empty({state_count}, edge_position.options().dtype(at::kBool));
    if (state_count == 0)
        return any_visible;

    AxialEdgeVisibilityParams params = {};
    params.trace.handle = scene.triangle_ias.traversable;
    params.tx = tx.data_ptr<float>();
    params.edge_position = edge_position.data_ptr<float>();
    params.edge_direction = edge_direction.data_ptr<float>();
    params.edge_t_min = edge_t_min.data_ptr<float>();
    params.edge_t_max = edge_t_max.data_ptr<float>();
    params.active = active == nullptr ? nullptr : mask_ptr(*active);
    params.state_count = checked_i32(state_count, "state_count");
    for (std::size_t index = 0; index < sample_fractions.size(); ++index)
        params.sample_fractions[index] = sample_fractions[index];
    params.out_any_visible = mutable_mask_ptr(any_visible);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    optix_pipeline_for_scene(scene, axial_edge_visibility_pipeline_config())
        ->launch(0, params, static_cast<unsigned int>(state_count), torch_ctx.stream);
    return any_visible;
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

} // namespace

VisibilityResult visibility_forward(
    const SceneResource &scene,
    const VisibilityRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    auto out = torch_backend::visibility_forward_native_impl(
        cache,
        request.start,
        request.end,
        present_optional(request.active));
    return {out.visible, out.blocker_prim, out.tape_t};
}

AxialEdgeVisibilityResult axial_edge_visibility_forward(
    const SceneResource &scene,
    const AxialEdgeVisibilityRequest &request) {
    auto &cache = detail::IntegrationAccess::scene_cache(scene);
    return {torch_backend::axial_edge_visibility_forward_native_impl(
        cache,
        request.tx,
        request.edge_position,
        request.edge_direction,
        request.edge_t_min,
        request.edge_t_max,
        present_optional(request.active),
        request.config.sample_fraction_bits)};
}

} // namespace rayd::torch
namespace rayd::torch_backend {

OptixPipelineConfig refl_visibility_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_segment_visibility_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_segment_visibility_optix_ptx);
    config.raygen_entries = {
        "__raygen__segment_visibility",
        "__raygen__segment_pair_visibility",
        "__raygen__axial_edge_visibility",
        "__raygen__segment_chain_visibility",
    };
    config.miss_entry = "__miss__segment_visibility";
    config.closesthit_entry = "__closesthit__segment_visibility";
    config.anyhit_entry = "__anyhit__segment_visibility";
    config.num_payload_values = shared::optix::VisibilityPayloadCount;
    config.params_size = sizeof(SegmentVisibilityParams);
    return config;
}

OptixPipelineConfig axial_edge_visibility_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_axial_edge_visibility_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_axial_edge_visibility_optix_ptx);
    config.raygen_entries = {"__raygen__axial_edge_visibility_exact"};
    config.miss_entry = "__miss__axial_edge_visibility_exact";
    config.closesthit_entry = "__closesthit__axial_edge_visibility_exact";
    config.num_payload_values = shared::optix::VisibilityPayloadCount;
    config.params_size = sizeof(AxialEdgeVisibilityParams);
    return config;
}

} // namespace rayd::torch_backend
