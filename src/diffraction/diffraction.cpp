// Copyright Xingyu Chen.
// Implements diffraction support for diffraction.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <src/diffraction/accum_params.h>
#include <src/diffraction/accum_ad.h>
#include <src/diffraction/accum_reduce.h>
#include <src/diffraction/paths_init.h>
#include <src/diffraction/paths_params.h>
#include <src/diffraction/pipeline.h>
#include <src/scene/geometry_kernels.h>
#include <src/runtime/optix_pipeline.h>
#include <src/reflection/kernels.h>
#include <src/reflection/pipeline.h>
#include <src/runtime/optix_context.h>
#include <src/reflection/accum_params.h>
#include <src/reflection/reflection_internal.h>
#include <src/reflection/epc_field.h>
#include <src/reflection/reflection_internal.h>
#include <src/visibility/visibility_params.h>
#include <src/scene/cache.h>
#include <src/scene/multipath_cuda.h>
#include <src/bindings/tensor_contract.h>
#include <rayd/diffraction.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace rayd::torch_backend {

namespace {

constexpr int64_t kStagedDfrAccumMinSamples = 2048;
constexpr int64_t kStagedDfrAccumMinSamplesPerCell = 4;

/// Rebase a per-lane device buffer for a sharded accumulation launch.
///
/// The diffraction accumulation device body indexes every per-lane buffer by
/// the *global* Monte-Carlo lane, while a shard only allocates the lanes it
/// launches. Subtracting `lane_offset` elements from the base pointer maps the
/// global lane back onto the local slot. Only lanes in
/// `[lane_offset, lane_offset + launch width)` run, so the rebased pointer is
/// never dereferenced outside the buffer. `lane_offset == 0` is a no-op, which
/// keeps the unsharded launch bit-identical.
template <typename T> T* rebase_lane_buffer(T* ptr, int32_t lane_offset, int32_t element_stride = 1) {
    if (ptr == nullptr || lane_offset == 0)
        return ptr;
    const std::uintptr_t back =
        static_cast<std::uintptr_t>(static_cast<int64_t>(lane_offset) * element_stride) * sizeof(T);
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(ptr) - back);
}

/// Number of lanes a `(lane_offset, lane_count)` window launches out of a
/// `total_samples`-wide Monte-Carlo lane space. `lane_count < 0` means "every
/// remaining lane", so the default `(0, -1)` window is the whole space.
int64_t resolve_lane_window(int64_t lane_offset, int64_t lane_count, int64_t total_samples) {
    if (lane_offset < 0)
        throw std::runtime_error("lane_offset must be non-negative.");
    if (lane_offset > total_samples)
        throw std::runtime_error("lane_offset must not exceed the total sample count.");
    const int64_t remaining = total_samples - lane_offset;
    if (lane_count < 0)
        return remaining;
    if (lane_count > remaining)
        throw std::runtime_error("lane_offset + lane_count must not exceed the total sample count.");
    return lane_count;
}

void require_same_batch(const at::Tensor& a, const at::Tensor& b, const char* name) {
    if (a.size(0) != b.size(0))
        throw std::runtime_error(std::string(name) + " tensors must have the same batch size.");
}

void require_flat_i32(const at::Tensor& tensor, const char* name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kInt, name);
    require_rank(tensor, 1, name);
}

void require_flat_i32_strided(const at::Tensor& tensor, const char* name) {
    require_cuda(tensor, name);
    require_dtype(tensor, at::kInt, name);
    require_rank(tensor, 1, name);
}

void require_flat_f32(const at::Tensor& tensor, const char* name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 1, name);
}

void require_flat_f32_strided(const at::Tensor& tensor, const char* name) {
    require_cuda(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 1, name);
}

void require_vec3f_strided(const at::Tensor& tensor, const char* name) {
    require_cuda(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 2, name);
    require_last_dim(tensor, 3, name);
}

void require_mask_strided(const at::Tensor& tensor, const char* name) {
    require_cuda(tensor, name);
    require_dtype(tensor, at::kBool, name);
    require_rank(tensor, 1, name);
}

void require_state_width(const at::Tensor& tensor, int64_t state_count, const char* name) {
    if (tensor.size(0) < state_count)
        throw std::runtime_error(std::string(name) + " must cover state_count.");
}

bool has_optional_tensor(const c10::optional<at::Tensor>& tensor) {
    return tensor.has_value() && tensor->defined() && tensor->numel() != 0;
}

bool has_defined_optional_tensor(const c10::optional<at::Tensor>& tensor) {
    return tensor.has_value() && tensor->defined();
}

const at::Tensor& require_defined_optional_tensor(const c10::optional<at::Tensor>& tensor, const char* name) {
    if (!has_defined_optional_tensor(tensor))
        throw std::runtime_error(std::string(name) + " must be provided.");
    return *tensor;
}

void require_optional_mask(const c10::optional<at::Tensor>& tensor, const char* name) {
    if (!has_defined_optional_tensor(tensor))
        return;
    require_cuda(*tensor, name);
    require_dtype(*tensor, at::kBool, name);
    require_rank(*tensor, 1, name);
}

void require_optional_vec3f(const c10::optional<at::Tensor>& tensor, const char* name) {
    if (has_optional_tensor(tensor))
        require_vec3f(*tensor, name);
}

void require_optional_vec3f_strided(const c10::optional<at::Tensor>& tensor, const char* name) {
    if (has_optional_tensor(tensor))
        require_vec3f_strided(*tensor, name);
}

void require_optional_scalar_f(const c10::optional<at::Tensor>& tensor, const char* name) {
    if (has_optional_tensor(tensor))
        require_scalar_f(*tensor, name);
}

void require_optional_scalar_f_strided(const c10::optional<at::Tensor>& tensor, const char* name) {
    if (has_optional_tensor(tensor))
        require_flat_f32_strided(*tensor, name);
}

void require_optional_state_width(const c10::optional<at::Tensor>& tensor, int64_t state_count, const char* name) {
    if (has_optional_tensor(tensor))
        require_state_width(*tensor, state_count, name);
}

at::Tensor flatten_optional_f32(const c10::optional<at::Tensor>& tensor, const char* name) {
    if (!has_optional_tensor(tensor))
        return at::Tensor();
    at::Tensor flat = tensor->reshape({-1}).contiguous();
    require_flat_f32(flat, name);
    return flat;
}

int32_t checked_i32(int64_t value, const char* name) {
    if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error(std::string(name) + " does not fit in int32.");
    return static_cast<int32_t>(value);
}

int32_t stride_i32(const at::Tensor& tensor, int64_t dim, const char* name) {
    return checked_i32(tensor.stride(dim), name);
}

const uint8_t* optional_mask_ptr(const at::Tensor& mask) {
    if (!mask.defined() || mask.numel() == 0)
        return nullptr;
    return reinterpret_cast<const uint8_t*>(mask.data_ptr<bool>());
}

at::Tensor active_mask_for_states(const at::Tensor& active, int64_t state_count, const char* name) {
    if (!active.defined() || active.numel() == 0)
        return active;
    if (active.size(0) == state_count || active.size(0) == 1)
        return active;
    throw std::runtime_error(std::string(name) + " active width must be 1 or match state_count.");
}

int32_t active_width_for_states(const at::Tensor& active, const char* name) {
    if (!active.defined() || active.numel() == 0)
        return 0;
    return checked_i32(active.size(0), name);
}

int32_t active_stride_for_states(const at::Tensor& active, const char* name) {
    if (!active.defined() || active.numel() == 0)
        return 0;
    return checked_i32(active.stride(0), name);
}

at::Tensor first_bounce_column(const at::Tensor& value, int64_t ray_count) {
    if (value.dim() == 1)
        return value.reshape({ray_count}).contiguous();
    return value.slice(1, 0, 1).reshape({ray_count}).contiguous();
}

struct Vec3SoA {
    at::Tensor x;
    at::Tensor y;
    at::Tensor z;
};

struct Vec3Input {
    const float* x = nullptr;
    const float* y = nullptr;
    const float* z = nullptr;
    int32_t stride = 0;
};

struct Vec3Output {
    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;
    int32_t stride = 0;
};

struct GridGradInput {
    const float* ptr = nullptr;
    int32_t rank = 0;
    int32_t stride0 = 0;
    int32_t stride1 = 0;
};

Vec3SoA split_vec3(const at::Tensor& value) {
    return {
        value.select(1, 0).contiguous(),
        value.select(1, 1).contiguous(),
        value.select(1, 2).contiguous(),
    };
}

Vec3SoA split_optional_vec3(const c10::optional<at::Tensor>& value) {
    if (!has_optional_tensor(value))
        return {};
    return split_vec3(*value);
}

Vec3Input vec3_input(const at::Tensor& value, const char* name) {
    const float* base = value.data_ptr<float>();
    const int32_t stride0 = stride_i32(value, 0, (std::string(name) + "_stride0").c_str());
    const int64_t stride1 = value.stride(1);
    return {
        base,
        base + checked_i32(stride1, (std::string(name) + "_stride1").c_str()),
        base + 2 * checked_i32(stride1, (std::string(name) + "_stride1").c_str()),
        stride0,
    };
}

Vec3Output vec3_output(at::Tensor& value, const char* name) {
    float* base = value.data_ptr<float>();
    const int32_t stride0 = stride_i32(value, 0, (std::string(name) + "_stride0").c_str());
    const int64_t stride1 = value.stride(1);
    return {
        base,
        base + checked_i32(stride1, (std::string(name) + "_stride1").c_str()),
        base + 2 * checked_i32(stride1, (std::string(name) + "_stride1").c_str()),
        stride0,
    };
}

Vec3Input optional_vec3_input(const c10::optional<at::Tensor>& value, const char* name) {
    if (!has_optional_tensor(value))
        return {};
    return vec3_input(*value, name);
}

float* vec3_ptr(at::Tensor& value) {
    return value.defined() ? value.data_ptr<float>() : nullptr;
}

const float* vec3_ptr(const at::Tensor& value) {
    return value.defined() ? value.data_ptr<float>() : nullptr;
}

const float* optional_scalar_ptr(const c10::optional<at::Tensor>& value) {
    return has_optional_tensor(value) ? value->data_ptr<float>() : nullptr;
}

int32_t optional_scalar_stride(const c10::optional<at::Tensor>& value, const char* name) {
    if (!has_optional_tensor(value))
        return 0;
    return stride_i32(*value, 0, name);
}

GridGradInput optional_grid_grad_input(const c10::optional<at::Tensor>& value, int64_t resolution0, int64_t resolution1,
                                       const char* name) {
    if (!has_optional_tensor(value))
        return {};
    require_cuda(*value, name);
    require_dtype(*value, at::kFloat, name);
    const int64_t cell_count = resolution0 * resolution1;
    GridGradInput view;
    view.ptr = value->data_ptr<float>();
    if (value->dim() == 2) {
        if (value->size(0) != resolution1 || value->size(1) != resolution0)
            throw std::runtime_error(std::string(name) + " must match grid shape.");
        view.rank = 2;
        view.stride0 = stride_i32(*value, 0, (std::string(name) + "_stride0").c_str());
        view.stride1 = stride_i32(*value, 1, (std::string(name) + "_stride1").c_str());
        return view;
    }
    if (value->dim() == 1) {
        if (value->size(0) < cell_count)
            throw std::runtime_error(std::string(name) + " must cover grid cells.");
        view.rank = 1;
        view.stride0 = stride_i32(*value, 0, (std::string(name) + "_stride0").c_str());
        return view;
    }
    throw std::runtime_error(std::string(name) + " must be rank 1 or rank 2.");
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

TriangleSoA make_triangle_soa(const MeshRecord& mesh) {
    at::Tensor faces_i64 = mesh.faces.to(at::kLong);
    at::Tensor v0 = mesh.vertices.index_select(0, faces_i64.select(1, 0)).contiguous();
    at::Tensor v1 = mesh.vertices.index_select(0, faces_i64.select(1, 1)).contiguous();
    at::Tensor v2 = mesh.vertices.index_select(0, faces_i64.select(1, 2)).contiguous();
    at::Tensor e1 = (v1 - v0).contiguous();
    at::Tensor e2 = (v2 - v0).contiguous();
    at::Tensor fn = at::cross(e1, e2, 1).contiguous();
    return {
        v0.select(1, 0).contiguous(),         v0.select(1, 1).contiguous(),
        v0.select(1, 2).contiguous(),         e1.select(1, 0).contiguous(),
        e1.select(1, 1).contiguous(),         e1.select(1, 2).contiguous(),
        e2.select(1, 0).contiguous(),         e2.select(1, 1).contiguous(),
        e2.select(1, 2).contiguous(),         fn.select(1, 0).contiguous(),
        fn.select(1, 1).contiguous(),         fn.select(1, 2).contiguous(),
        at::zeros({1}, mesh.faces.options()), static_cast<int32_t>(mesh.faces.size(0)),
    };
}

TriangleSoA make_scene_triangle_soa(const SceneCache& scene) {
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
        scene.face_offsets.contiguous(),
        static_cast<int32_t>(scene.global_faces.size(0)),
    };
}

const uint8_t* mask_ptr(const at::Tensor& mask) {
    return reinterpret_cast<const uint8_t*>(mask.data_ptr<bool>());
}

uint8_t* mutable_mask_ptr(const at::Tensor& mask) {
    return reinterpret_cast<uint8_t*>(mask.data_ptr<bool>());
}

at::Tensor stack_vec3(const at::Tensor& x, const at::Tensor& y, const at::Tensor& z) {
    return at::stack({x, y, z}, 1).contiguous();
}

std::shared_ptr<OptixLaunchPipeline> optix_pipeline_for_scene(const SceneCache& scene,
                                                              const OptixPipelineConfig& config) {
    OptixDeviceContextEntry& optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    return shared_optix_launch_pipeline(optix_entry.optix_context, static_cast<int>(scene.device_index), 1, config);
}

void require_scene_device(const SceneCache& scene, const at::Tensor& value, const char* name) {
    if (value.defined() && value.get_device() != scene.device_index)
        throw std::runtime_error(std::string(name) + " must be on the same CUDA device as the scene.");
}

void require_scene_device(const SceneCache& scene, const c10::optional<at::Tensor>& value, const char* name) {
    if (value.has_value())
        require_scene_device(scene, *value, name);
}

} // namespace

struct DiffractionPathOutputs {
    at::Tensor count;
    at::Tensor valid;
    at::Tensor tx_id;
    at::Tensor rx_id;
    at::Tensor order;
    at::Tensor edge0;
    at::Tensor edge1;
    at::Tensor edge2;
    at::Tensor delay;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
    at::Tensor field_y_re;
    at::Tensor field_y_im;
    at::Tensor field_z_re;
    at::Tensor field_z_im;
    at::Tensor p0;
    at::Tensor p1;
    at::Tensor p2;
};

DiffractionPathOutputs diffraction_paths_order1_forward_impl(
    SceneCache& scene, at::Tensor tx_pos, at::Tensor tx_pol, at::Tensor rx_pos, at::Tensor active,
    at::Tensor state_edge_index, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max, at::Tensor state_n0, at::Tensor state_n1, at::Tensor state_prim0,
    at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power,
    at::Tensor material_eta_r, at::Tensor material_sigma, at::Tensor material_mu_r, at::Tensor material_gain,
    at::Tensor material_valid, int64_t state_limit_arg, int64_t capacity, int output_layout, double wavelength,
    double isb_taper_width_scale) {
    require_vec3f_strided(tx_pos, "tx_pos");
    require_vec3f_strided(tx_pol, "tx_pol");
    require_vec3f_strided(rx_pos, "rx_pos");
    require_cuda(active, "active");
    require_contiguous(active, "active");
    require_dtype(active, at::kBool, "active");
    require_rank(active, 1, "active");
    require_flat_i32_strided(state_edge_index, "state_edge_index");
    require_vec3f_strided(state_edge_pos, "state_edge_pos");
    require_vec3f_strided(state_edge_dir, "state_edge_dir");
    require_flat_f32_strided(state_edge_t_min, "state_edge_t_min");
    require_flat_f32_strided(state_edge_t_max, "state_edge_t_max");
    require_vec3f_strided(state_n0, "state_n0");
    require_vec3f_strided(state_n1, "state_n1");
    require_flat_i32_strided(state_prim0, "state_prim0");
    require_flat_i32_strided(state_prim1, "state_prim1");
    require_flat_f32_strided(state_exterior_angle, "state_exterior_angle");
    require_vec3f_strided(state_src, "state_src");
    require_flat_f32_strided(state_src_power, "state_src_power");
    require_flat_f32_strided(material_eta_r, "material_eta_r");
    require_flat_f32_strided(material_sigma, "material_sigma");
    require_flat_f32_strided(material_mu_r, "material_mu_r");
    require_flat_f32_strided(material_gain, "material_gain");
    require_mask_strided(material_valid, "material_valid");
    if (state_limit_arg < 0)
        throw std::runtime_error("state_limit must be non-negative.");
    if (capacity < 0)
        throw std::runtime_error("capacity must be non-negative.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");

    const int64_t tx_count = tx_pos.size(0);
    if (tx_pol.size(0) != 1 && tx_pol.size(0) != tx_count)
        throw std::runtime_error("tx_pol width must be 1 or match tx_pos.");
    const int64_t rx_count = rx_pos.size(0);
    const int64_t state_physical_count = state_edge_index.size(0);
    if (state_limit_arg > state_physical_count)
        throw std::runtime_error("state_limit must not exceed state_edge_index width.");
    const int64_t state_limit = state_limit_arg;
    if (active.size(0) != state_limit)
        throw std::runtime_error("active must have shape [state_limit].");
    require_state_width(state_edge_pos, state_limit, "state_edge_pos");
    require_state_width(state_edge_dir, state_limit, "state_edge_dir");
    require_state_width(state_edge_t_min, state_limit, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_limit, "state_edge_t_max");
    require_state_width(state_n0, state_limit, "state_n0");
    require_state_width(state_n1, state_limit, "state_n1");
    require_state_width(state_prim0, state_limit, "state_prim0");
    require_state_width(state_prim1, state_limit, "state_prim1");
    require_state_width(state_exterior_angle, state_limit, "state_exterior_angle");
    require_state_width(state_src, state_limit, "state_src");
    require_state_width(state_src_power, state_limit, "state_src_power");
    const int64_t material_count = material_gain.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_eta_r.size(0) != material_count || material_sigma.size(0) != material_count ||
        material_mu_r.size(0) != material_count || material_valid.size(0) != material_count)
        throw std::runtime_error("diffraction material tensors must have matching widths.");

    const int64_t n_rays64 = tx_count * rx_count * state_limit;
    if (n_rays64 > capacity)
        throw std::runtime_error("capacity must be at least tx_count * rx_count * state_limit.");
    const int32_t n_rays = checked_i32(n_rays64, "n_rays");
    const int32_t capacity_i32 = checked_i32(capacity, "capacity");
    if (output_layout != kDiffractionPathLayoutCompact && output_layout != kDiffractionPathLayoutSourceLane)
        throw std::runtime_error("diffraction path layout is invalid.");

    require_scene_device(scene, tx_pos, "tx_pos");
    require_scene_device(scene, tx_pol, "tx_pol");
    require_scene_device(scene, rx_pos, "rx_pos");
    require_scene_device(scene, active, "active");
    require_scene_device(scene, state_edge_index, "state_edge_index");
    require_scene_device(scene, state_edge_pos, "state_edge_pos");
    require_scene_device(scene, state_edge_dir, "state_edge_dir");
    require_scene_device(scene, state_edge_t_min, "state_edge_t_min");
    require_scene_device(scene, state_edge_t_max, "state_edge_t_max");
    require_scene_device(scene, state_n0, "state_n0");
    require_scene_device(scene, state_n1, "state_n1");
    require_scene_device(scene, state_prim0, "state_prim0");
    require_scene_device(scene, state_prim1, "state_prim1");
    require_scene_device(scene, state_exterior_angle, "state_exterior_angle");
    require_scene_device(scene, state_src, "state_src");
    require_scene_device(scene, state_src_power, "state_src_power");
    require_scene_device(scene, material_eta_r, "material_eta_r");
    require_scene_device(scene, material_sigma, "material_sigma");
    require_scene_device(scene, material_mu_r, "material_mu_r");
    require_scene_device(scene, material_gain, "material_gain");
    require_scene_device(scene, material_valid, "material_valid");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));

    auto fopts = tx_pos.options();
    auto iopts = state_edge_index.options();
    auto bopts = state_edge_index.options().dtype(at::kBool);
    at::Tensor out_count = capacity_i32 == 0 ? at::zeros({1}, iopts) : at::empty({1}, iopts);
    at::Tensor out_valid = at::empty({capacity}, bopts);
    at::Tensor out_tx_id = at::empty({capacity}, iopts);
    at::Tensor out_rx_id = at::empty({capacity}, iopts);
    at::Tensor out_order = at::empty({capacity}, iopts);
    at::Tensor out_edge0 = at::empty({capacity}, iopts);
    at::Tensor out_edge1 = at::empty({capacity}, iopts);
    at::Tensor out_edge2 = at::empty({capacity}, iopts);
    at::Tensor out_delay = at::empty({capacity}, fopts);
    at::Tensor out_field_x_re = at::empty({capacity}, fopts);
    at::Tensor out_field_x_im = at::empty({capacity}, fopts);
    at::Tensor out_field_y_re = at::empty({capacity}, fopts);
    at::Tensor out_field_y_im = at::empty({capacity}, fopts);
    at::Tensor out_field_z_re = at::empty({capacity}, fopts);
    at::Tensor out_field_z_im = at::empty({capacity}, fopts);
    at::Tensor out_p0 = at::empty({capacity, 3}, fopts);
    at::Tensor out_p1 = at::empty({capacity, 3}, fopts);
    at::Tensor out_p2 = at::empty({capacity, 3}, fopts);
    if (capacity_i32 == 0) {
        return {out_count,      out_valid,      out_tx_id,      out_rx_id,      out_order,      out_edge0,
                out_edge1,      out_edge2,      out_delay,      out_field_x_re, out_field_x_im, out_field_y_re,
                out_field_y_im, out_field_z_re, out_field_z_im, out_p0,         out_p1,         out_p2};
    }
    init_dfr_path_outputs_cuda(capacity, out_count, out_valid, out_tx_id, out_rx_id, out_order, out_edge0, out_edge1,
                               out_edge2, out_delay, out_field_x_re, out_field_x_im, out_field_y_re, out_field_y_im,
                               out_field_z_re, out_field_z_im, out_p0, out_p1, out_p2);
    if (n_rays == 0) {
        return {out_count,      out_valid,      out_tx_id,      out_rx_id,      out_order,      out_edge0,
                out_edge1,      out_edge2,      out_delay,      out_field_x_re, out_field_x_im, out_field_y_re,
                out_field_y_im, out_field_z_re, out_field_z_im, out_p0,         out_p1,         out_p2};
    }

    const bool cuda_backend = scene.trace_backend == TraceBackend::Cuda;
    Vec3SoA cuda_tx_pos, cuda_rx_pos, cuda_state_edge_pos, cuda_state_edge_dir;
    Vec3SoA cuda_state_n0, cuda_state_n1, cuda_state_src;
    Vec3SoA cuda_out_p0, cuda_out_p1, cuda_out_p2;
    at::Tensor cuda_state_edge_index, cuda_state_edge_t_min, cuda_state_edge_t_max;
    at::Tensor cuda_state_prim0, cuda_state_prim1, cuda_state_exterior_angle;
    at::Tensor cuda_state_src_power, cuda_temp_visibility;
    if (cuda_backend) {
        cuda_tx_pos = split_vec3(tx_pos);
        cuda_rx_pos = split_vec3(rx_pos);
        cuda_state_edge_pos = split_vec3(state_edge_pos);
        cuda_state_edge_dir = split_vec3(state_edge_dir);
        cuda_state_n0 = split_vec3(state_n0);
        cuda_state_n1 = split_vec3(state_n1);
        cuda_state_src = split_vec3(state_src);
        cuda_state_edge_index = state_edge_index.contiguous();
        cuda_state_edge_t_min = state_edge_t_min.contiguous();
        cuda_state_edge_t_max = state_edge_t_max.contiguous();
        cuda_state_prim0 = state_prim0.contiguous();
        cuda_state_prim1 = state_prim1.contiguous();
        cuda_state_exterior_angle = state_exterior_angle.contiguous();
        cuda_state_src_power = state_src_power.contiguous();
        cuda_temp_visibility = at::zeros({n_rays}, active.options());
        cuda_out_p0 = split_vec3(out_p0);
        cuda_out_p1 = split_vec3(out_p1);
        cuda_out_p2 = split_vec3(out_p2);
    }

    DfrPathParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = n_rays;
    params.capacity = capacity_i32;
    params.output_layout = output_layout;
    params.tx_pos_x = cuda_backend ? cuda_tx_pos.x.data_ptr<float>() : nullptr;
    params.tx_pos_y = cuda_backend ? cuda_tx_pos.y.data_ptr<float>() : nullptr;
    params.tx_pos_z = cuda_backend ? cuda_tx_pos.z.data_ptr<float>() : nullptr;
    params.tx_pos_aos = tx_pos.data_ptr<float>();
    params.tx_pos_stride0 = stride_i32(tx_pos, 0, "tx_pos_stride0");
    params.tx_pos_stride1 = stride_i32(tx_pos, 1, "tx_pos_stride1");
    params.tx_count = checked_i32(tx_count, "tx_count");
    params.tx_pol_aos = tx_pol.data_ptr<float>();
    params.tx_pol_stride0 = stride_i32(tx_pol, 0, "tx_pol_stride0");
    params.tx_pol_stride1 = stride_i32(tx_pol, 1, "tx_pol_stride1");
    params.tx_pol_count = checked_i32(tx_pol.size(0), "tx_pol_count");
    params.rx_pos_x = cuda_backend ? cuda_rx_pos.x.data_ptr<float>() : nullptr;
    params.rx_pos_y = cuda_backend ? cuda_rx_pos.y.data_ptr<float>() : nullptr;
    params.rx_pos_z = cuda_backend ? cuda_rx_pos.z.data_ptr<float>() : nullptr;
    params.rx_pos_aos = rx_pos.data_ptr<float>();
    params.rx_pos_stride0 = stride_i32(rx_pos, 0, "rx_pos_stride0");
    params.rx_pos_stride1 = stride_i32(rx_pos, 1, "rx_pos_stride1");
    params.rx_count = checked_i32(rx_count, "rx_count");
    params.active_mask = reinterpret_cast<const uint8_t*>(active.data_ptr<bool>());
    params.state_count = checked_i32(state_limit, "state_count");
    params.state_limit = checked_i32(state_limit, "state_limit");
    params.state_edge_index = cuda_backend ? cuda_state_edge_index.data_ptr<int>() : state_edge_index.data_ptr<int>();
    params.state_edge_index_stride = stride_i32(state_edge_index, 0, "state_edge_index_stride");
    params.state_edge_pos_x = cuda_backend ? cuda_state_edge_pos.x.data_ptr<float>() : nullptr;
    params.state_edge_pos_y = cuda_backend ? cuda_state_edge_pos.y.data_ptr<float>() : nullptr;
    params.state_edge_pos_z = cuda_backend ? cuda_state_edge_pos.z.data_ptr<float>() : nullptr;
    params.state_edge_pos_aos = state_edge_pos.data_ptr<float>();
    params.state_edge_pos_stride0 = stride_i32(state_edge_pos, 0, "state_edge_pos_stride0");
    params.state_edge_pos_stride1 = stride_i32(state_edge_pos, 1, "state_edge_pos_stride1");
    params.state_edge_dir_x = cuda_backend ? cuda_state_edge_dir.x.data_ptr<float>() : nullptr;
    params.state_edge_dir_y = cuda_backend ? cuda_state_edge_dir.y.data_ptr<float>() : nullptr;
    params.state_edge_dir_z = cuda_backend ? cuda_state_edge_dir.z.data_ptr<float>() : nullptr;
    params.state_edge_dir_aos = state_edge_dir.data_ptr<float>();
    params.state_edge_dir_stride0 = stride_i32(state_edge_dir, 0, "state_edge_dir_stride0");
    params.state_edge_dir_stride1 = stride_i32(state_edge_dir, 1, "state_edge_dir_stride1");
    params.state_edge_t_min =
        cuda_backend ? cuda_state_edge_t_min.data_ptr<float>() : state_edge_t_min.data_ptr<float>();
    params.state_edge_t_min_stride = stride_i32(state_edge_t_min, 0, "state_edge_t_min_stride");
    params.state_edge_t_max =
        cuda_backend ? cuda_state_edge_t_max.data_ptr<float>() : state_edge_t_max.data_ptr<float>();
    params.state_edge_t_max_stride = stride_i32(state_edge_t_max, 0, "state_edge_t_max_stride");
    params.state_n0_x = cuda_backend ? cuda_state_n0.x.data_ptr<float>() : nullptr;
    params.state_n0_y = cuda_backend ? cuda_state_n0.y.data_ptr<float>() : nullptr;
    params.state_n0_z = cuda_backend ? cuda_state_n0.z.data_ptr<float>() : nullptr;
    params.state_n0_aos = state_n0.data_ptr<float>();
    params.state_n0_stride0 = stride_i32(state_n0, 0, "state_n0_stride0");
    params.state_n0_stride1 = stride_i32(state_n0, 1, "state_n0_stride1");
    params.state_n1_x = cuda_backend ? cuda_state_n1.x.data_ptr<float>() : nullptr;
    params.state_n1_y = cuda_backend ? cuda_state_n1.y.data_ptr<float>() : nullptr;
    params.state_n1_z = cuda_backend ? cuda_state_n1.z.data_ptr<float>() : nullptr;
    params.state_n1_aos = state_n1.data_ptr<float>();
    params.state_n1_stride0 = stride_i32(state_n1, 0, "state_n1_stride0");
    params.state_n1_stride1 = stride_i32(state_n1, 1, "state_n1_stride1");
    params.state_prim0 = cuda_backend ? cuda_state_prim0.data_ptr<int>() : state_prim0.data_ptr<int>();
    params.state_prim0_stride = stride_i32(state_prim0, 0, "state_prim0_stride");
    params.state_prim1 = cuda_backend ? cuda_state_prim1.data_ptr<int>() : state_prim1.data_ptr<int>();
    params.state_prim1_stride = stride_i32(state_prim1, 0, "state_prim1_stride");
    params.state_exterior_angle =
        cuda_backend ? cuda_state_exterior_angle.data_ptr<float>() : state_exterior_angle.data_ptr<float>();
    params.state_exterior_angle_stride = stride_i32(state_exterior_angle, 0, "state_exterior_angle_stride");
    params.state_src_x = cuda_backend ? cuda_state_src.x.data_ptr<float>() : nullptr;
    params.state_src_y = cuda_backend ? cuda_state_src.y.data_ptr<float>() : nullptr;
    params.state_src_z = cuda_backend ? cuda_state_src.z.data_ptr<float>() : nullptr;
    params.state_src_aos = state_src.data_ptr<float>();
    params.state_src_stride0 = stride_i32(state_src, 0, "state_src_stride0");
    params.state_src_stride1 = stride_i32(state_src, 1, "state_src_stride1");
    params.state_src_power = cuda_backend ? cuda_state_src_power.data_ptr<float>() : state_src_power.data_ptr<float>();
    params.state_src_power_stride = stride_i32(state_src_power, 0, "state_src_power_stride");
    params.material_eta_r = material_eta_r.data_ptr<float>();
    params.material_eta_r_stride = stride_i32(material_eta_r, 0, "material_eta_r_stride");
    params.material_sigma = material_sigma.data_ptr<float>();
    params.material_sigma_stride = stride_i32(material_sigma, 0, "material_sigma_stride");
    params.material_mu_r = material_mu_r.data_ptr<float>();
    params.material_mu_r_stride = stride_i32(material_mu_r, 0, "material_mu_r_stride");
    params.material_gain = material_gain.data_ptr<float>();
    params.material_gain_stride = stride_i32(material_gain, 0, "material_gain_stride");
    params.material_valid = mask_ptr(material_valid);
    params.material_valid_stride = stride_i32(material_valid, 0, "material_valid_stride");
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.omega = static_cast<float>(2.0 * 3.14159265358979323846 * 299792458.0 / wavelength);
    // ADR-017 ISB boundary-taper width scale (channel-native owned). 0 for
    // every existing caller reproduces the hard GO step bit-for-bit.
    params.isb_taper_width_scale = static_cast<float>(isb_taper_width_scale);
    params.seed = 0;
    params.max_order = 1;
    params.strategy_mask = RAYD_TORCH_DFR_DIRECT;
    params.sample_count = 1;
    params.return_geom = 1;
    params.receiver_model = RAYD_TORCH_DFR_MATCHED_ISO;
    params.temp_visibility = cuda_backend ? mutable_mask_ptr(cuda_temp_visibility) : nullptr;
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
    params.out_p0_x = cuda_backend ? cuda_out_p0.x.data_ptr<float>() : nullptr;
    params.out_p0_y = cuda_backend ? cuda_out_p0.y.data_ptr<float>() : nullptr;
    params.out_p0_z = cuda_backend ? cuda_out_p0.z.data_ptr<float>() : nullptr;
    params.out_p0_aos = out_p0.data_ptr<float>();
    params.out_p1_x = cuda_backend ? cuda_out_p1.x.data_ptr<float>() : nullptr;
    params.out_p1_y = cuda_backend ? cuda_out_p1.y.data_ptr<float>() : nullptr;
    params.out_p1_z = cuda_backend ? cuda_out_p1.z.data_ptr<float>() : nullptr;
    params.out_p2_x = cuda_backend ? cuda_out_p2.x.data_ptr<float>() : nullptr;
    params.out_p2_y = cuda_backend ? cuda_out_p2.y.data_ptr<float>() : nullptr;
    params.out_p2_z = cuda_backend ? cuda_out_p2.z.data_ptr<float>() : nullptr;

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (cuda_backend) {
        launch_diffraction_paths_cuda(scene, params, n_rays);
        out_p0 = at::stack({cuda_out_p0.x, cuda_out_p0.y, cuda_out_p0.z}, 1);
        out_p1 = at::stack({cuda_out_p1.x, cuda_out_p1.y, cuda_out_p1.z}, 1);
        out_p2 = at::stack({cuda_out_p2.x, cuda_out_p2.y, cuda_out_p2.z}, 1);
    } else {
        auto pipeline = optix_pipeline_for_scene(scene, diffraction_paths_pipeline_config());
        pipeline->launch(0, params, static_cast<unsigned int>(n_rays), torch_ctx.stream);
    }

    return {out_count,      out_valid,      out_tx_id,      out_rx_id,      out_order,      out_edge0,
            out_edge1,      out_edge2,      out_delay,      out_field_x_re, out_field_x_im, out_field_y_re,
            out_field_y_im, out_field_z_re, out_field_z_im, out_p0,         out_p1,         out_p2};
}

py::tuple diffraction_path_outputs_to_tuple(const DiffractionPathOutputs& result) {
    return py::make_tuple(result.count, result.valid, result.tx_id, result.rx_id, result.order, result.edge0,
                          result.edge1, result.edge2, result.delay, result.field_x_re, result.field_x_im,
                          result.field_y_re, result.field_y_im, result.field_z_re, result.field_z_im, result.p0,
                          result.p1, result.p2);
}

py::tuple diffraction_paths_order1_forward_op(
    int64_t scene_handle, at::Tensor tx_pos, at::Tensor tx_pol, at::Tensor rx_pos, at::Tensor active,
    at::Tensor state_edge_index, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max, at::Tensor state_n0, at::Tensor state_n1, at::Tensor state_prim0,
    at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power,
    at::Tensor material_eta_r, at::Tensor material_sigma, at::Tensor material_mu_r, at::Tensor material_gain,
    at::Tensor material_valid, int64_t state_limit, int64_t capacity, double wavelength, double isb_taper_width_scale,
    int64_t output_layout) {
    return diffraction_path_outputs_to_tuple(diffraction_paths_order1_forward_impl(
        get_scene(scene_handle), std::move(tx_pos), std::move(tx_pol), std::move(rx_pos), std::move(active),
        std::move(state_edge_index), std::move(state_edge_pos), std::move(state_edge_dir), std::move(state_edge_t_min),
        std::move(state_edge_t_max), std::move(state_n0), std::move(state_n1), std::move(state_prim0),
        std::move(state_prim1), std::move(state_exterior_angle), std::move(state_src), std::move(state_src_power),
        std::move(material_eta_r), std::move(material_sigma), std::move(material_mu_r), std::move(material_gain),
        std::move(material_valid), state_limit, capacity, checked_i32(output_layout, "output_layout"), wavelength,
        isb_taper_width_scale));
}

struct DiffractionAccumulationOutputs {
    at::Tensor power;
    at::Tensor field_x_re;
    at::Tensor field_x_im;
    at::Tensor field_y_re;
    at::Tensor field_y_im;
    at::Tensor field_z_re;
    at::Tensor field_z_im;
    at::Tensor direct_count;
    at::Tensor keller_count;
    at::Tensor suffix_count;
    at::Tensor visibility_rejects;
    at::Tensor edge_visibility_rejects;
    at::Tensor utd_rejects;
    at::Tensor edge_uses;
    at::Tensor tape_active;
    at::Tensor tape_state_idx;
    at::Tensor tape_cell;
    at::Tensor tape_material_idx;
    at::Tensor tape_edge_u;
};

DiffractionAccumulationOutputs diffraction_accumulation_forward_impl(
    SceneCache& scene, c10::optional<at::Tensor> active, at::Tensor state_edge_index, at::Tensor state_edge_pos,
    at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_n0,
    at::Tensor state_n1, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle,
    at::Tensor state_src, at::Tensor state_src_power, c10::optional<at::Tensor> state_wi,
    c10::optional<at::Tensor> state_d0, at::Tensor material_eta_r, at::Tensor material_sigma, at::Tensor material_mu_r,
    at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit_arg, int64_t grid_axis,
    double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min,
    double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area,
    double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed,
    int64_t max_order, int64_t recursive_state_limit_arg, c10::optional<at::Tensor> recursive_active,
    c10::optional<at::Tensor> recursive_state_edge_index, c10::optional<at::Tensor> recursive_state_edge_pos,
    c10::optional<at::Tensor> recursive_state_edge_dir, c10::optional<at::Tensor> recursive_state_edge_t_min,
    c10::optional<at::Tensor> recursive_state_edge_t_max, c10::optional<at::Tensor> recursive_state_n0,
    c10::optional<at::Tensor> recursive_state_n1, c10::optional<at::Tensor> recursive_state_prim0,
    c10::optional<at::Tensor> recursive_state_prim1, c10::optional<at::Tensor> recursive_state_exterior_angle,
    int64_t export_tape, c10::optional<at::Tensor> sample_state_index, c10::optional<at::Tensor> sample_edge_weight,
    int64_t lane_offset, int64_t lane_count) {
    require_optional_mask(active, "active");
    require_flat_i32_strided(state_edge_index, "state_edge_index");
    require_vec3f_strided(state_edge_pos, "state_edge_pos");
    require_vec3f_strided(state_edge_dir, "state_edge_dir");
    require_flat_f32_strided(state_edge_t_min, "state_edge_t_min");
    require_flat_f32_strided(state_edge_t_max, "state_edge_t_max");
    require_vec3f_strided(state_n0, "state_n0");
    require_vec3f_strided(state_n1, "state_n1");
    require_flat_i32_strided(state_prim0, "state_prim0");
    require_flat_i32_strided(state_prim1, "state_prim1");
    require_flat_f32_strided(state_exterior_angle, "state_exterior_angle");
    require_vec3f_strided(state_src, "state_src");
    require_flat_f32_strided(state_src_power, "state_src_power");
    require_optional_vec3f_strided(state_wi, "state_wi");
    require_optional_vec3f_strided(state_d0, "state_d0");
    require_flat_f32_strided(material_eta_r, "material_eta_r");
    require_flat_f32_strided(material_sigma, "material_sigma");
    require_flat_f32_strided(material_mu_r, "material_mu_r");
    require_flat_f32_strided(material_gain, "material_gain");
    require_mask_strided(material_valid, "material_valid");
    if (state_limit_arg < 0)
        throw std::runtime_error("state_limit must be non-negative.");
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
    if (recursive_state_limit_arg < 0)
        throw std::runtime_error("recursive_state_limit must be non-negative.");

    const int64_t state_physical_count = state_edge_index.size(0);
    if (state_limit_arg > state_physical_count)
        throw std::runtime_error("state_limit must not exceed state_edge_index width.");
    const int64_t state_count = state_limit_arg;
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
    require_optional_state_width(state_wi, state_count, "state_wi");
    require_optional_state_width(state_d0, state_count, "state_d0");
    const bool use_recursive = max_order > 1;
    int64_t recursive_state_count = 0;
    const at::Tensor* recursive_active_tensor = nullptr;
    const at::Tensor* recursive_state_edge_index_tensor = nullptr;
    const at::Tensor* recursive_state_edge_pos_tensor = nullptr;
    const at::Tensor* recursive_state_edge_dir_tensor = nullptr;
    const at::Tensor* recursive_state_edge_t_min_tensor = nullptr;
    const at::Tensor* recursive_state_edge_t_max_tensor = nullptr;
    const at::Tensor* recursive_state_n0_tensor = nullptr;
    const at::Tensor* recursive_state_n1_tensor = nullptr;
    const at::Tensor* recursive_state_prim0_tensor = nullptr;
    const at::Tensor* recursive_state_prim1_tensor = nullptr;
    const at::Tensor* recursive_state_exterior_angle_tensor = nullptr;
    if (use_recursive) {
        recursive_active_tensor = has_defined_optional_tensor(recursive_active) ? &*recursive_active : nullptr;
        recursive_state_edge_index_tensor =
            &require_defined_optional_tensor(recursive_state_edge_index, "recursive_state_edge_index");
        recursive_state_edge_pos_tensor =
            &require_defined_optional_tensor(recursive_state_edge_pos, "recursive_state_edge_pos");
        recursive_state_edge_dir_tensor =
            &require_defined_optional_tensor(recursive_state_edge_dir, "recursive_state_edge_dir");
        recursive_state_edge_t_min_tensor =
            &require_defined_optional_tensor(recursive_state_edge_t_min, "recursive_state_edge_t_min");
        recursive_state_edge_t_max_tensor =
            &require_defined_optional_tensor(recursive_state_edge_t_max, "recursive_state_edge_t_max");
        recursive_state_n0_tensor = &require_defined_optional_tensor(recursive_state_n0, "recursive_state_n0");
        recursive_state_n1_tensor = &require_defined_optional_tensor(recursive_state_n1, "recursive_state_n1");
        recursive_state_prim0_tensor = &require_defined_optional_tensor(recursive_state_prim0, "recursive_state_prim0");
        recursive_state_prim1_tensor = &require_defined_optional_tensor(recursive_state_prim1, "recursive_state_prim1");
        recursive_state_exterior_angle_tensor =
            &require_defined_optional_tensor(recursive_state_exterior_angle, "recursive_state_exterior_angle");
        if (recursive_active_tensor != nullptr)
            require_mask_strided(*recursive_active_tensor, "recursive_active");
        require_flat_i32_strided(*recursive_state_edge_index_tensor, "recursive_state_edge_index");
        require_vec3f_strided(*recursive_state_edge_pos_tensor, "recursive_state_edge_pos");
        require_vec3f_strided(*recursive_state_edge_dir_tensor, "recursive_state_edge_dir");
        require_flat_f32_strided(*recursive_state_edge_t_min_tensor, "recursive_state_edge_t_min");
        require_flat_f32_strided(*recursive_state_edge_t_max_tensor, "recursive_state_edge_t_max");
        require_vec3f_strided(*recursive_state_n0_tensor, "recursive_state_n0");
        require_vec3f_strided(*recursive_state_n1_tensor, "recursive_state_n1");
        require_flat_i32_strided(*recursive_state_prim0_tensor, "recursive_state_prim0");
        require_flat_i32_strided(*recursive_state_prim1_tensor, "recursive_state_prim1");
        require_flat_f32_strided(*recursive_state_exterior_angle_tensor, "recursive_state_exterior_angle");
        const int64_t recursive_state_physical_count = recursive_state_edge_index_tensor->size(0);
        if (recursive_state_limit_arg > recursive_state_physical_count)
            throw std::runtime_error("recursive_state_limit must not exceed recursive_state_edge_index width.");
        recursive_state_count = recursive_state_limit_arg;
        if (recursive_active_tensor != nullptr && recursive_active_tensor->numel() != 0)
            require_state_width(*recursive_active_tensor, recursive_state_count, "recursive_active");
        require_state_width(*recursive_state_edge_pos_tensor, recursive_state_count, "recursive_state_edge_pos");
        require_state_width(*recursive_state_edge_dir_tensor, recursive_state_count, "recursive_state_edge_dir");
        require_state_width(*recursive_state_edge_t_min_tensor, recursive_state_count, "recursive_state_edge_t_min");
        require_state_width(*recursive_state_edge_t_max_tensor, recursive_state_count, "recursive_state_edge_t_max");
        require_state_width(*recursive_state_n0_tensor, recursive_state_count, "recursive_state_n0");
        require_state_width(*recursive_state_n1_tensor, recursive_state_count, "recursive_state_n1");
        require_state_width(*recursive_state_prim0_tensor, recursive_state_count, "recursive_state_prim0");
        require_state_width(*recursive_state_prim1_tensor, recursive_state_count, "recursive_state_prim1");
        require_state_width(*recursive_state_exterior_angle_tensor, recursive_state_count,
                            "recursive_state_exterior_angle");
    }
    const int64_t material_count = material_eta_r.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_sigma.size(0) != material_count || material_mu_r.size(0) != material_count ||
        material_gain.size(0) != material_count || material_valid.size(0) != material_count) {
        throw std::runtime_error("material payload fields must have matching widths.");
    }

    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    const int32_t direct_launch_count = checked_i32(direct_samples, "direct_samples");
    const int32_t keller_launch_count = checked_i32(keller_samples, "keller_samples");
    const int32_t suffix_launch_count = checked_i32(suffix_samples, "suffix_samples");
    // direct/keller/suffix always describe the *global* Monte-Carlo lane space;
    // (lane_offset, lane_count) selects the window of it this launch executes.
    const int32_t total_samples = checked_i32(direct_samples + keller_samples + suffix_samples, "total_samples");
    const int32_t lane_begin = checked_i32(lane_offset, "lane_offset");
    const int32_t launch_count =
        checked_i32(resolve_lane_window(lane_offset, lane_count, total_samples), "launch_count");
    if (lane_begin != 0 && scene.trace_backend == TraceBackend::Cuda)
        throw std::runtime_error("diffraction accumulation lane_offset requires the OptiX trace backend.");
    const bool use_sample_state_index = has_defined_optional_tensor(sample_state_index);
    const bool use_sample_edge_weight = has_defined_optional_tensor(sample_edge_weight);
    if (use_sample_state_index) {
        require_flat_i32_strided(*sample_state_index, "sample_state_index");
        if (sample_state_index->size(0) < launch_count)
            throw std::runtime_error("sample_state_index must cover launch_count.");
    }
    if (use_sample_edge_weight) {
        require_flat_f32_strided(*sample_edge_weight, "sample_edge_weight");
        if (sample_edge_weight->size(0) < launch_count)
            throw std::runtime_error("sample_edge_weight must cover launch_count.");
    }
    require_scene_device(scene, active, "active");
    require_scene_device(scene, state_edge_index, "state_edge_index");
    require_scene_device(scene, state_edge_pos, "state_edge_pos");
    require_scene_device(scene, state_edge_dir, "state_edge_dir");
    require_scene_device(scene, state_edge_t_min, "state_edge_t_min");
    require_scene_device(scene, state_edge_t_max, "state_edge_t_max");
    require_scene_device(scene, state_n0, "state_n0");
    require_scene_device(scene, state_n1, "state_n1");
    require_scene_device(scene, state_prim0, "state_prim0");
    require_scene_device(scene, state_prim1, "state_prim1");
    require_scene_device(scene, state_exterior_angle, "state_exterior_angle");
    require_scene_device(scene, state_src, "state_src");
    require_scene_device(scene, state_src_power, "state_src_power");
    require_scene_device(scene, state_wi, "state_wi");
    require_scene_device(scene, state_d0, "state_d0");
    require_scene_device(scene, material_eta_r, "material_eta_r");
    require_scene_device(scene, material_sigma, "material_sigma");
    require_scene_device(scene, material_mu_r, "material_mu_r");
    require_scene_device(scene, material_gain, "material_gain");
    require_scene_device(scene, material_valid, "material_valid");
    require_scene_device(scene, recursive_active, "recursive_active");
    require_scene_device(scene, recursive_state_edge_index, "recursive_state_edge_index");
    require_scene_device(scene, recursive_state_edge_pos, "recursive_state_edge_pos");
    require_scene_device(scene, recursive_state_edge_dir, "recursive_state_edge_dir");
    require_scene_device(scene, recursive_state_edge_t_min, "recursive_state_edge_t_min");
    require_scene_device(scene, recursive_state_edge_t_max, "recursive_state_edge_t_max");
    require_scene_device(scene, recursive_state_n0, "recursive_state_n0");
    require_scene_device(scene, recursive_state_n1, "recursive_state_n1");
    require_scene_device(scene, recursive_state_prim0, "recursive_state_prim0");
    require_scene_device(scene, recursive_state_prim1, "recursive_state_prim1");
    require_scene_device(scene, recursive_state_exterior_angle, "recursive_state_exterior_angle");
    require_scene_device(scene, sample_state_index, "sample_state_index");
    require_scene_device(scene, sample_edge_weight, "sample_edge_weight");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    auto fopts = state_src.options();
    auto iopts = state_edge_index.options();
    auto bopts = state_edge_index.options().dtype(at::kBool);
    at::Tensor power = at::empty({cell_count}, fopts);
    at::Tensor field_x_re = at::empty({cell_count}, fopts);
    at::Tensor field_x_im = at::empty({cell_count}, fopts);
    at::Tensor field_y_re = at::empty({cell_count}, fopts);
    at::Tensor field_y_im = at::empty({cell_count}, fopts);
    at::Tensor field_z_re = at::empty({cell_count}, fopts);
    at::Tensor field_z_im = at::empty({cell_count}, fopts);
    at::Tensor direct_count = at::empty({1}, iopts);
    at::Tensor keller_count = at::empty({1}, iopts);
    at::Tensor suffix_count = at::empty({1}, iopts);
    at::Tensor vis_rejects = at::empty({1}, iopts);
    at::Tensor edge_vis_rejects = at::empty({1}, iopts);
    at::Tensor utd_rejects = at::empty({1}, iopts);
    at::Tensor edge_uses = at::empty({1}, iopts);
    const bool write_tape = export_tape != 0;
    at::Tensor tape_active = write_tape ? at::empty({launch_count}, bopts) : at::empty({0}, bopts);
    at::Tensor tape_state_idx = write_tape ? at::empty({launch_count}, iopts) : at::empty({0}, iopts);
    at::Tensor tape_cell = write_tape ? at::empty({launch_count}, iopts) : at::empty({0}, iopts);
    at::Tensor tape_material_idx = write_tape ? at::empty({launch_count}, iopts) : at::empty({0}, iopts);
    at::Tensor tape_edge_u = write_tape ? at::empty({launch_count}, fopts) : at::empty({0}, fopts);
    const bool staged_no_suffix_accum =
        !write_tape && !use_recursive && suffix_launch_count == 0 && (direct_launch_count + keller_launch_count) > 0 &&
        static_cast<int64_t>(launch_count) >= kStagedDfrAccumMinSamples &&
        static_cast<int64_t>(launch_count) >= cell_count * kStagedDfrAccumMinSamplesPerCell;
    at::Tensor stage_cell = staged_no_suffix_accum ? at::empty({launch_count}, iopts) : at::empty({0}, iopts);
    at::Tensor stage_value = staged_no_suffix_accum ? at::empty({launch_count, 4}, fopts) : at::empty({0, 4}, fopts);
    at::Tensor state_prefix_depth = at::empty({state_count}, iopts);
    at::Tensor temp_visibility = at::empty({launch_count}, bopts);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    DfrAccumInitArgs init_args;
    init_args.cell_count = checked_i32(cell_count, "cell_count");
    init_args.launch_count = launch_count;
    init_args.state_count = checked_i32(state_count, "state_count");
    float* const init_fields[7] = {power.data_ptr<float>(),      field_x_re.data_ptr<float>(),
                                   field_x_im.data_ptr<float>(), field_y_re.data_ptr<float>(),
                                   field_y_im.data_ptr<float>(), field_z_re.data_ptr<float>(),
                                   field_z_im.data_ptr<float>()};
    int* const init_counters[7] = {direct_count.data_ptr<int>(),     keller_count.data_ptr<int>(),
                                   suffix_count.data_ptr<int>(),     vis_rejects.data_ptr<int>(),
                                   edge_vis_rejects.data_ptr<int>(), utd_rejects.data_ptr<int>(),
                                   edge_uses.data_ptr<int>()};
    std::memcpy(init_args.fields, init_fields, sizeof(init_fields));
    std::memcpy(init_args.counters, init_counters, sizeof(init_counters));
    init_args.state_prefix_depth = state_count > 0 ? state_prefix_depth.data_ptr<int>() : nullptr;
    init_args.temp_visibility = launch_count > 0 ? mutable_mask_ptr(temp_visibility) : nullptr;
    if (write_tape && launch_count > 0) {
        init_args.tape_active = mutable_mask_ptr(tape_active);
        init_args.tape_state_idx = tape_state_idx.data_ptr<int>();
        init_args.tape_cell = tape_cell.data_ptr<int>();
        init_args.tape_material_idx = tape_material_idx.data_ptr<int>();
        init_args.tape_edge_u = tape_edge_u.data_ptr<float>();
    }
    if (staged_no_suffix_accum) {
        init_args.stage_cell = stage_cell.data_ptr<int>();
        init_args.stage_value = reinterpret_cast<float4*>(stage_value.data_ptr<float>());
    }
    init_dfr_accum_outputs_cuda(init_args, torch_ctx.stream);
    if (state_count == 0 || launch_count == 0) {
        return {power.reshape({grid_resolution1, grid_resolution0}),
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
                tape_edge_u};
    }

    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3Input state_edge_pos_view = vec3_input(state_edge_pos, "state_edge_pos");
    Vec3Input state_edge_dir_view = vec3_input(state_edge_dir, "state_edge_dir");
    Vec3Input state_n0_view = vec3_input(state_n0, "state_n0");
    Vec3Input state_n1_view = vec3_input(state_n1, "state_n1");
    Vec3Input state_src_view = vec3_input(state_src, "state_src");
    Vec3Input state_wi_view = optional_vec3_input(state_wi, "state_wi");
    Vec3Input state_d0_view = optional_vec3_input(state_d0, "state_d0");
    at::Tensor active_contig;
    if (has_defined_optional_tensor(active)) {
        active_contig = active_mask_for_states(*active, state_count, "diffraction_accumulation_forward");
    }
    at::Tensor recursive_active_contig;
    Vec3Input recursive_edge_pos_view;
    Vec3Input recursive_edge_dir_view;
    Vec3Input recursive_n0_view;
    Vec3Input recursive_n1_view;
    if (use_recursive) {
        if (recursive_active_tensor != nullptr) {
            recursive_active_contig = active_mask_for_states(*recursive_active_tensor, recursive_state_count,
                                                             "diffraction_accumulation_forward recursive_active");
        }
        recursive_edge_pos_view = vec3_input(*recursive_state_edge_pos_tensor, "recursive_state_edge_pos");
        recursive_edge_dir_view = vec3_input(*recursive_state_edge_dir_tensor, "recursive_state_edge_dir");
        recursive_n0_view = vec3_input(*recursive_state_n0_tensor, "recursive_state_n0");
        recursive_n1_view = vec3_input(*recursive_state_n1_tensor, "recursive_state_n1");
    }

    DfrAccumParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = total_samples;
    params.lane_offset = lane_begin;
    params.active_mask = optional_mask_ptr(active_contig);
    params.active_width = active_width_for_states(active_contig, "active_width");
    params.active_stride = active_stride_for_states(active_contig, "active_stride");
    params.sample_state_index_stride =
        use_sample_state_index ? stride_i32(*sample_state_index, 0, "sample_state_index_stride") : 0;
    params.sample_state_index =
        rebase_lane_buffer(use_sample_state_index ? sample_state_index->data_ptr<int>() : nullptr, lane_begin,
                           params.sample_state_index_stride);
    params.sample_edge_weight_stride =
        use_sample_edge_weight ? stride_i32(*sample_edge_weight, 0, "sample_edge_weight_stride") : 0;
    params.sample_edge_weight =
        rebase_lane_buffer(use_sample_edge_weight ? sample_edge_weight->data_ptr<float>() : nullptr, lane_begin,
                           params.sample_edge_weight_stride);
    params.state_count = checked_i32(state_count, "state_count");
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_index_stride = stride_i32(state_edge_index, 0, "state_edge_index_stride");
    params.state_edge_pos_x = state_edge_pos_view.x;
    params.state_edge_pos_y = state_edge_pos_view.y;
    params.state_edge_pos_z = state_edge_pos_view.z;
    params.state_edge_pos_stride = state_edge_pos_view.stride;
    params.state_edge_dir_x = state_edge_dir_view.x;
    params.state_edge_dir_y = state_edge_dir_view.y;
    params.state_edge_dir_z = state_edge_dir_view.z;
    params.state_edge_dir_stride = state_edge_dir_view.stride;
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_min_stride = stride_i32(state_edge_t_min, 0, "state_edge_t_min_stride");
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_edge_t_max_stride = stride_i32(state_edge_t_max, 0, "state_edge_t_max_stride");
    params.state_n0_x = state_n0_view.x;
    params.state_n0_y = state_n0_view.y;
    params.state_n0_z = state_n0_view.z;
    params.state_n0_stride = state_n0_view.stride;
    params.state_n1_x = state_n1_view.x;
    params.state_n1_y = state_n1_view.y;
    params.state_n1_z = state_n1_view.z;
    params.state_n1_stride = state_n1_view.stride;
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim0_stride = stride_i32(state_prim0, 0, "state_prim0_stride");
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_prim1_stride = stride_i32(state_prim1, 0, "state_prim1_stride");
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_exterior_angle_stride = stride_i32(state_exterior_angle, 0, "state_exterior_angle_stride");
    params.state_src_x = state_src_view.x;
    params.state_src_y = state_src_view.y;
    params.state_src_z = state_src_view.z;
    params.state_src_stride = state_src_view.stride;
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_src_power_stride = stride_i32(state_src_power, 0, "state_src_power_stride");
    params.state_wi_x = state_wi_view.x;
    params.state_wi_y = state_wi_view.y;
    params.state_wi_z = state_wi_view.z;
    params.state_wi_stride = state_wi_view.stride;
    params.state_d0_x = state_d0_view.x;
    params.state_d0_y = state_d0_view.y;
    params.state_d0_z = state_d0_view.z;
    params.state_d0_stride = state_d0_view.stride;
    params.state_prefix_depth = state_prefix_depth.data_ptr<int>();
    params.recursive_state_count = checked_i32(recursive_state_count, "recursive_state_count");
    if (use_recursive) {
        params.recursive_active_mask = optional_mask_ptr(recursive_active_contig);
        params.recursive_active_width = active_width_for_states(recursive_active_contig, "recursive_active_width");
        params.recursive_active_stride = active_stride_for_states(recursive_active_contig, "recursive_active_stride");
        params.recursive_state_edge_index = recursive_state_edge_index_tensor->data_ptr<int>();
        params.recursive_state_edge_index_stride =
            stride_i32(*recursive_state_edge_index_tensor, 0, "recursive_state_edge_index_stride");
        params.recursive_state_edge_pos_x = recursive_edge_pos_view.x;
        params.recursive_state_edge_pos_y = recursive_edge_pos_view.y;
        params.recursive_state_edge_pos_z = recursive_edge_pos_view.z;
        params.recursive_state_edge_pos_stride = recursive_edge_pos_view.stride;
        params.recursive_state_edge_dir_x = recursive_edge_dir_view.x;
        params.recursive_state_edge_dir_y = recursive_edge_dir_view.y;
        params.recursive_state_edge_dir_z = recursive_edge_dir_view.z;
        params.recursive_state_edge_dir_stride = recursive_edge_dir_view.stride;
        params.recursive_state_edge_t_min = recursive_state_edge_t_min_tensor->data_ptr<float>();
        params.recursive_state_edge_t_min_stride =
            stride_i32(*recursive_state_edge_t_min_tensor, 0, "recursive_state_edge_t_min_stride");
        params.recursive_state_edge_t_max = recursive_state_edge_t_max_tensor->data_ptr<float>();
        params.recursive_state_edge_t_max_stride =
            stride_i32(*recursive_state_edge_t_max_tensor, 0, "recursive_state_edge_t_max_stride");
        params.recursive_state_n0_x = recursive_n0_view.x;
        params.recursive_state_n0_y = recursive_n0_view.y;
        params.recursive_state_n0_z = recursive_n0_view.z;
        params.recursive_state_n0_stride = recursive_n0_view.stride;
        params.recursive_state_n1_x = recursive_n1_view.x;
        params.recursive_state_n1_y = recursive_n1_view.y;
        params.recursive_state_n1_z = recursive_n1_view.z;
        params.recursive_state_n1_stride = recursive_n1_view.stride;
        params.recursive_state_prim0 = recursive_state_prim0_tensor->data_ptr<int>();
        params.recursive_state_prim0_stride =
            stride_i32(*recursive_state_prim0_tensor, 0, "recursive_state_prim0_stride");
        params.recursive_state_prim1 = recursive_state_prim1_tensor->data_ptr<int>();
        params.recursive_state_prim1_stride =
            stride_i32(*recursive_state_prim1_tensor, 0, "recursive_state_prim1_stride");
        params.recursive_state_exterior_angle = recursive_state_exterior_angle_tensor->data_ptr<float>();
        params.recursive_state_exterior_angle_stride =
            stride_i32(*recursive_state_exterior_angle_tensor, 0, "recursive_state_exterior_angle_stride");
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
    params.material_gain_stride = stride_i32(material_gain, 0, "material_gain_stride");
    params.material_valid = mask_ptr(material_valid);
    params.material_valid_stride = stride_i32(material_valid, 0, "material_valid_stride");
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.seed = checked_i32(seed, "seed");
    params.samples = total_samples;
    params.max_order = checked_i32(max_order, "max_order");
    params.direct_samples = direct_launch_count;
    params.keller_samples = keller_launch_count;
    params.suffix_samples = suffix_launch_count;
    params.strategy_mask = (direct_launch_count > 0 ? RAYD_TORCH_DFR_DIRECT : 0) |
                           (keller_launch_count > 0 ? RAYD_TORCH_DFR_KELLER : 0) |
                           (suffix_launch_count > 0 ? RAYD_TORCH_DFR_SUFFIX_REFL : 0);
    params.sample_sequence = RAYD_TORCH_DFR_HASH;
    params.receiver_model = RAYD_TORCH_DFR_MATCHED_ISO;
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
    // Per-lane buffers stay shard-local; the device body reaches them through
    // the global lane, so their bases move back by lane_offset elements.
    params.temp_visibility = rebase_lane_buffer(mutable_mask_ptr(temp_visibility), lane_begin);
    params.tape_active = rebase_lane_buffer(write_tape ? mutable_mask_ptr(tape_active) : nullptr, lane_begin);
    params.tape_state_idx = rebase_lane_buffer(write_tape ? tape_state_idx.data_ptr<int>() : nullptr, lane_begin);
    params.tape_cell = rebase_lane_buffer(write_tape ? tape_cell.data_ptr<int>() : nullptr, lane_begin);
    params.tape_material_idx = rebase_lane_buffer(write_tape ? tape_material_idx.data_ptr<int>() : nullptr, lane_begin);
    params.tape_edge_u = rebase_lane_buffer(write_tape ? tape_edge_u.data_ptr<float>() : nullptr, lane_begin);
    params.stage_cell = rebase_lane_buffer(staged_no_suffix_accum ? stage_cell.data_ptr<int>() : nullptr, lane_begin);
    params.stage_value =
        rebase_lane_buffer(staged_no_suffix_accum ? reinterpret_cast<float4*>(stage_value.data_ptr<float>()) : nullptr,
                           lane_begin);

    std::shared_ptr<OptixLaunchPipeline> pipeline;
    if (scene.trace_backend != TraceBackend::Cuda)
        pipeline = optix_pipeline_for_scene(scene, diffraction_accumulation_pipeline_config());
    auto launch_variant = [&](int variant) {
        if (scene.trace_backend == TraceBackend::Cuda)
            launch_diffraction_accumulation_cuda(scene, params, variant, launch_count);
        else
            pipeline->launch(variant, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
    };
    if (use_recursive) {
        launch_variant(13);
    } else {
        launch_variant(6);
        if (direct_launch_count + keller_launch_count > 0)
            launch_variant(7);
        if (staged_no_suffix_accum) {
            reduce_dfr_accum_staged_cuda(launch_count, stage_cell, stage_value, power, field_x_re, direct_count,
                                         keller_count, edge_uses);
        }
        if (suffix_launch_count > 0) {
            launch_variant(8);
            launch_variant(9);
        }
    }

    return {power.reshape({grid_resolution1, grid_resolution0}),
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
            tape_edge_u};
}

py::tuple diffraction_accumulation_outputs_to_tuple(const DiffractionAccumulationOutputs& result) {
    return py::make_tuple(result.power, result.field_x_re, result.field_x_im, result.field_y_re, result.field_y_im,
                          result.field_z_re, result.field_z_im, result.direct_count, result.keller_count,
                          result.suffix_count, result.visibility_rejects, result.edge_visibility_rejects,
                          result.utd_rejects, result.edge_uses, result.tape_active, result.tape_state_idx,
                          result.tape_cell, result.tape_material_idx, result.tape_edge_u);
}

py::tuple diffraction_accumulation_forward_op(
    int64_t scene_handle, c10::optional<at::Tensor> active, at::Tensor state_edge_index, at::Tensor state_edge_pos,
    at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_n0,
    at::Tensor state_n1, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle,
    at::Tensor state_src, at::Tensor state_src_power, c10::optional<at::Tensor> state_wi,
    c10::optional<at::Tensor> state_d0, at::Tensor material_eta_r, at::Tensor material_sigma, at::Tensor material_mu_r,
    at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit, int64_t grid_axis, double grid_position,
    double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max,
    int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area, double wavelength,
    int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed, int64_t max_order,
    int64_t recursive_state_limit, c10::optional<at::Tensor> recursive_active,
    c10::optional<at::Tensor> recursive_state_edge_index, c10::optional<at::Tensor> recursive_state_edge_pos,
    c10::optional<at::Tensor> recursive_state_edge_dir, c10::optional<at::Tensor> recursive_state_edge_t_min,
    c10::optional<at::Tensor> recursive_state_edge_t_max, c10::optional<at::Tensor> recursive_state_n0,
    c10::optional<at::Tensor> recursive_state_n1, c10::optional<at::Tensor> recursive_state_prim0,
    c10::optional<at::Tensor> recursive_state_prim1, c10::optional<at::Tensor> recursive_state_exterior_angle,
    int64_t export_tape, c10::optional<at::Tensor> sample_state_index, c10::optional<at::Tensor> sample_edge_weight,
    int64_t lane_offset, int64_t lane_count) {
    return diffraction_accumulation_outputs_to_tuple(diffraction_accumulation_forward_impl(
        get_scene(scene_handle), active, state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min,
        state_edge_t_max, state_n0, state_n1, state_prim0, state_prim1, state_exterior_angle, state_src,
        state_src_power, state_wi, state_d0, material_eta_r, material_sigma, material_mu_r, material_gain,
        material_valid, state_limit, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min,
        grid_coord1_max, grid_resolution0, grid_resolution1, grid_cell_area, wavelength, direct_samples, keller_samples,
        suffix_samples, seed, max_order, recursive_state_limit, recursive_active, recursive_state_edge_index,
        recursive_state_edge_pos, recursive_state_edge_dir, recursive_state_edge_t_min, recursive_state_edge_t_max,
        recursive_state_n0, recursive_state_n1, recursive_state_prim0, recursive_state_prim1,
        recursive_state_exterior_angle, export_tape, sample_state_index, sample_edge_weight, lane_offset, lane_count));
}

py::tuple diffraction_accumulation_direct_backward_op(
    int64_t scene_handle, at::Tensor tape_active, at::Tensor tape_state_idx, at::Tensor tape_cell,
    at::Tensor tape_material_idx, at::Tensor tape_edge_u, at::Tensor state_edge_pos, at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_prim0, at::Tensor state_prim1,
    at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power,
    c10::optional<at::Tensor> state_wi, at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit_arg,
    int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min,
    double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area,
    double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed,
    c10::optional<at::Tensor> grad_power, c10::optional<at::Tensor> grad_field_x_re, int64_t lane_offset) {
    require_mask(tape_active, "tape_active");
    require_flat_i32(tape_state_idx, "tape_state_idx");
    require_flat_i32(tape_cell, "tape_cell");
    require_flat_i32(tape_material_idx, "tape_material_idx");
    require_flat_f32(tape_edge_u, "tape_edge_u");
    require_vec3f_strided(state_edge_pos, "state_edge_pos");
    require_vec3f_strided(state_edge_dir, "state_edge_dir");
    require_flat_f32_strided(state_edge_t_min, "state_edge_t_min");
    require_flat_f32_strided(state_edge_t_max, "state_edge_t_max");
    require_flat_i32_strided(state_prim0, "state_prim0");
    require_flat_i32_strided(state_prim1, "state_prim1");
    require_flat_f32_strided(state_exterior_angle, "state_exterior_angle");
    require_vec3f_strided(state_src, "state_src");
    require_flat_f32_strided(state_src_power, "state_src_power");
    require_optional_vec3f_strided(state_wi, "state_wi");
    require_flat_f32_strided(material_gain, "material_gain");
    require_mask_strided(material_valid, "material_valid");
    const int64_t launch_count = tape_active.size(0);
    require_state_width(tape_state_idx, launch_count, "tape_state_idx");
    require_state_width(tape_cell, launch_count, "tape_cell");
    require_state_width(tape_material_idx, launch_count, "tape_material_idx");
    require_state_width(tape_edge_u, launch_count, "tape_edge_u");
    if (state_limit_arg < 0)
        throw std::runtime_error("state_limit must be non-negative.");
    const int64_t state_physical_count = state_edge_pos.size(0);
    if (state_limit_arg > state_physical_count)
        throw std::runtime_error("state_limit must not exceed state_edge_pos width.");
    const int64_t state_count = state_limit_arg;
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    require_optional_state_width(state_wi, state_count, "state_wi");
    const int64_t material_count = material_gain.size(0);
    if (material_valid.size(0) != material_count)
        throw std::runtime_error("material_valid must match material_gain width.");

    SceneCache& scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3Input state_edge_pos_view = vec3_input(state_edge_pos, "state_edge_pos");
    Vec3Input state_edge_dir_view = vec3_input(state_edge_dir, "state_edge_dir");
    Vec3Input state_src_view = vec3_input(state_src, "state_src");
    Vec3Input state_wi_view = optional_vec3_input(state_wi, "state_wi");
    GridGradInput grad_power_view =
        optional_grid_grad_input(grad_power, grid_resolution0, grid_resolution1, "grad_power");
    GridGradInput grad_field_x_re_view =
        optional_grid_grad_input(grad_field_x_re, grid_resolution0, grid_resolution1, "grad_field_x_re");

    at::Tensor grad_edge_pos = at::zeros(state_edge_pos.sizes(), state_edge_pos.options());
    at::Tensor grad_edge_dir = at::zeros(state_edge_dir.sizes(), state_edge_dir.options());
    at::Tensor grad_edge_t_min = at::zeros_like(state_edge_t_min);
    at::Tensor grad_edge_t_max = at::zeros_like(state_edge_t_max);
    at::Tensor grad_src = at::zeros(state_src.sizes(), state_src.options());
    const bool state_wi_present = has_optional_tensor(state_wi);
    at::Tensor grad_wi = state_wi_present ? at::zeros(state_wi->sizes(), state_wi->options()) : at::Tensor();
    at::Tensor grad_src_power = at::zeros_like(state_src_power);
    at::Tensor grad_exterior_angle = at::zeros_like(state_exterior_angle);
    at::Tensor grad_material_gain = at::zeros_like(material_gain);
    at::Tensor grad_tri_p0_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_z = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_z = at::zeros({tri.n_triangles}, state_edge_pos.options());
    Vec3Output grad_edge_pos_view = vec3_output(grad_edge_pos, "grad_edge_pos");
    Vec3Output grad_edge_dir_view = vec3_output(grad_edge_dir, "grad_edge_dir");
    Vec3Output grad_src_view = vec3_output(grad_src, "grad_src");
    Vec3Output grad_wi_view = state_wi_present ? vec3_output(grad_wi, "grad_wi") : Vec3Output();

    DfrDirectAccumADParams params = {};
    // The tape rows are the shard's local lanes; the AD body replays them at the
    // global lanes [lane_offset, lane_offset + rows) the forward launch used.
    if (lane_offset < 0)
        throw std::runtime_error("lane_offset must be non-negative.");
    params.lane_offset = checked_i32(lane_offset, "lane_offset");
    params.n_rays = checked_i32(lane_offset + launch_count, "n_rays");
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
    params.tape_active = rebase_lane_buffer(mask_ptr(tape_active), params.lane_offset);
    params.tape_state_idx = rebase_lane_buffer(tape_state_idx.data_ptr<int>(), params.lane_offset);
    params.tape_cell = rebase_lane_buffer(tape_cell.data_ptr<int>(), params.lane_offset);
    params.tape_material_idx = rebase_lane_buffer(tape_material_idx.data_ptr<int>(), params.lane_offset);
    params.tape_edge_u = rebase_lane_buffer(tape_edge_u.data_ptr<float>(), params.lane_offset);
    params.state_edge_pos_x = state_edge_pos_view.x;
    params.state_edge_pos_y = state_edge_pos_view.y;
    params.state_edge_pos_z = state_edge_pos_view.z;
    params.state_edge_pos_stride = state_edge_pos_view.stride;
    params.state_edge_dir_x = state_edge_dir_view.x;
    params.state_edge_dir_y = state_edge_dir_view.y;
    params.state_edge_dir_z = state_edge_dir_view.z;
    params.state_edge_dir_stride = state_edge_dir_view.stride;
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_min_stride = stride_i32(state_edge_t_min, 0, "state_edge_t_min_stride");
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_edge_t_max_stride = stride_i32(state_edge_t_max, 0, "state_edge_t_max_stride");
    params.state_src_x = state_src_view.x;
    params.state_src_y = state_src_view.y;
    params.state_src_z = state_src_view.z;
    params.state_src_stride = state_src_view.stride;
    params.state_wi_x = state_wi_view.x;
    params.state_wi_y = state_wi_view.y;
    params.state_wi_z = state_wi_view.z;
    params.state_wi_stride = state_wi_view.stride;
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_src_power_stride = stride_i32(state_src_power, 0, "state_src_power_stride");
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_exterior_angle_stride = stride_i32(state_exterior_angle, 0, "state_exterior_angle_stride");
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim0_stride = stride_i32(state_prim0, 0, "state_prim0_stride");
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_prim1_stride = stride_i32(state_prim1, 0, "state_prim1_stride");
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
    params.material_gain_stride = stride_i32(material_gain, 0, "material_gain_stride");
    params.material_valid = mask_ptr(material_valid);
    params.material_valid_stride = stride_i32(material_valid, 0, "material_valid_stride");
    params.grad_out_power = grad_power_view.ptr;
    params.grad_out_power_rank = grad_power_view.rank;
    params.grad_out_power_stride0 = grad_power_view.stride0;
    params.grad_out_power_stride1 = grad_power_view.stride1;
    params.grad_out_field_x_re = grad_field_x_re_view.ptr;
    params.grad_out_field_x_re_rank = grad_field_x_re_view.rank;
    params.grad_out_field_x_re_stride0 = grad_field_x_re_view.stride0;
    params.grad_out_field_x_re_stride1 = grad_field_x_re_view.stride1;
    params.grad_state_edge_pos_x = grad_edge_pos_view.x;
    params.grad_state_edge_pos_y = grad_edge_pos_view.y;
    params.grad_state_edge_pos_z = grad_edge_pos_view.z;
    params.grad_state_edge_pos_stride = grad_edge_pos_view.stride;
    params.grad_state_edge_dir_x = grad_edge_dir_view.x;
    params.grad_state_edge_dir_y = grad_edge_dir_view.y;
    params.grad_state_edge_dir_z = grad_edge_dir_view.z;
    params.grad_state_edge_dir_stride = grad_edge_dir_view.stride;
    params.grad_state_edge_t_min = grad_edge_t_min.data_ptr<float>();
    params.grad_state_edge_t_min_stride = stride_i32(grad_edge_t_min, 0, "grad_state_edge_t_min_stride");
    params.grad_state_edge_t_max = grad_edge_t_max.data_ptr<float>();
    params.grad_state_edge_t_max_stride = stride_i32(grad_edge_t_max, 0, "grad_state_edge_t_max_stride");
    params.grad_state_src_x = grad_src_view.x;
    params.grad_state_src_y = grad_src_view.y;
    params.grad_state_src_z = grad_src_view.z;
    params.grad_state_src_stride = grad_src_view.stride;
    params.grad_state_wi_x = grad_wi_view.x;
    params.grad_state_wi_y = grad_wi_view.y;
    params.grad_state_wi_z = grad_wi_view.z;
    params.grad_state_wi_stride = grad_wi_view.stride;
    params.grad_state_src_power = grad_src_power.data_ptr<float>();
    params.grad_state_src_power_stride = stride_i32(grad_src_power, 0, "grad_state_src_power_stride");
    params.grad_state_exterior_angle = grad_exterior_angle.data_ptr<float>();
    params.grad_state_exterior_angle_stride = stride_i32(grad_exterior_angle, 0, "grad_state_exterior_angle_stride");
    params.grad_material_gain = grad_material_gain.data_ptr<float>();
    params.grad_material_gain_stride = stride_i32(grad_material_gain, 0, "grad_material_gain_stride");
    params.grad_tri_p0_x = grad_tri_p0_x.data_ptr<float>();
    params.grad_tri_p0_y = grad_tri_p0_y.data_ptr<float>();
    params.grad_tri_p0_z = grad_tri_p0_z.data_ptr<float>();
    params.grad_tri_fn_x = grad_tri_fn_x.data_ptr<float>();
    params.grad_tri_fn_y = grad_tri_fn_y.data_ptr<float>();
    params.grad_tri_fn_z = grad_tri_fn_z.data_ptr<float>();
    dfr_direct_accum_vjp_gpu(params);
    py::object grad_wi_obj = state_wi_present ? py::cast(grad_wi) : py::none();
    return py::make_tuple(grad_edge_pos, grad_edge_dir, grad_edge_t_min, grad_edge_t_max, grad_src, grad_wi_obj,
                          grad_src_power, grad_exterior_angle, grad_material_gain);
}

py::tuple diffraction_accumulation_direct_jvp_op(
    int64_t scene_handle, at::Tensor tape_active, at::Tensor tape_state_idx, at::Tensor tape_cell,
    at::Tensor tape_material_idx, at::Tensor tape_edge_u, at::Tensor state_edge_pos, at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_prim0, at::Tensor state_prim1,
    at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power,
    c10::optional<at::Tensor> state_wi, at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit_arg,
    int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min,
    double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area,
    double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed,
    c10::optional<at::Tensor> dot_state_edge_pos, c10::optional<at::Tensor> dot_state_edge_dir,
    c10::optional<at::Tensor> dot_state_edge_t_min, c10::optional<at::Tensor> dot_state_edge_t_max,
    c10::optional<at::Tensor> dot_state_exterior_angle, c10::optional<at::Tensor> dot_state_src,
    c10::optional<at::Tensor> dot_state_src_power, c10::optional<at::Tensor> dot_state_wi,
    c10::optional<at::Tensor> dot_material_gain, int64_t lane_offset) {
    require_mask(tape_active, "tape_active");
    require_flat_i32(tape_state_idx, "tape_state_idx");
    require_flat_i32(tape_cell, "tape_cell");
    require_flat_i32(tape_material_idx, "tape_material_idx");
    require_flat_f32(tape_edge_u, "tape_edge_u");
    require_vec3f_strided(state_edge_pos, "state_edge_pos");
    require_vec3f_strided(state_edge_dir, "state_edge_dir");
    require_flat_f32_strided(state_edge_t_min, "state_edge_t_min");
    require_flat_f32_strided(state_edge_t_max, "state_edge_t_max");
    require_flat_i32_strided(state_prim0, "state_prim0");
    require_flat_i32_strided(state_prim1, "state_prim1");
    require_flat_f32_strided(state_exterior_angle, "state_exterior_angle");
    require_vec3f_strided(state_src, "state_src");
    require_flat_f32_strided(state_src_power, "state_src_power");
    require_optional_vec3f_strided(state_wi, "state_wi");
    require_flat_f32_strided(material_gain, "material_gain");
    require_mask_strided(material_valid, "material_valid");
    require_optional_vec3f_strided(dot_state_edge_pos, "dot_state_edge_pos");
    require_optional_vec3f_strided(dot_state_edge_dir, "dot_state_edge_dir");
    require_optional_scalar_f_strided(dot_state_edge_t_min, "dot_state_edge_t_min");
    require_optional_scalar_f_strided(dot_state_edge_t_max, "dot_state_edge_t_max");
    require_optional_scalar_f_strided(dot_state_exterior_angle, "dot_state_exterior_angle");
    require_optional_vec3f_strided(dot_state_src, "dot_state_src");
    require_optional_scalar_f_strided(dot_state_src_power, "dot_state_src_power");
    require_optional_vec3f_strided(dot_state_wi, "dot_state_wi");
    require_optional_scalar_f_strided(dot_material_gain, "dot_material_gain");
    SceneCache& scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3Input state_edge_pos_view = vec3_input(state_edge_pos, "state_edge_pos");
    Vec3Input state_edge_dir_view = vec3_input(state_edge_dir, "state_edge_dir");
    Vec3Input state_src_view = vec3_input(state_src, "state_src");
    Vec3Input state_wi_view = optional_vec3_input(state_wi, "state_wi");
    Vec3Input dot_edge_pos_view = optional_vec3_input(dot_state_edge_pos, "dot_state_edge_pos");
    Vec3Input dot_edge_dir_view = optional_vec3_input(dot_state_edge_dir, "dot_state_edge_dir");
    Vec3Input dot_src_view = optional_vec3_input(dot_state_src, "dot_state_src");
    Vec3Input dot_wi_view = optional_vec3_input(dot_state_wi, "dot_state_wi");
    const int64_t launch_count = tape_active.size(0);
    require_state_width(tape_state_idx, launch_count, "tape_state_idx");
    require_state_width(tape_cell, launch_count, "tape_cell");
    require_state_width(tape_material_idx, launch_count, "tape_material_idx");
    require_state_width(tape_edge_u, launch_count, "tape_edge_u");
    if (state_limit_arg < 0)
        throw std::runtime_error("state_limit must be non-negative.");
    const int64_t state_physical_count = state_edge_pos.size(0);
    if (state_limit_arg > state_physical_count)
        throw std::runtime_error("state_limit must not exceed state_edge_pos width.");
    const int64_t state_count = state_limit_arg;
    const int64_t material_count = material_gain.size(0);
    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    require_optional_state_width(state_wi, state_count, "state_wi");
    require_optional_state_width(dot_state_edge_pos, state_count, "dot_state_edge_pos");
    require_optional_state_width(dot_state_edge_dir, state_count, "dot_state_edge_dir");
    require_optional_state_width(dot_state_edge_t_min, state_count, "dot_state_edge_t_min");
    require_optional_state_width(dot_state_edge_t_max, state_count, "dot_state_edge_t_max");
    require_optional_state_width(dot_state_exterior_angle, state_count, "dot_state_exterior_angle");
    require_optional_state_width(dot_state_src, state_count, "dot_state_src");
    require_optional_state_width(dot_state_src_power, state_count, "dot_state_src_power");
    require_optional_state_width(dot_state_wi, state_count, "dot_state_wi");
    require_optional_state_width(dot_material_gain, material_count, "dot_material_gain");
    if (material_valid.size(0) != material_count)
        throw std::runtime_error("material_valid must match material_gain width.");
    at::Tensor dot_power = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor dot_field_x_re = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor dot_zero = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor zero_tri = at::zeros({tri.n_triangles}, state_edge_pos.options());

    DfrDirectAccumADParams params = {};
    // The tape rows are the shard's local lanes; the AD body replays them at the
    // global lanes [lane_offset, lane_offset + rows) the forward launch used.
    if (lane_offset < 0)
        throw std::runtime_error("lane_offset must be non-negative.");
    params.lane_offset = checked_i32(lane_offset, "lane_offset");
    params.n_rays = checked_i32(lane_offset + launch_count, "n_rays");
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
    params.tape_active = rebase_lane_buffer(mask_ptr(tape_active), params.lane_offset);
    params.tape_state_idx = rebase_lane_buffer(tape_state_idx.data_ptr<int>(), params.lane_offset);
    params.tape_cell = rebase_lane_buffer(tape_cell.data_ptr<int>(), params.lane_offset);
    params.tape_material_idx = rebase_lane_buffer(tape_material_idx.data_ptr<int>(), params.lane_offset);
    params.tape_edge_u = rebase_lane_buffer(tape_edge_u.data_ptr<float>(), params.lane_offset);
    params.state_edge_pos_x = state_edge_pos_view.x;
    params.state_edge_pos_y = state_edge_pos_view.y;
    params.state_edge_pos_z = state_edge_pos_view.z;
    params.state_edge_pos_stride = state_edge_pos_view.stride;
    params.state_edge_dir_x = state_edge_dir_view.x;
    params.state_edge_dir_y = state_edge_dir_view.y;
    params.state_edge_dir_z = state_edge_dir_view.z;
    params.state_edge_dir_stride = state_edge_dir_view.stride;
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_min_stride = stride_i32(state_edge_t_min, 0, "state_edge_t_min_stride");
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_edge_t_max_stride = stride_i32(state_edge_t_max, 0, "state_edge_t_max_stride");
    params.state_src_x = state_src_view.x;
    params.state_src_y = state_src_view.y;
    params.state_src_z = state_src_view.z;
    params.state_src_stride = state_src_view.stride;
    params.state_wi_x = state_wi_view.x;
    params.state_wi_y = state_wi_view.y;
    params.state_wi_z = state_wi_view.z;
    params.state_wi_stride = state_wi_view.stride;
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_src_power_stride = stride_i32(state_src_power, 0, "state_src_power_stride");
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_exterior_angle_stride = stride_i32(state_exterior_angle, 0, "state_exterior_angle_stride");
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim0_stride = stride_i32(state_prim0, 0, "state_prim0_stride");
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_prim1_stride = stride_i32(state_prim1, 0, "state_prim1_stride");
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
    params.material_gain_stride = stride_i32(material_gain, 0, "material_gain_stride");
    params.material_valid = mask_ptr(material_valid);
    params.material_valid_stride = stride_i32(material_valid, 0, "material_valid_stride");
    params.dot_state_edge_pos_x = dot_edge_pos_view.x;
    params.dot_state_edge_pos_y = dot_edge_pos_view.y;
    params.dot_state_edge_pos_z = dot_edge_pos_view.z;
    params.dot_state_edge_pos_stride = dot_edge_pos_view.stride;
    params.dot_state_edge_dir_x = dot_edge_dir_view.x;
    params.dot_state_edge_dir_y = dot_edge_dir_view.y;
    params.dot_state_edge_dir_z = dot_edge_dir_view.z;
    params.dot_state_edge_dir_stride = dot_edge_dir_view.stride;
    params.dot_state_edge_t_min = optional_scalar_ptr(dot_state_edge_t_min);
    params.dot_state_edge_t_min_stride = optional_scalar_stride(dot_state_edge_t_min, "dot_state_edge_t_min_stride");
    params.dot_state_edge_t_max = optional_scalar_ptr(dot_state_edge_t_max);
    params.dot_state_edge_t_max_stride = optional_scalar_stride(dot_state_edge_t_max, "dot_state_edge_t_max_stride");
    params.dot_state_src_x = dot_src_view.x;
    params.dot_state_src_y = dot_src_view.y;
    params.dot_state_src_z = dot_src_view.z;
    params.dot_state_src_stride = dot_src_view.stride;
    params.dot_state_wi_x = dot_wi_view.x;
    params.dot_state_wi_y = dot_wi_view.y;
    params.dot_state_wi_z = dot_wi_view.z;
    params.dot_state_wi_stride = dot_wi_view.stride;
    params.dot_state_src_power = optional_scalar_ptr(dot_state_src_power);
    params.dot_state_src_power_stride = optional_scalar_stride(dot_state_src_power, "dot_state_src_power_stride");
    params.dot_state_exterior_angle = optional_scalar_ptr(dot_state_exterior_angle);
    params.dot_state_exterior_angle_stride =
        optional_scalar_stride(dot_state_exterior_angle, "dot_state_exterior_angle_stride");
    params.dot_material_gain = optional_scalar_ptr(dot_material_gain);
    params.dot_material_gain_stride = optional_scalar_stride(dot_material_gain, "dot_material_gain_stride");
    params.dot_tri_p0_x = zero_tri.data_ptr<float>();
    params.dot_tri_p0_y = zero_tri.data_ptr<float>();
    params.dot_tri_p0_z = zero_tri.data_ptr<float>();
    params.dot_tri_fn_x = zero_tri.data_ptr<float>();
    params.dot_tri_fn_y = zero_tri.data_ptr<float>();
    params.dot_tri_fn_z = zero_tri.data_ptr<float>();
    params.dot_out_power = dot_power.data_ptr<float>();
    params.dot_out_field_x_re = dot_field_x_re.data_ptr<float>();
    dfr_direct_accum_jvp_gpu(params);
    return py::make_tuple(dot_power.reshape({grid_resolution1, grid_resolution0}),
                          dot_field_x_re.reshape({grid_resolution1, grid_resolution0}),
                          dot_zero.reshape({grid_resolution1, grid_resolution0}));
}

py::tuple diffraction_accumulation_chain_backward_op(
    int64_t scene_handle, at::Tensor tape_active, at::Tensor tape_cell, at::Tensor state_edge_index,
    at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max,
    at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src,
    at::Tensor state_src_power, at::Tensor recursive_state_edge_index, at::Tensor recursive_state_edge_pos,
    at::Tensor recursive_state_edge_dir, at::Tensor recursive_state_edge_t_min, at::Tensor recursive_state_edge_t_max,
    at::Tensor recursive_state_prim0, at::Tensor recursive_state_prim1, at::Tensor recursive_state_exterior_angle,
    at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit_arg, int64_t recursive_state_limit_arg,
    int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min,
    double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area,
    double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed,
    int64_t max_order, c10::optional<at::Tensor> grad_power, c10::optional<at::Tensor> grad_field_x_re,
    int64_t lane_offset) {
    require_mask(tape_active, "tape_active");
    require_flat_i32(tape_cell, "tape_cell");
    require_flat_i32_strided(state_edge_index, "state_edge_index");
    require_vec3f_strided(state_edge_pos, "state_edge_pos");
    require_vec3f_strided(state_edge_dir, "state_edge_dir");
    require_flat_f32_strided(state_edge_t_min, "state_edge_t_min");
    require_flat_f32_strided(state_edge_t_max, "state_edge_t_max");
    require_flat_i32_strided(state_prim0, "state_prim0");
    require_flat_i32_strided(state_prim1, "state_prim1");
    require_flat_f32_strided(state_exterior_angle, "state_exterior_angle");
    require_vec3f_strided(state_src, "state_src");
    require_flat_f32_strided(state_src_power, "state_src_power");
    require_flat_i32_strided(recursive_state_edge_index, "recursive_state_edge_index");
    require_vec3f_strided(recursive_state_edge_pos, "recursive_state_edge_pos");
    require_vec3f_strided(recursive_state_edge_dir, "recursive_state_edge_dir");
    require_flat_f32_strided(recursive_state_edge_t_min, "recursive_state_edge_t_min");
    require_flat_f32_strided(recursive_state_edge_t_max, "recursive_state_edge_t_max");
    require_flat_i32_strided(recursive_state_prim0, "recursive_state_prim0");
    require_flat_i32_strided(recursive_state_prim1, "recursive_state_prim1");
    require_flat_f32_strided(recursive_state_exterior_angle, "recursive_state_exterior_angle");
    require_flat_f32_strided(material_gain, "material_gain");
    require_mask_strided(material_valid, "material_valid");
    const int64_t launch_count = tape_active.size(0);
    require_state_width(tape_cell, launch_count, "tape_cell");
    if (state_limit_arg < 0 || recursive_state_limit_arg < 0)
        throw std::runtime_error("state counts must be non-negative.");
    const int64_t state_count = state_limit_arg;
    const int64_t recursive_state_count = recursive_state_limit_arg;
    if (state_count > state_edge_pos.size(0))
        throw std::runtime_error("state_count must not exceed state_edge_pos width.");
    if (recursive_state_count > recursive_state_edge_pos.size(0))
        throw std::runtime_error("recursive_state_count must not exceed recursive_state_edge_pos width.");
    require_state_width(state_edge_index, state_count, "state_edge_index");
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    require_state_width(recursive_state_edge_index, recursive_state_count, "recursive_state_edge_index");
    require_state_width(recursive_state_edge_dir, recursive_state_count, "recursive_state_edge_dir");
    require_state_width(recursive_state_edge_t_min, recursive_state_count, "recursive_state_edge_t_min");
    require_state_width(recursive_state_edge_t_max, recursive_state_count, "recursive_state_edge_t_max");
    require_state_width(recursive_state_prim0, recursive_state_count, "recursive_state_prim0");
    require_state_width(recursive_state_prim1, recursive_state_count, "recursive_state_prim1");
    require_state_width(recursive_state_exterior_angle, recursive_state_count, "recursive_state_exterior_angle");
    const int64_t material_count = material_gain.size(0);
    if (material_valid.size(0) != material_count)
        throw std::runtime_error("material_valid must match material_gain width.");

    SceneCache& scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3Input state_edge_pos_view = vec3_input(state_edge_pos, "state_edge_pos");
    Vec3Input state_edge_dir_view = vec3_input(state_edge_dir, "state_edge_dir");
    Vec3Input state_src_view = vec3_input(state_src, "state_src");
    Vec3Input recursive_edge_pos_view = vec3_input(recursive_state_edge_pos, "recursive_state_edge_pos");
    Vec3Input recursive_edge_dir_view = vec3_input(recursive_state_edge_dir, "recursive_state_edge_dir");
    GridGradInput grad_power_view =
        optional_grid_grad_input(grad_power, grid_resolution0, grid_resolution1, "grad_power");
    GridGradInput grad_field_x_re_view =
        optional_grid_grad_input(grad_field_x_re, grid_resolution0, grid_resolution1, "grad_field_x_re");

    at::Tensor grad_edge_pos = at::zeros(state_edge_pos.sizes(), state_edge_pos.options());
    at::Tensor grad_edge_dir = at::zeros(state_edge_dir.sizes(), state_edge_dir.options());
    at::Tensor grad_edge_t_min = at::zeros_like(state_edge_t_min);
    at::Tensor grad_edge_t_max = at::zeros_like(state_edge_t_max);
    at::Tensor grad_src = at::zeros(state_src.sizes(), state_src.options());
    at::Tensor grad_src_power = at::zeros_like(state_src_power);
    at::Tensor grad_exterior_angle = at::zeros_like(state_exterior_angle);
    at::Tensor grad_recursive_edge_pos =
        at::zeros(recursive_state_edge_pos.sizes(), recursive_state_edge_pos.options());
    at::Tensor grad_recursive_edge_dir =
        at::zeros(recursive_state_edge_dir.sizes(), recursive_state_edge_dir.options());
    at::Tensor grad_recursive_edge_t_min = at::zeros_like(recursive_state_edge_t_min);
    at::Tensor grad_recursive_edge_t_max = at::zeros_like(recursive_state_edge_t_max);
    at::Tensor grad_recursive_exterior_angle = at::zeros_like(recursive_state_exterior_angle);
    at::Tensor grad_material_gain = at::zeros_like(material_gain);
    at::Tensor grad_tri_p0_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_z = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_z = at::zeros({tri.n_triangles}, state_edge_pos.options());
    Vec3Output grad_edge_pos_view = vec3_output(grad_edge_pos, "grad_edge_pos");
    Vec3Output grad_edge_dir_view = vec3_output(grad_edge_dir, "grad_edge_dir");
    Vec3Output grad_src_view = vec3_output(grad_src, "grad_src");
    Vec3Output grad_recursive_edge_pos_view = vec3_output(grad_recursive_edge_pos, "grad_recursive_edge_pos");
    Vec3Output grad_recursive_edge_dir_view = vec3_output(grad_recursive_edge_dir, "grad_recursive_edge_dir");

    DfrChainAccumADParams params = {};
    // The tape rows are the shard's local lanes; the AD body replays them at the
    // global lanes [lane_offset, lane_offset + rows) the forward launch used.
    if (lane_offset < 0)
        throw std::runtime_error("lane_offset must be non-negative.");
    params.lane_offset = checked_i32(lane_offset, "lane_offset");
    params.n_rays = checked_i32(lane_offset + launch_count, "n_rays");
    params.state_count = checked_i32(state_count, "state_count");
    params.recursive_state_count = checked_i32(recursive_state_count, "recursive_state_count");
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
    params.max_order = checked_i32(max_order, "max_order");
    params.wavelength = static_cast<float>(wavelength);
    params.seed = checked_i32(seed, "seed");
    params.n_triangles = tri.n_triangles;
    params.tape_active = rebase_lane_buffer(mask_ptr(tape_active), params.lane_offset);
    params.tape_cell = rebase_lane_buffer(tape_cell.data_ptr<int>(), params.lane_offset);
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_index_stride = stride_i32(state_edge_index, 0, "state_edge_index_stride");
    params.state_edge_pos_x = state_edge_pos_view.x;
    params.state_edge_pos_y = state_edge_pos_view.y;
    params.state_edge_pos_z = state_edge_pos_view.z;
    params.state_edge_pos_stride = state_edge_pos_view.stride;
    params.state_edge_dir_x = state_edge_dir_view.x;
    params.state_edge_dir_y = state_edge_dir_view.y;
    params.state_edge_dir_z = state_edge_dir_view.z;
    params.state_edge_dir_stride = state_edge_dir_view.stride;
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_min_stride = stride_i32(state_edge_t_min, 0, "state_edge_t_min_stride");
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_edge_t_max_stride = stride_i32(state_edge_t_max, 0, "state_edge_t_max_stride");
    params.state_src_x = state_src_view.x;
    params.state_src_y = state_src_view.y;
    params.state_src_z = state_src_view.z;
    params.state_src_stride = state_src_view.stride;
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_src_power_stride = stride_i32(state_src_power, 0, "state_src_power_stride");
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_exterior_angle_stride = stride_i32(state_exterior_angle, 0, "state_exterior_angle_stride");
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim0_stride = stride_i32(state_prim0, 0, "state_prim0_stride");
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_prim1_stride = stride_i32(state_prim1, 0, "state_prim1_stride");
    params.recursive_state_edge_index = recursive_state_edge_index.data_ptr<int>();
    params.recursive_state_edge_index_stride =
        stride_i32(recursive_state_edge_index, 0, "recursive_state_edge_index_stride");
    params.recursive_state_edge_pos_x = recursive_edge_pos_view.x;
    params.recursive_state_edge_pos_y = recursive_edge_pos_view.y;
    params.recursive_state_edge_pos_z = recursive_edge_pos_view.z;
    params.recursive_state_edge_pos_stride = recursive_edge_pos_view.stride;
    params.recursive_state_edge_dir_x = recursive_edge_dir_view.x;
    params.recursive_state_edge_dir_y = recursive_edge_dir_view.y;
    params.recursive_state_edge_dir_z = recursive_edge_dir_view.z;
    params.recursive_state_edge_dir_stride = recursive_edge_dir_view.stride;
    params.recursive_state_edge_t_min = recursive_state_edge_t_min.data_ptr<float>();
    params.recursive_state_edge_t_min_stride =
        stride_i32(recursive_state_edge_t_min, 0, "recursive_state_edge_t_min_stride");
    params.recursive_state_edge_t_max = recursive_state_edge_t_max.data_ptr<float>();
    params.recursive_state_edge_t_max_stride =
        stride_i32(recursive_state_edge_t_max, 0, "recursive_state_edge_t_max_stride");
    params.recursive_state_exterior_angle = recursive_state_exterior_angle.data_ptr<float>();
    params.recursive_state_exterior_angle_stride =
        stride_i32(recursive_state_exterior_angle, 0, "recursive_state_exterior_angle_stride");
    params.recursive_state_prim0 = recursive_state_prim0.data_ptr<int>();
    params.recursive_state_prim0_stride = stride_i32(recursive_state_prim0, 0, "recursive_state_prim0_stride");
    params.recursive_state_prim1 = recursive_state_prim1.data_ptr<int>();
    params.recursive_state_prim1_stride = stride_i32(recursive_state_prim1, 0, "recursive_state_prim1_stride");
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
    params.material_gain_stride = stride_i32(material_gain, 0, "material_gain_stride");
    params.material_valid = mask_ptr(material_valid);
    params.material_valid_stride = stride_i32(material_valid, 0, "material_valid_stride");
    params.grad_out_power = grad_power_view.ptr;
    params.grad_out_power_rank = grad_power_view.rank;
    params.grad_out_power_stride0 = grad_power_view.stride0;
    params.grad_out_power_stride1 = grad_power_view.stride1;
    params.grad_out_field_x_re = grad_field_x_re_view.ptr;
    params.grad_out_field_x_re_rank = grad_field_x_re_view.rank;
    params.grad_out_field_x_re_stride0 = grad_field_x_re_view.stride0;
    params.grad_out_field_x_re_stride1 = grad_field_x_re_view.stride1;
    params.grad_state_edge_pos_x = grad_edge_pos_view.x;
    params.grad_state_edge_pos_y = grad_edge_pos_view.y;
    params.grad_state_edge_pos_z = grad_edge_pos_view.z;
    params.grad_state_edge_pos_stride = grad_edge_pos_view.stride;
    params.grad_state_edge_dir_x = grad_edge_dir_view.x;
    params.grad_state_edge_dir_y = grad_edge_dir_view.y;
    params.grad_state_edge_dir_z = grad_edge_dir_view.z;
    params.grad_state_edge_dir_stride = grad_edge_dir_view.stride;
    params.grad_state_edge_t_min = grad_edge_t_min.data_ptr<float>();
    params.grad_state_edge_t_min_stride = stride_i32(grad_edge_t_min, 0, "grad_state_edge_t_min_stride");
    params.grad_state_edge_t_max = grad_edge_t_max.data_ptr<float>();
    params.grad_state_edge_t_max_stride = stride_i32(grad_edge_t_max, 0, "grad_state_edge_t_max_stride");
    params.grad_state_src_x = grad_src_view.x;
    params.grad_state_src_y = grad_src_view.y;
    params.grad_state_src_z = grad_src_view.z;
    params.grad_state_src_stride = grad_src_view.stride;
    params.grad_state_src_power = grad_src_power.data_ptr<float>();
    params.grad_state_src_power_stride = stride_i32(grad_src_power, 0, "grad_state_src_power_stride");
    params.grad_state_exterior_angle = grad_exterior_angle.data_ptr<float>();
    params.grad_state_exterior_angle_stride = stride_i32(grad_exterior_angle, 0, "grad_state_exterior_angle_stride");
    params.grad_recursive_state_edge_pos_x = grad_recursive_edge_pos_view.x;
    params.grad_recursive_state_edge_pos_y = grad_recursive_edge_pos_view.y;
    params.grad_recursive_state_edge_pos_z = grad_recursive_edge_pos_view.z;
    params.grad_recursive_state_edge_pos_stride = grad_recursive_edge_pos_view.stride;
    params.grad_recursive_state_edge_dir_x = grad_recursive_edge_dir_view.x;
    params.grad_recursive_state_edge_dir_y = grad_recursive_edge_dir_view.y;
    params.grad_recursive_state_edge_dir_z = grad_recursive_edge_dir_view.z;
    params.grad_recursive_state_edge_dir_stride = grad_recursive_edge_dir_view.stride;
    params.grad_recursive_state_edge_t_min = grad_recursive_edge_t_min.data_ptr<float>();
    params.grad_recursive_state_edge_t_min_stride =
        stride_i32(grad_recursive_edge_t_min, 0, "grad_recursive_state_edge_t_min_stride");
    params.grad_recursive_state_edge_t_max = grad_recursive_edge_t_max.data_ptr<float>();
    params.grad_recursive_state_edge_t_max_stride =
        stride_i32(grad_recursive_edge_t_max, 0, "grad_recursive_state_edge_t_max_stride");
    params.grad_recursive_state_exterior_angle = grad_recursive_exterior_angle.data_ptr<float>();
    params.grad_recursive_state_exterior_angle_stride =
        stride_i32(grad_recursive_exterior_angle, 0, "grad_recursive_state_exterior_angle_stride");
    params.grad_material_gain = grad_material_gain.data_ptr<float>();
    params.grad_material_gain_stride = stride_i32(grad_material_gain, 0, "grad_material_gain_stride");
    params.grad_tri_p0_x = grad_tri_p0_x.data_ptr<float>();
    params.grad_tri_p0_y = grad_tri_p0_y.data_ptr<float>();
    params.grad_tri_p0_z = grad_tri_p0_z.data_ptr<float>();
    params.grad_tri_fn_x = grad_tri_fn_x.data_ptr<float>();
    params.grad_tri_fn_y = grad_tri_fn_y.data_ptr<float>();
    params.grad_tri_fn_z = grad_tri_fn_z.data_ptr<float>();
    dfr_chain_accum_vjp_gpu(params);
    return py::make_tuple(grad_edge_pos, grad_edge_dir, grad_edge_t_min, grad_edge_t_max, grad_src, grad_src_power,
                          grad_exterior_angle, grad_recursive_edge_pos, grad_recursive_edge_dir,
                          grad_recursive_edge_t_min, grad_recursive_edge_t_max, grad_recursive_exterior_angle,
                          grad_material_gain);
}

py::tuple diffraction_accumulation_chain_jvp_op(
    int64_t scene_handle, at::Tensor tape_active, at::Tensor tape_cell, at::Tensor state_edge_index,
    at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max,
    at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src,
    at::Tensor state_src_power, at::Tensor recursive_state_edge_index, at::Tensor recursive_state_edge_pos,
    at::Tensor recursive_state_edge_dir, at::Tensor recursive_state_edge_t_min, at::Tensor recursive_state_edge_t_max,
    at::Tensor recursive_state_prim0, at::Tensor recursive_state_prim1, at::Tensor recursive_state_exterior_angle,
    at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit_arg, int64_t recursive_state_limit_arg,
    int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min,
    double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area,
    double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed,
    int64_t max_order, c10::optional<at::Tensor> dot_state_edge_pos, c10::optional<at::Tensor> dot_state_edge_dir,
    c10::optional<at::Tensor> dot_state_edge_t_min, c10::optional<at::Tensor> dot_state_edge_t_max,
    c10::optional<at::Tensor> dot_state_exterior_angle, c10::optional<at::Tensor> dot_state_src,
    c10::optional<at::Tensor> dot_state_src_power, c10::optional<at::Tensor> dot_recursive_state_edge_pos,
    c10::optional<at::Tensor> dot_recursive_state_edge_dir, c10::optional<at::Tensor> dot_recursive_state_edge_t_min,
    c10::optional<at::Tensor> dot_recursive_state_edge_t_max,
    c10::optional<at::Tensor> dot_recursive_state_exterior_angle, c10::optional<at::Tensor> dot_material_gain,
    int64_t lane_offset) {
    require_mask(tape_active, "tape_active");
    require_flat_i32(tape_cell, "tape_cell");
    require_flat_i32_strided(state_edge_index, "state_edge_index");
    require_vec3f_strided(state_edge_pos, "state_edge_pos");
    require_vec3f_strided(state_edge_dir, "state_edge_dir");
    require_flat_f32_strided(state_edge_t_min, "state_edge_t_min");
    require_flat_f32_strided(state_edge_t_max, "state_edge_t_max");
    require_flat_i32_strided(state_prim0, "state_prim0");
    require_flat_i32_strided(state_prim1, "state_prim1");
    require_flat_f32_strided(state_exterior_angle, "state_exterior_angle");
    require_vec3f_strided(state_src, "state_src");
    require_flat_f32_strided(state_src_power, "state_src_power");
    require_flat_i32_strided(recursive_state_edge_index, "recursive_state_edge_index");
    require_vec3f_strided(recursive_state_edge_pos, "recursive_state_edge_pos");
    require_vec3f_strided(recursive_state_edge_dir, "recursive_state_edge_dir");
    require_flat_f32_strided(recursive_state_edge_t_min, "recursive_state_edge_t_min");
    require_flat_f32_strided(recursive_state_edge_t_max, "recursive_state_edge_t_max");
    require_flat_i32_strided(recursive_state_prim0, "recursive_state_prim0");
    require_flat_i32_strided(recursive_state_prim1, "recursive_state_prim1");
    require_flat_f32_strided(recursive_state_exterior_angle, "recursive_state_exterior_angle");
    require_flat_f32_strided(material_gain, "material_gain");
    require_mask_strided(material_valid, "material_valid");
    SceneCache& scene = get_scene(scene_handle);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3Input state_edge_pos_view = vec3_input(state_edge_pos, "state_edge_pos");
    Vec3Input state_edge_dir_view = vec3_input(state_edge_dir, "state_edge_dir");
    Vec3Input state_src_view = vec3_input(state_src, "state_src");
    Vec3Input recursive_edge_pos_view = vec3_input(recursive_state_edge_pos, "recursive_state_edge_pos");
    Vec3Input recursive_edge_dir_view = vec3_input(recursive_state_edge_dir, "recursive_state_edge_dir");
    require_optional_vec3f_strided(dot_state_edge_pos, "dot_state_edge_pos");
    require_optional_vec3f_strided(dot_state_edge_dir, "dot_state_edge_dir");
    require_optional_scalar_f_strided(dot_state_edge_t_min, "dot_state_edge_t_min");
    require_optional_scalar_f_strided(dot_state_edge_t_max, "dot_state_edge_t_max");
    require_optional_scalar_f_strided(dot_state_exterior_angle, "dot_state_exterior_angle");
    require_optional_vec3f_strided(dot_state_src, "dot_state_src");
    require_optional_scalar_f_strided(dot_state_src_power, "dot_state_src_power");
    require_optional_vec3f_strided(dot_recursive_state_edge_pos, "dot_recursive_state_edge_pos");
    require_optional_vec3f_strided(dot_recursive_state_edge_dir, "dot_recursive_state_edge_dir");
    require_optional_scalar_f_strided(dot_recursive_state_edge_t_min, "dot_recursive_state_edge_t_min");
    require_optional_scalar_f_strided(dot_recursive_state_edge_t_max, "dot_recursive_state_edge_t_max");
    require_optional_scalar_f_strided(dot_recursive_state_exterior_angle, "dot_recursive_state_exterior_angle");
    require_optional_scalar_f_strided(dot_material_gain, "dot_material_gain");
    Vec3Input dot_edge_pos_view = optional_vec3_input(dot_state_edge_pos, "dot_state_edge_pos");
    Vec3Input dot_edge_dir_view = optional_vec3_input(dot_state_edge_dir, "dot_state_edge_dir");
    Vec3Input dot_src_view = optional_vec3_input(dot_state_src, "dot_state_src");
    Vec3Input dot_recursive_edge_pos_view =
        optional_vec3_input(dot_recursive_state_edge_pos, "dot_recursive_state_edge_pos");
    Vec3Input dot_recursive_edge_dir_view =
        optional_vec3_input(dot_recursive_state_edge_dir, "dot_recursive_state_edge_dir");
    const int64_t launch_count = tape_active.size(0);
    if (state_limit_arg < 0 || recursive_state_limit_arg < 0)
        throw std::runtime_error("state counts must be non-negative.");
    const int64_t state_count = state_limit_arg;
    const int64_t recursive_state_count = recursive_state_limit_arg;
    if (state_count > state_edge_pos.size(0))
        throw std::runtime_error("state_count must not exceed state_edge_pos width.");
    if (recursive_state_count > recursive_state_edge_pos.size(0))
        throw std::runtime_error("recursive_state_count must not exceed recursive_state_edge_pos width.");
    const int64_t material_count = material_gain.size(0);
    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    require_state_width(tape_cell, launch_count, "tape_cell");
    require_state_width(state_edge_index, state_count, "state_edge_index");
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    require_state_width(recursive_state_edge_index, recursive_state_count, "recursive_state_edge_index");
    require_state_width(recursive_state_edge_dir, recursive_state_count, "recursive_state_edge_dir");
    require_state_width(recursive_state_edge_t_min, recursive_state_count, "recursive_state_edge_t_min");
    require_state_width(recursive_state_edge_t_max, recursive_state_count, "recursive_state_edge_t_max");
    require_state_width(recursive_state_prim0, recursive_state_count, "recursive_state_prim0");
    require_state_width(recursive_state_prim1, recursive_state_count, "recursive_state_prim1");
    require_state_width(recursive_state_exterior_angle, recursive_state_count, "recursive_state_exterior_angle");
    require_optional_state_width(dot_state_edge_pos, state_count, "dot_state_edge_pos");
    require_optional_state_width(dot_state_edge_dir, state_count, "dot_state_edge_dir");
    require_optional_state_width(dot_state_edge_t_min, state_count, "dot_state_edge_t_min");
    require_optional_state_width(dot_state_edge_t_max, state_count, "dot_state_edge_t_max");
    require_optional_state_width(dot_state_exterior_angle, state_count, "dot_state_exterior_angle");
    require_optional_state_width(dot_state_src, state_count, "dot_state_src");
    require_optional_state_width(dot_state_src_power, state_count, "dot_state_src_power");
    require_optional_state_width(dot_recursive_state_edge_pos, recursive_state_count, "dot_recursive_state_edge_pos");
    require_optional_state_width(dot_recursive_state_edge_dir, recursive_state_count, "dot_recursive_state_edge_dir");
    require_optional_state_width(dot_recursive_state_edge_t_min, recursive_state_count,
                                 "dot_recursive_state_edge_t_min");
    require_optional_state_width(dot_recursive_state_edge_t_max, recursive_state_count,
                                 "dot_recursive_state_edge_t_max");
    require_optional_state_width(dot_recursive_state_exterior_angle, recursive_state_count,
                                 "dot_recursive_state_exterior_angle");
    require_optional_state_width(dot_material_gain, material_count, "dot_material_gain");
    at::Tensor dot_power = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor dot_field_x_re = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor dot_zero = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor zero_tri = at::zeros({tri.n_triangles}, state_edge_pos.options());

    DfrChainAccumADParams params = {};
    // The tape rows are the shard's local lanes; the AD body replays them at the
    // global lanes [lane_offset, lane_offset + rows) the forward launch used.
    if (lane_offset < 0)
        throw std::runtime_error("lane_offset must be non-negative.");
    params.lane_offset = checked_i32(lane_offset, "lane_offset");
    params.n_rays = checked_i32(lane_offset + launch_count, "n_rays");
    params.state_count = checked_i32(state_count, "state_count");
    params.recursive_state_count = checked_i32(recursive_state_count, "recursive_state_count");
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
    params.max_order = checked_i32(max_order, "max_order");
    params.wavelength = static_cast<float>(wavelength);
    params.seed = checked_i32(seed, "seed");
    params.n_triangles = tri.n_triangles;
    params.tape_active = rebase_lane_buffer(mask_ptr(tape_active), params.lane_offset);
    params.tape_cell = rebase_lane_buffer(tape_cell.data_ptr<int>(), params.lane_offset);
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_index_stride = stride_i32(state_edge_index, 0, "state_edge_index_stride");
    params.state_edge_pos_x = state_edge_pos_view.x;
    params.state_edge_pos_y = state_edge_pos_view.y;
    params.state_edge_pos_z = state_edge_pos_view.z;
    params.state_edge_pos_stride = state_edge_pos_view.stride;
    params.state_edge_dir_x = state_edge_dir_view.x;
    params.state_edge_dir_y = state_edge_dir_view.y;
    params.state_edge_dir_z = state_edge_dir_view.z;
    params.state_edge_dir_stride = state_edge_dir_view.stride;
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_min_stride = stride_i32(state_edge_t_min, 0, "state_edge_t_min_stride");
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_edge_t_max_stride = stride_i32(state_edge_t_max, 0, "state_edge_t_max_stride");
    params.state_src_x = state_src_view.x;
    params.state_src_y = state_src_view.y;
    params.state_src_z = state_src_view.z;
    params.state_src_stride = state_src_view.stride;
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_src_power_stride = stride_i32(state_src_power, 0, "state_src_power_stride");
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_exterior_angle_stride = stride_i32(state_exterior_angle, 0, "state_exterior_angle_stride");
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim0_stride = stride_i32(state_prim0, 0, "state_prim0_stride");
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_prim1_stride = stride_i32(state_prim1, 0, "state_prim1_stride");
    params.recursive_state_edge_index = recursive_state_edge_index.data_ptr<int>();
    params.recursive_state_edge_index_stride =
        stride_i32(recursive_state_edge_index, 0, "recursive_state_edge_index_stride");
    params.recursive_state_edge_pos_x = recursive_edge_pos_view.x;
    params.recursive_state_edge_pos_y = recursive_edge_pos_view.y;
    params.recursive_state_edge_pos_z = recursive_edge_pos_view.z;
    params.recursive_state_edge_pos_stride = recursive_edge_pos_view.stride;
    params.recursive_state_edge_dir_x = recursive_edge_dir_view.x;
    params.recursive_state_edge_dir_y = recursive_edge_dir_view.y;
    params.recursive_state_edge_dir_z = recursive_edge_dir_view.z;
    params.recursive_state_edge_dir_stride = recursive_edge_dir_view.stride;
    params.recursive_state_edge_t_min = recursive_state_edge_t_min.data_ptr<float>();
    params.recursive_state_edge_t_min_stride =
        stride_i32(recursive_state_edge_t_min, 0, "recursive_state_edge_t_min_stride");
    params.recursive_state_edge_t_max = recursive_state_edge_t_max.data_ptr<float>();
    params.recursive_state_edge_t_max_stride =
        stride_i32(recursive_state_edge_t_max, 0, "recursive_state_edge_t_max_stride");
    params.recursive_state_exterior_angle = recursive_state_exterior_angle.data_ptr<float>();
    params.recursive_state_exterior_angle_stride =
        stride_i32(recursive_state_exterior_angle, 0, "recursive_state_exterior_angle_stride");
    params.recursive_state_prim0 = recursive_state_prim0.data_ptr<int>();
    params.recursive_state_prim0_stride = stride_i32(recursive_state_prim0, 0, "recursive_state_prim0_stride");
    params.recursive_state_prim1 = recursive_state_prim1.data_ptr<int>();
    params.recursive_state_prim1_stride = stride_i32(recursive_state_prim1, 0, "recursive_state_prim1_stride");
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
    params.material_gain_stride = stride_i32(material_gain, 0, "material_gain_stride");
    params.material_valid = mask_ptr(material_valid);
    params.material_valid_stride = stride_i32(material_valid, 0, "material_valid_stride");
    params.dot_state_edge_pos_x = dot_edge_pos_view.x;
    params.dot_state_edge_pos_y = dot_edge_pos_view.y;
    params.dot_state_edge_pos_z = dot_edge_pos_view.z;
    params.dot_state_edge_pos_stride = dot_edge_pos_view.stride;
    params.dot_state_edge_dir_x = dot_edge_dir_view.x;
    params.dot_state_edge_dir_y = dot_edge_dir_view.y;
    params.dot_state_edge_dir_z = dot_edge_dir_view.z;
    params.dot_state_edge_dir_stride = dot_edge_dir_view.stride;
    params.dot_state_edge_t_min = optional_scalar_ptr(dot_state_edge_t_min);
    params.dot_state_edge_t_min_stride = optional_scalar_stride(dot_state_edge_t_min, "dot_state_edge_t_min_stride");
    params.dot_state_edge_t_max = optional_scalar_ptr(dot_state_edge_t_max);
    params.dot_state_edge_t_max_stride = optional_scalar_stride(dot_state_edge_t_max, "dot_state_edge_t_max_stride");
    params.dot_state_src_x = dot_src_view.x;
    params.dot_state_src_y = dot_src_view.y;
    params.dot_state_src_z = dot_src_view.z;
    params.dot_state_src_stride = dot_src_view.stride;
    params.dot_state_src_power = optional_scalar_ptr(dot_state_src_power);
    params.dot_state_src_power_stride = optional_scalar_stride(dot_state_src_power, "dot_state_src_power_stride");
    params.dot_state_exterior_angle = optional_scalar_ptr(dot_state_exterior_angle);
    params.dot_state_exterior_angle_stride =
        optional_scalar_stride(dot_state_exterior_angle, "dot_state_exterior_angle_stride");
    params.dot_recursive_state_edge_pos_x = dot_recursive_edge_pos_view.x;
    params.dot_recursive_state_edge_pos_y = dot_recursive_edge_pos_view.y;
    params.dot_recursive_state_edge_pos_z = dot_recursive_edge_pos_view.z;
    params.dot_recursive_state_edge_pos_stride = dot_recursive_edge_pos_view.stride;
    params.dot_recursive_state_edge_dir_x = dot_recursive_edge_dir_view.x;
    params.dot_recursive_state_edge_dir_y = dot_recursive_edge_dir_view.y;
    params.dot_recursive_state_edge_dir_z = dot_recursive_edge_dir_view.z;
    params.dot_recursive_state_edge_dir_stride = dot_recursive_edge_dir_view.stride;
    params.dot_recursive_state_edge_t_min = optional_scalar_ptr(dot_recursive_state_edge_t_min);
    params.dot_recursive_state_edge_t_min_stride =
        optional_scalar_stride(dot_recursive_state_edge_t_min, "dot_recursive_state_edge_t_min_stride");
    params.dot_recursive_state_edge_t_max = optional_scalar_ptr(dot_recursive_state_edge_t_max);
    params.dot_recursive_state_edge_t_max_stride =
        optional_scalar_stride(dot_recursive_state_edge_t_max, "dot_recursive_state_edge_t_max_stride");
    params.dot_recursive_state_exterior_angle = optional_scalar_ptr(dot_recursive_state_exterior_angle);
    params.dot_recursive_state_exterior_angle_stride =
        optional_scalar_stride(dot_recursive_state_exterior_angle, "dot_recursive_state_exterior_angle_stride");
    params.dot_material_gain = optional_scalar_ptr(dot_material_gain);
    params.dot_material_gain_stride = optional_scalar_stride(dot_material_gain, "dot_material_gain_stride");
    params.dot_tri_p0_x = zero_tri.data_ptr<float>();
    params.dot_tri_p0_y = zero_tri.data_ptr<float>();
    params.dot_tri_p0_z = zero_tri.data_ptr<float>();
    params.dot_tri_fn_x = zero_tri.data_ptr<float>();
    params.dot_tri_fn_y = zero_tri.data_ptr<float>();
    params.dot_tri_fn_z = zero_tri.data_ptr<float>();
    params.dot_out_power = dot_power.data_ptr<float>();
    params.dot_out_field_x_re = dot_field_x_re.data_ptr<float>();
    dfr_chain_accum_jvp_gpu(params);
    return py::make_tuple(dot_power.reshape({grid_resolution1, grid_resolution0}),
                          dot_field_x_re.reshape({grid_resolution1, grid_resolution0}),
                          dot_zero.reshape({grid_resolution1, grid_resolution0}));
}

struct CoherentDiffractionOutputs {
    at::Tensor direct_x_re;
    at::Tensor direct_x_im;
    at::Tensor direct_y_re;
    at::Tensor direct_y_im;
    at::Tensor direct_z_re;
    at::Tensor direct_z_im;
    at::Tensor multi_x_re;
    at::Tensor multi_x_im;
    at::Tensor multi_y_re;
    at::Tensor multi_y_im;
    at::Tensor multi_z_re;
    at::Tensor multi_z_im;
    at::Tensor direct_count;
    at::Tensor multi_count;
    at::Tensor visibility_reject_count;
    at::Tensor utd_reject_count;
};

CoherentDiffractionOutputs diffraction_coherent_accumulation_forward_impl(
    SceneCache& scene, c10::optional<at::Tensor> active, at::Tensor state_edge_index, at::Tensor state_edge_pos,
    at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_n0,
    at::Tensor state_n1, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle,
    at::Tensor state_src, at::Tensor state_src_power, c10::optional<at::Tensor> state_wi,
    c10::optional<at::Tensor> state_d0, at::Tensor material_eta_r, at::Tensor material_sigma, at::Tensor material_mu_r,
    at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit_arg, int64_t grid_axis,
    double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min,
    double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area,
    double wavelength, bool select_diffraction_point, bool prefilter_visibility, int64_t lane_offset,
    int64_t lane_count) {
    require_optional_mask(active, "active");
    require_flat_i32_strided(state_edge_index, "state_edge_index");
    require_vec3f_strided(state_edge_pos, "state_edge_pos");
    require_vec3f_strided(state_edge_dir, "state_edge_dir");
    require_flat_f32_strided(state_edge_t_min, "state_edge_t_min");
    require_flat_f32_strided(state_edge_t_max, "state_edge_t_max");
    require_vec3f_strided(state_n0, "state_n0");
    require_vec3f_strided(state_n1, "state_n1");
    require_flat_i32_strided(state_prim0, "state_prim0");
    require_flat_i32_strided(state_prim1, "state_prim1");
    require_flat_f32_strided(state_exterior_angle, "state_exterior_angle");
    require_vec3f_strided(state_src, "state_src");
    require_flat_f32_strided(state_src_power, "state_src_power");
    require_optional_vec3f_strided(state_wi, "state_wi");
    require_optional_vec3f_strided(state_d0, "state_d0");
    require_flat_f32_strided(material_eta_r, "material_eta_r");
    require_flat_f32_strided(material_sigma, "material_sigma");
    require_flat_f32_strided(material_mu_r, "material_mu_r");
    require_flat_f32_strided(material_gain, "material_gain");
    require_mask_strided(material_valid, "material_valid");
    if (state_limit_arg < 0)
        throw std::runtime_error("state_limit must be non-negative.");
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

    const int64_t state_physical_count = state_edge_index.size(0);
    if (state_limit_arg > state_physical_count)
        throw std::runtime_error("state_limit must not exceed state_edge_index width.");
    const int64_t state_count = state_limit_arg;
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
    require_optional_state_width(state_wi, state_count, "state_wi");
    require_optional_state_width(state_d0, state_count, "state_d0");
    const int64_t material_count = material_eta_r.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_sigma.size(0) != material_count || material_mu_r.size(0) != material_count ||
        material_gain.size(0) != material_count || material_valid.size(0) != material_count) {
        throw std::runtime_error("material payload fields must have matching widths.");
    }

    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    const int64_t total_lane_count64 = state_count * cell_count;
    const int32_t total_lane_count = checked_i32(total_lane_count64, "total_lane_count");
    const int64_t launch_count64 = resolve_lane_window(lane_offset, lane_count, total_lane_count64);
    const int32_t lane_begin = checked_i32(lane_offset, "lane_offset");
    const int32_t launch_count = checked_i32(launch_count64, "launch_count");
    if (lane_begin != 0 && scene.trace_backend == TraceBackend::Cuda)
        throw std::runtime_error("diffraction accumulation lane_offset requires the OptiX trace backend.");
    require_scene_device(scene, active, "active");
    require_scene_device(scene, state_edge_index, "state_edge_index");
    require_scene_device(scene, state_edge_pos, "state_edge_pos");
    require_scene_device(scene, state_edge_dir, "state_edge_dir");
    require_scene_device(scene, state_edge_t_min, "state_edge_t_min");
    require_scene_device(scene, state_edge_t_max, "state_edge_t_max");
    require_scene_device(scene, state_n0, "state_n0");
    require_scene_device(scene, state_n1, "state_n1");
    require_scene_device(scene, state_prim0, "state_prim0");
    require_scene_device(scene, state_prim1, "state_prim1");
    require_scene_device(scene, state_exterior_angle, "state_exterior_angle");
    require_scene_device(scene, state_src, "state_src");
    require_scene_device(scene, state_src_power, "state_src_power");
    require_scene_device(scene, state_wi, "state_wi");
    require_scene_device(scene, state_d0, "state_d0");
    require_scene_device(scene, material_eta_r, "material_eta_r");
    require_scene_device(scene, material_sigma, "material_sigma");
    require_scene_device(scene, material_mu_r, "material_mu_r");
    require_scene_device(scene, material_gain, "material_gain");
    require_scene_device(scene, material_valid, "material_valid");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(scene.device_index));
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
        return {direct_x_re,     direct_x_im, direct_y_re,  direct_y_im, direct_z_re,
                direct_z_im,     multi_x_re,  multi_x_im,   multi_y_re,  multi_y_im,
                multi_z_re,      multi_z_im,  direct_count, multi_count, visibility_reject_count,
                utd_reject_count};
    }
    const bool staged_coherent_accum =
        launch_count64 >= kStagedDfrAccumMinSamples && launch_count64 >= cell_count * kStagedDfrAccumMinSamplesPerCell;
    at::Tensor coherent_stage_key = staged_coherent_accum ? at::full({launch_count64}, -1, iopts) : at::Tensor();
    at::Tensor coherent_stage_value = staged_coherent_accum ? at::zeros({launch_count64, 8}, fopts) : at::Tensor();

    Vec3Input state_edge_pos_view = vec3_input(state_edge_pos, "state_edge_pos");
    Vec3Input state_edge_dir_view = vec3_input(state_edge_dir, "state_edge_dir");
    Vec3Input state_n0_view = vec3_input(state_n0, "state_n0");
    Vec3Input state_n1_view = vec3_input(state_n1, "state_n1");
    Vec3Input state_src_view = vec3_input(state_src, "state_src");
    Vec3Input state_wi_view = optional_vec3_input(state_wi, "state_wi");
    Vec3Input state_d0_view = optional_vec3_input(state_d0, "state_d0");
    TriangleSoA tri = make_scene_triangle_soa(scene);
    at::Tensor active_contig;
    if (has_defined_optional_tensor(active)) {
        active_contig = active_mask_for_states(*active, state_count, "diffraction_coherent_accumulation_forward");
    }
    at::Tensor state_prefix_depth = at::zeros({state_count}, iopts);

    DfrAccumParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = total_lane_count;
    params.lane_offset = lane_begin;
    params.active_mask = optional_mask_ptr(active_contig);
    params.active_width = active_width_for_states(active_contig, "active_width");
    params.active_stride = active_stride_for_states(active_contig, "active_stride");
    params.state_count = checked_i32(state_count, "state_count");
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_index_stride = stride_i32(state_edge_index, 0, "state_edge_index_stride");
    params.state_edge_pos_x = state_edge_pos_view.x;
    params.state_edge_pos_y = state_edge_pos_view.y;
    params.state_edge_pos_z = state_edge_pos_view.z;
    params.state_edge_pos_stride = state_edge_pos_view.stride;
    params.state_edge_dir_x = state_edge_dir_view.x;
    params.state_edge_dir_y = state_edge_dir_view.y;
    params.state_edge_dir_z = state_edge_dir_view.z;
    params.state_edge_dir_stride = state_edge_dir_view.stride;
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_min_stride = stride_i32(state_edge_t_min, 0, "state_edge_t_min_stride");
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_edge_t_max_stride = stride_i32(state_edge_t_max, 0, "state_edge_t_max_stride");
    params.state_n0_x = state_n0_view.x;
    params.state_n0_y = state_n0_view.y;
    params.state_n0_z = state_n0_view.z;
    params.state_n0_stride = state_n0_view.stride;
    params.state_n1_x = state_n1_view.x;
    params.state_n1_y = state_n1_view.y;
    params.state_n1_z = state_n1_view.z;
    params.state_n1_stride = state_n1_view.stride;
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim0_stride = stride_i32(state_prim0, 0, "state_prim0_stride");
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_prim1_stride = stride_i32(state_prim1, 0, "state_prim1_stride");
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_exterior_angle_stride = stride_i32(state_exterior_angle, 0, "state_exterior_angle_stride");
    params.state_src_x = state_src_view.x;
    params.state_src_y = state_src_view.y;
    params.state_src_z = state_src_view.z;
    params.state_src_stride = state_src_view.stride;
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_src_power_stride = stride_i32(state_src_power, 0, "state_src_power_stride");
    params.state_wi_x = state_wi_view.x;
    params.state_wi_y = state_wi_view.y;
    params.state_wi_z = state_wi_view.z;
    params.state_wi_stride = state_wi_view.stride;
    params.state_d0_x = state_d0_view.x;
    params.state_d0_y = state_d0_view.y;
    params.state_d0_z = state_d0_view.z;
    params.state_d0_stride = state_d0_view.stride;
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
    params.material_gain_stride = stride_i32(material_gain, 0, "material_gain_stride");
    params.material_valid = mask_ptr(material_valid);
    params.material_valid_stride = stride_i32(material_valid, 0, "material_valid_stride");
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.max_order = 1;
    params.receiver_model = RAYD_TORCH_DFR_MATCHED_ISO;
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
    params.coherent_stage_key =
        rebase_lane_buffer(staged_coherent_accum ? coherent_stage_key.data_ptr<int>() : nullptr, lane_begin);
    params.coherent_stage_value =
        rebase_lane_buffer(staged_coherent_accum
                               ? reinterpret_cast<DfrCoherentStagedValue*>(coherent_stage_value.data_ptr<float>())
                               : nullptr,
                           lane_begin);

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    if (scene.trace_backend == TraceBackend::Cuda) {
        launch_diffraction_accumulation_cuda(scene, params, 11, launch_count);
    } else {
        auto pipeline = optix_pipeline_for_scene(scene, diffraction_accumulation_pipeline_config());
        pipeline->launch(11, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
    }
    if (staged_coherent_accum) {
        reduce_dfr_coherent_accum_staged_cuda(launch_count64, cell_count, coherent_stage_key, coherent_stage_value,
                                              direct_x_re, direct_x_im, direct_y_re, direct_y_im, direct_z_re,
                                              direct_z_im, multi_x_re, multi_x_im, multi_y_re, multi_y_im, multi_z_re,
                                              multi_z_im, direct_count, multi_count);
    }

    return {direct_x_re.reshape({grid_resolution1, grid_resolution0}),
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
            utd_reject_count.reshape({grid_resolution1, grid_resolution0})};
}

py::tuple coherent_diffraction_outputs_to_tuple(const CoherentDiffractionOutputs& result) {
    return py::make_tuple(result.direct_x_re, result.direct_x_im, result.direct_y_re, result.direct_y_im,
                          result.direct_z_re, result.direct_z_im, result.multi_x_re, result.multi_x_im,
                          result.multi_y_re, result.multi_y_im, result.multi_z_re, result.multi_z_im,
                          result.direct_count, result.multi_count, result.visibility_reject_count,
                          result.utd_reject_count);
}

py::tuple diffraction_coherent_accumulation_forward_op(
    int64_t scene_handle, c10::optional<at::Tensor> active, at::Tensor state_edge_index, at::Tensor state_edge_pos,
    at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_n0,
    at::Tensor state_n1, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle,
    at::Tensor state_src, at::Tensor state_src_power, c10::optional<at::Tensor> state_wi,
    c10::optional<at::Tensor> state_d0, at::Tensor material_eta_r, at::Tensor material_sigma, at::Tensor material_mu_r,
    at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit, int64_t grid_axis, double grid_position,
    double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max,
    int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area, double wavelength,
    bool select_diffraction_point, bool prefilter_visibility, int64_t lane_offset, int64_t lane_count) {
    return coherent_diffraction_outputs_to_tuple(diffraction_coherent_accumulation_forward_impl(
        get_scene(scene_handle), active, state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min,
        state_edge_t_max, state_n0, state_n1, state_prim0, state_prim1, state_exterior_angle, state_src,
        state_src_power, state_wi, state_d0, material_eta_r, material_sigma, material_mu_r, material_gain,
        material_valid, state_limit, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min,
        grid_coord1_max, grid_resolution0, grid_resolution1, grid_cell_area, wavelength, select_diffraction_point,
        prefilter_visibility, lane_offset, lane_count));
}

} // namespace rayd::torch_backend

#include "../bindings/integration_internal.h"

namespace rayd::torch {

namespace {

c10::optional<at::Tensor> optional_defined_tensor(const std::optional<at::Tensor>& tensor) {
    if (!tensor.has_value() || !tensor->defined())
        return c10::nullopt;
    return *tensor;
}

} // namespace

DiffractionPathResult diffraction_paths_order1_forward(const SceneResource& scene,
                                                       const DiffractionPathConfig& config) {
    torch_backend::SceneCache& scene_cache = detail::IntegrationAccess::scene_cache(scene);
    auto result = torch_backend::diffraction_paths_order1_forward_impl(
        scene_cache, config.tx_pos, config.tx_pol, config.rx_pos, config.active, config.state.edge_index,
        config.state.edge_pos, config.state.edge_dir, config.state.edge_t_min, config.state.edge_t_max, config.state.n0,
        config.state.n1, config.state.prim0, config.state.prim1, config.state.exterior_angle, config.state.src,
        config.state.src_power, config.material.eta_r, config.material.sigma, config.material.mu_r,
        config.material.gain, config.material.valid, config.state_limit, config.capacity,
        static_cast<int>(config.layout), config.wavelength, config.isb_taper_width_scale);
    return {result.count,      result.valid,      result.tx_id,      result.rx_id,      result.order,
            result.edge0,      result.edge1,      result.edge2,      result.delay,      result.field_x_re,
            result.field_x_im, result.field_y_re, result.field_y_im, result.field_z_re, result.field_z_im,
            result.p0,         result.p1,         result.p2};
}

DiffractionAccumulationResult diffraction_accumulation_forward(const SceneResource& scene,
                                                               const DiffractionAccumulationConfig& config) {
    torch_backend::SceneCache& scene_cache = detail::IntegrationAccess::scene_cache(scene);

    std::int64_t recursive_state_limit = 0;
    c10::optional<at::Tensor> recursive_active;
    c10::optional<at::Tensor> recursive_edge_index;
    c10::optional<at::Tensor> recursive_edge_pos;
    c10::optional<at::Tensor> recursive_edge_dir;
    c10::optional<at::Tensor> recursive_edge_t_min;
    c10::optional<at::Tensor> recursive_edge_t_max;
    c10::optional<at::Tensor> recursive_n0;
    c10::optional<at::Tensor> recursive_n1;
    c10::optional<at::Tensor> recursive_prim0;
    c10::optional<at::Tensor> recursive_prim1;
    c10::optional<at::Tensor> recursive_exterior_angle;
    if (config.recursive_state.has_value()) {
        const RecursiveDiffractionState& recursive = *config.recursive_state;
        recursive_state_limit = recursive.state_limit;
        recursive_active = optional_defined_tensor(recursive.active);
        if (recursive.edge_index.defined())
            recursive_edge_index = recursive.edge_index;
        if (recursive.edge_pos.defined())
            recursive_edge_pos = recursive.edge_pos;
        if (recursive.edge_dir.defined())
            recursive_edge_dir = recursive.edge_dir;
        if (recursive.edge_t_min.defined())
            recursive_edge_t_min = recursive.edge_t_min;
        if (recursive.edge_t_max.defined())
            recursive_edge_t_max = recursive.edge_t_max;
        if (recursive.n0.defined())
            recursive_n0 = recursive.n0;
        if (recursive.n1.defined())
            recursive_n1 = recursive.n1;
        if (recursive.prim0.defined())
            recursive_prim0 = recursive.prim0;
        if (recursive.prim1.defined())
            recursive_prim1 = recursive.prim1;
        if (recursive.exterior_angle.defined())
            recursive_exterior_angle = recursive.exterior_angle;
    }

    auto result = torch_backend::diffraction_accumulation_forward_impl(
        scene_cache, optional_defined_tensor(config.active), config.state.edge_index, config.state.edge_pos,
        config.state.edge_dir, config.state.edge_t_min, config.state.edge_t_max, config.state.n0, config.state.n1,
        config.state.prim0, config.state.prim1, config.state.exterior_angle, config.state.src, config.state.src_power,
        optional_defined_tensor(config.state.wi), optional_defined_tensor(config.state.d0), config.material.eta_r,
        config.material.sigma, config.material.mu_r, config.material.gain, config.material.valid, config.state_limit,
        config.grid.axis, config.grid.position, config.grid.coord0_min, config.grid.coord0_max, config.grid.coord1_min,
        config.grid.coord1_max, config.grid.resolution0, config.grid.resolution1, config.grid.cell_area,
        config.wavelength, config.direct_samples, config.keller_samples, config.suffix_samples, config.seed,
        config.max_order, recursive_state_limit, recursive_active, recursive_edge_index, recursive_edge_pos,
        recursive_edge_dir, recursive_edge_t_min, recursive_edge_t_max, recursive_n0, recursive_n1, recursive_prim0,
        recursive_prim1, recursive_exterior_angle, config.export_tape ? 1 : 0,
        optional_defined_tensor(config.sample_state_index), optional_defined_tensor(config.sample_edge_weight),
        // The stable typed boundary always runs the whole Monte-Carlo lane
        // space; lane sharding stays a Python-side control for now.
        0, -1);
    return {result.power,        result.field_x_re,        result.field_x_im,         result.field_y_re,
            result.field_y_im,   result.field_z_re,        result.field_z_im,         result.direct_count,
            result.keller_count, result.suffix_count,      result.visibility_rejects, result.edge_visibility_rejects,
            result.utd_rejects,  result.edge_uses,         result.tape_active,        result.tape_state_idx,
            result.tape_cell,    result.tape_material_idx, result.tape_edge_u};
}

CoherentDiffractionResult diffraction_coherent_accumulation_forward(const SceneResource& scene,
                                                                    const CoherentDiffractionConfig& config) {
    torch_backend::SceneCache& scene_cache = detail::IntegrationAccess::scene_cache(scene);
    // The stable typed boundary keeps its existing whole-lane-space contract.
    auto result = torch_backend::diffraction_coherent_accumulation_forward_impl(
        scene_cache, optional_defined_tensor(config.active), config.state.edge_index, config.state.edge_pos,
        config.state.edge_dir, config.state.edge_t_min, config.state.edge_t_max, config.state.n0, config.state.n1,
        config.state.prim0, config.state.prim1, config.state.exterior_angle, config.state.src, config.state.src_power,
        optional_defined_tensor(config.state.wi), optional_defined_tensor(config.state.d0), config.material.eta_r,
        config.material.sigma, config.material.mu_r, config.material.gain, config.material.valid, config.state_limit,
        config.grid.axis, config.grid.position, config.grid.coord0_min, config.grid.coord0_max, config.grid.coord1_min,
        config.grid.coord1_max, config.grid.resolution0, config.grid.resolution1, config.grid.cell_area,
        config.wavelength, config.select_diffraction_point, config.prefilter_visibility, 0, -1);
    return {result.direct_x_re,     result.direct_x_im, result.direct_y_re,
            result.direct_y_im,     result.direct_z_re, result.direct_z_im,
            result.multi_x_re,      result.multi_x_im,  result.multi_y_re,
            result.multi_y_im,      result.multi_z_re,  result.multi_z_im,
            result.direct_count,    result.multi_count, result.visibility_reject_count,
            result.utd_reject_count};
}

} // namespace rayd::torch

// OptiX diffraction pipeline setup.

#include <src/diffraction/pipeline.h>

#include <src/diffraction/accum_params.h>
#include <src/diffraction/paths_params.h>
#include <src/runtime/rt_internal.h>
#include <rayd/diffraction/accumulation_torch_ptx.h>
#include <rayd/diffraction/paths_torch_ptx.h>

namespace rayd::torch_backend {

OptixPipelineConfig dfr_paths_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_diffraction_paths_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_diffraction_paths_optix_ptx);
    config.raygen_entries = {
        "__raygen__diffraction_paths_order1_primary",
        "__raygen__diffraction_paths_order1",
        "__raygen__diffraction_paths_order1_source_visibility_primary",
        "__raygen__diffraction_paths_order1_target_export_primary",
    };
    config.miss_entry = "__miss__diffraction_paths";
    config.closesthit_entry = "__closesthit__diffraction_paths";
    config.num_payload_values = shared::optix::DiffractionPayloadCount;
    config.params_size = sizeof(DfrPathParams);
    return config;
}

OptixPipelineConfig dfr_accum_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = rayd_torch_diffraction_accumulation_optix_ptx;
    config.ptx_size = sizeof(rayd_torch_diffraction_accumulation_optix_ptx);
    config.raygen_entries = {
        "__raygen__diffraction_order1_accumulation",
        "__raygen__diffraction_order1_accumulation_primary",
        "__raygen__diffraction_order1_accumulation_no_suffix",
        "__raygen__diffraction_order1_accumulation_no_suffix_primary",
        "__raygen__diffraction_order1_accumulation_suffix",
        "__raygen__diffraction_order1_accumulation_suffix_primary",
        "__raygen__diffraction_order1_source_visibility_primary",
        "__raygen__diffraction_order1_no_suffix_target_accumulation_primary",
        "__raygen__diffraction_order1_suffix_first_visibility_primary",
        "__raygen__diffraction_order1_suffix_target_accumulation_primary",
        "__raygen__diffraction_order1_coherent_accumulation",
        "__raygen__diffraction_order1_coherent_accumulation_primary",
        "__raygen__diffraction_chain_accumulation",
        "__raygen__diffraction_chain_accumulation_primary",
    };
    config.miss_entry = "__miss__diffraction_accumulation";
    config.closesthit_entry = "__closesthit__diffraction_accumulation";
    config.num_payload_values = shared::optix::DiffractionPayloadCount;
    config.params_size = sizeof(DfrAccumParams);
    return config;
}

} // namespace rayd::torch_backend
