// Copyright Xingyu Chen.
// Declares internal scattering support for scattering internal.

#pragma once

#include <rayd/scattering.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace rayd::torch::detail {

inline void check_tensor(const at::Tensor& tensor, const char* name, c10::ScalarType dtype, int64_t rank) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == rank, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_vec3_table(const at::Tensor& tensor, const char* name) {
    check_tensor(tensor, name, at::kFloat, 2);
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
}

inline void check_flat_tensor(const at::Tensor& tensor, const char* name, c10::ScalarType dtype) {
    check_tensor(tensor, name, dtype, 1);
}

// Scalars such as k0 and frequency_hz reach the kernels as float32 divisors;
// zero, negative, non-finite or float32-overflowing values only produce NaN or
// Inf rows, so every primal, backward and JVP entry rejects them alike.
inline void check_positive_finite_scalar(double value, const char* name) {
    constexpr double kFloatMax = static_cast<double>(std::numeric_limits<float>::max());
    TORCH_CHECK(std::isfinite(value) && value > 0.0 && value <= kFloatMax && static_cast<float>(value) > 0.0f, name,
                " must be a positive finite float32 value, got ", value);
}

// Height screens are sampled with h - 1 / w - 1 clamps and int texel indices.
inline void check_height_screen(const at::Tensor& heights, int64_t rows) {
    for (int64_t axis = 0; axis < 2; ++axis)
        TORCH_CHECK(rows == 0 || heights.size(axis) >= 1, "heights axis ", axis, " must not be empty");
    TORCH_CHECK(heights.numel() <= std::numeric_limits<int32_t>::max(), "heights exceeds int32 texel indexing");
}

// Dense (theta_i, phi_i, theta_o, phi_o) table. The device interpolation and
// nearest-cell lookups index every axis with n - 1 and take the extents as int,
// so a table that is read (rows > 0) needs every axis in [1, INT32_MAX].
inline void check_dense_table(const at::Tensor& tensor, const char* name, int64_t rows) {
    check_tensor(tensor, name, at::kFloat, 4);
    for (int64_t axis = 0; axis < 4; ++axis) {
        TORCH_CHECK(rows == 0 || tensor.size(axis) >= 1, name, " axis ", axis, " must not be empty");
        TORCH_CHECK(tensor.size(axis) <= std::numeric_limits<int32_t>::max(), name, " axis ", axis,
                    " exceeds int32 indexing");
    }
}

} // namespace rayd::torch::detail

namespace rayd::torch::detail {

inline void check_scattering_chain_leg(const at::Tensor& positions, const at::Tensor& normals, const at::Tensor& eps_r,
                                       const at::Tensor& sigma_e, const at::Tensor& mu_r, const at::Tensor& gain,
                                       const at::Tensor& thickness, const at::Tensor& depth, int64_t rows,
                                       const char* tag) {
    constexpr int64_t kMaxDepth = 8;
    check_tensor(positions, tag, at::kFloat, 3);
    check_tensor(normals, tag, at::kFloat, 3);
    TORCH_CHECK(positions.sizes() == at::IntArrayRef({rows, kMaxDepth, 3}), tag,
                " positions must have shape (R, 8, 3)");
    TORCH_CHECK(normals.sizes() == positions.sizes(), tag, " normals must match positions");
    for (const auto& tensor : {eps_r, sigma_e, mu_r, gain, thickness}) {
        check_tensor(tensor, tag, at::kFloat, 2);
        TORCH_CHECK(tensor.sizes() == at::IntArrayRef({rows, kMaxDepth}), tag,
                    " material tensors must have shape (R, 8)");
    }
    check_flat_tensor(depth, tag, at::kInt);
    TORCH_CHECK(depth.size(0) == rows, tag, " depth must have shape (R,)");
}

inline int64_t check_scattering_chain_ensemble_request(const ScatteringChainEnsembleEvalRequest& r) {
    check_vec3_table(r.tx_pol, "tx_pol");
    const int64_t rows = r.tx_pol.size(0);
    check_flat_tensor(r.valid, "valid", at::kBool);
    TORCH_CHECK(r.valid.size(0) == rows, "valid must have shape (R,)");
    for (const auto& tensor :
         {r.rx_pol, r.source, r.vertex, r.target, r.n_o, r.t1r, r.t2r, r.backup_axis, r.wi_local, r.d_i, r.d_o}) {
        check_vec3_table(tensor, "chain ensemble vec3");
        TORCH_CHECK(tensor.size(0) == rows, "chain ensemble vec3 inputs must have R rows");
    }
    check_scattering_chain_leg(r.c1_positions, r.c1_normals, r.c1_eps_r, r.c1_sigma_e, r.c1_mu_r, r.c1_gain,
                               r.c1_thickness, r.c1_depth, rows, "c1");
    check_scattering_chain_leg(r.c2_positions, r.c2_normals, r.c2_eps_r, r.c2_sigma_e, r.c2_mu_r, r.c2_gain,
                               r.c2_thickness, r.c2_depth, rows, "c2");
    for (const auto& tensor : {r.cos_i, r.cos_o, r.l1, r.l2, r.weights}) {
        check_flat_tensor(tensor, "chain ensemble scalar", at::kFloat);
        TORCH_CHECK(tensor.size(0) == rows, "chain ensemble scalar inputs must have R rows");
    }
    check_flat_tensor(r.material_id, "material_id", at::kInt);
    TORCH_CHECK(r.material_id.size(0) == rows, "material_id must have R rows");
    check_flat_tensor(r.f_te_flat, "f_te_flat", at::kFloat);
    check_flat_tensor(r.f_tm_flat, "f_tm_flat", at::kFloat);
    TORCH_CHECK(r.f_te_flat.size(0) == r.f_tm_flat.size(0), "f_te_flat and f_tm_flat must have equal lengths");
    check_flat_tensor(r.table_offset, "table_offset", at::kLong);
    check_tensor(r.table_dims, "table_dims", at::kInt, 2);
    TORCH_CHECK(r.table_dims.size(1) == 4, "table_dims must have shape (M, 4)");
    TORCH_CHECK(r.table_dims.size(0) == r.table_offset.size(0), "table_dims and table_offset must have equal rows");
    check_flat_tensor(r.material_slot, "material_slot", at::kInt);
    check_positive_finite_scalar(r.frequency_hz, "frequency_hz");

    for (const auto& tensor :
         {r.valid,      r.rx_pol,       r.source,     r.vertex,       r.target,       r.c1_positions, r.c1_normals,
          r.c1_eps_r,   r.c1_sigma_e,   r.c1_mu_r,    r.c1_gain,      r.c1_thickness, r.c1_depth,     r.c2_positions,
          r.c2_normals, r.c2_eps_r,     r.c2_sigma_e, r.c2_mu_r,      r.c2_gain,      r.c2_thickness, r.c2_depth,
          r.n_o,        r.t1r,          r.t2r,        r.backup_axis,  r.wi_local,     r.cos_i,        r.cos_o,
          r.d_i,        r.d_o,          r.l1,         r.l2,           r.weights,      r.material_id,  r.f_te_flat,
          r.f_tm_flat,  r.table_offset, r.table_dims, r.material_slot}) {
        TORCH_CHECK(tensor.get_device() == r.tx_pol.get_device(), "chain ensemble tensors must share device");
    }
    return rows;
}

} // namespace rayd::torch::detail

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <c10/util/Exception.h>

#include <optional>

#include <src/field_transport_ad.cuh>

// Backward / JVP companion kernels for the field transport forwards
// (plan 07 AD-1 materials/frequency, AD-2 geometry). Fixed-topology contract:
// the discrete winner (face sequence, validity, normal flips, polarizations,
// tx_power, material ids) is constant; the differentiable inputs are
// eps_r / sigma_e / gain / thickness (per bounce or CSR layer), the carrier
// frequency, and the continuous hit geometry (source, target,
// interaction_positions, interaction_normals) behind need_grad_geometry.
// path_length_m / delay_s are differentiable outputs of the geometry alone
// (their material/frequency cotangent is exactly zero).

namespace rayd::torch::scattering_internal {

constexpr int kBlockSize = 256;
constexpr int kMaxAdDepth = 8;
namespace field = rayd::shared::diffraction;
namespace em = rayd::shared::transmission;
namespace transport = rayd::shared::field_transport;
namespace ad = rayd::torch::field_transport_ad;

using ad::adj_dot;
using ad::DualC;
using ad::fold_output_cotangents;
using ad::write_output_tangents;

__device__ __forceinline__ field::float3a load3f(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::float3a load_sequence3f(const float* values, int64_t index, int64_t bounce,
                                                          int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ ad::DualF3 load_dual3f(const float* values, const float* tangents, int64_t index) {
    return {load3f(values, index), tangents != nullptr ? load3f(tangents, index) : field::f3_zero()};
}

__device__ __forceinline__ ad::DualF3 load_dual_sequence3f(const float* values, const float* tangents, int64_t index,
                                                           int64_t bounce, int64_t depth) {
    return {load_sequence3f(values, index, bounce, depth),
            tangents != nullptr ? load_sequence3f(tangents, index, bounce, depth) : field::f3_zero()};
}

__device__ __forceinline__ field::Complex complex_of(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

__device__ __forceinline__ c10::complex<float> to_c10(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

// ---------------------------------------------------------------------------
// Host entries.
// ---------------------------------------------------------------------------

inline int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

// Gradient accumulators must start at zero; allocate raw and memset on the
// current stream (same pattern as los.cu) instead of ATen zero-fill.
inline at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(
            cudaMemsetAsync(tensor.data_ptr(), 0, static_cast<size_t>(tensor.numel()) * tensor.element_size(), stream));
    }
    return tensor;
}

inline const at::Tensor* optional_grad(std::optional<at::Tensor> value, at::Tensor& storage, const char* name,
                                       c10::ScalarType dtype, at::IntArrayRef sizes, const at::Tensor& reference) {
    if (!value.has_value())
        return nullptr;
    storage = value.value().contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(storage.get_device() == reference.get_device(), name, " must share the primal device");
    return &storage;
}

template <typename T> const T* grad_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

} // namespace rayd::torch::scattering_internal

// Device-side range checks for row-addressed IDs and indices. The values are
// caller-owned device data and the boundary may not read them on the host, so
// every kernel tests them right after the validity gate and handles a row that
// fails exactly like an invalid row: bitwise-zero primal/JVP outputs and row
// gradients, and no shared-gradient atomics. Primal, backward and JVP apply the
// same test, so a rejected row is consistent across the three.
namespace rayd::torch::scattering_internal {

// Extents of the flat Kirchhoff table storage that material_id ->
// material_slot -> (table_offset, table_dims) addresses.
struct FlatTableBounds {
    int64_t material_count; // material_slot rows
    int64_t slot_count;     // table_offset / table_dims rows
    int64_t table_size;     // f_te_flat / f_tm_flat elements
};

inline FlatTableBounds flat_table_bounds(const at::Tensor& material_slot, const at::Tensor& table_offset,
                                         const at::Tensor& fte_flat) {
    return {material_slot.size(0), table_offset.size(0), fte_flat.size(0)};
}

// A negative slot means "no table" and stays valid; any other slot must name a
// table row whose four extents are >= 1 and whose block lies inside the flat
// storage.
__device__ __forceinline__ bool flat_table_lookup_in_range(int material, const int* material_slot,
                                                           const int64_t* table_offset, const int* table_dims,
                                                           FlatTableBounds bounds) {
    if (material < 0 || static_cast<int64_t>(material) >= bounds.material_count)
        return false;
    const int slot = material_slot[material];
    if (slot < 0)
        return true;
    if (static_cast<int64_t>(slot) >= bounds.slot_count)
        return false;
    int64_t extent = 1;
    for (int axis = 0; axis < 4; ++axis) {
        const int n = table_dims[static_cast<int64_t>(slot) * 4 + axis];
        if (n < 1 || extent > bounds.table_size / n)
            return false;
        extent *= n;
    }
    const int64_t base = table_offset[slot];
    return base >= 0 && base <= bounds.table_size - extent;
}

// Per-row chain depths index fixed (R, 8) leg blocks and fixed-size AD stacks.
__device__ __forceinline__ bool chain_depth_in_range(int depth) {
    return depth >= 0 && depth <= kMaxAdDepth;
}

} // namespace rayd::torch::scattering_internal
