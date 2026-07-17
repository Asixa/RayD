#include <optix.h>
#include <optix_device.h>

#include <rayd/shared/multipath/diffraction_accumulation_device.cuh>
#include <rayd/torch/common/math.cuh>
#include <rayd/torch/diffraction/accum_params.h>

namespace rayd::torch_backend {

extern "C" {
extern __constant__ DfrAccumParams params;
}

namespace {

struct DiffractionAccumulationPolicy {

  static __forceinline__ __device__ const DfrAccumParams &params() {
    return ::rayd::torch_backend::params;
  }
  static constexpr int kDirect = RAYD_TORCH_DFR_DIRECT;
  static constexpr int kKeller = RAYD_TORCH_DFR_KELLER;
  static constexpr int kSuffix = RAYD_TORCH_DFR_SUFFIX_REFL;
  using CellGroup = WarpCellGroup;

  static __forceinline__ __device__ bool active_state(int i) {
    return active_for_state(params().active_mask, params().active_width,
                            params().active_stride, i);
  }
  static __forceinline__ __device__ bool recursive_active_state(int i) {
    return active_for_state(params().recursive_active_mask,
                            params().recursive_active_width,
                            params().recursive_active_stride, i);
  }
  static __forceinline__ __device__ CellGroup cell_group(int cell) {
    return ::rayd::torch_backend::warp_cell_group(cell);
  }
  static __forceinline__ __device__ void
  atomic_add_same_cell(float *base, int i, float v, CellGroup g) {
    ::rayd::torch_backend::atomic_add_same_cell(base, i, v, g);
  }
  static __forceinline__ __device__ void
  atomic_add_same_cell(int *base, int i, int v, CellGroup g) {
    ::rayd::torch_backend::atomic_add_same_cell(base, i, v, g);
  }
  static __forceinline__ __device__ void atomic_add_warp(float *base, float v) {
    ::rayd::torch_backend::atomic_add_warp(base, v);
  }
  static __forceinline__ __device__ void atomic_add_warp(int *base, int v) {
    ::rayd::torch_backend::atomic_add_warp(base, v);
  }
  static __forceinline__ __device__ bool
  stage_order1(unsigned int lane, int cell, float power, float field_x_re,
               bool is_direct, bool is_keller) {
    if (params().stage_cell == nullptr || params().stage_value == nullptr)
      return false;
    params().stage_cell[lane] = cell;
    params().stage_value[lane] = make_float4(
        power, field_x_re, is_direct ? 1.f : 0.f, is_keller ? 1.f : 0.f);
    return true;
  }
  static __forceinline__ __device__ bool
  stage_coherent(int cell, int state_idx, bool is_multi, float xr, float xi,
                 float yr, float yi, float zr, float zi) {
    if (params().coherent_stage_key == nullptr ||
        params().coherent_stage_value == nullptr)
      return false;
    const int grid_cell_count =
        params().grid_resolution0 * params().grid_resolution1;
    const int key = is_multi ? grid_cell_count + cell : cell;
    const int slot = cell * params().state_count + state_idx;
    DfrCoherentStagedValue value;
    value.a = make_float4(xr, xi, yr, yi);
    value.b = make_float4(zr, zi, 1.f, 0.f);
    params().coherent_stage_key[slot] = key;
    params().coherent_stage_value[slot] = value;
    return true;
  }

  static __forceinline__ __device__ bool
  active_for_state(const uint8_t *mask, int width, int stride, int state_idx) {
    if (mask == nullptr) {
      return true;
    }
    const int active_idx = width == 1 ? 0 : state_idx;
    return mask[active_idx * stride] != 0u;
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f state_vec(
      const float *x, const float *y, const float *z, int stride, int idx) {
    const int offset = idx * stride;
    return ::rayd::shared::math::make_vec3(x[offset], y[offset], z[offset]);
  }

  static __forceinline__ __device__ float3 optional_state_vec(
      const float *x, const float *y, const float *z, int stride, int idx) {
    const int offset = idx * stride;
    return make_f3(x != nullptr ? x[offset] : 0.f,
                   y != nullptr ? y[offset] : 0.f,
                   z != nullptr ? z[offset] : 0.f);
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f recursive_state_vec(
      const float *x, const float *y, const float *z, int stride, int idx) {
    const int offset = idx * stride;
    return ::rayd::shared::math::make_vec3(x[offset], y[offset], z[offset]);
  }

  static __forceinline__ __device__ float read_f32(const float *ptr, int stride,
                                                   int idx) {
    return ptr[idx * stride];
  }

  static __forceinline__ __device__ int read_i32(const int *ptr, int stride,
                                                 int idx) {
    return ptr[idx * stride];
  }

  static __forceinline__ __device__ uint8_t read_u8(const uint8_t *ptr,
                                                    int stride, int idx) {
    return ptr[idx * stride];
  }

  static __forceinline__ __device__ int
  sample_state_index_for_lane(unsigned int lane) {
    if (params().sample_state_index == nullptr)
      return static_cast<int>(lane %
                              static_cast<unsigned int>(params().state_count));
    const int state_idx =
        read_i32(params().sample_state_index,
                 params().sample_state_index_stride, static_cast<int>(lane));
    return (state_idx >= 0 && state_idx < params().state_count) ? state_idx
                                                                : -1;
  }

  static __forceinline__ __device__ int state_edge_index_at(int idx) {
    return read_i32(params().state_edge_index, params().state_edge_index_stride,
                    idx);
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f state_edge_pos_at(int idx) {
    return state_vec(params().state_edge_pos_x, params().state_edge_pos_y,
                     params().state_edge_pos_z, params().state_edge_pos_stride,
                     idx);
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f state_edge_dir_at(int idx) {
    return state_vec(params().state_edge_dir_x, params().state_edge_dir_y,
                     params().state_edge_dir_z, params().state_edge_dir_stride,
                     idx);
  }

  static __forceinline__ __device__ float state_edge_t_min_at(int idx) {
    return read_f32(params().state_edge_t_min, params().state_edge_t_min_stride,
                    idx);
  }

  static __forceinline__ __device__ float state_edge_t_max_at(int idx) {
    return read_f32(params().state_edge_t_max, params().state_edge_t_max_stride,
                    idx);
  }

  static __forceinline__ __device__ float
  sample_edge_weight_for_lane(int state_idx, unsigned int lane,
                              int sample_count) {
    if (params().sample_edge_weight != nullptr) {
      return fmaxf(read_f32(params().sample_edge_weight,
                            params().sample_edge_weight_stride,
                            static_cast<int>(lane)),
                   0.f);
    }
    const float edge_length = fmaxf(
        state_edge_t_max_at(state_idx) - state_edge_t_min_at(state_idx), 0.f);
    return edge_length * static_cast<float>(params().state_count) /
           fmaxf(static_cast<float>(sample_count), 1.f);
  }

  static __forceinline__ __device__ int state_prim0_at(int idx) {
    return read_i32(params().state_prim0, params().state_prim0_stride, idx);
  }

  static __forceinline__ __device__ int state_prim1_at(int idx) {
    return read_i32(params().state_prim1, params().state_prim1_stride, idx);
  }

  static __forceinline__ __device__ float state_exterior_angle_at(int idx) {
    return read_f32(params().state_exterior_angle,
                    params().state_exterior_angle_stride, idx);
  }

  static __forceinline__ __device__ float state_src_power_at(int idx) {
    return read_f32(params().state_src_power, params().state_src_power_stride,
                    idx);
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f state_src_at(int idx) {
    return state_vec(params().state_src_x, params().state_src_y,
                     params().state_src_z, params().state_src_stride, idx);
  }

  static __forceinline__ __device__ float3 state_wi_at(int idx) {
    return optional_state_vec(params().state_wi_x, params().state_wi_y,
                              params().state_wi_z, params().state_wi_stride,
                              idx);
  }

  static __forceinline__ __device__ int recursive_state_edge_index_at(int idx) {
    return read_i32(params().recursive_state_edge_index,
                    params().recursive_state_edge_index_stride, idx);
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f
  recursive_state_edge_pos_at(int idx) {
    return recursive_state_vec(params().recursive_state_edge_pos_x,
                               params().recursive_state_edge_pos_y,
                               params().recursive_state_edge_pos_z,
                               params().recursive_state_edge_pos_stride, idx);
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f
  recursive_state_edge_dir_at(int idx) {
    return recursive_state_vec(params().recursive_state_edge_dir_x,
                               params().recursive_state_edge_dir_y,
                               params().recursive_state_edge_dir_z,
                               params().recursive_state_edge_dir_stride, idx);
  }

  static __forceinline__ __device__ float
  recursive_state_edge_t_min_at(int idx) {
    return read_f32(params().recursive_state_edge_t_min,
                    params().recursive_state_edge_t_min_stride, idx);
  }

  static __forceinline__ __device__ float
  recursive_state_edge_t_max_at(int idx) {
    return read_f32(params().recursive_state_edge_t_max,
                    params().recursive_state_edge_t_max_stride, idx);
  }

  static __forceinline__ __device__ int recursive_state_prim0_at(int idx) {
    return read_i32(params().recursive_state_prim0,
                    params().recursive_state_prim0_stride, idx);
  }

  static __forceinline__ __device__ int recursive_state_prim1_at(int idx) {
    return read_i32(params().recursive_state_prim1,
                    params().recursive_state_prim1_stride, idx);
  }

  static __forceinline__ __device__ float
  recursive_state_exterior_angle_at(int idx) {
    return read_f32(params().recursive_state_exterior_angle,
                    params().recursive_state_exterior_angle_stride, idx);
  }

  static __forceinline__ __device__ bool material_valid_at(int prim) {
    return params().material_valid == nullptr ||
           read_u8(params().material_valid, params().material_valid_stride,
                   prim) != 0u;
  }

  static __forceinline__ __device__ float material_gain_at(int prim) {
    return read_f32(params().material_gain, params().material_gain_stride,
                    prim);
  }
};

using Device = ::rayd::shared::multipath::diffraction_accumulation::
    DiffractionAccumulationDevice<DiffractionAccumulationPolicy>;

} // namespace

extern "C" {
__constant__ DfrAccumParams params;
}

extern "C" __global__ void __closesthit__diffraction_accumulation() {
  Device::closesthit();
}

extern "C" __global__ void __miss__diffraction_accumulation() {
  Device::miss();
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation() {
  Device::run_diffraction_order1_accumulation_raygen<false, false, true, true,
                                                     true>();
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation_primary() {
  Device::run_diffraction_order1_accumulation_raygen<true, false, true, true,
                                                     true>();
}

extern "C" __global__ void
__raygen__diffraction_order1_accumulation_no_suffix() {
  Device::run_diffraction_order1_accumulation_raygen<false, false, true, true,
                                                     false>();
}

extern "C" __global__ void
__raygen__diffraction_order1_accumulation_no_suffix_primary() {
  Device::run_diffraction_order1_accumulation_raygen<true, false, true, true,
                                                     false>();
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation_suffix() {
  Device::run_diffraction_order1_accumulation_raygen<false, false, false, false,
                                                     true>();
}

extern "C" __global__ void
__raygen__diffraction_order1_accumulation_suffix_primary() {
  Device::run_diffraction_order1_accumulation_raygen<true, false, false, false,
                                                     true>();
}

extern "C" __global__ void
__raygen__diffraction_order1_source_visibility_primary() {
  Device::run_diffraction_order1_source_visibility_raygen<true>();
}

extern "C" __global__ void
__raygen__diffraction_order1_no_suffix_target_accumulation_primary() {

  Device::run_diffraction_order1_no_suffix_target_accumulation_raygen<true>();
}

extern "C" __global__ void
__raygen__diffraction_order1_suffix_first_visibility_primary() {
  Device::run_diffraction_order1_suffix_first_visibility_raygen<true>();
}

extern "C" __global__ void
__raygen__diffraction_order1_suffix_target_accumulation_primary() {
  Device::run_diffraction_order1_suffix_target_accumulation_raygen<true>();
}

extern "C" __global__ void
__raygen__diffraction_order1_coherent_accumulation() {
  Device::run_diffraction_order1_coherent_accumulation_raygen<false>();
}

extern "C" __global__ void
__raygen__diffraction_order1_coherent_accumulation_primary() {
  Device::run_diffraction_order1_coherent_accumulation_raygen<true>();
}

extern "C" __global__ void __raygen__diffraction_chain_accumulation() {
  Device::run_diffraction_chain_accumulation_raygen<false>();
}

extern "C" __global__ void __raygen__diffraction_chain_accumulation_primary() {
  Device::run_diffraction_chain_accumulation_raygen<true>();
}

} // namespace rayd::torch_backend
