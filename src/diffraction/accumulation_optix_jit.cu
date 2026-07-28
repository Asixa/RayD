#include <optix.h>
#include <optix_device.h>

#include <rayd/multipath/diffraction_accumulation.h>
#include <rayd/multipath/diffraction_accumulation_params.h>
#include <rayd/shared/multipath/diffraction_accumulation_device.cuh>

namespace rayd {

extern "C" {
extern __constant__ DfrAccumParams params;
}

namespace {

struct DiffractionAccumulationPolicy {
  static __forceinline__ __device__ const DfrAccumParams &params() {
    return ::rayd::params;
  }

  static constexpr int kDirect = RAYD_DFR_DIRECT;
  static constexpr int kKeller = RAYD_DFR_KELLER;
  static constexpr int kSuffix = RAYD_DFR_SUFFIX_REFL;

  struct CellGroup {};

  static __forceinline__ __device__ int
  sample_state_index_for_lane(unsigned int lane) {
    return static_cast<int>(lane %
                            static_cast<unsigned int>(params().state_count));
  }

  static __forceinline__ __device__ bool active_state(int i) {
    return params().active_mask == nullptr || params().active_mask[i] != 0u;
  }

  static __forceinline__ __device__ bool recursive_active_state(int i) {
    return params().recursive_active_mask == nullptr ||
           params().recursive_active_mask[i] != 0u;
  }

  static __forceinline__ __device__ int state_edge_index_at(int i) {
    return params().state_edge_index[i];
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f state_edge_pos_at(int i) {
    return ::rayd::shared::math::make_vec3(params().state_edge_pos_x[i],
                                           params().state_edge_pos_y[i],
                                           params().state_edge_pos_z[i]);
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f state_edge_dir_at(int i) {
    return ::rayd::shared::math::make_vec3(params().state_edge_dir_x[i],
                                           params().state_edge_dir_y[i],
                                           params().state_edge_dir_z[i]);
  }

  static __forceinline__ __device__ float state_edge_t_min_at(int i) {
    return params().state_edge_t_min[i];
  }

  static __forceinline__ __device__ float state_edge_t_max_at(int i) {
    return params().state_edge_t_max[i];
  }

  static __forceinline__ __device__ float
  sample_edge_weight_for_lane(int state_idx, unsigned int, int sample_count) {
    const float edge_length = fmaxf(
        state_edge_t_max_at(state_idx) - state_edge_t_min_at(state_idx), 0.f);
    return edge_length / fmaxf(static_cast<float>(sample_count), 1.f);
  }

  static __forceinline__ __device__ int state_prim0_at(int i) {
    return params().state_prim0[i];
  }

  static __forceinline__ __device__ int state_prim1_at(int i) {
    return params().state_prim1[i];
  }

  static __forceinline__ __device__ float state_exterior_angle_at(int i) {
    return params().state_exterior_angle[i];
  }

  static __forceinline__ __device__ float state_src_power_at(int i) {
    return params().state_src_power[i];
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f state_src_at(int i) {
    return ::rayd::shared::math::make_vec3(params().state_src_x[i], params().state_src_y[i],
                                           params().state_src_z[i]);
  }

  static __forceinline__ __device__ float3 state_wi_at(int i) {
    return make_float3(params().state_wi_x[i], params().state_wi_y[i],
                       params().state_wi_z[i]);
  }

  static __forceinline__ __device__ int recursive_state_edge_index_at(int i) {
    return params().recursive_state_edge_index[i];
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f recursive_state_edge_pos_at(int i) {
    return ::rayd::shared::math::make_vec3(params().recursive_state_edge_pos_x[i],
                                           params().recursive_state_edge_pos_y[i],
                                           params().recursive_state_edge_pos_z[i]);
  }

  static __forceinline__ __device__ ::rayd::shared::math::Vec3f recursive_state_edge_dir_at(int i) {
    return ::rayd::shared::math::make_vec3(params().recursive_state_edge_dir_x[i],
                                           params().recursive_state_edge_dir_y[i],
                                           params().recursive_state_edge_dir_z[i]);
  }

  static __forceinline__ __device__ float recursive_state_edge_t_min_at(int i) {
    return params().recursive_state_edge_t_min[i];
  }

  static __forceinline__ __device__ float recursive_state_edge_t_max_at(int i) {
    return params().recursive_state_edge_t_max[i];
  }

  static __forceinline__ __device__ int recursive_state_prim0_at(int i) {
    return params().recursive_state_prim0[i];
  }

  static __forceinline__ __device__ int recursive_state_prim1_at(int i) {
    return params().recursive_state_prim1[i];
  }

  static __forceinline__ __device__ float
  recursive_state_exterior_angle_at(int i) {
    return params().recursive_state_exterior_angle[i];
  }

  static __forceinline__ __device__ bool material_valid_at(int prim) {
    return params().material_valid == nullptr ||
           params().material_valid[prim] != 0u;
  }

  static __forceinline__ __device__ float material_gain_at(int prim) {
    return params().material_gain[prim];
  }

  static __forceinline__ __device__ CellGroup cell_group(int) { return {}; }

  static __forceinline__ __device__ void
  atomic_add_same_cell(float *base, int i, float value, CellGroup) {
    atomicAdd(base + i, value);
  }

  static __forceinline__ __device__ void
  atomic_add_same_cell(int *base, int i, int value, CellGroup) {
    atomicAdd(base + i, value);
  }

  static __forceinline__ __device__ void atomic_add_warp(float *base,
                                                         float value) {
    atomicAdd(base, value);
  }

  static __forceinline__ __device__ void atomic_add_warp(int *base, int value) {
    atomicAdd(base, value);
  }

  static __forceinline__ __device__ bool stage_order1(unsigned int, int, float,
                                                      float, bool, bool) {
    return false;
  }

  static __forceinline__ __device__ bool
  stage_coherent(int, int, bool, float, float, float, float, float, float) {
    return false;
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

} // namespace rayd
