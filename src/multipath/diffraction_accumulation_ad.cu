#include <rayd/multipath/diffraction_accumulation_ad.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include <rayd/native_launch_audit.h>
#include <rayd/rayd.h>

namespace rayd {

namespace {

constexpr float kSmallEps = 1e-6f;
constexpr float kRayBias = 1e-4f;
constexpr float kPi = 3.14159265358979323846f;

static __forceinline__ __device__ float3 make_vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(float s, float3 v) {
    return make_vec3(s * v.x, s * v.y, s * v.z);
}

static __forceinline__ __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float norm3(float3 v) {
    return sqrtf(fmaxf(dot3(v, v), 0.f));
}

static __forceinline__ __device__ float3 cross3(float3 a, float3 b) {
    return make_vec3(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float3 normalize3(float3 v) {
    return rsqrtf(fmaxf(dot3(v, v), 1e-12f)) * v;
}

static __forceinline__ __device__ unsigned int hash_u32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __forceinline__ __device__ float uniform01(unsigned int lane,
                                                  unsigned int stream,
                                                  unsigned int seed) {
    const unsigned int h = hash_u32(lane ^ (stream * 0x9e3779b9u) ^ seed);
    return static_cast<float>(h & 0x00ffffffu) * (1.f / 16777216.f);
}

static __forceinline__ __device__ float component(float3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

static __forceinline__ __device__ float3 stable_perpendicular(float3 axis,
                                                              float3 preferred) {
    float3 projected = preferred - dot3(preferred, axis) * axis;
    if (dot3(projected, projected) > 1e-12f) {
        return normalize3(projected);
    }
    const float3 fallback = fabsf(axis.z) < 0.9f
                                ? make_vec3(0.f, 0.f, 1.f)
                                : make_vec3(0.f, 1.f, 0.f);
    return normalize3(fallback - dot3(fallback, axis) * axis);
}

static __forceinline__ __device__ float3 normalize_jvp(float3 unit,
                                                       float norm,
                                                       float3 dot_v) {
    if (!(norm > kSmallEps) || !isfinite(norm)) {
        return make_vec3(0.f, 0.f, 0.f);
    }
    return (1.f / norm) * (dot_v - dot3(unit, dot_v) * unit);
}

static __forceinline__ __device__ float3 grid_cell_center(
    const DfrDirectAccumADParams &params,
    int cell) {
    const int i = cell % params.grid_resolution0;
    const int j = cell / params.grid_resolution0;
    const float u = (static_cast<float>(i) + 0.5f) /
                    fmaxf(static_cast<float>(params.grid_resolution0), 1.f);
    const float v = (static_cast<float>(j) + 0.5f) /
                    fmaxf(static_cast<float>(params.grid_resolution1), 1.f);
    const float c0 = params.grid_coord0_min +
                     u * (params.grid_coord0_max - params.grid_coord0_min);
    const float c1 = params.grid_coord1_min +
                     v * (params.grid_coord1_max - params.grid_coord1_min);
    if (params.grid_axis == 0) {
        return make_vec3(params.grid_position, c0, c1);
    }
    if (params.grid_axis == 1) {
        return make_vec3(c0, params.grid_position, c1);
    }
    return make_vec3(c0, c1, params.grid_position);
}

struct DirectPrimal {
    int lane;
    int state_idx;
    int cell;
    int material_idx;
    bool is_keller;
    int sample_count;
    float edge_u;
    float3 edge_pos;
    float3 edge_dir_raw;
    float edge_dir_norm;
    float3 edge_dir;
    float edge_t_min;
    float edge_t_max;
    float edge_t;
    float edge_length;
    float3 edge_point;
    float3 source;
    float3 wi_raw;
    float wi_norm;
    float3 wi;
    float src_power;
    float exterior_angle;
    float material_gain;
    float wedge_scale;
    float3 target;
    float3 keller_ko;
    float keller_ray_t;
    float keller_sin;
    float keller_cos;
    float source_dist2;
    float target_dist2;
    float contribution;
    float common_no_src;
    bool edge_length_active;
    bool wedge_active;
    bool material_active;
};

struct DfrTangent {
    float3 edge_pos;
    float3 edge_dir_raw;
    float edge_t_min;
    float edge_t_max;
    float3 source;
    float3 wi_raw;
    float src_power;
    float exterior_angle;
    float material_gain;
};

static __forceinline__ __device__ bool keller_target_from_state(
    const DfrDirectAccumADParams &params,
    int lane,
    float3 edge_point,
    float3 edge_dir,
    float3 wi,
    float3 &target,
    float3 &ko,
    float &ray_t,
    float &sin_theta,
    float &cos_theta) {
    const float axial = fminf(fmaxf(dot3(wi, edge_dir), -1.f), 1.f);
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float3 basis0 = stable_perpendicular(edge_dir, wi);
    const float3 basis1 = normalize3(cross3(edge_dir, basis0));
    sincosf(2.f * kPi * uniform01(static_cast<unsigned int>(lane),
                                  1u,
                                  static_cast<unsigned int>(params.seed)),
            &sin_theta,
            &cos_theta);
    ko = normalize3(axial * edge_dir +
                    radial * (cos_theta * basis0 + sin_theta * basis1));
    const float denom = component(ko, params.grid_axis);
    if (fabsf(denom) <= kSmallEps) {
        return false;
    }
    ray_t = (params.grid_position - component(edge_point, params.grid_axis)) / denom;
    if (!(ray_t > kRayBias) || !isfinite(ray_t)) {
        return false;
    }
    target = edge_point + ray_t * ko;
    return true;
}

static __forceinline__ __device__ float3 stable_perpendicular_jvp(
    float3 axis,
    float3 dot_axis,
    float3 preferred,
    float3 dot_preferred,
    float3 basis) {
    const float axis_dot_preferred = dot3(preferred, axis);
    const float3 projected = preferred - axis_dot_preferred * axis;
    const float projected_norm2 = dot3(projected, projected);
    float3 dot_projected;
    float projected_norm;
    if (projected_norm2 > 1e-12f) {
        dot_projected =
            dot_preferred -
            (dot3(dot_preferred, axis) + dot3(preferred, dot_axis)) * axis -
            axis_dot_preferred * dot_axis;
        projected_norm = sqrtf(fmaxf(projected_norm2, 0.f));
    } else {
        const float3 fallback = fabsf(axis.z) < 0.9f
                                    ? make_vec3(0.f, 0.f, 1.f)
                                    : make_vec3(0.f, 1.f, 0.f);
        const float fallback_dot_axis = dot3(fallback, axis);
        const float3 fallback_projected = fallback - fallback_dot_axis * axis;
        dot_projected =
            -1.f * (dot3(fallback, dot_axis) * axis + fallback_dot_axis * dot_axis);
        projected_norm = norm3(fallback_projected);
    }
    return normalize_jvp(basis, projected_norm, dot_projected);
}

static __forceinline__ __device__ float3 keller_target_jvp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float3 dot_edge_point,
    float3 dot_edge_dir,
    float3 dot_wi_raw) {
    if (!p.is_keller) {
        return make_vec3(0.f, 0.f, 0.f);
    }

    const float3 dot_wi = normalize_jvp(p.wi, p.wi_norm, dot_wi_raw);
    const float unclamped_axial = dot3(p.wi, p.edge_dir);
    const bool axial_active = unclamped_axial > -1.f && unclamped_axial < 1.f;
    const float axial = fminf(fmaxf(unclamped_axial, -1.f), 1.f);
    const float dot_axial = axial_active
                                ? dot3(dot_wi, p.edge_dir) + dot3(p.wi, dot_edge_dir)
                                : 0.f;
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float dot_radial =
        radial > kSmallEps ? (-(axial / radial) * dot_axial) : 0.f;

    const float3 basis0 = stable_perpendicular(p.edge_dir, p.wi);
    const float3 dot_basis0 =
        stable_perpendicular_jvp(p.edge_dir, dot_edge_dir, p.wi, dot_wi, basis0);
    const float3 basis1_raw = cross3(p.edge_dir, basis0);
    const float3 basis1 = normalize3(basis1_raw);
    const float basis1_norm = norm3(basis1_raw);
    const float3 dot_basis1_raw =
        cross3(dot_edge_dir, basis0) + cross3(p.edge_dir, dot_basis0);
    const float3 dot_basis1 = normalize_jvp(basis1, basis1_norm, dot_basis1_raw);

    const float3 radial_basis = p.keller_cos * basis0 + p.keller_sin * basis1;
    const float3 dot_radial_basis =
        p.keller_cos * dot_basis0 + p.keller_sin * dot_basis1;
    const float3 ko_raw = axial * p.edge_dir + radial * radial_basis;
    const float3 dot_ko_raw =
        dot_axial * p.edge_dir +
        axial * dot_edge_dir +
        dot_radial * radial_basis +
        radial * dot_radial_basis;
    const float ko_norm = norm3(ko_raw);
    const float3 dot_ko = normalize_jvp(p.keller_ko, ko_norm, dot_ko_raw);

    const float denom = component(p.keller_ko, params.grid_axis);
    const float dot_denom = component(dot_ko, params.grid_axis);
    const float numerator =
        params.grid_position - component(p.edge_point, params.grid_axis);
    const float dot_numerator = -component(dot_edge_point, params.grid_axis);
    const float dot_t =
        (dot_numerator * denom - numerator * dot_denom) /
        fmaxf(denom * denom, kSmallEps);
    return dot_edge_point + dot_t * p.keller_ko + p.keller_ray_t * dot_ko;
}

static __forceinline__ __device__ bool load_primal(
    const DfrDirectAccumADParams &params,
    int lane,
    DirectPrimal &p) {
    if (lane >= params.n_rays ||
        params.tape_active == nullptr ||
        params.tape_active[lane] == 0u) {
        return false;
    }

    p.lane = lane;
    p.state_idx = params.tape_state_idx[lane];
    p.cell = params.tape_cell[lane];
    p.material_idx = params.tape_material_idx != nullptr
                         ? params.tape_material_idx[lane]
                         : -1;
    p.edge_u = params.tape_edge_u != nullptr ? params.tape_edge_u[lane] : 0.f;
    if (p.state_idx < 0 || p.state_idx >= params.state_count ||
        p.cell < 0 ||
        p.cell >= params.grid_resolution0 * params.grid_resolution1) {
        return false;
    }
    p.is_keller = lane >= params.direct_samples &&
                  lane < params.direct_samples + params.keller_samples;
    p.sample_count = p.is_keller ? params.keller_samples : params.direct_samples;
    if (p.sample_count <= 0) {
        return false;
    }

    p.edge_pos = make_vec3(params.state_edge_pos_x[p.state_idx],
                           params.state_edge_pos_y[p.state_idx],
                           params.state_edge_pos_z[p.state_idx]);
    p.edge_dir_raw = make_vec3(params.state_edge_dir_x[p.state_idx],
                               params.state_edge_dir_y[p.state_idx],
                               params.state_edge_dir_z[p.state_idx]);
    p.edge_dir_norm = norm3(p.edge_dir_raw);
    if (!(p.edge_dir_norm > kSmallEps) || !isfinite(p.edge_dir_norm)) {
        return false;
    }
    p.edge_dir = (1.f / p.edge_dir_norm) * p.edge_dir_raw;
    p.edge_t_min = params.state_edge_t_min[p.state_idx];
    p.edge_t_max = params.state_edge_t_max[p.state_idx];
    p.edge_t = p.edge_t_min + p.edge_u * (p.edge_t_max - p.edge_t_min);
    p.edge_length = fmaxf(p.edge_t_max - p.edge_t_min, 0.f);
    p.edge_length_active = (p.edge_t_max - p.edge_t_min) > 0.f;
    p.edge_point = p.edge_pos + p.edge_t * p.edge_dir;
    p.source = make_vec3(params.state_src_x[p.state_idx],
                         params.state_src_y[p.state_idx],
                         params.state_src_z[p.state_idx]);
    p.wi_raw = make_vec3(params.state_wi_x != nullptr ? params.state_wi_x[p.state_idx] : 0.f,
                         params.state_wi_y != nullptr ? params.state_wi_y[p.state_idx] : 0.f,
                         params.state_wi_z != nullptr ? params.state_wi_z[p.state_idx] : 0.f);
    p.wi_norm = norm3(p.wi_raw);
    p.wi = normalize3(p.wi_raw);
    p.src_power = params.state_src_power[p.state_idx];
    p.exterior_angle = params.state_exterior_angle[p.state_idx];
    const float exterior_clamped = fmaxf(p.exterior_angle, 0.25f * kPi);
    p.wedge_scale = fminf(exterior_clamped / (2.f * kPi), 2.f);
    p.wedge_active = p.exterior_angle > 0.25f * kPi &&
                     exterior_clamped / (2.f * kPi) < 2.f;
    p.material_gain = 1.f;
    p.material_active = false;
    if (p.material_idx >= 0 &&
        p.material_idx < params.material_count &&
        params.material_gain != nullptr) {
        const float raw_gain = params.material_gain[p.material_idx];
        p.material_gain = fmaxf(raw_gain, 0.f);
        p.material_active = raw_gain > 0.f;
    }
    p.keller_ko = make_vec3(0.f, 0.f, 0.f);
    p.keller_ray_t = 0.f;
    p.keller_sin = 0.f;
    p.keller_cos = 1.f;
    if (p.is_keller) {
        if (!keller_target_from_state(params,
                                      lane,
                                      p.edge_point,
                                      p.edge_dir,
                                      p.wi,
                                      p.target,
                                      p.keller_ko,
                                      p.keller_ray_t,
                                      p.keller_sin,
                                      p.keller_cos)) {
            return false;
        }
    } else {
        p.target = grid_cell_center(params, p.cell);
    }

    const float source_dist = fmaxf(norm3(p.edge_point - p.source), kSmallEps);
    const float target_dist = fmaxf(norm3(p.target - p.edge_point), kSmallEps);
    p.source_dist2 = source_dist * source_dist;
    p.target_dist2 = target_dist * target_dist;
    const float sample_norm =
        1.f / fmaxf(static_cast<float>(p.sample_count), 1.f);
    p.common_no_src = p.material_gain *
                      p.edge_length *
                      params.grid_cell_area *
                      p.wedge_scale *
                      sample_norm /
                      (p.source_dist2 * p.target_dist2);
    p.contribution = p.src_power * p.common_no_src;
    return p.contribution > 0.f && isfinite(p.contribution);
}

static __forceinline__ __device__ float read_or_zero(const float *ptr, int index) {
    return ptr != nullptr ? ptr[index] : 0.f;
}

static __forceinline__ __device__ float3 read_vec_or_zero(
    const float *x,
    const float *y,
    const float *z,
    int index) {
    return make_vec3(read_or_zero(x, index),
                     read_or_zero(y, index),
                     read_or_zero(z, index));
}

static __forceinline__ __device__ void atomic_add_vec(
    float *x,
    float *y,
    float *z,
    int index,
    float3 value) {
    if (x != nullptr) {
        atomicAdd(x + index, value.x);
    }
    if (y != nullptr) {
        atomicAdd(y + index, value.y);
    }
    if (z != nullptr) {
        atomicAdd(z + index, value.z);
    }
}

static __forceinline__ __device__ float contribution_jvp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    const DfrTangent &tangent) {
    const float dot_edge_t =
        (1.f - p.edge_u) * tangent.edge_t_min +
        p.edge_u * tangent.edge_t_max;
    const float dot_edge_length =
        p.edge_length_active ? (tangent.edge_t_max - tangent.edge_t_min) : 0.f;
    const float dot_wedge =
        p.wedge_active ? tangent.exterior_angle / (2.f * kPi) : 0.f;
    const float3 dot_edge_dir =
        normalize_jvp(p.edge_dir, p.edge_dir_norm, tangent.edge_dir_raw);
    const float3 dot_edge_point =
        tangent.edge_pos + dot_edge_t * p.edge_dir + p.edge_t * dot_edge_dir;
    const float3 dot_target =
        keller_target_jvp(params, p, dot_edge_point, dot_edge_dir, tangent.wi_raw);

    float dot_contribution = p.common_no_src * tangent.src_power;
    if (p.material_gain != 0.f) {
        dot_contribution += p.contribution * (tangent.material_gain / p.material_gain);
    }
    if (p.edge_length != 0.f) {
        dot_contribution += p.contribution * (dot_edge_length / p.edge_length);
    }
    if (p.wedge_scale != 0.f) {
        dot_contribution += p.contribution * (dot_wedge / p.wedge_scale);
    }

    const float3 source_delta = p.edge_point - p.source;
    const float3 target_delta = p.target - p.edge_point;
    const float dot_source_dist2 =
        2.f * dot3(source_delta, dot_edge_point - tangent.source);
    const float dot_target_dist2 =
        2.f * dot3(target_delta, dot_target - dot_edge_point);
    dot_contribution +=
        p.contribution *
        (-(dot_source_dist2 / p.source_dist2) -
         (dot_target_dist2 / p.target_dist2));
    return dot_contribution;
}

static __forceinline__ __device__ float direct_jvp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p) {
    DfrTangent tangent = {};
    tangent.edge_pos =
        read_vec_or_zero(params.dot_state_edge_pos_x,
                         params.dot_state_edge_pos_y,
                         params.dot_state_edge_pos_z,
                         p.state_idx);
    tangent.edge_dir_raw =
        read_vec_or_zero(params.dot_state_edge_dir_x,
                         params.dot_state_edge_dir_y,
                         params.dot_state_edge_dir_z,
                         p.state_idx);
    tangent.edge_t_min = read_or_zero(params.dot_state_edge_t_min, p.state_idx);
    tangent.edge_t_max = read_or_zero(params.dot_state_edge_t_max, p.state_idx);
    tangent.source =
        read_vec_or_zero(params.dot_state_src_x,
                         params.dot_state_src_y,
                         params.dot_state_src_z,
                         p.state_idx);
    tangent.wi_raw =
        read_vec_or_zero(params.dot_state_wi_x,
                         params.dot_state_wi_y,
                         params.dot_state_wi_z,
                         p.state_idx);
    tangent.src_power = read_or_zero(params.dot_state_src_power, p.state_idx);
    tangent.exterior_angle =
        read_or_zero(params.dot_state_exterior_angle, p.state_idx);
    tangent.material_gain =
        (p.material_active && p.material_idx >= 0)
            ? read_or_zero(params.dot_material_gain, p.material_idx)
            : 0.f;
    return contribution_jvp(params, p, tangent);
}

static __forceinline__ __device__ void add_unit_vjp(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution,
    float *ptr,
    int index,
    const DfrTangent &tangent) {
    if (ptr != nullptr) {
        const float partial = contribution_jvp(params, p, tangent);
        atomicAdd(ptr + index, grad_contribution * partial);
    }
}

static __forceinline__ __device__ void keller_vjp_by_unit_jvps(
    const DfrDirectAccumADParams &params,
    const DirectPrimal &p,
    float grad_contribution) {
    DfrTangent tangent = {};
    tangent.edge_pos = make_vec3(1.f, 0.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_x, p.state_idx, tangent);
    tangent = {};
    tangent.edge_pos = make_vec3(0.f, 1.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_y, p.state_idx, tangent);
    tangent = {};
    tangent.edge_pos = make_vec3(0.f, 0.f, 1.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_pos_z, p.state_idx, tangent);

    tangent = {};
    tangent.edge_dir_raw = make_vec3(1.f, 0.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_x, p.state_idx, tangent);
    tangent = {};
    tangent.edge_dir_raw = make_vec3(0.f, 1.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_y, p.state_idx, tangent);
    tangent = {};
    tangent.edge_dir_raw = make_vec3(0.f, 0.f, 1.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_dir_z, p.state_idx, tangent);

    tangent = {};
    tangent.edge_t_min = 1.f;
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_t_min, p.state_idx, tangent);
    tangent = {};
    tangent.edge_t_max = 1.f;
    add_unit_vjp(params, p, grad_contribution, params.grad_state_edge_t_max, p.state_idx, tangent);

    tangent = {};
    tangent.source = make_vec3(1.f, 0.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_src_x, p.state_idx, tangent);
    tangent = {};
    tangent.source = make_vec3(0.f, 1.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_src_y, p.state_idx, tangent);
    tangent = {};
    tangent.source = make_vec3(0.f, 0.f, 1.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_src_z, p.state_idx, tangent);

    tangent = {};
    tangent.wi_raw = make_vec3(1.f, 0.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_wi_x, p.state_idx, tangent);
    tangent = {};
    tangent.wi_raw = make_vec3(0.f, 1.f, 0.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_wi_y, p.state_idx, tangent);
    tangent = {};
    tangent.wi_raw = make_vec3(0.f, 0.f, 1.f);
    add_unit_vjp(params, p, grad_contribution, params.grad_state_wi_z, p.state_idx, tangent);

    tangent = {};
    tangent.src_power = 1.f;
    add_unit_vjp(params, p, grad_contribution, params.grad_state_src_power, p.state_idx, tangent);
    tangent = {};
    tangent.exterior_angle = 1.f;
    add_unit_vjp(params, p, grad_contribution, params.grad_state_exterior_angle, p.state_idx, tangent);
    if (p.material_active && p.material_idx >= 0) {
        tangent = {};
        tangent.material_gain = 1.f;
        add_unit_vjp(params, p, grad_contribution, params.grad_material_gain, p.material_idx, tangent);
    }
}

__global__ void dfr_direct_accum_jvp_kernel(DfrDirectAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    DirectPrimal p;
    if (!load_primal(params, lane, p)) {
        return;
    }
    const float dot_contribution = direct_jvp(params, p);
    if (params.dot_out_power != nullptr) {
        atomicAdd(params.dot_out_power + p.cell, dot_contribution);
    }
    if (params.dot_out_field_x_re != nullptr) {
        const float amp = sqrtf(fmaxf(p.contribution, 0.f));
        if (amp > kSmallEps) {
            atomicAdd(params.dot_out_field_x_re + p.cell,
                      0.5f * dot_contribution / amp);
        }
    }
}

__global__ void dfr_direct_accum_vjp_kernel(DfrDirectAccumADParams params) {
    const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    DirectPrimal p;
    if (!load_primal(params, lane, p)) {
        return;
    }

    float grad_contribution =
        read_or_zero(params.grad_out_power, p.cell);
    const float amp = sqrtf(fmaxf(p.contribution, 0.f));
    if (amp > kSmallEps) {
        grad_contribution +=
            read_or_zero(params.grad_out_field_x_re, p.cell) * 0.5f / amp;
    }
    if (grad_contribution == 0.f || !isfinite(grad_contribution)) {
        return;
    }

    if (p.is_keller) {
        keller_vjp_by_unit_jvps(params, p, grad_contribution);
        return;
    }

    const float grad_src_power = grad_contribution * p.common_no_src;
    if (params.grad_state_src_power != nullptr) {
        atomicAdd(params.grad_state_src_power + p.state_idx, grad_src_power);
    }

    if (p.material_active &&
        p.material_idx >= 0 &&
        params.grad_material_gain != nullptr) {
        const float grad_gain =
            grad_contribution * p.contribution / fmaxf(p.material_gain, kSmallEps);
        atomicAdd(params.grad_material_gain + p.material_idx, grad_gain);
    }

    float grad_edge_length = 0.f;
    if (p.edge_length_active && p.edge_length > kSmallEps) {
        grad_edge_length = grad_contribution * p.contribution / p.edge_length;
    }
    if (p.wedge_active && params.grad_state_exterior_angle != nullptr) {
        const float grad_wedge =
            grad_contribution * p.contribution / fmaxf(p.wedge_scale, kSmallEps);
        atomicAdd(params.grad_state_exterior_angle + p.state_idx,
                  grad_wedge / (2.f * kPi));
    }

    const float3 source_delta = p.edge_point - p.source;
    const float3 target_delta = p.target - p.edge_point;
    const float3 d_contribution_d_edge =
        p.contribution *
        ((-2.f / p.source_dist2) * source_delta +
         (2.f / p.target_dist2) * target_delta);
    const float3 d_contribution_d_source =
        p.contribution * ((2.f / p.source_dist2) * source_delta);

    const float3 grad_edge_point = grad_contribution * d_contribution_d_edge;
    const float3 grad_source = grad_contribution * d_contribution_d_source;
    atomic_add_vec(params.grad_state_src_x,
                   params.grad_state_src_y,
                   params.grad_state_src_z,
                   p.state_idx,
                   grad_source);
    atomic_add_vec(params.grad_state_edge_pos_x,
                   params.grad_state_edge_pos_y,
                   params.grad_state_edge_pos_z,
                   p.state_idx,
                   grad_edge_point);

    const float grad_edge_t = dot3(grad_edge_point, p.edge_dir);
    if (params.grad_state_edge_t_min != nullptr) {
        atomicAdd(params.grad_state_edge_t_min + p.state_idx,
                  (1.f - p.edge_u) * grad_edge_t - grad_edge_length);
    }
    if (params.grad_state_edge_t_max != nullptr) {
        atomicAdd(params.grad_state_edge_t_max + p.state_idx,
                  p.edge_u * grad_edge_t + grad_edge_length);
    }

    const float3 grad_edge_dir = p.edge_t * grad_edge_point;
    const float3 grad_edge_dir_raw =
        (1.f / p.edge_dir_norm) *
        (grad_edge_dir - dot3(p.edge_dir, grad_edge_dir) * p.edge_dir);
    atomic_add_vec(params.grad_state_edge_dir_x,
                   params.grad_state_edge_dir_y,
                   params.grad_state_edge_dir_z,
                   p.state_idx,
                   grad_edge_dir_raw);
}

void check_cuda_call(cudaError_t error, const char *message) {
    require(error == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(error));
}

void check_cuda_last_error(const char *message) {
    check_cuda_call(cudaGetLastError(), message);
}

template <typename Kernel>
void launch_ad_kernel(const char *name,
                      Kernel kernel,
                      const DfrDirectAccumADParams &params) {
    if (params.n_rays <= 0) {
        return;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(jit_cuda_stream());
    require(stream != nullptr,
            "dfr_direct_accum_ad_gpu(): CUDA stream is unavailable.");
    const int block_size = 128;
    const int block_count = (params.n_rays + block_size - 1) / block_size;
    audit_cuda_kernel_launch(name,
                             static_cast<uint32_t>(block_count),
                             1,
                             1,
                             static_cast<uint32_t>(block_size),
                             1,
                             1,
                             static_cast<uint64_t>(params.n_rays));
    kernel<<<block_count, block_size, 0, stream>>>(params);
    check_cuda_last_error("dfr_direct_accum_ad_gpu(): failed to launch kernel");
}

} // namespace

void dfr_direct_accum_jvp_gpu(const DfrDirectAccumADParams &params) {
    launch_ad_kernel("dfr_direct_accum_jvp_kernel",
                     dfr_direct_accum_jvp_kernel,
                     params);
}

void dfr_direct_accum_vjp_gpu(const DfrDirectAccumADParams &params) {
    launch_ad_kernel("dfr_direct_accum_vjp_kernel",
                     dfr_direct_accum_vjp_kernel,
                     params);
}

} // namespace rayd
