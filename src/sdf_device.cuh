// Copyright Xingyu Chen.
// Defines shared SDF sampling, tracing, and derivative algorithms.

#pragma once

#include <cmath>

#include <rayd/math.h>
#include <src/runtime/rt_device.cuh>

// Provides backend-neutral trilinear sampling for caller-owned SDF grids.

namespace rayd::shared::sdf {

// Sample counts per axis. ADR-0037 requires `N_i >= 2` on every axis; the host
// validation layer enforces that and this device math assumes it.
struct GridExtent {
    int nx;
    int ny;
    int nz;
};

// Base voxel index of a grid coordinate: the corner `(i, j, k)` of the cell the
// sample point falls in, always in `[0, N_i - 2]`.
struct BaseIndex {
    int i;
    int j;
    int k;
};

// The eight corners of one voxel: linear indices into `values` plus the
// trilinear weights `c_m` of ADR-0037 section 1. Corner order is
// `m = 4 * di + 2 * dj + dk`, so `m` counts along the fastest-varying axis last,
// matching the row-major storage. The weights sum to one and are exactly what a
// `values` gradient scatters through.
struct TrilinearCell {
    int index[8];
    float weight[8];
    math::Vec3f frac;
};

// Interpolated field value and its index-space gradient `dD/du` at one point.
struct GridSample {
    float value;
    math::Vec3f index_gradient;
};

// Per-axis `N_i - 1`: the span of the grid coordinate `u`, and the factor that
// turns an index-space gradient into a local-space one.
RAYD_HOST_DEVICE math::Vec3f grid_cells(GridExtent extent) {
    return math::make_vec3(static_cast<float>(extent.nx - 1), static_cast<float>(extent.ny - 1),
                           static_cast<float>(extent.nz - 1));
}

// Grid coordinate of a local-frame point, clamped to the closed sampled domain
// `[0, N_i - 1]` (ADR-0037 section 1), so the interpolant is never evaluated
// outside the box. The clamp absorbs float32 rounding at the box faces; a lane
// whose interval arithmetic is non-finite is rejected before it ever gets here.
RAYD_HOST_DEVICE math::Vec3f grid_coord(math::Vec3f local_point, math::Vec3f scale, math::Vec3f cells) {
    const math::Vec3f raw =
        math::make_vec3((local_point.x / scale.x + 0.5f) * cells.x, (local_point.y / scale.y + 0.5f) * cells.y,
                        (local_point.z / scale.z + 0.5f) * cells.z);
    return math::make_vec3(fminf(fmaxf(raw.x, 0.0f), cells.x), fminf(fmaxf(raw.y, 0.0f), cells.y),
                           fminf(fmaxf(raw.z, 0.0f), cells.z));
}

// Floor of a clamped grid coordinate, kept inside `[0, count - 2]`. Written with
// ternaries rather than `min`/`max` so the header needs neither the CUDA integer
// intrinsics nor `<algorithm>`.
RAYD_HOST_DEVICE int cell_of(float coordinate, int count) {
    const int index = static_cast<int>(floorf(coordinate));
    const int upper = count - 2;
    return index < 0 ? 0 : (index > upper ? upper : index);
}

RAYD_HOST_DEVICE BaseIndex base_index(math::Vec3f u, GridExtent extent) {
    // `u` is already clamped to `[0, N_i - 1]`, so flooring can only land one
    // past the last cell exactly at the far face; `cell_of` pulls it back.
    return BaseIndex{
        cell_of(u.x, extent.nx),
        cell_of(u.y, extent.ny),
        cell_of(u.z, extent.nz),
    };
}

// Corner indices and weights of the frozen voxel. The base index is passed in
// rather than recomputed so that the forward pass and the derivative passes read
// the same eight samples even if they contract the coordinate expression
// differently (ADR-0037 section 6).
RAYD_HOST_DEVICE TrilinearCell trilinear_cell(math::Vec3f u, BaseIndex base, GridExtent extent) {
    TrilinearCell cell{};
    cell.frac = math::make_vec3(u.x - static_cast<float>(base.i), u.y - static_cast<float>(base.j),
                                u.z - static_cast<float>(base.k));
    const float wx[2] = {1.0f - cell.frac.x, cell.frac.x};
    const float wy[2] = {1.0f - cell.frac.y, cell.frac.y};
    const float wz[2] = {1.0f - cell.frac.z, cell.frac.z};
    const int stride_i = extent.ny * extent.nz;
    const int stride_j = extent.nz;
    const int origin = (base.i * extent.ny + base.j) * extent.nz + base.k;
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                const int m = (di << 2) | (dj << 1) | dk;
                cell.index[m] = origin + di * stride_i + dj * stride_j + dk;
                cell.weight[m] = wx[di] * wy[dj] * wz[dk];
            }
        }
    }
    return cell;
}

// Trilinear value and analytic index-space gradient from one voxel's corners.
// The gradient is the exact derivative of the interpolant on this voxel, not a
// finite difference; it is C0-discontinuous across voxel faces, which is a
// property of the representation (ADR-0037 section 6).
RAYD_HOST_DEVICE GridSample sample_cell(const float* values, const TrilinearCell& cell) {
    const float wx[2] = {1.0f - cell.frac.x, cell.frac.x};
    const float wy[2] = {1.0f - cell.frac.y, cell.frac.y};
    const float wz[2] = {1.0f - cell.frac.z, cell.frac.z};
    GridSample sample{0.0f, math::make_vec3(0.0f, 0.0f, 0.0f)};
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                const int m = (di << 2) | (dj << 1) | dk;
                const float corner = values[cell.index[m]];
                const float slope_i = (di == 0) ? -1.0f : 1.0f;
                const float slope_j = (dj == 0) ? -1.0f : 1.0f;
                const float slope_k = (dk == 0) ? -1.0f : 1.0f;
                sample.value += cell.weight[m] * corner;
                sample.index_gradient.x += slope_i * wy[dj] * wz[dk] * corner;
                sample.index_gradient.y += wx[di] * slope_j * wz[dk] * corner;
                sample.index_gradient.z += wx[di] * wy[dj] * slope_k * corner;
            }
        }
    }
    return sample;
}

// Local-frame gradient from the index-space one: ADR-0037 section 2's
// `(grad_l D)_i = (dD/du_i) * (N_i - 1) / scale_i`.
RAYD_HOST_DEVICE math::Vec3f local_gradient(math::Vec3f index_gradient, math::Vec3f cells, math::Vec3f scale) {
    return math::make_vec3(index_gradient.x * cells.x / scale.x, index_gradient.y * cells.y / scale.y,
                           index_gradient.z * cells.z / scale.z);
}

} // namespace rayd::shared::sdf

// Provides the backend-neutral oriented-box clip and relaxed sphere trace.

namespace rayd::shared::sdf {

// ADR-0037 section 7. `eps_norm` and `eps_parallel` are contract constants, not
// caller parameters; `relaxation`, `max_steps` and `eps_hit` are caller
// parameters and appear here only as defaults and as a derivation rule.
inline constexpr float kSdfEpsNorm = 1.0e-12f;
inline constexpr float kSdfEpsParallel = 1.0e-7f;
inline constexpr float kSdfEpsHitVoxelFraction = 1.0e-3f;
inline constexpr float kSdfDefaultRelaxation = 0.9f;
inline constexpr int kSdfDefaultMaxSteps = 64;
inline constexpr int kSdfBisectionSteps = 32;

// Rigid local-to-world placement of the box: the three columns of `R(q)` plus
// the world-space centre. Rigid means `R^T == R^{-1}`, so a ray parameter is the
// same metric distance in both frames and the world gradient of the field is
// `R * grad_l D` with no inverse transpose (ADR-0037 section 2).
struct Placement {
    math::Vec3f axis_x;
    math::Vec3f axis_y;
    math::Vec3f axis_z;
    math::Vec3f position;
};

// Overlap of the ray with the box. `t_lo`/`t_hi` are meaningful only when
// `valid` is true; an invalid interval is a miss, not an error.
struct Interval {
    float t_lo;
    float t_hi;
    bool valid;
};

struct MarchConfig {
    float t_lo;
    float t_hi;
    float eps_hit;
    float relaxation;
    int max_steps;
};

// Outcome of one march. `t` and `value` are the frozen hit distance and the
// field value there; on a miss they are the last state the lane reached and
// carry no contract. `steps` counts every sampler call the lane made.
struct MarchResult {
    float t;
    float value;
    bool hit;
    int steps;
};

RAYD_HOST_DEVICE bool is_finite(float value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// Unit vector under the contract normalization floor: finite for any input,
// including a zero-length direction, and differentiable through in the
// derivative passes (ADR-0037 section 2).
RAYD_HOST_DEVICE math::Vec3f normalize_floor(math::Vec3f value) {
    const float length = fmaxf(sqrtf(math::squared_norm(value)), kSdfEpsNorm);
    return math::make_vec3(value.x / length, value.y / length, value.z / length);
}

// Placement from a scalar-first quaternion `(qw, qx, qy, qz)`, normalized
// internally with the same floor. The columns are `R(qh)` exactly as written in
// ADR-0037 section 2.
RAYD_HOST_DEVICE Placement make_placement(float qw, float qx, float qy, float qz, math::Vec3f position) {
    const float length = fmaxf(sqrtf(qw * qw + qx * qx + qy * qy + qz * qz), kSdfEpsNorm);
    const float w = qw / length;
    const float x = qx / length;
    const float y = qy / length;
    const float z = qz / length;
    Placement placement{};
    placement.axis_x = math::make_vec3(1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + w * z), 2.0f * (x * z - w * y));
    placement.axis_y = math::make_vec3(2.0f * (x * y - w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + w * x));
    placement.axis_z = math::make_vec3(2.0f * (x * z + w * y), 2.0f * (y * z - w * x), 1.0f - 2.0f * (x * x + y * y));
    placement.position = position;
    return placement;
}

// `R^T d`: a direction, or any vector that must not be translated.
RAYD_HOST_DEVICE math::Vec3f world_to_local_direction(const Placement& placement, math::Vec3f direction) {
    return math::make_vec3(math::dot(placement.axis_x, direction), math::dot(placement.axis_y, direction),
                           math::dot(placement.axis_z, direction));
}

// `R^T (p - position)`: the world-to-local point map.
RAYD_HOST_DEVICE math::Vec3f world_to_local_point(const Placement& placement, math::Vec3f point) {
    return world_to_local_direction(placement, math::subtract(point, placement.position));
}

// `R v`: carries a local-frame vector, in particular a field gradient, to world.
RAYD_HOST_DEVICE math::Vec3f local_to_world_direction(const Placement& placement, math::Vec3f local_vector) {
    return math::add(math::add(math::scale(placement.axis_x, local_vector.x),
                               math::scale(placement.axis_y, local_vector.y)),
                     math::scale(placement.axis_z, local_vector.z));
}

// One slab of the box. An axis the ray is parallel to constrains nothing when
// the origin lies inside it and forces a miss otherwise.
RAYD_HOST_DEVICE void clip_axis(float half, float origin, float direction, float& t_lo, float& t_hi, bool& inside) {
    if (fabsf(direction) <= kSdfEpsParallel) {
        inside = inside && (fabsf(origin) <= half);
        return;
    }
    const float t_a = (-half - origin) / direction;
    const float t_b = (half - origin) / direction;
    t_lo = fmaxf(t_lo, fminf(t_a, t_b));
    t_hi = fminf(t_hi, fmaxf(t_a, t_b));
}

// Traced interval of ADR-0037 section 3: the ray/box overlap on `[0, tmax]`,
// computed by a slab test in the local frame. The ray has no tmin of its own, so
// an origin inside the box starts at `t_lo = 0`, which is a supported case.
RAYD_HOST_DEVICE Interval clip_ray_to_box(math::Vec3f local_origin, math::Vec3f local_direction, math::Vec3f scale,
                                          float tmax) {
    float t_lo = 0.0f;
    float t_hi = tmax;
    bool inside = true;
    clip_axis(0.5f * scale.x, local_origin.x, local_direction.x, t_lo, t_hi, inside);
    clip_axis(0.5f * scale.y, local_origin.y, local_direction.y, t_lo, t_hi, inside);
    clip_axis(0.5f * scale.z, local_origin.z, local_direction.z, t_lo, t_hi, inside);
    const bool valid = inside && (t_lo <= t_hi) && is_finite(t_lo) && is_finite(t_hi);
    return Interval{t_lo, t_hi, valid};
}

// Smallest world-unit voxel edge `h_min = min_i(scale_i / (N_i - 1))`, the only
// length scale the representation has.
RAYD_HOST_DEVICE float min_voxel_edge(math::Vec3f scale, math::Vec3f cells) {
    return fminf(fminf(scale.x / cells.x, scale.y / cells.y), scale.z / cells.z);
}

// ADR-0037 section 7: a non-positive host scalar means "derive from the resident
// scale and the grid extents", which every lane computes identically and which
// costs no device-to-host read.
RAYD_HOST_DEVICE float resolve_eps_hit(float requested, math::Vec3f scale, math::Vec3f cells) {
    return requested > 0.0f ? requested : kSdfEpsHitVoxelFraction * min_voxel_edge(scale, cells);
}

// Bisection on a bracket that the march proved contains a crossing:
// `sigma * D(lo) >= 0 > sigma * D(hi)`. It always reports a hit, at the last
// midpoint if the budget runs out, because the sign change is a proof that the
// interpolant crosses zero inside the bracket and `eps_hit` is only a tolerance
// (ADR-0037 section 4). A non-finite sample is the one way out: that lane misses.
template <typename Sampler>
RAYD_HOST_DEVICE MarchResult bisect_bracket(Sampler& sample, float lo, float hi, float sigma, float eps_hit,
                                            int steps) {
    MarchResult result{0.5f * (lo + hi), 0.0f, true, steps};
    for (int iteration = 0; iteration < kSdfBisectionSteps; ++iteration) {
        const float mid = 0.5f * (lo + hi);
        const float value = sample(mid);
        result.t = mid;
        result.value = value;
        result.steps += 1;
        if (!is_finite(value)) {
            result.hit = false;
            return result;
        }
        if (fabsf(value) < eps_hit) {
            return result;
        }
        if (sigma * value >= 0.0f) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return result;
}

// Relaxed sphere trace of ADR-0037 section 4. `sample(t)` returns the field at
// `origin + t * direction`; it is called exactly once per field evaluation, and
// on a hit the last call is at `MarchResult::t`, so a sampler that caches its
// last base voxel index hands the caller the frozen winner for free.
//
// The entry sample fixes the marching sign, every step is clamped to `t_hi`
// before it is sampled so the interpolant is never read outside the box, and the
// sign-flip test runs before the exit test because a step whose raw target
// leaves the box may still cross the level set inside it.
template <typename Sampler> RAYD_HOST_DEVICE MarchResult sphere_trace(Sampler& sample, const MarchConfig& config) {
    float t = config.t_lo;
    float d = sample(t);
    MarchResult result{t, d, false, 1};
    if (!is_finite(d)) {
        return result;
    }
    const float sigma = (d >= 0.0f) ? 1.0f : -1.0f;
    for (int step = 0; step < config.max_steps; ++step) {
        if (fabsf(d) < config.eps_hit) {
            result.hit = true;
            return result;
        }
        const float t_raw = t + config.relaxation * sigma * d;
        const float t_next = fminf(t_raw, config.t_hi);
        const float d_next = sample(t_next);
        result.steps += 1;
        if (!is_finite(d_next)) {
            return result;
        }
        if (sigma * d_next < 0.0f) {
            return bisect_bracket(sample, t, t_next, sigma, config.eps_hit, result.steps);
        }
        if (t_raw > config.t_hi) {
            return result;
        }
        t = t_next;
        d = d_next;
        result.t = t;
        result.value = d;
    }
    return result;
}

} // namespace rayd::shared::sdf

#include <src/sdf_device.cuh>

// Device math the ADR-0037 SDF intersection kernels share between the forward
// translation unit and the derivative one. The shared headers own the field
// interpolant, the placement and the march; this header owns only what the
// Torch kernels need on top of them:
//
//   * the per-ray setup both passes must reproduce identically (placement, unit
//     direction, local ray, structural usability of the placement);
//   * the frozen-hit evaluation the derivative passes consume (interpolated
//     gradient, world gradient, normal, clamped IFT denominator);
//   * the second-order pieces only the derivatives need (the index-space
//     Hessian of the trilinear interpolant, the per-corner weight gradient, the
//     quaternion rotation derivative).
//
// Everything here is a pure function of its arguments, allocates nothing, and
// spells only RAYD_HOST_DEVICE, so the same expressions compile on the host.
// It deliberately includes no OptiX or CUDA SDK header and stays outside every
// committed-PTX include closure (ADR-0037 section 9).

namespace rayd::torch_backend::sdf_math {

namespace vmath = shared::math;
namespace core = shared::sdf;

using vmath::Vec3f;

// ADR-0037 section 7: the grazing clamp floor, the `constants.epsilon.small`
// value of `contracts/operations.json`. It is a contract constant, never
// a caller parameter.
inline constexpr float kSdfEpsGraze = 1.0e-6f;

using Mat3 = vmath::Mat3f;
using Quat = vmath::Quaternionf;
using vmath::add_outer;
using vmath::contract;
using vmath::multiply;
using vmath::quaternion_dot;
using vmath::quaternion_scale;
using vmath::quaternion_subtract;
using vmath::transpose_multiply;
using vmath::zero_mat3;

RAYD_HOST_DEVICE Quat make_quat(const float* values) {
    return Quat{values[0], values[1], values[2], values[3]};
}

// The same normalization floor `make_placement` applies, spelled once so the
// derivative and the primal cannot disagree about which quaternion `R` came
// from.
RAYD_HOST_DEVICE float quat_length_floor(Quat q) {
    return fmaxf(sqrtf(quaternion_dot(q, q)), core::kSdfEpsNorm);
}

// `dR(qh)/dqh_a` for `a` in `(w, x, y, z)` order, differentiating the matrix
// written in ADR-0037 section 2 entry by entry.
RAYD_HOST_DEVICE Mat3 rotation_derivative(Quat q, int axis) {
    Mat3 out = zero_mat3();
    if (axis == 0) {
        out.m[0][1] = -2.0f * q.z;
        out.m[0][2] = 2.0f * q.y;
        out.m[1][0] = 2.0f * q.z;
        out.m[1][2] = -2.0f * q.x;
        out.m[2][0] = -2.0f * q.y;
        out.m[2][1] = 2.0f * q.x;
    } else if (axis == 1) {
        out.m[0][1] = 2.0f * q.y;
        out.m[0][2] = 2.0f * q.z;
        out.m[1][0] = 2.0f * q.y;
        out.m[1][1] = -4.0f * q.x;
        out.m[1][2] = -2.0f * q.w;
        out.m[2][0] = 2.0f * q.z;
        out.m[2][1] = 2.0f * q.w;
        out.m[2][2] = -4.0f * q.x;
    } else if (axis == 2) {
        out.m[0][0] = -4.0f * q.y;
        out.m[0][1] = 2.0f * q.x;
        out.m[0][2] = 2.0f * q.w;
        out.m[1][0] = 2.0f * q.x;
        out.m[1][2] = 2.0f * q.z;
        out.m[2][0] = -2.0f * q.w;
        out.m[2][1] = 2.0f * q.z;
        out.m[2][2] = -4.0f * q.y;
    } else {
        out.m[0][0] = -4.0f * q.z;
        out.m[0][1] = -2.0f * q.w;
        out.m[0][2] = 2.0f * q.x;
        out.m[1][0] = 2.0f * q.w;
        out.m[1][1] = -4.0f * q.z;
        out.m[1][2] = 2.0f * q.y;
        out.m[2][0] = 2.0f * q.x;
        out.m[2][1] = 2.0f * q.y;
    }
    return out;
}

// `J_q^T` applied to a rotation-matrix gradient: contract with `dR/dqh` and then
// push through the internal normalization, so the answer is a gradient with
// respect to the raw caller quaternion (ADR-0037 section 6).
RAYD_HOST_DEVICE Quat quaternion_vjp(Quat raw, const Mat3& grad_rotation) {
    const float length = quat_length_floor(raw);
    const Quat unit = quaternion_scale(raw, 1.0f / length);
    const Quat grad_unit{
        contract(grad_rotation, rotation_derivative(unit, 0)),
        contract(grad_rotation, rotation_derivative(unit, 1)),
        contract(grad_rotation, rotation_derivative(unit, 2)),
        contract(grad_rotation, rotation_derivative(unit, 3)),
    };
    const Quat projected = quaternion_scale(unit, quaternion_dot(unit, grad_unit));
    return quaternion_scale(quaternion_subtract(grad_unit, projected), 1.0f / length);
}

// Forward-mode dual of `quaternion_vjp`: the differential of `R` induced by a
// tangent on the raw quaternion.
RAYD_HOST_DEVICE Mat3 rotation_differential(Quat raw, Quat tangent) {
    const float length = quat_length_floor(raw);
    const Quat unit = quaternion_scale(raw, 1.0f / length);
    const Quat projected = quaternion_scale(unit, quaternion_dot(unit, tangent));
    const Quat rate = quaternion_scale(quaternion_subtract(tangent, projected), 1.0f / length);
    const float components[4] = {rate.w, rate.x, rate.y, rate.z};
    Mat3 out = zero_mat3();
    for (int axis = 0; axis < 4; ++axis) {
        const Mat3 derivative = rotation_derivative(unit, axis);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                out.m[row][column] += components[axis] * derivative.m[row][column];
    }
    return out;
}

// Jacobian of `normalize_floor` at a vector of raw length `length` whose unit
// image is `unit_vector`, applied to `v`. It is symmetric, so the same call
// serves the VJP and the JVP of both the direction and the normal.
RAYD_HOST_DEVICE Vec3f normalize_floor_jacobian(float length, Vec3f unit_vector, Vec3f v) {
    if (length <= core::kSdfEpsNorm)
        return vmath::scale(v, 1.0f / core::kSdfEpsNorm);
    const Vec3f projected = vmath::scale(unit_vector, vmath::dot(unit_vector, v));
    return vmath::scale(vmath::subtract(v, projected), 1.0f / length);
}

// Gradient of the trilinear weight `c_m` with respect to the grid coordinate:
// the `dF/dvalues` row of ADR-0037 section 6 differentiated once more, which is
// what a `values` gradient of the *field gradient* scatters through.
RAYD_HOST_DEVICE Vec3f corner_weight_gradient(const core::TrilinearCell& cell, int corner) {
    const float wx[2] = {1.0f - cell.frac.x, cell.frac.x};
    const float wy[2] = {1.0f - cell.frac.y, cell.frac.y};
    const float wz[2] = {1.0f - cell.frac.z, cell.frac.z};
    const int di = (corner >> 2) & 1;
    const int dj = (corner >> 1) & 1;
    const int dk = corner & 1;
    const float slope_i = (di == 0) ? -1.0f : 1.0f;
    const float slope_j = (dj == 0) ? -1.0f : 1.0f;
    const float slope_k = (dk == 0) ? -1.0f : 1.0f;
    return vmath::make_vec3(slope_i * wy[dj] * wz[dk], wx[di] * slope_j * wz[dk], wx[di] * wy[dj] * slope_k);
}

// Index-space Hessian of the trilinear interpolant on one voxel. Its diagonal
// is exactly zero (the interpolant is affine along each axis), so only the three
// mixed terms exist (ADR-0037 section 6).
struct IndexHessian {
    float xy;
    float xz;
    float yz;
};

RAYD_HOST_DEVICE IndexHessian index_hessian(const float* values, const core::TrilinearCell& cell) {
    const float wx[2] = {1.0f - cell.frac.x, cell.frac.x};
    const float wy[2] = {1.0f - cell.frac.y, cell.frac.y};
    const float wz[2] = {1.0f - cell.frac.z, cell.frac.z};
    IndexHessian hessian{0.0f, 0.0f, 0.0f};
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                const int corner = (di << 2) | (dj << 1) | dk;
                const float value = values[cell.index[corner]];
                const float slope_i = (di == 0) ? -1.0f : 1.0f;
                const float slope_j = (dj == 0) ? -1.0f : 1.0f;
                const float slope_k = (dk == 0) ? -1.0f : 1.0f;
                hessian.xy += slope_i * slope_j * wz[dk] * value;
                hessian.xz += slope_i * wy[dj] * slope_k * value;
                hessian.yz += wx[di] * slope_j * slope_k * value;
            }
        }
    }
    return hessian;
}

RAYD_HOST_DEVICE Vec3f hessian_multiply(const IndexHessian& hessian, Vec3f v) {
    return vmath::make_vec3(hessian.xy * v.y + hessian.xz * v.z, hessian.xy * v.x + hessian.yz * v.z,
                            hessian.xz * v.x + hessian.yz * v.y);
}

// Everything about one ray that the forward and the derivative passes must
// agree on bit for bit. `usable` is the structural placement check ADR-0037
// section 8 defers to the device: a non-positive or non-finite `scale`, or a
// non-finite `position`/`rotation`, makes every lane a miss rather than letting
// the clamped grid coordinate manufacture a plausible sample.
struct Lane {
    core::Placement placement;
    core::GridExtent extent;
    Vec3f cells;
    Vec3f scale;
    Vec3f origin;
    Vec3f raw_direction;
    Vec3f unit_direction;
    float direction_length;
    Vec3f local_origin;
    Vec3f local_direction;
    bool usable;
};

RAYD_HOST_DEVICE Lane make_lane(const float* position, const float* rotation, const float* scale_values, Vec3f origin,
                                Vec3f direction, core::GridExtent extent) {
    Lane lane{};
    lane.extent = extent;
    lane.cells = core::grid_cells(extent);
    lane.scale = vmath::make_vec3(scale_values[0], scale_values[1], scale_values[2]);
    lane.placement = core::make_placement(rotation[0], rotation[1], rotation[2], rotation[3],
                                          vmath::make_vec3(position[0], position[1], position[2]));
    lane.origin = origin;
    lane.raw_direction = direction;
    lane.direction_length = sqrtf(vmath::squared_norm(direction));
    lane.unit_direction = core::normalize_floor(direction);
    lane.local_origin = core::world_to_local_point(lane.placement, origin);
    lane.local_direction = core::world_to_local_direction(lane.placement, lane.unit_direction);
    bool finite = core::is_finite(lane.scale.x) && core::is_finite(lane.scale.y) && core::is_finite(lane.scale.z);
    for (int axis = 0; axis < 3; ++axis)
        finite = finite && core::is_finite(position[axis]);
    for (int axis = 0; axis < 4; ++axis)
        finite = finite && core::is_finite(rotation[axis]);
    lane.usable = finite && lane.scale.x > 0.0f && lane.scale.y > 0.0f && lane.scale.z > 0.0f;
    return lane;
}

// Field sampler along one ray. `sphere_trace` calls it once per field
// evaluation, so after the march `base` holds the frozen winner's voxel.
struct GridSampler {
    const float* values;
    core::GridExtent extent;
    Vec3f cells;
    Vec3f box_scale;
    Vec3f local_origin;
    Vec3f local_direction;
    core::BaseIndex base;

    RAYD_HOST_DEVICE float operator()(float t) {
        const Vec3f point = vmath::add(local_origin, vmath::scale(local_direction, t));
        const Vec3f coordinate = core::grid_coord(point, box_scale, cells);
        base = core::base_index(coordinate, extent);
        return core::sample_cell(values, core::trilinear_cell(coordinate, base, extent)).value;
    }
};

RAYD_HOST_DEVICE GridSampler make_sampler(const float* values, const Lane& lane) {
    GridSampler sampler{};
    sampler.values = values;
    sampler.extent = lane.extent;
    sampler.cells = lane.cells;
    sampler.box_scale = lane.scale;
    sampler.local_origin = lane.local_origin;
    sampler.local_direction = lane.local_direction;
    sampler.base = core::BaseIndex{0, 0, 0};
    return sampler;
}

// The frozen hit every derivative is taken at: the winner voxel, the field
// gradient there in three frames, the normal, and the clamped IFT denominator.
struct FrozenHit {
    core::TrilinearCell cell;
    Vec3f local_point;
    Vec3f world_point;
    Vec3f index_gradient;
    Vec3f local_gradient;
    Vec3f world_gradient;
    Vec3f normal;
    float gradient_length;
    float denominator;
};

RAYD_HOST_DEVICE FrozenHit evaluate_frozen(const float* values, const Lane& lane, core::BaseIndex base, float t) {
    FrozenHit hit{};
    hit.local_point = vmath::add(lane.local_origin, vmath::scale(lane.local_direction, t));
    hit.world_point = vmath::add(lane.origin, vmath::scale(lane.unit_direction, t));
    const Vec3f coordinate = core::grid_coord(hit.local_point, lane.scale, lane.cells);
    hit.cell = core::trilinear_cell(coordinate, base, lane.extent);
    const core::GridSample sample = core::sample_cell(values, hit.cell);
    hit.index_gradient = sample.index_gradient;
    hit.local_gradient = core::local_gradient(sample.index_gradient, lane.cells, lane.scale);
    hit.world_gradient = core::local_to_world_direction(lane.placement, hit.local_gradient);
    hit.gradient_length = sqrtf(vmath::squared_norm(hit.world_gradient));
    hit.normal = vmath::scale(hit.world_gradient, 1.0f / fmaxf(hit.gradient_length, core::kSdfEpsNorm));
    const float along = vmath::dot(hit.world_gradient, lane.unit_direction);
    // ADR-0037 section 6: sign(0) is +1, and the magnitude floor bounds every
    // derivative of the hit distance by |dF/dtheta| / eps_graze.
    hit.denominator = (along >= 0.0f ? 1.0f : -1.0f) * fmaxf(fabsf(along), kSdfEpsGraze);
    return hit;
}

// `dF/dscale_i` at the frozen hit: `scale` enters only the grid coordinate
// mapping, so the row is exact and closed form (ADR-0037 section 6).
RAYD_HOST_DEVICE Vec3f scale_partial(const Lane& lane, const FrozenHit& hit) {
    return vmath::make_vec3(-hit.local_gradient.x * hit.local_point.x / lane.scale.x,
                            -hit.local_gradient.y * hit.local_point.y / lane.scale.y,
                            -hit.local_gradient.z * hit.local_point.z / lane.scale.z);
}

} // namespace rayd::torch_backend::sdf_math
