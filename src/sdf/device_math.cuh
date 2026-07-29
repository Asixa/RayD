#pragma once

#include <cmath>

#include <rayd/detail/vec3.h>
#include <rayd/detail/rt/qualifiers.h>
#include <rayd/detail/sdf/grid_sdf.cuh>
#include <rayd/detail/sdf/sphere_trace.h>

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

// Row-major 3x3, indexed `m[row][column]`, matching the `R_ij` spelling of
// ADR-0037 section 2.
struct Mat3 {
    float m[3][3];
};

RAYD_HOST_DEVICE Mat3 zero_mat3() {
    Mat3 out{};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            out.m[row][column] = 0.0f;
    return out;
}

// `acc += factor * a b^T`, the shape every rotation-gradient contribution takes.
RAYD_HOST_DEVICE void add_outer(Mat3 &acc, Vec3f a, Vec3f b, float factor) {
    const float left[3] = {a.x, a.y, a.z};
    const float right[3] = {b.x, b.y, b.z};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            acc.m[row][column] += factor * left[row] * right[column];
}

RAYD_HOST_DEVICE float contract(const Mat3 &a, const Mat3 &b) {
    float total = 0.0f;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            total += a.m[row][column] * b.m[row][column];
    return total;
}

// `(M^T v)_i = sum_j M[j][i] v_j`.
RAYD_HOST_DEVICE Vec3f transpose_mul(const Mat3 &matrix, Vec3f v) {
    const float source[3] = {v.x, v.y, v.z};
    float out[3] = {0.0f, 0.0f, 0.0f};
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            out[column] += matrix.m[row][column] * source[row];
    return vmath::make_vec3(out[0], out[1], out[2]);
}

RAYD_HOST_DEVICE Vec3f mul(const Mat3 &matrix, Vec3f v) {
    const float source[3] = {v.x, v.y, v.z};
    float out[3] = {0.0f, 0.0f, 0.0f};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            out[row] += matrix.m[row][column] * source[column];
    return vmath::make_vec3(out[0], out[1], out[2]);
}

// Scalar-first quaternion, the ADR-0037 section 2 convention.
struct Quat {
    float w;
    float x;
    float y;
    float z;
};

RAYD_HOST_DEVICE Quat make_quat(const float *values) {
    return Quat{values[0], values[1], values[2], values[3]};
}

RAYD_HOST_DEVICE float quat_dot(Quat a, Quat b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

RAYD_HOST_DEVICE Quat quat_scale(Quat q, float factor) {
    return Quat{q.w * factor, q.x * factor, q.y * factor, q.z * factor};
}

RAYD_HOST_DEVICE Quat quat_subtract(Quat a, Quat b) {
    return Quat{a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z};
}

// The same normalization floor `make_placement` applies, spelled once so the
// derivative and the primal cannot disagree about which quaternion `R` came
// from.
RAYD_HOST_DEVICE float quat_length_floor(Quat q) {
    return fmaxf(sqrtf(quat_dot(q, q)), core::kSdfEpsNorm);
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
RAYD_HOST_DEVICE Quat quaternion_vjp(Quat raw, const Mat3 &grad_rotation) {
    const float length = quat_length_floor(raw);
    const Quat unit = quat_scale(raw, 1.0f / length);
    const Quat grad_unit{
        contract(grad_rotation, rotation_derivative(unit, 0)),
        contract(grad_rotation, rotation_derivative(unit, 1)),
        contract(grad_rotation, rotation_derivative(unit, 2)),
        contract(grad_rotation, rotation_derivative(unit, 3)),
    };
    const Quat projected = quat_scale(unit, quat_dot(unit, grad_unit));
    return quat_scale(quat_subtract(grad_unit, projected), 1.0f / length);
}

// Forward-mode dual of `quaternion_vjp`: the differential of `R` induced by a
// tangent on the raw quaternion.
RAYD_HOST_DEVICE Mat3 rotation_differential(Quat raw, Quat tangent) {
    const float length = quat_length_floor(raw);
    const Quat unit = quat_scale(raw, 1.0f / length);
    const Quat projected = quat_scale(unit, quat_dot(unit, tangent));
    const Quat rate = quat_scale(quat_subtract(tangent, projected), 1.0f / length);
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
RAYD_HOST_DEVICE Vec3f corner_weight_gradient(const core::TrilinearCell &cell, int corner) {
    const float wx[2] = {1.0f - cell.frac.x, cell.frac.x};
    const float wy[2] = {1.0f - cell.frac.y, cell.frac.y};
    const float wz[2] = {1.0f - cell.frac.z, cell.frac.z};
    const int di = (corner >> 2) & 1;
    const int dj = (corner >> 1) & 1;
    const int dk = corner & 1;
    const float slope_i = (di == 0) ? -1.0f : 1.0f;
    const float slope_j = (dj == 0) ? -1.0f : 1.0f;
    const float slope_k = (dk == 0) ? -1.0f : 1.0f;
    return vmath::make_vec3(slope_i * wy[dj] * wz[dk],
                            wx[di] * slope_j * wz[dk],
                            wx[di] * wy[dj] * slope_k);
}

// Index-space Hessian of the trilinear interpolant on one voxel. Its diagonal
// is exactly zero (the interpolant is affine along each axis), so only the three
// mixed terms exist (ADR-0037 section 6).
struct IndexHessian {
    float xy;
    float xz;
    float yz;
};

RAYD_HOST_DEVICE IndexHessian index_hessian(const float *values, const core::TrilinearCell &cell) {
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

RAYD_HOST_DEVICE Vec3f hessian_mul(const IndexHessian &hessian, Vec3f v) {
    return vmath::make_vec3(hessian.xy * v.y + hessian.xz * v.z,
                            hessian.xy * v.x + hessian.yz * v.z,
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

RAYD_HOST_DEVICE Lane make_lane(
    const float *position,
    const float *rotation,
    const float *scale_values,
    Vec3f origin,
    Vec3f direction,
    core::GridExtent extent) {
    Lane lane{};
    lane.extent = extent;
    lane.cells = core::grid_cells(extent);
    lane.scale = vmath::make_vec3(scale_values[0], scale_values[1], scale_values[2]);
    lane.placement = core::make_placement(
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
        vmath::make_vec3(position[0], position[1], position[2]));
    lane.origin = origin;
    lane.raw_direction = direction;
    lane.direction_length = sqrtf(vmath::squared_norm(direction));
    lane.unit_direction = core::normalize_floor(direction);
    lane.local_origin = core::world_to_local_point(lane.placement, origin);
    lane.local_direction = core::world_to_local_direction(lane.placement, lane.unit_direction);
    bool finite = core::is_finite(lane.scale.x) && core::is_finite(lane.scale.y) &&
                  core::is_finite(lane.scale.z);
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
    const float *values;
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

RAYD_HOST_DEVICE GridSampler make_sampler(const float *values, const Lane &lane) {
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

RAYD_HOST_DEVICE FrozenHit evaluate_frozen(
    const float *values,
    const Lane &lane,
    core::BaseIndex base,
    float t) {
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
    hit.normal = vmath::scale(
        hit.world_gradient, 1.0f / fmaxf(hit.gradient_length, core::kSdfEpsNorm));
    const float along = vmath::dot(hit.world_gradient, lane.unit_direction);
    // ADR-0037 section 6: sign(0) is +1, and the magnitude floor bounds every
    // derivative of the hit distance by |dF/dtheta| / eps_graze.
    hit.denominator = (along >= 0.0f ? 1.0f : -1.0f) * fmaxf(fabsf(along), kSdfEpsGraze);
    return hit;
}

// `dF/dscale_i` at the frozen hit: `scale` enters only the grid coordinate
// mapping, so the row is exact and closed form (ADR-0037 section 6).
RAYD_HOST_DEVICE Vec3f scale_partial(const Lane &lane, const FrozenHit &hit) {
    return vmath::make_vec3(
        -hit.local_gradient.x * hit.local_point.x / lane.scale.x,
        -hit.local_gradient.y * hit.local_point.y / lane.scale.y,
        -hit.local_gradient.z * hit.local_point.z / lane.scale.z);
}

} // namespace rayd::torch_backend::sdf_math
