// Copyright Xingyu Chen.
// Defines shared sdf support for grid sdf.

#pragma once

#include <cmath>

#include <rayd/math.h>
#include <rayd/rt/qualifiers.h>

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
    return math::make_vec3(static_cast<float>(extent.nx - 1),
                           static_cast<float>(extent.ny - 1),
                           static_cast<float>(extent.nz - 1));
}

// Grid coordinate of a local-frame point, clamped to the closed sampled domain
// `[0, N_i - 1]` (ADR-0037 section 1), so the interpolant is never evaluated
// outside the box. The clamp absorbs float32 rounding at the box faces; a lane
// whose interval arithmetic is non-finite is rejected before it ever gets here.
RAYD_HOST_DEVICE math::Vec3f grid_coord(math::Vec3f local_point,
                                        math::Vec3f scale,
                                        math::Vec3f cells) {
    const math::Vec3f raw = math::make_vec3((local_point.x / scale.x + 0.5f) * cells.x,
                                            (local_point.y / scale.y + 0.5f) * cells.y,
                                            (local_point.z / scale.z + 0.5f) * cells.z);
    return math::make_vec3(fminf(fmaxf(raw.x, 0.0f), cells.x),
                           fminf(fmaxf(raw.y, 0.0f), cells.y),
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
RAYD_HOST_DEVICE TrilinearCell trilinear_cell(math::Vec3f u,
                                              BaseIndex base,
                                              GridExtent extent) {
    TrilinearCell cell{};
    cell.frac = math::make_vec3(u.x - static_cast<float>(base.i),
                                u.y - static_cast<float>(base.j),
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
RAYD_HOST_DEVICE GridSample sample_cell(const float *values, const TrilinearCell &cell) {
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
RAYD_HOST_DEVICE math::Vec3f local_gradient(math::Vec3f index_gradient,
                                            math::Vec3f cells,
                                            math::Vec3f scale) {
    return math::make_vec3(index_gradient.x * cells.x / scale.x,
                           index_gradient.y * cells.y / scale.y,
                           index_gradient.z * cells.z / scale.z);
}

} // namespace rayd::shared::sdf
