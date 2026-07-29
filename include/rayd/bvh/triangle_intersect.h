// Copyright Xingyu Chen.
// Defines shared bvh support for triangle intersect.

#pragma once

#include <cmath>

// Watertight ray/triangle intersection (Woop, Benthin, Wald 2013,
// "Watertight Ray/Triangle Intersection"). Host/device dual through the same
// __CUDACC__ inline pattern as rayd/math.h so the routine is both a
// device leaf test and a host-unit-testable pure function. It is
// primitive-only: no Dr.Jit, Torch, OptiX, or CUDA runtime dependency.
//
// The returned (u, v) match the Moller-Trumbore convention used by
// include/rayd/jit/utils.h ray_intersect_triangle: a triangle given
// as (p0, e1, e2) with vertices A = p0, B = p0 + e1, C = p0 + e2 has hit point
// P = p0 + u*e1 + v*e2 = (1 - u - v)*A + u*B + v*C. So u is the barycentric
// weight of B and v the weight of C, and t is measured along the (unnormalized)
// ray direction, exactly like the Moller-Trumbore reference.

#if defined(__CUDACC__)
#define RAYD_SHARED_BVH_TRI_INLINE __host__ __device__ __forceinline__
#else
#define RAYD_SHARED_BVH_TRI_INLINE inline
#endif

namespace rayd::shared::bvh {

/// Result of a watertight ray/triangle test. On a miss `hit` is false and the
/// remaining fields are unspecified.
struct WatertightTriangleHit {
    bool hit;
    float t;
    float u; ///< Barycentric weight of vertex B (p0 + e1).
    float v; ///< Barycentric weight of vertex C (p0 + e2).
};

/// Correctly-rounded `a*b - c*d` (Kahan). This is FMA-contraction-proof: when
/// `a*b == c*d` mathematically it returns exactly 0.0f regardless of the
/// compiler's `--fmad` setting, which is what keeps the watertight edge tests
/// from mistaking an on-edge hit (a genuinely zero edge function) for a
/// mixed-sign miss.
RAYD_SHARED_BVH_TRI_INLINE float diff_of_products(float a, float b, float c, float d) {
    const float cd = c * d;
    const float error = fmaf(-c, d, cd); // exact rounding error of c*d
    const float diff = fmaf(a, b, -cd);  // a*b - cd, with a*b exact
    return diff + error;
}

/// Watertight ray/triangle intersection with no backface culling. Accepts hits
/// on triangle boundaries (an edge function of exactly zero) so a ray grazing a
/// shared edge is reported by both adjacent triangles; the caller's closest-hit
/// (t, primitive-id) reduction then selects a single deterministic winner, which
/// is what keeps a shared-edge crossing exactly-once in closest-hit semantics.
///
/// \param ox,oy,oz  Ray origin.
/// \param dx,dy,dz  Ray direction (need not be normalized; t is in its units).
/// \param ax,ay,az  Vertex A = p0.
/// \param bx,by,bz  Vertex B = p0 + e1.
/// \param cx,cy,cz  Vertex C = p0 + e2.
/// \param t_min     Inclusive lower bound on the accepted hit distance.
/// \param t_max     Inclusive upper bound on the accepted hit distance.
RAYD_SHARED_BVH_TRI_INLINE WatertightTriangleHit intersect_triangle_watertight(float ox, float oy, float oz, float dx,
                                                                               float dy, float dz, float ax, float ay,
                                                                               float az, float bx, float by, float bz,
                                                                               float cx, float cy, float cz,
                                                                               float t_min, float t_max) {
    WatertightTriangleHit result{false, 0.0f, 0.0f, 0.0f};

    // Vertices relative to the ray origin.
    float a[3] = {ax - ox, ay - oy, az - oz};
    float b[3] = {bx - ox, by - oy, bz - oz};
    float c[3] = {cx - ox, cy - oy, cz - oz};
    const float d[3] = {dx, dy, dz};

    // Pick kz = axis of largest |direction| and a cyclic permutation kx, ky.
    int kz = 0;
    float max_component = fabsf(d[0]);
    if (fabsf(d[1]) > max_component) {
        max_component = fabsf(d[1]);
        kz = 1;
    }
    if (fabsf(d[2]) > max_component) {
        kz = 2;
    }
    int kx = kz + 1;
    if (kx == 3) {
        kx = 0;
    }
    int ky = kx + 1;
    if (ky == 3) {
        ky = 0;
    }
    // Preserve winding when the ray points down the chosen axis.
    if (d[kz] < 0.0f) {
        const int swap = kx;
        kx = ky;
        ky = swap;
    }

    const float dz_axis = d[kz];
    // A degenerate direction cannot form a shear frame.
    if (dz_axis == 0.0f) {
        return result;
    }
    const float sx = d[kx] / dz_axis;
    const float sy = d[ky] / dz_axis;
    const float sz = 1.0f / dz_axis;

    // Shear and scale the vertices into ray space.
    const float ax2 = a[kx] - sx * a[kz];
    const float ay2 = a[ky] - sy * a[kz];
    const float bx2 = b[kx] - sx * b[kz];
    const float by2 = b[ky] - sy * b[kz];
    const float cx2 = c[kx] - sx * c[kz];
    const float cy2 = c[ky] - sy * c[kz];

    // Scaled barycentric coordinates: u_A = U, u_B = V, u_C = W. The
    // FMA-contraction-proof difference of products makes a mathematically zero
    // edge function come out exactly 0.0f (rather than a tiny residual with an
    // arbitrary sign that would spuriously reject an on-edge crossing).
    float u = diff_of_products(cx2, by2, cy2, bx2);
    float v = diff_of_products(ax2, cy2, ay2, cx2);
    float w = diff_of_products(bx2, ay2, by2, ax2);

    // Exact-zero edge functions fall back to double precision so shared-edge
    // signs stay consistent between adjacent triangles (no gaps).
    if (u == 0.0f || v == 0.0f || w == 0.0f) {
        const double cxby = static_cast<double>(cx2) * static_cast<double>(by2);
        const double cybx = static_cast<double>(cy2) * static_cast<double>(bx2);
        u = static_cast<float>(cxby - cybx);
        const double axcy = static_cast<double>(ax2) * static_cast<double>(cy2);
        const double aycx = static_cast<double>(ay2) * static_cast<double>(cx2);
        v = static_cast<float>(axcy - aycx);
        const double bxay = static_cast<double>(bx2) * static_cast<double>(ay2);
        const double byax = static_cast<double>(by2) * static_cast<double>(ax2);
        w = static_cast<float>(bxay - byax);
    }

    // No backface culling: reject only when the edge functions disagree in sign.
    if ((u < 0.0f || v < 0.0f || w < 0.0f) && (u > 0.0f || v > 0.0f || w > 0.0f)) {
        return result;
    }

    const float det = u + v + w;
    if (det == 0.0f) {
        return result;
    }

    // Scaled hit distance along the sheared axis.
    const float az2 = sz * a[kz];
    const float bz2 = sz * b[kz];
    const float cz2 = sz * c[kz];
    const float scaled_t = u * az2 + v * bz2 + w * cz2;

    const float rcp_det = 1.0f / det;
    const float t = scaled_t * rcp_det;
    if (t < t_min || t > t_max) {
        return result;
    }

    result.hit = true;
    result.t = t;
    result.u = v * rcp_det; // weight of vertex B == Moller-Trumbore u
    result.v = w * rcp_det; // weight of vertex C == Moller-Trumbore v
    return result;
}

} // namespace rayd::shared::bvh

#undef RAYD_SHARED_BVH_TRI_INLINE
