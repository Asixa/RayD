#pragma once

// Fixed-winner reflection-EPC chain: the specular chain solve shared by the
// OptiX discovery raygen and the geometry adjoint, plus its reverse-mode
// companions.
//
// The forward primitives (reflect_point_across_plane, intersect_segment_plane)
// live in reflection_geometry.h and are called from both sides, so the math has
// exactly one implementation. What this header adds is (a) the chain that
// orchestrates them, so the discovery kernel and the adjoint kernel cannot
// drift apart, and (b) the adjoint of each primitive, following the house style
// already used for the UTD operators (adj_fresnel_reflection_face and friends
// in shared/utd/utd_math.h).
//
// Contract: the winner is FROZEN. Which primitive each bounce hits, whether the
// point lands inside the triangle, and whether a segment is occluded are all
// discrete decisions taken by the forward; the adjoint differentiates only the
// continuous geometry of an already-selected chain. Nothing here traces a ray,
// so the adjoint needs no OptiX.

#include <rayd/shared/math/vec3.h>
#include <rayd/shared/reflection/reflection_geometry.h>

#if defined(__CUDACC__)
#  define RAYD_SHARED_EPC_INLINE __host__ __device__ __forceinline__
#else
#  define RAYD_SHARED_EPC_INLINE inline
#endif

namespace rayd::shared::reflection {

// Matches the discovery kernel's guards (reflection_epc_device.cuh).
constexpr float kEpcChainParallelTolerance = 1.0e-7f;
constexpr float kEpcChainSegmentTolerance = 1.0e-4f;
constexpr float kEpcChainMinNorm = 1.0e-20f;
// epc_normalize clamps the squared norm at this floor (bit-identical to the
// discovery raygen's normalize3); its derivative below must branch on the
// same comparison the forward takes.
constexpr float kEpcChainNormalizeMinSquaredNorm = 1.0e-12f;
// The consumers build the plane-normal / face-normal tables that the vertex
// chaining below differentiates as raw / fmaxf(|raw|, kEpcFaceNormalMinNorm)
// (channel_native's deterministic_normalize_vec3 with eps = 1e-6). This
// constant mirrors that eps so the face-normal derivative takes the same
// clamp branch as the table build; keep the two in sync.
constexpr float kEpcFaceNormalMinNorm = 1.0e-6f;

// Bit-identical to the discovery raygen's normalize3 (reflection_epc_device.cuh):
// the adjoint re-solves the chain from the very inputs the forward consumed, so
// any divergence here would differentiate a slightly different function than the
// one that produced the winner - and, at a tolerance edge, could reject a row the
// forward accepted. rsqrtf has no host counterpart, hence the guarded fallback;
// only the device path has to match.
RAYD_SHARED_EPC_INLINE math::Vec3f epc_normalize(math::Vec3f value) {
    const float squared =
        fmaxf(math::squared_norm(value), kEpcChainNormalizeMinSquaredNorm);
#if defined(__CUDACC__)
    return math::scale(value, rsqrtf(squared));
#else
    return math::scale(value, 1.0f / sqrtf(squared));
#endif
}

// --------------------------------------------------------------------------
// Adjoint companions of the forward primitives.
// --------------------------------------------------------------------------

// Adjoint of epc_normalize. Above the clamp, unit = raw / |raw| and the
// Jacobian is the projection off `unit`, scaled by 1/|raw|; it is idempotent,
// so re-normalizing an already-unit input (which the discovery kernel does)
// leaves the adjoint unchanged. At or below the clamp, the primal is the
// constant-denominator scale v / sqrt(kEpcChainNormalizeMinSquaredNorm),
// whose exact Jacobian is that same constant times the identity: the
// cotangent passes through whole, with no projection and no 1/|raw| blow-up.
RAYD_SHARED_EPC_INLINE math::Vec3f adj_normalize(
    math::Vec3f raw,
    math::Vec3f unit,
    math::Vec3f grad_unit) {
    const float squared = math::squared_norm(raw);
    if (squared <= kEpcChainNormalizeMinSquaredNorm) {
        return math::scale(
            grad_unit, 1.0f / sqrtf(kEpcChainNormalizeMinSquaredNorm));
    }
    const math::Vec3f tangential = math::subtract(
        grad_unit, math::scale(unit, math::dot(grad_unit, unit)));
    return math::scale(tangential, 1.0f / sqrtf(squared));
}

// Adjoint of the face-normal table normalize (see kEpcFaceNormalMinNorm):
// unit = raw / fmaxf(|raw|, kEpcFaceNormalMinNorm). Above the clamp this is
// the standard projection Jacobian scaled by 1/|raw|; at or below the clamp
// the primal is the constant scale v / kEpcFaceNormalMinNorm, so the exact
// Jacobian is the identity over the same frozen denominator.
RAYD_SHARED_EPC_INLINE math::Vec3f adj_face_table_normalize(
    math::Vec3f raw,
    math::Vec3f unit,
    math::Vec3f grad_unit) {
    const float norm = sqrtf(math::squared_norm(raw));
    if (norm <= kEpcFaceNormalMinNorm) {
        return math::scale(grad_unit, 1.0f / kEpcFaceNormalMinNorm);
    }
    const math::Vec3f tangential = math::subtract(
        grad_unit, math::scale(unit, math::dot(grad_unit, unit)));
    return math::scale(tangential, 1.0f / norm);
}

// out = point - 2 * dot(point - plane_point, n) * n, with n a unit normal.
RAYD_SHARED_EPC_INLINE void adj_reflect_point_across_plane(
    math::Vec3f point,
    math::Vec3f plane_point,
    math::Vec3f unit_normal,
    math::Vec3f grad_out,
    math::Vec3f &grad_point,
    math::Vec3f &grad_plane_point,
    math::Vec3f &grad_unit_normal) {
    const math::Vec3f offset = math::subtract(point, plane_point);
    const float distance = math::dot(offset, unit_normal);
    const float grad_dot_normal = math::dot(grad_out, unit_normal);

    // The reflection operator (I - 2 n n^T) is symmetric: it is its own adjoint.
    grad_point = math::add(
        grad_point,
        math::subtract(grad_out, math::scale(unit_normal, 2.0f * grad_dot_normal)));
    grad_plane_point = math::add(
        grad_plane_point, math::scale(unit_normal, 2.0f * grad_dot_normal));
    // n appears twice (in the projection and as the reflection axis).
    grad_unit_normal = math::subtract(
        grad_unit_normal,
        math::add(
            math::scale(offset, 2.0f * grad_dot_normal),
            math::scale(grad_out, 2.0f * distance)));
}

// hit = start + t * (end - start), t = dot(plane_point - start, N) / dot(end - start, N).
// `N` is the plane normal exactly as the forward consumed it, `t` and `denominator`
// come from the forward so the adjoint never re-derives them.
RAYD_SHARED_EPC_INLINE void adj_intersect_segment_plane(
    math::Vec3f segment_start,
    math::Vec3f segment_end,
    math::Vec3f plane_point,
    math::Vec3f plane_normal,
    float t,
    float denominator,
    math::Vec3f grad_hit,
    math::Vec3f &grad_start,
    math::Vec3f &grad_end,
    math::Vec3f &grad_plane_point,
    math::Vec3f &grad_plane_normal) {
    const math::Vec3f span = math::subtract(segment_end, segment_start);
    const float grad_t = math::dot(grad_hit, span);
    const float grad_numerator = grad_t / denominator;
    const float grad_denominator = -grad_t * t / denominator;

    grad_start = math::add(
        grad_start,
        math::subtract(
            math::scale(grad_hit, 1.0f - t),
            math::scale(plane_normal, grad_numerator + grad_denominator)));
    grad_end = math::add(
        grad_end,
        math::add(
            math::scale(grad_hit, t),
            math::scale(plane_normal, grad_denominator)));
    grad_plane_point = math::add(
        grad_plane_point, math::scale(plane_normal, grad_numerator));
    grad_plane_normal = math::add(
        grad_plane_normal,
        math::add(
            math::scale(
                math::subtract(plane_point, segment_start), grad_numerator),
            math::scale(span, grad_denominator)));
}

// length = |end - start|.
RAYD_SHARED_EPC_INLINE void adj_segment_length(
    math::Vec3f segment_start,
    math::Vec3f segment_end,
    float grad_length,
    math::Vec3f &grad_start,
    math::Vec3f &grad_end) {
    const math::Vec3f span = math::subtract(segment_end, segment_start);
    const float norm = sqrtf(math::squared_norm(span));
    if (norm <= kEpcChainMinNorm) {
        // A degenerate segment has no direction to move along; the primal takes
        // the same zero-derivative branch by construction.
        return;
    }
    const math::Vec3f direction = math::scale(span, grad_length / norm);
    grad_end = math::add(grad_end, direction);
    grad_start = math::subtract(grad_start, direction);
}

// Unit face normal normalize(cross(v1 - v0, v2 - v0)), computed exactly as
// the consumers build the face-normal table: the denominator is clamped at
// kEpcFaceNormalMinNorm, so a sliver face yields the same short (non-unit)
// vector the table stores, and the forward value, the tangent and the adjoint
// all agree on which branch a degenerate triangle takes.
RAYD_SHARED_EPC_INLINE math::Vec3f face_unit_normal(
    math::Vec3f v0,
    math::Vec3f v1,
    math::Vec3f v2) {
    const math::Vec3f raw =
        math::cross(math::subtract(v1, v0), math::subtract(v2, v0));
    const float norm = sqrtf(math::squared_norm(raw));
    return math::scale(raw, 1.0f / fmaxf(norm, kEpcFaceNormalMinNorm));
}

// The scene's face normal is normalize(cross(v1 - v0, v2 - v0)) and its anchor
// is v0 (see the triangle SoA build in the torch backend); this is the adjoint
// of that pair, so plane cotangents land on the winner triangle's vertices.
RAYD_SHARED_EPC_INLINE void adj_face_normal(
    math::Vec3f v0,
    math::Vec3f v1,
    math::Vec3f v2,
    math::Vec3f unit_normal,
    math::Vec3f grad_unit_normal,
    math::Vec3f &grad_v0,
    math::Vec3f &grad_v1,
    math::Vec3f &grad_v2) {
    const math::Vec3f edge1 = math::subtract(v1, v0);
    const math::Vec3f edge2 = math::subtract(v2, v0);
    const math::Vec3f raw = math::cross(edge1, edge2);
    const math::Vec3f grad_raw =
        adj_face_table_normalize(raw, unit_normal, grad_unit_normal);
    // c = e1 x e2  =>  g_e1 = e2 x g_c, g_e2 = g_c x e1.
    const math::Vec3f grad_edge1 = math::cross(edge2, grad_raw);
    const math::Vec3f grad_edge2 = math::cross(grad_raw, edge1);
    grad_v1 = math::add(grad_v1, grad_edge1);
    grad_v2 = math::add(grad_v2, grad_edge2);
    grad_v0 = math::subtract(grad_v0, math::add(grad_edge1, grad_edge2));
}

// Forward-mode companion of adj_face_normal: the tangent of the unit face
// normal under vertex tangents. The normalize Jacobian is symmetric, so the
// shared adj_face_table_normalize serves both directions and the pair stays
// exactly transposed.
RAYD_SHARED_EPC_INLINE math::Vec3f jvp_face_normal(
    math::Vec3f v0,
    math::Vec3f v1,
    math::Vec3f v2,
    math::Vec3f tangent_v0,
    math::Vec3f tangent_v1,
    math::Vec3f tangent_v2) {
    const math::Vec3f edge1 = math::subtract(v1, v0);
    const math::Vec3f edge2 = math::subtract(v2, v0);
    const math::Vec3f raw = math::cross(edge1, edge2);
    const float norm = sqrtf(math::squared_norm(raw));
    const math::Vec3f unit =
        math::scale(raw, 1.0f / fmaxf(norm, kEpcFaceNormalMinNorm));
    const math::Vec3f tangent_raw = math::add(
        math::cross(math::subtract(tangent_v1, tangent_v0), edge2),
        math::cross(edge1, math::subtract(tangent_v2, tangent_v0)));
    return adj_face_table_normalize(raw, unit, tangent_raw);
}

// --------------------------------------------------------------------------
// The chain itself.
// --------------------------------------------------------------------------

template <int MaxBounces>
struct EpcChain {
    // image[0] is the source; image[b + 1] is the source mirrored through
    // planes 0..b. hit[b] is the interaction point on plane b.
    math::Vec3f image[MaxBounces + 1];
    math::Vec3f hit[MaxBounces];
    math::Vec3f unit_normal[MaxBounces];
    // Intersection parameter and denominator per bounce, kept so the adjoint
    // reuses the forward's values instead of recomputing them.
    float t[MaxBounces];
    float denominator[MaxBounces];
    float path_length;
    int bounces;
};

RAYD_SHARED_EPC_INLINE bool epc_is_finite(math::Vec3f v) {
#if defined(__CUDACC__)
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
#else
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
#endif
}

// Back-trace of an already-mirrored specular chain plus its path length: walk
// from the receiver, intersect each plane in reverse, then sum the B + 1 free
// segments. Shared verbatim by the discovery raygen and solve_epc_chain so the
// geometry has one implementation. ``unit_normals`` must already be unit;
// ``image_sources`` has bounces + 1 entries with image_sources[b + 1] the source
// mirrored through planes 0..b. ``t_out`` / ``denominator_out`` are optional
// outputs the adjoint reuses. Returns false on the same parallel / range /
// non-finite guards intersect_line_plane takes in the discovery kernel.
template <int MaxBounces>
RAYD_SHARED_EPC_INLINE bool epc_backtrace_and_length(
    const math::Vec3f *plane_points,
    const math::Vec3f *unit_normals,
    const math::Vec3f *image_sources,
    int bounces,
    math::Vec3f source,
    math::Vec3f receiver,
    math::Vec3f *hits,
    float *t_out,
    float *denominator_out,
    float &path_length) {
    math::Vec3f endpoint = receiver;
    for (int bounce = bounces - 1; bounce >= 0; --bounce) {
        const math::Vec3f start = image_sources[bounce + 1];
        const math::Vec3f unit_normal = unit_normals[bounce];
        const math::Vec3f span = math::subtract(endpoint, start);
        const float denominator = math::dot(span, unit_normal);
        if (fabsf(denominator) <= kEpcChainParallelTolerance) {
            return false;
        }
        const float t = math::dot(
            math::subtract(plane_points[bounce], start), unit_normal) / denominator;
        if (t < -kEpcChainSegmentTolerance || t > 1.0f + kEpcChainSegmentTolerance) {
            return false;
        }
        const math::Vec3f point = math::add(start, math::scale(span, t));
        if (!epc_is_finite(point)) {
            return false;
        }
        if (t_out != nullptr) {
            t_out[bounce] = t;
        }
        if (denominator_out != nullptr) {
            denominator_out[bounce] = denominator;
        }
        hits[bounce] = point;
        endpoint = point;
    }

    float length = 0.0f;
    for (int segment = 0; segment <= bounces; ++segment) {
        const math::Vec3f start = segment == 0 ? source : hits[segment - 1];
        const math::Vec3f end = segment == bounces ? receiver : hits[segment];
        length += sqrtf(math::squared_norm(math::subtract(end, start)));
    }
    path_length = length;
    return true;
}

// Solves the specular chain for an already-selected plane sequence. Mirrors the
// discovery kernel exactly: mirror the source through each plane in order, then
// back-trace and sum via the shared epc_backtrace_and_length. Returns false on
// the same degenerate guards the discovery kernel takes.
template <int MaxBounces>
RAYD_SHARED_EPC_INLINE bool solve_epc_chain(
    const math::Vec3f *plane_points,
    const math::Vec3f *plane_normals,
    int bounces,
    math::Vec3f source,
    math::Vec3f receiver,
    EpcChain<MaxBounces> &chain) {
    if (bounces <= 0 || bounces > MaxBounces) {
        return false;
    }
    chain.bounces = bounces;
    chain.image[0] = source;
    for (int bounce = 0; bounce < bounces; ++bounce) {
        const math::Vec3f unit_normal =
            epc_normalize(plane_normals[bounce]);
        if (math::squared_norm(unit_normal) <= 0.0f) {
            return false;
        }
        chain.unit_normal[bounce] = unit_normal;
        chain.image[bounce + 1] = reflect_point_across_plane(
            chain.image[bounce], plane_points[bounce], unit_normal);
    }
    return epc_backtrace_and_length<MaxBounces>(
        plane_points, chain.unit_normal, chain.image, bounces, source, receiver,
        chain.hit, chain.t, chain.denominator, chain.path_length);
}

// Reverse mode of solve_epc_chain. Cotangents of the hits, of the unit normals
// the forward emitted, and of the path length are pushed back to the source,
// the receiver, and each plane (anchor + normal as the forward consumed it).
// Chaining the plane cotangents to mesh vertices is the caller's job, since only
// the caller knows which triangle each bounce belongs to (adj_face_normal above).
template <int MaxBounces>
RAYD_SHARED_EPC_INLINE void adj_solve_epc_chain(
    const EpcChain<MaxBounces> &chain,
    const math::Vec3f *plane_points,
    const math::Vec3f *plane_normals,
    math::Vec3f source,
    math::Vec3f receiver,
    const math::Vec3f *grad_hits,
    const math::Vec3f *grad_unit_normals,
    float grad_path_length,
    math::Vec3f &grad_source,
    math::Vec3f &grad_receiver,
    math::Vec3f *grad_plane_points,
    math::Vec3f *grad_plane_normals) {
    const int bounces = chain.bounces;
    const math::Vec3f zero = math::make_vec3(0.0f, 0.0f, 0.0f);

    math::Vec3f grad_hit[MaxBounces];
    math::Vec3f grad_unit_normal[MaxBounces];
    math::Vec3f grad_image[MaxBounces + 1];
    for (int bounce = 0; bounce < bounces; ++bounce) {
        grad_hit[bounce] = grad_hits != nullptr ? grad_hits[bounce] : zero;
        grad_unit_normal[bounce] =
            grad_unit_normals != nullptr ? grad_unit_normals[bounce] : zero;
    }
    for (int bounce = 0; bounce <= bounces; ++bounce) {
        grad_image[bounce] = zero;
    }
    for (int bounce = 0; bounce < bounces; ++bounce) {
        grad_plane_points[bounce] = zero;
    }
    grad_source = zero;
    grad_receiver = zero;

    // 1. Path length: fold its cotangent into the endpoints it is built from,
    //    so the chain adjoints below see a complete cotangent on every hit.
    if (grad_path_length != 0.0f) {
        for (int segment = 0; segment <= bounces; ++segment) {
            const math::Vec3f start =
                segment == 0 ? source : chain.hit[segment - 1];
            const math::Vec3f end =
                segment == bounces ? receiver : chain.hit[segment];
            math::Vec3f grad_start = zero;
            math::Vec3f grad_end = zero;
            adj_segment_length(
                start, end, grad_path_length, grad_start, grad_end);
            if (segment == 0) {
                grad_source = math::add(grad_source, grad_start);
            } else {
                grad_hit[segment - 1] =
                    math::add(grad_hit[segment - 1], grad_start);
            }
            if (segment == bounces) {
                grad_receiver = math::add(grad_receiver, grad_end);
            } else {
                grad_hit[segment] = math::add(grad_hit[segment], grad_end);
            }
        }
    }

    // 2. Back-trace: hit[b] was solved against endpoint hit[b + 1] (the receiver
    //    for the last bounce), so cotangents flow from hit[0] outward. Running
    //    b upward means every hit's cotangent is complete when it is consumed.
    for (int bounce = 0; bounce < bounces; ++bounce) {
        const math::Vec3f start = chain.image[bounce + 1];
        const math::Vec3f endpoint =
            bounce == bounces - 1 ? receiver : chain.hit[bounce + 1];
        math::Vec3f grad_start = zero;
        math::Vec3f grad_end = zero;
        adj_intersect_segment_plane(
            start,
            endpoint,
            plane_points[bounce],
            chain.unit_normal[bounce],
            chain.t[bounce],
            chain.denominator[bounce],
            grad_hit[bounce],
            grad_start,
            grad_end,
            grad_plane_points[bounce],
            grad_unit_normal[bounce]);
        grad_image[bounce + 1] = math::add(grad_image[bounce + 1], grad_start);
        if (bounce == bounces - 1) {
            grad_receiver = math::add(grad_receiver, grad_end);
        } else {
            grad_hit[bounce + 1] = math::add(grad_hit[bounce + 1], grad_end);
        }
    }

    // 3. Image chain: image[b + 1] = mirror(image[b]) walks backwards to the source.
    for (int bounce = bounces - 1; bounce >= 0; --bounce) {
        adj_reflect_point_across_plane(
            chain.image[bounce],
            plane_points[bounce],
            chain.unit_normal[bounce],
            grad_image[bounce + 1],
            grad_image[bounce],
            grad_plane_points[bounce],
            grad_unit_normal[bounce]);
    }
    grad_source = math::add(grad_source, grad_image[0]);

    // 4. The forward normalizes the incoming plane normal once and uses that unit
    //    vector everywhere (mirror, intersection, emitted normal), so every
    //    contribution to it is complete only here. The Jacobian scales by the
    //    RAW normal's (clamped) length, which is why the raw array is needed
    //    and not just the unit vector the chain kept.
    for (int bounce = 0; bounce < bounces; ++bounce) {
        grad_plane_normals[bounce] = adj_normalize(
            plane_normals[bounce],
            chain.unit_normal[bounce],
            grad_unit_normal[bounce]);
    }
}

// Forward mode of solve_epc_chain: pushes tangents of the source, the
// receiver and each plane (anchor + raw normal, exactly the inputs the
// forward consumed) through the solved chain. Every step linearizes the same
// decomposition adj_solve_epc_chain differentiates (stored t / denominator,
// shared adj_normalize), so the pair is exactly transposed by construction.
template <int MaxBounces>
RAYD_SHARED_EPC_INLINE void jvp_solve_epc_chain(
    const EpcChain<MaxBounces> &chain,
    const math::Vec3f *plane_points,
    const math::Vec3f *plane_normals,
    math::Vec3f source,
    math::Vec3f receiver,
    math::Vec3f tangent_source,
    math::Vec3f tangent_receiver,
    const math::Vec3f *tangent_plane_points,
    const math::Vec3f *tangent_plane_normals,
    math::Vec3f *tangent_hits,
    math::Vec3f *tangent_unit_normals,
    float &tangent_path_length) {
    const int bounces = chain.bounces;
    const math::Vec3f zero = math::make_vec3(0.0f, 0.0f, 0.0f);

    // 1. Mirror loop: image[b + 1] = image[b] - 2 * dot(image[b] - p0, n) * n,
    //    with n = normalize(N). The normalize Jacobian is symmetric, so the
    //    adjoint helper doubles as the tangent map.
    math::Vec3f tangent_image[MaxBounces + 1];
    tangent_image[0] = tangent_source;
    for (int bounce = 0; bounce < bounces; ++bounce) {
        const math::Vec3f tangent_plane_point =
            tangent_plane_points != nullptr ? tangent_plane_points[bounce] : zero;
        const math::Vec3f tangent_raw_normal =
            tangent_plane_normals != nullptr ? tangent_plane_normals[bounce] : zero;
        const math::Vec3f unit_normal = chain.unit_normal[bounce];
        const math::Vec3f tangent_normal = adj_normalize(
            plane_normals[bounce], unit_normal, tangent_raw_normal);
        tangent_unit_normals[bounce] = tangent_normal;

        const math::Vec3f offset =
            math::subtract(chain.image[bounce], plane_points[bounce]);
        const float distance = math::dot(offset, unit_normal);
        const float tangent_distance =
            math::dot(
                math::subtract(tangent_image[bounce], tangent_plane_point),
                unit_normal) +
            math::dot(offset, tangent_normal);
        tangent_image[bounce + 1] = math::subtract(
            tangent_image[bounce],
            math::add(
                math::scale(unit_normal, 2.0f * tangent_distance),
                math::scale(tangent_normal, 2.0f * distance)));
    }

    // 2. Back-trace: hit[b] = S + t * (E - S) with t = num / den; iterate in
    //    the primal's order so tangent_hits[b + 1] is ready when consumed.
    for (int bounce = bounces - 1; bounce >= 0; --bounce) {
        const math::Vec3f start = chain.image[bounce + 1];
        const math::Vec3f endpoint =
            bounce == bounces - 1 ? receiver : chain.hit[bounce + 1];
        const math::Vec3f tangent_start = tangent_image[bounce + 1];
        const math::Vec3f tangent_end =
            bounce == bounces - 1 ? tangent_receiver : tangent_hits[bounce + 1];
        const math::Vec3f tangent_plane_point =
            tangent_plane_points != nullptr ? tangent_plane_points[bounce] : zero;
        const math::Vec3f unit_normal = chain.unit_normal[bounce];
        const math::Vec3f tangent_normal = tangent_unit_normals[bounce];
        const math::Vec3f span = math::subtract(endpoint, start);
        const math::Vec3f tangent_span = math::subtract(tangent_end, tangent_start);
        const float tangent_denominator =
            math::dot(tangent_span, unit_normal) + math::dot(span, tangent_normal);
        const float tangent_numerator =
            math::dot(
                math::subtract(tangent_plane_point, tangent_start), unit_normal) +
            math::dot(
                math::subtract(plane_points[bounce], start), tangent_normal);
        const float tangent_t =
            (tangent_numerator - chain.t[bounce] * tangent_denominator) /
            chain.denominator[bounce];
        tangent_hits[bounce] = math::add(
            tangent_start,
            math::add(
                math::scale(span, tangent_t),
                math::scale(tangent_span, chain.t[bounce])));
    }

    // 3. Path length: d|v| = dot(v, dv) / |v| per segment, zero on the same
    //    degenerate guard adj_segment_length takes.
    float tangent_length = 0.0f;
    for (int segment = 0; segment <= bounces; ++segment) {
        const math::Vec3f start =
            segment == 0 ? source : chain.hit[segment - 1];
        const math::Vec3f end =
            segment == bounces ? receiver : chain.hit[segment];
        const math::Vec3f tangent_start =
            segment == 0 ? tangent_source : tangent_hits[segment - 1];
        const math::Vec3f tangent_end =
            segment == bounces ? tangent_receiver : tangent_hits[segment];
        const math::Vec3f span = math::subtract(end, start);
        const float norm = sqrtf(math::squared_norm(span));
        if (norm <= kEpcChainMinNorm) {
            continue;
        }
        tangent_length +=
            math::dot(span, math::subtract(tangent_end, tangent_start)) / norm;
    }
    tangent_path_length = tangent_length;
}

} // namespace rayd::shared::reflection

#undef RAYD_SHARED_EPC_INLINE
