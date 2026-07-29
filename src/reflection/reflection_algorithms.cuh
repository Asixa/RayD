// Copyright Xingyu Chen.
// Defines host-safe reflection geometry, trace, accumulation, and EPC algorithms.

#pragma once

#include <rayd/math.h>

#if defined(__CUDACC__)
#define RAYD_SHARED_REFLECTION_INLINE __host__ __device__ __forceinline__
#else
#define RAYD_SHARED_REFLECTION_INLINE inline
#endif

namespace rayd::shared::reflection {

RAYD_SHARED_REFLECTION_INLINE math::Vec3f orient_normal_against(math::Vec3f incident_direction,
                                                                math::Vec3f unit_normal) {
    return math::dot(incident_direction, unit_normal) > 0.0f ? math::scale(unit_normal, -1.0f) : unit_normal;
}

RAYD_SHARED_REFLECTION_INLINE math::Vec3f reflect_direction(math::Vec3f incident_direction, math::Vec3f unit_normal) {
    return math::subtract(incident_direction,
                          math::scale(unit_normal, 2.0f * math::dot(incident_direction, unit_normal)));
}

RAYD_SHARED_REFLECTION_INLINE math::Vec3f reflect_point_across_plane(math::Vec3f point, math::Vec3f plane_point,
                                                                     math::Vec3f unit_normal) {
    return math::subtract(point,
                          math::scale(unit_normal, 2.0f * math::dot(math::subtract(point, plane_point), unit_normal)));
}

RAYD_SHARED_REFLECTION_INLINE bool intersect_segment_plane(math::Vec3f segment_start, math::Vec3f segment_end,
                                                           math::Vec3f plane_point, math::Vec3f plane_normal,
                                                           float parallel_tolerance, float segment_tolerance,
                                                           math::Vec3f& intersection) {
    const math::Vec3f direction = math::subtract(segment_end, segment_start);
    const float denominator = math::dot(direction, plane_normal);
    if (fabsf(denominator) <= parallel_tolerance) {
        return false;
    }
    const float t = math::dot(math::subtract(plane_point, segment_start), plane_normal) / denominator;
    if (t < -segment_tolerance || t > 1.0f + segment_tolerance) {
        return false;
    }
    intersection = math::add(segment_start, math::scale(direction, t));
    return true;
}

} // namespace rayd::shared::reflection

#undef RAYD_SHARED_REFLECTION_INLINE

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
// in shared/diffraction/utd.h).
//
// Contract: the winner is FROZEN. Which primitive each bounce hits, whether the
// point lands inside the triangle, and whether a segment is occluded are all
// discrete decisions taken by the forward; the adjoint differentiates only the
// continuous geometry of an already-selected chain. Nothing here traces a ray,
// so the adjoint needs no OptiX.

#include <rayd/math.h>

#if defined(__CUDACC__)
#define RAYD_SHARED_EPC_INLINE __host__ __device__ __forceinline__
#else
#define RAYD_SHARED_EPC_INLINE inline
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
    const float squared = fmaxf(math::squared_norm(value), kEpcChainNormalizeMinSquaredNorm);
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
RAYD_SHARED_EPC_INLINE math::Vec3f adj_normalize(math::Vec3f raw, math::Vec3f unit, math::Vec3f grad_unit) {
    const float squared = math::squared_norm(raw);
    if (squared <= kEpcChainNormalizeMinSquaredNorm) {
        return math::scale(grad_unit, 1.0f / sqrtf(kEpcChainNormalizeMinSquaredNorm));
    }
    const math::Vec3f tangential = math::subtract(grad_unit, math::scale(unit, math::dot(grad_unit, unit)));
    return math::scale(tangential, 1.0f / sqrtf(squared));
}

// Adjoint of the face-normal table normalize (see kEpcFaceNormalMinNorm):
// unit = raw / fmaxf(|raw|, kEpcFaceNormalMinNorm). Above the clamp this is
// the standard projection Jacobian scaled by 1/|raw|; at or below the clamp
// the primal is the constant scale v / kEpcFaceNormalMinNorm, so the exact
// Jacobian is the identity over the same frozen denominator.
RAYD_SHARED_EPC_INLINE math::Vec3f adj_face_table_normalize(math::Vec3f raw, math::Vec3f unit, math::Vec3f grad_unit) {
    const float norm = sqrtf(math::squared_norm(raw));
    if (norm <= kEpcFaceNormalMinNorm) {
        return math::scale(grad_unit, 1.0f / kEpcFaceNormalMinNorm);
    }
    const math::Vec3f tangential = math::subtract(grad_unit, math::scale(unit, math::dot(grad_unit, unit)));
    return math::scale(tangential, 1.0f / norm);
}

// out = point - 2 * dot(point - plane_point, n) * n, with n a unit normal.
RAYD_SHARED_EPC_INLINE void adj_reflect_point_across_plane(math::Vec3f point, math::Vec3f plane_point,
                                                           math::Vec3f unit_normal, math::Vec3f grad_out,
                                                           math::Vec3f& grad_point, math::Vec3f& grad_plane_point,
                                                           math::Vec3f& grad_unit_normal) {
    const math::Vec3f offset = math::subtract(point, plane_point);
    const float distance = math::dot(offset, unit_normal);
    const float grad_dot_normal = math::dot(grad_out, unit_normal);

    // The reflection operator (I - 2 n n^T) is symmetric: it is its own adjoint.
    grad_point = math::add(grad_point, math::subtract(grad_out, math::scale(unit_normal, 2.0f * grad_dot_normal)));
    grad_plane_point = math::add(grad_plane_point, math::scale(unit_normal, 2.0f * grad_dot_normal));
    // n appears twice (in the projection and as the reflection axis).
    grad_unit_normal = math::subtract(grad_unit_normal, math::add(math::scale(offset, 2.0f * grad_dot_normal),
                                                                  math::scale(grad_out, 2.0f * distance)));
}

// hit = start + t * (end - start), t = dot(plane_point - start, N) / dot(end - start, N).
// `N` is the plane normal exactly as the forward consumed it, `t` and `denominator`
// come from the forward so the adjoint never re-derives them.
RAYD_SHARED_EPC_INLINE void adj_intersect_segment_plane(math::Vec3f segment_start, math::Vec3f segment_end,
                                                        math::Vec3f plane_point, math::Vec3f plane_normal, float t,
                                                        float denominator, math::Vec3f grad_hit,
                                                        math::Vec3f& grad_start, math::Vec3f& grad_end,
                                                        math::Vec3f& grad_plane_point, math::Vec3f& grad_plane_normal) {
    const math::Vec3f span = math::subtract(segment_end, segment_start);
    const float grad_t = math::dot(grad_hit, span);
    const float grad_numerator = grad_t / denominator;
    const float grad_denominator = -grad_t * t / denominator;

    grad_start = math::add(grad_start, math::subtract(math::scale(grad_hit, 1.0f - t),
                                                      math::scale(plane_normal, grad_numerator + grad_denominator)));
    grad_end = math::add(grad_end, math::add(math::scale(grad_hit, t), math::scale(plane_normal, grad_denominator)));
    grad_plane_point = math::add(grad_plane_point, math::scale(plane_normal, grad_numerator));
    grad_plane_normal =
        math::add(grad_plane_normal, math::add(math::scale(math::subtract(plane_point, segment_start), grad_numerator),
                                               math::scale(span, grad_denominator)));
}

// length = |end - start|.
RAYD_SHARED_EPC_INLINE void adj_segment_length(math::Vec3f segment_start, math::Vec3f segment_end, float grad_length,
                                               math::Vec3f& grad_start, math::Vec3f& grad_end) {
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
RAYD_SHARED_EPC_INLINE math::Vec3f face_unit_normal(math::Vec3f v0, math::Vec3f v1, math::Vec3f v2) {
    const math::Vec3f raw = math::cross(math::subtract(v1, v0), math::subtract(v2, v0));
    const float norm = sqrtf(math::squared_norm(raw));
    return math::scale(raw, 1.0f / fmaxf(norm, kEpcFaceNormalMinNorm));
}

// The scene's face normal is normalize(cross(v1 - v0, v2 - v0)) and its anchor
// is v0 (see the triangle SoA build in the torch backend); this is the adjoint
// of that pair, so plane cotangents land on the winner triangle's vertices.
RAYD_SHARED_EPC_INLINE void adj_face_normal(math::Vec3f v0, math::Vec3f v1, math::Vec3f v2, math::Vec3f unit_normal,
                                            math::Vec3f grad_unit_normal, math::Vec3f& grad_v0, math::Vec3f& grad_v1,
                                            math::Vec3f& grad_v2) {
    const math::Vec3f edge1 = math::subtract(v1, v0);
    const math::Vec3f edge2 = math::subtract(v2, v0);
    const math::Vec3f raw = math::cross(edge1, edge2);
    const math::Vec3f grad_raw = adj_face_table_normalize(raw, unit_normal, grad_unit_normal);
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
RAYD_SHARED_EPC_INLINE math::Vec3f jvp_face_normal(math::Vec3f v0, math::Vec3f v1, math::Vec3f v2,
                                                   math::Vec3f tangent_v0, math::Vec3f tangent_v1,
                                                   math::Vec3f tangent_v2) {
    const math::Vec3f edge1 = math::subtract(v1, v0);
    const math::Vec3f edge2 = math::subtract(v2, v0);
    const math::Vec3f raw = math::cross(edge1, edge2);
    const float norm = sqrtf(math::squared_norm(raw));
    const math::Vec3f unit = math::scale(raw, 1.0f / fmaxf(norm, kEpcFaceNormalMinNorm));
    const math::Vec3f tangent_raw = math::add(math::cross(math::subtract(tangent_v1, tangent_v0), edge2),
                                              math::cross(edge1, math::subtract(tangent_v2, tangent_v0)));
    return adj_face_table_normalize(raw, unit, tangent_raw);
}

// --------------------------------------------------------------------------
// The chain itself.
// --------------------------------------------------------------------------

template <int MaxBounces> struct EpcChain {
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
RAYD_SHARED_EPC_INLINE bool epc_backtrace_and_length(const math::Vec3f* plane_points, const math::Vec3f* unit_normals,
                                                     const math::Vec3f* image_sources, int bounces, math::Vec3f source,
                                                     math::Vec3f receiver, math::Vec3f* hits, float* t_out,
                                                     float* denominator_out, float& path_length) {
    math::Vec3f endpoint = receiver;
    for (int bounce = bounces - 1; bounce >= 0; --bounce) {
        const math::Vec3f start = image_sources[bounce + 1];
        const math::Vec3f unit_normal = unit_normals[bounce];
        const math::Vec3f span = math::subtract(endpoint, start);
        const float denominator = math::dot(span, unit_normal);
        if (fabsf(denominator) <= kEpcChainParallelTolerance) {
            return false;
        }
        const float t = math::dot(math::subtract(plane_points[bounce], start), unit_normal) / denominator;
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
RAYD_SHARED_EPC_INLINE bool solve_epc_chain(const math::Vec3f* plane_points, const math::Vec3f* plane_normals,
                                            int bounces, math::Vec3f source, math::Vec3f receiver,
                                            EpcChain<MaxBounces>& chain) {
    if (bounces <= 0 || bounces > MaxBounces) {
        return false;
    }
    chain.bounces = bounces;
    chain.image[0] = source;
    for (int bounce = 0; bounce < bounces; ++bounce) {
        const math::Vec3f unit_normal = epc_normalize(plane_normals[bounce]);
        if (math::squared_norm(unit_normal) <= 0.0f) {
            return false;
        }
        chain.unit_normal[bounce] = unit_normal;
        chain.image[bounce + 1] = reflect_point_across_plane(chain.image[bounce], plane_points[bounce], unit_normal);
    }
    return epc_backtrace_and_length<MaxBounces>(plane_points, chain.unit_normal, chain.image, bounces, source, receiver,
                                                chain.hit, chain.t, chain.denominator, chain.path_length);
}

// Reverse mode of solve_epc_chain. Cotangents of the hits, of the unit normals
// the forward emitted, and of the path length are pushed back to the source,
// the receiver, and each plane (anchor + normal as the forward consumed it).
// Chaining the plane cotangents to mesh vertices is the caller's job, since only
// the caller knows which triangle each bounce belongs to (adj_face_normal above).
template <int MaxBounces>
RAYD_SHARED_EPC_INLINE void adj_solve_epc_chain(const EpcChain<MaxBounces>& chain, const math::Vec3f* plane_points,
                                                const math::Vec3f* plane_normals, math::Vec3f source,
                                                math::Vec3f receiver, const math::Vec3f* grad_hits,
                                                const math::Vec3f* grad_unit_normals, float grad_path_length,
                                                math::Vec3f& grad_source, math::Vec3f& grad_receiver,
                                                math::Vec3f* grad_plane_points, math::Vec3f* grad_plane_normals) {
    const int bounces = chain.bounces;
    const math::Vec3f zero = math::make_vec3(0.0f, 0.0f, 0.0f);

    math::Vec3f grad_hit[MaxBounces];
    math::Vec3f grad_unit_normal[MaxBounces];
    math::Vec3f grad_image[MaxBounces + 1];
    for (int bounce = 0; bounce < bounces; ++bounce) {
        grad_hit[bounce] = grad_hits != nullptr ? grad_hits[bounce] : zero;
        grad_unit_normal[bounce] = grad_unit_normals != nullptr ? grad_unit_normals[bounce] : zero;
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
            const math::Vec3f start = segment == 0 ? source : chain.hit[segment - 1];
            const math::Vec3f end = segment == bounces ? receiver : chain.hit[segment];
            math::Vec3f grad_start = zero;
            math::Vec3f grad_end = zero;
            adj_segment_length(start, end, grad_path_length, grad_start, grad_end);
            if (segment == 0) {
                grad_source = math::add(grad_source, grad_start);
            } else {
                grad_hit[segment - 1] = math::add(grad_hit[segment - 1], grad_start);
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
        const math::Vec3f endpoint = bounce == bounces - 1 ? receiver : chain.hit[bounce + 1];
        math::Vec3f grad_start = zero;
        math::Vec3f grad_end = zero;
        adj_intersect_segment_plane(start, endpoint, plane_points[bounce], chain.unit_normal[bounce], chain.t[bounce],
                                    chain.denominator[bounce], grad_hit[bounce], grad_start, grad_end,
                                    grad_plane_points[bounce], grad_unit_normal[bounce]);
        grad_image[bounce + 1] = math::add(grad_image[bounce + 1], grad_start);
        if (bounce == bounces - 1) {
            grad_receiver = math::add(grad_receiver, grad_end);
        } else {
            grad_hit[bounce + 1] = math::add(grad_hit[bounce + 1], grad_end);
        }
    }

    // 3. Image chain: image[b + 1] = mirror(image[b]) walks backwards to the source.
    for (int bounce = bounces - 1; bounce >= 0; --bounce) {
        adj_reflect_point_across_plane(chain.image[bounce], plane_points[bounce], chain.unit_normal[bounce],
                                       grad_image[bounce + 1], grad_image[bounce], grad_plane_points[bounce],
                                       grad_unit_normal[bounce]);
    }
    grad_source = math::add(grad_source, grad_image[0]);

    // 4. The forward normalizes the incoming plane normal once and uses that unit
    //    vector everywhere (mirror, intersection, emitted normal), so every
    //    contribution to it is complete only here. The Jacobian scales by the
    //    RAW normal's (clamped) length, which is why the raw array is needed
    //    and not just the unit vector the chain kept.
    for (int bounce = 0; bounce < bounces; ++bounce) {
        grad_plane_normals[bounce] =
            adj_normalize(plane_normals[bounce], chain.unit_normal[bounce], grad_unit_normal[bounce]);
    }
}

// Forward mode of solve_epc_chain: pushes tangents of the source, the
// receiver and each plane (anchor + raw normal, exactly the inputs the
// forward consumed) through the solved chain. Every step linearizes the same
// decomposition adj_solve_epc_chain differentiates (stored t / denominator,
// shared adj_normalize), so the pair is exactly transposed by construction.
template <int MaxBounces>
RAYD_SHARED_EPC_INLINE void jvp_solve_epc_chain(const EpcChain<MaxBounces>& chain, const math::Vec3f* plane_points,
                                                const math::Vec3f* plane_normals, math::Vec3f source,
                                                math::Vec3f receiver, math::Vec3f tangent_source,
                                                math::Vec3f tangent_receiver, const math::Vec3f* tangent_plane_points,
                                                const math::Vec3f* tangent_plane_normals, math::Vec3f* tangent_hits,
                                                math::Vec3f* tangent_unit_normals, float& tangent_path_length) {
    const int bounces = chain.bounces;
    const math::Vec3f zero = math::make_vec3(0.0f, 0.0f, 0.0f);

    // 1. Mirror loop: image[b + 1] = image[b] - 2 * dot(image[b] - p0, n) * n,
    //    with n = normalize(N). The normalize Jacobian is symmetric, so the
    //    adjoint helper doubles as the tangent map.
    math::Vec3f tangent_image[MaxBounces + 1];
    tangent_image[0] = tangent_source;
    for (int bounce = 0; bounce < bounces; ++bounce) {
        const math::Vec3f tangent_plane_point = tangent_plane_points != nullptr ? tangent_plane_points[bounce] : zero;
        const math::Vec3f tangent_raw_normal = tangent_plane_normals != nullptr ? tangent_plane_normals[bounce] : zero;
        const math::Vec3f unit_normal = chain.unit_normal[bounce];
        const math::Vec3f tangent_normal = adj_normalize(plane_normals[bounce], unit_normal, tangent_raw_normal);
        tangent_unit_normals[bounce] = tangent_normal;

        const math::Vec3f offset = math::subtract(chain.image[bounce], plane_points[bounce]);
        const float distance = math::dot(offset, unit_normal);
        const float tangent_distance =
            math::dot(math::subtract(tangent_image[bounce], tangent_plane_point), unit_normal) +
            math::dot(offset, tangent_normal);
        tangent_image[bounce + 1] =
            math::subtract(tangent_image[bounce], math::add(math::scale(unit_normal, 2.0f * tangent_distance),
                                                            math::scale(tangent_normal, 2.0f * distance)));
    }

    // 2. Back-trace: hit[b] = S + t * (E - S) with t = num / den; iterate in
    //    the primal's order so tangent_hits[b + 1] is ready when consumed.
    for (int bounce = bounces - 1; bounce >= 0; --bounce) {
        const math::Vec3f start = chain.image[bounce + 1];
        const math::Vec3f endpoint = bounce == bounces - 1 ? receiver : chain.hit[bounce + 1];
        const math::Vec3f tangent_start = tangent_image[bounce + 1];
        const math::Vec3f tangent_end = bounce == bounces - 1 ? tangent_receiver : tangent_hits[bounce + 1];
        const math::Vec3f tangent_plane_point = tangent_plane_points != nullptr ? tangent_plane_points[bounce] : zero;
        const math::Vec3f unit_normal = chain.unit_normal[bounce];
        const math::Vec3f tangent_normal = tangent_unit_normals[bounce];
        const math::Vec3f span = math::subtract(endpoint, start);
        const math::Vec3f tangent_span = math::subtract(tangent_end, tangent_start);
        const float tangent_denominator = math::dot(tangent_span, unit_normal) + math::dot(span, tangent_normal);
        const float tangent_numerator = math::dot(math::subtract(tangent_plane_point, tangent_start), unit_normal) +
                                        math::dot(math::subtract(plane_points[bounce], start), tangent_normal);
        const float tangent_t = (tangent_numerator - chain.t[bounce] * tangent_denominator) / chain.denominator[bounce];
        tangent_hits[bounce] = math::add(tangent_start, math::add(math::scale(span, tangent_t),
                                                                  math::scale(tangent_span, chain.t[bounce])));
    }

    // 3. Path length: d|v| = dot(v, dv) / |v| per segment, zero on the same
    //    degenerate guard adj_segment_length takes.
    float tangent_length = 0.0f;
    for (int segment = 0; segment <= bounces; ++segment) {
        const math::Vec3f start = segment == 0 ? source : chain.hit[segment - 1];
        const math::Vec3f end = segment == bounces ? receiver : chain.hit[segment];
        const math::Vec3f tangent_start = segment == 0 ? tangent_source : tangent_hits[segment - 1];
        const math::Vec3f tangent_end = segment == bounces ? tangent_receiver : tangent_hits[segment];
        const math::Vec3f span = math::subtract(end, start);
        const float norm = sqrtf(math::squared_norm(span));
        if (norm <= kEpcChainMinNorm) {
            continue;
        }
        tangent_length += math::dot(span, math::subtract(tangent_end, tangent_start)) / norm;
    }
    tangent_path_length = tangent_length;
}

} // namespace rayd::shared::reflection

#undef RAYD_SHARED_EPC_INLINE

#include <cmath>
#include <cstdint>

#include <vector_types.h> // float4 for the optional packed-triangle inputs.

#include <rayd/math.h>
#include <src/reflection/reflection_internal.h>

#include <rayd/math.h>
#include <src/runtime/rt_device.cuh>

// Host-compilable reflection-trace algorithm. This is the de-CUDA-ised body of
// the former reflection_trace_raygen: math is math::Vec3f throughout (mirroring
// the exact arithmetic op order of the old local CUDA vector helpers so device
// codegen stays bit-identical), every ray cast goes through an rt::is_traverser
// Traverser (so no OptiX ray-cast intrinsic, payload register, or launch-index
// query appears here), and the lane index is a plain parameter.
// reflection_trace_device.cuh instantiates it with
// TraceConfig<ReflectionTracePolicy, OptixTraverser>; the CUDA fused executor
// (P4d) will reuse it with CudaBvhTraverser.
//
// The P0 numeric-policy locks stay attached to the constants below.

namespace rayd::shared::multipath {

namespace reflection_trace_algo_detail {

using math::Vec3f;

constexpr float kTraceTMin = 1e-5f;
constexpr float kTraceTMax = 1e8f;
constexpr float kRayBias = 1e-5f;

static_assert(kTraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(kTraceTMax == ::rayd::shared::rt::kTraceTMaxFinite);
static_assert(kRayBias == ::rayd::shared::rt::kMultipathRayBias);
// This family clears missed slots to kTraceTMax rather than +inf.
static_assert(kTraceTMax == ::rayd::shared::rt::kReflectionTraceMissDistance);

using ::rayd::shared::optix::ReflectionTraceParams;

struct TriangleData {
    Vec3f p0;
    Vec3f e1;
    Vec3f e2;
    Vec3f fn;
};

RAYD_HOST_DEVICE float reciprocal_sqrt(float value) {
#if defined(__CUDA_ARCH__)
    return rsqrtf(value);
#else
    return 1.0f / std::sqrt(value);
#endif
}

RAYD_HOST_DEVICE bool is_finite(float value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

RAYD_HOST_DEVICE Vec3f load_vec3(const float* values) {
    return math::make_vec3(values[0], values[1], values[2]);
}

RAYD_HOST_DEVICE Vec3f load_vec3(float4 value) {
    return math::make_vec3(value.x, value.y, value.z);
}

RAYD_HOST_DEVICE Vec3f normalize(Vec3f value) {
    return math::scale(value, reciprocal_sqrt(fmaxf(math::dot(value, value), 1e-12f)));
}

RAYD_HOST_DEVICE ::rayd::shared::rt::TriangleHit choose_nearest_hit(::rayd::shared::rt::TriangleHit a,
                                                                    ::rayd::shared::rt::TriangleHit b) {
    if (a.hit == 0u)
        return b;
    if (b.hit == 0u)
        return a;
    return b.t < a.t ? b : a;
}

template <typename Policy>
RAYD_HOST_DEVICE int output_slot(const ReflectionTraceParams& params, unsigned int ray_index, int bounce) {
    if constexpr (Policy::honor_output_layout) {
        if (params.output_layout != 0)
            return bounce * params.n_rays + static_cast<int>(ray_index);
    }
    return static_cast<int>(ray_index) * params.max_bounces + bounce;
}

template <typename Policy>
RAYD_HOST_DEVICE void clear_output_slot(const ReflectionTraceParams& params, unsigned int ray_index, int bounce) {
    if constexpr (!Policy::clear_empty_slots) {
        return;
    } else {
        const int slot = output_slot<Policy>(params, ray_index, bounce);
        if (params.out_t != nullptr)
            params.out_t[slot] = kTraceTMax;
        if (params.out_shape_ids != nullptr)
            params.out_shape_ids[slot] = -1;
        if (params.out_prim_ids != nullptr)
            params.out_prim_ids[slot] = -1;
        if (params.out_global_prim_ids != nullptr)
            params.out_global_prim_ids[slot] = -1;
        if (params.out_valid != nullptr)
            params.out_valid[ray_index * params.max_bounces + bounce] = 0u;
        if (params.out_bary != nullptr) {
            params.out_bary[slot * 3 + 0] = 0.0f;
            params.out_bary[slot * 3 + 1] = 0.0f;
            params.out_bary[slot * 3 + 2] = 0.0f;
        }
        if (params.out_hit != nullptr) {
            params.out_hit[slot * 3 + 0] = 0.0f;
            params.out_hit[slot * 3 + 1] = 0.0f;
            params.out_hit[slot * 3 + 2] = 0.0f;
        }
        if (params.out_norm != nullptr) {
            params.out_norm[slot * 3 + 0] = 0.0f;
            params.out_norm[slot * 3 + 1] = 0.0f;
            params.out_norm[slot * 3 + 2] = 0.0f;
        }
        if (params.out_img != nullptr) {
            params.out_img[slot * 3 + 0] = 0.0f;
            params.out_img[slot * 3 + 1] = 0.0f;
            params.out_img[slot * 3 + 2] = 0.0f;
        }
    }
}

template <typename Policy> RAYD_HOST_DEVICE TriangleData load_triangle(const ReflectionTraceParams& params, int prim) {
    if constexpr (Policy::allow_packed_triangles) {
        if (params.tri_p0_packed != nullptr && params.tri_e1_packed != nullptr && params.tri_e2_packed != nullptr &&
            params.tri_fn_packed != nullptr) {
            return {load_vec3(params.tri_p0_packed[prim]), load_vec3(params.tri_e1_packed[prim]),
                    load_vec3(params.tri_e2_packed[prim]), load_vec3(params.tri_fn_packed[prim])};
        }
    }

    return {math::make_vec3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]),
            math::make_vec3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]),
            math::make_vec3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]),
            math::make_vec3(params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim])};
}

template <typename Policy>
RAYD_HOST_DEVICE void load_ray(const ReflectionTraceParams& params, unsigned int ray_index, Vec3f& origin,
                               Vec3f& direction) {
    if constexpr (Policy::allow_aos_inputs) {
        if (params.ray_o_aos != nullptr) {
            origin = load_vec3(params.ray_o_aos + ray_index * 3);
            direction = load_vec3(params.ray_d_aos + ray_index * 3);
            return;
        }
    }

    origin = math::make_vec3(params.ray_ox[ray_index], params.ray_oy[ray_index], params.ray_oz[ray_index]);
    direction = math::make_vec3(params.ray_dx[ray_index], params.ray_dy[ray_index], params.ray_dz[ray_index]);
}

template <typename Policy>
RAYD_HOST_DEVICE float first_trace_tmax(const ReflectionTraceParams& params, unsigned int ray_index) {
    if constexpr (Policy::nullable_ray_tmax) {
        return params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : kTraceTMax;
    }
    return params.ray_tmax[ray_index];
}

} // namespace reflection_trace_algo_detail

/// Reflection-path trace for one lane. `primary` / `secondary` are Config::
/// Traverser oracles over the two acceleration structures (secondary consulted
/// only when params.split_mode != 0), and `ray_index` is this lane's ray id.
template <typename Config>
RAYD_DEVICE void reflection_trace_algo(const ::rayd::shared::optix::ReflectionTraceParams& params,
                                       std::uint32_t ray_index, const typename Config::Traverser& primary,
                                       const typename Config::Traverser& secondary) {
    using namespace reflection_trace_algo_detail;
    using Policy = typename Config::Layout;
    using ::rayd::shared::rt::TriangleHit;
    namespace reflection = ::rayd::shared::reflection;

    if (ray_index >= static_cast<unsigned int>(params.n_rays))
        return;

    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        if constexpr (Policy::clear_empty_slots) {
            for (int bounce = 0; bounce < params.max_bounces; ++bounce)
                clear_output_slot<Policy>(params, ray_index, bounce);
        }
        if (params.out_bounce_count != nullptr)
            params.out_bounce_count[ray_index] = 0;
        return;
    }

    const int bounce_limit = params.max_bounces;
    Vec3f origin;
    Vec3f direction;
    load_ray<Policy>(params, ray_index, origin, direction);
    Vec3f image_source = origin;
    int bounce_count = 0;

    for (int bounce = 0; bounce < bounce_limit; ++bounce) {
        const float tmax_input = bounce == 0 ? first_trace_tmax<Policy>(params, ray_index) : kTraceTMax;
        const float trace_tmax = is_finite(tmax_input) ? tmax_input : kTraceTMax;

        const TriangleHit primary_hit = primary.trace_closest(origin, direction, kTraceTMin, trace_tmax);
        TriangleHit hit = primary_hit;
        if (params.split_mode != 0) {
            const TriangleHit secondary_hit = secondary.trace_closest(origin, direction, kTraceTMin, trace_tmax);
            hit = choose_nearest_hit(primary_hit, secondary_hit);
        }
        if (hit.hit == 0u)
            break;

        const int shape_id = hit.instance;
        const int local_prim = hit.prim;
        const int face_offset = shape_id >= 0 && shape_id < params.n_meshes ? params.face_offsets[shape_id] : 0;
        const int global_prim = face_offset + local_prim;
        const float t = hit.t;
        const float bary_u = hit.bary_u;
        const float bary_v = hit.bary_v;

        Vec3f hit_point = math::add(origin, math::scale(direction, t));
        Vec3f geo_normal = math::make_vec3(0.0f, 0.0f, 1.0f);
        if (global_prim >= 0 && global_prim < params.n_triangles) {
            const TriangleData tri = load_triangle<Policy>(params, global_prim);
            hit_point = math::add(math::add(tri.p0, math::scale(tri.e1, bary_u)), math::scale(tri.e2, bary_v));
            geo_normal = normalize(tri.fn);
        }
        geo_normal = reflection::orient_normal_against(direction, geo_normal);

        const bool write_image_source =
            (Policy::allow_extended_outputs && params.out_img != nullptr) ||
            (params.out_img_x != nullptr && params.out_img_y != nullptr && params.out_img_z != nullptr);
        if (write_image_source) {
            image_source = reflection::reflect_point_across_plane(image_source, hit_point, geo_normal);
        }

        const int slot = output_slot<Policy>(params, ray_index, bounce);
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_valid != nullptr)
                params.out_valid[ray_index * params.max_bounces + bounce] = 1u;
        }
        if (params.out_shape_ids != nullptr)
            params.out_shape_ids[slot] = shape_id;
        if (params.out_prim_ids != nullptr)
            params.out_prim_ids[slot] = local_prim;
        if (params.out_global_prim_ids != nullptr)
            params.out_global_prim_ids[slot] = global_prim;
        if (params.out_t != nullptr)
            params.out_t[slot] = t;
        if (params.out_bary_u != nullptr)
            params.out_bary_u[slot] = bary_u;
        if (params.out_bary_v != nullptr)
            params.out_bary_v[slot] = bary_v;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_bary != nullptr) {
                params.out_bary[slot * 3 + 0] = 1.0f - bary_u - bary_v;
                params.out_bary[slot * 3 + 1] = bary_u;
                params.out_bary[slot * 3 + 2] = bary_v;
            }
        }
        if (params.out_hit_x != nullptr)
            params.out_hit_x[slot] = hit_point.x;
        if (params.out_hit_y != nullptr)
            params.out_hit_y[slot] = hit_point.y;
        if (params.out_hit_z != nullptr)
            params.out_hit_z[slot] = hit_point.z;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_hit != nullptr) {
                params.out_hit[slot * 3 + 0] = hit_point.x;
                params.out_hit[slot * 3 + 1] = hit_point.y;
                params.out_hit[slot * 3 + 2] = hit_point.z;
            }
        }
        if (params.out_norm_x != nullptr)
            params.out_norm_x[slot] = geo_normal.x;
        if (params.out_norm_y != nullptr)
            params.out_norm_y[slot] = geo_normal.y;
        if (params.out_norm_z != nullptr)
            params.out_norm_z[slot] = geo_normal.z;
        if constexpr (Policy::allow_extended_outputs) {
            if (params.out_norm != nullptr) {
                params.out_norm[slot * 3 + 0] = geo_normal.x;
                params.out_norm[slot * 3 + 1] = geo_normal.y;
                params.out_norm[slot * 3 + 2] = geo_normal.z;
            }
        }
        if (write_image_source) {
            if constexpr (Policy::allow_extended_outputs) {
                if (params.out_img != nullptr) {
                    params.out_img[slot * 3 + 0] = image_source.x;
                    params.out_img[slot * 3 + 1] = image_source.y;
                    params.out_img[slot * 3 + 2] = image_source.z;
                } else {
                    params.out_img_x[slot] = image_source.x;
                    params.out_img_y[slot] = image_source.y;
                    params.out_img_z[slot] = image_source.z;
                }
            } else {
                params.out_img_x[slot] = image_source.x;
                params.out_img_y[slot] = image_source.y;
                params.out_img_z[slot] = image_source.z;
            }
        }

        direction = reflection::reflect_direction(direction, geo_normal);
        origin = math::add(hit_point, math::scale(direction, kRayBias));
        bounce_count = bounce + 1;
    }

    if constexpr (Policy::clear_empty_slots) {
        for (int bounce = bounce_count; bounce < bounce_limit; ++bounce)
            clear_output_slot<Policy>(params, ray_index, bounce);
    }

    if (bounce_count > 0 && params.return_trailing != 0) {
        if (params.out_trailing_dir_x != nullptr)
            params.out_trailing_dir_x[ray_index] = direction.x;
        if (params.out_trailing_dir_y != nullptr)
            params.out_trailing_dir_y[ray_index] = direction.y;
        if (params.out_trailing_dir_z != nullptr)
            params.out_trailing_dir_z[ray_index] = direction.z;
        if (params.out_trailing_origin_x != nullptr)
            params.out_trailing_origin_x[ray_index] = origin.x;
        if (params.out_trailing_origin_y != nullptr)
            params.out_trailing_origin_y[ray_index] = origin.y;
        if (params.out_trailing_origin_z != nullptr)
            params.out_trailing_origin_z[ray_index] = origin.z;

        const TriangleHit primary_hit = primary.trace_closest(origin, direction, kTraceTMin, kTraceTMax);
        TriangleHit trailing = primary_hit;
        if (params.split_mode != 0) {
            const TriangleHit secondary_hit = secondary.trace_closest(origin, direction, kTraceTMin, kTraceTMax);
            trailing = choose_nearest_hit(primary_hit, secondary_hit);
        }
        if (trailing.hit != 0u) {
            const int shape_id = trailing.instance;
            const int local_prim = trailing.prim;
            const int face_offset = shape_id >= 0 && shape_id < params.n_meshes ? params.face_offsets[shape_id] : 0;
            if (params.out_trailing_t != nullptr)
                params.out_trailing_t[ray_index] = trailing.t;
            if (params.out_trailing_prim != nullptr)
                params.out_trailing_prim[ray_index] = face_offset + local_prim;
        }
    }

    if (params.out_bounce_count != nullptr)
        params.out_bounce_count[ray_index] = bounce_count;
}

} // namespace rayd::shared::multipath

#include <cmath>
#include <cstdint>

#include <rayd/contracts.h>
#include <rayd/math.h>
#include <src/runtime/rt_device.cuh>

// Host-compilable reflection-accumulation algorithm. This is the de-CUDA-ised
// body of the former reflection_accumulation::raygen: math is math::Vec3f
// throughout (mirroring the exact arithmetic op order of the old local CUDA
// vector helpers so device codegen stays bit-identical), the closest-hit ray
// cast goes through an rt::is_traverser Traverser (so no OptiX ray-cast
// intrinsic, payload register, or launch-index query appears here), and the lane
// index is a plain parameter. The local 6-field HitPayload dissolves into
// rt::TriangleHit. accumulation_optix_device.cuh instantiates it with the
// shared OptixTraverser; the CUDA fused executor (P4d) will reuse it with
// CudaBvhTraverser. The wedge-event slot reservation and the grid commit remain
// the caller's Policy responsibility (device atomics); the host atomic_add
// fallback below only exists so this header parses under a pure host compiler.
//
// The P0 numeric-policy locks stay attached to the constants below.

namespace rayd::shared::multipath {

namespace reflection_accumulation_algo_detail {

using math::Vec3f;
namespace field = ::rayd::shared::field;
using field::Complex;
using field::Complex3;

inline constexpr float TraceTMin = 1.0e-5f;
inline constexpr float TraceTMax = 1.0e8f;
inline constexpr float RayBias = 1.0e-5f;
inline constexpr float Epsilon = shared::SmallEpsilon;
inline constexpr float SpeedOfLight = shared::SpeedOfLight;
inline constexpr float Pi = 3.14159265358979323846f;

static_assert(TraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(TraceTMax == ::rayd::shared::rt::kTraceTMaxFinite);
static_assert(RayBias == ::rayd::shared::rt::kMultipathRayBias);
// This family clears missed hits to TraceTMax rather than +inf.
static_assert(TraceTMax == ::rayd::shared::rt::kReflectionTraceMissDistance);

RAYD_HOST_DEVICE float reciprocal_sqrt(float value) {
#if defined(__CUDA_ARCH__)
    return rsqrtf(value);
#else
    return 1.0f / std::sqrt(value);
#endif
}

RAYD_HOST_DEVICE bool is_finite(float value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// Integer min/max, the host-safe form of the device min()/max() builtins.
RAYD_HOST_DEVICE int imax(int a, int b) {
    return a > b ? a : b;
}
RAYD_HOST_DEVICE int imin(int a, int b) {
    return a < b ? a : b;
}

// atomicAdd on device; a non-atomic byte-equivalent on the host so the wedge
// slot reservation compiles off-device (the host path is never executed).
RAYD_HOST_DEVICE int atomic_add(int* address, int value) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(address, value);
#else
    const int old = *address;
    *address += value;
    return old;
#endif
}

RAYD_HOST_DEVICE Vec3f fallback_axis(Vec3f direction) {
    return fabsf(direction.z) < 0.9f ? math::make_vec3(0.0f, 0.0f, 1.0f) : math::make_vec3(0.0f, 1.0f, 0.0f);
}

RAYD_HOST_DEVICE Vec3f stable_perpendicular(Vec3f direction, Vec3f preferred) {
    const Vec3f normalized_direction = math::normalize_f32(direction);
    Vec3f projected =
        math::subtract(preferred, math::scale(normalized_direction, math::dot(preferred, normalized_direction)));
    if (math::dot(projected, projected) > 1.0e-12f)
        return math::normalize_f32(projected);
    const Vec3f axis = fallback_axis(normalized_direction);
    projected = math::subtract(axis, math::scale(normalized_direction, math::dot(axis, normalized_direction)));
    return math::normalize_f32(projected);
}

RAYD_HOST_DEVICE float max_abs_component(Vec3f value) {
    return fmaxf(fabsf(value.x), fmaxf(fabsf(value.y), fabsf(value.z)));
}

RAYD_HOST_DEVICE Vec3f offset_surface_point(Vec3f point, Vec3f direction, Vec3f normal) {
    const float offset = RayBias * (1.0f + max_abs_component(point));
    const float signed_offset = math::dot(direction, normal) >= 0.0f ? offset : -offset;
    return math::add(point, math::scale(normal, signed_offset));
}

RAYD_HOST_DEVICE ::rayd::shared::rt::TriangleHit choose_hit(::rayd::shared::rt::TriangleHit primary,
                                                            ::rayd::shared::rt::TriangleHit secondary) {
    if (primary.hit == 0u)
        return secondary;
    if (secondary.hit == 0u)
        return primary;
    return secondary.t < primary.t ? secondary : primary;
}

template <typename Traverser>
RAYD_DEVICE ::rayd::shared::rt::TriangleHit trace_scene(int split_mode, const Traverser& primary,
                                                        const Traverser& secondary, Vec3f origin, Vec3f direction,
                                                        float tmax) {
    const ::rayd::shared::rt::TriangleHit primary_hit = primary.trace_closest(origin, direction, TraceTMin, tmax);
    if (split_mode == 0)
        return primary_hit;
    const ::rayd::shared::rt::TriangleHit secondary_hit = secondary.trace_closest(origin, direction, TraceTMin, tmax);
    return choose_hit(primary_hit, secondary_hit);
}

RAYD_HOST_DEVICE float component(Vec3f value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

RAYD_HOST_DEVICE void plane_coords(Vec3f value, int axis, float& coord0, float& coord1) {
    if (axis == 0) {
        coord0 = value.y;
        coord1 = value.z;
    } else if (axis == 1) {
        coord0 = value.x;
        coord1 = value.z;
    } else {
        coord0 = value.x;
        coord1 = value.y;
    }
}

RAYD_HOST_DEVICE Vec3f axis_plane_point(int axis, float position, float coord0, float coord1) {
    if (axis == 0)
        return math::make_vec3(position, coord0, coord1);
    if (axis == 1)
        return math::make_vec3(coord0, position, coord1);
    return math::make_vec3(coord0, coord1, position);
}

RAYD_HOST_DEVICE unsigned int hash_u32(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

RAYD_HOST_DEVICE float uniform01(unsigned int ray_index, unsigned int depth, unsigned int seed) {
    const unsigned int hash = hash_u32(ray_index ^ (depth * 0x9e3779b9u) ^ seed);
    return static_cast<float>(hash & 0x00ffffffu) * (1.0f / 16777216.0f);
}

template <typename Params>
RAYD_HOST_DEVICE bool material_reflection_coefficients(const Params& params, int global_primitive, float cos_theta,
                                                       Complex& r_te, Complex& r_tm) {
    r_te = field::c_make(0.0f, 0.0f);
    r_tm = field::c_make(0.0f, 0.0f);
    if (global_primitive < 0 || global_primitive >= params.material_count || params.material_valid == nullptr ||
        params.material_valid[global_primitive] == 0u)
        return false;
    const float omega = fmaxf(2.0f * Pi * SpeedOfLight / fmaxf(params.wavelength, Epsilon), Epsilon);
    return field::fresnel_reflection_coefficients(params.material_eta_r[global_primitive],
                                                  params.material_sigma[global_primitive],
                                                  params.material_mu_r[global_primitive],
                                                  params.material_gain[global_primitive], omega, cos_theta, r_te, r_tm,
                                                  Epsilon);
}

template <typename Params>
RAYD_HOST_DEVICE Complex3 reflect_field_vector(const Params& params, Complex3 field_value, Vec3f incident_direction,
                                               Vec3f normal, int global_primitive, Vec3f& reflected_direction) {
    const Vec3f incident_hat = math::normalize_f32(incident_direction);
    const Vec3f normal_hat = math::normalize_f32(normal);
    const float direction_dot_normal = math::dot(incident_hat, normal_hat);
    reflected_direction =
        math::normalize_f32(math::subtract(incident_hat, math::scale(normal_hat, 2.0f * direction_dot_normal)));

    Vec3f s_hat = math::cross(normal_hat, incident_hat);
    s_hat = math::dot(s_hat, s_hat) <= 1.0e-12f ? stable_perpendicular(incident_hat, normal_hat)
                                                : math::normalize_f32(s_hat);
    Vec3f p_in_hat = math::cross(s_hat, incident_hat);
    p_in_hat = math::dot(p_in_hat, p_in_hat) <= 1.0e-12f ? stable_perpendicular(incident_hat, normal_hat)
                                                         : math::normalize_f32(p_in_hat);
    Vec3f p_out_hat = math::cross(s_hat, reflected_direction);
    p_out_hat = math::dot(p_out_hat, p_out_hat) <= 1.0e-12f ? stable_perpendicular(reflected_direction, normal_hat)
                                                            : math::normalize_f32(p_out_hat);

    Complex r_te;
    Complex r_tm;
    if (!material_reflection_coefficients(params, global_primitive, fabsf(direction_dot_normal), r_te, r_tm))
        return field::c3_zero();
    const Complex e_s = field::c3_dot_real(field_value, s_hat);
    const Complex e_p = field::c3_dot_real(field_value, p_in_hat);
    return field::c3_add(field::c3_scale_complex(s_hat, field::c_mul(r_te, e_s)),
                         field::c3_scale_complex(p_out_hat, field::c_mul(r_tm, e_p)));
}

template <typename Params>
RAYD_HOST_DEVICE void store_wedge_event(const Params& params, unsigned int ray_index, int depth, int global_primitive,
                                        Vec3f hit_point, Vec3f normal, Vec3f incident_direction, Vec3f source_point,
                                        float source_power, Vec3f initial_direction) {
    if (params.collect_wedges == 0 || params.out_wedge_count == nullptr)
        return;
    if (depth > 0 && params.collect_wedge_prefixes == 0)
        return;
    float stored_source_power = source_power;
    const int sample_stride = imax(params.wedge_sample_stride, 1);
    if (params.collect_wedge_prefixes != 0 && sample_stride > 1) {
        const unsigned int max_prefix_depth = static_cast<unsigned int>(imax(params.max_bounces, 1));
        const unsigned int ordinal = ray_index * max_prefix_depth + static_cast<unsigned int>(depth);
        const unsigned int phase = static_cast<unsigned int>(params.seed) % static_cast<unsigned int>(sample_stride);
        if ((ordinal + phase) % static_cast<unsigned int>(sample_stride) != 0u)
            return;
        stored_source_power *= static_cast<float>(sample_stride);
    }

    const int slot = atomic_add(params.out_wedge_count, 1);
    if (slot < 0 || slot >= params.wedge_capacity)
        return;
    params.out_wedge_ray_index[slot] = static_cast<int>(ray_index);
    params.out_wedge_hit_x[slot] = hit_point.x;
    params.out_wedge_hit_y[slot] = hit_point.y;
    params.out_wedge_hit_z[slot] = hit_point.z;
    params.out_wedge_normal_x[slot] = normal.x;
    params.out_wedge_normal_y[slot] = normal.y;
    params.out_wedge_normal_z[slot] = normal.z;
    params.out_wedge_prim_id[slot] = global_primitive;
    params.out_wedge_dir_x[slot] = incident_direction.x;
    params.out_wedge_dir_y[slot] = incident_direction.y;
    params.out_wedge_dir_z[slot] = incident_direction.z;
    params.out_wedge_source_x[slot] = source_point.x;
    params.out_wedge_source_y[slot] = source_point.y;
    params.out_wedge_source_z[slot] = source_point.z;
    params.out_wedge_source_power[slot] = stored_source_power;
    params.out_wedge_initial_dir_x[slot] = initial_direction.x;
    params.out_wedge_initial_dir_y[slot] = initial_direction.y;
    params.out_wedge_initial_dir_z[slot] = initial_direction.z;
    params.out_wedge_bounce_depth[slot] = depth;
}

template <typename Params, typename Policy>
RAYD_HOST_DEVICE bool accumulate_plane(const Params& params, unsigned int ray_index, int depth, Vec3f origin,
                                       Vec3f direction, float blocker_t, Vec3f image_source, Complex3 field_value) {
    if (!Policy::include_depth(params, depth) || field::c3_power(field_value) <= 0.0f)
        return false;
    const int axis = params.grid_axis;
    const float axis_direction = component(direction, axis);
    if (fabsf(axis_direction) <= Epsilon)
        return false;
    const float safe_axis_direction = axis_direction + (axis_direction >= 0.0f ? Epsilon : -Epsilon);
    const float t_plane = (params.grid_position - component(origin, axis)) / safe_axis_direction;
    if (!(t_plane > RayBias && t_plane < blocker_t))
        return false;

    const Vec3f target = math::add(origin, math::scale(direction, t_plane));
    float coord0 = 0.0f;
    float coord1 = 0.0f;
    plane_coords(target, axis, coord0, coord1);
    if (coord0 < params.grid_coord0_min || coord0 >= params.grid_coord0_max || coord1 < params.grid_coord1_min ||
        coord1 >= params.grid_coord1_max)
        return false;
    const float span0 = params.grid_coord0_max - params.grid_coord0_min;
    const float span1 = params.grid_coord1_max - params.grid_coord1_min;
    if (span0 <= 0.0f || span1 <= 0.0f || params.grid_resolution0 <= 0 || params.grid_resolution1 <= 0)
        return false;

    const float u = (coord0 - params.grid_coord0_min) / span0;
    const float v = (coord1 - params.grid_coord1_min) / span1;
    const int ix = imin(imax(static_cast<int>(u * params.grid_resolution0), 0), params.grid_resolution0 - 1);
    const int iy = imin(imax(static_cast<int>(v * params.grid_resolution1), 0), params.grid_resolution1 - 1);
    const int cell = iy * params.grid_resolution0 + ix;

    const Vec3f target_plane = axis_plane_point(axis, params.grid_position, coord0, coord1);
    const float unfolded_distance = math::length_f32(math::subtract(target_plane, image_source));
    const float fspl = field::free_space_amplitude(params.wavelength, unfolded_distance, Epsilon);
    const float cos_theta = fmaxf(fabsf(axis_direction), Epsilon);
    const float geometry_power_scale = params.solid_angle_per_ray / fmaxf(params.cell_area, Epsilon) *
                                       unfolded_distance * unfolded_distance / cos_theta;
    const float amplitude_scale = fspl * sqrtf(fmaxf(geometry_power_scale, 0.0f));
    const float wave_number = fabsf(params.k) > Epsilon ? params.k : (2.0f * Pi / fmaxf(params.wavelength, Epsilon));
    const Complex phase = field::propagation_phase(wave_number, unfolded_distance);
    const Complex coefficient = field::c_scale(phase, amplitude_scale);
    const Complex3 contribution_field = field::c3_mul_complex(field_value, coefficient);
    if (!field::finite_complex3(contribution_field))
        return false;
    const float contribution_power = field::c3_power(contribution_field);
    if (!(contribution_power > 0.0f) || !is_finite(contribution_power))
        return false;
    Policy::commit(params, ray_index, depth, cell, contribution_field, contribution_power);
    return true;
}

} // namespace reflection_accumulation_algo_detail

/// Reflection-field accumulation for one lane (former reflection_accumulation::
/// raygen). `primary` / `secondary` are the Traverser oracles over the two
/// acceleration structures (secondary consulted only when params.split_mode !=
/// 0), and `ray_index` is this lane's ray id. `Policy` supplies the compile-time
/// include-depth predicate and the grid commit (device atomics).
template <typename Params, typename Policy, typename Traverser>
RAYD_DEVICE void reflection_accumulation_algo(const Params& params, std::uint32_t ray_index, const Traverser& primary,
                                              const Traverser& secondary) {
    using namespace reflection_accumulation_algo_detail;
    using field::Complex3;
    using math::Vec3f;
    using ::rayd::shared::rt::TriangleHit;

    if (ray_index >= static_cast<unsigned int>(params.n_rays))
        return;
    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u)
        return;

    Vec3f origin = math::make_vec3(params.ray_ox[ray_index], params.ray_oy[ray_index], params.ray_oz[ray_index]);
    Vec3f direction = math::normalize_f32(
        math::make_vec3(params.ray_dx[ray_index], params.ray_dy[ray_index], params.ray_dz[ray_index]));
    const Vec3f initial_direction = direction;
    Vec3f image_source = math::make_vec3(params.tx_x[ray_index], params.tx_y[ray_index], params.tx_z[ray_index]);
    const Vec3f tx_polarization =
        math::make_vec3(params.tx_pol_x[ray_index], params.tx_pol_y[ray_index], params.tx_pol_z[ray_index]);
    Vec3f transverse_polarization =
        math::subtract(tx_polarization, math::scale(direction, math::dot(tx_polarization, direction)));
    transverse_polarization = math::dot(transverse_polarization, transverse_polarization) <= 1.0e-12f
                                  ? stable_perpendicular(direction, tx_polarization)
                                  : math::normalize_f32(transverse_polarization);
    Complex3 field_value = field::c3_from_real(transverse_polarization);
    float path_length = 0.0f;

    for (int depth = 0; depth <= params.max_bounces; ++depth) {
        const float tmax_input = depth == 0 && params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : TraceTMax;
        const float trace_tmax = is_finite(tmax_input) ? tmax_input : TraceTMax;
        const TriangleHit hit = trace_scene(params.split_mode, primary, secondary, origin, direction, trace_tmax);
        const float blocker_t = hit.hit != 0u ? hit.t : TraceTMax;

        accumulate_plane<Params, Policy>(params, ray_index, depth, origin, direction, blocker_t, image_source,
                                         field_value);
        if (hit.hit == 0u || depth >= params.max_bounces)
            break;

        const int shape_id = static_cast<int>(hit.instance);
        const int local_primitive = static_cast<int>(hit.prim);
        const int face_offset = shape_id >= 0 && shape_id < params.n_meshes ? params.face_offsets[shape_id] : 0;
        const int global_primitive = face_offset + local_primitive;
        const float bary_u = hit.bary_u;
        const float bary_v = hit.bary_v;

        Vec3f hit_point = math::add(origin, math::scale(direction, blocker_t));
        Vec3f geometric_normal = math::make_vec3(0.0f, 0.0f, 1.0f);
        if (global_primitive >= 0 && global_primitive < params.n_triangles) {
            hit_point = math::make_vec3(params.tri_p0_x[global_primitive] + bary_u * params.tri_e1_x[global_primitive] +
                                            bary_v * params.tri_e2_x[global_primitive],
                                        params.tri_p0_y[global_primitive] + bary_u * params.tri_e1_y[global_primitive] +
                                            bary_v * params.tri_e2_y[global_primitive],
                                        params.tri_p0_z[global_primitive] + bary_u * params.tri_e1_z[global_primitive] +
                                            bary_v * params.tri_e2_z[global_primitive]);
            geometric_normal = math::normalize_f32(math::make_vec3(params.tri_fn_x[global_primitive],
                                                                   params.tri_fn_y[global_primitive],
                                                                   params.tri_fn_z[global_primitive]));
        }
        if (math::dot(direction, geometric_normal) > 0.0f)
            geometric_normal = math::scale(geometric_normal, -1.0f);

        Vec3f reflected_direction;
        const float source_power = field::c3_power(field_value) * params.solid_angle_per_ray;
        const Complex3 reflected_field = reflect_field_vector(params, field_value, direction, geometric_normal,
                                                              global_primitive, reflected_direction);
        if (field::c3_power(reflected_field) <= 0.0f)
            break;

        store_wedge_event(params, ray_index, depth, global_primitive, hit_point, geometric_normal, direction,
                          image_source, source_power, initial_direction);

        const float image_distance = math::dot(math::subtract(image_source, hit_point), geometric_normal);
        image_source = math::subtract(image_source, math::scale(geometric_normal, 2.0f * image_distance));
        path_length += blocker_t;
        field_value = reflected_field;
        direction = reflected_direction;
        origin = offset_surface_point(hit_point, direction, geometric_normal);

        const int next_depth = depth + 1;
        if (params.rr_depth > 0 && params.rr_prob < 1.0f && next_depth >= params.rr_depth) {
            const float field_power = field::c3_power(field_value);
            const float continue_probability = fminf(fmaxf(field_power, 1.0e-8f), fmaxf(params.rr_prob, 1.0e-8f));
            if (uniform01(ray_index, static_cast<unsigned int>(next_depth), static_cast<unsigned int>(params.seed)) >=
                continue_probability)
                break;
            const float roulette_scale = reciprocal_sqrt(fmaxf(continue_probability, 1.0e-8f));
            field_value.x = field::c_scale(field_value.x, roulette_scale);
            field_value.y = field::c_scale(field_value.y, roulette_scale);
            field_value.z = field::c_scale(field_value.z, roulette_scale);
        }

        if (params.stop_threshold > 0.0f) {
            const float fspl = field::free_space_amplitude(params.wavelength, path_length, Epsilon);
            if (field::c3_power(field_value) * fspl * fspl <= params.stop_threshold)
                break;
        }
    }
}

} // namespace rayd::shared::multipath

#include <cmath>
#include <cstdint>
#include <cstring>

#include <rayd/contracts.h>
#include <rayd/math.h>
#include <src/reflection/reflection_internal.h>

#include <rayd/math.h>
#include <src/runtime/rt_device.cuh>

// Host-compilable reflection-EPC discovery algorithm. This is the de-CUDA-ised
// body of the former run_reflection_epc_raygen: math is math::Vec3f throughout
// (mirroring the exact arithmetic op order of the old local CUDA vector helpers so
// device codegen stays bit-identical), the two ray-cast families (reflector scene
// trace + segment visibility) go through an rt::is_traverser Traverser, and the
// lane index is a plain parameter. reflection_epc_device.cuh instantiates it with
// TraceConfig<ReflEpc layout policy, ReflEpcOptixTraverser>; the same Traverser
// serves both trace families (trace_closest = reflector scene, trace_first_blocker
// = visibility), switched by the OptiX payload mode inside the shim.
//
// The P0 numeric-policy locks stay attached to the constants below.

namespace rayd::shared::multipath {

namespace reflection_epc_algo_detail {

using math::Vec3f;
using ::rayd::shared::optix::ReflEpcMaxBounces;
using ::rayd::shared::optix::ReflEpcParams;
using ::rayd::shared::optix::ReflEpcVisibilityIgnoreSurfaceGroup;
namespace reflection = ::rayd::shared::reflection;

constexpr float kTraceTMin = rayd::shared::GeneralEpsilon;
constexpr float kTraceTMax = 1e8f;
constexpr float kRayBias = rayd::shared::GeneralEpsilon;
constexpr float kMinSegmentLength = 2e-5f;
constexpr float kEpcTolerance = 1e-4f;

static_assert(kTraceTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(kTraceTMax == ::rayd::shared::rt::kTraceTMaxFinite);
static_assert(kRayBias == ::rayd::shared::rt::kMultipathRayBias);
static_assert(kMinSegmentLength == ::rayd::shared::rt::kMinSegmentLength);
static_assert(kEpcTolerance == ::rayd::shared::rt::kEpcBarycentricSlack);

// Bit-cast of a uint sentinel to float. On device this is __uint_as_float; on the
// host a byte copy. 0x7f800000 is +inf, the EPC out_path_length invalid sentinel.
RAYD_HOST_DEVICE float uint_as_float(unsigned int bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

RAYD_HOST_DEVICE bool is_finite(float value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// Host-compilable mirror of device_hit.h's global_primitive_id (that header is
// device-only). With shape_id outside [0, mesh_count) the face offset is 0 and the
// primitive passes through unchanged; the OptiX visibility traverser relies on this
// by reporting the already-global blocker prim with shape -1.
RAYD_HOST_DEVICE int global_primitive_id(int shape_id, int local_primitive, const int* face_offsets, int mesh_count) {
    const int face_offset = (shape_id >= 0 && shape_id < mesh_count) ? face_offsets[shape_id] : 0;
    return face_offset + local_primitive;
}

RAYD_HOST_DEVICE ::rayd::shared::rt::TriangleHit choose_nearest_hit(::rayd::shared::rt::TriangleHit a,
                                                                    ::rayd::shared::rt::TriangleHit b) {
    if (a.hit == 0u)
        return b;
    if (b.hit == 0u)
        return a;
    return b.t < a.t ? b : a;
}

RAYD_HOST_DEVICE Vec3f load_triangle_p0(const ReflEpcParams& params, int prim) {
    return math::make_vec3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]);
}

RAYD_HOST_DEVICE Vec3f load_triangle_e1(const ReflEpcParams& params, int prim) {
    return math::make_vec3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]);
}

RAYD_HOST_DEVICE Vec3f load_triangle_e2(const ReflEpcParams& params, int prim) {
    return math::make_vec3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]);
}

RAYD_HOST_DEVICE Vec3f load_triangle_normal(const ReflEpcParams& params, int prim) {
    return reflection::epc_normalize(
        math::make_vec3(params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim]));
}

RAYD_HOST_DEVICE bool has_surface_groups(const ReflEpcParams& params) {
    return params.surface_group_id != nullptr && params.surface_group_size != nullptr &&
           params.surface_group_members != nullptr && params.surface_group_count > 0 &&
           params.surface_max_group_size > 0;
}

RAYD_HOST_DEVICE int surface_group_for_prim(const ReflEpcParams& params, int prim) {
    if (!has_surface_groups(params) || prim < 0 || prim >= params.surface_group_id_count) {
        return -1;
    }
    const int group = params.surface_group_id[prim];
    return group >= 0 && group < params.surface_group_count ? group : -1;
}

RAYD_HOST_DEVICE int expected_prim_for_bounce(const ReflEpcParams& params, int slot) {
    if (params.expected_prim_ids == nullptr || slot < 0 || slot >= params.expected_prim_count) {
        return -1;
    }
    return params.expected_prim_ids[slot];
}

RAYD_HOST_DEVICE bool direct_plane_mode(const ReflEpcParams& params) {
    return params.direct_plane_point_x != nullptr && params.direct_plane_point_y != nullptr &&
           params.direct_plane_point_z != nullptr && params.direct_plane_normal_x != nullptr &&
           params.direct_plane_normal_y != nullptr && params.direct_plane_normal_z != nullptr;
}

RAYD_HOST_DEVICE int final_ignore_group_for_ray(const ReflEpcParams& params, int ray_index) {
    if (params.final_ignore_group_ids == nullptr || params.final_ignore_group_count <= 0) {
        return -1;
    }
    const int index = params.final_ignore_group_count == 1 ? 0 : ray_index;
    if (index < 0 || index >= params.final_ignore_group_count) {
        return -1;
    }
    return params.final_ignore_group_ids[index];
}

RAYD_HOST_DEVICE bool point_inside_triangle(const ReflEpcParams& params, int prim, Vec3f point) {
    if (prim < 0 || prim >= params.n_triangles) {
        return false;
    }
    const Vec3f p0 = load_triangle_p0(params, prim);
    const Vec3f e1 = load_triangle_e1(params, prim);
    const Vec3f e2 = load_triangle_e2(params, prim);
    const Vec3f vp = math::subtract(point, p0);
    const float d00 = math::dot(e1, e1);
    const float d01 = math::dot(e1, e2);
    const float d11 = math::dot(e2, e2);
    const float d20 = math::dot(vp, e1);
    const float d21 = math::dot(vp, e2);
    const float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) <= 1e-12f) {
        return false;
    }
    const float plane_deviation = math::dot(vp, math::cross(e1, e2));
    const float scale_sq = fmaxf(fmaxf(d00, d11), 1.0f);
    const float plane_tolerance = fmaxf(params.plane_tolerance, 0.0f);
    if (plane_deviation * plane_deviation > plane_tolerance * plane_tolerance * scale_sq * denom) {
        return false;
    }
    const float inv_denom = 1.0f / denom;
    const float u = (d11 * d20 - d01 * d21) * inv_denom;
    const float v = (d00 * d21 - d01 * d20) * inv_denom;
    return u >= -kEpcTolerance && v >= -kEpcTolerance && u + v <= 1.0f + kEpcTolerance;
}

RAYD_HOST_DEVICE bool point_inside_surface_group(const ReflEpcParams& params, int group, Vec3f point,
                                                 int& resolved_prim) {
    resolved_prim = -1;
    if (!has_surface_groups(params) || group < 0 || group >= params.surface_group_count ||
        params.surface_max_group_size <= 0) {
        return false;
    }

    int member_count = params.surface_group_size[group];
    if (member_count < 0) {
        member_count = 0;
    }
    if (member_count > params.surface_max_group_size) {
        member_count = params.surface_max_group_size;
    }
    const int base = group * params.surface_max_group_size;
    for (int i = 0; i < member_count; ++i) {
        const int prim = params.surface_group_members[base + i];
        if (prim < 0) {
            continue;
        }
        if (point_inside_triangle(params, prim, point)) {
            resolved_prim = prim;
            return true;
        }
    }
    return false;
}

// Segment-plane intersection with the discovery kernel's guards. Retained from the
// device header (parallel tolerance 1e-7, segment tolerance kEpcTolerance) so
// reflection_geometry.h's intersect_segment_plane stays host-exercised; the
// discovery kernel itself does its plane solves through epc_backtrace_and_length.
RAYD_HOST_DEVICE bool intersect_line_plane(Vec3f line_start, Vec3f line_end, Vec3f plane_point, Vec3f plane_normal,
                                           Vec3f& point) {
    Vec3f shared_point = {};
    if (!reflection::intersect_segment_plane(line_start, line_end, plane_point, plane_normal, 1e-7f, kEpcTolerance,
                                             shared_point)) {
        return false;
    }
    point = shared_point;
    return is_finite(point.x) && is_finite(point.y) && is_finite(point.z);
}

RAYD_HOST_DEVICE void store_invalid(const ReflEpcParams& params, unsigned int ray_index, int bounce_count,
                                    int first_blocked_segment, int first_blocked_prim, int first_blocked_group) {
    params.out_valid[ray_index] = 0u;
    params.out_bounce_count[ray_index] = bounce_count;
    params.out_path_length[ray_index] = uint_as_float(0x7f800000u);
    params.out_first_blocked_segment[ray_index] = first_blocked_segment;
    params.out_first_blocked_prim[ray_index] = first_blocked_prim;
    params.out_first_blocked_group[ray_index] = first_blocked_group;
}

/// Reflector scene trace (former trace_scene): closest hit against the primary and,
/// when split_mode is on, the secondary acceleration structure. Uses the Traverser's
/// trace_closest, which the OptiX shim maps to the reflection-mode ray cast.
template <typename Config>
RAYD_DEVICE ::rayd::shared::rt::TriangleHit trace_scene(const ReflEpcParams& params,
                                                        const typename Config::Traverser& primary,
                                                        const typename Config::Traverser& secondary, Vec3f origin,
                                                        Vec3f direction, float tmax) {
    const ::rayd::shared::rt::TriangleHit hit_primary = primary.trace_closest(origin, direction, kTraceTMin, tmax);
    if (params.split_mode == 0) {
        return hit_primary;
    }
    const ::rayd::shared::rt::TriangleHit hit_secondary = secondary.trace_closest(origin, direction, kTraceTMin, tmax);
    return choose_nearest_hit(hit_primary, hit_secondary);
}

struct VisibilityResult {
    std::uint32_t visible;
    int blocker;
};

/// Segment visibility (former trace_visibility / trace_visibility_primary): the
/// active/degenerate guard is algorithm semantics; the occlusion cast goes through
/// the Traverser's trace_first_blocker (visibility-mode ray cast with the ignore
/// filter). With PrimaryOnly the secondary structure is never consulted; otherwise
/// an unoccluded primary and split_mode fall through to the secondary. The blocker
/// is resolved to its global prim id via global_primitive_id (a pass-through when
/// the traverser already reports a global prim with shape -1).
template <typename Config, bool PrimaryOnly>
RAYD_DEVICE VisibilityResult trace_visibility_segment(const ReflEpcParams& params,
                                                      const typename Config::Traverser& primary,
                                                      const typename Config::Traverser& secondary, Vec3f start,
                                                      Vec3f end, int ignore0, int ignore1, int ignore2) {
    Vec3f direction = math::subtract(end, start);
    const float length = math::length_f32(direction);
    if (length <= kMinSegmentLength) {
        return {1u, -1};
    }
    direction = math::scale(direction, 1.0f / length);
    const Vec3f origin = math::add(start, math::scale(direction, kRayBias));
    const float tmax = fmaxf(length - 2.0f * kRayBias, 0.0f);
    const std::int32_t ignore[3] = {ignore0, ignore1, ignore2};

    ::rayd::shared::rt::TriangleHit result =
        primary.trace_first_blocker(origin, direction, kTraceTMin, tmax, ignore, 3);
    if constexpr (!PrimaryOnly) {
        if (result.hit == 0u && params.split_mode != 0) {
            result = secondary.trace_first_blocker(origin, direction, kTraceTMin, tmax, ignore, 3);
        }
    }

    VisibilityResult out;
    out.visible = result.hit == 0u ? 1u : 0u;
    out.blocker =
        result.hit != 0u ? global_primitive_id(result.instance, result.prim, params.face_offsets, params.n_meshes) : -1;
    return out;
}

} // namespace reflection_epc_algo_detail

/// Reflection-EPC discovery for one lane (former run_reflection_epc_raygen). Traces
/// the expected reflector sequence (or applies a supplied plane sequence in direct
/// mode), runs the shared fixed-winner back-trace, freezes containment, and checks
/// segment visibility to the receiver, writing per-slot geometry and per-ray
/// validity. `primary` / `secondary` are Config::Traverser oracles over the two
/// acceleration structures.
template <typename Config, bool DirectOnly, bool PrimaryVisibilityOnly>
RAYD_DEVICE void run_reflection_epc_algo(const ::rayd::shared::optix::ReflEpcParams& params, std::uint32_t ray_index,
                                         const typename Config::Traverser& primary,
                                         const typename Config::Traverser& secondary) {
    using namespace reflection_epc_algo_detail;
    namespace reflection = ::rayd::shared::reflection;

    if (ray_index >= static_cast<unsigned int>(params.n_rays)) {
        return;
    }

    const int B = params.max_bounces;
    const int base = static_cast<int>(ray_index) * B;
    for (int bounce = 0; bounce < B; ++bounce) {
        const int slot = base + bounce;
        params.out_point_x[slot] = 0.0f;
        params.out_point_y[slot] = 0.0f;
        params.out_point_z[slot] = 0.0f;
        params.out_trace_prim_ids[slot] = -1;
        params.out_resolved_prim_ids[slot] = -1;
        params.out_surface_group_ids[slot] = -1;
        params.out_plane_normal_x[slot] = 0.0f;
        params.out_plane_normal_y[slot] = 0.0f;
        params.out_plane_normal_z[slot] = 0.0f;
    }

    if (params.active_mask != nullptr && params.active_mask[ray_index] == 0u) {
        store_invalid(params, ray_index, 0, -1, -1, -1);
        return;
    }

    Vec3f origin = math::make_vec3(params.ray_ox[ray_index], params.ray_oy[ray_index], params.ray_oz[ray_index]);
    const int rx_id = params.rx_count == 1 ? 0 : static_cast<int>(ray_index);
    const Vec3f receiver = math::make_vec3(params.rx_x[rx_id], params.rx_y[rx_id], params.rx_z[rx_id]);

    Vec3f plane_points[ReflEpcMaxBounces];
    Vec3f plane_normals[ReflEpcMaxBounces];
    int trace_prim_ids[ReflEpcMaxBounces];
    int resolved_prim_ids[ReflEpcMaxBounces];
    int surface_group_ids[ReflEpcMaxBounces];
    Vec3f image_sources[ReflEpcMaxBounces + 1];
    Vec3f reflection_points[ReflEpcMaxBounces];
    image_sources[0] = origin;

    int bounce_count = 0;
    Vec3f image_source = origin;

    if (direct_plane_mode(params)) {
        for (int bounce = 0; bounce < B; ++bounce) {
            const int slot = base + bounce;
            const int expected_prim = expected_prim_for_bounce(params, slot);
            const int expected_group = surface_group_for_prim(params, expected_prim);
            if (expected_prim < 0 || expected_prim >= params.n_triangles ||
                !is_finite(params.direct_plane_point_x[slot]) || !is_finite(params.direct_plane_point_y[slot]) ||
                !is_finite(params.direct_plane_point_z[slot]) || !is_finite(params.direct_plane_normal_x[slot]) ||
                !is_finite(params.direct_plane_normal_y[slot]) || !is_finite(params.direct_plane_normal_z[slot])) {
                store_invalid(params, ray_index, bounce_count, -1, -1, -1);
                return;
            }

            const Vec3f plane_point =
                math::make_vec3(params.direct_plane_point_x[slot], params.direct_plane_point_y[slot],
                                params.direct_plane_point_z[slot]);
            const Vec3f plane_normal = reflection::epc_normalize(math::make_vec3(params.direct_plane_normal_x[slot],
                                                                                 params.direct_plane_normal_y[slot],
                                                                                 params.direct_plane_normal_z[slot]));
            if (math::length_f32(plane_normal) <= 0.0f) {
                store_invalid(params, ray_index, bounce_count, -1, -1, -1);
                return;
            }

            image_source = reflection::reflect_point_across_plane(image_source, plane_point, plane_normal);
            image_sources[bounce + 1] = image_source;
            plane_points[bounce] = plane_point;
            plane_normals[bounce] = plane_normal;
            trace_prim_ids[bounce] = expected_prim;
            resolved_prim_ids[bounce] = -1;
            surface_group_ids[bounce] = expected_group;
            ++bounce_count;
        }
    } else {
        if constexpr (DirectOnly) {
            store_invalid(params, ray_index, bounce_count, -1, -1, -1);
            return;
        } else {
            Vec3f trace_origin = origin;
            Vec3f trace_direction = reflection::epc_normalize(
                math::make_vec3(params.ray_dx[ray_index], params.ray_dy[ray_index], params.ray_dz[ray_index]));

            for (int bounce = 0; bounce < B; ++bounce) {
                const float tmax_input =
                    bounce == 0 && params.ray_tmax != nullptr ? params.ray_tmax[ray_index] : kTraceTMax;
                const float trace_tmax = is_finite(tmax_input) ? tmax_input : kTraceTMax;
                const ::rayd::shared::rt::TriangleHit hit =
                    trace_scene<Config>(params, primary, secondary, trace_origin, trace_direction, trace_tmax);
                if (hit.hit == 0u) {
                    break;
                }

                const int shape_id = hit.instance;
                const int local_prim = hit.prim;
                const int global_prim = global_primitive_id(shape_id, local_prim, params.face_offsets, params.n_meshes);
                const int actual_group = surface_group_for_prim(params, global_prim);
                const int slot = base + bounce;
                const int expected_prim = expected_prim_for_bounce(params, slot);
                const int expected_group = surface_group_for_prim(params, expected_prim);
                const bool expected_matches =
                    expected_prim < 0 ||
                    (has_surface_groups(params) ? (actual_group >= 0 && actual_group == expected_group)
                                                : (global_prim == expected_prim));
                if (!expected_matches) {
                    store_invalid(params, ray_index, bounce_count, -1, -1, -1);
                    return;
                }
                const float bary_u = hit.bary_u;
                const float bary_v = hit.bary_v;
                const float t = hit.t;

                Vec3f hit_point = math::add(trace_origin, math::scale(trace_direction, t));
                Vec3f geo_normal = math::make_vec3(0.0f, 0.0f, 1.0f);
                if (global_prim >= 0 && global_prim < params.n_triangles) {
                    const Vec3f p0 = load_triangle_p0(params, global_prim);
                    const Vec3f e1 = load_triangle_e1(params, global_prim);
                    const Vec3f e2 = load_triangle_e2(params, global_prim);
                    hit_point = math::add(math::add(p0, math::scale(e1, bary_u)), math::scale(e2, bary_v));
                    geo_normal = load_triangle_normal(params, global_prim);
                }
                if (math::dot(trace_direction, geo_normal) > 0.0f) {
                    geo_normal = math::scale(geo_normal, -1.0f);
                }

                const float image_distance = math::dot(math::subtract(image_source, hit_point), geo_normal);
                image_source = math::subtract(image_source, math::scale(geo_normal, 2.0f * image_distance));
                image_sources[bounce + 1] = image_source;
                plane_points[bounce] = hit_point;
                plane_normals[bounce] = geo_normal;
                trace_prim_ids[bounce] = global_prim;
                resolved_prim_ids[bounce] = -1;
                surface_group_ids[bounce] = expected_group >= 0 ? expected_group : actual_group;
                ++bounce_count;

                const float ray_dot_normal = math::dot(trace_direction, geo_normal);
                trace_direction = reflection::epc_normalize(
                    math::subtract(trace_direction, math::scale(geo_normal, 2.0f * ray_dot_normal)));
                trace_origin = math::add(hit_point, math::scale(trace_direction, kRayBias));
            }
        }
    }

    if (bounce_count != B) {
        store_invalid(params, ray_index, bounce_count, -1, -1, -1);
        return;
    }

    // Shared fixed-winner back-trace and path length (reflection/epc_chain.h): the
    // planes and image sources were built per discovery mode above; the geometry
    // from here on is mode-independent and everything is already math::Vec3f, so it
    // feeds epc_backtrace_and_length with no conversion.
    float path_length = 0.0f;
    if (!reflection::epc_backtrace_and_length<ReflEpcMaxBounces>(plane_points, plane_normals, image_sources, B, origin,
                                                                 receiver, reflection_points, nullptr, nullptr,
                                                                 path_length)) {
        store_invalid(params, ray_index, bounce_count, -1, -1, -1);
        return;
    }

    // Freeze which primitive each interaction lands in (the discrete winner). The
    // back-trace above is pure geometry, containment is the discovery decision, so
    // they are separated. resolved_prim_ids feeds the visibility ignore lists below
    // and must be fully populated before them.
    for (int bounce = B - 1; bounce >= 0; --bounce) {
        const Vec3f point = reflection_points[bounce];
        int resolved_prim = -1;
        bool inside;
        if (has_surface_groups(params) && surface_group_ids[bounce] >= 0) {
            inside = point_inside_surface_group(params, surface_group_ids[bounce], point, resolved_prim);
        } else {
            const int expected_prim = expected_prim_for_bounce(params, base + bounce);
            const int containment_prim = expected_prim >= 0 ? expected_prim : trace_prim_ids[bounce];
            inside = point_inside_triangle(params, containment_prim, point);
            resolved_prim = inside ? containment_prim : -1;
        }
        resolved_prim_ids[bounce] = resolved_prim;
        if (!inside) {
            store_invalid(params, ray_index, bounce_count, -1, -1, -1);
            return;
        }
    }

    bool valid = true;
    int first_blocked_segment = -1;
    int first_blocked_prim = -1;
    int first_blocked_group = -1;
    const int final_ignore_group = final_ignore_group_for_ray(params, static_cast<int>(ray_index));
    for (int segment = 0; segment <= B; ++segment) {
        const Vec3f start = segment == 0 ? origin : reflection_points[segment - 1];
        const Vec3f end = segment == B ? receiver : reflection_points[segment];

        const bool ignore_surface_group =
            params.visibility_ignore_mode == ReflEpcVisibilityIgnoreSurfaceGroup && has_surface_groups(params);
        const int ignore0 =
            segment > 0 ? (ignore_surface_group ? surface_group_ids[segment - 1] : resolved_prim_ids[segment - 1]) : -1;
        const int ignore1 =
            segment < B ? (ignore_surface_group ? surface_group_ids[segment] : resolved_prim_ids[segment]) : -1;
        const int ignore2 = ignore_surface_group && segment == B ? final_ignore_group : -1;
        VisibilityResult visibility;
        if constexpr (PrimaryVisibilityOnly) {
            visibility = trace_visibility_segment<Config, true>(params, primary, secondary, start, end, ignore0,
                                                                ignore1, ignore2);
        } else {
            visibility = trace_visibility_segment<Config, false>(params, primary, secondary, start, end, ignore0,
                                                                 ignore1, ignore2);
        }
        if (visibility.visible == 0u) {
            first_blocked_segment = segment;
            first_blocked_prim = visibility.blocker;
            first_blocked_group = surface_group_for_prim(params, first_blocked_prim);
            valid = false;
            break;
        }
    }

    if (!valid) {
        store_invalid(params, ray_index, bounce_count, first_blocked_segment, first_blocked_prim, first_blocked_group);
        return;
    }

    params.out_valid[ray_index] = 1u;
    params.out_bounce_count[ray_index] = bounce_count;
    params.out_path_length[ray_index] = path_length;
    params.out_first_blocked_segment[ray_index] = -1;
    params.out_first_blocked_prim[ray_index] = -1;
    params.out_first_blocked_group[ray_index] = -1;
    for (int bounce = 0; bounce < B; ++bounce) {
        const int slot = base + bounce;
        params.out_point_x[slot] = reflection_points[bounce].x;
        params.out_point_y[slot] = reflection_points[bounce].y;
        params.out_point_z[slot] = reflection_points[bounce].z;
        params.out_trace_prim_ids[slot] = trace_prim_ids[bounce];
        params.out_resolved_prim_ids[slot] = resolved_prim_ids[bounce];
        params.out_surface_group_ids[slot] = surface_group_ids[bounce];
        params.out_plane_normal_x[slot] = plane_normals[bounce].x;
        params.out_plane_normal_y[slot] = plane_normals[bounce].y;
        params.out_plane_normal_z[slot] = plane_normals[bounce].z;
    }
}

} // namespace rayd::shared::multipath
