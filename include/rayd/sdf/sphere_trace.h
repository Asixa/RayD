// Copyright Xingyu Chen.
// Defines shared sdf support for sphere trace.

#pragma once

#include <cmath>

#include <rayd/math.h>
#include <rayd/rt/qualifiers.h>

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
