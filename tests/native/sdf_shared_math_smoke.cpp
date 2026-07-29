// Copyright Xingyu Chen.
// Exercises sdf shared math smoke in a native smoke test.

#include <cmath>
#include <limits>
#include <vector>

#include <src/sdf_device.cuh>

namespace {

using rayd::shared::math::make_vec3;
using rayd::shared::math::Vec3f;
using rayd::shared::sdf::BaseIndex;
using rayd::shared::sdf::GridExtent;
using rayd::shared::sdf::GridSample;
using rayd::shared::sdf::Interval;
using rayd::shared::sdf::MarchConfig;
using rayd::shared::sdf::MarchResult;
using rayd::shared::sdf::Placement;
using rayd::shared::sdf::TrilinearCell;

static_assert(rayd::shared::sdf::kSdfBisectionSteps == 32, "ADR-0037 section 7");
static_assert(rayd::shared::sdf::kSdfDefaultMaxSteps == 64, "ADR-0037 section 7");
static_assert(rayd::shared::sdf::kSdfDefaultRelaxation == 0.9f, "ADR-0037 section 7");
static_assert(rayd::shared::sdf::kSdfEpsHitVoxelFraction == 1.0e-3f, "ADR-0037 section 7");
static_assert(rayd::shared::sdf::kSdfEpsNorm == 1.0e-12f, "ADR-0037 section 7");
static_assert(rayd::shared::sdf::kSdfEpsParallel == 1.0e-7f, "ADR-0037 section 7");

bool close(float actual, float expected, float tolerance) {
    return std::fabs(actual - expected) <= tolerance;
}

// A dense sphere field baked on a vertex-centred grid, exactly as a caller would
// hand it to the operation: world-metric distances, negative inside. `strength`
// scales the values away from unit gradient so the non-eikonal march can be
// exercised without changing where the zero level set is.
struct SphereGrid {
    GridExtent extent;
    Vec3f scale;
    std::vector<float> values;

    SphereGrid(int samples, float side, float radius, float strength)
        : extent{samples, samples, samples}, scale(make_vec3(side, side, side)) {
        const float cells = static_cast<float>(samples - 1);
        values.resize(static_cast<std::size_t>(samples) * samples * samples);
        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < samples; ++j) {
                for (int k = 0; k < samples; ++k) {
                    const float x = side * (static_cast<float>(i) / cells - 0.5f);
                    const float y = side * (static_cast<float>(j) / cells - 0.5f);
                    const float z = side * (static_cast<float>(k) / cells - 0.5f);
                    const std::size_t linear = (static_cast<std::size_t>(i) * samples + j) * samples + k;
                    values[linear] = strength * (std::sqrt(x * x + y * y + z * z) - radius);
                }
            }
        }
    }
};

// The sampler the march is templated on: it walks the ray in world space, maps
// each point through the placement into the grid, and caches the frozen base
// index and index-space gradient of its last evaluation. Caching is what lets a
// caller read the frozen winner straight off the sampler after a hit.
struct GridSampler {
    const SphereGrid& grid;
    Placement placement;
    Vec3f origin;
    Vec3f direction;
    BaseIndex base{};
    Vec3f index_gradient{};
    int calls = 0;
    bool saw_sign_flip = false;
    float entry_sign = 0.0f;

    float operator()(float t) {
        const Vec3f world = rayd::shared::math::add(origin, rayd::shared::math::scale(direction, t));
        const Vec3f local = rayd::shared::sdf::world_to_local_point(placement, world);
        const Vec3f cells = rayd::shared::sdf::grid_cells(grid.extent);
        const Vec3f u = rayd::shared::sdf::grid_coord(local, grid.scale, cells);
        base = rayd::shared::sdf::base_index(u, grid.extent);
        const TrilinearCell cell = rayd::shared::sdf::trilinear_cell(u, base, grid.extent);
        const GridSample sample = rayd::shared::sdf::sample_cell(grid.values.data(), cell);
        index_gradient = sample.index_gradient;
        if (calls == 0) {
            entry_sign = sample.value >= 0.0f ? 1.0f : -1.0f;
        } else if (entry_sign * sample.value < 0.0f) {
            saw_sign_flip = true;
        }
        ++calls;
        return sample.value;
    }

    // World-space field gradient at the last evaluation (ADR-0037 section 2).
    Vec3f world_gradient() const {
        const Vec3f cells = rayd::shared::sdf::grid_cells(grid.extent);
        return rayd::shared::sdf::local_to_world_direction(placement,
                                                           rayd::shared::sdf::local_gradient(index_gradient, cells,
                                                                                             grid.scale));
    }
};

// One traced ray against a baked sphere, returning the march outcome and leaving
// the sampler available for the frozen-winner state.
MarchResult trace(GridSampler& sampler, float tmax) {
    const Vec3f local_origin = rayd::shared::sdf::world_to_local_point(sampler.placement, sampler.origin);
    const Vec3f local_direction = rayd::shared::sdf::world_to_local_direction(sampler.placement, sampler.direction);
    const Interval interval =
        rayd::shared::sdf::clip_ray_to_box(local_origin, local_direction, sampler.grid.scale, tmax);
    if (!interval.valid) {
        return MarchResult{0.0f, 0.0f, false, 0};
    }
    const Vec3f cells = rayd::shared::sdf::grid_cells(sampler.grid.extent);
    MarchConfig config{
        interval.t_lo,
        interval.t_hi,
        rayd::shared::sdf::resolve_eps_hit(-1.0f, sampler.grid.scale, cells),
        rayd::shared::sdf::kSdfDefaultRelaxation,
        rayd::shared::sdf::kSdfDefaultMaxSteps,
    };
    return rayd::shared::sdf::sphere_trace(sampler, config);
}

int check_trilinear_interpolation() {
    // A field that is exactly affine in index space: trilinear interpolation
    // reproduces it, and the analytic index gradient is its constant slope.
    const GridExtent extent{2, 3, 4};
    std::vector<float> values(2 * 3 * 4);
    for (int i = 0; i < extent.nx; ++i) {
        for (int j = 0; j < extent.ny; ++j) {
            for (int k = 0; k < extent.nz; ++k) {
                values[(static_cast<std::size_t>(i) * extent.ny + j) * extent.nz + k] =
                    1.0f + 2.0f * i + 3.0f * j + 4.0f * k;
            }
        }
    }
    const Vec3f u = make_vec3(0.5f, 1.25f, 2.75f);
    const BaseIndex base = rayd::shared::sdf::base_index(u, extent);
    if (base.i != 0 || base.j != 1 || base.k != 2) {
        return 1;
    }
    const TrilinearCell cell = rayd::shared::sdf::trilinear_cell(u, base, extent);
    float weight_sum = 0.0f;
    for (int m = 0; m < 8; ++m) {
        weight_sum += cell.weight[m];
    }
    const GridSample sample = rayd::shared::sdf::sample_cell(values.data(), cell);
    if (!close(weight_sum, 1.0f, 1.0e-6f) || !close(sample.value, 16.75f, 1.0e-5f) ||
        !close(sample.index_gradient.x, 2.0f, 1.0e-5f) || !close(sample.index_gradient.y, 3.0f, 1.0e-5f) ||
        !close(sample.index_gradient.z, 4.0f, 1.0e-5f)) {
        return 2;
    }
    // The corner weights must address the row-major neighbourhood of the base.
    const int origin = (base.i * extent.ny + base.j) * extent.nz + base.k;
    if (cell.index[0] != origin || cell.index[7] != origin + extent.ny * extent.nz + extent.nz + 1) {
        return 3;
    }
    return 0;
}

int check_grid_coordinate_clamp() {
    const GridExtent extent{5, 5, 5};
    const Vec3f scale = make_vec3(2.0f, 2.0f, 2.0f);
    const Vec3f cells = rayd::shared::sdf::grid_cells(extent);
    // A point far outside the box clamps onto the closed sampled domain, and its
    // base index stays inside the last cell.
    const Vec3f u = rayd::shared::sdf::grid_coord(make_vec3(-7.0f, 7.0f, 0.0f), scale, cells);
    if (!close(u.x, 0.0f, 0.0f) || !close(u.y, 4.0f, 0.0f) || !close(u.z, 2.0f, 0.0f)) {
        return 4;
    }
    const BaseIndex base = rayd::shared::sdf::base_index(u, extent);
    if (base.i != 0 || base.j != 3 || base.k != 2) {
        return 5;
    }
    return 0;
}

int check_placement() {
    // A quarter turn about z, then a translation: local +x must come out as
    // world +y, and the point map must be the inverse of the frame it defines.
    const float half = std::sqrt(0.5f);
    const Placement placement = rayd::shared::sdf::make_placement(half, 0.0f, 0.0f, half, make_vec3(5.0f, 0.0f, 0.0f));
    const Vec3f mapped = rayd::shared::sdf::local_to_world_direction(placement, make_vec3(1.0f, 0.0f, 0.0f));
    if (!close(mapped.x, 0.0f, 1.0e-6f) || !close(mapped.y, 1.0f, 1.0e-6f) || !close(mapped.z, 0.0f, 1.0e-6f)) {
        return 6;
    }
    const Vec3f local = rayd::shared::sdf::world_to_local_point(placement, make_vec3(5.0f, 3.0f, -1.0f));
    if (!close(local.x, 3.0f, 1.0e-6f) || !close(local.y, 0.0f, 1.0e-6f) || !close(local.z, -1.0f, 1.0e-6f)) {
        return 7;
    }
    // An unnormalized quaternion is normalized internally, not rejected.
    const Placement scaled = rayd::shared::sdf::make_placement(3.0f, 0.0f, 0.0f, 0.0f, make_vec3(0, 0, 0));
    if (!close(scaled.axis_x.x, 1.0f, 1.0e-6f) || !close(scaled.axis_y.y, 1.0f, 1.0e-6f) ||
        !close(scaled.axis_z.z, 1.0f, 1.0e-6f)) {
        return 8;
    }
    return 0;
}

int check_slab_clip() {
    const Vec3f scale = make_vec3(2.0f, 4.0f, 6.0f);
    const Interval forward =
        rayd::shared::sdf::clip_ray_to_box(make_vec3(0.0f, 0.0f, -10.0f), make_vec3(0.0f, 0.0f, 1.0f), scale,
                                           std::numeric_limits<float>::infinity());
    if (!forward.valid || !close(forward.t_lo, 7.0f, 1.0e-6f) || !close(forward.t_hi, 13.0f, 1.0e-6f)) {
        return 9;
    }
    // tmax truncates the far end; the interval never reports past it.
    const Interval bounded =
        rayd::shared::sdf::clip_ray_to_box(make_vec3(0.0f, 0.0f, -10.0f), make_vec3(0.0f, 0.0f, 1.0f), scale, 8.0f);
    if (!bounded.valid || !close(bounded.t_hi, 8.0f, 1.0e-6f)) {
        return 10;
    }
    // A ray parallel to a slab it lies outside of misses without dividing.
    const Interval parallel =
        rayd::shared::sdf::clip_ray_to_box(make_vec3(0.0f, 5.0f, -10.0f), make_vec3(0.0f, 0.0f, 1.0f), scale,
                                           std::numeric_limits<float>::infinity());
    if (parallel.valid) {
        return 11;
    }
    // A ray pointing away from the box misses: the overlap is empty.
    const Interval behind =
        rayd::shared::sdf::clip_ray_to_box(make_vec3(0.0f, 0.0f, -10.0f), make_vec3(0.0f, 0.0f, -1.0f), scale,
                                           std::numeric_limits<float>::infinity());
    if (behind.valid) {
        return 12;
    }
    // An origin inside the box starts at zero; there is no ray tmin.
    const Interval inside = rayd::shared::sdf::clip_ray_to_box(make_vec3(0.0f, 0.0f, 0.0f), make_vec3(0.0f, 0.0f, 1.0f),
                                                               scale, std::numeric_limits<float>::infinity());
    if (!inside.valid || !close(inside.t_lo, 0.0f, 0.0f) || !close(inside.t_hi, 3.0f, 1.0e-6f)) {
        return 13;
    }
    return 0;
}

int check_eps_hit_derivation() {
    const Vec3f scale = make_vec3(4.0f, 4.0f, 4.0f);
    const Vec3f cells = make_vec3(32.0f, 32.0f, 8.0f);
    // h_min is the smallest world-unit voxel edge, here on the x/y axes.
    if (!close(rayd::shared::sdf::resolve_eps_hit(-1.0f, scale, cells), 1.0e-3f * 0.125f, 0.0f)) {
        return 14;
    }
    // A positive request is used verbatim; only a non-positive one derives.
    if (!close(rayd::shared::sdf::resolve_eps_hit(0.25f, scale, cells), 0.25f, 0.0f)) {
        return 15;
    }
    return 0;
}

int check_eikonal_march() {
    // A unit sphere baked on a 33^3 grid spanning a 4-unit box. Along the axis
    // line the interpolant is exactly |x| - 1, so the hit distance is analytic.
    const SphereGrid grid(33, 4.0f, 1.0f, 1.0f);
    const Placement placement = rayd::shared::sdf::make_placement(1.0f, 0.0f, 0.0f, 0.0f, make_vec3(0.0f, 0.0f, 0.0f));
    GridSampler sampler{grid, placement, make_vec3(-3.0f, 0.0f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f)};
    const MarchResult result = trace(sampler, std::numeric_limits<float>::infinity());
    if (!result.hit || !close(result.t, 2.0f, 1.0e-3f)) {
        return 16;
    }
    // The relaxed march is conservative on an eikonal field: no sign flip, so no
    // bisection, and the budget is nowhere near exhausted.
    if (sampler.saw_sign_flip || result.steps > 16) {
        return 17;
    }
    // The frozen winner is the state of the last sampler call.
    if (!close(std::fabs(result.value), 0.0f, 1.25e-4f)) {
        return 18;
    }
    const Vec3f gradient = rayd::shared::sdf::normalize_floor(sampler.world_gradient());
    if (!close(gradient.x, -1.0f, 0.15f) || std::fabs(gradient.y) > 0.15f || std::fabs(gradient.z) > 0.15f) {
        return 19;
    }
    return 0;
}

int check_placed_march_is_invariant() {
    // The same sphere, rotated a quarter turn and moved: a rigid placement can
    // not change where a ray meets a sphere centred on the box.
    const SphereGrid grid(33, 4.0f, 1.0f, 1.0f);
    const float half = std::sqrt(0.5f);
    const Placement placement = rayd::shared::sdf::make_placement(half, 0.0f, 0.0f, half, make_vec3(5.0f, 0.0f, 0.0f));
    GridSampler sampler{grid, placement, make_vec3(5.0f, -3.0f, 0.0f), make_vec3(0.0f, 1.0f, 0.0f)};
    const MarchResult result = trace(sampler, std::numeric_limits<float>::infinity());
    if (!result.hit || !close(result.t, 2.0f, 1.0e-3f)) {
        return 20;
    }
    // The gradient is carried to world through the same placement, so it points
    // back along the ray.
    const Vec3f gradient = rayd::shared::sdf::normalize_floor(sampler.world_gradient());
    if (!close(gradient.y, -1.0f, 0.15f) || std::fabs(gradient.x) > 0.15f) {
        return 21;
    }
    return 0;
}

int check_non_eikonal_march_bisects() {
    // Doubling the values keeps the zero level set and breaks the unit-gradient
    // assumption, so the relaxed step overshoots and the sign-flip bracket must
    // recover the same hit distance.
    const SphereGrid grid(33, 4.0f, 1.0f, 2.0f);
    const Placement placement = rayd::shared::sdf::make_placement(1.0f, 0.0f, 0.0f, 0.0f, make_vec3(0.0f, 0.0f, 0.0f));
    GridSampler sampler{grid, placement, make_vec3(-3.0f, 0.0f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f)};
    const MarchResult result = trace(sampler, std::numeric_limits<float>::infinity());
    if (!result.hit || !close(result.t, 2.0f, 1.0e-3f)) {
        return 22;
    }
    if (!sampler.saw_sign_flip) {
        return 23;
    }
    return 0;
}

int check_inside_start() {
    // Starting inside the surface is supported: the entry sample fixes sigma to
    // -1 and the march moves forward on a negative field.
    const SphereGrid grid(33, 4.0f, 1.0f, 1.0f);
    const Placement placement = rayd::shared::sdf::make_placement(1.0f, 0.0f, 0.0f, 0.0f, make_vec3(0.0f, 0.0f, 0.0f));
    GridSampler sampler{grid, placement, make_vec3(0.0f, 0.0f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f)};
    const MarchResult result = trace(sampler, std::numeric_limits<float>::infinity());
    if (!result.hit || !close(result.t, 1.0f, 1.0e-3f) || sampler.saw_sign_flip) {
        return 24;
    }
    return 0;
}

int check_misses() {
    const SphereGrid grid(33, 4.0f, 1.0f, 1.0f);
    const Placement placement = rayd::shared::sdf::make_placement(1.0f, 0.0f, 0.0f, 0.0f, make_vec3(0.0f, 0.0f, 0.0f));
    // Through the box, past the sphere: the march runs out of interval.
    GridSampler past{grid, placement, make_vec3(-3.0f, 1.5f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f)};
    if (trace(past, std::numeric_limits<float>::infinity()).hit) {
        return 25;
    }
    // Pointing away from the box: no interval at all, and no field evaluation.
    GridSampler away{grid, placement, make_vec3(-3.0f, 0.0f, 0.0f), make_vec3(-1.0f, 0.0f, 0.0f)};
    const MarchResult behind = trace(away, std::numeric_limits<float>::infinity());
    if (behind.hit || behind.steps != 0 || away.calls != 0) {
        return 26;
    }
    // tmax stops the ray short of the surface.
    GridSampler clipped{grid, placement, make_vec3(-3.0f, 0.0f, 0.0f), make_vec3(1.0f, 0.0f, 0.0f)};
    const MarchResult short_ray = trace(clipped, 1.5f);
    if (short_ray.hit) {
        return 27;
    }
    return 0;
}

int check_bisection_contract() {
    // An analytic bracket: the crossing is at 0.3 and the budget is ample.
    auto linear = [](float t) { return t - 0.3f; };
    const MarchResult found = rayd::shared::sdf::bisect_bracket(linear, 0.0f, 1.0f, -1.0f, 1.0e-5f, 0);
    if (!found.hit || !close(found.t, 0.3f, 1.0e-5f) || found.steps > 32) {
        return 28;
    }
    // A step discontinuity never reaches |D| < eps_hit, and the exhausted budget
    // must still report a hit at the final midpoint (ADR-0037 section 4).
    auto step = [](float t) { return t < 0.5f ? -1.0f : 1.0f; };
    const MarchResult exhausted = rayd::shared::sdf::bisect_bracket(step, 0.0f, 1.0f, -1.0f, 1.0e-9f, 0);
    if (!exhausted.hit || !close(exhausted.t, 0.5f, 1.0e-6f) || exhausted.steps != 32) {
        return 29;
    }
    // A non-finite sample is the one way a bracket does not produce a hit.
    auto poisoned = [](float) { return std::numeric_limits<float>::quiet_NaN(); };
    if (rayd::shared::sdf::bisect_bracket(poisoned, 0.0f, 1.0f, 1.0f, 1.0e-5f, 0).hit) {
        return 30;
    }
    return 0;
}

int check_non_finite_field_is_a_miss() {
    // A NaN anywhere the march samples makes the lane a miss rather than a hit
    // with a poisoned distance (ADR-0037 section 5).
    auto poisoned = [](float) { return std::numeric_limits<float>::quiet_NaN(); };
    const MarchConfig config{0.0f, 1.0f, 1.0e-4f, 0.9f, 64};
    const MarchResult result = rayd::shared::sdf::sphere_trace(poisoned, config);
    if (result.hit || result.steps != 1) {
        return 31;
    }
    return 0;
}

} // namespace

int main() {
    using Check = int (*)();
    const Check checks[] = {
        &check_trilinear_interpolation,
        &check_grid_coordinate_clamp,
        &check_placement,
        &check_slab_clip,
        &check_eps_hit_derivation,
        &check_eikonal_march,
        &check_placed_march_is_invariant,
        &check_non_eikonal_march_bisects,
        &check_inside_start,
        &check_misses,
        &check_bisection_contract,
        &check_non_finite_field_is_a_miss,
    };
    for (const auto& check : checks) {
        const int code = check();
        if (code != 0) {
            return code;
        }
    }
    return 0;
}
