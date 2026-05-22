#include <optix.h>
#include <optix_device.h>

#include <rayd/multipath/diffraction_accumulation.h>
#include <rayd/multipath/diffraction_accumulation_params.h>

namespace rayd {

extern "C" {
extern __constant__ DiffractionAccumParams params;
}

namespace {

constexpr float kTraceTMin = 1e-5f;
constexpr float kRayBias = 1e-4f;
constexpr float kSmallEps = 1e-6f;
constexpr float kPi = 3.14159265358979323846f;

struct HitPayload {
    unsigned int hit = 0u;
    unsigned int t = 0u;
    unsigned int prim = 0u;
    unsigned int instance = 0u;
};

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

static __forceinline__ __device__ void clear_payload(HitPayload &payload) {
    payload.hit = 0u;
    payload.t = __float_as_uint(1e8f);
    payload.prim = 0u;
    payload.instance = 0u;
}

static __forceinline__ __device__ void set_payload(const HitPayload &payload) {
    optixSetPayload_0(payload.hit);
    optixSetPayload_1(payload.t);
    optixSetPayload_2(payload.prim);
    optixSetPayload_3(payload.instance);
}

static __forceinline__ __device__ void trace_handle(OptixTraversableHandle handle,
                                                    float3 origin,
                                                    float3 direction,
                                                    float tmax,
                                                    HitPayload &payload) {
    clear_payload(payload);
    if (handle == 0ull || tmax <= kTraceTMin) {
        return;
    }

    optixTrace(handle,
               origin,
               direction,
               kTraceTMin,
               tmax,
               0.0f,
               255u,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0,
               1,
               0,
               payload.hit,
               payload.t,
               payload.prim,
               payload.instance);
}

static __forceinline__ __device__ HitPayload choose_hit(HitPayload a, HitPayload b) {
    if (a.hit == 0u) {
        return b;
    }
    if (b.hit == 0u) {
        return a;
    }
    return __uint_as_float(a.t) <= __uint_as_float(b.t) ? a : b;
}

static __forceinline__ __device__ HitPayload trace_scene(float3 origin,
                                                         float3 direction,
                                                         float tmax) {
    HitPayload primary;
    trace_handle(params.primary_handle, origin, direction, tmax, primary);
    if (params.split_mode == 0) {
        return primary;
    }
    HitPayload secondary;
    trace_handle(params.secondary_handle, origin, direction, tmax, secondary);
    return choose_hit(primary, secondary);
}

static __forceinline__ __device__ bool visible_segment(float3 start, float3 end) {
    const float3 delta = end - start;
    const float dist = norm3(delta);
    if (dist <= 1e-5f) {
        return true;
    }
    const float3 dir = (1.f / dist) * delta;
    const HitPayload hit = trace_scene(start + kRayBias * dir, dir, fmaxf(dist - 2.f * kRayBias, 0.f));
    return hit.hit == 0u;
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

static __forceinline__ __device__ float3 state_vec(const float *x,
                                                   const float *y,
                                                   const float *z,
                                                   int idx) {
    return make_vec3(x[idx], y[idx], z[idx]);
}

static __forceinline__ __device__ float3 recursive_state_vec(const float *x,
                                                             const float *y,
                                                             const float *z,
                                                             int idx) {
    return make_vec3(x[idx], y[idx], z[idx]);
}

static __forceinline__ __device__ float3 grid_cell_center(int cell) {
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

static __forceinline__ __device__ float component(float3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

static __forceinline__ __device__ bool grid_cell_from_point(float3 point, int &cell) {
    float c0;
    float c1;
    if (params.grid_axis == 0) {
        c0 = point.y;
        c1 = point.z;
    } else if (params.grid_axis == 1) {
        c0 = point.x;
        c1 = point.z;
    } else {
        c0 = point.x;
        c1 = point.y;
    }
    if (c0 < params.grid_coord0_min || c0 >= params.grid_coord0_max ||
        c1 < params.grid_coord1_min || c1 >= params.grid_coord1_max) {
        return false;
    }
    const float u = (c0 - params.grid_coord0_min) /
                    fmaxf(params.grid_coord0_max - params.grid_coord0_min, kSmallEps);
    const float v = (c1 - params.grid_coord1_min) /
                    fmaxf(params.grid_coord1_max - params.grid_coord1_min, kSmallEps);
    const int i = min(max(static_cast<int>(u * params.grid_resolution0), 0),
                      params.grid_resolution0 - 1);
    const int j = min(max(static_cast<int>(v * params.grid_resolution1), 0),
                      params.grid_resolution1 - 1);
    cell = j * params.grid_resolution0 + i;
    return true;
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

static __forceinline__ __device__ bool keller_grid_hit(int state_idx,
                                                       unsigned int lane,
                                                       float3 edge_point,
                                                       float3 edge_dir,
                                                       float3 &target,
                                                       int &cell) {
    const float3 incident =
        normalize3(state_vec(params.state_incident_dir_x,
                             params.state_incident_dir_y,
                             params.state_incident_dir_z,
                             state_idx));
    const float axial = fminf(fmaxf(dot3(incident, edge_dir), -1.f), 1.f);
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float3 basis0 = stable_perpendicular(edge_dir, incident);
    const float3 basis1 = normalize3(cross3(edge_dir, basis0));
    float s;
    float c;
    sincosf(2.f * kPi * uniform01(lane, 1u, static_cast<unsigned int>(params.seed)), &s, &c);
    const float3 ko = normalize3(axial * edge_dir + radial * (c * basis0 + s * basis1));
    const float denom = component(ko, params.grid_axis);
    if (fabsf(denom) <= kSmallEps) {
        return false;
    }
    const float t = (params.grid_position - component(edge_point, params.grid_axis)) / denom;
    if (!(t > kRayBias) || !isfinite(t)) {
        return false;
    }
    target = edge_point + t * ko;
    return grid_cell_from_point(target, cell);
}

static __forceinline__ __device__ float material_gain_for_faces(int face0_prim,
                                                                int face1_prim) {
    if (params.material_gain == nullptr || params.material_count <= 0) {
        return 1.f;
    }
    int prim = face0_prim;
    if (prim < 0 || prim >= params.material_count ||
        (params.material_valid != nullptr && params.material_valid[prim] == 0u)) {
        prim = face1_prim;
    }
    if (prim < 0 || prim >= params.material_count ||
        (params.material_valid != nullptr && params.material_valid[prim] == 0u)) {
        return 1.f;
    }
    return fmaxf(params.material_gain[prim], 0.f);
}

static __forceinline__ __device__ float material_gain_for_state(int state_idx) {
    return material_gain_for_faces(params.state_face0_prim_id[state_idx],
                                   params.state_face1_prim_id[state_idx]);
}

static __forceinline__ __device__ float diffraction_weight(int state_idx,
                                                           float3 edge_point,
                                                           float3 target,
                                                           int sample_count) {
    const float3 source =
        state_vec(params.state_source_x, params.state_source_y, params.state_source_z, state_idx);
    const float source_distance = fmaxf(norm3(edge_point - source), kSmallEps);
    const float target_distance = fmaxf(norm3(target - edge_point), kSmallEps);
    const float edge_length = fmaxf(
        params.state_edge_line_max[state_idx] - params.state_edge_line_min[state_idx],
        0.f);
    const float exterior_angle =
        fmaxf(params.state_exterior_angle[state_idx], 0.25f * kPi);
    const float wedge_scale = fminf(exterior_angle / (2.f * kPi), 2.f);
    const float material_gain = material_gain_for_state(state_idx);
    const float sample_norm = 1.f / fmaxf(static_cast<float>(sample_count), 1.f);
    return params.state_source_power[state_idx] *
           material_gain *
           edge_length *
           params.grid_cell_area *
           wedge_scale *
           sample_norm /
           (source_distance * source_distance * target_distance * target_distance);
}

static __forceinline__ __device__ float chain_event_weight(float source_power,
                                                           int face0_prim,
                                                           int face1_prim,
                                                           float edge_line_min,
                                                           float edge_line_max,
                                                           float exterior_angle,
                                                           float3 source,
                                                           float3 edge_point,
                                                           float3 target) {
    const float source_distance = fmaxf(norm3(edge_point - source), kSmallEps);
    const float target_distance = fmaxf(norm3(target - edge_point), kSmallEps);
    const float edge_length = fmaxf(edge_line_max - edge_line_min, 0.f);
    const float wedge_scale = fminf(fmaxf(exterior_angle, 0.25f * kPi) / (2.f * kPi), 2.f);
    const float material_gain = material_gain_for_faces(face0_prim, face1_prim);
    return source_power *
           material_gain *
           edge_length *
           wedge_scale /
           (source_distance * source_distance * target_distance * target_distance);
}

} // namespace

extern "C" {
__constant__ DiffractionAccumParams params;
}

extern "C" __global__ void __closesthit__diffraction_accumulation() {
    HitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_payload(payload);
}

extern "C" __global__ void __miss__diffraction_accumulation() {
    optixSetPayload_0(0u);
}

extern "C" __global__ void __raygen__diffraction_order1_accumulation() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.grid_resolution0 <= 0 ||
        params.grid_resolution1 <= 0) {
        return;
    }

    const int direct_limit =
        (params.strategy_mask & RAYD_DIFF_DIRECT) != 0 ? params.direct_samples : 0;
    const int keller_limit =
        (params.strategy_mask & RAYD_DIFF_KELLER) != 0 ? params.keller_samples : 0;
    const int total_samples = direct_limit + keller_limit;
    if (total_samples <= 0) {
        return;
    }
    const bool is_direct = static_cast<int>(lane) < direct_limit;
    const bool is_keller = !is_direct;
    if (is_keller && static_cast<int>(lane) >= total_samples) {
        return;
    }

    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(params.state_count));
    if (params.active_mask != nullptr && params.active_mask[state_idx] == 0u) {
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    int cell = static_cast<int>((lane / static_cast<unsigned int>(params.state_count)) %
                                static_cast<unsigned int>(grid_cell_count));
    const float edge_u = uniform01(lane, 0u, static_cast<unsigned int>(params.seed));
    const float edge_t = params.state_edge_line_min[state_idx] +
                         edge_u * (params.state_edge_line_max[state_idx] -
                                   params.state_edge_line_min[state_idx]);
    const float3 edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, state_idx);
    const float3 edge_dir =
        normalize3(state_vec(params.state_edge_dir_x, params.state_edge_dir_y, params.state_edge_dir_z, state_idx));
    const float3 edge_point = edge_pos + edge_t * edge_dir;
    const float3 source =
        state_vec(params.state_source_x, params.state_source_y, params.state_source_z, state_idx);
    float3 target = grid_cell_center(cell);
    if (is_keller && !keller_grid_hit(state_idx, lane, edge_point, edge_dir, target, cell)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_reject_count, 1);
        }
        return;
    }

    if (!visible_segment(source, edge_point) || !visible_segment(edge_point, target)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_visibility_reject_count, 1);
        }
        return;
    }

    const float contribution =
        diffraction_weight(state_idx, edge_point, target, total_samples);
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_reject_count, 1);
        }
        return;
    }

    atomicAdd(params.out_diffraction_power + cell, contribution);
    atomicAdd(params.out_field_x_re + cell, sqrtf(fmaxf(contribution, 0.f)));
    if (is_direct) {
        atomicAdd(params.out_direct_count, 1);
    } else {
        atomicAdd(params.out_keller_count, 1);
    }
    if (params.collect_edge_use != 0) {
        atomicAdd(params.out_edge_use_count, 1);
    }
}

extern "C" __global__ void __raygen__diffraction_chain_accumulation() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.state_count <= 0 ||
        params.recursive_state_count <= 0 ||
        params.grid_resolution0 <= 0 ||
        params.grid_resolution1 <= 0 ||
        (params.max_order != 2 && params.max_order != 3) ||
        (params.strategy_mask & RAYD_DIFF_DIRECT) == 0) {
        return;
    }

    const int direct_limit = params.direct_samples;
    if (direct_limit <= 0 || static_cast<int>(lane) >= direct_limit) {
        return;
    }

    const int first_idx = static_cast<int>(
        lane % static_cast<unsigned int>(params.state_count));
    const unsigned int second_hash = hash_u32(
        lane ^ (static_cast<unsigned int>(params.seed) * 0x9e3779b9u) ^ 0x51ed270bu);
    const int second_idx = static_cast<int>(
        second_hash % static_cast<unsigned int>(params.recursive_state_count));
    int third_idx = -1;
    if (params.active_mask != nullptr && params.active_mask[first_idx] == 0u) {
        return;
    }
    if (params.recursive_active_mask != nullptr &&
        params.recursive_active_mask[second_idx] == 0u) {
        return;
    }
    if (params.max_order == 3) {
        const unsigned int third_hash = hash_u32(
            lane ^ (static_cast<unsigned int>(params.seed) * 0x85ebca6bu) ^ 0xc2b2ae35u);
        third_idx = static_cast<int>(
            third_hash % static_cast<unsigned int>(params.recursive_state_count));
        if (params.recursive_active_mask != nullptr &&
            params.recursive_active_mask[third_idx] == 0u) {
            return;
        }
    }

    const int first_edge_index = params.state_edge_index[first_idx];
    const int second_edge_index = params.recursive_state_edge_index[second_idx];
    const int third_edge_index =
        params.max_order == 3 ? params.recursive_state_edge_index[third_idx] : -1;
    if (first_edge_index == second_edge_index ||
        (params.max_order == 3 &&
         (first_edge_index == third_edge_index || second_edge_index == third_edge_index))) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_reject_count, 1);
        }
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    const int cell = static_cast<int>(
        (lane / static_cast<unsigned int>(params.state_count)) %
        static_cast<unsigned int>(grid_cell_count));
    const float first_u = uniform01(lane, 0u, static_cast<unsigned int>(params.seed));
    const float second_u = uniform01(lane, 2u, static_cast<unsigned int>(params.seed));

    const float3 first_edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, first_idx);
    const float3 first_edge_dir =
        normalize3(state_vec(params.state_edge_dir_x,
                             params.state_edge_dir_y,
                             params.state_edge_dir_z,
                             first_idx));
    const float first_t = params.state_edge_line_min[first_idx] +
                          first_u * (params.state_edge_line_max[first_idx] -
                                     params.state_edge_line_min[first_idx]);
    const float3 first_point = first_edge_pos + first_t * first_edge_dir;

    const float3 second_edge_pos =
        recursive_state_vec(params.recursive_state_edge_pos_x,
                            params.recursive_state_edge_pos_y,
                            params.recursive_state_edge_pos_z,
                            second_idx);
    const float3 second_edge_dir =
        normalize3(recursive_state_vec(params.recursive_state_edge_dir_x,
                                       params.recursive_state_edge_dir_y,
                                       params.recursive_state_edge_dir_z,
                                       second_idx));
    const float second_t = params.recursive_state_edge_line_min[second_idx] +
                           second_u * (params.recursive_state_edge_line_max[second_idx] -
                                       params.recursive_state_edge_line_min[second_idx]);
    const float3 second_point = second_edge_pos + second_t * second_edge_dir;

    const float3 source =
        state_vec(params.state_source_x, params.state_source_y, params.state_source_z, first_idx);
    const float3 target = grid_cell_center(cell);
    float3 third_point = second_point;
    if (params.max_order == 3) {
        const float third_u = uniform01(lane, 4u, static_cast<unsigned int>(params.seed));
        const float3 third_edge_pos =
            recursive_state_vec(params.recursive_state_edge_pos_x,
                                params.recursive_state_edge_pos_y,
                                params.recursive_state_edge_pos_z,
                                third_idx);
        const float3 third_edge_dir =
            normalize3(recursive_state_vec(params.recursive_state_edge_dir_x,
                                           params.recursive_state_edge_dir_y,
                                           params.recursive_state_edge_dir_z,
                                           third_idx));
        const float third_t = params.recursive_state_edge_line_min[third_idx] +
                              third_u * (params.recursive_state_edge_line_max[third_idx] -
                                         params.recursive_state_edge_line_min[third_idx]);
        third_point = third_edge_pos + third_t * third_edge_dir;
    }
    const float3 terminal_point = params.max_order == 3 ? third_point : second_point;
    const bool source_visible = visible_segment(source, first_point);
    const bool first_inter_edge_visible = visible_segment(first_point, second_point);
    const bool second_inter_edge_visible =
        params.max_order == 3 ? visible_segment(second_point, third_point) : true;
    const bool target_visible = visible_segment(terminal_point, target);
    if (!source_visible || !target_visible) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_visibility_reject_count, 1);
        }
        return;
    }
    if (!first_inter_edge_visible || !second_inter_edge_visible) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_inter_edge_visibility_reject_count, 1);
        }
        return;
    }

    const float first_weight = chain_event_weight(
        params.state_source_power[first_idx],
        params.state_face0_prim_id[first_idx],
        params.state_face1_prim_id[first_idx],
        params.state_edge_line_min[first_idx],
        params.state_edge_line_max[first_idx],
        params.state_exterior_angle[first_idx],
        source,
        first_point,
        second_point);
    const float3 second_target = params.max_order == 3 ? third_point : target;
    const float second_weight = chain_event_weight(
        1.f,
        params.recursive_state_face0_prim_id[second_idx],
        params.recursive_state_face1_prim_id[second_idx],
        params.recursive_state_edge_line_min[second_idx],
        params.recursive_state_edge_line_max[second_idx],
        params.recursive_state_exterior_angle[second_idx],
        first_point,
        second_point,
        second_target);
    float chain_weight = first_weight * second_weight;
    if (params.max_order == 3) {
        const float third_weight = chain_event_weight(
            1.f,
            params.recursive_state_face0_prim_id[third_idx],
            params.recursive_state_face1_prim_id[third_idx],
            params.recursive_state_edge_line_min[third_idx],
            params.recursive_state_edge_line_max[third_idx],
            params.recursive_state_exterior_angle[third_idx],
            second_point,
            third_point,
            target);
        chain_weight *= third_weight;
    }
    const float wave_gain_per_event =
        (params.wavelength * (1.f / (4.f * kPi))) *
        (params.wavelength * (1.f / (4.f * kPi)));
    const float wave_gain =
        params.max_order == 3 ? wave_gain_per_event * wave_gain_per_event
                              : wave_gain_per_event;
    const float sample_norm = 1.f / fmaxf(static_cast<float>(direct_limit), 1.f);
    const float contribution =
        chain_weight * wave_gain * params.grid_cell_area * sample_norm;
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_reject_count, 1);
        }
        return;
    }

    atomicAdd(params.out_diffraction_power + cell, contribution);
    atomicAdd(params.out_field_x_re + cell, sqrtf(fmaxf(contribution, 0.f)));
    atomicAdd(params.out_direct_count, 1);
    if (params.collect_edge_use != 0) {
        atomicAdd(params.out_edge_use_count, 1);
    }
}

} // namespace rayd
