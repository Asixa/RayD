#include <optix.h>
#include <optix_device.h>

#include <rayd/multipath/diffraction_accumulation.h>
#include <rayd/multipath/diffraction_accumulation_params.h>

namespace rayd {

extern "C" {
extern __constant__ DfrAccumParams params;
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

static __forceinline__ __device__ int global_primitive_id(const HitPayload &hit) {
    if (hit.hit == 0u) {
        return -1;
    }
    const int instance = static_cast<int>(hit.instance);
    if (params.face_offsets != nullptr &&
        instance >= 0 &&
        instance < params.n_meshes) {
        return params.face_offsets[instance] + static_cast<int>(hit.prim);
    }
    return static_cast<int>(hit.prim);
}

static __forceinline__ __device__ bool visible_segment_ignore_prim(float3 start,
                                                                   float3 end,
                                                                   int ignore_prim) {
    const float3 delta = end - start;
    const float dist = norm3(delta);
    if (dist <= 1e-5f) {
        return true;
    }
    const float3 dir = (1.f / dist) * delta;
    const HitPayload hit = trace_scene(start + kRayBias * dir, dir, fmaxf(dist - 2.f * kRayBias, 0.f));
    if (hit.hit == 0u) {
        return true;
    }
    return global_primitive_id(hit) == ignore_prim;
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

static __forceinline__ __device__ bool keller_grid_hit_from_incident(float3 incident_vec,
                                                                     unsigned int lane,
                                                                     unsigned int stream,
                                                                     float3 edge_point,
                                                                     float3 edge_dir,
                                                                     float3 &target,
                                                                     int &cell) {
    const float3 incident = normalize3(incident_vec);
    const float axial = fminf(fmaxf(dot3(incident, edge_dir), -1.f), 1.f);
    const float radial = sqrtf(fmaxf(1.f - axial * axial, 0.f));
    const float3 basis0 = stable_perpendicular(edge_dir, incident);
    const float3 basis1 = normalize3(cross3(edge_dir, basis0));
    float s;
    float c;
    sincosf(2.f * kPi * uniform01(lane, stream, static_cast<unsigned int>(params.seed)), &s, &c);
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

static __forceinline__ __device__ bool keller_grid_hit(int state_idx,
                                                       unsigned int lane,
                                                       float3 edge_point,
                                                       float3 edge_dir,
                                                       float3 &target,
                                                       int &cell) {
    const float3 incident =
        state_vec(params.state_wi_x,
                  params.state_wi_y,
                  params.state_wi_z,
                  state_idx);
    return keller_grid_hit_from_incident(incident, lane, 1u, edge_point, edge_dir, target, cell);
}

static __forceinline__ __device__ int material_index_for_faces(int face0_prim,
                                                               int face1_prim) {
    if (params.material_gain == nullptr || params.material_count <= 0) {
        return -1;
    }
    int prim = face0_prim;
    if (prim < 0 || prim >= params.material_count ||
        (params.material_valid != nullptr && params.material_valid[prim] == 0u)) {
        prim = face1_prim;
    }
    if (prim < 0 || prim >= params.material_count ||
        (params.material_valid != nullptr && params.material_valid[prim] == 0u)) {
        return -1;
    }
    return prim;
}

static __forceinline__ __device__ float material_gain_for_faces(int face0_prim,
                                                                int face1_prim) {
    const int prim = material_index_for_faces(face0_prim, face1_prim);
    if (prim < 0) {
        return 1.f;
    }
    return fmaxf(params.material_gain[prim], 0.f);
}

static __forceinline__ __device__ float material_gain_for_state(int state_idx) {
    return material_gain_for_faces(params.state_prim0[state_idx],
                                   params.state_prim1[state_idx]);
}

static __forceinline__ __device__ float material_gain_for_prim(int prim) {
    if (params.material_gain == nullptr ||
        prim < 0 ||
        prim >= params.material_count ||
        (params.material_valid != nullptr && params.material_valid[prim] == 0u)) {
        return 1.f;
    }
    return fmaxf(params.material_gain[prim], 0.f);
}

static __forceinline__ __device__ bool suffix_candidate_valid(int prim) {
    return prim >= 0 &&
           prim < params.n_triangles &&
           prim < params.material_count &&
           params.material_valid != nullptr &&
           params.material_valid[prim] != 0u;
}

static __forceinline__ __device__ bool select_local_suffix_candidate(int face0_prim,
                                                                     int face1_prim,
                                                                     unsigned int lane,
                                                                     unsigned int stream,
                                                                     int &prim,
                                                                     float &candidate_count) {
    const bool face0_valid = suffix_candidate_valid(face0_prim);
    const bool face1_valid =
        suffix_candidate_valid(face1_prim) && face1_prim != face0_prim;
    const int count = (face0_valid ? 1 : 0) + (face1_valid ? 1 : 0);
    if (count <= 0) {
        return false;
    }
    const unsigned int candidate_hash = hash_u32(
        lane ^ (stream * 0x9e3779b9u) ^ static_cast<unsigned int>(params.seed));
    const int slot = static_cast<int>(candidate_hash % static_cast<unsigned int>(count));
    if (face0_valid && slot == 0) {
        prim = face0_prim;
    } else {
        prim = face1_prim;
    }
    candidate_count = static_cast<float>(count);
    return true;
}

static __forceinline__ __device__ bool load_triangle(int prim,
                                                     float3 &p0,
                                                     float3 &e1,
                                                     float3 &e2,
                                                     float3 &normal) {
    if (prim < 0 ||
        prim >= params.n_triangles ||
        params.tri_p0_x == nullptr ||
        params.tri_e1_x == nullptr ||
        params.tri_e2_x == nullptr ||
        params.tri_fn_x == nullptr) {
        return false;
    }
    p0 = make_vec3(params.tri_p0_x[prim], params.tri_p0_y[prim], params.tri_p0_z[prim]);
    e1 = make_vec3(params.tri_e1_x[prim], params.tri_e1_y[prim], params.tri_e1_z[prim]);
    e2 = make_vec3(params.tri_e2_x[prim], params.tri_e2_y[prim], params.tri_e2_z[prim]);
    normal = make_vec3(params.tri_fn_x[prim], params.tri_fn_y[prim], params.tri_fn_z[prim]);
    if (dot3(normal, normal) <= 1e-12f) {
        normal = cross3(e1, e2);
    }
    normal = normalize3(normal);
    return dot3(normal, normal) > 0.f;
}

static __forceinline__ __device__ bool intersect_reflection_triangle(float3 image_source,
                                                                     float3 target,
                                                                     int prim,
                                                                     float3 &reflection_point,
                                                                     float3 &normal) {
    float3 p0;
    float3 e1;
    float3 e2;
    if (!load_triangle(prim, p0, e1, e2, normal)) {
        return false;
    }
    const float3 delta = target - image_source;
    const float dist = norm3(delta);
    if (!(dist > kRayBias) || !isfinite(dist)) {
        return false;
    }
    const float3 dir = (1.f / dist) * delta;
    const float3 h = cross3(dir, e2);
    const float a = dot3(e1, h);
    if (fabsf(a) <= 1e-7f) {
        return false;
    }
    const float f = 1.f / a;
    const float3 s = image_source - p0;
    const float u = f * dot3(s, h);
    if (u < -1e-5f || u > 1.f + 1e-5f) {
        return false;
    }
    const float3 q = cross3(s, e1);
    const float v = f * dot3(dir, q);
    if (v < -1e-5f || u + v > 1.f + 1e-5f) {
        return false;
    }
    const float t = f * dot3(e2, q);
    if (!(t > kRayBias) || !(t < dist - kRayBias) || !isfinite(t)) {
        return false;
    }
    reflection_point = image_source + t * dir;
    return true;
}

static __forceinline__ __device__ bool suffix_reflection_connection(float3 diff_point,
                                                                    float3 target,
                                                                    int face0_prim,
                                                                    int face1_prim,
                                                                    unsigned int lane,
                                                                    unsigned int stream,
                                                                    float3 &reflection_point,
                                                                    int &prim,
                                                                    float &reflection_gain,
                                                                    float &suffix_fspl,
                                                                    float &candidate_count) {
    if (!select_local_suffix_candidate(face0_prim,
                                       face1_prim,
                                       lane,
                                       stream,
                                       prim,
                                       candidate_count)) {
        return false;
    }
    float3 p0;
    float3 e1;
    float3 e2;
    float3 normal;
    if (!load_triangle(prim, p0, e1, e2, normal)) {
        return false;
    }
    const float plane_distance = dot3(diff_point - p0, normal);
    const float3 image_source = diff_point - 2.f * plane_distance * normal;
    if (!intersect_reflection_triangle(image_source, target, prim, reflection_point, normal)) {
        return false;
    }

    const float3 incoming = reflection_point - diff_point;
    const float3 outgoing = target - reflection_point;
    const float incoming_dist = norm3(incoming);
    const float outgoing_dist = norm3(outgoing);
    if (!(incoming_dist > kSmallEps) || !(outgoing_dist > kSmallEps)) {
        return false;
    }
    const float3 incoming_hat = (1.f / incoming_dist) * incoming;
    const float3 oriented_normal =
        dot3(incoming_hat, normal) > 0.f ? (-1.f * normal) : normal;
    const float3 reflected_hat =
        incoming_hat - 2.f * dot3(incoming_hat, oriented_normal) * oriented_normal;
    const float3 outgoing_hat = (1.f / outgoing_dist) * outgoing;
    if (dot3(reflected_hat, outgoing_hat) <= 1.f - 1e-3f) {
        return false;
    }

    const float gain = material_gain_for_prim(prim);
    reflection_gain = gain * gain;
    suffix_fspl = (params.wavelength * (1.f / (4.f * kPi))) *
                  (params.wavelength * (1.f / (4.f * kPi))) /
                  fmaxf(outgoing_dist * outgoing_dist, kSmallEps);
    if (!(isfinite(reflection_gain) && isfinite(suffix_fspl))) {
        return false;
    }
    return true;
}

static __forceinline__ __device__ float diffraction_weight(int state_idx,
                                                           float3 edge_point,
                                                           float3 target,
                                                           int sample_count) {
    const float3 source =
        state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    const float source_distance = fmaxf(norm3(edge_point - source), kSmallEps);
    const float target_distance = fmaxf(norm3(target - edge_point), kSmallEps);
    const float edge_length = fmaxf(
        params.state_edge_t_max[state_idx] - params.state_edge_t_min[state_idx],
        0.f);
    const float exterior_angle =
        fmaxf(params.state_exterior_angle[state_idx], 0.25f * kPi);
    const float wedge_scale = fminf(exterior_angle / (2.f * kPi), 2.f);
    const float material_gain = material_gain_for_state(state_idx);
    const float sample_norm = 1.f / fmaxf(static_cast<float>(sample_count), 1.f);
    return params.state_src_power[state_idx] *
           material_gain *
           edge_length *
           params.grid_cell_area *
           wedge_scale *
           sample_norm /
           (source_distance * source_distance * target_distance * target_distance);
}

static __forceinline__ __device__ float chain_event_weight(float src_power,
                                                           int face0_prim,
                                                           int face1_prim,
                                                           float edge_t_min,
                                                           float edge_t_max,
                                                           float exterior_angle,
                                                           float3 source,
                                                           float3 edge_point,
                                                           float3 target) {
    const float source_distance = fmaxf(norm3(edge_point - source), kSmallEps);
    const float target_distance = fmaxf(norm3(target - edge_point), kSmallEps);
    const float edge_length = fmaxf(edge_t_max - edge_t_min, 0.f);
    const float wedge_scale = fminf(fmaxf(exterior_angle, 0.25f * kPi) / (2.f * kPi), 2.f);
    const float material_gain = material_gain_for_faces(face0_prim, face1_prim);
    return src_power *
           material_gain *
           edge_length *
           wedge_scale /
           (source_distance * source_distance * target_distance * target_distance);
}

} // namespace

extern "C" {
__constant__ DfrAccumParams params;
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
        (params.strategy_mask & RAYD_DFR_DIRECT) != 0 ? params.direct_samples : 0;
    const int keller_limit =
        (params.strategy_mask & RAYD_DFR_KELLER) != 0 ? params.keller_samples : 0;
    const int suffix_limit =
        (params.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? params.suffix_samples : 0;
    const int total_samples = direct_limit + keller_limit + suffix_limit;
    if (total_samples <= 0) {
        return;
    }
    const bool is_direct = static_cast<int>(lane) < direct_limit;
    const bool is_keller =
        !is_direct && static_cast<int>(lane) < direct_limit + keller_limit;
    const bool is_suffix =
        static_cast<int>(lane) >= direct_limit + keller_limit &&
        static_cast<int>(lane) < total_samples;
    if (!is_direct && !is_keller && !is_suffix) {
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
    const float edge_t = params.state_edge_t_min[state_idx] +
                         edge_u * (params.state_edge_t_max[state_idx] -
                                   params.state_edge_t_min[state_idx]);
    const float3 edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, state_idx);
    const float3 edge_dir =
        normalize3(state_vec(params.state_edge_dir_x, params.state_edge_dir_y, params.state_edge_dir_z, state_idx));
    const float3 edge_point = edge_pos + edge_t * edge_dir;
    const float3 source =
        state_vec(params.state_src_x, params.state_src_y, params.state_src_z, state_idx);
    float3 target = grid_cell_center(cell);
    if (is_keller && !keller_grid_hit(state_idx, lane, edge_point, edge_dir, target, cell)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    float suffix_reflection_gain = 1.f;
    float suffix_fspl = 1.f;
    float suffix_candidate_count = 1.f;
    int suffix_prim = -1;
    float3 connection_target = target;
    if (is_suffix) {
        if (!suffix_reflection_connection(edge_point,
                                          target,
                                          params.state_prim0[state_idx],
                                          params.state_prim1[state_idx],
                                          lane,
                                          17u,
                                          connection_target,
                                          suffix_prim,
                                          suffix_reflection_gain,
                                          suffix_fspl,
                                          suffix_candidate_count)) {
            if (params.collect_debug_counts != 0) {
                atomicAdd(params.out_utd_rejects, 1);
            }
            return;
        }
    }

    const bool source_visible = visible_segment(source, edge_point);
    const bool target_visible = is_suffix
        ? (visible_segment_ignore_prim(edge_point, connection_target, suffix_prim) &&
           visible_segment_ignore_prim(connection_target, target, suffix_prim))
        : visible_segment(edge_point, target);
    if (!source_visible || !target_visible) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_vis_rejects, 1);
        }
        return;
    }

    const int strategy_sample_count =
        is_direct ? direct_limit : (is_keller ? keller_limit : suffix_limit);
    float contribution =
        diffraction_weight(state_idx, edge_point, connection_target, strategy_sample_count);
    if (is_suffix) {
        contribution *= suffix_reflection_gain *
                        suffix_fspl *
                        fmaxf(suffix_candidate_count, 1.f);
    }
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    if (params.tape_active != nullptr) {
        params.tape_active[lane] = 1u;
        if (params.tape_state_idx != nullptr) {
            params.tape_state_idx[lane] = state_idx;
        }
        if (params.tape_cell != nullptr) {
            params.tape_cell[lane] = cell;
        }
        if (params.tape_material_idx != nullptr) {
            params.tape_material_idx[lane] =
                material_index_for_faces(params.state_prim0[state_idx],
                                         params.state_prim1[state_idx]);
        }
        if (params.tape_edge_u != nullptr) {
            params.tape_edge_u[lane] = edge_u;
        }
    }

    atomicAdd(params.out_power + cell, contribution);
    atomicAdd(params.out_field_x_re + cell, sqrtf(fmaxf(contribution, 0.f)));
    if (is_direct) {
        atomicAdd(params.out_direct_count, 1);
    } else if (is_keller) {
        atomicAdd(params.out_keller_count, 1);
    } else {
        atomicAdd(params.out_suffix_count, 1);
    }
    if (params.collect_edge_use != 0) {
        atomicAdd(params.out_edge_uses, 1);
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
        (params.strategy_mask & (RAYD_DFR_DIRECT | RAYD_DFR_KELLER)) == 0) {
        return;
    }

    const int direct_limit =
        (params.strategy_mask & RAYD_DFR_DIRECT) != 0 ? params.direct_samples : 0;
    const int keller_limit =
        (params.strategy_mask & RAYD_DFR_KELLER) != 0 ? params.keller_samples : 0;
    const int suffix_limit =
        (params.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? params.suffix_samples : 0;
    const int total_samples = direct_limit + keller_limit + suffix_limit;
    if (total_samples <= 0 || static_cast<int>(lane) >= total_samples) {
        return;
    }
    const bool is_direct = static_cast<int>(lane) < direct_limit;
    const bool is_keller =
        !is_direct && static_cast<int>(lane) < direct_limit + keller_limit;
    const bool is_suffix =
        static_cast<int>(lane) >= direct_limit + keller_limit &&
        static_cast<int>(lane) < total_samples;

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
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    const int grid_cell_count = params.grid_resolution0 * params.grid_resolution1;
    int cell = static_cast<int>(
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
    const float first_t = params.state_edge_t_min[first_idx] +
                          first_u * (params.state_edge_t_max[first_idx] -
                                     params.state_edge_t_min[first_idx]);
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
    const float second_t = params.recursive_state_edge_t_min[second_idx] +
                           second_u * (params.recursive_state_edge_t_max[second_idx] -
                                       params.recursive_state_edge_t_min[second_idx]);
    const float3 second_point = second_edge_pos + second_t * second_edge_dir;

    const float3 source =
        state_vec(params.state_src_x, params.state_src_y, params.state_src_z, first_idx);
    const float3 target = grid_cell_center(cell);
    float3 third_point = second_point;
    float3 third_edge_dir = second_edge_dir;
    if (params.max_order == 3) {
        const float third_u = uniform01(lane, 4u, static_cast<unsigned int>(params.seed));
        const float3 third_edge_pos =
            recursive_state_vec(params.recursive_state_edge_pos_x,
                                params.recursive_state_edge_pos_y,
                                params.recursive_state_edge_pos_z,
                                third_idx);
        third_edge_dir =
            normalize3(recursive_state_vec(params.recursive_state_edge_dir_x,
                                           params.recursive_state_edge_dir_y,
                                           params.recursive_state_edge_dir_z,
                                           third_idx));
        const float third_t = params.recursive_state_edge_t_min[third_idx] +
                              third_u * (params.recursive_state_edge_t_max[third_idx] -
                                         params.recursive_state_edge_t_min[third_idx]);
        third_point = third_edge_pos + third_t * third_edge_dir;
    }
    const float3 terminal_point = params.max_order == 3 ? third_point : second_point;
    const float3 terminal_edge_dir = params.max_order == 3 ? third_edge_dir : second_edge_dir;
    float3 final_target = target;
    if (is_keller) {
        const float3 terminal_incident =
            params.max_order == 3 ? (third_point - second_point) : (second_point - first_point);
        if (!keller_grid_hit_from_incident(terminal_incident,
                                           lane,
                                           7u + static_cast<unsigned int>(params.max_order),
                                           terminal_point,
                                           terminal_edge_dir,
                                           final_target,
                                           cell)) {
            if (params.collect_debug_counts != 0) {
                atomicAdd(params.out_utd_rejects, 1);
            }
            return;
        }
    }
    float suffix_reflection_gain = 1.f;
    float suffix_fspl = 1.f;
    float suffix_candidate_count = 1.f;
    int suffix_prim = -1;
    if (is_suffix) {
        const int suffix_face0_prim =
            params.max_order == 3
                ? params.recursive_state_prim0[third_idx]
                : params.recursive_state_prim0[second_idx];
        const int suffix_face1_prim =
            params.max_order == 3
                ? params.recursive_state_prim1[third_idx]
                : params.recursive_state_prim1[second_idx];
        if (!suffix_reflection_connection(terminal_point,
                                          target,
                                          suffix_face0_prim,
                                          suffix_face1_prim,
                                          lane,
                                          23u + static_cast<unsigned int>(params.max_order),
                                          final_target,
                                          suffix_prim,
                                          suffix_reflection_gain,
                                          suffix_fspl,
                                          suffix_candidate_count)) {
            if (params.collect_debug_counts != 0) {
                atomicAdd(params.out_utd_rejects, 1);
            }
            return;
        }
    }
    const bool source_visible = visible_segment(source, first_point);
    const bool first_inter_edge_visible = visible_segment(first_point, second_point);
    const bool second_inter_edge_visible =
        params.max_order == 3 ? visible_segment(second_point, third_point) : true;
    const bool target_visible = is_suffix
        ? (visible_segment_ignore_prim(terminal_point, final_target, suffix_prim) &&
           visible_segment_ignore_prim(final_target, target, suffix_prim))
        : visible_segment(terminal_point, final_target);
    if (!source_visible || !target_visible) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_vis_rejects, 1);
        }
        return;
    }
    if (!first_inter_edge_visible || !second_inter_edge_visible) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_edge_vis_rejects, 1);
        }
        return;
    }

    const float first_weight = chain_event_weight(
        params.state_src_power[first_idx],
        params.state_prim0[first_idx],
        params.state_prim1[first_idx],
        params.state_edge_t_min[first_idx],
        params.state_edge_t_max[first_idx],
        params.state_exterior_angle[first_idx],
        source,
        first_point,
        second_point);
    const float3 second_target = params.max_order == 3 ? third_point : final_target;
    const float second_weight = chain_event_weight(
        1.f,
        params.recursive_state_prim0[second_idx],
        params.recursive_state_prim1[second_idx],
        params.recursive_state_edge_t_min[second_idx],
        params.recursive_state_edge_t_max[second_idx],
        params.recursive_state_exterior_angle[second_idx],
        first_point,
        second_point,
        second_target);
    float chain_weight = first_weight * second_weight;
    if (params.max_order == 3) {
        const float third_weight = chain_event_weight(
            1.f,
            params.recursive_state_prim0[third_idx],
            params.recursive_state_prim1[third_idx],
            params.recursive_state_edge_t_min[third_idx],
            params.recursive_state_edge_t_max[third_idx],
            params.recursive_state_exterior_angle[third_idx],
            second_point,
            third_point,
            final_target);
        chain_weight *= third_weight;
    }
    const float wave_gain_per_event =
        (params.wavelength * (1.f / (4.f * kPi))) *
        (params.wavelength * (1.f / (4.f * kPi)));
    const float wave_gain =
        params.max_order == 3 ? wave_gain_per_event * wave_gain_per_event
                              : wave_gain_per_event;
    const int strategy_sample_count =
        is_direct ? direct_limit : (is_keller ? keller_limit : suffix_limit);
    const float sample_norm = 1.f / fmaxf(static_cast<float>(strategy_sample_count), 1.f);
    float contribution =
        chain_weight * wave_gain * params.grid_cell_area * sample_norm;
    if (is_suffix) {
        contribution *= suffix_reflection_gain *
                        suffix_fspl *
                        fmaxf(suffix_candidate_count, 1.f);
    }
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        if (params.collect_debug_counts != 0) {
            atomicAdd(params.out_utd_rejects, 1);
        }
        return;
    }

    if (params.tape_active != nullptr) {
        params.tape_active[lane] = 1u;
        if (params.tape_state_idx != nullptr) {
            params.tape_state_idx[lane] = first_idx;
        }
        if (params.tape_cell != nullptr) {
            params.tape_cell[lane] = cell;
        }
        if (params.tape_material_idx != nullptr) {
            params.tape_material_idx[lane] =
                material_index_for_faces(params.state_prim0[first_idx],
                                         params.state_prim1[first_idx]);
        }
        if (params.tape_edge_u != nullptr) {
            params.tape_edge_u[lane] = first_u;
        }
    }

    atomicAdd(params.out_power + cell, contribution);
    atomicAdd(params.out_field_x_re + cell, sqrtf(fmaxf(contribution, 0.f)));
    if (is_direct) {
        atomicAdd(params.out_direct_count, 1);
    } else if (is_keller) {
        atomicAdd(params.out_keller_count, 1);
    } else {
        atomicAdd(params.out_suffix_count, 1);
    }
    if (params.collect_edge_use != 0) {
        atomicAdd(params.out_edge_uses, 1);
    }
}

} // namespace rayd
