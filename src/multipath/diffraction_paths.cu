#include <optix.h>
#include <optix_device.h>

#include <rayd/multipath/diffraction_paths.h>
#include <rayd/multipath/diffraction_paths_params.h>

namespace rayd {

extern "C" {
extern __constant__ DiffractionPathParams params;
}

namespace {

constexpr float kTraceTMin = 1e-5f;
constexpr float kRayBias = 1e-4f;
constexpr float kSmallEps = 1e-6f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kSpeedOfLight = 299792458.f;

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
    const HitPayload hit =
        trace_scene(start + kRayBias * dir, dir, fmaxf(dist - 2.f * kRayBias, 0.f));
    return hit.hit == 0u;
}

static __forceinline__ __device__ float3 state_vec(const float *x,
                                                   const float *y,
                                                   const float *z,
                                                   int idx) {
    return make_vec3(x[idx], y[idx], z[idx]);
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

static __forceinline__ __device__ bool state_active(int state_idx) {
    if (params.active_mask == nullptr) {
        return true;
    }
    const int active_idx = params.active_width == 1 ? 0 : state_idx;
    return params.active_mask[active_idx] != 0u;
}

static __forceinline__ __device__ float path_weight(int state_idx,
                                                    float3 edge_point,
                                                    float3 receiver) {
    const float3 source =
        state_vec(params.state_source_x, params.state_source_y, params.state_source_z, state_idx);
    const float source_distance = fmaxf(norm3(edge_point - source), kSmallEps);
    const float receiver_distance = fmaxf(norm3(receiver - edge_point), kSmallEps);
    const float edge_length = fmaxf(
        params.state_edge_line_max[state_idx] - params.state_edge_line_min[state_idx],
        0.f);
    const float exterior_angle =
        fmaxf(params.state_exterior_angle[state_idx], 0.25f * kPi);
    const float wedge_scale = fminf(exterior_angle / (2.f * kPi), 2.f);
    const float material_gain = material_gain_for_faces(params.state_face0_prim_id[state_idx],
                                                       params.state_face1_prim_id[state_idx]);
    const float wave_gain = params.wavelength * (1.f / (4.f * kPi));
    return params.state_source_power[state_idx] *
           material_gain *
           edge_length *
           wedge_scale *
           wave_gain *
           wave_gain /
           (source_distance * source_distance * receiver_distance * receiver_distance);
}

static __forceinline__ __device__ void write_point(float *x,
                                                   float *y,
                                                   float *z,
                                                   int idx,
                                                   float3 value) {
    if (x == nullptr || y == nullptr || z == nullptr) {
        return;
    }
    x[idx] = value.x;
    y[idx] = value.y;
    z[idx] = value.z;
}

} // namespace

extern "C" {
__constant__ DiffractionPathParams params;
}

extern "C" __global__ void __closesthit__diffraction_paths() {
    HitPayload payload;
    payload.hit = 1u;
    payload.t = __float_as_uint(optixGetRayTmax());
    payload.prim = optixGetPrimitiveIndex();
    payload.instance = optixGetInstanceId();
    set_payload(payload);
}

extern "C" __global__ void __miss__diffraction_paths() {
    optixSetPayload_0(0u);
}

extern "C" __global__ void __raygen__diffraction_paths_order1() {
    const unsigned int lane = optixGetLaunchIndex().x;
    if (lane >= static_cast<unsigned int>(params.n_rays) ||
        params.capacity <= 0 ||
        params.tx_count <= 0 ||
        params.rx_count <= 0 ||
        params.state_count <= 0 ||
        params.state_limit <= 0 ||
        params.max_order != 1 ||
        (params.strategy_mask & RAYD_DIFF_DIRECT) == 0 ||
        params.receiver_model != RAYD_DIFF_MATCHED_ISOTROPIC) {
        return;
    }

    const int state_limit = params.state_limit;
    const int rx_count = params.rx_count;
    const int state_idx = static_cast<int>(lane % static_cast<unsigned int>(state_limit));
    const int pair_idx = static_cast<int>(lane / static_cast<unsigned int>(state_limit));
    const int rx_idx = pair_idx % rx_count;
    const int tx_idx = pair_idx / rx_count;
    if (tx_idx >= params.tx_count || state_idx >= params.state_count || !state_active(state_idx)) {
        return;
    }

    const float3 source =
        state_vec(params.state_source_x, params.state_source_y, params.state_source_z, state_idx);
    const float3 edge_pos =
        state_vec(params.state_edge_pos_x, params.state_edge_pos_y, params.state_edge_pos_z, state_idx);
    const float3 edge_dir =
        normalize3(state_vec(params.state_edge_dir_x,
                             params.state_edge_dir_y,
                             params.state_edge_dir_z,
                             state_idx));
    const float mid_t = 0.5f * (params.state_edge_line_min[state_idx] +
                               params.state_edge_line_max[state_idx]);
    const float3 edge_point = edge_pos + mid_t * edge_dir;
    const float3 receiver =
        make_vec3(params.rx_pos_x[rx_idx], params.rx_pos_y[rx_idx], params.rx_pos_z[rx_idx]);

    if (!isfinite(source.x) || !isfinite(source.y) || !isfinite(source.z) ||
        !isfinite(edge_point.x) || !isfinite(edge_point.y) || !isfinite(edge_point.z) ||
        !isfinite(receiver.x) || !isfinite(receiver.y) || !isfinite(receiver.z)) {
        return;
    }
    if (!visible_segment(source, edge_point) || !visible_segment(edge_point, receiver)) {
        return;
    }

    const float contribution = path_weight(state_idx, edge_point, receiver);
    if (!(contribution > 0.f) || !isfinite(contribution)) {
        return;
    }

    const int out_idx = atomicAdd(params.out_count, 1);
    if (out_idx < 0 || out_idx >= params.capacity) {
        return;
    }

    const float path_length = norm3(edge_point - source) + norm3(receiver - edge_point);
    float phase_s;
    float phase_c;
    sincosf(-params.k * path_length, &phase_s, &phase_c);
    const float amplitude = sqrtf(fmaxf(contribution, 0.f));

    params.out_valid[out_idx] = 1u;
    params.out_tx_index[out_idx] = tx_idx;
    params.out_rx_index[out_idx] = rx_idx;
    params.out_order[out_idx] = 1;
    params.out_edge_index_0[out_idx] = params.state_edge_index[state_idx];
    params.out_edge_index_1[out_idx] = -1;
    params.out_edge_index_2[out_idx] = -1;
    params.out_delay[out_idx] = path_length / kSpeedOfLight;
    params.out_field_x_re[out_idx] = amplitude * phase_c;
    params.out_field_x_im[out_idx] = amplitude * phase_s;
    params.out_field_y_re[out_idx] = 0.f;
    params.out_field_y_im[out_idx] = 0.f;
    params.out_field_z_re[out_idx] = 0.f;
    params.out_field_z_im[out_idx] = 0.f;
    write_point(params.out_point_0_x, params.out_point_0_y, params.out_point_0_z,
                out_idx, edge_point);
    write_point(params.out_point_1_x, params.out_point_1_y, params.out_point_1_z,
                out_idx, make_vec3(0.f, 0.f, 0.f));
    write_point(params.out_point_2_x, params.out_point_2_y, params.out_point_2_z,
                out_idx, make_vec3(0.f, 0.f, 0.f));
}

} // namespace rayd
