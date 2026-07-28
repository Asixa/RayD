#include <optix.h>
#include <optix_device.h>

#include <rayd/surfel/drjit/surfel_trace_params.h>

namespace rayd {

extern "C" {
__constant__ SurfelTraceParams params;
}

namespace {

constexpr float kInvalidT = 3.4028234663852886e38f;
constexpr float kSortEpsilon = 1.0e-6f;

static __forceinline__ __device__ float3 make_vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

static __forceinline__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float squared_norm(float3 a) {
    return dot3(a, a);
}

static __forceinline__ __device__ bool is_active(unsigned int ray) {
    return params.active_mask == nullptr || params.active_mask[ray] != 0u;
}

static __forceinline__ __device__ float3 load_ray_origin(unsigned int ray) {
    return make_vec3(params.ray_ox[ray], params.ray_oy[ray], params.ray_oz[ray]);
}

static __forceinline__ __device__ float3 load_ray_direction(unsigned int ray) {
    return make_vec3(params.ray_dx[ray], params.ray_dy[ray], params.ray_dz[ray]);
}

static __forceinline__ __device__ float3 load_center(int surfel) {
    return make_vec3(params.center_x[surfel],
                     params.center_y[surfel],
                     params.center_z[surfel]);
}

static __forceinline__ __device__ float3 load_tangent_u(int surfel) {
    return make_vec3(params.tangent_u_x[surfel],
                     params.tangent_u_y[surfel],
                     params.tangent_u_z[surfel]);
}

static __forceinline__ __device__ float3 load_tangent_v(int surfel) {
    return make_vec3(params.tangent_v_x[surfel],
                     params.tangent_v_y[surfel],
                     params.tangent_v_z[surfel]);
}

static __forceinline__ __device__ void write_invalid(unsigned int ray) {
    if (params.out_triangle_id != nullptr) {
        params.out_triangle_id[ray] = -1;
    }
    if (params.out_proxy_t != nullptr) {
        params.out_proxy_t[ray] = kInvalidT;
    }
    if (params.out_valid != nullptr) {
        params.out_valid[ray] = 0u;
    }
}

static __forceinline__ __device__ float infinity() {
    return __uint_as_float(0x7f800000u);
}

static __forceinline__ __device__ float ray_tmax(unsigned int ray) {
    float tmax = params.ray_tmax != nullptr ? params.ray_tmax[ray] : params.tmax_fallback;
    if (!(tmax > 0.0f) || !isfinite(tmax)) {
        tmax = params.tmax_fallback;
    }
    return tmax;
}

static __forceinline__ __device__ bool evaluate_gaussian_hit(float3 origin,
                                                             float3 direction,
                                                             float trace_tmax,
                                                             int surfel,
                                                             float &plane_t,
                                                             float *out_alpha = nullptr) {
    if (surfel < 0 || surfel >= params.surfel_count) {
        return false;
    }

    const float3 center = load_center(surfel);
    const float3 tangent_u = load_tangent_u(surfel);
    const float3 tangent_v = load_tangent_v(surfel);

    const float3 raw_normal = cross3(tangent_u, tangent_v);
    const float normal_len_sq = squared_norm(raw_normal);
    if (!(normal_len_sq > 1.0e-16f)) {
        return false;
    }

    float3 normal = raw_normal * rsqrtf(normal_len_sq);
    if (params.face_forward != 0 && dot3(normal, direction) > 0.0f) {
        normal = -normal;
    }

    const float denom = dot3(direction, normal);
    if (!(fabsf(denom) > 1.0e-8f)) {
        return false;
    }

    plane_t = dot3(center - origin, normal) / denom;
    if (!isfinite(plane_t) ||
        !(plane_t > params.ray_epsilon) ||
        !(plane_t < trace_tmax)) {
        return false;
    }

    const float3 hit_point = origin + direction * plane_t;
    const float3 delta = hit_point - center;

    const float uu = dot3(tangent_u, tangent_u);
    const float uv = dot3(tangent_u, tangent_v);
    const float vv = dot3(tangent_v, tangent_v);
    const float du = dot3(delta, tangent_u);
    const float dv = dot3(delta, tangent_v);
    const float basis_det = uu * vv - uv * uv;
    if (!(fabsf(basis_det) > 1.0e-16f)) {
        return false;
    }

    const float local_u = (du * vv - dv * uv) / basis_det;
    const float local_v = (dv * uu - du * uv) / basis_det;
    const float rho = local_u * local_u + local_v * local_v;
    const float alpha_uncapped = params.opacity[surfel] * expf(-0.5f * rho);
    const float alpha_accept_slack = 1.0e-6f * fmaxf(1.0f, params.alpha_min);
    const bool valid = alpha_uncapped + alpha_accept_slack >= params.alpha_min;
    if (out_alpha != nullptr) {
        *out_alpha = valid ? fminf(params.alpha_cap, fmaxf(0.0f, alpha_uncapped)) : 0.0f;
    }
    return valid;
}

static __forceinline__ __device__ float sh_basis(int basis, float3 view) {
    switch (basis) {
        case 0: return 0.28209479177387814f;
        case 1: return 0.4886025119029199f * view.y;
        case 2: return 0.4886025119029199f * view.z;
        case 3: return 0.4886025119029199f * view.x;
        case 4: return 1.0925484305920792f * view.x * view.y;
        case 5: return 1.0925484305920792f * view.y * view.z;
        case 6: return 0.31539156525252005f * (3.0f * view.z * view.z - 1.0f);
        case 7: return 1.0925484305920792f * view.x * view.z;
        case 8: return 0.5462742152960396f * (view.x * view.x - view.y * view.y);
        case 9: return 0.5900435899266435f * view.y *
                       (3.0f * view.x * view.x - view.y * view.y);
        case 10: return 2.890611442640554f * view.x * view.y * view.z;
        case 11: return 0.4570457994644658f * view.y *
                        (5.0f * view.z * view.z - 1.0f);
        case 12: return 0.3731763325901154f * view.z *
                        (5.0f * view.z * view.z - 3.0f);
        case 13: return 0.4570457994644658f * view.x *
                        (5.0f * view.z * view.z - 1.0f);
        case 14: return 1.445305721320277f * view.z *
                        (view.x * view.x - view.y * view.y);
        case 15: return 0.5900435899266435f * view.x *
                        (view.x * view.x - 3.0f * view.y * view.y);
        default: return 0.0f;
    }
}

static __forceinline__ __device__ float3 normalized_view_direction(float3 direction) {
    float3 view = -direction;
    const float len_sq = squared_norm(view);
    if (!(len_sq > 1.0e-16f)) {
        return make_float3(0.0f, 0.0f, 1.0f);
    }
    return view * rsqrtf(len_sq);
}

static __forceinline__ __device__ float3 oriented_surfel_normal(int surfel, float3 direction) {
    const float3 tangent_u = make_float3(params.tangent_u_x[surfel],
                                         params.tangent_u_y[surfel],
                                         params.tangent_u_z[surfel]);
    const float3 tangent_v = make_float3(params.tangent_v_x[surfel],
                                         params.tangent_v_y[surfel],
                                         params.tangent_v_z[surfel]);
    const float3 raw_normal = cross3(tangent_u, tangent_v);
    const float normal_len_sq = squared_norm(raw_normal);
    if (!(normal_len_sq > 1.0e-16f)) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float3 normal = raw_normal * rsqrtf(normal_len_sq);
    if (params.face_forward != 0 && dot3(normal, direction) > 0.0f) {
        normal = -normal;
    }
    return normal;
}

static __forceinline__ __device__ float evaluate_appearance_channel(int surfel,
                                                                    int channel,
                                                                    float3 view) {
    if (params.appearance_values == nullptr || surfel < 0 || surfel >= params.surfel_count) {
        return channel == 0 ? 1.0f : 0.0f;
    }

    if (params.color_model == 1) {
        const int degree = params.sh_degree < 0 ? 0 : (params.sh_degree > 3 ? 3 : params.sh_degree);
        const int basis_count = (degree + 1) * (degree + 1);
        const int storage_degree = params.appearance_sh_degree < 0
            ? 0
            : (params.appearance_sh_degree > 3 ? 3 : params.appearance_sh_degree);
        const int storage_basis_count = (storage_degree + 1) * (storage_degree + 1);
        const int rgb_channel = channel < 3 ? channel : 0;
        float value = 0.0f;
        const int base = surfel * storage_basis_count * 3;
        for (int basis = 0; basis < basis_count; ++basis) {
            value += params.appearance_values[base + basis * 3 + rgb_channel] *
                     sh_basis(basis, view);
        }
        return value;
    }

    const int channel_count = params.appearance_channel_count > 0
        ? params.appearance_channel_count
        : 1;
    const int source_channel = channel < channel_count ? channel : channel_count - 1;
    return params.appearance_values[surfel * channel_count + source_channel];
}

static __forceinline__ __device__ void insert_composite_hit(unsigned int ray,
                                                           int surfel,
                                                           float t,
                                                           float alpha,
                                                           float value) {
    const int k = params.composite_hit_capacity;
    if (k <= 0 || params.scratch_surfel_id == nullptr ||
        params.scratch_t == nullptr ||
        params.scratch_alpha == nullptr ||
        params.scratch_value == nullptr) {
        return;
    }

    const int base = static_cast<int>(ray) * k;
    for (int slot = 0; slot < k; ++slot) {
        if (params.scratch_surfel_id[base + slot] == surfel) {
            return;
        }
    }

    const int tail = base + k - 1;
    const int tail_id = params.scratch_surfel_id[tail];
    const float tail_t = params.scratch_t[tail];
    const bool before_tail =
        tail_id < 0 ||
        t < tail_t - kSortEpsilon ||
        (fabsf(t - tail_t) <= kSortEpsilon && surfel < tail_id);
    if (!before_tail) {
        return;
    }

    int insert = k - 1;
    while (insert > 0) {
        const int prev = base + insert - 1;
        const int prev_id = params.scratch_surfel_id[prev];
        const float prev_t = params.scratch_t[prev];
        const bool before_prev =
            prev_id < 0 ||
            t < prev_t - kSortEpsilon ||
            (fabsf(t - prev_t) <= kSortEpsilon && surfel < prev_id);
        if (!before_prev) {
            break;
        }
        const int dst = base + insert;
        params.scratch_surfel_id[dst] = params.scratch_surfel_id[prev];
        params.scratch_t[dst] = params.scratch_t[prev];
        params.scratch_alpha[dst] = params.scratch_alpha[prev];
        params.scratch_value[dst] = params.scratch_value[prev];
        --insert;
    }

    const int out = base + insert;
    params.scratch_surfel_id[out] = surfel;
    params.scratch_t[out] = t;
    params.scratch_alpha[out] = alpha;
    params.scratch_value[out] = value;
}

static __forceinline__ __device__ int count_candidate_buffer_for_ray(unsigned int ray, int k) {
    int filled = 0;
    if (k <= 0 || params.scratch_surfel_id == nullptr) {
        return filled;
    }

    const int base = static_cast<int>(ray) * k;
    for (int slot = 0; slot < k; ++slot) {
        if (params.scratch_surfel_id[base + slot] >= 0) {
            ++filled;
        }
    }
    return filled;
}

static __forceinline__ __device__ void clear_candidate_buffer_for_ray(unsigned int ray, int k) {
    if (k <= 0 || params.scratch_surfel_id == nullptr ||
        params.scratch_t == nullptr ||
        params.scratch_alpha == nullptr ||
        params.scratch_value == nullptr) {
        return;
    }

    const int base = static_cast<int>(ray) * k;
    for (int slot = 0; slot < k; ++slot) {
        const int index = base + slot;
        params.scratch_surfel_id[index] = -1;
        params.scratch_t[index] = kInvalidT;
        params.scratch_alpha[index] = 0.0f;
        params.scratch_value[index] = 0.0f;
    }
}

static __forceinline__ __device__ void composite_candidate_buffer_for_ray(unsigned int ray,
                                                                          int k,
                                                                          int channel_count,
                                                                          int channel_base,
                                                                          float3 view,
                                                                          float3 direction,
                                                                          float &intensity,
                                                                          float &alpha_accum,
                                                                          float &transmittance,
                                                                          float &depth_numerator,
                                                                          float3 &normal_numerator) {
    if (k <= 0 || params.scratch_surfel_id == nullptr ||
        params.scratch_t == nullptr ||
        params.scratch_alpha == nullptr) {
        return;
    }

    const int base = static_cast<int>(ray) * k;
    for (int slot = 0; slot < k; ++slot) {
        const int index = base + slot;
        if (params.scratch_surfel_id[index] < 0) {
            continue;
        }
        const float hit_alpha = params.scratch_alpha[index];
        if (!(hit_alpha > 0.0f)) {
            continue;
        }
        const float contribution = transmittance * hit_alpha;
        const int surfel = params.scratch_surfel_id[index];
        const float first_channel = evaluate_appearance_channel(surfel, 0, view);
        intensity += contribution * first_channel;
        if (params.out_channels != nullptr) {
            for (int channel = 0; channel < channel_count; ++channel) {
                params.out_channels[channel_base + channel] +=
                    contribution * evaluate_appearance_channel(surfel, channel, view);
            }
        }
        if (params.output_normal != 0) {
            normal_numerator = normal_numerator + oriented_surfel_normal(surfel, direction) * contribution;
        }
        alpha_accum += contribution;
        depth_numerator += contribution * params.scratch_t[index];
        transmittance *= 1.0f - hit_alpha;
        if (!(transmittance > params.transmittance_min)) {
            break;
        }
    }
}

static __forceinline__ __device__ void consider_intersection_hit(unsigned int ray,
                                                                 int triangle_id,
                                                                 int surfel,
                                                                 float plane_t) {
    if (params.out_triangle_id == nullptr || params.out_proxy_t == nullptr) {
        return;
    }

    const int current_triangle = params.out_triangle_id[ray];
    const float current_t = params.out_proxy_t[ray];
    int current_surfel = -1;
    if (current_triangle >= 0 && current_triangle < params.triangle_count) {
        current_surfel = params.triangle_to_surfel_id[current_triangle];
    }

    const bool take =
        current_triangle < 0 ||
        plane_t < current_t - kSortEpsilon ||
        (fabsf(plane_t - current_t) <= kSortEpsilon &&
         (current_surfel < 0 ||
          surfel < current_surfel ||
          (surfel == current_surfel && triangle_id < current_triangle)));
    if (take) {
        params.out_triangle_id[ray] = triangle_id;
        params.out_proxy_t[ray] = plane_t;
    }
}

} // namespace

extern "C" __global__ void __miss__surfel_trace() {
}

extern "C" __global__ void __anyhit__surfel_composite() {
    const unsigned int ray = optixGetLaunchIndex().x;
    const unsigned int primitive = optixGetPrimitiveIndex();
    if (ray >= static_cast<unsigned int>(params.ray_count) ||
        primitive >= static_cast<unsigned int>(params.triangle_count) ||
        params.triangle_to_surfel_id == nullptr) {
        optixIgnoreIntersection();
        return;
    }

    const int surfel = params.triangle_to_surfel_id[primitive];
    float plane_t = 0.0f;
    float alpha = 0.0f;
    if (evaluate_gaussian_hit(optixGetWorldRayOrigin(),
                              optixGetWorldRayDirection(),
                              ray_tmax(ray),
                              surfel,
                              plane_t,
                              &alpha)) {
        const float value = params.value != nullptr ? params.value[surfel] : 1.0f;
        insert_composite_hit(ray, surfel, plane_t, alpha, value);
    }
    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__surfel_intersect() {
    const unsigned int ray = optixGetLaunchIndex().x;
    const unsigned int primitive = optixGetPrimitiveIndex();
    if (ray >= static_cast<unsigned int>(params.ray_count) ||
        primitive >= static_cast<unsigned int>(params.triangle_count) ||
        params.triangle_to_surfel_id == nullptr) {
        optixIgnoreIntersection();
        return;
    }

    const int triangle_id = static_cast<int>(primitive);
    const int surfel = params.triangle_to_surfel_id[triangle_id];
    float plane_t = 0.0f;
    if (evaluate_gaussian_hit(optixGetWorldRayOrigin(),
                              optixGetWorldRayDirection(),
                              ray_tmax(ray),
                              surfel,
                              plane_t)) {
        consider_intersection_hit(ray, triangle_id, surfel, plane_t);
    }
    optixIgnoreIntersection();
}

extern "C" __global__ void __raygen__surfel_trace() {
    const unsigned int ray = optixGetLaunchIndex().x;
    if (ray >= static_cast<unsigned int>(params.ray_count) ||
        !is_active(ray) ||
        params.handle == 0ull ||
        params.triangle_count <= 0 ||
        params.surfel_count <= 0 ||
        params.triangle_to_surfel_id == nullptr ||
        params.out_triangle_id == nullptr ||
        params.out_proxy_t == nullptr) {
        write_invalid(ray);
        return;
    }

    const float trace_tmax = ray_tmax(ray);
    if (!(fmaxf(params.ray_epsilon, 0.0f) < trace_tmax)) {
        write_invalid(ray);
        return;
    }

    params.out_triangle_id[ray] = -1;
    params.out_proxy_t[ray] = kInvalidT;
    if (params.out_valid != nullptr) {
        params.out_valid[ray] = 0u;
    }

    uint32_t dummy = 0u;
    optixTrace(static_cast<OptixTraversableHandle>(params.handle),
               load_ray_origin(ray),
               load_ray_direction(ray),
               fmaxf(params.ray_epsilon, 0.0f),
               trace_tmax,
               0.0f,
               255u,
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
               1,
               1,
               0,
               dummy);

    const bool valid =
        params.out_triangle_id[ray] >= 0 &&
        params.out_triangle_id[ray] < params.triangle_count &&
        isfinite(params.out_proxy_t[ray]) &&
        params.out_proxy_t[ray] < trace_tmax;
    if (params.out_valid != nullptr) {
        params.out_valid[ray] = valid ? 1u : 0u;
    }
    if (!valid) {
        params.out_triangle_id[ray] = -1;
        params.out_proxy_t[ray] = kInvalidT;
    }
}

extern "C" __global__ void __raygen__surfel_composite() {
    const unsigned int ray = optixGetLaunchIndex().x;
    const int k = params.composite_hit_capacity;
    if (ray >= static_cast<unsigned int>(params.ray_count) || k <= 0) {
        return;
    }

    if (params.out_candidate_count != nullptr) {
        params.out_candidate_count[ray] = 0;
    }
    if (params.out_candidate_buffer_full != nullptr) {
        params.out_candidate_buffer_full[ray] = 0u;
    }
    const int channel_count = params.render_channel_count > 0 ? params.render_channel_count : 0;
    const int channel_base = static_cast<int>(ray) * channel_count;
    if (params.out_channels != nullptr) {
        for (int channel = 0; channel < channel_count; ++channel) {
            params.out_channels[channel_base + channel] = 0.0f;
        }
    }

    const bool trace_enabled =
        is_active(ray) &&
        params.handle != 0ull &&
        params.triangle_count > 0 &&
        params.surfel_count > 0 &&
        params.triangle_to_surfel_id != nullptr;

    float intensity = 0.0f;
    float alpha_accum = 0.0f;
    float transmittance = 1.0f;
    float depth_numerator = 0.0f;
    float3 normal_numerator = make_float3(0.0f, 0.0f, 0.0f);
    const float3 ray_direction = load_ray_direction(ray);
    const float3 view = normalized_view_direction(ray_direction);

    if (trace_enabled) {
        const float trace_tmax = ray_tmax(ray);
        float t_start = fmaxf(params.ray_epsilon, 0.0f);
        const int segment_limit = params.continue_after_full_buffer != 0
            ? (params.max_trace_segments > 1 ? params.max_trace_segments : 1)
            : 1;
        int total_filled = 0;
        bool buffer_full = false;

        for (int segment = 0;
             segment < segment_limit &&
             transmittance > params.transmittance_min &&
             t_start < trace_tmax;
             ++segment) {
            clear_candidate_buffer_for_ray(ray, k);
            uint32_t dummy = 0u;
            optixTrace(static_cast<OptixTraversableHandle>(params.handle),
                       load_ray_origin(ray),
                       load_ray_direction(ray),
                       t_start,
                       trace_tmax,
                       0.0f,
                       255u,
                       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
                       0,
                       1,
                       0,
                       dummy);

            const int filled = count_candidate_buffer_for_ray(ray, k);
            total_filled += filled;
            buffer_full = buffer_full || (filled == k);
            if (params.out_candidate_count != nullptr) {
                params.out_candidate_count[ray] = total_filled;
            }
            if (params.out_candidate_buffer_full != nullptr) {
                params.out_candidate_buffer_full[ray] = buffer_full ? 1u : 0u;
            }

            composite_candidate_buffer_for_ray(ray,
                                               k,
                                               channel_count,
                                               channel_base,
                                               view,
                                               ray_direction,
                                               intensity,
                                               alpha_accum,
                                               transmittance,
                                               depth_numerator,
                                               normal_numerator);

            if (params.continue_after_full_buffer == 0 || filled < k) {
                break;
            }
            const int last_index = static_cast<int>(ray) * k + k - 1;
            const float last_t = params.scratch_t[last_index];
            if (!isfinite(last_t) || last_t <= t_start) {
                break;
            }
            t_start = last_t + fmaxf(params.ray_epsilon, 1.0e-5f);
        }
    }

    if (params.out_channels != nullptr && channel_count >= 3 && is_active(ray)) {
        params.out_channels[channel_base + 0] += transmittance * params.background_rgb[0];
        params.out_channels[channel_base + 1] += transmittance * params.background_rgb[1];
        params.out_channels[channel_base + 2] += transmittance * params.background_rgb[2];
    }

    params.out_intensity[ray] = intensity;
    params.out_alpha[ray] = alpha_accum;
    params.out_transmittance[ray] = transmittance;
    params.out_depth[ray] = alpha_accum > 0.0f ? depth_numerator / alpha_accum : infinity();
    if (params.output_normal != 0 &&
        params.out_normal_x != nullptr &&
        params.out_normal_y != nullptr &&
        params.out_normal_z != nullptr) {
        if (alpha_accum > 0.0f) {
            const float inv_alpha = 1.0f / alpha_accum;
            params.out_normal_x[ray] = normal_numerator.x * inv_alpha;
            params.out_normal_y[ray] = normal_numerator.y * inv_alpha;
            params.out_normal_z[ray] = normal_numerator.z * inv_alpha;
        } else {
            params.out_normal_x[ray] = 0.0f;
            params.out_normal_y[ray] = 0.0f;
            params.out_normal_z[ray] = 0.0f;
        }
    }
    if (params.out_valid != nullptr) {
        params.out_valid[ray] = alpha_accum > 0.0f ? 1u : 0u;
    }
}

} // namespace rayd
