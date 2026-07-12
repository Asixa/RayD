#pragma once

#include <cmath>
#include <type_traits>

#include <rayd/shared/contracts.h>

#if defined(__CUDACC__)
#define RAYD_SHARED_FIELD_INLINE __host__ __device__ __forceinline__
#else
#define RAYD_SHARED_FIELD_INLINE inline
#endif

namespace rayd::shared::field {

struct Complex {
    float r;
    float i;
};

struct Complex3 {
    Complex x;
    Complex y;
    Complex z;
};

RAYD_SHARED_FIELD_INLINE Complex c_make(float r, float i = 0.f) {
    return {r, i};
}

RAYD_SHARED_FIELD_INLINE Complex c_add(Complex a, Complex b) {
    return c_make(a.r + b.r, a.i + b.i);
}

RAYD_SHARED_FIELD_INLINE Complex c_sub(Complex a, Complex b) {
    return c_make(a.r - b.r, a.i - b.i);
}

RAYD_SHARED_FIELD_INLINE Complex c_mul(Complex a, Complex b) {
    return c_make(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

RAYD_SHARED_FIELD_INLINE Complex c_scale(Complex a, float scale) {
    return c_make(a.r * scale, a.i * scale);
}

RAYD_SHARED_FIELD_INLINE Complex c_mul_real(Complex a, float scale) {
    return c_scale(a, scale);
}

RAYD_SHARED_FIELD_INLINE Complex c_div(Complex a, Complex b) {
    const float denominator = fmaxf(b.r * b.r + b.i * b.i, 1.0e-20f);
    return c_make((a.r * b.r + a.i * b.i) / denominator,
                  (a.i * b.r - a.r * b.i) / denominator);
}

RAYD_SHARED_FIELD_INLINE float c_abs2(Complex value) {
    return value.r * value.r + value.i * value.i;
}

RAYD_SHARED_FIELD_INLINE Complex c_sqrt(Complex value) {
    const float radius = hypotf(value.r, value.i);
    if (radius <= 0.f) {
        return c_make(0.f, 0.f);
    }
    const float real_magnitude = sqrtf(fmaxf(0.f, 0.5f * (radius + value.r)));
    const float imag_magnitude = sqrtf(fmaxf(0.f, 0.5f * (radius - value.r)));
    return c_make(real_magnitude, copysignf(imag_magnitude, value.i));
}

RAYD_SHARED_FIELD_INLINE Complex c_exp_neg_i(float phase) {
    float sine;
    float cosine;
#if defined(__CUDA_ARCH__)
    sincosf(phase, &sine, &cosine);
#else
    sine = std::sin(phase);
    cosine = std::cos(phase);
#endif
    return c_make(cosine, -sine);
}

RAYD_SHARED_FIELD_INLINE Complex c_exp_neg_i_product(float lhs, float rhs) {
    constexpr double TwoPi = 6.283185307179586476925286766559;
#if defined(__CUDA_ARCH__)
    const double reduced = fmod(static_cast<double>(lhs) * static_cast<double>(rhs), TwoPi);
#else
    const double reduced = std::fmod(static_cast<double>(lhs) * static_cast<double>(rhs), TwoPi);
#endif
    return c_exp_neg_i(static_cast<float>(reduced));
}

RAYD_SHARED_FIELD_INLINE Complex3 c3_zero() {
    return {c_make(0.f), c_make(0.f), c_make(0.f)};
}

template <typename Vec3>
RAYD_SHARED_FIELD_INLINE Complex3 c3_from_real(Vec3 value) {
    return {c_make(value.x), c_make(value.y), c_make(value.z)};
}

RAYD_SHARED_FIELD_INLINE Complex3 c3_add(Complex3 a, Complex3 b) {
    return {c_add(a.x, b.x), c_add(a.y, b.y), c_add(a.z, b.z)};
}

template <typename Vec3>
RAYD_SHARED_FIELD_INLINE Complex3 c3_scale_complex(Vec3 basis, Complex coefficient) {
    return {c_mul_real(coefficient, basis.x),
            c_mul_real(coefficient, basis.y),
            c_mul_real(coefficient, basis.z)};
}

RAYD_SHARED_FIELD_INLINE Complex3 c3_mul_complex(Complex3 value, Complex coefficient) {
    return {c_mul(value.x, coefficient),
            c_mul(value.y, coefficient),
            c_mul(value.z, coefficient)};
}

template <typename Vec3>
RAYD_SHARED_FIELD_INLINE Complex c3_dot_real(Complex3 value, Vec3 basis) {
    return c_add(c_add(c_mul_real(value.x, basis.x),
                       c_mul_real(value.y, basis.y)),
                 c_mul_real(value.z, basis.z));
}

RAYD_SHARED_FIELD_INLINE float c3_power(Complex3 value) {
    return c_abs2(value.x) + c_abs2(value.y) + c_abs2(value.z);
}

RAYD_SHARED_FIELD_INLINE bool finite_complex3(Complex3 value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value.x.r) && isfinite(value.x.i) &&
           isfinite(value.y.r) && isfinite(value.y.i) &&
           isfinite(value.z.r) && isfinite(value.z.i);
#else
    return std::isfinite(value.x.r) && std::isfinite(value.x.i) &&
           std::isfinite(value.y.r) && std::isfinite(value.y.i) &&
           std::isfinite(value.z.r) && std::isfinite(value.z.i);
#endif
}

RAYD_SHARED_FIELD_INLINE bool fresnel_reflection_coefficients(
    float eta_r_value,
    float sigma_value,
    float mu_r_value,
    float gain,
    float omega_value,
    float cos_theta,
    Complex &r_te,
    Complex &r_tm,
    float epsilon = SmallEpsilon) {
    const float eta_r = fmaxf(eta_r_value, epsilon);
    const float sigma = fmaxf(sigma_value, 0.f);
    const float mu_r = fmaxf(mu_r_value, epsilon);
    const float omega = fmaxf(omega_value, epsilon);
    const Complex eta = c_make(eta_r, -sigma / (omega * VacuumPermittivity));
    const Complex mu = c_make(mu_r);
    const float cosine = fminf(fmaxf(fabsf(cos_theta), epsilon), 1.f);
    const float sine_squared = fmaxf(0.f, 1.f - cosine * cosine);
    const Complex root = c_sqrt(c_sub(c_mul(mu, eta), c_make(sine_squared)));
    const Complex mu_cosine = c_make(mu_r * cosine);
    const Complex eta_cosine = c_make(eta.r * cosine, eta.i * cosine);
    r_te = c_scale(c_div(c_sub(mu_cosine, root), c_add(mu_cosine, root)), gain);
    r_tm = c_scale(c_div(c_sub(eta_cosine, root), c_add(eta_cosine, root)), gain);
#if defined(__CUDA_ARCH__)
    if (!isfinite(r_te.r) || !isfinite(r_te.i))
#else
    if (!std::isfinite(r_te.r) || !std::isfinite(r_te.i))
#endif
        r_te = c_make(0.f);
#if defined(__CUDA_ARCH__)
    if (!isfinite(r_tm.r) || !isfinite(r_tm.i))
#else
    if (!std::isfinite(r_tm.r) || !std::isfinite(r_tm.i))
#endif
        r_tm = c_make(0.f);
    return c_abs2(r_te) > 0.f || c_abs2(r_tm) > 0.f;
}

RAYD_SHARED_FIELD_INLINE float free_space_amplitude(
    float wavelength,
    float distance,
    float epsilon = SmallEpsilon) {
    constexpr float FourPi = 12.56637061435917295385f;
    return wavelength / (FourPi * fmaxf(distance, epsilon));
}

RAYD_SHARED_FIELD_INLINE Complex propagation_phase(float wave_number, float distance) {
    return c_exp_neg_i_product(wave_number, distance);
}

static_assert(std::is_standard_layout_v<Complex>);
static_assert(std::is_trivially_copyable_v<Complex>);
static_assert(std::is_standard_layout_v<Complex3>);
static_assert(std::is_trivially_copyable_v<Complex3>);

} // namespace rayd::shared::field

#undef RAYD_SHARED_FIELD_INLINE
