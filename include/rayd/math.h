// Copyright Xingyu Chen.
// Defines all reusable math types and functions.

#pragma once

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <rayd/contracts.h>
#include <rayd/rt/numeric_policy.h>
#include <rayd/rt/qualifiers.h>

#define RAYD_MATH_INLINE RAYD_HOST_DEVICE

namespace rayd::shared::math {

template <typename T> struct Vec3 {
    T x;
    T y;
    T z;
};

using Vec3f = Vec3<float>;

RAYD_MATH_INLINE constexpr Vec3f make_vec3(float x, float y, float z) {
    return {x, y, z};
}

template <typename V> RAYD_MATH_INLINE constexpr V add(V a, V b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename V> RAYD_MATH_INLINE constexpr V subtract(V a, V b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename V> RAYD_MATH_INLINE constexpr V negate(V value) {
    return {-value.x, -value.y, -value.z};
}

template <typename V, typename S> RAYD_MATH_INLINE constexpr V scale(V value, S factor) {
    return {value.x * factor, value.y * factor, value.z * factor};
}

template <typename V, typename S> RAYD_MATH_INLINE constexpr V divide(V value, S divisor) {
    return {value.x / divisor, value.y / divisor, value.z / divisor};
}

template <typename V> RAYD_MATH_INLINE constexpr auto dot(V a, V b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename V> RAYD_MATH_INLINE constexpr auto squared_norm(V value) {
    return dot(value, value);
}

RAYD_MATH_INLINE float length_f32(Vec3f value) {
    return sqrtf(fmaxf(dot(value, value), 0.0f));
}

RAYD_MATH_INLINE Vec3f normalize_f32(Vec3f value, float minimum_squared_norm = 1.0e-12f) {
    const float squared = fmaxf(dot(value, value), minimum_squared_norm);
#if defined(__CUDA_ARCH__)
    const float inverse_length = rsqrtf(squared);
#else
    const float inverse_length = 1.0f / sqrtf(squared);
#endif
    return scale(value, inverse_length);
}
template <typename V> RAYD_MATH_INLINE constexpr V cross(V a, V b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

template <typename T> RAYD_MATH_INLINE Vec3<T> load_vec3(const T* values, std::int64_t index) {
    const std::int64_t base = index * 3;
    return {values[base], values[base + 1], values[base + 2]};
}

RAYD_MATH_INLINE Vec3f component_min(Vec3f a, Vec3f b) {
    return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}

RAYD_MATH_INLINE Vec3f component_max(Vec3f a, Vec3f b) {
    return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

template <typename T> RAYD_MATH_INLINE T length(Vec3<T> value) {
    const T squared = dot(value, value);
    return sqrt(squared > T(0) ? squared : T(0));
}

template <typename T>
RAYD_MATH_INLINE Vec3<T> safe_normalize(Vec3<T> value, Vec3<T> alternate, T threshold, T epsilon) {
    const T norm = length(value);
    if (norm > threshold)
        return scale(value, T(1) / (norm + epsilon));
    const T alternate_norm = length(alternate);
    return scale(alternate, T(1) / (alternate_norm + epsilon));
}

template <typename T>
RAYD_MATH_INLINE Vec3<T> stable_perpendicular_basis(Vec3<T> direction, Vec3<T> preferred, T threshold, T epsilon) {
    const Vec3<T> projected = subtract(preferred, scale(direction, dot(preferred, direction)));
    const Vec3<T> alternate_axis = fabs(direction.z) < T(0.9) ? Vec3<T>{T(0), T(0), T(1)} : Vec3<T>{T(0), T(1), T(0)};
    const Vec3<T> alternate_projected = subtract(alternate_axis, scale(direction, dot(alternate_axis, direction)));
    return safe_normalize(projected, alternate_projected, threshold, epsilon);
}

template <typename T> RAYD_MATH_INLINE Vec3<T> transverse_project(Vec3<T> direction, Vec3<T> preferred) {
    return subtract(preferred, scale(direction, dot(preferred, direction)));
}
template <typename T> struct Complex {
    T re;
    T im;
};

template <typename T> struct Complex3 {
    Complex<T> x;
    Complex<T> y;
    Complex<T> z;
};

using Complexf = Complex<float>;
using Complex3f = Complex3<float>;

template <typename T> RAYD_MATH_INLINE constexpr Complex<T> make_complex(T re, T im) {
    return {re, im};
}

template <typename T> RAYD_MATH_INLINE constexpr Complex<T> complex_add(Complex<T> a, Complex<T> b) {
    return {a.re + b.re, a.im + b.im};
}

template <typename T> RAYD_MATH_INLINE constexpr Complex<T> complex_subtract(Complex<T> a, Complex<T> b) {
    return {a.re - b.re, a.im - b.im};
}

template <typename T> RAYD_MATH_INLINE constexpr Complex<T> complex_multiply(Complex<T> a, Complex<T> b) {
    return {
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    };
}

template <typename T, typename S> RAYD_MATH_INLINE constexpr Complex<T> complex_scale(Complex<T> value, S factor) {
    return {value.re * factor, value.im * factor};
}

template <typename T> RAYD_MATH_INLINE constexpr T complex_abs_squared(Complex<T> value) {
    return value.re * value.re + value.im * value.im;
}

struct Mat3f {
    float m[3][3];
};

struct Quaternionf {
    float w;
    float x;
    float y;
    float z;
};

RAYD_MATH_INLINE Mat3f zero_mat3() {
    Mat3f out{};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            out.m[row][column] = 0.0f;
    return out;
}

RAYD_MATH_INLINE void add_outer(Mat3f& acc, Vec3f a, Vec3f b, float factor) {
    const float left[3] = {a.x, a.y, a.z};
    const float right[3] = {b.x, b.y, b.z};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            acc.m[row][column] += factor * left[row] * right[column];
}

RAYD_MATH_INLINE float contract(const Mat3f& a, const Mat3f& b) {
    float total = 0.0f;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            total += a.m[row][column] * b.m[row][column];
    return total;
}

RAYD_MATH_INLINE Vec3f transpose_multiply(const Mat3f& matrix, Vec3f value) {
    return {
        matrix.m[0][0] * value.x + matrix.m[1][0] * value.y + matrix.m[2][0] * value.z,
        matrix.m[0][1] * value.x + matrix.m[1][1] * value.y + matrix.m[2][1] * value.z,
        matrix.m[0][2] * value.x + matrix.m[1][2] * value.y + matrix.m[2][2] * value.z,
    };
}

RAYD_MATH_INLINE Vec3f multiply(const Mat3f& matrix, Vec3f value) {
    return {
        matrix.m[0][0] * value.x + matrix.m[0][1] * value.y + matrix.m[0][2] * value.z,
        matrix.m[1][0] * value.x + matrix.m[1][1] * value.y + matrix.m[1][2] * value.z,
        matrix.m[2][0] * value.x + matrix.m[2][1] * value.y + matrix.m[2][2] * value.z,
    };
}

RAYD_MATH_INLINE float quaternion_dot(Quaternionf a, Quaternionf b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

RAYD_MATH_INLINE Quaternionf quaternion_scale(Quaternionf value, float factor) {
    return {
        value.w * factor,
        value.x * factor,
        value.y * factor,
        value.z * factor,
    };
}

RAYD_MATH_INLINE Quaternionf quaternion_subtract(Quaternionf a, Quaternionf b) {
    return {a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z};
}

static_assert(std::is_standard_layout_v<Vec3f>);
static_assert(std::is_trivially_copyable_v<Vec3f>);
static_assert(std::is_standard_layout_v<Complexf>);
static_assert(std::is_trivially_copyable_v<Complexf>);
static_assert(std::is_standard_layout_v<Complex3f>);
static_assert(std::is_trivially_copyable_v<Complex3f>);

} // namespace rayd::shared::math

namespace rayd::shared::diffraction {

struct Dual {
    float v;
    float d;

    Dual() = default;
    RAYD_MATH_INLINE constexpr Dual(float value) : v(value), d(0.0f) {}
    RAYD_MATH_INLINE constexpr Dual(float value, float tangent) : v(value), d(tangent) {}
};

RAYD_MATH_INLINE constexpr Dual operator+(Dual a, Dual b) {
    return {a.v + b.v, a.d + b.d};
}
RAYD_MATH_INLINE constexpr Dual operator-(Dual a, Dual b) {
    return {a.v - b.v, a.d - b.d};
}
RAYD_MATH_INLINE constexpr Dual operator-(Dual a) {
    return {-a.v, -a.d};
}
RAYD_MATH_INLINE constexpr Dual operator*(Dual a, Dual b) {
    return {a.v * b.v, a.d * b.v + a.v * b.d};
}
RAYD_MATH_INLINE Dual operator/(Dual a, Dual b) {
    const float inverse = 1.0f / b.v;
    const float quotient = a.v * inverse;
    return {quotient, (a.d - quotient * b.d) * inverse};
}
RAYD_MATH_INLINE Dual& operator+=(Dual& a, Dual b) {
    a = a + b;
    return a;
}
RAYD_MATH_INLINE Dual& operator-=(Dual& a, Dual b) {
    a = a - b;
    return a;
}
RAYD_MATH_INLINE Dual& operator*=(Dual& a, Dual b) {
    a = a * b;
    return a;
}
RAYD_MATH_INLINE Dual& operator/=(Dual& a, Dual b) {
    a = a / b;
    return a;
}
RAYD_MATH_INLINE constexpr bool operator<(Dual a, Dual b) {
    return a.v < b.v;
}
RAYD_MATH_INLINE constexpr bool operator>(Dual a, Dual b) {
    return a.v > b.v;
}
RAYD_MATH_INLINE constexpr bool operator<=(Dual a, Dual b) {
    return a.v <= b.v;
}
RAYD_MATH_INLINE constexpr bool operator>=(Dual a, Dual b) {
    return a.v >= b.v;
}
RAYD_MATH_INLINE constexpr bool operator==(Dual a, Dual b) {
    return a.v == b.v;
}
RAYD_MATH_INLINE constexpr bool operator!=(Dual a, Dual b) {
    return a.v != b.v;
}

RAYD_MATH_INLINE float sqrtf(float value) {
    return ::sqrtf(value);
}
RAYD_MATH_INLINE float fmaxf(float a, float b) {
    return ::fmaxf(a, b);
}
RAYD_MATH_INLINE float fminf(float a, float b) {
    return ::fminf(a, b);
}
RAYD_MATH_INLINE float fabsf(float value) {
    return ::fabsf(value);
}
RAYD_MATH_INLINE float sinf(float value) {
    return ::sinf(value);
}
RAYD_MATH_INLINE float cosf(float value) {
    return ::cosf(value);
}
RAYD_MATH_INLINE float expf(float value) {
    return ::expf(value);
}
RAYD_MATH_INLINE float atan2f(float y, float x) {
    return ::atan2f(y, x);
}
RAYD_MATH_INLINE float roundf(float value) {
    return ::roundf(value);
}
RAYD_MATH_INLINE float floorf(float value) {
    return ::floorf(value);
}
RAYD_MATH_INLINE bool isfinite(float value) {
    return ::isfinite(value);
}
RAYD_MATH_INLINE void sincosf(float value, float* sine, float* cosine) {
#if defined(__CUDA_ARCH__)
    ::sincosf(value, sine, cosine);
#else
    *sine = ::sinf(value);
    *cosine = ::cosf(value);
#endif
}
RAYD_MATH_INLINE Dual sqrtf(Dual value) {
    const float root = ::sqrtf(value.v);
    return {root, value.v > 0.0f ? 0.5f * value.d / root : 0.0f};
}
RAYD_MATH_INLINE Dual fmaxf(Dual a, Dual b) {
    return a.v >= b.v ? a : b;
}
RAYD_MATH_INLINE Dual fminf(Dual a, Dual b) {
    return a.v <= b.v ? a : b;
}
RAYD_MATH_INLINE Dual fabsf(Dual value) {
    if (value.v > 0.0f)
        return value;
    if (value.v < 0.0f)
        return {-value.v, -value.d};
    return {0.0f, 0.0f};
}
RAYD_MATH_INLINE Dual sinf(Dual value) {
    return {::sinf(value.v), ::cosf(value.v) * value.d};
}
RAYD_MATH_INLINE Dual cosf(Dual value) {
    return {::cosf(value.v), -::sinf(value.v) * value.d};
}
RAYD_MATH_INLINE Dual expf(Dual value) {
    const float exponential = ::expf(value.v);
    return {exponential, exponential * value.d};
}
RAYD_MATH_INLINE Dual atan2f(Dual y, Dual x) {
    const float denominator = x.v * x.v + y.v * y.v;
    const float slope = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    return {
        ::atan2f(y.v, x.v),
        (x.v * y.d - y.v * x.d) * slope,
    };
}
RAYD_MATH_INLINE Dual roundf(Dual value) {
    return {::roundf(value.v), 0.0f};
}
RAYD_MATH_INLINE Dual floorf(Dual value) {
    return {::floorf(value.v), 0.0f};
}
RAYD_MATH_INLINE bool isfinite(Dual value) {
    return ::isfinite(value.v);
}
RAYD_MATH_INLINE void sincosf(Dual value, Dual* sine, Dual* cosine) {
    float sine_value;
    float cosine_value;
#if defined(__CUDA_ARCH__)
    ::sincosf(value.v, &sine_value, &cosine_value);
#else
    sine_value = ::sinf(value.v);
    cosine_value = ::cosf(value.v);
#endif
    *sine = {sine_value, cosine_value * value.d};
    *cosine = {cosine_value, -sine_value * value.d};
}

RAYD_MATH_INLINE constexpr float scalar_value(float value) {
    return value;
}
RAYD_MATH_INLINE constexpr float scalar_value(Dual value) {
    return value.v;
}
RAYD_MATH_INLINE constexpr float scalar_tangent(float) {
    return 0.0f;
}
RAYD_MATH_INLINE constexpr float scalar_tangent(Dual value) {
    return value.d;
}

template <typename T> using Vec3T = math::Vec3<T>;
using float3a = math::Vec3f;

template <typename T> using ComplexT = math::Complex<T>;
using Complex = math::Complexf;

template <typename T> using Complex3T = math::Complex3<T>;
using Complex3 = math::Complex3f;

RAYD_MATH_INLINE float3a make_f3(float x, float y, float z) {
    return math::make_vec3(x, y, z);
}

template <typename T> RAYD_MATH_INLINE Vec3T<T> v3_const(float x, float y, float z) {
    return {T(x), T(y), T(z)};
}

template <typename T = float> RAYD_MATH_INLINE Vec3T<T> f3_zero() {
    return {T(0.0f), T(0.0f), T(0.0f)};
}

template <typename T> RAYD_MATH_INLINE Vec3T<T> f3_add(Vec3T<T> a, Vec3T<T> b) {
    return math::add(a, b);
}

template <typename T> RAYD_MATH_INLINE Vec3T<T> f3_sub(Vec3T<T> a, Vec3T<T> b) {
    return math::subtract(a, b);
}

template <typename T, typename S> RAYD_MATH_INLINE Vec3T<T> f3_mul(Vec3T<T> value, S factor) {
    const T scale = T(factor);
    return {value.x * scale, value.y * scale, value.z * scale};
}

template <typename T> RAYD_MATH_INLINE Vec3T<T> f3_neg(Vec3T<T> value) {
    return math::negate(value);
}

template <typename T> RAYD_MATH_INLINE T f3_dot(Vec3T<T> a, Vec3T<T> b) {
    return math::dot(a, b);
}

template <typename T> RAYD_MATH_INLINE Vec3T<T> f3_cross(Vec3T<T> a, Vec3T<T> b) {
    return math::cross(a, b);
}

template <typename T> RAYD_MATH_INLINE T f3_len(Vec3T<T> value) {
    return sqrtf(fmaxf(f3_dot(value, value), T(0.0f)));
}

template <typename T, typename S> RAYD_MATH_INLINE Vec3T<T> f3_div(Vec3T<T> value, S divisor) {
    const T scale = T(divisor);
    return {value.x / scale, value.y / scale, value.z / scale};
}

template <typename T, typename = std::enable_if_t<!std::is_arithmetic_v<T>>>
RAYD_MATH_INLINE ComplexT<T> cplx(T re, T im) {
    return math::make_complex(re, im);
}

RAYD_MATH_INLINE Complex cplx(float re, float im) {
    return math::make_complex(re, im);
}

template <typename T> RAYD_MATH_INLINE ComplexT<T> c_const(float re, float im) {
    return {T(re), T(im)};
}

template <typename T = float> RAYD_MATH_INLINE ComplexT<T> cplx_zero() {
    return {T(0.0f), T(0.0f)};
}

template <typename T> RAYD_MATH_INLINE ComplexT<T> cplx_add(ComplexT<T> a, ComplexT<T> b) {
    return math::complex_add(a, b);
}

template <typename T> RAYD_MATH_INLINE ComplexT<T> cplx_sub(ComplexT<T> a, ComplexT<T> b) {
    return math::complex_subtract(a, b);
}

template <typename T> RAYD_MATH_INLINE ComplexT<T> cplx_mul(ComplexT<T> a, ComplexT<T> b) {
    return math::complex_multiply(a, b);
}

template <typename T, typename S> RAYD_MATH_INLINE ComplexT<T> cplx_mul_real(ComplexT<T> value, S factor) {
    const T scale = T(factor);
    return {value.re * scale, value.im * scale};
}

template <typename T, typename S> RAYD_MATH_INLINE ComplexT<T> cplx_div_real(ComplexT<T> value, S divisor) {
    const T scale = T(divisor);
    return {value.re / scale, value.im / scale};
}

template <typename T> RAYD_MATH_INLINE ComplexT<T> cplx_div(ComplexT<T> a, ComplexT<T> b) {
    const T denominator = b.re * b.re + b.im * b.im + T(1.0e-10f);
    return {
        (a.re * b.re + a.im * b.im) / denominator,
        (a.im * b.re - a.re * b.im) / denominator,
    };
}

template <typename T> RAYD_MATH_INLINE ComplexT<T> cplx_conj(ComplexT<T> value) {
    return {value.re, -value.im};
}

template <typename T> RAYD_MATH_INLINE ComplexT<T> cplx_exp_phase(T phase) {
    T sine;
    T cosine;
    sincosf(phase, &sine, &cosine);
    return {cosine, sine};
}

template <typename T> RAYD_MATH_INLINE T cplx_abs_sqr(ComplexT<T> value) {
    return math::complex_abs_squared(value);
}

RAYD_MATH_INLINE float cplx_adj_dot(Complex gradient, Complex value) {
    return gradient.re * value.re + gradient.im * value.im;
}

RAYD_MATH_INLINE bool cplx_any_nonzero(Complex value) {
    return fabsf(value.re) > 0.0f || fabsf(value.im) > 0.0f;
}

template <typename T = float> RAYD_MATH_INLINE Complex3T<T> c3_zero() {
    return {cplx_zero<T>(), cplx_zero<T>(), cplx_zero<T>()};
}

template <typename T> RAYD_MATH_INLINE Complex3T<T> c3_add(Complex3T<T> a, Complex3T<T> b) {
    return {cplx_add(a.x, b.x), cplx_add(a.y, b.y), cplx_add(a.z, b.z)};
}

template <typename T> RAYD_MATH_INLINE Complex3T<T> c3_scale(Complex3T<T> value, ComplexT<T> factor) {
    return {
        cplx_mul(value.x, factor),
        cplx_mul(value.y, factor),
        cplx_mul(value.z, factor),
    };
}

template <typename T> RAYD_MATH_INLINE ComplexT<T> cplx_dot_real(Complex3T<T> value, Vec3T<T> basis) {
    ComplexT<T> sum = cplx_zero<T>();
    sum = cplx_add(sum, cplx_mul_real(value.x, basis.x));
    sum = cplx_add(sum, cplx_mul_real(value.y, basis.y));
    sum = cplx_add(sum, cplx_mul_real(value.z, basis.z));
    return sum;
}

template <typename T> RAYD_MATH_INLINE Complex3T<T> cplx_scale_real(Vec3T<T> basis, ComplexT<T> factor) {
    return {
        cplx_mul_real(factor, basis.x),
        cplx_mul_real(factor, basis.y),
        cplx_mul_real(factor, basis.z),
    };
}

RAYD_MATH_INLINE bool c3_grad_any_nonzero(Complex3 value) {
    return cplx_any_nonzero(value.x) || cplx_any_nonzero(value.y) || cplx_any_nonzero(value.z);
}

} // namespace rayd::shared::diffraction

namespace rayd::shared::field {

using Complex = math::Complexf;
using Complex3 = math::Complex3f;

RAYD_MATH_INLINE Complex c_make(float re, float im = 0.0f) {
    return math::make_complex(re, im);
}

RAYD_MATH_INLINE Complex c_add(Complex a, Complex b) {
    return math::complex_add(a, b);
}

RAYD_MATH_INLINE Complex c_sub(Complex a, Complex b) {
    return math::complex_subtract(a, b);
}

RAYD_MATH_INLINE Complex c_mul(Complex a, Complex b) {
    return math::complex_multiply(a, b);
}

RAYD_MATH_INLINE Complex c_scale(Complex value, float factor) {
    return math::complex_scale(value, factor);
}

RAYD_MATH_INLINE Complex c_mul_real(Complex value, float factor) {
    return c_scale(value, factor);
}

RAYD_MATH_INLINE Complex c_div(Complex a, Complex b) {
    const float denominator = fmaxf(b.re * b.re + b.im * b.im, 1.0e-20f);
    return {
        (a.re * b.re + a.im * b.im) / denominator,
        (a.im * b.re - a.re * b.im) / denominator,
    };
}

RAYD_MATH_INLINE float c_abs2(Complex value) {
    return math::complex_abs_squared(value);
}

RAYD_MATH_INLINE Complex c_sqrt(Complex value) {
    const float radius = hypotf(value.re, value.im);
    if (radius <= 0.0f)
        return c_make(0.0f);
    const float real_magnitude = sqrtf(fmaxf(0.0f, 0.5f * (radius + value.re)));
    const float imag_magnitude = sqrtf(fmaxf(0.0f, 0.5f * (radius - value.re)));
    return c_make(real_magnitude, copysignf(imag_magnitude, value.im));
}

RAYD_MATH_INLINE Complex c_exp_neg_i(float phase) {
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

RAYD_MATH_INLINE Complex c_exp_neg_i_product(float lhs, float rhs) {
    constexpr double TwoPi = 6.283185307179586476925286766559;
#if defined(__CUDA_ARCH__)
    const double reduced = fmod(static_cast<double>(lhs) * static_cast<double>(rhs), TwoPi);
#else
    const double reduced = std::fmod(static_cast<double>(lhs) * static_cast<double>(rhs), TwoPi);
#endif
    return c_exp_neg_i(static_cast<float>(reduced));
}

RAYD_MATH_INLINE Complex3 c3_zero() {
    return {c_make(0.0f), c_make(0.0f), c_make(0.0f)};
}

template <typename Vec3> RAYD_MATH_INLINE Complex3 c3_from_real(Vec3 value) {
    return {c_make(value.x), c_make(value.y), c_make(value.z)};
}

RAYD_MATH_INLINE Complex3 c3_add(Complex3 a, Complex3 b) {
    return {c_add(a.x, b.x), c_add(a.y, b.y), c_add(a.z, b.z)};
}

template <typename Vec3> RAYD_MATH_INLINE Complex3 c3_scale_complex(Vec3 basis, Complex coefficient) {
    return {
        c_mul_real(coefficient, basis.x),
        c_mul_real(coefficient, basis.y),
        c_mul_real(coefficient, basis.z),
    };
}

RAYD_MATH_INLINE Complex3 c3_mul_complex(Complex3 value, Complex coefficient) {
    return {
        c_mul(value.x, coefficient),
        c_mul(value.y, coefficient),
        c_mul(value.z, coefficient),
    };
}

template <typename Vec3> RAYD_MATH_INLINE Complex c3_dot_real(Complex3 value, Vec3 basis) {
    return c_add(c_add(c_mul_real(value.x, basis.x), c_mul_real(value.y, basis.y)), c_mul_real(value.z, basis.z));
}

RAYD_MATH_INLINE float c3_power(Complex3 value) {
    return c_abs2(value.x) + c_abs2(value.y) + c_abs2(value.z);
}

RAYD_MATH_INLINE bool finite_complex3(Complex3 value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value.x.re) && isfinite(value.x.im) && isfinite(value.y.re) && isfinite(value.y.im) &&
           isfinite(value.z.re) && isfinite(value.z.im);
#else
    return std::isfinite(value.x.re) && std::isfinite(value.x.im) && std::isfinite(value.y.re) &&
           std::isfinite(value.y.im) && std::isfinite(value.z.re) && std::isfinite(value.z.im);
#endif
}

RAYD_MATH_INLINE bool fresnel_reflection_coefficients(float eta_r_value, float sigma_value, float mu_r_value,
                                                      float gain, float omega_value, float cos_theta, Complex& r_te,
                                                      Complex& r_tm, float epsilon = SmallEpsilon) {
    const float eta_r = fmaxf(eta_r_value, epsilon);
    const float sigma = fmaxf(sigma_value, 0.0f);
    const float mu_r = fmaxf(mu_r_value, epsilon);
    const float omega = fmaxf(omega_value, epsilon);
    const Complex eta = c_make(eta_r, -sigma / (omega * VacuumPermittivity));
    const Complex mu = c_make(mu_r);
    const float cosine = fminf(fmaxf(fabsf(cos_theta), epsilon), 1.0f);
    const float sine_squared = fmaxf(0.0f, 1.0f - cosine * cosine);
    const Complex root = c_sqrt(c_sub(c_mul(mu, eta), c_make(sine_squared)));
    const Complex mu_cosine = c_make(mu_r * cosine);
    const Complex eta_cosine = c_make(eta.re * cosine, eta.im * cosine);
    r_te = c_scale(c_div(c_sub(mu_cosine, root), c_add(mu_cosine, root)), gain);
    r_tm = c_scale(c_div(c_sub(eta_cosine, root), c_add(eta_cosine, root)), gain);
#if defined(__CUDA_ARCH__)
    if (!isfinite(r_te.re) || !isfinite(r_te.im))
#else
    if (!std::isfinite(r_te.re) || !std::isfinite(r_te.im))
#endif
        r_te = c_make(0.0f);
#if defined(__CUDA_ARCH__)
    if (!isfinite(r_tm.re) || !isfinite(r_tm.im))
#else
    if (!std::isfinite(r_tm.re) || !std::isfinite(r_tm.im))
#endif
        r_tm = c_make(0.0f);
    return c_abs2(r_te) > 0.0f || c_abs2(r_tm) > 0.0f;
}

RAYD_MATH_INLINE float free_space_amplitude(float wavelength, float distance, float epsilon = SmallEpsilon) {
    constexpr float FourPi = 12.56637061435917295385f;
    return wavelength / (FourPi * fmaxf(distance, epsilon));
}

RAYD_MATH_INLINE Complex propagation_phase(float wave_number, float distance) {
    return c_exp_neg_i_product(wave_number, distance);
}

} // namespace rayd::shared::field

namespace rayd::shared::field_transport {

using LegacySlabComplex = math::Complexf;

RAYD_MATH_INLINE LegacySlabComplex legacy_add(LegacySlabComplex a, LegacySlabComplex b) {
    return math::complex_add(a, b);
}

RAYD_MATH_INLINE LegacySlabComplex legacy_sub(LegacySlabComplex a, LegacySlabComplex b) {
    return math::complex_subtract(a, b);
}

RAYD_MATH_INLINE LegacySlabComplex legacy_mul(LegacySlabComplex a, LegacySlabComplex b) {
    return math::complex_multiply(a, b);
}

RAYD_MATH_INLINE LegacySlabComplex legacy_scale(LegacySlabComplex value, float factor) {
    return math::complex_scale(value, factor);
}

RAYD_MATH_INLINE LegacySlabComplex legacy_div_floor(LegacySlabComplex a, LegacySlabComplex b, float denominator_floor) {
    const float denominator = fmaxf(b.re * b.re + b.im * b.im, denominator_floor);
    return {
        (a.re * b.re + a.im * b.im) / denominator,
        (a.im * b.re - a.re * b.im) / denominator,
    };
}

RAYD_MATH_INLINE LegacySlabComplex legacy_div(LegacySlabComplex a, LegacySlabComplex b) {
    return legacy_div_floor(a, b, 1.0e-30f);
}

RAYD_MATH_INLINE LegacySlabComplex legacy_sqrt(LegacySlabComplex value) {
    const float magnitude = hypotf(value.re, value.im);
    const float real = sqrtf(fmaxf(0.0f, 0.5f * (magnitude + value.re)));
    const float imaginary = copysignf(sqrtf(fmaxf(0.0f, 0.5f * (magnitude - value.re))), value.im);
    return {real, imaginary};
}

RAYD_MATH_INLINE LegacySlabComplex legacy_interface_sqrt(LegacySlabComplex value) {
    const float magnitude = hypotf(value.re, value.im);
    const float real = sqrtf(fmaxf(0.0f, 0.5f * (magnitude + value.re)));
    const float imaginary_sign = value.im < 0.0f ? -1.0f : 1.0f;
    const float imaginary = imaginary_sign * sqrtf(fmaxf(0.0f, 0.5f * (magnitude - value.re)));
    return {real, imaginary};
}

RAYD_MATH_INLINE LegacySlabComplex legacy_exp_neg_2i(LegacySlabComplex value) {
    const float amplitude = expf(fminf(2.0f * value.im, 80.0f));
    float sine;
    float cosine;
#if defined(__CUDA_ARCH__)
    sincosf(2.0f * value.re, &sine, &cosine);
#else
    sine = std::sin(2.0f * value.re);
    cosine = std::cos(2.0f * value.re);
#endif
    return {amplitude * cosine, -amplitude * sine};
}

} // namespace rayd::shared::field_transport
namespace rayd::shared::transmission {

namespace utd = ::rayd::shared::diffraction;

RAYD_MATH_INLINE utd::Complex c_sqrt_passive(utd::Complex value) {
    const float magnitude = hypotf(value.re, value.im);
    const float real = sqrtf(fmaxf(0.5f * (magnitude + value.re), 0.0f));
    const float imaginary = -sqrtf(fmaxf(0.5f * (magnitude - value.re), 0.0f));
    return utd::cplx(real, imaginary);
}

RAYD_MATH_INLINE utd::Complex c_exp_neg_j(double phase) {
#if defined(__CUDA_ARCH__)
    const double reduced = fmod(phase, 6.283185307179586476925287);
#else
    const double reduced = std::fmod(phase, 6.283185307179586476925287);
#endif
    float sine;
    float cosine;
#if defined(__CUDA_ARCH__)
    sincosf(static_cast<float>(reduced), &sine, &cosine);
#else
    sine = std::sin(static_cast<float>(reduced));
    cosine = std::cos(static_cast<float>(reduced));
#endif
    return utd::cplx(cosine, -sine);
}

RAYD_MATH_INLINE utd::Complex c_div(utd::Complex a, utd::Complex b) {
    const float denominator = fmaxf(b.re * b.re + b.im * b.im, 1.0e-30f);
    return utd::cplx((a.re * b.re + a.im * b.im) / denominator, (a.im * b.re - a.re * b.im) / denominator);
}

RAYD_MATH_INLINE float c_abs2(utd::Complex value) {
    return utd::cplx_abs_sqr(value);
}

} // namespace rayd::shared::transmission

namespace rayd::torch_backend {

using shared::field::c3_add;
using shared::field::c3_dot_real;
using shared::field::c3_from_real;
using shared::field::c3_mul_complex;
using shared::field::c3_power;
using shared::field::c3_scale_complex;
using shared::field::c3_zero;
using shared::field::c_abs2;
using shared::field::c_add;
using shared::field::c_div;
using shared::field::c_exp_neg_i;
using shared::field::c_exp_neg_i_product;
using shared::field::c_make;
using shared::field::c_mul;
using shared::field::c_mul_real;
using shared::field::c_scale;
using shared::field::c_sqrt;
using shared::field::c_sub;
using shared::field::Complex;
using shared::field::Complex3;
using shared::field::finite_complex3;

} // namespace rayd::torch_backend

#if defined(__CUDACC__)

namespace rayd::shared::cuda_math {

constexpr float kSmallEps = 1e-12f;
constexpr float kDistanceEps = 1e-20f;
constexpr float kRayTMin = 1e-5f;
constexpr float kRayBias = 1e-5f;
constexpr float kDfrRayBias = 1e-4f;
constexpr float kRayTMax = 1e8f;
constexpr float kPi = 3.14159265358979323846f;

static_assert(kRayTMin == ::rayd::shared::rt::kMultipathTraceTMin);
static_assert(kRayBias == ::rayd::shared::rt::kMultipathRayBias);
static_assert(kRayTMax == ::rayd::shared::rt::kTraceTMaxFinite);

__forceinline__ __host__ __device__ float3 make_vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}
__forceinline__ __host__ __device__ float3 make_f3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__forceinline__ __device__ float3 make_f3(const float* ptr) {
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__forceinline__ __host__ __device__ float3 f3_zero() {
    return make_float3(0.0f, 0.0f, 0.0f);
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __host__ __device__ float3 operator*(float s, float3 a) {
    return a * s;
}

__forceinline__ __host__ __device__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__forceinline__ __host__ __device__ float3 add3(float3 a, float3 b) {
    return a + b;
}

__forceinline__ __host__ __device__ float3 sub3(float3 a, float3 b) {
    return a - b;
}

__forceinline__ __host__ __device__ float3 mul3(float s, float3 a) {
    return s * a;
}

__forceinline__ __host__ __device__ float3 mul3(float3 a, float s) {
    return a * s;
}

__forceinline__ __host__ __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __host__ __device__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__forceinline__ __host__ __device__ float3 cross(float3 a, float3 b) {
    return cross3(a, b);
}

__forceinline__ __host__ __device__ float norm2_3(float3 a) {
    return dot3(a, a);
}

__forceinline__ __host__ __device__ float squared_norm(float3 a) {
    return norm2_3(a);
}

__forceinline__ __host__ __device__ float norm3(float3 a) {
    return sqrtf(fmaxf(dot3(a, a), 0.0f));
}

__forceinline__ __host__ __device__ float length3(float3 a) {
    return norm3(a);
}

__forceinline__ __device__ float3 normalize3(float3 v, float eps = kSmallEps) {
    const float inv_len = rsqrtf(fmaxf(dot3(v, v), eps));
    return inv_len * v;
}

__forceinline__ __device__ float3 normalize3_finite_or_zero(float3 value) {
    const float squared = dot3(value, value);
    if (!(squared > kSmallEps) || !isfinite(squared))
        return make_float3(0.0f, 0.0f, 0.0f);
    return rsqrtf(squared) * value;
}
__forceinline__ __device__ void atomic_add3(float* base, int index, float3 value) {
    atomicAdd(&base[index * 3 + 0], value.x);
    atomicAdd(&base[index * 3 + 1], value.y);
    atomicAdd(&base[index * 3 + 2], value.z);
}

__forceinline__ __device__ float warp_sum_masked(unsigned int mask, float value) {
    float sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane) {
        if ((mask & (1u << lane)) != 0u)
            sum += __shfl_sync(mask, value, lane);
    }
    return sum;
}

__forceinline__ __device__ int warp_sum_masked(unsigned int mask, int value) {
    int sum = 0;
    for (int lane = 0; lane < 32; ++lane) {
        if ((mask & (1u << lane)) != 0u)
            sum += __shfl_sync(mask, value, lane);
    }
    return sum;
}

__forceinline__ __device__ bool warp_mask_leader(unsigned int mask) {
    return static_cast<int>(threadIdx.x & 31u) == (__ffs(mask) - 1);
}

struct WarpCellGroup {
    unsigned int peers = 0u;
    int count = 0;
    bool leader = false;
};

__forceinline__ __device__ WarpCellGroup warp_cell_group(int index) {
    WarpCellGroup group;
    group.peers = __match_any_sync(__activemask(), index);
    group.count = __popc(group.peers);
    group.leader = warp_mask_leader(group.peers);
    return group;
}

__forceinline__ __device__ void atomic_add_same_cell(float* base, int index, float value) {
    const unsigned int active = __activemask();
    const unsigned int peers = __match_any_sync(active, index);
    if (__popc(peers) == 1) {
        atomicAdd(base + index, value);
        return;
    }
    const float sum = warp_sum_masked(peers, value);
    if (warp_mask_leader(peers))
        atomicAdd(base + index, sum);
}

__forceinline__ __device__ void atomic_add_same_cell(float* base, int index, float value, WarpCellGroup group) {
    if (group.count == 1) {
        atomicAdd(base + index, value);
        return;
    }
    const float sum = warp_sum_masked(group.peers, value);
    if (group.leader)
        atomicAdd(base + index, sum);
}

__forceinline__ __device__ void atomic_add_same_cell(int* base, int index, int value) {
    const unsigned int active = __activemask();
    const unsigned int peers = __match_any_sync(active, index);
    if (__popc(peers) == 1) {
        atomicAdd(base + index, value);
        return;
    }
    const int sum = warp_sum_masked(peers, value);
    if (warp_mask_leader(peers))
        atomicAdd(base + index, sum);
}

__forceinline__ __device__ void atomic_add_same_cell(int* base, int index, int value, WarpCellGroup group) {
    if (group.count == 1) {
        atomicAdd(base + index, value);
        return;
    }
    const int sum = warp_sum_masked(group.peers, value);
    if (group.leader)
        atomicAdd(base + index, sum);
}

__forceinline__ __device__ void atomic_add_warp(float* base, float value) {
    const unsigned int active = __activemask();
    const float sum = warp_sum_masked(active, value);
    if (warp_mask_leader(active))
        atomicAdd(base, sum);
}

__forceinline__ __device__ void atomic_add_warp(int* base, int value) {
    const unsigned int active = __activemask();
    const int sum = warp_sum_masked(active, value);
    if (warp_mask_leader(active))
        atomicAdd(base, sum);
}

} // namespace rayd::shared::cuda_math

namespace rayd::torch_backend {
using namespace shared::cuda_math;
} // namespace rayd::torch_backend

#endif

#undef RAYD_MATH_INLINE
