#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <rayd/shared/contracts.h>

#ifdef __CUDACC__
// host+device so the UTD helpers are callable from the shared, host-compilable
// multipath algorithm bodies (which nvcc compiles for both passes in an object
// unit). Device codegen for the OptiX -ptx compiles is unchanged (the device
// pass emits the same instructions; -ptx drops the host side).
#define UTD_DEVICE   __host__ __device__
#define UTD_DINLINE  __host__ __device__ __forceinline__
#define UTD_GLOBAL   __global__
#else
#define UTD_DEVICE
#define UTD_DINLINE  inline
#define UTD_GLOBAL
#endif

namespace rayd::shared::diffraction {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr float UTD_PI             = 3.14159265358979323846f;
constexpr float UTD_TWO_PI         = 6.28318530717958647692f;
constexpr float UTD_EPS            = 1.0e-10f;
constexpr float UTD_SMALL_EPS      = 1.0e-6f;
constexpr float UTD_EPSILON_0      = rayd::shared::VacuumPermittivity;
// Numerical guard only; UTD is evaluated outside its asymptotic regime below
// ~lambda, which beats hard-zeroing the field there.
constexpr float UTD_MIN_DISTANCE   = 1.0e-4f;
constexpr float UTD_SLOPE_STEP     = 1.0e-4f;

constexpr int OWNERSHIP_DIRECT   = 0;
constexpr int OWNERSHIP_MIXED    = 1;
constexpr int UTD_PAIR_VALID_FLAG = 1;

// ---------------------------------------------------------------------------
// Forward-mode dual scalar. The UTD math below is templated on the scalar
// type; instantiating it with Dual turns the SAME forward implementation into
// its exact directional derivative (plan 07 AD-4). All discrete branches
// compare primal values only, so the dual pass follows the float pass's
// branch decisions exactly (fixed-winner contract). Ties in fmaxf/fminf keep
// the pass-through side (>= / <=), matching the AD-1 clamp-boundary
// convention validated by the tests/ad oracle.
// ---------------------------------------------------------------------------
struct Dual {
    float v;
    float d;

    Dual() = default;
    UTD_DINLINE constexpr Dual(float value) : v(value), d(0.0f) {}
    UTD_DINLINE constexpr Dual(float value, float tangent)
        : v(value), d(tangent) {}
};

UTD_DINLINE constexpr Dual operator+(Dual a, Dual b) {
    return {a.v + b.v, a.d + b.d};
}
UTD_DINLINE constexpr Dual operator-(Dual a, Dual b) {
    return {a.v - b.v, a.d - b.d};
}
UTD_DINLINE constexpr Dual operator-(Dual a) { return {-a.v, -a.d}; }
UTD_DINLINE constexpr Dual operator*(Dual a, Dual b) {
    return {a.v * b.v, a.d * b.v + a.v * b.d};
}
UTD_DINLINE Dual operator/(Dual a, Dual b) {
    const float inv = 1.0f / b.v;
    const float q = a.v * inv;
    return {q, (a.d - q * b.d) * inv};
}
UTD_DINLINE Dual& operator+=(Dual& a, Dual b) { a = a + b; return a; }
UTD_DINLINE Dual& operator-=(Dual& a, Dual b) { a = a - b; return a; }
UTD_DINLINE Dual& operator*=(Dual& a, Dual b) { a = a * b; return a; }
UTD_DINLINE Dual& operator/=(Dual& a, Dual b) { a = a / b; return a; }
UTD_DINLINE constexpr bool operator<(Dual a, Dual b) { return a.v < b.v; }
UTD_DINLINE constexpr bool operator>(Dual a, Dual b) { return a.v > b.v; }
UTD_DINLINE constexpr bool operator<=(Dual a, Dual b) { return a.v <= b.v; }
UTD_DINLINE constexpr bool operator>=(Dual a, Dual b) { return a.v >= b.v; }
UTD_DINLINE constexpr bool operator==(Dual a, Dual b) { return a.v == b.v; }
UTD_DINLINE constexpr bool operator!=(Dual a, Dual b) { return a.v != b.v; }

// Scalar math shims. The float overloads forward to the CRT/CUDA builtins so
// the float instantiation of the templated math is operation-for-operation
// identical to the pre-template code; the Dual overloads carry the tangent.
// (They must exist because the Dual overloads in this namespace would
// otherwise hide the global functions from unqualified calls.)
UTD_DINLINE float sqrtf(float x) { return ::sqrtf(x); }
UTD_DINLINE float fmaxf(float a, float b) { return ::fmaxf(a, b); }
UTD_DINLINE float fminf(float a, float b) { return ::fminf(a, b); }
UTD_DINLINE float fabsf(float x) { return ::fabsf(x); }
UTD_DINLINE float sinf(float x) { return ::sinf(x); }
UTD_DINLINE float cosf(float x) { return ::cosf(x); }
UTD_DINLINE float expf(float x) { return ::expf(x); }
UTD_DINLINE float atan2f(float y, float x) { return ::atan2f(y, x); }
UTD_DINLINE float roundf(float x) { return ::roundf(x); }
UTD_DINLINE float floorf(float x) { return ::floorf(x); }
UTD_DINLINE bool isfinite(float x) { return ::isfinite(x); }
#ifdef __CUDA_ARCH__
UTD_DINLINE void sincosf(float x, float* s, float* c) { ::sincosf(x, s, c); }
#else
// Host fallback keeps the pre-template behavior (separate sinf/cosf calls).
UTD_DINLINE void sincosf(float x, float* s, float* c) {
    *s = ::sinf(x);
    *c = ::cosf(x);
}
#endif

UTD_DINLINE Dual sqrtf(Dual x) {
    const float r = ::sqrtf(x.v);
    return {r, x.v > 0.0f ? 0.5f * x.d / r : 0.0f};
}
// Tie convention: the first (pass-through) operand wins at equality, which is
// the >= / <= subgradient convention of plan 07 AD-1.
UTD_DINLINE Dual fmaxf(Dual a, Dual b) { return a.v >= b.v ? a : b; }
UTD_DINLINE Dual fminf(Dual a, Dual b) { return a.v <= b.v ? a : b; }
UTD_DINLINE Dual fabsf(Dual x) {
    if (x.v > 0.0f) return x;
    if (x.v < 0.0f) return {-x.v, -x.d};
    return {0.0f, 0.0f};
}
UTD_DINLINE Dual sinf(Dual x) { return {::sinf(x.v), ::cosf(x.v) * x.d}; }
UTD_DINLINE Dual cosf(Dual x) { return {::cosf(x.v), -::sinf(x.v) * x.d}; }
UTD_DINLINE Dual expf(Dual x) {
    const float e = ::expf(x.v);
    return {e, e * x.d};
}
UTD_DINLINE Dual atan2f(Dual y, Dual x) {
    const float denom = x.v * x.v + y.v * y.v;
    const float slope = denom > 0.0f ? 1.0f / denom : 0.0f;
    return {::atan2f(y.v, x.v), (x.v * y.d - y.v * x.d) * slope};
}
UTD_DINLINE Dual roundf(Dual x) { return {::roundf(x.v), 0.0f}; }
UTD_DINLINE Dual floorf(Dual x) { return {::floorf(x.v), 0.0f}; }
UTD_DINLINE bool isfinite(Dual x) { return ::isfinite(x.v); }
UTD_DINLINE void sincosf(Dual x, Dual* s, Dual* c) {
    float sv;
    float cv;
#ifdef __CUDA_ARCH__
    ::sincosf(x.v, &sv, &cv);
#else
    sv = ::sinf(x.v);
    cv = ::cosf(x.v);
#endif
    *s = {sv, cv * x.d};
    *c = {cv, -sv * x.d};
}

UTD_DINLINE constexpr float scalar_value(float x) { return x; }
UTD_DINLINE constexpr float scalar_value(Dual x) { return x.v; }
UTD_DINLINE constexpr float scalar_tangent(float) { return 0.0f; }
UTD_DINLINE constexpr float scalar_tangent(Dual x) { return x.d; }

// ---------------------------------------------------------------------------
// Primitive types, templated on the scalar. The float aliases keep every
// pre-existing consumer (RayD kernels, channel_native transport) compiling
// and behaving exactly as before.
// ---------------------------------------------------------------------------
template <typename T>
struct Vec3T {
    T x, y, z;
};
using float3a = Vec3T<float>;

template <typename T>
struct ComplexT {
    T re, im;
};
using Complex = ComplexT<float>;

template <typename T>
struct Complex3T {
    ComplexT<T> x, y, z;
};
using Complex3 = Complex3T<float>;

template <typename T>
struct Jones2T {
    ComplexT<T> u, v;
};
using Jones2 = Jones2T<float>;

template <typename T>
struct JonesOperatorT {
    ComplexT<T> m00, m01, m10, m11;
};
using JonesOperator = JonesOperatorT<float>;

template <typename T>
struct Basis3T {
    Vec3T<T> u, v, k;
};
using Basis3 = Basis3T<float>;

// useFresnel / present are discrete flags and stay float in every
// instantiation (frozen branches under the fixed-winner contract).
template <typename T>
struct FaceMaterialParamsT {
    T etaR;
    T muR;
    T sigma;
    T gain;
    float useFresnel;
    float present;
};
using FaceMaterialParams = FaceMaterialParamsT<float>;

// Full state for a source-edge pair (SoA -> loaded per thread).
// selectStationaryPoint is a discrete mode flag and stays float.
template <typename T>
struct PairInputsT {
    Vec3T<T> edgePos;
    Vec3T<T> edgeDir;
    Vec3T<T> n0;
    Vec3T<T> nn;
    T       wedgeN;
    T       edgeLineMin;
    T       edgeLineMax;
    Vec3T<T> sourcePos;
    ComplexT<T> incidentField;
    ComplexT<T> incidentNormalDerivative;
    ComplexT<T> r0;
    ComplexT<T> rn;
    Complex3T<T> incidentVector;
    Complex3T<T> incidentDerivativeVector;
    Jones2T<T>  incidentJones;
    Jones2T<T>  incidentDerivativeJones;
    Basis3T<T>  incidentBasis;
    JonesOperatorT<T> face0Operator;
    JonesOperatorT<T> face1Operator;
    FaceMaterialParamsT<T> face0Material;
    FaceMaterialParamsT<T> face1Material;
    float   selectStationaryPoint;
    // Discrete mode flag (like selectStationaryPoint, always float): when set on
    // the stationary path (selectStationaryPoint > 0.5) the incident field is the
    // frozen EXTERNAL incidentJones (a coupled image-source spherical wave)
    // re-extrapolated from the frozen edge point to the re-anchored stationary
    // point, instead of the direct transmitter source. Default 0 preserves the
    // order-1 diffraction and MC callers bit-for-bit.
    float   stationaryExternalIncident;
    // ADR-017 (channel-native) ISB boundary-taper width scale. When > 0 on the
    // stationary path, the incident-boundary (ISB) beta-terms' odd (GO-step
    // carrying) part is notched over the congruent angular half-width
    // widthScale * w_F / s (w_F = sqrt(lambda s s'/(s+s'))), matching the
    // caller's smoothed LoS occlusion gate so the compensation pair transitions
    // together. Reflection-boundary terms are untouched. Default 0 preserves
    // every existing caller bit-for-bit (aggregate zero-init).
    float   isbTaperWidthScale;
};
using PairInputs = PairInputsT<float>;

template <typename T>
struct PairOutputsT {
    ComplexT<T>  field;
    Complex3T<T> vectorField;
};
using PairOutputs = PairOutputsT<float>;

template <typename T>
struct DiffractionOperatorTermsT {
    ComplexT<T> direct;
    ComplexT<T> face0;
    ComplexT<T> face1;
    ComplexT<T> directDphiPrime;
    ComplexT<T> face0DphiPrime;
    ComplexT<T> face1DphiPrime;
};
using DiffractionOperatorTerms = DiffractionOperatorTermsT<float>;

struct EdgeAngleCache {
    float3a sourceToEdge;
    float3a sourceToEdgeProj;
    float   sourceToEdgeProjNorm;
    float3a edgeToTarget;
    float3a edgeToTargetProj;
    float   edgeToTargetProjNorm;
    float3a toHatBase;
    float3a toHat;
    float   toHatBaseNorm;
    float3a kiProj;
    float3a koProj;
    float   phi;
    float   phiPrime;
    float   s;
    float   sPrime;
};

struct BetaTermCache {
    float n, kL, cotSign, cotArg;
    float cotValue, cot1, cot2;
    float a, a1, a2, aN, a1N;
    Complex transition, transition1, transition2;
    Complex value, first, second;
};

struct PairScalarInputs {
    float phi, phiPrime, s, sPrime, wedgeN;
    Complex incidentField;
    Complex incidentNormalDerivative;
    Complex r0, rn;
};

template <typename T>
struct MaterialParamsT {
    int   useFresnel;
    T etaR;
    T muR;
    T sigma;
    T gain;
    T omega;
    T txPolX;
    T txPolY;
    T txPolZ;
};
using MaterialParams = MaterialParamsT<float>;

// Gradient / tangent accumulator for PairInputs (mirrors the continuous
// fields). Serves both as the VJP output contract and as the tangent-seed
// container for the dual-instantiated JVP.
struct PairInputsGrad {
    float3a edgePos;
    float3a edgeDir;
    float3a n0;
    float3a nn;
    float   wedgeN;
    float   edgeLineMin;
    float   edgeLineMax;
    float3a sourcePos;
    Complex incidentField;
    Complex incidentNormalDerivative;
    Complex r0;
    Complex rn;
    Complex3 incidentVector;
    Complex3 incidentDerivativeVector;
    Jones2  incidentJones;
    Jones2  incidentDerivativeJones;
    Basis3  incidentBasis;
    JonesOperator face0Operator;
    JonesOperator face1Operator;
    FaceMaterialParams face0Material;
    FaceMaterialParams face1Material;
};

// ---------------------------------------------------------------------------
// Vec3 inline helpers
// ---------------------------------------------------------------------------
UTD_DINLINE float3a make_f3(float x, float y, float z) { return {x, y, z}; }

// Constant vector inside scalar-templated code (T must be spelled at the
// call site because the arguments are plain float literals).
template <typename T>
UTD_DINLINE Vec3T<T> v3_const(float x, float y, float z) {
    return {T(x), T(y), T(z)};
}

template <typename T = float>
UTD_DINLINE Vec3T<T> f3_zero() { return {T(0.0f), T(0.0f), T(0.0f)}; }

template <typename T>
UTD_DINLINE Vec3T<T> f3_add(Vec3T<T> a, Vec3T<T> b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template <typename T>
UTD_DINLINE Vec3T<T> f3_sub(Vec3T<T> a, Vec3T<T> b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
template <typename T, typename S>
UTD_DINLINE Vec3T<T> f3_mul(Vec3T<T> a, S s) {
    const T scale = T(s);
    return {a.x * scale, a.y * scale, a.z * scale};
}
template <typename T>
UTD_DINLINE Vec3T<T> f3_neg(Vec3T<T> a) { return {-a.x, -a.y, -a.z}; }
template <typename T>
UTD_DINLINE T f3_dot(Vec3T<T> a, Vec3T<T> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T>
UTD_DINLINE Vec3T<T> f3_cross(Vec3T<T> a, Vec3T<T> b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
template <typename T>
UTD_DINLINE T f3_len(Vec3T<T> a) {
    return sqrtf(fmaxf(f3_dot(a, a), T(0.0f)));
}
template <typename T, typename S>
UTD_DINLINE Vec3T<T> f3_div(Vec3T<T> a, S s) {
    const T scale = T(s);
    return {a.x / scale, a.y / scale, a.z / scale};
}

// ---------------------------------------------------------------------------
// Complex inline helpers
// ---------------------------------------------------------------------------
// Arithmetic literals (int/float) fall through to the float overload so
// pre-template call sites like cplx(0, 1) keep building a float Complex.
template <typename T, typename = std::enable_if_t<!std::is_arithmetic_v<T>>>
UTD_DINLINE ComplexT<T> cplx(T re, T im) { return {re, im}; }
UTD_DINLINE Complex cplx(float re, float im) { return {re, im}; }

// Constant complex inside scalar-templated code.
template <typename T>
UTD_DINLINE ComplexT<T> c_const(float re, float im) {
    return {T(re), T(im)};
}

template <typename T = float>
UTD_DINLINE ComplexT<T> cplx_zero() { return {T(0.0f), T(0.0f)}; }

template <typename T>
UTD_DINLINE ComplexT<T> cplx_add(ComplexT<T> a, ComplexT<T> b) {
    return {a.re + b.re, a.im + b.im};
}
template <typename T>
UTD_DINLINE ComplexT<T> cplx_sub(ComplexT<T> a, ComplexT<T> b) {
    return {a.re - b.re, a.im - b.im};
}
template <typename T>
UTD_DINLINE ComplexT<T> cplx_mul(ComplexT<T> a, ComplexT<T> b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <typename T, typename S>
UTD_DINLINE ComplexT<T> cplx_mul_real(ComplexT<T> a, S b) {
    const T scale = T(b);
    return {a.re * scale, a.im * scale};
}
template <typename T, typename S>
UTD_DINLINE ComplexT<T> cplx_div_real(ComplexT<T> a, S b) {
    const T scale = T(b);
    return {a.re / scale, a.im / scale};
}
template <typename T>
UTD_DINLINE ComplexT<T> cplx_div(ComplexT<T> a, ComplexT<T> b) {
    const T d = b.re * b.re + b.im * b.im + T(UTD_EPS);
    return {(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
}
template <typename T>
UTD_DINLINE ComplexT<T> cplx_conj(ComplexT<T> a) { return {a.re, -a.im}; }
template <typename T>
UTD_DINLINE ComplexT<T> cplx_exp_phase(T p) {
    T s;
    T c;
    sincosf(p, &s, &c);
    return {c, s};
}
template <typename T>
UTD_DINLINE T cplx_abs_sqr(ComplexT<T> a) { return a.re * a.re + a.im * a.im; }
UTD_DINLINE float cplx_adj_dot(Complex g, Complex v) {
    return g.re * v.re + g.im * v.im;
}
UTD_DINLINE bool cplx_any_nonzero(Complex v) {
    return fabsf(v.re) > 0.f || fabsf(v.im) > 0.f;
}

// ---------------------------------------------------------------------------
// Complex3 inline helpers
// ---------------------------------------------------------------------------
template <typename T = float>
UTD_DINLINE Complex3T<T> c3_zero() {
    return {cplx_zero<T>(), cplx_zero<T>(), cplx_zero<T>()};
}
template <typename T>
UTD_DINLINE Complex3T<T> c3_add(Complex3T<T> a, Complex3T<T> b) {
    return {cplx_add(a.x, b.x), cplx_add(a.y, b.y), cplx_add(a.z, b.z)};
}
template <typename T>
UTD_DINLINE Complex3T<T> c3_scale(Complex3T<T> v, ComplexT<T> c) {
    return {cplx_mul(v.x, c), cplx_mul(v.y, c), cplx_mul(v.z, c)};
}
template <typename T>
UTD_DINLINE ComplexT<T> cplx_dot_real(Complex3T<T> v, Vec3T<T> b) {
    ComplexT<T> s = cplx_zero<T>();
    s = cplx_add(s, cplx_mul_real(v.x, b.x));
    s = cplx_add(s, cplx_mul_real(v.y, b.y));
    s = cplx_add(s, cplx_mul_real(v.z, b.z));
    return s;
}
template <typename T>
UTD_DINLINE Complex3T<T> cplx_scale_real(Vec3T<T> b, ComplexT<T> c) {
    return {cplx_mul_real(c, b.x), cplx_mul_real(c, b.y), cplx_mul_real(c, b.z)};
}
UTD_DINLINE bool c3_grad_any_nonzero(Complex3 v) {
    return cplx_any_nonzero(v.x) || cplx_any_nonzero(v.y) || cplx_any_nonzero(v.z);
}

// ---------------------------------------------------------------------------
// Jones inline helpers
// ---------------------------------------------------------------------------
template <typename T = float>
UTD_DINLINE Jones2T<T> jones_zero() { return {cplx_zero<T>(), cplx_zero<T>()}; }
template <typename T>
UTD_DINLINE Jones2T<T> jones_add(Jones2T<T> a, Jones2T<T> b) {
    return {cplx_add(a.u, b.u), cplx_add(a.v, b.v)};
}
template <typename T>
UTD_DINLINE Jones2T<T> jones_scale(Jones2T<T> v, ComplexT<T> c) {
    return {cplx_mul(v.u, c), cplx_mul(v.v, c)};
}

template <typename T = float>
UTD_DINLINE JonesOperatorT<T> jop_zero() {
    return {cplx_zero<T>(), cplx_zero<T>(), cplx_zero<T>(), cplx_zero<T>()};
}
template <typename T = float>
UTD_DINLINE JonesOperatorT<T> jop_identity() {
    return {c_const<T>(1.f, 0.f), cplx_zero<T>(), cplx_zero<T>(),
            c_const<T>(1.f, 0.f)};
}
template <typename T>
UTD_DINLINE JonesOperatorT<T> jop_add(JonesOperatorT<T> a, JonesOperatorT<T> b) {
    return {cplx_add(a.m00, b.m00), cplx_add(a.m01, b.m01),
            cplx_add(a.m10, b.m10), cplx_add(a.m11, b.m11)};
}
template <typename T>
UTD_DINLINE JonesOperatorT<T> jop_scale(JonesOperatorT<T> v, ComplexT<T> c) {
    return {cplx_mul(v.m00, c), cplx_mul(v.m01, c),
            cplx_mul(v.m10, c), cplx_mul(v.m11, c)};
}
template <typename T>
UTD_DINLINE Jones2T<T> apply_jop(Jones2T<T> v, JonesOperatorT<T> op) {
    return {cplx_add(cplx_mul(op.m00, v.u), cplx_mul(op.m01, v.v)),
            cplx_add(cplx_mul(op.m10, v.u), cplx_mul(op.m11, v.v))};
}
template <typename T>
UTD_DINLINE Complex3T<T> vector_from_jones(Jones2T<T> v, Basis3T<T> b) {
    return c3_add(cplx_scale_real(b.u, v.u), cplx_scale_real(b.v, v.v));
}
template <typename T>
UTD_DINLINE Jones2T<T> jones_from_vector(Complex3T<T> v, Basis3T<T> b) {
    return {cplx_dot_real(v, b.u), cplx_dot_real(v, b.v)};
}

// ---------------------------------------------------------------------------
// Gradient helper: zero-initialise a PairInputsGrad
// ---------------------------------------------------------------------------
UTD_DINLINE PairInputsGrad pig_zero() {
    PairInputsGrad g{};
    g.edgePos = f3_zero(); g.edgeDir = f3_zero();
    g.n0 = f3_zero(); g.nn = f3_zero(); g.wedgeN = 0.f;
    g.edgeLineMin = 0.f; g.edgeLineMax = 0.f;
    g.sourcePos = f3_zero();
    g.incidentField = cplx_zero(); g.incidentNormalDerivative = cplx_zero();
    g.r0 = cplx_zero(); g.rn = cplx_zero();
    g.incidentVector = c3_zero(); g.incidentDerivativeVector = c3_zero();
    g.incidentJones = jones_zero(); g.incidentDerivativeJones = jones_zero();
    g.incidentBasis = {f3_zero(), f3_zero(), f3_zero()};
    g.face0Operator = jop_zero(); g.face1Operator = jop_zero();
    g.face0Material = {0, 0, 0, 0, 0, 0}; g.face1Material = {0, 0, 0, 0, 0, 0};
    return g;
}

// ---------------------------------------------------------------------------
// Dual seeding helpers: value + tangent -> dual-typed structures. These are
// what turns the templated forward into an exact JVP.
// ---------------------------------------------------------------------------
UTD_DINLINE Dual dual_seed(float value, float tangent) { return {value, tangent}; }
UTD_DINLINE Vec3T<Dual> dual_seed(float3a value, float3a tangent) {
    return {{value.x, tangent.x}, {value.y, tangent.y}, {value.z, tangent.z}};
}
UTD_DINLINE ComplexT<Dual> dual_seed(Complex value, Complex tangent) {
    return {{value.re, tangent.re}, {value.im, tangent.im}};
}
UTD_DINLINE Complex3T<Dual> dual_seed(Complex3 value, Complex3 tangent) {
    return {dual_seed(value.x, tangent.x), dual_seed(value.y, tangent.y),
            dual_seed(value.z, tangent.z)};
}
UTD_DINLINE Jones2T<Dual> dual_seed(Jones2 value, Jones2 tangent) {
    return {dual_seed(value.u, tangent.u), dual_seed(value.v, tangent.v)};
}
UTD_DINLINE JonesOperatorT<Dual> dual_seed(JonesOperator value, JonesOperator tangent) {
    return {dual_seed(value.m00, tangent.m00), dual_seed(value.m01, tangent.m01),
            dual_seed(value.m10, tangent.m10), dual_seed(value.m11, tangent.m11)};
}
UTD_DINLINE Basis3T<Dual> dual_seed(Basis3 value, Basis3 tangent) {
    return {dual_seed(value.u, tangent.u), dual_seed(value.v, tangent.v),
            dual_seed(value.k, tangent.k)};
}
UTD_DINLINE FaceMaterialParamsT<Dual> dual_seed(
    FaceMaterialParams value, FaceMaterialParams tangent) {
    return {{value.etaR, tangent.etaR},
            {value.muR, tangent.muR},
            {value.sigma, tangent.sigma},
            {value.gain, tangent.gain},
            value.useFresnel,
            value.present};
}
UTD_DINLINE Complex dual_tangent(ComplexT<Dual> value) {
    return {value.re.d, value.im.d};
}
UTD_DINLINE Complex3 dual_tangent(Complex3T<Dual> value) {
    return {dual_tangent(value.x), dual_tangent(value.y), dual_tangent(value.z)};
}
UTD_DINLINE Complex dual_value(ComplexT<Dual> value) {
    return {value.re.v, value.im.v};
}
UTD_DINLINE Complex3 dual_value(Complex3T<Dual> value) {
    return {dual_value(value.x), dual_value(value.y), dual_value(value.z)};
}
UTD_DINLINE Vec3T<Dual> dual_const3(float3a value) {
    return dual_seed(value, f3_zero());
}

// Seed the full pair state: every continuous field of the tangent enters the
// dual instantiation; the discrete flags are copied.
UTD_DINLINE PairInputsT<Dual> pair_inputs_seed(
    const PairInputs& value, const PairInputsGrad& tangent) {
    PairInputsT<Dual> out;
    out.edgePos = dual_seed(value.edgePos, tangent.edgePos);
    out.edgeDir = dual_seed(value.edgeDir, tangent.edgeDir);
    out.n0 = dual_seed(value.n0, tangent.n0);
    out.nn = dual_seed(value.nn, tangent.nn);
    out.wedgeN = {value.wedgeN, tangent.wedgeN};
    out.edgeLineMin = {value.edgeLineMin, tangent.edgeLineMin};
    out.edgeLineMax = {value.edgeLineMax, tangent.edgeLineMax};
    out.sourcePos = dual_seed(value.sourcePos, tangent.sourcePos);
    out.incidentField = dual_seed(value.incidentField, tangent.incidentField);
    out.incidentNormalDerivative = dual_seed(
        value.incidentNormalDerivative, tangent.incidentNormalDerivative);
    out.r0 = dual_seed(value.r0, tangent.r0);
    out.rn = dual_seed(value.rn, tangent.rn);
    out.incidentVector = dual_seed(value.incidentVector, tangent.incidentVector);
    out.incidentDerivativeVector = dual_seed(
        value.incidentDerivativeVector, tangent.incidentDerivativeVector);
    out.incidentJones = dual_seed(value.incidentJones, tangent.incidentJones);
    out.incidentDerivativeJones = dual_seed(
        value.incidentDerivativeJones, tangent.incidentDerivativeJones);
    out.incidentBasis = dual_seed(value.incidentBasis, tangent.incidentBasis);
    out.face0Operator = dual_seed(value.face0Operator, tangent.face0Operator);
    out.face1Operator = dual_seed(value.face1Operator, tangent.face1Operator);
    out.face0Material = dual_seed(value.face0Material, tangent.face0Material);
    out.face1Material = dual_seed(value.face1Material, tangent.face1Material);
    out.selectStationaryPoint = value.selectStationaryPoint;
    out.stationaryExternalIncident = value.stationaryExternalIncident;
    out.isbTaperWidthScale = value.isbTaperWidthScale;
    return out;
}

// Seed the shared material params: omega is the only continuous field the
// pair math reads (the tx polarization is a frozen unit vector).
UTD_DINLINE MaterialParamsT<Dual> material_params_seed(
    const MaterialParams& value, float tangentOmega) {
    MaterialParamsT<Dual> out;
    out.useFresnel = value.useFresnel;
    out.etaR = value.etaR;
    out.muR = value.muR;
    out.sigma = value.sigma;
    out.gain = value.gain;
    out.omega = {value.omega, tangentOmega};
    out.txPolX = value.txPolX;
    out.txPolY = value.txPolY;
    out.txPolZ = value.txPolZ;
    return out;
}

// ---------------------------------------------------------------------------
// Adjoint helper macros for complex mul (float-only reverse-mode building
// blocks; consumed by channel_native's field companions)
// ---------------------------------------------------------------------------
UTD_DINLINE void adj_cplx_mul(Complex a, Complex b, Complex gO,
                              Complex& gA, Complex& gB) {
    gA.re += gO.re*b.re + gO.im*b.im;
    gA.im += -gO.re*b.im + gO.im*b.re;
    gB.re += gO.re*a.re + gO.im*a.im;
    gB.im += -gO.re*a.im + gO.im*a.re;
}
UTD_DINLINE void adj_cplx_mul_real(Complex a, float b, Complex gO,
                                   Complex& gA, float& gB) {
    gA.re += gO.re*b;
    gA.im += gO.im*b;
    gB += cplx_adj_dot(gO, a);
}
UTD_DINLINE void adj_cplx_scale_real(float3a basis, Complex coeff,
                                     Complex3 gO, float3a& gBasis, Complex& gCoeff) {
    gBasis.x += cplx_adj_dot(gO.x, coeff);
    gBasis.y += cplx_adj_dot(gO.y, coeff);
    gBasis.z += cplx_adj_dot(gO.z, coeff);
    gCoeff.re += gO.x.re*basis.x + gO.y.re*basis.y + gO.z.re*basis.z;
    gCoeff.im += gO.x.im*basis.x + gO.y.im*basis.y + gO.z.im*basis.z;
}
UTD_DINLINE void adj_cplx_dot_real(Complex3 v, float3a b, Complex gO,
                                   Complex3& gV, float3a& gB) {
    gV.x.re += gO.re*b.x; gV.x.im += gO.im*b.x;
    gV.y.re += gO.re*b.y; gV.y.im += gO.im*b.y;
    gV.z.re += gO.re*b.z; gV.z.im += gO.im*b.z;
    gB.x += cplx_adj_dot(gO, v.x);
    gB.y += cplx_adj_dot(gO, v.y);
    gB.z += cplx_adj_dot(gO, v.z);
}
} // namespace rayd::shared::diffraction

// Temporary source-compatibility bridge for downstream code that included the
// original pre-RayD namespace directly.
namespace witwin::channel {
namespace native_ext = ::rayd::shared::diffraction;
}
