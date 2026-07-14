#pragma once

#include <rayd/shared/utd/utd_types.h>

// The UTD math below is templated on the scalar type T (float | Dual). The
// float instantiation is the production forward, operation-for-operation
// identical to the pre-template implementation (the scalar shims in
// utd_types.h forward to the same CRT/CUDA builtins). The Dual instantiation
// IS the derivative: seeding one input tangent turns the same forward pass
// into an exact JVP (pair_vector_output_jvp below), and seeded probes give
// the reverse mode (pair_vector_output_vjp). Discrete branches compare primal
// values only, so both instantiations always follow the same control flow
// (fixed-winner contract).

namespace rayd::shared::utd {

// ===================================================================
// Safe length / normalize
// ===================================================================
template <typename T>
UTD_DINLINE T safe_length(Vec3T<T> v) {
    return sqrtf(fmaxf(f3_dot(v,v), T(0.f)));
}

template <typename T>
UTD_DINLINE Vec3T<T> safe_normalize(Vec3T<T> v, Vec3T<T> fallback) {
    T n = safe_length(v);
    if (n > UTD_SMALL_EPS) return f3_div(v, n + T(UTD_EPS));
    T fn = safe_length(fallback);
    return f3_div(fallback, fn + T(UTD_EPS));
}

template <typename T>
UTD_DINLINE T safe_acos(T v) {
    T c = fminf(fmaxf(v, T(-1.f)), T(1.f));
    T s = sqrtf(fmaxf(T(1.f) - c*c, T(0.f)));
    return atan2f(s, c);
}

template <typename T>
UTD_DINLINE T cot_val(T v) {
    T s, c;
    sincosf(v, &s, &c);
    T d = (fabsf(s) < UTD_SMALL_EPS)
        ? T((scalar_value(s) + UTD_SMALL_EPS) >= 0.f ? UTD_SMALL_EPS : -UTD_SMALL_EPS)
        : s;
    T r = c / d;
    return isfinite(r) ? r : T(0.f);
}

// ===================================================================
// Wedge geometry helpers
// ===================================================================
template <typename T>
UTD_DINLINE Vec3T<T> project_to_wedge_plane(Vec3T<T> v, Vec3T<T> e) {
    return f3_sub(v, f3_mul(e, f3_dot(v,e)));
}

template <typename T>
UTD_DINLINE Vec3T<T> rotate_vector_around_axis(Vec3T<T> v, Vec3T<T> axis, T angle) {
    T s, c;
    sincosf(angle, &s, &c);
    Vec3T<T> term0 = f3_mul(v, c);
    Vec3T<T> term1 = f3_mul(f3_cross(axis, v), s);
    Vec3T<T> term2 = f3_mul(axis, f3_dot(axis, v) * (T(1.f) - c));
    return f3_add(f3_add(term0, term1), term2);
}

template <typename T>
UTD_DINLINE Vec3T<T> normalize_in_wedge_plane(Vec3T<T> v, Vec3T<T> e) {
    return safe_normalize(project_to_wedge_plane(v,e), v3_const<T>(1,0,0));
}

template <typename T>
UTD_DINLINE Vec3T<T> stable_perp_basis(Vec3T<T> rayDir, Vec3T<T> preferred) {
    Vec3T<T> proj = f3_sub(preferred, f3_mul(rayDir, f3_dot(preferred, rayDir)));
    Vec3T<T> altAxis = (fabsf(rayDir.z) < 0.9f) ? v3_const<T>(0,0,1) : v3_const<T>(0,1,0);
    Vec3T<T> altProj = f3_sub(altAxis, f3_mul(rayDir, f3_dot(altAxis, rayDir)));
    return safe_normalize(proj, altProj);
}

template <typename T>
UTD_DINLINE Basis3T<T> basis_from_first_vector(Vec3T<T> rayDir, Vec3T<T> firstVec, Vec3T<T> fallback) {
    Vec3T<T> rayHat = safe_normalize(rayDir, v3_const<T>(0,0,1));
    Vec3T<T> uVec = f3_sub(firstVec, f3_mul(rayHat, f3_dot(firstVec, rayHat)));
    Vec3T<T> uHat = safe_normalize(uVec, fallback);
    Vec3T<T> vFallback = stable_perp_basis(rayHat, v3_const<T>(0,1,0));
    Vec3T<T> vHat = safe_normalize(f3_cross(rayHat, uHat), vFallback);
    return {uHat, vHat, rayHat};
}

template <typename T>
UTD_DINLINE Basis3T<T> diffraction_edge_basis(Vec3T<T> rayDir, Vec3T<T> edgeDir, bool outgoing) {
    Vec3T<T> rayHat = safe_normalize(rayDir, v3_const<T>(0,0,1));
    Vec3T<T> edgeHat = safe_normalize(edgeDir, v3_const<T>(0,0,1));
    Vec3T<T> phiHat = f3_cross(rayHat, edgeHat);
    if (outgoing) phiHat = f3_neg(phiHat);
    Vec3T<T> fallback = stable_perp_basis(rayHat, edgeHat);
    return basis_from_first_vector(rayHat, phiHat, fallback);
}

template <typename T>
UTD_DINLINE JonesOperatorT<T> jop_in_basis(JonesOperatorT<T> op,
    Basis3T<T> srcIn, Basis3T<T> srcOut, Basis3T<T> dstIn, Basis3T<T> dstOut)
{
    Jones2T<T> unitU = {c_const<T>(1,0), cplx_zero<T>()};
    Jones2T<T> unitV = {cplx_zero<T>(), c_const<T>(1,0)};
    Complex3T<T> fieldU = vector_from_jones(unitU, dstIn);
    Jones2T<T> srcU = jones_from_vector(fieldU, srcIn);
    Jones2T<T> srcOutU = apply_jop(srcU, op);
    Jones2T<T> mappedU = jones_from_vector(vector_from_jones(srcOutU, srcOut), dstOut);
    Complex3T<T> fieldV = vector_from_jones(unitV, dstIn);
    Jones2T<T> srcV = jones_from_vector(fieldV, srcIn);
    Jones2T<T> srcOutV = apply_jop(srcV, op);
    Jones2T<T> mappedV = jones_from_vector(vector_from_jones(srcOutV, srcOut), dstOut);
    return {mappedU.u, mappedV.u, mappedU.v, mappedV.v};
}

template <typename T = float>
UTD_DINLINE Basis3T<T> basis_zero() {
    return {f3_zero<T>(), f3_zero<T>(), f3_zero<T>()};
}

// ===================================================================
// Exterior region / pole safety
// ===================================================================
template <typename T>
UTD_DINLINE bool wedge_exterior_mask(Vec3T<T> dirFromEdge, Vec3T<T> edgeDir,
                                     Vec3T<T> n0, Vec3T<T> nn) {
    Vec3T<T> dp = project_to_wedge_plane(dirFromEdge, edgeDir);
    T sd0 = f3_dot(dp, n0);
    T sdn = f3_dot(dp, nn);
    return (safe_length(dp) > UTD_SMALL_EPS) &&
           ((sd0 >= -UTD_SMALL_EPS) || (sdn >= -UTD_SMALL_EPS));
}

template <typename T>
UTD_DINLINE T distance_to_cot_pole(T v) {
    T np = roundf(v / UTD_PI) * UTD_PI;
    return fabsf(v - np);
}

template <typename T>
UTD_DINLINE bool cot_pole_safe_mask(T phi, T phiP, T n, float guard) {
    T twoN = 2.f * n;
    T args[4] = {
        (UTD_PI + phi - phiP) / twoN,
        (UTD_PI - phi + phiP) / twoN,
        (UTD_PI + phi + phiP) / twoN,
        (UTD_PI - phi - phiP) / twoN
    };
    for (int i = 0; i < 4; ++i)
        if (distance_to_cot_pole(args[i]) <= guard) return false;
    return true;
}

template <typename T>
UTD_DINLINE bool slope_safe_mask(T phi, T phiP, T n, float step) {
    T npi = n * UTD_PI;
    bool interior = (phi >= step) && (phi <= npi-T(step)) &&
                    (phiP >= step) && (phiP <= npi-T(step));
    T guard = step / (2.f * n);
    return interior && cot_pole_safe_mask(phi, phiP, n, scalar_value(guard));
}

// ===================================================================
// Boersma Fresnel integral with 1st and 2nd derivatives
// ===================================================================
template <typename T>
UTD_DINLINE void poly12(T x,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6, float c7,
    float c8, float c9, float c10, float c11,
    T& val, T& fst, T& snd)
{
    val = c11; fst = 0.f; snd = 0.f;
    #define POLY_STEP(ci) snd = snd*x + 2.f*fst; fst = fst*x + val; val = val*x + ci;
    POLY_STEP(c10) POLY_STEP(c9) POLY_STEP(c8) POLY_STEP(c7)
    POLY_STEP(c6)  POLY_STEP(c5) POLY_STEP(c4) POLY_STEP(c3)
    POLY_STEP(c2)  POLY_STEP(c1) POLY_STEP(c0)
    #undef POLY_STEP
}

template <typename T>
UTD_DINLINE void fresnel_boersma(T x, ComplexT<T>& val, ComplexT<T>& fst, ComplexT<T>& snd) {
    const float SE = 1.0e-12f;
    bool xPos = x >= 0.f;
    T xA = fabsf(x);
    T safeX = fmaxf(xA, T(SE));
    bool cond = xA < 4.f;

    T argS = 0.25f * xA;
    T argL = 4.f / safeX;
    T a1S = T(0.25f), a2S = T(0.f);
    T a1L = -4.f/(safeX*safeX);
    T a2L = 8.f/(safeX*safeX*safeX);

    T rS, rS1, rS2;
    poly12(argS, +1.595769140f,-0.000001702f,-6.808568854f,-0.000576361f,
           +6.920691902f,-0.016898657f,-3.050485660f,-0.075752419f,
           +0.850663781f,-0.025639041f,-0.150230960f,+0.034404779f, rS,rS1,rS2);
    T iS,iS1,iS2;
    poly12(argS, -0.000000033f,+4.255387524f,-0.000092810f,-7.780020400f,
           -0.009520895f,+5.075161298f,-0.138341947f,-1.363729124f,
           -0.403349276f,+0.702222016f,-0.216195929f,+0.019547031f, iS,iS1,iS2);
    T rL,rL1,rL2;
    poly12(argL, +0.000000000f,-0.024933975f,+0.000003936f,+0.005770956f,
           +0.000689892f,-0.009497136f,+0.011948809f,-0.006748873f,
           +0.000246420f,+0.002102967f,-0.001217930f,+0.000233939f, rL,rL1,rL2);
    T iL,iL1,iL2;
    poly12(argL, +0.199471140f,+0.000000023f,-0.009351341f,+0.000023006f,
           +0.004851466f,+0.001903218f,-0.017122914f,+0.029064067f,
           -0.027928955f,+0.016497308f,-0.005598515f,+0.000838386f, iL,iL1,iL2);

    T rC  = cond ? rS : rL;
    T iC  = cond ? iS : iL;
    T rC1 = cond ? rS1*a1S : rL1*a1L;
    T iC1 = cond ? iS1*a1S : iL1*a1L;
    T rC2 = cond ? rS2*a1S*a1S : rL2*a1L*a1L + rL1*a2L;
    T iC2 = cond ? iS2*a1S*a1S : iL2*a1L*a1L + iL1*a2L;

    T arg = cond ? argS : argL;
    T a1  = cond ? a1S : a1L;
    T a2  = cond ? a2S : a2L;
    T argSafe = fmaxf(arg, T(SE));
    T aSqrt  = sqrtf(argSafe);
    T aSqrt1 = 0.5f*a1/aSqrt;
    T aSqrt2 = 0.5f*a2/aSqrt - 0.25f*a1*a1/(argSafe*aSqrt);

    T rP  = rC*aSqrt;
    T rP1 = rC1*aSqrt + rC*aSqrt1;
    T rP2 = rC2*aSqrt + 2.f*rC1*aSqrt1 + rC*aSqrt2;
    T iP  = -iC*aSqrt;
    T iP1 = -(iC1*aSqrt + iC*aSqrt1);
    T iP2 = -(iC2*aSqrt + 2.f*iC1*aSqrt1 + iC*aSqrt2);

    T sinX, cosX;
    sincosf(xA, &sinX, &cosX);
    T vR = cosX*rP - sinX*iP;
    T vI = cosX*iP + sinX*rP;
    T f1R = cosX*(rP1-iP) - sinX*(rP+iP1);
    T f1I = cosX*(iP1+rP) + sinX*(rP1-iP);
    T f2R = cosX*(rP2-rP-2.f*iP1) - sinX*(2.f*rP1-iP+iP2);
    T f2I = cosX*(iP2+2.f*rP1-iP) + sinX*(rP2-rP-2.f*iP1);

    if (!cond) { vR += 0.5f; vI += 0.5f; }

    val = cplx(xPos ? vR : -vR, xPos ? vI : -vI);
    fst = cplx(f1R, f1I);
    snd = cplx(xPos ? f2R : -f2R, xPos ? f2I : -f2I);
}

template <typename T>
UTD_DINLINE T first_order_diffraction_parameter(
    Vec3T<T> sourcePos,
    Vec3T<T> targetPos,
    Vec3T<T> edgeOrigin,
    Vec3T<T> edgeDir)
{
    Vec3T<T> zeta = safe_normalize(edgeDir, v3_const<T>(0.f, 0.f, 1.f));
    Vec3T<T> targetOffset = f3_sub(targetPos, edgeOrigin);
    Vec3T<T> sourceOffset = f3_sub(sourcePos, edgeOrigin);
    Vec3T<T> targetProjection = f3_mul(zeta, f3_dot(targetOffset, zeta));
    Vec3T<T> sourceProjection = f3_mul(zeta, f3_dot(sourceOffset, zeta));
    Vec3T<T> targetRadial = f3_sub(targetOffset, targetProjection);
    Vec3T<T> sourceRadial = f3_sub(sourceOffset, sourceProjection);
    T targetRadialNorm = safe_length(targetRadial);
    T sourceRadialNorm = safe_length(sourceRadial);
    Vec3T<T> v1 = f3_div(targetRadial, fmaxf(targetRadialNorm, T(UTD_SMALL_EPS)));
    Vec3T<T> v2 = f3_div(sourceRadial, fmaxf(sourceRadialNorm, T(UTD_SMALL_EPS)));
    T theta = UTD_PI - safe_acos(f3_dot(v1, v2));
    Vec3T<T> rotationAxis = f3_cross(sourceRadial, targetRadial);
    T rotationAxisNorm = safe_length(rotationAxis);
    rotationAxis = rotationAxisNorm > UTD_SMALL_EPS
        ? f3_div(rotationAxis, rotationAxisNorm + T(UTD_EPS))
        : zeta;
    Vec3T<T> coplanarTarget = rotate_vector_around_axis(targetOffset, rotationAxis, theta);
    Vec3T<T> sourceToTarget = f3_sub(coplanarTarget, sourceOffset);
    T sourceToTargetNorm = safe_length(sourceToTarget);
    Vec3T<T> u0 = f3_div(sourceToTarget, fmaxf(sourceToTargetNorm, T(UTD_SMALL_EPS)));
    Vec3T<T> u1 = f3_cross(sourceOffset, u0);
    Vec3T<T> u2 = f3_cross(zeta, u0);
    T u2Norm = safe_length(u2);
    T sign = T(scalar_value(f3_dot(u1, u2)) >= 0.f ? 1.f : -1.f);
    return sign * safe_length(u1) / fmaxf(u2Norm, T(UTD_SMALL_EPS));
}

template <typename T>
struct FiniteEdgePointSelectionT {
    Vec3T<T> point;
    T edgeLineMin;
    T edgeLineMax;
    bool valid;
    bool inside;
};
using FiniteEdgePointSelection = FiniteEdgePointSelectionT<float>;

template <typename T>
UTD_DINLINE FiniteEdgePointSelectionT<T> finite_edge_diffraction_point(
    PairInputsT<T> state,
    Vec3T<T> targetPos)
{
    Vec3T<T> edgeHat = safe_normalize(state.edgeDir, v3_const<T>(0.f, 0.f, 1.f));
    Vec3T<T> edgeOrigin = f3_add(state.edgePos, f3_mul(edgeHat, state.edgeLineMin));
    T edgeLength = state.edgeLineMax - state.edgeLineMin;
    T parameter = first_order_diffraction_parameter(
        state.sourcePos,
        targetPos,
        edgeOrigin,
        edgeHat
    );
    bool valid = (edgeLength > UTD_SMALL_EPS) && isfinite(parameter);
    bool inside = valid && (parameter > 0.f) && (parameter < edgeLength);
    return {
        f3_add(edgeOrigin, f3_mul(edgeHat, parameter)),
        -parameter,
        edgeLength - parameter,
        valid,
        inside,
    };
}

template <typename T>
UTD_DINLINE PairInputsT<T> pair_state_at_stationary_point(
    PairInputsT<T> state,
    Vec3T<T> targetPos,
    bool& selected,
    bool& inside,
    bool& valid)
{
    selected = false;
    inside = false;
    valid = true;
    if (state.selectStationaryPoint <= 0.5f) {
        return state;
    }
    FiniteEdgePointSelectionT<T> point = finite_edge_diffraction_point(state, targetPos);
    if (!point.valid) {
        valid = false;
        return state;
    }
    state.edgePos = point.point;
    state.edgeLineMin = point.edgeLineMin;
    state.edgeLineMax = point.edgeLineMax;
    selected = true;
    inside = point.inside;
    return state;
}

template <typename T>
UTD_DINLINE ComplexT<T> direct_source_field(Vec3T<T> sourcePos, Vec3T<T> targetPos, T k) {
    T distance = safe_length(f3_sub(targetPos, sourcePos)) + T(UTD_EPS);
    T fspl = 1.f / (2.f * fmaxf(k, T(UTD_SMALL_EPS)) * distance);
    return cplx_mul_real(cplx_exp_phase(-k * distance), fspl);
}

template <typename T>
UTD_DINLINE Complex3T<T> direct_source_vector(
    Vec3T<T> sourcePos,
    Vec3T<T> targetPos,
    T k,
    MaterialParamsT<T> mat)
{
    Vec3T<T> rayDir = safe_normalize(f3_sub(targetPos, sourcePos), v3_const<T>(0.f, 0.f, 1.f));
    Vec3T<T> txPol = {mat.txPolX, mat.txPolY, mat.txPolZ};
    Vec3T<T> polDir = stable_perp_basis(rayDir, txPol);
    return cplx_scale_real(polDir, direct_source_field(sourcePos, targetPos, k));
}

template <typename T>
UTD_DINLINE T poly12_value(T x,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6, float c7,
    float c8, float c9, float c10, float c11)
{
    T val = T(c11);
    val = val * x + c10;
    val = val * x + c9;
    val = val * x + c8;
    val = val * x + c7;
    val = val * x + c6;
    val = val * x + c5;
    val = val * x + c4;
    val = val * x + c3;
    val = val * x + c2;
    val = val * x + c1;
    val = val * x + c0;
    return val;
}

template <typename T>
UTD_DINLINE ComplexT<T> fresnel_boersma_value(T x) {
    bool xPos = x > 0.f;
    T xA = fabsf(x);
    bool cond = xA < 4.f;
    T arg = cond ? (0.25f * xA) : (4.f / xA);
    T root = sqrtf(arg);

    T rS = poly12_value(arg, +1.595769140f,-0.000001702f,-6.808568854f,-0.000576361f,
        +6.920691902f,-0.016898657f,-3.050485660f,-0.075752419f,
        +0.850663781f,-0.025639041f,-0.150230960f,+0.034404779f);
    T iS = poly12_value(arg, -0.000000033f,+4.255387524f,-0.000092810f,-7.780020400f,
        -0.009520895f,+5.075161298f,-0.138341947f,-1.363729124f,
        -0.403349276f,+0.702222016f,-0.216195929f,+0.019547031f);
    T rL = poly12_value(arg, +0.000000000f,-0.024933975f,+0.000003936f,+0.005770956f,
        +0.000689892f,-0.009497136f,+0.011948809f,-0.006748873f,
        +0.000246420f,+0.002102967f,-0.001217930f,+0.000233939f);
    T iL = poly12_value(arg, +0.199471140f,+0.000000023f,-0.009351341f,+0.000023006f,
        +0.004851466f,+0.001903218f,-0.017122914f,+0.029064067f,
        -0.027928955f,+0.016497308f,-0.005598515f,+0.000838386f);

    T rP = (cond ? rS : rL) * root;
    T iP = -(cond ? iS : iL) * root;
    T sinX, cosX;
    sincosf(xA, &sinX, &cosX);
    T vR = cosX * rP - sinX * iP;
    T vI = cosX * iP + sinX * rP;
    if (!cond) { vR += 0.5f; vI += 0.5f; }
    return cplx(xPos ? vR : -vR, xPos ? vI : -vI);
}

// ===================================================================
// UTD transition function f(x) with 1st and 2nd derivatives
// ===================================================================
template <typename T>
UTD_DINLINE void f_utd_with_derivatives(T x, ComplexT<T>& val, ComplexT<T>& fst, ComplexT<T>& snd) {
    T sx = fmaxf(x, T(UTD_SMALL_EPS));
    ComplexT<T> fV, fF, fS;
    fresnel_boersma(x, fV, fF, fS);
    ComplexT<T> fcV = cplx_conj(fV), fcF = cplx_conj(fF), fcS = cplx_conj(fS);

    T pf  = sqrtf(UTD_PI*sx*0.5f);
    T pf1 = 0.5f*pf/sx;
    T pf2 = -0.25f*pf/(sx*sx);
    ComplexT<T> ph  = cplx_exp_phase(x);
    ComplexT<T> ph1 = cplx_mul(c_const<T>(0,1), ph);
    ComplexT<T> ph2 = cplx(-ph.re, -ph.im);
    ComplexT<T> br  = cplx_sub(c_const<T>(1,1), cplx_mul(c_const<T>(0,2), fcV));
    ComplexT<T> br1 = cplx_mul(c_const<T>(0,-2), fcF);
    ComplexT<T> br2 = cplx_mul(c_const<T>(0,-2), fcS);

    val = cplx_mul_real(cplx_mul(ph, br), pf);
    fst = cplx_add(cplx_add(
        cplx_mul_real(cplx_mul(ph, br), pf1),
        cplx_mul_real(cplx_mul(ph1, br), pf)),
        cplx_mul_real(cplx_mul(ph, br1), pf));
    snd = cplx_add(
        cplx_add(
            cplx_add(
                cplx_mul_real(cplx_mul(ph, br), pf2),
                cplx_mul_real(cplx_mul(ph1, br), 2.f*pf1)),
            cplx_mul_real(cplx_mul(ph, br1), 2.f*pf1)),
        cplx_add(
            cplx_add(
                cplx_mul_real(cplx_mul(ph2, br), pf),
                cplx_mul_real(cplx_mul(ph1, br1), 2.f*pf)),
        cplx_mul_real(cplx_mul(ph, br2), pf)));
}

template <typename T>
UTD_DINLINE ComplexT<T> f_utd_value(T x) {
    T sx = fmaxf(x, T(0.0f));
    ComplexT<T> fV = fresnel_boersma_value(x);
    ComplexT<T> bracket = cplx_sub(c_const<T>(1.0f, 1.0f), cplx_mul(c_const<T>(0.0f, 2.0f), cplx_conj(fV)));
    ComplexT<T> phase = cplx_exp_phase(x);
    T prefactor = sqrtf(UTD_PI * sx * 0.5f);
    return cplx_mul_real(cplx_mul(phase, bracket), prefactor);
}

// ===================================================================
// Beta term values + assembly
// ===================================================================
template <typename T>
UTD_DINLINE T shadow_a_threshold(T n) {
    return 8.0e-12f * fmaxf(n * n, T(1.0f));
}

template <typename T>
UTD_DINLINE ComplexT<T> cot_transition_product_value(
    T cotV,
    ComplexT<T> transition,
    T x,
    T x1,
    T kL,
    T n,
    float cotSign)
{
    ComplexT<T> raw = cplx_mul_real(transition, cotV);
    T safeKL = fabsf(kL) > UTD_EPS ? kL : T(0.0f);
    T a = safeKL != 0.0f ? fmaxf(x / safeKL, T(0.0f)) : T(0.0f);
    T a1 = safeKL != 0.0f ? x1 / safeKL : T(0.0f);
    T threshold = shadow_a_threshold(n);
    if (a > threshold) {
        return raw;
    }

    float fallbackSign = cotV >= 0.0f ? 1.0f : -1.0f;
    float a1Sign = a1 >= 0.0f ? 1.0f : -1.0f;
    float limitSign = fabsf(scalar_value(a1)) > UTD_SMALL_EPS ? cotSign * a1Sign : fallbackSign;
    T limitScale = limitSign * n * sqrtf(UTD_PI * fmaxf(kL, T(0.0f)));
    ComplexT<T> limit = cplx(limitScale, limitScale);
    T blend = fminf(T(1.0f), a / fmaxf(threshold, T(1.0e-20f)));
    return cplx_add(limit, cplx_mul_real(cplx_sub(raw, limit), blend));
}

template <typename T>
UTD_DINLINE void beta_term_values(T beta, T n, T kL, float cotSign,
    bool plusBranch, T& cotV, T& c1, T& c2,
    T& xo, T& x1, T& x2)
{
    T twoN = 2.f*n;
    T twoNPi = 2.f*n*UTD_PI;
    T ri = plusBranch ? roundf((beta+UTD_PI)/twoNPi) : roundf((beta-UTD_PI)/twoNPi);
    T po = twoNPi*ri - beta;
    T chp = cosf(0.5f*po);
    T a = 2.f*chp*chp;
    T a1v = sinf(po);
    T a2v = 1.f-a;
    T ca = (UTD_PI + cotSign*beta)/twoN;
    cotV = cot_val(ca);
    c1 = -(cotSign/twoN)*(1.f + cotV*cotV);
    c2 = 0.5f*cotV*(1.f + cotV*cotV)/(n*n);
    xo = kL*a;
    x1 = kL*a1v;
    x2 = kL*a2v;
}

template <typename T>
UTD_DINLINE void assemble_beta_term(T cotV, T c1, T c2,
    T x, T x1, T x2, T kL, T n, float cotSign,
    ComplexT<T> tr, ComplexT<T> tr1, ComplexT<T> tr2,
    ComplexT<T>& val, ComplexT<T>& fst, ComplexT<T>& snd)
{
    ComplexT<T> forwardTransition = f_utd_value(x);
    val = cot_transition_product_value(cotV, forwardTransition, x, x1, kL, n, cotSign);
    fst = cplx_add(cplx_mul_real(tr, c1), cplx_mul_real(tr1, cotV*x1));
    snd = cplx_add(
        cplx_add(cplx_mul_real(tr, c2), cplx_mul_real(tr1, 2.f*c1*x1)),
        cplx_mul_real(cplx_add(cplx_mul_real(tr2, x1*x1), cplx_mul_real(tr1, x2)), cotV));
}

// ===================================================================
// Diffraction beta groups (2D / 3D)
// ===================================================================
template <typename T>
UTD_DINLINE T endpoint_unpaired_direct_beta(T beta, T n) {
    T period = fmaxf(2.f * n * UTD_PI, T(UTD_SMALL_EPS));
    T centered = beta - period * floorf((beta + 0.5f * period) / period);
    if (centered > UTD_PI) {
        return 2.f * UTD_PI - centered;
    }
    if (centered < -UTD_PI) {
        return -2.f * UTD_PI - centered;
    }
    return centered;
}

template <typename T>
UTD_DINLINE void diffraction_beta_groups_from_betas(T dP, T sP2, T n, T k,
    T s, T sP, ComplexT<T> r0, ComplexT<T> rn,
    ComplexT<T>& factor, ComplexT<T>& dG, ComplexT<T>& dG1,
    ComplexT<T>& sG, ComplexT<T>& sG1, ComplexT<T>& dG2, ComplexT<T>& sG2)
{
    T l = s*sP/(s+sP+T(UTD_EPS));
    T kL = k*l;
    factor = cplx_mul_real(cplx_exp_phase(T(-0.25f*UTD_PI)),
                           -1.f/(2.f*n*sqrtf(UTD_TWO_PI*k+T(UTD_EPS))));
    T cv[4],c1[4],c2[4],xv[4],x1[4],x2[4];
    beta_term_values(dP, n, kL, +1.f, true,  cv[0],c1[0],c2[0],xv[0],x1[0],x2[0]);
    beta_term_values(dP, n, kL, -1.f, false, cv[1],c1[1],c2[1],xv[1],x1[1],x2[1]);
    beta_term_values(sP2,n, kL, +1.f, true,  cv[2],c1[2],c2[2],xv[2],x1[2],x2[2]);
    beta_term_values(sP2,n, kL, -1.f, false, cv[3],c1[3],c2[3],xv[3],x1[3],x2[3]);
    ComplexT<T> tr[4],tr1[4],tr2[4];
    for (int i=0;i<4;++i) f_utd_with_derivatives(xv[i],tr[i],tr1[i],tr2[i]);
    ComplexT<T> tv[4],tf[4],ts[4];
    for (int i=0;i<4;++i) {
        float cotSign = (i == 0 || i == 2) ? +1.f : -1.f;
        assemble_beta_term(cv[i],c1[i],c2[i],xv[i],x1[i],x2[i],kL,n,cotSign,tr[i],tr1[i],tr2[i],tv[i],tf[i],ts[i]);
    }
    dG  = cplx_add(tv[0],tv[1]);
    dG1 = cplx_add(tf[0],tf[1]);
    dG2 = cplx_add(ts[0],ts[1]);
    sG  = cplx_add(cplx_mul(rn,tv[2]), cplx_mul(r0,tv[3]));
    sG1 = cplx_add(cplx_mul(rn,tf[2]), cplx_mul(r0,tf[3]));
    sG2 = cplx_add(cplx_mul(rn,ts[2]), cplx_mul(r0,ts[3]));
}

template <typename T>
UTD_DINLINE void diffraction_beta_groups(T phi, T phiP, T n, T k,
    T s, T sP, ComplexT<T> r0, ComplexT<T> rn,
    ComplexT<T>& factor, ComplexT<T>& dG, ComplexT<T>& dG1,
    ComplexT<T>& sG, ComplexT<T>& sG1, ComplexT<T>& dG2, ComplexT<T>& sG2)
{
    diffraction_beta_groups_from_betas(phi - phiP, phi + phiP, n, k, s, sP,
                                       r0, rn, factor, dG, dG1, sG, sG1, dG2, sG2);
}

template <typename T>
UTD_DINLINE void diffraction_beta_groups_with_direct_beta(T phi, T phiP,
    T directBeta, T n, T k, T s, T sP, ComplexT<T> r0, ComplexT<T> rn,
    ComplexT<T>& factor, ComplexT<T>& dG, ComplexT<T>& dG1,
    ComplexT<T>& sG, ComplexT<T>& sG1, ComplexT<T>& dG2, ComplexT<T>& sG2)
{
    diffraction_beta_groups_from_betas(directBeta, phi + phiP, n, k, s, sP,
                                       r0, rn, factor, dG, dG1, sG, sG1, dG2, sG2);
}

template <typename T>
UTD_DINLINE void diffraction_beta_groups_3d_from_betas(T dP, T sP2, T n, T k,
    T s, T sP, T sinBeta0, ComplexT<T> r0, ComplexT<T> rn,
    ComplexT<T>& factor, ComplexT<T>& dG, ComplexT<T>& dG1,
    ComplexT<T>& sG, ComplexT<T>& sG1, ComplexT<T>& dG2, ComplexT<T>& sG2)
{
    T sb = fmaxf(sinBeta0, T(UTD_SMALL_EPS));
    T l = s*sP/(s+sP+T(UTD_EPS))*sb*sb;
    T kL = k*l;
    factor = cplx_mul_real(cplx_exp_phase(T(-0.25f*UTD_PI)),
                           -1.f/(2.f*n*sqrtf(UTD_TWO_PI*k+T(UTD_EPS))*sb));
    T cv[4],c1[4],c2[4],xv[4],x1[4],x2[4];
    beta_term_values(dP, n, kL, +1.f, true,  cv[0],c1[0],c2[0],xv[0],x1[0],x2[0]);
    beta_term_values(dP, n, kL, -1.f, false, cv[1],c1[1],c2[1],xv[1],x1[1],x2[1]);
    beta_term_values(sP2,n, kL, +1.f, true,  cv[2],c1[2],c2[2],xv[2],x1[2],x2[2]);
    beta_term_values(sP2,n, kL, -1.f, false, cv[3],c1[3],c2[3],xv[3],x1[3],x2[3]);
    ComplexT<T> tr[4],tr1[4],tr2[4];
    for (int i=0;i<4;++i) f_utd_with_derivatives(xv[i],tr[i],tr1[i],tr2[i]);
    ComplexT<T> tv[4],tf[4],ts[4];
    for (int i=0;i<4;++i) {
        float cotSign = (i == 0 || i == 2) ? +1.f : -1.f;
        assemble_beta_term(cv[i],c1[i],c2[i],xv[i],x1[i],x2[i],kL,n,cotSign,tr[i],tr1[i],tr2[i],tv[i],tf[i],ts[i]);
    }
    dG  = cplx_add(tv[0],tv[1]);
    dG1 = cplx_add(tf[0],tf[1]);
    dG2 = cplx_add(ts[0],ts[1]);
    sG  = cplx_add(cplx_mul(rn,tv[2]), cplx_mul(r0,tv[3]));
    sG1 = cplx_add(cplx_mul(rn,tf[2]), cplx_mul(r0,tf[3]));
    sG2 = cplx_add(cplx_mul(rn,ts[2]), cplx_mul(r0,ts[3]));
}

template <typename T>
UTD_DINLINE void diffraction_beta_groups_3d(T phi, T phiP, T n, T k,
    T s, T sP, T sinBeta0, ComplexT<T> r0, ComplexT<T> rn,
    ComplexT<T>& factor, ComplexT<T>& dG, ComplexT<T>& dG1,
    ComplexT<T>& sG, ComplexT<T>& sG1, ComplexT<T>& dG2, ComplexT<T>& sG2)
{
    diffraction_beta_groups_3d_from_betas(phi - phiP, phi + phiP, n, k, s, sP,
                                          sinBeta0, r0, rn, factor, dG, dG1,
                                          sG, sG1, dG2, sG2);
}

template <typename T>
UTD_DINLINE void diffraction_beta_groups_3d_with_direct_beta(T phi, T phiP,
    T directBeta, T n, T k, T s, T sP, T sinBeta0,
    ComplexT<T> r0, ComplexT<T> rn, ComplexT<T>& factor, ComplexT<T>& dG, ComplexT<T>& dG1,
    ComplexT<T>& sG, ComplexT<T>& sG1, ComplexT<T>& dG2, ComplexT<T>& sG2)
{
    diffraction_beta_groups_3d_from_betas(directBeta, phi + phiP, n, k, s, sP,
                                          sinBeta0, r0, rn, factor, dG, dG1,
                                          sG, sG1, dG2, sG2);
}

// ===================================================================
// Diffraction coefficients (2D / 3D)
// ===================================================================
template <typename T>
UTD_DINLINE ComplexT<T> diff_coeff_2d(T phi, T phiP, T n, T k,
                                   T s, T sP, ComplexT<T> r0, ComplexT<T> rn) {
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups(phi,phiP,n,k,s,sP,r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    return cplx_mul(fac, cplx_add(dG,sG));
}

template <typename T>
UTD_DINLINE ComplexT<T> diff_coeff_2d_endpoint_continued(T phi, T phiP,
    T n, T k, T s, T sP, ComplexT<T> r0, ComplexT<T> rn) {
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    T directBeta = endpoint_unpaired_direct_beta(phi - phiP, n);
    diffraction_beta_groups_with_direct_beta(phi,phiP,directBeta,n,k,s,sP,r0,rn,
                                             fac,dG,dG1,sG,sG1,dG2,sG2);
    return cplx_mul(fac, cplx_add(dG,sG));
}

template <typename T>
UTD_DINLINE ComplexT<T> diff_coeff_3d(T phi, T phiP, T n, T k,
    T s, T sP, T sb, ComplexT<T> r0, ComplexT<T> rn) {
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups_3d(phi,phiP,n,k,s,sP,sb,r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    return cplx_mul(fac, cplx_add(dG,sG));
}

template <typename T>
UTD_DINLINE ComplexT<T> diff_coeff_3d_endpoint_continued(T phi, T phiP,
    T n, T k, T s, T sP, T sb, ComplexT<T> r0, ComplexT<T> rn) {
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    T directBeta = endpoint_unpaired_direct_beta(phi - phiP, n);
    diffraction_beta_groups_3d_with_direct_beta(phi,phiP,directBeta,n,k,s,sP,sb,
                                                r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    return cplx_mul(fac, cplx_add(dG,sG));
}

template <typename T>
UTD_DINLINE ComplexT<T> diff_coeff_2d_angle_deriv(T phi, T phiP, T n, T k,
    T s, T sP, bool wrtPhi, ComplexT<T> r0, ComplexT<T> rn) {
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups(phi,phiP,n,k,s,sP,r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    ComplexT<T> combined = wrtPhi ? cplx_add(dG1,sG1)
                              : cplx_add(cplx_mul_real(dG1,-1.f), sG1);
    return cplx_mul(fac, combined);
}

template <typename T>
UTD_DINLINE ComplexT<T> diff_coeff_3d_angle_deriv(T phi, T phiP, T n, T k,
    T s, T sP, T sb, bool wrtPhi, ComplexT<T> r0, ComplexT<T> rn) {
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups_3d(phi,phiP,n,k,s,sP,sb,r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    ComplexT<T> combined = wrtPhi ? cplx_add(dG1,sG1)
                              : cplx_add(cplx_mul_real(dG1,-1.f), sG1);
    return cplx_mul(fac, combined);
}

template <typename T>
UTD_DINLINE ComplexT<T> slope_diff_2d(T phi, T phiP, T n, T k,
                                  T s, T sP, ComplexT<T> r0, ComplexT<T> rn) {
    ComplexT<T> d = diff_coeff_2d_angle_deriv(phi,phiP,n,k,s,sP,false,r0,rn);
    return cplx_div_real(cplx_mul(c_const<T>(0,-1), d), k);
}

template <typename T>
UTD_DINLINE ComplexT<T> slope_diff_3d(T phi, T phiP, T n, T k,
    T s, T sP, T sb, ComplexT<T> r0, ComplexT<T> rn) {
    ComplexT<T> d = diff_coeff_3d_angle_deriv(phi,phiP,n,k,s,sP,sb,false,r0,rn);
    return cplx_div_real(cplx_mul(c_const<T>(0,-1), d), k);
}

// ===================================================================
// Edge angle computation
// ===================================================================
template <typename T>
UTD_DINLINE T oriented_angle_positive(T y, T x) {
    T a = atan2f(y, x);
    return a < 0.f ? a + T(UTD_TWO_PI) : a;
}

template <typename T>
UTD_DINLINE void compute_edge_angles(Vec3T<T> srcPos, Vec3T<T> edgePos, Vec3T<T> edgeDir,
    Vec3T<T> n0, Vec3T<T> tgtPos,
    T& phi, T& phiP, T& s, T& sP)
{
    Vec3T<T> srcToEdge = f3_sub(edgePos, srcPos);
    Vec3T<T> srcProj = project_to_wedge_plane(srcToEdge, edgeDir);
    sP = safe_length(srcProj) + T(UTD_EPS);
    Vec3T<T> toHat = safe_normalize(f3_cross(n0, edgeDir), v3_const<T>(0,1,0));
    Vec3T<T> kiProj = f3_div(srcProj, sP);
    float signP = (scalar_value(-f3_dot(kiProj, n0)) >= 0.f ? 1.f : -1.f);
    phiP = UTD_PI - safe_acos(-f3_dot(kiProj, toHat));
    phiP = phiP * (-signP) + UTD_PI;

    Vec3T<T> edgeToTgt = f3_sub(tgtPos, edgePos);
    Vec3T<T> tgtProj = project_to_wedge_plane(edgeToTgt, edgeDir);
    s = safe_length(tgtProj) + T(UTD_EPS);
    Vec3T<T> koProj = f3_div(tgtProj, s);
    float signPhi = (scalar_value(f3_dot(koProj, n0)) >= 0.f ? 1.f : -1.f);
    phi = UTD_PI - safe_acos(f3_dot(koProj, toHat));
    phi = phi * (-signPhi) + UTD_PI;
}

template <typename T>
UTD_DINLINE void compute_edge_geometry_3d(Vec3T<T> srcPos, Vec3T<T> edgePos, Vec3T<T> edgeDir,
    Vec3T<T> n0, Vec3T<T> tgtPos,
    T& phi, T& phiP, T& s, T& sP, T& sinBeta0)
{
    T sProj, sPProj;
    compute_edge_angles(srcPos, edgePos, edgeDir, n0, tgtPos, phi, phiP, sProj, sPProj);
    Vec3T<T> srcToEdge = f3_sub(edgePos, srcPos);
    Vec3T<T> edgeToTgt = f3_sub(tgtPos, edgePos);
    sP = safe_length(srcToEdge) + T(UTD_EPS);
    s  = safe_length(edgeToTgt) + T(UTD_EPS);
    T sbP = fminf(fmaxf(sPProj/sP, T(UTD_SMALL_EPS)), T(1.f));
    T sb  = fminf(fmaxf(sProj/s,  T(UTD_SMALL_EPS)), T(1.f));
    sinBeta0 = sqrtf(fmaxf(sb*sbP, T(UTD_SMALL_EPS)));
}

// Float-only reverse-mode leaves kept for channel_native's hand-written field
// companions (plan 07 AD-1/AD-2).
UTD_DINLINE void adj_normalize_branch(float3a v, float3a gO, float3a& gV) {
    float vn = safe_length(v);
    if (vn <= UTD_SMALL_EPS) return;
    float d = vn + UTD_EPS;
    float dg = f3_dot(gO, v);
    gV = f3_add(gV, f3_sub(f3_div(gO, d), f3_mul(v, dg / (vn * d * d))));
}

UTD_DINLINE void adj_safe_normalize(float3a v, float3a fallback, float3a gO,
                                    float3a& gV, float3a& gFallback) {
    float vn = safe_length(v);
    if (vn > UTD_SMALL_EPS) {
        adj_normalize_branch(v, gO, gV);
    } else {
        adj_normalize_branch(fallback, gO, gFallback);
    }
}

UTD_DINLINE void adj_stable_perp_basis(float3a rayDir, float3a preferred, float3a gO,
                                       float3a& gRayDir, float3a& gPreferred) {
    float projDot = f3_dot(preferred, rayDir);
    float3a proj = f3_sub(preferred, f3_mul(rayDir, projDot));
    float3a altAxis = (fabsf(rayDir.z) < 0.9f) ? make_f3(0,0,1) : make_f3(0,1,0);
    float altDot = f3_dot(altAxis, rayDir);
    float3a altProj = f3_sub(altAxis, f3_mul(rayDir, altDot));

    float3a gProj = f3_zero();
    float3a gAltProj = f3_zero();
    adj_safe_normalize(proj, altProj, gO, gProj, gAltProj);

    gPreferred = f3_add(gPreferred, gProj);
    gRayDir = f3_sub(gRayDir, f3_mul(gProj, projDot));
    float gProjDot = -f3_dot(gProj, rayDir);
    gPreferred = f3_add(gPreferred, f3_mul(rayDir, gProjDot));
    gRayDir = f3_add(gRayDir, f3_mul(preferred, gProjDot));

    gRayDir = f3_sub(gRayDir, f3_mul(gAltProj, altDot));
    float gAltDot = -f3_dot(gAltProj, rayDir);
    gRayDir = f3_add(gRayDir, f3_mul(altAxis, gAltDot));
}

// ===================================================================
// Complex sqrt for Fresnel
// ===================================================================
template <typename T>
UTD_DINLINE ComplexT<T> cplx_sqrt(ComplexT<T> z) {
    T x = z.re, y = z.im;
    T r = sqrtf(x*x + y*y);
    bool nz = r > 0.f;
    bool xnn = x >= 0.f;
    T rMag = sqrtf((xnn && nz) ? T(0.5f)*(r+x) : T(0.f));
    T iMag = sqrtf((!xnn && nz) ? T(0.5f)*(r-x) : T(0.f));
    T srMag = rMag > 0.f ? rMag : T(1.f);
    T siMag = iMag > 0.f ? iMag : T(1.f);
    T rPart = xnn ? rMag : fabsf(y)/(2.f*siMag);
    T iPart = xnn ? y/(2.f*srMag) : (y < 0.f ? -iMag : iMag);
    return cplx(nz ? rPart : T(0.f), nz ? iPart : T(0.f));
}

UTD_DINLINE void adj_cplx_sqrt(Complex z, Complex gO, Complex& gZ) {
    Complex y = cplx_sqrt(z);
    float mag2 = cplx_abs_sqr(y);
    if (mag2 <= UTD_EPS)
        return;
    Complex denom = cplx_mul_real(cplx_conj(y), 2.f);
    gZ = cplx_add(gZ, cplx_div(gO, denom));
}

// ===================================================================
// Fresnel reflection
// ===================================================================
template <typename T>
UTD_DINLINE void fresnel_reflection_face(T cosTheta, T etaR, T muR, T sigma,
    T omega, ComplexT<T>& rTE, ComplexT<T>& rTM)
{
    T ct = fminf(fmaxf(cosTheta, T(UTD_SMALL_EPS)), T(1.f));
    T sinSq = 1.f - ct*ct;
    T so = fmaxf(omega, T(UTD_SMALL_EPS));
    ComplexT<T> eta = cplx(etaR, -sigma/(so*UTD_EPSILON_0));
    ComplexT<T> mu = cplx(muR, T(0.f));
    ComplexT<T> a = cplx_sqrt(cplx_sub(cplx_mul(mu, eta), cplx(sinSq, T(0))));
    ComplexT<T> muCt = cplx_mul_real(mu, ct);
    rTE = cplx_div(cplx_sub(muCt, a), cplx_add(muCt, a));
    rTM = cplx_div(cplx_sub(cplx_mul_real(eta,ct), a),
                   cplx_add(cplx_mul_real(eta,ct), a));
}

template <typename T>
UTD_DINLINE JonesOperatorT<T> face_reflection_operator(FaceMaterialParamsT<T> fm,
    T cosTheta, Vec3T<T> normal, Vec3T<T> inHat, Vec3T<T> outHat,
    Basis3T<T> inEdgeBasis, Basis3T<T> outEdgeBasis, T omega)
{
    ComplexT<T> gain = cplx(fm.gain, T(0));
    bool useFr = fm.useFresnel > 0.5f;
    ComplexT<T> rTE, rTM;
    fresnel_reflection_face(cosTheta, fm.etaR, fm.muR, fm.sigma, omega, rTE, rTM);
    JonesOperatorT<T> diagOp = useFr
        ? JonesOperatorT<T>{cplx_mul(gain,rTE), cplx_zero<T>(), cplx_zero<T>(), cplx_mul(gain,rTM)}
        : JonesOperatorT<T>{cplx(-fm.gain,T(0)), cplx_zero<T>(), cplx_zero<T>(), cplx(-fm.gain,T(0))};
    Vec3T<T> faceSIn = f3_cross(normal, inHat);
    Vec3T<T> faceSOutRaw = f3_cross(normal, outHat);
    Vec3T<T> fallbackOut = stable_perp_basis(outHat, faceSIn);
    Vec3T<T> faceSOut = f3_dot(faceSOutRaw, fallbackOut) < 0.0f
        ? f3_neg(faceSOutRaw)
        : faceSOutRaw;
    Basis3T<T> fIn  = basis_from_first_vector(inHat,  faceSIn, stable_perp_basis(inHat,  v3_const<T>(0,0,1)));
    Basis3T<T> fOut = basis_from_first_vector(outHat, faceSOut, fallbackOut);
    return jop_in_basis(diagOp, fIn, fOut, inEdgeBasis, outEdgeBasis);
}

template <typename T>
UTD_DINLINE JonesOperatorT<T> fallback_face_operator(JonesOperatorT<T> stored,
    Vec3T<T> normal, Vec3T<T> inHat, Vec3T<T> outHat,
    Basis3T<T> inEdgeBasis, Basis3T<T> outEdgeBasis)
{
    (void) normal;
    (void) inHat;
    (void) outHat;
    (void) inEdgeBasis;
    (void) outEdgeBasis;
    // Stored face operators in the state are already represented in the
    // diffraction edge basis. Re-basing them again corrupts both the forward
    // value and the operator gradients.
    return stored;
}

// ===================================================================
// Operator term computation (3D / 2D)
// ===================================================================
template <typename T>
UTD_DINLINE DiffractionOperatorTermsT<T> compute_op_terms_3d(T phi, T phiP,
    T wedgeN, T k, T s, T sP, T sinBeta0)
{
    ComplexT<T> z = cplx_zero<T>(), one = c_const<T>(1,0);
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups_3d(phi,phiP,wedgeN,k,s,sP,sinBeta0,z,z,fac,dG,dG1,sG,sG1,dG2,sG2);
    ComplexT<T> fac0,dF0,dF01,sF0,sF01,dF02,sF02;
    diffraction_beta_groups_3d(phi,phiP,wedgeN,k,s,sP,sinBeta0,one,z,fac0,dF0,dF01,sF0,sF01,dF02,sF02);
    ComplexT<T> fac1,dF1,dF11,sF1,sF11,dF12,sF12;
    diffraction_beta_groups_3d(phi,phiP,wedgeN,k,s,sP,sinBeta0,z,one,fac1,dF1,dF11,sF1,sF11,dF12,sF12);
    return {
        cplx_mul(fac, dG),
        cplx_mul(fac0, sF0),
        cplx_mul(fac1, sF1),
        cplx_mul(fac, cplx_mul_real(dG1, -1.f)),
        cplx_mul(fac0, sF01),
        cplx_mul(fac1, sF11)
    };
}

template <typename T>
UTD_DINLINE DiffractionOperatorTermsT<T> compute_op_terms_3d_endpoint_continued(T phi, T phiP,
    T wedgeN, T k, T s, T sP, T sinBeta0)
{
    ComplexT<T> z = cplx_zero<T>(), one = c_const<T>(1,0);
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    T directBeta = endpoint_unpaired_direct_beta(phi - phiP, wedgeN);
    diffraction_beta_groups_3d_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,sinBeta0,z,z,fac,dG,dG1,sG,sG1,dG2,sG2);
    ComplexT<T> fac0,dF0,dF01,sF0,sF01,dF02,sF02;
    diffraction_beta_groups_3d_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,sinBeta0,one,z,fac0,dF0,dF01,sF0,sF01,dF02,sF02);
    ComplexT<T> fac1,dF1,dF11,sF1,sF11,dF12,sF12;
    diffraction_beta_groups_3d_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,sinBeta0,z,one,fac1,dF1,dF11,sF1,sF11,dF12,sF12);
    return {
        cplx_mul(fac, dG),
        cplx_mul(fac0, sF0),
        cplx_mul(fac1, sF1),
        cplx_mul(fac, cplx_mul_real(dG1, -1.f)),
        cplx_mul(fac0, sF01),
        cplx_mul(fac1, sF11)
    };
}

template <typename T>
UTD_DINLINE DiffractionOperatorTermsT<T> compute_op_terms_2d(T phi, T phiP,
    T wedgeN, T k, T s, T sP)
{
    T l = s*sP/(s+sP+T(UTD_EPS));
    T kL = k*l;
    T dPhi = phi - phiP;
    T sPhi = phi + phiP;
    // Build beta term caches inline
    T cv[4],c1v[4],c2v[4],xv[4],x1v[4],x2v[4];
    beta_term_values(dPhi, wedgeN, kL, +1.f, true,  cv[0],c1v[0],c2v[0],xv[0],x1v[0],x2v[0]);
    beta_term_values(dPhi, wedgeN, kL, -1.f, false, cv[1],c1v[1],c2v[1],xv[1],x1v[1],x2v[1]);
    beta_term_values(sPhi, wedgeN, kL, +1.f, true,  cv[2],c1v[2],c2v[2],xv[2],x1v[2],x2v[2]);
    beta_term_values(sPhi, wedgeN, kL, -1.f, false, cv[3],c1v[3],c2v[3],xv[3],x1v[3],x2v[3]);
    ComplexT<T> tr[4],tr1[4],tr2[4];
    for (int i=0;i<4;++i) f_utd_with_derivatives(xv[i],tr[i],tr1[i],tr2[i]);
    ComplexT<T> tv[4],tf[4],ts[4];
    for (int i=0;i<4;++i) {
        float cotSign = (i == 0 || i == 2) ? +1.f : -1.f;
        assemble_beta_term(cv[i],c1v[i],c2v[i],xv[i],x1v[i],x2v[i],kL,wedgeN,cotSign,tr[i],tr1[i],tr2[i],tv[i],tf[i],ts[i]);
    }
    ComplexT<T> factor = cplx_mul_real(cplx_exp_phase(T(-0.25f*UTD_PI)),
                     -1.f/(2.f*wedgeN*sqrtf(UTD_TWO_PI*k+T(UTD_EPS))));
    ComplexT<T> difV = cplx_add(tv[0],tv[1]);
    ComplexT<T> difF = cplx_add(tf[0],tf[1]);
    return {
        cplx_mul(factor, difV),
        cplx_mul(factor, tv[3]),
        cplx_mul(factor, tv[2]),
        cplx_mul(factor, cplx_mul_real(difF, -1.f)),
        cplx_mul(factor, tf[3]),
        cplx_mul(factor, tf[2])
    };
}

template <typename T>
UTD_DINLINE DiffractionOperatorTermsT<T> compute_op_terms_2d_endpoint_continued(T phi, T phiP,
    T wedgeN, T k, T s, T sP)
{
    ComplexT<T> z = cplx_zero<T>(), one = c_const<T>(1,0);
    ComplexT<T> fac,dG,dG1,sG,sG1,dG2,sG2;
    T directBeta = endpoint_unpaired_direct_beta(phi - phiP, wedgeN);
    diffraction_beta_groups_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,z,z,fac,dG,dG1,sG,sG1,dG2,sG2);
    ComplexT<T> fac0,dF0,dF01,sF0,sF01,dF02,sF02;
    diffraction_beta_groups_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,one,z,fac0,dF0,dF01,sF0,sF01,dF02,sF02);
    ComplexT<T> fac1,dF1,dF11,sF1,sF11,dF12,sF12;
    diffraction_beta_groups_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,z,one,fac1,dF1,dF11,sF1,sF11,dF12,sF12);
    return {
        cplx_mul(fac, dG),
        cplx_mul(fac0, sF0),
        cplx_mul(fac1, sF1),
        cplx_mul(fac, cplx_mul_real(dG1, -1.f)),
        cplx_mul(fac0, sF01),
        cplx_mul(fac1, sF11)
    };
}

// ===================================================================
// Assemble diffraction operator (Jones) from terms
// ===================================================================
template <typename T>
UTD_DINLINE JonesOperatorT<T> assemble_diff_operator(ComplexT<T> free_term,
    ComplexT<T> face0_term, ComplexT<T> face1_term,
    JonesOperatorT<T> face0Op, JonesOperatorT<T> face1Op)
{
    JonesOperatorT<T> total = jop_scale(jop_identity<T>(), free_term);
    total = jop_add(total, jop_scale(face0Op, face0_term));
    total = jop_add(total, jop_scale(face1Op, face1_term));
    return total;
}

// ===================================================================
// Scalar field terms (for computePairFieldTerms)
// ===================================================================
template <typename T>
UTD_DINLINE ComplexT<T> finite_wedge_truncation_factor_bounds(
    PairInputsT<T> state,
    Vec3T<T> tgtPos,
    T k,
    T lineMin,
    T lineMax,
    bool stationaryAtOrigin)
{
    Vec3T<T> edgeHat = safe_normalize(state.edgeDir, v3_const<T>(0.f, 0.f, 1.f));
    Vec3T<T> edgePos = state.edgePos;
    Vec3T<T> sourcePos = state.sourcePos;

    T sourceAxial = f3_dot(f3_sub(sourcePos, edgePos), edgeHat);
    T targetAxial = f3_dot(f3_sub(tgtPos, edgePos), edgeHat);

    Vec3T<T> sourceToEdge = f3_sub(edgePos, sourcePos);
    Vec3T<T> edgeToTarget = f3_sub(tgtPos, edgePos);
    T sPrimeProj = safe_length(project_to_wedge_plane(sourceToEdge, edgeHat)) + T(UTD_EPS);
    T sProj = safe_length(project_to_wedge_plane(edgeToTarget, edgeHat)) + T(UTD_EPS);

    T stationaryU = stationaryAtOrigin
        ? T(0.f)
        : (sPrimeProj * targetAxial + sProj * sourceAxial) / (sProj + sPrimeProj + T(UTD_EPS));
    T sourceOffset = stationaryU - sourceAxial;
    T targetOffset = targetAxial - stationaryU;
    T sourceRange =
        sqrtf(sPrimeProj * sPrimeProj + sourceOffset * sourceOffset + T(UTD_EPS));
    T targetRange =
        sqrtf(sProj * sProj + targetOffset * targetOffset + T(UTD_EPS));
    T curvature =
        sPrimeProj * sPrimeProj / (sourceRange * sourceRange * sourceRange + T(UTD_EPS))
        + sProj * sProj / (targetRange * targetRange * targetRange + T(UTD_EPS));
    T scale = sqrtf(fmaxf(k * curvature, T(UTD_EPS)) / UTD_PI);

    ComplexT<T> fMin, fMin1, fMin2;
    ComplexT<T> fMax, fMax1, fMax2;
    fresnel_boersma(scale * (lineMin - stationaryU), fMin, fMin1, fMin2);
    fresnel_boersma(scale * (lineMax - stationaryU), fMax, fMax1, fMax2);
    ComplexT<T> delta = cplx_sub(fMax, fMin);
    return cplx_mul(c_const<T>(0.5f, 0.5f), cplx_conj(delta));
}

template <typename T>
UTD_DINLINE ComplexT<T> finite_wedge_truncation_factor_bounds(
    PairInputsT<T> state,
    Vec3T<T> tgtPos,
    T k,
    T lineMin,
    T lineMax)
{
    return finite_wedge_truncation_factor_bounds(
        state,
        tgtPos,
        k,
        lineMin,
        lineMax,
        false
    );
}

template <typename T>
UTD_DINLINE ComplexT<T> finite_wedge_truncation_factor(PairInputsT<T> state, Vec3T<T> tgtPos, T k) {
    return finite_wedge_truncation_factor_bounds(
        state,
        tgtPos,
        k,
        state.edgeLineMin,
        state.edgeLineMax
    );
}

template <typename T>
UTD_DINLINE ComplexT<T> finite_wedge_stationary_completion_factor(
    PairInputsT<T> state,
    Vec3T<T> tgtPos,
    T k,
    bool inside)
{
    if (inside) {
        return c_const<T>(1.f, 0.f);
    }
    T edgeLength = state.edgeLineMax - state.edgeLineMin;
    T outsideDistance = fmaxf(fmaxf(state.edgeLineMin, -state.edgeLineMax), T(0.f));
    T wavelength = (2.f * UTD_PI) / fmaxf(k, T(UTD_SMALL_EPS));
    T taperLength = fminf(0.25f * edgeLength, fmaxf(0.5f * wavelength, T(UTD_EPS)));
    T endpointU = fmaxf(outsideDistance / fmaxf(taperLength, T(UTD_EPS)), T(0.f));
    T endpointWeight = expf(-endpointU * endpointU);
    ComplexT<T> raw = finite_wedge_truncation_factor_bounds(
        state,
        tgtPos,
        k,
        state.edgeLineMin,
        state.edgeLineMax,
        true
    );
    ComplexT<T> boundary = state.edgeLineMin >= 0.f
        ? finite_wedge_truncation_factor_bounds(state, tgtPos, k, T(0.f), edgeLength, true)
        : finite_wedge_truncation_factor_bounds(state, tgtPos, k, -edgeLength, T(0.f), true);
    T boundaryPower = cplx_abs_sqr(boundary);
    if (boundaryPower <= UTD_EPS) {
        return cplx_mul_real(raw, endpointWeight);
    }
    return cplx_mul_real(cplx_div(raw, boundary), endpointWeight);
}

template <typename T>
UTD_DINLINE void compute_pair_field_terms(PairInputsT<T> state, Vec3T<T> tgtPos, T k,
    MaterialParamsT<T> mat, bool& geomValid, ComplexT<T>& field,
    ComplexT<T>& directGain, ComplexT<T>& derivativeGain)
{
    geomValid = false;
    field = cplx_zero<T>(); directGain = cplx_zero<T>(); derivativeGain = cplx_zero<T>();

    bool selectedStationary = false;
    bool selectedInside = false;
    bool selectedValid = true;
    state = pair_state_at_stationary_point(
        state,
        tgtPos,
        selectedStationary,
        selectedInside,
        selectedValid
    );
    if (!selectedValid) return;

    bool srcExt = wedge_exterior_mask(f3_sub(state.sourcePos, state.edgePos), state.edgeDir, state.n0, state.nn);
    bool tgtExt = wedge_exterior_mask(f3_sub(tgtPos, state.edgePos), state.edgeDir, state.n0, state.nn);
    T phi,phiP,s,sP,sb;
    compute_edge_geometry_3d(state.sourcePos, state.edgePos, state.edgeDir, state.n0, tgtPos, phi,phiP,s,sP,sb);

    geomValid = srcExt && (sP > UTD_MIN_DISTANCE) && (s > UTD_MIN_DISTANCE);
    if (!geomValid) return;

    ComplexT<T> r0 = state.r0, rn = state.rn;
    T w = state.wedgeN;
    bool poleSafe = cot_pole_safe_mask(phi,phiP,w,1.0e-6f);
    T safePhi  = poleSafe ? phi  : T(0.5f)*w*UTD_PI;
    T safePhiP = poleSafe ? phiP : T(0.5f)*w*UTD_PI;
    bool slopeSafe = slope_safe_mask(safePhi,safePhiP,w,UTD_SLOPE_STEP);
    bool useFace = (state.face0Material.present > 0.5f) || (state.face1Material.present > 0.5f);
    bool endpointContinuation = selectedStationary && !tgtExt;
    ComplexT<T> d = endpointContinuation
        ? (useFace ? diff_coeff_3d_endpoint_continued(phi,phiP,w,k,s,sP,sb,r0,rn)
                   : diff_coeff_2d_endpoint_continued(phi,phiP,w,k,s,sP,r0,rn))
        : (useFace ? diff_coeff_3d(phi,phiP,w,k,s,sP,sb,r0,rn)
                   : diff_coeff_2d(phi,phiP,w,k,s,sP,r0,rn));
    if (!poleSafe) { d.re = d.re; d.im = d.im; } // detach (no AD in CUDA anyway)
    ComplexT<T> dSlope = cplx_zero<T>();
    bool hasSlope = (cplx_abs_sqr(state.incidentNormalDerivative) > 1.0e-24f) && slopeSafe;
    if (hasSlope) {
        dSlope = useFace ? slope_diff_3d(safePhi,safePhiP,w,k,s,sP,sb,r0,rn)
                         : slope_diff_2d(safePhi,safePhiP,w,k,s,sP,r0,rn);
    }
    T ls = sqrtf(sP/(s*(s+sP)+T(UTD_EPS)));
    ComplexT<T> phase = cplx_exp_phase(-k*s);
    directGain = cplx_mul_real(cplx_mul(d,phase), ls);
    derivativeGain = cplx_mul_real(cplx_mul(dSlope,phase), ls);
    ComplexT<T> finiteFactor = selectedStationary
        ? finite_wedge_stationary_completion_factor(state, tgtPos, k, selectedInside)
        : finite_wedge_truncation_factor(state, tgtPos, k);
    directGain = cplx_mul(directGain, finiteFactor);
    derivativeGain = cplx_mul(derivativeGain, finiteFactor);
    ComplexT<T> incidentField = selectedStationary
        ? direct_source_field(state.sourcePos, state.edgePos, k)
        : state.incidentField;
    ComplexT<T> incidentNormalDerivative = selectedStationary
        ? cplx_zero<T>()
        : state.incidentNormalDerivative;
    field = cplx_add(cplx_mul(incidentField, directGain),
                     cplx_mul(incidentNormalDerivative, derivativeGain));
}

// ===================================================================
// Vector field contribution (mega-kernel core)
// ===================================================================
template <typename T>
UTD_DINLINE Complex3T<T> c3_scale_real(Complex3T<T> value, T scale) {
    ComplexT<T> s = cplx(scale, T(0.0f));
    return c3_scale(value, s);
}

template <typename T>
UTD_DINLINE Complex3T<T> compute_pair_vector_at_angles(
    PairInputsT<T> state,
    Vec3T<T> tgtPos,
    T k,
    MaterialParamsT<T> mat,
    T phi,
    T phiP,
    T s,
    T sP,
    T sb,
    Basis3T<T> inEB,
    Basis3T<T> outEB,
    ComplexT<T> finiteFactor,
    bool endpointContinuation)
{
    bool selectedStationary = state.selectStationaryPoint > 0.5f;
    Complex3T<T> incidentVector = selectedStationary
        ? direct_source_vector(state.sourcePos, state.edgePos, k, mat)
        : vector_from_jones(state.incidentJones, state.incidentBasis);
    Complex3T<T> incidentDerivativeVector = selectedStationary
        ? c3_zero<T>()
        : vector_from_jones(state.incidentDerivativeJones, state.incidentBasis);
    Jones2T<T> incJE  = jones_from_vector(incidentVector, inEB);
    Jones2T<T> incDJE = jones_from_vector(incidentDerivativeVector, inEB);
    bool poleSafe = cot_pole_safe_mask(phi, phiP, state.wedgeN, 1.0e-6f);
    T safePhi = poleSafe ? phi : T(0.5f) * state.wedgeN * UTD_PI;
    T safePhiP = poleSafe ? phiP : T(0.5f) * state.wedgeN * UTD_PI;
    bool slopeSafe = slope_safe_mask(safePhi, safePhiP, state.wedgeN, UTD_SLOPE_STEP);
    T derivativePower = cplx_abs_sqr(incDJE.u) + cplx_abs_sqr(incDJE.v);
    bool hasSlope = (derivativePower > 1.0e-24f) && slopeSafe;

    bool useFace = (state.face0Material.present > 0.5f) || (state.face1Material.present > 0.5f);
    bool f0HasMat = state.face0Material.present > 0.5f;
    bool f1HasMat = state.face1Material.present > 0.5f;
    bool useStoredFaceOps = mat.omega <= 0.f;

    JonesOperatorT<T> f0Op = (f0HasMat && !useStoredFaceOps)
        ? face_reflection_operator(state.face0Material,
            fminf(fmaxf(fabsf(sinf(phiP)), T(1.0e-6f)), T(1.f)),
            state.n0, inEB.k, outEB.k, inEB, outEB, mat.omega)
        : fallback_face_operator(state.face0Operator, state.n0, inEB.k, outEB.k, inEB, outEB);
    JonesOperatorT<T> f1Op = (f1HasMat && !useStoredFaceOps)
        ? face_reflection_operator(state.face1Material,
            fminf(fmaxf(fabsf(sinf(state.wedgeN*UTD_PI - phi)), T(1.0e-6f)), T(1.f)),
            state.nn, inEB.k, outEB.k, inEB, outEB, mat.omega)
        : fallback_face_operator(state.face1Operator, state.nn, inEB.k, outEB.k, inEB, outEB);

    DiffractionOperatorTermsT<T> terms = endpointContinuation
        ? (useFace
            ? compute_op_terms_3d_endpoint_continued(phi,phiP,state.wedgeN,k,s,sP,sb)
            : compute_op_terms_2d_endpoint_continued(phi,phiP,state.wedgeN,k,s,sP))
        : (useFace
            ? compute_op_terms_3d(phi,phiP,state.wedgeN,k,s,sP,sb)
            : compute_op_terms_2d(phi,phiP,state.wedgeN,k,s,sP));
    DiffractionOperatorTermsT<T> slopeTerms = endpointContinuation
        ? (useFace
            ? compute_op_terms_3d_endpoint_continued(safePhi,safePhiP,state.wedgeN,k,s,sP,sb)
            : compute_op_terms_2d_endpoint_continued(safePhi,safePhiP,state.wedgeN,k,s,sP))
        : (useFace
            ? compute_op_terms_3d(safePhi,safePhiP,state.wedgeN,k,s,sP,sb)
            : compute_op_terms_2d(safePhi,safePhiP,state.wedgeN,k,s,sP));
    JonesOperatorT<T> directOp = assemble_diff_operator(
        cplx_mul_real(terms.direct, -1.f),
        terms.face0,
        terms.face1,
        f0Op,
        f1Op
    );
    ComplexT<T> slopeFactor = cplx(T(0), -1.f/k);
    JonesOperatorT<T> slopeOp = hasSlope
        ? assemble_diff_operator(
            cplx_mul(slopeFactor, cplx_mul_real(slopeTerms.directDphiPrime, -1.f)),
            cplx_mul(slopeFactor, slopeTerms.face0DphiPrime),
            cplx_mul(slopeFactor, slopeTerms.face1DphiPrime),
            f0Op, f1Op)
        : jop_zero<T>();

    Jones2T<T> slopeFieldJ = hasSlope ? apply_jop(incDJE, slopeOp) : jones_zero<T>();
    Jones2T<T> fieldJ = jones_add(apply_jop(incJE, directOp), slopeFieldJ);
    fieldJ = jones_scale(fieldJ, finiteFactor);
    T ls = sqrtf(sP/(s*(s+sP)+T(UTD_EPS)));
    ComplexT<T> scale = cplx_mul_real(cplx_exp_phase(-k*s), ls);
    return c3_scale(vector_from_jones(fieldJ, outEB), scale);
}

template <typename T>
UTD_DINLINE Complex3T<T> compute_pair_vector_contribution_no_completion(PairInputsT<T> state, Vec3T<T> tgtPos,
    T k, MaterialParamsT<T> mat)
{
    bool selectedStationary = false;
    bool selectedInside = false;
    bool selectedValid = true;
    state = pair_state_at_stationary_point(
        state,
        tgtPos,
        selectedStationary,
        selectedInside,
        selectedValid
    );
    if (!selectedValid) return c3_zero<T>();

    bool srcExt = wedge_exterior_mask(f3_sub(state.sourcePos, state.edgePos), state.edgeDir, state.n0, state.nn);
    T phi,phiP,s,sP,sb;
    compute_edge_geometry_3d(state.sourcePos, state.edgePos, state.edgeDir, state.n0, tgtPos, phi,phiP,s,sP,sb);
    bool geomValid = srcExt && (sP > UTD_MIN_DISTANCE) && (s > UTD_MIN_DISTANCE);
    if (!geomValid) return c3_zero<T>();

    Basis3T<T> inEB  = diffraction_edge_basis(f3_sub(state.edgePos, state.sourcePos), state.edgeDir, false);
    Basis3T<T> outEB = diffraction_edge_basis(f3_sub(tgtPos, state.edgePos), state.edgeDir, true);
    ComplexT<T> finiteFactor = selectedStationary
        ? finite_wedge_stationary_completion_factor(state, tgtPos, k, selectedInside)
        : finite_wedge_truncation_factor(state, tgtPos, k);
    bool tgtExt = wedge_exterior_mask(f3_sub(tgtPos, state.edgePos), state.edgeDir, state.n0, state.nn);
    bool endpointContinuation = selectedStationary && !tgtExt;
    return compute_pair_vector_at_angles(
        state, tgtPos, k, mat, phi, phiP, s, sP, sb, inEB, outEB, finiteFactor,
        endpointContinuation);
}

template <typename T>
UTD_DINLINE Complex3T<T> compute_pair_vector_contribution(PairInputsT<T> state, Vec3T<T> tgtPos,
    T k, MaterialParamsT<T> mat)
{
    return compute_pair_vector_contribution_no_completion(state, tgtPos, k, mat);
}

// ===================================================================
// Full pair contribution (scalar field + vector field)
// ===================================================================
template <typename T>
UTD_DINLINE PairOutputsT<T> compute_pair_contribution(PairInputsT<T> state, Vec3T<T> tgtPos,
    T k, MaterialParamsT<T> mat)
{
    PairOutputsT<T> out;
    out.field = cplx_zero<T>(); out.vectorField = c3_zero<T>();
    bool gv; ComplexT<T> dg, dvg;
    compute_pair_field_terms(state, tgtPos, k, mat, gv, out.field, dg, dvg);
    out.vectorField = compute_pair_vector_contribution(state, tgtPos, k, mat);
    return out;
}

// ===================================================================
// Exact pair-vector JVP and VJP (plan 07 AD-4).
//
// The JVP seeds every continuous input with its tangent and runs the SAME
// templated forward once with Dual scalars -- no second implementation of
// the physics, no finite differences. k and mat.omega carry independent
// tangents; a frequency derivative chains both (dk = 2*pi/c * df,
// domega = 2*pi * df).
//
// The VJP contracts the output cotangent against one seeded JVP per input
// scalar (complex-linear inputs need a single probe each). It is a reference
// implementation: kernels that only need a subset of gradients should run
// their own seeded probes instead.
// ===================================================================
UTD_DINLINE Complex3 pair_vector_output_jvp(
    const PairInputs& pi,
    const PairInputsGrad& tangentState,
    float3a tgt,
    float3a tangentTgt,
    float k,
    float tangentK,
    const MaterialParams& mat,
    float tangentOmega)
{
    const PairInputsT<Dual> state = pair_inputs_seed(pi, tangentState);
    const Vec3T<Dual> target = dual_seed(tgt, tangentTgt);
    const MaterialParamsT<Dual> material = material_params_seed(mat, tangentOmega);
    const Complex3T<Dual> out = compute_pair_vector_contribution(
        state, target, Dual(k, tangentK), material);
    return dual_tangent(out);
}

namespace detail {

UTD_DINLINE float pair_vjp_contract(Complex3 vecGrad, Complex3 tangent) {
    return cplx_adj_dot(vecGrad.x, tangent.x) +
           cplx_adj_dot(vecGrad.y, tangent.y) +
           cplx_adj_dot(vecGrad.z, tangent.z);
}

// Gradient of a complex-linear input from a single (1, 0) probe: the probe
// tangent is the complex column a, and the real-pair gradient is
// (dot(g, a), dot(g, i*a)) -- the adj_cplx_mul contraction.
UTD_DINLINE Complex pair_vjp_complex_linear(Complex3 vecGrad, Complex3 probe) {
    const Complex3 rotated = {
        cplx(-probe.x.im, probe.x.re),
        cplx(-probe.y.im, probe.y.re),
        cplx(-probe.z.im, probe.z.re)};
    return cplx(pair_vjp_contract(vecGrad, probe),
                pair_vjp_contract(vecGrad, rotated));
}

}  // namespace detail

UTD_DINLINE void pair_vector_output_vjp(
    const PairInputs& pi,
    float3a tgt,
    float k,
    const MaterialParams& mat,
    Complex3 vecGrad,
    PairInputsGrad& sg,
    float3a& gTgt,
    float& gK,
    float& gOmega)
{
    if (!c3_grad_any_nonzero(vecGrad))
        return;

    const float3a zero3 = f3_zero();
    PairInputsGrad seed = pig_zero();

    // Real scalar seeds: one dual forward per component.
    float* slots[] = {
        &seed.sourcePos.x, &seed.sourcePos.y, &seed.sourcePos.z,
        &seed.edgePos.x, &seed.edgePos.y, &seed.edgePos.z,
        &seed.edgeDir.x, &seed.edgeDir.y, &seed.edgeDir.z,
        &seed.n0.x, &seed.n0.y, &seed.n0.z,
        &seed.nn.x, &seed.nn.y, &seed.nn.z,
        &seed.wedgeN,
        &seed.edgeLineMin, &seed.edgeLineMax,
        &seed.incidentBasis.u.x, &seed.incidentBasis.u.y, &seed.incidentBasis.u.z,
        &seed.incidentBasis.v.x, &seed.incidentBasis.v.y, &seed.incidentBasis.v.z,
        &seed.incidentBasis.k.x, &seed.incidentBasis.k.y, &seed.incidentBasis.k.z,
    };
    float* outputs[] = {
        &sg.sourcePos.x, &sg.sourcePos.y, &sg.sourcePos.z,
        &sg.edgePos.x, &sg.edgePos.y, &sg.edgePos.z,
        &sg.edgeDir.x, &sg.edgeDir.y, &sg.edgeDir.z,
        &sg.n0.x, &sg.n0.y, &sg.n0.z,
        &sg.nn.x, &sg.nn.y, &sg.nn.z,
        &sg.wedgeN,
        &sg.edgeLineMin, &sg.edgeLineMax,
        &sg.incidentBasis.u.x, &sg.incidentBasis.u.y, &sg.incidentBasis.u.z,
        &sg.incidentBasis.v.x, &sg.incidentBasis.v.y, &sg.incidentBasis.v.z,
        &sg.incidentBasis.k.x, &sg.incidentBasis.k.y, &sg.incidentBasis.k.z,
    };
    constexpr int kRealSlots = 27;
    const bool stationary = pi.selectStationaryPoint > 0.5f;
    for (int slot = 0; slot < kRealSlots; ++slot) {
        // The incident basis is unused under stationary-point selection (the
        // incident field is rebuilt from the source); skip the dead probes.
        if (stationary && slot >= 18)
            break;
        *slots[slot] = 1.f;
        const Complex3 tangent = pair_vector_output_jvp(
            pi, seed, tgt, zero3, k, 0.f, mat, 0.f);
        *slots[slot] = 0.f;
        *outputs[slot] += detail::pair_vjp_contract(vecGrad, tangent);
    }

    // Target position.
    for (int axis = 0; axis < 3; ++axis) {
        float3a tgtSeed = zero3;
        (axis == 0 ? tgtSeed.x : axis == 1 ? tgtSeed.y : tgtSeed.z) = 1.f;
        const Complex3 tangent = pair_vector_output_jvp(
            pi, seed, tgt, tgtSeed, k, 0.f, mat, 0.f);
        float& out = axis == 0 ? gTgt.x : axis == 1 ? gTgt.y : gTgt.z;
        out += detail::pair_vjp_contract(vecGrad, tangent);
    }

    // Wave number and face-operator angular frequency.
    {
        const Complex3 tangent = pair_vector_output_jvp(
            pi, seed, tgt, zero3, k, 1.f, mat, 0.f);
        gK += detail::pair_vjp_contract(vecGrad, tangent);
    }
    const bool storedOps = mat.omega <= 0.f;
    if (!storedOps) {
        const Complex3 tangent = pair_vector_output_jvp(
            pi, seed, tgt, zero3, k, 0.f, mat, 1.f);
        gOmega += detail::pair_vjp_contract(vecGrad, tangent);
    }

    // Face materials (used only on the omega > 0 path with present faces).
    const bool f0Mat = pi.face0Material.present > 0.5f && !storedOps;
    const bool f1Mat = pi.face1Material.present > 0.5f && !storedOps;
    if (f0Mat || f1Mat) {
        float* materialSlots[] = {
            &seed.face0Material.etaR, &seed.face0Material.sigma,
            &seed.face0Material.gain, &seed.face0Material.muR,
            &seed.face1Material.etaR, &seed.face1Material.sigma,
            &seed.face1Material.gain, &seed.face1Material.muR,
        };
        float* materialOutputs[] = {
            &sg.face0Material.etaR, &sg.face0Material.sigma,
            &sg.face0Material.gain, &sg.face0Material.muR,
            &sg.face1Material.etaR, &sg.face1Material.sigma,
            &sg.face1Material.gain, &sg.face1Material.muR,
        };
        for (int slot = 0; slot < 8; ++slot) {
            if ((slot < 4 && !f0Mat) || (slot >= 4 && !f1Mat))
                continue;
            *materialSlots[slot] = 1.f;
            const Complex3 tangent = pair_vector_output_jvp(
                pi, seed, tgt, zero3, k, 0.f, mat, 0.f);
            *materialSlots[slot] = 0.f;
            *materialOutputs[slot] += detail::pair_vjp_contract(vecGrad, tangent);
        }
    }

    // Complex-linear inputs: one probe per complex scalar. The stored face
    // operators only act when the material path is off; the incident Jones
    // vectors only act without stationary-point selection.
    if (!f0Mat) {
        Complex* opSeeds[] = {
            &seed.face0Operator.m00, &seed.face0Operator.m01,
            &seed.face0Operator.m10, &seed.face0Operator.m11};
        Complex* opOutputs[] = {
            &sg.face0Operator.m00, &sg.face0Operator.m01,
            &sg.face0Operator.m10, &sg.face0Operator.m11};
        for (int slot = 0; slot < 4; ++slot) {
            *opSeeds[slot] = cplx(1.f, 0.f);
            const Complex3 probe = pair_vector_output_jvp(
                pi, seed, tgt, zero3, k, 0.f, mat, 0.f);
            *opSeeds[slot] = cplx_zero();
            *opOutputs[slot] = cplx_add(
                *opOutputs[slot], detail::pair_vjp_complex_linear(vecGrad, probe));
        }
    }
    if (!f1Mat) {
        Complex* opSeeds[] = {
            &seed.face1Operator.m00, &seed.face1Operator.m01,
            &seed.face1Operator.m10, &seed.face1Operator.m11};
        Complex* opOutputs[] = {
            &sg.face1Operator.m00, &sg.face1Operator.m01,
            &sg.face1Operator.m10, &sg.face1Operator.m11};
        for (int slot = 0; slot < 4; ++slot) {
            *opSeeds[slot] = cplx(1.f, 0.f);
            const Complex3 probe = pair_vector_output_jvp(
                pi, seed, tgt, zero3, k, 0.f, mat, 0.f);
            *opSeeds[slot] = cplx_zero();
            *opOutputs[slot] = cplx_add(
                *opOutputs[slot], detail::pair_vjp_complex_linear(vecGrad, probe));
        }
    }
    if (!stationary) {
        Complex* jonesSeeds[] = {
            &seed.incidentJones.u, &seed.incidentJones.v,
            &seed.incidentDerivativeJones.u, &seed.incidentDerivativeJones.v};
        Complex* jonesOutputs[] = {
            &sg.incidentJones.u, &sg.incidentJones.v,
            &sg.incidentDerivativeJones.u, &sg.incidentDerivativeJones.v};
        const bool hasDerivative =
            cplx_abs_sqr(pi.incidentDerivativeJones.u) +
                cplx_abs_sqr(pi.incidentDerivativeJones.v) >
            0.f;
        for (int slot = 0; slot < 4; ++slot) {
            // At an exactly-zero incident normal derivative the slope branch
            // is gated off in a neighborhood of the primal, so the frozen
            // gate's zero IS the fixed-topology derivative.
            if (slot >= 2 && !hasDerivative)
                continue;
            *jonesSeeds[slot] = cplx(1.f, 0.f);
            const Complex3 probe = pair_vector_output_jvp(
                pi, seed, tgt, zero3, k, 0.f, mat, 0.f);
            *jonesSeeds[slot] = cplx_zero();
            *jonesOutputs[slot] = cplx_add(
                *jonesOutputs[slot], detail::pair_vjp_complex_linear(vecGrad, probe));
        }
    }
}

} // namespace rayd::shared::utd
