# Copyright Xingyu Chen.
# Tests the host-compiled UTD coefficient against float64 Kouyoumjian-Pathak and dipole references.

import cmath
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CXX = shutil.which("c++") or shutil.which("g++")

HARNESS = r"""
#include <rayd/utd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
using namespace rayd::shared::diffraction;
// argv: n phiP phi sP s halfLength pol n0x n0y nnx nny ex ey ez [eps_r sigma]  (degrees, metres; pol 0 soft,
// 1 hard; perfectly conducting faces unless eps_r and sigma are given)
// prints the receiver-projected vector field of the deterministic (stationary-point) path.
static int pair(char** a, float3a sourceSide = make_f3(0, 0, 0), float3a targetSide = make_f3(0, 0, 0)) {
    const double n = atof(a[1]), pp = atof(a[2]) * M_PI / 180.0, ph = atof(a[3]) * M_PI / 180.0;
    const double sP = atof(a[4]), s = atof(a[5]), half = atof(a[6]);
    const int pol = atoi(a[7]);
    const float k = static_cast<float>(2.0 * M_PI * 3.5e9 / 299792458.0);
    PairInputs p = {};
    p.edgePos = make_f3(0, 0, 0);
    p.edgeDir = make_f3(atof(a[12]), atof(a[13]), atof(a[14]));
    p.n0 = make_f3(atof(a[8]), atof(a[9]), 0);
    p.nn = make_f3(atof(a[10]), atof(a[11]), 0);
    p.wedgeN = static_cast<float>(n);
    p.edgeLineMin = static_cast<float>(-half);
    p.edgeLineMax = static_cast<float>(half);
    p.sourcePos = make_f3(sP * cos(pp), sP * sin(pp), 0);
    p.selectStationaryPoint = 1.f;
    p.sourceSide = sourceSide;
    p.targetSide = targetSide;
    const FaceMaterialParams face = {a[15] ? static_cast<float>(atof(a[15])) : 1.f, 1.f,
                                     a[15] ? static_cast<float>(atof(a[16])) : 1.0e9f, 1.f, 1.f, 1.f};
    p.face0Material = face;
    p.face1Material = face;
    MaterialParams mat = {};
    mat.useFresnel = 1;
    mat.etaR = mat.muR = mat.gain = 1.f;
    mat.omega = static_cast<float>(2.0 * M_PI * 3.5e9);
    if (pol == 0) {
        mat.txPolZ = 1.f;
    } else {
        mat.txPolX = static_cast<float>(sin(pp));
        mat.txPolY = static_cast<float>(-cos(pp));
    }
    const PairOutputs o = compute_pair_contribution(p, make_f3(s * cos(ph), s * sin(ph), 0), k, mat);
    const double rx = pol == 0 ? 0.0 : -sin(ph), ry = pol == 0 ? 0.0 : cos(ph), rz = pol == 0 ? 1.0 : 0.0;
    printf("%.9e %.9e\n", o.vectorField.x.re * rx + o.vectorField.y.re * ry + o.vectorField.z.re * rz,
           o.vectorField.x.im * rx + o.vectorField.y.im * ry + o.vectorField.z.im * rz);
    return 0;
}
// argv: dipole theta_deg -> |E| / |scalar field|
static int dipole(char** a) {
    const double t = atof(a[2]) * M_PI / 180.0;
    MaterialParams mat = {};
    mat.txPolZ = 1.f;
    const float3a target = make_f3(10.0 * sin(t), 0, 10.0 * cos(t));
    const Complex3 v = direct_source_vector(make_f3(0, 0, 0), target, 3.f, mat);
    const Complex c = direct_source_field(make_f3(0, 0, 0), target, 3.f);
    printf("%.9e\n", sqrt((cplx_abs_sqr(v.x) + cplx_abs_sqr(v.y) + cplx_abs_sqr(v.z)) / cplx_abs_sqr(c)));
    return 0;
}
// argv: p, then source, target, edge origin and edge direction (12 floats) -> stationary edge coordinate
static int parameter(char** a) {
    float v[12];
    for (int i = 0; i < 12; ++i)
        v[i] = static_cast<float>(atof(a[2 + i]));
    printf("%.9e\n", first_order_diffraction_parameter(make_f3(v[0], v[1], v[2]), make_f3(v[3], v[4], v[5]),
                                                       make_f3(v[6], v[7], v[8]), make_f3(v[9], v[10], v[11])));
    return 0;
}
// argv: j eps_r sigma zs zr polx poly polz -> residual of the total (GO reflection + diffraction) step across the
// face-0 reflection boundary of a half-plane in the xz plane (z edge, sheet on +x), relative to the GO step. The
// source and receiver sit at heights zs and zr, so the incidence on the edge is oblique when they differ.
static Complex3 c3_minus(Complex3 a, Complex3 b) {
    return {cplx_sub(a.x, b.x), cplx_sub(a.y, b.y), cplx_sub(a.z, b.z)};
}
static Complex3 go_reflection(float3a src, float3a rx, float3a pol, float k, double eps, double sigma, double omega) {
    // Image through y = 0; the specular point must lie on the sheet (x > 0).
    const float3a img = make_f3(src.x, -src.y, src.z);
    const float t = src.y / (src.y + rx.y);
    const float3a p = f3_add(src, f3_mul(f3_sub(make_f3(rx.x, -rx.y, rx.z), src), t));
    if (!(p.x > 0.f))
        return c3_zero<float>();
    const float3a ki = safe_normalize(f3_sub(p, src), make_f3(0, 0, 1));
    const float3a kr = safe_normalize(f3_sub(rx, p), make_f3(0, 0, 1));
    const float3a n = make_f3(0.f, 1.f, 0.f);
    const float3a sx = safe_normalize(f3_cross(n, ki), make_f3(0, 0, 1));
    const float3a pin = f3_cross(sx, ki), pout = f3_cross(sx, kr);
    Complex rte, rtm;
    fresnel_reflection_face(static_cast<float>(fabs(f3_dot(ki, n))), static_cast<float>(eps), 1.f,
                            static_cast<float>(sigma), static_cast<float>(omega), rte, rtm);
    const float3a e = f3_sub(pol, f3_mul(ki, f3_dot(pol, ki)));
    const float d = safe_length(f3_sub(rx, img));
    const Complex prop = cplx_mul_real(cplx_exp_phase(-k * d), 1.f / (2.f * k * d));
    const Complex3 out = c3_add(cplx_scale_real(sx, cplx_mul_real(rte, f3_dot(e, sx))),
                                cplx_scale_real(pout, cplx_mul_real(rtm, f3_dot(e, pin))));
    return c3_scale(out, prop);
}
static int jump(char** a) {
    const double eps = atof(a[2]), sigma = atof(a[3]), zs = atof(a[4]), zr = atof(a[5]);
    const float3a pol = make_f3(atof(a[6]), atof(a[7]), atof(a[8]));
    const double omega = 2.0 * M_PI * 3.5e9;
    const float k = static_cast<float>(omega / 299792458.0);
    const float3a src = make_f3(-3.0f, 2.0f, static_cast<float>(zs));
    // Receivers on a line across the face-0 reflection boundary (the image of the source through the edge).
    PairInputs p = {};
    p.edgePos = make_f3(0, 0, 0);
    p.edgeDir = make_f3(0, 0, 1);
    p.n0 = make_f3(0, 1, 0);
    p.nn = make_f3(0, -1, 0);
    p.wedgeN = 2.f;
    p.edgeLineMin = -400.f;
    p.edgeLineMax = 400.f;
    p.sourcePos = src;
    p.selectStationaryPoint = 1.f;
    const FaceMaterialParams face = {static_cast<float>(eps), 1.f, static_cast<float>(sigma), 1.f, 1.f, 1.f};
    p.face0Material = face;
    p.face1Material = face;
    MaterialParams mat = {};
    mat.useFresnel = 1;
    mat.etaR = mat.muR = mat.gain = 1.f;
    mat.omega = static_cast<float>(omega);
    mat.txPolX = pol.x;
    mat.txPolY = pol.y;
    mat.txPolZ = pol.z;
    // Boundary direction in the xy plane: the image (-3, -2) seen through the edge, continued to radius 4.
    const double bx = 3.0 / sqrt(13.0) * 4.0, by = 2.0 / sqrt(13.0) * 4.0;
    Complex3 total[2];
    Complex3 go_step = c3_zero<float>();
    for (int side = 0; side < 2; ++side) {
        const double ang = (side == 0 ? -1.0 : 1.0) * 2.0e-5;
        const float3a rx = make_f3(static_cast<float>(bx * cos(ang) - by * sin(ang)),
                                   static_cast<float>(bx * sin(ang) + by * cos(ang)), static_cast<float>(zr));
        const Complex3 d = compute_pair_contribution(p, rx, k, mat).vectorField;
        const Complex3 g = go_reflection(src, rx, pol, k, eps, sigma, omega);
        total[side] = c3_add(d, g);
        go_step = side == 0 ? g : c3_minus(go_step, g);
    }
    const Complex3 step = c3_minus(total[0], total[1]);
    const double num = cplx_abs_sqr(step.x) + cplx_abs_sqr(step.y) + cplx_abs_sqr(step.z);
    const double den = cplx_abs_sqr(go_step.x) + cplx_abs_sqr(go_step.y) + cplx_abs_sqr(go_step.z);
    printf("%.9e\n", sqrt(num / den));
    return 0;
}
// argv: f dir(3) n0(3) nn(3) edge(3) -> grazing_face_side of the ray leaving the edge along dir
static int face_side(char** a) {
    float v[12];
    for (int i = 0; i < 12; ++i)
        v[i] = static_cast<float>(atof(a[2 + i]));
    const float3a side = grazing_face_side(make_f3(v[0], v[1], v[2]), make_f3(v[9], v[10], v[11]),
                                           make_f3(v[3], v[4], v[5]), make_f3(v[6], v[7], v[8]));
    printf("%.9e %.9e %.9e\n", side.x, side.y, side.z);
    return 0;
}
int main(int argc, char** argv) {
    // argv: s sourceSideY targetSideY, then the pair arguments: the pair with side hints (0, y, 0).
    if (argc > 3 && argv[1][0] == 's')
        return pair(argv + 3, make_f3(0, static_cast<float>(atof(argv[2])), 0),
                    make_f3(0, static_cast<float>(atof(argv[3])), 0));
    if (argc > 1 && argv[1][0] == 'f')
        return face_side(argv);
    if (argc > 1 && argv[1][0] == 'j')
        return jump(argv);
    if (argc > 1 && argv[1][0] == 'd')
        return dipole(argv);
    if (argc > 1 && argv[1][0] == 'p')
        return parameter(argv);
    return pair(argv);
}
"""

K = 2.0 * math.pi * 3.5e9 / 299792458.0
WEDGE = (0, 1, -1, 0, 0, 0, 1)  # faces on phi = 0 (normal +y) and phi = 270 deg (normal -x), edge +z
HALF_PLANE = (0, 1, 0, -1, 0, 0, 1)  # one sheet on phi = 0, normals +-y


def _fresnel_tail(x: float) -> complex:
    """``int_x^inf exp(-j t^2) dt`` for ``x >= 0`` by direct quadrature plus the asymptotic tail."""
    try:
        from scipy import special
    except ImportError as error:  # pragma: no cover - environment dependent
        raise unittest.SkipTest("scipy is required for the float64 Fresnel reference") from error
    s, c = special.fresnel(x * math.sqrt(2.0 / math.pi))
    return math.sqrt(math.pi / 2.0) * ((0.5 - c) - 1j * (0.5 - s))


def _transition(x: float) -> complex:
    return 2j * math.sqrt(x) * cmath.exp(1j * x) * _fresnel_tail(math.sqrt(x))


def kouyoumjian_pathak(phi: float, phi_p: float, n: float, distance_l: float, hard: bool) -> complex:
    """Textbook perfectly conducting wedge coefficient (normal incidence on the edge)."""

    def a(beta: float, sign: int) -> float:
        big_n = round((beta + sign * math.pi) / (2.0 * math.pi * n))
        return 2.0 * math.cos((2.0 * math.pi * n * big_n - beta) / 2.0) ** 2

    terms = []
    for beta, sign in ((phi - phi_p, 1), (phi - phi_p, -1), (phi + phi_p, 1), (phi + phi_p, -1)):
        cot = 1.0 / math.tan((math.pi + sign * beta) / (2.0 * n))
        terms.append(cot * _transition(K * distance_l * a(beta, sign)))
    prefactor = -cmath.exp(-1j * math.pi / 4.0) / (2.0 * n * math.sqrt(2.0 * math.pi * K))
    reflected = terms[2] + terms[3]
    return prefactor * (terms[0] + terms[1] + (reflected if hard else -reflected))


@unittest.skipIf(CXX is None, "a host C++17 compiler is required")
class UtdReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        source = Path(cls._dir.name) / "harness.cpp"
        source.write_text(HARNESS, encoding="utf-8")
        cls._exe = str(Path(cls._dir.name) / "harness")
        subprocess.run([CXX, "-std=c++17", "-O1", f"-I{ROOT / 'include'}", str(source), "-o", cls._exe], check=True)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _field(self, n, phi_p, phi, s_p, s, half, hard, geometry, material=()):
        args = [self._exe, *(str(v) for v in (n, phi_p, phi, s_p, s, half, int(hard), *geometry, *material))]
        re, im = (float(v) for v in subprocess.run(args, capture_output=True, text=True, check=True).stdout.split())
        return complex(re, im)

    def _sided_field(self, source_side, target_side, n, phi_p, phi, s_p, s, half, hard, geometry):
        args = [self._exe, "s", str(source_side), str(target_side),
                *(str(v) for v in (n, phi_p, phi, s_p, s, half, int(hard), *geometry))]
        re, im = (float(v) for v in subprocess.run(args, capture_output=True, text=True, check=True).stdout.split())
        return complex(re, im)

    def _face_side(self, direction, n0, nn, edge):
        args = [self._exe, "f", *(str(v) for v in (*direction, *n0, *nn, *edge))]
        return tuple(float(v) for v in subprocess.run(args, capture_output=True, text=True, check=True).stdout.split())

    def _coefficient(self, n, phi_p, phi, s_p, s, half, hard, geometry):
        incident = cmath.exp(-1j * K * s_p) / (2.0 * K * s_p)
        spreading = math.sqrt(s_p / (s * (s + s_p)))
        field = self._field(n, phi_p, phi, s_p, s, half, hard, geometry)
        return field / (incident * spreading * cmath.exp(-1j * K * s))

    def _check(self, n, phi_p, phis, geometry, tolerance):
        for s_p, s in ((100.0, 10.0), (400.0, 10.0), (100.0, 30.0), (30.0, 30.0)):
            for hard in (False, True):
                for phi in phis:
                    value = self._coefficient(n, phi_p, phi, s_p, s, 400.0, hard, geometry)
                    reference = kouyoumjian_pathak(math.radians(phi), math.radians(phi_p), n, s * s_p / (s + s_p), hard)
                    with self.subTest(s_p=s_p, s=s, hard=hard, phi=phi):
                        self.assertLess(abs(value - reference) / abs(reference), tolerance)

    def test_right_angle_wedge_matches_kouyoumjian_pathak(self):
        # 2e-3: the 800 m edge's finite-aperture tails (|F| tails ~ 1/(pi v), v ~ 600) plus float32.
        self._check(1.5, 30.0, (100.0, 180.0, 240.0), WEDGE, 2e-3)

    def test_half_plane_matches_kouyoumjian_pathak_including_normal_exit(self):
        # phi = 270 deg leaves the edge along the sheet normal; the face s axis must not come from rounding noise.
        self._check(2.0, 60.0, (150.0, 200.0, 270.0, 320.0), HALF_PLANE, 2e-3)

    def test_long_edge_truncation_converges_to_the_infinite_edge(self):
        reference = kouyoumjian_pathak(math.radians(100.0), math.radians(30.0), 1.5, 1000.0 / 110.0, False)
        errors = [
            abs(self._coefficient(1.5, 30.0, 100.0, 100.0, 10.0, half, False, WEDGE) / reference - 1.0)
            for half in (40.0, 400.0, 4000.0)
        ]
        # The aperture tails fall as 1/half-length (not 1/sqrt), so each decade gains about 10x.
        self.assertLess(errors[1], 2e-3)
        self.assertLess(errors[2], 3e-4)
        self.assertGreater(errors[0] / errors[1], 5.0)

    def test_lossy_wedge_is_reciprocal_and_independent_of_the_face_labels(self):
        # Concrete faces, where the reflection coefficients depend strongly on the angle. Relabelling the faces
        # (face 0 <-> face n, edge reversed) is the same wedge; swapping source and observer is reciprocity.
        concrete = (5.24, 0.123)
        relabelled = (-1, 0, 0, 1, 0, 0, -1)
        for phi_p in (20.0, 60.0, 100.0, 200.0):
            for phi in (30.0, 150.0, 250.0):
                for hard in (False, True):
                    value = self._field(1.5, phi_p, phi, 40.0, 12.0, 400.0, hard, WEDGE, concrete)
                    other = self._field(1.5, phi_p, phi, 40.0, 12.0, 400.0, hard, relabelled, concrete)
                    with self.subTest(phi_p=phi_p, phi=phi, hard=hard, check="faces"):
                        self.assertLess(abs(value - other) / abs(value), 1e-3)
                    if not hard:
                        swapped = self._field(1.5, phi, phi_p, 12.0, 40.0, 400.0, hard, WEDGE, concrete)
                        with self.subTest(phi_p=phi_p, phi=phi, check="reciprocity"):
                            self.assertLess(abs(value - swapped) / abs(value), 1e-5)

    def test_stationary_point_is_the_float64_fermat_point(self):
        import random

        rng = random.Random(1)
        worst = 0.0
        for _ in range(200):
            v = [rng.uniform(-5.0, 5.0) for _ in range(12)]
            out = float(
                subprocess.run(
                    [self._exe, "p", *(repr(x) for x in v)], capture_output=True, text=True, check=True
                ).stdout
            )
            s, t, o, d = (v[i : i + 3] for i in (0, 3, 6, 9))
            norm = math.sqrt(sum(x * x for x in d))
            z = [x / norm for x in d]
            so = [a - b for a, b in zip(s, o)]
            to = [a - b for a, b in zip(t, o)]
            a_s, a_t = sum(a * b for a, b in zip(so, z)), sum(a * b for a, b in zip(to, z))
            r_s = math.sqrt(max(sum(x * x for x in so) - a_s * a_s, 0.0))
            r_t = math.sqrt(max(sum(x * x for x in to) - a_t * a_t, 0.0))
            worst = max(worst, abs(out - (r_t * a_s + r_s * a_t) / (r_s + r_t)))
        # float32 inputs of 10 m scale: a few ulps of 10 m.
        self.assertLess(worst, 2e-5)

    def test_a_receiver_exactly_on_a_boundary_takes_the_side_the_geometric_tests_choose(self):
        # Half-plane, source at 60 deg: the face-0 reflection boundary is phi = 120 deg (lit below) and the
        # incident boundary phi = 240 deg (lit below). Exactly on a boundary the angles cannot tell the side;
        # the coefficient takes the lit side of the reflection boundary (the in-face test includes the edge)
        # and the shadow side of the incident boundary (a segment through the edge is blocked).
        delta = 1.0e-3
        for boundary, lit_below, takes_lit in ((120.0, True, True), (240.0, True, False)):
            for hard in (False, True):
                on = self._field(2.0, 60.0, boundary, 40.0, 12.0, 400.0, hard, HALF_PLANE)
                below = self._field(2.0, 60.0, boundary - delta, 40.0, 12.0, 400.0, hard, HALF_PLANE)
                above = self._field(2.0, 60.0, boundary + delta, 40.0, 12.0, 400.0, hard, HALF_PLANE)
                lit, shadow = (below, above) if lit_below else (above, below)
                expected, other = (lit, shadow) if takes_lit else (shadow, lit)
                with self.subTest(boundary=boundary, hard=hard):
                    self.assertLess(abs(on - expected), 0.05 * abs(on - other))

    def test_grazing_incidence_keeps_one_side_for_both_terms(self):
        # A source within float noise of a face plane puts the incident and reflection boundaries on one angle;
        # the two terms must take one side there, or the pair leaves a spurious one-sided limit (it was 12x the
        # grazing value). Sources 2e-5 deg either side of either face plane match the exactly grazing value.
        for hard in (False, True):
            scale = abs(self._field(2.0, 0.0, 180.0, 40.0, 12.0, 400.0, True, HALF_PLANE))
            for phi in (180.0 - 1.0e-3, 180.0, 180.0 + 1.0e-3):
                grazing = self._field(2.0, 0.0, phi, 40.0, 12.0, 400.0, hard, HALF_PLANE)
                for phi_p in (-2.0e-5, 2.0e-5, 360.0 - 2.0e-5, 360.0, 360.0 + 2.0e-5):
                    value = self._field(2.0, phi_p, phi, 40.0, 12.0, 400.0, hard, HALF_PLANE)
                    with self.subTest(hard=hard, phi=phi, phi_p=phi_p):
                        self.assertLess(abs(value - grazing), 3.0e-2 * scale)

    def test_a_side_hint_picks_the_face_of_a_sheet_a_grazing_ray_runs_on(self):
        # A target or source exactly on a sheet's plane has no side of its own: the angle takes face 0 (phi = 0).
        # A side hint selects the face; each side's value is the limit from that side, and the two differ.
        for hard in (False, True):
            above = self._field(2.0, 60.0, 1.0e-3, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            below = self._field(2.0, 60.0, 360.0 - 1.0e-3, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            scale = max(abs(above), abs(below), 1.0e-30)
            unhinted = self._field(2.0, 60.0, 0.0, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            hinted_up = self._sided_field(0.0, 1.0, 2.0, 60.0, 0.0, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            hinted_down = self._sided_field(0.0, -1.0, 2.0, 60.0, 0.0, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            with self.subTest(hard=hard, end="target"):
                self.assertEqual(unhinted, hinted_up)
                self.assertLess(abs(hinted_up - above), 1.0e-2 * scale)
                self.assertLess(abs(hinted_down - below), 1.0e-2 * scale)
                if hard:
                    self.assertGreater(abs(above - below), 0.1 * scale)
            # The same at the source end: a source on the sheet's plane, seen from the side the hint names.
            src_above = self._field(2.0, 1.0e-3, 120.0, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            src_below = self._field(2.0, 360.0 - 1.0e-3, 120.0, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            src_scale = max(abs(src_above), abs(src_below), 1.0e-30)
            src_down = self._sided_field(-1.0, 0.0, 2.0, 0.0, 120.0, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            src_up = self._sided_field(1.0, 0.0, 2.0, 0.0, 120.0, 40.0, 12.0, 400.0, hard, HALF_PLANE)
            with self.subTest(hard=hard, end="source"):
                self.assertLess(abs(src_up - src_above), 1.0e-2 * src_scale)
                self.assertLess(abs(src_down - src_below), 1.0e-2 * src_scale)
        # A hint never moves a ray that is off the plane.
        self.assertEqual(self._field(2.0, 60.0, 30.0, 40.0, 12.0, 400.0, True, HALF_PLANE),
                         self._sided_field(0.0, -1.0, 2.0, 60.0, 30.0, 40.0, 12.0, 400.0, True, HALF_PLANE))

    def test_grazing_face_side_names_the_one_face_a_ray_runs_along(self):
        # 270-degree wedge: face 0 runs along +x (normal +y), face n along -y (normal -x); edge +z.
        n0, nn, edge = (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        self.assertEqual(self._face_side((3.0, 0.0, 0.5), n0, nn, edge), n0)
        self.assertEqual(self._face_side((0.0, -2.0, -1.0), n0, nn, edge), nn)
        for direction in ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            with self.subTest(direction=direction):
                self.assertEqual(self._face_side(direction, n0, nn, edge), (0.0, 0.0, 0.0))
        # A sheet holds a ray along it on both faces: no side.
        self.assertEqual(self._face_side((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), edge), (0.0, 0.0, 0.0))

    def test_the_coefficient_is_smooth_up_to_the_boundary(self):
        # Within 1e-4 rad of a GO boundary the cotangent pole and the transition zero must come from one small angle:
        # computed apart they missed the limit by up to 8 % at 2e-6 rad. Each side's values agree with that side's
        # value 1e-3 deg away to 0.5 %.
        for boundary in (120.0, 240.0):
            for hard in (False, True):
                far = {
                    side: self._field(2.0, 60.0, boundary + side * 1.0e-3, 40.0, 12.0, 400.0, hard, HALF_PLANE)
                    for side in (-1.0, 1.0)
                }
                for side in (-1.0, 1.0):
                    for offset in (1.0e-4, 3.0e-4):
                        near = self._field(2.0, 60.0, boundary + side * offset, 40.0, 12.0, 400.0, hard, HALF_PLANE)
                        with self.subTest(boundary=boundary, hard=hard, side=side, offset=offset):
                            self.assertLess(abs(near - far[side]), 5.0e-3 * abs(far[side]))

    def test_dielectric_face_term_cancels_the_reflection_step_under_oblique_incidence(self):
        # A concrete half-plane: across the face-0 reflection boundary the total field (GO reflection plus the
        # diffraction) must not step, for normal and oblique incidence on the edge and every polarization. The GO
        # reflection is the 3D Fresnel reflection at the specular point (an independent oracle in the harness).
        for zs, zr in ((0.0, 0.0), (1.5, -1.5), (3.0, 0.5)):
            for pol in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, 0.0, 0.8)):
                out = subprocess.run(
                    [self._exe, "j", "5.24", "0.123", str(zs), str(zr), *(str(v) for v in pol)],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                with self.subTest(zs=zs, zr=zr, pol=pol):
                    self.assertLess(float(out), 0.02)

    def test_direct_source_keeps_the_short_dipole_pattern(self):
        for theta in (90.0, 60.0, 30.0, 0.0):
            out = float(subprocess.run([self._exe, "d", str(theta)], capture_output=True, text=True, check=True).stdout)
            self.assertAlmostEqual(out, math.sin(math.radians(theta)), delta=2e-5)


if __name__ == "__main__":
    unittest.main()
