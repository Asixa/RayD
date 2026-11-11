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
static int pair(char** a) {
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
int main(int argc, char** argv) {
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
                    reference = kouyoumjian_pathak(math.radians(phi), math.radians(phi_p), n, s * s_p / (s + s_p),
                                                   hard)
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
        errors = [abs(self._coefficient(1.5, 30.0, 100.0, 100.0, 10.0, half, False, WEDGE) / reference - 1.0)
                  for half in (40.0, 400.0, 4000.0)]
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
            out = float(subprocess.run([self._exe, "p", *(repr(x) for x in v)], capture_output=True, text=True,
                                       check=True).stdout)
            s, t, o, d = (v[i:i + 3] for i in (0, 3, 6, 9))
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

    def test_direct_source_keeps_the_short_dipole_pattern(self):
        for theta in (90.0, 60.0, 30.0, 0.0):
            out = float(subprocess.run([self._exe, "d", str(theta)], capture_output=True, text=True,
                                       check=True).stdout)
            self.assertAlmostEqual(out, math.sin(math.radians(theta)), delta=2e-5)


if __name__ == "__main__":
    unittest.main()
