# Copyright Xingyu Chen.
# Tests shared field math.

import cmath
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "include" / "rayd"


class SharedFieldMathTests(unittest.TestCase):
    def test_utd_uses_only_the_rayd_namespace(self):
        types = (SHARED / "utd.h").read_text(encoding="utf-8")
        math_header = types
        self.assertIn("namespace rayd::shared::diffraction", types)
        self.assertIn("namespace rayd::shared::diffraction", math_header)
        self.assertNotIn("namespace witwin::channel", types)

    def test_accumulation_backends_delegate_diffraction_parameter(self):
        # Since P4c the algorithm body (and its UTD delegation) lives in the
        # host-compilable algo header; the device header keeps the OptiX layer.
        shared_algo = (ROOT / "src" / "diffraction" / "accumulation.h").read_text(encoding="utf-8")
        self.assertIn("<rayd/utd.h>", shared_algo)
        self.assertIn("::rayd::shared::diffraction::first_order_diffraction_parameter(", shared_algo)

        paths = (
            ROOT / "src" / "diffraction" / "accumulation_optix_jit.cu",
            ROOT / "src" / "diffraction" / "accumulation_optix.cu",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("<src/diffraction/accumulation_optix.cuh>", source)
            self.assertNotIn("first_order_diffraction_parameter", source)
            self.assertNotIn("rotate_around_axis", source)

    def test_complex_and_field_scalars_have_one_implementation(self):
        header = (SHARED / "math.h").read_text(encoding="utf-8")
        self.assertIn("struct Complex", header)
        self.assertIn("fresnel_reflection_coefficients", header)
        self.assertIn("free_space_amplitude", header)
        self.assertIn("propagation_phase", header)
        self.assertIn("is_standard_layout_v<Complexf>", header)

        shared_reflection_algo = (ROOT / "src" / "reflection" / "reflection_algorithms.cuh").read_text(encoding="utf-8")
        self.assertIn("field::fresnel_reflection_coefficients(", shared_reflection_algo)

        accumulation_adapters = (
            ROOT / "src" / "reflection" / "accumulation_optix_jit.cu",
            ROOT / "src" / "reflection" / "accumulation_optix.cu",
        )
        for path in accumulation_adapters:
            source = path.read_text(encoding="utf-8")
            self.assertIn("<src/reflection/reflection_accumulation_optix.cuh>", source)
            self.assertNotIn("fresnel_reflection_coefficients", source)
            self.assertNotIn("kEpsilon0", source)
            self.assertNotIn("struct Complex {", source)

        epc_consumers = (
            ROOT / "src" / "reflection" / "reflection_kernels_jit.cu",
            ROOT / "src" / "reflection" / "reflection_kernels.cu",
        )
        for path in epc_consumers:
            source = path.read_text(encoding="utf-8")
            self.assertIn("fresnel_reflection_coefficients", source)
            self.assertNotIn("kEpsilon0", source)
            self.assertNotIn("struct Complex {", source)

    def test_scalar_formula_reference_values(self):
        wavelength = 0.125
        distance = 3.75
        self.assertAlmostEqual(wavelength / (4.0 * math.pi * distance), 0.0026525823848649224, places=15)

        wave_number = 2.0 * math.pi / wavelength
        phase = cmath.exp(-1j * math.fmod(wave_number * distance, 2.0 * math.pi))
        self.assertAlmostEqual(phase.real, 1.0, places=12)
        self.assertAlmostEqual(phase.imag, 0.0, places=12)

        # A vacuum-to-vacuum interface must have zero TE/TM reflection.
        eta = complex(1.0, 0.0)
        mu = complex(1.0, 0.0)
        cosine = 0.6
        root = cmath.sqrt(mu * eta - complex(1.0 - cosine * cosine, 0.0))
        r_te = (mu * cosine - root) / (mu * cosine + root)
        r_tm = (eta * cosine - root) / (eta * cosine + root)
        self.assertAlmostEqual(abs(r_te), 0.0, places=12)
        self.assertAlmostEqual(abs(r_tm), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
