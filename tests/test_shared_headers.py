# Copyright Xingyu Chen.
# Tests shared headers.

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SharedHeaderTests(unittest.TestCase):
    def test_utd_has_one_canonical_public_owner(self):
        canonical = ROOT / "include" / "rayd" / "utd.h"
        self.assertTrue(canonical.is_file())
        self.assertFalse((ROOT / "include" / "rayd" / "diffraction").exists())
        text = canonical.read_text(encoding="utf-8")
        self.assertIn("namespace rayd::shared::diffraction", text)
        self.assertIn("first_order_diffraction_parameter", text)

    def test_backends_use_canonical_include(self):
        shared_algo = (ROOT / "src" / "diffraction" / "accumulation.h").read_text(encoding="utf-8")
        self.assertIn("<rayd/utd.h>", shared_algo)
        self.assertNotIn("<utd/", shared_algo)
        shared_device = (ROOT / "src" / "diffraction" / "accumulation_optix.cuh").read_text(encoding="utf-8")
        self.assertNotIn("<utd/", shared_device)

        sources = [
            ROOT / "src" / "diffraction" / "accumulation_optix_jit.cu",
            ROOT / "src" / "diffraction" / "accumulation_optix.cu",
        ]
        for source in sources:
            text = source.read_text(encoding="utf-8")
            self.assertIn("<src/diffraction/accumulation_optix.cuh>", text)
            self.assertNotIn("<utd/", text)


if __name__ == "__main__":
    unittest.main()
