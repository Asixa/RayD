# Copyright Xingyu Chen.
# Tests shared headers.

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SharedHeaderTests(unittest.TestCase):
    def test_utd_headers_have_one_canonical_copy(self):
        canonical = ROOT / "include" / "rayd" / "diffraction"
        self.assertTrue((canonical / "utd.h").is_file())
        self.assertTrue((canonical / "utd_types.h").is_file())
        self.assertFalse((ROOT / "include" / "rayd" / "utd").exists())

    def test_backends_use_canonical_include(self):
        diffraction = ROOT / "include" / "rayd" / "diffraction"
        shared_algo = (diffraction / "accumulation_algo.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("<rayd/diffraction/utd.h>", shared_algo)
        self.assertNotIn("<utd/", shared_algo)
        shared_device = (diffraction / "accumulation_optix_device.cuh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("<utd/", shared_device)

        sources = [
            ROOT / "src" / "diffraction" / "accumulation_optix_jit.cu",
            ROOT / "src" / "diffraction" / "accumulation_optix.cu",
        ]
        for source in sources:
            text = source.read_text(encoding="utf-8")
            self.assertIn(
                "<rayd/diffraction/accumulation_optix_device.cuh>",
                text,
            )
            self.assertNotIn("<utd/", text)


if __name__ == "__main__":
    unittest.main()
