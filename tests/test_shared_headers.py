import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SharedHeaderTests(unittest.TestCase):
    def test_utd_headers_have_one_canonical_copy(self):
        canonical = ROOT / "shared" / "include" / "rayd" / "shared" / "utd"
        self.assertTrue((canonical / "utd_math.h").is_file())
        self.assertTrue((canonical / "utd_types.h").is_file())
        self.assertFalse(list((ROOT / "backends" / "drjit" / "include" / "utd").glob("*.h")))
        self.assertFalse(list((ROOT / "backends" / "torch" / "include" / "utd").glob("*.h")))

    def test_backends_use_canonical_include(self):
        sources = [
            ROOT / "backends" / "drjit" / "src" / "multipath" / "diffraction_accumulation.cu",
            ROOT / "backends" / "torch" / "src" / "torch_ext" / "diffraction" / "accum_optix.cu",
        ]
        for source in sources:
            text = source.read_text(encoding="utf-8")
            self.assertIn("<rayd/shared/utd/utd_math.h>", text)
            self.assertNotIn("<utd/", text)


if __name__ == "__main__":
    unittest.main()
