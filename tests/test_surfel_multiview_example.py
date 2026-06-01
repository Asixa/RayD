import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "basics" / "surfel_multiview_color_fit.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("surfel_multiview_color_fit", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SurfelMultiviewColorExampleTests(unittest.TestCase):
    def test_degree_one_sh_color_uses_view_direction_coefficients(self):
        module = load_example_module()
        coeffs = np.zeros((2, 3, 4), dtype=np.float32)
        coeffs[:, :, 0] = 0.25
        coeffs[0, 0, 1] = 0.50
        coeffs[0, 1, 2] = 0.25
        coeffs[0, 2, 3] = -0.10
        coeffs[1, :, 1:] = 0.10
        view_dirs = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        rgb = module.evaluate_sh_rgb(coeffs, view_dirs, degree=1)

        np.testing.assert_allclose(rgb[0], [0.75, 0.25, 0.35], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(rgb[1], [0.35, 0.35, 0.35], rtol=0.0, atol=1e-6)

    def test_examples_readme_links_multiview_color_fit(self):
        readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
        self.assertIn("surfel_multiview_color_fit.py", readme)

    def test_convergence_frame_contains_three_panels(self):
        module = load_example_module()
        target = np.zeros((4, 4, 3), dtype=np.float32)
        pred = np.ones((4, 4, 3), dtype=np.float32) * 0.5

        frame = module.make_convergence_frame(target, pred, iteration=2, loss=0.25)

        self.assertEqual(frame.size, (4 * 3 + 32, 4 + 60))


if __name__ == "__main__":
    unittest.main()
