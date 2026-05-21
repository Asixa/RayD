import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectMetadataTests(unittest.TestCase):
    def test_torch_and_slang_frontends_are_not_shipped(self):
        removed_paths = [
            ROOT / "rayd" / "torch",
            ROOT / "rayd" / "slang",
            ROOT / "tests" / "torch",
            ROOT / "tests" / "slang",
            ROOT / "include" / "rayd" / "slang",
            ROOT / "include" / "rayd_slang.slang",
            ROOT / "src" / "slang_interop.cpp",
        ]

        for path in removed_paths:
            self.assertFalse(path.exists(), f"Unexpected frontend artifact remains: {path}")

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertNotIn("rayd.torch", readme)
        self.assertNotIn("Slang", readme)
        self.assertNotIn("torch =", pyproject)

    def test_core_camera_api_is_not_shipped(self):
        removed_paths = [
            ROOT / "include" / "rayd" / "camera.h",
            ROOT / "src" / "camera.cpp",
        ]

        for path in removed_paths:
            self.assertFalse(path.exists(), f"Unexpected Camera artifact remains: {path}")

        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        bindings = (ROOT / "src" / "rayd.cpp").read_text(encoding="utf-8")
        fwd = (ROOT / "include" / "rayd" / "rayd.h").read_text(encoding="utf-8")
        scene_header = (ROOT / "include" / "rayd" / "scene" / "scene.h").read_text(encoding="utf-8")

        self.assertNotIn("camera.h", cmake)
        self.assertNotIn("camera.cpp", cmake)
        self.assertNotIn("Camera", bindings)
        self.assertNotIn("PrimaryEdgeSample", bindings)
        self.assertNotIn("class Camera", fwd)
        self.assertNotIn("Camera *", scene_header)

    def test_cornell_renderer_uses_example_local_camera(self):
        renderer_camera = ROOT / "examples" / "renderer" / "camera.py"
        cornell_box = ROOT / "examples" / "renderer" / "cornell_box.py"

        self.assertTrue(renderer_camera.is_file(), "Cornell renderer should carry its own example-local camera.")
        renderer_source = cornell_box.read_text(encoding="utf-8")

        self.assertIn("from camera import ExampleCamera", renderer_source)
        self.assertNotIn("rd.Camera", renderer_source)

    def test_readme_matches_pinned_nanobind_version(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('nanobind==2.9.2', pyproject)
        self.assertIn('nanobind==2.9.2', readme)
        self.assertNotIn('nanobind==2.11.0', readme)

    def test_reflection_trace_ptx_header_is_committed(self):
        self.assertTrue(
            (
                ROOT
                / "include"
                / "rayd"
                / "multipath"
                / "reflection_trace_ptx.h"
            ).is_file(),
            "Expected committed reflection_trace PTX header for wheel builds.",
        )

    def test_reflection_epc_ptx_header_is_committed(self):
        self.assertTrue(
            (
                ROOT
                / "include"
                / "rayd"
                / "multipath"
                / "reflection_epc_ptx.h"
            ).is_file(),
            "Expected committed reflection_epc PTX header for wheel builds.",
        )


if __name__ == "__main__":
    unittest.main()
