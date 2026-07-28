from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_HEADER = ROOT / "include/rayd/torch/scene/cache.h"
CACHE_SOURCE = ROOT / "src/scene/scene.cpp"
TYPES_SOURCE = ROOT / "python/rayd/_impl/geometry.py"


class TorchGlobalGeometryContractTests(unittest.TestCase):
    def test_native_helper_is_declared_for_scene_handles(self) -> None:
        header = CACHE_HEADER.read_text(encoding="utf-8")
        self.assertIn(
            "std::vector<at::Tensor> scene_global_geometry(c10::intrusive_ptr<SceneHandle> scene);",
            header,
        )

    def test_native_result_matches_python_six_field_order(self) -> None:
        source = CACHE_SOURCE.read_text(encoding="utf-8")
        body_match = re.search(
            r"std::vector<at::Tensor> scene_global_geometry\(.*?\n\}",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(body_match)
        body = body_match.group(0)
        expected = [
            "scene.global_vertices",
            "scene.global_faces",
            "face_normal",
            "scene.face_shape_id",
            "scene.face_local_id",
            "global_prim_id",
        ]
        return_body = body.split("return {", 1)[1]
        positions = [return_body.index(field) for field in expected]
        self.assertEqual(positions, sorted(positions))

        types_source = TYPES_SOURCE.read_text(encoding="utf-8")
        dataclass_match = re.search(
            r"class SceneGlobalGeometry:\s*"
            r"vertices: torch\.Tensor\s*faces: torch\.Tensor\s*"
            r"face_normal: torch\.Tensor\s*shape_id: torch\.Tensor\s*"
            r"local_prim_id: torch\.Tensor\s*global_prim_id: torch\.Tensor",
            types_source,
        )
        self.assertIsNotNone(dataclass_match)

    def test_normals_and_global_primitive_ids_are_scene_global(self) -> None:
        source = CACHE_SOURCE.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r"at::stack\(\s*\{scene\.tri_fn_x, scene\.tri_fn_y, scene\.tri_fn_z\}, 1\)",
        )
        self.assertIn("face_normal.square().sum(1, true)", source)
        self.assertRegex(
            source,
            r"at::where\(\s*squared_normal\.gt\(0\.0f\), face_normal \* inverse_normal,\s*"
            r"at::zeros_like\(face_normal\)\)",
        )
        self.assertRegex(
            source,
            r"at::arange\(\s*scene\.global_faces\.size\(0\), scene\.face_local_id\.options\(\)\)",
        )


if __name__ == "__main__":
    unittest.main()
