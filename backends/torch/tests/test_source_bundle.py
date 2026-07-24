import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "backends" / "torch" / "scripts" / "generate_source_bundle.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceBundleTests(unittest.TestCase):
    def test_bundle_is_relocatable_complete_and_integrity_described(self):
        spec = importlib.util.spec_from_file_location("rayd_source_bundle", SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle with spaces"
            module.generate(
                ROOT,
                output,
                distribution_version="0.7.0",
                commit="1" * 40,
                repository_url="https://github.com/Asixa/RayD.git",
            )
            metadata = json.loads((output / "rayd-source.json").read_text())
            manifest_path = output / metadata["source_manifest"]["path"]
            manifest = json.loads(manifest_path.read_text())
            source_root = output / metadata["source_root"]

            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(
                metadata["distribution"],
                {"name": "rayd-torch", "version": "0.7.0"},
            )
            self.assertEqual(metadata["commit"], "1" * 40)
            self.assertEqual(metadata["source_manifest"]["sha256"], sha256(manifest_path))
            self.assertEqual(metadata["integration_abi"]["api_version"], 6)
            self.assertEqual(metadata["integration_abi"]["identity"], "rayd.torch.integration")

            described = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
            actual = {
                path.relative_to(source_root).as_posix(): sha256(path)
                for path in source_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(described, actual)
            self.assertIn("backends/torch/CMakeLists.txt", actual)
            self.assertIn("backends/torch/include/rayd/torch/integration.h", actual)
            self.assertTrue(any(path.startswith("backends/torch/src/") for path in actual))
            self.assertTrue(any(path.startswith("shared/include/") for path in actual))
            self.assertTrue(any(path.startswith("shared/src/") for path in actual))
            self.assertFalse(any("/.git/" in f"/{path}/" for path in actual))
            self.assertFalse(any("__pycache__" in path for path in actual))
            self.assertFalse(any(Path(path).is_absolute() for path in actual))

    def test_cmake_installs_fixed_passive_metadata_location(self):
        cmake = (ROOT / "backends" / "torch" / "CMakeLists.txt").read_text()
        self.assertIn("RAYD_TORCH_INSTALL_SOURCE_BUNDLE", cmake)
        self.assertIn("scripts/generate_source_bundle.py", cmake)
        self.assertIn("DESTINATION rayd/torch/_source", cmake)


if __name__ == "__main__":
    unittest.main()
