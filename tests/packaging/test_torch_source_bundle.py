import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "torch" / "scripts" / "generate_source_bundle.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def header_set_sha256(headers: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for header in sorted(headers, key=lambda item: item["path"]):
        digest.update(header["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(header["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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

            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(
                metadata["distribution"],
                {"name": "rayd-torch", "version": "0.7.0"},
            )
            self.assertEqual(metadata["commit"], "1" * 40)
            self.assertEqual(metadata["source_manifest"]["sha256"], sha256(manifest_path))
            integration_abi = metadata["integration_abi"]
            self.assertEqual(integration_abi["kind"], "source-header-set-sha256")
            self.assertEqual(integration_abi["api_version"], 7)
            self.assertEqual(integration_abi["identity"], "rayd.torch.integration")
            self.assertEqual(
                integration_abi["entrypoint"], "include/rayd/integration/torch.h"
            )
            self.assertEqual(
                {header["path"] for header in integration_abi["headers"]},
                {
                    "include/rayd/integration/torch.h",
                    "include/rayd/diffraction/torch.h",
                    "include/rayd/field_transport/torch_ad.cuh",
                    "include/rayd/penetration/torch.h",
                    "include/rayd/reflection/torch.h",
                    "include/rayd/scattering/torch.h",
                    "include/rayd/scene/torch.h",
                    "include/rayd/transmission/torch.h",
                    "include/rayd/visibility/torch.h",
                },
            )
            self.assertEqual(integration_abi["sha256"], header_set_sha256(integration_abi["headers"]))
            for header in integration_abi["headers"]:
                self.assertEqual(
                    header["sha256"], normalized_text_sha256(source_root / header["path"])
                )

            described = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
            actual = {
                path.relative_to(source_root).as_posix(): sha256(path)
                for path in source_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(described, actual)
            self.assertIn("torch/CMakeLists.txt", actual)
            self.assertIn("include/rayd/integration/torch.h", actual)
            self.assertTrue(any(path.startswith("src/") for path in actual))
            self.assertTrue(any(path.startswith("include/") for path in actual))
            self.assertTrue(any(path.startswith("cmake/") for path in actual))
            self.assertFalse(any(path.startswith("backends/") for path in actual))
            self.assertFalse(any(path.startswith("shared/") for path in actual))
            self.assertFalse(any("/.git/" in f"/{path}/" for path in actual))
            self.assertFalse(any("__pycache__" in path for path in actual))
            self.assertFalse(any(Path(path).is_absolute() for path in actual))

    def test_cmake_installs_fixed_passive_metadata_location(self):
        cmake = (ROOT / "torch" / "CMakeLists.txt").read_text()
        self.assertIn("RAYD_TORCH_INSTALL_SOURCE_BUNDLE", cmake)
        self.assertIn("scripts/generate_source_bundle.py", cmake)
        self.assertIn("DESTINATION rayd/torch/_source", cmake)


if __name__ == "__main__":
    unittest.main()
