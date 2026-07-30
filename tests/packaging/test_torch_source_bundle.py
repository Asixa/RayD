# Copyright Xingyu Chen.
# Tests torch source bundle.

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.support.hashing import header_set_sha256, normalized_text_sha256, sha256_file as sha256


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "torch" / "scripts" / "generate_source_bundle.py"


class SourceBundleTests(unittest.TestCase):
    @staticmethod
    def _load_generator():
        spec = importlib.util.spec_from_file_location("rayd_source_bundle", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load Torch source-bundle generator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _git(repository: Path, *arguments: str) -> None:
        subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _minimal_git_workspace(self, module, root: Path) -> Path:
        workspace = root / "workspace"
        workspace.mkdir()
        for relative in module.SOURCE_INPUTS:
            path = workspace / relative
            if relative in {"src", "cmake"}:
                path.mkdir(parents=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            content = "placeholder\n"
            if relative == module.INTEGRATION_ABI_PATH:
                content = (
                    "inline constexpr int kIntegrationApiVersion = 8;\n"
                    'inline constexpr const char* kIntegrationIdentity = "rayd.torch.integration";\n'
                )
            path.write_text(content, encoding="utf-8", newline="\n")
        (workspace / "src" / "bundled.cpp").write_text("tracked bundle source\n", encoding="utf-8", newline="\n")
        (workspace / "cmake" / "bundled.cmake").write_text("set(BUNDLED ON)\n", encoding="utf-8", newline="\n")

        self._git(workspace, "init")
        self._git(workspace, "config", "user.name", "RayD Test")
        self._git(workspace, "config", "user.email", "rayd-test@example.invalid")
        self._git(workspace, "remote", "add", "origin", "https://github.com/Asixa/RayD.git")
        self._git(workspace, "add", "--all")
        self._git(workspace, "commit", "-m", "fixture")
        return workspace

    @staticmethod
    def _generate_metadata(module, workspace: Path, output: Path) -> dict:
        module.generate(workspace, output, distribution_version="0.8.0", commit=None, repository_url=None)
        return json.loads((output / "rayd-source.json").read_text(encoding="utf-8"))

    def test_bundle_is_relocatable_complete_and_integrity_described(self):
        module = self._load_generator()

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle with spaces"
            module.generate(
                ROOT,
                output,
                distribution_version="0.8.0",
                commit="1" * 40,
                repository_url="https://github.com/Asixa/RayD.git",
            )
            metadata = json.loads((output / "rayd-source.json").read_text())
            manifest_path = output / metadata["source_manifest"]["path"]
            manifest = json.loads(manifest_path.read_text())
            source_root = output / metadata["source_root"]

            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["distribution"], {"name": "rayd-torch", "version": "0.8.0"})
            self.assertEqual(metadata["commit"], "1" * 40)
            self.assertEqual(metadata["source_manifest"]["sha256"], sha256(manifest_path))
            integration_abi = metadata["integration_abi"]
            self.assertEqual(integration_abi["kind"], "source-header-set-sha256")
            self.assertEqual(integration_abi["api_version"], 8)
            self.assertEqual(integration_abi["identity"], "rayd.torch.integration")
            self.assertEqual(integration_abi["entrypoint"], "include/rayd/integration.h")
            self.assertEqual(
                {header["path"] for header in integration_abi["headers"]},
                {
                    "include/rayd/integration.h",
                    "include/rayd/diffraction.h",
                    "include/rayd/penetration.h",
                    "include/rayd/reflection.h",
                    "include/rayd/scattering.h",
                    "include/rayd/scene.h",
                    "include/rayd/transmission.h",
                    "include/rayd/visibility.h",
                },
            )
            self.assertEqual(integration_abi["sha256"], header_set_sha256(integration_abi["headers"]))
            for header in integration_abi["headers"]:
                self.assertEqual(header["sha256"], normalized_text_sha256((source_root / header["path"]).read_bytes()))

            described = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
            actual = {
                path.relative_to(source_root).as_posix(): sha256(path)
                for path in source_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(described, actual)
            expected_include = {
                "include/rayd/contracts.h",
                "include/rayd/diffraction.h",
                "include/rayd/field_transport.cuh",
                "include/rayd/integration.h",
                "include/rayd/math.h",
                "include/rayd/path_exchange.h",
                "include/rayd/penetration.h",
                "include/rayd/reflection.h",
                "include/rayd/scattering.h",
                "include/rayd/scattering_table.cuh",
                "include/rayd/scene.h",
                "include/rayd/transmission.h",
                "include/rayd/utd.h",
                "include/rayd/visibility.h",
            }
            expected_src = {path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*") if path.is_file()}
            self.assertEqual({path for path in actual if path.startswith("include/")}, expected_include)
            self.assertEqual({path for path in actual if path.startswith("src/")}, expected_src)
            self.assertIn("torch/CMakeLists.txt", actual)
            self.assertIn("include/rayd/integration.h", actual)
            self.assertIn("include/rayd/path_exchange.h", actual)
            self.assertIn("include/rayd/utd.h", actual)
            self.assertIn("src/field_transport_ad.cuh", actual)
            self.assertIn("src/edge/edge_bvh_jit.h", actual)
            self.assertFalse(any(path.startswith("include/rayd/jit/") for path in actual))
            self.assertTrue(any(path.startswith("src/") for path in actual))
            self.assertTrue(any(path.startswith("include/") for path in actual))
            self.assertTrue(any(path.startswith("cmake/") for path in actual))
            self.assertFalse(any(path.startswith("backends/") for path in actual))
            self.assertFalse(any(path.startswith("shared/") for path in actual))
            self.assertFalse(any("/.git/" in f"/{path}/" for path in actual))
            self.assertFalse(any("__pycache__" in path for path in actual))
            self.assertFalse(any(Path(path).is_absolute() for path in actual))

            bundled_cmake = (source_root / "torch" / "CMakeLists.txt").read_text()
            self.assertFalse((source_root / "torch" / "scripts" / "generate_source_bundle.py").exists())
            self.assertIn('if(EXISTS "${RAYD_TORCH_SOURCE_BUNDLE_GENERATOR}")', bundled_cmake)
            self.assertIn("set(RAYD_TORCH_INSTALL_SOURCE_BUNDLE_DEFAULT OFF)", bundled_cmake)
            self.assertIn("${RAYD_TORCH_INSTALL_SOURCE_BUNDLE_DEFAULT})", bundled_cmake)
            self.assertIn("RAYD_TORCH_INSTALL_SOURCE_BUNDLE=ON requires ", bundled_cmake)

    def test_dirty_identity_is_scoped_to_bundle_owned_git_paths(self):
        module = self._load_generator()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = self._minimal_git_workspace(module, root)
            (workspace / ".optix").mkdir()
            (workspace / ".optix" / "unrelated-cache.bin").write_bytes(b"cache")

            clean = self._generate_metadata(module, workspace, root / "bundle-clean")
            self.assertFalse(clean["dirty"], "an unrelated untracked .optix checkout must not dirty the bundle")

            tracked_source = workspace / "src" / "bundled.cpp"
            tracked_source.write_text("modified bundle source\n", encoding="utf-8", newline="\n")
            modified = self._generate_metadata(module, workspace, root / "bundle-modified")
            self.assertTrue(modified["dirty"], "a modified tracked bundle source must dirty the bundle")

            tracked_source.write_text("tracked bundle source\n", encoding="utf-8", newline="\n")
            (workspace / "src" / "untracked.cu").write_text(
                "// untracked bundle source\n", encoding="utf-8", newline="\n"
            )
            untracked = self._generate_metadata(module, workspace, root / "bundle-untracked")
            self.assertTrue(untracked["dirty"], "an untracked source covered by SOURCE_INPUTS must dirty the bundle")

    def test_cmake_installs_fixed_passive_metadata_location(self):
        cmake = (ROOT / "torch" / "CMakeLists.txt").read_text()
        self.assertTrue(SCRIPT.is_file())
        self.assertIn("RAYD_TORCH_INSTALL_SOURCE_BUNDLE", cmake)
        self.assertIn("set(RAYD_TORCH_INSTALL_SOURCE_BUNDLE_DEFAULT ON)", cmake)
        self.assertIn("scripts/generate_source_bundle.py", cmake)
        self.assertIn("DESTINATION rayd/torch/_source", cmake)


if __name__ == "__main__":
    unittest.main()
