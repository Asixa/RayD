import hashlib
import json
import os
import posixpath
import unittest
import zipfile
from email.parser import BytesParser
from pathlib import Path


class WheelLayoutTests(unittest.TestCase):
    SOURCE_PREFIX = "rayd/torch/_source/"
    INTEGRATION_HEADERS = {
        "include/rayd/integration/torch.h",
        "include/rayd/diffraction/torch.h",
        "include/rayd/field_transport/torch_ad.cuh",
        "include/rayd/penetration/torch.h",
        "include/rayd/reflection/torch.h",
        "include/rayd/scattering/torch.h",
        "include/rayd/scene/torch.h",
        "include/rayd/transmission/torch.h",
        "include/rayd/visibility/torch.h",
    }
    @classmethod
    def setUpClass(cls):
        meta = os.environ.get("RAYD_META_WHEEL")
        drjit = os.environ.get("RAYD_DRJIT_WHEEL")
        torch = os.environ.get("RAYD_TORCH_WHEEL")
        if not meta or not drjit or not torch:
            raise unittest.SkipTest("wheel paths are provided by the packaging job")
        cls.meta_wheel = Path(meta)
        cls.drjit_wheel = Path(drjit)
        cls.torch_wheel = Path(torch)

    @staticmethod
    def names(path):
        with zipfile.ZipFile(path) as wheel:
            return set(wheel.namelist())

    @staticmethod
    def metadata(path):
        with zipfile.ZipFile(path) as wheel:
            metadata_path = next(
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            return BytesParser().parsebytes(wheel.read(metadata_path))

    @staticmethod
    def sha256(content):
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def normalized_text_sha256(cls, content):
        normalized = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return cls.sha256(normalized)

    @classmethod
    def header_set_sha256(cls, headers):
        digest = hashlib.sha256()
        for header in sorted(headers, key=lambda item: item["path"]):
            digest.update(header["path"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(header["sha256"].encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def test_namespace_root_is_implicit(self):
        for wheel in (self.drjit_wheel, self.torch_wheel):
            self.assertNotIn("rayd/__init__.py", self.names(wheel))

    def test_backend_files_are_disjoint(self):
        drjit = {
            name
            for name in self.names(self.drjit_wheel)
            if name.startswith("rayd/") and not name.endswith("/")
        }
        torch = {
            name
            for name in self.names(self.torch_wheel)
            if name.startswith("rayd/") and not name.endswith("/")
        }
        drjit_impl = {
            "rayd/_impl/runtime_jit.py",
            "rayd/_impl/capabilities_jit.py",
            "rayd/_impl/path_exchange_jit.py",
        }
        torch_impl = {
            "rayd/_impl/runtime.py",
            "rayd/_impl/capabilities.py",
            "rayd/_impl/path_exchange.py",
            "rayd/_impl/geometry.py",
            "rayd/_impl/scene.py",
            "rayd/_impl/multi.py",
            "rayd/_impl/multipath.py",
            "rayd/_impl/camera.py",
            "rayd/_impl/sdf.py",
        }
        self.assertTrue(drjit)
        self.assertTrue(torch)
        self.assertFalse(drjit & torch)
        self.assertTrue(drjit_impl <= drjit)
        self.assertTrue(torch_impl <= torch)
        self.assertEqual(
            {name for name in drjit if not name.startswith("rayd/drjit/")},
            drjit_impl,
        )
        self.assertEqual(
            {name for name in torch if not name.startswith("rayd/torch/")},
            torch_impl,
        )
        self.assertNotIn("rayd/_impl/__init__.py", drjit | torch)
    def test_torch_wheel_contains_untagged_stable_abi_library(self):
        names = self.names(self.torch_wheel)
        stable = [
            name
            for name in names
            if name.startswith("rayd/torch/_stable_ops")
            and name.endswith((".dll", ".so", ".dylib"))
        ]
        self.assertEqual(len(stable), 1, stable)
        self.assertNotRegex(stable[0], r"cp3(?:10|11|12|13|14)")

    def test_torch_wheel_contains_integrity_described_source_bundle(self):
        prefix = self.SOURCE_PREFIX
        with zipfile.ZipFile(self.torch_wheel) as wheel:
            names = {
                name
                for name in wheel.namelist()
                if name.startswith(prefix) and not name.endswith("/")
            }
            metadata_name = f"{prefix}rayd-source.json"
            manifest_name = f"{prefix}source-files.json"
            self.assertIn(metadata_name, names)
            self.assertIn(manifest_name, names)

            metadata = json.loads(wheel.read(metadata_name))
            manifest_bytes = wheel.read(manifest_name)
            manifest = json.loads(manifest_bytes)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["source_root"], "source")
            self.assertEqual(
                metadata["source_manifest"]["path"], "source-files.json"
            )
            self.assertEqual(
                metadata["source_manifest"]["sha256"],
                self.sha256(manifest_bytes),
            )
            self.assertEqual(manifest["schema_version"], 1)

            entries = manifest["files"]
            paths = [entry["path"] for entry in entries]
            self.assertEqual(len(paths), len(set(paths)))
            manifest_by_path = {entry["path"]: entry for entry in entries}
            for source_path in paths:
                with self.subTest(source_path=source_path):
                    self.assertEqual(source_path, posixpath.normpath(source_path))
                    self.assertFalse(source_path.startswith(("/", "../")))
                    self.assertNotIn("\\", source_path)
                    member = f"{prefix}source/{source_path}"
                    self.assertIn(member, names)
                    self.assertEqual(
                        manifest_by_path[source_path]["sha256"],
                        self.sha256(wheel.read(member)),
                    )

            expected_names = {
                metadata_name,
                manifest_name,
                *(f"{prefix}source/{source_path}" for source_path in paths),
            }
            self.assertEqual(names, expected_names)

            integration = metadata["integration_abi"]
            self.assertEqual(integration["kind"], "source-header-set-sha256")
            self.assertEqual(
                integration["entrypoint"], "include/rayd/integration/torch.h"
            )
            self.assertEqual(integration["api_version"], 7)
            self.assertEqual(integration["identity"], "rayd.torch.integration")
            headers = integration["headers"]
            header_paths = [header["path"] for header in headers]
            self.assertEqual(header_paths, sorted(header_paths))
            self.assertEqual(set(header_paths), self.INTEGRATION_HEADERS)
            self.assertEqual(len(headers), len(self.INTEGRATION_HEADERS))
            for header in headers:
                with self.subTest(integration_header=header["path"]):
                    self.assertIn(header["path"], manifest_by_path)
                    content = wheel.read(f"{prefix}source/{header['path']}")
                    self.assertEqual(
                        header["sha256"],
                        self.normalized_text_sha256(content),
                    )
            self.assertEqual(
                integration["sha256"], self.header_set_sha256(headers)
            )

            forbidden = ("/.git/", "/__pycache__/", ".obj", ".pdb", ".pyc")
            self.assertFalse(
                any(token in name for name in names for token in forbidden)
            )

    def test_torch_wheel_separates_legacy_dispatcher_and_compatibility_shim(self):
        names = self.names(self.torch_wheel)
        legacy = [
            name
            for name in names
            if name.startswith("rayd/torch/_legacy_ops")
            and name.endswith((".dll", ".so", ".dylib"))
        ]
        compat = [
            name
            for name in names
            if name.startswith("rayd/torch/_C")
            and name.endswith((".pyd", ".so"))
        ]
        self.assertEqual(len(legacy), 1, legacy)
        self.assertEqual(len(compat), 1, compat)

    def test_backend_wheels_include_complete_typing_metadata(self):
        """Both wheels stay PEP 561 typed distributions.

        The backend packages are typed inline, so a module's `.py` is its own
        type surface and `py.typed` is what lets a downstream type checker read
        it. Two stubs still ship, both from the Dr.Jit wheel:
        `rayd/drjit/_C.pyi`, because the nanobind extension it describes has no
        Python source to annotate, and `rayd/drjit/__init__.pyi`, which shields
        a type checker from Dr.Jit 1.3.1's syntactically invalid own stub (see
        `DRJIT_TOP_LEVEL_STUB` in tests/test_public_api_manifest.py).
        """
        drjit = self.names(self.drjit_wheel)
        torch = self.names(self.torch_wheel)
        self.assertIn("rayd/drjit/py.typed", drjit)
        self.assertIn("rayd/drjit/_C.pyi", drjit)
        self.assertIn("rayd/drjit/__init__.pyi", drjit)
        self.assertIn("rayd/torch/py.typed", torch)
        for name in ("__init__.py", "path_exchange.py", "py.typed"):
            self.assertIn(f"rayd/drjit/{name}", drjit)
        for name in ("__init__.py", "path_exchange.py", "py.typed"):
            self.assertIn(f"rayd/torch/{name}", torch)
        # No other shipped stub may shadow an inline-annotated module: a stale
        # stub silently wins over the annotations next to it.
        self.assertEqual(
            {name for name in drjit if name.endswith(".pyi")},
            {"rayd/drjit/_C.pyi", "rayd/drjit/__init__.pyi"},
        )
        self.assertEqual({name for name in torch if name.endswith(".pyi")}, set())

    def test_meta_wheel_is_file_free_and_pins_both_backends(self):
        self.assertFalse(any(name.startswith("rayd/") for name in self.names(self.meta_wheel)))
        meta = self.metadata(self.meta_wheel)
        drjit = self.metadata(self.drjit_wheel)
        torch = self.metadata(self.torch_wheel)
        self.assertEqual(meta["Version"], drjit["Version"])
        self.assertEqual(meta["Version"], torch["Version"])
        self.assertEqual(
            set(meta.get_all("Requires-Dist", [])),
            {f"rayd-drjit=={meta['Version']}", f"rayd-torch=={meta['Version']}"},
        )


if __name__ == "__main__":
    unittest.main()
