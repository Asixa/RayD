import ast
import hashlib
import json
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "shared" / "contracts"
MANIFEST_PATH = CONTRACT_DIR / "public_api.json"
SCHEMA_PATH = CONTRACT_DIR / "public_api.schema.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


class PublicApiManifestTests(unittest.TestCase):
    def test_manifest_matches_schema_enums_and_required_fields(self):
        self.assertEqual(MANIFEST["version"], 1)
        self.assertEqual(
            set(SCHEMA["required"]),
            {
                "version",
                "capability_keys",
                "stability_levels",
                "naming_conventions",
                "apis",
                "aliases",
                "backends",
            },
        )
        categories = {"core", "multipath", "surfel", "experimental"}
        stability = set(MANIFEST["stability_levels"])
        self.assertEqual(stability, {"stable", "provisional", "experimental", "deprecated"})
        self.assertEqual(set(MANIFEST["apis"]), set(MANIFEST["capability_keys"]))
        for name, metadata in MANIFEST["apis"].items():
            with self.subTest(api=name):
                self.assertIn(metadata["category"], categories)
                self.assertIn(metadata["stability"], stability)
            self.assertTrue(metadata["summary"])

        naming = MANIFEST["naming_conventions"]
        self.assertEqual(set(naming), {"options", "results", "fields"})
        self.assertIn("<Operation>Options", naming["options"])
        self.assertIn("PascalCase", naming["results"])
        self.assertIn("global_", naming["fields"])

    def test_backend_capabilities_are_complete_and_boolean(self):
        required = set(MANIFEST["capability_keys"])
        operations = json.loads(
            (CONTRACT_DIR / "operations.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(operations["required_capability_keys"]), {"backend"} | required)
        for backend in ("drjit", "torch"):
            entry = MANIFEST["backends"][backend]
            self.assertEqual(set(entry["capabilities"]), required)
            self.assertTrue(all(type(value) is bool for value in entry["capabilities"].values()))
            self.assertEqual(entry["typing"], "partial")

    def test_runtime_modules_are_validated_copies_of_shared_manifest(self):
        schema_hash = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
        for backend in ("drjit", "torch"):
            module_path = (
                ROOT / "backends" / backend / "python" / "rayd" / backend / "_capabilities.py"
            )
            namespace = runpy.run_path(str(module_path))
            flat = namespace["backend_capabilities"]()
            rich = namespace["api_manifest"]()
            self.assertEqual(flat["backend"], backend)
            self.assertEqual(
                {key: value for key, value in flat.items() if key != "backend"},
                MANIFEST["backends"][backend]["capabilities"],
            )
            self.assertEqual(rich["schema_sha256"], schema_hash)
            self.assertEqual(rich["typing"], MANIFEST["backends"][backend]["typing"])
            self.assertEqual(rich["naming_conventions"], MANIFEST["naming_conventions"])
            for name, metadata in rich["apis"].items():
                self.assertEqual(metadata["category"], MANIFEST["apis"][name]["category"])
                self.assertEqual(metadata["stability"], MANIFEST["apis"][name]["stability"])
            self.assertEqual(rich["aliases"], MANIFEST["aliases"])

    def test_hybrid_is_only_a_deprecated_compatibility_alias(self):
        aliases = MANIFEST["aliases"]["edge_bvh_backend"]
        self.assertEqual(aliases["hybrid"]["canonical"], "optix_drjit")
        self.assertEqual(aliases["hybrid"]["stability"], "deprecated")
        self.assertIn("unrelated", aliases["hybrid"]["summary"])

    def test_partial_typing_markers_and_stubs_are_shipped(self):
        for backend in ("drjit", "torch"):
            package = ROOT / "backends" / backend / "python" / "rayd" / backend
            self.assertEqual((package / "py.typed").read_text(encoding="utf-8").strip(), "partial")
            stub = (package / "_capabilities.pyi").read_text(encoding="utf-8")
            self.assertIn("def backend_capabilities()", stub)
            self.assertIn("def api_manifest()", stub)
            for stub_path in package.glob("*.pyi"):
                ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))

        torch_package = ROOT / "backends" / "torch" / "python" / "rayd" / "torch"
        public_stub = (torch_package / "__init__.pyi").read_text(encoding="utf-8")
        for name in (
            "NearestEdgesTopK",
            "SegmentPairVisibility",
            "AxialEdgeVisibility",
            "SegmentChainVisibility",
        ):
            self.assertIn(name, public_stub)


if __name__ == "__main__":
    unittest.main()
