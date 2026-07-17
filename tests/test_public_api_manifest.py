import ast
import hashlib
import json
import runpy
import re
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
        self.assertEqual(MANIFEST["version"], 2)
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
                "trace",
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
            self.assertEqual(entry["typing"], "complete")

    def test_runtime_modules_are_validated_copies_of_shared_manifest(self):
        schema_hash = hashlib.sha256(
            MANIFEST_PATH.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
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
            self.assertEqual(rich["version"], MANIFEST["version"])
            self.assertEqual(rich["schema_sha256"], schema_hash)
            self.assertEqual(rich["typing"], MANIFEST["backends"][backend]["typing"])
            self.assertEqual(rich["naming_conventions"], MANIFEST["naming_conventions"])
            for name, metadata in rich["apis"].items():
                self.assertEqual(metadata["category"], MANIFEST["apis"][name]["category"])
                self.assertEqual(metadata["stability"], MANIFEST["apis"][name]["stability"])
            self.assertEqual(rich["aliases"], MANIFEST["aliases"])
            self.assertEqual(rich["trace"], MANIFEST["trace"])

    def test_trace_axis_records_the_optix_and_cuda_backends(self):
        trace = MANIFEST["trace"]
        self.assertEqual(set(trace), {"backends", "integration_modes", "frontend_support"})
        self.assertEqual(set(trace["backends"]), {"optix", "cuda"})
        self.assertEqual(trace["backends"]["optix"]["stability"], "stable")
        self.assertTrue(trace["backends"]["optix"]["summary"])
        self.assertEqual(trace["backends"]["cuda"]["stability"], "provisional")
        self.assertTrue(trace["backends"]["cuda"]["summary"])
        self.assertEqual(trace["integration_modes"], ["jit_symbolic", "eager_native"])
        # The CUDA backend is eager-native only: it never folds into a Dr.Jit
        # symbolic megakernel, and it has no Torch frontend in this phase.
        self.assertEqual(trace["frontend_support"], {
            "drjit": {"optix": ["jit_symbolic", "eager_native"], "cuda": ["eager_native"]},
            "torch": {"optix": ["eager_native"]},
        })

    def test_hybrid_is_only_a_deprecated_compatibility_alias(self):
        aliases = MANIFEST["aliases"]["edge_bvh_backend"]
        self.assertEqual(aliases["hybrid"]["canonical"], "optix_drjit")
        self.assertEqual(aliases["hybrid"]["stability"], "deprecated")
        self.assertIn("unrelated", aliases["hybrid"]["summary"])

    def test_complete_typing_markers_and_stubs_are_shipped(self):
        for backend in ("drjit", "torch"):
            package = ROOT / "backends" / backend / "python" / "rayd" / backend
            self.assertEqual((package / "py.typed").read_text(encoding="utf-8"), "")
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

        drjit_package = ROOT / "backends" / "drjit" / "python" / "rayd" / "drjit"
        drjit_public_stub = (drjit_package / "__init__.pyi").read_text(encoding="utf-8")
        self.assertIn("from ._C import *", drjit_public_stub)

    def test_torch_top_level_stub_reexports_runtime_all(self):
        package = ROOT / "backends" / "torch" / "python" / "rayd" / "torch"
        runtime = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
        stub = ast.parse((package / "__init__.pyi").read_text(encoding="utf-8"))
        all_node = next(
            node for node in runtime.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )
        runtime_exports = {
            element.value for element in all_node.value.elts if isinstance(element, ast.Constant)
        }
        stub_exports = {
            alias.asname or alias.name
            for node in stub.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertEqual(runtime_exports, stub_exports)

    def test_drjit_native_stub_covers_bound_public_symbols(self):
        source = (ROOT / "backends" / "drjit" / "src" / "rayd.cpp").read_text(
            encoding="utf-8"
        )
        stub_path = (
            ROOT / "backends" / "drjit" / "python" / "rayd" / "drjit" / "_C.pyi"
        )
        tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
        stub_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        stub_names.update(
            node.target.id
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        )
        bound_names = set(
            re.findall(r'nb::(?:class_|enum_)<[^;\n]*?\(m,\s*"([A-Za-z0-9_]+)"', source)
        )
        bound_names.update(re.findall(r'\bm\.(?:def|attr)\("([A-Za-z0-9_]+)"', source))
        bound_names.discard("__name__")
        self.assertFalse(bound_names - stub_names, sorted(bound_names - stub_names))

    def test_drjit_key_classes_have_typed_members(self):
        stub_path = (
            ROOT / "backends" / "drjit" / "python" / "rayd" / "drjit" / "_C.pyi"
        )
        tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }

        def members(name):
            node = classes[name]
            result = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            result.update(
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
            )
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in classes:
                    result.update(members(base.id))
            return result

        required = {
            "Mesh": {
                "vertex_positions", "face_indices", "to_world", "build",
                "set_transform", "append_transform", "secondary_edges",
            },
            "Scene": {
                "intersect", "nearest_edge", "nearest_edges", "set_edge_mask",
                "visible", "visible_pair", "visible_edge", "visible_chain",
                "trace_reflections", "trace_refl_epc_field", "trace_dfr_paths",
                "accumulate_reflections", "accum_dfr_direct", "accum_dfr",
            },
            "ReflectionTraceOptions": {
                "deduplicate", "canonical_prim_table", "export_mode", "return_trailing",
            },
            "DfrOptions": {
                "strategy_mask", "sample_sequence", "receiver_model", "max_order",
            },
            "Intersection": {"is_valid", "t", "p", "global_prim_id"},
            "NearestEdgesTopK": {"query_count", "k", "distances", "global_edge_ids"},
            "ReflectionChain": {"is_valid", "bounce_count", "global_prim_ids"},
        }
        for class_name, expected in required.items():
            with self.subTest(class_name=class_name):
                self.assertLessEqual(expected, members(class_name))

    def test_torch_public_python_modules_match_their_stubs(self):
        package = ROOT / "backends" / "torch" / "python" / "rayd" / "torch"
        for stem in ("autograd", "camera", "mesh", "path_exchange", "scene", "types"):
            source_tree = ast.parse((package / f"{stem}.py").read_text(encoding="utf-8"))
            stub_tree = ast.parse((package / f"{stem}.pyi").read_text(encoding="utf-8"))
            source_defs = {
                node.name: node
                for node in source_tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
            }
            stub_defs = {
                node.name: node
                for node in stub_tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
            }
            with self.subTest(module=stem):
                self.assertLessEqual(set(source_defs), set(stub_defs))
            for name, source_node in source_defs.items():
                if not isinstance(source_node, ast.ClassDef):
                    continue
                source_methods = {
                    child.name
                    for child in source_node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not child.name.startswith("_")
                }
                stub_node = stub_defs[name]
                stub_methods = {
                    child.name
                    for child in stub_node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not child.name.startswith("_")
                }
                with self.subTest(module=stem, class_name=name):
                    self.assertLessEqual(source_methods, stub_methods)
                source_fields = {
                    child.target.id
                    for child in source_node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and not child.target.id.startswith("_")
                }
                stub_fields = {
                    child.target.id
                    for child in stub_node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and not child.target.id.startswith("_")
                }
                with self.subTest(module=stem, class_name=name, surface="fields"):
                    self.assertLessEqual(source_fields, stub_fields)


if __name__ == "__main__":
    unittest.main()
