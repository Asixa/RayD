"""Architecture gate for ADR-0039's concept-major repository layout."""

import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
PRODUCTION_ROOTS = (ROOT / "src", ROOT / "include", ROOT / "python")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


class ConceptAxisLayoutTests(unittest.TestCase):
    def test_retired_backend_container_is_absent(self):
        self.assertFalse((ROOT / "backends").exists())

    def test_agent_guides_are_synchronized(self):
        self.assertEqual(
            (ROOT / "AGENTS.md").read_bytes(),
            (ROOT / "CLAUDE.md").read_bytes(),
        )

    def test_no_tracked_production_implementation_under_backends(self):
        offenders = []
        for path in tracked_files():
            rel = path.relative_to(ROOT)
            if not rel.parts or rel.parts[0] != "backends":
                continue
            if any(part in {"tests", "examples", "docs", "build", "scripts"}
                   for part in rel.parts):
                continue
            if path.suffix.lower() in PRODUCTION_SUFFIXES:
                offenders.append(rel.as_posix())
        self.assertEqual(offenders, [], "production code must be concept-owned at root")

    def test_backend_frontends_do_not_own_native_implementation_trees(self):
        for frontend in (ROOT / "drjit", ROOT / "torch"):
            with self.subTest(frontend=frontend.name):
                self.assertFalse((frontend / "src").exists())
                self.assertFalse((frontend / "include").exists())

    def test_concept_variant_filename_convention(self):
        bad_backend_suffixes = []
        for root in PRODUCTION_ROOTS:
            for path in root.rglob("*"):
                if path.is_file() and re.search(r"_(?:torch|drjit)(?=\.)", path.name):
                    bad_backend_suffixes.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(bad_backend_suffixes, [])

    def test_rf_is_not_a_production_owner(self):
        rf_directories = []
        rf_namespaces = []
        namespace_re = re.compile(r"namespace\s+(?:::)?rayd(?:::\w+)*::rf\b")
        for root in PRODUCTION_ROOTS:
            for path in root.rglob("*"):
                if path.is_dir() and path.name == "rf":
                    rf_directories.append(path.relative_to(ROOT).as_posix())
                elif path.is_file() and path.suffix.lower() in PRODUCTION_SUFFIXES:
                    try:
                        text = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue
                    if namespace_re.search(text):
                        rf_namespaces.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(rf_directories, [])
        self.assertEqual(rf_namespaces, [])

    def test_shared_physical_sources_have_one_root_owner(self):
        expected = {
            "src/bvh/build_shared.cu",
            "src/bvh/triangle_query_shared.cu",
            "src/edge/edge_shared.cu",
            "src/reflection/dedup_shared.cu",
            "src/scene/packing_shared.cu",
        }
        actual = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "src").rglob("*_shared.*")
            if path.is_file()
        }
        self.assertEqual(actual, expected)
        for rel in expected:
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_compile_policy_preserves_logical_tu_identity(self):
        contract = json.loads((ROOT / "contracts" / "compile_policy.json").read_text(
            encoding="utf-8"))
        units = contract["translation_units"]
        identities = {(entry["backend"], entry["unit"]) for entry in units}
        self.assertEqual(len(units), 80)
        self.assertEqual(len(identities), 80)
        self.assertEqual({entry["backend"] for entry in units}, {"drjit", "torch"})
        for entry in units:
            with self.subTest(unit=f"{entry['backend']}/{entry['unit']}"):
                self.assertTrue((ROOT / entry["source"]).is_file(), entry["source"])

    def test_private_python_implementation_ownership_is_disjoint(self):
        implementation = ROOT / "python" / "rayd" / "_impl"
        self.assertFalse((implementation / "__init__.py").exists())
        files = {path.name for path in implementation.glob("*.py")}
        jit = {name for name in files if name.endswith("_jit.py")}
        torch_owned = files - jit
        self.assertTrue(jit)
        self.assertTrue(torch_owned)
        self.assertFalse(jit & torch_owned)


if __name__ == "__main__":
    unittest.main()