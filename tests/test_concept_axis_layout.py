# Copyright Xingyu Chen.
# Tests concept axis layout.

"""Checks the repository concept-oriented layout."""

import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
PRODUCTION_ROOTS = (ROOT / "src", ROOT / "include", ROOT / "python")


def tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


class ConceptAxisLayoutTests(unittest.TestCase):
    def test_retired_backend_container_is_absent(self):
        self.assertFalse((ROOT / "backends").exists())

    def test_agent_guides_are_synchronized(self):
        self.assertEqual((ROOT / "AGENTS.md").read_bytes(), (ROOT / "CLAUDE.md").read_bytes())

    def test_no_tracked_production_implementation_under_backends(self):
        offenders = []
        for path in tracked_files():
            rel = path.relative_to(ROOT)
            if not rel.parts or rel.parts[0] != "backends":
                continue
            if any(part in {"tests", "examples", "docs", "build", "scripts"} for part in rel.parts):
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

    def test_public_headers_use_flat_default_and_jit_surfaces(self):
        include_root = ROOT / "include" / "rayd"
        self.assertEqual(
            {path.name for path in include_root.iterdir() if path.is_file()},
            {
                "diffraction.h",
                "integration.h",
                "path_exchange.h",
                "penetration.h",
                "reflection.h",
                "scattering.h",
                "scene.h",
                "transmission.h",
                "visibility.h",
                "contracts.h",
                "field_transport.cuh",
                "math.h",
                "utd.h",
                "scattering_table.cuh",
            },
        )
        self.assertEqual({path.name for path in include_root.iterdir() if path.is_dir()}, {"jit"})

        jit_root = include_root / "jit"
        self.assertEqual([path for path in jit_root.iterdir() if path.is_dir()], [])
        self.assertEqual(
            {path.name for path in jit_root.iterdir() if path.is_file()},
            {
                "core.h",
                "diffraction.h",
                "edge.h",
                "mesh.h",
                "native_launch_audit.h",
                "optix.h",
                "reflection.h",
                "scene.h",
                "sdf.h",
                "surfel.h",
                "visibility.h",
            },
        )

        self.assertFalse((include_root / "detail").exists())
        self.assertTrue((include_root / "math.h").is_file())
        self.assertFalse(any(include_root.rglob("torch.h")))
        self.assertFalse(any(include_root.rglob("drjit.h")))
        self.assertFalse((include_root / "shared").exists())

    def test_public_headers_do_not_reach_private_sources(self):
        offenders = []
        for path in (ROOT / "include" / "rayd").rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".h", ".hpp", ".cuh"}:
                continue
            for target in re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', path.read_text(encoding="utf-8")):
                normalized = target.replace("\\", "/")
                if normalized.startswith("src/") or "/src/" in normalized or ".." in normalized.split("/"):
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}: {target}")
        self.assertEqual(offenders, [])

    def test_public_headers_are_not_forwarders(self):
        offenders = []
        for path in (ROOT / "include" / "rayd").rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".h", ".hpp", ".cuh"}:
                continue
            text = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
            text = re.sub(r"//.*", "", text)
            body = [
                line
                for line in text.splitlines()
                if line.strip()
                and not re.match(r"\s*#\s*pragma\s+once\b", line)
                and not re.match(r"\s*#\s*include\b", line)
            ]
            if not body:
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [], "public forwarding headers are forbidden")

    def test_src_directories_are_real_multifile_modules(self):
        native_suffixes = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp"}
        offenders = {}
        for directory in (ROOT / "src").iterdir():
            if not directory.is_dir():
                continue
            native_files = [
                path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in native_suffixes
            ]
            if native_files and len(native_files) < 3:
                offenders[directory.relative_to(ROOT).as_posix()] = sorted(path.name for path in native_files)
        self.assertEqual(offenders, {}, "one- and two-file native concepts belong directly under src/")

    def test_private_headers_have_multiple_production_consumers(self):
        native_suffixes = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp"}
        src_root = ROOT / "src"
        production_sources = [
            path for path in src_root.rglob("*") if path.is_file() and path.suffix.lower() in native_suffixes
        ]
        native_test_sources = [
            path
            for path in (ROOT / "tests" / "native").rglob("*")
            if path.is_file() and path.suffix.lower() in native_suffixes
        ]
        sources = production_sources + native_test_sources
        headers = [path for path in production_sources if path.suffix.lower() in {".h", ".hpp", ".cuh"}]
        basename_counts = {}
        for header in headers:
            basename_counts[header.name] = basename_counts.get(header.name, 0) + 1
        include_re = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')
        include_tokens = {
            source: {target.replace("\\", "/") for target in include_re.findall(source.read_text(encoding="utf-8"))}
            for source in sources
        }
        consumers = {}
        for header in headers:
            relative_root = header.relative_to(ROOT).as_posix()
            relative_src = header.relative_to(src_root).as_posix()
            spellings = {relative_root, relative_src}
            if basename_counts[header.name] == 1:
                spellings.add(header.name)
            consumers[header] = {
                source
                for source, tokens in include_tokens.items()
                if source != header and any(token in spellings or Path(token).name in spellings for token in tokens)
            }
        offenders = {
            path.relative_to(ROOT).as_posix(): len(users) for path, users in consumers.items() if len(users) < 2
        }
        self.assertEqual(offenders, {}, "single-consumer private headers must be folded into their consumer")

    def test_torch_backend_private_headers_are_concept_owned(self):
        namespace = "namespace rayd::torch_backend"
        private_headers = []
        for path in (ROOT / "src").rglob("*"):
            if path.is_file() and path.suffix.lower() in {".h", ".hpp", ".cuh"}:
                if namespace in path.read_text(encoding="utf-8"):
                    private_headers.append(path.relative_to(ROOT).as_posix())

        self.assertTrue(private_headers)
        self.assertFalse((ROOT / "include/rayd/torch").exists())

    def test_torch_generated_ptx_headers_are_concept_owned(self):
        cmake = (ROOT / "torch/CMakeLists.txt").read_text(encoding="utf-8")
        expected = {
            "rayd/scene/intersection_torch_ptx.h",
            "rayd/edge/point_ray_torch_ptx.h",
            "rayd/edge/topk_torch_ptx.h",
            "rayd/reflection/trace_torch_ptx.h",
            "rayd/reflection/epc_torch_ptx.h",
            "rayd/reflection/accumulation_torch_ptx.h",
            "rayd/visibility/segment_torch_ptx.h",
            "rayd/visibility/axial_edge_torch_ptx.h",
            "rayd/diffraction/paths_torch_ptx.h",
            "rayd/diffraction/accumulation_torch_ptx.h",
            "rayd/penetration/segment_torch_ptx.h",
        }
        actual = set(re.findall(r"generated/(rayd/[^\"]+_torch_ptx\.h)", cmake))
        self.assertEqual(actual, expected)
        self.assertNotIn("generated/rayd/torch", cmake)
        self.assertEqual(cmake.count('-I "${RAYD_ROOT_DIR}"'), 11)
        for path in (ROOT / "src").rglob("*"):
            if not path.is_file() or path.suffix not in PRODUCTION_SUFFIXES:
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"#include\s*<rayd/torch/[^>]*ptx\.h>")

    def test_shared_physical_sources_have_one_root_owner(self):
        expected = {
            "src/bvh_build_shared.cu",
            "src/bvh_triangle_query_shared.cu",
            "src/edge/edge_shared.cu",
            "src/reflection/dedup_shared.cu",
            "src/scene/packing_shared.cu",
        }
        actual = {path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*_shared.*") if path.is_file()}
        self.assertEqual(actual, expected)
        for rel in expected:
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_compile_policy_preserves_logical_tu_identity(self):
        contract = json.loads((ROOT / "contracts" / "compile_policy.json").read_text(encoding="utf-8"))
        units = contract["translation_units"]
        identities = {(entry["backend"], entry["unit"]) for entry in units}
        self.assertEqual(len(units), 81)
        self.assertEqual(len(identities), 81)
        self.assertEqual({entry["backend"] for entry in units}, {"drjit", "torch"})
        for entry in units:
            with self.subTest(unit=f"{entry['backend']}/{entry['unit']}"):
                self.assertTrue((ROOT / entry["source"]).is_file(), entry["source"])

    def test_public_python_frontends_share_one_namespace_source_root(self):
        namespace = ROOT / "python" / "rayd"
        self.assertFalse((namespace / "__init__.py").exists())
        self.assertFalse((ROOT / "drjit" / "python").exists())
        self.assertFalse((ROOT / "torch" / "python").exists())
        self.assertEqual(
            {path.name for path in (namespace / "drjit").iterdir() if path.is_file()},
            {"__init__.py", "__init__.pyi", "_C.pyi", "path_exchange.py", "py.typed"},
        )
        self.assertEqual(
            {path.name for path in (namespace / "torch").iterdir() if path.is_file()},
            {"__init__.py", "path_exchange.py", "py.typed"},
        )

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
