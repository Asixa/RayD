# Copyright Xingyu Chen.
# Tests multi device.

"""Checks the replicated multi-device contract against the implementation."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS_PATH = ROOT / "contracts" / "operations.json"
PUBLIC_API_PATH = ROOT / "contracts" / "public_api.json"
TORCH_PACKAGE = ROOT / "python" / "rayd" / "_impl"
MULTI_PATH = TORCH_PACKAGE / "multi.py"
SCENE_PATH = TORCH_PACKAGE / "scene.py"
AUTOGRAD_PATH = TORCH_PACKAGE / "multipath.py"
LIBRARY_PATH = ROOT / "src" / "bindings" / "library.cpp"
DIFFRACTION_OPS_PATH = ROOT / "src" / "diffraction" / "diffraction.cpp"
CAPABILITY_MODULES = {
    "drjit": ROOT / "python" / "rayd" / "_impl" / "capabilities_jit.py",
    "torch": ROOT / "python" / "rayd" / "_impl" / "capabilities.py",
}

CAPABILITY = "multi_device_replicated"
CLASSES = ("per_ray", "grid_reduce", "source_lane", "variant_specific", "batch_coupled")
DISPOSITIONS = ("sharded", "unsupported", "single_device", "variant_specific")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def defaults_of(path: Path, function: str) -> dict[str, object]:
    """Checks the replicated multi-device contract against the implementation."""
    tree = ast.parse(read(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            args = node.args
            found: dict[str, object] = {}
            positional = args.posonlyargs + args.args
            for arg, default in zip(positional[len(positional) - len(args.defaults) :], args.defaults):
                found[arg.arg] = ast.literal_eval(default)
            for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                if default is not None:
                    found[arg.arg] = ast.literal_eval(default)
            return found
    raise AssertionError(f"{path.name} defines no function {function!r}")


def dataclass_defaults(path: Path, name: str) -> dict[str, object]:
    """Checks the replicated multi-device contract against the implementation."""
    tree = ast.parse(read(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            found: dict[str, object] = {}
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    if child.value is None:
                        continue
                    found[child.target.id] = ast.literal_eval(child.value)
            return found
    raise AssertionError(f"{path.name} defines no dataclass {name!r}")


def py_constant(text: str, name: str) -> object:
    match = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", text, re.M)
    if match is None:
        raise AssertionError(f"no module-level assignment to {name!r}")
    return ast.literal_eval(match.group(1).strip())


def load_module(backend: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"_adr0038_capabilities_{backend}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_multi_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom) and node.level == 1:
        return node.module == "multi" or (node.module is None and any(a.name == "multi" for a in node.names))
    return False


class Adr0038ContractStateTests(unittest.TestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.public_api = json.loads(read(PUBLIC_API_PATH))
        self.operations = json.loads(read(OPERATIONS_PATH))

    def test_capability_is_declared_in_every_place_that_carries_capabilities(self) -> None:
        self.assertIn(CAPABILITY, self.public_api["capability_keys"])
        self.assertIn(CAPABILITY, self.public_api["apis"])
        self.assertIn(CAPABILITY, self.operations["required_capability_keys"])
        for backend in ("drjit", "torch"):
            self.assertIn(CAPABILITY, self.public_api["backends"][backend]["capabilities"], msg=backend)

    def test_declared_capability_carries_the_adr_values(self) -> None:
        metadata = self.public_api["apis"][CAPABILITY]
        self.assertEqual(metadata["category"], "core")
        self.assertEqual(metadata["stability"], "provisional")
        self.assertTrue(metadata["summary"])
        backends = self.public_api["backends"]
        # Dr.Jit is process-per-GPU only, Torch owns the in-process layer.
        self.assertFalse(backends["drjit"]["capabilities"][CAPABILITY])
        self.assertTrue(backends["torch"]["capabilities"][CAPABILITY])

    def test_the_capability_is_not_an_operation(self) -> None:
        # Multi-device execution is a property of the existing operations, not
        # a fifteenth one.
        self.assertNotIn(CAPABILITY, self.operations["operations"])
        self.assertEqual(len(self.operations["operations"]), 14)

    def test_shardability_classes_are_declared_with_their_dispositions(self) -> None:
        declared = self.operations["shardability_classes"]
        self.assertTrue((ROOT / declared["record"]).is_file())
        self.assertEqual(set(declared["classes"]), set(CLASSES))
        self.assertEqual(set(declared["torch_multi_device"]), set(DISPOSITIONS))
        for name, text in declared["classes"].items():
            with self.subTest(cls=name):
                self.assertTrue(text)

    def test_every_operation_carries_a_declared_class_and_disposition(self) -> None:
        for name, operation in self.operations["operations"].items():
            with self.subTest(operation=name):
                shardability = operation["shardability"]
                self.assertIn(shardability["class"], CLASSES)
                self.assertIn(shardability["torch_multi_device"], DISPOSITIONS)


class Adr0038ClassificationTableTests(unittest.TestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.operations = json.loads(read(OPERATIONS_PATH))["operations"]

    def test_the_classification_is_the_one_the_layer_implements(self) -> None:
        declared = {
            name: (entry["shardability"]["class"], entry["shardability"]["torch_multi_device"])
            for name, entry in self.operations.items()
        }
        self.assertEqual(
            declared,
            {
                "intersect": ("per_ray", "sharded"),
                "nearest_edge_point": ("per_ray", "sharded"),
                "nearest_edge_ray": ("per_ray", "sharded"),
                "nearest_edges_topk": ("per_ray", "sharded"),
                "visibility": ("per_ray", "sharded"),
                "visibility_pair": ("per_ray", "sharded"),
                "visibility_edge": ("per_ray", "sharded"),
                "visibility_chain": ("per_ray", "sharded"),
                "reflection_trace": ("variant_specific", "variant_specific"),
                "reflection_accumulation": ("grid_reduce", "sharded"),
                "diffraction_direct": ("variant_specific", "sharded"),
                "diffraction_chain": ("grid_reduce", "sharded"),
                "sdf_intersect": ("per_ray", "single_device"),
                "mixed_scene": ("per_ray", "single_device"),
            },
        )


class Adr0038LaneWindowTests(unittest.TestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.window = json.loads(read(OPERATIONS_PATH))["shardability_classes"]["lane_window"]
        self.ops_cpp = read(DIFFRACTION_OPS_PATH)
        self.multi = read(MULTI_PATH)

    def test_the_declared_defaults_are_the_public_python_defaults(self) -> None:
        self.assertEqual(self.window["defaults"], {"lane_offset": 0, "lane_count": -1})
        for path, function in (
            (SCENE_PATH, "accum_dfr_direct"),
            (SCENE_PATH, "accum_dfr"),
            (AUTOGRAD_PATH, "accum_dfr_direct_native"),
            (AUTOGRAD_PATH, "accum_dfr_chain_native"),
        ):
            with self.subTest(function=f"{path.name}:{function}"):
                found = defaults_of(path, function)
                self.assertEqual(found["lane_offset"], self.window["defaults"]["lane_offset"])
                self.assertEqual(found["lane_count"], self.window["defaults"]["lane_count"])

    def test_the_dispatcher_schema_carries_the_same_defaults(self) -> None:
        library = read(LIBRARY_PATH)
        offsets = re.findall(r"\blane_offset=(-?\d+)", library)
        counts = re.findall(r"\blane_count=(-?\d+)", library)
        # Every native schema default is the declared window default.
        self.assertTrue(offsets)
        self.assertTrue(counts)
        self.assertEqual({int(value) for value in offsets}, {self.window["defaults"]["lane_offset"]})
        self.assertEqual({int(value) for value in counts}, {self.window["defaults"]["lane_count"]})

    def test_the_warp_alignment_matches_the_orchestrator(self) -> None:
        self.assertEqual(py_constant(self.multi, "_LANE_ALIGNMENT"), self.window["warp_alignment"])

    def test_the_host_twin_rejects_exactly_what_the_native_window_rejects(self) -> None:
        for message in (
            "lane_offset must be non-negative.",
            "lane_offset must not exceed the total sample count.",
            "lane_offset + lane_count must not exceed the total sample count.",
        ):
            with self.subTest(message=message):
                self.assertIn(message, self.ops_cpp)
                self.assertIn(message, self.multi)


class Adr0038PythonDefaultTests(unittest.TestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.multi = read(MULTI_PATH)

    def test_multi_device_options_defaults_are_the_documented_ones(self) -> None:
        self.assertEqual(
            dataclass_defaults(MULTI_PATH, "MultiDeviceOptions"),
            {
                "weights": None,
                "operation_weights": None,
                "require_peer_access": True,
                "require_homogeneous_devices": True,
                "warm_up": True,
                "chunk_rays": None,
                "offload": None,
                "tape_memory_budget_bytes": None,
                "pipeline": True,
                "pipeline_chunks_per_device": 4,
                "min_rays_per_device": 262144,
                "min_lanes_per_device": 262144,
            },
        )

    def test_the_module_constants_and_the_dataclass_agree(self) -> None:
        defaults = dataclass_defaults(MULTI_PATH, "MultiDeviceOptions")
        self.assertEqual(py_constant(self.multi, "_MIN_RAYS_PER_DEVICE"), defaults["min_rays_per_device"])
        self.assertEqual(py_constant(self.multi, "_PIPELINE_CHUNKS_PER_DEVICE"), defaults["pipeline_chunks_per_device"])
        self.assertEqual(py_constant(self.multi, "_MIN_LANES_PER_DEVICE"), defaults["min_lanes_per_device"])


class Adr0038SingleDeviceInvarianceTests(unittest.TestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def test_the_orchestration_layer_is_imported_only_inside_a_function(self) -> None:
        tree = ast.parse(read(SCENE_PATH))
        # No runtime module-level import; the TYPE_CHECKING block is nested in an `if`.
        self.assertEqual([node for node in tree.body if is_multi_import(node)], [])
        lazy = [
            inner
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for inner in ast.walk(node)
            if is_multi_import(inner)
        ]
        self.assertTrue(lazy, "scene.py no longer imports the orchestrator lazily")


class Adr0038CapabilityModuleTests(unittest.TestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.sources = {backend: read(path) for backend, path in CAPABILITY_MODULES.items()}

    def test_each_backend_declares_the_capability_with_its_own_value(self) -> None:
        expected = {"drjit": False, "torch": True}
        for backend, path in CAPABILITY_MODULES.items():
            with self.subTest(backend=backend):
                module = load_module(backend, path)
                self.assertIs(module._CAPABILITIES[CAPABILITY], expected[backend])
                self.assertEqual(module._API_CLASSIFICATION[CAPABILITY], ("core", "provisional"))

    def test_the_copies_diverge_on_exactly_the_three_enumerated_lines(self) -> None:
        drjit = self.sources["drjit"].splitlines()
        torch = self.sources["torch"].splitlines()
        self.assertEqual(len(drjit), len(torch))
        divergent = [left.strip().split(":")[0].strip() for left, right in zip(drjit, torch) if left != right]
        self.assertEqual(divergent, ['_BACKEND = "drjit"', '"torch_compile"', f'"{CAPABILITY}"'])

    def test_both_copies_repinned_the_manifest_hash(self) -> None:
        expected = hashlib.sha256(PUBLIC_API_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for backend, source in self.sources.items():
            with self.subTest(backend=backend):
                self.assertIn(f'_SCHEMA_SHA256 = "{expected}"', source)


if __name__ == "__main__":
    unittest.main()
