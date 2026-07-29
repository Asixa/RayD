# Copyright Xingyu Chen.
# Tests multi device.

"""Checks the replicated multi-device contract against the implementation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = ROOT / "docs" / "adr" / "0038-replicated-multi-device-execution.md"
ADR0036_PATH = ROOT / "docs" / "adr" / "0036-backend-mirrored-python-modules.md"
PLAN_PATH = ROOT / "docs" / "dev" / "multi_gpu_plan.md"
OPERATIONS_DOC_PATH = ROOT / "docs" / "dev" / "multi_gpu_operations.md"
OPERATIONS_PATH = ROOT / "contracts" / "operations.json"
PUBLIC_API_PATH = ROOT / "contracts" / "public_api.json"
PTX_SOURCES_PATH = ROOT / "drjit" / "ptx_sources.json"
TORCH_PACKAGE = ROOT / "python" / "rayd" / "_impl"
MULTI_PATH = TORCH_PACKAGE / "multi.py"
SCENE_PATH = TORCH_PACKAGE / "scene.py"
AUTOGRAD_PATH = TORCH_PACKAGE / "multipath.py"
LIBRARY_PATH = ROOT / "src" / "bindings" / "library.cpp"
DIFFRACTION_OPS_PATH = (
    ROOT / "src" / "diffraction" / "diffraction.cpp"
)
DRJIT_PARITY_TEST_PATH = (
    ROOT / "tests" / "native" / "test_cuda_multipath_jit.py"
)
CAPABILITY_MODULES = {
    "drjit": ROOT / "python" / "rayd" / "_impl" / "capabilities_jit.py",
    "torch": ROOT / "python" / "rayd" / "_impl" / "capabilities.py",
}

CAPABILITY = "multi_device_replicated"
CLASSES = ("per_ray", "grid_reduce", "batch_coupled")
DISPOSITIONS = ("sharded", "unsupported", "single_device")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flat(text: str) -> str:
    """Checks the replicated multi-device contract against the implementation."""
    return re.sub(r"\s+", " ", re.sub(r"^[ \t]*>[ \t]?", " ", text, flags=re.M))


def sections(text: str, level: int) -> dict[str, str]:
    """Checks the replicated multi-device contract against the implementation."""
    marker = "#" * level + " "
    found: dict[str, str] = {}
    title: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and not line.startswith(marker):
            if line.split(" ", 1)[0].count("#") <= level:
                if title is not None:
                    found[title] = "\n".join(body)
                title, body = None, []
            continue
        if line.startswith(marker):
            if title is not None:
                found[title] = "\n".join(body)
            title, body = line[len(marker) :].strip(), []
            continue
        if title is not None:
            body.append(line)
    if title is not None:
        found[title] = "\n".join(body)
    return found


def table_rows(body: str, columns: int) -> list[list[str]]:
    """Checks the replicated multi-device contract against the implementation."""
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) != columns:
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows[1:] if rows else rows


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


class AdrTestCase(unittest.TestCase):
    def assertPhrase(self, phrase: str, text: str) -> None:
        self.assertIn(flat(phrase), flat(text))

    def assertNoPhrase(self, phrase: str, text: str) -> None:
        self.assertNotIn(flat(phrase), flat(text))


class Adr0038RecordTests(AdrTestCase):
    def setUp(self) -> None:
        self.adr = read(ADR_PATH)
        self.top = sections(self.adr, 2)
        self.decision = sections(self.adr, 3)

    def test_header_block_matches_the_repository_adr_shape(self) -> None:
        head = self.adr.splitlines()[:6]
        self.assertEqual(head[0], "# ADR-0038: Replicated multi-device and chunked execution")
        self.assertRegex(head[2], r"^- Status: Accepted(?:; .+)?$")
        self.assertIn("ADR-0040", head[2])
        self.assertEqual(head[3], "- Date: 2026-07-27")
        self.assertEqual(head[4], "- Decision ID: `replicated-multi-device-execution`")
        self.assertTrue(head[5].startswith("- Scope:"))

    def test_required_top_level_sections_are_present(self) -> None:
        self.assertLessEqual(
            {
                "Context",
                "Decision",
                "Measured results",
                "Platform note (observation)",
                "Contract impact",
                "Consequences",
                "Non-goals",
                "Deferred",
                "Stop conditions",
            },
            set(self.top),
        )

    def test_every_plan_decision_has_its_own_subsection(self) -> None:
        titles = " | ".join(self.decision)
        for tag in ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"):
            with self.subTest(decision=tag):
                self.assertIn(f"({tag})", titles)
        for subject in (
            "Small-batch fallback and calibration",
            "Shardability classification",
        ):
            self.assertIn(subject, titles)

    def test_the_record_points_at_its_plan_and_its_operational_note(self) -> None:
        self.assertPhrase("docs/dev/multi_gpu_plan.md", self.adr)
        self.assertPhrase("multi_gpu_operations.md", self.adr)
        self.assertTrue(PLAN_PATH.is_file())
        self.assertTrue(OPERATIONS_DOC_PATH.is_file())

    def test_the_frozen_guarantee_names_the_records_it_leaves_alone(self) -> None:
        body = self.decision["3. Single-device numerics are untouched; merge-layer float semantics (D3)"]
        for record in ("ADR-0026", "ADR-0030", "ADR-0032", "ADR-0035"):
            with self.subTest(record=record):
                self.assertPhrase(record, body)
        self.assertPhrase("No kernel launch shape", body)
        self.assertPhrase("the only kernel-visible change", body)

    def test_stop_conditions_forbid_the_expensive_mistakes(self) -> None:
        body = self.top["Stop conditions"]
        for condition in (
            "reduction order",
            "bitwise equality",
            "warp multiples",
            "batch_coupled",
            "detaching, zeroing, or approximating a gradient",
            "row floor",
            "parallel public multi-device API surface",
            "collective",
            "partitioning geometry",
        ):
            with self.subTest(condition=condition):
                self.assertPhrase(condition, body)

    def test_deferred_items_are_the_ones_the_layer_actually_refuses(self) -> None:
        body = self.top["Deferred"]
        for item in (
            "trace_dfr_paths",
            "SourceLane",
            "ADR-0032",
            "deduplicate = true",
            "ADR-0033 failure-bit",
            "accum_dfr_coherent_direct",
            "Appendix A",
        ):
            with self.subTest(item=item):
                self.assertPhrase(item, body)
        self.assertPhrase("None is authorized here", body)

    def test_contract_impact_names_every_file_this_change_touches(self) -> None:
        body = self.top["Contract impact"]
        for path in (
            "shared/contracts/public_api.json",
            "shared/contracts/operations.json",
            "backends/drjit/python/rayd/drjit/_capabilities.py",
            "backends/torch/python/rayd/torch/_capabilities.py",
            "tests/test_shared_operation_contract.py",
            "tests/test_adr0038_multi_device.py",
            "shared/contracts/compile_policy.json",
            "backends/drjit/ptx_sources.json",
        ):
            with self.subTest(path=path):
                self.assertPhrase(path, body)
        self.assertPhrase("ADR-0036", body)
        self.assertPhrase("**five** lines", body)


class Adr0038ContractStateTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.public_api = json.loads(read(PUBLIC_API_PATH))
        self.operations = json.loads(read(OPERATIONS_PATH))

    def test_capability_is_declared_in_every_place_that_carries_capabilities(self) -> None:
        self.assertIn(CAPABILITY, self.public_api["capability_keys"])
        self.assertIn(CAPABILITY, self.public_api["apis"])
        self.assertIn(CAPABILITY, self.operations["required_capability_keys"])
        for backend in ("drjit", "torch"):
            self.assertIn(
                CAPABILITY,
                self.public_api["backends"][backend]["capabilities"],
                msg=backend,
            )

    def test_declared_capability_carries_the_adr_values(self) -> None:
        metadata = self.public_api["apis"][CAPABILITY]
        self.assertEqual(metadata["category"], "core")
        self.assertEqual(metadata["stability"], "provisional")
        self.assertTrue(metadata["summary"])
        backends = self.public_api["backends"]
        # Section 2: Dr.Jit is process-per-GPU only, Torch owns the in-process layer.
        self.assertFalse(backends["drjit"]["capabilities"][CAPABILITY])
        self.assertTrue(backends["torch"]["capabilities"][CAPABILITY])
        impact = sections(read(ADR_PATH), 2)["Contract impact"]
        self.assertPhrase(
            'backends.drjit.capabilities.multi_device_replicated` is `false`', impact
        )
        self.assertPhrase(
            'backends.torch.capabilities.multi_device_replicated` is `true`', impact
        )

    def test_the_capability_is_not_an_operation(self) -> None:
        # Section 11: multi-device execution is a property of the existing
        # operations, not a fourteenth one.
        self.assertNotIn(CAPABILITY, self.operations["operations"])
        self.assertEqual(len(self.operations["operations"]), 13)

    def test_shardability_classes_are_declared_with_their_dispositions(self) -> None:
        declared = self.operations["shardability_classes"]
        self.assertEqual(declared["record"], "docs/adr/0038-replicated-multi-device-execution.md")
        self.assertTrue((ROOT / declared["record"]).is_file())
        self.assertEqual(set(declared["classes"]), set(CLASSES))
        self.assertEqual(set(declared["torch_multi_device"]), set(DISPOSITIONS))
        for name, text in declared["classes"].items():
            with self.subTest(cls=name):
                self.assertTrue(text)
        self.assertPhrase("float32 summation order", declared["merge_order"])
        self.assertPhrase("devices order", declared["merge_order"])

    def test_every_operation_carries_a_declared_class_and_disposition(self) -> None:
        for name, operation in self.operations["operations"].items():
            with self.subTest(operation=name):
                shardability = operation["shardability"]
                self.assertIn(shardability["class"], CLASSES)
                self.assertIn(shardability["torch_multi_device"], DISPOSITIONS)


class Adr0038ClassificationTableTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.operations = json.loads(read(OPERATIONS_PATH))["operations"]
        self.body = sections(read(ADR_PATH), 3)["11. Shardability classification"]

    def test_the_record_table_is_the_contract_table(self) -> None:
        rows = {
            operation: (cls, disposition)
            for operation, cls, disposition in table_rows(self.body, 3)
        }
        declared = {
            name: (entry["shardability"]["class"], entry["shardability"]["torch_multi_device"])
            for name, entry in self.operations.items()
        }
        self.assertEqual(rows, declared)

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
                "reflection_trace": ("per_ray", "sharded"),
                "reflection_accumulation": ("grid_reduce", "single_device"),
                "diffraction_direct": ("grid_reduce", "sharded"),
                "diffraction_chain": ("grid_reduce", "sharded"),
                "sdf_intersect": ("per_ray", "single_device"),
            },
        )


class Adr0038LaneWindowTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.window = json.loads(read(OPERATIONS_PATH))["shardability_classes"]["lane_window"]
        self.body = sections(read(ADR_PATH), 3)["5. The Monte-Carlo lane window (D5)"]
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
        # One windowed forward, four AD ops that inherit their width from the tape.
        self.assertEqual(
            library.count("int lane_offset=0, int lane_count=-1) -> Tensor?[]"), 1
        )
        self.assertEqual(library.count("int lane_offset=0) -> Tensor?[]"), 4)

    def test_the_zero_offset_launch_is_declared_and_implemented_as_a_no_op(self) -> None:
        self.assertPhrase("bitwise the unwindowed single launch", self.window["invariance"])
        self.assertPhrase(
            "`lane_offset = 0` with the default `lane_count` is bitwise the pre-ADR launch",
            self.body,
        )
        # `rebase_lane_buffer` is what makes that true, and it returns early.
        self.assertIn("if (ptr == nullptr || lane_offset == 0)", self.ops_cpp)

    def test_the_warp_alignment_caveat_matches_the_orchestrator(self) -> None:
        self.assertEqual(self.window["warp_alignment"], 32)
        self.assertEqual(py_constant(self.multi, "_LANE_ALIGNMENT"), 32)
        self.assertPhrase("warp-multiple windows", self.window["warp_alignment_rule"])
        self.assertPhrase("only for warp-multiple windows", self.body)
        self.assertPhrase("aggregates a warp's contributions before its atomic", self.body)

    def test_a_windowed_launch_requires_the_optix_backend(self) -> None:
        self.assertPhrase("requires the OptiX trace backend", self.window["trace_backend"])
        self.assertIn(
            '"diffraction accumulation lane_offset requires the OptiX trace backend."',
            self.ops_cpp,
        )
        self.assertPhrase("requires the OptiX trace backend", self.body)

    def test_the_host_twin_rejects_exactly_what_the_native_window_rejects(self) -> None:
        for message in (
            "lane_offset must be non-negative.",
            "lane_offset must not exceed the total sample count.",
            "lane_offset + lane_count must not exceed the total sample count.",
        ):
            with self.subTest(message=message):
                self.assertIn(message, self.ops_cpp)
                self.assertIn(message, self.multi)

    def test_the_global_lane_space_is_the_sum_of_the_three_sample_counts(self) -> None:
        self.assertPhrase(
            "direct_samples + keller_samples + suffix_samples", self.window["semantics"]
        )
        self.assertPhrase(
            "direct_samples + keller_samples + suffix_samples", self.body
        )
        self.assertIn(
            "checked_i32(direct_samples + keller_samples + suffix_samples, \"total_samples\")",
            self.ops_cpp,
        )


class Adr0038PythonDefaultTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.multi = read(MULTI_PATH)
        self.decision = sections(read(ADR_PATH), 3)

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
        body = self.decision["8. Multi-GPU is invisible at the top-level API (D8)"]
        for pinned in (
            "`weights=None` (equal split)",
            "`operation_weights=None`",
            "`require_peer_access=True`",
            "`require_homogeneous_devices=True`",
            "`warm_up=True`",
            "`chunk_rays=None`",
            "`offload=None`",
            "`tape_memory_budget_bytes=None`",
            "`pipeline=True`",
            "`pipeline_chunks_per_device=4`",
            "`min_rays_per_device=262144`",
            "`min_lanes_per_device=262144`",
        ):
            with self.subTest(default=pinned):
                self.assertPhrase(pinned, body)

    def test_the_module_constants_and_the_dataclass_agree(self) -> None:
        defaults = dataclass_defaults(MULTI_PATH, "MultiDeviceOptions")
        self.assertEqual(
            py_constant(self.multi, "_MIN_RAYS_PER_DEVICE"), defaults["min_rays_per_device"]
        )
        self.assertEqual(
            py_constant(self.multi, "_PIPELINE_CHUNKS_PER_DEVICE"),
            defaults["pipeline_chunks_per_device"],
        )
        self.assertEqual(
            py_constant(self.multi, "_MIN_LANES_PER_DEVICE"),
            defaults["min_lanes_per_device"],
        )

    def test_the_work_floor_in_the_record_is_the_shipped_policy(self) -> None:
        body = self.decision["10. Small-batch fallback and calibration semantics"]
        self.assertPhrase("weighted remote shard", body)
        self.assertPhrase("copied input plus", body)
        self.assertPhrase("returned output bytes per row", body)
        self.assertPhrase("`min_rays_per_device`", body)
        self.assertPhrase("`min_lanes_per_device`", body)
        self.assertPhrase("runs on the master replica", body)

    def test_the_calibration_ladder_and_tolerance_are_the_shipped_ones(self) -> None:
        body = self.decision["10. Small-batch fallback and calibration semantics"]
        self.assertEqual(py_constant(self.multi, "_REFINE_SHARES"), (1.0, 0.5, 0.25, 0.1, 0.0))
        self.assertEqual(py_constant(self.multi, "_REFINE_TOLERANCE"), 0.03)
        self.assertPhrase("`1, 1/2, 1/4, 1/10, 0`", body)
        self.assertPhrase("within 3%", body)
        # The strength of the claim is the point of the section.
        self.assertPhrase(
            "Calibration will not knowingly keep a split that it measured as more "
            "than the refinement tolerance (3%) slower than the master alone",
            body,
        )
        self.assertPhrase(
            'It is **not** "calibration cannot leave you slower than one GPU"', body
        )


class Adr0038SingleDeviceInvarianceTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.scene = read(SCENE_PATH)
        self.body = sections(read(ADR_PATH), 3)["9. Zero single-GPU regression (D9)"]

    def test_the_orchestration_layer_is_imported_only_when_devices_are_requested(self) -> None:
        # `from .multi import ...` appears exactly once, inside the branch that
        # only runs when the caller asked for devices or options.
        self.assertEqual(self.scene.count("from .multi import"), 1)
        self.assertIn(
            "if devices is not None or options is not None:\n"
            "            from .multi import plan as _plan_multi_device",
            self.scene,
        )
        module_level = [
            node
            for node in ast.parse(self.scene).body
            if isinstance(node, ast.ImportFrom) and node.module == "_multi"
        ]
        self.assertEqual(module_level, [])
        self.assertPhrase("imported **only** when `devices=` is passed", self.body)

    def test_a_one_device_scene_without_chunking_gets_no_orchestrator(self) -> None:
        multi = read(MULTI_PATH)
        self.assertIn("if len(indices) == 1 and not chunking:\n        return None", multi)
        self.assertPhrase(
            "`plan()` returns `None` for a one-device `Scene(devices=[d])`", self.body
        )

    def test_every_dispatch_site_is_one_comparison(self) -> None:
        # The branch the record promises: `if self._multi is not None`, nothing else.
        self.assertGreaterEqual(self.scene.count("if self._multi is not None:"), 15)
        self.assertPhrase("one `if self._multi is not None` comparison", self.body)

    def test_the_phase0_guards_are_named_as_the_only_single_gpu_change(self) -> None:
        self.assertPhrase("Phase 0 device guards are the only change", self.body)
        self.assertPhrase("c10::cuda::CUDAGuard", self.body)
        self.assertPhrase("ADR-0026 pinned RF source hashes", self.body)


class Adr0038RefusalTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.scene = read(SCENE_PATH)
        self.multi = read(MULTI_PATH)
        self.body = sections(read(ADR_PATH), 3)[
            "6. Batch-coupled operations get explicit semantics or they fail (D6)"
        ]

    def test_the_two_refused_operations_are_the_ones_the_record_names(self) -> None:
        refused = set(re.findall(r'_multi\.unsupported\("([a-z_]+)"\)', self.scene))
        self.assertEqual(refused, {"trace_dfr_paths", "accum_dfr_coherent_direct"})
        for name in sorted(refused):
            with self.subTest(operation=name):
                self.assertPhrase(name, self.body)
        self.assertIn("raise NotImplementedError(", self.multi)
        self.assertPhrase("raise `NotImplementedError`", self.body)

    def test_dedup_and_penetration_keep_their_single_launch_meaning(self) -> None:
        self.assertPhrase("reflection_dedup_forward", self.body)
        self.assertPhrase("ADR-0033", self.body)
        # The Torch dedup op exists and is not a Scene method, which is why the
        # replicated layer never sees it.
        self.assertIn("reflection_dedup_forward", read(LIBRARY_PATH))
        self.assertNotIn("deduplicate", self.scene)
        self.assertNotIn("dedup", self.multi)


class Adr0038FrozenSurfaceTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def test_no_committed_ptx_closure_reaches_the_torch_backend(self) -> None:
        modules = json.loads(read(PTX_SOURCES_PATH))["modules"]
        self.assertTrue(modules)
        for name, module in modules.items():
            for source in module["sources"]:
                with self.subTest(module=name, source=source):
                    self.assertFalse(source.startswith("backends/torch/"))

    def test_the_record_states_the_untouched_contracts(self) -> None:
        impact = sections(read(ADR_PATH), 2)["Contract impact"]
        self.assertPhrase("No compile-flag change", impact)
        self.assertPhrase("compile_policy.json` is untouched", impact)
        self.assertPhrase("No PTX change", impact)
        self.assertPhrase("ptx_sources.json` is untouched", impact)


class Adr0038CapabilityModuleTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.sources = {
            backend: read(path) for backend, path in CAPABILITY_MODULES.items()
        }

    def test_each_backend_declares_the_capability_with_its_own_value(self) -> None:
        self.assertIn(f'"{CAPABILITY}": False,', self.sources["drjit"])
        self.assertIn(f'"{CAPABILITY}": True,', self.sources["torch"])
        for backend, source in self.sources.items():
            with self.subTest(backend=backend):
                self.assertIn(f'"{CAPABILITY}": ("core", "provisional"),', source)

    def test_the_copies_diverge_on_exactly_the_five_enumerated_lines(self) -> None:
        drjit = self.sources["drjit"].splitlines()
        torch = self.sources["torch"].splitlines()
        self.assertEqual(len(drjit), len(torch))
        divergent = [
            left.strip().split(":")[0].strip()
            for left, right in zip(drjit, torch)
            if left != right
        ]
        self.assertEqual(
            divergent,
            [
                '_BACKEND = "drjit"',
                '"surfel"',
                '"sdf_intersect"',
                '"torch_compile"',
                f'"{CAPABILITY}"',
            ],
        )

    def test_both_copies_repinned_the_manifest_hash(self) -> None:
        expected = hashlib.sha256(
            PUBLIC_API_PATH.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for backend, source in self.sources.items():
            with self.subTest(backend=backend):
                self.assertIn(f'_SCHEMA_SHA256 = "{expected}"', source)

    def test_adr0036_was_amended_rather_than_left_false(self) -> None:
        adr0036 = read(ADR0036_PATH)
        self.assertPhrase("diverges on exactly five lines", adr0036)
        self.assertPhrase(
            f'`"{CAPABILITY}"` (`False` versus `True`, per ADR-0038)', adr0036
        )
        self.assertNoPhrase("diverges on exactly four lines", adr0036)
        self.assertNoPhrase("diverges on exactly three lines", adr0036)


class Adr0038EvidenceTests(AdrTestCase):
    """Checks the replicated multi-device contract against the implementation."""

    def setUp(self) -> None:
        self.adr = read(ADR_PATH)
        self.measured = sections(self.adr, 2)["Measured results"]
        self.note = read(OPERATIONS_DOC_PATH)

    def test_the_headline_numbers_come_from_the_recorded_runs(self) -> None:
        rows = {
            (row[0].split(" ")[0], row[1]): row[2:] for row in table_rows(self.measured, 5)
        }
        self.assertEqual(rows[("compute", "intersect")], ["19.09 ms", "11.83 ms", "1.61x"])
        self.assertEqual(
            rows[("compute", "trace_reflections")], ["53.33 ms", "28.38 ms", "1.88x"]
        )
        self.assertEqual(
            rows[("compute", "accum_dfr_direct")], ["34.76 ms", "18.83 ms", "1.85x"]
        )
        self.assertEqual(rows[("light", "intersect")][2], "0.27x")
        for cell in ("19.09 ms", "11.83 ms", "53.33 ms", "28.38 ms", "34.76 ms", "18.83 ms"):
            with self.subTest(cell=cell):
                self.assertIn(cell, self.note)

    def test_the_machine_and_the_link_speed_are_the_measured_ones(self) -> None:
        for pinned in ("2x NVIDIA RTX A6000", "49.1 GB/s", "Torch 2.13.0+cu130", "maxwell"):
            with self.subTest(pinned=pinned):
                self.assertPhrase(pinned, self.measured)
                self.assertPhrase(pinned, self.note)

    def test_the_merge_deviation_and_the_bitwise_claim_are_the_recorded_ones(self) -> None:
        for pinned in ("6.5e-08", "2.9e-07"):
            with self.subTest(pinned=pinned):
                self.assertPhrase(pinned, self.adr)
                self.assertPhrase(pinned, self.note)
        self.assertPhrase("bitwise the single-device result", self.measured)

    def test_the_platform_note_is_an_observation_with_a_live_test_behind_it(self) -> None:
        body = sections(self.adr, 2)["Platform note (observation)"]
        self.assertPhrase("test_diffraction_paths_parity", body)
        self.assertPhrase("bit-identically before and after", body)
        self.assertPhrase("observation, not as a decision", body)
        self.assertIn("def test_diffraction_paths_parity(", read(DRJIT_PARITY_TEST_PATH))


if __name__ == "__main__":
    unittest.main()
