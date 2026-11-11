# Copyright Xingyu Chen.
# Tests sdf intersection.

"""Checks the SDF intersection contract against the implementation."""

from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS_PATH = ROOT / "contracts" / "operations.json"
PUBLIC_API_PATH = ROOT / "contracts" / "public_api.json"
COMPILE_POLICY_PATH = ROOT / "contracts" / "compile_policy.json"
DEVICE_MATH_PATH = ROOT / "src" / "sdf_device.cuh"
TORCH_PACKAGE = ROOT / "python" / "rayd" / "_impl"
CAPABILITY_MODULES = {
    "drjit": ROOT / "python" / "rayd" / "_impl" / "capabilities_jit.py",
    "torch": ROOT / "python" / "rayd" / "_impl" / "capabilities.py",
}

CAPABILITY = "sdf_intersect"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def cpp_constant(text: str, name: str) -> float:
    """Checks the SDF intersection contract against the implementation."""
    match = re.search(rf"constexpr\s+\w+\s+{re.escape(name)}\s*=\s*([^;]+);", text)
    if match is None:
        raise AssertionError(f"no constexpr definition of {name!r}")
    return float(match.group(1).strip().rstrip("f"))


def py_constant(text: str, name: str) -> float:
    match = re.search(rf"^{re.escape(name)}\s*=\s*(\S+)$", text, re.M)
    if match is None:
        raise AssertionError(f"no module-level assignment to {name!r}")
    return float(match.group(1))


def load_module(backend: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"_adr0037_capabilities_{backend}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Adr0037ContractStateTests(unittest.TestCase):
    """Checks the SDF intersection contract against the implementation."""

    def setUp(self) -> None:
        self.public_api = json.loads(read(PUBLIC_API_PATH))
        self.operations = json.loads(read(OPERATIONS_PATH))
        self.compile_policy = json.loads(read(COMPILE_POLICY_PATH))

    def test_capability_is_declared_in_every_place_that_carries_capabilities(self) -> None:
        self.assertIn(CAPABILITY, self.public_api["capability_keys"])
        self.assertIn(CAPABILITY, self.public_api["apis"])
        self.assertIn(CAPABILITY, self.operations["required_capability_keys"])
        self.assertIn(CAPABILITY, self.operations["operations"])
        for backend in ("drjit", "torch"):
            self.assertIn(CAPABILITY, self.public_api["backends"][backend]["capabilities"], msg=backend)

    def test_declared_capability_carries_the_adr_values(self) -> None:
        metadata = self.public_api["apis"][CAPABILITY]
        self.assertEqual(metadata["category"], "core")
        self.assertEqual(metadata["stability"], "provisional")
        self.assertTrue(metadata["summary"])
        backends = self.public_api["backends"]
        # ADR-0042 supersedes the original Torch-only backend scope.
        self.assertTrue(backends["drjit"]["capabilities"][CAPABILITY])
        self.assertTrue(backends["torch"]["capabilities"][CAPABILITY])

    def test_no_sdf_translation_unit_may_leave_the_nvcc_default_profile(self) -> None:
        units = [unit for unit in self.compile_policy["translation_units"] if unit["source"] == "src/sdf.cu"]
        self.assertTrue(units, "the SDF translation units are not declared at all")
        for unit in units:
            with self.subTest(unit=f"{unit['backend']}:{unit['unit']}"):
                self.assertEqual(unit["profile"], "nvcc_default")
                self.assertEqual(unit["kind"], "object")


class Adr0037OperationContractTests(unittest.TestCase):
    """Checks the SDF intersection contract against the implementation."""

    def setUp(self) -> None:
        self.operations = json.loads(read(OPERATIONS_PATH))
        self.operation = self.operations["operations"][CAPABILITY]
        self.result = self.operations["result_contracts"]["sdf_intersection"]

    def test_operation_is_cross_backend_and_names_its_numeric_record(self) -> None:
        self.assertEqual(self.operation["integration"], {"drjit": ["eager_native"], "torch": ["eager_native"]})
        self.assertTrue((ROOT / self.operation["record"]).is_file())

    def test_the_six_differentiable_inputs_are_the_operation_inputs(self) -> None:
        # The contract's input list must contain the six gradient inputs plus
        # the four non-differentiable host scalars.
        self.assertLessEqual(
            {"values", "position", "rotation", "scale", "origins", "directions"}, set(self.operation["inputs"])
        )
        self.assertLessEqual({"tmax", "max_steps", "relaxation", "eps_hit"}, set(self.operation["inputs"]))

    def test_reused_epsilons_are_the_shared_contract_values(self) -> None:
        policy = self.operation["numeric_policy"]
        # The reused epsilons are the shared contract's own values, not copies
        # that happen to agree today.
        self.assertEqual(policy["eps_graze"], self.operations["constants"]["epsilon"]["small"])
        self.assertEqual(policy["eps_norm"], self.operations["numeric_policy"]["shared_multipath"]["normalize_floor"])
        self.assertEqual(
            policy["eps_parallel"], self.operations["numeric_policy"]["backend_profiles"]["torch"]["parallel_epsilon"]
        )

    def test_miss_sentinel_is_the_shared_distance_sentinel(self) -> None:
        policy = self.operation["numeric_policy"]
        self.assertEqual(policy["miss_sentinel"], self.operations["miss_sentinels"]["distance"])
        self.assertEqual(self.result["miss"]["t"], self.operations["miss_sentinels"]["distance"])
        self.assertIs(self.result["miss"]["valid"], False)
        self.assertEqual(self.result["miss"]["position"], 0.0)
        self.assertEqual(self.result["miss"]["normal"], 0.0)

    def test_result_fields_match_both_public_result_types(self) -> None:
        source = read(TORCH_PACKAGE / "geometry.py")
        start = source.index("class SdfIntersection(")
        block = source[start : source.index("@dataclass", start)]
        fields = re.findall(r"^    ([a-z][a-z0-9_]*): torch\.Tensor$", block, re.M)
        self.assertEqual(fields, self.result["canonical_fields"])
        self.assertEqual(fields, self.result["backend_fields"]["torch"])
        self.assertEqual(fields, self.result["backend_fields"]["drjit"])
        self.assertEqual(self.result["differentiable_fields"], ["t", "position", "normal"])
        self.assertEqual(set(self.result["field_types"]), set(fields))


class Adr0037CodeConstantTests(unittest.TestCase):
    """Checks the SDF intersection contract against the implementation."""

    def setUp(self) -> None:
        self.device_math = read(DEVICE_MATH_PATH)
        self.operations = json.loads(read(OPERATIONS_PATH))
        self.policy = self.operations["operations"][CAPABILITY]["numeric_policy"]

    def test_device_constants_equal_the_operation_contract(self) -> None:
        expected = {
            "kSdfBisectionSteps": self.policy["bisection_steps"],
            "kSdfDefaultMaxSteps": self.policy["max_steps_default"],
            "kSdfDefaultRelaxation": self.policy["relaxation_default"],
            "kSdfEpsHitVoxelFraction": self.policy["eps_hit_voxel_fraction"],
            "kSdfEpsNorm": self.policy["eps_norm"],
            "kSdfEpsParallel": self.policy["eps_parallel"],
            "kSdfEpsGraze": self.operations["constants"]["epsilon"]["small"],
        }
        for name, value in expected.items():
            with self.subTest(constant=name):
                self.assertEqual(cpp_constant(self.device_math, name), float(value))

    def test_python_defaults_equal_the_device_defaults(self) -> None:
        source = read(TORCH_PACKAGE / "sdf.py")
        self.assertEqual(
            py_constant(source, "DEFAULT_MAX_STEPS"), cpp_constant(self.device_math, "kSdfDefaultMaxSteps")
        )
        self.assertEqual(
            py_constant(source, "DEFAULT_RELAXATION"), cpp_constant(self.device_math, "kSdfDefaultRelaxation")
        )
        # The host scalar is a non-positive sentinel meaning "derive eps_hit
        # from the resident scale", which is what keeps the operation free of a
        # device-to-host read.
        self.assertLess(py_constant(source, "_EPS_HIT_DEVICE_DERIVED"), 0.0)


class Adr0037CapabilityModuleTests(unittest.TestCase):
    """Checks the SDF intersection contract against the implementation."""

    def test_each_backend_declares_the_capability_true(self) -> None:
        for backend, path in CAPABILITY_MODULES.items():
            with self.subTest(backend=backend):
                module = load_module(backend, path)
                self.assertIs(module._CAPABILITIES[CAPABILITY], True)
                self.assertEqual(module._API_CLASSIFICATION[CAPABILITY], ("core", "provisional"))

    def test_sdf_is_no_longer_a_capability_divergence(self) -> None:
        drjit = read(CAPABILITY_MODULES["drjit"]).splitlines()
        torch = read(CAPABILITY_MODULES["torch"]).splitlines()
        self.assertEqual(len(drjit), len(torch))
        divergent = [left.strip().split(":")[0].strip() for left, right in zip(drjit, torch) if left != right]
        self.assertEqual(divergent, ['_BACKEND = "drjit"', '"torch_compile"', '"multi_device_replicated"'])


if __name__ == "__main__":
    unittest.main()
