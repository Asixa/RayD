# Copyright Xingyu Chen.
# Tests axial edge visibility.

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH = ROOT / "torch"


class TypedAxialEdgeVisibilityGovernanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adr = (ROOT / "docs/adr/0029-typed-axial-edge-visibility.md").read_text(
            encoding="utf-8"
        )
        cls.header = (ROOT / "include/rayd/visibility.h").read_text(
            encoding="utf-8"
        )
        cls.ops = (ROOT / "src/visibility/visibility.cpp").read_text(
            encoding="utf-8"
        )
        cls.device = (
            ROOT / "src/visibility/axial_edge_visibility_optix.cu"
        ).read_text(encoding="utf-8")
        cls.cmake = (TORCH / "CMakeLists.txt").read_text(encoding="utf-8")
        cls.direct_test = (ROOT / "tests/native/integration_test.cpp").read_text(
            encoding="utf-8"
        )

    def test_accepted_contract_and_exact_fraction_bits(self):
        self.assertIn("Status: Accepted", self.adr)
        for token in (
            "AxialEdgeVisibilityRequest",
            "AxialEdgeVisibilityResult",
            "axial_edge_visibility_forward",
            "0x3ca3d70a",
            "0x3eaaaaab",
            "0x3f2aaaab",
            "0x3f7ae148",
        ):
            self.assertIn(token, self.header)

    def test_exact_device_arithmetic_has_no_torch_numerical_path(self):
        for token in (
            'asm volatile("sub.rn.f32',
            'asm volatile("mul.rn.f32',
            'asm volatile("add.rn.f32',
        ):
            self.assertIn(token, self.device)
        for forbidden in (".ftz", "fma.", "__fsub_rn", "__fmul_rn", "__fadd_rn"):
            self.assertNotIn(forbidden, self.device)
        for forbidden in ("at::", "torch::", "at::isfinite", "finite_vec3_rows"):
            self.assertNotIn(forbidden, self.device)

    def test_separate_ptx_inherits_legacy_visibility_compile_policy(self):
        start = self.cmake.index('OUTPUT "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX}"')
        end = self.cmake.index("add_custom_command(", start + 1)
        block = self.cmake[start:end]
        self.assertIn("${RAYD_TORCH_OPTIX_NVCC_FLAGS}", block)
        for forbidden in ("--fmad=false", "--prec-div=true", "--prec-sqrt=true"):
            self.assertNotIn(forbidden, block)
        self.assertIn("axial_edge_visibility_pipeline_config", self.ops)

    def test_one_launch_and_empty_returns_before_launch(self):
        start = self.ops.index("axial_edge_visibility_forward_native_impl")
        end = self.ops.index("} // namespace rayd::torch_backend", start)
        body = self.ops[start:end]
        self.assertEqual(body.count("->launch("), 1)
        self.assertLess(body.index("state_count == 0"), body.index("->launch("))

    def test_parity_and_existing_staging_sync_are_explicit(self):
        common_launch = (
            ROOT / "src/runtime/optix.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("cudaEventSynchronize(params_staging_events_[slot])", common_launch)
        self.assertIn("reducing four public launch-parameter staging checks to one", self.adr)
        self.assertIn("separate Phase 12 optimization", self.adr)
        self.assertIn("random_count = 257", self.direct_test)
        self.assertIn("single axial launch versus four segment launches", self.direct_test)

    def test_candidate_is_not_python_or_legacy_dispatcher_exposed(self):
        library = (ROOT / "src/bindings/library.cpp").read_text(encoding="utf-8")
        module = (ROOT / "src/bindings/module.cpp").read_text(encoding="utf-8")
        legacy_device = (
            ROOT / "src/visibility/visibility_optix.cu"
        ).read_text(encoding="utf-8")
        self.assertNotIn("axial_edge_visibility", library)
        self.assertNotIn("axial_edge_visibility", module)
        self.assertNotIn("axial_edge_visibility_exact", legacy_device)
        self.assertIn("__raygen__segment_visibility", legacy_device)

    def test_repository_guardrails_match(self):
        self.assertEqual(
            (ROOT / "AGENTS.md").read_bytes(), (ROOT / "CLAUDE.md").read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
