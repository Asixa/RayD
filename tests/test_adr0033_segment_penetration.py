from __future__ import annotations

from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
TORCH = ROOT / "backends" / "torch"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class Adr0033SegmentPenetrationTests(unittest.TestCase):
    def test_stable_typed_api_is_complete(self) -> None:
        header = read("include/rayd/torch/integration.h")
        self.assertIn("kIntegrationApiVersion = 6", header)
        self.assertIn('"rayd.torch.integration"', header)
        for name in (
            "SegmentPenetrationPolicy",
            "EnumeratedFullDistance",
            "MonteCarloTargetInset",
            "SegmentPenetrationRequest",
            "SegmentPenetrationResult",
            "SegmentPenetrationTapeResult",
            "SegmentPenetrationBackwardRequest",
            "SegmentPenetrationBackwardResult",
            "SegmentPenetrationJvpRequest",
            "SegmentPenetrationJvpResult",
            "segment_penetration_forward",
            "segment_penetration_forward_tape",
            "segment_penetration_backward",
            "segment_penetration_jvp",
            "input_active_any",
            "hit_capacity",
            "capacity_failure_state",
            "failure_bit",
        ):
            self.assertIn(name, header)
        family = header[header.index("enum class SegmentPenetrationPolicy") :]
        family = family[: family.index("struct ReflectionTraceRequest")]
        self.assertNotIn("v2", family.lower())
        self.assertNotIn("wip", family.lower())
        self.assertNotIn("next", family.lower())

    def test_one_optix_launch_contains_the_d_plus_one_march(self) -> None:
        host = read("src/penetration/penetration.cpp")
        device = read(
            "src/penetration/penetration_optix.cu"
        )
        self.assertEqual(host.count("->launch("), 1)
        self.assertIn(
            "validated.segment_count > 0 && request.input_active_any", host
        )
        self.assertEqual(device.count("optixTrace("), 1)
        self.assertIn("probe <= params.hit_capacity", device)
        self.assertIn("probe == params.hit_capacity", device)
        self.assertIn("atomicOr(params.capacity_failure_state, params.failure_bit)", device)
        self.assertIn("hit_t < remaining", device)
        self.assertIn("hit_t > 0.0f && hit_t <= remaining", device)
        self.assertIn("nextafterf(remaining", device)
        self.assertIn("isfinite(hit_t)", device)
        self.assertIn("!isfinite(delta.x)", device)
        self.assertIn("rayd::shared::SmallEpsilon", device)

    def test_all_inactive_validation_and_batch_sanitize_are_device_resident(self) -> None:
        cuda = read(
            "src/penetration/penetration.cu"
        )
        host = read("src/penetration/penetration.cpp")
        self.assertIn("!input_active_any && input_active != nullptr", cuda)
        self.assertIn("atomicOr(capacity_failure_state, failure_bit)", cuda)
        self.assertIn("segment_penetration_initialize_cuda", host)
        self.assertIn("segment_penetration_sanitize_cuda", host)
        self.assertIn("capacity_failure_state[0] == 0", cuda)
        self.assertIn("valid[index] = 0u", cuda)
        self.assertIn("num_hits[index] = 0", cuda)
        self.assertIn("reached_target[index] = 0u", cuda)
        self.assertIn("global_primitive_id[index] = -1", cuda)
        self.assertIn("t[index] = -1.0f", cuda)
        self.assertNotIn("overflow[index] = 0u", cuda[cuda.index("sanitize_kernel") :])
        family = host + cuda
        for forbidden in (
            "cudaMemcpyDeviceToHost",
            "cudaStreamSynchronize",
            ".item(",
            "nonzero",
        ):
            self.assertNotIn(forbidden, family)

    def test_tape_freezes_restart_and_ad_never_retraces(self) -> None:
        header = read("include/rayd/torch/integration.h")
        cuda = read(
            "src/penetration/penetration.cu"
        )
        for field in (
            "tape_primitive_id",
            "tape_barycentric",
            "tape_restart_epsilon",
            "tape_restart_branch",
            "tape_restart_tie_mask",
            "tape_direction_denominator_branch",
        ):
            self.assertIn(field, header)
        self.assertIn("backward_kernel", cuda)
        self.assertIn("jvp_kernel", cuda)
        self.assertNotIn("optixTrace", cuda)
        self.assertIn("restart_tie_mask", cuda)
        self.assertIn("direction_denominator_branch", cuda)

    def test_cmake_owns_precise_ptx_native_and_direct_test(self) -> None:
        cmake = read("torch/CMakeLists.txt")
        self.assertIn("rayd_torch_segment_penetration_optix_ptx", cmake)
        self.assertIn("penetration/penetration_optix.cu", cmake)
        self.assertIn("penetration/penetration.cu", cmake)
        self.assertIn("penetration/penetration.cpp", cmake)
        self.assertIn("penetration/penetration.cpp", cmake)
        self.assertIn("--ftz=false", cmake)
        self.assertIn("--prec-div=true", cmake)
        self.assertIn("--prec-sqrt=true", cmake)
        start = cmake.index('OUTPUT "${RAYD_TORCH_SEGMENT_PENETRATION_PTX}"')
        command = cmake[start : cmake.index("add_custom_command(", start)]
        self.assertNotIn("RAYD_TORCH_OPTIX_NVCC_FLAGS", command)
        self.assertIn("rayd_torch_segment_penetration_test", cmake)
        self.assertIn("tests/penetration/segment_penetration_oracle.cu", cmake)
        self.assertIn("NAME rayd_torch_segment_penetration", cmake)

    def test_direct_contract_matrix_is_present(self) -> None:
        direct = read("tests/penetration/segment_penetration_test.cpp")
        for evidence in (
            "plain/tape primal mismatch",
            "exact-D hit count",
            "D=0 clear segment did not reach target",
            "D=0 first hit did not overflow",
            "mixed overflow diagnostic",
            "all-inactive request without mask accepted",
            "mask contradiction failure bit",
            "nonfinite input did not fail device transaction",
            "nonfinite tape was not inert",
            "origin NaN",
            "target NaN",
            "origin +Inf",
            "origin -Inf",
            "target +Inf",
            "target -Inf",
            "finite subtraction overflow",
            "finite squared-norm overflow",
            "input-inactive JVP was nonzero",
            "input-inactive VJP was nonzero",
            "active degenerate reached_target",
            "zero inset reached_target",
            "L2 epsilon",
            "Linf epsilon",
            "inclusive inset endpoint hit",
            "strict full-distance endpoint rejected",
            "geometric normal vs intersect",
            "non-axis enumerated direction bits",
            "non-axis Monte Carlo direction bits",
            "non-axis hit t vs typed intersect",
            "non-axis hit position vs typed intersect",
            "non-axis Monte Carlo hit t vs typed intersect",
            "non-axis Monte Carlo hit position vs typed intersect",
            "non-axis geometric normal bits vs typed intersect",
            "non-axis Monte Carlo shading normal bits vs typed intersect",
            "non-axis enumerated second-normalization bits",
            "non-axis L2 restart epsilon bits",
            "non-axis Linf restart epsilon bits",
            "non-axis post-restart enumerated t bits",
            "non-axis post-restart enumerated position bits",
            "non-axis post-restart enumerated geometric normal bits",
            "non-axis post-restart Monte Carlo t bits",
            "non-axis post-restart Monte Carlo position bits",
            "non-axis post-restart Monte Carlo shading normal bits",
            "fixed-winner JVP/VJP duality",
            "penetration changed current stream",
            "optional zero JVP",
        ):
            self.assertIn(evidence, direct)

    def test_family_is_typed_only_and_documented(self) -> None:
        library = read("src/bindings/library.cpp")
        python_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (TORCH / "python").rglob("*.py")
        )
        self.assertNotIn("segment_penetration", library)
        self.assertNotIn("segment_penetration", python_sources)
        audit = json.loads(read("torch/abi_audit.json"))
        self.assertEqual(
            audit["migration"]["typed_native_candidates"][
                "segment_penetration_complete_family"
            ],
            "dormant_same_graph_batched_optix_fixed_winner_ad",
        )
        self.assertTrue((ROOT / "docs/adr/0033-batched-segment-penetration.md").is_file())
        agents = read("AGENTS.md")
        claude = read("CLAUDE.md")
        self.assertEqual(agents, claude)
        self.assertIn("ADR-0033", agents)


if __name__ == "__main__":
    unittest.main()
