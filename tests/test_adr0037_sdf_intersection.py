"""ADR-0037 guard: the SDF intersection decision record is complete and consistent.

Phase 0 of `docs/dev/sdf_intersection_plan.md` produces only the decision record,
so this suite checks the record against itself, against the plan it governs, and
against the contract values it reuses. The contract and code assertions are
written so that they are exact in both states: they pass while `sdf_intersect` is
absent from the shared contracts, and become real cross-checks the moment Phase 4
adds it.

Prose assertions run on whitespace-flattened text so that reflowing a paragraph
is not a test failure; assertions that parse table rows keep the raw text.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = ROOT / "docs" / "adr" / "0037-differentiable-sdf-intersection.md"
ADR_INDEX_PATH = ROOT / "docs" / "adr" / "README.md"
PLAN_PATH = ROOT / "docs" / "dev" / "sdf_intersection_plan.md"
OPERATIONS_PATH = ROOT / "shared" / "contracts" / "operations.json"
PUBLIC_API_PATH = ROOT / "shared" / "contracts" / "public_api.json"
COMPILE_POLICY_PATH = ROOT / "shared" / "contracts" / "compile_policy.json"

CAPABILITY = "sdf_intersect"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def sections(text: str, level: int) -> dict[str, str]:
    """Map heading title -> body for every heading at exactly `level` hashes."""
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


class AdrTestCase(unittest.TestCase):
    def assertPhrase(self, phrase: str, text: str) -> None:
        self.assertIn(flat(phrase), flat(text))

    def assertNoPhrase(self, phrase: str, text: str) -> None:
        self.assertNotIn(flat(phrase), flat(text))


class Adr0037RecordTests(AdrTestCase):
    def setUp(self) -> None:
        self.adr = read(ADR_PATH)
        self.top = sections(self.adr, 2)
        self.decision = sections(self.adr, 3)

    def test_header_block_matches_the_repository_adr_shape(self) -> None:
        head = self.adr.splitlines()[:6]
        self.assertEqual(head[0], "# ADR-0037: Differentiable SDF ray intersection")
        self.assertEqual(head[2], "- Status: Accepted")
        self.assertEqual(head[3], "- Date: 2026-07-26")
        self.assertEqual(head[4], "- Decision ID: `differentiable-sdf-intersection`")
        self.assertTrue(head[5].startswith("- Scope:"))

    def test_record_is_indexed_and_the_sequence_range_is_updated(self) -> None:
        index = read(ADR_INDEX_PATH)
        self.assertIn(
            "| [0037](0037-differentiable-sdf-intersection.md) "
            "| Differentiable SDF ray intersection "
            "| `differentiable-sdf-intersection` | 2026-07-26 | Accepted |",
            index,
        )
        self.assertIn("`0028`-`0037`", index)
        self.assertNotIn("`0028`-`0036`", index)

    def test_required_top_level_sections_are_present(self) -> None:
        self.assertLessEqual(
            {
                "Context",
                "Decision",
                "Contract impact",
                "Consequences",
                "Non-goals",
                "Phase 5 backlog",
                "Stop conditions",
            },
            set(self.top),
        )

    def test_decision_subsections_cover_every_pinned_subject(self) -> None:
        titles = " | ".join(self.decision)
        for subject in (
            "Field representation",
            "Placement",
            "Traced interval",
            "Sphere trace",
            "Outputs and miss semantics",
            "Differentiability",
            "Numeric constants",
            "Validation",
            "Structural constraints",
        ):
            self.assertIn(subject, titles)


class Adr0037RepresentationTests(AdrTestCase):
    def setUp(self) -> None:
        self.decision = sections(read(ADR_PATH), 3)

    def test_grid_is_vertex_centred_trilinear_with_the_core_sign_convention(self) -> None:
        body = self.decision["1. Field representation"]
        for pinned in (
            "[Nx, Ny, Nz]",
            "N_i >= 2",
            "vertex-centered",
            "align_corners=True",
            "negative inside, positive outside",
            "world-metric distances",
            "RayD never bakes a field",
        ):
            self.assertPhrase(pinned, body)

    def test_oriented_box_transform_math_is_fully_written_out(self) -> None:
        body = self.decision["2. Placement: the oriented bounding box"]
        for pinned in (
            "scalar-first",
            "(qw, qx, qy, qz)",
            "**full** side lengths",
            "[-scale_i / 2, +scale_i / 2]",
            "x_l = R^T (x_w - position)",
            "u_i = (x_l_i / scale_i + 0.5) * (N_i - 1)",
            "h_i = scale_i / (N_i - 1)",
            "(grad_l D)_i = (dD/du_i) * (N_i - 1) / scale_i",
            "grad_w D = R * grad_l D",
            "1-2(y^2+z^2)",
            # The rigid-map property is what makes a world-metric field
            # traceable with no Lipschitz correction; it must be stated.
            "rigid",
        ):
            self.assertPhrase(pinned, body)

    def test_slab_clip_defines_the_parallel_axis_and_inside_start_cases(self) -> None:
        body = self.decision["3. Traced interval"]
        for pinned in (
            "t_lo = 0",
            "t_hi = tmax",
            "eps_parallel",
            "t_lo > t_hi` is a miss",
            "starting inside is a supported case",
        ):
            self.assertPhrase(pinned, body)


class Adr0037AlgorithmTests(AdrTestCase):
    def setUp(self) -> None:
        self.march = sections(read(ADR_PATH), 3)[
            "4. Sphere trace with relaxation and sign-flip bisection"
        ]

    def test_relaxed_step_freezes_the_entry_sign(self) -> None:
        self.assertPhrase("t_k+1 = t_k + lambda * sigma * d_k", self.march)
        self.assertPhrase("sigma = +1 if d_0 >= 0 else -1", self.march)
        self.assertPhrase("`(0, 1]`, default `0.9`", self.march)

    def test_termination_order_is_explicit(self) -> None:
        for rule in (
            "`|d_k| < eps_hit` terminates as a hit",
            "sigma * d_k+1 < 0",
            "t_k+1 > t_hi` terminates as a miss",
            "exhausting `max_steps` iterations terminates as a miss",
        ):
            self.assertPhrase(rule, self.march)

    def test_bisection_fallback_is_bounded_and_always_reports_a_hit(self) -> None:
        for pinned in (
            "kSdfBisectionSteps = 32",
            "sigma * D(a) >= 0 > sigma * D(b)",
            "Exhausting the bisection budget still reports a hit",
            "stalled march",
        ):
            self.assertPhrase(pinned, self.march)


class Adr0037OutputAndAdTests(AdrTestCase):
    def setUp(self) -> None:
        decision = sections(read(ADR_PATH), 3)
        self.outputs = decision["5. Outputs and miss semantics"]
        self.ad = decision[
            "6. Differentiability: frozen-winner implicit function theorem"
        ]

    def test_result_fields_and_miss_inertness_are_pinned(self) -> None:
        for pinned in (
            "`t`",
            "`hit_mask`",
            "`position`",
            "`normal`",
            "`steps`",
            "float32 `[N]`",
            "float32 `[N, 3]`",
            "int32 `[N]`",
            "bitwise inert",
            "exact positive zero",
            "contributes no atomic",
            "`t = +inf` is the only non-finite value any output may ever contain",
        ):
            self.assertPhrase(pinned, self.outputs)

    def test_frozen_winner_and_tape_are_defined(self) -> None:
        for pinned in (
            "F(theta) = D(u(x_l(o, w, t*, theta))) = 0",
            "dt*/dtheta = -(dF/dtheta) / g_clamped",
            "g = grad_w D . wh",
            "`t*` (float32 `[N]`)",
            "`hit_mask` (bool `[N]`)",
            "`base_index` (int32",
            "FMA contraction",
        ):
            self.assertPhrase(pinned, self.ad)

    def test_all_six_gradient_inputs_have_a_partial_derivative_row(self) -> None:
        rows = {
            line.split("|")[1].strip()
            for line in self.ad.splitlines()
            if line.startswith("|") and line.count("|") >= 3
        }
        for name in (
            "`values[b + m]`",
            "`origins`",
            "`position`",
            "`directions`",
            "`scale_i`",
            "`rotation_a`",
        ):
            self.assertIn(name, rows)
        # The two internal normalizations must be differentiated through, not
        # asserted away as a caller precondition.
        self.assertPhrase("(I - wh wh^T)", self.ad)
        self.assertPhrase("(I - qh qh^T)", self.ad)
        self.assertPhrase("-(grad_l D)_i * x_l_i / scale_i", self.ad)

    def test_normal_is_recomputed_differentiably_and_its_limits_are_stated(self) -> None:
        for pinned in (
            "recomputed differentiably",
            "C0-discontinuous across voxel faces",
            "`hit_mask` and `steps` carry no derivative",
            "JVP and VJP are exact duals",
        ):
            self.assertPhrase(pinned, self.ad)

    def test_grazing_clamp_and_determinism_are_part_of_the_numeric_contract(self) -> None:
        for pinned in (
            "g_clamped = sign(g) * max(|g|, eps_graze), with sign(0) := +1",
            "|dF/dtheta| / eps_graze",
            "No NaN and no infinity may leave the operation",
            "forward pass is bitwise deterministic",
            "float32 atomics",
        ):
            self.assertPhrase(pinned, self.ad)


class Adr0037NumericConstantTests(AdrTestCase):
    """The reused epsilons must equal the values the shared contract already owns."""

    def setUp(self) -> None:
        self.constants = sections(read(ADR_PATH), 3)["7. Numeric constants"]
        self.operations = json.loads(read(OPERATIONS_PATH))

    def cell(self, name: str) -> str:
        for line in self.constants.splitlines():
            if line.startswith("|") and line.split("|")[1].strip() == name:
                return line.split("|")[2].strip()
        raise AssertionError(f"ADR-0037 numeric table has no row {name!r}")

    def test_reused_epsilons_match_the_shared_operation_contract(self) -> None:
        expected = {
            "`eps_graze`": self.operations["constants"]["epsilon"]["small"],
            "`eps_norm`": self.operations["numeric_policy"]["shared_multipath"][
                "normalize_floor"
            ],
            "`eps_parallel`": self.operations["numeric_policy"]["backend_profiles"][
                "torch"
            ]["parallel_epsilon"],
        }
        for name, value in expected.items():
            self.assertEqual(float(self.cell(name).strip("`")), value, msg=name)

    def test_miss_sentinel_is_the_existing_distance_sentinel(self) -> None:
        self.assertEqual(self.operations["miss_sentinels"]["distance"], "inf")
        self.assertEqual(self.cell("miss `t`"), "`+inf`")
        self.assertPhrase("this operation introduces no new sentinel", read(ADR_PATH))

    def test_new_constants_are_named_and_defaulted(self) -> None:
        self.assertEqual(self.cell("`relaxation` default"), "`0.9`")
        self.assertEqual(self.cell("`max_steps` default"), "`64`")
        self.assertEqual(self.cell("`kSdfBisectionSteps`"), "`32`")
        self.assertEqual(
            self.cell("`eps_hit` default"),
            "`kSdfEpsHitVoxelFraction * h_min`, `kSdfEpsHitVoxelFraction = 1e-3`",
        )

    def test_eps_hit_default_is_derived_on_device_without_a_sync(self) -> None:
        for pinned in (
            "h_min = min_i(scale_i / (N_i - 1))",
            "derived on the device",
            "non-positive sentinel",
            "The operation performs no device-to-host copy, no stream "
            "synchronization, and no host read of any device tensor, anywhere.",
        ):
            self.assertPhrase(pinned, self.constants)

    def test_caller_parameters_and_contract_constants_are_separated(self) -> None:
        self.assertPhrase(
            "`eps_graze`, `eps_norm`, and `eps_parallel` are contract constants "
            "and are not caller parameters",
            self.constants,
        )
        self.assertPhrase("none of them is differentiable", self.constants)


class Adr0037ScopeTests(AdrTestCase):
    def setUp(self) -> None:
        self.adr = read(ADR_PATH)
        self.top = sections(self.adr, 2)
        self.decision = sections(self.adr, 3)

    def test_structural_constraints_match_the_frozen_plan_decisions(self) -> None:
        body = self.decision["9. Structural constraints"]
        for pinned in (
            "`drjit: false, torch: true`",
            "No OptiX and no `Scene`",
            "ptx_sources.json",
            "`nvcc_default` only",
            "ADR-0035",
            "kIntegrationApiVersion",
            "GIL-free",
            "sdf_intersect_forward",
            "sdf_intersect_backward",
            "sdf_intersect_jvp",
        ):
            self.assertPhrase(pinned, body)

    def test_non_goals_carry_the_v1_exclusions(self) -> None:
        body = self.top["Non-goals"]
        for exclusion in (
            "Silhouette",
            "OptiX",
            "Baking",
            "Analytic SDF primitives",
            "Multi-grid batching",
            "Dr.Jit implementation",
            "CPU path",
            "integration.h",
            "BSDF",
        ):
            self.assertPhrase(exclusion, body)

    def test_phase5_backlog_reproduces_the_plan_backlog_exactly(self) -> None:
        body = self.top["Phase 5 backlog"]
        items = re.findall(r"^\d+\. (.+)$", body, re.M)
        self.assertEqual(len(items), 6)
        joined = flat(" ".join(items)).lower()
        for topic in (
            "dr.jit backend port",
            "custom-aabb",
            "silhouette",
            "integration.h",
            "analytic-sdf fast path",
            "multi-grid batching",
        ):
            self.assertIn(topic, joined)
        self.assertPhrase("None of these is authorized by this record", body)

    def test_contract_impact_names_every_file_phase4_must_touch(self) -> None:
        body = self.top["Contract impact"]
        for path in (
            "shared/contracts/public_api.json",
            "shared/contracts/operations.json",
            "tests/test_shared_operation_contract.py",
            "backends/drjit/python/rayd/drjit/_capabilities.py",
            "backends/torch/python/rayd/torch/_capabilities.py",
            "shared/contracts/compile_policy.json",
            "backends/torch/CMakeLists.txt",
            "tests/test_public_api_manifest.py",
            "tests/test_ptx_source_digest.py",
            "tests/test_compile_flag_policy_contract.py",
        ):
            self.assertPhrase(path, body)
        # The four-line _capabilities.py divergence contradicts ADR-0036's prose
        # and the record must say so rather than let it rot.
        self.assertPhrase("ADR-0036", body)
        self.assertPhrase("**four** lines", body)

    def test_stop_conditions_forbid_the_expensive_mistakes(self) -> None:
        body = self.top["Stop conditions"]
        for condition in (
            "fallback",
            "stream synchronization",
            "NaN",
            "frozen discrete decision",
            "sign convention",
            "grazing clamp",
            "missed lane",
            "nvcc_default",
            "ptx_sources.json",
            "dispatcher",
        ):
            self.assertPhrase(condition, body)


class Adr0037PlanConsistencyTests(AdrTestCase):
    """The record must not contradict the plan it was written from."""

    def setUp(self) -> None:
        self.adr = read(ADR_PATH)
        self.plan = read(PLAN_PATH)

    def test_the_record_points_at_its_plan(self) -> None:
        self.assertPhrase("docs/dev/sdf_intersection_plan.md", self.adr)

    def test_shared_scope_decisions_agree_verbatim(self) -> None:
        for shared in (
            "negative inside, positive outside",
            "u_i = (x_l_i / scale_i + 0.5) * (N_i - 1)",
        ):
            self.assertPhrase(shared, self.plan)
            self.assertPhrase(shared, self.adr)

    def test_plan_public_names_are_the_names_the_record_governs(self) -> None:
        for name in ("SdfGrid", "sdf_intersect"):
            self.assertIn(name, self.plan)
            self.assertIn(name, self.adr)


class Adr0037ContractStateTests(AdrTestCase):
    """Phase 4 has not landed; whichever state the contracts are in must be coherent.

    These assertions are exact before and after Phase 4: the capability is either
    absent everywhere, or present everywhere with the values ADR-0037 declares.
    """

    def setUp(self) -> None:
        self.public_api = json.loads(read(PUBLIC_API_PATH))
        self.operations = json.loads(read(OPERATIONS_PATH))
        self.compile_policy = json.loads(read(COMPILE_POLICY_PATH))
        self.declared = CAPABILITY in self.public_api["capability_keys"]

    def test_capability_presence_is_all_or_nothing(self) -> None:
        self.assertEqual(CAPABILITY in self.public_api["apis"], self.declared)
        self.assertEqual(
            CAPABILITY in self.operations["required_capability_keys"], self.declared
        )
        for backend in ("drjit", "torch"):
            entry = self.public_api["backends"][backend]["capabilities"]
            self.assertEqual(CAPABILITY in entry, self.declared, msg=backend)

    def test_declared_capability_would_carry_the_adr_values(self) -> None:
        if not self.declared:
            self.skipTest("Phase 4 has not declared sdf_intersect yet")
        metadata = self.public_api["apis"][CAPABILITY]
        self.assertEqual(metadata["category"], "core")
        self.assertEqual(metadata["stability"], "provisional")
        backends = self.public_api["backends"]
        self.assertFalse(backends["drjit"]["capabilities"][CAPABILITY])
        self.assertTrue(backends["torch"]["capabilities"][CAPABILITY])
        self.assertIn(CAPABILITY, self.operations["operations"])

    def test_no_sdf_translation_unit_may_leave_the_nvcc_default_profile(self) -> None:
        units = self.compile_policy["translation_units"]
        if "sdf" not in json.dumps(units):
            self.skipTest("Phase 3a has not added an SDF translation unit yet")
        for backend, entry in units.items():
            for unit in entry.get("objects", []):
                if "sdf" in unit["source"]:
                    self.assertEqual(unit["profile"], "nvcc_default", msg=backend)


if __name__ == "__main__":
    unittest.main()
