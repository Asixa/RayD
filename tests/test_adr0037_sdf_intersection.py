# Copyright Xingyu Chen.
# Tests sdf intersection.

"""Checks the SDF intersection contract against the implementation."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = ROOT / "docs" / "adr" / "0037-differentiable-sdf-intersection.md"
ADR0036_PATH = ROOT / "docs" / "adr" / "0036-backend-mirrored-python-modules.md"
ADR_INDEX_PATH = ROOT / "docs" / "adr" / "README.md"
PLAN_PATH = ROOT / "docs" / "dev" / "sdf_intersection_plan.md"
OPERATIONS_PATH = ROOT / "contracts" / "operations.json"
PUBLIC_API_PATH = ROOT / "contracts" / "public_api.json"
COMPILE_POLICY_PATH = ROOT / "contracts" / "compile_policy.json"
SPHERE_TRACE_PATH = (
    ROOT / "include" / "rayd" / "sdf" / "sphere_trace.h"
)
DEVICE_MATH_PATH = ROOT / "src" / "sdf" / "derivatives.cuh"
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


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def sections(text: str, level: int) -> dict[str, str]:
    """Checks the SDF intersection contract against the implementation."""
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
        self.assertRegex(head[2], r"^- Status: Accepted(?:; .+)?$")
        self.assertIn("ADR-0041", head[2])
        self.assertEqual(head[3], "- Date: 2026-07-26")
        self.assertEqual(head[4], "- Decision ID: `differentiable-sdf-intersection`")
        self.assertTrue(head[5].startswith("- Scope:"))

    def test_record_is_indexed_and_the_sequence_range_is_updated(self) -> None:
        index = read(ADR_INDEX_PATH)
        self.assertIn(
            "| [0037](0037-differentiable-sdf-intersection.md) "
            "| Differentiable SDF ray intersection "
            "| `differentiable-sdf-intersection` | 2026-07-26 "
            "| Accepted; integration include and API-version clauses "
            "superseded by ADR-0041 |",
            index,
        )
        # The sequence sentence must cover 0037; later ADRs extend the range
        # (0038 did), so assert the endpoint is at least 0037 rather than
        # pinning the exact string.
        range_match = re.search(r"`0028`-`(\d{4})`", index)
        self.assertIsNotNone(range_match)
        self.assertGreaterEqual(int(range_match.group(1)), 37)

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
        self.assertPhrase("t_raw_k = t_k + lambda * sigma * d_k", self.march)
        self.assertPhrase("sigma = +1 if d_0 >= 0 else -1", self.march)
        self.assertPhrase("`(0, 1]`, default `0.9`", self.march)

    def test_termination_order_is_explicit(self) -> None:
        for rule in (
            "`|d_k| < eps_hit` terminates as a hit",
            "sigma * d_k+1 < 0",
            "t_raw_k > t_hi` terminates as a miss",
            "exhausting `max_steps` iterations terminates as a miss",
        ):
            self.assertPhrase(rule, self.march)

    def test_the_step_is_clamped_so_no_hit_can_land_outside_the_interval(self) -> None:
        """Checks the SDF intersection contract against the implementation."""
        for pinned in (
            "t_k+1   = min(t_raw_k, t_hi)",
            "The step is clamped to `t_hi` before it is sampled",
            "t_lo <= t* <= t_hi <= tmax",
        ):
            self.assertPhrase(pinned, self.march)

    def test_the_interpolant_is_never_evaluated_outside_its_domain(self) -> None:
        body = sections(read(ADR_PATH), 3)["1. Field representation"]
        self.assertPhrase("u_i := clamp(u_i, 0, N_i - 1)", body)
        self.assertPhrase(
            "`D` is never evaluated at a `u` outside `[0, N_i - 1]`", body
        )

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
    """Checks the SDF intersection contract against the implementation."""

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
    """Checks the SDF intersection contract against the implementation."""

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
        # Section 9: Torch only in v1, and the Dr.Jit port is Phase 5 backlog.
        self.assertFalse(backends["drjit"]["capabilities"][CAPABILITY])
        self.assertTrue(backends["torch"]["capabilities"][CAPABILITY])

    def test_no_sdf_translation_unit_may_leave_the_nvcc_default_profile(self) -> None:
        units = [
            unit
            for unit in self.compile_policy["translation_units"]
            if "/sdf/" in unit["source"]
        ]
        self.assertTrue(units, "the SDF translation units are not declared at all")
        for unit in units:
            with self.subTest(unit=f"{unit['backend']}:{unit['unit']}"):
                self.assertEqual(unit["profile"], "nvcc_default")
                self.assertEqual(unit["kind"], "object")


class Adr0037OperationContractTests(AdrTestCase):
    """Checks the SDF intersection contract against the implementation."""

    def setUp(self) -> None:
        self.operations = json.loads(read(OPERATIONS_PATH))
        self.operation = self.operations["operations"][CAPABILITY]
        self.result = self.operations["result_contracts"]["sdf_intersection"]
        self.constants = sections(read(ADR_PATH), 3)["7. Numeric constants"]

    def adr_cell(self, name: str) -> str:
        for line in self.constants.splitlines():
            if line.startswith("|") and line.split("|")[1].strip() == name:
                return line.split("|")[2].strip().strip("`")
        raise AssertionError(f"ADR-0037 numeric table has no row {name!r}")

    def test_operation_is_torch_only_and_names_its_record(self) -> None:
        self.assertEqual(
            self.operation["integration"], {"drjit": [], "torch": ["eager_native"]}
        )
        self.assertEqual(self.operation["record"], "docs/adr/0037-differentiable-sdf-intersection.md")
        self.assertTrue((ROOT / self.operation["record"]).is_file())

    def test_the_six_differentiable_inputs_are_the_operation_inputs(self) -> None:
        # Section 6 lists six gradient inputs; the contract's input list must
        # contain all of them plus the four non-differentiable host scalars.
        self.assertLessEqual(
            {"values", "position", "rotation", "scale", "origins", "directions"},
            set(self.operation["inputs"]),
        )
        self.assertLessEqual(
            {"tmax", "max_steps", "relaxation", "eps_hit"},
            set(self.operation["inputs"]),
        )
        self.assertPhrase("fixed-winner", self.operation["ad"])
        for name in ("values", "position", "rotation", "scale", "origins", "directions"):
            self.assertPhrase(name, self.operation["ad"])

    def test_numeric_policy_repeats_the_adr_constant_table(self) -> None:
        policy = self.operation["numeric_policy"]
        self.assertEqual(policy["relaxation_default"], float(self.adr_cell("`relaxation` default")))
        self.assertEqual(policy["max_steps_default"], float(self.adr_cell("`max_steps` default")))
        self.assertEqual(policy["bisection_steps"], float(self.adr_cell("`kSdfBisectionSteps`")))
        self.assertEqual(policy["eps_graze"], float(self.adr_cell("`eps_graze`")))
        self.assertEqual(policy["eps_norm"], float(self.adr_cell("`eps_norm`")))
        self.assertEqual(policy["eps_parallel"], float(self.adr_cell("`eps_parallel`")))
        # The reused epsilons are the shared contract's own values, not copies
        # that happen to agree today.
        self.assertEqual(policy["eps_graze"], self.operations["constants"]["epsilon"]["small"])
        self.assertEqual(
            policy["eps_norm"],
            self.operations["numeric_policy"]["shared_multipath"]["normalize_floor"],
        )
        self.assertEqual(
            policy["eps_parallel"],
            self.operations["numeric_policy"]["backend_profiles"]["torch"]["parallel_epsilon"],
        )

    def test_grazing_clamp_and_miss_sentinel_are_declared_not_implied(self) -> None:
        policy = self.operation["numeric_policy"]
        self.assertPhrase(
            "g_clamped = sign(g) * max(|g|, eps_graze) with sign(0) := +1",
            policy["grazing_clamp"],
        )
        self.assertEqual(policy["miss_sentinel"], self.operations["miss_sentinels"]["distance"])
        self.assertEqual(self.result["miss"]["t"], self.operations["miss_sentinels"]["distance"])
        self.assertIs(self.result["miss"]["hit_mask"], False)
        self.assertEqual(self.result["miss"]["position"], 0.0)
        self.assertEqual(self.result["miss"]["normal"], 0.0)
        self.assertPhrase("bitwise inert", self.result["miss_inertness"])
        self.assertPhrase("no atomic", self.result["miss_inertness"])

    def test_eps_hit_default_is_the_device_derived_voxel_fraction(self) -> None:
        policy = self.operation["numeric_policy"]
        self.assertEqual(policy["eps_hit_voxel_fraction"], 1e-3)
        self.assertPhrase("derived on the device", policy["eps_hit_default"])
        self.assertPhrase("min_i(scale_i / (N_i - 1))", policy["eps_hit_default"])

    def test_result_fields_match_the_public_torch_result_type(self) -> None:
        source = read(TORCH_PACKAGE / "geometry.py")
        start = source.index("class SdfIntersection:")
        block = source[start : source.index("@dataclass", start)]
        fields = re.findall(r"^    ([a-z][a-z0-9_]*): torch\.Tensor$", block, re.M)
        self.assertEqual(fields, self.result["canonical_fields"])
        self.assertEqual(fields, self.result["backend_fields"]["torch"])
        # The Dr.Jit port is Phase 5; declaring fields for it would claim a
        # surface that does not exist.
        self.assertEqual(self.result["backend_fields"]["drjit"], [])
        self.assertEqual(self.result["differentiable_fields"], ["t", "position", "normal"])
        self.assertEqual(set(self.result["field_types"]), set(fields))


class Adr0037CodeConstantTests(AdrTestCase):
    """Checks the SDF intersection contract against the implementation."""

    def setUp(self) -> None:
        self.constants = sections(read(ADR_PATH), 3)["7. Numeric constants"]
        self.shared = read(SPHERE_TRACE_PATH)
        self.device_math = read(DEVICE_MATH_PATH)
        self.operations = json.loads(read(OPERATIONS_PATH))

    def test_shared_device_constants_equal_the_adr_table(self) -> None:
        expected = {
            "kSdfBisectionSteps": 32.0,
            "kSdfDefaultMaxSteps": 64.0,
            "kSdfDefaultRelaxation": 0.9,
            "kSdfEpsHitVoxelFraction": 1e-3,
            "kSdfEpsNorm": 1e-12,
            "kSdfEpsParallel": 1e-7,
        }
        for name, value in expected.items():
            with self.subTest(constant=name):
                self.assertEqual(cpp_constant(self.shared, name), value)

    def test_grazing_epsilon_is_the_shared_small_epsilon(self) -> None:
        self.assertEqual(
            cpp_constant(self.device_math, "kSdfEpsGraze"),
            self.operations["constants"]["epsilon"]["small"],
        )

    def test_python_defaults_equal_the_device_defaults(self) -> None:
        source = read(TORCH_PACKAGE / "sdf.py")
        self.assertEqual(
            py_constant(source, "DEFAULT_MAX_STEPS"),
            cpp_constant(self.shared, "kSdfDefaultMaxSteps"),
        )
        self.assertEqual(
            py_constant(source, "DEFAULT_RELAXATION"),
            cpp_constant(self.shared, "kSdfDefaultRelaxation"),
        )
        # Section 7: the host scalar is a non-positive sentinel meaning "derive
        # eps_hit from the resident scale", which is what keeps the operation
        # free of a device-to-host read.
        self.assertLess(py_constant(source, "_EPS_HIT_DEVICE_DERIVED"), 0.0)


class Adr0037CapabilityModuleTests(AdrTestCase):
    """Checks the SDF intersection contract against the implementation."""

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

    def test_this_records_line_is_one_of_the_enumerated_divergences(self) -> None:
        # ADR-0038 added a fifth divergent line (`multi_device_replicated`), so
        # the count belongs to ADR-0036 and to the ADR-0038 guard; what this
        # record still owns is that `sdf_intersect` is one of them, in position.
        drjit = self.sources["drjit"].splitlines()
        torch = self.sources["torch"].splitlines()
        self.assertEqual(len(drjit), len(torch))
        divergent = [
            left.strip().split(":")[0].strip()
            for left, right in zip(drjit, torch)
            if left != right
        ]
        self.assertEqual(
            divergent[:4],
            ["_BACKEND = \"drjit\"", '"surfel"', f'"{CAPABILITY}"', '"torch_compile"'],
        )

    def test_adr0036_was_amended_rather_than_left_false(self) -> None:
        adr0036 = read(ADR0036_PATH)
        self.assertPhrase(f'`"{CAPABILITY}"` (`False` versus `True`, per ADR-0037)', adr0036)
        self.assertNoPhrase("diverges on exactly three lines", adr0036)
        self.assertNoPhrase("diverges on exactly four lines", adr0036)
        self.assertNoPhrase("ADR-0037 adds a fourth divergent line", adr0036)


if __name__ == "__main__":
    unittest.main()
