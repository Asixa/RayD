# Copyright Xingyu Chen.
# Tests multi device benchmark evidence.

"""Checks committed multi-device benchmark evidence and GPU CI metadata."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "adr" / "0038-replicated-multi-device-execution.md"
OPERATIONS = ROOT / "docs" / "dev" / "multi_gpu_operations.md"
BENCHMARK = ROOT / "benchmarks" / "torch" / "benchmark_multi_device.py"
WORKFLOW = ROOT / ".github" / "workflows" / "multi_gpu.yml"
SCHEMA = ROOT / "benchmarks" / "multi_device_result.schema.json"
MANIFEST = ROOT / "benchmarks" / "multi_device_manifest.json"
RECORD = ROOT / "benchmarks" / "baselines" / "multi_device_2xa6000_20260727.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_row(text: str, configuration: str, operation: str) -> list[str]:
    pattern = re.compile(
        rf"^\|\s*{re.escape(configuration)}[^|]*\|\s*"
        rf"`?{re.escape(operation)}`?\s*\|(?P<rest>.*)\|$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"missing row {configuration!r}/{operation!r}")
    return [cell.strip() for cell in match.group("rest").split("|")]


class MultiDeviceEvidenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load(SCHEMA)
        cls.manifest = load(MANIFEST)
        cls.record = load(RECORD)
        cls.adr = ADR.read_text(encoding="utf-8")
        cls.operations = OPERATIONS.read_text(encoding="utf-8")

    def test_manifest_pins_schema_and_every_record_by_content(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(self.manifest["benchmark"], "rayd_multi_device")
        schema = self.manifest["schema"]
        self.assertEqual(ROOT / schema["path"], SCHEMA)
        self.assertEqual(schema["sha256"], sha256(SCHEMA))
        self.assertEqual(len(self.manifest["records"]), 1)
        entry = self.manifest["records"][0]
        self.assertEqual(ROOT / entry["path"], RECORD)
        self.assertEqual(entry["sha256"], sha256(RECORD))
        self.assertEqual(entry["provenance_kind"], "historical_documentation_import")

    def test_schema_requires_auditable_machine_and_provenance_fields(self) -> None:
        self.assertEqual(self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], 1)
        required = set(self.schema["required"])
        self.assertEqual(required, {"schema_version", "benchmark", "provenance", "parameters", "machine", "configs"})
        machine_required = set(self.schema["properties"]["machine"]["required"])
        self.assertIn("device_count", machine_required)
        self.assertIn("peer_access", machine_required)
        device_properties = self.schema["properties"]["machine"]["properties"]["devices"]["items"]["properties"]
        self.assertIn("compute_capability", device_properties)
        self.assertIn("peak_memory", self.schema["$defs"])

    def test_historical_record_is_explicitly_not_a_new_execution(self) -> None:
        provenance = self.record["provenance"]
        self.assertEqual(provenance["kind"], "historical_documentation_import")
        self.assertEqual(provenance["measured_at"], "2026-07-27")
        self.assertIn("not a new benchmark execution", provenance["statement"])
        self.assertEqual(
            provenance["source_documents"],
            [
                "docs/adr/0038-replicated-multi-device-execution.md#measured-results",
                "docs/dev/multi_gpu_operations.md#51-measured-2026-07-27",
            ],
        )
        for config in self.record["configs"].values():
            self.assertEqual(config["intersect_offload"]["peak_memory"]["status"], "not_recorded")

    def test_machine_record_does_not_invent_unmeasured_peer_directions(self) -> None:
        machine = self.record["machine"]
        count = machine["device_count"]
        self.assertEqual(count, len(machine["devices"]))
        self.assertEqual(count, 2)
        pairs = machine["peer_access"]["pairs"]
        self.assertEqual(machine["peer_access"]["status"], "partially_recorded")
        self.assertNotIn("all_pairs_accessible", machine["peer_access"])
        self.assertEqual(len(pairs), 1)
        self.assertEqual({(pair["source"], pair["destination"]) for pair in pairs}, {(0, 1)})
        self.assertTrue(all(pair["can_access"] for pair in pairs))
        self.assertIn("reverse direction", self.record["provenance"]["statement"])

    def test_headline_record_matches_both_published_tables(self) -> None:
        expected = {
            "intersect": (19.09, 11.83, 1.61),
            "trace_reflections": (53.33, 28.38, 1.88),
            "accum_dfr_direct": (34.76, 18.83, 1.85),
        }
        for operation, values in expected.items():
            with self.subTest(operation=operation):
                row = self.record["configs"]["compute"][operation]
                self.assertEqual((row["single_ms"], row["multi_ms"], row["speedup"]), values)
                rendered = [f"{values[0]:.2f} ms", f"{values[1]:.2f} ms", f"{values[2]:.2f}x"]
                self.assertEqual(table_row(self.adr, "compute", operation)[-3:], rendered)
                self.assertEqual(table_row(self.operations, "compute", operation)[1:4], rendered)

    def test_correctness_evidence_matches_the_published_claims(self) -> None:
        configs = self.record["configs"]
        for name in ("light", "compute"):
            for operation in ("intersect", "trace_reflections"):
                with self.subTest(config=name, operation=operation):
                    self.assertEqual(configs[name][operation]["bitwise_agreement"], 1.0)
        self.assertEqual(configs["light"]["accum_dfr_direct"]["relative_grid_deviation"], 6.5e-08)
        self.assertEqual(configs["compute"]["accum_dfr_direct"]["relative_grid_deviation"], 2.9e-07)
        for claim in ("6.5e-08", "2.9e-07"):
            self.assertIn(claim, self.adr)
            self.assertIn(claim, self.operations)


class MultiDeviceBenchmarkProducerTests(unittest.TestCase):
    def test_live_benchmark_emits_the_schema_identity_and_audit_fields(self) -> None:
        source = BENCHMARK.read_text(encoding="utf-8")
        for literal in (
            '"schema_version": 1',
            '"benchmark": "rayd_multi_device"',
            '"kind": "live_measurement"',
            '"device_count": len(devices)',
            '"compute_capability": [',
            'machine["peer_access"]',
            '"all_pairs_accessible"',
            '"pairs": peer_pairs',
            'record["peak_memory"]',
            '"master_device_index"',
            '"streamed_bytes"',
            '"concatenated_bytes"',
            '"per_device_streamed_bytes"',
        ):
            with self.subTest(literal=literal):
                self.assertIn(literal, source)
        self.assertRegex(source, r'parser\.add_argument\(\s*"--json"')
        self.assertIn("args.json.parent.mkdir(parents=True, exist_ok=True)", source)

    def test_live_peer_evidence_is_all_pairs_not_only_the_first_two_devices(self) -> None:
        source = BENCHMARK.read_text(encoding="utf-8")
        self.assertIn("for source in devices", source)
        self.assertIn("for destination in devices", source)
        self.assertIn("if source != destination", source)
        self.assertIn("torch.cuda.can_device_access_peer(source, destination)", source)


class MultiDeviceWorkflowEvidenceTests(unittest.TestCase):
    def test_ci_has_explicit_manual_weekly_and_labelled_pr_routes(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("schedule:", workflow)
        self.assertRegex(workflow, r'cron:\s*"17 9 \* \* 1"')
        self.assertIn("pull_request:", workflow)
        self.assertIn("types: [labeled]", workflow)
        self.assertIn("run-multi-gpu-ci", workflow)

    def test_benchmark_failure_is_fatal_and_json_is_retained(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("continue-on-error", workflow)
        self.assertIn("test_multi_device_policy", workflow)
        self.assertIn("test_multi_device_resilience", workflow)
        self.assertIn("--json artifacts/multi_gpu/benchmark.json", workflow)
        self.assertIn("actions/upload-artifact@v4", workflow)
        self.assertIn("if-no-files-found: error", workflow)
        self.assertIn("retention-days: 30", workflow)

    def test_docs_do_not_mistake_a_trigger_for_runner_capacity(self) -> None:
        operations = OPERATIONS.read_text(encoding="utf-8")
        for phrase in (
            "external self-hosted runner",
            "run-multi-gpu-ci",
            "jobs remain queued",
            "provide no acceptance or performance evidence",
            "at least two CUDA devices",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, operations)


if __name__ == "__main__":
    unittest.main()
