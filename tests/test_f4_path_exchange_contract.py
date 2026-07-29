# Copyright Xingyu Chen.
# Tests f4 path exchange contract.

import json
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "contracts" / "path_exchange.json").read_text(encoding="utf-8")
)
IMPLEMENTATIONS = {
    "drjit": ROOT / "python" / "rayd" / "_impl" / "path_exchange_jit.py",
    "torch": ROOT / "python" / "rayd" / "_impl" / "path_exchange.py",
}


def load_adapter(backend):
    return runpy.run_path(
        str(IMPLEMENTATIONS[backend])
    )


class F4PathExchangeContractTests(unittest.TestCase):
    def test_schema_covers_required_path_and_interaction_fields(self):
        self.assertEqual(CONTRACT["version"], 1)
        self.assertEqual(
            CONTRACT["layout"]["path_order"],
            [
                "valid", "fixed_winner", "order", "source_index", "receiver_index",
                "provenance", "available_fields", "differentiable_fields",
                "interaction_offset", "interaction_count", "total_length", "delay",
                "aod", "aoa", "field", "power",
            ],
        )
        self.assertEqual(
            CONTRACT["layout"]["interaction_order"],
            ["kind", "global_primitive_id", "global_edge_id", "position", "normal"],
        )
        self.assertEqual(CONTRACT["interaction_fields"]["global_primitive_id"]["id_space"], "scene-global")
        self.assertEqual(CONTRACT["interaction_fields"]["global_edge_id"]["id_space"], "scene-global")

    def test_scope_explicitly_excludes_full_simulation_framework(self):
        excluded = set(CONTRACT["scope"]["excluded"])
        self.assertTrue(
            {"scene loader", "integrator", "channel model", "material model", "antenna model"}
            <= excluded
        )

    def test_fixed_winner_contract_separates_discrete_and_continuous_fields(self):
        ad = CONTRACT["fixed_winner_ad"]
        for name in ("valid", "order", "IDs", "provenance", "interaction offsets"):
            self.assertIn(name, ad["selection"])
        self.assertIn("interaction_position", ad["continuous_fields"])
        self.assertIn("complex_field", ad["continuous_fields"])
        self.assertIn("primal-identical indexing", ad["payload"])

    def test_backend_mapping_records_real_availability_gaps(self):
        mappings = CONTRACT["backend_mappings"]
        self.assertEqual(
            mappings["torch"]["reflection"]["global_primitive_id"],
            "ReflectionChain.prim_ids (scene-global)",
        )
        self.assertIn("position", mappings["torch"]["reflection"]["availability_gaps"])
        for backend in ("drjit", "torch"):
            gaps = mappings[backend]["diffraction"]["availability_gaps"]
            self.assertIn("normal", gaps)
            self.assertIn("aod", gaps)
            self.assertIn("aoa", gaps)

    def test_cpp_contract_freezes_pod_layout_and_enums(self):
        header = (
            ROOT / "include" / "rayd" / "path_exchange.h"
        ).read_text(encoding="utf-8")
        for token in (
            "PathInteractionKind",
            "PathProvenance",
            "PathDerivativeMode",
            "PathRecordBatchView",
            "sizeof(PathRecord) == 96",
            "offsetof(PathRecord, field) == 68",
            "std::is_standard_layout_v",
            "std::is_trivially_copyable_v",
        ):
            self.assertIn(token, header)

    def test_private_implementations_are_identical_and_conversion_parity_holds(self):
        drjit_path = IMPLEMENTATIONS["drjit"]
        torch_path = IMPLEMENTATIONS["torch"]
        self.assertEqual(drjit_path.read_bytes(), torch_path.read_bytes())

        drjit_frontend = (
            ROOT / "python" / "rayd" / "drjit" / "path_exchange.py"
        ).read_text(encoding="utf-8")
        torch_frontend = (
            ROOT / "python" / "rayd" / "torch" / "path_exchange.py"
        ).read_text(encoding="utf-8")
        self.assertIn("rayd._impl.path_exchange_jit", drjit_frontend)
        self.assertIn("rayd._impl.path_exchange", torch_frontend)
        modules = [load_adapter("drjit"), load_adapter("torch")]
        records = []
        for module in modules:
            fields = module["PathDerivativeField"]
            record = module["reflection_path_record"](
                [3, 8],
                segment_lengths=[2.0, 4.0],
                positions=[(0, 0, 1), (0, 0, 3)],
                normals=[(0, 1, 0), (0, 1, 0)],
                source_index=1,
                receiver_index=5,
                delay=2.0e-8,
                aod=(0, 0, 1),
                aoa=(0, 0, 1),
                differentiable_fields=fields.INTERACTION_POSITION | fields.TOTAL_LENGTH,
                derivative_mode=module["PathDerivativeMode"].TANGENT,
                derivative=module["PathDerivative"](total_length=0.5),
                interaction_derivatives=[
                    module["PathInteractionDerivative"](position=(0.1, 0, 0)),
                    module["PathInteractionDerivative"](position=(0.2, 0, 0)),
                ],
            )
            records.append(record.as_exchange_dict())
        self.assertEqual(records[0], records[1])
        self.assertEqual(records[0]["order"], 2)
        self.assertEqual(records[0]["total_length"], 6.0)
        self.assertEqual(records[0]["interactions"][0]["global_primitive_id"], 3)
        self.assertEqual(records[0]["derivative_mode"], 1)
        self.assertEqual(records[0]["derivative"]["total_length"], 0.5)
        self.assertEqual(len(records[0]["interaction_derivatives"]), 2)

    def test_diffraction_adapter_preserves_global_edges_field_and_power(self):
        module = load_adapter("torch")
        record = module["diffraction_path_record"](
            [11, 12],
            positions=[(1, 0, 0), (2, 0, 0)],
            source_index=4,
            receiver_index=9,
            delay=1.0e-8,
            field=(1 + 2j, 2 + 0j, 0 + 1j),
        )
        payload = record.as_exchange_dict()
        self.assertEqual([item["global_edge_id"] for item in payload["interactions"]], [11, 12])
        self.assertAlmostEqual(payload["power"], 10.0)
        self.assertEqual(payload["source_index"], 4)
        self.assertEqual(payload["receiver_index"], 9)

    def test_adapter_rejects_non_fixed_or_non_available_derivatives(self):
        module = load_adapter("drjit")
        with self.assertRaises(ValueError):
            module["PathRecord"](
                True,
                0,
                0,
                0,
                module["PathProvenance"].IMPORTED,
                fixed_winner=False,
            )
        with self.assertRaises(ValueError):
            module["reflection_path_record"](
                [1],
                differentiable_fields=module["PathDerivativeField"].INTERACTION_NORMAL,
            )


if __name__ == "__main__":
    unittest.main()
