# Copyright Xingyu Chen.
# Tests shared operation contract.

import json
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "operations.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class SharedOperationContractTests(unittest.TestCase):
    def test_schema_v5_has_per_operation_contracts(self):
        self.assertEqual(CONTRACT["version"], 5)
        operations = CONTRACT["operations"]
        self.assertEqual(
            set(operations),
            {
                "intersect",
                "nearest_edge_point",
                "nearest_edge_ray",
                "nearest_edges_topk",
                "visibility",
                "visibility_pair",
                "visibility_edge",
                "visibility_chain",
                "reflection_trace",
                "reflection_accumulation",
                "diffraction_direct",
                "diffraction_chain",
                "sdf_intersect",
                "mixed_scene",
            },
        )
        for name, operation in operations.items():
            with self.subTest(operation=name):
                self.assertTrue(operation["inputs"])
                self.assertIn("result", operation)
                self.assertIn("ad", operation)
                self.assertTrue(operation["derivative_variants"])

    def test_derivative_capabilities_resolve_every_variant_backend_and_domain(self):
        statuses = set(CONTRACT["derivative_manifest"]["statuses"])
        self.assertEqual(statuses, {"supported", "unsupported", "not_applicable"})
        derivatives = CONTRACT["derivative_capabilities"]
        self.assertEqual(set(derivatives), set(CONTRACT["operations"]))
        for operation, variants in derivatives.items():
            with self.subTest(operation=operation):
                self.assertEqual(set(variants), set(CONTRACT["operations"][operation]["derivative_variants"]))
            for variant, metadata in variants.items():
                with self.subTest(operation=operation, variant=variant):
                    self.assertEqual(set(metadata["primal"]), {"drjit", "torch"})
                    self.assertTrue(all(type(value) is bool for value in metadata["primal"].values()))
                    self.assertTrue(metadata["input_domains"])
                for domain, backend_modes in metadata["input_domains"].items():
                    with self.subTest(operation=operation, variant=variant, domain=domain):
                        self.assertEqual(set(backend_modes), {"drjit", "torch"})
                    for backend, modes in backend_modes.items():
                        with self.subTest(operation=operation, variant=variant, domain=domain, backend=backend):
                            self.assertEqual(set(modes), {"vjp", "jvp"})
                            self.assertLessEqual(set(modes.values()), statuses)

    def test_forward_only_variants_are_not_reported_as_not_applicable(self):
        derivatives = CONTRACT["derivative_capabilities"]
        forward_only = {
            ("reflection_accumulation", "accumulate_reflections", "torch"),
            ("diffraction_direct", "trace_dfr_paths", "torch"),
            ("diffraction_direct", "accum_dfr_coherent_direct", "drjit"),
            ("diffraction_direct", "accum_dfr_coherent_direct", "torch"),
        }
        for operation, variant, backend in forward_only:
            domains = derivatives[operation][variant]["input_domains"]
            with self.subTest(operation=operation, variant=variant, backend=backend):
                self.assertTrue(
                    all(modes[backend] == {"vjp": "unsupported", "jvp": "unsupported"} for modes in domains.values())
                )

    def test_reflection_variants_do_not_inherit_the_flat_ad_capability(self):
        reflection = CONTRACT["derivative_capabilities"]["reflection_trace"]
        self.assertEqual(
            reflection["trace_reflections"]["input_domains"]["ray"]["torch"], {"vjp": "supported", "jvp": "supported"}
        )
        self.assertEqual(
            reflection["trace_refl_epc"]["input_domains"]["receiver"]["torch"],
            {"vjp": "unsupported", "jvp": "unsupported"},
        )
        self.assertEqual(
            reflection["trace_refl_epc_field"]["input_domains"]["receiver"]["torch"],
            {"vjp": "unsupported", "jvp": "unsupported"},
        )
        self.assertEqual(
            reflection["trace_refl_epc_field"]["input_domains"]["material"]["torch"],
            {"vjp": "not_applicable", "jvp": "not_applicable"},
        )
        self.assertEqual(
            reflection["trace_refl_epc"]["input_domains"]["plane_geometry"]["torch"],
            {"vjp": "unsupported", "jvp": "unsupported"},
        )
        self.assertEqual(
            CONTRACT["derivative_capabilities"]["mixed_scene"]["transmittance"]["input_domains"]["ray"]["torch"],
            {"vjp": "supported", "jvp": "supported"},
        )

    def test_capability_names_cover_operation_names(self):
        required = set(CONTRACT["required_capability_keys"])
        self.assertLessEqual(set(CONTRACT["operations"]), required)

    def test_invalid_values_and_ray_flags_are_canonical(self):
        constants = CONTRACT["constants"]
        self.assertEqual(constants["invalid_signed_id"], -1)
        self.assertEqual(constants["invalid_unsigned_id"], 0xFFFFFFFF)
        self.assertEqual(constants["ray_flags"], {"None": 0, "Geometric": 1, "ShadingN": 2, "UV": 4, "All": 7})

    def test_intersection_field_order_matches_torch_public_result(self):
        source = (ROOT / "python" / "rayd" / "_impl" / "geometry.py").read_text(encoding="utf-8")
        block = source[
            source.index("class Intersection:") : source.index("    def is_valid", source.index("class Intersection:"))
        ]
        fields = re.findall(r"^    ([a-z][a-z0-9_]*): torch\.Tensor$", block, re.MULTILINE)
        self.assertEqual(fields, CONTRACT["result_contracts"]["intersection"]["canonical_fields"])

    def test_backend_specific_intersection_epsilon_is_explicit(self):
        overrides = CONTRACT["operations"]["intersect"]["backend_overrides"]
        self.assertEqual(overrides["drjit"]["ray_tmin"], 1e-3)
        self.assertEqual(overrides["torch"]["ray_tmin"], 1e-6)
        self.assertNotEqual(overrides["drjit"], overrides["torch"])

    def test_result_differences_are_extensions_not_canonical_reordering(self):
        point = CONTRACT["result_contracts"]["nearest_edge_point"]
        ray = CONTRACT["result_contracts"]["nearest_edge_ray"]
        self.assertNotIn("is_boundary", point["backend_fields"]["torch"])
        self.assertNotIn("is_boundary", ray["backend_fields"]["torch"])
        self.assertIn("is_boundary", point["backend_fields"]["drjit"])
        self.assertIn("is_boundary", ray["backend_fields"]["drjit"])
        self.assertLessEqual(set(point["canonical_semantics"]), set(point["backend_fields"]["torch"]))
        self.assertLessEqual(set(ray["canonical_semantics"]), set(ray["backend_fields"]["torch"]))

    def test_tensor_invalid_ids_match_constant_contract(self):
        invalid = CONTRACT["tensor_contract"]["invalid"]
        signed_invalid = CONTRACT["constants"]["invalid_signed_id"]
        self.assertEqual(invalid["shape_id"], signed_invalid)
        self.assertEqual(invalid["prim_id"], signed_invalid)
        self.assertEqual(invalid["edge_id"], signed_invalid)

    def test_numeric_policy_shared_multipath_constants(self):
        shared = CONTRACT["numeric_policy"]["shared_multipath"]
        self.assertEqual(shared["trace_tmin"], 1e-5)
        self.assertEqual(shared["trace_tmax_finite"], 1e8)
        self.assertEqual(shared["ray_bias"], 1e-5)
        self.assertEqual(shared["min_segment_length"], 2e-5)
        self.assertEqual(shared["epc_barycentric_slack"], 1e-4)
        self.assertEqual(shared["normalize_floor"], 1e-12)
        self.assertEqual(shared["edge_distance_epsilon"], 1e-7)

    def test_numeric_policy_backend_profiles_freeze_divergence(self):
        profiles = CONTRACT["numeric_policy"]["backend_profiles"]
        drjit = profiles["drjit"]
        torch = profiles["torch"]
        self.assertEqual(drjit["ray_tmin"], 1e-3)
        self.assertEqual(drjit["shadow_tmin"], 1e-3)
        self.assertEqual(torch["ray_tmin"], 1e-6)
        self.assertEqual(torch["shadow_tmin"], 1e-5)
        self.assertNotEqual(drjit["ray_tmin"], torch["ray_tmin"])
        self.assertEqual(drjit["endpoint_offset"], 1e-5)
        self.assertEqual(torch["endpoint_offset"], 1e-5)
        self.assertEqual(drjit["parallel_epsilon"], 1e-7)
        self.assertEqual(torch["parallel_epsilon"], 1e-7)
        self.assertFalse(drjit["watertight_triangles"])
        self.assertFalse(torch["watertight_triangles"])
        self.assertEqual(drjit["surfel_endpoint_offset"], 1e-3)
        self.assertEqual(torch["surfel_endpoint_offset"], 1e-3)
        self.assertTrue(CONTRACT["numeric_policy"]["notes"])

    def test_miss_sentinels(self):
        sentinels = CONTRACT["miss_sentinels"]
        self.assertEqual(sentinels["distance"], "inf")
        self.assertEqual(sentinels["reflection_trace_distance"], 1e8)
        self.assertEqual(sentinels["invalid_id"], -1)
        self.assertNotEqual(sentinels["reflection_trace_distance"], sentinels["distance"])

    def test_edge_topk_tie_break_asymmetry_is_recorded(self):
        tie = CONTRACT["edge_topk_tie_break"]
        self.assertEqual(tie["contract"], "(distance_squared, global_edge_id)")
        self.assertEqual(tie["implemented_in"], ["shared_bvh", "torch_topk"])
        self.assertEqual(tie["not_implemented_in"], ["drjit_optix", "drjit_native_bvh"])
        self.assertTrue(tie["note"])

    def test_operation_integration_matrix(self):
        expected = {
            "intersect": {"drjit": ["jit_symbolic"], "torch": ["eager_native"]},
            "nearest_edge_point": {"drjit": ["eager_native"], "torch": ["eager_native"]},
            "nearest_edge_ray": {"drjit": ["eager_native"], "torch": ["eager_native"]},
            "nearest_edges_topk": {"drjit": ["eager_native"], "torch": ["eager_native"]},
            "visibility": {"drjit": ["jit_symbolic", "eager_native"], "torch": ["eager_native"]},
            "visibility_pair": {"drjit": ["jit_symbolic", "eager_native"], "torch": ["eager_native"]},
            "visibility_edge": {"drjit": ["jit_symbolic", "eager_native"], "torch": ["eager_native"]},
            "visibility_chain": {"drjit": ["jit_symbolic", "eager_native"], "torch": ["eager_native"]},
            "reflection_trace": {"drjit": ["jit_symbolic", "eager_native"], "torch": ["eager_native"]},
            "reflection_accumulation": {"drjit": ["eager_native"], "torch": ["eager_native"]},
            "diffraction_direct": {"drjit": ["eager_native"], "torch": ["eager_native"]},
            "diffraction_chain": {"drjit": ["eager_native"], "torch": ["eager_native"]},
            "sdf_intersect": {"drjit": ["eager_native"], "torch": ["eager_native"]},
            "mixed_scene": {"drjit": ["eager_native"], "torch": ["eager_native"]},
        }
        operations = CONTRACT["operations"]
        self.assertEqual(set(expected), set(operations))
        for name, integration in expected.items():
            with self.subTest(operation=name):
                self.assertEqual(operations[name]["integration"], integration)

    def test_operation_shardability_classification(self):
        # ADR-0038 classifies every operation once: how it shards at all, and
        # what the Torch replicated layer does with it. `single_device` means
        # the operation is outside the Scene surface that layer wraps, not that
        # it is missing an entry.
        expected = {
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
        }
        declared = CONTRACT["shardability_classes"]
        operations = CONTRACT["operations"]
        self.assertEqual(set(expected), set(operations))
        for name, (klass, disposition) in expected.items():
            with self.subTest(operation=name):
                shardability = operations[name]["shardability"]
                self.assertEqual(shardability["class"], klass)
                self.assertEqual(shardability["torch_multi_device"], disposition)
                self.assertIn(klass, declared["classes"])
                self.assertIn(disposition, declared["torch_multi_device"])

    def test_variant_shardability_resolves_every_family_variant(self):
        declared = CONTRACT["shardability_classes"]
        for name, operation in CONTRACT["operations"].items():
            shardability = operation["shardability"]
            if "variant_specific" not in (shardability["class"], shardability["torch_multi_device"]):
                self.assertNotIn("variant_shardability", shardability)
                continue
            variants = shardability["variant_shardability"]
            with self.subTest(operation=name):
                self.assertEqual(set(variants), set(operation["derivative_variants"]))
            for variant, metadata in variants.items():
                with self.subTest(operation=name, variant=variant):
                    self.assertIn(metadata["class"], declared["classes"])
                    self.assertIn(metadata["torch_multi_device"], declared["torch_multi_device"])

    def test_shardability_lane_window_defaults_are_declared(self):
        window = CONTRACT["shardability_classes"]["lane_window"]
        self.assertEqual(window["parameters"], ["lane_offset", "lane_count"])
        self.assertEqual(window["defaults"], {"lane_offset": 0, "lane_count": -1})
        self.assertEqual(window["warp_alignment"], 32)
        self.assertTrue(window["invariance"])

    def test_raw_hit_result_contract(self):
        raw_hit = CONTRACT["result_contracts"]["raw_hit"]
        self.assertEqual(raw_hit["fields"], ["t", "bary_u", "bary_v", "global_prim_id", "shape_id", "local_prim_id"])
        self.assertEqual(raw_hit["sizeof_bytes"], 24)
        self.assertEqual(raw_hit["miss"]["t"], "inf")
        self.assertEqual(raw_hit["miss"]["global_prim_id"], -1)
        self.assertEqual(raw_hit["blocker_fields"], ["global_prim_id"])


if __name__ == "__main__":
    unittest.main()
