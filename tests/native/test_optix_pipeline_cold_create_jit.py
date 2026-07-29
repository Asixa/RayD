# Copyright Xingyu Chen.
# Exercises optix pipeline cold create Dr.Jit in a native smoke test.

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class OptixPipelineColdCreateTests(unittest.TestCase):
    CASES = (
        ("intersect", "tests.scene.test_geometry_jit.GeometryCoreTests.test_constant_hit_returns_minimal_intersection"),
        (
            "shadow_test",
            "tests.scene.test_geometry_jit.GeometryCoreTests."
            "test_shadow_test_returns_hit_mask_without_intersection_payload",
        ),
        (
            "trace_reflections",
            "tests.scene.test_geometry_jit.GeometryCoreTests."
            "test_trace_reflections_cold_pipeline_survives_materialized_ad_inputs",
        ),
        (
            "trace_refl_epc",
            "tests.reflection.test_epc_jit.ReflEpcTests."
            "test_one_bounce_method_of_images_path_is_extracted_in_one_launch",
        ),
        (
            "trace_refl_epc_field",
            "tests.reflection.test_epc_jit.ReflEpcTests.test_epc_field_batched_direct_path_creates_cold_pipeline",
        ),
        (
            "accumulate_reflections",
            "tests.reflection.test_accumulation_jit.ReflectionAccumulationTests."
            "test_accumulate_reflections_large_cold_launch",
        ),
        (
            "accum_dfr_direct",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests.test_accum_dfr_direct_writes_grid",
        ),
        (
            "accum_dfr_direct_ad_custom_op",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests.test_accum_dfr_direct_supports_ad_inputs",
        ),
        (
            "accum_dfr_direct_suffix",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests."
            "test_accum_dfr_direct_suffix_reflection_writes_grid",
        ),
        (
            "accum_dfr",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests."
            "test_accum_dfr_order2_direct_and_keller_writes_grid",
        ),
        (
            "trace_dfr_paths",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests."
            "test_trace_dfr_paths_order1_exports_compact_paths",
        ),
        (
            "trace_dfr_paths_ad",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests."
            "test_trace_dfr_paths_order1_supports_ad_inputs",
        ),
        (
            "build_dfr_coherent_tx_states",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests."
            "test_coherent_tx_state_builder_returns_compact_direct_states",
        ),
        (
            "build_dfr_coherent_higher_candidates",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests."
            "test_coherent_higher_candidate_builder_cold_visibility_filter",
        ),
        (
            "accum_dfr_coherent_direct",
            "tests.diffraction.test_accumulation_jit.DfrAccumulationTests."
            "test_accum_dfr_coherent_direct_writes_direct_field",
        ),
        (
            "visible_native_no_ignore",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_visible_native_backend_cold_create_no_ignore",
        ),
        (
            "visible_native_ignore",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_visible_ignore_uses_one_native_segment_launch",
        ),
        (
            "visible_pair_native_no_ignore",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_visible_pair_native_backend_cold_create_no_ignore",
        ),
        (
            "visible_pair_and_visible_edge_default",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_segment_visibility_ignore_pair_and_axial",
        ),
        (
            "visible_pair_native_backend",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_trace_visibility_native_backend_keeps_optixlaunch_path_available",
        ),
        (
            "visible_pair_large_ignore_table",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_visibility_ignore_tables_accept_more_than_eight_entries",
        ),
        (
            "visible_edge_native",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_visible_edge_native_backend_cold_create",
        ),
        (
            "visible_chain_native_no_ignore",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_visible_chain_native_backend_cold_create_no_ignore",
        ),
        (
            "visible_chain",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_segment_chain_visibility_reports_first_blocker_and_uses_segment_ignores",
        ),
        (
            "visible_chain_native_ignore_blocker",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests."
            "test_segment_chain_visibility_with_ignore_reports_native_blocker",
        ),
        (
            "nearest_edge_point",
            "tests.scene.test_geometry_jit.GeometryCoreTests."
            "test_scene_nearest_edge_point_queries_return_expected_fields_and_batches",
        ),
        (
            "nearest_edge_ray",
            "tests.scene.test_geometry_jit.GeometryCoreTests."
            "test_scene_nearest_edge_ray_queries_use_segment_semantics_and_batches",
        ),
        (
            "nearest_edges_optix_custom_op",
            "tests.visibility.test_visibility_topk_jit.VisibilityAndTopKTests.test_nearest_edges_point_k2",
        ),
        (
            "surfel_intersect",
            "tests.surfel.test_surfel_jit.SurfelCoreTests.test_quad_surfel_intersection_returns_2dgs_fields",
        ),
    )

    def test_public_optix_api_cold_create_matrix(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONSAFEPATH"] = "1"

        for api_name, target in self.CASES:
            with self.subTest(api=api_name):
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", target, "-v"],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=240,
                    check=False,
                )
                combined = result.stdout + "\n" + result.stderr
                self.assertEqual(result.returncode, 0, f"{api_name} cold-create subprocess failed.\n{combined}")
                self.assertNotIn("optixPipelineCreate", combined)
                self.assertNotIn("[COMPILER] COMPILE ERROR", combined)


if __name__ == "__main__":
    unittest.main()
