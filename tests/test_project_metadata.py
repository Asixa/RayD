import unittest
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectMetadataTests(unittest.TestCase):
    def test_torch_and_slang_frontends_are_not_shipped(self):
        removed_paths = [
            ROOT / "rayd" / "torch",
            ROOT / "rayd" / "slang",
            ROOT / "tests" / "torch",
            ROOT / "tests" / "slang",
            ROOT / "include" / "rayd" / "slang",
            ROOT / "include" / "rayd_slang.slang",
            ROOT / "src" / "slang_interop.cpp",
        ]

        for path in removed_paths:
            self.assertFalse(path.exists(), f"Unexpected frontend artifact remains: {path}")

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertNotIn("rayd.torch", readme)
        self.assertNotIn("Slang", readme)
        self.assertNotIn("torch =", pyproject)

    def test_core_camera_api_is_not_shipped(self):
        removed_paths = [
            ROOT / "include" / "rayd" / "camera.h",
            ROOT / "src" / "camera.cpp",
        ]

        for path in removed_paths:
            self.assertFalse(path.exists(), f"Unexpected Camera artifact remains: {path}")

        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        bindings = (ROOT / "src" / "rayd.cpp").read_text(encoding="utf-8")
        fwd = (ROOT / "include" / "rayd" / "rayd.h").read_text(encoding="utf-8")
        scene_header = (ROOT / "include" / "rayd" / "scene" / "scene.h").read_text(encoding="utf-8")

        self.assertNotIn("camera.h", cmake)
        self.assertNotIn("camera.cpp", cmake)
        self.assertNotIn("Camera", bindings)
        self.assertNotIn("PrimaryEdgeSample", bindings)
        self.assertNotIn("class Camera", fwd)
        self.assertNotIn("Camera *", scene_header)

    def test_cornell_renderer_uses_example_local_camera(self):
        renderer_camera = ROOT / "examples" / "renderer" / "camera.py"
        cornell_box = ROOT / "examples" / "renderer" / "cornell_box.py"

        self.assertTrue(renderer_camera.is_file(), "Cornell renderer should carry its own example-local camera.")
        renderer_source = cornell_box.read_text(encoding="utf-8")

        self.assertIn("from camera import ExampleCamera", renderer_source)
        self.assertNotIn("rd.Camera", renderer_source)

    def test_readme_matches_pinned_nanobind_version(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('nanobind==2.9.2', pyproject)
        self.assertIn('nanobind==2.9.2', readme)
        self.assertNotIn('nanobind==2.11.0', readme)

    def test_reflection_trace_ptx_header_is_committed(self):
        self.assertTrue(
            (
                ROOT
                / "include"
                / "rayd"
                / "multipath"
                / "reflection_trace_ptx.h"
            ).is_file(),
            "Expected committed reflection_trace PTX header for wheel builds.",
        )

    def test_reflection_epc_ptx_header_is_committed(self):
        self.assertTrue(
            (
                ROOT
                / "include"
                / "rayd"
                / "multipath"
                / "reflection_epc_ptx.h"
            ).is_file(),
            "Expected committed reflection_epc PTX header for wheel builds.",
        )

    def test_multipath_optix_pipeline_uses_dedicated_exception_flags(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        pipelines = (ROOT / "src" / "multipath" / "pipelines.cpp").read_text(encoding="utf-8")

        self.assertIn(
            "set(RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL ${RAYD_OPTIX_MODULE_OPT_LEVEL})",
            cmake,
        )
        self.assertIn("set(RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS 11)", cmake)
        self.assertIn(
            "RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS=${RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS}",
            cmake,
        )
        self.assertIn(
            "module_options.optLevel = RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL;",
            pipelines,
        )
        self.assertIn(
            "pipeline_options.exceptionFlags = RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS;",
            pipelines,
        )
        self.assertNotIn(
            "pipeline_options.exceptionFlags = RAYD_OPTIX_EXCEPTION_FLAGS;",
            pipelines,
        )

    def test_trace_reflections_builds_cold_pipeline_before_drjit_materialization(self):
        source = (ROOT / "src" / "scene" / "scene_multipath.cpp").read_text(encoding="utf-8")

        anchor = source.find("const OptixSceneSelection scenes = select_optix_scenes();")
        self.assertGreaterEqual(anchor, 0, "Missing trace_reflections OptiX scene selection block.")

        pipeline = source.find("ensure_pipeline(reflection_pipeline_", anchor)
        active_mask = source.find("const Mask active_detached = sanitize_reflection_active", anchor)
        ad_geometry_eval = source.find("drjit::eval(triangle_info_.p0", anchor)
        detached_eval = source.find("drjit::eval(broadphase_ray.o", anchor)
        launch = source.find("reflection_pipeline_->launch", anchor)

        for name, pos in (
            ("reflection pipeline ensure", pipeline),
            ("active mask sanitization", active_mask),
            ("AD geometry eval", ad_geometry_eval),
            ("detached launch input eval", detached_eval),
            ("reflection launch", launch),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {name} in trace_reflections.")

        self.assertLess(pipeline, active_mask)
        self.assertLess(active_mask, ad_geometry_eval)
        self.assertLess(ad_geometry_eval, detached_eval)
        self.assertLess(detached_eval, launch)

    def test_visibility_utilities_use_single_trace_segment_pipeline_launches(self):
        source = (ROOT / "src" / "scene" / "scene_multipath.cpp").read_text(encoding="utf-8")
        segment_source = (ROOT / "src" / "multipath" / "segment_visibility.cu").read_text(encoding="utf-8")

        visible_start = source.find("SegmentVisibilityT<Detached> Scene::visible(")
        self.assertGreaterEqual(visible_start, 0, "Missing Scene::visible().")
        visible_end = source.find("\ntemplate <bool Detached>", visible_start + 1)
        self.assertGreaterEqual(visible_end, 0, "Missing end of Scene::visible().")
        visible_body = source[visible_start:visible_end]
        self.assertIn("trace_segment_visibility_native<Detached>", visible_body)
        self.assertNotIn("visible_pair<Detached>", visible_body)

        start = source.find("SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_native(")
        self.assertGreaterEqual(start, 0, "Missing native segment-pair visibility helper.")
        end = source.find("\ntemplate <bool Detached>", start + 1)
        self.assertGreaterEqual(end, 0, "Missing end of native segment-pair visibility helper.")
        body = source[start:end]

        self.assertGreaterEqual(body.count("launch_segment_visibility_detached("), 2)
        self.assertNotIn("out_visible_b", body)
        self.assertIn("params.out_first_blocked_prim[ray]", segment_source)

    def test_public_optix_cold_create_matrix_covers_multipath_apis(self):
        source = (ROOT / "tests" / "drjit" / "test_optix_pipeline_cold_create.py").read_text(
            encoding="utf-8"
        )
        expected_apis = {
            "trace_reflections",
            "trace_refl_epc",
            "trace_refl_epc_field",
            "accumulate_reflections",
            "trace_dfr_paths",
            "accum_dfr_direct",
            "accum_dfr_coherent_direct",
            "accum_dfr",
            "build_dfr_coherent_tx_states",
            "build_dfr_coherent_higher_candidates",
            "visible_pair_native_backend",
            "visible_chain",
        }
        for api in expected_apis:
            self.assertIn(api, source)

    def test_shared_multipath_pipeline_cache_key_includes_pipeline_shape(self):
        pipelines = (ROOT / "src" / "multipath" / "pipelines.cpp").read_text(encoding="utf-8")
        for marker in (
            "config.ptx_size",
            "pipeline_raygen_entries_key(config.raygen_entries)",
            "pipeline_entry_key(config.miss_entry)",
            "pipeline_entry_key(config.closesthit_entry)",
            "pipeline_entry_key(config.anyhit_entry)",
            "config.num_payload_values",
            "config.params_size",
        ):
            self.assertIn(marker, pipelines)

    def test_multipath_pipeline_order_guards_cover_staged_launches(self):
        source = (ROOT / "src" / "scene" / "scene_multipath.cpp").read_text(encoding="utf-8")

        def function_body(signature: str) -> str:
            start = source.find(signature)
            self.assertGreaterEqual(start, 0, f"Missing function signature: {signature}")
            end = source.find("\ntemplate <", start + 1)
            self.assertGreaterEqual(end, 0, f"Missing function end after: {signature}")
            return source[start:end]

        trace_dfr_paths = function_body("DfrPathsT<Detached> Scene::trace_dfr_paths(")
        accum_dfr_direct = function_body("DfrAccumT<Detached> Scene::accum_dfr_direct(")
        accum_dfr = function_body("DfrAccumT<Detached> Scene::accum_dfr(")

        def assert_order(name: str, body: str, before: str, after: str):
            before_pos = body.find(before)
            after_pos = body.find(after)
            self.assertGreaterEqual(before_pos, 0, f"Missing guard marker: {before}")
            self.assertGreaterEqual(after_pos, 0, f"Missing eval marker: {after}")
            self.assertLess(
                before_pos,
                after_pos,
                f"{name} must cold-create its staged OptiX pipeline before Dr.Jit eval.",
            )

        assert_order(
            "trace_dfr_paths split-scene export",
            trace_dfr_paths,
            "ensure_pipeline(diffraction_paths_pipeline_",
            "drjit::eval(tx_positions,",
        )
        assert_order(
            "trace_dfr_paths source visibility",
            trace_dfr_paths,
            "ensure_pipeline(diffraction_paths_source_visibility_primary_pipeline_",
            "diffraction_paths_source_visibility_primary_pipeline_->launch",
        )
        assert_order(
            "trace_dfr_paths target export",
            trace_dfr_paths,
            "ensure_pipeline(diffraction_paths_target_export_primary_pipeline_",
            "diffraction_paths_target_export_primary_pipeline_->launch",
        )
        assert_order(
            "accum_dfr_direct source visibility",
            accum_dfr_direct,
            "ensure_pipeline(diffraction_order1_source_visibility_primary_pipeline_",
            "diffraction_order1_source_visibility_primary_pipeline_->launch",
        )
        assert_order(
            "accum_dfr_direct no-suffix target",
            accum_dfr_direct,
            "ensure_pipeline(diffraction_order1_no_suffix_target_primary_pipeline_",
            "diffraction_order1_no_suffix_target_primary_pipeline_->launch",
        )
        assert_order(
            "accum_dfr_direct suffix first visibility",
            accum_dfr_direct,
            "ensure_pipeline(diffraction_order1_suffix_first_visibility_primary_pipeline_",
            "diffraction_order1_suffix_first_visibility_primary_pipeline_->launch",
        )
        assert_order(
            "accum_dfr_direct suffix target",
            accum_dfr_direct,
            "ensure_pipeline(diffraction_order1_suffix_target_primary_pipeline_",
            "diffraction_order1_suffix_target_primary_pipeline_->launch",
        )
        assert_order(
            "accum_dfr",
            accum_dfr,
            "ensure_pipeline(dfr_pipeline,",
            "drjit::eval(initial_states.edge_index,",
        )


if __name__ == "__main__":
    unittest.main()
