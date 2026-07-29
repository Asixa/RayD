# Copyright Xingyu Chen.
# Tests multiview example Dr.Jit.

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "drjit" / "basics" / "surfel_multiview_color_fit.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("surfel_multiview_color_fit", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SurfelMultiviewColorExampleTests(unittest.TestCase):
    def test_degree_one_sh_color_uses_view_direction_coefficients(self):
        module = load_example_module()
        coeffs = np.zeros((2, 3, 4), dtype=np.float32)
        coeffs[:, :, 0] = 0.25
        coeffs[0, 0, 1] = 0.50
        coeffs[0, 1, 2] = 0.25
        coeffs[0, 2, 3] = -0.10
        coeffs[1, :, 1:] = 0.10
        view_dirs = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        rgb = module.evaluate_sh_rgb(coeffs, view_dirs, degree=1)

        np.testing.assert_allclose(rgb[0], [0.75, 0.25, 0.35], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(rgb[1], [0.35, 0.35, 0.35], rtol=0.0, atol=1e-6)

    def test_examples_readme_links_multiview_color_fit(self):
        readme = (ROOT / "examples" / "drjit" / "README.md").read_text(encoding="utf-8")
        self.assertIn("surfel_multiview_color_fit.py", readme)

    def test_convergence_frame_contains_three_panels(self):
        module = load_example_module()
        target = np.zeros((4, 4, 3), dtype=np.float32)
        pred = np.ones((4, 4, 3), dtype=np.float32) * 0.5

        frame = module.make_convergence_frame(target, pred, iteration=2, loss=0.25)

        self.assertEqual(frame.size, (4 * 3 + 32, 4 + 60))

    def test_size_zero_uses_native_square_image_resolution(self):
        module = load_example_module()
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "native.png"
            Image.new("RGBA", (13, 13), (255, 255, 255, 255)).save(image_path)
            args = SimpleNamespace(size=0)

            self.assertEqual(module.resolve_fit_size(args, image_path), 13)

    def test_training_pixel_batch_samples_native_indices(self):
        module = load_example_module()
        rng = np.random.default_rng(4)
        mask = np.zeros((16,), dtype=np.float32)
        mask[[2, 5, 7, 11]] = 1.0
        view = {"mask": mask, "pixel_count": 16}

        indices = module.select_training_pixel_indices(view, batch_size=10, rng=rng, foreground_fraction=0.5)

        self.assertEqual(indices.shape, (10,))
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 16))
        self.assertGreaterEqual(int(np.isin(indices, np.flatnonzero(mask > 0.0)).sum()), 5)

    def test_native_training_defaults_disable_fixed_screen_pruning(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertRegex(source, r'parser\.add_argument\(\s*"--max-screen-radius",\s*type=float,\s*default=0\.0')

    def test_metrics_record_full_resolution_mse(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn('"final_full_image_mse"', source)
        self.assertIn('"initial_full_image_mse"', source)

    def test_training_writes_interrupt_safe_progress_log(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn("append_progress_entry", source)
        self.assertIn("progress.jsonl", source)
        self.assertIn("--progress-every", source)

    def test_training_exposes_resource_stop_guards(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn("--stop-at-vram-gb", source)
        self.assertIn("--stop-at-host-available-gb", source)
        self.assertIn("--max-wall-seconds", source)

    def test_training_batches_do_not_require_full_resolution_ad_rays(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn('view["ray_origins"][indices]', source)
        self.assertIn("rays_ad = None", source)
        self.assertIn("target_ad = None", source)

    def test_capacity_probe_can_skip_full_resolution_previews(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn('choices=["none", "gif", "mp4", "both"]', source)
        self.assertIn("--final-render", source)
        self.assertIn('if args.video_format != "none"', source)

    def test_dense_multiview_options_use_configurable_candidate_hits(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn("--max-candidate-hits", source)
        self.assertIn("opts.max_candidate_hits = int(args.max_candidate_hits)", source)

    def test_training_loop_keeps_coefficients_gpu_resident(self):
        source = EXAMPLE.read_text(encoding="utf-8")
        start = source.index("def fit_color_coefficients")
        end = source.index("\ndef make_convergence_frame", start)
        fit_body = source[start:end]

        self.assertIn("class GpuSurfelFitState", source)
        self.assertIn("self.train_scene.build()", source)
        self.assertIn("self.preview_scene.build()", source)
        self.assertIn("state.optimizer_step", fit_body)
        self.assertNotIn("grad_coeffs = np.zeros_like", fit_body)
        self.assertNotIn("np.array([float(grad", fit_body)
        self.assertNotIn("active_coeff", fit_body)

    def test_full_loss_mask_keeps_background_constrained(self):
        module = load_example_module()
        foreground = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        full = module.select_loss_mask(foreground, "full")
        fg = module.select_loss_mask(foreground, "foreground")

        np.testing.assert_allclose(full, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(fg, foreground)

    def test_example_has_no_depth_or_fallback_initializer(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertNotIn("fallback_sphere", source)
        self.assertNotIn("initialize_from_depth", source)
        self.assertNotIn("resolve_frame_depth", source)
        self.assertNotIn("depth_path", source)
        self.assertNotIn("--init-from-depth", source)
        self.assertNotIn("--depth-scale", source)

    def test_random_initializer_returns_2dgs_field(self):
        module = load_example_module()
        args = SimpleNamespace(surfels=16, seed=5, random_radius=1.25, initial_scale=0.08)

        centers, colors, tangent_u, tangent_v, info = module.initialize_random_surfel_field(args)

        self.assertEqual(info["source"], "random")
        self.assertEqual(centers.shape, (16, 3))
        self.assertEqual(colors.shape, (16, 3))
        self.assertEqual(tangent_u.shape, (16, 3))
        self.assertEqual(tangent_v.shape, (16, 3))
        self.assertLessEqual(float(np.abs(centers).max()), 1.25)
        self.assertTrue(np.all(np.linalg.norm(tangent_u, axis=1) > 0.0))
        self.assertTrue(np.all(np.linalg.norm(tangent_v, axis=1) > 0.0))

    def test_training_render_options_keep_configured_background(self):
        source = EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("background_rgb=[self.background, self.background, self.background]", source)
        self.assertIn("--render-normals", source)
        self.assertIn("render_normals=args.render_normals", source)
        self.assertIn("self.render_options", source)
        fit_start = source.index("def fit_color_coefficients")
        fit_end = source.index("\ndef make_convergence_frame", fit_start)
        fit_body = source[fit_start:fit_end]
        self.assertIn("state.render_options", fit_body)
        self.assertNotIn("SurfelRenderOptions.rgb(sh_degree=state.degree)", fit_body)

    def test_opacity_is_a_gpu_resident_trainable_parameter(self):
        source = EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("self.opacity_values = ad.Float(opacity.tolist())", source)
        self.assertIn("dr.enable_grad(self.opacity_values)", source)
        self.assertIn(
            "self.opacity_values, self.opacity_momentum, self.opacity_velocity = self.optimizer_step_param", source
        )
        self.assertIn("--fit-opacity", source)

    def test_geometry_is_gpu_resident_and_trainable(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn("self.center_values = array3_ad(centers)", source)
        self.assertIn("dr.enable_grad(self.center_values)", source)
        self.assertIn("dr.enable_grad(self.tangent_u_values)", source)
        self.assertIn("dr.enable_grad(self.tangent_v_values)", source)
        self.assertIn("self.rebuild_train_scene()", source)
        self.assertIn(
            "self.center_values, self.center_momentum, self.center_velocity = self.optimizer_step_param", source
        )

    def test_geometry_updates_use_surfel_scene_update_geometry(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn("self.train_scene.update_geometry", source)
        self.assertIn("self.preview_scene.update_geometry", source)
        self.assertIn("train_geometry_updates", source)
        self.assertIn("preview_geometry_updates", source)

    def test_metrics_include_reconstruction_performance_summary(self):
        source = EXAMPLE.read_text(encoding="utf-8")

        self.assertIn('"performance"', source)
        self.assertIn('"iterations_per_second"', source)
        self.assertIn('"train_pixels_per_second"', source)
        self.assertIn('"geometry_update_count"', source)

    def test_long_training_lr_schedule_warms_up_and_decays(self):
        module = load_example_module()

        start = module.scheduled_learning_rate(1, base_lr=0.01, final_lr=0.001, warmup_iters=4, total_iters=100)
        warm = module.scheduled_learning_rate(4, base_lr=0.01, final_lr=0.001, warmup_iters=4, total_iters=100)
        end = module.scheduled_learning_rate(100, base_lr=0.01, final_lr=0.001, warmup_iters=4, total_iters=100)

        self.assertLess(start, warm)
        self.assertAlmostEqual(warm, 0.01, places=6)
        self.assertAlmostEqual(end, 0.001, places=6)

    def test_densify_and_prune_clones_splits_and_removes_opacity_outliers(self):
        module = load_example_module()
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        tangent_u = np.array([[0.05, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
        tangent_v = np.array([[0.0, 0.05, 0.0], [0.0, 0.4, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
        opacity = np.array([0.5, 0.6, 0.001], dtype=np.float32)
        values = np.arange(9, dtype=np.float32)
        grad_norm = np.array([0.2, 0.3, 0.0], dtype=np.float32)
        args = SimpleNamespace(
            densify_grad_threshold=0.1,
            percent_dense=0.1,
            scene_extent=2.0,
            split_child_count=2,
            prune_opacity_threshold=0.01,
            min_scale=0.001,
            max_scale=1.0,
            max_world_scale_fraction=0.5,
            max_screen_radius=0.0,
            max_surfels=8,
            max_new_surfels_per_refine=8,
        )

        result = module.densify_and_prune_surfel_arrays(
            centers, tangent_u, tangent_v, opacity, values, degree=0, grad_norm=grad_norm, args=args, seed=9
        )

        self.assertEqual(result["stats"]["pruned"], 2)
        self.assertEqual(result["stats"]["cloned"], 1)
        self.assertEqual(result["stats"]["split_parents"], 1)
        self.assertEqual(result["stats"]["split_children"], 2)
        self.assertEqual(result["centers"].shape[0], 4)
        self.assertEqual(result["values"].shape[0], 12)
        self.assertFalse(np.any(np.all(np.isclose(result["centers"], [1.0, 0.0, 0.0]), axis=1)))
        self.assertLessEqual(float(module.surfel_scales(result["tangent_u"], result["tangent_v"]).max()), 0.25)

    def test_auto_candidate_hit_capacity_estimates_dense_overlap(self):
        module = load_example_module()
        count = 300
        centers = np.zeros((count, 3), dtype=np.float32)
        tangent_u = np.tile(np.array([[0.2, 0.0, 0.0]], dtype=np.float32), (count, 1))
        tangent_v = np.tile(np.array([[0.0, 0.2, 0.0]], dtype=np.float32), (count, 1))
        opacity = np.full((count,), 0.1, dtype=np.float32)
        views = [
            {
                "ray_origins": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                "ray_directions": np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
            }
        ]
        args = SimpleNamespace(
            max_candidate_hits=0,
            min_candidate_hits=256,
            max_candidate_hit_cap=512,
            candidate_hit_sample_rays=64,
            candidate_hit_quantile=0.99,
            candidate_hit_safety=1.1,
            seed=3,
        )

        estimate = module.estimate_candidate_hit_capacity(centers, tangent_u, tangent_v, opacity, views, args)

        self.assertGreaterEqual(estimate["capacity"], 256)
        self.assertEqual(estimate["capacity"] & (estimate["capacity"] - 1), 0)
        self.assertEqual(estimate["hit_count_max"], count)


if __name__ == "__main__":
    unittest.main()
