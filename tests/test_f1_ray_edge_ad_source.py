import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class F1RayEdgeAdSourceTests(unittest.TestCase):
    def test_autograd_calls_fixed_winner_ray_edge_ops(self):
        source = (
            ROOT / "backends/torch/python/rayd/torch/autograd.py"
        ).read_text(encoding="utf-8")
        self.assertIn("nearest_edge_ray_backward_optional", source)
        self.assertIn("nearest_edge_ray_jvp_optional", source)
        self.assertNotIn(
            "def backward(ctx, *grad_outputs):\n        return None, None, None, None, None, None",
            source,
        )

    def test_dispatcher_registers_ray_edge_derivative_ops(self):
        source = (
            ROOT / "backends/torch/src/torch_ext/library.cpp"
        ).read_text(encoding="utf-8")
        for operation in (
            "nearest_edge_ray_backward_optional",
            "nearest_edge_ray_jvp_optional",
        ):
            self.assertIn(f'm.def("{operation}(', source)
            self.assertIn(f'm.impl("{operation}"', source)

    def test_cuda_adapter_uses_shared_ray_segment_derivatives(self):
        source = (
            ROOT / "backends/torch/src/torch_ext/edge/edge_backward.cu"
        ).read_text(encoding="utf-8")
        self.assertIn("shared::edge::ray_segment_vjp_fixed_winner", source)
        self.assertIn("shared::edge::ray_segment_jvp_fixed_winner", source)
        self.assertIn("tape_edge_id[ray_idx]", source)

    def test_tmax_is_a_detached_domain_boundary(self):
        source = (
            ROOT / "backends/torch/python/rayd/torch/autograd.py"
        ).read_text(encoding="utf-8")
        jvp_start = source.index("class _NearestEdgeRayFunction")
        jvp_end = source.index("\ndef nearest_edge_ray", jvp_start)
        ray_ad = source[jvp_start:jvp_end]
        self.assertIn("grad_ray_tmax", ray_ad)
        self.assertIn(
            "return None, grad_vertices, grad_ray_o, grad_ray_d, None, None",
            ray_ad,
        )
        self.assertNotIn("_native_tangent_or_none(grad_ray_tmax)", ray_ad)


if __name__ == "__main__":
    unittest.main()
