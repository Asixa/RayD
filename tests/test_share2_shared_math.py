import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDGE_HEADER = ROOT / "shared" / "include" / "rayd" / "shared" / "edge" / "edge_distance_math.h"
REFLECTION_HEADER = (
    ROOT / "shared" / "include" / "rayd" / "shared" / "reflection" / "reflection_geometry.h"
)


class Share2SharedMathTests(unittest.TestCase):
    def test_headers_are_backend_neutral(self):
        for path in (EDGE_HEADER, REFLECTION_HEADER):
            source = path.read_text(encoding="utf-8")
            for forbidden in (
                "at::Tensor",
                "torch/",
                "drjit",
                "nanobind",
                "optix",
                "cudaStream",
                "cudaMalloc",
                "cudaFree",
            ):
                self.assertNotIn(forbidden, source, f"{forbidden} remains in {path.relative_to(ROOT)}")

    def test_edge_formula_surface_is_complete(self):
        source = EDGE_HEADER.read_text(encoding="utf-8")
        for symbol in (
            "point_segment_distance",
            "segment_segment_distance",
            "ray_segment_distance",
            "point_segment_jvp_fixed_winner",
            "point_segment_vjp_fixed_winner",
            "ray_segment_jvp_fixed_winner",
            "ray_segment_vjp_fixed_winner",
        ):
            self.assertIn(symbol, source)

    def test_edge_callers_use_shared_formulas(self):
        callers = {
            "backends/drjit/src/edge/edge_optix.cu": (
                "shared::edge::point_segment_distance",
                "shared::edge::segment_segment_distance",
            ),
            "backends/torch/src/torch_ext/edge/edge_optix.cu": (
                "shared::edge::point_segment_distance",
                "shared::edge::segment_segment_distance",
            ),
            "backends/torch/src/torch_ext/edge/edge_forward.cu": (
                "shared::edge::point_segment_distance",
            ),
            "backends/torch/src/torch_ext/edge/edge_backward.cu": (
                "shared::edge::point_segment_jvp_fixed_winner",
                "shared::edge::point_segment_vjp_fixed_winner",
                "shared::edge::ray_segment_jvp_fixed_winner",
                "shared::edge::ray_segment_vjp_fixed_winner",
            ),
        }
        for relative, symbols in callers.items():
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("<rayd/shared/edge/edge_distance_math.h>", source)
            for symbol in symbols:
                self.assertIn(symbol, source)
        for relative in (
            "backends/drjit/src/edge/edge_optix.cu",
            "backends/torch/src/torch_ext/edge/edge_optix.cu",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("update_segment_best", source)
            self.assertNotIn("const float query_t_line", source)

    def test_reflection_trace_callers_use_shared_primitives(self):
        required = (
            "reflection::orient_normal_against",
            "reflection::reflect_direction",
            "reflection::reflect_point_across_plane",
        )
        shared_device = (
            ROOT / "shared/include/rayd/shared/optix/reflection_trace_device.cuh"
        ).read_text(encoding="utf-8")
        self.assertIn("<rayd/shared/reflection/reflection_geometry.h>", shared_device)
        for symbol in required:
            self.assertIn(symbol, shared_device)

        epc_device = (
            ROOT / "shared/include/rayd/shared/optix/reflection_epc_device.cuh"
        ).read_text(encoding="utf-8")
        self.assertIn("<rayd/shared/reflection/reflection_geometry.h>", epc_device)
        self.assertIn("reflection::intersect_segment_plane", epc_device)
        self.assertIn("reflection::reflect_point_across_plane", epc_device)

        for relative in (
            "backends/drjit/src/multipath/reflection_trace.cu",
            "backends/torch/src/torch_ext/reflection/trace_optix.cu",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("<rayd/shared/optix/reflection_trace_device.cuh>", source)


if __name__ == "__main__":
    unittest.main()
