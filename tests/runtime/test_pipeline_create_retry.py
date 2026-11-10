# Copyright Xingyu Chen.
# Tests Torch OptiX pipeline-build failure recovery by injecting driver failures in a fresh process.

"""The Torch backend retries a multipath pipeline whose creation failed with a compile-class OptiX error, first
after releasing idle cached pipelines and then with the OptiX disk cache disabled, and reports any other failure
as is. The scene and edge pipelines are not retried but must never be left half-built: a failed build is
reported and the next query rebuilds from scratch. Each case loads the fault-injection driver wrapper in a fresh
process, arms it right before the first use of one pipeline, and checks the counters it keeps.
"""

import unittest

import torch

from tests.support.optix_fault_injection import build_wrapper, run_with_wrapper


SCENE = """
import torch
import rayd.torch as rt

vertices = torch.tensor([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]], device="cuda")
faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda", dtype=torch.int32)


def make_scene(**kwargs):
    scene = rt.Scene(**kwargs)
    scene.add_mesh(rt.Mesh(vertices, faces))
    scene.build()
    return scene


# One ray reflects off the square once and leaves; the other misses it.
ray = rt.Ray(torch.tensor([[0.1, 0.2, 1.0], [5.0, 5.0, 1.0]], device="cuda"),
             torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], device="cuda"))
points = torch.tensor([[0.1, -1.9, 0.3], [1.7, 0.4, -0.2]], device="cuda")


def reflections(scene):
    chain = scene.trace_reflections(ray, 2)
    return {"valid": chain.valid.tolist(), "t": chain.t.tolist(), "prim": chain.prim_ids.tolist()}


def hits(scene):
    hit = scene.intersect(ray)
    return {"t": hit.t.tolist(), "prim": hit.global_prim_id.tolist()}


def edges(scene):
    nearest = scene.nearest_edge(points)
    topk = scene.nearest_edges(points, 2)
    return {"distance": nearest.distance.tolist(), "edge": nearest.global_edge_id.tolist(),
            "topk_distance": topk.distances.tolist(), "topk_edge": topk.global_edge_ids.tolist()}
"""

EXPECTED_REFLECTIONS = {"valid": [[True, False], [False, False]], "t": [[1.0, 1.0e8], [1.0e8, 1.0e8]]}


def _wrapper_reason() -> str:
    return build_wrapper()[1] if torch.cuda.is_available() else "CUDA torch is required"


@unittest.skipUnless(torch.cuda.is_available() and build_wrapper()[0] is not None, _wrapper_reason())
class TorchMultipathPipelineRetryTests(unittest.TestCase):
    """Compile-class failures of the shared multipath pipeline are retried twice, with distinct strategies."""

    def run_reflections_with_failures(self, failures: int):
        return run_with_wrapper(
            SCENE
            + f"""
scene = make_scene()
hits(scene)
before = counters()
shim.rayd_fault_arm_pipeline_create(0, {failures})
error = error_of(lambda: reflections(scene))
during = counters()
disarm()
print(json.dumps({{"before": before, "during": during, "error": error, "after": reflections(scene)}}))
"""
        )

    def assert_reflections(self, result) -> None:
        self.assertEqual(result["valid"], EXPECTED_REFLECTIONS["valid"])
        for row, expected_row in zip(result["t"], EXPECTED_REFLECTIONS["t"]):
            for value, expected in zip(row, expected_row):
                self.assertAlmostEqual(value, expected, delta=1e-5 * max(1.0, expected))

    def test_wrapper_is_loaded_for_the_torch_backend(self):
        data, _ = self.run_reflections_with_failures(0)
        self.assertGreaterEqual(data["before"]["patched_tables"], 1)
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 1)
        self.assertIsNone(data["error"])

    def test_one_link_failure_is_retried_after_releasing_idle_pipelines(self):
        data, stderr = self.run_reflections_with_failures(1)
        self.assertIsNone(data["error"])
        self.assertEqual(data["during"]["create_injected"], 1)
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 2)
        self.assertEqual(data["during"]["cache_disable"], data["before"]["cache_disable"])
        self.assertIn("succeeded on retry after releasing this context's idle cached pipelines", stderr)
        self.assertIn("OPTIX_ERROR_PIPELINE_LINK_ERROR", stderr)
        self.assert_reflections(data["after"])

    def test_two_link_failures_are_retried_without_the_disk_cache_and_restore_it(self):
        data, stderr = self.run_reflections_with_failures(2)
        self.assertIsNone(data["error"])
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 3)
        self.assertEqual(data["during"]["cache_disable"] - data["before"]["cache_disable"], 1)
        self.assertEqual(data["during"]["cache_enable"] - data["before"]["cache_enable"], 1)
        self.assertIn("with the OptiX disk cache disabled", stderr)
        self.assert_reflections(data["after"])

    def test_three_link_failures_are_reported_and_the_next_call_rebuilds(self):
        data, _ = self.run_reflections_with_failures(3)
        self.assertIsNotNone(data["error"])
        self.assertIn("optixPipelineCreate(multipath)", data["error"])
        self.assertIn("OPTIX_ERROR_PIPELINE_LINK_ERROR", data["error"])
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 3)
        self.assertEqual(data["during"]["cache_disable"] - data["before"]["cache_disable"], 1)
        self.assertEqual(data["during"]["cache_enable"] - data["before"]["cache_enable"], 1)
        self.assert_reflections(data["after"])

    def test_non_compile_failure_is_not_retried(self):
        data, stderr = run_with_wrapper(
            SCENE
            + """
scene = make_scene()
hits(scene)
before = counters()
shim.rayd_fault_arm_set_stack_size(0, 1)
error = error_of(lambda: reflections(scene))
during = counters()
disarm()
print(json.dumps({"before": before, "during": during, "error": error, "after": reflections(scene)}))
"""
        )
        self.assertIn("optixPipelineSetStackSize", data["error"])
        self.assertEqual(data["during"]["stack_injected"], 1)
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 1)
        self.assertEqual(data["during"]["cache_disable"], data["before"]["cache_disable"])
        self.assertNotIn("succeeded on retry", stderr)
        self.assert_reflections(data["after"])


@unittest.skipUnless(torch.cuda.is_available() and build_wrapper()[0] is not None, _wrapper_reason())
class TorchScenePipelineRebuildTests(unittest.TestCase):
    """A failed scene or edge pipeline build leaves nothing marked ready, so the next query rebuilds it."""

    def test_intersect_pipeline_failures_are_reported_and_rebuilt(self):
        for arm in ("shim.rayd_fault_arm_pipeline_create(0, 1)", "shim.rayd_fault_arm_set_stack_size(0, 1)"):
            with self.subTest(arm=arm):
                data, _ = run_with_wrapper(
                    SCENE
                    + f"""
reference = hits(make_scene(trace_backend="cuda", edge_bvh_backend="cuda"))
scene = make_scene(trace_backend="optix")
{arm}
error = error_of(lambda: hits(scene))
disarm()
print(json.dumps({{"error": error, "after": hits(scene), "reference": reference}}))
"""
                )
                self.assertIsNotNone(data["error"])
                self.assertEqual(data["after"]["prim"], data["reference"]["prim"])
                self.assertEqual(data["after"]["prim"], [1, -1])
                for value, expected in zip(data["after"]["t"], data["reference"]["t"]):
                    self.assertAlmostEqual(value, expected, delta=1e-6)
                self.assertIn("intersect" if "create" in arm else "optixPipelineSetStackSize", data["error"])

    def test_edge_pipeline_partial_build_failure_is_rebuilt_completely(self):
        # The edge build creates the point/ray pipeline and then the top-k pipeline; failing the second one must
        # also discard the first, so the next query builds both again.
        data, _ = run_with_wrapper(
            SCENE
            + """
reference = edges(make_scene(trace_backend="cuda", edge_bvh_backend="cuda"))
scene = make_scene(trace_backend="optix", edge_bvh_backend="optix")
hits(scene)
shim.rayd_fault_arm_pipeline_create(1, 1)
before = counters()
error = error_of(lambda: edges(scene))
during = counters()
disarm()
after = edges(scene)
print(json.dumps({"before": before, "during": during, "after_counters": counters(), "error": error,
                  "after": after, "reference": reference}))
"""
        )
        self.assertIsNotNone(data["error"])
        self.assertEqual(data["after"]["edge"], data["reference"]["edge"])
        self.assertEqual(data["after"]["topk_edge"], data["reference"]["topk_edge"])
        for name in ("distance", "topk_distance"):
            flat_after = torch.tensor(data["after"][name]).flatten().tolist()
            flat_reference = torch.tensor(data["reference"][name]).flatten().tolist()
            for value, expected in zip(flat_after, flat_reference):
                self.assertAlmostEqual(value, expected, delta=1e-6)
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 2)
        self.assertEqual(data["after_counters"]["create_calls"] - data["during"]["create_calls"], 2)
        self.assertIn("optixPipelineCreate(edge", data["error"])


if __name__ == "__main__":
    unittest.main()
