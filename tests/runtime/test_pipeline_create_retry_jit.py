# Copyright Xingyu Chen.
# Tests Dr.Jit OptiX multipath pipeline-build failure recovery by injecting driver failures in a fresh process.

"""The Dr.Jit backend retries a multipath pipeline whose creation failed with a compile-class OptiX error: after
releasing idle cached pipelines when there are any, and then with the OptiX disk cache disabled. Other failures
are reported without a retry. Faults are armed only around RayD's own multipath launch: a failed pipeline build
inside Dr.Jit itself (for example the one behind `Scene.intersect`) aborts the process in Dr.Jit and is outside
RayD's control.
"""

import unittest

from tests.support.backends import drjit_available
from tests.support.optix_fault_injection import build_wrapper, run_with_wrapper


SCENE = """
import gc
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as pj


def make_scene():
    scene = pj.Scene()
    scene.add_mesh(pj.Mesh(cuda.Array3f([-2.0, 2.0, 2.0, -2.0], [-2.0, -2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]),
                           cuda.Array3i([0, 0], [1, 2], [2, 3])))
    scene.build()
    return scene


ray = pj.Ray(cuda.Array3f([0.1, 5.0], [0.2, 5.0], [1.0, 1.0]), cuda.Array3f([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0]))


def reflections(scene):
    chain = scene.trace_reflections(ray, 2, symbolic=False)
    valid = chain.is_valid()
    dr.eval(valid, chain.t)
    return {"valid": [bool(v) for v in valid], "t": [float(v) for v in chain.t]}


def warm(scene):
    hit = scene.intersect(ray)
    dr.eval(hit.t)


def leave_idle_pipeline():
    # A scene that is gone leaves its multipath pipeline in the cache with no other owner.
    other = make_scene()
    result = other.trace_refl_epc(ray, cuda.Array3f([0.3, 0.3], [0.2, 0.2], [1.0, 1.0]), max_bounces=1)
    dr.eval(result.valid)
    del result, other
    gc.collect()
"""

FAULT_CASE = """
scene = make_scene()
warm(scene)
{setup}
before = counters()
{arm}
error = error_of(lambda: reflections(scene))
during = counters()
disarm()
print(json.dumps({{"before": before, "during": during, "error": error, "after": reflections(scene)}}))
"""

# Two rays by two bounces: only the first ray's first bounce exists; every other slot is a miss at +inf.
EXPECTED_VALID = [True, False, False, False]


def _skip_reason() -> str:
    if not drjit_available():
        return "a Dr.Jit CUDA device and an importable rayd.drjit are required"
    return build_wrapper()[1]


@unittest.skipUnless(drjit_available() and build_wrapper()[0] is not None, _skip_reason())
class DrJitMultipathPipelineRetryTests(unittest.TestCase):
    def run_case(self, arm: str, setup: str = ""):
        return run_with_wrapper(SCENE + FAULT_CASE.format(setup=setup, arm=arm))

    def assert_reflections(self, result) -> None:
        self.assertEqual(result["valid"], EXPECTED_VALID)
        self.assertAlmostEqual(result["t"][0], 1.0, delta=1e-5)
        self.assertEqual(result["t"][1:], [float("inf")] * 3)

    def test_wrapper_is_loaded_for_the_drjit_backend(self):
        data, _ = self.run_case("pass")
        self.assertGreaterEqual(data["before"]["patched_tables"], 1)
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 1)
        self.assertIsNone(data["error"])

    def test_link_failure_with_an_idle_pipeline_is_retried_after_releasing_it(self):
        data, stderr = self.run_case("shim.rayd_fault_arm_pipeline_create(0, 1)", setup="leave_idle_pipeline()")
        self.assertIsNone(data["error"])
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 2)
        self.assertEqual(data["during"]["cache_disable"], data["before"]["cache_disable"])
        self.assertIn("succeeded after releasing 1 idle multipath pipeline(s)", stderr)
        self.assert_reflections(data["after"])

    def test_link_failure_without_idle_pipelines_is_retried_without_the_disk_cache(self):
        data, stderr = self.run_case("shim.rayd_fault_arm_pipeline_create(0, 1)")
        self.assertIsNone(data["error"])
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 2)
        self.assertEqual(data["during"]["cache_disable"] - data["before"]["cache_disable"], 1)
        self.assertEqual(data["during"]["cache_enable"] - data["before"]["cache_enable"], 1)
        self.assertIn("succeeded only with the OptiX disk cache disabled", stderr)
        self.assert_reflections(data["after"])

    def test_repeated_link_failures_report_every_attempt_and_restore_the_cache(self):
        data, _ = self.run_case("shim.rayd_fault_arm_pipeline_create(0, 3)", setup="leave_idle_pipeline()")
        self.assertIsNotNone(data["error"])
        self.assertIn("optixPipelineCreate(multipath)", data["error"])
        self.assertIn("after releasing 1 idle pipeline(s)", data["error"])
        self.assertIn("with the OptiX disk cache disabled", data["error"])
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 3)
        self.assertEqual(data["during"]["cache_disable"] - data["before"]["cache_disable"], 1)
        self.assertEqual(data["during"]["cache_enable"] - data["before"]["cache_enable"], 1)
        self.assert_reflections(data["after"])

    def test_non_compile_failure_is_not_retried(self):
        data, stderr = self.run_case("shim.rayd_fault_arm_set_stack_size(0, 1)")
        self.assertIsNotNone(data["error"])
        self.assertIn("optixPipelineSetStackSize(multipath)", data["error"])
        self.assertEqual(data["during"]["create_calls"] - data["before"]["create_calls"], 1)
        self.assertEqual(data["during"]["cache_disable"], data["before"]["cache_disable"])
        self.assertNotIn("the rebuild succeeded", stderr)
        self.assert_reflections(data["after"])


if __name__ == "__main__":
    unittest.main()
