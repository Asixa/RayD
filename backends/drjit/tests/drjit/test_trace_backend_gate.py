"""P1 gate: TraceBackend abstraction + decoupled OptiX GAS build.

Exercises the two axes the P1 acceptance criteria demand:

* OptiX artificially blocked (env ``RAYD_DISABLE_OPTIX=1``) -- ``optix_available()``
  reports False and the default scene automatically selects the CUDA triangle
  and Dr.Jit edge backends while preserving the checked-in OptiX baseline.
* OptiX available -- the default scene reports the OptiX trace backend, a
  ``trace_backend='none'`` scene answers edge queries while triangle traces raise,
  a reserved backend name raises not-implemented, and the golden edge baseline
  still matches bit-for-bit after the refactor.

Each case runs in a fresh subprocess so Dr.Jit/CUDA/OptiX state never leaks
between cases (mirrors ``test_geometry`` / ``test_golden_scenes``).
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `tests.golden` lives at the repository root; see backends/drjit/tests/__init__.py
# for why it resolves from here under both documented invocations.
from tests.golden import scenes as scene_defs  # noqa: E402

EDGE_BASELINE = (
    ROOT / "tests" / "golden" / "baselines" / "optix" / "edge_queries.json"
)


def _run_json(script: str, disable_optix: bool = False, timeout: int = 300):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if disable_optix:
        env["RAYD_DISABLE_OPTIX"] = "1"
    else:
        env.pop("RAYD_DISABLE_OPTIX", None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON.\nSTDERR:\n{result.stderr}")
    return json.loads(lines[-1])


# Reproduce the golden "edge_queries" geometry and queries so the blocked (drjit)
# edge scene can be compared against the checked-in OptiX baseline directly.
_EDGE_SCENE = next(s for s in scene_defs.SCENES if s["name"] == "edge_queries")
_QUAD = _EDGE_SCENE["meshes"][0]

_EDGE_GATE_SCRIPT = """
import json
import drjit.cuda as cuda
import rayd.drjit as rd

VERTS = {verts}
FACES = {faces}


def vec3(points):
    return cuda.Array3f(
        [float(p[0]) for p in points],
        [float(p[1]) for p in points],
        [float(p[2]) for p in points],
    )


def ints(a):
    return [int(v) for v in list(a)]


def bools(a):
    return [int(bool(v)) for v in list(a)]


scene = rd.Scene()
mesh = rd.Mesh(
    vec3(VERTS),
    cuda.Array3i(
        [int(f[0]) for f in FACES],
        [int(f[1]) for f in FACES],
        [int(f[2]) for f in FACES],
    ),
)
scene.add_mesh(mesh)
scene.build()
its = scene.intersect(rd.Ray(
    vec3([[0.25, 0.25, -1.0]]),
    vec3([[0.0, 0.0, 1.0]]),
))

out = {{"optix_available": rd.optix_available(),
       "trace_backend_name": scene.trace_backend_name(),
       "edge_bvh_backend": scene.edge_bvh_backend,
       "intersect_valid": bools(its.is_valid()),
       "is_ready": bool(scene.is_ready()),
       "queries": {{}}}}


def point_query(name, point):
    res = scene.nearest_edge(vec3([point]))
    out["queries"][name] = {{
        "valid": bools(res.is_valid()),
        "shape_id": ints(res.shape_id),
        "edge_id": ints(res.edge_id),
        "global_edge_id": ints(res.global_edge_id),
        "is_boundary": bools(res.is_boundary),
    }}


def ray_query(name, origin, direction, tmax=None):
    ray = rd.Ray(vec3([origin]), vec3([direction]))
    if tmax is not None:
        ray.tmax = cuda.Float([float(tmax)])
    res = scene.nearest_edge(ray)
    out["queries"][name] = {{
        "valid": bools(res.is_valid()),
        "shape_id": ints(res.shape_id),
        "edge_id": ints(res.edge_id),
        "global_edge_id": ints(res.global_edge_id),
        "is_boundary": bools(res.is_boundary),
    }}


def topk_query(name, point, k):
    res = scene.nearest_edges(vec3([point]), k)
    out["queries"][name] = {{
        "query_count": int(res.query_count),
        "k": int(res.k),
        "is_valid": bools(res.is_valid),
        "shape_ids": ints(res.shape_ids),
        "edge_ids": ints(res.edge_ids),
        "global_edge_ids": ints(res.global_edge_ids),
        "is_boundary": bools(res.is_boundary),
    }}


point_query("point_near_boundary_edge", [0.5, -0.2, 0.0])
point_query("point_near_internal_edge", [0.52, 0.48, 0.0])
ray_query("ray_finite_segment", [0.5, 0.0, 1.0], [0.0, 0.0, -1.0], tmax=2.0)
ray_query("ray_infinite", [0.5, 0.0, 1.0], [0.0, 0.0, -1.0])
topk_query("topk_k4", [0.35, 0.2, 0.0], 4)

print(json.dumps(out))
""".format(verts=_QUAD["vertices"], faces=_QUAD["faces"])


class TraceBackendGateBlockedTests(unittest.TestCase):
    """OptiX artificially unavailable (RAYD_DISABLE_OPTIX=1)."""

    def test_optix_available_reports_false(self):
        data = _run_json(
            "import json, rayd.drjit as rd; "
            'print(json.dumps({"optix_available": rd.optix_available()}))',
            disable_optix=True,
        )
        self.assertIs(data["optix_available"], False)

    def test_default_scene_uses_cuda_and_matches_optix_edge_baseline(self):
        data = _run_json(_EDGE_GATE_SCRIPT, disable_optix=True)
        self.assertIs(data["optix_available"], False)
        self.assertEqual(data["trace_backend_name"], "cuda")
        self.assertEqual(data["edge_bvh_backend"], "drjit")
        self.assertEqual(data["intersect_valid"], [1])
        self.assertTrue(data["is_ready"])

        baseline = json.loads(EDGE_BASELINE.read_text(encoding="utf-8"))["queries"]
        for name, produced in data["queries"].items():
            with self.subTest(query=name):
                expected = baseline[name]["discrete"]
                for field, value in produced.items():
                    self.assertEqual(
                        value,
                        expected[field],
                        f"discrete field {field!r} of {name!r} drifted vs OptiX baseline",
                    )

    def test_auto_capabilities_set_device_and_explicit_optix_errors(self):
        script = """
        import json
        import drjit.cuda as cuda
        import rayd.drjit as rd

        out = {}

        current = rd.current_device()
        out["current_device"] = int(current)
        out["set_device"] = int(rd.set_device(current))

        # Both default selectors must resolve to their software implementations.
        scene = rd.Scene()
        mesh = rd.Mesh(
            cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            cuda.Array3i([0], [1], [2]),
        )
        scene.add_mesh(mesh)
        scene.build()
        caps = scene.capabilities()
        out["caps"] = {
            "trace_backend": caps["trace_backend"],
            "optix_available": caps["optix_available"],
            "intersect": caps["intersect"],
            "nearest_edge": caps["nearest_edge"],
            "integration": list(caps["integration"]),
            "edge_backend": caps["edge_backend"],
        }
        its = scene.intersect(rd.Ray(
            cuda.Array3f([0.25], [0.25], [-1.0]),
            cuda.Array3f([0.0], [0.0], [1.0]),
        ))
        out["intersect_valid"] = bool(its.is_valid()[0])

        # trace_backend='optix' must fail at build() naming OptiX unavailable.
        forced = rd.Scene(edge_bvh_backend="drjit", trace_backend="optix")
        forced.add_mesh(rd.Mesh(
            cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            cuda.Array3i([0], [1], [2]),
        ))
        try:
            forced.build()
            out["forced_optix_build"] = {"raised": False}
        except Exception as exc:  # noqa: BLE001
            out["forced_optix_build"] = {"raised": True, "msg": str(exc)}

        # Explicit edge OptiX must likewise fail rather than silently fallback.
        forced_edge = rd.Scene(edge_bvh_backend="optix", trace_backend="cuda")
        forced_edge.add_mesh(rd.Mesh(
            cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            cuda.Array3i([0], [1], [2]),
        ))
        try:
            forced_edge.build()
            out["forced_edge_optix_build"] = {"raised": False}
        except Exception as exc:  # noqa: BLE001
            out["forced_edge_optix_build"] = {"raised": True, "msg": str(exc)}

        print(json.dumps(out))
        """
        data = _run_json(script, disable_optix=True)

        self.assertEqual(data["set_device"], data["current_device"])
        self.assertEqual(data["caps"]["trace_backend"], "cuda")
        self.assertIs(data["caps"]["optix_available"], False)
        self.assertIs(data["caps"]["intersect"], True)
        self.assertIs(data["caps"]["nearest_edge"], True)
        self.assertEqual(data["caps"]["integration"], ["eager_native"])
        self.assertEqual(data["caps"]["edge_backend"], "drjit")
        self.assertTrue(data["intersect_valid"])

        self.assertTrue(data["forced_optix_build"]["raised"])
        self.assertIn("optix", data["forced_optix_build"]["msg"].lower())
        self.assertIn("unavailable", data["forced_optix_build"]["msg"])

        self.assertTrue(data["forced_edge_optix_build"]["raised"])
        self.assertIn("edge_bvh_backend", data["forced_edge_optix_build"]["msg"])
        self.assertIn("unavailable", data["forced_edge_optix_build"]["msg"])

    def test_default_symbolic_reflections_run_eagerly_for_one_to_three_bounces(self):
        data = _run_json(
            """
            import json
            import drjit.cuda as cuda
            import rayd.drjit as rd

            # Two parallel quads form a corridor so one ray has at least three
            # deterministic reflections. The public default symbolic=True path
            # remains usable on the automatically selected CUDA backend: outside
            # a recording region its bounce loop executes eagerly.
            yz = [-2.0, 2.0, 2.0, -2.0]
            zz = [-2.0, -2.0, 2.0, 2.0]
            faces = cuda.Array3i([0, 0], [1, 2], [2, 3])
            scene = rd.Scene()
            scene.add_mesh(rd.Mesh(cuda.Array3f([0.0] * 4, yz, zz), faces))
            scene.add_mesh(rd.Mesh(cuda.Array3f([1.0] * 4, yz, zz), faces))
            scene.build()
            ray = rd.Ray(
                cuda.Array3f([0.5], [0.0], [0.0]),
                cuda.Array3f([1.0], [0.0], [0.0]),
            )
            results = {}
            for max_bounces in (1, 2, 3):
                trace = scene.trace_reflections(ray, max_bounces)
                results[str(max_bounces)] = {
                    "bounce_count": int(trace.bounce_count[0]),
                    "stored_bounces": len(trace.bounces),
                    "prim_ids": [int(b.global_prim_ids[0]) for b in trace.bounces],
                }
            print(json.dumps({
                "trace_backend": scene.trace_backend_name(),
                "results": results,
            }))
            """,
            disable_optix=True,
        )
        self.assertEqual(data["trace_backend"], "cuda")
        for max_bounces in (1, 2, 3):
            result = data["results"][str(max_bounces)]
            self.assertEqual(result["bounce_count"], max_bounces)
            self.assertEqual(result["stored_bounces"], max_bounces)
            self.assertTrue(all(prim_id >= 0 for prim_id in result["prim_ids"]))


class TraceBackendGateAvailableTests(unittest.TestCase):
    """OptiX present (normal environment)."""

    def test_optix_available_reports_true(self):
        data = _run_json(
            "import json, rayd.drjit as rd; "
            'print(json.dumps({"optix_available": rd.optix_available()}))'
        )
        self.assertIs(data["optix_available"], True)

    def test_default_scene_reports_optix_trace_backend(self):
        script = """
        import json
        import rayd.drjit as rd
        scene = rd.Scene()
        caps = scene.capabilities()
        print(json.dumps({
            "trace_backend": caps["trace_backend"],
            "optix_available": caps["optix_available"],
            "intersect": caps["intersect"],
            "integration": list(caps["integration"]),
            "trace_backend_name": scene.trace_backend_name(),
            "edge_bvh_backend": scene.edge_bvh_backend,
        }))
        """
        data = _run_json(script)
        self.assertEqual(data["trace_backend"], "optix")
        self.assertIs(data["optix_available"], True)
        self.assertIs(data["intersect"], True)
        self.assertEqual(data["integration"], ["jit_symbolic", "eager_native"])
        self.assertEqual(data["trace_backend_name"], "optix")
        self.assertEqual(data["edge_bvh_backend"], "optix")

    def test_trace_none_scene_answers_edges_but_not_triangles(self):
        script = """
        import json
        import drjit.cuda as cuda
        import rayd.drjit as rd

        scene = rd.Scene(trace_backend="none", edge_bvh_backend="drjit")
        mesh = rd.Mesh(
            cuda.Array3f([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
            cuda.Array3i([0, 0], [1, 2], [2, 3]),
        )
        scene.add_mesh(mesh)
        scene.build()

        res = scene.nearest_edge(cuda.Array3f([0.5], [-0.2], [0.0]))
        out = {
            "trace_backend_name": scene.trace_backend_name(),
            "nearest_global_edge_id": [int(v) for v in list(res.global_edge_id)],
        }
        try:
            scene.intersect(rd.Ray(
                cuda.Array3f([0.25], [0.25], [-1.0]),
                cuda.Array3f([0.0], [0.0], [1.0]),
            ))
            out["intersect"] = {"raised": False}
        except Exception as exc:  # noqa: BLE001
            out["intersect"] = {"raised": True, "msg": str(exc)}
        print(json.dumps(out))
        """
        data = _run_json(script)
        self.assertEqual(data["trace_backend_name"], "none")
        self.assertEqual(data["nearest_global_edge_id"], [0])
        self.assertTrue(data["intersect"]["raised"])
        self.assertIn("trace backend", data["intersect"]["msg"])

    def test_reserved_backend_name_raises_not_implemented(self):
        # 'cuda' now resolves to the pure-CUDA backend; only 'embree' remains
        # reserved for a later phase.
        script = """
        import json
        import rayd.drjit as rd
        try:
            rd.Scene(trace_backend="embree")
            out = {"raised": False}
        except Exception as exc:  # noqa: BLE001
            out = {"raised": True, "msg": str(exc)}
        print(json.dumps(out))
        """
        data = _run_json(script)
        self.assertTrue(data["raised"])
        self.assertIn("not implemented", data["msg"])

    def test_cuda_backend_constructs_and_reports_capabilities(self):
        script = """
        import json
        import rayd.drjit as rd
        scene = rd.Scene(trace_backend="cuda")
        caps = scene.capabilities()
        print(json.dumps({
            "trace_backend": caps["trace_backend"],
            "intersect": caps["intersect"],
            "shadow_test": caps["shadow_test"],
            "visibility": caps["visibility"],
            "integration": list(caps["integration"]),
            "trace_backend_name": scene.trace_backend_name(),
        }))
        """
        data = _run_json(script)
        self.assertEqual(data["trace_backend"], "cuda")
        self.assertIs(data["intersect"], True)
        self.assertIs(data["shadow_test"], True)
        # P4 fused executor: the CUDA backend now serves the full multipath surface.
        self.assertIs(data["visibility"], True)
        self.assertEqual(data["integration"], ["eager_native"])
        self.assertEqual(data["trace_backend_name"], "cuda")

    def test_golden_edge_queries_still_match_baseline(self):
        # Determinism guard: with OptiX available, the default-backend golden
        # edge scene must remain bit-identical to the checked-in baseline after
        # the trace-backend refactor.
        data = _run_json(
            """
            import json
            from tests.golden.runner import collect_golden
            print(json.dumps(collect_golden()["edge_queries"], sort_keys=True))
            """
        )
        baseline = json.loads(EDGE_BASELINE.read_text(encoding="utf-8"))["queries"]
        for name, record in data["queries"].items():
            with self.subTest(query=name):
                self.assertEqual(record["discrete"], baseline[name]["discrete"])


if __name__ == "__main__":
    unittest.main()
