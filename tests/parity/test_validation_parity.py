# Copyright Xingyu Chen.
# Tests that both backends reject or answer the same invalid inputs without crashing, hanging or poisoning CUDA.

"""Cross-backend validation parity.

Every case runs once per backend in its own interpreter (see `tests.support.validation_cases`): an invalid
input that reaches a kernel can fault the device, and a sticky CUDA error would poison every later case. Each
record says whether the backend `accepted` (with a summary of its answer) or `rejected` the input, or whether
the process crashed or timed out, and carries `healthy`, the `t` of an ordinary query run afterwards in the
same process. Exception types and messages may differ between the backends; where their answers legitimately
differ, each backend's own documented behaviour is asserted explicitly. Run this module directly to print the
whole case x backend table.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import textwrap
import unittest

from tests.support.backends import both_backends_available, cuda_device_count
from tests.support.validation_cases import outcome_summary, run_backend_case


BACKENDS = ("torch", "drjit")
INF = float("inf")


@dataclass(frozen=True)
class Case:
    """One invalid-input probe; `body` defines `case()` in the adapter vocabulary of both preludes."""

    body: str = ""
    torch_body: str = ""
    drjit_body: str = ""
    timeout: float = 240.0
    gpus: int = 1

    def body_for(self, backend: str) -> str:
        specific = self.torch_body if backend == "torch" else self.drjit_body
        return specific or self.body


def _faces_case(faces: str) -> str:
    return f"""
def case():
    s = scene_from([(QUAD, {faces})])
    return {{"hit": intersect(s, [[0.25, 0.25, -1.0], [0.75, 0.75, -1.0]], [[0.0, 0.0, 1.0]] * 2)}}
"""


def _bad_vertex_case(vertex: str) -> str:
    # Face 0 is entirely finite; only face 1 uses the bad vertex 3. Both trace backends are probed.
    return f"""
def case():
    out = {{}}
    for trace_backend in ("optix", "cuda"):
        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], {vertex}]
        def probe(trace_backend=trace_backend, vertices=vertices):
            s = scene_from([(vertices, QUAD_FACES)], trace_backend=trace_backend)
            return {{"hit": intersect(s, [[0.25, 0.25, -1.0], [0.75, 0.75, -1.0]], [[0.0, 0.0, 1.0]] * 2),
                     "visible": visible(s, [[0.25, 0.25, -1.0]], [[0.25, 0.25, 1.0]]),
                     "bounces": reflections(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]], 1)["bounce_count"],
                     "edge": nearest_point(s, [[0.5, -0.5, 0.0]])}}
        out[trace_backend] = attempt(probe)
    return out
"""


def _fan(count: int) -> str:
    # `count` distinct faces around edge (0, 1): a non-manifold fan, not duplicates.
    return f"""
def case():
    n = {count}
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] + [
        [0.5, math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n)] for i in range(n)]
    faces = [[0, 1, 2 + i] for i in range(n)]
    s = scene_from([(vertices, faces)])
    return {{"edge": nearest_point(s, [[0.5, 0.0, 0.0]]), "hit": intersect(s, [[0.5, 0.3, -2.0]], [[0.0, 0.0, 1.0]])}}
"""


# Degenerate segments for nearest_edge(ray): rows 0..6 use direction -z from (0.5, -0.5, 1) towards edge 0 with the
# listed tmax, row 7 has a zero-length direction and tmax 1. The half-ray answer is distance 0.5 at ray_t 1; a point
# query at the origin is distance sqrt(1.25) at ray_t 0; tmax 0.5 ends at distance sqrt(0.5).
EDGE_RAY_TMAX = ["NAN", "INF", "-INF", "-1.0", "0.0", "0.5", "2.0"]
EDGE_RAY_DEGENERATE_ROWS = (0, 2, 3, 4, 7)
EDGE_BACKENDS = {"torch": ("optix", "cuda"), "drjit": ("optix", "drjit")}

_EDGE_RAY = f"""
def case():
    out = {{}}
    tmax = [{", ".join(EDGE_RAY_TMAX)}, 1.0]
    o = [[0.5, -0.5, 1.0]] * len(tmax)
    d = [[0.0, 0.0, -1.0]] * (len(tmax) - 1) + [[0.0, 0.0, 0.0]]
    for edge_backend in EDGE_BACKENDS + ("auto",):
        s = scene_from([(TRIANGLE, [[0, 1, 2]])], edge_backend=edge_backend)
        out[edge_backend] = nearest_ray(s, o, d, tmax)
    return out
"""

# The update moves vertex 1, so a gradient that read the current geometry differs from the one at query time: the
# barycentric weights of (0.25, 0.25) change from (0.5, 0.25, 0.25) to (0.625, 0.125, 0.25).
_AD_UPDATE = [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5], [0.0, 1.0, 0.5]]
QUERY_TIME_GRAD_Z = [0.5, 0.25, 0.25]

_DRJIT_AD_SCENE = """
def ad_scene(dynamic):
    mesh = pj.Mesh(V(TRIANGLE), F([[0, 1, 2]]))
    v = ad.Array3f(V(TRIANGLE))
    dr.enable_grad(v)
    mesh.vertex_positions = v
    s = pj.Scene()
    s.add_mesh(mesh, dynamic=dynamic)
    s.build()
    ray = pj.RayAD(ad.Array3f([0.25], [0.25], [-1.0]), ad.Array3f([0.0], [0.0], [1.0]))
    return s, mesh, v, ray
"""

_TORCH_AD_SCENE = """
def ad_scene(dynamic, vertices=None):
    v = V(TRIANGLE).requires_grad_() if vertices is None else vertices
    s = rt.Scene()
    s.add_mesh(rt.Mesh(v, F([[0, 1, 2]])), dynamic=dynamic)
    s.build()
    return s, v, make_ray([[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])
"""


def _with(prefix: str, body: str) -> str:
    return prefix + textwrap.dedent(body)


CASES: dict[str, Case] = {
    # Face indices outside [0, V) on multi-face meshes.
    "face_index_past_end": Case(_faces_case("[[0, 1, 2], [1, 4, 2]]")),
    "face_index_huge": Case(_faces_case("[[0, 1, 2], [1, 2000000000, 2]]")),
    "face_index_negative": Case(_faces_case("[[0, 1, 2], [1, -1, 2]]")),
    "face_index_second_mesh": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]]), (QUAD, [[0, 1, 2], [1, 3, 9]])])
            return {"hit": intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])}
        """
    ),
    # Empty geometry.
    "zero_vertices": Case(
        """
        def case():
            s = scene_from([([], [])])
            return {"hit": intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])}
        """
    ),
    "zero_vertices_with_faces": Case(
        """
        def case():
            s = scene_from([([], [[0, 1, 2]])])
            return {"hit": intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])}
        """
    ),
    "zero_faces": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [])])
            return {"hit": intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])}
        """
    ),
    "zero_faces_beside_mesh": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]]), (TRIANGLE, [])])
            return {"hit": intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]]),
                    "edge": nearest_point(s, [[0.5, -0.5, 0.0]])}
        """
    ),
    # Non-finite geometry and queries.
    "nan_vertex": Case(_bad_vertex_case("[NAN, 1.0, 0.0]")),
    "inf_vertex": Case(_bad_vertex_case("[INF, 1.0, 0.0]")),
    "non_finite_ray": Case(
        """
        def case():
            s = scene_from([(QUAD, QUAD_FACES)])
            good = [0.25, 0.25, -1.0]
            origins = [[NAN, 0.25, -1.0], [0.25, 0.25, -INF], good, good, good, good]
            directions = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, NAN, 1.0], [0.0, 0.0, INF], [0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]]
            return {"hit": attempt(lambda: intersect(s, origins, directions)),
                    "bounces": attempt(lambda: reflections(s, origins, directions, 2)["bounce_count"])}
        """
    ),
    "non_finite_tmax": Case(
        """
        def case():
            s = scene_from([(QUAD, QUAD_FACES)])
            return {"hit": attempt(lambda: intersect(s, [[0.25, 0.25, -1.0]] * 5, [[0.0, 0.0, 1.0]] * 5,
                                                     [NAN, INF, -1.0, 0.0, 2.0]))}
        """
    ),
    "non_finite_points": Case(
        """
        def case():
            s = scene_from([(QUAD, QUAD_FACES)])
            points = [[NAN, 0.0, 0.0], [INF, 0.0, 0.0], [0.4, -0.5, 0.0]]
            return {"edge": attempt(lambda: nearest_point(s, points)),
                    "topk": attempt(lambda: nearest_k(s, points, 2)),
                    "edge_ray": attempt(lambda: nearest_ray(s, [[NAN, -0.5, 0.0], [0.5, -0.5, 1.0]],
                                                            [[1.0, 0.0, 0.0], [NAN, 0.0, -1.0]]))}
        """
    ),
    "non_finite_visibility": Case(
        """
        def case():
            s = scene_from([(QUAD, QUAD_FACES)])
            start = [[NAN, 0.0, 0.0], [0.0, 0.0, -1.0], [2.0, 2.0, -1.0], [0.25, 0.25, -1.0]]
            end = [[2.0, 2.0, 1.0], [INF, 0.0, 1.0], [2.0, 2.0, 1.0], [0.25, 0.25, 1.0]]
            return {"visible": visible(s, start, end)}
        """
    ),
    "edge_ray_degenerate_segments": Case(
        "EDGE_BACKENDS = " + repr(EDGE_BACKENDS["torch"]) + "\n" + _EDGE_RAY,
        drjit_body="EDGE_BACKENDS = " + repr(EDGE_BACKENDS["drjit"]) + "\n" + _EDGE_RAY,
    ),
    "zero_length_direction": Case(
        """
        def case():
            s = scene_from([(QUAD, QUAD_FACES)])
            zero = [[0.0, 0.0, 0.0]]
            return {"hit": attempt(lambda: intersect(s, [[0.25, 0.25, -1.0]], zero)),
                    "bounces": attempt(lambda: reflections(s, [[0.25, 0.25, -1.0]], zero, 2)["bounce_count"]),
                    "visible_point_segment": attempt(lambda: visible(s, [[0.25, 0.25, -1.0]], [[0.25, 0.25, -1.0]]))}
        """
    ),
    "empty_batches": Case(
        """
        def case():
            s = scene_from([(QUAD, QUAD_FACES)])
            return {"intersect": attempt(lambda: intersect(s, [], [])),
                    "edge_point": attempt(lambda: nearest_point(s, [])),
                    "edge_ray": attempt(lambda: nearest_ray(s, [], [])),
                    "topk": attempt(lambda: nearest_k(s, [], 2)),
                    "visible": attempt(lambda: visible(s, [], [])),
                    "reflections": attempt(lambda: reflections(s, [], [], 2)["bounce_count"])}
        """
    ),
    # Lifecycle.
    "query_before_build": Case(
        """
        def case():
            s = scene_from([(QUAD, QUAD_FACES)], build=False)
            o, d = [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]]
            return {"intersect": attempt(lambda: intersect(s, o, d)),
                    "edge_point": attempt(lambda: nearest_point(s, [[0.5, -0.5, 0.0]])),
                    "topk": attempt(lambda: nearest_k(s, [[0.5, -0.5, 0.0]], 2)),
                    "visible": attempt(lambda: visible(s, o, [[0.25, 0.25, 1.0]])),
                    "reflections": attempt(lambda: reflections(s, o, d, 2))}
        """
    ),
    "query_with_pending_updates": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]])], dynamic=True)
            update(s, 0, [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
            o, d = [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]]
            out = {"pending": bool(s.has_pending_updates()),
                   "intersect": attempt(lambda: intersect(s, o, d)),
                   "edge_point": attempt(lambda: nearest_point(s, [[0.5, -0.5, 0.0]])),
                   "topk": attempt(lambda: nearest_k(s, [[0.5, -0.5, 0.0]], 2)),
                   "visible": attempt(lambda: visible(s, o, [[0.25, 0.25, 0.25]])),
                   "reflections": attempt(lambda: reflections(s, o, d, 1))}
            s.sync()
            out["after_sync"] = intersect(s, o, d)
            return out
        """
    ),
    # Topology stress.
    "duplicate_faces_x100": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]] * 100)])
            return {"hit": intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]]),
                    "edge": nearest_point(s, [[0.5, -0.5, 0.0]])["distance"]}
        """
    ),
    "duplicate_faces_x5000": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]] * 5000)])
            return {"hit": intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])}
        """
    ),
    "degenerate_faces_x20000": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 0, 0]] * 20000 + [[0, 1, 2]])])
            return {"hit": intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]]),
                    "edge": nearest_point(s, [[0.5, -0.5, 0.0]])["distance"]}
        """
    ),
    "edge_shared_by_256_faces": Case(_fan(256)),
    "edge_shared_by_257_faces": Case(_fan(257)),
    # Argument ranges.
    "nearest_edges_k_range": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]])])
            p = [[0.4, -0.5, 0.0]]
            return {str(k): attempt(lambda k=k: nearest_k(s, p, k))
                    for k in (0, -1, 3, 10, 16, 17, (1 << 31) - 1, 1 << 31, 1 << 40)}
        """
    ),
    "max_bounces_range": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]])])
            o, d = [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]]
            return {str(b): attempt(lambda b=b: reflections(s, o, d, b)["bounce_count"])
                    for b in (0, -1, 70000, 1 << 31, 1 << 40)}
        """,
        timeout=300.0,
    ),
    "update_bad_mesh_id": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]])], dynamic=True)
            out = {str(m): attempt(lambda m=m: update(s, m, TRIANGLE)) for m in (99, -1, 1)}
            out["pending"] = bool(s.has_pending_updates())
            out["intact"] = intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])
            return out
        """
    ),
    "update_wrong_vertex_count": Case(
        """
        def case():
            out = {}
            for name, vertices in (("more", QUAD), ("fewer", TRIANGLE[:2]), ("none", [])):
                s = scene_from([(TRIANGLE, [[0, 1, 2]])], dynamic=True)
                out[name] = attempt(lambda s=s, vertices=vertices: update(s, 0, vertices))
                out[name + "_intact"] = attempt(lambda s=s: intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]]))
                # A rejected update must not leak into the geometry a later build() uses.
                out[name + "_rebuilt"] = attempt(
                    lambda s=s: (s.build(), intersect(s, [[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]]))[1])
            return out
        """
    ),
    "edge_mask_length": Case(
        """
        def case():
            s = scene_from([(TRIANGLE, [[0, 1, 2]])])
            out = {name: attempt(lambda mask=mask: set_mask(s, mask))
                   for name, mask in (("short", [True, True]), ("long", [True] * 4), ("empty", []))}
            out["pending_after_bad_masks"] = bool(s.has_pending_updates())
            out["edge_after_bad_masks"] = nearest_point(s, [[0.4, -0.5, 0.0]])
            set_mask(s, [False, True, True])
            out["pending_after_valid_mask"] = bool(s.has_pending_updates())
            s.sync()
            out["edge_with_valid_mask"] = nearest_point(s, [[0.4, -0.5, 0.0]])
            return out
        """
    ),
    # Derivatives against a geometry-changing sync or a deleted scene.
    "backward_after_sync": Case(
        torch_body=_with(
            _TORCH_AD_SCENE,
            f"""
            def case():
                s, v, ray = ad_scene(True)
                t = s.intersect(ray).t
                update(s, 0, {_AD_UPDATE})
                s.sync()
                t.sum().backward()
                return {{"grad_z": floats(v.grad[:, 2])}}
            """,
        ),
        drjit_body=_with(
            _DRJIT_AD_SCENE,
            f"""
            def case():
                s, mesh, v, ray = ad_scene(True)
                t = s.intersect(ray).t
                update(s, 0, {_AD_UPDATE})
                s.sync()
                dr.backward(dr.sum(t))
                return {{"grad_z": floats(dr.grad(v)[2])}}
            """,
        ),
    ),
    "jvp_after_sync": Case(
        torch_body=_with(
            _TORCH_AD_SCENE,
            f"""
            import torch.autograd.forward_ad as fwad

            def case():
                with fwad.dual_level():
                    tangent = torch.zeros((3, 3), device=DEVICE)
                    tangent[0, 2] = 1.0
                    s, v, ray = ad_scene(True, fwad.make_dual(V(TRIANGLE), tangent))
                    hit = s.intersect(ray)
                    update(s, 0, {_AD_UPDATE})
                    s.sync()
                    return {{"t": attempt(lambda: floats(fwad.unpack_dual(hit.t).tangent)),
                             "p_z": attempt(lambda: floats(fwad.unpack_dual(hit.p).tangent[:, 2]))}}
            """,
        ),
        drjit_body=_with(
            _DRJIT_AD_SCENE,
            f"""
            def case():
                s, mesh, v, ray = ad_scene(True)
                hit = s.intersect(ray)
                update(s, 0, {_AD_UPDATE})
                s.sync()
                dr.set_grad(v, ad.Array3f([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
                dr.forward_to(hit.t, hit.p)
                return {{"t": {{"ok": floats(dr.grad(hit.t))}}, "p_z": {{"ok": floats(dr.grad(hit.p)[2])}}}}
            """,
        ),
    ),
    "scene_deleted_before_backward": Case(
        torch_body=_with(
            _TORCH_AD_SCENE,
            """
            def case():
                s, v, ray = ad_scene(False)
                t = s.intersect(ray).t
                del s
                gc.collect()
                t.sum().backward()
                return {"grad_z": floats(v.grad[:, 2])}
            """,
        ),
        drjit_body=_with(
            _DRJIT_AD_SCENE,
            """
            def case():
                s, mesh, v, ray = ad_scene(False)
                t = s.intersect(ray).t
                del s, mesh
                gc.collect()
                dr.backward(dr.sum(t))
                return {"grad_z": floats(dr.grad(v)[2])}
            """,
        ),
    ),
    # Inputs that live on a second GPU.
    "cross_device_inputs": Case(
        torch_body="""
        def case():
            s = scene_from([(QUAD, QUAD_FACES)])
            o, d = V([[0.25, 0.25, -1.0]], "cuda:1"), V([[0.0, 0.0, 1.0]], "cuda:1")
            def foreign_mesh():
                other = rt.Scene()
                other.add_mesh(rt.Mesh(V(TRIANGLE, "cuda:1"), F([[0, 1, 2]])))
                other.build()
                return floats(other.intersect(make_ray([[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])).t)
            return {"intersect": attempt(lambda: floats(s.intersect(rt.Ray(o, d)).t)),
                    "edge_point": attempt(lambda: floats(s.nearest_edge(o).distance)),
                    "edge_ray": attempt(lambda: floats(s.nearest_edge(rt.Ray(o, d)).distance)),
                    "topk": attempt(lambda: floats(s.nearest_edges(o, 2).distances)),
                    "visible": attempt(lambda: s.visible(o, d).tolist()),
                    "reflections": attempt(lambda: floats(s.trace_reflections(rt.Ray(o, d), 2).t)),
                    "mesh_vertices": attempt(foreign_mesh)}
        """,
        drjit_body="""
        def case():
            pj.set_device(0)
            s = scene_from([(QUAD, QUAD_FACES)])
            pj.set_device(1)
            # Distinct lanes keep these from folding into literals, so eval() gives them device-1 buffers.
            o = cuda.Array3f([0.25, 0.3], [0.25, 0.3], [-1.0, -1.0])
            d = cuda.Array3f([0.0, 0.01], [0.0, 0.01], [1.0, 1.0])
            dr.eval(o, d)
            pj.set_device(0)
            ray = pj.Ray(o, d)
            return {"intersect": attempt(lambda: floats(s.intersect(ray).t)),
                    "edge_point": attempt(lambda: floats(s.nearest_edge(o).distance)),
                    "edge_ray": attempt(lambda: floats(s.nearest_edge(ray).distance)),
                    "topk": attempt(lambda: floats(s.nearest_edges(o, 2).distances)),
                    "visible": attempt(lambda: [bool(x) for x in s.visible(o, d).visible]),
                    "reflections": attempt(lambda: ints(s.trace_reflections(ray, 2).bounce_count))}
        """,
        gpus=2,
    ),
}


def run_matrix(names=None, workers: int = 4, env=None) -> dict[tuple[str, str], dict]:
    """Run every selected case on both backends concurrently; one fresh interpreter per (case, backend)."""
    gpus = cuda_device_count()
    jobs = [
        (name, backend)
        for name, case in CASES.items()
        if (names is None or name in names) and case.gpus <= gpus
        for backend in BACKENDS
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            job: pool.submit(run_backend_case, job[1], CASES[job[0]].body_for(job[1]), CASES[job[0]].timeout, env)
            for job in jobs
        }
        return {job: future.result() for job, future in futures.items()}


@unittest.skipUnless(both_backends_available(), "CUDA Torch, rayd.torch and a CUDA rayd.drjit are required")
class ValidationParityTests(unittest.TestCase):
    matrix: dict[tuple[str, str], dict] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = run_matrix()

    def record(self, name: str, backend: str) -> dict:
        if (name, backend) not in self.matrix:
            self.skipTest(f"{name} needs {CASES[name].gpus} CUDA devices")
        return self.matrix[(name, backend)]

    def value(self, name: str, backend: str) -> dict:
        record = self.record(name, backend)
        self.assertEqual(record["outcome"], "accepted", f"{name} on {backend}: {outcome_summary(record, 600)}")
        return record["value"]

    def assert_rejected(self, name: str, backend: str, *fragments: str) -> None:
        record = self.record(name, backend)
        self.assertEqual(record["outcome"], "rejected", f"{name} on {backend}: {outcome_summary(record, 600)}")
        for fragment in fragments:
            self.assertIn(fragment, record["message"], f"{name} on {backend}")

    def assert_sub_rejected(self, probe: dict, label: str, *fragments: str) -> None:
        self.assertIn("error", probe, f"{label} was accepted: {probe}")
        for fragment in fragments:
            self.assertIn(fragment, probe["error"], label)

    def ok(self, probe: dict, label: str):
        self.assertIn("ok", probe, f"{label} was rejected: {probe}")
        return probe["ok"]

    # Every case: no crash, no hang, and an ordinary query still answers correctly afterwards.

    def test_no_case_crashes_hangs_or_poisons_the_cuda_context(self):
        for (name, backend), record in sorted(self.matrix.items()):
            with self.subTest(case=name, backend=backend):
                self.assertIn(record["outcome"], ("accepted", "rejected"), outcome_summary(record, 2000))
                self.assertEqual(record.get("healthy"), 1.0, f"context unusable after {name}")

    # Geometry validation at build.

    def test_out_of_range_face_indices_are_rejected_by_both(self):
        for name, index in (
            ("face_index_past_end", "4"),
            ("face_index_huge", "2000000000"),
            ("face_index_negative", "-1"),
            ("face_index_second_mesh", "9"),
        ):
            for backend in BACKENDS:
                with self.subTest(case=name, backend=backend):
                    self.assert_rejected(name, backend, index, "outside the valid range [0, 4)")
        self.assert_rejected("face_index_second_mesh", "torch", "mesh 1")

    def test_zero_vertex_meshes_are_rejected_by_both(self):
        for name in ("zero_vertices", "zero_vertices_with_faces"):
            for backend in BACKENDS:
                with self.subTest(case=name, backend=backend):
                    self.assert_rejected(name, backend)
        self.assert_rejected("zero_vertices", "torch", "has no vertices")
        self.assert_rejected("zero_vertices", "drjit", "has no vertices")

    def test_zero_face_mesh_is_an_empty_mesh_in_torch_and_rejected_by_drjit(self):
        # Pinned divergence: Torch builds a face-less mesh that nothing can hit; Dr.Jit's Mesh::build refuses it.
        self.assertEqual(self.value("zero_faces", "torch")["hit"], {"t": [INF], "prim": [-1]})
        self.assertEqual(
            self.value("zero_faces_beside_mesh", "torch"),
            {"hit": {"t": [1.0], "prim": [0]}, "edge": {"distance": [0.5], "edge": [0]}},
        )
        for name in ("zero_faces", "zero_faces_beside_mesh"):
            with self.subTest(case=name):
                self.assert_rejected(name, "drjit", "mesh has no faces")

    # Rule: a NaN or +-Inf vertex coordinate rejects Scene.build() on every trace backend, with an error naming the
    # mesh; one non-finite vertex once made a whole OptiX GAS unhittable. Torch also rejects such vertices when
    # update_mesh_vertices() is committed by sync() (tests.scene.test_non_finite_vertices).
    def _assert_non_finite_vertices_rejected(self, backend: str, *fragments: str) -> None:
        for name in ("nan_vertex", "inf_vertex"):
            for trace_backend, probe in self.value(name, backend).items():
                with self.subTest(case=name, trace_backend=trace_backend):
                    self.assert_sub_rejected(probe, f"{name} on {backend} {trace_backend}", *fragments)

    def test_torch_rejects_non_finite_vertices_at_build(self):
        self._assert_non_finite_vertices_rejected(
            "torch", "mesh 0 has a NaN or infinite world-space coordinate at vertex 3"
        )

    def test_drjit_rejects_non_finite_vertices_at_build(self):
        self._assert_non_finite_vertices_rejected(
            "drjit", "mesh 0 has a NaN or infinite world-space coordinate at vertex 3"
        )

    # Non-finite and degenerate queries.

    def test_non_finite_rays_miss_on_both(self):
        # Rows: NaN origin, -Inf origin, NaN direction, Inf direction, zero direction, and one ordinary hit.
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                value = self.value("non_finite_ray", backend)
                hit = self.ok(value["hit"], f"{backend} intersect")
                self.assertEqual(hit, {"t": [INF] * 5 + [1.0], "prim": [-1] * 5 + [0]})
                self.assertEqual(self.ok(value["bounces"], f"{backend} reflections"), [0] * 5 + [1])

    def test_non_finite_and_non_positive_tmax_intersect_identically(self):
        # tmax rows: NaN, Inf, -1, 0, 2. NaN and non-positive tmax are misses, never hits.
        expected = {"t": [INF, 1.0, INF, INF, 1.0], "prim": [-1, 0, -1, -1, 0]}
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.assertEqual(self.ok(self.value("non_finite_tmax", backend)["hit"], backend), expected)

    def test_non_finite_points_have_no_nearest_edge_on_both(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                value = self.value("non_finite_points", backend)
                self.assertEqual(self.ok(value["edge"], backend), {"distance": [INF, INF, 0.5], "edge": [-1, -1, 0]})
                topk = self.ok(value["topk"], backend)
                self.assertEqual(topk["edges"], [-1, -1, -1, -1, 0, 1])
                self.assertEqual(topk["distances"][:4], [INF] * 4)
                edge_ray = self.ok(value["edge_ray"], backend)
                self.assertEqual(edge_ray["edge"], [-1, -1])
                self.assertEqual(edge_ray["distance"], [INF, INF])

    def test_non_finite_visibility_endpoints_are_not_visible_on_both(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.assertEqual(self.value("non_finite_visibility", backend)["visible"], [False, False, True, False])

    def test_zero_length_direction_is_a_miss_on_both(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                value = self.value("zero_length_direction", backend)
                self.assertEqual(self.ok(value["hit"], backend), {"t": [INF], "prim": [-1]})
                self.assertEqual(self.ok(value["bounces"], backend), [0])
                # A point segment (start == end) is unobstructed on both backends.
                self.assertEqual(self.ok(value["visible_point_segment"], backend), [True])

    def _edge_ray(self, backend: str) -> dict:
        return self.value("edge_ray_degenerate_segments", backend)

    def test_nearest_edge_ray_regular_segments_agree_everywhere(self):
        expected = {1: (0.5, 0, 1.0), 5: (0.5**0.5, 0, 0.5), 6: (0.5, 0, 1.0)}
        for backend in BACKENDS:
            for edge_backend, result in self._edge_ray(backend).items():
                for row, (distance, edge, ray_t) in expected.items():
                    with self.subTest(backend=backend, edge_backend=edge_backend, row=row):
                        self.assertAlmostEqual(result["distance"][row], distance, places=6)
                        self.assertEqual(result["edge"][row], edge)
                        self.assertAlmostEqual(result["ray_t"][row], ray_t, places=6)

    def test_drjit_nearest_edge_ray_degenerate_segments(self):
        # contracts/operations.json (nearest_edge_ray.degenerate_rows): NaN, -Inf, zero and negative tmax and a
        # zero-length direction are empty segments with no nearest edge; +Inf is the half-ray. Both Dr.Jit edge
        # backends follow it.
        half_ray, empty = (0.5, 0), (INF, -1)
        expected = {0: empty, 1: half_ray, 2: empty, 3: empty, 4: empty, 7: empty}
        for edge_backend, result in self._edge_ray("drjit").items():
            for row, (distance, edge) in expected.items():
                with self.subTest(edge_backend=edge_backend, row=row):
                    self.assertEqual((result["distance"][row], result["edge"][row]), (distance, edge))

    # Regression: Torch nearest_edge(ray) answers once depended on edge_bvh_backend for degenerate segments. Both
    # edge backends now follow the rule in contracts/operations.json (nearest_edge_ray.degenerate_rows): NaN, -Inf,
    # zero and negative tmax and a zero direction find no edge.
    def test_torch_nearest_edge_ray_degenerate_segments_do_not_depend_on_edge_backend(self):
        torch_rows = self._edge_ray("torch")
        for name, result in torch_rows.items():
            for row in EDGE_RAY_DEGENERATE_ROWS:
                with self.subTest(edge_backend=name, row=row):
                    self.assertEqual((result["distance"][row], result["edge"][row]), (INF, -1))
        results = self._edge_ray("torch")
        for row in EDGE_RAY_DEGENERATE_ROWS:
            with self.subTest(row=row):
                answers = {name: (r["distance"][row], r["edge"][row]) for name, r in results.items()}
                self.assertEqual(len(set(answers.values())), 1, answers)

    # Both backends follow contracts/operations.json (nearest_edge_ray.degenerate_rows) with their default edge
    # backends.
    def test_nearest_edge_ray_degenerate_segments_agree_across_backends(self):
        torch_default, drjit_default = self._edge_ray("torch")["auto"], self._edge_ray("drjit")["auto"]
        for row in EDGE_RAY_DEGENERATE_ROWS:
            with self.subTest(row=row):
                self.assertEqual(
                    (torch_default["distance"][row], torch_default["edge"][row]),
                    (drjit_default["distance"][row], drjit_default["edge"][row]),
                )

    def test_empty_batches(self):
        empty_answers = {
            "edge_point": {"distance": [], "edge": []},
            "edge_ray": {"distance": [], "edge": [], "ray_t": []},
            "topk": {"k": 2, "distances": [], "edges": []},
            "visible": [],
            "reflections": [],
        }
        for backend in BACKENDS:
            value = self.value("empty_batches", backend)
            for operation, expected in empty_answers.items():
                with self.subTest(backend=backend, operation=operation):
                    self.assertEqual(self.ok(value[operation], f"{backend} {operation}"), expected)
        # Pinned divergence: Torch answers an empty intersect batch with empty results, Dr.Jit rejects it.
        self.assertEqual(self.ok(self.value("empty_batches", "torch")["intersect"], "torch"), {"t": [], "prim": []})
        self.assert_sub_rejected(self.value("empty_batches", "drjit")["intersect"], "drjit", "empty ray batch")

    # Lifecycle.

    def test_queries_before_build_are_rejected_by_both(self):
        for backend in BACKENDS:
            for operation, probe in self.value("query_before_build", backend).items():
                with self.subTest(backend=backend, operation=operation):
                    self.assert_sub_rejected(probe, f"{backend} {operation}")
                    self.assertRegex(probe["error"], "build|built")

    def test_queries_with_pending_updates_are_rejected_by_both(self):
        for backend in BACKENDS:
            value = self.value("query_with_pending_updates", backend)
            self.assertTrue(value["pending"])
            for operation in ("intersect", "edge_point", "topk", "visible", "reflections"):
                with self.subTest(backend=backend, operation=operation):
                    self.assert_sub_rejected(value[operation], f"{backend} {operation}", "pending updates")
            with self.subTest(backend=backend, operation="after_sync"):
                self.assertEqual(value["after_sync"], {"t": [1.5], "prim": [0]})

    # Topology stress.

    def test_duplicate_and_degenerate_faces_build_and_answer_identically(self):
        for name, prims in (("duplicate_faces_x100", range(100)), ("degenerate_faces_x20000", [20000])):
            for backend in BACKENDS:
                with self.subTest(case=name, backend=backend):
                    value = self.value(name, backend)
                    self.assertEqual(value["hit"]["t"], [1.0])
                    self.assertIn(value["hit"]["prim"][0], prims)
                    self.assertEqual(value["edge"], [0.5])

    def test_edge_incidence_cap_is_256_faces_on_both(self):
        torch_value = self.value("edge_shared_by_256_faces", "torch")
        self.assertEqual(torch_value, self.value("edge_shared_by_256_faces", "drjit"))
        self.assertEqual(torch_value["edge"], {"distance": [0.0], "edge": [0]})
        for name, count in (("duplicate_faces_x5000", 5000), ("edge_shared_by_257_faces", 257)):
            for backend in BACKENDS:
                with self.subTest(case=name, backend=backend):
                    self.assert_rejected(name, backend, f"shared by {count} non-degenerate faces", "at most 256")

    # Argument ranges.

    def test_nearest_edges_k_range(self):
        for backend in BACKENDS:
            value = self.value("nearest_edges_k_range", backend)
            for k in ("0", "-1", "17", str((1 << 31) - 1), str(1 << 31), str(1 << 40)):
                with self.subTest(backend=backend, k=k):
                    self.assert_sub_rejected(value[k], f"{backend} k={k}")
            with self.subTest(backend=backend, k="3"):
                # Query (0.4, -0.5, 0) has no distance ties, so the order is fixed even where Dr.Jit has no
                # tie-break rule.
                result = self.ok(value["3"], backend)
                self.assertEqual(result["edges"], [0, 1, 2])
                for got, want in zip(result["distances"], (0.5, 0.41**0.5, 1.1 / 2**0.5)):
                    self.assertAlmostEqual(got, want, places=6)
            for k in (10, 16):
                with self.subTest(backend=backend, k=k):
                    # k beyond the edge count pads with invalid ids and infinite distances.
                    result = self.ok(value[str(k)], backend)
                    self.assertEqual(result["k"], k)
                    self.assertEqual(result["edges"], [0, 1, 2] + [-1] * (k - 3))
                    self.assertEqual(result["distances"][3:], [INF] * (k - 3))

    def test_max_bounces_range(self):
        for backend in BACKENDS:
            value = self.value("max_bounces_range", backend)
            for bounces in ("0", "-1", str(1 << 31), str(1 << 40)):
                with self.subTest(backend=backend, bounces=bounces):
                    self.assert_sub_rejected(value[bounces], f"{backend} max_bounces={bounces}")
        # Pinned divergence: Torch traces 70000 bounces in one launch; Dr.Jit's default symbolic chain keeps one
        # record per bounce and caps it at 65536.
        self.assertEqual(self.ok(self.value("max_bounces_range", "torch")["70000"], "torch"), [1])
        self.assert_sub_rejected(self.value("max_bounces_range", "drjit")["70000"], "drjit", "at most 65536 bounces")

    def test_update_with_invalid_mesh_id_is_rejected_by_both(self):
        for backend in BACKENDS:
            value = self.value("update_bad_mesh_id", backend)
            for mesh_id in ("99", "-1", "1"):
                with self.subTest(backend=backend, mesh_id=mesh_id):
                    self.assert_sub_rejected(value[mesh_id], f"{backend} mesh_id={mesh_id}")
            with self.subTest(backend=backend):
                self.assertFalse(value["pending"])
                self.assertEqual(value["intact"], {"t": [1.0], "prim": [0]})

    def test_update_with_wrong_vertex_count_is_rejected_by_both(self):
        for backend in BACKENDS:
            value = self.value("update_wrong_vertex_count", backend)
            for name in ("more", "fewer", "none"):
                with self.subTest(backend=backend, update=name):
                    self.assert_sub_rejected(value[name], f"{backend} {name}", "vertex count")
                    self.assertEqual(self.ok(value[name + "_intact"], backend), {"t": [1.0], "prim": [0]})

    def test_edge_mask_of_wrong_length_is_rejected_by_both(self):
        for backend in BACKENDS:
            value = self.value("edge_mask_length", backend)
            for name in ("short", "long", "empty"):
                with self.subTest(backend=backend, mask=name):
                    self.assert_sub_rejected(value[name], f"{backend} {name}")
            with self.subTest(backend=backend):
                self.assertFalse(value["pending_after_bad_masks"])
                self.assertEqual(value["edge_after_bad_masks"], {"distance": [0.5], "edge": [0]})
                self.assertEqual(value["edge_with_valid_mask"]["edge"], [1])
                self.assertAlmostEqual(value["edge_with_valid_mask"]["distance"][0], 0.41**0.5, places=6)
        # Pinned divergence: Torch applies a valid edge mask immediately; Dr.Jit queues it until sync().
        self.assertFalse(self.value("edge_mask_length", "torch")["pending_after_valid_mask"])
        self.assertTrue(self.value("edge_mask_length", "drjit")["pending_after_valid_mask"])

    # Derivatives after the scene changed or disappeared: refuse, or return the derivative of the geometry the
    # forward pass saw. A derivative of the current geometry would be a silent wrong answer.

    def test_backward_after_geometry_changing_sync(self):
        self.assert_rejected("backward_after_sync", "torch", "scene geometry changed")
        self.assertEqual(self.value("backward_after_sync", "drjit")["grad_z"], QUERY_TIME_GRAD_Z)

    def test_jvp_after_geometry_changing_sync(self):
        for backend in BACKENDS:
            value = self.value("jvp_after_sync", backend)
            for field in ("t", "p_z"):
                with self.subTest(backend=backend, field=field):
                    if "error" in value[field]:
                        self.assertIn("scene geometry changed", value[field]["error"])
                    else:
                        self.assertEqual(value[field]["ok"], [QUERY_TIME_GRAD_Z[0]])

    def test_backward_after_scene_deletion(self):
        self.assert_rejected("scene_deleted_before_backward", "torch", "Invalid RayD Torch scene handle")
        self.assertEqual(self.value("scene_deleted_before_backward", "drjit")["grad_z"], QUERY_TIME_GRAD_Z)

    def test_inputs_on_another_device_are_rejected_by_both(self):
        for backend in BACKENDS:
            value = self.value("cross_device_inputs", backend)
            for operation, probe in value.items():
                with self.subTest(backend=backend, operation=operation):
                    self.assert_sub_rejected(probe, f"{backend} {operation}", "device")


if __name__ == "__main__":
    import json
    import sys

    matrix = run_matrix(sys.argv[1:] or None)
    for (name, backend), record in sorted(matrix.items()):
        print(f"{name:32s} {backend:6s} {outcome_summary(record, 400)} | healthy={record.get('healthy')}")
    if os.environ.get("RAYD_VALIDATION_JSON"):
        with open(os.environ["RAYD_VALIDATION_JSON"], "w") as stream:
            json.dump({f"{n}|{b}": r for (n, b), r in matrix.items()}, stream, indent=1, default=str)
