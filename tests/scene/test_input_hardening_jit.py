# Copyright Xingyu Chen.
# Tests Dr.Jit backend rejection of oversized, malformed, and cross-device inputs.

"""Regression tests for Dr.Jit input hardening.

Every case runs in a fresh subprocess: the failures these tests guard against were
out-of-bounds device accesses that poison the CUDA context, unbounded host memory,
and hangs, so each case also proves that the process can still answer an ordinary
query after the rejected call.
"""

import json
import unittest

from tests.support.subprocess_cases import compose, run_json_case, run_script


def drjit_device_count() -> int:
    """Dr.Jit CUDA device count, or 0 when rayd.drjit cannot be imported."""
    result = run_script(
        """
        import json
        try:
            import rayd.drjit as pj
            devices = int(pj.device_count())
        except Exception:
            devices = 0
        print(json.dumps({"devices": devices}))
        """,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        return 0
    try:
        return int(json.loads(lines[-1])["devices"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0


DEVICE_COUNT = drjit_device_count()

PRELUDE = """
import json
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as pj


def triangle_scene(trace_backend="optix", edge_bvh_backend="optix", faces=None, vertices=None):
    vertices = cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]) if vertices is None else vertices
    faces = cuda.Array3i([0], [1], [2]) if faces is None else faces
    scene = pj.Scene(edge_bvh_backend=edge_bvh_backend, trace_backend=trace_backend)
    scene.add_mesh(pj.Mesh(vertices, faces))
    scene.build()
    return scene


def probe_ray():
    return pj.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))


def healthy():
    return float(triangle_scene().intersect(probe_ray()).t[0])


def error_of(fn):
    try:
        fn()
    except Exception as error:
        return str(error)
    return None
"""


def run_case(body: str, timeout: int = 300):
    return run_json_case(compose(PRELUDE, body), timeout=timeout)


@unittest.skipUnless(DEVICE_COUNT >= 1, "a Dr.Jit CUDA device and an importable rayd.drjit are required")
class OversizedBatchTests(unittest.TestCase):
    def test_cuda_trace_stack_beyond_array_limit_is_rejected(self):
        # 2^26 rays need 2^32 traversal-stack slots, one more than a Dr.Jit array holds; the old int product
        # wrapped to an empty buffer that the kernel then wrote past.
        data = run_case(
            """
            scene = triangle_scene(trace_backend="cuda", edge_bvh_backend="drjit")
            n = 1 << 26
            ray = pj.Ray(cuda.Array3f(dr.full(cuda.Float, 0.25, n), dr.full(cuda.Float, 0.25, n),
                                      dr.full(cuda.Float, -1.0, n)),
                         cuda.Array3f(dr.zeros(cuda.Float, n), dr.zeros(cuda.Float, n), dr.full(cuda.Float, 1.0, n)))
            print(json.dumps({"error": error_of(lambda: scene.intersect(ray)),
                              "shadow": error_of(lambda: scene.shadow_test(ray)), "healthy": healthy()}))
            """
        )
        self.assertIn("traversal stack slots", data["error"])
        self.assertIn("traversal stack slots", data["shadow"])
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)

    def test_edge_bvh_stack_beyond_index_range_is_rejected(self):
        data = run_case(
            """
            scene = triangle_scene(edge_bvh_backend="drjit")
            n = (1 << 25) + 1
            points = cuda.Array3f(dr.full(cuda.Float, 0.1, n), dr.full(cuda.Float, 0.1, n), dr.full(cuda.Float, 1.0, n))
            print(json.dumps({"error": error_of(lambda: scene.nearest_edge(points)), "healthy": healthy()}))
            """
        )
        self.assertIn("32-bit index range", data["error"])
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)

    def test_reflection_slot_overflow_is_rejected(self):
        data = run_case(
            """
            scene = triangle_scene()
            n = 4096
            ray = pj.Ray(cuda.Array3f(dr.full(cuda.Float, 0.25, n), dr.full(cuda.Float, 0.25, n),
                                      dr.full(cuda.Float, -1.0, n)),
                         cuda.Array3f(dr.zeros(cuda.Float, n), dr.zeros(cuda.Float, n), dr.full(cuda.Float, 1.0, n)))
            print(json.dumps({
                "native": error_of(lambda: scene.trace_reflections(ray, 600000, symbolic=False)),
                "symbolic": error_of(lambda: scene.trace_reflections(ray, 600000)),
                "healthy": healthy(),
            }))
            """
        )
        self.assertIn("4096 x 600000 elements exceed the 32-bit index range", data["native"])
        self.assertIn("at most 65536 bounces", data["symbolic"])
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)

    def test_accumulation_grid_overflow_is_rejected(self):
        data = run_case(
            """
            scene = triangle_scene()
            grid = pj.AccumGrid()
            grid.axis = 2
            grid.resolution0 = 70000
            grid.resolution1 = 70000
            grid.coord0_min, grid.coord0_max, grid.coord1_min, grid.coord1_max = 0.0, 1.0, 0.0, 1.0
            error = error_of(lambda: scene.accumulate_reflections(probe_ray(), cuda.Array3f([0.0], [0.0], [-1.0]),
                                                                   grid, pj.Material(), 2))
            print(json.dumps({"error": error, "healthy": healthy()}))
            """
        )
        self.assertIn("70000 x 70000 elements exceed the 32-bit index range", data["error"])
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)


@unittest.skipUnless(DEVICE_COUNT >= 1, "a Dr.Jit CUDA device and an importable rayd.drjit are required")
class SymbolicReflectionChainTests(unittest.TestCase):
    def test_long_symbolic_chain_matches_native_trace(self):
        # Rays bounce inside a closed cube, so every one of the 20 bounces hits; the symbolic per-bounce path
        # evaluates its carried state every few bounces and must agree with the single native launch.
        data = run_case(
            """
            corners = [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
            quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
            faces = [f for a, b, c, d in quads for f in ((a, b, c), (a, c, d))]
            vertices = cuda.Array3f(*[[c[axis] for c in corners] for axis in range(3)])
            scene = triangle_scene(faces=cuda.Array3i(*[[f[k] for f in faces] for k in range(3)]), vertices=vertices)
            ray = pj.Ray(cuda.Array3f([0.3, 0.6, 0.5], [0.4, 0.2, 0.5], [0.5, 0.7, 0.3]),
                         dr.normalize(cuda.Array3f([0.31, -0.7, 0.2], [0.53, 0.33, -0.81], [0.79, 0.21, 0.45])))
            bounces = 20
            symbolic = scene.trace_reflections(ray, bounces)
            native = scene.trace_reflections(ray, bounces, symbolic=False)
            sym_t = [[float(b.t[i]) for b in symbolic.bounces] for i in range(3)]
            nat_t = [[float(native.t[i * bounces + k]) for k in range(bounces)] for i in range(3)]
            print(json.dumps({"count": len(symbolic.bounces), "sym_hits": [int(v) for v in symbolic.bounce_count],
                              "nat_hits": [int(v) for v in native.bounce_count], "sym_t": sym_t, "nat_t": nat_t}))
            """
        )
        self.assertEqual(data["count"], 20)
        self.assertEqual(data["sym_hits"], [20, 20, 20])
        self.assertEqual(data["nat_hits"], [20, 20, 20])
        for sym_row, nat_row in zip(data["sym_t"], data["nat_t"]):
            for sym, nat in zip(sym_row, nat_row):
                self.assertAlmostEqual(sym, nat, delta=1e-3 * max(1.0, abs(nat)))

    def test_long_symbolic_chain_finishes_and_pads_misses(self):
        # Before the periodic evaluation, 64 bounces fused into one kernel took 18 s to compile and 256 did not
        # finish within 300 s.
        data = run_case(
            """
            scene = triangle_scene()
            chain = scene.trace_reflections(pj.Ray(cuda.Array3f([0.25, 0.3], [0.25, 0.3], [-1.0, -1.0]),
                                                   cuda.Array3f([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])), 1000)
            print(json.dumps({"count": len(chain.bounces), "hits": [int(v) for v in chain.bounce_count],
                              "first_t": [float(v) for v in chain.bounces[0].t],
                              "last_t": [float(v) for v in chain.bounces[-1].t],
                              "last_prim": [int(v) for v in chain.bounces[-1].prim_ids]}))
            """,
            timeout=120,
        )
        self.assertEqual(data["count"], 1000)
        self.assertEqual(data["hits"], [1, 1])
        self.assertEqual(data["first_t"], [1.0, 1.0])
        self.assertEqual(data["last_t"], [float("inf"), float("inf")])
        self.assertEqual(data["last_prim"], [-1, -1])


@unittest.skipUnless(DEVICE_COUNT >= 1, "a Dr.Jit CUDA device and an importable rayd.drjit are required")
class MeshTopologyTests(unittest.TestCase):
    def test_degenerate_faces_create_no_edges(self):
        # 20000 [0, 0, 0] padding faces used to put 60000 incident faces on edge (0, 0): about 1.8e9 wedge
        # records, an int overflow, and more than 80 GB of host memory before the 300 s timeout.
        data = run_case(
            """
            pad = 20000
            faces = cuda.Array3i([0] * pad + [0, 1], [0] * pad + [1, 1], [0] * pad + [2, 2])
            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]), faces)
            edges = [list(map(int, column)) for column in mesh.edge_indices()]
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            hit = scene.intersect(probe_ray())
            print(json.dumps({"edges": edges, "prim": int(hit.prim_id[0]),
                              "distance": float(scene.nearest_edge(cuda.Array3f([0.5], [-1.0], [0.0])).distance[0])}))
            """
        )
        # Only face `pad` = [0, 1, 2] contributes edges; [1, 1, 2] repeats a vertex.
        self.assertEqual(data["edges"], [[0, 0, 1], [1, 2, 2], [20000] * 3, [-1] * 3, [2, 1, 0]])
        self.assertEqual(data["prim"], 20000)
        self.assertAlmostEqual(data["distance"], 1.0, places=5)

    def test_edge_shared_by_too_many_faces_is_reported_and_edges_can_be_disabled(self):
        data = run_case(
            """
            k = 5000
            vertices = cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
            faces = cuda.Array3i([0] * k, [1] * k, [2] * k)
            with_edges = error_of(lambda: triangle_scene(faces=faces, vertices=vertices))
            mesh = pj.Mesh(vertices, faces)
            mesh.edges_enabled = False
            scene = pj.Scene()
            scene.add_mesh(mesh)
            scene.build()
            print(json.dumps({"error": with_edges, "t": float(scene.intersect(probe_ray()).t[0])}))
            """,
            timeout=120,
        )
        self.assertIn(
            "is shared by 5000 non-degenerate faces; edge topology supports at most 256 faces per edge", data["error"]
        )
        self.assertIn("duplicated or coincident faces", data["error"])
        self.assertAlmostEqual(data["t"], 1.0, places=5)

    def test_edge_incidence_cap_matches_torch_backend(self):
        # The Torch twin (src/scene/cache.cu) accepts at most 256 non-degenerate faces per edge, so the same mesh
        # must be accepted or rejected by both backends.
        data = run_case(
            """
            vertices = cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])

            def stacked(k):
                return cuda.Array3i([0] * k, [1] * k, [2] * k)

            at_cap = triangle_scene(faces=stacked(256), vertices=vertices)
            point = cuda.Array3f([0.5], [-1.0], [0.0])
            print(json.dumps({"at_cap": float(at_cap.nearest_edge(point).distance[0]),
                              "over_cap": error_of(lambda: triangle_scene(faces=stacked(257), vertices=vertices))}))
            """,
            timeout=120,
        )
        self.assertAlmostEqual(data["at_cap"], 1.0, places=5)
        self.assertIn("is shared by 257 non-degenerate faces; edge topology supports at most 256", data["over_cap"])

    def test_manifold_and_non_manifold_edges_keep_their_records(self):
        data = run_case(
            """
            import random
            n = 6
            vertices = cuda.Array3f([float(i % n) for i in range(n * n)], [float(i // n) for i in range(n * n)],
                                    [0.0] * (n * n))
            faces = []
            for i in range(n - 1):
                for j in range(n - 1):
                    a = i * n + j
                    faces += [(a, a + 1, a + n), (a + 1, a + n + 1, a + n)]
            # A fin of three extra faces on edge (0, 1) makes it non-manifold with four incident faces.
            vertices = cuda.Array3f(list(vertices.x) + [0.5, 0.5, 0.5], list(vertices.y) + [0.0, 0.0, -1.0],
                                    list(vertices.z) + [1.0, -1.0, 0.0])
            faces += [(0, 1, n * n), (1, 0, n * n + 1), (0, 1, n * n + 2)]
            mesh = pj.Mesh(vertices, cuda.Array3i(*[[f[k] for f in faces] for k in range(3)]))
            edges = [list(map(int, column)) for column in mesh.edge_indices()]

            # All-pairs reference: one boundary record per single-face edge, one wedge per face pair otherwise.
            incident = {}
            for face_index, face in enumerate(faces):
                for local in range(3):
                    a, b, opposite = face[local], face[(local + 1) % 3], face[(local + 2) % 3]
                    incident.setdefault((min(a, b), max(a, b)), []).append((face_index, opposite))
            reference = [[], [], [], [], []]
            for (a, b), entries in sorted(incident.items()):
                pairs = [(entries[0], None)] if len(entries) == 1 else [
                    (entries[p], entries[q]) for p in range(len(entries)) for q in range(p + 1, len(entries))]
                for first, second in pairs:
                    for column, value in zip(reference, (a, b, first[0], -1 if second is None else second[0],
                                                         first[1])):
                        column.append(value)
            print(json.dumps({"edges": edges, "reference": reference}))
            """
        )
        self.assertEqual(data["edges"], data["reference"])
        wedges_on_fin_edge = sum(1 for a, b in zip(data["edges"][0], data["edges"][1]) if (a, b) == (0, 1))
        self.assertEqual(wedges_on_fin_edge, 6)

    def test_set_face_indices_rebuilds_topology_at_build(self):
        data = run_case(
            """
            mesh = pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0] * 4),
                           cuda.Array3i([0], [1], [2]))
            before = len(mesh.edge_indices()[0])
            mesh.face_indices = dr.cuda.ad.Array3i(cuda.Array3i([0, 1], [1, 3], [2, 2]))
            stale = error_of(lambda: mesh.edge_indices())
            mesh.build()
            print(json.dumps({"before": before, "stale": stale, "after": len(mesh.edge_indices()[0])}))
            """
        )
        self.assertEqual(data["before"], 3)
        self.assertIn("call build() first", data["stale"])
        self.assertEqual(data["after"], 5)


@unittest.skipUnless(DEVICE_COUNT >= 1, "a Dr.Jit CUDA device and an importable rayd.drjit are required")
class FaceIndexValidationTests(unittest.TestCase):
    def test_out_of_range_face_indices_are_rejected(self):
        # Dr.Jit bounds-checks a gather only in debug mode or for a literal index (one face). With several faces a
        # negative or huge index faulted the device (a sticky CUDA error) and one just past the end read zeros.
        data = run_case(
            """
            vertices = cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
            errors = {}
            for name, face in (("past_end", [0, 1, 3]), ("negative", [0, 1, -1]), ("huge", [0, 1, 2000000000])):
                faces = cuda.Array3i(*[[0, v] for v in face])
                errors[name] = error_of(lambda: triangle_scene(faces=faces, vertices=vertices))
            mesh = pj.Mesh(vertices, cuda.Array3i([0], [1], [2]))
            mesh.vertex_positions = dr.cuda.ad.Array3f(cuda.Array3f([0.0, 1.0], [0.0, 0.0], [0.0, 0.0]))
            scene = pj.Scene()
            scene.add_mesh(mesh)
            errors["shrunk_vertices"] = error_of(scene.build)
            errors["healthy"] = healthy()
            print(json.dumps(errors))
            """
        )
        self.assertIn("face 1 references vertex 3, outside the valid range [0, 3)", data["past_end"])
        self.assertIn("face 1 references vertex -1, outside the valid range [0, 3)", data["negative"])
        self.assertIn("references vertex 2000000000", data["huge"])
        self.assertIn("references index 2, outside the valid range [0, 2)", data["shrunk_vertices"])
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)

    def test_non_finite_vertices_are_rejected_at_build_and_update(self):
        # OptiX drops every triangle of a GAS that holds an infinite vertex, so finite faces stopped being hit.
        # Non-finite vertices are rejected instead, on every trace backend, at build and at sync().
        data = run_case(
            """
            nan, inf = float("nan"), float("inf")
            errors = {}
            for backend in ("optix", "cuda"):
                for name, bad in (("nan", nan), ("inf", inf), ("-inf", -inf)):
                    vertices = cuda.Array3f([0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, bad])
                    faces = cuda.Array3i([0, 1], [1, 3], [2, 2])
                    errors[f"build_{backend}_{name}"] = error_of(
                        lambda: triangle_scene(trace_backend=backend, faces=faces, vertices=vertices))
                    scene = pj.Scene(trace_backend=backend)
                    scene.add_mesh(pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
                                           cuda.Array3i([0], [1], [2])))
                    scene.add_mesh(pj.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [5.0, 5.0, 5.0]),
                                           cuda.Array3i([0], [1], [2])), dynamic=True)
                    scene.build()
                    scene.update_mesh_vertices(1, dr.cuda.ad.Array3f([0.0, 1.0, bad], [0.0, 0.0, 1.0], [5.0] * 3))
                    errors[f"sync_{backend}_{name}"] = error_of(scene.sync)
                    errors[f"pending_{backend}_{name}"] = bool(scene.has_pending_updates())
                    scene.update_mesh_vertices(1, dr.cuda.ad.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [5.0] * 3))
                    scene.sync()
                    errors[f"recovered_{backend}_{name}"] = float(scene.intersect(probe_ray()).t[0])
            errors["healthy"] = healthy()
            print(json.dumps(errors))
            """
        )
        for backend in ("optix", "cuda"):
            for name in ("nan", "inf", "-inf"):
                with self.subTest(backend=backend, value=name):
                    self.assertIn(
                        "mesh 0 has a NaN or infinite world-space coordinate at vertex 3",
                        data[f"build_{backend}_{name}"],
                    )
                    self.assertIn(
                        "Scene::sync(): mesh 1 has a NaN or infinite world-space coordinate at vertex 2",
                        data[f"sync_{backend}_{name}"],
                    )
                    self.assertTrue(data[f"pending_{backend}_{name}"])
                    self.assertAlmostEqual(data[f"recovered_{backend}_{name}"], 1.0, places=5)
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)


@unittest.skipUnless(DEVICE_COUNT >= 1, "a Dr.Jit CUDA device and an importable rayd.drjit are required")
class EmptyQueryTests(unittest.TestCase):
    def test_empty_queries_on_both_trace_backends(self):
        data = run_case(
            """
            out = {}
            for trace_backend in ("optix", "cuda"):
                scene = triangle_scene(trace_backend=trace_backend, edge_bvh_backend="drjit")
                empty = cuda.Array3f([], [], [])
                ray = pj.Ray(empty, empty)
                out[trace_backend] = {
                    "intersect": error_of(lambda: scene.intersect(ray)),
                    "shadow": len(scene.shadow_test(ray)),
                    "edge_point": len(scene.nearest_edge(empty).distance),
                    "edge_ray": len(scene.nearest_edge(ray).distance),
                }
            out["healthy"] = healthy()
            print(json.dumps(out))
            """
        )
        for trace_backend in ("optix", "cuda"):
            self.assertIn("empty ray batch", data[trace_backend]["intersect"])
            self.assertEqual(data[trace_backend]["shadow"], 0)
            self.assertEqual(data[trace_backend]["edge_point"], 0)
            self.assertEqual(data[trace_backend]["edge_ray"], 0)
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)


@unittest.skipUnless(DEVICE_COUNT >= 2, "two Dr.Jit CUDA devices and an importable rayd.drjit are required")
class CrossDeviceInputTests(unittest.TestCase):
    def test_inputs_evaluated_on_another_device_are_rejected(self):
        # Dr.Jit does not record an array's device: a device-0 kernel reading these device-1 buffers faulted.
        data = run_case(
            """
            out = {}
            for trace_backend in ("optix", "cuda"):
                pj.set_device(0)
                scene = triangle_scene(trace_backend=trace_backend, edge_bvh_backend="drjit")
                pj.set_device(1)
                # Distinct lanes keep these from folding into literals, so eval() gives them device-1 buffers.
                origin = cuda.Array3f([0.25, 0.3], [0.25, 0.3], [-1.0, -1.0])
                direction = cuda.Array3f([0.0, 0.01], [0.0, 0.01], [1.0, 1.0])
                dr.eval(origin, direction)
                pj.set_device(0)
                ray = pj.Ray(origin, direction)
                out[trace_backend] = {
                    "intersect": error_of(lambda: scene.intersect(ray)),
                    "reflections": error_of(lambda: scene.trace_reflections(ray, 2, symbolic=False)),
                    "edge": error_of(lambda: scene.nearest_edge(origin)),
                    "visible": error_of(lambda: scene.visible(origin, direction)),
                }
            out["healthy"] = healthy()
            print(json.dumps(out))
            """
        )
        for trace_backend in ("optix", "cuda"):
            for operation, message in data[trace_backend].items():
                self.assertIsNotNone(message, f"{trace_backend} {operation}")
                self.assertIn("an input array lives on CUDA device 1", message)
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)

    def test_epc_diffraction_and_surfel_inputs_on_another_device_are_rejected(self):
        # Equivalent-path, diffraction (whose inputs are Dr.Jit structs), and surfel entry points read raw device
        # pointers too, so a device-1 array handed to a device-0 scene must be rejected before any launch.
        data = run_case(
            """
            pj.set_device(0)
            scene = triangle_scene(edge_bvh_backend="drjit")
            surfels = pj.SurfelScene(pj.SurfelGeometry(cuda.Array3f([0.0], [0.0], [0.0]),
                                                       cuda.Array3f([1.0], [0.0], [0.0]),
                                                       cuda.Array3f([0.0], [1.0], [0.0])))
            surfels.build()
            pj.set_device(1)
            origin = cuda.Array3f([0.25, 0.3], [0.25, 0.3], [-1.0, -1.0])
            direction = cuda.Array3f([0.0, 0.01], [0.0, 0.01], [1.0, 1.0])
            receiver = cuda.Array3f([0.2, 0.1], [0.2, 0.1], [-2.0, -2.0])
            edge_index = cuda.Int([0, 1])
            dr.eval(origin, direction, receiver, edge_index)
            pj.set_device(0)
            ray = pj.Ray(origin, direction)
            states = pj.DfrStates()
            states.count = 2
            states.edge_index = edge_index
            grid = pj.DfrGrid()
            grid.axis = 2
            grid.position = -1.0
            grid.coord0_min = -1.0
            grid.coord0_max = 1.0
            grid.coord1_min = -1.0
            grid.coord1_max = 1.0
            grid.resolution0 = 1
            grid.resolution1 = 1
            grid.cell_area = 4.0
            material = pj.DfrMaterial()
            options = pj.DfrOptions()
            options.wavelength = 0.125
            options.max_order = 1
            out = {
                "trace_refl_epc": error_of(lambda: scene.trace_refl_epc(ray, receiver, 1)),
                "accum_dfr_direct": error_of(lambda: scene.accum_dfr_direct(states, grid, material, options, True)),
                "surfel_intersect": error_of(lambda: surfels.intersect(ray)),
                "surfel_visible": error_of(lambda: surfels.visible(origin, receiver)),
            }
            out["healthy"] = healthy()
            print(json.dumps(out))
            """
        )
        for operation in ("trace_refl_epc", "accum_dfr_direct", "surfel_intersect", "surfel_visible"):
            self.assertIsNotNone(data[operation], operation)
            self.assertIn("an input array lives on CUDA device 1", data[operation])
        self.assertAlmostEqual(data["healthy"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
