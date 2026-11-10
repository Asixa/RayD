# Copyright Xingyu Chen.
# Tests Torch scene input validation, edge topology bounds and stale-tape refusal.

import subprocess
import sys
import textwrap
import time
import unittest

import torch
import rayd.torch as rt

_PRELUDE = """
import torch
import rayd.torch as rt
dev = "cuda:0"
def T(x, dt=torch.float32, d=None):
    return torch.tensor(x, device=d or dev, dtype=dt)
"""


def _run_isolated(body: str, timeout: float = 300.0) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter, since a regression here poisons the CUDA context."""
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _triangle():
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    return vertices, faces


def _reference_edge_records(faces: list[list[int]]) -> list[tuple[int, int, int, int, int]]:
    """Edge records the topology must emit: one per pair of incident faces, degenerate faces skipped."""
    incident: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
            continue
        for local_edge in range(3):
            a, b = face[local_edge], face[(local_edge + 1) % 3]
            opposite = face[(local_edge + 2) % 3]
            incident.setdefault((min(a, b), max(a, b)), []).append((face_index, opposite))
    records = []
    for key in sorted(incident):
        entries = incident[key]
        if len(entries) == 1:
            records.append((key[0], key[1], entries[0][0], -1, entries[0][1]))
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                records.append((key[0], key[1], entries[i][0], entries[j][0], entries[i][1]))
    return records


def _native_edge_records(scene: rt.Scene) -> list[tuple[int, int, int, int, int]]:
    values = scene._native_scene.edge_records()
    v0, v1, face0, face1, opposite = (values[i].cpu().tolist() for i in (5, 6, 7, 8, 11))
    return list(zip(v0, v1, face0, face1, opposite))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class FaceIndexValidationTests(unittest.TestCase):
    def test_out_of_range_face_indices_are_rejected_at_build(self):
        for index in (3, 7, 2000000000, -1):
            with self.subTest(index=index):
                result = _run_isolated(
                    f"""
                    s = rt.Scene()
                    s.add_mesh(rt.Mesh(T([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), T([[0, 1, {index}]], torch.int32)))
                    try:
                        s.build()
                    except RuntimeError as error:
                        print("REJECTED", error)
                    good = rt.Scene()
                    good.add_mesh(rt.Mesh(T([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), T([[0, 1, 2]], torch.int32)))
                    good.build()
                    print("HIT", float(good.intersect(rt.Ray(T([[.25, .25, -1]]), T([[0., 0, 1]]))).t[0]))
                    """
                )
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                self.assertIn("REJECTED", result.stdout)
                self.assertIn(f"face index {index} outside the valid range [0, 3)", result.stdout)
                self.assertIn("HIT 1.0", result.stdout)

    def test_zero_vertex_mesh_is_rejected_with_a_clear_error(self):
        scene = rt.Scene()
        scene.add_mesh(
            rt.Mesh(torch.empty((0, 3), device="cuda"), torch.empty((0, 3), device="cuda", dtype=torch.int32))
        )
        with self.assertRaisesRegex(RuntimeError, "mesh 0 has no vertices"):
            scene.build()

    def test_second_mesh_is_named_in_the_error(self):
        vertices, faces = _triangle()
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.add_mesh(rt.Mesh(vertices.clone(), torch.tensor([[0, 1, 5]], device="cuda", dtype=torch.int32)))
        with self.assertRaisesRegex(RuntimeError, "mesh 1 has face index 5"):
            scene.build()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class EdgeTopologyBoundsTests(unittest.TestCase):
    def _scene(self, vertices, faces):
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, torch.tensor(faces, device="cuda", dtype=torch.int32)))
        scene.build()
        return scene

    def test_records_match_reference_for_degenerate_and_non_manifold_faces(self):
        vertices = torch.rand((8, 3), device="cuda", generator=None)
        faces = [
            [0, 1, 2],
            [1, 0, 3],  # manifold neighbour across edge (0, 1)
            [0, 1, 4],  # third and fourth faces make (0, 1) a fan
            [5, 1, 0],
            [2, 2, 6],  # degenerate: emits no edges
            [6, 7, 7],  # degenerate
            [3, 6, 7],
            [3, 6, 7],  # duplicate face
            [2, 1, 6],
        ]
        scene = self._scene(vertices, faces)
        self.assertEqual(_native_edge_records(scene), _reference_edge_records(faces))

    def test_only_degenerate_faces_produce_no_edges(self):
        vertices, _ = _triangle()
        scene = self._scene(vertices, [[0, 0, 0]] * 20000 + [[1, 1, 2]])
        self.assertEqual(scene._native_scene.edge_count(), 0)

    def test_moderately_repeated_face_builds_in_bounded_time(self):
        vertices, _ = _triangle()
        start = time.perf_counter()
        scene = self._scene(vertices, [[0, 1, 2]] * 100)
        elapsed = time.perf_counter() - start
        self.assertEqual(scene._native_scene.edge_count(), 3 * (100 * 99 // 2))
        self.assertLess(elapsed, 30.0)

    def test_highly_repeated_face_is_rejected_quickly(self):
        result = _run_isolated(
            """
            import time
            s = rt.Scene()
            s.add_mesh(rt.Mesh(T([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), T([[0, 1, 2]] * 5000, torch.int32)))
            start = time.perf_counter()
            try:
                s.build()
                print("BUILT")
            except RuntimeError as error:
                print("REJECTED", error)
            print("SECONDS", time.perf_counter() - start)
            """,
            timeout=180.0,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("REJECTED", result.stdout)
        self.assertIn("shared by 5000 non-degenerate faces", result.stdout)
        self.assertIn("edges_enabled=False", result.stdout)

    def test_repeated_face_without_edges_still_builds(self):
        vertices, _ = _triangle()
        scene = rt.Scene()
        faces = torch.tensor([[0, 1, 2]] * 5000, device="cuda", dtype=torch.int32)
        scene.add_mesh(rt.Mesh(vertices, faces, edges_enabled=False))
        scene.build()
        self.assertEqual(scene._native_scene.edge_count(), 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class StaleTapeTests(unittest.TestCase):
    def _dynamic_scene(self):
        vertices, faces = _triangle()
        vertices = vertices.requires_grad_()
        scene = rt.Scene()
        mesh_id = scene.add_mesh(rt.Mesh(vertices, faces), dynamic=True)
        scene.build()
        return scene, mesh_id, vertices

    def _ray(self):
        return rt.Ray(torch.tensor([[0.25, 0.25, -1.0]], device="cuda"), torch.tensor([[0.0, 0.0, 1.0]], device="cuda"))

    def _commit_update(self, scene, mesh_id, vertices):
        scene.update_mesh_vertices(mesh_id, vertices.detach() + torch.tensor([0.0, 0.0, 0.5], device="cuda"))
        scene.sync()

    def test_intersect_backward_after_sync_is_refused(self):
        scene, mesh_id, vertices = self._dynamic_scene()
        t = scene.intersect(self._ray()).t
        self._commit_update(scene, mesh_id, vertices)
        with self.assertRaisesRegex(RuntimeError, "scene geometry changed"):
            t.sum().backward()

    def test_lazy_intersection_read_after_sync_is_refused(self):
        scene, mesh_id, vertices = self._dynamic_scene()
        hit = scene.intersect(self._ray())
        self._commit_update(scene, mesh_id, vertices)
        with self.assertRaisesRegex(RuntimeError, "lazily evaluated result"):
            hit.p.sum().backward()

    def test_full_intersection_backward_after_sync_is_refused(self):
        scene, mesh_id, vertices = self._dynamic_scene()
        p = scene.intersect(self._ray()).p
        self._commit_update(scene, mesh_id, vertices)
        with self.assertRaisesRegex(RuntimeError, "scene geometry changed"):
            p.sum().backward()

    def test_nearest_edge_backward_after_sync_is_refused(self):
        scene, mesh_id, vertices = self._dynamic_scene()
        distance = scene.nearest_edge(torch.tensor([[0.3, 0.3, 0.2]], device="cuda")).distance
        self._commit_update(scene, mesh_id, vertices)
        with self.assertRaisesRegex(RuntimeError, "scene geometry changed"):
            distance.sum().backward()

    def test_backward_before_sync_is_unchanged(self):
        scene, mesh_id, vertices = self._dynamic_scene()
        t = scene.intersect(self._ray()).t
        t.sum().backward()
        expected = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.25], [0.0, 0.0, 0.25]], device="cuda")
        torch.testing.assert_close(vertices.grad, expected)

    def test_query_with_pending_updates_is_refused(self):
        scene, mesh_id, vertices = self._dynamic_scene()
        scene.update_mesh_vertices(mesh_id, vertices.detach() + 1.0)
        with self.assertRaisesRegex(RuntimeError, "pending updates. Call Scene.sync"):
            scene.intersect(self._ray())
        scene.sync()
        self.assertFalse(scene.has_pending_updates())
        self.assertTrue(torch.isinf(scene.intersect(self._ray()).t).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class ErrorContractTests(unittest.TestCase):
    def _scene(self, **mesh_kwargs):
        vertices, faces = _triangle()
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces, **mesh_kwargs))
        scene.build()
        return scene

    def test_non_finite_visibility_endpoints_are_not_visible(self):
        scene = self._scene()
        nan, inf = float("nan"), float("inf")
        start = torch.tensor([[nan, 0.0, 0.0], [0.0, 0.0, -1.0], [0.2, 0.2, 1.0], [0.2, 0.2, -1.0]], device="cuda")
        end = torch.tensor([[0.0, 0.0, 1.0], [inf, 0.0, 1.0], [0.2, 0.2, 2.0], [0.2, 0.2, 1.0]], device="cuda")
        self.assertEqual(scene.visible(start, end).tolist(), [False, False, True, False])

    def test_trace_reflections_rejects_non_positive_bounces(self):
        scene = self._scene()
        ray = rt.Ray(torch.zeros((1, 3), device="cuda"), torch.tensor([[0.0, 0.0, 1.0]], device="cuda"))
        for bounces in (0, -1):
            with self.assertRaisesRegex(ValueError, "max_bounces must be at least 1"):
                scene.trace_reflections(ray, bounces)

    def test_update_with_invalid_mesh_id_is_a_clear_error(self):
        scene = self._scene()
        for mesh_id in (99, -1):
            with self.assertRaisesRegex(ValueError, "invalid mesh_id"):
                scene.update_mesh_vertices(mesh_id, torch.zeros((3, 3), device="cuda"))

    def test_double_backward_is_refused(self):
        vertices, faces = _triangle()
        vertices = vertices.requires_grad_()
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
        ray = rt.Ray(torch.tensor([[0.25, 0.25, -1.0]], device="cuda"), torch.tensor([[1.0, 0.0, 1.0]], device="cuda"))
        (grad,) = torch.autograd.grad(scene.intersect(ray).t.sum(), vertices, create_graph=True)
        with self.assertRaisesRegex(RuntimeError, "double backward"):
            grad.sum().backward()

    def test_first_order_create_graph_still_returns_the_gradient(self):
        vertices, faces = _triangle()
        vertices = vertices.requires_grad_()
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
        ray = rt.Ray(torch.tensor([[0.25, 0.25, -1.0]], device="cuda"), torch.tensor([[0.0, 0.0, 1.0]], device="cuda"))
        (grad,) = torch.autograd.grad(scene.intersect(ray).t.sum(), vertices, create_graph=True)
        expected = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.25], [0.0, 0.0, 0.25]], device="cuda")
        torch.testing.assert_close(grad.detach(), expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class FailedSyncTests(unittest.TestCase):
    def test_sync_failing_mid_commit_stays_pending_and_can_be_retried(self):
        result = _run_isolated(
            """
            n = 512
            xs = torch.linspace(0, 1, n, device=dev)
            X, Y = torch.meshgrid(xs, xs, indexing="ij")
            v = torch.stack([X.flatten(), Y.flatten(), torch.zeros(n * n, device=dev)], 1).contiguous()
            i = torch.arange(n - 1, device=dev)
            I, J = torch.meshgrid(i, i, indexing="ij")
            a = (I * n + J).flatten(); b = a + 1; c = a + n; d = c + 1
            f = torch.cat([torch.stack([a, b, c], 1), torch.stack([b, d, c], 1)]).int().contiguous()
            s = rt.Scene()
            m = s.add_mesh(rt.Mesh(v, f), dynamic=True)
            s.build()
            ray = rt.Ray(T([[0.3, 0.3, 1.0]]), T([[0.0, 0.0, -1.0]]))
            s.update_mesh_vertices(m, v + T([0.0, 0.0, 0.5]))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Keep one small-pool segment cached so the small allocations at the start of sync() (the non-finite
            # vertex flags) succeed and the commit fails later, at its first acceleration-structure buffer, after the
            # geometry versions have advanced.
            warm = torch.empty(1024, device=dev)
            del warm
            before = int(s._native_scene.geometry_version())
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(torch.cuda.memory_reserved(0) / total, 0)
            try:
                s.sync()
                print("NOT_PROVOKED")
            except torch.OutOfMemoryError:
                print("SYNC_FAILED")
            print("FAILED_MID_COMMIT", int(s._native_scene.geometry_version()) > before)
            print("PENDING", s.has_pending_updates())
            try:
                s.intersect(ray)
                print("QUERY_ALLOWED")
            except RuntimeError as error:
                print("QUERY_REFUSED", "pending updates" in str(error))
            torch.cuda.set_per_process_memory_fraction(1.0, 0)
            s.sync()
            hit = s.intersect(ray)
            print("RETRY", float(hit.t[0]), float(hit.p[0, 2]), s.has_pending_updates())
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        if "NOT_PROVOKED" in result.stdout:
            self.skipTest("the allocator did not fail inside sync() on this device")
        self.assertIn("SYNC_FAILED", result.stdout)
        self.assertIn("FAILED_MID_COMMIT True", result.stdout)
        self.assertIn("PENDING True", result.stdout)
        self.assertIn("QUERY_REFUSED True", result.stdout)
        self.assertIn("RETRY 0.5 0.5 False", result.stdout)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class FreshThreadBuildTests(unittest.TestCase):
    def test_build_on_a_thread_that_never_used_cuda(self):
        result = _run_isolated(
            """
            import threading
            vertices = T([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
            faces = T([[0, 1, 2]], torch.int32)
            ray = rt.Ray(T([[.25, .25, -1]]), T([[0., 0, 1]]))
            outcome = []
            def worker():
                try:
                    scene = rt.Scene()
                    scene.add_mesh(rt.Mesh(vertices, faces))
                    scene.build()
                    outcome.append((scene.trace_backend, float(scene.intersect(ray).t[0])))
                except BaseException as error:
                    outcome.append(repr(error))
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            print("OUTCOME", outcome)
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("OUTCOME [('optix', 1.0)]", result.stdout)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices are required")
class CrossDeviceQueryTests(unittest.TestCase):
    def test_queries_on_another_device_are_rejected_without_poisoning_the_context(self):
        result = _run_isolated(
            """
            s = rt.Scene()
            s.add_mesh(rt.Mesh(T([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), T([[0, 1, 2]], torch.int32)))
            s.build()
            o, d = T([[.25, .25, -1]], d="cuda:1"), T([[0., 0, 1]], d="cuda:1")
            v = s._meshes[0][0].vertices.detach().clone().requires_grad_()
            for name, call in (
                ("intersect", lambda: s.intersect(rt.Ray(o, d)).t),
                ("intersect_p", lambda: s.intersect(rt.Ray(o, d)).p),
                ("intersect_ad_t", lambda: torch.ops.rayd_torch.intersect_ad_t(
                    s._native_scene, v, o, d, torch.empty(0, device="cuda:1"), None)),
                ("nearest_edge_point", lambda: s.nearest_edge(o).distance),
                ("nearest_edge_ray", lambda: s.nearest_edge(rt.Ray(o, d)).distance),
                ("nearest_edges", lambda: s.nearest_edges(o, 2).distances),
                ("visible", lambda: s.visible(o, d)),
                ("trace_reflections", lambda: s.trace_reflections(rt.Ray(o, d), 2).t),
            ):
                try:
                    call()
                    print("ACCEPTED", name)
                except RuntimeError as error:
                    print("REJECTED", name, "same CUDA device" in str(error) or "scene CUDA device" in str(error))
            print("HIT", float(s.intersect(rt.Ray(T([[.25, .25, -1]]), T([[0., 0, 1]]))).t[0]))
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertNotIn("ACCEPTED", result.stdout)
        self.assertNotIn(" False", result.stdout)
        self.assertIn("HIT 1.0", result.stdout)


if __name__ == "__main__":
    unittest.main()
