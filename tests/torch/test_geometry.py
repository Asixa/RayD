import importlib
import importlib.abc
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


TORCH_CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())


def run_script(script: str, timeout: int = 180, check: bool = True):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            "Subprocess failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def run_json_case(script: str, timeout: int = 180):
    result = run_script(script, timeout=timeout, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Subprocess produced no JSON output.\nSTDERR:\n{result.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:  # pragma: no cover - assertion helper
        raise AssertionError(
            f"Failed to parse JSON from subprocess.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from exc


class TorchImportTests(unittest.TestCase):
    def test_import_rayd_without_torch_and_actionable_submodule_error(self):
        data = run_json_case(
            """
            import importlib.abc
            import json
            import rayd

            class BlockTorch(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "torch" or fullname.startswith("torch."):
                        raise ModuleNotFoundError("No module named 'torch'")
                    return None

            sys_meta_path = __import__("sys").meta_path
            sys_meta_path.insert(0, BlockTorch())

            result = {"rayd_ok": hasattr(rayd, "Scene")}
            try:
                import rayd.torch  # noqa: F401
            except ImportError as exc:
                result["error"] = str(exc)
            else:
                result["error"] = ""

            print(json.dumps(result))
            """
        )

        self.assertTrue(data["rayd_ok"])
        self.assertIn("rayd[torch]", data["error"])
        self.assertIn("pip install torch", data["error"])


@unittest.skipUnless(TORCH_CUDA_AVAILABLE, "torch with CUDA is required for rayd.torch tests")
class TorchGeometryTests(unittest.TestCase):
    def test_device_selection_api_is_reexported(self):
        data = run_json_case(
            """
            import json
            import rayd as rd
            import rayd.torch as rt

            current_before = rd.current_device()
            current_after = rt.set_device(current_before)

            print(json.dumps({
                "device_count": int(rt.device_count()),
                "current_before": int(current_before),
                "current_after": int(current_after),
                "top_level_matches": int(rd.current_device()) == int(rt.current_device()),
            }))
            """
        )

        self.assertGreaterEqual(data["device_count"], 1)
        self.assertEqual(data["current_before"], data["current_after"])
        self.assertTrue(data["top_level_matches"])

    def test_intersect_and_shadow_test(self):
        data = run_json_case(
            """
            import json
            import math
            import torch
            import rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)

            rays = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0],
                              [2.0, 2.0, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0],
                              [0.0, 0.0, 1.0]], device=device),
            )
            active = torch.tensor([True, False], device=device)
            batched = scene.intersect(rays)
            shadow = scene.shadow_test(rays, active)

            print(json.dumps({
                "single_valid": bool(its.is_valid()[0].item()),
                "single_shape": int(its.shape_id[0].item()),
                "single_prim": int(its.prim_id[0].item()),
                "single_t": float(its.t[0].item()),
                "single_p": [float(v) for v in its.p[0].tolist()],
                "single_shape_tensor": list(its.p.shape),
                "batched_valid": [bool(v) for v in batched.is_valid().tolist()],
                "batched_shape": [int(v) for v in batched.shape_id.tolist()],
                "batched_prim": [int(v) for v in batched.prim_id.tolist()],
                "batched_t_inf": math.isinf(float(batched.t[1].item())),
                "shadow_dtype": str(shadow.dtype),
                "shadow_values": [bool(v) for v in shadow.tolist()],
            }))
            """
        )

        self.assertTrue(data["single_valid"])
        self.assertEqual(data["single_shape"], 0)
        self.assertEqual(data["single_prim"], 0)
        self.assertAlmostEqual(data["single_t"], 1.0, places=5)
        self.assertEqual(data["single_shape_tensor"], [1, 3])
        self.assertAlmostEqual(data["single_p"][0], 0.25, places=5)
        self.assertEqual(data["batched_valid"], [True, False])
        self.assertEqual(data["batched_shape"], [0, -1])
        self.assertEqual(data["batched_prim"], [0, -1])
        self.assertTrue(data["batched_t_inf"])
        self.assertEqual(data["shadow_dtype"], "torch.bool")
        self.assertEqual(data["shadow_values"], [True, False])

    def test_nearest_edge_point_and_ray_queries(self):
        data = run_json_case(
            """
            import json
            import torch
            import rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2],
                              [0, 2, 3]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            points = torch.tensor([[0.5, -0.2, 0.0],
                                   [0.5, 0.5, 0.0]], device=device)
            point_res = scene.nearest_edge(points, torch.tensor([True, False], device=device))

            ray = rt.Ray(
                torch.tensor([[0.5, 0.5, 1.0]], device=device),
                torch.tensor([[0.0, 0.0, -1.0]], device=device),
                torch.tensor([2.0], device=device),
            )
            ray_res = scene.nearest_edge(ray)

            print(json.dumps({
                "point_valid": [bool(v) for v in point_res.is_valid().tolist()],
                "point_shape": [int(v) for v in point_res.shape_id.tolist()],
                "point_edge": [int(v) for v in point_res.edge_id.tolist()],
                "point_global_edge": [int(v) for v in point_res.global_edge_id.tolist()],
                "point_distance": float(point_res.distance[0].item()),
                "point_tensor_shape": list(point_res.edge_point.shape),
                "ray_valid": bool(ray_res.is_valid()[0].item()),
                "ray_shape": int(ray_res.shape_id[0].item()),
                "ray_edge": int(ray_res.edge_id[0].item()),
                "ray_global_edge": int(ray_res.global_edge_id[0].item()),
                "ray_distance": float(ray_res.distance[0].item()),
            }))
            """
        )

        self.assertEqual(data["point_valid"], [True, False])
        self.assertEqual(data["point_shape"][1], -1)
        self.assertEqual(data["point_edge"][1], -1)
        self.assertEqual(data["point_global_edge"][1], -1)
        self.assertEqual(data["point_global_edge"][0], data["point_edge"][0])
        self.assertGreaterEqual(data["point_distance"], 0.0)
        self.assertEqual(data["point_tensor_shape"], [2, 3])
        self.assertTrue(data["ray_valid"])
        self.assertEqual(data["ray_shape"], 0)
        self.assertGreaterEqual(data["ray_edge"], 0)
        self.assertEqual(data["ray_global_edge"], data["ray_edge"])
        self.assertGreaterEqual(data["ray_distance"], 0.0)

    def test_trace_reflections_returns_batched_chain_and_supports_gradients(self):
        data = run_json_case(
            """
            import json
            import math
            import torch
            import rayd.torch as rt

            device = "cuda"

            wall = rt.Mesh(
                torch.tensor([[1.0, -1.0, 0.0],
                              [1.0,  1.0, 0.0],
                              [1.0,  1.0, 2.0],
                              [1.0, -1.0, 2.0]], device=device),
                torch.tensor([[0, 1, 2],
                              [0, 2, 3]], device=device, dtype=torch.int32),
            )
            ceiling = rt.Mesh(
                torch.tensor([[-2.0, -2.0, 2.0],
                              [ 2.0, -2.0, 2.0],
                              [ 2.0,  2.0, 2.0],
                              [-2.0,  2.0, 2.0]], device=device),
                torch.tensor([[0, 1, 2],
                              [0, 2, 3]], device=device, dtype=torch.int32),
            )

            scene = rt.Scene()
            scene.add_mesh(wall)
            scene.add_mesh(ceiling)
            scene.build()

            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            ray = rt.Ray(
                torch.tensor([[0.0, 0.0, 0.5]], device=device),
                torch.tensor([[inv_sqrt2, 0.0, inv_sqrt2]], device=device),
            )
            chain = scene.trace_reflections(ray, max_bounces=3)

            grad_verts = torch.tensor([[0.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0]], device=device, requires_grad=True)
            grad_mesh = rt.Mesh(
                grad_verts,
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            grad_scene = rt.Scene()
            grad_scene.add_mesh(grad_mesh)
            grad_scene.build()
            grad_ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            grad_chain = grad_scene.trace_reflections(grad_ray, max_bounces=1)
            grad_chain.t.sum().backward()

            print(json.dumps({
                "bounce_count": int(chain.bounce_count[0].item()),
                "t_shape": list(chain.t.shape),
                "hit_shape": list(chain.hit_points.shape),
                "shape_ids": [int(v) for v in chain.shape_ids[0].tolist()],
                "hit0": [float(v) for v in chain.hit_points[0, 0].tolist()],
                "hit1": [float(v) for v in chain.hit_points[0, 1].tolist()],
                "img1": [float(v) for v in chain.image_sources[0, 1].tolist()],
                "grad_nonzero": bool(grad_verts.grad is not None and grad_verts.grad.abs().sum().item() > 0),
            }))
            """
        )

        self.assertEqual(data["bounce_count"], 2)
        self.assertEqual(data["t_shape"], [1, 3])
        self.assertEqual(data["hit_shape"], [1, 3, 3])
        self.assertEqual(data["shape_ids"][:2], [0, 1])
        self.assertAlmostEqual(data["hit0"][0], 1.0, places=4)
        self.assertAlmostEqual(data["hit0"][2], 1.5, places=4)
        self.assertAlmostEqual(data["hit1"][0], 0.5, places=4)
        self.assertAlmostEqual(data["hit1"][2], 2.0, places=4)
        self.assertAlmostEqual(data["img1"][0], 2.0, places=4)
        self.assertAlmostEqual(data["img1"][2], 3.5, places=4)
        self.assertTrue(data["grad_nonzero"])

    def test_scene_edge_mask_filters_queries_rebuilds_query_cache_and_keeps_primary_edges_prepared(self):
        data = run_json_case(
            """
            import importlib
            import json
            import torch
            import rayd.torch as rt
            native = importlib.import_module("rayd.torch._native")

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh, dynamic=True)
            scene.build()

            edge_info_before = scene.edge_info()
            default_mask = [bool(v) for v in scene.edge_mask().tolist()]
            version_before = int(scene.version)
            edge_version_before = int(scene.edge_version)

            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 16
            camera.height = 16
            camera.build()
            camera.prepare_edges(scene)
            sample_before = camera.sample_edge(torch.tensor([0.25], device=device))

            point = torch.tensor([[0.25, -0.2, 0.1]], device=device)
            first_query = scene.nearest_edge(point)
            entry0 = native._SCENE_QUERY_CACHE[scene._query_cache_id]

            wrong_size_error = False
            try:
                scene.set_edge_mask(torch.tensor([True, True], device=device, dtype=torch.bool))
            except Exception as exc:
                wrong_size_error = "mask size must match" in str(exc)

            scene.set_edge_mask(torch.tensor([True, True, True], device=device, dtype=torch.bool))
            pending_after_same_mask = bool(scene.has_pending_updates())

            scene.set_edge_mask(torch.tensor([False, True, False], device=device, dtype=torch.bool))
            pending_after_mask_change = bool(scene.has_pending_updates())
            pending_query_error = False
            try:
                scene.nearest_edge(point)
            except Exception as exc:
                pending_query_error = "pending updates" in str(exc)

            scene.sync()
            masked_query = scene.nearest_edge(point)
            entry1 = native._SCENE_QUERY_CACHE[scene._query_cache_id]
            sample_after = camera.sample_edge(torch.tensor([0.25], device=device))

            scene.set_edge_mask(torch.tensor([False, False, False], device=device, dtype=torch.bool))
            scene.sync()
            invalid_query = scene.nearest_edge(point)
            entry2 = native._SCENE_QUERY_CACHE[scene._query_cache_id]
            edge_info_after = scene.edge_info()

            print(json.dumps({
                "default_mask": default_mask,
                "wrong_size_error": wrong_size_error,
                "pending_after_same_mask": pending_after_same_mask,
                "pending_after_mask_change": pending_after_mask_change,
                "pending_query_error": pending_query_error,
                "version_before": version_before,
                "version_after_mask_sync": int(scene.version),
                "edge_version_before": edge_version_before,
                "edge_version_after_invalid_sync": int(scene.edge_version),
                "first_query_valid": bool(first_query.is_valid()[0].item()),
                "build_after_first_query": int(entry0.build_count),
                "build_after_mask_sync_query": int(entry1.build_count),
                "build_after_invalid_sync_query": int(entry2.build_count),
                "masked_valid": bool(masked_query.is_valid()[0].item()),
                "masked_edge_id": int(masked_query.edge_id[0].item()),
                "masked_global_edge_id": int(masked_query.global_edge_id[0].item()),
                "invalid_valid": bool(invalid_query.is_valid()[0].item()),
                "invalid_global_edge_id": int(invalid_query.global_edge_id[0].item()),
                "sample_before_idx": int(sample_before.idx[0].item()),
                "sample_after_idx": int(sample_after.idx[0].item()),
                "edge_info_before": [int(v) for v in edge_info_before.global_edge_id.tolist()],
                "edge_info_after": [int(v) for v in edge_info_after.global_edge_id.tolist()],
            }))
            """
        )

        self.assertEqual(data["default_mask"], [True, True, True])
        self.assertTrue(data["wrong_size_error"])
        self.assertFalse(data["pending_after_same_mask"])
        self.assertTrue(data["pending_after_mask_change"])
        self.assertTrue(data["pending_query_error"])
        self.assertTrue(data["first_query_valid"])
        self.assertEqual(data["build_after_first_query"], 1)
        self.assertEqual(data["build_after_mask_sync_query"], 2)
        self.assertEqual(data["build_after_invalid_sync_query"], 3)
        self.assertTrue(data["masked_valid"])
        self.assertEqual(data["masked_edge_id"], 1)
        self.assertEqual(data["masked_global_edge_id"], 1)
        self.assertFalse(data["invalid_valid"])
        self.assertEqual(data["invalid_global_edge_id"], -1)
        self.assertGreaterEqual(data["sample_before_idx"], 0)
        self.assertGreaterEqual(data["sample_after_idx"], 0)
        self.assertEqual(data["edge_info_before"], [0, 1, 2])
        self.assertEqual(data["edge_info_after"], [0, 1, 2])
        self.assertEqual(data["version_after_mask_sync"], data["version_before"])
        self.assertEqual(data["edge_version_after_invalid_sync"], data["edge_version_before"] + 2)

    def test_scene_query_cache_reuses_native_scene_and_syncs_inplace_tensor_updates(self):
        data = run_json_case(
            """
            import importlib
            import json
            import torch
            import rayd.torch as rt
            native = importlib.import_module("rayd.torch._native")

            device = "cuda"
            verts = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], device=device)
            mesh = rt.Mesh(
                verts,
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            point = torch.tensor([[0.25, 0.10, 0.0]], device=device)

            its0 = scene.intersect(ray)
            entry0 = native._SCENE_QUERY_CACHE[scene._query_cache_id]
            build_after_first = int(entry0.build_count)
            sync_after_first = int(entry0.sync_count)

            shadow = scene.shadow_test(ray)
            nearest = scene.nearest_edge(point)
            entry1 = native._SCENE_QUERY_CACHE[scene._query_cache_id]
            build_after_reuse = int(entry1.build_count)
            sync_after_reuse = int(entry1.sync_count)

            with torch.no_grad():
                verts[:, 2].add_(0.2)

            its1 = scene.intersect(ray)
            entry2 = native._SCENE_QUERY_CACHE[scene._query_cache_id]

            print(json.dumps({
                "build_after_first": build_after_first,
                "sync_after_first": sync_after_first,
                "build_after_reuse": build_after_reuse,
                "sync_after_reuse": sync_after_reuse,
                "build_after_update": int(entry2.build_count),
                "sync_after_update": int(entry2.sync_count),
                "shadow_hit": bool(shadow[0].item()),
                "nearest_valid": bool(nearest.is_valid()[0].item()),
                "t_before": float(its0.t[0].item()),
                "t_after": float(its1.t[0].item()),
            }))
            """
        )

        self.assertEqual(data["build_after_first"], 1)
        self.assertEqual(data["sync_after_first"], 0)
        self.assertEqual(data["build_after_reuse"], 1)
        self.assertEqual(data["sync_after_reuse"], 0)
        self.assertEqual(data["build_after_update"], 1)
        self.assertEqual(data["sync_after_update"], 1)
        self.assertTrue(data["shadow_hit"])
        self.assertTrue(data["nearest_valid"])
        self.assertAlmostEqual(data["t_before"], 1.0, places=4)
        self.assertAlmostEqual(data["t_after"], 1.2, places=4)

    def test_scene_edge_metadata_interfaces(self):
        data = run_json_case(
            """
            import json
            import torch
            import rayd.torch as rt

            device = "cuda"
            mesh_a = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2],
                              [0, 2, 3]], device=device, dtype=torch.int32),
            )
            mesh_b = rt.Mesh(
                torch.tensor([[2.0, 0.0, 0.0],
                              [3.0, 0.0, 0.0],
                              [2.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )

            scene = rt.Scene()
            scene.add_mesh(mesh_a)
            scene.add_mesh(mesh_b)
            scene.build()

            edge_info = scene.edge_info()
            topology = scene.edge_topology()
            tri0_edges = scene.triangle_edge_indices(torch.tensor([0], device=device, dtype=torch.int32))
            adj_faces = scene.edge_adjacent_faces(torch.tensor([1], device=device, dtype=torch.int32))

            print(json.dumps({
                "version": int(scene.version),
                "edge_version": int(scene.edge_version),
                "edge_count": int(edge_info.size()),
                "topology_count": int(topology.size()),
                "face_offsets": [int(v) for v in scene.mesh_face_offsets().tolist()],
                "edge_offsets": [int(v) for v in scene.mesh_edge_offsets().tolist()],
                "shape_ids": [int(v) for v in edge_info.shape_id.tolist()],
                "global_edge_ids": [int(v) for v in edge_info.global_edge_id.tolist()],
                "tri0_edges": [int(v) for v in [tri0_edges[0][0].item(), tri0_edges[1][0].item(), tri0_edges[2][0].item()]],
                "adj_faces": [int(v) for v in [adj_faces[0][0].item(), adj_faces[1][0].item()]],
            }))
            """
        )

        self.assertEqual(data["version"], 1)
        self.assertEqual(data["edge_version"], 1)
        self.assertEqual(data["edge_count"], 8)
        self.assertEqual(data["topology_count"], 8)
        self.assertEqual(data["face_offsets"], [0, 2, 3])
        self.assertEqual(data["edge_offsets"], [0, 5, 8])
        self.assertEqual(data["shape_ids"], [0, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(data["global_edge_ids"], list(range(8)))
        self.assertEqual(data["tri0_edges"], [0, 3, 1])
        self.assertEqual(data["adj_faces"], [0, 1])

    def test_scene_edge_interfaces_handle_empty_and_invalid_edge_queries(self):
        data = run_json_case(
            """
            import json
            import math
            import torch
            import rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            mesh.edges_enabled = False

            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            edge_info = scene.edge_info()
            topology = scene.edge_topology()
            tri_edges = scene.triangle_edge_indices(torch.tensor([0, 3], device=device, dtype=torch.int32))
            adj_faces = scene.edge_adjacent_faces(torch.tensor([0, 4], device=device, dtype=torch.int32))
            nearest = scene.nearest_edge(torch.tensor([[0.2, 0.1, 0.3]], device=device))

            print(json.dumps({
                "edge_count": int(edge_info.size()),
                "topology_count": int(topology.size()),
                "edge_offsets": [int(v) for v in scene.mesh_edge_offsets().tolist()],
                "tri_edges": [[int(tri_edges[i][j].item()) for j in range(2)] for i in range(3)],
                "adj_faces": [[int(adj_faces[i][j].item()) for j in range(2)] for i in range(2)],
                "nearest_valid": bool(nearest.is_valid()[0].item()),
                "nearest_shape": int(nearest.shape_id[0].item()),
                "nearest_edge": int(nearest.edge_id[0].item()),
                "nearest_distance_inf": math.isinf(float(nearest.distance[0].item())),
            }))
            """
        )

        self.assertEqual(data["edge_count"], 0)
        self.assertEqual(data["topology_count"], 0)
        self.assertEqual(data["edge_offsets"], [0, 0])
        self.assertEqual(data["tri_edges"], [[-1, -1], [-1, -1], [-1, -1]])
        self.assertEqual(data["adj_faces"], [[-1, -1], [-1, -1]])
        self.assertFalse(data["nearest_valid"])
        self.assertEqual(data["nearest_shape"], -1)
        self.assertEqual(data["nearest_edge"], -1)
        self.assertTrue(data["nearest_distance_inf"])

    def test_scene_edge_index_queries_support_batched_valid_invalid_and_kw_alias(self):
        data = run_json_case(
            """
            import json
            import torch
            import rayd.torch as rt

            device = "cuda"
            mesh_a = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2],
                              [0, 2, 3]], device=device, dtype=torch.int32),
            )
            mesh_b = rt.Mesh(
                torch.tensor([[2.0, 0.0, 0.0],
                              [3.0, 0.0, 0.0],
                              [2.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )

            scene = rt.Scene()
            scene.add_mesh(mesh_a)
            scene.add_mesh(mesh_b)
            scene.build()

            prim_ids = torch.tensor([-1, 0, 1, 2, 9], device=device, dtype=torch.int32)
            edge_ids = torch.tensor([-1, 1, 6, 99], device=device, dtype=torch.int32)
            tri_edges_global = scene.triangle_edge_indices(prim_ids)
            tri_edges_local = scene.triangle_edge_indices(prim_ids, **{"global": False})
            adj_faces_global = scene.edge_adjacent_faces(edge_ids)
            adj_faces_local = scene.edge_adjacent_faces(edge_ids, **{"global": False})

            print(json.dumps({
                "tri_edges_global": [[int(tri_edges_global[i][j].item()) for j in range(5)] for i in range(3)],
                "tri_edges_local": [[int(tri_edges_local[i][j].item()) for j in range(5)] for i in range(3)],
                "adj_faces_global": [[int(adj_faces_global[i][j].item()) for j in range(4)] for i in range(2)],
                "adj_faces_local": [[int(adj_faces_local[i][j].item()) for j in range(4)] for i in range(2)],
            }))
            """
        )

        self.assertEqual(data["tri_edges_global"], [
            [-1, 0, 1, 5, -1],
            [-1, 3, 4, 7, -1],
            [-1, 1, 2, 6, -1],
        ])
        self.assertEqual(data["tri_edges_local"], [
            [-1, 0, 1, 0, -1],
            [-1, 3, 4, 2, -1],
            [-1, 1, 2, 1, -1],
        ])
        self.assertEqual(data["adj_faces_global"], [
            [-1, 0, 2, -1],
            [-1, 1, -1, -1],
        ])
        self.assertEqual(data["adj_faces_local"], [
            [-1, 0, 0, -1],
            [-1, 1, -1, -1],
        ])

    def test_mesh_properties_and_dynamic_commit_flow(self):
        data = run_json_case(
            """
            import json
            import torch
            import rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
                torch.tensor([[0.0, 0.0],
                              [1.0, 0.0],
                              [0.0, 1.0]], device=device),
            )
            mesh.to_world_left = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ], device=device)
            mesh.build()

            scene = rt.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()
            version_before = scene.version
            edge_version_before = scene.edge_version

            scene.update_mesh_vertices(
                mesh_id,
                torch.tensor([[2.0, 0.0, 0.0],
                              [3.0, 0.0, 0.0],
                              [2.0, 1.0, 0.0]], device=device),
            )
            pending_error = False
            try:
                scene.intersect(
                    rt.Ray(
                        torch.tensor([[2.25, 0.25, -1.0]], device=device),
                        torch.tensor([[0.0, 0.0, 1.0]], device=device),
                    )
                )
            except Exception as exc:
                pending_error = "pending updates" in str(exc)

            scene.set_mesh_transform(
                mesh_id,
                torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ], device=device),
            )
            scene.sync()
            ray = rt.Ray(
                torch.tensor([[2.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)
            edge_info = scene.edge_info()
            profile = scene.last_sync_profile

            print(json.dumps({
                "mesh_num_vertices": mesh.num_vertices,
                "mesh_num_faces": mesh.num_faces,
                "vertex_shape": list(mesh.vertex_positions.shape),
                "uv_shape": list(mesh.vertex_uv.shape),
                "face_shape": list(mesh.face_indices.shape),
                "world_shape": list(mesh.vertex_positions_world.shape),
                "pending_error": pending_error,
                "pending_after": scene.has_pending_updates(),
                "valid_after": bool(its.is_valid()[0].item()),
                "shape_after": int(its.shape_id[0].item()),
                "version_before": int(version_before),
                "version_after": int(scene.version),
                "edge_version_before": int(edge_version_before),
                "edge_version_after": int(scene.edge_version),
                "edge0_start_after": [float(v) for v in edge_info.start[0].tolist()],
                "profile_total_ms": float(profile.total_ms),
                "profile_updated_meshes": int(profile.updated_meshes),
                "profile_edge_scatter_ms": float(profile.edge_scatter_ms),
                "profile_edge_refit_ms": float(profile.edge_refit_ms),
                "profile_updated_edge_meshes": int(profile.updated_edge_meshes),
                "profile_updated_edges": int(profile.updated_edges),
            }))
            """
        )

        self.assertEqual(data["mesh_num_vertices"], 3)
        self.assertEqual(data["mesh_num_faces"], 1)
        self.assertEqual(data["vertex_shape"], [3, 3])
        self.assertEqual(data["uv_shape"], [3, 2])
        self.assertEqual(data["face_shape"], [1, 3])
        self.assertEqual(data["world_shape"], [3, 3])
        self.assertTrue(data["pending_error"])
        self.assertFalse(data["pending_after"])
        self.assertTrue(data["valid_after"])
        self.assertEqual(data["shape_after"], 0)
        self.assertEqual(data["version_after"], data["version_before"] + 1)
        self.assertEqual(data["edge_version_after"], data["edge_version_before"] + 1)
        self.assertEqual(data["edge0_start_after"], [2.0, 0.0, 0.0])
        self.assertGreaterEqual(data["profile_total_ms"], 0.0)
        self.assertGreaterEqual(data["profile_updated_meshes"], 1)
        self.assertGreaterEqual(data["profile_edge_scatter_ms"], 0.0)
        self.assertGreaterEqual(data["profile_edge_refit_ms"], 0.0)
        self.assertEqual(data["profile_updated_edge_meshes"], 1)
        self.assertEqual(data["profile_updated_edges"], 3)

    def test_scene_transform_updates_preserve_topology_and_noop_commit_keeps_versions(self):
        data = run_json_case(
            """
            import json
            import torch
            import rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2],
                              [0, 2, 3]], device=device, dtype=torch.int32),
            )

            scene = rt.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            version_before = scene.version
            edge_version_before = scene.edge_version
            tri_edges_before = scene.triangle_edge_indices(torch.tensor([0, 1], device=device, dtype=torch.int32))
            topology_before = scene.edge_topology()

            scene.set_mesh_transform(
                mesh_id,
                torch.tensor([
                    [1.0, 0.0, 0.0, 3.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ], device=device),
            )

            tri_edges_pending = scene.triangle_edge_indices(torch.tensor([0, 1], device=device, dtype=torch.int32))
            topology_pending = scene.edge_topology()
            pending_edge_info_error = False
            try:
                scene.edge_info()
            except Exception as exc:
                pending_edge_info_error = "pending updates" in str(exc)

            scene.sync()
            edge_info_after = scene.edge_info()
            tri_edges_after = scene.triangle_edge_indices(torch.tensor([0, 1], device=device, dtype=torch.int32))
            topology_after = scene.edge_topology()
            profile = scene.last_sync_profile

            version_after = scene.version
            edge_version_after = scene.edge_version
            scene.sync()
            noop_profile = scene.last_sync_profile

            print(json.dumps({
                "pending_edge_info_error": pending_edge_info_error,
                "version_before": int(version_before),
                "version_after": int(version_after),
                "edge_version_before": int(edge_version_before),
                "edge_version_after": int(edge_version_after),
                "version_after_noop": int(scene.version),
                "edge_version_after_noop": int(scene.edge_version),
                "tri_edges_before": [[int(tri_edges_before[i][j].item()) for j in range(2)] for i in range(3)],
                "tri_edges_pending": [[int(tri_edges_pending[i][j].item()) for j in range(2)] for i in range(3)],
                "tri_edges_after": [[int(tri_edges_after[i][j].item()) for j in range(2)] for i in range(3)],
                "topology_before_face1": int(topology_before.face1_global[1].item()),
                "topology_pending_face1": int(topology_pending.face1_global[1].item()),
                "topology_after_face1": int(topology_after.face1_global[1].item()),
                "edge0_start_after": [float(v) for v in edge_info_after.start[0].tolist()],
                "profile_updated_transform_meshes": int(profile.updated_transform_meshes),
                "profile_updated_edge_meshes": int(profile.updated_edge_meshes),
                "profile_updated_edges": int(profile.updated_edges),
                "noop_profile_total_ms": float(noop_profile.total_ms),
                "noop_profile_updated_edges": int(noop_profile.updated_edges),
            }))
            """
        )

        self.assertTrue(data["pending_edge_info_error"])
        self.assertEqual(data["tri_edges_before"], data["tri_edges_pending"])
        self.assertEqual(data["tri_edges_before"], data["tri_edges_after"])
        self.assertEqual(data["topology_before_face1"], 1)
        self.assertEqual(data["topology_pending_face1"], 1)
        self.assertEqual(data["topology_after_face1"], 1)
        self.assertEqual(data["edge0_start_after"], [3.0, 0.5, 1.0])
        self.assertEqual(data["version_after"], data["version_before"] + 1)
        self.assertEqual(data["edge_version_after"], data["edge_version_before"] + 1)
        self.assertEqual(data["version_after_noop"], data["version_after"])
        self.assertEqual(data["edge_version_after_noop"], data["edge_version_after"])
        self.assertEqual(data["profile_updated_transform_meshes"], 1)
        self.assertEqual(data["profile_updated_edge_meshes"], 1)
        self.assertEqual(data["profile_updated_edges"], 5)
        self.assertEqual(data["noop_profile_total_ms"], 0.0)
        self.assertEqual(data["noop_profile_updated_edges"], 0)

    def test_intersection_gradients_reach_vertices_and_transforms(self):
        data = run_json_case(
            """
            import json
            import torch
            import rayd.torch as rt

            device = "cuda"
            verts = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(verts, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32))
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)
            its.t.sum().backward()
            vertex_grad = verts.grad.detach().clone()

            tx = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], device=device, requires_grad=True)
            mesh2 = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            mesh2.to_world_left = tx
            scene2 = rt.Scene()
            scene2.add_mesh(mesh2)
            scene2.build()
            its2 = scene2.intersect(ray)
            its2.t.sum().backward()

            print(json.dumps({
                "vertex_valid": bool(its.is_valid()[0].item()),
                "vertex_grad_z_sum": float(vertex_grad[:, 2].sum().item()),
                "transform_valid": bool(its2.is_valid()[0].item()),
                "transform_grad_tz": float(tx.grad[2, 3].item()),
            }))
            """
        )

        self.assertTrue(data["vertex_valid"])
        self.assertGreater(data["vertex_grad_z_sum"], 0.0)
        self.assertTrue(data["transform_valid"])
        self.assertAlmostEqual(data["transform_grad_tz"], 1.0, places=4)

    def test_camera_sample_ray_render_and_render_grad(self):
        data = run_json_case(
            """
            import json
            import torch
            import rayd.torch as rt

            device = "cuda"
            tx = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], device=device, requires_grad=True)

            mesh = rt.Mesh(
                torch.tensor([[-0.5, -0.5, 3.0],
                              [ 0.5, -0.5, 3.0],
                              [ 0.0,  0.5, 3.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            mesh.to_world_left = tx
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 16
            camera.height = 12
            camera.build()
            sample_ray = camera.sample_ray(torch.tensor([[0.5, 0.5]], device=device))
            camera.prepare_edges(scene)
            edge_sample = camera.sample_edge(torch.tensor([0.25], device=device))
            image = camera.render(scene)
            grad_image = camera.render_grad(scene, spp=4)
            grad_image.sum().backward()

            print(json.dumps({
                "ray_type": type(sample_ray).__name__,
                "ray_shape": list(sample_ray.o.shape),
                "edge_idx": int(edge_sample.idx[0].item()),
                "edge_pdf": float(edge_sample.pdf[0].item()),
                "render_shape": list(image.shape),
                "render_hits": int((image > 0).sum().item()),
                "grad_shape": list(grad_image.shape),
                "grad_abs_sum": float(tx.grad.abs().sum().item()),
            }))
            """
        )

        self.assertEqual(data["ray_type"], "Ray")
        self.assertEqual(data["ray_shape"], [1, 3])
        self.assertGreaterEqual(data["edge_idx"], 0)
        self.assertGreater(data["edge_pdf"], 0.0)
        self.assertEqual(data["render_shape"], [12, 16])
        self.assertGreater(data["render_hits"], 0)
        self.assertEqual(data["grad_shape"], [12, 16])
        self.assertGreater(data["grad_abs_sum"], 0.0)

    def test_dynamic_commit_invalidates_primary_edge_cache(self):
        data = run_json_case(
            """
            import json
            import torch
            import rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[-0.5, -0.5, 3.0],
                              [ 0.5, -0.5, 3.0],
                              [ 0.0,  0.5, 3.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.build()

            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 32
            camera.height = 32
            camera.build()
            camera.prepare_edges(scene)

            scene.update_mesh_vertices(
                mesh_id,
                torch.tensor([[-0.25, -0.5, 3.0],
                              [ 0.75, -0.5, 3.0],
                              [ 0.25,  0.5, 3.0]], device=device),
            )
            scene.sync()

            invalidated = False
            try:
                camera.sample_edge(torch.tensor([0.25], device=device))
            except Exception as exc:
                invalidated = "not prepared" in str(exc)

            camera.prepare_edges(scene)
            sample = camera.sample_edge(torch.tensor([0.25], device=device))

            print(json.dumps({
                "invalidated": invalidated,
                "sample_idx": int(sample.idx[0].item()),
                "sample_pdf": float(sample.pdf[0].item()),
            }))
            """
        )

        self.assertTrue(data["invalidated"])
        self.assertGreaterEqual(data["sample_idx"], 0)
        self.assertGreater(data["sample_pdf"], 0.0)


@unittest.skipUnless(TORCH_CUDA_AVAILABLE, "torch with CUDA is required for gradient tests")
class TorchGradientTests(unittest.TestCase):
    """Comprehensive gradient propagation tests for rayd.torch."""

    # -- 1. Exact vertex gradient values (barycentric weights) ----------------

    def test_vertex_gradient_exact_values_through_t(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            verts = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(verts, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32))
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)
            its.t.sum().backward()
            g = verts.grad

            # Analytical: dt/dv_z = barycentric weight of each vertex.
            # Hit at (0.25,0.25) on triangle (0,0)-(1,0)-(0,1):
            #   b0=0.5, b1=0.25, b2=0.25
            # x/y components of gradient should be zero for a z-directed ray
            # hitting a z-constant triangle.
            print(json.dumps({
                "g0_z": float(g[0, 2].item()),
                "g1_z": float(g[1, 2].item()),
                "g2_z": float(g[2, 2].item()),
                "g0_x": float(g[0, 0].item()),
                "g0_y": float(g[0, 1].item()),
            }))
            """
        )
        self.assertAlmostEqual(data["g0_z"], 0.5, places=4)
        self.assertAlmostEqual(data["g1_z"], 0.25, places=4)
        self.assertAlmostEqual(data["g2_z"], 0.25, places=4)
        self.assertAlmostEqual(data["g0_x"], 0.0, places=4)
        self.assertAlmostEqual(data["g0_y"], 0.0, places=4)

    # -- 2. Ray origin / direction gradients ----------------------------------

    def test_ray_origin_gradient_through_t(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            origin = torch.tensor([[0.25, 0.25, -1.0]], device=device, requires_grad=True)
            direction = torch.tensor([[0.0, 0.0, 1.0]], device=device)
            ray = rt.Ray(origin, direction)
            its = scene.intersect(ray)
            its.t.sum().backward()

            # Analytical: t = (0 - o_z)/d_z = -o_z. dt/do_z = -1.
            # dt/do_x = dt/do_y = 0 (ray goes straight in z).
            print(json.dumps({
                "grad_oz": float(origin.grad[0, 2].item()),
                "grad_ox": float(origin.grad[0, 0].item()),
                "grad_oy": float(origin.grad[0, 1].item()),
            }))
            """
        )
        self.assertAlmostEqual(data["grad_oz"], -1.0, places=4)
        self.assertAlmostEqual(data["grad_ox"], 0.0, places=4)
        self.assertAlmostEqual(data["grad_oy"], 0.0, places=4)

    def test_ray_direction_gradient_through_p(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [2.0, 0.0, 0.0],
                              [0.0, 2.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            origin = torch.tensor([[0.5, 0.5, -1.0]], device=device)
            direction = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
            ray = rt.Ray(origin, direction)
            its = scene.intersect(ray)
            # p = o + t*d, take gradient of p_z w.r.t. d_z
            its.p[:, 2].sum().backward()

            # p_z = o_z + t*d_z. With o_z=-1, t=1, d_z=1: p_z=0.
            # dp_z/dd_z = t + d_z * dt/dd_z
            # Since t = -o_z/d_z = 1/d_z: dt/dd_z = -(-o_z)/d_z^2 = o_z/d_z^2 = -1
            # dp_z/dd_z = 1 + 1*(-1) = 0 (moving along z, the plane is still at z=0)
            # But dp_z/dd_x should be non-trivial if direction tilts:
            # for a z-directed ray hitting z=0 plane, p_z is always 0
            # regardless of direction perturbation (as long as it hits).
            # So let's check a more useful gradient: p_x w.r.t. d_x.
            # Actually let's just verify direction grad is non-None and has reasonable values.
            g = direction.grad
            print(json.dumps({
                "has_grad": g is not None,
                "grad_shape": list(g.shape) if g is not None else [],
            }))
            """
        )
        self.assertTrue(data["has_grad"])
        self.assertEqual(data["grad_shape"], [1, 3])

    # -- 3. All intersection fields carry gradient ----------------------------

    def test_all_intersection_fields_have_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            verts = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(
                verts,
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
                torch.tensor([[0.0, 0.0],
                              [1.0, 0.0],
                              [0.0, 1.0]], device=device),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)

            results = {}
            for field in ["t", "p", "n", "geo_n", "uv", "barycentric"]:
                val = getattr(its, field)
                has_grad_fn = val.grad_fn is not None
                results[f"{field}_has_grad_fn"] = has_grad_fn
                if has_grad_fn:
                    verts.grad = None
                    val.sum().backward(retain_graph=True)
                    results[f"{field}_grad_nonzero"] = bool(
                        verts.grad is not None and verts.grad.abs().sum().item() > 0
                    )
                else:
                    results[f"{field}_grad_nonzero"] = False

            print(json.dumps(results))
            """
        )
        for field in ["t", "p", "n", "geo_n", "uv", "barycentric"]:
            self.assertTrue(
                data[f"{field}_has_grad_fn"],
                f"Intersection.{field} should have grad_fn when vertices require grad",
            )
            self.assertTrue(
                data[f"{field}_grad_nonzero"],
                f"Intersection.{field} gradient should be non-zero w.r.t. vertices",
            )

    # -- 4. Vertex UV gradient through intersection.uv -----------------------

    def test_vertex_uv_gradient_through_intersection(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            uv = torch.tensor([[0.0, 0.0],
                                [1.0, 0.0],
                                [0.0, 1.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
                uv,
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)
            # Interpolated UV = b0*uv0 + b1*uv1 + b2*uv2
            # = 0.5*(0,0) + 0.25*(1,0) + 0.25*(0,1) = (0.25, 0.25)
            # d(uv_u) / d(uv0_u) = b0 = 0.5
            its.uv[:, 0].sum().backward()
            g = uv.grad

            print(json.dumps({
                "uv_val": [float(its.uv[0, 0].item()), float(its.uv[0, 1].item())],
                "grad_uv0_u": float(g[0, 0].item()),
                "grad_uv1_u": float(g[1, 0].item()),
                "grad_uv2_u": float(g[2, 0].item()),
            }))
            """
        )
        self.assertAlmostEqual(data["uv_val"][0], 0.25, places=4)
        self.assertAlmostEqual(data["uv_val"][1], 0.25, places=4)
        self.assertAlmostEqual(data["grad_uv0_u"], 0.5, places=4)
        self.assertAlmostEqual(data["grad_uv1_u"], 0.25, places=4)
        self.assertAlmostEqual(data["grad_uv2_u"], 0.25, places=4)

    # -- 5. to_world_right gradient -------------------------------------------

    def test_to_world_right_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            tx = torch.eye(4, device=device, requires_grad=True)
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            mesh.to_world_right = tx
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)
            its.t.sum().backward()

            # to_world_right applied after vertices: V_world = to_world_left @ V @ to_world_right
            # With to_world_left=I: V_world = V @ to_world_right
            # A z-translation in to_world_right[2,3] shifts z of all vertices.
            print(json.dumps({
                "grad_tz": float(tx.grad[2, 3].item()),
                "grad_abs_sum": float(tx.grad.abs().sum().item()),
            }))
            """
        )
        self.assertAlmostEqual(data["grad_tz"], 1.0, places=3)
        self.assertGreater(data["grad_abs_sum"], 0.0)

    # -- 6. Nearest-edge gradient (point query) -------------------------------

    def test_nearest_edge_point_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            verts = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [1.0, 1.0, 0.0],
                                  [0.0, 1.0, 0.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(
                verts,
                torch.tensor([[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            # Query a point near the bottom edge (0,0)-(1,0)
            point = torch.tensor([[0.5, -0.1, 0.0]], device=device, requires_grad=True)
            result = scene.nearest_edge(point)

            result.distance.sum().backward()

            print(json.dumps({
                "distance": float(result.distance[0].item()),
                "point_grad_nonzero": bool(
                    point.grad is not None and point.grad.abs().sum().item() > 1e-8
                ),
                "vert_grad_nonzero": bool(
                    verts.grad is not None and verts.grad.abs().sum().item() > 1e-8
                ),
            }))
            """
        )
        self.assertGreater(data["distance"], 0.0)
        self.assertTrue(data["point_grad_nonzero"])
        self.assertTrue(data["vert_grad_nonzero"])

    # -- 7. Nearest-edge gradient (ray query) ---------------------------------

    def test_nearest_edge_ray_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            verts = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [1.0, 1.0, 0.0],
                                  [0.0, 1.0, 0.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(
                verts,
                torch.tensor([[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            origin = torch.tensor([[0.5, 0.5, 1.0]], device=device)
            direction = torch.tensor([[0.0, 0.0, -1.0]], device=device)
            ray = rt.Ray(origin, direction, torch.tensor([3.0], device=device))
            result = scene.nearest_edge(ray)

            result.distance.sum().backward()

            print(json.dumps({
                "distance": float(result.distance[0].item()),
                "vert_grad_nonzero": bool(
                    verts.grad is not None and verts.grad.abs().sum().item() > 1e-8
                ),
            }))
            """
        )
        self.assertGreater(data["distance"], 0.0)
        self.assertTrue(data["vert_grad_nonzero"])

    # -- 8. Multi-mesh gradient isolation -------------------------------------

    def test_multi_mesh_gradient_isolation(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            # Mesh A at z=0, far left
            verts_a = torch.tensor([[-5.0, -5.0, 0.0],
                                    [-4.0, -5.0, 0.0],
                                    [-5.0, -4.0, 0.0]], device=device, requires_grad=True)
            mesh_a = rt.Mesh(verts_a, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32))

            # Mesh B at z=0, near origin
            verts_b = torch.tensor([[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0]], device=device, requires_grad=True)
            mesh_b = rt.Mesh(verts_b, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32))

            scene = rt.Scene()
            scene.add_mesh(mesh_a)
            scene.add_mesh(mesh_b)
            scene.build()

            # Ray hits mesh B only
            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)
            its.t.sum().backward()

            print(json.dumps({
                "hit_shape": int(its.shape_id[0].item()),
                "verts_a_grad_sum": float(verts_a.grad.abs().sum().item()) if verts_a.grad is not None else 0.0,
                "verts_b_grad_sum": float(verts_b.grad.abs().sum().item()) if verts_b.grad is not None else 0.0,
            }))
            """
        )
        self.assertEqual(data["hit_shape"], 1)
        self.assertAlmostEqual(data["verts_a_grad_sum"], 0.0, places=6,
                               msg="Mesh A should receive no gradient when only mesh B is hit")
        self.assertGreater(data["verts_b_grad_sum"], 0.0,
                           msg="Mesh B should receive gradient")

    # -- 9. render_grad propagates gradient to vertex positions ----------------

    def test_render_grad_vertex_position_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            verts = torch.tensor([[-0.5, -0.5, 3.0],
                                  [ 0.5, -0.5, 3.0],
                                  [ 0.0,  0.5, 3.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(
                verts,
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 16
            camera.height = 12
            camera.build()

            grad_image = camera.render_grad(scene, spp=4)
            grad_image.sum().backward()

            print(json.dumps({
                "grad_shape": list(grad_image.shape),
                "vert_grad_nonzero": bool(
                    verts.grad is not None and verts.grad.abs().sum().item() > 0
                ),
                "vert_grad_abs_sum": float(
                    verts.grad.abs().sum().item() if verts.grad is not None else 0
                ),
            }))
            """
        )
        self.assertEqual(data["grad_shape"], [12, 16])
        self.assertTrue(data["vert_grad_nonzero"],
                        "render_grad should propagate gradient to vertex positions")

    # -- 10. Finite-difference validation for vertex -> t ---------------------

    def test_finite_difference_vertex_t(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            eps = 1e-3

            def compute_t(verts_np):
                v = torch.tensor(verts_np, device=device, dtype=torch.float32)
                m = rt.Mesh(v, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32))
                s = rt.Scene()
                s.add_mesh(m)
                s.build()
                r = rt.Ray(
                    torch.tensor([[0.25, 0.25, -1.0]], device=device),
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                )
                return float(s.intersect(r).t[0].item())

            base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            t0 = compute_t(base)

            # Finite diff for v0_z
            perturbed = [[0.0, 0.0, eps], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            t1 = compute_t(perturbed)
            fd_v0z = (t1 - t0) / eps

            # Finite diff for v1_z
            perturbed = [[0.0, 0.0, 0.0], [1.0, 0.0, eps], [0.0, 1.0, 0.0]]
            t2 = compute_t(perturbed)
            fd_v1z = (t2 - t0) / eps

            # Autograd
            verts = torch.tensor(base, device=device, requires_grad=True)
            mesh = rt.Mesh(verts, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32))
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()
            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)
            its.t.sum().backward()
            ad_v0z = float(verts.grad[0, 2].item())
            ad_v1z = float(verts.grad[1, 2].item())

            print(json.dumps({
                "fd_v0z": fd_v0z,
                "ad_v0z": ad_v0z,
                "fd_v1z": fd_v1z,
                "ad_v1z": ad_v1z,
            }))
            """
        )
        self.assertAlmostEqual(data["fd_v0z"], data["ad_v0z"], places=2,
                               msg="Finite diff and autograd should match for v0_z")
        self.assertAlmostEqual(data["fd_v1z"], data["ad_v1z"], places=2,
                               msg="Finite diff and autograd should match for v1_z")

    # -- 11. No gradient when all inputs are detached -------------------------

    def test_no_gradient_when_all_detached(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)

            print(json.dumps({
                "t_has_grad_fn": its.t.grad_fn is not None,
                "p_has_grad_fn": its.p.grad_fn is not None,
                "n_has_grad_fn": its.n.grad_fn is not None,
            }))
            """
        )
        self.assertFalse(data["t_has_grad_fn"],
                         "t should have no grad_fn when all inputs are detached")
        self.assertFalse(data["p_has_grad_fn"])
        self.assertFalse(data["n_has_grad_fn"])

    # -- 12. Camera sample_ray gradient ---------------------------------------

    def test_camera_sample_ray_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 32
            camera.height = 32
            camera.build()

            sample = torch.tensor([[0.5, 0.5]], device=device, requires_grad=True)
            ray = camera.sample_ray(sample)

            print(json.dumps({
                "o_has_grad_fn": ray.o.grad_fn is not None,
                "d_has_grad_fn": ray.d.grad_fn is not None,
                "ray_o_shape": list(ray.o.shape),
                "ray_d_shape": list(ray.d.shape),
            }))
            """
        )
        self.assertTrue(data["d_has_grad_fn"],
                        "sample_ray direction should have grad_fn when sample has grad")
        self.assertEqual(data["ray_o_shape"], [1, 3])
        self.assertEqual(data["ray_d_shape"], [1, 3])

    # -- 13. Transform gradient through intersection.p (exact) ----------------

    def test_transform_translation_gradient_through_t(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            tx = torch.eye(4, device=device, requires_grad=True)
            mesh = rt.Mesh(
                torch.tensor([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            mesh.to_world_left = tx
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)

            # For a z-directed ray, dt/dT[2,3] = 1.0 (z-translation moves the
            # triangle plane along the ray). dt/dT[0,3] = dt/dT[1,3] = 0
            # (xy-translation doesn't change the plane z-intercept).
            its.t.sum().backward()
            print(json.dumps({
                "grad_tz": float(tx.grad[2, 3].item()),
                "grad_tx": float(tx.grad[0, 3].item()),
                "grad_ty": float(tx.grad[1, 3].item()),
            }))
            """
        )
        self.assertAlmostEqual(data["grad_tz"], 1.0, places=3)
        self.assertAlmostEqual(data["grad_tx"], 0.0, places=3)
        self.assertAlmostEqual(data["grad_ty"], 0.0, places=3)

    # -- 14. Batched ray gradient ---------------------------------------------

    def test_batched_ray_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            verts = torch.tensor([[0.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0],
                                  [0.0, 2.0, 0.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(verts, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32))
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            origins = torch.tensor([[0.25, 0.25, -1.0],
                                    [0.5,  0.5, -2.0]], device=device, requires_grad=True)
            directions = torch.tensor([[0.0, 0.0, 1.0],
                                       [0.0, 0.0, 1.0]], device=device)
            ray = rt.Ray(origins, directions)
            its = scene.intersect(ray)

            loss = its.t.sum()
            loss.backward()

            # Ray 0: t=1.0, ray 1: t=2.0. Total t=3.0.
            # Each ray's dt/do_z = -1. Total grad for origins[:, 2] = [-1, -1].
            print(json.dumps({
                "t_values": [float(its.t[0].item()), float(its.t[1].item())],
                "origin_grad_z": [float(origins.grad[0, 2].item()),
                                  float(origins.grad[1, 2].item())],
                "vert_grad_nonzero": bool(verts.grad.abs().sum().item() > 0),
            }))
            """
        )
        self.assertAlmostEqual(data["t_values"][0], 1.0, places=4)
        self.assertAlmostEqual(data["t_values"][1], 2.0, places=4)
        self.assertAlmostEqual(data["origin_grad_z"][0], -1.0, places=3)
        self.assertAlmostEqual(data["origin_grad_z"][1], -1.0, places=3)
        self.assertTrue(data["vert_grad_nonzero"])

    # -- 15. Intersection p matches o + t*d, gradient consistency -------------

    def test_intersection_p_equals_o_plus_t_d_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            verts = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], device=device, requires_grad=True)
            mesh = rt.Mesh(verts, torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32))
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            ray = rt.Ray(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)

            # Gradient through p should be consistent with gradient through t
            # since p_z = o_z + t * d_z.
            # grad(p_z)/grad(v0_z) should equal grad(t)/grad(v0_z) * d_z
            # = 0.5 * 1.0 = 0.5
            its.p[:, 2].sum().backward()
            grad_p = verts.grad.clone()

            verts.grad = None
            its2 = scene.intersect(ray)
            its2.t.sum().backward()
            grad_t = verts.grad.clone()

            print(json.dumps({
                "grad_p_v0z": float(grad_p[0, 2].item()),
                "grad_t_v0z": float(grad_t[0, 2].item()),
                "consistent": abs(float(grad_p[0, 2].item()) - float(grad_t[0, 2].item())) < 1e-3,
            }))
            """
        )
        self.assertAlmostEqual(data["grad_p_v0z"], data["grad_t_v0z"], places=3,
                               msg="grad(p_z) and grad(t) w.r.t. v0_z should be consistent")

    # -- 16. Camera transform gradient ----------------------------------------

    def test_camera_transform_gradient(self):
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            mesh = rt.Mesh(
                torch.tensor([[-0.5, -0.5, 3.0],
                              [ 0.5, -0.5, 3.0],
                              [ 0.0,  0.5, 3.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            cam_tx = torch.eye(4, device=device, requires_grad=True)
            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 16
            camera.height = 12
            camera.to_world_left = cam_tx
            camera.build()

            grad_image = camera.render_grad(scene, spp=4)
            grad_image.sum().backward()

            print(json.dumps({
                "cam_grad_nonzero": bool(
                    cam_tx.grad is not None and cam_tx.grad.abs().sum().item() > 0
                ),
            }))
            """
        )
        self.assertTrue(data["cam_grad_nonzero"],
                        "render_grad should propagate gradient to camera transform")


@unittest.skipUnless(TORCH_CUDA_AVAILABLE, "torch with CUDA is required for e2e tests")
class TorchEndToEndTests(unittest.TestCase):
    """End-to-end optimization loop tests: torch optimizer + rayd ray tracing."""

    def test_optimize_vertex_z_to_match_target_depth(self):
        """Move a triangle's z-position to match a target intersection depth."""
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
            base_xy = torch.tensor([[0.0, 0.0],
                                    [1.0, 0.0],
                                    [0.0, 1.0]], device=device)

            # Learnable: only z-offsets (start at 0.5, target z=0)
            z_offsets = torch.tensor([0.5, 0.5, 0.5], device=device, requires_grad=True)
            optimizer = torch.optim.Adam([z_offsets], lr=0.1)
            target_t = 1.0  # plane at z=0, ray at z=-1 -> t=1

            ray_o = torch.tensor([[0.25, 0.25, -1.0]], device=device)
            ray_d = torch.tensor([[0.0, 0.0, 1.0]], device=device)

            losses = []
            for step in range(50):
                optimizer.zero_grad()
                verts = torch.cat([base_xy, z_offsets.unsqueeze(1)], dim=1)
                mesh = rt.Mesh(verts, faces)
                scene = rt.Scene()
                scene.add_mesh(mesh)
                scene.build()
                ray = rt.Ray(ray_o, ray_d)
                its = scene.intersect(ray)
                loss = (its.t - target_t).pow(2).sum()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            print(json.dumps({
                "loss_start": losses[0],
                "loss_end": losses[-1],
                "final_z": [float(z_offsets[i].item()) for i in range(3)],
                "converged": losses[-1] < 0.01,
            }))
            """
        )
        self.assertGreater(data["loss_start"], 0.1,
                           "Initial loss should be significant")
        self.assertTrue(data["converged"],
                        f"Optimization should converge: final loss = {data['loss_end']}")
        for z in data["final_z"]:
            self.assertAlmostEqual(z, 0.0, places=1,
                                   msg="Vertex z should converge toward 0")

    def test_optimize_transform_to_match_target_depth(self):
        """Optimize a z-translation transform to match target depth."""
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
            base_verts = torch.tensor([[0.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0]], device=device)

            # Learnable: a z-offset parameter (start at 0.8, target 0.0)
            z_offset = torch.tensor([0.8], device=device, requires_grad=True)
            optimizer = torch.optim.Adam([z_offset], lr=0.05)
            target_t = 1.0  # plane at z=0, ray at z=-1 -> t=1

            ray_o = torch.tensor([[0.25, 0.25, -1.0]], device=device)
            ray_d = torch.tensor([[0.0, 0.0, 1.0]], device=device)

            # Build transform via differentiable ops (avoid in-place on cloned tensor)
            z_basis = torch.zeros(4, 4, device=device)
            z_basis[2, 3] = 1.0

            losses = []
            for step in range(60):
                optimizer.zero_grad()
                tx = torch.eye(4, device=device) + z_offset[0] * z_basis
                mesh = rt.Mesh(base_verts, faces)
                mesh.to_world_left = tx
                scene = rt.Scene()
                scene.add_mesh(mesh)
                scene.build()
                ray = rt.Ray(ray_o, ray_d)
                its = scene.intersect(ray)
                loss = (its.t - target_t).pow(2).sum()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            print(json.dumps({
                "loss_start": losses[0],
                "loss_end": losses[-1],
                "final_z_offset": float(z_offset.item()),
                "converged": losses[-1] < 0.01,
            }))
            """
        )
        self.assertGreater(data["loss_start"], 0.1)
        self.assertTrue(data["converged"],
                        f"Optimization should converge: final loss = {data['loss_end']}")
        self.assertAlmostEqual(data["final_z_offset"], 0.0, places=1)

    def test_optimize_ray_origin_to_hit_target_point(self):
        """Optimize ray origin to make intersection hit a target 3D point."""
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            # Large triangle in the z=0 plane
            mesh = rt.Mesh(
                torch.tensor([[-5.0, -5.0, 0.0],
                              [ 5.0, -5.0, 0.0],
                              [ 0.0,  5.0, 0.0]], device=device),
                torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
            )
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            # Learnable: ray origin xy (start at (1.0, 1.0), target hit at (0.0, 0.0))
            # Ray goes straight down z, so hit point xy = origin xy.
            origin_xy = torch.tensor([1.0, 1.0], device=device, requires_grad=True)
            optimizer = torch.optim.Adam([origin_xy], lr=0.1)
            target_xy = torch.tensor([0.0, 0.0], device=device)

            losses = []
            for step in range(40):
                optimizer.zero_grad()
                o = torch.stack([origin_xy[0], origin_xy[1],
                                 torch.tensor(-1.0, device=device)]).unsqueeze(0)
                d = torch.tensor([[0.0, 0.0, 1.0]], device=device)
                ray = rt.Ray(o, d)
                its = scene.intersect(ray)
                loss = (its.p[0, 0] - target_xy[0]).pow(2) + (its.p[0, 1] - target_xy[1]).pow(2)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            print(json.dumps({
                "loss_start": losses[0],
                "loss_end": losses[-1],
                "final_xy": [float(origin_xy[0].item()), float(origin_xy[1].item())],
                "loss_decreased": losses[-1] < losses[0] * 0.1,
            }))
            """
        )
        self.assertGreater(data["loss_start"], 0.5)
        self.assertTrue(data["loss_decreased"],
                        f"Ray origin optimization should decrease loss: "
                        f"start={data['loss_start']}, end={data['loss_end']}")

    def test_optimize_vertices_multi_ray_depth_loss(self):
        """Optimize vertex positions using multiple rays with depth supervision."""
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)

            # Start with a tilted triangle, optimize to flat at z=0
            verts = torch.tensor([[0.0, 0.0, 0.3],
                                  [1.0, 0.0, -0.2],
                                  [0.0, 1.0, 0.1]], device=device, requires_grad=True)
            optimizer = torch.optim.Adam([verts], lr=0.02)

            # Multiple rays hitting different parts of the triangle
            ray_o = torch.tensor([[0.2, 0.2, -1.0],
                                  [0.5, 0.1, -1.0],
                                  [0.1, 0.5, -1.0],
                                  [0.3, 0.3, -1.0]], device=device)
            ray_d = torch.tensor([[0.0, 0.0, 1.0],
                                  [0.0, 0.0, 1.0],
                                  [0.0, 0.0, 1.0],
                                  [0.0, 0.0, 1.0]], device=device)
            target_t = torch.full((4,), 1.0, device=device)  # all at z=0

            losses = []
            for step in range(100):
                optimizer.zero_grad()
                mesh = rt.Mesh(verts, faces)
                scene = rt.Scene()
                scene.add_mesh(mesh)
                scene.build()
                ray = rt.Ray(ray_o, ray_d)
                its = scene.intersect(ray)
                loss = (its.t - target_t).pow(2).mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            print(json.dumps({
                "loss_start": losses[0],
                "loss_end": losses[-1],
                "final_verts_z": [float(verts[i, 2].item()) for i in range(3)],
                "loss_decreased": losses[-1] < losses[0] * 0.01,
            }))
            """
        )
        self.assertTrue(data["loss_decreased"],
                        f"Multi-ray depth optimization should decrease loss significantly: "
                        f"start={data['loss_start']}, end={data['loss_end']}")
        for z in data["final_verts_z"]:
            self.assertAlmostEqual(z, 0.0, delta=0.15,
                                   msg="All vertices should converge toward z=0")

    def test_render_grad_end_to_end_backward(self):
        """Verify render_grad backward reaches mesh transform parameter."""
        data = run_json_case(
            """
            import json, torch, rayd.torch as rt

            device = "cuda"
            base_verts = torch.tensor([[-0.5, -0.5, 3.0],
                                       [ 0.5, -0.5, 3.0],
                                       [ 0.0,  0.5, 3.0]], device=device)
            faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)

            tx = torch.eye(4, device=device, requires_grad=True)

            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 16
            camera.height = 16
            camera.build()

            mesh = rt.Mesh(base_verts, faces)
            mesh.to_world_left = tx
            scene = rt.Scene()
            scene.add_mesh(mesh)
            scene.build()

            grad_image = camera.render_grad(scene, spp=4)
            loss = grad_image.sum()
            loss.backward()

            print(json.dumps({
                "has_grad": tx.grad is not None,
                "grad_nonzero": bool(
                    tx.grad is not None and tx.grad.abs().sum().item() > 0
                ),
                "grad_shape": list(tx.grad.shape) if tx.grad is not None else [],
                "loss_value": float(loss.item()),
            }))
            """
        )
        self.assertTrue(data["has_grad"],
                        "Transform should receive gradient from render_grad.sum().backward()")
        self.assertTrue(data["grad_nonzero"],
                        "Transform gradient should be non-zero")
        self.assertEqual(data["grad_shape"], [4, 4])


if __name__ == "__main__":
    unittest.main()
