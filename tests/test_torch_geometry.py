import importlib
import importlib.abc
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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
            scene.configure()

            ray = rt.RayDetached(
                torch.tensor([[0.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)

            rays = rt.RayDetached(
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
            scene.configure()

            points = torch.tensor([[0.5, -0.2, 0.0],
                                   [0.5, 0.5, 0.0]], device=device)
            point_res = scene.nearest_edge(points, torch.tensor([True, False], device=device))

            ray = rt.RayDetached(
                torch.tensor([[0.5, 0.5, 1.0]], device=device),
                torch.tensor([[0.0, 0.0, -1.0]], device=device),
                torch.tensor([2.0], device=device),
            )
            ray_res = scene.nearest_edge(ray)

            print(json.dumps({
                "point_valid": [bool(v) for v in point_res.is_valid().tolist()],
                "point_shape": [int(v) for v in point_res.shape_id.tolist()],
                "point_edge": [int(v) for v in point_res.edge_id.tolist()],
                "point_distance": float(point_res.distance[0].item()),
                "point_tensor_shape": list(point_res.edge_point.shape),
                "ray_valid": bool(ray_res.is_valid()[0].item()),
                "ray_shape": int(ray_res.shape_id[0].item()),
                "ray_edge": int(ray_res.edge_id[0].item()),
                "ray_distance": float(ray_res.distance[0].item()),
            }))
            """
        )

        self.assertEqual(data["point_valid"], [True, False])
        self.assertEqual(data["point_shape"][1], -1)
        self.assertEqual(data["point_edge"][1], -1)
        self.assertGreaterEqual(data["point_distance"], 0.0)
        self.assertEqual(data["point_tensor_shape"], [2, 3])
        self.assertTrue(data["ray_valid"])
        self.assertEqual(data["ray_shape"], 0)
        self.assertGreaterEqual(data["ray_edge"], 0)
        self.assertGreaterEqual(data["ray_distance"], 0.0)

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
            mesh.configure()

            scene = rt.Scene()
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            scene.configure()

            scene.update_mesh_vertices(
                mesh_id,
                torch.tensor([[2.0, 0.0, 0.0],
                              [3.0, 0.0, 0.0],
                              [2.0, 1.0, 0.0]], device=device),
            )
            pending_error = False
            try:
                scene.intersect(
                    rt.RayDetached(
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
            scene.commit_updates()
            ray = rt.RayDetached(
                torch.tensor([[2.25, 0.25, -1.0]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            its = scene.intersect(ray)
            profile = scene.last_commit_profile

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
                "profile_total_ms": float(profile.total_ms),
                "profile_updated_meshes": int(profile.updated_meshes),
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
        self.assertGreaterEqual(data["profile_total_ms"], 0.0)
        self.assertGreaterEqual(data["profile_updated_meshes"], 1)

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
            scene.configure()

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
            scene2.configure()
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
            scene.configure()

            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 16
            camera.height = 12
            camera.configure()
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

        self.assertEqual(data["ray_type"], "RayDetached")
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
            scene.configure()

            camera = rt.Camera(45.0, 1e-4, 1e4)
            camera.width = 32
            camera.height = 32
            camera.configure()
            camera.prepare_edges(scene)

            scene.update_mesh_vertices(
                mesh_id,
                torch.tensor([[-0.25, -0.5, 3.0],
                              [ 0.75, -0.5, 3.0],
                              [ 0.25,  0.5, 3.0]], device=device),
            )
            scene.commit_updates()

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


if __name__ == "__main__":
    unittest.main()
