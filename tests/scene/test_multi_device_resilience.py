# Copyright Xingyu Chen.
# Tests multi device resilience.

"""Failure-atomicity gates for replicated multi-device execution."""

import ast
from contextlib import nullcontext
import inspect
from types import SimpleNamespace
import textwrap
import unittest
from unittest import mock

import torch
import rayd.torch as rt

from rayd._impl.multi import _ReplicatedScene


class _FakeTensor:
    def to(self, _device):
        return self


class _FakeReplica:
    def __init__(self, *, fail: str | None = None, advance: bool = True):
        self.version = 7
        self.fail = fail
        self.advance = advance
        self.calls = []
        self._native_scene = object()

    def _call(self, operation):
        self.calls.append(operation)
        if self.fail == operation:
            raise RuntimeError(f"injected {operation} failure")
        if self.advance:
            self.version += 1

    def update_mesh_vertices(self, _mesh_id, _positions):
        self._call("update_mesh_vertices")

    def sync(self):
        self._call("sync")

    def set_edge_mask(self, _mask):
        self._call("set_edge_mask")


def _fake_layer(*replicas):
    layer = object.__new__(_ReplicatedScene)
    layer.devices = tuple(SimpleNamespace(index=index) for index in range(len(replicas)))
    layer._replicas = tuple(replicas)
    layer._poisoned = None
    return layer


class NoGpuResilienceTests(unittest.TestCase):
    def test_chunk_executor_has_an_unconditional_stream_exit_edge(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(_ReplicatedScene._run_chunked)))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        guarded = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Try)
            and any(
                isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "_exit"
                for statement in node.finalbody
                for call in ast.walk(statement)
            )
        ]
        self.assertTrue(guarded, "_run_chunked must join private streams in finally")
        finalbody = guarded[0].finalbody
        self.assertTrue(
            any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute) and target.attr == "_active_stream" for target in node.targets
                )
                and isinstance(node.value, ast.Constant)
                and node.value.value is None
                for statement in finalbody
                for node in ast.walk(statement)
            ),
            "_run_chunked must clear _active_stream on every exit",
        )

    def test_every_partial_broadcast_poison_fails_closed(self):
        for operation in ("update_mesh_vertices", "sync", "set_edge_mask"):
            with self.subTest(operation=operation):
                first = _FakeReplica()
                second = _FakeReplica(fail=operation)
                layer = _fake_layer(first, second)
                arguments = (
                    (3, _FakeTensor())
                    if operation == "update_mesh_vertices"
                    else ((_FakeTensor(),) if operation == "set_edge_mask" else ())
                )
                with mock.patch.object(torch.cuda, "device", return_value=nullcontext()):
                    with self.assertRaisesRegex(RuntimeError, f"injected {operation} failure"):
                        getattr(layer, operation)(*arguments)

                self.assertTrue(layer.is_poisoned)
                self.assertEqual(first.calls, [operation])
                self.assertEqual(second.calls, [operation])
                for access in (layer.master, layer.master_native_scene, layer.sync, lambda: layer._shards(4)):
                    with self.assertRaisesRegex(RuntimeError, "Call Scene.build"):
                        access()

    def test_version_divergence_also_poisons_the_replica_set(self):
        layer = _fake_layer(_FakeReplica(), _FakeReplica(advance=False))
        with mock.patch.object(torch.cuda, "device", return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "replicas diverged"):
                layer.sync()
        self.assertTrue(layer.is_poisoned)
        with self.assertRaisesRegex(RuntimeError, "Call Scene.build"):
            layer.master()


def _dynamic_scene(options=None):
    device = torch.device("cuda", 0)
    vertices = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    mesh = rt.Mesh(vertices, faces)
    scene = rt.Scene(devices=[0, 1], options=options or rt.MultiDeviceOptions(warm_up=False, min_rays_per_device=1))
    scene.add_mesh(mesh, dynamic=True)
    scene.build()
    return scene, mesh


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices are required")
class GpuFailureInjectionTests(unittest.TestCase):
    def test_partial_vertex_broadcast_requires_a_complete_rebuild(self):
        scene, mesh = _dynamic_scene()
        old_vertices = mesh.vertices
        remote = scene._multi._replicas[1]

        def fail_update(_mesh_id, _positions):
            raise RuntimeError("injected remote update failure")

        remote.update_mesh_vertices = fail_update
        with self.assertRaisesRegex(RuntimeError, "injected remote update failure"):
            scene.update_mesh_vertices(0, old_vertices + 0.25)

        self.assertIs(mesh.vertices, old_vertices)
        self.assertFalse(scene.is_ready())
        self.assertTrue(scene.has_pending_updates())
        with self.assertRaisesRegex(RuntimeError, "Call Scene.build"):
            _ = scene.version
        with self.assertRaisesRegex(RuntimeError, "Call Scene.build"):
            scene.sync()

        scene.build()
        self.assertTrue(scene.is_ready())
        self.assertFalse(scene.has_pending_updates())
        self.assertIsInstance(scene.version, int)

    def test_offload_exception_joins_streams_before_mutation(self):
        def fail_offload(_start, _result):
            raise RuntimeError("injected offload failure")

        options = rt.MultiDeviceOptions(warm_up=False, min_rays_per_device=1, chunk_rays=8, offload=fail_offload)
        scene, mesh = _dynamic_scene(options)
        origins = torch.zeros((64, 3), dtype=torch.float32, device="cuda:0")
        origins[:, 2] = -1.0
        directions = torch.zeros_like(origins)
        directions[:, 2] = 1.0

        with self.assertRaisesRegex(RuntimeError, "injected offload failure"):
            scene.intersect(rt.Ray(origins, directions))
        self.assertIsNone(scene._multi._active_stream)

        # If the private compute streams were not handed back, this immediate
        # mutation could race an already-submitted traversal.
        object.__setattr__(scene._multi.options, "offload", None)
        scene.update_mesh_vertices(0, mesh.vertices + 0.125)
        scene.sync()
        torch.cuda.synchronize()
        self.assertTrue(scene.is_ready())


if __name__ == "__main__":
    unittest.main()
