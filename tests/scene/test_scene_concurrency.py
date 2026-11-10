# Copyright Xingyu Chen.
# Tests that scene mutation and deletion are safe against concurrent GIL-free queries and backward passes.

import json
import os
import subprocess
import sys
import textwrap
import unittest

import torch

# Worker threads run the GIL-free AD intersect and its backward (which holds only the integer scene handle) while the
# main thread commits vertex updates, rebuilds, or replaces and drops the scene. Before scenes were shared-owned and
# locked this produced illegal memory accesses and segmentation faults within seconds.
_STRESS = """
import json, sys, threading, time
import torch
import rayd.torch as rt

mode, seconds = sys.argv[1], float(sys.argv[2])
dev = "cuda:0"
n = 128
xs = torch.linspace(0, 1, n, device=dev)
X, Y = torch.meshgrid(xs, xs, indexing="ij")
V = torch.stack([X.flatten(), Y.flatten(), torch.zeros(n * n, device=dev)], 1).contiguous()
i = torch.arange(n - 1, device=dev)
I, J = torch.meshgrid(i, i, indexing="ij")
a = (I * n + J).flatten(); b = a + 1; c = a + n; d = c + 1
F = torch.cat([torch.stack([a, b, c], 1), torch.stack([b, d, c], 1)]).int().contiguous()

def make_scene():
    scene = rt.Scene()
    mesh_id = scene.add_mesh(rt.Mesh(V.clone(), F), dynamic=True)
    scene.build()
    return scene, mesh_id

holder = {"scene": make_scene()}
stop = threading.Event()
errors, completed, refused = [], [0, 0], [0, 0]

def worker(k):
    count = 1 << 14
    o = torch.rand(count, 3, device=dev); o[:, 2] = 1.0
    direction = torch.zeros(count, 3, device=dev); direction[:, 2] = -1.0
    tmax = torch.empty(0, device=dev)
    while not stop.is_set():
        try:
            native = holder["scene"][0]._native_scene
            vertices = V.clone().requires_grad_()
            t = torch.ops.rayd_torch.intersect_ad_t(native, vertices, o, direction, tmax, None)
            del native
            t.sum().backward()
            completed[k] += 1
        except RuntimeError as error:
            message = str(error)
            if "scene geometry changed" in message or "Invalid RayD Torch scene handle" in message:
                refused[k] += 1
                continue
            errors.append(message[:300])
            stop.set()

threads = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
for thread in threads:
    thread.start()
deadline = time.time() + seconds
step = 0
while time.time() < deadline and not stop.is_set():
    scene, mesh_id = holder["scene"]
    if mode == "sync":
        scene.update_mesh_vertices(mesh_id, V + 0.001 * (step % 7))
        scene.sync()
    elif mode == "build":
        scene.build()
    else:
        holder["scene"] = make_scene()
        del scene
    step += 1
    # Leave the workers windows in which a forward and its backward see one geometry, so they complete queries
    # between mutations instead of only being refused.
    time.sleep(0.002)
stop.set()
for thread in threads:
    thread.join()
torch.cuda.synchronize()
report = {"mode": mode, "steps": step, "completed": completed, "refused": refused, "errors": errors[:3]}
print("RESULT " + json.dumps(report))
sys.exit(1 if errors else 0)
"""


# A query is queued on a side stream behind a long device-side spin, so it reads the scene only after the main
# thread's sync() or scene release has been issued on the default stream. Unless the scene orders those writes after
# the queued read, a refit moves the plane under the query (t = 0.5), or the NaN fills overwrite the acceleration
# structures the commit or the release freed before the query traverses them.
_STREAM_ORDER = """
import json, sys
import torch
import rayd.torch as rt
from tests.support.geometry import grid_mesh

mode = sys.argv[1]
dev = torch.device("cuda", 0)
vertices, faces = grid_mesh(dev, 64)
scene = rt.Scene()
mesh_id = scene.add_mesh(rt.Mesh(vertices.clone(), faces), dynamic=True)
scene.build()
count = 1 << 16
generator = torch.Generator(device="cpu").manual_seed(3)
origin = ((torch.rand(count, 3, generator=generator) * 1.8) - 0.9).to(dev)
origin[:, 2] = 1.0
direction = torch.zeros(count, 3, device=dev)
direction[:, 2] = -1.0
moved = vertices + torch.tensor([0.0, 0.0, 0.5], device=dev)
side = torch.cuda.Stream(device=dev)
with torch.cuda.stream(side):
    # The first launch loads modules and blocks the host; only a warm launch is queued behind the spin, and nothing
    # below may wait for the host before the commit is issued.
    scene.intersect(rt.Ray(origin, direction)).t
torch.cuda.synchronize()
with torch.cuda.stream(side):
    torch.cuda._sleep(400_000_000)
    t = scene.intersect(rt.Ray(origin, direction)).t
if mode == "sync":
    scene.update_mesh_vertices(mesh_id, moved)
    scene.sync()
else:
    del scene
junk = [torch.full((1 << shift,), float("nan"), device=dev) for shift in range(8, 24) for _ in range(4)]
torch.cuda.synchronize()
print("RESULT " + json.dumps({"wrong": int((t != 1.0).sum().item())}))
"""

# An edge-mask commit is queued behind a device-side spin and keeps a single edge. In "chain" a second commit on
# another stream and then a query on a third stream follow; in "device" the scene lives on cuda:1 and the query runs
# on that device's default stream while cuda:0 is current. Blocks of the mask's size are zeroed first, so a query
# that is not ordered after the mask commit reads an all-zero mask and finds no edge.
_COMMIT_ORDER = """
import json, sys
import torch
import rayd.torch as rt
from tests.support.geometry import grid_mesh

mode = sys.argv[1]
dev = torch.device("cuda", 1 if mode == "device" else 0)
torch.cuda.set_device(dev)
vertices, faces = grid_mesh(dev, 8)
scene = rt.Scene()
mesh_id = scene.add_mesh(rt.Mesh(vertices.clone(), faces), dynamic=True)
scene.build()
point = torch.tensor([[-1.0, -1.0, 0.5]], device=dev)
edge_count = scene.edge_mask().shape[0]
target = edge_count // 2
mask = torch.zeros(edge_count, dtype=torch.bool, device=dev)
mask[target] = True
moved = vertices.clone()
writer, other, reader = (torch.cuda.Stream(device=dev) for _ in range(3))
# Run every step once on its stream first: module loads, fresh allocator segments and the pinned read-back buffer
# of sync() would otherwise block the host and hide the ordering under test.
for stream in (writer, other, reader):
    with torch.cuda.stream(stream):
        scene.set_edge_mask(torch.ones(edge_count, dtype=torch.bool, device=dev))
        scene.update_mesh_vertices(mesh_id, moved)
        scene.sync()
        scene.nearest_edge(point).global_edge_id
torch.cuda.synchronize(dev)
with torch.cuda.stream(writer):
    zeros = [torch.zeros(edge_count, dtype=torch.uint8, device=dev) for _ in range(2)]
torch.cuda.synchronize(dev)
del zeros
with torch.cuda.stream(writer):
    torch.cuda._sleep(400_000_000)
    scene.set_edge_mask(mask)
if mode == "chain":
    with torch.cuda.stream(other):
        scene.update_mesh_vertices(mesh_id, moved)
        scene.sync()
    with torch.cuda.stream(reader):
        found = scene.nearest_edge(point).global_edge_id
else:
    torch.cuda.set_device(0)
    found = scene.nearest_edge(point).global_edge_id
torch.cuda.synchronize(dev)
print("RESULT " + json.dumps({"wrong": int((found != target).sum().item())}))
"""


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class SceneStreamOrderTests(unittest.TestCase):
    def _run(self, mode: str, script: str = _STREAM_ORDER) -> None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script), mode],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            cwd=root,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-3000:])
        line = next((row for row in result.stdout.splitlines() if row.startswith("RESULT ")), None)
        self.assertIsNotNone(line, result.stdout[-2000:])
        self.assertEqual(json.loads(line[len("RESULT ") :])["wrong"], 0)

    def test_sync_waits_for_a_query_queued_on_another_stream(self):
        self._run("sync")

    def test_scene_release_waits_for_a_query_queued_on_another_stream(self):
        self._run("delete")

    def test_a_commit_and_its_readers_wait_for_an_earlier_commit_on_another_stream(self):
        self._run("chain", _COMMIT_ORDER)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices are required")
    def test_default_stream_query_on_a_non_current_device_waits_for_the_commit(self):
        self._run("device", _COMMIT_ORDER)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class SceneConcurrencyTests(unittest.TestCase):
    def _stress(self, mode: str) -> None:
        seconds = os.environ.get("RAYD_SCENE_CONCURRENCY_SECONDS", "3")
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_STRESS), mode, seconds],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-3000:])
        line = next((row for row in result.stdout.splitlines() if row.startswith("RESULT ")), None)
        self.assertIsNotNone(line, result.stdout[-2000:])
        report = json.loads(line[len("RESULT ") :])
        self.assertEqual(report["errors"], [])
        self.assertGreater(report["steps"], 0, report)
        # Each worker must have finished whole forward and backward passes while the scene was being mutated, so
        # the run exercised the race instead of only refusing every query.
        for worker, done in enumerate(report["completed"]):
            self.assertGreater(done, 0, f"worker {worker} completed no query: {report}")

    def test_sync_during_gil_free_queries_and_backward(self):
        self._stress("sync")

    def test_rebuild_during_gil_free_queries_and_backward(self):
        self._stress("build")

    def test_scene_deletion_during_backward(self):
        self._stress("delete")


if __name__ == "__main__":
    unittest.main()
