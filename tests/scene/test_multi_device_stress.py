"""Phase 1 acceptance tests for concurrent multi-device Torch execution.

Phase 0 proved that a single host thread can drive any device correctly. This
module raises the bar to the Phase 1 criteria: two host threads, each owning one
device and its own non-default stream, must produce the same bits a
single-threaded single-device run produces; a non-zero device must survive an
OptiX cold create as the first CUDA work of a process; and two devices driven
concurrently must actually overlap instead of serializing behind a shared lock.

Nothing here shares a tensor between threads. Every worker builds its own scene,
its own inputs, and its own stream, so a difference can only come from device or
stream handling inside the backend.
"""

import contextlib
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import torch
import rayd.torch as rt


# The correctness workload stays small: it stores every iteration's result on the
# host for a bitwise comparison, and the contract it checks is device handling,
# not throughput.
_STRESS_CELLS = 8
_STRESS_RAYS = 1 << 13
_STRESS_POINTS = 1 << 12
_STRESS_ITERATIONS = 24

# The overlap workload is sized so GPU time dominates the host time spent
# enqueuing it; otherwise the measurement would report Python contention rather
# than cross-device serialization.
_TIMED_CELLS = 32
_TIMED_RAYS = 1 << 20
_TIMED_POINTS = 1 << 16
_TIMED_ITERATIONS = 20
_TIMED_REPEATS = 5

# Measured on this repository's 2x RTX A6000 machine: the best-of-5 concurrent
# ratio sits at 0.52 with individual trials as high as 0.78, so 0.80 is the
# strictest threshold that repeated runs cleared. A shared launch lock would push
# the ratio to ~1.0, which is what this bound exists to catch.
_OVERLAP_RATIO = 0.80

# Each iteration shifts the ray origins by an exactly representable step so no
# two iterations are the same launch, and so the sequence stays bitwise
# reproducible between the reference run and the threaded run.
_RAY_STEP = 0.001953125

_BARRIER_TIMEOUT = 120.0

# `rayd.torch` is imported from the thin root `python` frontend; the cold-create
# subprocess needs the same directory on its `PYTHONPATH`.
_PACKAGE_ROOT = Path(rt.__file__).resolve().parents[2]


def _grid_mesh(device: torch.device, cells: int, span: float = 2.0):
    """Deterministic z=0 triangle grid; identical bits on every CUDA device."""
    axis = torch.linspace(-0.5 * span, 0.5 * span, cells + 1, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    flat_x = x.reshape(-1)
    vertices = torch.stack((flat_x, y.reshape(-1), torch.zeros_like(flat_x)), dim=1)
    index = torch.arange((cells + 1) * (cells + 1), dtype=torch.int32).reshape(
        cells + 1, cells + 1
    )
    a = index[:-1, :-1].reshape(-1)
    b = index[:-1, 1:].reshape(-1)
    c = index[1:, :-1].reshape(-1)
    d = index[1:, 1:].reshape(-1)
    faces = torch.cat((torch.stack((a, b, c), dim=1), torch.stack((b, d, c), dim=1)))
    return vertices.contiguous().to(device), faces.contiguous().to(device)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Host copy compared bit-for-bit, so NaN, -0.0, and inf all compare exactly."""
    host = tensor.detach().contiguous().cpu()
    if host.dtype == torch.float32:
        return host.view(torch.int32)
    if host.dtype == torch.float64:
        return host.view(torch.int64)
    return host


def _spread(count: int, stride: int, period: int) -> torch.Tensor:
    """Integer-exact scatter in [-0.5, 0.5); no transcendental, so no ULP doubt.

    Every thread rebuilds its own inputs instead of receiving them, so the host
    construction has to be reproducible bit for bit across calls and threads.
    """
    index = torch.arange(count, dtype=torch.int64)
    return ((index * stride) % period).to(torch.float32) / float(period) - 0.5


def _query_inputs(device: torch.device, rays: int, points: int):
    """Per-thread ray and point batches; never shared, always the same bits."""
    u = _spread(rays, 37, 211)
    v = _spread(rays, 61, 197)
    origin = torch.stack(
        (1.8 * u, 1.8 * v, torch.full_like(u, -1.0)), dim=1
    ).contiguous()
    direction = torch.stack((0.25 * v, 0.25 * u, torch.ones_like(u)), dim=1)
    direction = (direction / direction.norm(dim=1, keepdim=True)).contiguous()

    pu = _spread(points, 29, 173)
    pv = _spread(points, 53, 149)
    query = torch.stack((1.6 * pu, 1.6 * pv, 0.25 + 0.5 * pu), dim=1).contiguous()
    return origin.to(device), direction.to(device), query.to(device)


def _build_scene(index: int, cells: int) -> rt.Scene:
    """Scene.build() requires its own device to be current; nothing else does."""
    vertices, faces = _grid_mesh(torch.device("cuda", index), cells)
    with torch.cuda.device(index):
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(vertices, faces))
        scene.build()
    return scene


class _DeviceWorkload:
    """Everything one device needs, allocated by the thread that owns it.

    `own_stream=False` keeps the loop on the device's default stream, which is
    how the single-threaded reference runs; the threaded workers take their own
    stream, so the comparison covers stream choice as well as thread ownership.
    """

    def __init__(
        self, index: int, cells: int, rays: int, points: int, own_stream: bool = True
    ) -> None:
        self.index = index
        self.scene = _build_scene(index, cells)
        device = torch.device("cuda", index)
        self.ray_o, self.ray_d, self.points = _query_inputs(device, rays, points)
        self.stream = torch.cuda.Stream(device=device) if own_stream else None

    def _stream_context(self):
        if self.stream is None:
            return contextlib.nullcontext()
        return torch.cuda.stream(self.stream)

    def _drain(self) -> None:
        if self.stream is None:
            torch.cuda.synchronize(self.index)
        else:
            self.stream.synchronize()

    def _iteration(self, step: int) -> dict[str, torch.Tensor]:
        ray = rt.Ray(self.ray_o + step * _RAY_STEP, self.ray_d)
        intersection = self.scene.intersect(ray)
        nearest = self.scene.nearest_edge(self.points)
        chain = self.scene.trace_reflections(ray, max_bounces=2)
        return {
            "intersect.t": intersection.t,
            "intersect.p": intersection.p,
            "intersect.n": intersection.n,
            "intersect.prim_id": intersection.prim_id,
            "nearest_edge.distance": nearest.distance,
            "nearest_edge.edge_point": nearest.edge_point,
            "nearest_edge.edge_t": nearest.edge_t,
            "nearest_edge.global_edge_id": nearest.global_edge_id,
            "trace_reflections.valid": chain.valid,
            "trace_reflections.t": chain.t,
            "trace_reflections.prim_ids": chain.prim_ids,
        }

    def collect(self, iterations: int):
        """Run the loop on the owned stream and read the results back as host bits.

        The host copies happen only after the stream drains, so nothing observes
        a result the device has not written yet.
        """
        device_results = []
        with self._stream_context():
            for step in range(iterations):
                device_results.append(self._iteration(step))
        self._drain()

        devices = {
            value.device.index
            for iteration in device_results
            for value in iteration.values()
        }
        host_results = [
            {name: _bits(value) for name, value in iteration.items()}
            for iteration in device_results
        ]
        return host_results, devices

    def run_untimed(self, iterations: int) -> None:
        """Same loop without result retention, for the overlap measurement."""
        with self._stream_context():
            for step in range(iterations):
                self._iteration(step)
        self._drain()


def _cold_create_script(device_index: int) -> str:
    """Body of the cold-create subprocess: all CUDA work lands on one device."""
    return f'''\
import torch
import rayd.torch as rt

DEVICE_INDEX = {device_index}

count = torch.cuda.device_count()
assert count > DEVICE_INDEX, f"need more than {{DEVICE_INDEX}} devices, saw {{count}}"

device = torch.device("cuda", DEVICE_INDEX)

axis = torch.linspace(-1.0, 1.0, 5, dtype=torch.float32)
y, x = torch.meshgrid(axis, axis, indexing="ij")
flat_x = x.reshape(-1)
vertices = torch.stack((flat_x, y.reshape(-1), torch.zeros_like(flat_x)), dim=1)
index = torch.arange(25, dtype=torch.int32).reshape(5, 5)
a = index[:-1, :-1].reshape(-1)
b = index[:-1, 1:].reshape(-1)
c = index[1:, :-1].reshape(-1)
d = index[1:, 1:].reshape(-1)
faces = torch.cat((torch.stack((a, b, c), dim=1), torch.stack((b, d, c), dim=1)))

# First CUDA work of this process: the scene, its OptiX context, and its
# pipelines are all created cold on a non-zero device.
with torch.cuda.device(DEVICE_INDEX):
    scene = rt.Scene()
    mesh = rt.Mesh(vertices.contiguous().to(device), faces.contiguous().to(device))
    scene.add_mesh(mesh)
    scene.build()

    ray_o = torch.tensor(
        [[-0.6, -0.6, -1.0], [0.1, -0.3, -1.0], [0.4, 0.5, -1.0], [-0.2, 0.7, -1.0]],
        dtype=torch.float32,
        device=device,
    )
    ray_d = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32, device=device)
    points = torch.tensor(
        [[0.0, 0.0, 0.25], [0.9, -0.9, 0.1], [-0.5, 0.5, -0.3], [0.3, 0.2, 0.5]],
        dtype=torch.float32,
        device=device,
    )

    intersection = scene.intersect(rt.Ray(ray_o, ray_d))
    chain = scene.trace_reflections(rt.Ray(ray_o, ray_d), max_bounces=2)
    nearest = scene.nearest_edge(points)

torch.cuda.synchronize(DEVICE_INDEX)

for name, value in (
    ("intersect.t", intersection.t),
    ("trace_reflections.t", chain.t),
    ("nearest_edge.distance", nearest.distance),
):
    assert value.device.index == DEVICE_INDEX, f"{{name}} left cuda:{{DEVICE_INDEX}}"

# The mesh is a z=0 plate spanning the four +z rays, so every ray hits it once.
assert bool(torch.isfinite(intersection.t).all()), "cold intersect produced no hit"
assert bool((intersection.t > 0.0).all()), "cold intersect produced a degenerate t"
assert bool(chain.valid[:, 0].all()), "cold trace_reflections lost the first bounce"
assert bool(torch.equal(chain.t[:, 0], intersection.t)), "cold reflection t disagrees"
assert bool(torch.isfinite(nearest.distance).all()), "cold nearest_edge found no edge"
assert bool((nearest.distance >= 0.0).all()), "cold nearest_edge went negative"
assert bool((nearest.global_edge_id >= 0).all()), "cold nearest_edge has no edge id"

# Nothing may have been staged on device 0 on the way to device {device_index}.
assert torch.cuda.memory_allocated(0) == 0, "cold run allocated on cuda:0"

print("cold-create OK on cuda:%d" % DEVICE_INDEX)
'''


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "two CUDA devices are required",
)
class MultiDeviceStressTests(unittest.TestCase):
    def setUp(self) -> None:
        self._entry_device = torch.cuda.current_device()
        torch.cuda.set_device(0)

    def tearDown(self) -> None:
        torch.cuda.set_device(self._entry_device)

    def assert_same_bits(self, reference, observed, context: str) -> None:
        self.assertEqual(len(reference), len(observed), f"{context}: iteration count")
        for step, (left, right) in enumerate(zip(reference, observed)):
            self.assertEqual(
                sorted(left), sorted(right), f"{context}: iteration {step}"
            )
            for name, value in left.items():
                other = right[name]
                self.assertEqual(
                    value.dtype, other.dtype, f"{context}: {name} dtype at {step}"
                )
                self.assertEqual(
                    value.shape, other.shape, f"{context}: {name} shape at {step}"
                )
                self.assertTrue(
                    torch.equal(value, other),
                    f"{context}: {name} is not bitwise equal at iteration {step}",
                )

    def assert_non_degenerate(self, results, context: str) -> None:
        """Guard against a vacuous pass: equal bits must be interesting bits.

        The rays are aimed at the plate and the query points sit beside it, so a
        run that missed everything would compare equal while proving nothing.
        """
        first = results[0]
        t = first["intersect.t"].view(torch.float32)
        hit = torch.isfinite(t)
        self.assertGreater(
            hit.float().mean().item(), 0.9, f"{context}: intersect mostly missed"
        )
        self.assertTrue(
            bool((t[hit] > 0.0).all()), f"{context}: intersect returned a degenerate t"
        )
        self.assertGreater(
            first["intersect.prim_id"][hit].unique().numel(),
            1,
            f"{context}: intersect resolved a single primitive",
        )
        self.assertGreater(
            first["trace_reflections.valid"][:, 0].float().mean().item(),
            0.9,
            f"{context}: trace_reflections lost the first bounce",
        )
        distance = first["nearest_edge.distance"].view(torch.float32)
        self.assertTrue(
            bool(torch.isfinite(distance).all()),
            f"{context}: nearest_edge found no edge",
        )
        self.assertTrue(
            bool((first["nearest_edge.global_edge_id"] >= 0).all()),
            f"{context}: nearest_edge returned no edge id",
        )

    def run_two_threads(self, body, concurrent: bool) -> None:
        """Drive both devices from their own threads, concurrently or one at a time.

        `body(index)` owns everything it touches. Concurrent runs meet at a
        barrier so neither thread can finish its loop before the other starts.
        """
        errors: dict[int, BaseException] = {}
        barrier = threading.Barrier(2) if concurrent else None

        def worker(index: int) -> None:
            try:
                torch.cuda.set_device(index)
                body(index, barrier)
            except BaseException as error:  # re-raised on the main thread
                errors[index] = error
                if barrier is not None:
                    barrier.abort()

        if concurrent:
            threads = [
                threading.Thread(target=worker, args=(index,)) for index in (0, 1)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for index in (0, 1):
                thread = threading.Thread(target=worker, args=(index,))
                thread.start()
                thread.join()

        if errors:
            raise AssertionError(
                "worker thread failed: "
                + "; ".join(
                    f"cuda:{index}: {error!r}" for index, error in errors.items()
                )
            )

    def test_two_threads_on_two_devices_reproduce_single_device_results(self):
        reference = {}
        for index in (0, 1):
            torch.cuda.set_device(index)
            workload = _DeviceWorkload(
                index, _STRESS_CELLS, _STRESS_RAYS, _STRESS_POINTS, own_stream=False
            )
            results, devices = workload.collect(_STRESS_ITERATIONS)
            self.assertEqual(
                devices, {index}, f"cuda:{index} reference left its device"
            )
            self.assert_non_degenerate(results, f"cuda:{index} reference")
            reference[index] = results
        torch.cuda.set_device(0)

        observed: dict[int, list] = {}
        seen: dict[int, set] = {}

        def body(index: int, barrier) -> None:
            # Built inside the thread: the two threads share no tensor, no scene,
            # and no stream, so only backend device handling is under test.
            workload = _DeviceWorkload(
                index, _STRESS_CELLS, _STRESS_RAYS, _STRESS_POINTS
            )
            barrier.wait(timeout=_BARRIER_TIMEOUT)
            observed[index], seen[index] = workload.collect(_STRESS_ITERATIONS)

        self.run_two_threads(body, concurrent=True)

        self.assertEqual(sorted(observed), [0, 1])
        for index in (0, 1):
            self.assertEqual(
                seen[index], {index}, f"cuda:{index} thread left its device"
            )
            self.assert_same_bits(
                reference[index], observed[index], f"cuda:{index} concurrent thread"
            )
        self.assertEqual(
            torch.cuda.current_device(), 0, "a worker thread leaked the main device"
        )

    def test_cold_create_on_a_non_zero_device_in_a_fresh_process(self):
        env = os.environ.copy()
        # The subprocess must see the real device topology, not an inherited mask
        # that would rename cuda:1 or hide it entirely.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(_PACKAGE_ROOT), env.get("PYTHONPATH", "")) if part
        )

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "cold_create_device_one.py"
            script.write_text(_cold_create_script(1), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                timeout=600,
                check=False,
            )

        combined = result.stdout + "\n" + result.stderr
        self.assertEqual(
            result.returncode, 0, f"cold-create subprocess failed.\n{combined}"
        )
        self.assertIn("cold-create OK on cuda:1", result.stdout, combined)
        self.assertNotIn("optixPipelineCreate", combined)

    def test_two_devices_overlap_instead_of_serializing(self):
        """Coarse gate on Phase 1's "no cross-device serialization" criterion."""
        prepared: dict[int, _DeviceWorkload] = {}
        for index in (0, 1):
            torch.cuda.set_device(index)
            prepared[index] = _DeviceWorkload(
                index, _TIMED_CELLS, _TIMED_RAYS, _TIMED_POINTS
            )
        torch.cuda.set_device(0)

        def body(index: int, barrier) -> None:
            if barrier is not None:
                barrier.wait(timeout=_BARRIER_TIMEOUT)
            prepared[index].run_untimed(_TIMED_ITERATIONS)

        # Warm both devices so pipeline creation and allocator growth stay out of
        # the measurement.
        self.run_two_threads(body, concurrent=False)

        def measure(concurrent: bool) -> float:
            start = time.perf_counter()
            self.run_two_threads(body, concurrent=concurrent)
            return time.perf_counter() - start

        serialized = min(measure(False) for _ in range(_TIMED_REPEATS))
        concurrent = min(measure(True) for _ in range(_TIMED_REPEATS))

        self.assertGreater(
            serialized, 0.02, "workload too small to time; raise the iteration count"
        )
        self.assertLess(
            concurrent,
            _OVERLAP_RATIO * serialized,
            "two devices did not overlap: "
            f"{concurrent * 1e3:.1f} ms concurrent versus "
            f"{serialized * 1e3:.1f} ms serialized "
            f"(ratio {concurrent / serialized:.3f}, bound {_OVERLAP_RATIO})",
        )


if __name__ == "__main__":
    unittest.main()
