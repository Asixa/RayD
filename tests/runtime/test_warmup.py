# Copyright Xingyu Chen.
# Tests warmup.

import threading
import unittest
from unittest import mock

import torch

import rayd.torch as rt
from rayd._impl import runtime as _warmup


class WarmUpImportSafetyTests(unittest.TestCase):
    """Everything here must hold on a CPU-only machine: no CUDA is touched."""

    def test_module_is_not_public_api(self):
        self.assertNotIn("_warmup", rt.__all__)
        self.assertNotIn("warm_up_devices", rt.__all__)
        self.assertFalse(hasattr(rt, "warm_up_devices"))

    def test_no_devices_is_a_no_op(self):
        self.assertEqual(_warmup.warm_up_devices([]), {})

    def test_a_bare_device_index_is_rejected(self):
        with self.assertRaises(TypeError):
            _warmup.warm_up_devices(0)

    def test_a_bare_op_name_is_rejected(self):
        with self.assertRaises(TypeError):
            _warmup.warm_up_devices([], ops="intersect")

    def test_unknown_ops_are_rejected(self):
        with self.assertRaises(ValueError):
            _warmup.warm_up_devices([], ops=("intersect", "teleport"))

    def test_non_cuda_devices_are_rejected(self):
        with self.assertRaises(ValueError):
            _warmup.warm_up_devices([torch.device("cpu")])


@unittest.skipUnless(torch.cuda.is_available(), "a CUDA device is required")
class WarmUpSingleDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._entry_device = torch.cuda.current_device()

    def tearDown(self) -> None:
        torch.cuda.set_device(self._entry_device)

    def test_warm_up_device_zero(self):
        elapsed = _warmup.warm_up_devices([0])

        self.assertEqual(list(elapsed), [0])
        self.assertGreater(elapsed[0], 0.0)

    def test_warm_up_accepts_torch_devices(self):
        elapsed = _warmup.warm_up_devices([torch.device("cuda", 0)])

        self.assertEqual(list(elapsed), [0])

    def test_warm_up_leaves_the_calling_thread_device_untouched(self):
        torch.cuda.set_device(0)
        _warmup.warm_up_devices([torch.cuda.device_count() - 1])

        self.assertEqual(torch.cuda.current_device(), 0)

    def test_each_op_can_be_warmed_on_its_own(self):
        for name in _warmup.DEFAULT_OPS:
            with self.subTest(op=name):
                elapsed = _warmup.warm_up_devices([0], ops=(name,))
                self.assertEqual(list(elapsed), [0])

    def test_an_invalid_device_index_is_rejected(self):
        with self.assertRaises(ValueError):
            _warmup.warm_up_devices([torch.cuda.device_count()])

    def test_duplicate_devices_are_rejected(self):
        with self.assertRaises(ValueError):
            _warmup.warm_up_devices([0, 0])

    def test_a_worker_failure_names_its_device(self):
        def explode(index, ops):
            raise ValueError("synthetic warm-up failure")

        with mock.patch.object(_warmup, "_warm_up_device", explode):
            with self.assertRaises(RuntimeError) as caught:
                _warmup.warm_up_devices([0])

        self.assertIn("cuda:0", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_an_empty_op_list_does_no_device_work(self):
        self.assertEqual(_warmup.warm_up_devices([0], ops=()), {})

    def test_the_warmed_scene_is_not_retained(self):
        # Measured from the second call on, so one-time process-wide caches are
        # not charged to the steady state; the throwaway scene and its query
        # tensors must be dropped by the worker on every call.
        _warmup.warm_up_devices([0])
        before = torch.cuda.memory_allocated(0)
        _warmup.warm_up_devices([0])
        after = torch.cuda.memory_allocated(0)

        self.assertLessEqual(after, before + 4096)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices are required")
class WarmUpMultiDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._entry_device = torch.cuda.current_device()
        torch.cuda.set_device(0)

    def tearDown(self) -> None:
        torch.cuda.set_device(self._entry_device)

    def test_both_devices_are_warmed(self):
        elapsed = _warmup.warm_up_devices([0, 1])

        self.assertEqual(list(elapsed), [0, 1])
        for index, seconds in elapsed.items():
            self.assertGreater(seconds, 0.0, f"cuda:{index} reported no work")
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_warmed_devices_still_answer_a_real_query(self):
        _warmup.warm_up_devices([0, 1])

        for index in (0, 1):
            with self.subTest(device=index):
                device = torch.device("cuda", index)
                vertices = torch.tensor(
                    [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device=device
                )
                faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
                with torch.cuda.device(index):
                    scene = rt.Scene()
                    scene.add_mesh(rt.Mesh(vertices, faces))
                    scene.build()
                ray = rt.Ray(
                    torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device),
                    torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device),
                )
                hit = scene.intersect(ray)
                self.assertEqual(hit.t.device.index, index)
                self.assertAlmostEqual(float(hit.t[0]), 1.0, places=5)

    def test_device_work_overlaps(self):
        """Nothing serializes the per-device work any more.

        The real warm-up runs here; only the observation is injected. The
        helper exists to overlap the per-device OptiX build, so two devices
        must be inside `_run_op` at the same time at least once. This is also
        the regression guard for the native concurrent-driving deadlock: it
        hangs rather than fails if that ever comes back.
        """
        real_run_op = _warmup._run_op
        lock = threading.Lock()
        entered = threading.Barrier(2, timeout=120.0)
        active: list[int] = []
        overlaps: list[tuple[int, ...]] = []

        def tracked(scene, name, device):
            with lock:
                active.append(device.index)
                if len(active) > 1:
                    overlaps.append(tuple(active))
            try:
                # Both workers must reach a real op before either may leave it,
                # so the overlap is forced instead of raced for.
                entered.wait()
                real_run_op(scene, name, device)
            finally:
                with lock:
                    active.remove(device.index)

        with mock.patch.object(_warmup, "_run_op", tracked):
            elapsed = _warmup.warm_up_devices([0, 1], ops=("intersect",))

        self.assertEqual(sorted(elapsed), [0, 1])
        self.assertNotEqual(overlaps, [], "device work never overlapped")

    def test_workers_are_dispatched_concurrently(self):
        """The executor itself must dispatch every device at once.

        One live worker thread per device, all started before any of them
        finishes. A rendezvous barrier is the assertion — dispatching the
        devices one after another would break it instead.
        """
        barrier = threading.Barrier(2, timeout=30.0)
        observed: dict[int, str] = {}
        lock = threading.Lock()

        def rendezvous(index, ops):
            barrier.wait()
            with lock:
                observed[index] = threading.current_thread().name
            return 0.125

        with mock.patch.object(_warmup, "_warm_up_device", rendezvous):
            elapsed = _warmup.warm_up_devices([0, 1])

        self.assertEqual(elapsed, {0: 0.125, 1: 0.125})
        self.assertEqual(sorted(observed), [0, 1])
        self.assertNotEqual(observed[0], observed[1], "both devices ran on the same thread")

    def test_a_failure_on_one_device_does_not_hide_the_other(self):
        started = threading.Event()

        def half_broken(index, ops):
            if index == 0:
                started.set()
                raise ValueError("synthetic warm-up failure")
            started.wait(timeout=30.0)
            return 0.5

        with mock.patch.object(_warmup, "_warm_up_device", half_broken):
            with self.assertRaises(RuntimeError) as caught:
                _warmup.warm_up_devices([0, 1])

        self.assertIn("cuda:0", str(caught.exception))


@unittest.skipUnless(torch.cuda.is_available(), "a CUDA device is required")
class ConcurrentHostThreadTests(unittest.TestCase):
    """Regression guard for the concurrent-driving deadlock fixed 2026-07-27.

    Op wrappers hold the GIL and then take the scene-registry mutex, so nothing
    holding that mutex may wait for the GIL. `destroy_scene` used to release
    the scene's mesh tensors under it, and releasing a tensor that carries a
    Python object gives the GIL up and takes it back — the reverse order, and
    an ABBA deadlock against any thread inside an op.

    Building, querying and dropping scenes from several threads at once is the
    shape that reproduced it. One device is enough; the second is used when it
    is there because that is the documented multi-device shape.
    """

    ROUNDS = 12
    TIMEOUT_S = 120.0

    @staticmethod
    def _churn(index: int, rounds: int, failures: list[BaseException]) -> None:
        device = torch.device("cuda", index)
        try:
            with torch.cuda.device(index):
                for _ in range(rounds):
                    scene = _warmup._throwaway_scene(device)
                    _warmup._run_op(scene, "intersect", device)
                    # Dropping the scene here is the point: its destructor runs
                    # while the other thread is inside an op.
                    del scene
        except BaseException as error:  # noqa: BLE001 - reported by the test
            failures.append(error)

    def test_building_and_dropping_scenes_concurrently_completes(self):
        indices = list(range(min(2, torch.cuda.device_count())))
        failures: list[BaseException] = []
        threads = [
            threading.Thread(
                target=self._churn, args=(index, self.ROUNDS, failures), name=f"rayd-churn-{index}", daemon=True
            )
            # Two threads even on one device: the deadlock is a host-thread
            # defect, not a multi-device one.
            for index in (indices * 2)[:2]
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=self.TIMEOUT_S)

        stuck = [thread.name for thread in threads if thread.is_alive()]
        self.assertEqual(stuck, [], f"threads never finished: {stuck}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
