# Copyright Xingyu Chen.
# Tests the GPU compatibility check against mocked device queries.

import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python" / "rayd" / "_impl" / "device_support.py"


def _load():
    spec = importlib.util.spec_from_file_location("rayd_device_support_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ds = _load()

RELEASE_DRJIT = ds.BuildTargets("drjit", (70, 75, 80, 86, 87, 89, 90, 100, 101, 120), (120,), 70)
RELEASE_TORCH = ds.BuildTargets("torch", (70, 75, 80, 86, 87, 89, 90, 100, 101, 120), (120,), 75)
CLEAN_ENVIRONMENT = {ds.SKIP_ENVIRONMENT: "", ds.DISABLE_OPTIX_ENVIRONMENT: ""}


class CompatibilityRuleTests(unittest.TestCase):
    def test_release_builds_run_on_covered_gpus(self):
        for cc in (75, 80, 86, 89, 90, 100, 103, 120, 121):
            with self.subTest(cc=cc):
                self.assertEqual(ds.problems(RELEASE_DRJIT, cc), [])
                self.assertEqual(ds.problems(RELEASE_TORCH, cc), [])

    def test_future_gpu_runs_forward_compatible_ptx(self):
        self.assertEqual(ds.problems(RELEASE_DRJIT, 130), [])

    def test_newer_minor_revision_runs_older_cubin(self):
        targets = ds.BuildTargets("drjit", (80,), (), 70)
        self.assertEqual(ds.problems(targets, 86), [])
        self.assertTrue(ds.problems(targets, 90))

    def test_ptx_runs_forward_only(self):
        targets = ds.BuildTargets("drjit", (), (75,), 70)
        self.assertEqual(ds.problems(targets, 120), [])
        self.assertIn("no CUDA kernel image runs on sm_70", ds.problems(targets, 70)[0])

    def test_torch_release_optix_excludes_volta(self):
        # The Torch OptiX PTX targets compute_75, so Volta runs only its CUDA paths.
        self.assertEqual(ds.problems(RELEASE_DRJIT, 70), [])
        found = ds.problems(RELEASE_TORCH, 70)
        self.assertEqual(len(found), 1)
        self.assertIn("compute_75", found[0])
        self.assertEqual(ds.problems(RELEASE_TORCH, 70, optix=False), [])

    def test_torch_source_build_for_volta_runs_optix(self):
        self.assertEqual(ds.problems(ds.BuildTargets("torch", (70, 89), (), 70), 70), [])

    def test_below_volta_is_never_supported(self):
        for cc in (37, 52, 61):
            with self.subTest(cc=cc):
                found = ds.problems(ds.BuildTargets("torch", (cc,), (cc,), cc), cc)
                self.assertEqual(len(found), 1)
                self.assertIn("Volta", found[0])

    def test_unknown_cuda_targets_check_only_optix(self):
        targets = ds.BuildTargets("torch", None, (), 75)
        self.assertEqual(ds.problems(targets, 75), [])
        self.assertEqual(len(ds.problems(targets, 70)), 1)

    def test_lowest_supported(self):
        self.assertEqual(RELEASE_DRJIT.lowest_supported, 70)
        self.assertEqual(RELEASE_TORCH.lowest_supported, 75)
        self.assertEqual(ds.BuildTargets("torch", (70, 89), (), 70).lowest_supported, 70)
        self.assertEqual(ds.BuildTargets("drjit", (89,), (), 70).lowest_supported, 89)
        self.assertIsNone(ds.BuildTargets("drjit", None, (), None).lowest_supported)


class MessageTests(unittest.TestCase):
    def _error(self, targets, device):
        with mock.patch.dict(os.environ, CLEAN_ENVIRONMENT):
            with self.assertRaises(ds.UnsupportedDeviceError) as caught:
                ds.check(targets, device)
        return str(caught.exception)

    def test_message_names_gpu_range_and_build_command(self):
        text = self._error(RELEASE_TORCH, ds.Device(0, "Tesla V100-SXM2-16GB", 70))
        self.assertIn("rayd-torch cannot run on CUDA device 0 'Tesla V100-SXM2-16GB'", text)
        self.assertIn("compute capability 7.0, sm_70", text)
        self.assertIn("supports sm_75 and newer", text)
        for line in ds.build_command("torch", 70).splitlines():
            self.assertIn(line, text)
        self.assertIn("CUDA 12.x", text)
        self.assertIn(ds.DISABLE_OPTIX_ENVIRONMENT, text)
        self.assertIn(ds.SKIP_ENVIRONMENT, text)

    def test_message_for_a_source_build_that_omits_the_gpu(self):
        text = self._error(ds.BuildTargets("drjit", (89,), (), 70), ds.Device(1, "NVIDIA A100", 80))
        self.assertIn("no CUDA kernel image runs on sm_80 (cubins: sm_89; PTX: none)", text)
        self.assertIn("supports sm_89 and newer", text)
        self.assertIn("80", ds.build_command("drjit", 80))

    def test_build_command_per_platform(self):
        self.assertIn(
            "RAYD_CUDA_ARCHS=70 python -m pip install --no-build-isolation ./drjit",
            ds.build_command("drjit", 70, platform="linux"),
        )
        self.assertIn(
            ".\\scripts\\build_local.cmd -Backend torch -CudaArch 86", ds.build_command("torch", 86, platform="win32")
        )

    def test_below_volta_message_offers_no_build(self):
        text = self._error(RELEASE_DRJIT, ds.Device(1, "NVIDIA GeForce GTX 1080", 61))
        self.assertIn("compute capability 6.1, sm_61", text)
        self.assertIn("Volta", text)
        self.assertNotIn("git clone", text)

    def test_supported_device_passes(self):
        with mock.patch.dict(os.environ, CLEAN_ENVIRONMENT):
            ds.check(RELEASE_DRJIT, ds.Device(0, "RTX 4090", 89))
            ds.check(None, ds.Device(0, "anything", 30))

    def test_environment_overrides(self):
        with mock.patch.dict(os.environ, {**CLEAN_ENVIRONMENT, ds.DISABLE_OPTIX_ENVIRONMENT: "1"}):
            ds.check(RELEASE_TORCH, ds.Device(0, "Tesla V100", 70))
        with mock.patch.dict(os.environ, {**CLEAN_ENVIRONMENT, ds.SKIP_ENVIRONMENT: "1"}):
            ds.check(RELEASE_TORCH, ds.Device(0, "GTX 1080", 61))


class VisibleDeviceTests(unittest.TestCase):
    def test_no_usable_device_raises(self):
        devices = [ds.Device(0, "GTX 1080", 61), ds.Device(1, "GTX 1070", 61)]
        with mock.patch.dict(os.environ, CLEAN_ENVIRONMENT):
            with self.assertRaisesRegex(ds.UnsupportedDeviceError, "device 0 'GTX 1080'"):
                ds.check_visible(RELEASE_DRJIT, devices)

    def test_partially_usable_machine_warns(self):
        devices = [ds.Device(0, "GTX 1080", 61), ds.Device(1, "RTX 4090", 89)]
        with mock.patch.dict(os.environ, CLEAN_ENVIRONMENT):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ds.check_visible(RELEASE_DRJIT, devices)
        self.assertEqual(len(caught), 1)
        self.assertIn("GTX 1080", str(caught[0].message))

    def test_nothing_to_check(self):
        with mock.patch.dict(os.environ, CLEAN_ENVIRONMENT):
            ds.check_visible(RELEASE_DRJIT, None)
            ds.check_visible(RELEASE_DRJIT, [])
            ds.check_visible(None, [ds.Device(0, "GTX 1080", 61)])


class BuildRecordTests(unittest.TestCase):
    def test_record_is_found_beside_the_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {"backend": "torch", "cuda_sass": [89, 70], "cuda_ptx": [], "optix_ptx": 70}
            Path(tmp, "_build_config.json").write_text(json.dumps(record), encoding="utf-8")
            spec = types.SimpleNamespace(origin=str(Path(tmp, "_C.pyd")))
            with mock.patch.object(ds.importlib.util, "find_spec", return_value=spec):
                targets = ds.load_build_targets("torch", "rayd.torch._C")
                self.assertEqual(targets, ds.BuildTargets("torch", (70, 89), (), 70))
                with self.assertRaises(RuntimeError):
                    ds.load_build_targets("drjit", "rayd.drjit._C")

    def test_missing_record_or_extension_disables_the_check(self):
        with mock.patch.object(ds.importlib.util, "find_spec", return_value=None):
            self.assertIsNone(ds.load_build_targets("drjit", "rayd.drjit._C"))
        with tempfile.TemporaryDirectory() as tmp:
            spec = types.SimpleNamespace(origin=str(Path(tmp, "_C.pyd")))
            with mock.patch.object(ds.importlib.util, "find_spec", return_value=spec):
                self.assertIsNone(ds.load_build_targets("drjit", "rayd.drjit._C"))


class BuildSourceAgreementTests(unittest.TestCase):
    def test_cmake_and_python_share_the_floor(self):
        text = (ROOT / "cmake" / "RayDCudaArchitectures.cmake").read_text(encoding="utf-8")
        floor = re.search(r"^set\(RAYD_MIN_CUDA_ARCH (\d+)\)$", text, re.MULTILINE)
        self.assertIsNotNone(floor)
        self.assertEqual(int(floor.group(1)), ds.MIN_COMPUTE_CAPABILITY)

    def test_drjit_record_matches_the_committed_ptx_arch(self):
        optix = (ROOT / "cmake" / "RayDOptix.cmake").read_text(encoding="utf-8")
        emitted = set(re.findall(r"-arch=compute_(\d+)", optix))
        recorded = re.search(
            r"^set\(RAYD_DRJIT_OPTIX_PTX_ARCH (\d+)\)$",
            (ROOT / "drjit" / "CMakeLists.txt").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        self.assertIsNotNone(recorded)
        self.assertEqual(emitted, {recorded.group(1)})


class DriverQueryTests(unittest.TestCase):
    def _fake_driver(self, devices):
        def write(pointer, value):
            pointer._obj.value = value

        def get_attribute(pointer, attribute, handle):
            major, minor = divmod(devices[handle.value][1], 10)
            write(pointer, major if attribute == 75 else minor)
            return 0

        def get_name(buffer, size, handle):
            buffer.value = devices[handle.value][0].encode()
            return 0

        return types.SimpleNamespace(
            cuInit=lambda flags: 0,
            cuDeviceGetCount=lambda pointer: write(pointer, len(devices)) or 0,
            cuDeviceGet=lambda pointer, ordinal: write(pointer, ordinal) or 0,
            cuDeviceGetAttribute=get_attribute,
            cuDeviceGetName=get_name,
        )

    def test_reads_every_visible_device(self):
        fake = self._fake_driver([("GTX 1080", 61), ("RTX 4090", 89)])
        with mock.patch.object(ds.ctypes, "CDLL", return_value=fake):
            self.assertEqual(ds.driver_devices(), [ds.Device(0, "GTX 1080", 61), ds.Device(1, "RTX 4090", 89)])

    def test_missing_driver_returns_none(self):
        with mock.patch.object(ds.ctypes, "CDLL", side_effect=OSError("no driver")):
            self.assertIsNone(ds.driver_devices())

    def test_driver_init_failure_returns_none(self):
        fake = self._fake_driver([])
        fake.cuInit = lambda flags: 100  # CUDA_ERROR_NO_DEVICE
        with mock.patch.object(ds.ctypes, "CDLL", return_value=fake):
            self.assertIsNone(ds.driver_devices())


if __name__ == "__main__":
    unittest.main()
