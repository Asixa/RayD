"""The process-per-GPU recipes under `backends/torch/examples/distributed`.

The examples are the documented multi-GPU route for the Dr.Jit backend and the
cluster-scale route for both backends, so they are covered the way a user runs
them: `torchrun --nproc_per_node=2` in a subprocess, not an in-process
imitation of one. What the launched runs have to prove is the two properties
the recipes claim:

- the ranks stay a single replicated model -- identical final parameters after
  a train loop whose only cross-rank traffic is one gradient all-reduce;
- the rank-sharded lane windows are a partition of one Monte-Carlo launch --
  the merged grid reproduces a single-process, single-device launch of the full
  sample count, up to the summation order of the all-reduce.

The lane arithmetic itself is checked in-process so a single-GPU machine still
runs something.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "distributed"
_INTERSECT_EXAMPLE = _EXAMPLES / "ddp_intersect_train.py"
_ACCUM_EXAMPLE = _EXAMPLES / "ddp_accum_grids.py"

# Generous: a cold OptiX pipeline is built once per rank, and the ranks compile
# concurrently into separate caches. The point of the bound is to kill a hang,
# not to time the run.
_LAUNCH_TIMEOUT_SECONDS = 900.0


def _load_example(path: Path):
    """Import an example by path without putting the directory on `sys.path`."""
    spec = importlib.util.spec_from_file_location(f"rayd_example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _torchrun() -> str | None:
    """`torchrun` belonging to *this* interpreter, not to whatever is on PATH."""
    candidate = Path(sys.executable).parent / "torchrun"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("torchrun")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_env(cache_root: str) -> dict[str, str]:
    """The child's environment: this process's `rayd` package plus a cache root.

    The examples give each rank a private `OPTIX_CACHE_PATH` under the root, so
    two ranks never compile into one cache database.
    """
    import rayd.torch as rt

    package_root = str(Path(rt.__file__).resolve().parents[2])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        package_root if not existing else os.pathsep.join((package_root, existing))
    )
    env["RAYD_EXAMPLE_OPTIX_CACHE_ROOT"] = cache_root
    env.pop("OPTIX_CACHE_PATH", None)
    return env


class _Launcher:
    """`torchrun --nproc_per_node=2 <example>`, killed as a group if it hangs."""

    def __init__(self, test: unittest.TestCase, cache_root: str) -> None:
        self._test = test
        self._env = _launch_env(cache_root)

    def run(self, script: Path, *args: str) -> str:
        launcher = _torchrun()
        assert launcher is not None
        command = [
            launcher,
            "--nnodes=1",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={_free_port()}",
            str(script),
            *args,
        ]
        # Own session: a torchrun that wedges takes its workers down with it.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self._env,
            cwd=str(script.parent),
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=_LAUNCH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            output, _ = process.communicate()
            self._test.fail(
                f"{script.name} did not finish within "
                f"{_LAUNCH_TIMEOUT_SECONDS:.0f}s; output:\n{output}"
            )
        self._test.assertEqual(
            process.returncode,
            0,
            f"{script.name} exited {process.returncode}; output:\n{output}",
        )
        return output


class DistributedLaneArithmeticTests(unittest.TestCase):
    """Pure host arithmetic, so it runs wherever the test suite runs."""

    def test_ray_shards_partition_the_global_batch(self):
        module = _load_example(_INTERSECT_EXAMPLE)
        for count in (0, 1, 7, 1024, 262144):
            for world_size in (1, 2, 3, 8):
                covered: list[int] = []
                for rank in range(world_size):
                    begin, end = module.shard_bounds(count, rank, world_size)
                    self.assertLessEqual(begin, end)
                    covered.extend(range(begin, end))
                self.assertEqual(covered, list(range(count)))

    def test_lane_windows_partition_the_sample_space(self):
        module = _load_example(_ACCUM_EXAMPLE)
        for requested in (1, 31, 1024, 1 << 20, 100000):
            for world_size in (1, 2, 3, 8):
                total = module.resolve_total_samples(requested, world_size)
                self.assertGreaterEqual(total, requested)
                self.assertEqual(total % (module.LANE_ALIGNMENT * world_size), 0)
                cursor = 0
                for rank in range(world_size):
                    begin, count = module.lane_window(total, rank, world_size)
                    self.assertEqual(begin, cursor, "windows must be contiguous")
                    self.assertGreater(count, 0)
                    if rank + 1 < world_size:
                        self.assertEqual(count % module.LANE_ALIGNMENT, 0)
                    cursor = begin + count
                self.assertEqual(cursor, total, "windows must cover the space")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
@unittest.skipUnless(
    torch.cuda.device_count() >= 2, "two CUDA devices are required"
)
@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "torch.distributed with NCCL is required",
)
@unittest.skipUnless(_torchrun() is not None, "torchrun was not found")
class DistributedRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cache = tempfile.TemporaryDirectory(prefix="rayd-optix-cache-")
        self.addCleanup(self._cache.cleanup)
        self.launcher = _Launcher(self, self._cache.name)

    def test_ddp_intersect_train_keeps_the_ranks_bitwise_identical(self):
        output = self.launcher.run(
            _INTERSECT_EXAMPLE,
            "--steps=6",
            "--check-every=3",
            "--rays=65536",
            "--cells=16",
        )
        # Two ranks writing to one pipe can interleave inside a line, so match
        # the tokens rather than parsing lines.
        digests = re.findall(r"final_param_sha256=([0-9a-f]{64})", output)
        self.assertEqual(len(digests), 2, f"expected two rank hashes; got:\n{output}")
        self.assertEqual(
            digests[0],
            digests[1],
            f"ranks diverged: {digests[0]} vs {digests[1]}\n{output}",
        )
        # The in-run drift assertion has to have actually fired, or the hash
        # agreement above is the only thing that ran.
        self.assertGreaterEqual(len(re.findall(r"drift=0", output)), 4, output)

    def test_ddp_accum_grids_merges_to_the_single_process_reference(self):
        module = _load_example(_ACCUM_EXAMPLE)
        samples = 1 << 20
        seed = 7
        resolution = 8
        with tempfile.TemporaryDirectory(prefix="rayd-ddp-grids-") as directory:
            merged_path = os.path.join(directory, "merged.pt")
            output = self.launcher.run(
                _ACCUM_EXAMPLE,
                f"--samples={samples}",
                f"--seed={seed}",
                f"--resolution={resolution}",
                f"--out={merged_path}",
            )
            self.assertTrue(os.path.isfile(merged_path), output)
            merged = torch.load(merged_path, weights_only=True)

        total = module.resolve_total_samples(samples, 2)
        self.assertEqual(merged["total_samples"], total)
        self.assertEqual(merged["world_size"], 2)

        # The reference: one process, one device, one launch of the *whole*
        # lane space. The two-rank run must reproduce it, because its windows
        # partition exactly that space.
        device = torch.device("cuda", 0)
        scene, states, material, grid = module.build_fixture(device, resolution)
        reference = module.accumulate(
            scene,
            states,
            material,
            grid,
            total=total,
            seed=seed,
            begin=0,
            count=-1,
        )

        for name in module.COUNT_GRIDS:
            # Counts are integers: the partition is exact or it is wrong.
            self.assertTrue(
                torch.equal(merged[name], getattr(reference, name).cpu()),
                f"{name} differs between the merged and reference grids",
            )
        for name in module.FLOAT_GRIDS:
            expected = getattr(reference, name).cpu()
            # The only admissible difference is the order the identical set of
            # sample contributions was summed in.
            torch.testing.assert_close(
                merged[name], expected, rtol=1e-4, atol=1e-9, msg=f"{name} mismatch"
            )
        self.assertGreater(
            float(getattr(reference, "power").double().sum()),
            0.0,
            "the fixture accumulated nothing, so the comparison is vacuous",
        )
        self.assertIn("merged_grid_checksum=", output)


if __name__ == "__main__":
    unittest.main()
