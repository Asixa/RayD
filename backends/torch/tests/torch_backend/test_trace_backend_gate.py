from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


_SCENE_SETUP = r"""
import torch
import rayd.torch as rt

device = torch.device("cuda")
vertices = torch.tensor(
    [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]],
    dtype=torch.float32,
    device=device,
)
faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
"""


def _run_fresh(body: str, *, disable_optix: bool | str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    python_root = str(Path(__file__).resolve().parents[2] / "python")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (python_root, env.get("PYTHONPATH", "")) if part
    )
    if isinstance(disable_optix, str):
        env["RAYD_DISABLE_OPTIX"] = disable_optix
    elif disable_optix:
        env["RAYD_DISABLE_OPTIX"] = "1"
    else:
        env.pop("RAYD_DISABLE_OPTIX", None)
        env.pop("RAYD_TORCH_DISABLE_OPTIX", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCENE_SETUP + body)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA is required")
class TorchTraceBackendGateTests(unittest.TestCase):
    def test_false_kill_switch_value_does_not_change_selection(self) -> None:
        body = r"""
scene = rt.Scene()
scene.add_mesh(rt.Mesh(vertices, faces))
scene.build()
print(scene.trace_backend, scene.edge_bvh_backend)
"""
        baseline = _run_fresh(body, disable_optix=False)
        false_value = _run_fresh(body, disable_optix="false")
        self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
        self.assertEqual(false_value.returncode, 0, false_value.stdout + false_value.stderr)
        self.assertEqual(false_value.stdout.strip(), baseline.stdout.strip())

    def test_kill_switch_is_checked_after_context_cache_warmup(self) -> None:
        result = _run_fresh(
            r"""
import os
first = rt.Scene()
first.add_mesh(rt.Mesh(vertices, faces))
first.build()
os.environ["RAYD_DISABLE_OPTIX"] = "1"
second = rt.Scene()
second.add_mesh(rt.Mesh(vertices, faces))
second.build()
assert second.trace_backend == "cuda", second.trace_backend
assert second.edge_bvh_backend == "cuda", second.edge_bvh_backend
explicit = rt.Scene(trace_backend="optix")
explicit.add_mesh(rt.Mesh(vertices, faces))
try:
    explicit.build()
except RuntimeError as exc:
    assert "OptiX" in str(exc), str(exc)
else:
    raise AssertionError("cached OptiX context bypassed the kill switch")
""",
            disable_optix=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_auto_selects_cuda_when_optix_is_disabled(self) -> None:
        result = _run_fresh(
            r"""
scene = rt.Scene()
scene.add_mesh(rt.Mesh(vertices, faces))
scene.build()
assert scene.trace_backend == "cuda", scene.trace_backend
assert scene.edge_bvh_backend == "cuda", scene.edge_bvh_backend
ray = rt.Ray(
    torch.tensor([[0.0, 0.0, 0.0]], device=device),
    torch.tensor([[0.0, 0.0, 1.0]], device=device),
)
hit = scene.intersect(ray)
assert hit.global_prim_id.item() == 0
assert torch.equal(hit.t, torch.ones_like(hit.t))
edge = scene.nearest_edge(torch.tensor([[0.0, -1.0, 1.0]], device=device))
assert edge.global_edge_id.item() >= 0
""",
            disable_optix=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_optix_fails_instead_of_falling_back(self) -> None:
        result = _run_fresh(
            r"""
for kwargs in (
    {"trace_backend": "optix"},
    {"trace_backend": "cuda", "edge_bvh_backend": "optix"},
):
    scene = rt.Scene(**kwargs)
    scene.add_mesh(rt.Mesh(vertices, faces))
    try:
        scene.build()
    except RuntimeError as exc:
        assert "OptiX" in str(exc), str(exc)
    else:
        raise AssertionError(f"explicit OptiX request silently succeeded: {kwargs}")
""",
            disable_optix=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_cuda_does_not_initialize_optix(self) -> None:
        result = _run_fresh(
            r"""
scene = rt.Scene(trace_backend="cuda", edge_bvh_backend="cuda")
scene.add_mesh(rt.Mesh(vertices, faces))
scene.build()
assert scene.trace_backend == "cuda"
assert scene.edge_bvh_backend == "cuda"
""",
            disable_optix=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
