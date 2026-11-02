# Copyright Xingyu Chen.
# Tests that Git-checkout source builds stay the only route for GPUs the wheels do not cover.

import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "pypi.yml"
HOSTED = WORKFLOWS / "ci.yml"
SINGLE_GPU = WORKFLOWS / "single_gpu.yml"


def _build_commands(text: str) -> list[str]:
    """Every `python -m build` or `pip wheel` line in a workflow."""
    return [line.strip() for line in text.splitlines() if re.search(r"-m build\b|pip wheel\b", line)]


class SourceBuildRouteTests(unittest.TestCase):
    def test_only_the_meta_distribution_builds_an_sdist(self):
        commands = [command for command in _build_commands(RELEASE.read_text(encoding="utf-8")) if "--sdist" in command]
        self.assertEqual(len(commands), 1, commands)
        self.assertIn("dist/meta", commands[0])

    def test_release_and_hosted_builds_never_use_the_source_build_switch(self):
        # Release artifacts must name every architecture explicitly; the
        # source-build switch and its `native` detection are local only.
        for workflow in (RELEASE, HOSTED):
            with self.subTest(workflow=workflow.name):
                self.assertNotIn("RAYD_CUDA_ARCHS", workflow.read_text(encoding="utf-8"))

    def test_gpu_acceptance_builds_through_the_documented_route(self):
        text = SINGLE_GPU.read_text(encoding="utf-8")
        self.assertIn("RAYD_CUDA_ARCHS: native", text)

    def test_documented_commands_match_the_runtime_message(self):
        # rayd/_impl/device_support.py prints these exact commands.
        guide = (ROOT / "docs" / "dev" / "gpu_support.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for text in (guide, readme):
            self.assertIn("RAYD_CUDA_ARCHS=native python -m pip install --no-build-isolation ./drjit", text)
            self.assertIn("RAYD_CUDA_ARCHS=native python -m pip install --no-build-isolation ./torch", text)
            self.assertIn(r".\scripts\build_local.cmd -Backend torch -CudaArch native", text)
            self.assertIn("git clone https://github.com/Asixa/RayD", text)


class ReleaseArtifactSdistTests(unittest.TestCase):
    def test_backend_distributions_publish_no_sdist(self):
        dist_dir = os.environ.get("RAYD_RELEASE_DIST_DIR")
        if not dist_dir:
            self.skipTest("release artifact directory is provided by distribution CI")
        for stem in ("rayd_drjit", "rayd-drjit", "rayd_torch", "rayd-torch"):
            self.assertEqual(list(Path(dist_dir).glob(f"{stem}-*.tar.gz")), [], stem)


if __name__ == "__main__":
    unittest.main()
