# Copyright Xingyu Chen.
# Tests torch ptx depfiles.

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "torch" / "CMakeLists.txt"

EXPECTED_OUTPUTS = {
    "RAYD_TORCH_OPTIX_PTX",
    "RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX",
    "RAYD_TORCH_EDGE_OPTIX_TOPK_PTX",
    "RAYD_TORCH_REFLECTION_TRACE_PTX",
    "RAYD_TORCH_SEGMENT_VISIBILITY_PTX",
    "RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX",
    "RAYD_TORCH_REFLECTION_EPC_PTX",
    "RAYD_TORCH_REFLECTION_ACCUMULATION_PTX",
    "RAYD_TORCH_DIFFRACTION_PATHS_PTX",
    "RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX",
    "RAYD_TORCH_SEGMENT_PENETRATION_PTX",
}


class TorchPtxDepfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = CMAKE.read_text(encoding="utf-8")
        cls.ptx_blocks = [
            block
            for block in re.findall(r"    add_custom_command\(\s*(.*?)\r?\n    \)", text, re.DOTALL)
            if re.search(r'COMMAND\s+"\$\{CMAKE_CUDA_COMPILER\}"\s+--ptx\b', block)
        ]

    def test_all_raw_ptx_commands_are_accounted_for(self):
        outputs = []
        for block in self.ptx_blocks:
            match = re.search(r'OUTPUT "\$\{([^}]+)\}"', block)
            self.assertIsNotNone(match, block)
            outputs.append(match.group(1))

        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertEqual(set(outputs), EXPECTED_OUTPUTS)

    def test_every_ptx_command_emits_and_registers_a_depfile(self):
        for block in self.ptx_blocks:
            output = re.search(r'OUTPUT "\$\{([^}]+)\}"', block).group(1)
            with self.subTest(output=output):
                self.assertRegex(block, r"(?m)^\s+-MD\s*$")
                self.assertIn(f'-MF "${{{output}}}.d"', block)
                self.assertIn(f'-MT "${{{output}}}"', block)
                self.assertIn(f'DEPFILE "${{{output}}}.d"', block)
                self.assertRegex(block, r"(?m)^\s+VERBATIM\s*$")


if __name__ == "__main__":
    unittest.main()
