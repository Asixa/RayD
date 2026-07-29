# Copyright Xingyu Chen.
# Tests verify cuda binary arches Dr.Jit.

import importlib.util
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest import TestCase
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[2] / "drjit" / "scripts" / "verify_cuda_binary_arches.py"
_SPEC = importlib.util.spec_from_file_location("_verify_cuda_binary_arches", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_cuobjdump = _MODULE._cuobjdump


class VerifyCudaBinaryArchesTests(TestCase):
    def test_linux_elf_falls_back_to_extracted_fatbin(self):
        calls: list[list[str]] = []

        def run(command, **kwargs):
            calls.append(command)
            if command[0] == "objcopy":
                fatbin = Path(command[2].split("=", 1)[1])
                fatbin.write_bytes(b"fatbin")
                return CompletedProcess(command, 0, "", "")
            if command[-1].endswith(".fatbin"):
                return CompletedProcess(command, 0, "ELF file: kernel.sm_87.cubin", "")
            return CompletedProcess(command, 255, "", "host ELF is too large")

        with patch("_verify_cuda_binary_arches.subprocess.run", side_effect=run):
            output = _cuobjdump("--list-elf", Path("rayd/torch/_C.so"))

        self.assertIn("sm_87", output)
        self.assertEqual([call[0] for call in calls], ["cuobjdump", "objcopy", "cuobjdump"])

    def test_failure_reports_cuobjdump_stderr(self):
        failed = CompletedProcess(["cuobjdump"], 255, "", "unsupported input")
        with patch("_verify_cuda_binary_arches.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(SystemExit, "unsupported input"):
                _cuobjdump("--list-elf", Path("_C.pyd"))
