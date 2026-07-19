import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "adr" / "0026-generic-scattering-runtime-ownership.md"

EXPECTED_FAMILIES = {
    "resident table evaluation AD": {
        "scattering_table_eval",
        "scattering_table_eval_backward",
        "scattering_table_eval_jvp",
    },
    "resident table sampling": {
        "scattering_table_sample",
        "scattering_table_pdf",
    },
    "single-bounce ensemble": {
        "scattering_ensemble_eval",
        "scattering_ensemble_eval_backward",
        "scattering_ensemble_eval_jvp",
    },
    "phase-screen patch integral": {
        "scattering_patch_integral_eval",
        "scattering_patch_integral_eval_backward",
        "scattering_patch_integral_eval_jvp",
    },
    "v2 chain ensemble": {
        "scattering_chain_ensemble_eval",
        "scattering_chain_ensemble_eval_backward",
        "scattering_chain_ensemble_eval_jvp",
    },
    "v2 chain realization": {
        "scattering_chain_realization_eval",
        "scattering_chain_realization_eval_backward",
        "scattering_chain_realization_eval_jvp",
    },
}

PHASE10A_CONTRACT_HASHES = {
    "backends/torch/include/rayd/torch/rf/scattering.h":
        "66d75a20be16057f03cdfb79e3b9dcc85cacec79b555cd73b019259aa510262a",
    "shared/include/rayd/shared/rf/scattering_table.cuh":
        "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38",
    "backends/torch/include/rayd/torch/integration_v2.h":
        "9f95ad9e8e3b790d00f8e762a3e6a09252d46afb65bfc3aba7c42325836cb1fb",
    "backends/torch/src/torch_ext/rf/scattering.cu":
        "72fb84a4158652a70c5f4f17e5d1ce61371773cdd54db6835148ee065e474c50",
    "backends/torch/src/torch_ext/rf/scattering_table_eval_ad.cu":
        "e09cb3992737b028222e205318baea1aa070d300f0126def9759edaa17ad5b7c",
    "backends/torch/src/torch_ext/rf/scattering_ensemble.cu":
        "be38ff966dd06afe3f1df46d2eb16094c97111c76534e22d5f3fec6685f1f1fc",
    "backends/torch/src/torch_ext/rf/scattering_ensemble_ad.cu":
        "8c094b3a6542b1da26e662e38c405ec1d90cf53aaf8934147b0549f66a8fb0ea",
    "backends/torch/src/torch_ext/rf/scattering_patch_integral.cu":
        "e1d8555874a1832067e92e9f1973cee38d9ce2f18dac230b56bb1c6504c0c08b",
    "backends/torch/src/torch_ext/rf/scattering_patch_integral_ad.cu":
        "0d3bffe34ecd22656f1c5bdb10a6fe903ad059803547e29ccb95f5fd390858aa",
}

PHASE10A_CUDA_SOURCES = tuple(
    path for path in PHASE10A_CONTRACT_HASHES if path.endswith(".cu")
)


class Adr0026ScatteringOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = ADR.read_text(encoding="utf-8")

    def test_accepted_adr_freezes_exact_six_family_seventeen_symbol_matrix(self):
        self.assertIn("- Status: Accepted", self.text)
        section = self.text.split("### Complete operation families", 1)[1].split(
            "### Unique typed and device-header owners", 1
        )[0]
        actual = {}
        for line in section.splitlines():
            if not line.startswith("|") or "`scattering_" not in line:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            actual[cells[0]] = set(re.findall(r"`(scattering_[a-z0-9_]+)`", cells[1]))

        self.assertEqual(actual, EXPECTED_FAMILIES)
        symbols = set().union(*actual.values())
        self.assertEqual(len(symbols), 17)

    def test_governance_files_are_byte_identical_and_link_the_adr(self):
        self.assertEqual(
            (ROOT / "AGENTS.md").read_bytes(),
            (ROOT / "CLAUDE.md").read_bytes(),
        )
        link = "0026-generic-scattering-runtime-ownership.md"
        self.assertIn(link, (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn(
            link,
            (ROOT / "backends" / "torch" / "README.md").read_text(encoding="utf-8"),
        )

    def test_adr_freezes_asymmetric_geometry_ad_and_tu_flags(self):
        required = (
            "VJP request for continuous chain\n  geometry fails loudly",
            "chain-realization backward/VJP and JVP continue to support",
            "`scattering.cu` uses the RayD target's default CUDA flags",
            "`scattering_chain_realization_ad.cu` retain `--fmad=false`",
            "Rollback changes Channel's lock",
            "high-level BSDF framework",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text)

    def test_phase10a_typed_header_identity_and_dormant_source_scope(self):
        header = (
            ROOT
            / "backends"
            / "torch"
            / "include"
            / "rayd"
            / "torch"
            / "integration_v2.h"
        ).read_text(encoding="utf-8")
        self.assertIn("#include <rayd/torch/rf/scattering.h>", header)
        self.assertIn(
            "rayd.torch.integration.v2.20260719.rf-transmission-sequence."
            "pure-wedge-diffraction.scattering-table-single-bounce",
            header,
        )

        cmake = (ROOT / "backends" / "torch" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        source_list = cmake.split("set(\n        RAYD_TORCH_NATIVE_CORE_SOURCES", 1)[
            1
        ].split("add_library(rayd_torch_native_core", 1)[0]
        scattering_sources = set(
            re.findall(r"src/torch_ext/rf/(scattering[a-z0-9_]*)\.cu", source_list)
        )
        self.assertEqual(
            scattering_sources,
            {
                "scattering",
                "scattering_table_eval_ad",
                "scattering_ensemble",
                "scattering_ensemble_ad",
                "scattering_patch_integral",
                "scattering_patch_integral_ad",
            },
        )
        self.assertNotIn("scattering_chain_", source_list)
        self.assertNotIn("scattering_event", source_list)
        self.assertIn("tests/cpp/scattering_test.cpp", cmake)
        self.assertIn("NAME rayd_torch_scattering", cmake)

    def test_phase10a_source_local_fmad_policy(self):
        cmake = (ROOT / "backends" / "torch" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        fmad_blocks = [
            block
            for block in re.findall(
                r"set_source_files_properties\((.*?)\)", cmake, re.DOTALL
            )
            if "--fmad=false" in block
        ]
        self.assertEqual(len(fmad_blocks), 1)
        fmad_sources = set(
            re.findall(r"src/torch_ext/rf/(scattering[a-z0-9_]*)\.cu", fmad_blocks[0])
        )
        self.assertEqual(
            fmad_sources,
            {
                "scattering_table_eval_ad",
                "scattering_ensemble",
                "scattering_ensemble_ad",
                "scattering_patch_integral",
                "scattering_patch_integral_ad",
            },
        )
        self.assertNotIn("src/torch_ext/rf/scattering.cu", fmad_blocks[0])

        fast_math_blocks = [
            block
            for block in re.findall(
                r"set_source_files_properties\((.*?)\)", cmake, re.DOTALL
            )
            if "--use_fast_math" in block
        ]
        for block in fast_math_blocks:
            self.assertNotIn("src/torch_ext/rf/scattering.cu", block)

    def test_phase10a_source_contract_hashes_are_pinned(self):
        for relative_path, expected_hash in PHASE10A_CONTRACT_HASHES.items():
            with self.subTest(path=relative_path):
                data = (
                    (ROOT / relative_path)
                    .read_bytes()
                    .replace(b"\r\n", b"\n")
                    .replace(b"\r", b"\n")
                )
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected_hash)
                self.assertIn(expected_hash, self.text)

    def test_phase10a_cuda_sources_have_no_shim_fallback_or_scope_leak(self):
        forbidden = (
            "pybind11",
            "TensorResultMap",
            "cn_scattering",
            "channel_native",
            "scattering_event_kernel",
            "scattering_chain_",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "GetProcAddress",
            "dlsym",
        )
        for relative_path in PHASE10A_CUDA_SOURCES:
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(path=relative_path):
                for token in forbidden:
                    self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
