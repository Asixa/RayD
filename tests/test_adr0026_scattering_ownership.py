import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "adr" / "0026-generic-scattering-runtime-ownership.md"
STABLE_INTEGRATION_ADR = (
    ROOT / "docs" / "adr" / "0028-stable-typed-integration-naming.md"
)
STABLE_INTEGRATION_HEADER = (
    "backends/torch/include/rayd/torch/integration.h"
)
STABLE_INTEGRATION_HEADER_HASH = (
    "57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14"
)

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
    "chain ensemble": {
        "scattering_chain_ensemble_eval",
        "scattering_chain_ensemble_eval_backward",
        "scattering_chain_ensemble_eval_jvp",
    },
    "chain realization": {
        "scattering_chain_realization_eval",
        "scattering_chain_realization_eval_backward",
        "scattering_chain_realization_eval_jvp",
    },
}

PHASE10B_CONTRACT_HASHES = {
    "backends/torch/include/rayd/torch/rf/scattering.h":
        "7a29ff216f11a08256ee271ef5dcad817e4b8379d88bc07772685fa3da439aa9",
    "shared/include/rayd/shared/rf/scattering_table.cuh":
        "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38",
    "backends/torch/src/torch_ext/rf/scattering.cu":
        "aae7d33ae78d8886c8d1d1a665336e9027d35a3b2ba180a85b7211cd3ee22e21",
    "backends/torch/src/torch_ext/rf/scattering_table_eval_ad.cu":
        "fe2046c2a3ba45bb073cb43272a89282593c0c2a1659f4cc7e85d2ebd5335039",
    "backends/torch/src/torch_ext/rf/scattering_ensemble.cu":
        "ed6f9225e1d987b8624062dafce617755a793d5646f8866bc6106637e8c4d492",
    "backends/torch/src/torch_ext/rf/scattering_ensemble_ad.cu":
        "3951fef2cac6759c05b57167a4491cdf421f575843ca33a9e1d761c713237573",
    "backends/torch/src/torch_ext/rf/scattering_patch_integral.cu":
        "a37459f03879199cb0365a20b0cc06fca5fc24369a7efda858e189d11822af33",
    "backends/torch/src/torch_ext/rf/scattering_patch_integral_ad.cu":
        "bf40adb74e029a520162363383ca16c39e4829ee9e0ff3252816b0bcf04bff82",
    "backends/torch/src/torch_ext/rf/scattering_chain_ad_common.cuh":
        "2551c33533dc7ea0a0c1680d67e5432587f8c2f77833d5a717fcb2d20597b507",
    "backends/torch/src/torch_ext/rf/scattering_chain_checks.h":
        "f848b268bbca8835ac091bc49f223d0f64532925361090bb1409c93d1d50278c",
    "backends/torch/src/torch_ext/rf/scattering_chain_ensemble.cu":
        "1121a2f276d982bb2bf6efe3e20aa0d82eb7251e224b7efea8c1099c92e9afe7",
    "backends/torch/src/torch_ext/rf/scattering_chain_ensemble_ad.cu":
        "554a6ad5cdfd1aac37913e3526e8c5d252ec514db6992eafd7d43882a14956bc",
    "backends/torch/src/torch_ext/rf/scattering_chain_realization.cu":
        "e61dd957af9a2fdc9a5035040c874a12b3834a00cb8285b64e921f1d01cb72b3",
    "backends/torch/src/torch_ext/rf/scattering_chain_realization_ad.cu":
        "63cf0704157d591307eb788b9de08c22aadd919177f8545e8cb4f8b037bb27bd",
}

PHASE10B_CUDA_SOURCES = tuple(
    path for path in PHASE10B_CONTRACT_HASHES if path.endswith(".cu")
)


class Adr0026ScatteringOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = ADR.read_text(encoding="utf-8")
        cls.stable_integration_text = STABLE_INTEGRATION_ADR.read_text(
            encoding="utf-8"
        )

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

    def test_phase10b_typed_header_identity_and_dormant_source_scope(self):
        header = (
            ROOT
            / "backends"
            / "torch"
            / "include"
            / "rayd"
            / "torch"
            / "integration.h"
        ).read_text(encoding="utf-8")
        self.assertIn("#include <rayd/torch/rf/scattering.h>", header)
        self.assertIn(
            "rayd.torch.integration",
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
                "scattering_chain_ensemble",
                "scattering_chain_ensemble_ad",
                "scattering_chain_realization",
                "scattering_chain_realization_ad",
            },
        )
        self.assertNotIn("scattering_event", source_list)
        self.assertIn("tests/cpp/scattering_test.cpp", cmake)
        self.assertIn("NAME rayd_torch_scattering", cmake)
        self.assertIn("tests/cpp/scattering_chain_test.cpp", cmake)
        self.assertIn("NAME rayd_torch_scattering_chain", cmake)

    def test_phase10b_source_local_fmad_policy(self):
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
                "scattering_chain_ensemble",
                "scattering_chain_ensemble_ad",
                "scattering_chain_realization",
                "scattering_chain_realization_ad",
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

    def test_phase10b_source_contract_hashes_are_pinned(self):
        for relative_path, expected_hash in PHASE10B_CONTRACT_HASHES.items():
            with self.subTest(path=relative_path):
                data = (
                    (ROOT / relative_path)
                    .read_bytes()
                    .replace(b"\r\n", b"\n")
                    .replace(b"\r", b"\n")
                )
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected_hash)
                self.assertIn(expected_hash, self.text)

    def test_stable_integration_header_hash_is_pinned(self):
        data = (
            (ROOT / STABLE_INTEGRATION_HEADER)
            .read_bytes()
            .replace(b"\r\n", b"\n")
            .replace(b"\r", b"\n")
        )
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            STABLE_INTEGRATION_HEADER_HASH,
        )
        self.assertIn(
            STABLE_INTEGRATION_HEADER_HASH,
            self.stable_integration_text,
        )

    def test_phase10b_cuda_sources_have_no_shim_fallback_or_scope_leak(self):
        forbidden = (
            "pybind11",
            "torch/extension.h",
            "TensorResultMap",
            "cn_scattering",
            "channel_native",
            "scattering_event_kernel",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "GetProcAddress",
            "dlsym",
        )
        for relative_path in PHASE10B_CUDA_SOURCES:
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(path=relative_path):
                for token in forbidden:
                    self.assertNotIn(token, text)

        chain_sources = tuple(
            path for path in PHASE10B_CUDA_SOURCES if "scattering_chain_" in path
        )
        for relative_path in chain_sources:
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(path=relative_path):
                for token in (
                    "scattering_event_probabilities",
                    "bdpt",
                    "montecarlo",
                    "mis_weight",
                    "curand",
                    "solver",
                ):
                    self.assertNotIn(token, text.lower())


if __name__ == "__main__":
    unittest.main()
