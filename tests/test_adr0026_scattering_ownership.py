import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "adr" / "0026-generic-scattering-runtime-ownership.md"
STABLE_INTEGRATION_ADR = (
    ROOT / "docs" / "adr" / "0028-stable-typed-integration-naming.md"
)
STABLE_INTEGRATION_HEADER = (
    "include/rayd/integration.h"
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

HISTORICAL_PHASE10B_HASHES = frozenset(
    {
        "061f41fe99435a60eb2afd5763f7422ccba800595e126963f2efe81d599569dd",
        "28e520b86ed622ab65509e2d8fa46a1f5f04c7cdfe64f79943fcd805adddb545",
        "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38",
        "49afe510215b5251ce4d220712f96f0b876a529401306683bb7439ade031c01f",
        "529e8777750c26cef2aed691a8799dda1f5035af02fdaa0a71725cf8584044ac",
        "55db93ec294f91b3355876eedf6089170f49fad43f1608197e848bd53ce17eb5",
        "61a9e2e86854880bd60ab35c77bc3d0308c07c3c61f560f4cce4f05b109a874c",
        "7a29ff216f11a08256ee271ef5dcad817e4b8379d88bc07772685fa3da439aa9",
        "89f50f631233775d10bf33719482ec06ad16861bae7d9696d2d793fbf934910b",
        "8b41199b7e3f8c796bf933de5d8aa43432df2fcce2cfbf19764e5292f763733d",
        "e77f5a3888186ef675ba88516fa059fb2d252db6bb1420099e8b37614637d544",
        "e96a4a0229d626a6ad55cacdbf71a16a48c438b248c18442b2d63a7a1850d60c",
        "f5db3d5f93efe38273e28c9dad548da56cbccfc43a53f634064cc592545bfb1b",
        "f848b268bbca8835ac091bc49f223d0f64532925361090bb1409c93d1d50278c",
    }
)

CANONICAL_SCATTERING_SOURCES = (
    "src/scattering/table.cu",
    "src/scattering/table_ad.cu",
    "src/scattering/ensemble.cu",
    "src/scattering/patch.cu",
    "src/scattering/chain_ensemble.cu",
    "src/scattering/chain_realization.cu",
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
            (ROOT / "torch" / "README.md").read_text(encoding="utf-8"),
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
            ROOT / "include" / "rayd" / "integration.h"
        ).read_text(encoding="utf-8")
        self.assertIn("#include <rayd/scattering.h>", header)
        self.assertIn(
            "rayd.torch.integration",
            header,
        )

        cmake = (ROOT / "torch" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        source_list = cmake.split("set(\n        RAYD_TORCH_NATIVE_CORE_SOURCES", 1)[
            1
        ].split("add_library(rayd_torch_native_core", 1)[0]
        scattering_sources = set(
            re.findall(r"src/scattering/([a-z0-9_]+)\.cu", source_list)
        )
        self.assertEqual(
            scattering_sources,
            {
                "table",
                "table_ad",
                "ensemble",
                "patch",
                "chain_ensemble",
                "chain_realization",
            },
        )
        self.assertNotIn("scattering_event", source_list)
        self.assertIn("tests/scattering/scattering_test.cpp", cmake)
        self.assertIn("NAME rayd_torch_scattering", cmake)
        self.assertIn("tests/scattering/scattering_chain_test.cpp", cmake)
        self.assertIn("NAME rayd_torch_scattering_chain", cmake)

    def test_phase10b_source_local_fmad_policy(self):
        cmake = (ROOT / "torch" / "CMakeLists.txt").read_text(
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
            re.findall(r"src/scattering/([a-z0-9_]+)\.cu", fmad_blocks[0])
        )
        self.assertEqual(
            fmad_sources,
            {
                "table_ad",
                "ensemble",
                "patch",
                "chain_ensemble",
                "chain_realization",
            },
        )
        self.assertNotIn("src/scattering/table.cu", fmad_blocks[0])

        fast_math_blocks = [
            block
            for block in re.findall(
                r"set_source_files_properties\((.*?)\)", cmake, re.DOTALL
            )
            if "--use_fast_math" in block
        ]
        for block in fast_math_blocks:
            self.assertNotIn("src/scattering/table.cu", block)

    def test_phase10b_historical_hashes_remain_evidence_and_live_sources_exist(self):
        for expected_hash in HISTORICAL_PHASE10B_HASHES:
            with self.subTest(hash=expected_hash):
                self.assertIn(expected_hash, self.text)
        for relative_path in CANONICAL_SCATTERING_SOURCES:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_stable_integration_historical_hash_and_live_identity_are_pinned(self):
        self.assertIn(
            STABLE_INTEGRATION_HEADER_HASH,
            self.stable_integration_text,
        )
        header = (ROOT / STABLE_INTEGRATION_HEADER).read_text(encoding="utf-8")
        self.assertIn("kIntegrationApiVersion = 8", header)
        self.assertIn('"rayd.torch.integration"', header)

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
        for relative_path in CANONICAL_SCATTERING_SOURCES:
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(path=relative_path):
                for token in forbidden:
                    self.assertNotIn(token, text)

        chain_sources = tuple(
            path for path in CANONICAL_SCATTERING_SOURCES if "chain_" in path
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
