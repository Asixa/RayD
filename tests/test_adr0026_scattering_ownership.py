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


if __name__ == "__main__":
    unittest.main()
