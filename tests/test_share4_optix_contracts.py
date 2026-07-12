import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared" / "include" / "rayd" / "shared" / "optix"


class Share4DiffractionOptixContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_path = SHARED / "diffraction_contracts.h"
        cls.contract = cls.contract_path.read_text(encoding="utf-8")
        cls.path_headers = (
            ROOT / "backends" / "drjit" / "include" / "rayd" / "multipath" / "diffraction_paths_params.h",
            ROOT / "backends" / "torch" / "include" / "rayd" / "torch" / "diffraction" / "paths_params.h",
        )
        cls.accum_headers = (
            ROOT / "backends" / "drjit" / "include" / "rayd" / "multipath" / "diffraction_accumulation_params.h",
            ROOT / "backends" / "torch" / "include" / "rayd" / "torch" / "diffraction" / "accum_params.h",
        )

    def test_contract_is_backend_neutral_pod_only(self):
        self.assertTrue(self.contract_path.is_file())
        lowered = self.contract.lower()
        for forbidden in (
            "at::tensor",
            "torch/",
            "torch::",
            "drjit",
            "nanobind",
            "scenehandle",
            "std::vector",
            "std::unique_ptr",
            "std::shared_ptr",
            "cudamalloc",
            "cudafree",
        ):
            self.assertNotIn(forbidden, lowered)
        structs = re.findall(r"^struct\s+(\w+)\s*\{", self.contract, re.MULTILINE)
        self.assertEqual(len(structs), 5)
        for struct in structs:
            self.assertIn(f"RAYD_SHARED_DIFFRACTION_ASSERT_POD({struct})", self.contract)

    def test_strategy_and_field_slots_are_frozen(self):
        for token in (
            "Direct = 1 << 0",
            "Keller = 1 << 1",
            "SuffixReflection = 1 << 2",
            "Hash = 0",
            "Sobol = 1",
            "MatchedIsotropic = 0",
            "DiffractionPathField::Count) == 23u",
            "DiffractionAccumOutputField::Count) == 29u",
            "DiffractionAccumTapeField::Count) == 5u",
        ):
            self.assertIn(token, self.contract)

    def test_backend_public_enums_derive_from_shared_values(self):
        headers = (
            ROOT / "backends" / "drjit" / "include" / "rayd" / "multipath" / "diffraction_accumulation.h",
            ROOT / "backends" / "torch" / "include" / "rayd" / "torch" / "diffraction" / "common.h",
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in headers)
        self.assertGreaterEqual(combined.count("DiffractionStrategyBit::"), 6)
        self.assertGreaterEqual(combined.count("DiffractionSampleSequence::"), 4)
        self.assertGreaterEqual(combined.count("DiffractionReceiverModel::"), 2)
        self.assertNotRegex(combined, r"RAYD_(?:TORCH_)?DFR_DIRECT\s*=\s*1\s*<<")

    def test_path_params_validate_only_common_sub_layouts(self):
        for path in self.path_headers:
            source = path.read_text(encoding="utf-8")
            self.assertIn("<rayd/shared/optix/diffraction_contracts.h>", source)
            self.assertIn("is_standard_layout_v<DfrPathParams>", source)
            self.assertEqual(source.count("RAYD_ASSERT_DFR_PATH_PREFIX("), 19)
            self.assertEqual(source.count("RAYD_ASSERT_DFR_PATH_TAIL("), 7)
            self.assertIn("DiffractionPathOutputPrefix", source)
            self.assertIn("DiffractionPathGeometryTail", source)

        torch = self.path_headers[1].read_text(encoding="utf-8")
        drjit = self.path_headers[0].read_text(encoding="utf-8")
        self.assertIn("tx_pos_aos", torch)
        self.assertIn("state_edge_index_stride", torch)
        self.assertNotIn("tx_pos_aos", drjit)
        self.assertNotIn("state_edge_index_stride", drjit)
        self.assertNotIn("using DfrPathParams =", torch + drjit)

    def test_accum_params_validate_grid_output_and_tape_sub_layouts(self):
        for path in self.accum_headers:
            source = path.read_text(encoding="utf-8")
            self.assertIn("<rayd/shared/optix/diffraction_contracts.h>", source)
            self.assertIn("is_standard_layout_v<DfrAccumParams>", source)
            self.assertEqual(source.count("RAYD_ASSERT_DFR_GRID("), 10)
            self.assertEqual(source.count("RAYD_ASSERT_DFR_ACCUM_OUTPUT("), 30)
            self.assertEqual(source.count("RAYD_ASSERT_DFR_ACCUM_TAPE("), 6)
            self.assertIn("DiffractionGridParams", source)
            self.assertIn("DiffractionAccumOutputPointers", source)
            self.assertIn("DiffractionAccumTapePointers", source)

        torch = self.accum_headers[1].read_text(encoding="utf-8")
        drjit = self.accum_headers[0].read_text(encoding="utf-8")
        self.assertIn("active_stride", torch)
        self.assertIn("coherent_stage_value", torch)
        self.assertNotIn("active_stride", drjit)
        self.assertNotIn("coherent_stage_value", drjit)
        self.assertNotIn("using DfrAccumParams =", torch + drjit)

    def test_device_programs_consume_backend_params_without_host_ownership(self):
        device_sources = (
            ROOT / "backends" / "drjit" / "src" / "multipath" / "diffraction_paths.cu",
            ROOT / "backends" / "drjit" / "src" / "multipath" / "diffraction_accumulation.cu",
            ROOT / "backends" / "torch" / "src" / "torch_ext" / "diffraction" / "paths_optix.cu",
            ROOT / "backends" / "torch" / "src" / "torch_ext" / "diffraction" / "accum_optix.cu",
        )
        for path in device_sources:
            source = path.read_text(encoding="utf-8")
            self.assertIn("__constant__", source)
            self.assertIn("params", source)
        for forbidden in ("OptixDeviceContext", "OptixPipeline", "at::Tensor", "drjit::"):
            self.assertNotIn(forbidden, self.contract)


if __name__ == "__main__":
    unittest.main()
