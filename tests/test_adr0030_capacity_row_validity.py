import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_INCLUDE = ROOT / "include" / "rayd" / "torch"
RF_SOURCE = ROOT / "src"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def struct_body(text: str, name: str) -> str:
    match = re.search(rf"struct {name}\s*\{{(?P<body>.*?)\n\}};", text, re.S)
    if match is None:
        raise AssertionError(f"missing struct {name}")
    return match.group("body")


def kernel_body(text: str, name: str) -> str:
    start = text.index(f"__global__ void {name}(")
    next_kernel = text.find("__global__ void ", start + 1)
    return text[start:] if next_kernel < 0 else text[start:next_kernel]


class Adr0030CapacityRowValidityTests(unittest.TestCase):
    def test_api_version_and_stable_identity(self):
        integration = read(TORCH_INCLUDE / "integration.h")
        self.assertIn("kIntegrationApiVersion = 6", integration)
        self.assertIn('"rayd.torch.integration"', integration)
        self.assertFalse((TORCH_INCLUDE / "integration_v2.h").exists())

    def test_top_level_primal_requests_require_cuda_row_validity(self):
        contracts = {
            TORCH_INCLUDE / "diffraction" / "wedge.h": {
                "DiffractionWedgeRequest": "at::Tensor valid;",
            },
            TORCH_INCLUDE / "transmission" / "transmission.h": {
                "TransmissionSequenceRequest": "at::Tensor path_valid;",
            },
            TORCH_INCLUDE / "scattering" / "scattering.h": {
                "ScatteringTableEvalRequest": "at::Tensor valid;",
                "ScatteringTableSampleRequest": "at::Tensor valid;",
                "ScatteringTablePdfRequest": "at::Tensor valid;",
                "ScatteringEnsembleEvalRequest": "at::Tensor valid;",
                "ScatteringPatchIntegralEvalRequest": "at::Tensor valid;",
                "ScatteringChainEnsembleEvalRequest": "at::Tensor valid;",
                "ScatteringChainRealizationEvalRequest": "at::Tensor valid;",
            },
        }
        for path, requests in contracts.items():
            text = read(path)
            for request, declaration in requests.items():
                with self.subTest(request=request):
                    body = struct_body(text, request)
                    self.assertEqual(body.count(declaration), 1)
                    self.assertNotIn("std::optional<at::Tensor> valid", body)
                    self.assertNotIn("std::optional<at::Tensor> path_valid", body)

        scattering = read(TORCH_INCLUDE / "scattering" / "scattering.h")
        for name in (
            "ScatteringTableEvalBackwardRequest",
            "ScatteringTableEvalJvpRequest",
            "ScatteringEnsembleEvalBackwardRequest",
            "ScatteringEnsembleEvalJvpRequest",
            "ScatteringPatchIntegralEvalBackwardRequest",
            "ScatteringPatchIntegralEvalJvpRequest",
            "ScatteringChainEnsembleEvalBackwardRequest",
            "ScatteringChainEnsembleEvalJvpRequest",
            "ScatteringChainRealizationEvalBackwardRequest",
            "ScatteringChainRealizationEvalJvpRequest",
        ):
            self.assertIn(" primal;", struct_body(scattering, name))

    def test_every_row_kernel_gates_before_poisonable_payload(self):
        cases = {
            "diffraction/wedge.cu": {
                "diffraction_wedge_forward_kernel": ("if (!valid[index])", "load_wedge_row"),
                "diffraction_wedge_backward_kernel": ("if (!valid[index])", "load_wedge_row"),
                "diffraction_wedge_jvp_kernel": ("if (!valid[index])", "load_wedge_row"),
            },
            "transmission/transmission.cu": {
                "transmission_sequence_kernel": ("if (!path_valid[index])", "load3(source"),
                "transmission_sequence_backward_kernel": ("if (!path_valid[index])", "transmission_chain_eval"),
                "transmission_sequence_jvp_kernel": ("if (!path_valid[index])", "transmission_chain_eval"),
            },
            "scattering/table.cu": {
                "scattering_eval_kernel": ("if (!valid[row])", "wi + 3 * row"),
                "scattering_sample_kernel": ("if (!valid[row])", "wi + 3 * row"),
                "scattering_pdf_kernel": ("if (!valid[row])", "wi + 3 * row"),
            },
            "scattering/table_ad.cu": {
                "table_eval_backward_kernel": ("if (!valid[row])", "st::eval_te_tm"),
                "table_eval_jvp_kernel": ("if (!valid[row])", "st::eval_te_tm"),
            },
            "scattering/ensemble.cu": {
                "ensemble_eval_kernel": ("if (!valid[row])", "r2_rows[row]"),
                "ensemble_eval_backward_kernel": ("if (!valid[row])", "recompute_row"),
                "ensemble_eval_jvp_kernel": ("if (!valid[row])", "recompute_row"),
            },
            "scattering/patch.cu": {
                "patch_integral_rows_kernel": ("if (!valid[row])", "rows[row]"),
                "patch_integral_backward_kernel": ("if (!valid[row])", "rows[row]"),
                "patch_integral_jvp_kernel": ("if (!valid[row])", "rows[row]"),
            },
            "scattering/chain_ensemble.cu": {
                "chain_ensemble_eval_kernel": ("if (!valid[row])", "c1_depth[row]"),
                "chain_ensemble_backward_kernel": ("if (!valid[row])", "c1_depth[row]"),
                "chain_ensemble_jvp_kernel": ("if (!valid[row])", "c1_depth[row]"),
            },
            "scattering/chain_realization.cu": {
                "chain_realization_rows_kernel": ("if (!valid[row])", "rows[row]"),
                "chain_realization_backward_kernel": ("if (!valid[row])", "rows[row]"),
                "chain_realization_jvp_kernel": ("if (!valid[row])", "rows[row]"),
            },
        }
        for source_name, kernels in cases.items():
            text = read(RF_SOURCE / source_name)
            for kernel_name, (gate, first_payload) in kernels.items():
                with self.subTest(source=source_name, kernel=kernel_name):
                    body = kernel_body(text, kernel_name)
                    self.assertIn(gate, body)
                    self.assertIn(first_payload, body)
                    self.assertLess(body.index(gate), body.index(first_payload))
                    if "atomicAdd(" in body:
                        self.assertLess(body.index(gate), body.index("atomicAdd("))

    def test_contract_validation_and_guardrails_are_locked(self):
        sources = "\n".join(
            read(RF_SOURCE / path)
            for path in (
                "diffraction/wedge.cu",
                "transmission/transmission.cu",
                "scattering/table.cu",
                "scattering/table_ad.cu",
                "scattering/ensemble.cu",
                "scattering/patch.cu",
                "scattering/chain_ensemble.cu",
                "scattering/chain_realization.cu",
            )
        )
        self.assertIn('check_flat_tensor(valid, "valid", at::kBool)', sources)
        self.assertIn('check_flat_tensor(request.path_valid, "path_valid", at::kBool)', sources)
        self.assertNotIn("valid.value_or", sources)
        self.assertNotIn("path_valid.value_or", sources)
        self.assertEqual(read(ROOT / "AGENTS.md"), read(ROOT / "CLAUDE.md"))


if __name__ == "__main__":
    unittest.main()
