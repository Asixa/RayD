# Copyright Xingyu Chen.
# Tests required diffraction path validity.

import unittest
from pathlib import Path

from tests.support.source_inspection import read_text as read, struct_body


ROOT = Path(__file__).resolve().parents[1]
RAYD_INCLUDE = ROOT / "include" / "rayd"
TORCH_SOURCE = ROOT / "src"
TORCH_PYTHON = ROOT / "python" / "rayd" / "_impl"


class Adr0031RequiredDiffractionPathValidityTests(unittest.TestCase):
    def test_required_path_export_validity_survives_api6(self):
        integration = read(RAYD_INCLUDE / "integration.h")
        self.assertIn("kIntegrationApiVersion = 8", integration)
        diffraction = read(RAYD_INCLUDE / "diffraction.h")
        config = struct_body(diffraction, "DiffractionPathConfig")
        self.assertEqual(config.count("at::Tensor active;"), 1)
        self.assertNotIn("std::optional<at::Tensor> active", config)
        self.assertIn("std::optional<at::Tensor> active;", struct_body(diffraction, "DiffractionAccumulationConfig"))
        self.assertIn("std::optional<at::Tensor> active;", struct_body(diffraction, "CoherentDiffractionConfig"))

    def test_dispatch_and_public_python_require_active(self):
        library = read(TORCH_SOURCE / "bindings" / "library.cpp")
        schema_start = library.index('m.def("diffraction_paths_order1_forward(')
        schema = library[schema_start : library.index(");", schema_start)]
        self.assertIn("Tensor active", schema)
        self.assertNotIn("Tensor? active", schema)

        autograd = read(TORCH_PYTHON / "multipath.py")
        trace_native = autograd.split("def trace_dfr_paths_order1_native", 1)[1].split(") -> DfrPaths:", 1)[0]
        self.assertIn("active: torch.Tensor,", trace_native)
        self.assertNotIn("active: torch.Tensor | None", trace_native)

        scene = read(TORCH_PYTHON / "scene.py")
        trace_scene = scene.split("def trace_dfr_paths", 1)[1].split(") -> DfrPaths:", 1)[0]
        self.assertIn("active: torch.Tensor,", trace_scene)
        self.assertNotIn("active: torch.Tensor | None", trace_scene)

    def test_host_contract_is_cuda_bool_contiguous_exact_width(self):
        ops = read(TORCH_SOURCE / "diffraction" / "diffraction.cpp")
        body = ops.split("DiffractionPathOutputs diffraction_paths_order1_forward_impl", 1)[1]
        body = body.split("py::tuple diffraction_path_outputs_to_tuple", 1)[0]
        for contract in (
            'require_cuda(active, "active")',
            'require_contiguous(active, "active")',
            'require_dtype(active, at::kBool, "active")',
            'require_rank(active, 1, "active")',
            "active.size(0) != state_limit",
            'require_scene_device(scene, active, "active")',
        ):
            self.assertIn(contract, body)
        self.assertNotIn("require_optional_mask(active", body)
        self.assertNotIn("optional_mask_ptr(active", body)

    def test_device_paths_have_no_implicit_all_valid_branch(self):
        optix = read(TORCH_SOURCE / "diffraction" / "paths_optix.cu")
        shared = read(ROOT / "include" / "rayd" / "diffraction" / "paths_algo.h")
        for source in (optix, shared):
            self.assertIn("return params.active_mask[state_idx] != 0u;", source)
            self.assertNotIn("params.active_mask == nullptr", source)

        optix_lane = optix.split("static __forceinline__ __device__ void trace_paths_order1_impl", 1)[1]
        optix_lane = optix_lane.split("static __forceinline__ __device__", 1)[0]
        self.assertLess(optix_lane.index("!state_active(state_idx)"), optix_lane.index("vec_from_storage("))

        shared_lane = shared.split("RAYD_DEVICE void trace_paths_order1_algo", 1)[1]
        shared_lane = shared_lane.split("RAYD_DEVICE void trace_paths_source_visibility_algo", 1)[0]
        self.assertLess(shared_lane.index("!state_active(params, state_idx)"), shared_lane.index("state_vec("))

    def test_governance_files_are_identical_and_link_decision(self):
        self.assertEqual((ROOT / "AGENTS.md").read_bytes(), (ROOT / "CLAUDE.md").read_bytes())
        link = "0031-required-diffraction-path-validity.md"
        self.assertIn(link, read(ROOT / "AGENTS.md"))
        self.assertIn(link, read(ROOT / "torch" / "README.md"))


if __name__ == "__main__":
    unittest.main()
