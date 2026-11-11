# Copyright Xingyu Chen.
# Tests the Torch EPC field options, its Ray input, and the default that keeps the fixed-material result.

import math
import unittest

import torch

import rayd.torch as rt


# The material and source the EPC field used before it took options.
LEGACY_ETA_R, LEGACY_MU_R, LEGACY_SIGMA, LEGACY_GAIN = 4.0, 1.0, 0.0, 1.0
FIELDS = ("field_re", "field_im", "path_length", "valid", "resolved_prim_ids")
SOURCES = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.03, 0.0, 0.0], [0.0, 0.0, -0.05], [-0.02, 0.0, 0.02]]
RECEIVERS = [[0.0, 2.0, 0.0], [0.0, 2.0, 0.05], [0.03, 2.0, 0.0], [0.0, 2.1, 0.0], [-0.02, 1.95, 0.0]]


def _mirror_scene():
    """Two mirrors that carry each source to its receiver over two bounces."""
    from tests.parity.test_share2_ad import _rotate_two_bounce_vector, _two_bounce_vertices

    vertices = torch.tensor([_rotate_two_bounce_vector(vertex) for vertex in _two_bounce_vertices()], device="cuda")
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(vertices, torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)))
    scene.build()
    return scene


def _slots(value: float, count: int = len(SOURCES), bounces: int = 2) -> torch.Tensor:
    return torch.full((count, bounces), value, device="cuda", dtype=torch.float32)


class TorchEpcFieldOptionsRecordTests(unittest.TestCase):
    def test_defaults_match_the_drjit_defaults(self):
        options = rt.ReflEpcFieldOptions()
        self.assertEqual(options.wavelength, 1.0)
        self.assertEqual(options.omega, 2.0 * math.pi * 299792458.0)
        for name in ("slot_eta_r", "slot_mu_r", "slot_sigma", "slot_gain", "tx_polarization"):
            self.assertIsNone(getattr(options, name))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TorchEpcFieldOptionsTests(unittest.TestCase):
    def setUp(self):
        self.scene = _mirror_scene()
        self.source = torch.tensor(SOURCES, device="cuda")
        self.receiver = torch.tensor(RECEIVERS, device="cuda")

    def assert_fields_equal(self, actual, expected):
        for name in FIELDS:
            with self.subTest(field=name):
                self.assertTrue(torch.equal(getattr(actual, name), getattr(expected, name)), name)

    def test_default_options_are_bit_identical_to_the_fixed_material(self):
        default = self.scene.trace_refl_epc_field(self.source, self.receiver, 2)
        self.assertTrue(default.valid.all())
        self.assertGreater(float(default.field_re.abs().min()), 1.0e-4)
        legacy = rt.ReflEpcFieldOptions(
            slot_eta_r=_slots(LEGACY_ETA_R),
            slot_mu_r=_slots(LEGACY_MU_R),
            slot_sigma=_slots(LEGACY_SIGMA),
            slot_gain=_slots(LEGACY_GAIN),
            tx_polarization=torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
            omega=2.0 * math.pi * 299792458.0,
            wavelength=1.0,
        )
        self.assert_fields_equal(self.scene.trace_refl_epc_field(self.source, self.receiver, 2, legacy), default)
        self.assert_fields_equal(
            self.scene.trace_refl_epc_field(self.source, self.receiver, 2, rt.ReflEpcFieldOptions()), default
        )

    def test_a_ray_toward_the_receiver_reproduces_the_source_call(self):
        default = self.scene.trace_refl_epc_field(self.source, self.receiver, 2)
        direction = self.receiver - self.source
        ray = rt.Ray(self.source, direction, torch.linalg.vector_norm(direction, dim=1))
        self.assert_fields_equal(self.scene.trace_refl_epc_field(ray, self.receiver, 2), default)

    def test_a_ray_is_traced_along_its_own_direction(self):
        away = rt.Ray(self.source, torch.tensor([[0.0, -1.0, 0.0]] * len(SOURCES), device="cuda"))
        self.assertFalse(self.scene.trace_refl_epc_field(away, self.receiver, 2).valid.any())
        bounded = rt.Ray(self.source, self.receiver - self.source, torch.full((len(SOURCES),), 0.5, device="cuda"))
        self.assertFalse(self.scene.trace_refl_epc_field(bounded, self.receiver, 2).valid.any())

    def test_material_and_wavelength_change_the_field(self):
        default = self.scene.trace_refl_epc_field(self.source, self.receiver, 2)
        half_gain = self.scene.trace_refl_epc_field(
            self.source, self.receiver, 2, rt.ReflEpcFieldOptions(slot_gain=_slots(0.5))
        )
        # A gain of 1/2 on both bounces scales the field by exactly 1/4 and leaves the path alone.
        torch.testing.assert_close(half_gain.field_re, 0.25 * default.field_re, rtol=1e-6, atol=0.0)
        torch.testing.assert_close(half_gain.field_im, 0.25 * default.field_im, rtol=1e-6, atol=0.0)
        self.assertTrue(torch.equal(half_gain.path_length, default.path_length))
        for options in (
            rt.ReflEpcFieldOptions(slot_eta_r=_slots(2.5)),
            rt.ReflEpcFieldOptions(slot_mu_r=_slots(1.5)),
            rt.ReflEpcFieldOptions(slot_sigma=_slots(0.05), omega=2.0 * math.pi * 1.0e9),
            rt.ReflEpcFieldOptions(wavelength=0.125),
            rt.ReflEpcFieldOptions(tx_polarization=torch.tensor([[0.0, 0.0, 1.0]], device="cuda")),
        ):
            with self.subTest(options=options):
                changed = self.scene.trace_refl_epc_field(self.source, self.receiver, 2, options)
                self.assertTrue(torch.equal(changed.valid, default.valid))
                self.assertFalse(torch.equal(changed.field_re, default.field_re))

    def test_per_row_polarization_matches_a_shared_row(self):
        polarization = torch.tensor([[0.0, 0.6, 0.8]], device="cuda")
        shared = self.scene.trace_refl_epc_field(
            self.source, self.receiver, 2, rt.ReflEpcFieldOptions(tx_polarization=polarization)
        )
        per_row = self.scene.trace_refl_epc_field(
            self.source,
            self.receiver,
            2,
            rt.ReflEpcFieldOptions(tx_polarization=polarization.expand(len(SOURCES), 3).contiguous()),
        )
        self.assert_fields_equal(per_row, shared)

    def test_invalid_options_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, r"slot_eta_r must have shape \(N, max_bounces\)"):
            self.scene.trace_refl_epc_field(
                self.source, self.receiver, 2, rt.ReflEpcFieldOptions(slot_eta_r=_slots(4.0, bounces=1))
            )
        with self.assertRaisesRegex(RuntimeError, "tx_polarization"):
            self.scene.trace_refl_epc_field(
                self.source,
                self.receiver,
                2,
                rt.ReflEpcFieldOptions(tx_polarization=torch.ones((2, 3), device="cuda")),
            )
        with self.assertRaisesRegex(RuntimeError, "wavelength"):
            self.scene.trace_refl_epc_field(self.source, self.receiver, 2, rt.ReflEpcFieldOptions(wavelength=0.0))
        with self.assertRaisesRegex(TypeError, "ReflEpcFieldOptions"):
            self.scene.trace_refl_epc_field(self.source, self.receiver, 2, torch.ones(5, device="cuda", dtype=bool))

    def test_option_tensors_are_forward_only(self):
        tracked = _slots(4.0).requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            self.scene.trace_refl_epc_field(self.source, self.receiver, 2, rt.ReflEpcFieldOptions(slot_eta_r=tracked))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(_slots(1.0), _slots(1.0))
            with self.assertRaisesRegex(RuntimeError, "JVP"):
                self.scene.trace_refl_epc_field(self.source, self.receiver, 2, rt.ReflEpcFieldOptions(slot_gain=dual))
        ray = rt.Ray(self.source.clone().requires_grad_(True), self.receiver - self.source)
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            self.scene.trace_refl_epc_field(ray, self.receiver, 2)


if __name__ == "__main__":
    unittest.main()
