# Copyright Xingyu Chen.
# Tests that Torch forward-mode AD accepts an explicit active mask on intersect and trace_reflections.

import unittest

import torch
from torch.autograd import forward_ad

try:
    import rayd.torch as rt
except ImportError:  # pragma: no cover - exercised only without the Torch backend
    rt = None


@unittest.skipUnless(rt is not None and torch.cuda.is_available(), "CUDA Torch backend is required")
class ForwardAdActiveMaskTests(unittest.TestCase):
    # The native forward ops hand the caller's `active` tensor back unchanged; the autograd Functions must not save
    # that tensor as an output, which forward-mode AD refuses ("returned as-is ... saved for backward").
    def setUp(self):
        vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
        self.scene = rt.Scene()
        self.scene.add_mesh(rt.Mesh(vertices, torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)))
        self.scene.build()
        self.origins = torch.tensor([[0.25, 0.25, -1.0], [0.3, 0.2, -2.0]], device="cuda")
        self.directions = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], device="cuda")
        self.tangent = torch.tensor([[0.0, 0.0, 1.0], [0.2, -0.1, 0.5]], device="cuda")

    def _tangents(self, query):
        values = {}
        for label, active in (("none", None), ("mask", torch.ones(2, dtype=torch.bool, device="cuda"))):
            with forward_ad.dual_level():
                origins = forward_ad.make_dual(self.origins, self.tangent)
                values[label] = forward_ad.unpack_dual(query(rt.Ray(origins, self.directions), active)).tangent
        return values

    def test_intersect_jvp_with_active_mask(self):
        values = self._tangents(lambda ray, active: self.scene.intersect(ray, active).t)
        self.assertIsNotNone(values["mask"])
        torch.testing.assert_close(values["mask"], values["none"], atol=0.0, rtol=0.0)

    def test_trace_reflections_jvp_with_active_mask(self):
        values = self._tangents(lambda ray, active: self.scene.trace_reflections(ray, 1, active).t)
        self.assertIsNotNone(values["mask"])
        torch.testing.assert_close(values["mask"], values["none"], atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
