# Copyright Xingyu Chen.
# Tests legacy loader compat.

import unittest

import rayd.torch as rt
from rayd._impl import runtime as _legacy


class LegacyLoaderCompatibilityTests(unittest.TestCase):
    def test_c_availability_tracks_legacy_dispatcher_registration(self):
        self.assertEqual(rt._C is not None, _legacy.is_registered())

    def test_metadata_shim_remains_available_with_native_dispatcher(self):
        if not _legacy.is_registered():
            self.skipTest(f"legacy dispatcher is unavailable: {_legacy.LOAD_ERROR}")
        info = rt._C.build_info()
        self.assertEqual(info["backend"], "torch")
        self.assertFalse(info["uses_dr_jit"])
        values = rt._C.contract_values()
        self.assertEqual(values["invalid_signed_id"], -1)
        self.assertEqual(values["ray_flags_all"], 7)


if __name__ == "__main__":
    unittest.main()
