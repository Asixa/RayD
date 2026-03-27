import importlib.util
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "rayd" / "torch" / "_interop_patch.py"


def load_interop_patch_module():
    spec = importlib.util.spec_from_file_location("rayd_torch_interop_patch_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InteropPatchTests(unittest.TestCase):
    def setUp(self):
        self.module = load_interop_patch_module()

    def _buggy_flatten(self, a, flat, desc, /):
        tp = type(a)
        desc.append(tp)
        if tp is list or tp is tuple:
            desc.append(len(a))
            for value in a:
                self._buggy_flatten(value, flat, desc)
        elif tp is dict:
            desc.append(tuple(a.keys()))
            for value in a.values():
                self._buggy_flatten(value, flat, desc)
        else:
            desc = getattr(tp, "DRJIT_STRUCT", None)
            if type(desc) is dict:
                for key in desc:
                    self._buggy_flatten(getattr(a, key), flat, desc)
            else:
                flat.append(a)

    def _buggy_unflatten(self, flat, desc, /):
        tp = desc.pop()
        if tp is list or tp is tuple:
            n = desc.pop()
            return tp(self._buggy_unflatten(flat, desc) for _ in range(n))
        if tp is dict:
            keys = desc.pop()
            return {key: self._buggy_unflatten(flat, desc) for key in keys}

        desc = getattr(tp, "DRJIT_STRUCT", None)
        if type(desc) is dict:
            result = tp()
            for key in desc:
                setattr(result, key, self._buggy_unflatten(flat, desc))
            return result
        return flat.pop()

    def _fixed_flatten(self, a, flat, desc, /):
        tp = type(a)
        desc.append(tp)
        if tp is list or tp is tuple:
            desc.append(len(a))
            for value in a:
                self._fixed_flatten(value, flat, desc)
        elif tp is dict:
            desc.append(tuple(a.keys()))
            for value in a.values():
                self._fixed_flatten(value, flat, desc)
        else:
            struct_desc = getattr(tp, "DRJIT_STRUCT", None)
            if type(struct_desc) is dict:
                for key in struct_desc:
                    self._fixed_flatten(getattr(a, key), flat, desc)
            else:
                flat.append(a)

    def _fixed_unflatten(self, flat, desc, /):
        tp = desc.pop()
        if tp is list or tp is tuple:
            n = desc.pop()
            return tp(self._fixed_unflatten(flat, desc) for _ in range(n))
        if tp is dict:
            keys = desc.pop()
            return {key: self._fixed_unflatten(flat, desc) for key in keys}

        struct_desc = getattr(tp, "DRJIT_STRUCT", None)
        if type(struct_desc) is dict:
            result = tp()
            for key in struct_desc:
                setattr(result, key, self._fixed_unflatten(flat, desc))
            return result
        return flat.pop()

    def test_detects_buggy_interop_helpers(self):
        fake_interop = types.SimpleNamespace(
            _flatten=self._buggy_flatten,
            _unflatten=self._buggy_unflatten,
        )
        self.assertTrue(self.module._needs_interop_patch(fake_interop))

    def test_skips_fixed_interop_helpers(self):
        fake_interop = types.SimpleNamespace(
            _flatten=self._fixed_flatten,
            _unflatten=self._fixed_unflatten,
        )
        self.assertFalse(self.module._needs_interop_patch(fake_interop))
        self.assertFalse(self.module.install_drjit_interop_patch(fake_interop))

    def test_install_replaces_buggy_helpers_once(self):
        fake_interop = types.SimpleNamespace(
            _flatten=self._buggy_flatten,
            _unflatten=self._buggy_unflatten,
        )
        self.assertTrue(self.module.install_drjit_interop_patch(fake_interop))
        self.assertIs(fake_interop._flatten, self.module._flatten_fixed)
        self.assertIs(fake_interop._unflatten, self.module._unflatten_fixed)
        self.assertFalse(self.module.install_drjit_interop_patch(fake_interop))


if __name__ == "__main__":
    unittest.main()
