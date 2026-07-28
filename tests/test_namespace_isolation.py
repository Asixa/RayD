import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "contracts" / "operations.json").read_text())


def run_script(script):
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class NamespaceIsolationTests(unittest.TestCase):
    def test_parent_is_namespace_only(self):
        import rayd

        self.assertIsNone(rayd.__file__)
        self.assertFalse(hasattr(rayd, "Scene"))
        self.assertFalse(hasattr(rayd, "Mesh"))

    def test_backend_import_order_is_stable(self):
        for order in (("rayd.drjit", "rayd.torch"), ("rayd.torch", "rayd.drjit")):
            script = f"""
import importlib
a = importlib.import_module({order[0]!r})
b = importlib.import_module({order[1]!r})
assert a.backend_capabilities()["backend"] == {order[0].split('.')[-1]!r}
assert b.backend_capabilities()["backend"] == {order[1].split('.')[-1]!r}
"""
            result = run_script(script)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_torch_backend_does_not_import_drjit(self):
        result = run_script("import rayd.torch; import sys; assert 'drjit' not in sys.modules")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_drjit_backend_does_not_import_torch(self):
        result = run_script("import rayd.drjit; import sys; assert 'torch' not in sys.modules")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_obsolete_imports_fail(self):
        result = run_script("""
for name in ("raydn", "rayd.native"):
    try:
        __import__(name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"obsolete import still succeeds: {name}")
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_capability_manifests_match_contract(self):
        import rayd.drjit as rdd
        import rayd.torch as rdt

        required = set(CONTRACT["required_capability_keys"])
        self.assertEqual(set(rdd.backend_capabilities()), required)
        self.assertEqual(set(rdt.backend_capabilities()), required)

    def test_drjit_all_matches_the_built_extension(self):
        import rayd.drjit as rdd

        # tests/test_public_api_manifest.py derives __all__ statically from
        # rayd.cpp; this closes the loop against the extension actually built.
        declared = set(rdd.__all__)
        runtime = {name for name in dir(rdd._C) if not name.startswith("_")}
        runtime |= {"api_manifest", "backend_capabilities"}
        self.assertEqual(declared, runtime, sorted(declared ^ runtime))


if __name__ == "__main__":
    unittest.main()
