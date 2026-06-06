import subprocess
import sys
import textwrap
import unittest


class TorchNativeImportTests(unittest.TestCase):
    def test_raydtorch_import_does_not_import_drjit(self):
        code = textwrap.dedent(
            """
            import sys
            import raydtorch as rt
            print("drjit" in sys.modules)
            print(hasattr(rt, "Scene"))
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        lines = proc.stdout.strip().splitlines()
        self.assertEqual(lines[0], "False")
        self.assertEqual(lines[1], "True")

    def test_native_extension_loads(self):
        import raydtorch as rt
        self.assertTrue(hasattr(rt, "_C"))
        self.assertTrue(hasattr(rt._C, "build_info"))
        info = rt._C.build_info()
        self.assertEqual(info["backend"], "raydtorch-native")


if __name__ == "__main__":
    unittest.main()
