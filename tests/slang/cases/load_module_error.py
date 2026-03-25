"""Verify clear ImportError when slangtorch is not installed."""
import importlib.abc, json, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "slangtorch" or fullname.startswith("slangtorch."):
            raise ModuleNotFoundError("No module named 'slangtorch'")
        return None

sys.meta_path.insert(0, Block())
import rayd.slang as rs

try:
    rs.load_module("x.slang")
except ImportError as e:
    print(json.dumps({"error": str(e)}))
