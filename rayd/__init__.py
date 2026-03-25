import os as _os
import pathlib as _pathlib
import sys as _sys

import drjit as _drjit

if _sys.platform == "win32" and hasattr(_os, "add_dll_directory"):
    _os.add_dll_directory(str(_pathlib.Path(_drjit.__file__).resolve().parent))

from .rayd import *  # noqa: F401,F403

try:
    del rayd
except NameError:
    pass
