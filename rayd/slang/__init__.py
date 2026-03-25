"""rayd.slang -- Slang interop utilities for RayD.

``include_dir()`` returns the path to add with ``-I`` when compiling Slang
code that does ``import rayd.slang.rayd;``.

``load_module()`` is a thin ``slangtorch.loadModule`` wrapper that adds the
RayD include path automatically (requires ``pip install slangtorch``).
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path

_PACKAGE_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]


def include_dir() -> _Path:
    """Return the directory to pass as ``-I`` to ``slangc``.

    For a development checkout this is ``<repo>/include/``.
    For an installed wheel this is the package root (``rayd/``).
    Either way Slang can resolve ``import rayd.slang.rayd;``.
    """
    dev = _REPO_ROOT / "include"
    if (dev / "rayd" / "slang" / "rayd.slang").is_file():
        return dev
    installed = _PACKAGE_DIR.parent
    if (_PACKAGE_DIR / "rayd.slang").is_file():
        return installed
    raise FileNotFoundError(
        "Cannot locate rayd.slang module.  "
        "Ensure the package is properly installed or you are running from "
        "the repository root."
    )


def load_module(filename, *, include_paths=None, **kwargs):
    """Load a Slang module via ``slangtorch.loadModule`` with RayD paths.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.slang`` file.
    include_paths : list[str], optional
        Extra include paths.  The RayD include directory is prepended
        automatically.
    **kwargs
        Forwarded to ``slangtorch.loadModule``.
    """
    try:
        import slangtorch
    except ImportError:
        raise ImportError(
            "slangtorch is required for rayd.slang.load_module().  "
            "Install it with: pip install slangtorch"
        ) from None

    # Windows workarounds for non-English MSVC locale and missing ninja on PATH.
    if _sys.platform == "win32":
        scripts = _os.path.join(_os.path.dirname(_sys.executable), "Scripts")
        if scripts not in _os.environ.get("PATH", ""):
            _os.environ["PATH"] = scripts + _os.pathsep + _os.environ.get("PATH", "")
        try:
            import torch.utils.cpp_extension as _ce
            _ce.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')
        except Exception:
            pass
        try:
            import setuptools.msvc as _msvc
            _orig = _os.path.isfile
            _msvc.isfile = lambda p: False if p is None else _orig(p)
        except Exception:
            pass

    paths = [str(include_dir())] + (include_paths or [])
    return slangtorch.loadModule(str(filename), includePaths=paths, **kwargs)
