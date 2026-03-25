"""rayd.slang -- Slang / slangtorch interop utilities for RayD.

This module provides helpers for using RayD alongside slangtorch:

- ``include_dir()`` returns the directory containing the C++ interop header
  and the Slang module so that custom ``.slang`` files can
  ``import rayd.slang.rayd;``
- ``shader_dir()`` returns the directory containing bundled Slang shaders
  for common operations (shading, ray generation, etc.)
- ``load_module(filename, **kw)`` is a thin wrapper around
  ``slangtorch.loadModule`` that automatically adds the RayD include path.

The C++ interop layer (``include/rayd/slang/interop.h`` and
``include/rayd/slang/rayd.slang``) targets Slang's **cpp** backend for
host-side scalar queries.  ``slangtorch`` compiles for the **CUDA** backend,
so the host-side query functions (``raydSceneIntersect``, etc.) are **not**
callable from slangtorch kernels.  The recommended pattern is:

1. Use ``rayd.torch`` for geometry queries (intersection, nearest-edge, etc.)
2. Pass the resulting ``torch.Tensor`` data to a slangtorch-compiled Slang
   kernel for custom GPU processing (shading, field computation, etc.)
3. Gradients flow through both ``rayd.torch`` and ``slangtorch`` via
   ``torch.autograd``.

Requires: ``pip install slangtorch``
"""

from pathlib import Path as _Path

_PACKAGE_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]


def include_dir() -> _Path:
    """Return the directory to add to Slang include paths.

    For an installed wheel this is the package root (``rayd/``).
    For a development checkout this is ``<repo>/include/``.
    The returned path lets Slang resolve ``import rayd.slang.rayd;``.
    """
    # Development layout: include/rayd/slang/rayd.slang exists
    dev = _REPO_ROOT / "include"
    if (dev / "rayd" / "slang" / "rayd.slang").is_file():
        return dev
    # Installed layout: rayd/slang/rayd.slang sits next to this __init__.py
    installed = _PACKAGE_DIR.parent
    if (_PACKAGE_DIR / "rayd.slang").is_file():
        return installed
    raise FileNotFoundError(
        "Cannot locate rayd.slang module.  "
        "Ensure the package is properly installed or you are running from "
        "the repository root."
    )


def shader_dir() -> _Path:
    """Return the directory containing bundled Slang shaders."""
    d = _PACKAGE_DIR / "shaders"
    if not d.is_dir():
        raise FileNotFoundError(f"Shader directory not found: {d}")
    return d


def load_module(filename, *, include_paths=None, **kwargs):
    """Load a Slang module via ``slangtorch.loadModule`` with RayD paths.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.slang`` file.
    include_paths : list[str], optional
        Additional include paths.  The RayD include directory is prepended
        automatically.
    **kwargs
        Forwarded to ``slangtorch.loadModule``.

    Returns
    -------
    The compiled module object returned by ``slangtorch.loadModule``.
    """
    try:
        import slangtorch
    except ImportError:
        raise ImportError(
            "slangtorch is required for rayd.slang.load_module().  "
            "Install it with: pip install slangtorch"
        ) from None

    rayd_include = str(include_dir())
    paths = [rayd_include] + (include_paths or [])
    return slangtorch.loadModule(str(filename), includePaths=paths, **kwargs)
