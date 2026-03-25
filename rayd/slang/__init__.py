"""rayd.slang -- compile Slang modules that call RayD scene queries.

Public API
----------
``include_dir()``
    Path to pass as ``-I`` to ``slangc`` so Slang can resolve
    ``import rayd.slang.rayd;``.

``load_module(filename)``
    Compiles a ``.slang`` file with ``slangc -target cpp``, auto-generates
    ``pybind11`` bindings for every ``export`` function, and links the result
    against ``rayd_core`` + drjit.  Returns a Python extension module whose
    attributes are the exported functions.

``load_module(filename, link_rayd=False)``
    Falls back to ``slangtorch.loadModule`` for pure-CUDA kernels that do
    **not** call into the RayD interop layer.

Requires ``pip install slangtorch`` (provides ``slangc``).
"""

from __future__ import annotations

import glob as _glob
import hashlib as _hashlib
import os as _os
import re as _re
import subprocess as _subprocess
import sys as _sys
from pathlib import Path as _Path

_PACKAGE_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]

# pybind11 struct bindings injected into every compiled module.
_PYBIND11_STRUCT_BINDINGS = r"""
    namespace py = pybind11;
    using namespace rayd::slang;
    py::class_<Float2>(m, "Float2")
        .def_readonly("x", &Float2::x)
        .def_readonly("y", &Float2::y);
    py::class_<Float3>(m, "Float3")
        .def_readonly("x", &Float3::x)
        .def_readonly("y", &Float3::y)
        .def_readonly("z", &Float3::z);
    py::class_<Intersection>(m, "Intersection")
        .def_readonly("valid", &Intersection::valid)
        .def_readonly("t", &Intersection::t)
        .def_readonly("p", &Intersection::p)
        .def_readonly("n", &Intersection::n)
        .def_readonly("geo_n", &Intersection::geo_n)
        .def_readonly("uv", &Intersection::uv)
        .def_readonly("barycentric", &Intersection::barycentric)
        .def_readonly("shape_id", &Intersection::shape_id)
        .def_readonly("prim_id", &Intersection::prim_id);
    py::class_<IntersectionAD>(m, "IntersectionAD")
        .def_readonly("valid", &IntersectionAD::valid)
        .def_readonly("t", &IntersectionAD::t)
        .def_readonly("p", &IntersectionAD::p)
        .def_readonly("n", &IntersectionAD::n)
        .def_readonly("geo_n", &IntersectionAD::geo_n)
        .def_readonly("uv", &IntersectionAD::uv)
        .def_readonly("barycentric", &IntersectionAD::barycentric)
        .def_readonly("shape_id", &IntersectionAD::shape_id)
        .def_readonly("prim_id", &IntersectionAD::prim_id)
        .def_readonly("dt_do", &IntersectionAD::dt_do)
        .def_readonly("dt_dd", &IntersectionAD::dt_dd);
    py::class_<Ray>(m, "Ray")
        .def_readonly("o", &Ray::o)
        .def_readonly("d", &Ray::d)
        .def_readonly("tmax", &Ray::tmax);
    py::class_<NearestPointEdge>(m, "NearestPointEdge")
        .def_readonly("valid", &NearestPointEdge::valid)
        .def_readonly("distance", &NearestPointEdge::distance)
        .def_readonly("point", &NearestPointEdge::point)
        .def_readonly("edge_t", &NearestPointEdge::edge_t)
        .def_readonly("edge_point", &NearestPointEdge::edge_point)
        .def_readonly("shape_id", &NearestPointEdge::shape_id)
        .def_readonly("edge_id", &NearestPointEdge::edge_id)
        .def_readonly("is_boundary", &NearestPointEdge::is_boundary);
    py::class_<NearestRayEdge>(m, "NearestRayEdge")
        .def_readonly("valid", &NearestRayEdge::valid)
        .def_readonly("distance", &NearestRayEdge::distance)
        .def_readonly("ray_t", &NearestRayEdge::ray_t)
        .def_readonly("point", &NearestRayEdge::point)
        .def_readonly("edge_t", &NearestRayEdge::edge_t)
        .def_readonly("edge_point", &NearestRayEdge::edge_point)
        .def_readonly("shape_id", &NearestRayEdge::shape_id)
        .def_readonly("edge_id", &NearestRayEdge::edge_id)
        .def_readonly("is_boundary", &NearestRayEdge::is_boundary);
    py::class_<PrimaryEdgeSample>(m, "PrimaryEdgeSample")
        .def_readonly("valid", &PrimaryEdgeSample::valid)
        .def_readonly("x_dot_n", &PrimaryEdgeSample::x_dot_n)
        .def_readonly("idx", &PrimaryEdgeSample::idx)
        .def_readonly("ray_n", &PrimaryEdgeSample::ray_n)
        .def_readonly("ray_p", &PrimaryEdgeSample::ray_p)
        .def_readonly("pdf", &PrimaryEdgeSample::pdf);
"""

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def include_dir() -> _Path:
    """Directory to pass as ``-I`` to ``slangc``.

    Development layout → ``<repo>/include``
    Installed wheel    → package root (``rayd/``)
    """
    dev = _REPO_ROOT / "include"
    if (dev / "rayd" / "slang" / "rayd.slang").is_file():
        return dev
    installed = _PACKAGE_DIR.parent
    if (_PACKAGE_DIR / "rayd.slang").is_file():
        return installed
    raise FileNotFoundError(
        "Cannot locate rayd.slang module. "
        "Ensure the package is properly installed or you are running "
        "from the repository root.")


def _slangc() -> str:
    import slangtorch
    ext = ".exe" if _sys.platform == "win32" else ""
    p = _Path(slangtorch.__file__).parent / "bin" / f"slangc{ext}"
    if not p.is_file():
        raise FileNotFoundError(f"slangc not found at {p}")
    return str(p)


def _lib_dir() -> _Path:
    """Directory containing ``rayd_core`` static library."""
    # Development: build/<tag>/lib/Release/  (Windows) or lib/ (Linux)
    for pattern in ("*/lib/Release", "*/lib"):
        for d in sorted((_REPO_ROOT / "build").glob(pattern), reverse=True):
            if list(d.glob("rayd_core*")):
                return d
    # Installed: rayd/lib/
    d = _PACKAGE_DIR.parent / "lib"
    if d.is_dir() and list(d.glob("rayd_core*")):
        return d
    raise FileNotFoundError(
        "Cannot locate rayd_core library. "
        "Ensure the package is built (pip install -e .) or installed.")


def _drjit_dir() -> _Path:
    import drjit
    return _Path(drjit.__file__).resolve().parent


# ---------------------------------------------------------------------------
# load_module
# ---------------------------------------------------------------------------

def load_module(
    filename: str | _Path,
    *,
    link_rayd: bool = True,
    include_paths: list[str] | None = None,
    **kwargs,
):
    """Compile a ``.slang`` file into a Python extension module.

    Parameters
    ----------
    filename : path-like
        The ``.slang`` source file.
    link_rayd : bool
        ``True`` (default): compile to C++ host code via ``slangc`` and link
        against ``rayd_core`` so that ``raydSceneIntersect`` etc. work.
        ``False``: delegate to ``slangtorch.loadModule`` for CUDA kernels.
    include_paths : list[str], optional
        Additional ``-I`` paths for ``slangc``.
    **kwargs
        Forwarded to ``slangtorch.loadModule`` when *link_rayd=False*.
    """
    try:
        import slangtorch  # noqa: F401
    except ImportError:
        raise ImportError(
            "slangtorch is required for rayd.slang.load_module(). "
            "Install it with: pip install slangtorch") from None

    _apply_platform_workarounds()

    rayd_inc = str(include_dir())
    paths = [rayd_inc] + (include_paths or [])

    if not link_rayd:
        return slangtorch.loadModule(str(filename), includePaths=paths, **kwargs)

    return _compile_host_module(str(filename), paths)


# ---------------------------------------------------------------------------
# Core pipeline: slangc → patch C++ → torch.utils.cpp_extension.load
# ---------------------------------------------------------------------------

def _compile_host_module(slang_file: str, inc_paths: list[str]):
    import torch.utils.cpp_extension

    slang_file = _os.path.abspath(slang_file)
    stem = _Path(slang_file).stem

    hasher = _hashlib.md5(_Path(slang_file).read_bytes())
    _inc = include_dir()
    for dep in ("rayd/slang/rayd.slang", "rayd/slang/interop_types.h"):
        dep_path = _inc / dep
        if dep_path.is_file():
            hasher.update(dep_path.read_bytes())
    content_hash = hasher.hexdigest()[:12]
    cache = _Path(slang_file).parent / ".rayd_slang_cache" / f"{stem}_{content_hash}"
    cache.mkdir(parents=True, exist_ok=True)

    mod_name = f"rayd_slang_{stem}_{content_hash}"
    cpp_raw = cache / f"{stem}.cpp"
    cpp_patched = cache / f"{stem}_patched.cpp"

    # ---- Step 1: slangc -target cpp ----
    if not cpp_raw.exists():
        cmd = [_slangc(), slang_file, "-target", "cpp",
               "-o", str(cpp_raw), "-ignore-capabilities"]
        for p in inc_paths:
            cmd += ["-I", p]
        r = _subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"slangc failed on {slang_file}:\n{r.stderr}")

    # ---- Step 2: patch generated C++ — prepend header, append pybind11 bindings ----
    if not cpp_patched.exists():
        gen = cpp_raw.read_text(encoding="utf-8", errors="replace")

        # Extract exported functions (slangc mangles Name → Name_0).
        # They appear after the huge Slang prelude, so skip the first half.
        func_re = _re.compile(
            r'^([\w:]+)\s+(\w+_\d+)\s*\(([^)]*)\)\s*$', _re.MULTILINE)
        funcs = [(m.group(1), m.group(2))
                 for m in func_re.finditer(gen)
                 if m.start() > len(gen) // 2]

        fn_defs = "\n".join(
            f'    m.def("{_re.sub(r"_[0-9]+$", "", fn)}", &{fn});'
            for _, fn in funcs)

        # Register POD structs so pybind11 can return them from functions.
        struct_defs = _PYBIND11_STRUCT_BINDINGS

        cpp_patched.write_text(
            f"#include <rayd/slang/interop_types.h>\n"
            f"{gen}\n"
            f"#include <pybind11/pybind11.h>\n"
            f"PYBIND11_MODULE({mod_name}, m) {{\n"
            f"{struct_defs}\n{fn_defs}\n}}\n",
            encoding="utf-8")

    # ---- Step 3: collect include / lib / flags ----
    rayd_inc = str(include_dir())
    drjit = _drjit_dir()
    drjit_inc = str(drjit / "include")
    drjit_lib = str(drjit)
    lib_dir = str(_lib_dir())

    extra_include = [rayd_inc, drjit_inc]
    cuda_path = _os.environ.get("CUDA_PATH", "")
    if cuda_path:
        cuda_inc = _Path(cuda_path) / "include"
        if cuda_inc.is_dir():
            extra_include.append(str(cuda_inc))

    if _sys.platform == "win32":
        cflags = ["/std:c++17", "/wd4624", "/wd4267", "/wd4244"]
        ldflags = [f"/LIBPATH:{lib_dir}", f"/LIBPATH:{drjit_lib}",
                   "rayd_core.lib", "drjit-core.lib",
                   "drjit-extra.lib", "nanothread.lib", "version.lib"]
        if cuda_path:
            cuda_lib = str(_Path(cuda_path) / "lib" / "x64")
            if _os.path.isdir(cuda_lib):
                ldflags += [f"/LIBPATH:{cuda_lib}", "cudart.lib"]
    else:
        cflags = ["-std=c++17", "-fPIC"]
        ldflags = [f"-L{lib_dir}", f"-L{drjit_lib}",
                   "-lrayd_core", "-ldrjit-core",
                   "-ldrjit-extra", "-lnanothread",
                   f"-Wl,-rpath,{lib_dir}", f"-Wl,-rpath,{drjit_lib}"]
        if cuda_path:
            cuda_lib = str(_Path(cuda_path) / "lib64")
            if _os.path.isdir(cuda_lib):
                ldflags += [f"-L{cuda_lib}", "-lcudart",
                            f"-Wl,-rpath,{cuda_lib}"]

    # ---- Step 4: compile & load ----
    return torch.utils.cpp_extension.load(
        name=mod_name,
        sources=[str(cpp_patched)],
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        extra_include_paths=extra_include,
        build_directory=str(cache),
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Platform workarounds
# ---------------------------------------------------------------------------

def _apply_platform_workarounds():
    """One-time fixes applied before any compilation."""
    if _sys.platform != "win32":
        return

    # 1. Conda env Scripts/ may not be on PATH (ninja lives there).
    scripts = _os.path.join(_os.path.dirname(_sys.executable), "Scripts")
    if scripts not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = scripts + _os.pathsep + _os.environ["PATH"]

    # 2. cl.exe may not be on PATH when running outside Developer Prompt.
    try:
        _subprocess.check_output(["where", "cl"], stderr=_subprocess.DEVNULL)
    except Exception:
        _add_msvc_to_path()

    # 3. MSVC on Chinese-locale Windows outputs non-ASCII that the default
    #    OEM codec cannot decode.
    try:
        import torch.utils.cpp_extension as _ce
        _ce.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
    except Exception:
        pass

    # 4. setuptools.msvc may return None for VCRuntimeRedist on some setups.
    try:
        import setuptools.msvc as _msvc
        _orig = _os.path.isfile
        _msvc.isfile = lambda p: False if p is None else _orig(p)
    except Exception:
        pass


def _add_msvc_to_path():
    """Find the latest MSVC cl.exe and add it to PATH."""
    try:
        vswhere = _os.path.join(
            _os.environ.get("ProgramFiles(x86)", ""),
            "Microsoft Visual Studio", "Installer", "vswhere.exe")
        r = _subprocess.run(
            [vswhere, "-latest", "-products", "*",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=10)
        vs = r.stdout.strip()
        if not vs:
            return
        bins = _glob.glob(_os.path.join(
            vs, "**", "VC", "Tools", "MSVC", "**",
            "bin", "Hostx64", "x64"), recursive=True)
        if bins:
            bins.sort(key=_os.path.getmtime, reverse=True)
            _os.environ["PATH"] = bins[0] + _os.pathsep + _os.environ["PATH"]
    except Exception:
        pass


