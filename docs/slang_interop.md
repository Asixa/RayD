# Slang Interop

RayD provides a Slang interop layer that lets Slang code call RayD scene queries (intersection, shadow test, nearest-edge, camera sampling) from host-side C++.

## Files

| File | Purpose |
|---|---|
| `include/rayd/slang/rayd.slang` | Slang module — types, constructors, accessors, query wrappers |
| `include/rayd/slang/interop.h` | Full C++ header (includes rayd internals + drjit) |
| `include/rayd/slang/interop_types.h` | Lightweight C++ header (POD types + declarations only) |
| `src/slang_interop.cpp` | Compiled implementations of query functions (in `rayd_core`) |
| `rayd/slang/__init__.py` | Python helper: `include_dir()`, `load_module()` |

## Quick Start (Python)

```python
import rayd.slang as rs

# Compile a .slang file that calls raydSceneIntersect, link against rayd_core
m = rs.load_module("my_shader.slang")  # use rayd.slang.load_module, not slangtorch.loadModule

# Create a scene and get its handle
import rayd as rd, drjit.cuda as cuda
scene = rd.Scene()
scene.add_mesh(rd.Mesh(v, f))
scene.configure()

# Call the Slang function from Python
t = m.traceRayT(scene.slang_handle, 0.25, 0.25, -1.0, 0.0, 0.0, 1.0)
```

## Writing Slang Code

Import the module and use the provided constructors and accessor functions. **Do not access struct fields directly** — use the `raydIts*`, `raydF3*`, `raydRay*` accessors instead, because `slangc` mangles field names when targeting C++.

Two import styles are supported:

```slang
import rayd_slang;           // shorthand (recommended)
import rayd.slang.rayd;      // full path (also works)
```

```slang
import rayd_slang;

export float traceRayT(uint64_t sceneHandle,
                       float ox, float oy, float oz,
                       float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersection hit = raydSceneIntersect(scene, ray);
    return raydItsT(hit);       // NOT hit.t — use the accessor
}
```

### Available Functions

**Constructors:**
`raydFloat2(x,y)`, `raydFloat3(x,y,z)`, `raydMakeRay(o,d,tmax)`, `raydMakeSceneHandle(uint64)`, `raydMakeCameraHandle(uint64)`

**Intersection accessors:**
`raydItsValid(h)`, `raydItsT(h)`, `raydItsP(h)`, `raydItsN(h)`, `raydItsGeoN(h)`, `raydItsUV(h)`, `raydItsBarycentric(h)`, `raydItsShapeId(h)`, `raydItsPrimId(h)`

**Float/Ray accessors:**
`raydF2X/Y`, `raydF3X/Y/Z`, `raydRayO`, `raydRayD`, `raydRayTmax`

**Scene queries:**
`raydSceneIntersect(scene, ray)`, `raydSceneShadowTest(scene, ray)`, `raydSceneClosestEdgePoint(scene, point)`, `raydSceneClosestEdgeRay(scene, ray)`

**Camera queries:**
`raydCameraSampleRay(camera, sample)`, `raydCameraSamplePrimaryEdge(camera, s)`, `raydCameraSetResolution(camera, w, h)`, `raydCameraConfigure(camera)`, `raydCameraPrepareEdges(camera, scene)`

## Compilation Pipeline

`rayd.slang.load_module()` automates this entire pipeline:

```
my_shader.slang
  │  slangc -target cpp -I $(python -c "import rayd.slang; print(rayd.slang.include_dir())")
  ▼
my_shader.cpp                  (generated C++, uses rayd::slang:: types)
  │  prepend #include <rayd/slang/interop_types.h>
  │  append PYBIND11_MODULE(...) with m.def() for each export function
  ▼
my_shader_patched.cpp          (self-contained compilable source)
  │  torch.utils.cpp_extension.load(
  │      sources=[patched.cpp],
  │      extra_ldflags=[rayd_core, drjit-core, drjit-extra, nanothread, ...]
  │  )
  ▼
my_shader.pyd / .so            (Python extension module)
```

### Manual Compilation (without Python)

```bash
# 1. Generate C++
slangc my_shader.slang -target cpp -o my_shader.cpp \
    -I $(python -c "import rayd.slang; print(rayd.slang.include_dir())") \
    -ignore-capabilities

# 2. Compile and link against rayd_core
#    (you must provide your own main() or pybind11 module)
g++ -std=c++17 -o my_app my_shader.cpp my_main.cpp \
    -I$(python -c "import rayd.slang; print(rayd.slang.include_dir())") \
    -I$(python -c "import drjit; import pathlib; print(pathlib.Path(drjit.__file__).parent / 'include')") \
    -L<rayd_core_lib_dir> -lrayd_core \
    -L<drjit_lib_dir> -ldrjit-core -ldrjit-extra -lnanothread \
    -lcudart
```

## Architecture Notes

- The Slang module uses `__target_intrinsic(cpp, "rayd::slang::Foo")` to map Slang struct types onto C++ POD structs.
- All constructors and field accesses go through `__intrinsic_asm` on the `cpp` target, because `slangc` mangles Slang field names (e.g. `t` → `t_0`) which would not match the C++ struct definitions.
- The query functions (`scene_intersect`, etc.) are implemented in `src/slang_interop.cpp` and compiled into `rayd_core`. They convert a scalar Slang query into a 1-lane detached Dr.Jit call and extract the result back into a POD struct.
- Non-`cpp` targets (CUDA, HLSL, etc.) fall back to invalid sentinels — RayD's implementation depends on host-side Dr.Jit + OptiX and is not device-callable.

## Handles from Python

The `.slang_handle` property returns the raw C++ pointer as `uint64`, suitable for passing to Slang `export` functions:

```python
# Scene handle
scene = rd.Scene()
scene.add_mesh(mesh)
scene.configure()
scene_handle = scene.slang_handle  # uint64

# Camera handle
camera = rd.Camera(45.0)
camera.width, camera.height = 512, 512
camera.configure()
camera_handle = camera.slang_handle  # uint64
```

## Performance Characteristics

### Scalar-mode execution

All Slang interop queries execute in **scalar mode**: each call constructs a 1-lane Dr.Jit array, launches a GPU kernel, and synchronizes (`drjit::sync_thread()`). This means every call incurs full GPU launch + sync overhead.

For latency-sensitive applications (e.g., per-pixel queries in a rendering loop), batch your queries on the Dr.Jit side using the vectorized Python API instead of calling the Slang interop layer in a loop.

### Handle lifetime

`scene.slang_handle` and `camera.slang_handle` are raw C++ pointers encoded as `uint64`. They are **not** reference-counted. If the Python `Scene` or `Camera` object is garbage-collected while the handle is still in use, the pointer becomes dangling and any subsequent call through it is undefined behavior.

**Best practice:** Keep a Python reference to the `Scene`/`Camera` object alive for as long as you use its `slang_handle`.

## Known Workarounds

### `__requirePrelude` not supported by slangc 2026.4

The original `rayd.slang` used `__requirePrelude(R"(#include <rayd/slang/interop.h>)")` to inject the C++ header. This is not supported by slangc 2026.4. The workaround is to prepend the include when generating the patched C++ file (handled automatically by `load_module()`).

### slangc field-name mangling

`slangc` appends `_0`, `_1`, etc. to struct field names in generated C++ code, even for types annotated with `__target_intrinsic(cpp, ...)`. This means `hit.t` in Slang becomes `hit.t_0` in C++, which does not match the actual C++ struct. The solution is to **never access struct fields directly** in Slang — use the provided accessor functions (`raydItsT`, `raydF3X`, etc.) which use `__intrinsic_asm` to emit correct C++ field access.

### Windows non-English locale

MSVC outputs compilation messages in the system locale. The default OEM codec in Python cannot decode Chinese (or other non-Latin) text. `load_module()` patches `torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS` to `('utf-8', 'ignore')` automatically.

### cl.exe / ninja not on PATH

When running outside a Visual Studio Developer Command Prompt, `cl.exe` and `ninja` may not be on `PATH`. `load_module()` automatically locates them via `vswhere.exe` and the conda `Scripts/` directory.

### setuptools.msvc VCRuntimeRedist is None

On some Visual Studio installations, `setuptools.msvc.EnvironmentInfo.VCRuntimeRedist` returns `None`, causing `isfile(None)` to raise `TypeError`. `load_module()` patches `isfile` to tolerate `None`.

## Packaging

The build installs:

- the public `include/rayd/**` header tree (`.h` + `.slang` files)
- the linkable `rayd_core` library under `rayd/lib/`

This lets the site-packages root serve as both the include root for `#include <rayd/...>` and the module search root for `import rayd_slang;`.
