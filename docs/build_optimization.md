# Build Optimization

RayD now defaults to a production-oriented build.

## Defaults

The default build enables:

- OptiX module optimization level `LEVEL_3`
- OptiX exception flags `NONE`
- Release-oriented host C++ optimization
- `nanobind` `NOMINSIZE`, which disables nanobind's default `/Os` or `-Os` bias in optimized builds

In CMake, the defaults are:

```cmake
RAYD_OPTIX_PRODUCTION_CONFIG=ON
RAYD_NANOBIND_NOMINSIZE=ON
RAYD_ENABLE_LTO=OFF
```

Meaning:

- `RAYD_OPTIX_PRODUCTION_CONFIG=ON`
  - `RAYD_OPTIX_MODULE_OPT_LEVEL = 0x2343`
  - `RAYD_OPTIX_EXCEPTION_FLAGS = 0`
- `RAYD_NANOBIND_NOMINSIZE=ON`
  - removes nanobind's optimized-build `/Os` or `-Os`
  - lets the package stay speed-oriented instead of size-oriented

## Where It Is built

- CMake options live in [CMakeLists.txt](/E:/Code/psdr-jit/CMakeLists.txt)
- OptiX module config is consumed in [scene_optix.cpp](/E:/Code/psdr-jit/src/scene/scene_optix.cpp)
- OptiX target constants, stubs, and runtime probe declarations are defined in [optix.h](/E:/Code/psdr-jit/include/rayd/optix.h)

## Debug-Friendly Build

If you need the old debug-oriented OptiX behavior, turn production mode off.

PowerShell example:

```powershell
$env:CMAKE_ARGS='-DRAYD_OPTIX_PRODUCTION_CONFIG=OFF -DRAYD_NANOBIND_NOMINSIZE=OFF'
python -m pip install --no-build-isolation -ve .
```

That switches to:

- OptiX optimization level `LEVEL_0`
- OptiX exception flags `STACK_OVERFLOW | TRACE_DEPTH | DEBUG`
- size-oriented nanobind defaults restored

To go back to the default production build:

```powershell
$env:CMAKE_ARGS='-DRAYD_OPTIX_PRODUCTION_CONFIG=ON -DRAYD_NANOBIND_NOMINSIZE=ON'
python -m pip install --no-build-isolation -ve .
```

## Whole-Package C++ Optimization Status

For the host C++ package on Windows:

- Release already uses `/O2`
- The main extra issue was that `nanobind_add_module()` also injected `/Os`
- With `RAYD_NANOBIND_NOMINSIZE=ON`, that size bias is removed

So the package is now aligned with a speed-first build by default.

`RAYD_ENABLE_LTO` is left `OFF` by default because it needs separate validation against the current mixed C++/CUDA build setup and link times.
