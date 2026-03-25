# Slang Interop

RayD now includes a small Slang-facing host interop layer for the existing C++ API.

Files:

- `include/rayd/slang/interop.h`
- `include/rayd/slang/rayd.slang`

The design follows Slang's target-specific interop model for the `cpp` target:

- Slang-visible POD types (`RayDFloat3`, `RayDRay`, `RayDIntersection`, ...)
- opaque host handles (`RayDSceneHandle`, `RayDCameraHandle`)
- `__target_intrinsic(cpp, "...")` to map Slang types onto C++ wrapper types
- `__intrinsic_asm` calls into `rayd::slang::*` helper functions

## Scope

This layer is intentionally narrow.

It is meant for **host-side scalar queries** from Slang-generated C++ code:

- scene intersection
- shadow test
- closest-edge point query
- closest-edge ray query
- camera sample-ray
- camera primary-edge preparation and sampling

It does **not** expose the full Dr.Jit array API, batched queries, or differentiable AD plumbing to Slang. Internally the wrapper converts a scalar Slang query into a 1-lane detached Dr.Jit query and then extracts the result back into POD structs.

## Host-side usage

Create and configure RayD objects in normal C++, then pass handles to generated Slang/C++ entry points:

```cpp
#include <rayd/slang/interop.h>
#include <rayd/scene/scene.h>
#include <rayd/camera.h>

rayd::Scene scene;
// ... add meshes and configure ...

rayd::PerspectiveCamera camera;
// ... set resolution / configure ...

rayd::slang::SceneHandle scene_handle = rayd::slang::make_scene_handle(scene);
rayd::slang::CameraHandle camera_handle = rayd::slang::make_camera_handle(camera);
```

The generated C++ should link against the exported `rayd_core` library from this build, because the interop helpers call into the existing RayD C++ implementation rather than re-implementing scene logic in the header.

## Slang-side usage

Import the module and query through the wrapper functions:

```slang
import rayd.slang.rayd;

RayDRay ray = raydMakeRay(
    raydFloat3(0.25f, 0.25f, -1.0f),
    raydFloat3(0.0f, 0.0f, 1.0f));

RayDIntersection hit = raydSceneIntersect(sceneHandle, ray);
if (hit.valid)
{
    float t = hit.t;
    RayDFloat3 p = hit.p;
}
```

For camera-side sampling:

```slang
raydCameraSetResolution(cameraHandle, 512, 512);
raydCameraConfigure(cameraHandle);
raydCameraPrepareEdges(cameraHandle, sceneHandle);

RayDRay primary = raydCameraSampleRay(cameraHandle, raydFloat2(0.5f, 0.5f));
RayDPrimaryEdgeSample edgeSample = raydCameraSamplePrimaryEdge(cameraHandle, 0.25f);
```

## Non-`cpp` targets

The module is intentionally wired for Slang's `cpp` target. Other targets fall back to invalid sentinel values or `false`, because RayD's current C++ implementation depends on host-side Dr.Jit and OptiX objects and is not device-callable through Slang.

## Packaging

The build now installs:

- the public `include/rayd/**` header tree into the package root
- the Slang module under `rayd/slang/rayd.slang`
- the linkable `rayd_core` library under `rayd/lib/`

This lets a site-packages root act as the include root for `#include <rayd/...>` and as the module search root for `import rayd.slang.rayd;`.
