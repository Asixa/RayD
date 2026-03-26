// Differentiable rayd query from Slang.
//
//   traceAD       — returns IntersectionAD (forward result + analytic gradients dt_do, dt_dd)
//   traceTFwd     — returns just t (convenience for forward-only use)
//
// Python backward pass uses traceAD to get both dt_do and dt_dd in a single call.

import rayd_slang;

// ---- Full AD result (one-shot forward + backward) ----

export IntersectionAD traceAD(
    uint64_t sceneHandle,
    float ox, float oy, float oz,
    float dx, float dy, float dz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz));
    return sceneIntersectAD(scene, ray);
}

// ---- Forward: returns t ----

export float traceTFwd(
    uint64_t sceneHandle,
    float ox, float oy, float oz,
    float dx, float dy, float dz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz));
    return itsADT(sceneIntersectAD(scene, ray));
}
