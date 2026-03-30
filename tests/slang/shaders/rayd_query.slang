// Slang module that calls rayd interop functions via the cpp target.
// Uses accessor functions to avoid slangc field-name mangling.

import rayd_slang;

// ---- basic queries ----

export float traceRayT(uint64_t sceneHandle,
                       float ox, float oy, float oz,
                       float dx, float dy, float dz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz));
    Intersection hit = sceneIntersect(scene, ray);
    return itsT(hit);
}

export bool traceRayHit(uint64_t sceneHandle,
                        float ox, float oy, float oz,
                        float dx, float dy, float dz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz));
    Intersection hit = sceneIntersect(scene, ray);
    return itsValid(hit);
}

export bool shadowTest(uint64_t sceneHandle,
                       float ox, float oy, float oz,
                       float dx, float dy, float dz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz));
    return sceneShadowTest(scene, ray);
}

export bool nearestPointEdgeValid(uint64_t sceneHandle,
                                  float px, float py, float pz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    NearestPointEdge edge = sceneNearestEdgePoint(scene, float3(px, py, pz));
    return npeValid(edge);
}

export NearestPointEdge nearestPointEdgeStruct(uint64_t sceneHandle,
                                               float px, float py, float pz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    return sceneNearestEdgePoint(scene, float3(px, py, pz));
}

export int nearestPointGlobalEdgeId(uint64_t sceneHandle,
                                    float px, float py, float pz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    NearestPointEdge edge = sceneNearestEdgePoint(scene, float3(px, py, pz));
    return npeGlobalEdgeId(edge);
}

export int nearestRayGlobalEdgeId(uint64_t sceneHandle,
                                  float ox, float oy, float oz,
                                  float dx, float dy, float dz,
                                  float tmax)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz), tmax);
    NearestRayEdge edge = sceneNearestEdgeRay(scene, ray);
    return nreGlobalEdgeId(edge);
}

export NearestRayEdge nearestRayEdgeStruct(uint64_t sceneHandle,
                                           float ox, float oy, float oz,
                                           float dx, float dy, float dz,
                                           float tmax)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz), tmax);
    return sceneNearestEdgeRay(scene, ray);
}

export int getSceneEdgeCount(uint64_t sceneHandle)
{
    return sceneEdgeCount(makeSceneHandle(sceneHandle));
}

export bool getSceneEdgeMaskValue(uint64_t sceneHandle, int index)
{
    return sceneEdgeMaskValue(makeSceneHandle(sceneHandle), index);
}

export void setSceneEdgeMask(uint64_t sceneHandle, uint64_t maskPtr, int count)
{
    sceneSetEdgeMask(makeSceneHandle(sceneHandle), maskPtr, count);
}

export void syncScene(uint64_t sceneHandle)
{
    sceneSync(makeSceneHandle(sceneHandle));
}

// ---- queries returning processed scalar results (for gradient tests) ----

// Inverse depth: 1/t (useful as a differentiable signal)
export float inverseDepth(uint64_t sceneHandle,
                          float ox, float oy, float oz,
                          float dx, float dy, float dz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz));
    Intersection hit = sceneIntersect(scene, ray);
    float t = itsT(hit);
    return itsValid(hit) ? 1.0f / t : 0.0f;
}

// Hit-point z coordinate
export float hitPointZ(uint64_t sceneHandle,
                       float ox, float oy, float oz,
                       float dx, float dy, float dz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz));
    Intersection hit = sceneIntersect(scene, ray);
    Float3 p = itsP(hit);
    return itsValid(hit) ? f3Z(p) : 0.0f;
}

// Squared depth: t^2
export float squaredDepth(uint64_t sceneHandle,
                          float ox, float oy, float oz,
                          float dx, float dy, float dz)
{
    SceneHandle scene = makeSceneHandle(sceneHandle);
    Ray ray = makeRay(float3(ox, oy, oz), float3(dx, dy, dz));
    Intersection hit = sceneIntersect(scene, ray);
    float t = itsT(hit);
    return itsValid(hit) ? t * t : 0.0f;
}
