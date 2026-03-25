// Slang module that calls rayd interop functions via the cpp target.
// Uses accessor functions to avoid slangc field-name mangling.

import rayd.slang.rayd;

// ---- basic queries ----

export float traceRayT(uint64_t sceneHandle,
                       float ox, float oy, float oz,
                       float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersection hit = raydSceneIntersect(scene, ray);
    return raydItsT(hit);
}

export bool traceRayHit(uint64_t sceneHandle,
                        float ox, float oy, float oz,
                        float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersection hit = raydSceneIntersect(scene, ray);
    return raydItsValid(hit);
}

export bool shadowTest(uint64_t sceneHandle,
                       float ox, float oy, float oz,
                       float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydSceneShadowTest(scene, ray);
}

// ---- queries returning processed scalar results (for gradient tests) ----

// Inverse depth: 1/t (useful as a differentiable signal)
export float inverseDepth(uint64_t sceneHandle,
                          float ox, float oy, float oz,
                          float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersection hit = raydSceneIntersect(scene, ray);
    float t = raydItsT(hit);
    return raydItsValid(hit) ? 1.0f / t : 0.0f;
}

// Hit-point z coordinate
export float hitPointZ(uint64_t sceneHandle,
                       float ox, float oy, float oz,
                       float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersection hit = raydSceneIntersect(scene, ray);
    RayDFloat3 p = raydItsP(hit);
    return raydItsValid(hit) ? raydF3Z(p) : 0.0f;
}

// Squared depth: t^2
export float squaredDepth(uint64_t sceneHandle,
                          float ox, float oy, float oz,
                          float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersection hit = raydSceneIntersect(scene, ray);
    float t = raydItsT(hit);
    return raydItsValid(hit) ? t * t : 0.0f;
}
