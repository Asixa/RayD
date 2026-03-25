// Slang module that calls rayd interop functions via the cpp target.
// Uses accessor functions (raydItsT, raydItsValid, etc.) instead of
// direct field access to avoid slangc field-name mangling.

import rayd.slang.rayd;

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
