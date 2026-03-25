// Test: Slang code that imports the RayD interop module and calls query
// functions using proper accessor functions (no direct field access).

import rayd_slang;

export bool testIntersect(RayDSceneHandle scene)
{
    RayDRay ray = raydMakeRay(raydFloat3(0.25f, 0.25f, -1.0f), raydFloat3(0.0f, 0.0f, 1.0f));
    RayDIntersection hit = raydSceneIntersect(scene, ray);
    return raydItsValid(hit);
}

export float testIntersectT(RayDSceneHandle scene)
{
    RayDRay ray = raydMakeRay(raydFloat3(0.25f, 0.25f, -1.0f), raydFloat3(0.0f, 0.0f, 1.0f));
    RayDIntersection hit = raydSceneIntersect(scene, ray);
    return raydItsT(hit);
}

export bool testShadow(RayDSceneHandle scene)
{
    RayDRay ray = raydMakeRay(raydFloat3(0.25f, 0.25f, -1.0f), raydFloat3(0.0f, 0.0f, 1.0f));
    return raydSceneShadowTest(scene, ray);
}

export bool testClosestEdgePoint(RayDSceneHandle scene)
{
    RayDFloat3 point = raydFloat3(0.25f, 0.1f, 0.0f);
    RayDNearestPointEdge edge = raydSceneClosestEdgePoint(scene, point);
    return edge.valid;  // ok: only accessed on non-cpp targets via default
}

export bool testClosestEdgeRay(RayDSceneHandle scene)
{
    RayDRay ray = raydMakeRay(raydFloat3(0.2f, 0.4f, 1.0f), raydFloat3(0.0f, 0.0f, -1.0f));
    RayDNearestRayEdge edge = raydSceneClosestEdgeRay(scene, ray);
    return edge.valid;
}

export RayDRay testCameraSampleRay(RayDCameraHandle camera)
{
    RayDFloat2 sample = raydFloat2(0.5f, 0.5f);
    return raydCameraSampleRay(camera, sample);
}

export RayDFloat3 testFloat3()
{
    return raydFloat3(1.0f, 2.0f, 3.0f);
}

export RayDFloat2 testFloat2()
{
    return raydFloat2(4.0f, 5.0f);
}

export RayDRay testMakeRay()
{
    return raydMakeRay(raydFloat3(0.0f, 0.0f, 0.0f), raydFloat3(0.0f, 0.0f, 1.0f), 100.0f);
}
