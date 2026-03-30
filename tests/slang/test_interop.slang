// Test: Slang code that imports the RayD interop module and calls query
// functions using proper accessor functions (no direct field access).

import rayd_slang;

export bool testIntersect(SceneHandle scene)
{
    Ray ray = makeRay(float3(0.25f, 0.25f, -1.0f), float3(0.0f, 0.0f, 1.0f));
    Intersection hit = sceneIntersect(scene, ray);
    return itsValid(hit);
}

export float testIntersectT(SceneHandle scene)
{
    Ray ray = makeRay(float3(0.25f, 0.25f, -1.0f), float3(0.0f, 0.0f, 1.0f));
    Intersection hit = sceneIntersect(scene, ray);
    return itsT(hit);
}

export bool testShadow(SceneHandle scene)
{
    Ray ray = makeRay(float3(0.25f, 0.25f, -1.0f), float3(0.0f, 0.0f, 1.0f));
    return sceneShadowTest(scene, ray);
}

export bool testNearestEdgePoint(SceneHandle scene)
{
    Float3 point = float3(0.25f, 0.1f, 0.0f);
    NearestPointEdge edge = sceneNearestEdgePoint(scene, point);
    return npeValid(edge);
}

export bool testNearestEdgeRay(SceneHandle scene)
{
    Ray ray = makeRay(float3(0.2f, 0.4f, 1.0f), float3(0.0f, 0.0f, -1.0f));
    NearestRayEdge edge = sceneNearestEdgeRay(scene, ray);
    return nreValid(edge);
}

export int testNearestEdgePointGlobalId(SceneHandle scene)
{
    Float3 point = float3(0.25f, 0.1f, 0.0f);
    NearestPointEdge edge = sceneNearestEdgePoint(scene, point);
    return npeGlobalEdgeId(edge);
}

export int testSceneEdgeCount(SceneHandle scene)
{
    return sceneEdgeCount(scene);
}

export Ray testCameraSampleRay(CameraHandle camera)
{
    Float2 sample = float2(0.5f, 0.5f);
    return cameraSampleRay(camera, sample);
}

export Float3 testFloat3()
{
    return float3(1.0f, 2.0f, 3.0f);
}

export Float2 testFloat2()
{
    return float2(4.0f, 5.0f);
}

export Ray testMakeRay()
{
    return makeRay(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f), 100.0f);
}
