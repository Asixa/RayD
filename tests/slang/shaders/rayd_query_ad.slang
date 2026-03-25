// Differentiable rayd query from Slang.
//
// Provides a forward/backward pair:
//   traceAD       — returns IntersectionAD (forward result + analytic gradients)
//   traceT_fwd    — returns just t (for use as the forward pass)
//   traceT_bwd    — given upstream gradient dresult, returns d(loss)/d(ray params)
//
// Both forward and backward logic are in Slang/C++.
// Python only calls the compiled host functions — no Python-side autograd wrapper needed.

import rayd.slang.rayd;

// ---- Full AD result (one-shot forward + backward) ----

export RayDIntersectionAD traceAD(
    uint64_t sceneHandle,
    float ox, float oy, float oz,
    float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydSceneIntersectAD(scene, ray);
}

// ---- Forward: returns t ----

export float traceTFwd(
    uint64_t sceneHandle,
    float ox, float oy, float oz,
    float dx, float dy, float dz)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydItsADT(raydSceneIntersectAD(scene, ray));
}

// ---- Backward: given dresult = d(loss)/d(t), returns d(loss)/d(ray_params) ----
// Returns a RayDFloat3 pair:  grad_origin and grad_direction.
// grad_origin  = dresult * dt_do
// grad_direction = dresult * dt_dd

export RayDFloat3 traceTBwdOrigin(
    uint64_t sceneHandle,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float dresult)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersectionAD hit = raydSceneIntersectAD(scene, ray);
    RayDFloat3 dt_do = raydItsADDtDo(hit);
    return raydFloat3(
        dresult * raydF3X(dt_do),
        dresult * raydF3Y(dt_do),
        dresult * raydF3Z(dt_do));
}

export RayDFloat3 traceTBwdDirection(
    uint64_t sceneHandle,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float dresult)
{
    RayDSceneHandle scene = raydMakeSceneHandle(sceneHandle);
    RayDRay ray = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersectionAD hit = raydSceneIntersectAD(scene, ray);
    RayDFloat3 dt_dd = raydItsADDtDd(hit);
    return raydFloat3(
        dresult * raydF3X(dt_dd),
        dresult * raydF3Y(dt_dd),
        dresult * raydF3Z(dt_dd));
}
