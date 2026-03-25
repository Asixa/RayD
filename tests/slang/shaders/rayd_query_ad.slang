// AD-enabled rayd queries from Slang.
// raydSceneIntersectAD runs forward + backward internally via Dr.Jit AD,
// returning both the intersection result and dt/d(ray).

import rayd.slang.rayd;

// Forward result
export float traceT(uint64_t sh, float ox, float oy, float oz,
                    float dx, float dy, float dz)
{
    RayDSceneHandle s = raydMakeSceneHandle(sh);
    RayDRay r = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersectionAD h = raydSceneIntersectAD(s, r);
    return raydItsADT(h);
}

export bool traceHit(uint64_t sh, float ox, float oy, float oz,
                     float dx, float dy, float dz)
{
    RayDSceneHandle s = raydMakeSceneHandle(sh);
    RayDRay r = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    RayDIntersectionAD h = raydSceneIntersectAD(s, r);
    return raydItsADValid(h);
}

// Gradient: dt/d(ray_origin)
export float dtdox(uint64_t sh, float ox, float oy, float oz,
                   float dx, float dy, float dz)
{
    RayDSceneHandle s = raydMakeSceneHandle(sh);
    RayDRay r = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydF3X(raydItsADDtDo(raydSceneIntersectAD(s, r)));
}

export float dtdoy(uint64_t sh, float ox, float oy, float oz,
                   float dx, float dy, float dz)
{
    RayDSceneHandle s = raydMakeSceneHandle(sh);
    RayDRay r = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydF3Y(raydItsADDtDo(raydSceneIntersectAD(s, r)));
}

export float dtdoz(uint64_t sh, float ox, float oy, float oz,
                   float dx, float dy, float dz)
{
    RayDSceneHandle s = raydMakeSceneHandle(sh);
    RayDRay r = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydF3Z(raydItsADDtDo(raydSceneIntersectAD(s, r)));
}

// Gradient: dt/d(ray_direction)
export float dtddx(uint64_t sh, float ox, float oy, float oz,
                   float dx, float dy, float dz)
{
    RayDSceneHandle s = raydMakeSceneHandle(sh);
    RayDRay r = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydF3X(raydItsADDtDd(raydSceneIntersectAD(s, r)));
}

export float dtddy(uint64_t sh, float ox, float oy, float oz,
                   float dx, float dy, float dz)
{
    RayDSceneHandle s = raydMakeSceneHandle(sh);
    RayDRay r = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydF3Y(raydItsADDtDd(raydSceneIntersectAD(s, r)));
}

export float dtddz(uint64_t sh, float ox, float oy, float oz,
                   float dx, float dy, float dz)
{
    RayDSceneHandle s = raydMakeSceneHandle(sh);
    RayDRay r = raydMakeRay(raydFloat3(ox, oy, oz), raydFloat3(dx, dy, dz));
    return raydF3Z(raydItsADDtDd(raydSceneIntersectAD(s, r)));
}
