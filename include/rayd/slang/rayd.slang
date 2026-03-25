// Slang interop layer for RayD's host-side C++ API.
// Intended for Slang's C++ target. Other targets return invalid sentinels.

// NOTE: When compiling the generated C++ output, the build system must
// ensure <rayd/slang/interop.h> is on the include path and that the
// generated source includes it (e.g. via a force-include flag or by
// prepending the directive to the generated .cpp file).

__target_intrinsic(cpp, "rayd::slang::Float2")
struct RayDFloat2
{
    float x;
    float y;
};

__target_intrinsic(cpp, "rayd::slang::Float3")
struct RayDFloat3
{
    float x;
    float y;
    float z;
};

__target_intrinsic(cpp, "rayd::slang::Ray")
struct RayDRay
{
    RayDFloat3 o;
    RayDFloat3 d;
    float tmax;
};

__target_intrinsic(cpp, "rayd::slang::Intersection")
struct RayDIntersection
{
    bool valid;
    float t;
    RayDFloat3 p;
    RayDFloat3 n;
    RayDFloat3 geo_n;
    RayDFloat2 uv;
    RayDFloat3 barycentric;
    int shape_id;
    int prim_id;
};

__target_intrinsic(cpp, "rayd::slang::NearestPointEdge")
struct RayDNearestPointEdge
{
    bool valid;
    float distance;
    RayDFloat3 point;
    float edge_t;
    RayDFloat3 edge_point;
    int shape_id;
    int edge_id;
    bool is_boundary;
};

__target_intrinsic(cpp, "rayd::slang::NearestRayEdge")
struct RayDNearestRayEdge
{
    bool valid;
    float distance;
    float ray_t;
    RayDFloat3 point;
    float edge_t;
    RayDFloat3 edge_point;
    int shape_id;
    int edge_id;
    bool is_boundary;
};

__target_intrinsic(cpp, "rayd::slang::PrimaryEdgeSample")
struct RayDPrimaryEdgeSample
{
    bool valid;
    float x_dot_n;
    int idx;
    RayDRay ray_n;
    RayDRay ray_p;
    float pdf;
};

__target_intrinsic(cpp, "rayd::slang::SceneHandle")
struct RayDSceneHandle
{
    uint64_t value;
};

__target_intrinsic(cpp, "rayd::slang::CameraHandle")
struct RayDCameraHandle
{
    uint64_t value;
};

float raydInfinity()
{
    return 1.0f / 0.0f;
}

RayDFloat2 raydFloat2(float x, float y)
{
    RayDFloat2 result;
    result.x = x;
    result.y = y;
    return result;
}

RayDFloat3 raydFloat3(float x, float y, float z)
{
    RayDFloat3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

RayDRay raydMakeRay(RayDFloat3 origin, RayDFloat3 direction, float tmax = raydInfinity())
{
    RayDRay result;
    result.o = origin;
    result.d = direction;
    result.tmax = tmax;
    return result;
}

RayDIntersection raydInvalidIntersection()
{
    RayDIntersection result;
    result.valid = false;
    result.t = raydInfinity();
    result.p = raydFloat3(0.0f, 0.0f, 0.0f);
    result.n = raydFloat3(0.0f, 0.0f, 0.0f);
    result.geo_n = raydFloat3(0.0f, 0.0f, 0.0f);
    result.uv = raydFloat2(0.0f, 0.0f);
    result.barycentric = raydFloat3(0.0f, 0.0f, 0.0f);
    result.shape_id = -1;
    result.prim_id = -1;
    return result;
}

RayDNearestPointEdge raydInvalidNearestPointEdge()
{
    RayDNearestPointEdge result;
    result.valid = false;
    result.distance = raydInfinity();
    result.point = raydFloat3(0.0f, 0.0f, 0.0f);
    result.edge_t = 0.0f;
    result.edge_point = raydFloat3(0.0f, 0.0f, 0.0f);
    result.shape_id = -1;
    result.edge_id = -1;
    result.is_boundary = false;
    return result;
}

RayDNearestRayEdge raydInvalidNearestRayEdge()
{
    RayDNearestRayEdge result;
    result.valid = false;
    result.distance = raydInfinity();
    result.ray_t = 0.0f;
    result.point = raydFloat3(0.0f, 0.0f, 0.0f);
    result.edge_t = 0.0f;
    result.edge_point = raydFloat3(0.0f, 0.0f, 0.0f);
    result.shape_id = -1;
    result.edge_id = -1;
    result.is_boundary = false;
    return result;
}

RayDPrimaryEdgeSample raydInvalidPrimaryEdgeSample()
{
    RayDPrimaryEdgeSample result;
    result.valid = false;
    result.x_dot_n = 0.0f;
    result.idx = -1;
    result.ray_n = raydMakeRay(raydFloat3(0.0f, 0.0f, 0.0f),
                               raydFloat3(0.0f, 0.0f, 1.0f));
    result.ray_p = raydMakeRay(raydFloat3(0.0f, 0.0f, 0.0f),
                               raydFloat3(0.0f, 0.0f, 1.0f));
    result.pdf = 0.0f;
    return result;
}

bool raydSceneIsReady(RayDSceneHandle scene)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::scene_is_ready($0)";
    default:
        return false;
    }
}

bool raydSceneHasPendingUpdates(RayDSceneHandle scene)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::scene_has_pending_updates($0)";
    default:
        return false;
    }
}

void raydSceneConfigure(RayDSceneHandle scene)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::scene_configure($0)";
    default:
        return;
    }
}

void raydSceneCommitUpdates(RayDSceneHandle scene)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::scene_commit_updates($0)";
    default:
        return;
    }
}

RayDIntersection raydSceneIntersect(RayDSceneHandle scene, RayDRay ray, bool active = true)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::scene_intersect($0, $1, $2)";
    default:
        return raydInvalidIntersection();
    }
}

bool raydSceneShadowTest(RayDSceneHandle scene, RayDRay ray, bool active = true)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::scene_shadow_test($0, $1, $2)";
    default:
        return false;
    }
}

RayDNearestPointEdge raydSceneClosestEdgePoint(RayDSceneHandle scene,
                                               RayDFloat3 point,
                                               bool active = true)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::scene_closest_edge_point($0, $1, $2)";
    default:
        return raydInvalidNearestPointEdge();
    }
}

RayDNearestRayEdge raydSceneClosestEdgeRay(RayDSceneHandle scene,
                                           RayDRay ray,
                                           bool active = true)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::scene_closest_edge_ray($0, $1, $2)";
    default:
        return raydInvalidNearestRayEdge();
    }
}

bool raydCameraIsReady(RayDCameraHandle camera)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::camera_is_ready($0)";
    default:
        return false;
    }
}

bool raydCameraPrimaryEdgesReady(RayDCameraHandle camera)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::camera_primary_edges_ready($0)";
    default:
        return false;
    }
}

void raydCameraSetResolution(RayDCameraHandle camera, int width, int height)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::camera_set_resolution($0, $1, $2)";
    default:
        return;
    }
}

void raydCameraConfigure(RayDCameraHandle camera, bool cache = true)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::camera_configure($0, $1)";
    default:
        return;
    }
}

void raydCameraPrepareEdges(RayDCameraHandle camera, RayDSceneHandle scene)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::camera_prepare_edges($0, $1)";
    default:
        return;
    }
}

RayDRay raydCameraSampleRay(RayDCameraHandle camera, RayDFloat2 sample)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::camera_sample_ray($0, $1)";
    default:
        return raydMakeRay(raydFloat3(0.0f, 0.0f, 0.0f),
                           raydFloat3(0.0f, 0.0f, 1.0f));
    }
}

RayDPrimaryEdgeSample raydCameraSamplePrimaryEdge(RayDCameraHandle camera, float sample)
{
    __target_switch
    {
    case cpp:
        __intrinsic_asm "rayd::slang::camera_sample_primary_edge($0, $1)";
    default:
        return raydInvalidPrimaryEdgeSample();
    }
}
