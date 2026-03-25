// Slang interop layer for RayD's host-side C++ API.
// Intended for Slang's C++ target. Other targets return invalid sentinels.
//
// NOTE: When compiling the generated C++ output, the build system must
// ensure <rayd/slang/interop.h> is on the include path and that the
// generated source includes it (e.g. via a force-include flag or by
// prepending the directive to the generated .cpp file).
//
// All constructors and field accessors use __intrinsic_asm on the cpp target
// so that field-name mangling from slangc does not cause mismatches with the
// actual C++ struct definitions in interop.h.

// ---- POD types ----

__target_intrinsic(cpp, "rayd::slang::Float2")
struct RayDFloat2 { float x; float y; };

__target_intrinsic(cpp, "rayd::slang::Float3")
struct RayDFloat3 { float x; float y; float z; };

__target_intrinsic(cpp, "rayd::slang::Ray")
struct RayDRay { RayDFloat3 o; RayDFloat3 d; float tmax; };

__target_intrinsic(cpp, "rayd::slang::Intersection")
struct RayDIntersection
{
    bool valid; float t;
    RayDFloat3 p; RayDFloat3 n; RayDFloat3 geo_n;
    RayDFloat2 uv; RayDFloat3 barycentric;
    int shape_id; int prim_id;
};

__target_intrinsic(cpp, "rayd::slang::NearestPointEdge")
struct RayDNearestPointEdge
{
    bool valid; float distance;
    RayDFloat3 point; float edge_t; RayDFloat3 edge_point;
    int shape_id; int edge_id; bool is_boundary;
};

__target_intrinsic(cpp, "rayd::slang::NearestRayEdge")
struct RayDNearestRayEdge
{
    bool valid; float distance; float ray_t;
    RayDFloat3 point; float edge_t; RayDFloat3 edge_point;
    int shape_id; int edge_id; bool is_boundary;
};

__target_intrinsic(cpp, "rayd::slang::PrimaryEdgeSample")
struct RayDPrimaryEdgeSample
{
    bool valid; float x_dot_n; int idx;
    RayDRay ray_n; RayDRay ray_p; float pdf;
};

__target_intrinsic(cpp, "rayd::slang::IntersectionAD")
struct RayDIntersectionAD
{
    bool valid; float t;
    RayDFloat3 p; RayDFloat3 n; RayDFloat3 geo_n; RayDFloat3 barycentric;
    RayDFloat2 uv;
    int shape_id; int prim_id;
    RayDFloat3 dt_do; RayDFloat3 dt_dd;
};

__target_intrinsic(cpp, "rayd::slang::SceneHandle")
struct RayDSceneHandle { uint64_t value; };

__target_intrinsic(cpp, "rayd::slang::CameraHandle")
struct RayDCameraHandle { uint64_t value; };

// ---- Constructors ----

float raydInfinity() { return 1.0f / 0.0f; }

RayDFloat2 raydFloat2(float x, float y)
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_float2($0, $1)";
    default:  RayDFloat2 r; r.x = x; r.y = y; return r;
    }
}

RayDFloat3 raydFloat3(float x, float y, float z)
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_float3($0, $1, $2)";
    default:  RayDFloat3 r; r.x = x; r.y = y; r.z = z; return r;
    }
}

RayDRay raydMakeRay(RayDFloat3 origin, RayDFloat3 direction, float tmax = raydInfinity())
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_ray($0, $1, $2)";
    default:  RayDRay r; r.o = origin; r.d = direction; r.tmax = tmax; return r;
    }
}

RayDSceneHandle raydMakeSceneHandle(uint64_t v)
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_scene_handle_raw($0)";
    default:  RayDSceneHandle h; h.value = v; return h;
    }
}

RayDCameraHandle raydMakeCameraHandle(uint64_t v)
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_camera_handle_raw($0)";
    default:  RayDCameraHandle h; h.value = v; return h;
    }
}

// ---- Float2/3 field accessors ----

float raydF2X(RayDFloat2 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f2_x($0)"; default: return v.x; } }
float raydF2Y(RayDFloat2 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f2_y($0)"; default: return v.y; } }
float raydF3X(RayDFloat3 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f3_x($0)"; default: return v.x; } }
float raydF3Y(RayDFloat3 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f3_y($0)"; default: return v.y; } }
float raydF3Z(RayDFloat3 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f3_z($0)"; default: return v.z; } }

// ---- Ray field accessors ----

RayDFloat3 raydRayO(RayDRay r)    { __target_switch { case cpp: __intrinsic_asm "rayd::slang::ray_o($0)";    default: return r.o; } }
RayDFloat3 raydRayD(RayDRay r)    { __target_switch { case cpp: __intrinsic_asm "rayd::slang::ray_d($0)";    default: return r.d; } }
float      raydRayTmax(RayDRay r) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::ray_tmax($0)"; default: return r.tmax; } }

// ---- Intersection field accessors ----

bool       raydItsValid(RayDIntersection h)       { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_valid($0)";       default: return h.valid; } }
float      raydItsT(RayDIntersection h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_t($0)";           default: return h.t; } }
RayDFloat3 raydItsP(RayDIntersection h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_p($0)";           default: return h.p; } }
RayDFloat3 raydItsN(RayDIntersection h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_n($0)";           default: return h.n; } }
RayDFloat3 raydItsGeoN(RayDIntersection h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_geo_n($0)";       default: return h.geo_n; } }
RayDFloat2 raydItsUV(RayDIntersection h)          { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_uv($0)";          default: return h.uv; } }
RayDFloat3 raydItsBarycentric(RayDIntersection h) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_barycentric($0)"; default: return h.barycentric; } }
int        raydItsShapeId(RayDIntersection h)     { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_shape_id($0)";    default: return h.shape_id; } }
int        raydItsPrimId(RayDIntersection h)      { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_prim_id($0)";     default: return h.prim_id; } }

// ---- Invalid-result constructors (non-cpp targets) ----

RayDIntersection raydInvalidIntersection()
{
    RayDIntersection result;
    result.valid = false; result.t = raydInfinity();
    result.p = raydFloat3(0,0,0); result.n = raydFloat3(0,0,0); result.geo_n = raydFloat3(0,0,0);
    result.uv = raydFloat2(0,0); result.barycentric = raydFloat3(0,0,0);
    result.shape_id = -1; result.prim_id = -1;
    return result;
}

RayDIntersectionAD raydInvalidIntersectionAD()
{
    RayDIntersectionAD r;
    r.valid = false; r.t = raydInfinity();
    r.p = raydFloat3(0,0,0); r.n = raydFloat3(0,0,0); r.geo_n = raydFloat3(0,0,0);
    r.uv = raydFloat2(0,0); r.barycentric = raydFloat3(0,0,0);
    r.shape_id = -1; r.prim_id = -1;
    r.dt_do = raydFloat3(0,0,0); r.dt_dd = raydFloat3(0,0,0);
    return r;
}

RayDNearestPointEdge raydInvalidNearestPointEdge()
{
    RayDNearestPointEdge r;
    r.valid = false; r.distance = raydInfinity();
    r.point = raydFloat3(0,0,0); r.edge_t = 0; r.edge_point = raydFloat3(0,0,0);
    r.shape_id = -1; r.edge_id = -1; r.is_boundary = false;
    return r;
}

RayDNearestRayEdge raydInvalidNearestRayEdge()
{
    RayDNearestRayEdge r;
    r.valid = false; r.distance = raydInfinity(); r.ray_t = 0;
    r.point = raydFloat3(0,0,0); r.edge_t = 0; r.edge_point = raydFloat3(0,0,0);
    r.shape_id = -1; r.edge_id = -1; r.is_boundary = false;
    return r;
}

RayDPrimaryEdgeSample raydInvalidPrimaryEdgeSample()
{
    RayDPrimaryEdgeSample r;
    r.valid = false; r.x_dot_n = 0; r.idx = -1;
    r.ray_n = raydMakeRay(raydFloat3(0,0,0), raydFloat3(0,0,1));
    r.ray_p = raydMakeRay(raydFloat3(0,0,0), raydFloat3(0,0,1));
    r.pdf = 0;
    return r;
}

// ---- Scene queries ----

bool raydSceneIsReady(RayDSceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_is_ready($0)"; default: return false; } }

bool raydSceneHasPendingUpdates(RayDSceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_has_pending_updates($0)"; default: return false; } }

void raydSceneConfigure(RayDSceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_configure($0)"; default: return; } }

void raydSceneCommitUpdates(RayDSceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_commit_updates($0)"; default: return; } }

RayDIntersection raydSceneIntersect(RayDSceneHandle scene, RayDRay ray, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_intersect($0, $1, $2)"; default: return raydInvalidIntersection(); } }

bool raydSceneShadowTest(RayDSceneHandle scene, RayDRay ray, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_shadow_test($0, $1, $2)"; default: return false; } }

// ---- AD scene query (forward + backward in one call) ----

RayDIntersectionAD raydSceneIntersectAD(RayDSceneHandle scene, RayDRay ray, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_intersect_ad($0, $1, $2)"; default: return raydInvalidIntersectionAD(); } }

// ---- IntersectionAD field accessors ----

bool       raydItsADValid(RayDIntersectionAD h)       { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_valid($0)";       default: return h.valid; } }
float      raydItsADT(RayDIntersectionAD h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_t($0)";           default: return h.t; } }
RayDFloat3 raydItsADP(RayDIntersectionAD h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_p($0)";           default: return h.p; } }
RayDFloat3 raydItsADN(RayDIntersectionAD h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_n($0)";           default: return h.n; } }
RayDFloat3 raydItsADGeoN(RayDIntersectionAD h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_geo_n($0)";       default: return h.geo_n; } }
RayDFloat2 raydItsADUV(RayDIntersectionAD h)          { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_uv($0)";          default: return h.uv; } }
RayDFloat3 raydItsADBarycentric(RayDIntersectionAD h) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_barycentric($0)"; default: return h.barycentric; } }
int        raydItsADShapeId(RayDIntersectionAD h)     { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_shape_id($0)";    default: return h.shape_id; } }
int        raydItsADPrimId(RayDIntersectionAD h)      { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_prim_id($0)";     default: return h.prim_id; } }
RayDFloat3 raydItsADDtDo(RayDIntersectionAD h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_dt_do($0)";       default: return h.dt_do; } }
RayDFloat3 raydItsADDtDd(RayDIntersectionAD h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_dt_dd($0)";       default: return h.dt_dd; } }

RayDNearestPointEdge raydSceneClosestEdgePoint(RayDSceneHandle scene, RayDFloat3 point, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_closest_edge_point($0, $1, $2)"; default: return raydInvalidNearestPointEdge(); } }

RayDNearestRayEdge raydSceneClosestEdgeRay(RayDSceneHandle scene, RayDRay ray, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_closest_edge_ray($0, $1, $2)"; default: return raydInvalidNearestRayEdge(); } }

// ---- Camera queries ----

bool raydCameraIsReady(RayDCameraHandle camera)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_is_ready($0)"; default: return false; } }

bool raydCameraPrimaryEdgesReady(RayDCameraHandle camera)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_primary_edges_ready($0)"; default: return false; } }

void raydCameraSetResolution(RayDCameraHandle camera, int width, int height)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_set_resolution($0, $1, $2)"; default: return; } }

void raydCameraConfigure(RayDCameraHandle camera, bool cache = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_configure($0, $1)"; default: return; } }

void raydCameraPrepareEdges(RayDCameraHandle camera, RayDSceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_prepare_edges($0, $1)"; default: return; } }

RayDRay raydCameraSampleRay(RayDCameraHandle camera, RayDFloat2 sample)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_sample_ray($0, $1)"; default: return raydMakeRay(raydFloat3(0,0,0), raydFloat3(0,0,1)); } }

RayDPrimaryEdgeSample raydCameraSamplePrimaryEdge(RayDCameraHandle camera, float sample)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_sample_primary_edge($0, $1)"; default: return raydInvalidPrimaryEdgeSample(); } }
