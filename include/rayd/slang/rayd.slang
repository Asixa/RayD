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
struct Float2 { float x; float y; };

__target_intrinsic(cpp, "rayd::slang::Float3")
struct Float3 { float x; float y; float z; };

__target_intrinsic(cpp, "rayd::slang::Ray")
struct Ray { Float3 o; Float3 d; float tmax; };

__target_intrinsic(cpp, "rayd::slang::Intersection")
struct Intersection
{
    bool valid; float t;
    Float3 p; Float3 n; Float3 geo_n;
    Float2 uv; Float3 barycentric;
    int shape_id; int prim_id;
};

__target_intrinsic(cpp, "rayd::slang::NearestPointEdge")
struct NearestPointEdge
{
    bool valid; float distance;
    Float3 point; float edge_t; Float3 edge_point;
    int shape_id; int edge_id; int global_edge_id; bool is_boundary;
};

__target_intrinsic(cpp, "rayd::slang::NearestRayEdge")
struct NearestRayEdge
{
    bool valid; float distance; float ray_t;
    Float3 point; float edge_t; Float3 edge_point;
    int shape_id; int edge_id; int global_edge_id; bool is_boundary;
};

__target_intrinsic(cpp, "rayd::slang::PrimaryEdgeSample")
struct PrimaryEdgeSample
{
    bool valid; float x_dot_n; int idx;
    Ray ray_n; Ray ray_p; float pdf;
};

__target_intrinsic(cpp, "rayd::slang::IntersectionAD")
struct IntersectionAD
{
    bool valid; float t;
    Float3 p; Float3 n; Float3 geo_n; Float3 barycentric;
    Float2 uv;
    int shape_id; int prim_id;
    Float3 dt_do; Float3 dt_dd;
};

__target_intrinsic(cpp, "rayd::slang::SceneHandle")
struct SceneHandle { uint64_t value; };

__target_intrinsic(cpp, "rayd::slang::CameraHandle")
struct CameraHandle { uint64_t value; };

// ---- Constructors ----

float infinity() { return 1.0f / 0.0f; }

Float2 float2(float x, float y)
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_float2($0, $1)";
    default:  Float2 r; r.x = x; r.y = y; return r;
    }
}

Float3 float3(float x, float y, float z)
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_float3($0, $1, $2)";
    default:  Float3 r; r.x = x; r.y = y; r.z = z; return r;
    }
}

Ray makeRay(Float3 origin, Float3 direction, float tmax = infinity())
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_ray($0, $1, $2)";
    default:  Ray r; r.o = origin; r.d = direction; r.tmax = tmax; return r;
    }
}

SceneHandle makeSceneHandle(uint64_t v)
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_scene_handle_raw($0)";
    default:  SceneHandle h; h.value = v; return h;
    }
}

CameraHandle makeCameraHandle(uint64_t v)
{
    __target_switch {
    case cpp: __intrinsic_asm "rayd::slang::make_camera_handle_raw($0)";
    default:  CameraHandle h; h.value = v; return h;
    }
}

// ---- Float2/3 field accessors ----

float f2X(Float2 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f2_x($0)"; default: return v.x; } }
float f2Y(Float2 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f2_y($0)"; default: return v.y; } }
float f3X(Float3 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f3_x($0)"; default: return v.x; } }
float f3Y(Float3 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f3_y($0)"; default: return v.y; } }
float f3Z(Float3 v) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::f3_z($0)"; default: return v.z; } }

// ---- Ray field accessors ----

Float3 rayO(Ray r)    { __target_switch { case cpp: __intrinsic_asm "rayd::slang::ray_o($0)";    default: return r.o; } }
Float3 rayD(Ray r)    { __target_switch { case cpp: __intrinsic_asm "rayd::slang::ray_d($0)";    default: return r.d; } }
float  rayTmax(Ray r) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::ray_tmax($0)"; default: return r.tmax; } }

// ---- Intersection field accessors ----

bool   itsValid(Intersection h)       { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_valid($0)";       default: return h.valid; } }
float  itsT(Intersection h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_t($0)";           default: return h.t; } }
Float3 itsP(Intersection h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_p($0)";           default: return h.p; } }
Float3 itsN(Intersection h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_n($0)";           default: return h.n; } }
Float3 itsGeoN(Intersection h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_geo_n($0)";       default: return h.geo_n; } }
Float2 itsUV(Intersection h)          { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_uv($0)";          default: return h.uv; } }
Float3 itsBarycentric(Intersection h) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_barycentric($0)"; default: return h.barycentric; } }
int    itsShapeId(Intersection h)     { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_shape_id($0)";    default: return h.shape_id; } }
int    itsPrimId(Intersection h)      { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_prim_id($0)";     default: return h.prim_id; } }

// ---- Invalid-result constructors (non-cpp targets) ----

Intersection invalidIntersection()
{
    Intersection result;
    result.valid = false; result.t = infinity();
    result.p = float3(0,0,0); result.n = float3(0,0,0); result.geo_n = float3(0,0,0);
    result.uv = float2(0,0); result.barycentric = float3(0,0,0);
    result.shape_id = -1; result.prim_id = -1;
    return result;
}

IntersectionAD invalidIntersectionAD()
{
    IntersectionAD r;
    r.valid = false; r.t = infinity();
    r.p = float3(0,0,0); r.n = float3(0,0,0); r.geo_n = float3(0,0,0);
    r.uv = float2(0,0); r.barycentric = float3(0,0,0);
    r.shape_id = -1; r.prim_id = -1;
    r.dt_do = float3(0,0,0); r.dt_dd = float3(0,0,0);
    return r;
}

NearestPointEdge invalidNearestPointEdge()
{
    NearestPointEdge r;
    r.valid = false; r.distance = infinity();
    r.point = float3(0,0,0); r.edge_t = 0; r.edge_point = float3(0,0,0);
    r.shape_id = -1; r.edge_id = -1; r.global_edge_id = -1; r.is_boundary = false;
    return r;
}

NearestRayEdge invalidNearestRayEdge()
{
    NearestRayEdge r;
    r.valid = false; r.distance = infinity(); r.ray_t = 0;
    r.point = float3(0,0,0); r.edge_t = 0; r.edge_point = float3(0,0,0);
    r.shape_id = -1; r.edge_id = -1; r.global_edge_id = -1; r.is_boundary = false;
    return r;
}

PrimaryEdgeSample invalidPrimaryEdgeSample()
{
    PrimaryEdgeSample r;
    r.valid = false; r.x_dot_n = 0; r.idx = -1;
    r.ray_n = makeRay(float3(0,0,0), float3(0,0,1));
    r.ray_p = makeRay(float3(0,0,0), float3(0,0,1));
    r.pdf = 0;
    return r;
}

// ---- Scene queries ----

bool sceneIsReady(SceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_is_ready($0)"; default: return false; } }

bool sceneHasPendingUpdates(SceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_has_pending_updates($0)"; default: return false; } }

void sceneBuild(SceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_build($0)"; default: return; } }

void sceneSync(SceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_sync($0)"; default: return; } }

Intersection sceneIntersect(SceneHandle scene, Ray ray, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_intersect($0, $1, $2)"; default: return invalidIntersection(); } }

bool sceneShadowTest(SceneHandle scene, Ray ray, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_shadow_test($0, $1, $2)"; default: return false; } }

// ---- AD scene query (forward + backward in one call) ----

IntersectionAD sceneIntersectAD(SceneHandle scene, Ray ray, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_intersect_ad($0, $1, $2)"; default: return invalidIntersectionAD(); } }

// ---- IntersectionAD field accessors ----

bool   itsADValid(IntersectionAD h)       { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_valid($0)";       default: return h.valid; } }
float  itsADT(IntersectionAD h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_t($0)";           default: return h.t; } }
Float3 itsADP(IntersectionAD h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_p($0)";           default: return h.p; } }
Float3 itsADN(IntersectionAD h)           { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_n($0)";           default: return h.n; } }
Float3 itsADGeoN(IntersectionAD h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_geo_n($0)";       default: return h.geo_n; } }
Float2 itsADUV(IntersectionAD h)          { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_uv($0)";          default: return h.uv; } }
Float3 itsADBarycentric(IntersectionAD h) { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_barycentric($0)"; default: return h.barycentric; } }
int    itsADShapeId(IntersectionAD h)     { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_shape_id($0)";    default: return h.shape_id; } }
int    itsADPrimId(IntersectionAD h)      { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_prim_id($0)";     default: return h.prim_id; } }
Float3 itsADDtDo(IntersectionAD h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_dt_do($0)";       default: return h.dt_do; } }
Float3 itsADDtDd(IntersectionAD h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::its_ad_dt_dd($0)";       default: return h.dt_dd; } }

// ---- Nearest-edge field accessors ----

bool   npeValid(NearestPointEdge h)         { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_valid($0)";          default: return h.valid; } }
float  npeDistance(NearestPointEdge h)      { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_distance($0)";       default: return h.distance; } }
Float3 npePoint(NearestPointEdge h)         { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_point($0)";          default: return h.point; } }
float  npeEdgeT(NearestPointEdge h)         { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_edge_t($0)";         default: return h.edge_t; } }
Float3 npeEdgePoint(NearestPointEdge h)     { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_edge_point($0)";     default: return h.edge_point; } }
int    npeShapeId(NearestPointEdge h)       { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_shape_id($0)";       default: return h.shape_id; } }
int    npeEdgeId(NearestPointEdge h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_edge_id($0)";        default: return h.edge_id; } }
int    npeGlobalEdgeId(NearestPointEdge h)  { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_global_edge_id($0)"; default: return h.global_edge_id; } }
bool   npeIsBoundary(NearestPointEdge h)    { __target_switch { case cpp: __intrinsic_asm "rayd::slang::npe_is_boundary($0)";    default: return h.is_boundary; } }

bool   nreValid(NearestRayEdge h)         { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_valid($0)";          default: return h.valid; } }
float  nreDistance(NearestRayEdge h)      { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_distance($0)";       default: return h.distance; } }
float  nreRayT(NearestRayEdge h)          { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_ray_t($0)";          default: return h.ray_t; } }
Float3 nrePoint(NearestRayEdge h)         { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_point($0)";          default: return h.point; } }
float  nreEdgeT(NearestRayEdge h)         { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_edge_t($0)";         default: return h.edge_t; } }
Float3 nreEdgePoint(NearestRayEdge h)     { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_edge_point($0)";     default: return h.edge_point; } }
int    nreShapeId(NearestRayEdge h)       { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_shape_id($0)";       default: return h.shape_id; } }
int    nreEdgeId(NearestRayEdge h)        { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_edge_id($0)";        default: return h.edge_id; } }
int    nreGlobalEdgeId(NearestRayEdge h)  { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_global_edge_id($0)"; default: return h.global_edge_id; } }
bool   nreIsBoundary(NearestRayEdge h)    { __target_switch { case cpp: __intrinsic_asm "rayd::slang::nre_is_boundary($0)";    default: return h.is_boundary; } }

NearestPointEdge sceneNearestEdgePoint(SceneHandle scene, Float3 point, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_nearest_edge_point($0, $1, $2)"; default: return invalidNearestPointEdge(); } }

NearestRayEdge sceneNearestEdgeRay(SceneHandle scene, Ray ray, bool active = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_nearest_edge_ray($0, $1, $2)"; default: return invalidNearestRayEdge(); } }

int sceneEdgeCount(SceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_edge_count($0)"; default: return 0; } }

bool sceneEdgeMaskValue(SceneHandle scene, int index)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_edge_mask_value($0, $1)"; default: return false; } }

void sceneSetEdgeMask(SceneHandle scene, uint64_t maskPtr, int count)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::scene_set_edge_mask($0, $1, $2)"; default: return; } }

// ---- Camera queries ----

bool cameraIsReady(CameraHandle camera)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_is_ready($0)"; default: return false; } }

bool cameraPrimaryEdgesReady(CameraHandle camera)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_primary_edges_ready($0)"; default: return false; } }

void cameraSetResolution(CameraHandle camera, int width, int height)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_set_resolution($0, $1, $2)"; default: return; } }

void cameraBuild(CameraHandle camera, bool cache = true)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_build($0, $1)"; default: return; } }

void cameraPrepareEdges(CameraHandle camera, SceneHandle scene)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_prepare_edges($0, $1)"; default: return; } }

Ray cameraSampleRay(CameraHandle camera, Float2 sample)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_sample_ray($0, $1)"; default: return makeRay(float3(0,0,0), float3(0,0,1)); } }

PrimaryEdgeSample cameraSamplePrimaryEdge(CameraHandle camera, float sample)
{ __target_switch { case cpp: __intrinsic_asm "rayd::slang::camera_sample_primary_edge($0, $1)"; default: return invalidPrimaryEdgeSample(); } }
